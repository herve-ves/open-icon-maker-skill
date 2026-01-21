# /// script
# dependencies = [
#   "openai",
#   "pillow",
#   "pydantic",
#   "jinja2",
# ]
# ///

from __future__ import annotations

import argparse
import asyncio
import base64
import logging
import os
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Optional, Tuple
from uuid import uuid4

from jinja2 import Environment, FileSystemLoader, TemplateNotFound
from openai import AsyncOpenAI
from PIL import Image, ImageChops, ImageDraw
from pydantic import BaseModel, Field

_TEMPLATE_CACHE_MAX_SIZE = 8
_TEMPLATE_DIR_ENV = "ICON_MAKER_TEMPLATE_DIR"


def _resolve_template_dir() -> Path:
    configured = os.getenv(_TEMPLATE_DIR_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parent.parent / "assets" / "templates"


@lru_cache(maxsize=1)
def _get_template_env() -> Environment:
    template_dir = _resolve_template_dir()
    if not template_dir.exists():
        raise FileNotFoundError(
            f"Template directory not found: {template_dir} "
            f"(override with {_TEMPLATE_DIR_ENV})"
        )
    return Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )


@lru_cache(maxsize=_TEMPLATE_CACHE_MAX_SIZE)
def _get_template(name: str):
    try:
        return _get_template_env().get_template(name)
    except TemplateNotFound as exc:
        raise FileNotFoundError(
            f"Missing template '{name}' in {_resolve_template_dir()}"
        ) from exc


def render_icon_style_prompt(category: str, description: str) -> str:
    return _get_template("icon_style.j2").render(
        category=category,
        description=description,
    )


def render_icon_brief_instructions() -> str:
    return _get_template("icon_brief_instructions.j2").render()


def render_icon_brief_user_prompt(style_prompt: str) -> str:
    return _get_template("icon_brief_request.j2").render(style_prompt=style_prompt)


class DesignBrief(BaseModel):
    """Structured design brief for icon generation."""

    visual_concept: str = Field(
        description=(
            "The main subject or object to depict in the icon. "
            "Should be specific and clearly identifiable."
        )
    )
    composition: str = Field(
        description=(
            "Layout, positioning, and perspective details. "
            "E.g., centered, front view, isometric, etc."
        )
    )
    style: str = Field(
        description=(
            "Visual style specifications. "
            "E.g., minimalist, flat design, geometric shapes, rounded lines."
        )
    )
    color: str = Field(
        description=(
            "Color specifications including hex codes. "
            "E.g., single solid color #000000 (pure black), no gradients."
        )
    )
    restrictions: str = Field(
        description=(
            "What to avoid in the design. "
            "E.g., no outlines, no shadows, no text, no 3D effects."
        )
    )
    background: str = Field(
        description="Background specifications. E.g., transparent, solid color."
    )
    context: str = Field(
        description=(
            "Usage context and format requirements. "
            "E.g., health tracking app icon, 1:1 square format."
        )
    )

    def to_prompt(self) -> str:
        return self.model_dump_json(indent=4)


@dataclass
class IconConfig:
    quality: str = "auto"
    gpt_model: str = "gpt-5.2"
    gpt_reasoning_effort: str = "medium"
    image_model: str = "gpt-image-1.5"

    foreground_color: Optional[str] = "#000000"
    foreground_opacity: float = 1.0
    foreground_alpha_threshold: int = 0


@dataclass
class BackgroundConfig:
    canvas_size: int = 88
    icon_size: int = 48
    corner_radius: int = 32
    bg_color: str = "#000000"
    bg_opacity: float = 0.02
    crop_to_content: bool = True
    crop_alpha_threshold: int = 64
    debug_save_cropped: bool = False


@dataclass
class GenerationResult:
    prompt: str
    output_path: Path
    elapsed_ms: float


def _parse_hex_rgb(color: str) -> Tuple[int, int, int]:
    normalized = color.strip().lstrip("#")
    if len(normalized) != 6:
        raise ValueError(f"Invalid hex color: {color!r}")

    r = int(normalized[0:2], 16)
    g = int(normalized[2:4], 16)
    b = int(normalized[4:6], 16)
    return r, g, b


def _create_rounded_rectangle_mask(
    size: Tuple[int, int],
    radius: int,
    opacity: int = 255,
) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle([(0, 0), (size[0], size[1])], radius=radius, fill=opacity)
    return mask


def _crop_to_content_square(
    img: Image.Image, alpha_threshold: int = 128
) -> Image.Image:
    alpha = img.getchannel("A")
    lut = [255 if i >= alpha_threshold else 0 for i in range(256)]
    alpha_thresholded = alpha.point(lut)
    bbox = alpha_thresholded.getbbox()
    if bbox is None:
        return img

    left, top, right, bottom = bbox
    content_width = right - left
    content_height = bottom - top
    side = max(content_width, content_height)

    center_x = (left + right) // 2
    center_y = (top + bottom) // 2
    half_side = side // 2

    sq_left = center_x - half_side
    sq_top = center_y - half_side
    sq_right = sq_left + side
    sq_bottom = sq_top + side

    img_width, img_height = img.size

    if sq_left < 0:
        sq_right -= sq_left
        sq_left = 0
    if sq_top < 0:
        sq_bottom -= sq_top
        sq_top = 0
    if sq_right > img_width:
        sq_left -= sq_right - img_width
        sq_right = img_width
    if sq_bottom > img_height:
        sq_top -= sq_bottom - img_height
        sq_bottom = img_height

    sq_left = max(0, sq_left)
    sq_top = max(0, sq_top)

    cropped = img.crop((sq_left, sq_top, sq_right, sq_bottom))

    crop_w, crop_h = cropped.size
    if crop_w != crop_h:
        new_side = max(crop_w, crop_h)
        square = Image.new("RGBA", (new_side, new_side), (0, 0, 0, 0))
        paste_x = (new_side - crop_w) // 2
        paste_y = (new_side - crop_h) // 2
        square.paste(cropped, (paste_x, paste_y))
        return square

    return cropped


def add_rounded_background(
    image_path: str,
    output_path: str,
    config: Optional[BackgroundConfig] = None,
) -> None:
    if config is None:
        config = BackgroundConfig()

    input_path = Path(image_path)
    out_path = Path(output_path)

    img = Image.open(input_path).convert("RGBA")

    if config.crop_to_content:
        img = _crop_to_content_square(img, config.crop_alpha_threshold)
        if config.debug_save_cropped:
            cropped_path = out_path.with_name(
                f"{out_path.stem}_cropped{out_path.suffix}"
            )
            cropped_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(cropped_path, "PNG")

    r, g, b = _parse_hex_rgb(config.bg_color)
    alpha = int(round(config.bg_opacity * 255))

    icon = img.resize((config.icon_size, config.icon_size), Image.Resampling.LANCZOS)

    canvas = Image.new(
        "RGBA", (config.canvas_size, config.canvas_size), (r, g, b, alpha)
    )

    offset_x = (config.canvas_size - config.icon_size) // 2
    offset_y = (config.canvas_size - config.icon_size) // 2
    canvas.paste(icon, (offset_x, offset_y), icon)

    rounded_mask = _create_rounded_rectangle_mask(
        (config.canvas_size, config.canvas_size),
        config.corner_radius,
        opacity=255,
    )

    _r_channel, _g_channel, _b_channel, a_channel = canvas.split()
    canvas.putalpha(ImageChops.multiply(a_channel, rounded_mask))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, "PNG")


def _postprocess_icon_png(
    input_path: str,
    output_path: str,
    foreground_color: Optional[str],
    foreground_opacity: float,
    foreground_alpha_threshold: int,
) -> Path:
    opacity = max(0.0, min(float(foreground_opacity), 1.0))
    threshold = max(0, min(int(foreground_alpha_threshold), 255))

    in_path = Path(input_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if foreground_color is None and opacity == 1.0 and threshold == 0:
        if in_path != out_path:
            shutil.copyfile(in_path, out_path)
        return out_path

    img = Image.open(in_path).convert("RGBA")

    alpha = img.getchannel("A")
    lut = [0] * 256
    for i in range(256):
        if i < threshold:
            lut[i] = 0
            continue
        lut[i] = min(255, int(round(i * opacity)))

    new_alpha = alpha.point(lut)

    if foreground_color is not None:
        r, g, b = _parse_hex_rgb(foreground_color)
        recolored = Image.new("RGBA", img.size, (r, g, b, 0))
        recolored.putalpha(new_alpha)
        img = recolored
    else:
        r_channel, g_channel, b_channel, _ = img.split()
        img = Image.merge("RGBA", (r_channel, g_channel, b_channel, new_alpha))

    img.save(out_path, format="PNG")
    return out_path


class IconGenerator:
    def __init__(
        self,
        *,
        config: Optional[IconConfig] = None,
        openai_client: Optional[AsyncOpenAI] = None,
    ) -> None:
        self.config = config or IconConfig()
        self.client = openai_client or AsyncOpenAI()

    async def create_brief(
        self,
        category: str,
        description: str,
        model: Optional[str] = None,
    ) -> DesignBrief:
        model = model or self.config.gpt_model

        instructions = render_icon_brief_instructions()
        style_prompt = render_icon_style_prompt(category, description)
        user_prompt = render_icon_brief_user_prompt(style_prompt)

        response = await self.client.responses.parse(
            model=model,
            instructions=instructions,
            input=user_prompt,
            text_format=DesignBrief,
            reasoning={"effort": self.config.gpt_reasoning_effort},  # type: ignore
        )

        brief = response.output_parsed
        if brief is None:
            raise ValueError("Failed to parse design brief from response")

        return brief

    async def generate(
        self,
        brief: object,
        output_folder: Path,
        size: str = "1024x1024",
    ) -> GenerationResult:
        start_time = perf_counter()

        prompt = brief.to_prompt() if isinstance(brief, DesignBrief) else str(brief)

        response = await self.client.images.generate(
            model=self.config.image_model,
            prompt=prompt,
            size=size,  # type: ignore
            quality=self.config.quality,  # type: ignore
            moderation="low",
            n=1,
            background="transparent",
            output_format="png",
        )

        image_b64 = response.data[0].b64_json if response.data else None
        if not image_b64:
            raise ValueError("No image data returned from API")

        raw_image_data = base64.b64decode(image_b64)

        file_id = uuid4()
        output_folder.mkdir(parents=True, exist_ok=True)

        raw_output_path = output_folder / f"{file_id}_raw.png"
        raw_output_path.write_bytes(raw_image_data)

        output_path = output_folder / f"{file_id}.png"
        _postprocess_icon_png(
            str(raw_output_path),
            str(output_path),
            self.config.foreground_color,
            self.config.foreground_opacity,
            self.config.foreground_alpha_threshold,
        )

        elapsed_ms = (perf_counter() - start_time) * 1000
        return GenerationResult(
            prompt=prompt, output_path=output_path, elapsed_ms=elapsed_ms
        )


def _add_icon_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output",
        "-o",
        default="./output",
        help="Output folder for generated icons.",
    )
    parser.add_argument(
        "--quality",
        choices=["auto", "high", "medium", "low"],
        default="auto",
        help="Image quality setting.",
    )
    parser.add_argument(
        "--size",
        choices=["1024x1024", "1536x1024", "1024x1536", "auto"],
        default="1024x1024",
        help="Image size.",
    )
    parser.add_argument(
        "--gpt-model",
        default="gpt-5.2",
        help="OpenAI model for generating the design brief.",
    )
    parser.add_argument(
        "--gpt-reasoning-effort",
        choices=["none", "low", "medium", "high"],
        default="medium",
        help="Reasoning effort (only for reasoning models).",
    )
    parser.add_argument(
        "--image-model",
        default="gpt-image-1.5",
        help="OpenAI model for image generation.",
    )
    parser.add_argument(
        "--no-recolor",
        action="store_true",
        help="Do not recolor foreground; keep model output colors.",
    )
    parser.add_argument(
        "--foreground-color",
        default="#000000",
        help="Recolor icon foreground to hex color (ignored with --no-recolor).",
    )
    parser.add_argument(
        "--foreground-opacity",
        type=float,
        default=0.9,
        help="Multiply icon alpha by 0..1.",
    )
    parser.add_argument(
        "--foreground-alpha-threshold",
        type=int,
        default=64,
        help="Alpha cutoff 0..255 before opacity scaling.",
    )


def _add_background_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--canvas-size",
        type=int,
        default=88,
        help="Final background canvas size in pixels.",
    )
    parser.add_argument(
        "--icon-size",
        type=int,
        default=48,
        help="Icon size before placing on canvas.",
    )
    parser.add_argument(
        "--corner-radius",
        type=int,
        default=32,
        help="Corner radius for rounded rectangle background.",
    )
    parser.add_argument(
        "--bg-color",
        default="#F0F0F0",
        help="Background color in hex format.",
    )
    parser.add_argument(
        "--bg-opacity",
        type=float,
        default=1.0,
        help="Background opacity (0.0-1.0).",
    )
    parser.add_argument(
        "--crop-to-content",
        action="store_true",
        default=True,
        help="Crop to content bounding square before resizing.",
    )
    parser.add_argument(
        "--no-crop-to-content",
        action="store_false",
        dest="crop_to_content",
        help="Disable cropping to content bounding square before resizing.",
    )
    parser.add_argument(
        "--crop-alpha-threshold",
        type=int,
        default=64,
        help="Alpha threshold (0..255) used when cropping to content.",
    )
    parser.add_argument(
        "--debug-save-cropped",
        action="store_true",
        help="Save cropped image with '_cropped' suffix for debugging.",
    )


async def _run_generate(args: argparse.Namespace) -> None:
    config = IconConfig(
        quality=args.quality,
        gpt_model=args.gpt_model,
        gpt_reasoning_effort=args.gpt_reasoning_effort,
        image_model=args.image_model,
        foreground_color=None if args.no_recolor else args.foreground_color,
        foreground_opacity=args.foreground_opacity,
        foreground_alpha_threshold=args.foreground_alpha_threshold,
    )
    generator = IconGenerator(config=config, openai_client=AsyncOpenAI())

    brief = await generator.create_brief(args.category, args.description)
    result = await generator.generate(brief, Path(args.output), size=args.size)

    raw_path = result.output_path.with_name(f"{result.output_path.stem}_raw.png")
    print(f"Output: {result.output_path}")
    print(f"Raw: {raw_path}")
    print(f"Elapsed: {result.elapsed_ms:.0f}ms")


async def _run_add_background(args: argparse.Namespace) -> None:
    config = BackgroundConfig(
        canvas_size=args.canvas_size,
        icon_size=args.icon_size,
        corner_radius=args.corner_radius,
        bg_color=args.bg_color,
        bg_opacity=args.bg_opacity,
        crop_to_content=args.crop_to_content,
        crop_alpha_threshold=args.crop_alpha_threshold,
        debug_save_cropped=args.debug_save_cropped,
    )
    await asyncio.to_thread(
        add_rounded_background, args.input_path, args.output_path, config
    )
    print(f"Output saved to: {args.output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="icon_maker.py",
        description="Generate and post-process UI icons using OpenAI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate",
        help="Generate an icon from category + description (brief → image).",
    )
    generate.add_argument(
        "--category",
        "-c",
        required=True,
        help="Icon category name.",
    )
    generate.add_argument(
        "--description",
        "-d",
        required=True,
        help="Detailed description for the category icon.",
    )

    add_bg = subparsers.add_parser(
        "add_background", help="Add a rounded background to an existing PNG."
    )
    add_bg.add_argument("input_path", help="Path to input PNG.")
    add_bg.add_argument("output_path", help="Path to output PNG.")

    _add_icon_options(generate)
    _add_background_options(add_bg)

    generate.set_defaults(_runner=_run_generate)
    add_bg.set_defaults(_runner=_run_add_background)

    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(args._runner(args))


if __name__ == "__main__":
    main()
