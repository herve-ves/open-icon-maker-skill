---
name: open-icon-maker
description: "Generate and post-process UI icons (PNG) with OpenAI: create a structured design brief, generate a transparent icon body, enforce single-color/alpha rules, and optionally add a rounded background tile. Use for app/category icon generation, prompt template tuning, and background/crop/alpha troubleshooting."
---

# Open Icon Maker

## Quick start

- Paths below are relative to the skill folder.
- Prereqs: Python 3.10+ and packages `openai`, `pillow`, `pydantic`, `jinja2`.
- If `uv` is available, run without manual installs: `uv run scripts/icon_maker.py --help`
- Install deps: `python -m pip install openai pillow pydantic jinja2`
- Run: `python scripts/icon_maker.py --help`

## Default workflow (recommended)

1. Clarify inputs (only if missing/ambiguous):
   - `category`: what feature/domain the icon represents (e.g. "Food Intake", "Sleep").
   - `description`: what the glyph should depict, plus any hard requirements (no text, single subject, etc.).
   - Background tile: confirm whether a rounded background is required; if yes, confirm sizing/colors or use defaults.
2. Generate icon (brief → image): use the `generate` command.
3. Add rounded background tile: take the printed `Output: <path>` PNG and run `add_background` to produce a `*_bg.png`.

## Generate an icon (brief → image)

1. Ensure `OPENAI_API_KEY` is set (avoid passing keys on the command line).
2. Run:
   - `python scripts/icon_maker.py generate --category "Food Intake" --description "A healthy salad bowl" -o ./output`

Outputs:
- `<uuid>.png`: postprocessed icon body (transparent background)
- `<uuid>_raw.png`: raw API output

## Add background to an existing PNG

- `python scripts/icon_maker.py add_background input.png output.png --bg-opacity 0.02 --canvas-size 88 --icon-size 48`

## Prompt templates (Jinja2)

- Templates live in `assets/templates/`:
  - `assets/templates/icon_style.j2`
  - `assets/templates/icon_brief_instructions.j2`
  - `assets/templates/icon_brief_request.j2`
- Edit these files to change style policy and brief-generation instructions.
- Optional: set `ICON_MAKER_TEMPLATE_DIR` to load templates from a different folder.

## Knobs to tune

- Foreground cleanup: `--foreground-alpha-threshold` (edge noise), `--foreground-opacity`
- Recolor to single solid color: `--foreground-color` or disable with `--no-recolor`
- Crop to content before resizing: `--crop-to-content` / `--no-crop-to-content` + `--crop-alpha-threshold`
- Background tile: `--canvas-size`, `--icon-size`, `--corner-radius`, `--bg-color`, `--bg-opacity`
