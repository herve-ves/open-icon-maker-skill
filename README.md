# Open Icon Maker (Codex Skill)

| Americano (hot) | Americano (takeaway) | Grilled seafood | Pizza (person) | Pizza (profile) |
| --- | --- | --- | --- | --- |
| <img src="open-icon-maker/references/americano_hot_final.png" width="120" height="120" /> | <img src="open-icon-maker/references/americano_takeaway_final.png" width="120" height="120" /> | <img src="open-icon-maker/references/grilled_seafood_final.png" width="120" height="120" /> | <img src="open-icon-maker/references/pizza_eating_person_final.png" width="120" height="120" /> | <img src="open-icon-maker/references/pizza_eating_profile_final.png" width="120" height="120" /> |

## Install

1. Find your Codex home directory:
   - If `CODEX_HOME` is set, use that.
   - Otherwise, the default is `~/.codex`.
2. Copy the `open-icon-maker/` folder into `$CODEX_HOME/skills/` so you end up with:
   - `$CODEX_HOME/skills/open-icon-maker/SKILL.md`

### macOS / Linux

```sh
CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
mkdir -p "$CODEX_HOME/skills"
cp -R open-icon-maker "$CODEX_HOME/skills/"
```

If you are currently inside the `open-icon-maker/` folder:

```sh
CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
mkdir -p "$CODEX_HOME/skills/open-icon-maker"
cp -R . "$CODEX_HOME/skills/open-icon-maker/"
```

### Windows (PowerShell)

```powershell
$codexHome = if ($env:CODEX_HOME) { $env:CODEX_HOME } else { "$HOME\.codex" }
New-Item -ItemType Directory -Force -Path "$codexHome\skills" | Out-Null
Copy-Item -Recurse -Force .\open-icon-maker "$codexHome\skills\"
```

If you are currently inside the `open-icon-maker/` folder:

```powershell
$codexHome = if ($env:CODEX_HOME) { $env:CODEX_HOME } else { "$HOME\.codex" }
New-Item -ItemType Directory -Force -Path "$codexHome\skills\open-icon-maker" | Out-Null
Copy-Item -Recurse -Force .\* "$codexHome\skills\open-icon-maker\"
```

## Verify

- Restart your Codex session (or reload skills if your Codex build supports it).
- The installed folder should exist at `$CODEX_HOME/skills/open-icon-maker/` and contain `SKILL.md`.

## Update / Uninstall

- Update: replace `$CODEX_HOME/skills/open-icon-maker/` with the updated `open-icon-maker/` folder.
- Uninstall: delete `$CODEX_HOME/skills/open-icon-maker/`.

## Notes

- The skill includes a runnable Python script at `scripts/icon_maker.py`. Dependencies are not bundled; use `uv run scripts/icon_maker.py --help` (if you have `uv`) or install packages listed in `SKILL.md`.
