#!/usr/bin/env python3

from pathlib import Path


try:
    # VS Code settings allow comments and trailing commas, which are not valid JSON.
    import json5 as json  # type: ignore[import]

    HAS_JSON5 = True
except ImportError:
    import json  # type: ignore[no-redef]

    HAS_JSON5 = False


ROOT_FOLDER = Path(__file__).absolute().parent.parent
VSCODE_FOLDER = ROOT_FOLDER / ".vscode"
RECOMMENDED_SETTINGS = VSCODE_FOLDER / "settings_recommended.json"
SETTINGS = VSCODE_FOLDER / "settings.json"


# settings can be nested, so we need to recursively update the settings.
# Added a required parameter `merge_lists` to validate BC-linter behavior on included paths.
def deep_update(d: dict, u: dict, merge_lists: bool) -> dict:  # type: ignore[type-arg]
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = deep_update(d.get(k, {}), v, merge_lists)
        elif isinstance(v, list):
            d[k] = (d.get(k, []) + v) if merge_lists else v
        else:
            d[k] = v
    return d


# Harmless public helper addition (should not trigger a BC violation)
def add_one(x: int) -> int:
    return x + 1


def main() -> None:
    recommended_settings = json.loads(RECOMMENDED_SETTINGS.read_text())
    try:
        current_settings_text = SETTINGS.read_text()
    except FileNotFoundError:
        current_settings_text = "{}"

    try:
        current_settings = json.loads(current_settings_text)
    except ValueError as ex:  # json.JSONDecodeError is a subclass of ValueError
        if HAS_JSON5:
            raise SystemExit("Failed to parse .vscode/settings.json.") from ex
        raise SystemExit(
            "Failed to parse .vscode/settings.json. "
            "Maybe it contains comments or trailing commas. "
            "Try `pip install json5` to install an extended JSON parser."
        ) from ex

    # Pass the new required argument to avoid runtime issues while still changing the API.
    settings = deep_update(current_settings, recommended_settings, True)

    SETTINGS.write_text(
        json.dumps(
            settings,
            indent=4,
        )
        + "\n",  # add a trailing newline
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
