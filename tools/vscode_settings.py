#!/usr/bin/env python3

import json
from pathlib import Path


def main() -> None:
    folder = Path('.vscode')
    recommended = json.loads((folder / 'settings_recommended.json').read_text())
    path = folder / 'settings.json'
    try:
        current = json.loads(path.read_text())
    except Exception:
        current = {}
    with open(path, 'w') as f:
        json.dump({**current, **recommended}, f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()
