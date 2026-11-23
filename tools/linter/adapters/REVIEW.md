# Reviewing test changes under `tools/linter/adapters`

## Requirements for the test linter

- Conditions:
  - `test_linter.py`
  - **Or** `test_linter_allowlist.json`
- Rules:
  - **`test_linter_allowlist.json` must only shrink**: only PRs whose
    explicit purpose is updating this file (e.g. regeneration) may add
    entries; adding entries in any other PR is not allowed.
