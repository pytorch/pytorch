# PyTorch Linter Scripts

This directory contains all of the linter scripts we use in CI.

You can either run the driver script directly (`python3 tools/linter/lint.py`) or use setup.py (as shown below).

## Usage

```bash
# Run all lint checks
python3 setup.py lint 

# Run all lint checks on changes
python3 setup.py lint --changed-only

# Run a specific lint
python3 setup.py lint mypy <linter options>

# Run a specific lint on changes
python3 setup.py lint mypy --changed-only <linter options>
```

## Installation

```
python3 setup.py lint --install
```
