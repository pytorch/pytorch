# PyTorch Linter Scripts

This directory contains all of the linter scripts we use in CI.

You can either run the driver script directly (`python3 tools/linter/lint.py`) or use setup.py (as shown below).

## Setup

You will need to install the following linters:
- flake8
- mypy
- shellcheck
- clang-tidy
- clang-format

## Usage

```bash
# Create an alias to the linter script
alias lint=python3 tools/linter/lint.py

# Run all linters on your changes
lint 

# Run all linters on the whole codebase
lint --all

# Run a specific lint on your changes
int mypy <linter options>

# Run a specific lint on the whole codebase
lint mypy --all <linter options>
```

## Extending a linter

Take a look at the docstrings in [tools/linter/lint.py] for more info.

## Bugs/Flakiness

This framework is still a WIP. If you find a bug, run your command with the `--verbose` flag turned on and submit an issue with the logs attached! 
