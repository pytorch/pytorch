# PyTorch Linter Scripts

This directory contains all of the linter scripts we use in CI.

## Usage

```bash
# Run all lint checks
python3 tools/linter/lint.py

# Run all lint checks on changes
python3 tools/linter/lint.py --changed-only

# Run a specific lint
python3 tools/linter/lint.py clang-tidy -- <linter options>

# Run a specific lint on changes
python3 tools/linter/lint.py clang-tidy --changed-only -- <linter options>
```

## Installation

We provide custom binaries for `clang-format` and `clang-tidy`. You can install them by running:

```bash
python3 tools/linter/install clang-tidy clang-format
```
