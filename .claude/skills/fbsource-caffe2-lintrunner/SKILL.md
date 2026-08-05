---
name: fbsource-caffe2-lintrunner
description: Run PyTorch (caffe2) lintrunner linters locally in fbsource to reproduce the OSS PR lint jobs (e.g. Lint / lintrunner-noclang-partial). Use when lintrunner is not installed, when `uv`-based linters fail to download CPython, or when running CLANGFORMAT, RUFF, PYFMT, FLAKE8, CODESPELL, or PYREFLY on changed caffe2 files from a devserver.
---

# Run caffe2 lintrunner linters locally in fbsource

Reproduce the PyTorch OSS PR lint jobs (e.g. `Lint / lintrunner-noclang-partial`) against
local changes in `fbcode/caffe2` (and the `xplat/caffe2` mirror) from an fbsource
devserver, where `lintrunner` is not preinstalled and network egress is restricted.

## When to use this skill

Use this skill when:
- Asked to run the linters from a PyTorch PR / a specific lint CI job locally.
- `lintrunner` is not on `PATH` and there is no `.venv`.
- A `uv`-based linter fails with `Failed to download ... cpython ... github.com` (blocked egress).
- Running `CLANGFORMAT`, `RUFF`, `PYFMT`, `FLAKE8`, `CODESPELL`, or `PYREFLY` on changed
  `caffe2` files.
- User mentions "lintrunner", "lintrunner-noclang", "noclang-partial", "docstring linter",
  or "reproduce the lint job".

## Key facts

- Run all `lintrunner` commands from `fbcode/caffe2` (that is where `.lintrunner.toml`,
  `pyproject.toml`, and `pyrefly.toml` live). `-I/-X` and config paths are relative to CWD.
- fbsource is Sapling/EdenFS, not git. `lintrunner` still works when you pass explicit
  file paths; the `fatal: not a git repository` messages from some adapters are non-fatal.
- Network: AWS S3 (`oss-clang-format.s3.us-east-2.amazonaws.com`) is reachable, so
  `clang-format` / `clang-tidy` / `actionlint` download fine. `github.com` direct downloads
  are blocked, so `uv run --python 3.10 ...` (used by the Python linters) fails. Work around
  it by running each Python linter's adapter script directly with the venv Python and
  installing the adapter's pinned deps from the internal PyPI mirror (plain `pip` works).
- `xplat/caffe2` is a byte-identical mirror of `fbcode/caffe2`; `.lintrunner.toml` only lives
  under `fbcode/caffe2`. Lint the fbcode copy (maps to OSS paths) and mirror any fix into xplat.
- An adapter prints LintMessage JSON per finding; no JSON output == clean.

## Step 1: Environment setup

```bash
# venv (system python3 is 3.12; matches pyrefly.toml python-version = "3.12")
python3 -m venv /tmp/lr_venv
/tmp/lr_venv/bin/python -m pip install --upgrade pip lintrunner

# Download the S3-hosted binaries into fbcode/caffe2/.lintbin
# (full `lintrunner init` also works; its uv-based init steps fail harmlessly)
cd /data/users/adelesun/fbsource/fbcode/caffe2
python3 tools/linter/adapters/s3_init.py \
    --config-json=tools/linter/adapters/s3_init_config.json \
    --linter=clang-format --dry-run=0 --output-dir=.lintbin --output-name=clang-format

# Put lintrunner on PATH so the LINTRUNNER_VERSION adapter (calls bare `lintrunner -V`) passes
export PATH="/tmp/lr_venv/bin:$PATH"
```

Cleanup when done: `rm -rf /tmp/lr_venv` and `fbcode/caffe2/.lintbin`.

## Step 2: Offline linters via lintrunner (no uv, no build)

Run from `fbcode/caffe2` with explicit file paths. Each linter applies its own
include/exclude patterns, so a mixed C++/py/md file list is fine.

Faithful "noclang" run = everything except the clang linters (need a build) and the
uv-based linters (run separately in Step 3):

```bash
cd /data/users/adelesun/fbsource/fbcode/caffe2
export PATH="/tmp/lr_venv/bin:$PATH"

SKIP="CLANGFORMAT,CLANGTIDY,CLANGTIDY_EXECUTORCH_COMPATIBILITY,\
FLAKE8,PYREFLY,NATIVEFUNCTIONS,GHA,CMAKE,SHELLCHECK,TEST_HAS_MAIN,\
WORKFLOWSYNC,NO_WORKFLOWS_ON_FORK,CODESPELL,PYFMT,PYPROJECT,\
CMAKE_MINIMUM_REQUIRED,RUFF"

lintrunner --skip "$SKIP" <files...>

# CLANGFORMAT is offline too (uses .lintbin/clang-format); run it explicitly
lintrunner --take CLANGFORMAT <cpp/h files...>
```

This covers `NEWLINE`, `TABS`, `SPACES`, `INCLUDE`, `TYPEIGNORE`, `NOQA`, `RAWTHROW`,
`COPYRIGHT`, `C10_UNUSED`, `C10_NODISCARD`, `RAWCUDA`, `RAWCUDADEVICE`, `CALL_ONCE`,
`ONCE_FLAG`, `ATEN_CPU_GPU_AGNOSTIC`, `PYBIND11_INCLUDE`, `PYBIND11_SPECIALIZATION`,
`ERROR_PRONE_ISINSTANCE`, `SET_LINTER`, `IMPORT_LINTER`, `DOCSTRING_LINTER`,
`META_NO_CREATE_UNBACKED`, `SYMPY_MINMAX`, and other offline grep/pattern linters.

Notes:
- `DOCSTRING_LINTER`: fails on an undocumented `class` >100 lines or `def` >80 lines that is
  not listed in `tools/linter/adapters/docstring_linter-grandfather.json`. Fix by adding a
  docstring (a documented block is exempt regardless of size).
- Filter noise in output with:
  `2>&1 | grep -vE 'INFO lintrunner|not a git repo|Stopping at filesystem|hint:|sl root|sl help'`

## Step 3: Python linters via their adapter scripts (uv bypassed)

Run each adapter `.py` directly with the venv Python and install its pinned deps (from the
adapter's PEP-723 `# dependencies` block) from the internal mirror.

```bash
cd /data/users/adelesun/fbsource/fbcode/caffe2
PY=/tmp/lr_venv/bin/python
PYFILES="<changed .py files...>"
ALLFILES="<all changed files incl C++/md...>"

# RUFF (pinned 0.14.4)
/tmp/lr_venv/bin/pip install 'ruff==0.14.4'
$PY tools/linter/adapters/ruff_linter.py --config=pyproject.toml --show-disable -- $PYFILES

# PYFMT (ruff-format + usort + isort); ruff==0.14.4 already installed
/tmp/lr_venv/bin/pip install 'usort==1.1.3' 'isort==6.0.1'
$PY tools/linter/adapters/pyfmt_linter.py $PYFILES

# FLAKE8 (7.3.0 + plugins). flake8-logging-format imports pkg_resources, removed in
# setuptools>=81, so pin setuptools<81.
/tmp/lr_venv/bin/pip install \
  'flake8==7.3.0' 'flake8-bugbear==24.12.12' 'flake8-comprehensions==3.16.0' \
  'flake8-executable==2.1.3' 'flake8-logging-format==2024.24.12' 'flake8-pyi==25.5.0' \
  'flake8-simplify==0.30.0' 'mccabe==0.7.0' 'pycodestyle==2.14.0' 'pyflakes==3.4.0' \
  'setuptools<81'
$PY tools/linter/adapters/flake8_linter.py $PYFILES

# CODESPELL (2.4.1) - runs on all file types
/tmp/lr_venv/bin/pip install 'codespell[toml]==2.4.1'
$PY tools/linter/adapters/codespell_linter.py $ALLFILES
```

Always pin to the adapter's PEP-723 `# dependencies` block; a version mismatch (e.g. a newer
ruff) can change results.

### Auto-fixing offline (offline equivalent of `lintrunner -a`)

`uv`-based `lintrunner -a` is blocked, but the pinned tools auto-fix in place. From
`fbcode/caffe2`:

```bash
/tmp/lr_venv/bin/ruff check --config=pyproject.toml --fix $PYFILES   # RUFF safe fixes (e.g. F401)
/tmp/lr_venv/bin/usort format $PYFILES                               # import ordering (PYFMT step 1)
/tmp/lr_venv/bin/ruff format --config=pyproject.toml $PYFILES        # formatting (PYFMT step 2)
```

Together these reproduce the RUFF + PYFMT autofixes that `lintrunner -a` would apply. `arc f`
(which runs automatically on `sl amend` / `sl commit`) is ruff-compatible for caffe2, so a file
that is clean under these stays clean after amend/commit.

### Common findings and fixes

- `RUFF S101` ("use of `assert`"): `S101` is enabled repo-wide in `pyproject.toml`'s ruff
  `select` with no test-directory exemption, so a bare `assert` is flagged even in test files
  (e.g. under `torch/_inductor/tests/`). For a deliberate invariant assert, append
  `# noqa: S101` (a short assert plus the noqa still fits under the B950 line limit).

## Step 4: PYREFLY (type checker)

```bash
cd /data/users/adelesun/fbsource/fbcode/caffe2
/tmp/lr_venv/bin/pip install \
  'pyrefly==0.58.0' 'numpy==2.1.0' 'expecttest==0.3.0' 'sympy==1.13.3' \
  'types-requests==2.27.25' 'types-pyyaml==6.0.2' 'types-tabulate==0.8.8' \
  'types-protobuf==5.29.1.20250403' 'types-setuptools==79.0.0.20250422' \
  'types-jinja2==2.11.9' 'types-colorama==0.4.6' 'filelock==3.18.0' \
  'junitparser==2.1.1' 'rich==14.1.0'
# Optional: 'spmd_types==0.2.1' (pulls in a PyPI torch + CUDA wheels as deps)

/tmp/lr_venv/bin/pyrefly check --config pyrefly.toml --output-format=json <changed .py files...>
```

Caveat: without a built PyTorch, the local `torch` source lacks generated `_C` stubs
(`_VariableFunctions.pyi`, etc.), so pyrefly reports many baseline `missing-attribute` errors
on `torch.*` builtins (e.g. `torch.rand`, `torch._scaled_mm`). These are environment noise,
not code defects. To isolate your change, attribute errors to the lines your commit added
(parse the added-line ranges from `sl diff -c <rev> -g <file>` hunks and intersect with each
error's line number). A clean-baseline CI-fidelity run requires a built torch so the
generated stubs exist.

## Linters that can NOT run locally

- `CLANGTIDY`, `CLANGTIDY_EXECUTORCH_COMPATIBILITY`: need `compile_commands.json` from a
  CUDA/ROCm build (`--build_dir=./build`). Not available in a plain checkout.
- `PYREFLY` at CI fidelity: needs a built torch (generated `.pyi` stubs) for a clean baseline.
- uv-based non-`.py` linters (`GHA`, `ACTIONLINT`, `CMAKE`, `SHELLCHECK`, `NATIVEFUNCTIONS`,
  `WORKFLOWSYNC`, `PYPROJECT`, `CMAKE_MINIMUM_REQUIRED`): uv is blocked, and they do not apply
  to C++/py/md source changes anyway, so they are no-ops for typical caffe2 diffs.

## Meta-native alternatives (do apply here)

```bash
arc f <files...>          # format
arc lint -a <files...>    # lint + autofix (BLACK, import sort, autodeps). Run before amend.
```
