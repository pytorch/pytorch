# Optional type checking with Mypy

Mypy is an optional static typechecker that works with Python 3.
To use it, install the following dependencies:

```bash
# Install dependencies
pip install mypy mypy-extensions

# Run type checker in the pytorch/ directory
mypy
```

Note that the minimum version of MyPy that is supported is 0.770


## What we're aiming for

1. Complete type annotations for the whole code base, and shipping those in a
   PEP 561 compatible manner (adding a `py.typed` file so the installed package
   supports typechecking.
2. Inline type annotations for all Python code, _except_ if there are too many
   overloads for functions/methods - in that case a stub file should be
   preferred (what's too many is a bit of a judgement call, I'd suggest two per
   function is a reasonable threshold).
3. Stub files for the extension modules (e.g. `torch._C`).
4. Good type annotation test coverage, by using `check_untyped_defs=True` for
   the test suite (or adding type annotations to tests).


## How to go about improving type annotations

_The tracking issue for this is
https://github.com/pytorch/pytorch/issues/16574_

Before starting, install MyPy (0.770 or newer), build PyTorch with `python
setup.py develop`, and run `mypy` in the root of the repo. This should give
output like:

```
Success: no issues found in 969 source files
```

In `mypy.ini` there's a long list of `ignore_missing_imports` and
`ignore_errors` for specific modules or files. If you remove one and re-run
`mypy`, then errors should appear. For example, deleting

```
[mypy-torch._C]
ignore_missing_imports = True
```

will show (currently):

```
...
torch/utils/data/_utils/signal_handling.py:39: error: Cannot find implementation or library stub for module named 'torch._C'  [import]
Found 14 errors in 14 files (checked 969 source files)
```

_Note that MyPy caching can be flaky, in particular removing
`ignore_missing_imports` has a caching bug
(https://github.com/python/mypy/issues/7777). If you don't see any errors
appear, try `rm -rf .mypy_cache` and try again._
