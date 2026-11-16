# PyTorch is built with scikit-build-core through the PEP 517 interface
# declared in pyproject.toml; pip and `python -m build` never run this
# file. It exists only for direct `python setup.py <command>` calls:
# install and develop are forwarded to pip, everything else fails with
# instructions (#180248).
#
# Deprecation schedule:
#   PyTorch 2.14-2.15: install and develop forward to pip
#   PyTorch 2.16-2.17: all commands fail with instructions
#   PyTorch 2.18:      this file is removed
#
# See https://github.com/pytorch/pytorch/issues/152276 for background.

import shlex
import subprocess
import sys
import warnings


PIP_INSTALL = [sys.executable, "-m", "pip", "install", "-v", "--no-build-isolation"]
FORWARDS = {
    "install": [*PIP_INSTALL, "."],
    "develop": [*PIP_INSTALL, "-e", "."],
}

DEPRECATION_NOTICE = """\
`python setup.py {command}` is deprecated: PyTorch is built with
scikit-build-core via the standard PEP 517 interface (pyproject.toml),
and setup.py is no longer part of the build.

Deprecation schedule:
  PyTorch 2.14-2.15: install and develop forward to pip
                     (current behavior); other commands fail
  PyTorch 2.16-2.17: all commands fail with instructions
  PyTorch 2.18:      setup.py is removed

Replacement commands:
  install:           spin install  (or: pip install --no-build-isolation -v .)
  editable install:  spin develop  (or: pip install --no-build-isolation -v -e .)
  wheel:             python -m build --wheel --no-isolation
  sdist:             python -m build --sdist

Build customization through environment variables (DEBUG=1, USE_CUDA=0,
MAX_JOBS=..., etc.) works unchanged with the replacement commands. See
the "From Source" section of README.md for details and
https://github.com/pytorch/pytorch/issues/152276 for background.
"""


def main() -> int:
    command = sys.argv[1] if len(sys.argv) > 1 else "<command>"
    notice = DEPRECATION_NOTICE.format(command=command)
    replacement = FORWARDS.get(command)
    if replacement is None:
        raise SystemExit(f"error: {notice}")
    warnings.warn(notice, DeprecationWarning)
    if len(sys.argv) > 2:
        warnings.warn(f"ignoring extra arguments: {shlex.join(sys.argv[2:])}")
    print(f"Forwarding to `{shlex.join(replacement)}`.", file=sys.stderr)
    return subprocess.run(replacement).returncode


if __name__ == "__main__":
    sys.exit(main())
