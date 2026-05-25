"""Shim that forwards legacy setup.py commands to their modern equivalents.

PyTorch has migrated from setuptools to scikit-build-core. This script
intercepts common setup.py invocations and translates them to the
corresponding pip/build/spin commands.

Usage examples (all produce the same result as before):
    python setup_.py install          -> pip install . -v --no-build-isolation
    python setup_.py develop          -> pip install -e . -v --no-build-isolation
    python setup_.py bdist_wheel      -> python -m build --wheel --no-isolation
    python setup_.py clean            -> spin clean
    python setup_.py build            -> pip install -e . -v --no-build-isolation
"""

import subprocess
import sys


_PIP_INSTALL = [
    sys.executable,
    "-m",
    "pip",
    "install",
    ".",
    "-v",
    "--no-build-isolation",
]
_PIP_INSTALL_E = [
    sys.executable,
    "-m",
    "pip",
    "install",
    "-e",
    ".",
    "-v",
    "--no-build-isolation",
]

COMMANDS: dict[str, list[str]] = {
    "install": _PIP_INSTALL,
    "develop": _PIP_INSTALL_E,
    "build": _PIP_INSTALL_E,
    "bdist_wheel": [sys.executable, "-m", "build", "--wheel", "--no-isolation"],
    "clean": [sys.executable, "-m", "spin", "clean"],
}


def main() -> None:
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print(__doc__.strip())
        sys.exit(0)

    command = args[0]
    cmd = COMMANDS.get(command)
    if cmd is None:
        print(
            f"Unknown command: {command}\n\n"
            f"Supported commands: {', '.join(COMMANDS)}\n"
            "See 'python setup_.py --help' for details.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"NOTE: 'python setup.py {command}' is no longer supported.\n"
        f"Forwarding to: {' '.join(cmd)}\n",
        file=sys.stderr,
    )
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
