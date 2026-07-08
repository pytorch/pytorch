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

import os
import subprocess
import sys


def _check_build_requires() -> None:
    """Verify build-system.requires is satisfied before a no-isolation build.

    The build commands below pass --no-build-isolation / --no-isolation, which
    means pip does not enforce the build-system.requires pins; a missing or
    too-old build dependency (e.g. scikit-build-core) then surfaces as a deep
    BackendUnavailable traceback. Check the pins up front and fail with an
    actionable message instead. Best-effort: if the tooling needed to run the
    check is itself absent, skip rather than block the build.
    """
    try:
        import tomllib
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError:
            return
    try:
        from importlib.metadata import (
            PackageNotFoundError,
            version as installed_version,
        )

        from packaging.requirements import Requirement
    except ModuleNotFoundError:
        return

    pyproject = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "pyproject.toml"
    )
    with open(pyproject, "rb") as f:
        requires = tomllib.load(f).get("build-system", {}).get("requires", [])

    problems = []
    for spec in requires:
        req = Requirement(spec)
        if req.marker is not None and not req.marker.evaluate():
            continue
        try:
            found = installed_version(req.name)
        except PackageNotFoundError:
            problems.append(f"{spec} (not installed)")
            continue
        if not req.specifier.contains(found, prereleases=True):
            problems.append(f"{spec} (found {found})")

    if problems:
        listing = "\n  ".join(problems)
        sys.exit(
            "ERROR: build dependencies are missing or too old for a "
            "--no-build-isolation build:\n  "
            f"{listing}\n\n"
            "Install them first:\n"
            "  python -m pip install -r requirements-build.txt"
        )


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
    if any(flag in cmd for flag in ("--no-build-isolation", "--no-isolation")):
        _check_build_requires()
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
