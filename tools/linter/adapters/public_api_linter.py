# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "expecttest",
#   "hypothesis",
# ]
# ///
"""
This lint runs the public API checks from test/test_public_bindings.py against
only the torch submodules touched by the changed files, instead of walking
every submodule in the package. This keeps lintrunner fast: importing every
torch submodule takes seconds on every invocation, while checking a handful of
edited modules is much cheaper.

Checks that only make sense for a scoped set of submodules:
- every edited submodule can be imported
- edited public submodules follow the public API naming guidelines

Checks that evaluate the whole package (new re-exported callables, new
torch._C bindings) run for any change under torch/, since they only need
`import torch`, which is already paid for.

If `torch` is not importable, or the importable `torch` does not come from this
source tree (e.g. an installed wheel), no messages are emitted: the
authoritative check in that case is test/test_public_bindings.py itself, which
continues to run in CI.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import unittest
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import NamedTuple


LINTER_CODE = "PUBLIC_API_CHECKS"


class LintSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"


class LintMessage(NamedTuple):
    path: str | None
    line: int | None
    char: int | None
    code: str
    severity: LintSeverity
    name: str
    original: str | None
    replacement: str | None
    description: str | None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def path_to_module(relpath: str) -> str | None:
    """Map a repo-relative path to the torch submodule it defines.

    Returns None for paths outside the torch package and for files that do
    not live inside real packages (every directory up to torch/ must have an
    __init__.py), mirroring what pkgutil.walk_packages(torch.__path__)
    can discover.
    """
    parts = PurePosixPath(relpath).parts
    if len(parts) < 2 or parts[0] != "torch":
        return None
    name = parts[-1]
    if name == "__init__.py":
        module_parts = parts[:-1]
    elif name.endswith(".py"):
        module_parts = parts[:-1] + (name[: -len(".py")],)
    else:
        return None
    root = repo_root()
    for i in range(1, len(module_parts)):
        if not (root.joinpath(*module_parts[:i], "__init__.py")).is_file():
            return None
    return ".".join(module_parts)


def torch_matches_source_tree() -> bool:
    """True when the importable `torch` is the one in this checkout.

    Checks against an installed wheel would inspect the wheel's code rather
    than the tree being linted (and would false-positive on modules the
    wheel does not have yet), so we only run when the two match, e.g. in a
    development environment with an editable install.
    """
    try:
        import torch
    except ImportError:
        return False
    return (
        Path(torch.__file__).resolve()
        == (repo_root() / "torch" / "__init__.py").resolve()
    )


def load_test_module():
    spec = importlib.util.spec_from_file_location(
        "test_public_bindings", repo_root() / "test" / "test_public_bindings.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load test/test_public_bindings.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_message(path: str, name: str, description: str) -> LintMessage:
    return LintMessage(
        path=path,
        line=None,
        char=None,
        code=LINTER_CODE,
        severity=LintSeverity.ERROR,
        name=name,
        original=None,
        replacement=None,
        description=description,
    )


def run_checks(filenames: list[str]) -> list[LintMessage]:
    test_mod = load_test_module()
    case = test_mod.TestPublicBindings("test_no_new_reexport_callables")
    messages = []

    # Whole-package checks: cheap once `import torch` has happened, and they
    # are exactly the "trivial" failures that are annoying to discover only
    # when the test job runs.
    for method_name, name in (
        ("test_no_new_reexport_callables", "[reexported-callables]"),
        ("test_no_new_bindings", "[torch-c-bindings]"),
    ):
        try:
            getattr(case, method_name)()
        except AssertionError as e:
            messages.append(make_message("torch/__init__.py", name, str(e)))

    # Scoped checks: only the submodules the changed files define.
    module_to_path: dict[str, str] = {}
    for filename in filenames:
        if not (repo_root() / filename).is_file():
            # A deleted file no longer defines an importable module.
            continue
        modname = path_to_module(filename)
        if modname is not None:
            module_to_path[modname] = filename

    import_skipped = False
    names_skipped = False
    for modname in sorted(module_to_path):
        path = module_to_path[modname]
        if not import_skipped:
            try:
                case.test_modules_can_be_imported(modnames=[modname])
            except unittest.SkipTest:
                # Same platform skips as the test (e.g. Windows/macOS).
                import_skipped = True
            except AssertionError as e:
                messages.append(make_message(path, "[failed-import]", str(e)))
        if not names_skipped:
            try:
                case.test_correct_module_names(modnames=[modname])
            except unittest.SkipTest:
                names_skipped = True
            except AssertionError as e:
                messages.append(make_message(path, "[module-names]", str(e)))
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="check public API bindings of changed torch submodules",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )
    args = parser.parse_args()

    if not torch_matches_source_tree():
        print(
            "PUBLIC_API_CHECKS: skipped because a `torch` built from this source "
            "tree is not importable. Run this from a development environment "
            "(editable install); CI still checks these invariants via "
            "test/test_public_bindings.py.",
            file=sys.stderr,
        )
        return

    for message in run_checks(args.filenames):
        print(json.dumps(message._asdict()), flush=True)


if __name__ == "__main__":
    main()
