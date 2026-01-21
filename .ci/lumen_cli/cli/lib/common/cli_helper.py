"""
Cli Argparser Utility helpers for CLI tasks.

"""

import argparse
from abc import ABC, abstractmethod


try:
    from collections.abc import Callable  # Python 3.11+
    from typing import Any, Required, TypedDict
except ImportError:
    from collections.abc import Callable
    from typing import Any, TypedDict

    from typing_extensions import Required  # Fallback for Python <3.11


class BaseRunner(ABC):
    def __init__(self, args: Any) -> None:
        self.args = args

    @abstractmethod
    def run(self) -> None:
        """runs main logics, required"""


# Pretty help: keep newlines + show defaults
class RichHelp(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
):
    pass


class TargetSpec(TypedDict, total=False):
    """CLI subcommand specification with bA."""

    runner: Required[type[BaseRunner]]
    help: str
    description: str
    add_arguments: Callable[[argparse.ArgumentParser], None]


def register_targets(
    parser: argparse.ArgumentParser,
    target_specs: dict[str, TargetSpec],
    common_args: Callable[[argparse.ArgumentParser], None] = lambda _: None,
) -> None:
    """Register target subcommands."""
    targets = parser.add_subparsers(
        dest="target",
        required=True,
        metavar="{" + ",".join(target_specs.keys()) + "}",
    )

    for name, spec in target_specs.items():
        desc = spec.get("description") or spec["runner"].__doc__ or ""

        p = targets.add_parser(
            name,
            help=spec.get("help", ""),
            description=desc.strip(),
            formatter_class=RichHelp,
        )
        p.set_defaults(
            func=lambda args, cls=spec["runner"]: cls(args).run(),
            _runner_class=spec["runner"],
        )
        if "add_arguments" in spec and callable(spec["add_arguments"]):
            spec["add_arguments"](p)
        if common_args:
            common_args(p)
