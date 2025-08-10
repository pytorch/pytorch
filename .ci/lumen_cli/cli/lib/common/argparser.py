"""
Argparse Utility helpers for CLI tasks.
"""

import argparse
from abc import ABC, abstractmethod
from typing import Any, Callable, Required, TypedDict


# Pretty help: keep newlines + show defaults
class RichHelp(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
):
    pass


class BaseRunner(ABC):
    def __init__(self, args: Any) -> None:
        self.args = args

    @abstractmethod
    def run(self, args: Any) -> None:
        """runs main logics, required"""


class TargetSpec(TypedDict, total=False):
    runner: Required[type[BaseRunner]]
    help: str
    description: str
    add_arguments: Callable[[argparse.ArgumentParser], None]


def register_target_commands_and_runner(
    parser: argparse.ArgumentParser, target_specs: dict[str, TargetSpec]
) -> None:
    targets = parser.add_subparsers(
        dest="target",
        required=True,
        metavar="{" + ",".join(target_specs.keys()) + "}",
    )

    for name, spec in target_specs.items():
        desc = (spec.get("description") or (spec["runner"].__doc__ or "")).strip()
        env = spec.get("env") or {}
        epilog = None
        if env:
            env_block = "\n".join(f"  {k:<22} {v}" for k, v in env.items())
            epilog = "Environment variables:\n" + env_block

        p = targets.add_parser(
            name,
            help=spec.get("help", ""),
            description=desc,
            epilog=epilog,
            formatter_class=RichHelp,
        )
        p.set_defaults(
            func=lambda args, _cls=spec["runner"]: _cls(args).run(),
            _runner_class=spec["runner"],
        )
