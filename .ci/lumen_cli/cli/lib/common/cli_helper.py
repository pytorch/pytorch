"""
Argparse Utility helpers for CLI tasks.
"""

import argparse
from typing import Callable, Protocol, Required, TypedDict


# Pretty help: keep newlines + show defaults
class RichHelp(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
):
    pass


# Any class that can be constructed with (args) and has .run()
class RunnerLike(Protocol):
    def run(self) -> None: ...


class TargetSpec(TypedDict, total=False):
    runner: Required[type[RunnerLike]]
    help: str
    description: str
    add_arguments: Callable[[argparse.ArgumentParser], None]


def register_target_commands_and_runner(
    parser: argparse.ArgumentParser,
    target_specs: dict[str, TargetSpec],
    common_args: Callable[[argparse.ArgumentParser], None] = lambda _: None,
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

        if "add_arguments" in spec and callable(spec["add_arguments"]):
            spec["add_arguments"](p)
        if common_args:
            common_args(p)
