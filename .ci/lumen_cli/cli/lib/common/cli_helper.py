"""
Cli Argparser Utility helpers for CLI tasks.

"""

import argparse
from abc import ABC, abstractmethod


try:
    from typing import Any, Callable, Required, TypedDict  # Python 3.11+
except ImportError:
    from typing import Any, Callable, TypedDict

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
    # Registry entry for a CLI subcommand.
    #
    # Keys:
    #   runner        – class implementing BaseRunner; will be constructed
    #                   with the parsed argparse Namespace.
    #   help          – short help text for the subparser (shows in parent help).
    #   description   – long help text (falls back to runner.__doc__ if missing).
    #   add_arguments – optional function to add subparser-specific CLI args.

    runner: Required[type[BaseRunner]]
    help: str
    description: str
    add_arguments: Callable[[argparse.ArgumentParser], None]


def register_target_commands_and_runner(
    parser: argparse.ArgumentParser,
    target_specs: dict[str, TargetSpec],
    common_args: Callable[[argparse.ArgumentParser], None] = lambda _: None,
    placeholder_name: str = "target",
) -> None:
    """
    Given an argparse parser and a mapping of target names → TargetSpec,
    register each target as a subcommand and wire it to its runner class.
    - Creates a subparser for each target name.
    - description: defaults to TargetSpec['description'] or runner.__doc__.
      - description has higher priority than runner.__doc__.
    - add_arguments[Optional]: add target-specific CLI args, if 'add_arguments' exists
    - common_args[Optional]: add shared CLI args to each target parser.
    - Sets parser defaults:
        func:         lambda that constructs the runner with parsed args
                      and calls its .run().
        _runner_class: stored runner class for introspection/testing.
    """
    targets = parser.add_subparsers(
        dest=placeholder_name,
        required=True,
        metavar="{" + ",".join(target_specs.keys()) + "}",
    )

    for name, spec in target_specs.items():
        desc = (spec.get("description") or (spec["runner"].__doc__ or "")).strip()

        p = targets.add_parser(
            name,
            help=spec.get("help", ""),
            description=desc,
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
