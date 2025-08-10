import argparse
import logging
import textwrap
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol, Required, Type, TypedDict

from cli.lib.common.argparser import (
    register_target_commands_and_runner,
    RichHelp,
    TargetSpec,
)
from cli.lib.core.vllm import VllmBuildRunner


logger = logging.getLogger(__name__)

# tarfets dicts that maps target name to target spec
_TARGETS: Dict[str, TargetSpec] = {
    "vllm": {
        "runner": VllmBuildRunner,
        "help": "Build vLLM using docker buildx.",
    }
    # add yours ...
}


def register_build_commands(subparsers: argparse._SubParsersAction) -> None:
    build_parser = subparsers.add_parser(
        "build",
        help="Build related commands",
        formatter_class=RichHelp,
    )
    build_subparsers = build_parser.add_subparsers(dest="build_command", required=True)
    overview = "\n".join(
        f"  {name:12} {spec.get('help', '')}" for name, spec in _TARGETS.items()
    )
    external_parser = build_subparsers.add_parser(
        "external",
        help="Build external targets",
        description="Build third-party targets.\n\nAvailable targets:\n" + overview,
        formatter_class=RichHelp,
    )
    register_target_commands_and_runner(external_parser, _TARGETS)
