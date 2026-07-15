import argparse
import logging

from cli.lib.common.cli_helper import register_targets, RichHelp, TargetSpec
from cli.lib.core.vllm.vllm_test import VllmTestRunner


logger = logging.getLogger(__name__)

# Maps targets to their argparse configuration and runner
# it adds new target to path python -m cli.run build external {target} with buildrunner
_TARGETS: dict[str, TargetSpec] = {
    "vllm": {
        "runner": VllmTestRunner,
        "help": "test vLLM with pytorch main",
    }
    # add yours ...
}


def common_args(parser: argparse.ArgumentParser) -> None:
    """
    Add common CLI arguments to the given parser.
    """
    parser.add_argument(
        "--shard-id",
        type=int,
        default=1,
        help="a shard id to run, e.g. '0,1,2,3'",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="a number of shards to run, e.g. '4'",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-tp",
        "--test-plan",
        type=str,
        help="a pre-defined test plan to run, e.g. 'basic_correctness_test'",
    )


def register_test_commands(subparsers: argparse._SubParsersAction) -> None:
    build_parser = subparsers.add_parser(
        "test",
        help="test related commands",
        formatter_class=RichHelp,
    )
    build_subparsers = build_parser.add_subparsers(dest="test_command", required=True)
    overview = "\n".join(
        f"  {name:12} {spec.get('help', '')}" for name, spec in _TARGETS.items()
    )
    external_parser = build_subparsers.add_parser(
        "external",
        help="Test external targets",
        description="Test third-party targets.\n\nAvailable targets:\n" + overview,
        formatter_class=RichHelp,
    )
    register_targets(external_parser, _TARGETS, common_args=common_args)
