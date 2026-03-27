from __future__ import annotations

import argparse  # noqa: TC003
import logging

from cli.lib.common.cli_helper import register_targets, RichHelp, TargetSpec
from cli.lib.core.torchtitan.torchtitan_test import TorchtitanTestRunner
from cli.lib.core.vllm.vllm_test import VllmTestRunner
from cli.lib.pytorch.lint_test.lint_plans import LINT_PLANS
from cli.lib.pytorch.runner import PytorchTestRunner


logger = logging.getLogger(__name__)

# Maps targets to their argparse configuration and runner
# it adds new target to path python -m cli.run build external {target} with buildrunner
_TARGETS: dict[str, TargetSpec] = {
    "vllm": {
        "runner": VllmTestRunner,
        "help": "test vLLM with pytorch main",
    },
    "torchtitan": {
        "runner": TorchtitanTestRunner,
        "help": "test torchtitan with pytorch main",
    },
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
    _register_pytorch_commands(build_subparsers)


def _register_pytorch_commands(subparsers: argparse._SubParsersAction) -> None:
    _register_lint_commands(subparsers)


def _register_lint_commands(subparsers: argparse._SubParsersAction) -> None:
    available = "\n".join(
        f"  {gid:30} {plan.title}" for gid, plan in LINT_PLANS.items()
    )
    parser = subparsers.add_parser(
        "lint",
        help="Run lint test plans",
        description="Run PyTorch lint test.\n\nAvailable group IDs:\n" + available,
        formatter_class=RichHelp,
    )
    parser.add_argument(
        "--group-id",
        required=True,
        help="lint plan to run, e.g. 'lintrunner_noclang'",
    )
    parser.add_argument(
        "--input",
        metavar="KEY=VALUE",
        action="append",
        default=[],
        help="override plan inputs, e.g. --input changed_files='src/foo.py src/bar.py'",
    )
    parser.set_defaults(func=lambda args: PytorchTestRunner(args).run())
