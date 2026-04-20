from __future__ import annotations

import argparse  # noqa: TC003
import logging

from cli.lib.common.cli_helper import register_targets, RichHelp, TargetSpec
from cli.lib.core.pytorch.lib import PYTORCH_TEST_LIBRARY
from cli.lib.core.pytorch.pytorch_test import PytorchTestRunner
from cli.lib.core.torchtitan.torchtitan_test import TorchtitanTestRunner
from cli.lib.core.vllm.vllm_test import VllmTestRunner


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


def _register_pytorch_core_commands(subparsers: argparse._SubParsersAction) -> None:
    available = "\n".join(
        f"  {gid:40} {plan.title}" for gid, plan in PYTORCH_TEST_LIBRARY.items()
    )
    parser = subparsers.add_parser(
        "pytorch-core",
        help="Run PyTorch core tests",
        description="Run PyTorch CI test plans.\n\nAvailable group IDs:\n" + available,
        formatter_class=RichHelp,
    )

    # Mutually exclusive: either name the plan directly, or let TEST_CONFIG drive it.
    dispatch = parser.add_mutually_exclusive_group(required=True)
    dispatch.add_argument(
        "--group-id",
        metavar="GROUP_ID",
        help="run a specific plan by name, e.g. 'pytorch_cpuonly'",
    )
    dispatch.add_argument(
        "--test-config",
        metavar="TEST_CONFIG",
        help="resolve plan from TEST_CONFIG (+ --build-env), replacing test.sh dispatch",
    )

    parser.add_argument(
        "--build-env",
        required=True,
        metavar="BUILD_ENVIRONMENT",
        help="build environment string for plan resolution and env_vars",
    )
    parser.add_argument(
        "--test-id",
        default=None,
        metavar="TEST_ID",
        help="run a single step within the group (for reproduction)",
    )
    parser.add_argument(
        "--cmd",
        default=None,
        metavar="CMD",
        help="replay setup context of --test-id but run this command instead "
        "(e.g. a specific pytest line)",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=1,
        help="current shard index (1-based)",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="total number of shards",
    )
    parser.add_argument(
        "--filter",
        metavar="KEY=VALUE",
        action="append",
        default=[],
        help="filter steps by params, e.g. --filter mode=training --filter dtype=float16",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        default=False,
        help="strip artifact-upload flags (e.g. --upload-artifacts-while-running) from test invocations",
    )
    parser.set_defaults(func=lambda args: PytorchTestRunner(args).run())


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
    _register_pytorch_core_commands(build_subparsers)
