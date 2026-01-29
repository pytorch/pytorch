import argparse
import logging

from cli.lib.common.cli_helper import register_targets, RichHelp, TargetSpec
from cli.lib.core.pytorch.runners.python_tests import (
    PythonLegacyJITTestRunner,
    PythonSmokeTestRunner,
    PythonTestRunner,
    QuantizationTestRunner,
    WithoutNumpyTestRunner,
)
from cli.lib.core.vllm.vllm_test import VllmTestRunner


logger = logging.getLogger(__name__)

# ============================================================================
# External Targets (third-party libraries tested against PyTorch)
# ============================================================================
_EXTERNAL_TARGETS: dict[str, TargetSpec] = {
    "vllm": {
        "runner": VllmTestRunner,
        "help": "test vLLM with pytorch main",
    }
    # add more external targets here...
}

# ============================================================================
# Internal Targets (PyTorch's own test suite)
# ============================================================================
_INTERNAL_TARGETS: dict[str, TargetSpec] = {
    "python": {
        "runner": PythonTestRunner,
        "help": "run Python test shards (excludes JIT/distributed/quantization)",
        "description": """
Run PyTorch Python tests with optional sharding.

This runner executes the standard Python test suite, excluding JIT executor,
distributed, and quantization tests (which have their own runners).

Equivalent to test_python_shard() in test.sh.

Examples:
    # Run shard 1 of 4
    python -m cli.run test internal python --shard-id 1 --num-shards 4

    # Run all tests without sharding
    python -m cli.run test internal python --shard-id 1 --num-shards 1
""",
    },
    "python-legacy-jit": {
        "runner": PythonLegacyJITTestRunner,
        "help": "run legacy JIT tests (test_jit_legacy, test_jit_fuser_legacy)",
    },
    "python-smoke": {
        "runner": PythonSmokeTestRunner,
        "help": "run smoke tests for H100/B200",
    },
    "quantization": {
        "runner": QuantizationTestRunner,
        "help": "run quantization tests",
    },
    "without-numpy": {
        "runner": WithoutNumpyTestRunner,
        "help": "test that torch works without numpy installed",
    },
    # More internal targets can be added here:
    # "inductor": { ... },
    # "dynamo": { ... },
    # "distributed": { ... },
}


def external_common_args(parser: argparse.ArgumentParser) -> None:
    """
    Add common CLI arguments for external test targets.
    """
    parser.add_argument(
        "--shard-id",
        type=int,
        default=1,
        help="a shard id to run, e.g. '1'",
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


def internal_common_args(parser: argparse.ArgumentParser) -> None:
    """
    Add common CLI arguments for internal PyTorch test targets.
    """
    parser.add_argument(
        "--shard-id",
        type=int,
        default=1,
        help="shard ID to run (1-indexed)",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="total number of shards",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="enable verbose output",
    )
    parser.add_argument(
        "--no-upload-artifacts",
        action="store_false",
        dest="upload_artifacts",
        help="disable artifact upload during test run",
    )
    parser.add_argument(
        "--include-tests",
        type=str,
        nargs="+",
        help="specific tests to include (space-separated)",
    )


def register_test_commands(subparsers: argparse._SubParsersAction) -> None:
    """
    Register all test-related CLI commands.

    Creates the following command structure:
        cli.run test external <target>  - Test third-party libraries (vLLM, etc.)
        cli.run test internal <target>  - Run PyTorch's own test suite
    """
    test_parser = subparsers.add_parser(
        "test",
        help="test related commands",
        formatter_class=RichHelp,
    )
    test_subparsers = test_parser.add_subparsers(dest="test_command", required=True)

    # ========================================================================
    # External targets (third-party libraries)
    # ========================================================================
    external_overview = "\n".join(
        f"  {name:12} {spec.get('help', '')}"
        for name, spec in _EXTERNAL_TARGETS.items()
    )
    external_parser = test_subparsers.add_parser(
        "external",
        help="Test external/third-party targets",
        description="Test third-party targets.\n\nAvailable targets:\n"
        + external_overview,
        formatter_class=RichHelp,
    )
    register_targets(
        external_parser, _EXTERNAL_TARGETS, common_args=external_common_args
    )

    # ========================================================================
    # Internal targets (PyTorch's own tests)
    # ========================================================================
    internal_overview = "\n".join(
        f"  {name:20} {spec.get('help', '')}"
        for name, spec in _INTERNAL_TARGETS.items()
    )
    internal_parser = test_subparsers.add_parser(
        "internal",
        help="Run PyTorch internal tests",
        description="Run PyTorch's own test suite.\n\nAvailable targets:\n"
        + internal_overview,
        formatter_class=RichHelp,
    )
    register_targets(
        internal_parser, _INTERNAL_TARGETS, common_args=internal_common_args
    )
