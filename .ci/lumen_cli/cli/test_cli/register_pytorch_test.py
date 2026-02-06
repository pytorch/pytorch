"""
CLI registration for PyTorch test utilities.

This module registers PyTorch-specific test commands under the `test pytorch` subcommand.
"""

import argparse
import logging

from cli.lib.common.cli_helper import BaseRunner, register_targets, RichHelp, TargetSpec
from cli.lib.core.pytorch import PytorchTestEnvironment


logger = logging.getLogger(__name__)


class PytorchTestEnvRunner(BaseRunner):
    """
    CLI runner for PyTorch test environment configuration.

    This runner initializes and displays the computed environment variables
    based on the provided CLI flags or environment variables.
    """

    def run(self) -> None:
        """Initialize and display PyTorch test environment configuration."""
        env = PytorchTestEnvironment.from_args(self.args)

        # Export mode: output shell export statements only (all variables)
        if self.args.export:
            for key, value in sorted(env.get_updates().items()):
                # Escape single quotes in values for shell safety
                escaped_value = value.replace("'", "'\\''")
                print(f"export {key}='{escaped_value}'")
            return
        if self.args.verify_build_env:
            env.apply()
            env.verify_build_configuration()


        # Display mode: show detailed information
        print(f"Build Environment: {env.build_environment}")
        print(f"Test Config: {env.test_config}")
        print(f"Shard: {env.shard_number}/{env.num_test_shards}")
        print()

        print("Build Environment Properties:")
        print(f"  is_cuda: {env.is_cuda}")
        print(f"  is_rocm: {env.is_rocm}")
        print(f"  is_xpu: {env.is_xpu}")
        print(f"  is_asan: {env.is_asan}")
        print(f"  is_debug: {env.is_debug}")
        print()

        print("Test Config Properties:")
        print(f"  is_distributed_test: {env.is_distributed_test}")
        print(f"  is_inductor_test: {env.is_inductor_test}")
        print(f"  is_dynamo_test: {env.is_dynamo_test}")
        print()

        # Display environment variables by category
        categorized = env.get_categorized_updates()
        for category, vars_dict in categorized.items():
            if vars_dict:
                print(f"{category} ({len(vars_dict)}):")
                for key, value in sorted(vars_dict.items()):
                    print(f"  {key}={value}")
                print()


def add_pytorch_env_args(parser: argparse.ArgumentParser) -> None:
    """Add PyTorch test environment specific CLI arguments."""
    parser.add_argument(
        "--build-environment",
        type=str,
        dest="build_environment",
        help="CI build environment string (e.g., 'linux-focal-cuda12.1-py3.10')",
    )
    parser.add_argument(
        "--test-config",
        type=str,
        dest="test_config",
        help="Test configuration name (e.g., 'default', 'slow', 'inductor')",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        dest="shard_id",
        help="Current test shard number (1-indexed)",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        dest="num_shards",
        help="Total number of test shards",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Output shell export statements (use with eval)",
    )
    parser.add_argument(
        "--verify-build-env",
        action="store_true",
        help="Output shell export statements (use with eval)",
    )


# PyTorch test targets
_PYTORCH_TARGETS: dict[str, TargetSpec] = {
    "env": {
        "runner": PytorchTestEnvRunner,
        "help": "display PyTorch test environment configuration",
        "add_arguments": add_pytorch_env_args,
    }
}


def register_pytorch_test_commands(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """
    Register PyTorch test utilities under the test subcommand.

    Args:
        subparsers: The subparsers action from the parent 'test' parser.

    Returns:
        The created pytorch parser for further customization if needed.
    """
    overview = "\n".join(
        f"  {name:12} {spec.get('help', '')}" for name, spec in _PYTORCH_TARGETS.items()
    )
    pytorch_parser = subparsers.add_parser(
        "pytorch",
        help="PyTorch test utilities",
        description="PyTorch test utilities.\n\nAvailable targets:\n" + overview,
        formatter_class=RichHelp,
    )
    register_targets(pytorch_parser, _PYTORCH_TARGETS)
    return pytorch_parser
