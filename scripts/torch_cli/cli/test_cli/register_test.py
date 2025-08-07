import logging

from cli.lib.core.vllm import VllmBuildRunner, VllmTestRunner


logger = logging.getLogger(__name__)


def register_test_external_commands(subparsers):
    """
    register build external commands, this is a subcommand of build
    """
    external_parser = subparsers.add_parser("external", help="Build external targets")
    external_parser.add_argument(
        "target", help="Name of the external target to build (e.g., vllm)"
    )
    external_parser.set_defaults(func=run_test_external)
    return external_parser


def register_test_commands(subparsers):
    """
    register test commands, this is a subcommand of torch_cli
    """
    test_subcommand_registry = [
        register_test_external_commands,
        # add more register_test_whatever_commands,
    ]

    test_parser = subparsers.add_parser("test", help="test related commands")
    test_parser.add_argument(
        "--test-name",
        type=str,
        required=False,
        help="Name of the test to run (applies to all test subcommands)",
    )
    test_subparsers = test_parser.add_subparsers(dest="test_command")
    for register_fn in test_subcommand_registry:
        sub = register_fn(test_subparsers)
        add_common_test_parser(sub)


def add_common_test_parser(parser):
    """
    Add common parser for all test commands
    """
    parser.add_argument(
        "--test-names",
        type=str,
        nargs="*",  # zero or more values (use "+" for one or more)
        required=False,
        help="Name(s) of the test(s) to run",
    )
    return parser


# Mappings to test external targets
# add new build external targets here
EXTERNAL_BUILD_TARGET_DISPATCH = {
    "vllm": lambda args: VllmTestRunner(config_path=args.config),
}


def run_test_external(args):
    target = args.target
    print(f"Running external build for target: {args.target}")
    print(args.config)
    if target not in EXTERNAL_BUILD_TARGET_DISPATCH:
        raise ValueError(f"Unknown build target: {target}")
    EXTERNAL_BUILD_TARGET_DISPATCH[target](args).prepare()
