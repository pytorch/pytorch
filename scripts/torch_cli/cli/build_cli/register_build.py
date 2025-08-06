import logging

from cli.lib.vllm import build_vllm


logger = logging.getLogger(__name__)


def register_build_commands(subparsers):
    """
    register build commands, this is a subcommand of torch_cli
    """
    build_parser = subparsers.add_parser("build", help="Build related commands")
    build_subparsers = build_parser.add_subparsers(dest="build_command")

    register_build_external_commands(build_subparsers)


def register_build_external_commands(subparsers):
    """
    register build external commands, this is a subcommand of build
    """
    external_parser = subparsers.add_parser("external", help="Build external targets")
    external_parser.add_argument(
        "target", help="Name of the external target to build (e.g., vllm)"
    )
    external_parser.set_defaults(func=run_build_external)


# Mappings to build external targets
# add new build external targets here
EXTERNAL_BUILD_TARGET_DISPATCH = {
    "vllm": lambda args: build_vllm(config_path=args.config),
}


def run_build_external(args):
    target = args.target
    print(f"Running external build for target: {args.target}")
    print(args.config)
    if target not in EXTERNAL_BUILD_TARGET_DISPATCH:
        raise ValueError(f"Unknown build target: {target}")
    EXTERNAL_BUILD_TARGET_DISPATCH[target](args)
