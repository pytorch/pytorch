# main.py

import argparse
import logging

from cli.build_cli.register_build import register_build_commands
from cli.lib.common.logger import setup_logging


logger = logging.getLogger(__name__)


def main():
    # Define top-level parser
    parser = argparse.ArgumentParser(description="Lumos CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    parser.add_argument(
        "--log-level", default="INFO", help="Log level (DEBUG, INFO, WARNING, ERROR)"
    )

    # registers second-level subcommands
    register_build_commands(subparsers)

    # parse args after all options are registered
    args = parser.parse_args()

    # setup global logging
    setup_logging(getattr(logging, args.log_level.upper(), logging.INFO))
    logger.debug("Parsed args: %s", args)

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
