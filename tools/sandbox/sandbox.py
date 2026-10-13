import argparse
import sys

import modal

from tools.sandbox import cmd
from tools.sandbox.common import username
from tools.sandbox.logger import log


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m tools.sandbox")
    subparsers = parser.add_subparsers()
    for command in (cmd.create, cmd.exec, cmd.list, cmd.rm, cmd.stop):
        command.add_parser(subparsers)
    args = parser.parse_args()
    if not hasattr(args, "run"):
        parser.print_help()
        sys.exit(1)
    if args.run is cmd.create.run:
        log.info("checking GitHub authentication")
    log.debug("signed in as %s", username())
    try:
        args.run(args)
    except (RuntimeError, modal.exception.Error) as error:
        log.error("%s", error)
        sys.exit(1)
