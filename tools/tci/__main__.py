"""Run from the repository root: python -m tools.tci <command>"""

from __future__ import annotations

import argparse

from tools.tci import collect, run, shard


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m tools.tci")
    commands = parser.add_subparsers(dest="command", required=True)
    modules = (("collect", collect), ("shard", shard), ("run", run))
    for name, module in modules:
        command = commands.add_parser(name, help=module.__doc__)
        module.add_arguments(command)
        command.set_defaults(run=module.run)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
