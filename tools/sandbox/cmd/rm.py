import argparse

from tools.sandbox.common import output
from tools.sandbox.filesystem import Filesystem


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "rm", help="permanently delete a filesystem and all its snapshots"
    )
    parser.add_argument("filesystem", help="filesystem id from `list`")
    parser.set_defaults(run=run)


def run(args: argparse.Namespace) -> None:
    filesystem = Filesystem(args.filesystem)
    with filesystem.lock():
        sandbox_ids = filesystem.delete()
    output({"filesystem": filesystem.id, "sandboxes": sandbox_ids})
