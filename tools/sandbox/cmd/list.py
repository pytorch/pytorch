import argparse

from tools.sandbox.common import output, sandbox_entry, sandbox_infos
from tools.sandbox.filesystem import filesystem_entries


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("list", help="list your sandboxes and filesystems")
    parser.add_argument(
        "--historical",
        action="store_true",
        help="also list sandboxes that have ended",
    )
    parser.set_defaults(run=run)


def run(args: argparse.Namespace) -> None:
    infos = sandbox_infos(include_finished=True)
    output(
        {
            "sandboxes": [
                sandbox_entry(info)
                for info in infos
                if args.historical or not info.task_info.finished_at
            ],
            "filesystems": filesystem_entries(infos),
        }
    )
