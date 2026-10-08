import argparse

from tools.sandbox.common import FILESYSTEM_TAG, output, owned_sandbox
from tools.sandbox.filesystem import Filesystem
from tools.sandbox.logger import log


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "stop", help="stop a sandbox and save its filesystem"
    )
    parser.add_argument("sandbox_id")
    parser.set_defaults(run=run)


def run(args: argparse.Namespace) -> None:
    sandbox = owned_sandbox(args.sandbox_id)
    filesystem_id = sandbox.get_tags().get(FILESYSTEM_TAG)
    if filesystem_id is None:
        raise RuntimeError(f"sandbox {args.sandbox_id} has no filesystem")
    filesystem = Filesystem(filesystem_id)
    with filesystem.lock():
        log.info("stopping sandbox and saving filesystem %s", filesystem.id)
        filesystem.save(sandbox)
    output({"sandbox_id": args.sandbox_id, "filesystem": filesystem.id})
