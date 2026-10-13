import argparse
import sys

from modal.stream_type import StreamType

from tools.sandbox.common import running_sandbox, sandboxes
from tools.sandbox.logger import log


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "exec",
        help="run a command in a sandbox and stream its output",
        description=(
            "Runs COMMAND in a login shell inside the sandbox, like ssh does: the words are joined"
            " with spaces, so pipes, redirections and && work when quoted. The exit code is the"
            " command's."
        ),
    )
    parser.add_argument(
        "-s", "--sandbox", help="sandbox id (default: your only running sandbox)"
    )
    parser.add_argument(
        "command", nargs=argparse.REMAINDER, help="command and arguments"
    )
    parser.set_defaults(run=run)


def run(args: argparse.Namespace) -> None:
    command = " ".join(args.command[1:] if args.command[:1] == ["--"] else args.command)
    if not command:
        log.error("no command given")
        sys.exit(2)
    sandbox = running_sandbox(args.sandbox or only_running_sandbox_id())
    # A login shell picks up /etc/profile.d/sandbox.sh (venv python, CUDA, ccache).
    process = sandbox.exec(
        "bash", "-lc", command, stdout=StreamType.STDOUT, stderr=StreamType.STDOUT
    )
    # Nothing is forwarded to the command, so tell it so; otherwise `cat`-style readers hang.
    process.stdin.write_eof()
    process.stdin.drain()
    sys.exit(process.wait())


def only_running_sandbox_id() -> str:
    running = sandboxes()
    if len(running) != 1:
        ids = ", ".join(sandbox.object_id for sandbox in running) or "none"
        log.error(
            "pass --sandbox: you have %s running sandboxes (%s)", len(running), ids
        )
        sys.exit(1)
    return running[0].object_id
