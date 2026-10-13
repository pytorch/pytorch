import argparse
import colorsys
import math
import sys
from contextlib import redirect_stdout

import modal

from tools.sandbox.common import (
    app,
    FILESYSTEM_TAG,
    github_keys,
    output,
    sandbox_entry,
    sandbox_info,
    SSH_PORT,
    username,
)
from tools.sandbox.filesystem import Filesystem
from tools.sandbox.flavor import FLAVORS, hourly_rate, parse_flavor
from tools.sandbox.image import base_image
from tools.sandbox.logger import log, no_color


SANDBOX_TIMEOUT_SECONDS = 24 * 60 * 60  # the maximum Modal allows
SANDBOX_IDLE_TIMEOUT_SECONDS = 60 * 60


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "create",
        help="create a sandbox on a filesystem",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Prices come from Modal, so fetch them only when the help is actually shown.
    default_format_help = parser.format_help

    def format_help_with_prices() -> str:
        parser.epilog = flavor_prices()
        return default_format_help()

    parser.format_help = format_help_with_prices
    parser.add_argument(
        "--filesystem",
        help="filesystem id from `list` (default: a new filesystem from a fresh CI image)",
    )
    parser.add_argument(
        "-f",
        "--flavor",
        metavar="FLAVOR",
        choices=FLAVORS,
        default="cpu",
        help="hardware; GPU flavors get 16 cores and 64 GiB per GPU (default: %(default)s)",
    )
    parser.set_defaults(run=run)


def run(args: argparse.Namespace) -> None:
    flavor = parse_flavor(args.flavor)
    filesystem = Filesystem(args.filesystem) if args.filesystem else Filesystem.new()
    log.info("loading GitHub SSH keys")
    keys = github_keys()
    tags = {
        "username": username(),
        "flavor": flavor.name,
        FILESYSTEM_TAG: filesystem.id,
    }
    # The entrypoint installs the user's GitHub keys and runs sshd in the foreground.
    start_sshd = (
        "mkdir -p /root/.ssh /run/sshd"
        " && printf '%s\\n' \"$AUTHORIZED_KEYS\" > /root/.ssh/authorized_keys"
        " && chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys"
        " && ssh-keygen -A > /dev/null && exec /usr/sbin/sshd -D -e"
    )
    log.info("creating %s sandbox on %s", flavor.name, filesystem.id)
    try:
        with filesystem.lock(), redirect_stdout(sys.stderr), modal.enable_output():
            application = app()
            if args.filesystem:
                log.info("restoring filesystem %s", filesystem.id)
                image = filesystem.image()
            else:
                log.info("preparing CI image; the first build may take several minutes")
                image = base_image()
                image.build(app=application)
            log.info("waiting for %s allocation and startup", flavor.name)
            sandbox = modal.Sandbox.create(
                "bash",
                "-c",
                start_sshd,
                app=application,
                name=filesystem.sandbox_name,
                image=image,
                runtime="gvisor",
                gpu=flavor.gpu,
                cpu=flavor.cpu,
                memory=flavor.memory_mib,
                timeout=SANDBOX_TIMEOUT_SECONDS,
                idle_timeout=SANDBOX_IDLE_TIMEOUT_SECONDS,
                tags=tags,
                env={"AUTHORIZED_KEYS": keys},
                unencrypted_ports=[SSH_PORT],
                experimental_options={"enable_exit_snapshot": True},
            )
    except modal.exception.AlreadyExistsError:
        log.error("filesystem %s already has a running sandbox", filesystem.id)
        sys.exit(1)
    except modal.exception.ResourceExhaustedError:
        log.error("no capacity for %s right now, try fewer GPUs or later", flavor.name)
        sys.exit(1)
    except modal.exception.NotFoundError:
        log.error("filesystem %s no longer exists, `rm` it", filesystem.id)
        sys.exit(1)
    log.info("sandbox %s started; fetching connection details", sandbox.object_id)
    entry = sandbox_entry(sandbox_info(sandbox.object_id))
    ssh = entry["connect"]["ssh"]
    log.info("ssh: ssh -p %s %s@%s", ssh["port"], ssh["user"], ssh["host"])
    output(entry)


def flavor_prices() -> str:
    """Help text: a table of every flavor with its hardware and current hourly price."""
    header = ("flavor", "GPUs", "cores", "memory", "rate")
    rows, costs = [], []
    for name in FLAVORS:
        flavor = parse_flavor(name)
        gpus = (
            f"{flavor.gpu_count} x {flavor.gpu_type.upper()}"
            if flavor.gpu_type
            else "-"
        )
        memory = f"{flavor.memory_mib // 1024} GiB"
        costs.append(hourly_rate(flavor))
        rows.append((name, gpus, str(flavor.cpu), memory, f"${costs[-1]:.2f}/h"))
    widths = [
        max(len(row[column]) for row in (header, *rows))
        for column in range(len(header))
    ]
    right_aligned = {2, 3, 4}
    colorful = not no_color() and sys.stdout.isatty()

    def line(cells: tuple[str, ...], cost: float | None = None) -> str:
        padded = [
            cell.rjust(widths[column])
            if column in right_aligned
            else cell.ljust(widths[column])
            for column, cell in enumerate(cells)
        ]
        if cost is not None and colorful:
            # Log scale: prices span 35x, so a linear ramp would leave most rows the same color.
            fraction = math.log(cost / min(costs)) / math.log(max(costs) / min(costs))
            padded[-1] = f"{cost_gradient(fraction)}{padded[-1]}\033[0m"
        return "│ " + " │ ".join(padded) + " │"

    def rule(left: str, middle: str, right: str) -> str:
        return left + middle.join("─" * (width + 2) for width in widths) + right

    body = [rule("┌", "┬", "┐"), line(header), rule("├", "┼", "┤")]
    body += [line(row, cost) for row, cost in zip(rows, costs)]
    body.append(rule("└", "┴", "┘"))
    return "flavors, at Modal's current list prices:\n" + "\n".join(body) + "\n"


def cost_gradient(fraction: float) -> str:
    """ANSI 24-bit color sweeping from green (0.0) through yellow to red (1.0)."""
    hue = (1 - fraction) / 3  # 1/3 is green, 0 is red on the color wheel
    red, green, blue = (
        round(255 * channel) for channel in colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    )
    return f"\033[38;2;{red};{green};{blue}m"
