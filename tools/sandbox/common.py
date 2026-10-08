import datetime
import json
import shutil
import subprocess
import sys
from functools import cache
from typing import Any

import modal
from modal._utils.async_utils import synchronize_api
from modal.client import _Client
from modal_proto import api_pb2

from tools.sandbox.flavor import FLAVORS, hourly_rate, parse_flavor
from tools.sandbox.logger import log


SSH_PORT = 22
SSH_USER = "root"
SNAPSHOT_TIMEOUT_SECONDS = 600
# A stable filesystem id connects every sandbox generation to its saved image.
FILESYSTEM_TAG = "filesystem"


@cache
def username() -> str:
    github_cli = shutil.which("gh")
    if github_cli is None:
        log.error("gh is not installed, install it and run `gh auth login`")
        sys.exit(1)
    result = subprocess.run(
        [github_cli, "api", "user", "--jq", ".login"], capture_output=True, text=True
    )
    if result.returncode != 0:
        log.error(
            "not logged in to GitHub, run `gh auth login`: %s", result.stderr.strip()
        )
        sys.exit(1)
    return result.stdout.strip()


@cache
def github_keys() -> str:
    """The user's public SSH keys registered on GitHub, one per line."""
    result = subprocess.run(
        ["gh", "api", f"users/{username()}/keys", "--jq", ".[].key"],
        capture_output=True,
        text=True,
        check=True,
    )
    if not result.stdout.strip():
        log.error(
            "no SSH keys on GitHub for %s, add one at https://github.com/settings/keys",
            username(),
        )
        sys.exit(1)
    return result.stdout


@cache
def repository_root() -> str:
    """Top-level directory of the git checkout the command runs in."""
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


@cache
def app() -> modal.App:
    return modal.App.lookup("pytorch", create_if_missing=True)


def exit_snapshot(sandbox_id: str, timeout: float) -> str | None:
    """Id of the snapshot Modal took of the sandbox's filesystem as it exited, or None if there is none.

    None covers a missing sandbox, disabled exit snapshots, or a deleted snapshot.
    Snapshot creation failures and timeouts propagate to the caller.
    """
    try:
        sandbox = modal.Sandbox.from_id(sandbox_id)
        image = sandbox._experimental_get_exit_snapshot(timeout)
    except (modal.exception.NotFoundError, modal.exception.InvalidError):
        return None
    return image.object_id if image_exists(image.object_id) else None


def delete_image(image_id: str) -> str | None:
    """Delete an image, returning an error unless its absence is confirmed.

    image_delete returns nothing, even for an image that does not exist, so the image is probed afterwards.
    """
    try:
        modal.experimental.image_delete(image_id)
    except modal.exception.NotFoundError:
        return None
    except modal.exception.Error as exception:
        return str(exception) or type(exception).__name__
    try:
        return "still exists after delete" if image_exists(image_id) else None
    except modal.exception.Error as exception:
        return str(exception) or type(exception).__name__


def image_exists(image_id: str) -> bool:
    """Look up the image, propagating failures that do not establish its absence."""
    return synchronize_api(_image_exists)(image_id)


async def _image_exists(image_id: str) -> bool:
    client = await _Client.from_env()
    try:
        await client._stub.ImageFromId(api_pb2.ImageFromIdRequest(image_id=image_id))
    except modal.exception.NotFoundError:
        return False
    return True


def owned_sandbox(sandbox_id: str) -> modal.Sandbox:
    """The user's sandbox with this id; logs an error and exits otherwise."""
    try:
        sandbox = modal.Sandbox.from_id(sandbox_id)
        tags = sandbox.get_tags()
    except (modal.exception.NotFoundError, modal.exception.InvalidError):
        log.error("no sandbox: %s", sandbox_id)
        sys.exit(1)
    if tags.get("username") != username():
        log.error("sandbox %s does not belong to %s", sandbox_id, username())
        sys.exit(1)
    return sandbox


def running_sandbox(sandbox_id: str) -> modal.Sandbox:
    """The user's running sandbox with this id; logs an error and exits otherwise."""
    sandbox = owned_sandbox(sandbox_id)
    if sandbox.poll() is not None:
        log.error(
            "sandbox %s has already exited, `create` will restore its exit snapshot",
            sandbox_id,
        )
        sys.exit(1)
    return sandbox


def sandboxes() -> list[modal.Sandbox]:
    return list(modal.Sandbox.list(app_id=app().app_id, tags={"username": username()}))


def sandbox_infos(include_finished: bool, **tags: str) -> list[api_pb2.SandboxInfo]:
    """This user's sandboxes, newest first, narrowed to those carrying every given tag.

    The public Sandbox.list() cannot include finished ones, so this goes through the RPC.
    """
    all_tags = {"username": username(), **tags}
    return synchronize_api(_sandbox_infos)(app().app_id, all_tags, include_finished)


async def _sandbox_infos(
    app_id: str, tags: dict[str, str], include_finished: bool
) -> list[api_pb2.SandboxInfo]:
    client = await _Client.from_env()
    sandbox_tags = [
        api_pb2.SandboxTag(tag_name=name, tag_value=value)
        for name, value in tags.items()
    ]
    infos: list[api_pb2.SandboxInfo] = []
    before_timestamp = None
    while True:
        request = api_pb2.SandboxListRequest(
            app_id=app_id,
            include_finished=include_finished,
            tags=sandbox_tags,
            before_timestamp=before_timestamp,
        )
        response = await client._stub.SandboxList(request)
        if not response.sandboxes:
            return infos
        infos.extend(response.sandboxes)
        before_timestamp = response.sandboxes[-1].created_at


def sandbox_info(sandbox_id: str) -> api_pb2.SandboxInfo:
    """The listing of one running sandbox."""
    return next(
        info for info in sandbox_infos(include_finished=False) if info.id == sandbox_id
    )


def sandbox_entry(info: api_pb2.SandboxInfo) -> dict[str, Any]:
    """What `create` and `list` print for a sandbox: identity, state, uptime, cost and how to connect."""
    now = datetime.datetime.now(datetime.timezone.utc).timestamp()
    tags = {tag.tag_name: tag.tag_value for tag in info.tags}
    started_at, finished_at = info.task_info.started_at, info.task_info.finished_at
    status = api_pb2.GenericResult.GenericStatus.Name(info.task_info.result.status)
    flavor = tags.get("flavor")
    rate = (
        hourly_rate(parse_flavor(flavor))
        if flavor is not None and flavor in FLAVORS
        else None
    )
    # Uptime runs from start until exit (or now) and is billed at list rates. Modal only records
    # started_at for some sandboxes; creation is within seconds of start since create blocks.
    uptime_hours = (
        max(0.0, (finished_at or now) - (started_at or info.created_at)) / 3600
    )
    state = status.removeprefix("GENERIC_STATUS_").lower() if finished_at else "running"
    connect = None
    if not finished_at:
        # The list RPC does not report tunnels, so ask the sandbox itself. The VS Code link is a
        # Remote-SSH link to /root whose authority is hex-encoded JSON, the form the extension
        # itself produces for user@host:port; VS Code @ Meta registers the fb-vscode:// scheme.
        try:
            host, port = modal.Sandbox.from_id(info.id).tunnels()[SSH_PORT].tcp_socket
        except (modal.exception.ConflictError, modal.exception.SandboxTimeoutError):
            # The listing lags a little behind a sandbox that is shutting down.
            state = "exiting"
        else:
            authority = (
                json.dumps(
                    {"hostName": host, "user": SSH_USER, "port": port},
                    separators=(",", ":"),
                )
                .encode()
                .hex()
            )
            connect = {
                "ssh": {"user": SSH_USER, "host": host, "port": port},
                "vscode": f"fb-vscode://vscode-remote/ssh-remote+{authority}/root",
            }
    return {
        "sandbox_id": info.id,
        "filesystem": tags.get(FILESYSTEM_TAG),
        "flavor": flavor,
        "state": state,
        "created_at": timestamp(info.created_at),
        "finished_at": timestamp(finished_at) if finished_at else None,
        "uptime_hours": round(uptime_hours, 2),
        "hourly_rate_usd": rate,
        "total_cost_usd": round(uptime_hours * rate, 4) if rate is not None else None,
        "dashboard_url": f"https://modal.com/id/{info.id}",
        "connect": connect,
    }


def timestamp(epoch_seconds: float) -> str:
    return datetime.datetime.fromtimestamp(
        epoch_seconds, datetime.timezone.utc
    ).isoformat(timespec="seconds")


def output(result: dict[str, object]) -> None:
    """Print a command result as JSON on stdout; logs go to stderr via `log`."""
    print(json.dumps(result, indent=2))
