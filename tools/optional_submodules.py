from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from subprocess import CalledProcessError, check_call, PIPE, run


repo_root = Path(__file__).absolute().parent.parent
third_party_path = repo_root / "third_party"


def _read_file(path: Path) -> str:
    with path.open(encoding="utf-8") as f:
        return f.read().strip()


# Failures of a clone that are worth another attempt, matched against git's
# stderr (English: _clone runs git in the C locale):
#   - gitlab.com's "unable to handle this request due to load" page;
#   - HTTP 429 and 5xx, as `The requested URL returned error: 503` (git prints it
#     for a failed POST too, after `RPC failed; HTTP 503 curl 22`);
#   - `RPC failed; curl 18`, `curl 56` and `curl 92 ... CANCEL/INTERNAL_ERROR`, a
#     transfer that broke part way (partial file, receive failure, HTTP/2 stream
#     cancelled). The curl code is only checked when the clone fails before any
#     pack data has arrived. Other curl errors, e.g. `curl 16` from a broken
#     HTTP/2 framing layer, usually come back on every attempt and are not
#     retried then;
#   - `early EOF`, `unexpected disconnect` and `GnuTLS recv error`, the same for a
#     connection that dropped while git was reading. These are matched anywhere in
#     stderr, whatever curl error came before them: a failure while the pack is
#     streaming is normally followed by `unexpected disconnect` and `early EOF`, so
#     it is retried even for a curl code listed above as not retried;
#   - `SSL_read: Connection reset` and `SSL_read: SSL_ERROR_SYSCALL`, a connection
#     that was reset while data was being read;
#   - `Operation too slow`, a transfer aborted by the low-speed limit below. A bare
#     `curl 28` is not matched: the same code is the connect timeout, which is not
#     retried.
# Only these read-time signatures are matched. A reset while the connection is
# being set up (`SSL_connect: Connection reset ... in connection to`) is not
# retried: it usually never goes away, and each attempt would stall for the 10 to
# 60 s wait. The same goes for a bare `Recv failure: Connection reset by peer`:
# curl prints it both for a proxy that refuses the CONNECT and, on current curl,
# for a reset while a response is being read, and the message cannot tell the two
# apart, so neither is retried. (In a POST, git adds `RPC failed; curl 56`, which
# is retried.)
# Everything else fails at once instead of stalling for minutes, e.g. in the
# `cmake` configure step: a tag that does not exist, HTTP 4xx other than 429, no
# DNS, an unwritable directory. Connect timeouts are not retried either: on a
# network that silently drops traffic each attempt would wait for the full connect
# timeout (~2 minutes) before failing.
_TRANSIENT_CLONE_ERROR = re.compile(
    r"unable to handle this request due to load"
    r"|The requested URL returned error: (?:429|5\d\d)"
    r"|RPC failed; (?:curl (?:18|56)\b|curl 92\b[^\n]*(?:CANCEL|INTERNAL_ERROR))"
    r"|early EOF|unexpected disconnect|GnuTLS recv error"
    r"|SSL_read: (?:Connection reset|SSL_ERROR_SYSCALL)"
    r"|Operation too slow",
    re.IGNORECASE,
)


# Abort a transfer that stays under 1000 bytes/s for 60 s, so a stalled clone fails
# (and is retried) instead of blocking until the job's timeout.
_LOW_SPEED_ARGS = ["-c", "http.lowSpeedLimit=1000", "-c", "http.lowSpeedTime=60"]


def _is_transient_clone_error(stderr: str) -> bool:
    return _TRANSIENT_CLONE_ERROR.search(stderr) is not None


def _clone(
    cmd: list[str],
    /,
    *,
    cwd: Path | None = None,
    attempts: int = 8,
    backoff: int = 10,
    max_delay: int = 60,
) -> None:
    # git has no retry or backoff setting for `clone` (`http.lowSpeedLimit` and
    # `http.lowSpeedTime`, set by the callers, only abort a stalled transfer), so
    # retry here. A failed clone removes the directory it created, so there is
    # nothing to clean up.
    #
    # stderr is captured so that it can be matched and kept on the exception. git
    # shows its progress meter only when stderr is a terminal, so an interactive
    # run prints nothing until each attempt ends. In CI too (including under
    # cmake's execute_process), git's output appears only when an attempt exits,
    # and is lost if the attempt is killed, e.g. by a step or job timeout.
    if attempts < 1:
        raise ValueError(f"attempts must be at least 1, got {attempts}")
    for attempt in range(1, attempts + 1):
        # The messages are matched as English text, so force the C locale for git.
        # errors="replace" keeps non-ASCII output (paths, `remote:` text) from
        # raising UnicodeDecodeError when it is decoded here.
        result = run(
            cmd,
            cwd=cwd,
            stderr=PIPE,
            text=True,
            errors="replace",
            env={**os.environ, "LC_ALL": "C"},
        )
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr, flush=True)
        if result.returncode == 0:
            return
        if attempt == attempts or not _is_transient_clone_error(result.stderr):
            raise CalledProcessError(result.returncode, cmd, stderr=result.stderr)
        delay = min(backoff * 2 ** (attempt - 1), max_delay)
        print(
            f"-- Clone failed (attempt {attempt}/{attempts}), retry in {delay}s",
            flush=True,
        )
        time.sleep(delay)


def _checkout_by_tag(repo: str, tag: str) -> None:
    _clone(
        ["git", *_LOW_SPEED_ARGS, "clone", "--depth", "1", "--branch", tag, repo],
        cwd=third_party_path,
    )


def read_nccl_pin(cuda_version: str = "") -> str:
    # Default NCCL version
    nccl_file = "nccl.txt"

    # If NCCL version diverges for different CUDA versions, uncomment the
    # following block and add the appropriate file (using CUDA 11 as an example)

    # 12.6 builds for sm50, needs a lower version
    if not cuda_version:
        cuda_version = os.getenv("DESIRED_CUDA", os.getenv("CUDA_VERSION", ""))
    if cuda_version.startswith("12.6") or cuda_version == "cu126":
        nccl_file = "nccl-cu126.txt"

    nccl_pin_path = repo_root / ".ci" / "docker" / "ci_commit_pins" / nccl_file
    return _read_file(nccl_pin_path)


def checkout_nccl() -> None:
    release_tag = read_nccl_pin()
    print(f"-- Checkout nccl release tag: {release_tag}")
    nccl_basedir = third_party_path / "nccl"
    if not nccl_basedir.exists():
        _checkout_by_tag("https://github.com/NVIDIA/nccl", release_tag)


def checkout_nccl_ep() -> None:
    revision = _read_file(third_party_path / "nccl_ep" / "pin.txt")
    print(f"-- Checkout nccl-extensions revision: {revision}")
    source_dir = third_party_path / "nccl-extensions"
    if not source_dir.exists():
        _clone(
            [
                "git",
                *_LOW_SPEED_ARGS,
                "clone",
                "--no-checkout",
                "https://github.com/NVIDIA/nccl-extensions.git",
                str(source_dir),
            ]
        )
        check_call(["git", "checkout", "--detach", revision], cwd=source_dir)


def checkout_eigen() -> None:
    eigen_tag = _read_file(third_party_path / "eigen_pin.txt")
    print(f"-- Checkout Eigen release tag: {eigen_tag}")
    eigen_basedir = third_party_path / "eigen"
    if not eigen_basedir.exists():
        _checkout_by_tag("https://gitlab.com/libeigen/eigen", eigen_tag)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # If no arguments are given checkout all optional dependency
        checkout_nccl()
        checkout_nccl_ep()
        checkout_eigen()
    else:
        # Otherwise just call top-level function of choice
        globals()[sys.argv[1]]()
