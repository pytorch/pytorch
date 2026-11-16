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


# Transient failures worth retrying, matched against git's stderr (English, see
# _clone): gitlab.com's load page, HTTP 429/5xx, and signatures of a transfer that
# broke while reading. Connection-setup failures, connect timeouts and permanent
# errors (missing tag, other 4xx, DNS) fail at once. Examples are in the tests.
_TRANSIENT_CLONE_ERROR = re.compile(
    r"unable to handle this request due to load"
    r"|The requested URL returned error: (?:429|5\d\d)"
    r"|RPC failed; (?:curl (?:18|56)\b|curl 92\b[^\n]*(?:CANCEL|INTERNAL_ERROR))"
    r"|early EOF|unexpected disconnect|GnuTLS recv error"
    r"|SSL_read: (?:Connection reset|SSL_ERROR_SYSCALL)"
    r"|Operation too slow",
    re.IGNORECASE,
)


# Abort a transfer under 1000 bytes/s for 60 s, so a stalled clone fails and is retried.
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
    # git has no retry setting for `clone`, so retry here. stderr is captured to
    # match it and keep it on the exception, so git's output appears when an
    # attempt ends.
    if attempts < 1:
        raise ValueError(f"attempts must be at least 1, got {attempts}")
    for attempt in range(1, attempts + 1):
        # C locale so the messages match as English; errors="replace" so non-ASCII
        # output cannot raise UnicodeDecodeError.
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
