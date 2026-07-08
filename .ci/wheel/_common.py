"""Helpers shared across the Linux, Windows, and macOS CD wheel pipelines.

Each platform's build_env_setup.py / build_install_deps.py runs an
independent stage, streams binaries from ossci mirrors, pins numpy by
Python version, and hands env back to a parent bash wrapper via a
`--env-out` file. Keeping these helpers here means a fix in one place
(e.g. the cygpath -up PATH conversion, the BASH_FUNC_retry%% filter, or a
numpy pin) reaches every pipeline.

Import from a sibling stage script with, e.g.:

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # .ci/wheel
    from _common import numpy_pin, pip_install, write_env_exports
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
import urllib.request
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from pathlib import Path


_BASH_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


# NumPy build-time pin per supported CPython version. Each entry is the
# oldest numpy *minor* that publishes a wheel for that interpreter (verified
# on PyPI) -- which gives the built extension the widest runtime-numpy range
# -- at the newest *patch* of that minor that has been shipped in production
# by one of the CD platforms, so no numpy bugfix is dropped for a marginal
# floor change. Listed explicitly (rather than a prefix match with a default)
# so an unsupported version -- e.g. a new Python without numpy wheels yet --
# fails loudly instead of silently picking a stale fallback. Freethreaded
# builds (e.g. 3.14t) share their base version's pin, since sys.version_info
# does not distinguish them.
NUMPY_PINS: dict[str, str] = {
    "3.10": "2.0.2",
    "3.11": "2.0.2",
    "3.12": "2.0.2",
    "3.13": "2.1.2",
    "3.14": "2.3.4",
}


def shell_quote(value: str) -> str:
    if value and all(c.isalnum() or c in "_-./:=" for c in value):
        return value
    return "'" + value.replace("'", "'\\''") + "'"


def _to_posix_path_list(windows_path_list: str) -> str:
    """Convert a Windows `;`-separated path list to POSIX `:`-separated.

    vcvarsall.bat (and other cmd-side env-setup scripts) write PATH in
    Windows format with `;` separators and `\\` directory separators. The
    parent bash uses `:` separators and POSIX-style paths to find
    executables; sourcing PATH unmodified leaves bash with a single
    bogus PATH entry and the next `python` lookup dies with exit 127.

    `cygpath -up` is the canonical translator (Git Bash / MSYS ship it
    and the rest of PyTorch's Windows CI already uses it). Calling it
    once on the whole list is fine for this hot path. Only invoked on
    Windows (guarded by a `;` check in write_env_exports), so the absence
    of cygpath on Linux/macOS is never hit.
    """
    if not windows_path_list:
        return windows_path_list
    result = subprocess.run(
        ["cygpath", "-up", windows_path_list],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def write_env_exports(env: dict[str, str], path: Path | None) -> None:
    """Write `export KEY=VALUE` lines for the parent bash to source.

    On Windows, PATH is converted from Windows-format (`;`-separated,
    backslashes) to POSIX-format so the parent bash can find executables in
    subsequent steps; the `;` check makes this a no-op on Linux/macOS, where
    PATH is already POSIX. Other path-like env vars (INCLUDE, LIB, ...) are
    left in Windows format because the MSVC tools that consume them expect
    that.

    Keys that aren't valid bash identifiers are skipped. When CI bash
    exports a function (e.g. `export -f retry` in binary_populate_env.sh),
    bash serializes it into the env as `BASH_FUNC_retry%%=() { ... }`. That
    leaks into the Python interpreter's `os.environ` and, since bash cannot
    re-export an identifier containing `%`, sourcing the env file would die
    on `not a valid identifier`.
    """
    if path is None:
        return
    lines = []
    for k, v in env.items():
        if not _BASH_IDENT_RE.match(k):
            continue
        if k.upper() == "PATH" and ";" in v:
            # ';' separator marks the captured PATH as Windows-form; if
            # some future caller already feeds POSIX form (no ';') we
            # leave it alone rather than double-convert.
            v = _to_posix_path_list(v)
        lines.append(f"export {k}={shell_quote(v)}")
    path.write_text("\n".join(lines) + "\n")


def download(url: str, dest: Path, attempts: int = 5) -> None:
    """Stream `url` to `dest`, retrying with exponential backoff."""
    for attempt in range(1, attempts + 1):
        try:
            print(f"Downloading {url} -> {dest} (attempt {attempt}/{attempts})")
            with urllib.request.urlopen(url) as response, open(dest, "wb") as out:
                while chunk := response.read(1 << 20):
                    out.write(chunk)
            return
        except Exception as exc:
            if attempt == attempts:
                sys.exit(f"Failed to download {url}: {exc}")
            time.sleep(2**attempt)


def retry(cmd: list[str], delays: tuple[int, ...] = (1, 2, 4, 8)) -> None:
    """Run cmd, retrying with backoff on failure (mirrors the shell retry helper)."""
    last_rc = 0
    for delay in (0, *delays):
        if delay:
            time.sleep(delay)
        result = subprocess.run(cmd)
        if result.returncode == 0:
            return
        last_rc = result.returncode
    sys.exit(last_rc)


def pip_install(*args: str) -> None:
    retry([sys.executable, "-m", "pip", "install", *args])


def numpy_pin() -> str:
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    pin = NUMPY_PINS.get(version)
    if pin is None:
        sys.exit(
            f"Unsupported Python version {version}: add a numpy pin to "
            "NUMPY_PINS in .ci/wheel/_common.py"
        )
    return pin
