"""`gh api` wrapper shared by the triage hooks and the outcome check."""

import subprocess
import time
from collections.abc import Callable


def gh_api(
    args: list[str],
    attempts: int = 3,
    timeout: int = 10,
    log: Callable[[str], None] = lambda _: None,
) -> str:
    """Run `gh api` with retry and backoff. Returns stdout; raises RuntimeError
    after the last attempt. Any non-zero exit is retried: gh exits 1 for
    HTTP errors and the transient ones (429, 5xx) are what we care about."""
    for attempt in range(1, attempts + 1):
        result = subprocess.run(
            ["gh", "api", *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout
        error = result.stderr.strip()
        log(
            f"gh api {args[0]} failed (attempt {attempt}, exit {result.returncode}): {error}"
        )
        if attempt < attempts:
            time.sleep(2**attempt)
    raise RuntimeError(
        f"gh api {' '.join(args)} failed after {attempts} attempts: {error}"
    )
