"""Run a ClickHouse query through the ci-metrics Grafana wrapper and return rows as dicts.

This goes through the Grafana datasource the ci-metrics skill uses because
developers do not have direct ClickHouse credentials.
"""

from __future__ import annotations

import json
import re
import subprocess
import threading
import time

from tools.testing.llm_selector.eval import REPO


# The ci-metrics skill's wrapper and the grafana-clickhouse-datasource UID its SKILL.md gives; update both if the skill moves or the UID rotates.
GCX = REPO / ".agents/skills/ci-metrics/gcx-wrapper.sh"
CLICKHOUSE_UID = "ceczcsck1b20wb"
# The wrapper rewrites gcx's config (use-context, login on token expiry) on each run, so runs are serialized; backoffs are not.
LOCK = threading.Lock()
TIMEOUT = 600
ATTEMPTS = 3
# Server-side "Timeout exceeded" is deterministic, so it is not here; code 202 is "Too many simultaneous queries".
# HTTP codes need an HTTP/status prefix or their reason phrase, so "elapsed 500 seconds" is not transient.
# Go's (gcx's) and curl's network errors are transient, and so is the wrapper's re-login, which it runs when its token check fails for any reason.
TRANSIENT = re.compile(r"timed out|throttl|rate.?limit|connection (reset|refused|closed)|temporar(il)?y unavailable|too many simultaneous queries|dial tcp|i/o timeout|TLS handshake timeout|unexpected EOF|: EOF\b|no such host|broken pipe|network is unreachable|could not resolve host|failed to connect|gcx login failed|failed to fetch gcx token|(HTTP|status)\D{0,12}\b(429|500|502|503|504)\b|returned error: (429|5\d\d)\b|\b(429 Too Many|500 Internal Server|502 Bad Gateway|503 Service Unavailable|504 Gateway Time)", re.IGNORECASE)  # fmt: skip


def query(sql: str) -> list[dict]:
    """The rows of `sql`. A failure whose stderr is transient is retried, for ATTEMPTS attempts in all; a local timeout, malformed output or any other failure raises at once."""
    for attempt in range(ATTEMPTS):
        if attempt:
            time.sleep(30 * attempt)
        cmd = [str(GCX), "datasources", "clickhouse", "query", "-d", CLICKHOUSE_UID, "--limit", "0", "-o", "json", sql]  # fmt: skip
        try:
            with LOCK:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            err = f"attempt {attempt + 1} of {ATTEMPTS} timed out after {TIMEOUT}s"
            break
        if p.returncode == 0:
            try:
                if (m := re.search(r'\{\s*"columns"', p.stdout)) is None:
                    raise ValueError("no JSON result")
                d = json.JSONDecoder().raw_decode(p.stdout, m.start())[0]
                cols = [c["name"] for c in d["columns"]]
                return [dict(zip(cols, r)) for r in d["rows"] or []]
            except (ValueError, KeyError, TypeError):
                err = f"attempt {attempt + 1} of {ATTEMPTS} returned no result: {p.stdout[-2000:]}"
                break
        err = f"attempt {attempt + 1} of {ATTEMPTS} exited with code {p.returncode}: {p.stderr[-2000:] or p.stdout[-2000:]}"  # fmt: skip
        if not TRANSIENT.search(p.stderr or p.stdout):
            break
    raise RuntimeError(f"ClickHouse query failed, {err}\nSQL: {sql[:500]}")
