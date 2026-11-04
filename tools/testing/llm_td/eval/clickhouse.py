"""Run a ClickHouse query through the ci-metrics Grafana wrapper and return rows as dicts.

This goes through the Grafana datasource the ci-metrics skill uses because
developers do not have direct ClickHouse credentials.
"""

from __future__ import annotations

import json
import re
import subprocess
import time

from tools.testing.llm_td.eval import REPO


GCX = REPO / ".agents/skills/ci-metrics/gcx-wrapper.sh"
# The grafana-clickhouse-datasource UID (see the ci-metrics skill).
CLICKHOUSE_UID = "ceczcsck1b20wb"
TIMEOUT = 600
ATTEMPTS = 3
# Server-side "Timeout exceeded" is deterministic, so it is not here; code 202 is "Too many simultaneous queries".
TRANSIENT = re.compile(r"timed out|throttl|rate.?limit|connection (reset|refused|closed)|temporar|too many simultaneous queries|\b(429|500|502|503|504)\b", re.IGNORECASE)  # fmt: skip


def query(sql: str) -> list[dict]:
    """The rows of `sql`. A failure whose stderr is transient is retried up to ATTEMPTS times; a local timeout, malformed output or any other failure raises at once."""
    for attempt in range(ATTEMPTS):
        if attempt:
            time.sleep(30 * attempt)
        cmd = [str(GCX), "datasources", "clickhouse", "query", "-d", CLICKHOUSE_UID, "--limit", "0", "-o", "json", sql]  # fmt: skip
        try:
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
