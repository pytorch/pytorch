"""Run a ClickHouse query through the ci-metrics Grafana wrapper and return rows as dicts.

This goes through the Grafana datasource the ci-metrics skill uses because
developers do not have direct ClickHouse credentials.
"""

from __future__ import annotations

import contextlib
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
TRANSIENT = re.compile(r"timed? ?out|throttl|rate.?limit|connection|temporar|\b(429|500|502|503|504)\b", re.IGNORECASE)  # fmt: skip


def query(sql: str) -> list[dict]:
    """The rows of `sql`. A timeout or a transient failure is retried up to ATTEMPTS times; any other failure raises at once."""
    err = ""
    for attempt in range(ATTEMPTS):
        if attempt:
            time.sleep(30 * attempt)
        cmd = [str(GCX), "datasources", "clickhouse", "query", "-d", CLICKHOUSE_UID, "--limit", "0", "-o", "json", sql]  # fmt: skip
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            err = f"attempt {attempt + 1} of {ATTEMPTS} timed out after {TIMEOUT}s"
            continue
        m = re.search(r'\{\s*"columns"', p.stdout)
        if p.returncode == 0 and m:
            with contextlib.suppress(ValueError):
                d = json.JSONDecoder().raw_decode(p.stdout, m.start())[0]
                cols = [c["name"] for c in d["columns"]]
                return [dict(zip(cols, r)) for r in d["rows"] or []]
        out = p.stdout[-2000:] + p.stderr[-2000:]
        err = f"attempt {attempt + 1} of {ATTEMPTS} exited with code {p.returncode}: {out}"
        if not TRANSIENT.search(out):
            break
    raise RuntimeError(f"ClickHouse query failed, {err}\nSQL: {sql[:500]}")
