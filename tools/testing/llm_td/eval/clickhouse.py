"""Run a ClickHouse query through the ci-metrics Grafana wrapper and return rows as dicts.

This goes through the Grafana datasource the ci-metrics skill uses because
developers do not have direct ClickHouse credentials.
"""

from __future__ import annotations

import json
import re
import subprocess

from tools.testing.llm_td.eval import REPO


GCX = REPO / ".agents/skills/ci-metrics/gcx-wrapper.sh"


def query(sql: str) -> list[dict]:
    p = subprocess.run(
        [
            str(GCX),
            "datasources",
            "clickhouse",
            "query",
            "--limit",
            "0",
            "-o",
            "json",
            sql,
        ],
        capture_output=True,
        text=True,
    )
    m = re.search(r'\{\s*"columns"', p.stdout)
    try:
        if m is None:
            raise ValueError("no result in the output")
        d = json.JSONDecoder().raw_decode(p.stdout, m.start())[0]
    except ValueError as e:
        raise RuntimeError(p.stdout[-2000:] + p.stderr[-2000:]) from e
    cols = [c["name"] for c in d["columns"]]
    return [dict(zip(cols, r)) for r in d["rows"] or []]
