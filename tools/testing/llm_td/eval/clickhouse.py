"""Run a ClickHouse query through the ci-metrics Grafana wrapper and return rows as dicts."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


GCX = Path(__file__).resolve().parents[4] / ".agents/skills/ci-metrics/gcx-wrapper.sh"


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
    start = p.stdout.find('{"columns"')
    try:
        d = json.JSONDecoder().raw_decode(p.stdout[start:])[0]
    except (ValueError, IndexError) as e:
        raise RuntimeError(p.stdout[-2000:] + p.stderr[-2000:]) from e
    cols = [c["name"] for c in d["columns"]]
    return [dict(zip(cols, r)) for r in d["rows"]]
