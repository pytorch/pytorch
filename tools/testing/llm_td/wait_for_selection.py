"""Hold a workflow's test jobs until an enforcing llm-td.yml run on the PR head finishes.

Nothing orders llm-td.yml (a pull_request_target workflow) before the test jobs,
and on a Python-only change the build can finish first. Run once per workflow
run in the target-determination job, which every test job needs, so no test
runner idles. Shadow runs never wait: their selection changes nothing that runs.
llm-td.yml names each run "llm-td enforce" or "llm-td shadow", since a ciflow
tag push carries no PR labels.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request


WAIT_SECONDS = 600
PENDING = {"queued", "in_progress", "requested", "pending"}


def get(path: str) -> dict:
    url = f"https://api.github.com/repos/{os.environ['GITHUB_REPOSITORY']}/{path}"
    auth = {"Authorization": f"Bearer {os.environ['GH_TOKEN']}"}
    req = urllib.request.Request(url, headers=auth)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def pending_enforce_runs(sha: str) -> list[dict]:
    q = f"event=pull_request_target&head_sha={sha}"
    runs = get(f"actions/workflows/llm-td.yml/runs?{q}")["workflow_runs"]
    enforce = [r for r in runs if r["display_title"] == "llm-td enforce"]
    return [r for r in enforce if r["status"] in PENDING]


def main() -> None:
    sha = os.environ.get("HEAD_SHA", "")
    tag = re.fullmatch(r"refs/tags/ciflow/.+/(\d+)", os.environ.get("GITHUB_REF", ""))
    if not sha or not (os.environ.get("PR_NUMBER") or tag):
        return
    deadline, delay = time.monotonic() + WAIT_SECONDS, 15
    while (runs := pending_enforce_runs(sha)) and time.monotonic() < deadline:
        print(f"Waiting for {runs[0]['html_url']}", file=sys.stderr)
        time.sleep(delay)
        delay = min(2 * delay, 60)


if __name__ == "__main__":
    main()
