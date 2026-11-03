"""Resolve the LLM test selection once per workflow run.

Runs in the target-determination job, which uploads the result for the test
jobs of the same run, so every shard of the run applies the same selection.
The selection comes from the newest llm-td.yml run for the PR head. It is
enforced only while the PR is labeled ci-llm-td and not ci-no-td, and only in
the workflows where run_test.py enables TD (TD_ENABLED). Then, while that run
is pending, this waits for it, polling with backoff up to once a minute until
WAIT_SECONDS have passed (plus at most one API call), and takes its selection
only if it succeeded. Otherwise the newest successful run's selection is used
in shadow. Nothing orders llm-td.yml (a pull_request_target workflow) before
the test jobs, and on a Python-only change the build can finish first.
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import time
import urllib.request
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tools.testing.llm_td.selection import LLM_TD_SELECTION_FILE, Selection


sys.path.remove(str(REPO_ROOT))

WAIT_SECONDS = 600
PENDING = {"queued", "in_progress", "waiting", "requested", "pending"}
WORKFLOW = ".github/workflows/llm-td.yml"


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    # Not forwarded to the blob storage an artifact download redirects to.
    req.add_unredirected_header("Authorization", f"Bearer {os.environ['GITHUB_TOKEN']}")
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def get(path: str) -> dict:
    repo = os.environ["GITHUB_REPOSITORY"]
    return json.loads(fetch(f"https://api.github.com/repos/{repo}/{path}"))


def selector_runs(sha: str) -> list[dict]:
    """llm-td.yml's runs for this head, newest first. Any workflow can upload an
    artifact named like the selection, so only main's pull_request_target copy
    of llm-td.yml, run for a same-repo head, counts."""
    q = f"event=pull_request_target&head_sha={sha}&per_page=100"
    runs = [
        r
        for r in get(f"actions/workflows/llm-td.yml/runs?{q}")["workflow_runs"]
        if r["path"] == WORKFLOW
        and r["event"] == "pull_request_target"
        and r["head_sha"] == sha
        and (r["head_repository"] or {}).get("id") == r["repository"]["id"]
        # Label events other than ci-llm-td skip the job.
        and r["conclusion"] != "skipped"
    ]
    return sorted(runs, key=lambda r: (r["created_at"], r["id"]), reverse=True)


def download(run: dict, name: str) -> Selection | None:
    for a in get(f"actions/runs/{run['id']}/artifacts?name={name}")["artifacts"]:
        if a["name"] == name and not a["expired"]:
            with zipfile.ZipFile(io.BytesIO(fetch(a["archive_download_url"]))) as z:
                return Selection.from_json(json.loads(z.read(LLM_TD_SELECTION_FILE)))
    print(f"{run['html_url']} has no {name} artifact")
    return None


def resolve(pr: str, sha: str, enforce: bool) -> Selection | None:
    deadline, delay = time.monotonic() + WAIT_SECONDS, 15
    while True:
        runs = selector_runs(sha)
        if not enforce:
            run = next((r for r in runs if r["conclusion"] == "success"), None)
            break
        run = runs[0] if runs else None
        if run is None or run["status"] not in PENDING:
            break
        left = deadline - time.monotonic()
        if left <= 0:
            print(f"Gave up after {WAIT_SECONDS}s waiting for {run['html_url']}")
            return None
        print(f"Waiting for {run['html_url']} ({run['status']})")
        time.sleep(min(delay, left))
        delay = min(2 * delay, 60)
    if run is None:
        print(f"No llm-td.yml run for {sha}")
        return None
    if run["conclusion"] != "success":
        print(f"{run['html_url']} concluded {run['conclusion']}")
        return None
    print(f"Using the selection from {run['html_url']}")
    return download(run, f"llm_td_selection_{pr}_{sha}")


def main(output: Path = Path(LLM_TD_SELECTION_FILE)) -> None:
    sha = os.environ.get("HEAD_SHA", "")
    # ciflow/<label>/<PR number> tag pushes carry the PR in the ref.
    tag = re.fullmatch(r"refs/tags/ciflow/.+/(\d+)", os.environ.get("GITHUB_REF", ""))
    pr = os.environ.get("PR_NUMBER") or (tag and tag.group(1))
    if not sha or not pr:
        return
    info = get(f"pulls/{pr}")
    if (info["head"]["repo"] or {}).get("full_name") != os.environ["GITHUB_REPOSITORY"]:
        return
    # The PR's current labels, not the event's: a tag push carries none, and
    # removing ci-llm-td must stop enforcement even at an already selected head.
    labels = {label["name"] for label in info["labels"]}
    enforce = (
        os.environ.get("TD_ENABLED") == "true"
        and "ci-llm-td" in labels
        and "ci-no-td" not in labels
    )
    selection = resolve(pr, sha, enforce)
    if selection is not None:
        # select_tests clears enforce when its model call failed.
        selection.enforce = enforce and selection.enforce
        output.write_text(json.dumps(selection.to_json()))
        print(f"Resolved the LLM test selection, enforce={selection.enforce}")


if __name__ == "__main__":
    main()
