"""Bring the LLM test selection of this run's PR head to the TD job.

llm_td_selection.yml writes one selection per PR head, as the artifact
llm_selection_<PR>_<head sha> of its run for that head. It is a
pull_request_target workflow, so nothing orders it before the test workflows:
this waits while its newest run for the head is pending, up to WAIT_SECONDS,
then writes the newest successful run's selection where the LLMSelection
heuristic reads it. Without a PR, a successful run or a selection it writes
nothing, and the heuristic raises: TD then runs without it.
"""

from __future__ import annotations

import io
import json
import os
import re
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
# heuristics/llm_selection.py's LLM_SELECTION_PATH; importing it would need the
# TD job's dependencies.
LLM_SELECTION_FILE = "llm_selection.json"
LLM_SELECTION_PATH = REPO_ROOT / ".additional_ci_files" / LLM_SELECTION_FILE
WORKFLOW_FILE = "llm_td_selection.yml"
WAIT_SECONDS = 600
PENDING = {"queued", "in_progress", "waiting", "requested", "pending"}
MAX_SELECTION_BYTES = 1 << 20


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    # Not forwarded to the blob storage an artifact download redirects to.
    req.add_unredirected_header("Authorization", f"Bearer {os.environ['GITHUB_TOKEN']}")
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()  # type: ignore[no-any-return]


def get(path: str) -> Any:
    repo = os.environ["GITHUB_REPOSITORY"]
    return json.loads(fetch(f"https://api.github.com/repos/{repo}/{path}"))


def selector_runs(sha: str) -> list[dict[str, Any]]:
    """llm_td_selection.yml's runs for this head, newest first. Any workflow can
    upload an artifact named like the selection, so only main's
    pull_request_target copy of llm_td_selection.yml, run for a same-repo head,
    counts."""
    q = f"event=pull_request_target&head_sha={sha}&per_page=100"
    runs = [
        r
        for r in get(f"actions/workflows/{WORKFLOW_FILE}/runs?{q}")["workflow_runs"]
        if r["path"] == f".github/workflows/{WORKFLOW_FILE}"
        and r["event"] == "pull_request_target"
        and r["head_sha"] == sha
        and (r["head_repository"] or {}).get("id") == r["repository"]["id"]
        # Fork and non-pytorch runs skip the job.
        and r["conclusion"] != "skipped"
    ]
    return sorted(runs, key=lambda r: (r["created_at"], r["id"]), reverse=True)


def download(run: dict[str, Any], name: str, sha: str) -> bytes | None:
    """The selection file of the run's artifact `name`, if it is for `sha`."""
    for a in get(f"actions/runs/{run['id']}/artifacts?name={name}")["artifacts"]:
        if a["name"] != name or a["expired"]:
            continue
        with zipfile.ZipFile(io.BytesIO(fetch(a["archive_download_url"]))) as z:
            if z.getinfo(LLM_SELECTION_FILE).file_size > MAX_SELECTION_BYTES:
                print(f"{run['html_url']}: the selection is too large")
                return None
            content = z.read(LLM_SELECTION_FILE)
        if json.loads(content).get("head_sha") != sha:
            print(f"{run['html_url']}: the selection is for another head")
            return None
        return content
    print(f"{run['html_url']} has no {name} artifact")
    return None


def resolve(pr: str, sha: str) -> bytes | None:
    deadline, delay = time.monotonic() + WAIT_SECONDS, 15
    while True:
        runs = selector_runs(sha)
        if not runs or runs[0]["status"] not in PENDING:
            break
        left = deadline - time.monotonic()
        if left <= 0:
            print(f"Gave up after {WAIT_SECONDS}s waiting for {runs[0]['html_url']}")
            break
        print(f"Waiting for {runs[0]['html_url']} ({runs[0]['status']})")
        time.sleep(min(delay, left))
        delay = min(2 * delay, 60)
    run = next((r for r in runs if r["conclusion"] == "success"), None)
    if run is None:
        print(f"No successful {WORKFLOW_FILE} run for {sha}")
        return None
    print(f"Using the selection from {run['html_url']}")
    return download(run, f"llm_selection_{pr}_{sha}", sha)


def main() -> None:
    LLM_SELECTION_PATH.unlink(missing_ok=True)
    sha = os.environ.get("HEAD_SHA", "")
    # ciflow/<label>/<PR number> tag pushes carry the PR in the ref.
    tag = re.fullmatch(r"refs/tags/ciflow/.+/(\d+)", os.environ.get("GITHUB_REF", ""))
    pr = os.environ.get("PR_NUMBER") or (tag.group(1) if tag else "")
    if not sha or not pr:
        print("Not a PR run: no LLM test selection")
        return
    try:
        selection = resolve(pr, sha)
    except Exception as e:
        # The selection is optional: a GitHub API error must not look like a TD failure.
        print(f"No LLM test selection: {type(e).__name__}: {e}")
        return
    if selection is not None:
        LLM_SELECTION_PATH.parent.mkdir(parents=True, exist_ok=True)
        LLM_SELECTION_PATH.write_bytes(selection)
        print(f"Wrote {LLM_SELECTION_PATH}")


if __name__ == "__main__":
    main()
