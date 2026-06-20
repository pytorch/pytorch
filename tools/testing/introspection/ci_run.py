"""CI entry point: compute the test diff for the current PR and write a result JSON.

Runs inside the build container (wheel-installed torch), as the *parent* process
(imports no torch), via:
    python -m tools.testing.introspection.ci_run

Env:
  PR_NUMBER          the PR number (carried into the artifact for the comment stage)
  HEAD_SHA           the PR head sha (diff `to`)
  BASE_REF           the PR base branch (e.g. main); fetched to find the merge-base
  TESTINTRO_OUTPUT   output path (default test-diff-result.json)
  TESTINTRO_PLATFORMS  optional comma-separated platform names (default: all)

Writes {"pr", "skipped", ...diff()...}. If the PR changes nothing test-relevant it
writes a skip result so the comment stage can no-op.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

from tools.testing.introspection import diff as diff_mod, platforms


def main() -> int:
    t0 = time.perf_counter()
    pr = int(os.environ["PR_NUMBER"])
    head = os.environ["HEAD_SHA"]
    base_ref = os.environ["BASE_REF"]
    out = os.environ.get("TESTINTRO_OUTPUT", "test-diff-result.json")
    repo = str(diff_mod.collector.REPO)

    # Make the base reachable, then find the fork point.
    subprocess.run(
        ["git", "fetch", "--no-tags", "--quiet", "origin", base_ref],
        cwd=repo,
        check=True,
    )
    merge_base = subprocess.run(
        ["git", "merge-base", f"origin/{base_ref}", head],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    # Pre-check: skip unless the PR can change which tests exist.
    changed = diff_mod._changed_files(merge_base, head)
    broad = any(diff_mod._is_broad(f) for f in changed)
    relevant = broad or any(diff_mod._is_test_py(f) for f in changed)
    if not relevant:
        result: dict = {"pr": pr, "skipped": True}
    else:
        names = os.environ.get("TESTINTRO_PLATFORMS")
        plat_list = names.split(",") if names else sorted(platforms.REGISTRY)
        # A broad change (generation surface) expands the scope to the whole selected
        # set; running that across every platform would blow the CI timeout, so limit
        # broad diffs to cpu only and flag it for the comment.
        if broad:
            plat_list = ["linux-cpu"]
        jobs = [platforms.get_job(p.strip()) for p in plat_list]
        result = {
            "pr": pr,
            "skipped": False,
            "broad": broad,
            **diff_mod.diff(jobs, merge_base, head, locations=True),
        }
        result["elapsed_s"] = round(time.perf_counter() - t0, 1)

    with open(out, "w") as f:
        json.dump(result, f)
    print(f"wrote {out} (skipped={result['skipped']})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
