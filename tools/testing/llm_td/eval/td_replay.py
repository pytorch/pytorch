"""Replay the in-repo TD heuristics on each eval sample.

    python -m tools.testing.llm_td.eval.td_replay --data DIR [--out DIR/td] [--shard I N]

Runs get_test_prioritizations over each sample's candidates, as the
target-determination job does, with the change (changed files, commit text)
patched in and every PR-local input (previous failures, the pytest cache, LLM
mappings, linked issue bodies) empty. Writes OUT/<id>.json with:

  scored     files TD gives a positive score, best first (what the hedge draws from)
  top25      CI's top-25% cut over all candidates
  top25_cfg  config -> files of that cut per config of test-times.json

CI computes the scores once and each job cuts them to its own test list
(TestPrioritizations.amend_tests), so the per-config cut reuses the global
scores. A file counts as run in a config if any TestRun of it is in the cut,
which slightly favors TD when a heuristic scored only some of its classes.

The historical-stats artifacts are read from STATS (default
REPO/.additional_ci_files, fetched by tools/stats/import_test_stats.py). This
is optimistic for TD: today's artifacts already contain the failures being
predicted.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

from tools.testing.llm_td.eval import REPO
from tools.testing.llm_td.eval.score import Seconds, test_seconds


if TYPE_CHECKING:
    from tools.testing.target_determination.heuristics.interface import (
        TestPrioritizations,
    )


H = "tools.testing.target_determination.heuristics."
ARTIFACTS = [
    "test-file-ratings.json",
    "td_heuristic_historical_edited_files.json",
    "td_heuristic_profiling.json",
]


def top25_per_config(
    tp: TestPrioritizations, cands: list[str], seconds: Seconds
) -> dict[str, list[str]]:
    """The files of CI's top-25% cut in each config, cutting the scores to the candidates in that config."""
    out = {}
    for c, files in seconds.items():
        job = copy.deepcopy(tp)
        job.amend_tests([f for f in cands if f in files])
        out[c] = sorted({t.test_file for t in job.get_top_per_tests(25)[0]})
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, help="default: DATA/td")
    p.add_argument("--stats", type=Path, default=REPO / ".additional_ci_files")
    p.add_argument("--ids", type=Path, help="only these samples")
    p.add_argument("--shard", type=int, nargs=2, default=[0, 1])
    args = p.parse_args()
    if args.stats.name != ".additional_ci_files":
        raise SystemExit("--stats must be a directory named .additional_ci_files")
    missing = [a for a in ARTIFACTS if not (args.stats / a).exists()]
    if missing:
        raise SystemExit(
            f"{args.stats} lacks {missing}; fetch them with tools/stats/import_test_stats.py"
        )

    from tools.testing.target_determination.determinator import get_test_prioritizations

    seconds = test_seconds(args.data / "test-times.json")
    current: dict = {}
    stats_root = args.stats.resolve().parent
    patches = {
        # The heuristics read their artifacts as REPO_ROOT / ".additional_ci_files/...".
        "utils.REPO_ROOT": stats_root,
        "historical_class_failure_correlation.REPO_ROOT": stats_root,
        "previously_failed_in_pr.get_previous_failures": set,
        "previously_failed_in_pr.read_additional_test_failures_file": set,
        "llm.LLM.get_mappings": lambda self: {},
        "mentioned_in_pr.get_git_commit_info": lambda: current["title"]
        + "\n"
        + (current["body"] or ""),
        "mentioned_in_pr.get_pr_number": lambda: None,
        "mentioned_in_pr.get_issue_or_pr_body": lambda n: "",
    }
    for m in [
        "utils",
        "edited_by_pr",
        "filepath",
        "historical_class_failure_correlation",
        "public_bindings",
    ]:
        patches[f"{m}.query_changed_files"] = lambda: current["changed_files"]

    out = args.out or args.data / "td"
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted((args.data / "samples").glob("*.json"))
    if args.ids:
        ids = set(args.ids.read_text().split())
        paths = [x for x in paths if x.stem in ids]
    which, n = args.shard
    with contextlib.ExitStack() as stack:
        for target, new in patches.items():
            stack.enter_context(mock.patch(H + target, new))
        for path in paths[which::n]:
            if (out / path.name).exists():
                continue
            current.clear()
            current.update(json.loads(path.read_text()))
            cands = current["candidates"]
            log = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()):
                tp = get_test_prioritizations(
                    cands, file=log
                ).get_aggregated_priorities()
            for line in log.getvalue().splitlines():
                if line.startswith("Error in "):
                    print(path.stem, line, flush=True)
            scored = [t.test_file for score, t in tp._traverse_scores() if score > 0]
            (out / path.name).write_text(json.dumps({
                "scored": list(dict.fromkeys(scored)),
                "top25": sorted({t.test_file for t in tp.get_top_per_tests(25)[0]}),
                "top25_cfg": top25_per_config(tp, cands, seconds),
            }))  # fmt: skip


if __name__ == "__main__":
    main()
