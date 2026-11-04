"""Replay the in-repo TD heuristics on each eval sample.

    python -m tools.testing.llm_selector.eval.td_replay --data DIR [--out DIR/td] [--shard I N]

--shard I N (0 <= I < N, 0-indexed unlike run_test.py's) processes every
N-th sample starting at the I-th.

Runs get_test_prioritizations over each sample's candidates, as the
target-determination job does, with the change (changed files, commit text)
patched in, the sample's `previous_failures` (head samples; see
build_dataset) as the PR's previous failures, and every other PR-local input
(LLM mappings, linked issue bodies) empty. A sample on which a heuristic raised
gets no file, and the run exits non-zero, so a rerun recomputes it; an
error in a trial_mode heuristic, which TD's aggregate leaves out, is ignored. Writes
OUT/<id>.json with:

  inputs     a hash of test-times.json, the sample and `stats` (see
             score.replay_inputs); a file with another hash is recomputed,
             and score.py drops one that does not match its inputs
  stats      a hash of the STATS artifacts and the replay code (see score.replay_stats)
  scored     files TD gives a positive score, best first (what the hedge draws from)
  top25      CI's top-25% cut over all candidates
  top25_cfg  config -> files of that cut per config of test-times.json, over
             the config's job test list (see score.job_files)

CI computes the scores once and each job cuts them to its own test list
(TestPrioritizations.amend_tests), so the per-config cut reuses the global
scores. A file counts as run in a config if any TestRun of it is in the cut,
which slightly favors TD when a heuristic scored only some of its classes.

The historical-stats artifacts are read from STATS (default
REPO/.additional_ci_files, written by get_test_file_ratings() and its siblings
in tools/stats/import_test_stats.py). This is optimistic for TD: today's
artifacts already contain the failures being predicted.
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

from tools.stats.import_test_stats import ADDITIONAL_CI_FILES_FOLDER
from tools.testing.llm_selector.eval import REPO, write_json
from tools.testing.llm_selector.eval.score import (
    job_files,
    replay_inputs,
    replay_stats,
    STATS_ARTIFACTS,
    test_seconds,
)


if TYPE_CHECKING:
    from tools.testing.target_determination.heuristics.interface import (
        TestPrioritizations,
    )


H = "tools.testing.target_determination.heuristics."
# The heuristics modules that call query_changed_files, which reads this clone's git.
CHANGED_FILES_READERS = ("utils", "edited_by_pr", "filepath", "historical_class_failure_correlation", "public_bindings")  # fmt: skip


def top25_per_config(
    tp: TestPrioritizations, jobs: dict[str, set[str]]
) -> dict[str, list[str]]:
    """The files of CI's top-25% cut in each config, cutting the scores to that config's job test list `jobs[c]` (see score.job_files)."""
    out = {}
    for c, files in jobs.items():
        job = copy.copy(tp)
        job.amend_tests(sorted(files))
        out[c] = sorted({t.test_file for t in job.get_top_per_tests(25)[0]})
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, help="default: DATA/td")
    p.add_argument("--stats", type=Path, default=REPO / ADDITIONAL_CI_FILES_FOLDER)
    p.add_argument("--ids", type=Path, help="only these samples")
    p.add_argument("--shard", type=int, nargs=2, default=[0, 1], metavar=("I", "N"), help="0-indexed: every N-th sample from the I-th")  # fmt: skip
    args = p.parse_args()
    which, n = args.shard
    if not 0 <= which < n:
        p.error(f"--shard I N needs 0 <= I < N, got {which} {n}")
    if args.stats.name != ADDITIONAL_CI_FILES_FOLDER.name:
        raise SystemExit(f"--stats must be a directory named {ADDITIONAL_CI_FILES_FOLDER}")  # fmt: skip
    missing = [a for a in STATS_ARTIFACTS if not (args.stats / a).exists()]
    if missing:
        raise SystemExit(
            f"{args.stats} lacks {missing}; fetch them with get_test_file_ratings(), get_td_heuristic_historial_edited_files_json() and get_td_heuristic_profiling_json() from tools/stats/import_test_stats.py"
        )

    from tools.testing.target_determination.determinator import get_test_prioritizations
    from tools.testing.target_determination.heuristics import HEURISTICS

    required = {h.name for h in HEURISTICS if not h.trial_mode}

    seconds = test_seconds(args.data / "test-times.json")
    times = (args.data / "test-times.json").read_bytes()
    stats = replay_stats(args.stats)
    current: dict = {}
    stats_root = args.stats.absolute().parent
    patches = {
        # The heuristics read their artifacts as REPO_ROOT / ".additional_ci_files/...".
        "utils.REPO_ROOT": stats_root,
        "historical_class_failure_correlation.REPO_ROOT": stats_root,
        "previously_failed_in_pr.get_previous_failures": lambda: set(
            current.get("previous_failures", ())
        ),
        "previously_failed_in_pr.read_additional_test_failures_file": set,
        "llm.LLM.get_mappings": lambda self: {},
        "mentioned_in_pr.get_git_commit_info": lambda: current["title"]
        + "\n"
        + (current["body"] or ""),
        "mentioned_in_pr.get_pr_number": lambda: None,
        "mentioned_in_pr.get_issue_or_pr_body": lambda n: "",
    }
    for m in CHANGED_FILES_READERS:
        patches[f"{m}.query_changed_files"] = lambda: current["changed_files"]

    out = args.out or args.data / "td"
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted((args.data / "samples").glob("*.json"))
    if args.ids:
        ids = set(args.ids.read_text().split())
        paths = [x for x in paths if x.stem in ids]
    failed = []
    with contextlib.ExitStack() as stack:
        for target, new in patches.items():
            stack.enter_context(mock.patch(H + target, new))
        for path in paths[which::n]:
            done, sample = out / path.name, path.read_bytes()
            inputs = replay_inputs(stats, times, sample)
            if done.exists() and json.loads(done.read_text()).get("inputs") == inputs:
                continue
            current.clear()
            current.update(json.loads(sample))
            cands = current["candidates"]
            log = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()):
                agg = get_test_prioritizations(cands, file=log)
            # get_test_prioritizations leaves out a heuristic that raised and goes on without it.
            if raised := sorted(required - set(agg.to_json())):
                errors = [x for x in log.getvalue().splitlines() if x.startswith("Error in ")]  # fmt: skip
                print(path.stem, f"raised: {raised}", *errors, sep="\n", flush=True)
                failed.append(path.stem)
                done.unlink(missing_ok=True)
                continue
            tp = agg.get_aggregated_priorities()
            scored = [t.test_file for t in tp.get_all_tests() if tp.get_score(t) > 0]
            write_json(done, {
                "inputs": inputs,
                "stats": stats,
                "scored": list(dict.fromkeys(scored)),
                "top25": sorted({t.test_file for t in tp.get_top_per_tests(25)[0]}),
                "top25_cfg": top25_per_config(tp, job_files(current, seconds)),
            })  # fmt: skip
    if failed:
        raise SystemExit(
            f"a heuristic raised on {len(failed)} samples, left for a rerun: {failed}"
        )


if __name__ == "__main__":
    main()
