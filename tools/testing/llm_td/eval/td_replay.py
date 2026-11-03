"""Replay the in-repo TD heuristics on each eval sample.

    python -m tools.testing.llm_td.eval.td_replay --data DIR [--shard I N]

Feeds each sample's changed files and commit text to HEURISTICS and writes
DIR/td/<id>.json with the ranked files and the top 25% that CI would run.
This is optimistic for TD: it uses today's historical-failure artifacts,
which already contain the failures being predicted.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--shard", type=int, nargs=2, default=[0, 1])
    args = p.parse_args()
    data = args.data.resolve()
    # The heuristics read their artifacts relative to the repo root.
    os.chdir(REPO)
    from tools.stats import import_test_stats as stats

    for fetch in (stats.get_test_times, stats.get_test_class_times, stats.get_test_file_ratings,
                  stats.get_test_class_ratings, stats.get_td_heuristic_historial_edited_files_json,
                  stats.get_td_heuristic_profiling_json):  # fmt: skip
        fetch()

    import tools.testing.target_determination.heuristics.utils as utils
    from tools.testing.target_determination.heuristics import HEURISTICS
    from tools.testing.target_determination.heuristics.interface import (
        AggregatedHeuristics,
    )

    current: dict = {}
    for m in [sys.modules[type(h).__module__] for h in HEURISTICS] + [utils]:
        # The heuristics read the change through these module-level helpers.
        stubs = {
            "query_changed_files": lambda: current["changed_files"],
            "get_git_commit_info": lambda: current["title"]
            + "\n"
            + (current["body"] or ""),
            "get_pr_number": lambda: None,
        }
        for name, fn in stubs.items():
            if hasattr(m, name):
                setattr(m, name, fn)

    out = data / "td"
    out.mkdir(exist_ok=True)
    which, n = args.shard
    for i, path in enumerate(sorted((data / "samples").glob("*.json"))):
        if i % n != which or (out / path.name).exists():
            continue
        current.clear()
        current.update(json.loads(path.read_text()))
        agg = AggregatedHeuristics(current["candidates"])
        for h in HEURISTICS:
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    agg.add_heuristic_results(
                        h, h.get_prediction_confidence(current["candidates"])
                    )
            except Exception as e:
                print(path.stem, type(h).__name__, "failed:", e)
        prioritizations = agg.get_aggregated_priorities()
        ranked = list(
            dict.fromkeys(t.test_file for _, t in prioritizations._traverse_scores())
        )
        include, _ = prioritizations.get_top_per_tests(25)
        (out / path.name).write_text(
            json.dumps(
                {"ranked": ranked, "top25": sorted({t.test_file for t in include})}
            )
        )


if __name__ == "__main__":
    main()
