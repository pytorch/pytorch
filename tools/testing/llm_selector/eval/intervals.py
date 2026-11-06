"""Paired 95% intervals and McNemar p-values for new minus old on the headline metrics, and model-call latency.

    python -m tools.testing.llm_selector.eval.intervals rows_eval.json rows_head.json [--new TAG:POLICY] [--old TAG:POLICY]
    python -m tools.testing.llm_selector.eval.intervals --data DIR --latency TAG [--ids IDS]

ROWS are score.py --json outputs (the revert rows and the head rows). Reverts
are resampled per commit; failing PR pushes are resampled per PR (a cluster
bootstrap, since pushes of one PR are correlated) and, for reference, per
push. A revert counts as caught when any failure is selected, a push only
when every one is, as in chart.caught. Also prints the discordant counts and
an exact two-sided McNemar p-value. --latency prints the p50 and p90 of
run_eval's wall_s (the last attempt's latency) over TAG's results on IDS,
timed-out calls included.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import statistics as st
from pathlib import Path

from tools.testing.llm_selector.eval.chart import PRODUCTION
from tools.testing.llm_selector.eval.dollars import current_results


def outcomes(rows: list[dict], tag: str, pol: str, kind: str) -> dict[str, bool]:
    """id -> whether `tag` `pol` caught the failing `kind` sample."""
    return {
        r["id"]: r["hit"] > 0 if kind == "revert" else r["hit"] == r["pairs"]
        for r in rows
        if (r["tag"], r["policy"], r["kind"]) == (tag, pol, kind) and r["pairs"]
    }


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value for `b` and `c` discordant pairs."""
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def bootstrap(
    clusters: dict[str, list[int]], b: int, rng: random.Random
) -> tuple[float, float]:
    """95% percentile interval, in points, of the mean difference over `b` cluster resamples."""
    cs = list(clusters.values())
    stats = []
    for _ in range(b):
        pick = [cs[rng.randrange(len(cs))] for _ in cs]
        stats.append(100 * sum(sum(c) for c in pick) / sum(len(c) for c in pick))
    stats.sort()
    return stats[int(0.025 * b)], stats[int(0.975 * b) - 1]


def compare(rows: list[dict], new: list[str], old: list[str], b: int, seed: int) -> list[str]:  # fmt: skip
    """One line per headline metric comparing (tag, policy) `new` with `old` on the ids both scored, with `b` resamples."""
    rng = random.Random(seed)
    lines = []
    for kind, label in (("revert", "reverts caught"), ("head", "failing pushes, every failure")):  # fmt: skip
        n, o = (outcomes(rows, t, p, kind) for t, p in (new, old))
        ids = sorted(n.keys() & o.keys())
        if not ids:
            raise SystemExit(f"no paired failing {kind} rows of {':'.join(new)} and {':'.join(old)}")  # fmt: skip
        d = {i: int(n[i]) - int(o[i]) for i in ids}
        nb, nc = sum(v == 1 for v in d.values()), sum(v == -1 for v in d.values())
        line = (
            f"{label}: n={len(ids)} new={100 * sum(n[i] for i in ids) / len(ids):.1f}% "
            f"old={100 * sum(o[i] for i in ids) / len(ids):.1f}% new-old={100 * sum(d.values()) / len(ids):+.1f} pts "
            f"discordant new-only={nb} old-only={nc} McNemar p={mcnemar(nb, nc):.3f}"
        )
        lo, hi = bootstrap({i: [v] for i, v in d.items()}, b, rng)
        line += f" | 95% CI per {'commit' if kind == 'revert' else 'push'} [{lo:+.1f}, {hi:+.1f}]"  # fmt: skip
        if kind == "head":
            by_pr = collections.defaultdict(list)
            for i, v in d.items():
                # head_<pr>_<sha>
                by_pr[i.split("_")[1]].append(v)
            lo, hi = bootstrap(by_pr, b, rng)
            line += f" | per PR ({len(by_pr)} PRs) [{lo:+.1f}, {hi:+.1f}]"
        lines.append(line)
    return lines


def latency(data: Path, tag: str, ids: list[str]) -> str:
    """p50 and p90 of `tag`'s wall_s on its current results on `ids` (see dollars.current_results): the last attempt's latency, without earlier attempts or their backoff.

    Timed-out calls count, at the time they were cut off."""
    walls = sorted(r["wall_s"] for r in current_results(data, tag, ids, data / "samples"))  # fmt: skip
    if len(walls) < 2:
        raise SystemExit(f"fewer than 2 {tag} results on the ids")
    q = st.quantiles(walls, n=10, method="inclusive")
    return f"{tag} latency: n={len(walls)} p50={st.median(walls):.0f}s p90={q[8]:.0f}s"


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("rows", nargs="*", type=Path, help="score.py --json rows")
    p.add_argument("--new", default=":".join(PRODUCTION), help="TAG:POLICY")
    p.add_argument("--old", default="td:td25", help="TAG:POLICY")
    p.add_argument("--b", type=int, default=10000, help="bootstrap resamples")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", type=Path, help="run_eval.py's --data, for --latency")
    p.add_argument("--latency", metavar="TAG", help="print TAG's call latency instead")
    p.add_argument("--ids", type=Path, default=Path(__file__).parent / "ids/eval.txt", help="--latency's samples")  # fmt: skip
    args = p.parse_args()
    if args.b < 40:
        p.error(f"--b {args.b}: a 95% interval needs at least 40 resamples")
    if args.latency:
        if args.data is None:
            p.error("--latency needs --data")
        print(latency(args.data, args.latency, args.ids.read_text().split()))
        return
    if not args.rows:
        p.error("no rows")
    if bad := [x for x in (args.new, args.old) if x.count(":") != 1]:
        p.error(f"--new and --old are TAG:POLICY, got {bad[0]!r}")
    rows = [r for f in args.rows for r in json.loads(f.read_text())]
    for line in compare(rows, args.new.split(":"), args.old.split(":"), args.b, args.seed):  # fmt: skip
        print(line)


if __name__ == "__main__":
    main()
