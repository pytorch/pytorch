"""Score selector results: recall of real failures, and the share of CI test time selected.

    python -m tools.testing.llm_td.eval.score --data DIR --ids DIR/ids_test.txt sonnet5_medium opus55_high td

A failure is a (test file, CI config) pair. It counts as caught when the
selection names the file, requests the config as a sweep, or runs
everything. Cost is the share of one full CI pass's CPU and GPU test-seconds
(from test-times.json) that the selection would run. The tag `td` scores the
replayed in-repo TD (see td_replay.py) at its CI cut of the top 25% of files.
"""

from __future__ import annotations

import argparse
import collections
import functools
import json
import re
import statistics as st
from pathlib import Path

from tools.testing.llm_td.selection import Selection


GPU_ENV = re.compile(r"cuda|rocm|sm\d+|xpu|h100|b200|a100|gpu", re.IGNORECASE)
CONFIDENCE = {"high": 0, "medium": 1, "low": 2}


def config_of(job: str) -> str:
    """'linux-jammy-py3.10-clang21 / test (crossref, 1, 2, lf.linux.2xlarge)' -> 'crossref'"""
    m = re.search(r"\(([^,()]+), \d+, \d+", job) or re.search(r"\(([^,()]+)\)", job)
    return m.group(1).strip() if m else job.strip()


@functools.cache
def test_seconds(path: Path) -> dict[str, dict[str, tuple[float, float]]]:
    """config -> file -> (cpu, gpu) seconds of one full CI pass."""
    out: dict = collections.defaultdict(
        lambda: collections.defaultdict(lambda: [0.0, 0.0])
    )
    for env, configs in json.loads(path.read_text()).items():
        gpu = int(bool(GPU_ENV.search(env)))
        for c, files in configs.items():
            for f, s in files.items():
                out[c][f][gpu] += s
    return out


def truth_pairs(s: dict, strong: bool) -> list[tuple[str, str]]:
    truth: list[str] = s["truth"]
    if strong and s["kind"] == "pr":
        truth = s.get("truth_strong", truth)
    cands = set(s["candidates"])
    pairs = set()
    for f in truth:
        if f in cands:
            jobs = s.get("truth_jobs", {}).get(f) or ["(default)"]
            pairs |= {(f, config_of(j)) for j in jobs}
    return sorted(pairs)


def policy(
    name: str, s: dict, r: dict | None, td: dict | None, file_seconds: dict[str, float]
) -> Selection | None:
    cands = set(s["candidates"])
    if name == "td25":
        return None if td is None else Selection(set(td["top25"]))
    out = (r or {}).get("out")
    if name == "model":
        return (
            Selection(run_all=True)
            if out is None
            else Selection.from_model_output(out, cands, [])
        )
    if name.startswith("final+td"):
        # The LLM selection plus TD's highest-ranked files that fit in N% of the test time,
        # the same hedge run_test.py applies when enforcing.
        if td is None:
            return None
        sel = Selection.from_model_output(out, cands, s["changed_files"])
        budget = sum(file_seconds.get(f, 0.0) for f in cands) * int(name[8:]) / 100
        for f in td["ranked"]:
            if file_seconds.get(f, 0.0) <= budget:
                budget -= file_seconds.get(f, 0.0)
                sel.files.add(f)
        return sel
    if name == "final-hm" and out is not None:
        out = {
            **out,
            "tests": [
                t for t in out["tests"] if CONFIDENCE.get(t.get("confidence"), 2) <= 1
            ],
        }
    return Selection.from_model_output(out, cands, s["changed_files"])


def cost_fraction(
    sel: Selection, cands: set[str], seconds: dict
) -> tuple[float, float]:
    if sel.run_all:
        return 1.0, 1.0
    tot, run = [0.0, 0.0], [0.0, 0.0]
    for c, files in seconds.items():
        for f, (cpu, gpu) in files.items():
            if f in cands:
                tot[0] += cpu
                tot[1] += gpu
                if sel.covers(f, c):
                    run[0] += cpu
                    run[1] += gpu
    return run[0] / max(tot[0], 1), run[1] / max(tot[1], 1)


def score(
    data: Path, ids: list[str], tag: str, policies: list[str], strong: bool
) -> list[dict]:
    seconds = test_seconds(data / "test-times.json")
    file_seconds: dict[str, float] = collections.Counter()
    for files in seconds.values():
        for f, (cpu, gpu) in files.items():
            file_seconds[f] += cpu + gpu
    rows = []
    for sid in ids:
        s = json.loads((data / "samples" / f"{sid}.json").read_text())
        rp, tp = data / "results" / tag / f"{sid}.json", data / "td" / f"{sid}.json"
        r = json.loads(rp.read_text()) if rp.exists() else None
        td = json.loads(tp.read_text()) if tp.exists() else None
        if tag != "td" and r is None:
            continue
        pairs = truth_pairs(s, strong)
        cands = set(s["candidates"])
        for pol in policies:
            sel = policy(pol, s, r, td, file_seconds)
            if sel is None:
                continue
            hit = [sel.covers(f, c) for f, c in pairs]
            cpu, gpu = cost_fraction(sel, cands, seconds)
            rows.append({
                "tag": tag, "policy": pol, "id": sid, "kind": s["kind"], "pairs": len(pairs), "hit": sum(hit),
                "cpu": cpu, "gpu": gpu, "run_all": sel.run_all, "files": len(sel.files),
                "cost_usd": (r or {}).get("cost_usd") or 0.0, "error": bool(r and r["error"]),
                "missed": [p for p, h in zip(pairs, hit) if not h],
            })  # fmt: skip
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    groups = collections.defaultdict(list)
    for x in rows:
        groups[(x["tag"], x["policy"], x["kind"])].append(x)
    out = []
    for (tag, pol, kind), xs in groups.items():
        failing = [x for x in xs if x["pairs"]]
        out.append({
            "tag": tag, "policy": pol, "kind": kind, "n": len(xs), "n_failing": len(failing),
            "caught_any": sum(x["hit"] > 0 for x in failing) / max(len(failing), 1),
            "caught_all": sum(x["hit"] == x["pairs"] for x in failing) / max(len(failing), 1),
            "pair_recall": sum(x["hit"] for x in failing) / max(sum(x["pairs"] for x in failing), 1),
            "cpu": st.mean(x["cpu"] for x in xs), "gpu": st.mean(x["gpu"] for x in xs),
            "run_all": sum(x["run_all"] for x in xs), "median_files": st.median(x["files"] for x in xs),
            "usd_per_call": st.mean(x["cost_usd"] for x in xs), "errors": sum(x["error"] for x in xs),
        })  # fmt: skip
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--ids", type=Path, required=True)
    p.add_argument("--policies", default="model,final,final-hm")
    p.add_argument(
        "--all-truth",
        action="store_true",
        help="score PRs on every truth, not only reproduced ones",
    )
    p.add_argument("--json", type=Path, help="also write per-sample rows here")
    p.add_argument("tags", nargs="+")
    args = p.parse_args()
    ids = args.ids.read_text().split()
    rows = []
    for tag in args.tags:
        rows += score(
            args.data,
            ids,
            tag,
            ["td25"] if tag == "td" else args.policies.split(","),
            not args.all_truth,
        )
    for x in summarize(rows):
        print(
            f"{x['tag']:24s} {x['policy']:9s} {x['kind']:7s} n={x['n']:3d} caught={x['caught_any']:6.1%} "
            f"all={x['caught_all']:6.1%} pairs={x['pair_recall']:6.1%} cpu={x['cpu']:6.1%} gpu={x['gpu']:6.1%} "
            f"run_all={x['run_all']:3d} files={x['median_files']:4.0f} $/call={x['usd_per_call']:.3f} err={x['errors']}"
        )
    if args.json:
        args.json.write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
