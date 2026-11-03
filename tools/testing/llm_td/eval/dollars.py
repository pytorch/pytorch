"""Monthly runner-dollar model of replacing TD with an LLM selection policy.

    python -m tools.testing.llm_td.eval.dollars --data DIR --td-dir DIR/td v2_opus55_low final+td5 [--overhead 0.2]

Selection fractions come from the neutral samples (random landed main
commits), the closest thing to a typical PR push. Each workflow's 30-day PR
spend is rescaled by the job time the policy runs relative to replayed TD's
top-25% cut (or to running everything, where TD is off), where a job's time
is a fixed `overhead` share (setup, build download) plus the rest scaling
with the share of test time run. Inductor workflows only count changes that
the ciflow/inductor labeler rule would trigger them on.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import statistics as st
from math import isnan, nan
from pathlib import Path

from tools.testing.llm_td.eval import REPO
from tools.testing.llm_td.eval.score import (
    ci_jobs,
    cost_fraction,
    load,
    policy,
    runs_td,
    test_seconds,
)


# 30-day runner cost of test jobs ($k), the PR share of test hours, and whether
# TD runs in that workflow today. Inductor is the run_test part only. Only the
# TD workflows enforce the selection in this stack; the rest are follow-ups.
WORKFLOWS = [
    ("pull", "cpu", 463.6, 0.81, True),
    ("trunk", "cpu", 157.0, 0.63, True),
    ("trunk", "gpu", 158.0, 0.63, True),
    ("inductor (unit tests)", "ind_cpu", 79.0 * 0.83, 0.78, False),
    ("inductor (unit tests)", "ind_gpu", 54.5 * 0.38, 0.78, False),
    ("periodic", "gpu", 17.0, 0.31, False),
    ("slow", "cpu", 6.3, 0.07, False),
]
PR_PUSHES_PER_MONTH_K = 11.4
# The samples the report is scored on (eval) and the prompt was developed on (dev).
IDS = Path(__file__).parent / "ids"


def inductor_globs() -> list[str]:
    block = (
        (REPO / ".github/labeler.yml")
        .read_text()
        .split('"ciflow/inductor":')[1]
        .split("\n\n")[0]
    )
    return [
        line.strip()[2:].replace("**", "*")
        for line in block.splitlines()
        if line.strip().startswith("- ")
    ]


def fractions(
    data: Path, ids: list[str], tag: str, pol: str, samples: Path, td_dir: Path
) -> dict[str, float]:
    seconds = test_seconds(data / "test-times.json")
    globs = inductor_globs()
    # The configs of inductor.yml's unit-test job (inductor-unittest.yml) that TD never runs.
    unit = {c for n, c in ci_jobs()[False] if n.startswith("unit-test / ")}
    inductor = {c for c in unit if not runs_td(c)}
    gen, ind = [], []
    for sid in ids:
        s, r, td = load(data, tag, sid, samples, td_dir)
        choice = None if tag != "td" and r is None else policy(pol, s, r, td, seconds)
        if choice is None:
            continue
        cands = set(s["candidates"])
        gen.append(cost_fraction(choice, cands, seconds, runs_td))
        if any(fnmatch.fnmatch(f, g) for f in s["changed_files"] for g in globs):
            ind.append(cost_fraction(choice, cands, seconds, inductor.__contains__))
    if not gen:
        raise SystemExit(f"no {tag} results for the given ids")
    if not ind:
        print(
            f"warning: no {tag} sample matches the ciflow/inductor globs, so the inductor rows are skipped"
        )
    return {
        "cpu": st.mean(x[0] for x in gen), "gpu": st.mean(x[1] for x in gen),
        "ind_cpu": st.mean(x[0] for x in ind) if ind else nan, "ind_gpu": st.mean(x[1] for x in ind) if ind else nan,
        "n": len(gen), "n_ind": len(ind),
    }  # fmt: skip


def mean_call_usd(data: Path, tag: str, ids: list[str]) -> float:
    paths = [p for i in ids if (p := data / "results" / tag / f"{i}.json").exists()]
    if not paths:
        raise SystemExit(f"no {tag} results for the cost ids")
    return st.mean(json.loads(p.read_text()).get("cost_usd") or 0.0 for p in paths)


def model(
    data: Path,
    tag: str,
    pol: str,
    overhead: float,
    ids: list[str],
    cost_ids: list[str],
    samples: Path,
    td_dir: Path,
) -> dict:
    f = fractions(data, ids, tag, pol, samples, td_dir)
    ftd = fractions(data, ids, "td", "td25", samples, td_dir)
    total = {"cpu": 0.0, "gpu": 0.0}
    stack = 0.0
    rows = []
    for wf, k, cost, share, has_td in WORKFLOWS:
        if isnan(f[k]):
            continue
        pr = cost * share
        new_rel = overhead + (1 - overhead) * f[k]
        old_rel = overhead + (1 - overhead) * ftd[k] if has_td else 1.0
        save = pr * (1 - new_rel / old_rel)
        hw = "gpu" if "gpu" in k else "cpu"
        total[hw] += save
        stack += save if has_td else 0.0
        rows.append(
            {
                "workflow": wf,
                "hw": hw,
                "pr_k": pr,
                "fraction": f[k],
                "old_rel": old_rel,
                "new_rel": new_rel,
                "save_k": save,
            }
        )
    llm_k = PR_PUSHES_PER_MONTH_K * mean_call_usd(data, tag, cost_ids)
    return {"tag": tag, "policy": pol, "overhead": overhead, "fractions": f, "td_fractions": ftd, "rows": rows,
            "total_k_per_month": total, "stack_k_per_month": stack, "llm_k_per_month": llm_k}  # fmt: skip


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data", type=Path, required=True)
    p.add_argument(
        "--ids",
        type=Path,
        help="samples the fractions come from; default: the report's neutral samples",
    )
    p.add_argument(
        "--cost-ids",
        type=Path,
        help="samples $/call is averaged over; default: the report's eval samples",
    )
    p.add_argument("--samples-dir", type=Path)
    p.add_argument("--td-dir", type=Path)
    p.add_argument("--overhead", type=float, default=0.2)
    p.add_argument("--json", type=Path, help="also write the per-workflow rows here")
    p.add_argument("tag")
    p.add_argument("policy")
    args = p.parse_args()
    ids = (args.ids or IDS / "neutral.txt").read_text().split()
    cost_ids = (args.cost_ids or IDS / "eval.txt").read_text().split()
    d = model(args.data, args.tag, args.policy, args.overhead, ids, cost_ids,
              args.samples_dir or args.data / "samples", args.td_dir or args.data / "td")  # fmt: skip
    f, t = d["fractions"], d["total_k_per_month"]
    print(
        f"{d['tag']} {d['policy']} overhead={d['overhead']} n={f['n']} n_ind={f['n_ind']} "
        f"run cpu={f['cpu']:.3f} gpu={f['gpu']:.3f} ind_cpu={f['ind_cpu']:.3f} ind_gpu={f['ind_gpu']:.3f} "
        f"save cpu=${t['cpu']:.0f}k gpu=${t['gpu']:.0f}k (this stack ${d['stack_k_per_month']:.0f}k) llm=${d['llm_k_per_month']:.1f}k/month"
    )
    if args.json:
        args.json.write_text(json.dumps(d, indent=1))


if __name__ == "__main__":
    main()
