"""Monthly runner-dollar model of replacing TD with an LLM selection policy.

    python -m tools.testing.llm_td.eval.dollars --data DIR --td-dir DIR/td v2_opus55_medium final+td20 [--overhead 0.2]

Selection fractions come from the neutral samples (random landed main
commits), the closest thing to a typical PR push, scored on the samples that
have both the policy's result and a TD replay. Each workflow's 30-day
same-repo PR spend (fork PRs never get a selection) is rescaled by the job
time the policy runs relative to replayed TD's top-25% cut (or to running
everything, where TD is off), where a job's time is a fixed `overhead` share
(setup, build download) plus the rest scaling with the share of test time
run. Inductor workflows only count changes that the ciflow/inductor labeler
rule would trigger them on. The constants and the queries behind them are in
REPORT.md's Reproducing section.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
from pathlib import Path

import yaml

from tools.testing.llm_td.eval import REPO
from tools.testing.llm_td.eval.score import (
    ci_jobs,
    cost_fraction,
    load,
    policy,
    runs_td,
    test_seconds,
)


# 30-day runner cost of test jobs ($k), the PR share of test hours, the fork-PR
# share of those PR hours, and whether TD runs in that workflow today.
# Inductor is the run_test part only. Only the TD workflows enforce the
# selection in this stack; the rest are follow-ups.
WORKFLOWS = [
    ("pull", "cpu", 463.6, 0.81, 0.36, True),
    ("trunk", "cpu", 157.0, 0.63, 0.46, True),
    ("trunk", "gpu", 158.0, 0.63, 0.46, True),
    ("inductor (unit tests)", "ind_cpu", 79.0 * 0.83, 0.78, 0.30, False),
    ("inductor (unit tests)", "ind_gpu", 54.5 * 0.38, 0.78, 0.30, False),
    ("periodic", "gpu", 17.0, 0.31, 0.37, False),
    ("slow", "cpu", 6.3, 0.07, 0.38, False),
]
# Same-repo PR heads with a pull run per month; fork PRs make no model call.
PR_PUSHES_PER_MONTH_K = 12.5
IDS = Path(__file__).parent / "ids"
LABELER = REPO / ".github/labeler.yml"


def inductor_globs(labeler: Path) -> re.Pattern[str]:
    """The ciflow/inductor labeler globs as one regex, with the labeler's
    minimatch semantics: `*` stays within a path segment, `**` crosses them."""
    globs = (yaml.safe_load(labeler.read_text()) or {}).get("ciflow/inductor")
    if not globs or not all(isinstance(g, str) for g in globs):
        raise SystemExit(f"{labeler} has no list of ciflow/inductor globs")
    parts = [
        re.escape(g)
        .replace(r"\*\*/", "(?:.*/)?")
        .replace(r"\*\*", ".*")
        .replace(r"\*", "[^/]*")
        .replace(r"\?", "[^/]")
        for g in globs
    ]
    return re.compile("|".join(parts))


def fractions(
    data: Path,
    ids: list[str],
    tag: str,
    pol: str,
    samples: Path,
    td_dir: Path,
    labeler: Path = LABELER,
) -> dict[str, float]:
    """Mean (cpu, gpu) share of test time `pol` runs on `ids`, which must all have its inputs."""
    seconds = test_seconds(data / "test-times.json")
    globs = inductor_globs(labeler)
    # The configs of inductor.yml's unit-test job (inductor-unittest.yml) that TD never runs.
    unit = {c for _, n, c in ci_jobs()[False] if n.startswith("unit-test / ")}
    inductor = {c for c in unit if not runs_td(c)}
    if not inductor:
        raise SystemExit("no inductor unit-test configs without TD in workflows")
    gen, ind = [], []
    for sid in ids:
        s, r, td = load(data, tag, sid, samples, td_dir)
        choice = None if tag != "td" and r is None else policy(pol, s, r, td, seconds)
        if choice is None:
            raise SystemExit(f"{sid} has no {tag} result or TD replay")
        cands = set(s["candidates"])
        gen.append(cost_fraction(choice, cands, seconds, runs_td))
        if any(globs.fullmatch(f) for f in s["changed_files"]):
            ind.append(cost_fraction(choice, cands, seconds, inductor.__contains__))
    if not gen:
        raise SystemExit(f"no {tag} samples")
    if not ind:
        raise SystemExit(f"no {tag} sample matches the ciflow/inductor globs")
    return {
        "cpu": st.mean(x[0] for x in gen), "gpu": st.mean(x[1] for x in gen),
        "ind_cpu": st.mean(x[0] for x in ind), "ind_gpu": st.mean(x[1] for x in ind),
        "n": len(gen), "n_ind": len(ind),
    }  # fmt: skip


def mean_call_usd(data: Path, tag: str, ids: list[str]) -> float:
    """Mean cost of the calls with a recorded cost; timed-out calls have none and are left out."""
    paths = [p for i in ids if (p := data / "results" / tag / f"{i}.json").exists()]
    costs = [c for p in paths if (c := json.loads(p.read_text()).get("cost_usd"))]
    if not costs:
        raise SystemExit(f"no {tag} results with a cost for the cost ids")
    if len(costs) < len(paths):
        print(f"{len(paths) - len(costs)} of {len(paths)} {tag} results have no cost")
    return st.mean(costs)


def model(
    data: Path,
    tag: str,
    pol: str,
    overhead: float,
    ids: list[str],
    cost_ids: list[str],
    samples: Path,
    td_dir: Path,
    labeler: Path = LABELER,
    workflows: list[tuple[str, str, float, float, float, bool]] = WORKFLOWS,
) -> dict:
    # New and old are averaged over the same samples.
    both = [
        i
        for i in ids
        if (td_dir / f"{i}.json").exists()
        and (tag == "td" or (data / "results" / tag / f"{i}.json").exists())
    ]
    if missing := len(ids) - len(both):
        print(f"{missing} of {len(ids)} ids lack a {tag} result or TD replay")
    f = fractions(data, both, tag, pol, samples, td_dir, labeler)
    ftd = fractions(data, both, "td", "td25", samples, td_dir, labeler)
    total = {"cpu": 0.0, "gpu": 0.0}
    stack = 0.0
    rows = []
    for wf, k, cost, share, fork, has_td in workflows:
        pr = cost * share * (1 - fork)
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
    p.add_argument("--labeler", type=Path, default=LABELER)
    p.add_argument("--overhead", type=float, default=0.2)
    p.add_argument("--json", type=Path, help="also write the per-workflow rows here")
    p.add_argument("tag")
    p.add_argument("policy")
    args = p.parse_args()
    ids = (args.ids or IDS / "neutral.txt").read_text().split()
    cost_ids = (args.cost_ids or IDS / "eval.txt").read_text().split()
    d = model(args.data, args.tag, args.policy, args.overhead, ids, cost_ids,
              args.samples_dir or args.data / "samples", args.td_dir or args.data / "td", args.labeler)  # fmt: skip
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
