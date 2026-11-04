"""Monthly runner-dollar model of replacing TD with an LLM selection policy.

    python -m tools.testing.llm_td.eval.dollars --data DIR v3_opus55_medium final+td20 [--overhead 0.2]

The selection fractions are score.score's rows on the neutral samples (random
landed main commits), the closest thing to a typical PR push, for the policy
and for TD's replayed top-25% cut on the same samples, so they are the
Accuracy table's typical-change shares. Each workflow's 30-day PR spend is cut
to its same-repo share (fork PRs never get a selection) and rescaled by the
job time the policy runs relative to TD's cut (or to running everything,
where TD is off), where a job's time is a fixed `overhead` share (setup,
build download) plus the rest scaling with the share of test time run.
Inductor workflows count only the samples whose changes the ciflow/inductor
labeler rule triggers them on, and only the inductor unit-test configs' time.
The model-call cost is PR_PUSHES_PER_MONTH_K times the mean cost of a call on
an asked eval sample. The constants and the queries behind them are in
REPORT.md's Reproducing section.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from pathlib import Path

import numpy as np
import yaml

from tools.testing.llm_td.eval import REPO, score
from tools.testing.llm_td.eval.run_eval import TIMED_OUT
from tools.testing.llm_td.select_tests import test_minutes


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
    """The ciflow/inductor labeler globs as one regex: `*` and `?` stay within a
    path segment and `**` crosses them. Other glob syntax ({}, [], !) is
    rejected rather than matched literally."""
    config = yaml.safe_load(labeler.read_text())
    globs = config.get("ciflow/inductor") if isinstance(config, dict) else None
    if not isinstance(globs, list) or not globs or not all(isinstance(g, str) for g in globs):  # fmt: skip
        raise SystemExit(f"{labeler} has no list of ciflow/inductor globs")
    if bad := [g for g in globs if set(g) & set("{}[]!")]:
        raise SystemExit(f"{labeler} has ciflow/inductor globs with unsupported syntax: {bad}")  # fmt: skip
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
    tag: str,
    pol: str,
    *,
    ids: list[str],
    samples: Path,
    td_dir: Path,
    stats: str,
    labeler: Path = LABELER,
) -> tuple[dict, dict]:
    """The mean (cpu, gpu) shares of test time `pol` of `tag` and TD's cut run, from score.score's rows on `ids`.

    The inductor shares ("ind_cpu", "ind_gpu") are over the rows whose change
    the ciflow/inductor labeler rule matches, of the time of inductor.yml's
    unit-test configs (inductor-unittest.yml) that TD never runs."""
    rows = score.score(data, ids, list(dict.fromkeys([tag, "td"])), [pol], strong=True, samples=samples, td_dir=td_dir, stats=stats)  # fmt: skip
    seconds = score.test_seconds(data / "test-times.json")
    fallback = score.fallback_seconds(data / "test-times.json")
    globs = inductor_globs(labeler)
    unit = {c for w, n, c in score.ci_jobs()[False] if w == "inductor" and n.startswith("unit-test / ")}  # fmt: skip
    inductor = {c for c in unit if not score.runs_td(c)}
    if not inductor:
        raise SystemExit("no inductor unit-test configs without TD in workflows")
    out = {}
    for t, p in dict.fromkeys([(tag, pol), ("td", "td25")]):
        xs = [x for x in rows if (x["tag"], x["policy"]) == (t, p)]
        if not xs:
            raise SystemExit(f"no scored {t} {p} rows")
        ind = []
        for x in xs:
            s, r, td = score.load(data, t, x["id"], samples, td_dir)
            if any(globs.fullmatch(f) for f in s["changed_files"]):
                grep = score.grep_head(s, x["id"], set())
                choice = score.policy(p, s, r, td, seconds, grep, fallback)
                if choice is None:
                    raise AssertionError(f"score.score scored {x['id']} without {t} {p}'s inputs")  # fmt: skip
                ind.append(score.cost_fraction(choice, set(s["candidates"]), seconds, inductor.__contains__))  # fmt: skip
        if not ind:
            raise SystemExit(f"no scored {t} sample matches the ciflow/inductor globs")
        out[t] = {
            "cpu": st.mean(x["cpu"] for x in xs), "gpu": st.mean(x["gpu"] for x in xs),
            "ind_cpu": st.mean(x[0] for x in ind), "ind_gpu": st.mean(x[1] for x in ind),
            "n": len(xs), "n_ind": len(ind),
        }  # fmt: skip
    return out[tag], out["td"]


def current_results(data: Path, tag: str, ids: list[str], samples: Path) -> list[dict]:
    """`tag`'s current results on the `ids` select_tests asks the model about.

    As in score.score, a result is current if it was made from the prompt the
    sample gives now (see score.prompt_sha)."""
    minutes = test_minutes(json.loads((data / "test-times.json").read_text()))
    results, stale = [], 0
    for i in ids:
        s = json.loads((samples / f"{i}.json").read_text())
        if score.unasked(s) or not (p := data / "results" / tag / f"{i}.json").exists():
            continue
        r = json.loads(p.read_text())
        if r.get("prompt_sha") == score.prompt_sha(score.sample_prompt(s, minutes), bool(r.get("agentic"))):  # fmt: skip
            results.append(r)
        else:
            stale += 1
    if stale:
        print(f"left out {stale} {tag} results of another prompt", file=sys.stderr)
    return results


def call_results(data: Path, tag: str, ids: list[str], samples: Path) -> list[dict]:
    """current_results without the timed-out calls, since a call run_eval timed out records only its earlier attempts' cost."""
    results = current_results(data, tag, ids, samples)
    kept = [r for r in results if r["error"] != TIMED_OUT]
    if not kept:
        raise SystemExit(f"no {tag} results that did not time out for the cost ids")
    if n := len(results) - len(kept):
        print(f"left out {n} of {len(results)} {tag} calls that timed out", file=sys.stderr)  # fmt: skip
    return kept


def mean_call_usd(data: Path, tag: str, ids: list[str], samples: Path) -> float:
    """Mean recorded cost of `tag`'s calls on the `ids` select_tests asks the model about (see call_results)."""
    return st.mean(r["cost_usd"] for r in call_results(data, tag, ids, samples))


USAGE = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")  # fmt: skip


def cold_call_usd(data: Path, tag: str, ids: list[str], samples: Path) -> tuple[float, list[float]]:  # fmt: skip
    """Mean cost of `tag`'s calls on `ids` with a cold prompt cache, and the per-Mtok prices of USAGE fitted to the recorded costs.

    The eval ran its calls back to back, so most read the cached prompt;
    production calls are spread out and may not. Cold prices every
    cache-read token as a cache write. A result's cost sums its attempts
    while its usage is the last one's, so both the fit and the mean are over
    the single-attempt results with a recorded usage only, and the fit must
    be full rank with at least four results per price."""
    every = call_results(data, tag, ids, samples)
    rs = [r for r in every if r.get("attempts", 1) == 1 and r["usage"] is not None]
    u = np.array([[r["usage"].get(k, 0) for k in USAGE] for r in rs], float) / 1e6
    c = np.array([r["cost_usd"] for r in rs], float)
    price, _, rank, _ = np.linalg.lstsq(u, c, rcond=None)
    if rank < len(USAGE) or len(rs) < 4 * len(USAGE):
        raise SystemExit(f"{tag}: cannot fit {len(USAGE)} prices to {len(rs)} results of rank {rank}")  # fmt: skip
    rel = np.linalg.norm(u @ price - c) / np.linalg.norm(c)
    print(f"{tag}: price fit relative residual {rel:.2e} on {len(rs)} of {len(every)} results, cache-read share of their input tokens {u[:, 2].sum() / u[:, :3].sum():.2f}", file=sys.stderr)  # fmt: skip
    return float((c + u[:, 2] * (price[1] - price[2])).mean()), [float(x) for x in price]  # fmt: skip


def model(
    data: Path,
    tag: str,
    pol: str,
    *,
    overhead: float,
    ids: list[str],
    cost_ids: list[str],
    samples: Path,
    td_dir: Path,
    stats: str,
    labeler: Path = LABELER,
    workflows: list[tuple[str, str, float, float, float, bool]] = WORKFLOWS,
) -> dict:
    """Per-workflow monthly savings ($k) of `pol` of `tag` over TD, with the fractions on `ids` and the model-call cost averaged over `cost_ids`."""
    f, ftd = fractions(data, tag, pol, ids=ids, samples=samples, td_dir=td_dir, stats=stats, labeler=labeler)  # fmt: skip
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
        rows.append({"workflow": wf, "hw": hw, "pr_k": pr, "fraction": f[k], "old_rel": old_rel, "new_rel": new_rel, "save_k": save})  # fmt: skip
    calls = 0.0 if tag == "td" else mean_call_usd(data, tag, cost_ids, samples)
    llm_k = PR_PUSHES_PER_MONTH_K * calls
    return {"tag": tag, "policy": pol, "overhead": overhead, "fractions": f, "td_fractions": ftd, "rows": rows,
            "total_k_per_month": total, "stack_k_per_month": stack, "usd_per_call": calls, "llm_k_per_month": llm_k}  # fmt: skip


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data", type=Path, required=True, help="run_eval.py's --data: test-times.json and results/")  # fmt: skip
    p.add_argument("--ids", type=Path, help="samples the fractions come from; default: the report's neutral samples")  # fmt: skip
    p.add_argument("--cost-ids", type=Path, help="samples $/call is averaged over; default: the report's eval samples")  # fmt: skip
    p.add_argument("--td-dir", type=Path, help="td_replay.py output; default: DATA/td")
    p.add_argument("--stats", type=Path, default=REPO / score.ADDITIONAL_CI_FILES_FOLDER, help="td_replay.py's --stats")  # fmt: skip
    p.add_argument("--labeler", type=Path, default=LABELER, help="the labeler config with the ciflow/inductor globs")  # fmt: skip
    p.add_argument("--overhead", type=float, default=0.2, help="share of a test job's time that does not scale with the tests run")  # fmt: skip
    p.add_argument("--cold", action="store_true", help="also print $/call with a cold prompt cache (see cold_call_usd)")  # fmt: skip
    p.add_argument("--json", type=Path, help="also write the whole result (fractions, per-workflow rows, totals) here")  # fmt: skip
    p.add_argument("tag", help="results/TAG, or td for TD's own cut")
    p.add_argument("policy", help="a score.py policy, td25 with tag td")
    args = p.parse_args()
    if not 0 <= args.overhead < 1:
        p.error(f"--overhead must be in [0, 1), not {args.overhead}")
    if missing := [a for a in score.STATS_ARTIFACTS if not (args.stats / a).exists()]:
        p.error(f"{args.stats} lacks {missing}, which td_replay.py read")
    ids = (args.ids or IDS / "neutral.txt").read_text().split()
    cost_ids = (args.cost_ids or IDS / "eval.txt").read_text().split()
    d = model(args.data, args.tag, args.policy, overhead=args.overhead, ids=ids, cost_ids=cost_ids,
              samples=args.data / "samples", td_dir=args.td_dir or args.data / "td",
              stats=score.replay_stats(args.stats), labeler=args.labeler)  # fmt: skip
    f, t = d["fractions"], d["total_k_per_month"]
    print(
        f"{d['tag']} {d['policy']} overhead={d['overhead']} n={f['n']} n_ind={f['n_ind']} "
        f"run cpu={f['cpu']:.3f} gpu={f['gpu']:.3f} ind_cpu={f['ind_cpu']:.3f} ind_gpu={f['ind_gpu']:.3f} "
        f"save cpu=${t['cpu']:.0f}k gpu=${t['gpu']:.0f}k (this stack ${d['stack_k_per_month']:.0f}k) "
        f"$/call={d['usd_per_call']:.3f} llm=${d['llm_k_per_month']:.1f}k/month"
    )
    if args.cold and d["tag"] != "td":
        cold, price = cold_call_usd(args.data, d["tag"], cost_ids, args.data / "samples")  # fmt: skip
        prices = ", ".join(f"{k}=${x:.2f}" for k, x in zip(USAGE, price))
        print(f"cold cache $/call={cold:.3f} llm=${PR_PUSHES_PER_MONTH_K * cold:.1f}k/month; fitted per-Mtok {prices}")  # fmt: skip
        d["cold_usd_per_call"], d["fitted_usd_per_mtok"] = cold, dict(zip(USAGE, price))
    if args.json:
        args.json.write_text(json.dumps(d, indent=1))


if __name__ == "__main__":
    main()
