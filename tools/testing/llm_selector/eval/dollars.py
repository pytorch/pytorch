"""Monthly runner-dollar model of replacing TD with an LLM selection policy.

    python -m tools.testing.llm_selector.eval.dollars --data DIR v3_opus55_medium final+td20 [--overhead 0.2]

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
an asked eval sample. `--constants END` recomputes the constants from
ClickHouse.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics as st
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from tools.testing.llm_selector.eval import clickhouse, land_time, REPO, score
from tools.testing.llm_selector.eval.run_eval import TIMED_OUT
from tools.testing.llm_selector.select_tests import test_minutes


# 30-day runner cost ($k) of the test jobs the selection can shrink
# (land_time.selectable), the PR share of their hours, the fork-PR share of
# those PR hours, and whether TD runs in that workflow today, from
# `--constants 1793672225`. Only the TD workflows enforce the selection in
# this stack; the rest are follow-ups.
WORKFLOWS = [
    ("pull", "cpu", 458.1, 0.80, 0.36, True),
    ("trunk", "cpu", 133.4, 0.62, 0.46, True),
    ("trunk", "gpu", 136.5, 0.60, 0.47, True),
    ("inductor (unit tests)", "ind_cpu", 54.9, 0.78, 0.30, False),
    ("inductor (unit tests)", "ind_gpu", 48.5, 0.77, 0.32, False),
    ("periodic", "gpu", 17.2, 0.08, 0.37, False),
    ("slow", "cpu", 6.2, 0.04, 0.37, False),
]
# Same-repo PR heads with a pull run per month; fork PRs make no model call.
PR_PUSHES_PER_MONTH_K = 12.2
COST_SQL = """
SELECT workflow_name wf, gpu, job_name name, sum(cost) / 1000 k FROM misc.runner_cost
WHERE group_repo = 'pytorch/pytorch' AND date >= toDate(toDateTime({end})) - 30 AND date < toDate(toDateTime({end}))
  AND job_name LIKE '%/ test (%' AND workflow_name IN ('pull', 'trunk', 'inductor', 'periodic', 'slow')
GROUP BY wf, gpu, name
"""
# PR jobs are those of pull_request runs and ciflow/<label>/<PR> pushes, the ones that can get a selection; same is whether the PR's head is in this repo.
HOURS_SQL = """
WITH runs AS (
  SELECT id, event, head_branch, head_repository.full_name = 'pytorch/pytorch' run_same,
    toInt64OrZero(extract(head_branch, '^ciflow/[^/]+/([0-9]+)$')) prnum
  FROM default.workflow_run FINAL
  WHERE created_at >= toDateTime({end}) - INTERVAL 31 DAY AND created_at < toDateTime({end})
    AND repository.full_name = 'pytorch/pytorch' AND name IN ('pull', 'trunk', 'inductor', 'periodic', 'slow')),
prs AS (
  SELECT number, any(head.repo.full_name) = 'pytorch/pytorch' same FROM default.pull_request FINAL
  WHERE number IN (SELECT prnum FROM runs WHERE prnum > 0) GROUP BY number)
SELECT j.workflow_name wf, j.name name, arrayExists(l -> match(l, '{gpu}'), j.labels) gpu,
  r.event = 'pull_request' OR startsWith(r.head_branch, 'ciflow/') pr, if(r.event = 'pull_request', r.run_same, ifNull(p.same, 0)) same,
  sum(dateDiff('second', j.started_at, j.completed_at)) / 3600 h
FROM default.workflow_job j FINAL JOIN runs r ON j.run_id = r.id LEFT JOIN prs p ON p.number = r.prnum
WHERE j.repository_full_name = 'pytorch/pytorch' AND j.started_at >= toDateTime({end}) - INTERVAL 30 DAY AND j.started_at < toDateTime({end})
  AND j.completed_at > j.started_at AND j.name LIKE '%/ test (%' AND j.workflow_name IN ('pull', 'trunk', 'inductor', 'periodic', 'slow')
GROUP BY wf, name, gpu, pr, same
"""
HEADS_SQL = """
SELECT uniqExact(head_sha) heads FROM default.workflow_run FINAL
WHERE created_at >= toDateTime({end}) - INTERVAL 30 DAY AND created_at < toDateTime({end}) AND repository.full_name = 'pytorch/pytorch'
  AND name = 'pull' AND event = 'pull_request' AND head_repository.full_name = 'pytorch/pytorch'
"""
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
    the ciflow/inductor labeler rule matches, of the time of
    land_time.inductor_unit_configs(), the configs the Time to land replay
    shrinks in inductor."""
    rows = score.score(data, ids, list(dict.fromkeys([tag, "td"])), [pol], strong=True, samples=samples, td_dir=td_dir, stats=stats)  # fmt: skip
    seconds = score.test_seconds(data / "test-times.json")
    fallback = score.fallback_seconds(data / "test-times.json")
    globs = inductor_globs(labeler)
    inductor = land_time.inductor_unit_configs()
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
    sample gives now (see score.prompt_sha), and a sample whose prompt cannot
    be built because its base is not in the clone is left out."""
    minutes = test_minutes(json.loads((data / "test-times.json").read_text()))
    results, stale, no_base = [], 0, 0
    for i in ids:
        s = json.loads((samples / f"{i}.json").read_text())
        if score.unasked(s) or not (p := data / "results" / tag / f"{i}.json").exists():
            continue
        r = json.loads(p.read_text())
        try:
            prompt = score.sample_prompt(s, minutes)
        except subprocess.CalledProcessError:
            no_base += 1
            continue
        if r.get("prompt_sha") == score.prompt_sha(prompt, bool(r.get("agentic"))):
            results.append(r)
        else:
            stale += 1
    if stale:
        print(f"left out {stale} {tag} results of another prompt", file=sys.stderr)
    if no_base:
        print(f"left out {no_base} {tag} results whose sample's base is not in the clone", file=sys.stderr)  # fmt: skip
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
    if len(rs) < 4 * len(USAGE):
        raise SystemExit(f"{tag}: cannot fit {len(USAGE)} prices to {len(rs)} results")
    price, _, rank, _ = np.linalg.lstsq(u, c, rcond=None)
    if rank < len(USAGE):
        raise SystemExit(f"{tag}: cannot fit {len(USAGE)} prices to {len(rs)} results of rank {rank}")  # fmt: skip
    rel = np.linalg.norm(u @ price - c) / np.linalg.norm(c)
    print(f"{tag}: price fit relative residual {rel:.2e} on {len(rs)} of {len(every)} results, cache-read share of their input tokens {u[:, 2].sum() / u[:, :3].sum():.2f}", file=sys.stderr)  # fmt: skip
    return float((c + u[:, 2] * (price[1] - price[2])).mean()), [float(x) for x in price]  # fmt: skip


def constants(end: int) -> tuple[list[tuple[str, str, float, float, float, bool]], float]:  # fmt: skip
    """WORKFLOWS and PR_PUSHES_PER_MONTH_K over the 30 days to `end`.

    A row counts only the test jobs the selection can shrink, as the Time to
    land replay does (land_time.selectable): its cost, the PR share of their
    hours and the fork share of those PR hours. Selectable buckets without a
    WORKFLOWS row, and cost whose misc.runner_cost GPU flag disagrees with
    land_time.GPU_LABEL on the job's runner, are printed to stderr; a row
    with hours but no cost raises."""
    unit = land_time.inductor_unit_configs()
    gpu_label = re.compile(land_time.GPU_LABEL)

    def key(wf: str, name: str, gpu: bool) -> tuple[str, str] | None:
        if not land_time.selectable(wf, name, unit):
            return None
        return wf, ("ind_" if wf == "inductor" else "") + ("gpu" if gpu else "cpu")

    cost: collections.defaultdict[tuple[str, str], float] = collections.defaultdict(float)  # fmt: skip
    disagree: collections.Counter[tuple[str, str]] = collections.Counter()
    for r in clickhouse.query(COST_SQL.format(end=end)):
        if k := key(r["wf"], r["name"], r["gpu"]):
            cost[k] += r["k"]
            if bool(r["gpu"]) != bool(gpu_label.search(r["name"].partition("/ test (")[2])):  # fmt: skip
                disagree[k] += r["k"]
    # (all, PR, fork-PR) hours.
    hours: dict[tuple[str, str], list[float]] = collections.defaultdict(lambda: [0.0, 0.0, 0.0])  # fmt: skip
    for r in clickhouse.query(HOURS_SQL.format(end=end, gpu=land_time.GPU_LABEL)):
        if k := key(r["wf"], r["name"], r["gpu"]):
            hours[k][0] += r["h"]
            hours[k][1] += r["h"] if r["pr"] else 0.0
            hours[k][2] += r["h"] if r["pr"] and not r["same"] else 0.0
    out, listed = [], set()
    for wf, k, *_, has_td in WORKFLOWS:
        total, pr, fork = hours[w := ("inductor" if k.startswith("ind_") else wf, k)]
        if not pr:
            raise SystemExit(f"no PR test hours for {w}")
        if not cost[w]:
            raise SystemExit(f"{w} has {total:.0f} test hours but no misc.runner_cost")
        listed.add(w)
        out.append((wf, k, round(cost[w], 1), round(pr / total, 2), round(fork / pr, 2), has_td))  # fmt: skip
    for w in sorted((cost.keys() | hours.keys()) - listed):
        total, pr, _ = hours[w]
        print(f"left out {w}, which has no WORKFLOWS row: ${cost[w]:.1f}k, {total:.0f} test hours, {pr:.0f} of them PR", file=sys.stderr)  # fmt: skip
    for w, k in sorted(disagree.items()):
        if round(k, 1):
            print(f"${k:.1f}k of {w} is on runners misc.runner_cost and land_time.GPU_LABEL classify differently", file=sys.stderr)  # fmt: skip
    heads = clickhouse.query(HEADS_SQL.format(end=end))[0]["heads"]
    return out, round(heads / 1000, 1)


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
    p.add_argument("--data", type=Path, help="run_eval.py's --data: test-times.json and results/")  # fmt: skip
    p.add_argument("--ids", type=Path, help="samples the fractions come from; default: the report's neutral samples")  # fmt: skip
    p.add_argument("--cost-ids", type=Path, help="samples $/call is averaged over; default: the report's eval samples")  # fmt: skip
    p.add_argument("--td-dir", type=Path, help="td_replay.py output; default: DATA/td")
    p.add_argument("--stats", type=Path, default=REPO / score.ADDITIONAL_CI_FILES_FOLDER, help="td_replay.py's --stats")  # fmt: skip
    p.add_argument("--labeler", type=Path, default=LABELER, help="the labeler config with the ciflow/inductor globs")  # fmt: skip
    p.add_argument("--overhead", type=float, default=0.2, help="share of a test job's time that does not scale with the tests run")  # fmt: skip
    p.add_argument("--cold", action="store_true", help="also print $/call with a cold prompt cache (see cold_call_usd)")  # fmt: skip
    p.add_argument("--json", type=Path, help="also write the whole result (fractions, per-workflow rows, totals) here")  # fmt: skip
    p.add_argument("--constants", type=int, metavar="END", help="only print WORKFLOWS and PR_PUSHES_PER_MONTH_K for the 30 days to unix time END, from ClickHouse")  # fmt: skip
    p.add_argument("tag", nargs="?", help="results/TAG, or td for TD's own cut")
    p.add_argument("policy", nargs="?", help="a score.py policy, td25 with tag td")
    args = p.parse_args()
    if args.constants is not None:
        workflows, heads = constants(args.constants)
        print("WORKFLOWS = [\n" + "".join(f"    {w!r},\n" for w in workflows) + f"]\nPR_PUSHES_PER_MONTH_K = {heads}")  # fmt: skip
        return
    if args.data is None or args.policy is None:
        p.error("--data, tag and policy are required without --constants")
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
