"""Replay of PR CI time to signal and merge wait with the selection's test time, for REPORT.md's Time to land.

    python -m tools.testing.llm_selector.eval.land_time fetch --cache DIR [--end UNIX_TIME]
    python -m tools.testing.llm_selector.eval.land_time report --cache DIR --dollars DOLLARS_JSON [--all-prs]

fetch caches, from ClickHouse, 31 days of pull, trunk, inductor and rocm-mi300
runs (wfruns.json), each run's first-attempt job aggregates (runs_WF.json),
the head repository of each PR run and ciflow PR (fork.json), and the
merges of the last 30 days (merges.json), all up to --end (meta.json).

report replays the runs created between 29.5 days and 12 hours before the
end. Each same-repo test job's `Test` step is shrunk by r, new over old test
time on a typical change, from a dollars.py --json output: fractions over
td_fractions where TD runs, and the inductor unit-test fractions (old = 1)
in inductor, whose model benchmarks keep r = 1 (see selectable). Jobs
run_test.py runs without TD keep r = 1, and so do fork-PR runs, which keep
TD, unless --all-prs. Each job keeps its start time; a run's time to signal
(TTS) is its last job's completion minus its creation. Only first attempts
count, and runs that did not conclude success or failure, have a cancelled
job, ran no test job or have a TTS outside (0, 24 h] are left out. A PR
head's merge wait is the latest end of its first included pull and trunk
runs (and inductor's) minus the earliest creation, over the heads whose trunk (and
inductor) run was created within 10 minutes of pull. The merges
whose comment started trunk are replayed by taking that trunk run's saving
off their comment to merge time; those whose trunk run the replay left out
are left out of both the old and the new times.
"""

from __future__ import annotations

import argparse
import collections
import fnmatch
import json
import re
import statistics as st
import time
from pathlib import Path
from typing import TYPE_CHECKING

from tools.testing.llm_selector.eval import clickhouse, score, write_json


if TYPE_CHECKING:
    from collections.abc import Sequence


WORKFLOWS = ("pull", "trunk", "inductor", "rocm-mi300")
# Configs of TD workflows whose .ci/pytorch/test.sh branch does not run run_test.py.
NO_RUN_TEST = {"backwards_compat", "docs_test", "libtorch_agnostic_targetting", "pr_time_benchmarks", "verify_cachebench", "aoti_cross_compile_for_windows"}  # fmt: skip
# The job name before ' / test (' and the config, in workflow_job's and misc.runner_cost's job names.
TEST_JOB = re.compile(r"^(.*) / test \(([^,]+),")
CIFLOW_PR = re.compile(r"^ciflow/[^/]+/(\d+)$")
GPU_LABEL = "gpu|nvidia|-a10g|-l4|-t4|h100|h200|b200|gb[0-9]00|a100|rocm|xpu"
CONCURRENT_S = 600

RUNS_SQL = """
SELECT id, name, event, head_branch, head_sha, toUnixTimestamp(created_at) created, run_attempt, conclusion,
  length(pull_requests) npr, if(length(pull_requests)>0, pull_requests[1].number, 0) prnum
FROM default.workflow_run FINAL
WHERE created_at >= toDateTime({end}) - INTERVAL 31 DAY AND created_at < toDateTime({end})
  AND repository.full_name = 'pytorch/pytorch' AND name IN ('pull','trunk','inductor','rocm-mi300')
"""

# Per run: cancelled job count, last build and non-test job end; per test job: (end, Test step seconds, job name, GPU, job seconds, queue seconds, labels).
JOBS_SQL = """
WITH jobs AS (
  SELECT id, any(run_id) run_id, any(head_sha) head_sha, any(name) nm, any(conclusion) concl,
    any(created_at) jc, any(started_at) js, any(completed_at) je, any(labels) lbl,
    any(steps.name) sn, any(steps.started_at) ss, any(steps.completed_at) se
  FROM default.workflow_job
  WHERE created_at >= toDateTime({end}) - INTERVAL {hi} DAY AND created_at < toDateTime({end}) - INTERVAL {lo} DAY
    AND workflow_name = '{wf}' AND run_attempt = 1 AND conclusion != ''
    AND repository_full_name = 'pytorch/pytorch' AND {branch}
  GROUP BY id
), j2 AS (
  SELECT *, indexOf(sn, 'Test') ti,
    if(ti > 0 AND se[ti] >= ss[ti] AND ss[ti] >= js AND se[ti] <= je, dateDiff('second', ss[ti], se[ti]), 0) tsec,
    multiIf(nm LIKE '%/ test (%', 'test', match(nm, '/ build($| )'), 'build', 'other') kind,
    arrayExists(l -> match(l, '{gpu}'), lbl) gpu,
    arrayStringConcat(lbl, ',') lab
  FROM jobs
)
SELECT run_id, any(head_sha) sha,
  countIf(concl = 'cancelled') ncancel,
  maxIf(toUnixTimestamp(je), kind='build' AND concl != 'skipped') last_build,
  maxIf(toUnixTimestamp(je), kind='other' AND concl != 'skipped') last_other,
  groupArrayIf((toUnixTimestamp(je), tsec, nm, gpu, dateDiff('second', js, je), dateDiff('second', jc, js), lab), kind='test' AND concl != 'skipped') tests
FROM j2 GROUP BY run_id
"""

FORK_SQL = """
SELECT toString(id) k, head_repository.full_name repo FROM default.workflow_run FINAL
WHERE created_at >= toDateTime({end}) - INTERVAL 31 DAY AND created_at < toDateTime({end})
  AND repository.full_name = 'pytorch/pytorch' AND name IN ('pull','trunk','inductor','rocm-mi300')
  AND event = 'pull_request'
"""

PR_REPO_SQL = """
SELECT toString(number) k, any(head.repo.full_name) repo FROM default.pull_request FINAL
WHERE number IN (
  SELECT DISTINCT toInt64OrZero(extract(head_branch, '^ciflow/[^/]+/([0-9]+)$')) FROM default.workflow_run FINAL
  WHERE created_at >= toDateTime({end}) - INTERVAL 31 DAY AND created_at < toDateTime({end})
    AND repository.full_name = 'pytorch/pytorch' AND name IN ('pull','trunk','inductor','rocm-mi300'))
GROUP BY number
"""

# Merge comment (cc), the main pull run of the merge commit (mt) and the first PR pull run of the merged head (pt).
MERGES_SQL = """
WITH m AS (
  SELECT pr_num, comment_id, last_commit_sha, merge_commit_sha, skip_mandatory_checks f, ignore_current i
  FROM default.merges
  WHERE owner='pytorch' AND project='pytorch' AND merge_commit_sha != '' AND comment_id > 0 AND NOT dry_run AND pr_num > 160000
),
c AS (SELECT id, min(created_at) cc FROM default.issue_comment
  WHERE created_at >= toDateTime({end}) - INTERVAL 30 DAY AND created_at < toDateTime({end}) AND id IN (SELECT comment_id FROM m) GROUP BY id),
mainrun AS (SELECT head_sha, min(created_at) mt FROM default.workflow_run
  WHERE created_at >= toDateTime({end}) - INTERVAL 31 DAY AND created_at < toDateTime({end}) AND name='pull' AND event='push' AND head_branch='main'
    AND repository.full_name='pytorch/pytorch' GROUP BY head_sha),
prrun AS (SELECT head_sha, min(created_at) pt FROM default.workflow_run
  WHERE created_at >= toDateTime({end}) - INTERVAL 40 DAY AND created_at < toDateTime({end}) AND name='pull' AND event='pull_request'
    AND repository.full_name='pytorch/pytorch' GROUP BY head_sha)
SELECT m.pr_num pr, m.f f, m.i i, toUnixTimestamp(c.cc) cc, toUnixTimestamp(mainrun.mt) mt, toUnixTimestamp(prrun.pt) pt, m.last_commit_sha sha
FROM m JOIN c ON c.id = m.comment_id JOIN mainrun ON mainrun.head_sha = m.merge_commit_sha LEFT JOIN prrun ON prrun.head_sha = m.last_commit_sha
"""


def fetch(cache: Path, end: int) -> None:
    """meta.json is written last, so a fetch that fails partway leaves a cache report refuses."""
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "meta.json").unlink(missing_ok=True)
    write_json(cache / "wfruns.json", clickhouse.query(RUNS_SQL.format(end=end)))
    for wf in WORKFLOWS:
        branch = "head_branch NOT LIKE 'release/%'" if wf == "pull" else "head_branch LIKE 'ciflow/%'"  # fmt: skip
        rows = [r for lo in range(0, 30, 3) for r in clickhouse.query(JOBS_SQL.format(end=end, lo=lo, hi=lo + 3, wf=wf, branch=branch, gpu=GPU_LABEL))]  # fmt: skip
        write_json(cache / f"runs_{wf}.json", rows)
    fork = {name: {r["k"]: r["repo"] for r in clickhouse.query(sql.format(end=end))} for name, sql in (("run_head_repo", FORK_SQL), ("pr_head_repo", PR_REPO_SQL))}  # fmt: skip
    write_json(cache / "fork.json", fork)
    write_json(cache / "merges.json", clickhouse.query(MERGES_SQL.format(end=end)))
    write_json(cache / "meta.json", {"end": end})


def pct(a: list[float], p: int) -> float:
    """numpy.percentile's default (linear) interpolation, or nan with no points."""
    if len(a) < 2:
        return a[0] if a else float("nan")
    return st.median(a) if p == 50 else st.quantiles(a, n=100, method="inclusive")[p - 1]  # fmt: skip


def ratios(dollars: dict) -> dict[str, float]:
    f, t = dollars["fractions"], dollars["td_fractions"]
    return {"cpu": round(f["cpu"] / t["cpu"], 3), "gpu": round(f["gpu"] / t["gpu"], 3), "ind_cpu": round(f["ind_cpu"], 3), "ind_gpu": round(f["ind_gpu"], 3)}  # fmt: skip


def inductor_unit_configs() -> set[str]:
    """The configs of inductor's unit-test jobs (inductor-unittest.yml, run as 'unit-test / ...') that TD never runs; the rest of inductor are model benchmarks."""
    unit = {c for w, n, c in score.ci_jobs()[False] if w == "inductor" and n.startswith("unit-test / ")}  # fmt: skip
    return {c for c in unit if not score.runs_td(c)}


def selectable(wf: str, job: str, unit: set[str]) -> bool:
    """Whether the selection shrinks the `wf` test job named `job`: inductor's unit tests (`unit` is inductor_unit_configs()), and elsewhere the jobs run_test.py runs with TD (score.job_runs_td).

    TD is off in periodic and slow, so ci_jobs has no job-level verdict there;
    their follow-up rows apply run_test.py's config and build-environment
    rules (periodic.yml and slow.yml set no force-no-td)."""
    if (m := TEST_JOB.search(job)) is None:
        return False
    name, cfg = m.groups()
    if wf == "inductor":
        return cfg in unit
    if cfg in NO_RUN_TEST:
        return False
    if wf in ("periodic", "slow"):
        return not score.NO_TD_ENV.search(name) and not any(fnmatch.fnmatchcase(cfg, n) for n in score.NO_TD_CONFIGS)  # fmt: skip
    # misc.runner_cost's names have no shard numbers, which score.config_of needs.
    return score.job_runs_td(f"{name} / test ({cfg}, 1, 1)", wf)


def fork_runs(cache: Path, wfruns: list[dict]) -> set[int]:
    """Runs whose PR head repository is not pytorch/pytorch or is unknown: pull's pull_request runs by their own, ciflow/<label>/<PR> pushes by the PR's."""
    fk = json.loads((cache / "fork.json").read_text())
    out = set()
    for w in wfruns:
        if w["event"] == "pull_request":
            repo = fk["run_head_repo"].get(str(w["id"]))
        elif m := CIFLOW_PR.match(w["head_branch"] or ""):
            repo = fk["pr_head_repo"].get(m.group(1))
        else:
            continue
        if repo != "pytorch/pytorch":
            out.add(w["id"])
    return out


def replay(cache: Path, r: dict[str, float], keep: set[int]) -> dict[str, list[dict]]:
    """The included PR runs of each workflow, with their old and new TTS. Runs in `keep` are not shrunk."""
    end = json.loads((cache / "meta.json").read_text())["end"]
    wfr = {w["id"]: w for w in json.loads((cache / "wfruns.json").read_text())}
    unit = inductor_unit_configs()

    def ratio(wf: str, job: str, gpu: bool) -> float:
        if not selectable(wf, job, unit):
            return 1.0
        return r[("ind_" if wf == "inductor" else "") + ("gpu" if gpu else "cpu")]

    out = {}
    for wf in WORKFLOWS:
        merged: dict[int, dict] = {}
        for x in json.loads((cache / f"runs_{wf}.json").read_text()):
            if (m := merged.get(x["run_id"])) is None:
                merged[x["run_id"]] = x
                continue
            m["ncancel"] += x["ncancel"]
            m["last_build"] = max(m["last_build"], x["last_build"])
            m["last_other"] = max(m["last_other"], x["last_other"])
            m["tests"] += x["tests"]
        rs = []
        for rid, x in merged.items():
            w = wfr.get(rid)
            if w is None or (w["event"] != "pull_request" if wf == "pull" else not w["head_branch"].startswith("ciflow/")):  # fmt: skip
                continue
            c = w["created"]
            if not end - 29.5 * 86400 <= c <= end - 12 * 3600 or w["conclusion"] not in ("success", "failure") or x["ncancel"] or not x["tests"]:  # fmt: skip
                continue
            last_build, last_other = x["last_build"] or c, x["last_other"] or c
            for t in x["tests"]:
                t.append(1.0 if rid in keep else ratio(wf, t[2], t[3]))
            t_old = max(t[0] for t in x["tests"])
            t_new = max(t[0] - t[1] * (1 - t[-1]) for t in x["tests"])
            old = max(last_build, last_other, t_old) - c
            if not 0 < old <= 24 * 3600:
                continue
            new = max(last_build, last_other, t_new) - c
            rs.append({"sha": w["head_sha"], "c": c, "end_old": c + old, "end_new": c + new, "old": old / 60, "new": new / 60, "build_bound_old": t_old < max(last_build, last_other), "build_bound_new": t_new < max(last_build, last_other), "tests": x["tests"]})  # fmt: skip
        out[wf] = rs
    return out


def wait(heads: Sequence[Sequence[dict]], end: str) -> list[float]:
    return [(max(r[end] for r in h) - min(r["c"] for r in h)) / 60 for h in heads]


def report(cache: Path, dollars: dict, all_prs: bool) -> list[str]:
    wfruns = json.loads((cache / "wfruns.json").read_text())
    r = ratios(dollars)
    keep = set() if all_prs else fork_runs(cache, wfruns)
    runs = replay(cache, r, keep)
    lines = [f"r {r}, {len(keep)} fork runs kept", "", "| Workflow | Runs | Old p50 | New p50 | Old p90 | New p90 | Mean saved | Build-bound old | Build-bound new |", "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    if empty := [wf for wf, rs in runs.items() if not rs]:
        raise SystemExit(f"no included {', '.join(empty)} runs in {cache}")
    for wf, rs in runs.items():
        o, n = [x["old"] for x in rs], [x["new"] for x in rs]
        bo, bn = (100 * sum(x[k] for x in rs) / len(rs) for k in ("build_bound_old", "build_bound_new"))  # fmt: skip
        lines.append(f"| {wf} | {len(rs)} | {pct(o, 50):.0f} | {pct(n, 50):.0f} | {pct(o, 90):.0f} | {pct(n, 90):.0f} | {st.mean(o) - st.mean(n):.0f} | {bo:.1f}% | {bn:.1f}% |")  # fmt: skip

    jo = sum(t[4] for rs in runs.values() for x in rs for t in x["tests"]) / 3600
    jn = sum(t[4] - t[1] * (1 - t[-1]) for rs in runs.values() for x in rs for t in x["tests"]) / 3600  # fmt: skip
    lines += ["", f"Test job-hours: {jo:,.0f} -> {jn:,.0f}"]

    first: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    for wf in ("pull", "trunk", "inductor"):
        for x in runs[wf]:
            if wf not in first[x["sha"]] or x["c"] < first[x["sha"]][wf]["c"]:
                first[x["sha"]][wf] = x

    def heads(wfs: tuple[str, ...]) -> list[tuple[dict, ...]]:
        return [tuple(h[w] for w in wfs) for h in first.values() if all(w in h and abs(h[w]["c"] - h["pull"]["c"]) <= CONCURRENT_S for w in wfs)]  # fmt: skip

    conc, ind = heads(("pull", "trunk")), heads(("pull", "trunk", "inductor"))
    if not conc or not ind:
        raise SystemExit(f"{len(conc)} pull and trunk heads, {len(ind)} with inductor: too few to report")  # fmt: skip
    lines += ["", "| Merge wait | Heads | Old p50 | Old p90 | Old mean | New p50 | New p90 | New mean |", "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for name, heads in (("pull and trunk", conc), ("pull, trunk and inductor", ind)):
        o, n = wait(heads, "end_old"), wait(heads, "end_new")
        lines.append(f"| {name} | {len(heads)} | {pct(o, 50):.0f} | {pct(o, 90):.0f} | {st.mean(o):.0f} | {pct(n, 50):.0f} | {pct(n, 90):.0f} | {st.mean(n):.0f} |")  # fmt: skip
    later = sum(t["end_old"] >= p["end_old"] for p, t in conc), sum(t["end_new"] >= p["end_new"] for p, t in conc)  # fmt: skip
    lines.append(f"Trunk ends last for {100 * later[0] / len(conc):.0f}% of the pull and trunk heads, {100 * later[1] / len(conc):.0f}% after.")  # fmt: skip
    last = [sum(i[k] >= max(p[k], t[k]) for p, t, i in ind) for k in ("end_old", "end_new")]  # fmt: skip
    lines.append(f"Inductor ends last for {100 * last[0] / len(ind):.0f}% of the pull, trunk and inductor heads, {100 * last[1] / len(ind):.0f}% after.")  # fmt: skip

    unforced = [m for m in json.loads((cache / "merges.json").read_text()) if not m["f"]]  # fmt: skip
    p2m = [(m["mt"] - m["pt"]) / 3600 for m in unforced if m["pt"] and m["mt"] >= m["pt"]]  # fmt: skip
    merges = [m for m in unforced if m["mt"] >= m["cc"]]
    c2m = [(m["mt"] - m["cc"]) / 60 for m in merges]
    trunk = collections.defaultdict(list)
    for w in wfruns:
        if w["name"] == "trunk" and w["head_branch"].startswith("ciflow/"):
            trunk[w["head_sha"]].append(w["created"])
    saved = {(x["sha"], x["c"]): x["old"] - x["new"] for x in runs["trunk"]}
    started, started_new, before, unreplayed = [], [], [], 0
    for m in merges:
        if m["i"] or not (ts := trunk.get(m["sha"])):
            continue
        if min(ts) < m["cc"] - 300:
            before.append((m["mt"] - m["cc"]) / 60)
        elif abs((t := min(ts, key=lambda t: abs(t - m["cc"]))) - m["cc"]) <= 300:
            if (k := (m["sha"], t)) not in saved:
                unreplayed += 1
                continue
            started.append((m["mt"] - m["cc"]) / 60)
            started_new.append(started[-1] - saved[k])
    lines += [
        "",
        f"Merges without -f (n={len(c2m)}): comment to merge p50 {pct(c2m, 50):.0f} / p90 {pct(c2m, 90):.0f} min; first PR pull run to merge p50 {pct(p2m, 50):.1f} / p90 {pct(p2m, 90):.1f} h.",
        f"Merge comment started trunk (n={len(started)}, {unreplayed} more whose trunk run was not replayed left out): comment to merge p50 {pct(started, 50):.0f} / p90 {pct(started, 90):.0f} min, new p50 {pct(started_new, 50):.0f} / p90 {pct(started_new, 90):.0f} min. Trunk ran before the comment (n={len(before)}): p50 {pct(before, 50):.0f} / p90 {pct(before, 90):.0f} min.",
    ]

    queues: dict[str, list] = collections.defaultdict(lambda: [[], 0.0, 0.0])
    for rs in runs.values():
        for x in rs:
            for t in x["tests"]:
                q = queues[t[6]]
                q[0].append(t[5] / 60)
                q[1] += t[4] / 3600
                q[2] += t[1] / 3600
    lines += ["", "| Runner labels | Jobs | Queue p50 (min) | Queue p90 (min) | Job-hours | Test-step share |", "|---|---|---|---|---|---|"]  # fmt: skip
    for lab, (qs, jh, th) in sorted(queues.items(), key=lambda kv: -kv[1][1])[:22]:
        lines.append(f"| {lab} | {len(qs)} | {pct(qs, 50):.1f} | {pct(qs, 90):.1f} | {jh:,.0f} | {100 * th / jh:.0f}% |")  # fmt: skip
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--cache", type=Path, required=True)
    f.add_argument("--end", type=int, default=int(time.time()), help="unix time the windows end at")  # fmt: skip
    r = sub.add_parser("report")
    r.add_argument("--cache", type=Path, required=True)
    r.add_argument("--dollars", type=Path, required=True, help="a dollars.py --json output")  # fmt: skip
    r.add_argument("--all-prs", action="store_true", help="shrink fork-PR runs too")
    args = p.parse_args()
    if args.cmd == "fetch":
        fetch(args.cache, args.end)
    else:
        print("\n".join(report(args.cache, json.loads(args.dollars.read_text()), args.all_prs)))  # fmt: skip


if __name__ == "__main__":
    main()
