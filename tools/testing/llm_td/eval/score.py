"""Score selector results: recall of real failures, and the share of CI test time selected.

    python -m tools.testing.llm_td.eval.score --data DIR --ids DIR/ids_test.txt sonnet5_medium opus55_high td

A failure is a (test file, CI config) pair. It counts as caught when the
selection runs that file in that config. Cost is the share of one full CI
pass's CPU and GPU test-seconds (from test-times.json) that the selection
would run.

Policies: `model` is the raw model output, `final` adds what CI adds (edited
test files, the run-all rules), `final-hm` keeps only high/medium-confidence
picks, and `final+tdN` hedges `final` with TD's files that fit in N% of each
config's test time (N defaults to LLM_TD_HEDGE). The tag `td` scores the
replayed in-repo TD (see td_replay.py) at its CI cut of the top 25% per config.

run_test.py applies TD, and so the selection, only in the pull, trunk and
ROCm workflows. Jobs of any other workflow (inductor's unit tests, periodic,
slow, ...) run every test under every policy, so their failures are left out
of the truth, and configs no TD job runs are left out of the cost share
(the report's dollar model prices the inductor ones separately). Jobs are
classified from .github/workflows (see job_runs_td). test-times.json is
merged over jobs per config, so a config that TD and non-TD jobs share (e.g.
`default` in periodic) is priced as one.
CI applies the TD cut and the hedge per job over that job's own test list;
this models a job as a config of test-times.json, and for a failure in a
config missing from test-times.json falls back to a cut over all candidates.

`files` in the summary is the median number of files selected among the
selections that do not run everything.
"""

from __future__ import annotations

import argparse
import collections
import functools
import json
import re
import statistics as st
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from tools.testing.llm_td.eval import REPO
from tools.testing.llm_td.selection import LLM_TD_HEDGE, Selection


GPU_ENV = re.compile(r"cuda|rocm|sm\d+|xpu|h100|b200|a100|gpu", re.IGNORECASE)
CONFIDENCE = {"high": 0, "medium": 1, "low": 2}
HEDGE = re.compile(r"final\+td(\d+(?:\.\d+)?)?")
MATRIX_CONFIG = re.compile(r'config: "([^"]+)"')
MATRIX_NEEDS = re.compile(r"needs\.([\w-]+)\.outputs\.test-matrix")
VERSION = re.compile(r"\d+(\.\d+)*")
MATRIX_SUFFIX = re.compile(r" \([^()]*\)")

# config -> file -> [cpu, gpu] seconds
Seconds = dict[str, dict[str, list[float]]]


@dataclass
class Choice:
    """A Selection plus the files run on top of it in each config (TD's cut or the hedge).

    `other` applies to configs absent from test-times.json."""

    sel: Selection
    per_config: dict[str, set[str]] = field(default_factory=dict)
    other: set[str] = field(default_factory=set)

    def covers(self, test_file: str, test_config: str) -> bool:
        return self.sel.covers(
            test_file, test_config
        ) or test_file in self.per_config.get(test_config, self.other)

    def files(self) -> set[str]:
        return self.sel.files.union(*self.per_config.values())


def config_of(job: str) -> str:
    """'linux-jammy-py3.10-clang21 / test (crossref, 1, 2, lf.linux.2xlarge)' -> 'crossref'"""
    m = re.search(r"\(([^,()]+), \d+, \d+", job) or re.search(r"\(([^,()]+)\)", job)
    return m.group(1).strip() if m else job.strip()


@functools.cache
def test_seconds(path: Path) -> Seconds:
    """config -> file -> [cpu, gpu] seconds of one full CI pass."""
    out: dict = collections.defaultdict(
        lambda: collections.defaultdict(lambda: [0.0, 0.0])
    )
    for env, configs in json.loads(path.read_text()).items():
        gpu = int(bool(GPU_ENV.search(env)))
        for c, files in configs.items():
            for f, s in files.items():
                out[c][f][gpu] += s
    return {c: dict(files) for c, files in out.items()}


def td_workflow(name: str) -> bool:
    """Whether run_test.py enables TD under GITHUB_WORKFLOW `name`."""
    return name in ("trunk", "pull") or name.startswith(("rocm-", "periodic-rocm-"))


@functools.cache
def ci_jobs(
    workflows: Path = REPO / ".github/workflows",
) -> dict[bool, set[tuple[str, str]]]:
    """Whether TD runs -> the (job name, config) pairs of the test jobs that run so.

    A reusable workflow's jobs are named and classified under each caller, as
    GitHub reports them ('unit-test / inductor-test') and sets GITHUB_WORKFLOW,
    and also under its own name if it has triggers of its own."""
    wfs = {p.name: yaml.safe_load(p.read_text()) for p in workflows.glob("*.yml")}

    def test_jobs(file: str) -> set[tuple[str, str]]:
        jobs = wfs[file].get("jobs") or {}

        def configs(key: str) -> set[str]:
            matrix = str((jobs[key].get("with") or {}).get("test-matrix", ""))
            if (m := MATRIX_NEEDS.search(matrix)) and m.group(1) in jobs:
                return configs(m.group(1))
            return set(MATRIX_CONFIG.findall(matrix))

        out = set()
        for key, job in jobs.items():
            name, uses = job.get("name", key), Path(job.get("uses", "")).name
            if uses in wfs and not uses.startswith("_"):
                out |= {(f"{name} / {n}", c) for n, c in test_jobs(uses)}
            elif "test" in uses:
                out |= {(name, c) for c in configs(key)}
        return out

    out: dict[bool, set[tuple[str, str]]] = {True: set(), False: set()}
    for file, wf in wfs.items():
        on = wf.get(True) if isinstance(wf, dict) else None
        if set([on] if isinstance(on, str) else on or []) - {"workflow_call"}:
            out[td_workflow(wf.get("name", ""))] |= test_jobs(file)
    return out


@functools.cache
def runs_td(config: str) -> bool:
    return any(c == config for _, c in ci_jobs()[True])


@functools.cache
def job_runs_td(job: str) -> bool:
    """Whether the test job named `job` runs with TD.

    It is looked up by name, then without matrix suffixes ('dynamo-test (3.11)')
    and with versions wildcarded (CI has moved to newer CUDA and Python since),
    then by config alone. A job that matches both TD and non-TD jobs counts as TD."""
    name, c = job.rsplit(" / ", 1)[0], config_of(job)
    jobs = ci_jobs()
    for norm in (lambda n: n, lambda n: VERSION.sub("#", MATRIX_SUFFIX.sub("", n))):
        td, other = ({(norm(n), k) for n, k in jobs[b]} for b in (True, False))
        if (norm(name), c) in td | other:
            return (norm(name), c) in td
    return runs_td(c)


def truth_pairs(s: dict, strong: bool) -> list[tuple[str, str]]:
    """The (file, config) failures of `s` in jobs that run TD.

    A file with no recorded job counts as failing in `default`."""
    truth: list[str] = s["truth"]
    if strong and s["kind"] == "pr":
        truth = s.get("truth_strong", truth)
    cands = set(s["candidates"])
    pairs = set()
    for f in truth:
        if f in cands:
            jobs = s.get("truth_jobs", {}).get(f)
            td_jobs = [j for j in jobs if job_runs_td(j)] if jobs else ["(default)"]
            pairs |= {(f, config_of(j)) for j in td_jobs}
    return sorted(pairs)


def hedge(
    sel: Selection,
    scored: list[str],
    seconds: dict[str, float],
    share: float,
    config: str,
) -> set[str]:
    """TD's positive-score files that fit in `share` of `seconds`' total, filled greedily in TD's order.

    As in run_test.py, files the selection already covers cost nothing, and a
    file that does not fit is skipped while smaller ones after it may still
    fit. Files with no recorded time are skipped rather than added for free."""
    budget = share * sum(seconds.values())
    out = set()
    for f in scored:
        t = seconds.get(f, 0.0)
        if t > 0 and t <= budget and not sel.covers(f, config):
            budget -= t
            out.add(f)
    return out


def policy(
    name: str, s: dict, r: dict | None, td: dict | None, seconds: Seconds
) -> Choice | None:
    """The files `name` runs on sample `s`, or None when its inputs are missing."""
    cands = set(s["candidates"])
    if name == "td25":
        if td is None:
            return None
        return Choice(
            Selection(),
            {c: set(fs) for c, fs in td["top25_cfg"].items()},
            set(td["top25"]),
        )
    out = (r or {}).get("out")
    if name == "model":
        return Choice(
            Selection(run_all=True)
            if out is None
            else Selection.from_model_output(out, cands, [])
        )
    if m := HEDGE.fullmatch(name):
        if td is None:
            return None
        share = float(m.group(1)) / 100 if m.group(1) else LLM_TD_HEDGE
        sel = Selection.from_model_output(out, cands, s["changed_files"])
        if sel.run_all:
            return Choice(sel)
        times = {
            c: {f: cpu + gpu for f, (cpu, gpu) in fs.items() if f in cands}
            for c, fs in seconds.items()
        }
        total: dict[str, float] = collections.defaultdict(float)
        for fs in times.values():
            for f, t in fs.items():
                total[f] += t
        per_config = {
            c: hedge(sel, td["scored"], fs, share, c) for c, fs in times.items()
        }
        return Choice(sel, per_config, hedge(sel, td["scored"], total, share, ""))
    if name == "final-hm":
        if out is not None:
            out = {
                **out,
                "tests": [
                    t
                    for t in out.get("tests", [])
                    if CONFIDENCE.get(t.get("confidence"), 2) <= 1
                ],
            }
    elif name != "final":
        raise ValueError(f"unknown policy {name!r}")
    return Choice(Selection.from_model_output(out, cands, s["changed_files"]))


def cost_fraction(
    choice: Choice, cands: set[str], seconds: Seconds, configs=lambda c: True
) -> tuple[float, float]:
    """Share of the candidates' (cpu, gpu) seconds in the configs `configs` accepts that `choice` runs."""
    tot, run = [0.0, 0.0], [0.0, 0.0]
    for c, files in seconds.items():
        if not configs(c):
            continue
        for f, (cpu, gpu) in files.items():
            if f in cands:
                tot[0] += cpu
                tot[1] += gpu
                if choice.covers(f, c):
                    run[0] += cpu
                    run[1] += gpu
    return run[0] / max(tot[0], 1), run[1] / max(tot[1], 1)


def load(
    data: Path, tag: str, sid: str, samples: Path, td_dir: Path
) -> tuple[dict, dict | None, dict | None]:
    s = json.loads((samples / f"{sid}.json").read_text())
    rp, tp = data / "results" / tag / f"{sid}.json", td_dir / f"{sid}.json"
    r = json.loads(rp.read_text()) if rp.exists() else None
    td = json.loads(tp.read_text()) if tp.exists() else None
    return s, r, td


def score(
    data: Path,
    ids: list[str],
    tag: str,
    policies: list[str],
    strong: bool,
    samples: Path,
    td_dir: Path,
) -> list[dict]:
    seconds = test_seconds(data / "test-times.json")
    rows = []
    for sid in ids:
        s, r, td = load(data, tag, sid, samples, td_dir)
        if tag != "td" and r is None:
            continue
        pairs = truth_pairs(s, strong)
        cands = set(s["candidates"])
        for pol in policies:
            choice = policy(pol, s, r, td, seconds)
            if choice is None:
                continue
            hit = [choice.covers(f, c) for f, c in pairs]
            cpu, gpu = cost_fraction(choice, cands, seconds, runs_td)
            rows.append({
                "tag": tag, "policy": pol, "id": sid, "kind": s["kind"], "pairs": len(pairs), "hit": sum(hit),
                "cpu": cpu, "gpu": gpu, "run_all": choice.sel.run_all, "files": len(choice.files()),
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
        subset = [x["files"] for x in xs if not x["run_all"]] or [0]
        out.append({
            "tag": tag, "policy": pol, "kind": kind, "n": len(xs), "n_failing": len(failing),
            "caught_any": sum(x["hit"] > 0 for x in failing) / max(len(failing), 1),
            "caught_all": sum(x["hit"] == x["pairs"] for x in failing) / max(len(failing), 1),
            "pair_recall": sum(x["hit"] for x in failing) / max(sum(x["pairs"] for x in failing), 1),
            "cpu": st.mean(x["cpu"] for x in xs), "gpu": st.mean(x["gpu"] for x in xs),
            "run_all": sum(x["run_all"] for x in xs), "median_files": st.median(subset),
            "usd_per_call": st.mean(x["cost_usd"] for x in xs), "errors": sum(x["error"] for x in xs),
        })  # fmt: skip
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--ids", type=Path, required=True)
    p.add_argument("--samples-dir", type=Path, help="default: DATA/samples")
    p.add_argument("--td-dir", type=Path, help="td_replay.py output; default: DATA/td")
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
    samples, td_dir = (
        args.samples_dir or args.data / "samples",
        args.td_dir or args.data / "td",
    )
    rows = []
    for tag in args.tags:
        pols = ["td25"] if tag == "td" else args.policies.split(",")
        rows += score(args.data, ids, tag, pols, not args.all_truth, samples, td_dir)
    for x in summarize(rows):
        print(
            f"{x['tag']:24s} {x['policy']:10s} {x['kind']:7s} n={x['n']:3d} fail={x['n_failing']:3d} caught={x['caught_any']:6.1%} "
            f"all={x['caught_all']:6.1%} pairs={x['pair_recall']:6.1%} cpu={x['cpu']:6.1%} gpu={x['gpu']:6.1%} "
            f"run_all={x['run_all']:3d} files={x['median_files']:4.0f} $/call={x['usd_per_call']:.3f} err={x['errors']}"
        )
    if args.json:
        args.json.write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
