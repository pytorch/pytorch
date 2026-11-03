"""Score selector results: recall of real failures, and the share of CI test time selected.

    python -m tools.testing.llm_td.eval.score --data DIR --ids tools/testing/llm_td/eval/ids/eval.txt sonnet5_medium opus55_high td

A failure is a (test file, CI config) pair. It counts as caught when the
selection runs that file in that config. Cost is the share of one full CI
pass's CPU and GPU test-seconds (from test-times.json) that the selection
would run.

Policies: `model` is the raw model output, `final` adds what CI adds (edited
test files, the run-all rules), `final-hm` keeps only high/medium-confidence
picks of a well-formed answer and otherwise falls back as `final` does, and
`final+tdN` hedges `final` with TD's files that fit in N% of each config's
test time (N defaults to 100*LLM_TD_HEDGE), and keeps TD's cut of a config in
which that runs nothing, as Selection.split does. `final+tdN` falls back to
TD's own cut when the model call failed or its answer is malformed, as CI
does: select_tests never enforces such a selection, so
selection.apply_selection keeps TD's split. Under enforcement
Selection.split also always runs the files that already failed on the PR and
that TD's split runs; the samples carry no such signal, so the hedge here has
no `always` set, which understates both what enforcement runs and what it
catches on a PR's later pushes.
As in select_tests.select, a sample with no changed files, a change to test
infrastructure (needs_run_all) or more than MAX_CHANGED_FILES runs everything
under every model policy, at no model cost (with no changed files it is an
error, so `final+tdN` keeps TD's cut). A malformed answer counts as an error,
as select_tests records it.
The tag `td` scores the replayed in-repo TD (see td_replay.py) under the
single policy `td25`, its CI cut of the top 25% per config, and ignores
--policies.

Every tag and policy is scored on the same samples: those with a TD replay
of the current test-times.json, sample, STATS artifacts and TD heuristics'
code (see replay_inputs and replay_stats) and a result for every model tag; the rest are counted on stderr. The test files that
import an edited test file are found with git grep at the sample's head; a
sample on which that fails (its head is not in this clone?) is left out and
counted on stderr too, as in CI the grep works.

run_test.py applies TD, and so the selection, only in the pull, trunk and
ROCm workflows, and not on macOS, in xpu or onnx builds, under NO_TD (from
force-no-td or .ci/pytorch/test.sh, see NO_TD_CONFIGS) or with
TESTS_TO_INCLUDE (test/run_test.py's --enable-td default and main(); see
td_job). Other jobs (inductor's unit tests, periodic, slow, ...) run
every test under every policy, so their failures are left out
of the truth, and configs no TD job runs are left out of the cost share
(the report's dollar model prices the inductor ones separately). Jobs are
classified by workflow and name from .github/workflows (see job_runs_td), as
periodic reuses trunk's job names. test-times.json is merged over jobs per
config, so a config that TD and non-TD jobs share (e.g. `default` in
periodic) is priced as one.
CI applies the TD cut and the hedge per job over that job's own test list;
this models a job as a config of test-times.json, and hedges a config no job
of test-times.json records over its fallback job's times, as run_test.py
does: default/<config>, else default/default (see fallback_seconds).

`files` in the summary is the median number of files selected among the
selections that do not run everything.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import functools
import hashlib
import json
import re
import statistics as st
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from tools.stats.import_test_stats import (
    ADDITIONAL_CI_FILES_FOLDER,
    TD_HEURISTIC_HISTORICAL_EDITED_FILES,
    TD_HEURISTIC_PROFILING_FILE,
    TEST_FILE_RATINGS_FILE,
)
from tools.testing.llm_td.eval import REPO
from tools.testing.llm_td.select_tests import git_grep, MAX_CHANGED_FILES
from tools.testing.llm_td.selection import (
    LLM_TD_HEDGE,
    needs_run_all,
    parse_model_output,
    Selection,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Collection


GPU_ENV = re.compile(r"cuda|rocm|sm\d+|xpu|h100|b200|a100|gpu", re.IGNORECASE)
HEDGE = re.compile(r"final\+td(\d+(?:\.\d+)?)?")
MATRIX_CONFIG = re.compile(r'config: "([^"]+)"')
VERSION = re.compile(r"\d+(\.\d+)*")
MATRIX_SUFFIX = re.compile(r" \([^()]*\)")
# .ci/pytorch/test.sh runs a TEST_CONFIG containing one of these through a function that exports NO_TD=1.
NO_TD_CONFIGS = ("dynamo_cpython",)
# The historical-stats artifacts td_replay reads, and the TD code it runs.
STATS_ARTIFACTS = [TEST_FILE_RATINGS_FILE, TD_HEURISTIC_HISTORICAL_EDITED_FILES, TD_HEURISTIC_PROFILING_FILE]  # fmt: skip
TD_CODE = REPO / "tools/testing/target_determination"

# config -> file -> [cpu, gpu] seconds
Seconds = dict[str, dict[str, list[float]]]


@dataclass
class Choice:
    """A Selection plus the files run on top of it in each config (TD's cut or the hedge).

    `other` applies to configs absent from test-times.json and its fallback job."""

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
    """config -> file -> [cpu, gpu] seconds of one full CI pass.

    The "default" job is run_test.py's fallback for unknown jobs, not a job (see select_tests.test_minutes)."""
    out: dict = collections.defaultdict(
        lambda: collections.defaultdict(lambda: [0.0, 0.0])
    )
    for env, configs in json.loads(path.read_text()).items():
        if env == "default":
            continue
        gpu = int(bool(GPU_ENV.search(env)))
        for c, files in configs.items():
            for f, s in files.items():
                out[c][f][gpu] += s
    return {c: dict(files) for c, files in out.items()}


@functools.cache
def fallback_seconds(path: Path) -> dict[str, dict[str, float]]:
    """config -> file -> seconds of test-times.json's "default" job, which run_test.py uses for a job it has no times of."""
    return json.loads(path.read_text()).get("default", {})


def td_workflow(name: str) -> bool:
    """Whether run_test.py enables TD under GITHUB_WORKFLOW `name` (see test/run_test.py --enable-td)."""
    return name in ("trunk", "pull") or name.startswith(("rocm-", "periodic-rocm-"))


def td_job(
    uses: str, build_environment: str, force_no_td: object, tests_to_include: str
) -> bool:
    """The job-level rest of run_test.py's TD check: not IS_MACOS, no xpu/onnx BUILD_ENVIRONMENT, no NO_TD from force-no-td, no TESTS_TO_INCLUDE.

    A tests-to-include expression (pull's and trunk's workflow_dispatch input)
    is empty on PRs. ci_jobs also applies test.sh's NO_TD, which depends on
    the config (NO_TD_CONFIGS). Not modelled, as no workflow file records
    them: NO_TD from the PR's ci-no-td label, the PR-number check (every
    sample is scored as a PR push), and TD switching off when it has no
    prioritizations."""
    include = tests_to_include.strip()
    return (
        uses != "_mac-test.yml"
        and not re.search("xpu|onnx", build_environment)
        and force_no_td not in (True, "true")
        and (not include or include.startswith("${{"))
    )


@functools.cache
def ci_jobs(
    workflows: Path = REPO / ".github/workflows",
) -> dict[bool, set[tuple[str, str, str]]]:
    """Whether TD runs -> the (workflow, job name, config) of the test jobs that run so.

    A reusable workflow's jobs are named and classified under each caller, as
    GitHub reports them ('unit-test / inductor-test') and sets GITHUB_WORKFLOW,
    and also under its own name if it has triggers of its own."""
    wfs = {p.name: yaml.safe_load(p.read_text()) for p in workflows.glob("*.yml")}

    def test_jobs(file: str) -> set[tuple[str, str, bool]]:
        """(job name, config, whether the job itself allows TD) of `file`'s test jobs."""
        jobs = wfs[file].get("jobs") or {}

        def arg(key: str, input: str) -> str:
            """Job `key`'s `with: input`, following `needs.<job>.outputs.<input>` to the build job."""
            value = str((jobs[key].get("with") or {}).get(input, ""))
            if (m := re.search(rf"needs\.([\w-]+)\.outputs\.{input}\b", value)) and m.group(1) in jobs:  # fmt: skip
                return arg(m.group(1), input)
            return value

        out = set()
        for key, job in jobs.items():
            name, uses = job.get("name", key), Path(job.get("uses", "")).name
            if uses in wfs and not uses.startswith("_"):
                out |= {(f"{name} / {n}", c, ok) for n, c, ok in test_jobs(uses)}
            elif "test" in uses:
                w = job.get("with") or {}
                ok = td_job(uses, arg(key, "build-environment"), w.get("force-no-td"), str(w.get("tests-to-include", "")))  # fmt: skip
                out |= {(name, c, ok and not any(n in c for n in NO_TD_CONFIGS)) for c in MATRIX_CONFIG.findall(arg(key, "test-matrix"))}  # fmt: skip
        return out

    out: dict[bool, set[tuple[str, str, str]]] = {True: set(), False: set()}
    for file, wf in wfs.items():
        on = wf.get(True) if isinstance(wf, dict) else None
        if set([on] if isinstance(on, str) else on or []) - {"workflow_call"}:
            workflow = wf.get("name", "")
            for name, c, ok in test_jobs(file):
                out[td_workflow(workflow) and ok].add((workflow, name, c))
    return out


@functools.cache
def runs_td(config: str) -> bool:
    return any(c == config for _, _, c in ci_jobs()[True])


@functools.cache
def job_runs_td(job: str, workflow: str | None = None) -> bool:
    """Whether the test job named `job` of `workflow` runs with TD.

    It is looked up among that workflow's jobs by name, then without matrix
    suffixes ('dynamo-test (3.11)') and with versions wildcarded (CI has moved
    to newer CUDA and Python since), then by config alone unless the name is
    of a macOS, xpu or onnx job. With no `workflow`
    (samples built before it was recorded) every workflow is searched, and a
    job that matches both TD and non-TD jobs, such as a trunk job periodic
    reuses the name of, counts as TD."""
    if workflow is not None and not td_workflow(workflow):
        return False
    name, c = job.rsplit(" / ", 1)[0], config_of(job)
    jobs = ci_jobs()
    for norm in (lambda n: n, lambda n: VERSION.sub("#", MATRIX_SUFFIX.sub("", n))):
        td, other = ({(norm(n), k) for w, n, k in jobs[b] if workflow in (None, w)} for b in (True, False))  # fmt: skip
        if (key := (norm(name), c)) in td | other:
            return key in td and (workflow is None or key not in other)
    return not re.search("macos|xpu|onnx", name) and runs_td(c)


def truth_jobs(s: dict, f: str) -> list[tuple[str | None, str]]:
    """The (workflow, job) `f` failed in; older samples list the job alone, with no workflow."""
    return [(None, j) if isinstance(j, str) else (j[0], j[1]) for j in s.get("truth_jobs", {}).get(f, [])]  # fmt: skip


def truth_files(s: dict, strong: bool) -> list[str]:
    """The truth files of `s` that are candidates, the reproduced ones only for a PR when `strong`."""
    truth: list[str] = s["truth"]
    if strong and s["kind"] == "pr":
        truth = s.get("truth_strong", truth)
    cands = set(s["candidates"])
    return [f for f in truth if f in cands]


def truth_pairs(s: dict, strong: bool) -> list[tuple[str, str]]:
    """The (file, config) failures of `s` in jobs that run TD.

    A file with no recorded job counts as failing in `default`."""
    pairs = set()
    for f in truth_files(s, strong):
        jobs = truth_jobs(s, f)
        td_jobs = [j for wf, j in jobs if job_runs_td(j, wf)] if jobs else ["(default)"]  # fmt: skip
        pairs |= {(f, config_of(j)) for j in td_jobs}
    return sorted(pairs)


def hedge(
    sel: Selection,
    scored: list[str],
    seconds: dict[str, float],
    share: float,
    config: str,
    td_cut: Collection[str],
    untimed: Collection[str],
) -> set[str]:
    """TD's positive-score files that fit in `share` of `seconds`' total, filled greedily in TD's order.

    This is selection.Selection.split for any share and with a job's test
    list taken to be `seconds` and `untimed` (test_hedge_matches_split pins
    the two together). As in split, files the selection already covers cost
    nothing, and a file that does not fit is skipped while smaller ones after
    it may still fit. Files with no recorded time are skipped rather than added
    for free. If the selection covers none of `seconds`' files and the
    `untimed` ones and nothing is hedged, it is TD's cut `td_cut`. Unlike
    split it takes any share and has no `always` files (see the module
    docstring)."""
    budget = share * sum(seconds.values())
    out = set()
    for f in scored:
        t = seconds.get(f, 0.0)
        if t > 0 and t <= budget and not sel.covers(f, config):
            budget -= t
            out.add(f)
    if not out and not any(sel.covers(f, config) for f in [*seconds, *untimed]):
        return set(td_cut)
    return out


grep_at = functools.cache(git_grep)


def grep_head(s: dict, sid: str, failed: set[str]) -> Callable[[str], list[str]]:
    """grep_at at `s`'s head, finding nothing and adding `sid` to `failed` once git grep fails there."""

    def grep(pattern: str) -> list[str]:
        if sid not in failed:
            try:
                return grep_at(pattern, s["sha"])
            except RuntimeError:
                failed.add(sid)
        return []

    return grep


def policy(
    name: str,
    s: dict,
    r: dict | None,
    td: dict | None,
    seconds: Seconds,
    grep: Callable[[str], list[str]] | None = None,
    fallback: dict[str, dict[str, float]] | None = None,
) -> Choice | None:
    """The files `name` runs on sample `s`, or None when its inputs are missing.

    `grep` finds the importers of edited test files (see selection.importers),
    and `fallback` is fallback_seconds."""
    cands = set(s["candidates"])
    if name == "td25":
        if td is None:
            return None
        return Choice(
            Selection(),
            {c: set(fs) for c, fs in td["top25_cfg"].items()},
            set(td["top25"]),
        )
    if name not in ("model", "final", "final-hm") and not HEDGE.fullmatch(name):
        raise ValueError(f"unknown policy {name!r}")
    if unasked(s):
        if HEDGE.fullmatch(name) and not s["changed_files"]:
            return policy("td25", s, r, td, seconds)
        return Choice(Selection(run_all=True))
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
        if malformed(out, cands):
            # Not enforced, so apply_selection keeps TD's split.
            return policy("td25", s, r, td, seconds)
        sel = Selection.from_model_output(out, cands, s["changed_files"], grep)
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
        # As in td_replay.top25_per_config, untimed files are in every job.
        cut, untimed = td["top25_cfg"], cands - set(total)
        per_config = {
            c: hedge(sel, td["scored"], fs, share, c, cut.get(c, ()), untimed)
            for c, fs in times.items()
        }
        # A config no job records is hedged over the fallback job's times, and its test list is every candidate.
        fb = {c: {f: t for f, t in fs.items() if f in cands} for c, fs in (fallback or {}).items()}  # fmt: skip
        for c in fb.keys() - times.keys() - {"default"}:
            per_config[c] = hedge(sel, td["scored"], fb[c], share, c, td["top25"], cands - set(fb[c]))  # fmt: skip
        fs = fb.get("default", {})
        other = hedge(sel, td["scored"], fs, share, "", td["top25"], cands - set(fs))
        return Choice(sel, per_config, other)
    if name == "final-hm":
        # A malformed answer is left for from_model_output to fall back on.
        with contextlib.suppress(ValueError):
            parse_model_output(out, cands)
            if isinstance(out, dict):
                out = {**out, "tests": [t for t in out["tests"] if t.get("confidence") in ("high", "medium")]}  # fmt: skip
    return Choice(Selection.from_model_output(out, cands, s["changed_files"], grep))


def unasked(s: dict) -> bool:
    """Whether select_tests.select runs everything on `s` without asking the model."""
    changed = s["changed_files"]
    return not changed or needs_run_all(changed) or len(changed) > MAX_CHANGED_FILES


def malformed(out: object, cands: set[str]) -> bool:
    """Whether select_tests.select records `out` as a failed call: missing or malformed."""
    try:
        parse_model_output(out, cands)
    except ValueError:
        return True
    return False


def replay_inputs(stats: str, times: bytes, sample: bytes) -> str:
    """The `inputs` hash of a TD replay of `sample` under `times` and the digest `stats` (see replay_stats)."""
    parts = [stats, *(hashlib.sha256(b).hexdigest() for b in (times, sample))]
    return hashlib.sha256(" ".join(parts).encode()).hexdigest()


def replay_stats(stats_dir: Path) -> str:
    """A hash of the STATS_ARTIFACTS in `stats_dir` and of the TD code (TD_CODE) a replay runs."""
    paths = [stats_dir / a for a in STATS_ARTIFACTS] + sorted(TD_CODE.rglob("*.py"))
    parts = [f"{p.name} {hashlib.sha256(p.read_bytes()).hexdigest()}" for p in paths]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


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
    tags: list[str],
    policies: list[str],
    strong: bool,
    samples: Path,
    td_dir: Path,
    stats: str,
) -> list[dict]:
    """Rows of every tag and policy, all on the ids that have a TD replay of the current inputs under `stats` (see replay_stats), git grep at their head and, unless `unasked`, a result for every tag but `td`."""
    seconds = test_seconds(data / "test-times.json")
    fallback = fallback_seconds(data / "test-times.json")
    times = (data / "test-times.json").read_bytes()
    rows = []
    dropped: collections.Counter[str] = collections.Counter()
    ungrepped: set[str] = set()
    no_workflow: set[str] = set()
    for sid in ids:
        loaded = {tag: load(data, tag, sid, samples, td_dir) for tag in tags}
        s, _, td = next(iter(loaded.values()))
        missing = [] if td else ["td replay"]
        if td and td.get("inputs") != replay_inputs(stats, times, (samples / f"{sid}.json").read_bytes()):  # fmt: skip
            missing = [
                "td replay of the current test-times.json, sample, STATS and TD code"
            ]
        if not unasked(s):
            missing += [f"{tag} result" for tag, (_, r, _) in loaded.items() if tag != "td" and r is None]  # fmt: skip
        dropped.update(missing)
        if missing:
            continue
        pairs, cands = truth_pairs(s, strong), set(s["candidates"])
        sample_rows = []
        for tag, (_, r, td) in loaded.items():
            skipped = tag != "td" and unasked(s)
            for pol in ["td25"] if tag == "td" else policies:
                choice = policy(
                    pol, s, r, td, seconds, grep_head(s, sid, ungrepped), fallback
                )
                if choice is None:
                    continue
                hit = [choice.covers(f, c) for f, c in pairs]
                cpu, gpu = cost_fraction(choice, cands, seconds, runs_td)
                sample_rows.append({
                    "tag": tag, "policy": pol, "id": sid, "kind": s["kind"], "pairs": len(pairs), "hit": sum(hit),
                    "cpu": cpu, "gpu": gpu, "run_all": choice.sel.run_all, "files": len(choice.files()),
                    "cost_usd": 0.0 if skipped else (r or {}).get("cost_usd") or 0.0,
                    "error": not s["changed_files"] if skipped else bool(r) and (bool(r["error"]) or malformed(r["out"], cands)),
                    "missed": [p for p, h in zip(pairs, hit) if not h],
                })  # fmt: skip
        if sid in ungrepped:
            dropped["working git grep at their head (not in this clone?)"] += 1
            continue
        rows += sample_rows
        if any(wf is None for f in truth_files(s, strong) for wf, _ in truth_jobs(s, f)):  # fmt: skip
            no_workflow.add(sid)
    for why, n in sorted(dropped.items()):
        print(f"dropped {n} of {len(ids)} ids with no {why}", file=sys.stderr)
    if no_workflow:
        print(f"scored {len(no_workflow)} of {len(ids)} ids with truth jobs that record no workflow: a job name TD and non-TD workflows share counts as TD", file=sys.stderr)  # fmt: skip
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
    p.add_argument("--stats", type=Path, default=REPO / ADDITIONAL_CI_FILES_FOLDER, help="td_replay.py's --stats")  # fmt: skip
    p.add_argument("--policies", default="model,final,final-hm,final+td20")
    p.add_argument(
        "--all-truth",
        action="store_true",
        help="score PRs on every truth, not only reproduced ones",
    )
    p.add_argument("--json", type=Path, help="also write per-sample rows here")
    p.add_argument("tags", nargs="+")
    args = p.parse_args()
    if missing := [a for a in STATS_ARTIFACTS if not (args.stats / a).exists()]:
        p.error(f"{args.stats} lacks {missing}, which td_replay.py read")
    ids = args.ids.read_text().split()
    samples, td_dir = (
        args.samples_dir or args.data / "samples",
        args.td_dir or args.data / "td",
    )
    rows = score(args.data, ids, args.tags, args.policies.split(","), not args.all_truth, samples, td_dir, replay_stats(args.stats))  # fmt: skip
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
