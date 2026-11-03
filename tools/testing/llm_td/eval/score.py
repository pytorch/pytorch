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
test time (N defaults to 100*LLM_TD_HEDGE). `final+tdN` falls back
to TD's own cut when the model call failed or its answer is malformed, as
run_test.py does under enforcement (see the PR above this one in the stack).
The tag `td` scores the replayed in-repo TD (see td_replay.py) under the
single policy `td25`, its CI cut of the top 25% per config, and ignores
--policies.

Every tag and policy is scored on the same samples: those with a TD replay
and a result for every model tag; the rest are counted on stderr. The test
files that import an edited test file are found with git grep at the sample's
head; a sample whose head is not in this clone is scored without them and
counted on stderr too.

run_test.py applies TD, and so the selection, only in the pull, trunk and
ROCm workflows, and not on macOS, in xpu or onnx builds or under NO_TD
(test/run_test.py's --enable-td default; _linux-test.yml sets NO_TD from
force-no-td). Other jobs (inductor's unit tests, periodic, slow, ...) run
every test under every policy, so their failures are left out
of the truth, and configs no TD job runs are left out of the cost share
(the report's dollar model prices the inductor ones separately). Jobs are
classified by workflow and name from .github/workflows (see job_runs_td), as
periodic reuses trunk's job names. test-times.json is merged over jobs per
config, so a config that TD and non-TD jobs share (e.g. `default` in
periodic) is priced as one.
CI applies the TD cut and the hedge per job over that job's own test list;
this models a job as a config of test-times.json, and for a failure in a
config missing from test-times.json falls back to a cut over all candidates.

`files` in the summary is the median number of files selected among the
selections that do not run everything.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import functools
import json
import re
import statistics as st
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from tools.testing.llm_td.eval import REPO
from tools.testing.llm_td.select_tests import git_grep
from tools.testing.llm_td.selection import (
    LLM_TD_HEDGE,
    needs_run_all,
    parse_model_output,
    Selection,
)


if TYPE_CHECKING:
    from collections.abc import Callable


GPU_ENV = re.compile(r"cuda|rocm|sm\d+|xpu|h100|b200|a100|gpu", re.IGNORECASE)
HEDGE = re.compile(r"final\+td(\d+(?:\.\d+)?)?")
MATRIX_CONFIG = re.compile(r'config: "([^"]+)"')
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
    """Whether run_test.py enables TD under GITHUB_WORKFLOW `name` (see test/run_test.py --enable-td)."""
    return name in ("trunk", "pull") or name.startswith(("rocm-", "periodic-rocm-"))


def td_job(uses: str, build_environment: str, force_no_td: object) -> bool:
    """The rest of run_test.py's --enable-td check: not IS_MACOS, no xpu/onnx BUILD_ENVIRONMENT, no NO_TD."""
    return uses != "_mac-test.yml" and not re.search("xpu|onnx", build_environment) and force_no_td not in (True, "true")  # fmt: skip


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
                ok = td_job(uses, arg(key, "build-environment"), (job.get("with") or {}).get("force-no-td"))  # fmt: skip
                out |= {(name, c, ok) for c in MATRIX_CONFIG.findall(arg(key, "test-matrix"))}  # fmt: skip
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
    to newer CUDA and Python since), then by config alone. With no `workflow`
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
    return runs_td(c)


def truth_jobs(s: dict, f: str) -> list[tuple[str | None, str]]:
    """The (workflow, job) `f` failed in; older samples list the job alone, with no workflow."""
    return [(None, j) if isinstance(j, str) else (j[0], j[1]) for j in s.get("truth_jobs", {}).get(f, [])]  # fmt: skip


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
) -> set[str]:
    """TD's positive-score files that fit in `share` of `seconds`' total, filled greedily in TD's order.

    As in run_test.py's split (see the PR above), files the selection already
    covers cost nothing, and a file that does not fit is skipped while smaller
    ones after it may still fit. Files with no recorded time are skipped rather
    than added for free."""
    budget = share * sum(seconds.values())
    out = set()
    for f in scored:
        t = seconds.get(f, 0.0)
        if t > 0 and t <= budget and not sel.covers(f, config):
            budget -= t
            out.add(f)
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
) -> Choice | None:
    """The files `name` runs on sample `s`, or None when its inputs are missing.

    `grep` finds the importers of edited test files (see selection.importers)."""
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
        failed = out is None
        try:
            parse_model_output(out, cands)
        except ValueError:
            failed = True
        if failed and not needs_run_all(s["changed_files"]):
            # As run_test.py does under enforcement, see the PR above.
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
        per_config = {
            c: hedge(sel, td["scored"], fs, share, c) for c, fs in times.items()
        }
        return Choice(sel, per_config, hedge(sel, td["scored"], total, share, ""))
    if name == "final-hm":
        # A malformed answer is left for from_model_output to fall back on.
        with contextlib.suppress(ValueError):
            parse_model_output(out, cands)
            if isinstance(out, dict):
                out = {**out, "tests": [t for t in out["tests"] if t.get("confidence") in ("high", "medium")]}  # fmt: skip
    elif name != "final":
        raise ValueError(f"unknown policy {name!r}")
    return Choice(Selection.from_model_output(out, cands, s["changed_files"], grep))


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
) -> list[dict]:
    """Rows of every tag and policy, all on the ids that have a TD replay and a result for every tag but `td`."""
    seconds = test_seconds(data / "test-times.json")
    rows = []
    dropped: collections.Counter[str] = collections.Counter()
    ungrepped: set[str] = set()
    no_workflow: set[str] = set()
    for sid in ids:
        loaded = {tag: load(data, tag, sid, samples, td_dir) for tag in tags}
        missing = [] if (td_dir / f"{sid}.json").exists() else ["td replay"]
        missing += [f"{tag} result" for tag, (_, r, _) in loaded.items() if tag != "td" and r is None]  # fmt: skip
        dropped.update(missing)
        if missing:
            continue
        for tag, (s, r, td) in loaded.items():
            if any(wf is None for f in s["truth"] for wf, _ in truth_jobs(s, f)):
                no_workflow.add(sid)
            pairs = truth_pairs(s, strong)
            cands = set(s["candidates"])
            for pol in ["td25"] if tag == "td" else policies:
                choice = policy(pol, s, r, td, seconds, grep_head(s, sid, ungrepped))
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
    for why, n in sorted(dropped.items()):
        print(f"dropped {n} of {len(ids)} ids with no {why}", file=sys.stderr)
    if no_workflow:
        print(f"scored {len(no_workflow)} of {len(ids)} ids with truth jobs that record no workflow: a job name TD and non-TD workflows share counts as TD", file=sys.stderr)  # fmt: skip
    if ungrepped:
        print(f"scored {len(ungrepped)} of {len(ids)} ids without importers: git grep failed at their head (not in this clone?)", file=sys.stderr)  # fmt: skip
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
    p.add_argument("--policies", default="model,final,final-hm,final+td20")
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
    rows = score(args.data, ids, args.tags, args.policies.split(","), not args.all_truth, samples, td_dir)  # fmt: skip
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
