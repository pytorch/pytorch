"""Score selector results: recall of real failures, and the share of CI test time selected.

    python -m tools.testing.llm_selector.eval.score --data DIR --ids tools/testing/llm_selector/eval/ids/eval.txt sonnet5_medium opus55_high td

A failure is a (test file, CI config) pair. It counts as caught when the
selection runs that file in that config. Cost is the share of one full CI
pass's CPU and GPU test-seconds (from test-times.json) that the selection
would run, counting only the jobs and environments that run TD (see below).

Policies: `model` is the raw model output, `final` adds what CI adds (edited
test files, the run-all rules), `final-hm` keeps only high/medium-confidence
picks of a well-formed answer and otherwise falls back as `final` does, and
`final+tdN` hedges `final` with TD's files that fit in N% of each config's
test time (N defaults to 100*LLM_SELECTOR_HEDGE), and keeps TD's cut alone in a
config in which that runs nothing or that has no test time, as
Selection.split and apply_selection do. `final+tdN` falls back to
TD's own cut when the model call failed or its answer is malformed, as CI
does: select_tests never enforces such a selection, so
selection.apply_selection keeps TD's split. As Selection.split does under
enforcement, `final+tdN` also runs a sample's `previous_failures` (head
samples only, see build_dataset) that are in TD's cut of the config, outside
the hedge budget; td_replay ranks them first, as TD's PreviouslyFailedInPR
does.
As in select_tests.select, a sample with no changed files, a change to test
infrastructure or more than MAX_CHANGED_FILES (select_tests.run_all_reason) runs everything
under every model policy, at no model cost; `final+tdN` keeps TD's cut in a
config with no test time there too, as it does for any run-all. A malformed answer counts as an
error, as select_tests records it, and errors still transient (throttled or
5xx) after run_eval's retries are counted on stderr.
The tag `td` scores the replayed in-repo TD (see td_replay.py) under the
single policy `td25`, its CI cut of the top 25% per config, and ignores
--policies.

Every tag and policy is scored on the same samples: those with a TD replay
of the current test-times.json, sample, STATS artifacts and replay code
(see replay_inputs and replay_stats) and a result for every model tag made
from the prompt the sample gives now (see prompt_sha); the rest are counted on stderr. The test files that
import an edited test file are found with git grep at the sample's head; a
sample whose head is not in this clone, or on which that grep fails, is left
out under every policy and counted on stderr too, as in CI the grep works; so
is one whose diff adds a candidate test file and whose main_base (default
base) is not in this clone.

run_test.py applies TD, and so the selection, only in the pull, trunk and
ROCm workflows, and not on macOS, in xpu or onnx builds, under NO_TD (from
force-no-td or .ci/pytorch/test.sh, see NO_TD_CONFIGS) or with
TESTS_TO_INCLUDE (test/run_test.py's --enable-td default, td_enabled_default,
and its main(); see td_job). Other jobs (inductor's unit tests, periodic, slow, ...) run
every test under every policy, so their failures are left out
of the truth, and configs no TD job runs are left out of the cost share
(the report's dollar model prices the inductor ones separately). Jobs are
classified by workflow and name from .github/workflows (see job_runs_td), as
periodic reuses trunk's job names. test-times.json is merged over jobs per
config, so a config that TD and non-TD jobs share (e.g. `default` in
periodic) is priced as one.
CI applies the TD cut and the hedge per job over that job's own test list;
this models a job as a config of test-times.json (see job_files), and hedges a config no job
of test-times.json records over its fallback job's times, as run_test.py
does: default/<config>, else default/default (see fallback_seconds).
CI applies them, and their fallbacks, per run_test.py call, and this pools
a config's calls into one list. That differs where test.sh makes several
calls with their own --include lists: inductor's two, which run no newly
added file, and small extra calls on default's shard 1. Modelling inductor's
calls apart moves no failure and the mean cost shares by under 0.6 points,
except td25's GPU share on the eval set (-2 points).

`files` in the summary is the median number of files selected among the
selections that do not run everything.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import fnmatch
import functools
import hashlib
import inspect
import json
import re
import statistics as st
import subprocess
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
from tools.testing.llm_selector.eval import REPO
from tools.testing.llm_selector.select_tests import (
    added_test_files,
    AGENTIC_TOOLS,
    build_prompt,
    CHECKOUT_PREAMBLE,
    git_grep,
    git_paths,
    run_all_reason,
    SCHEMA,
    SYSTEM_PROMPT,
    test_minutes,
)
from tools.testing.llm_selector.selection import (
    LLM_SELECTOR_HEDGE,
    parse_model_output,
    Selection,
)
from tools.testing.target_determination.heuristics.utils import (
    NO_TD_BUILD_ENVS,
    td_enabled_job,
    td_enabled_workflow,
)
from tools.testing.test_run import TestRun


if TYPE_CHECKING:
    from collections.abc import Callable, Collection


GPU_ENV = re.compile(r"cuda|rocm|sm\d+|h100|b200|a100|gpu", re.IGNORECASE)
HEDGE = re.compile(r"final\+td(\d+(?:\.\d+)?)?")
# The policies of a model tag besides the final+tdN ones.
POLICIES = ("model", "final", "final-hm")
MATRIX_CONFIG = re.compile(r"""\{\s*config:\s*['"]?([\w.-]+)""")
VERSION = re.compile(r"\d+(\.\d+)*")
MATRIX_SUFFIX = re.compile(r" \([^()]*\)")
# run_test.py never runs TD on macOS or in NO_TD_BUILD_ENVS builds.
NO_TD_ENV = re.compile("|".join(("macos", *NO_TD_BUILD_ENVS)))
# The patterns of .ci/pytorch/test.sh's TEST_CONFIG branches that export NO_TD=1.
NO_TD_CONFIGS = ("*dynamo_cpython*",)
# The historical-stats artifacts td_replay reads, and the TD code it runs.
STATS_ARTIFACTS = [TEST_FILE_RATINGS_FILE, TD_HEURISTIC_HISTORICAL_EDITED_FILES, TD_HEURISTIC_PROFILING_FILE]  # fmt: skip
TD_CODE = REPO / "tools/testing/target_determination"
REPLAY_CODE = [Path(__file__).with_name("td_replay.py"), REPO / "tools/testing/test_run.py"]  # fmt: skip

# config -> file -> [cpu, gpu] seconds
Seconds = dict[str, dict[str, list[float]]]


@dataclass
class Choice:
    """A Selection plus the files run on top of it in each config (TD's cut or the hedge).

    `other` applies to configs absent from test-times.json and its fallback job,
    and "" in `unenforced` to those. A config in `unenforced` runs only TD's cut."""

    sel: Selection
    per_config: dict[str, set[str]] = field(default_factory=dict)
    other: set[str] = field(default_factory=set)
    unenforced: set[str] = field(default_factory=set)

    def covers(self, test_file: str, test_config: str) -> bool:
        known = test_config in self.per_config
        cut = self.per_config[test_config] if known else self.other
        if (test_config if known else "") in self.unenforced:
            return test_file in cut
        return self.sel.covers(test_file, test_config) or test_file in cut

    def files(self) -> set[str]:
        return self.sel.files.union(*self.per_config.values())


def config_of(job: str) -> str:
    """'linux-jammy-py3.10-clang21 / test (crossref, 1, 2, lf.linux.2xlarge)' -> 'crossref'"""
    m = re.search(r"\(([^,()]+), \d+, \d+", job) or re.search(r"\(([^,()]+)\)", job)
    return m.group(1).strip() if m else job.strip()


@functools.cache
def test_seconds(path: Path) -> Seconds:
    """config -> file -> [cpu, gpu] seconds of one full CI pass.

    The "default" job is run_test.py's fallback for unknown jobs, not a job
    (see select_tests.test_minutes), and environments that never run TD are left out."""
    out: dict = collections.defaultdict(
        lambda: collections.defaultdict(lambda: [0.0, 0.0])
    )
    for env, configs in json.loads(path.read_text()).items():
        if env == "default" or NO_TD_ENV.search(env):
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


def td_job(
    uses: str, build_environment: str, force_no_td: object, tests_to_include: str
) -> bool:
    """The job-level rest of run_test.py's TD check: td_enabled_job, with NO_TD from force-no-td, and main()'s TESTS_TO_INCLUDE override from tests-to-include.

    A tests-to-include expression (pull's and trunk's workflow_dispatch input)
    is empty on PRs. ci_jobs also applies test.sh's NO_TD, which depends on
    the config (NO_TD_CONFIGS). Not modelled, as no workflow file records
    them: NO_TD from the PR's ci-no-td label, the PR-number check (every
    sample is scored as a PR push), and TD switching off when it has no
    prioritizations."""
    include = "" if tests_to_include.strip().startswith("${{") else tests_to_include
    macos, no_td = uses == "_mac-test.yml", force_no_td in (True, "true")
    return not include.strip() and td_enabled_job(build_environment, macos=macos, no_td=no_td)  # fmt: skip


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
                out |= {(name, c, ok and not any(fnmatch.fnmatchcase(c, n) for n in NO_TD_CONFIGS)) for c in MATRIX_CONFIG.findall(arg(key, "test-matrix"))}  # fmt: skip
        return out

    out: dict[bool, set[tuple[str, str, str]]] = {True: set(), False: set()}
    for file, wf in wfs.items():
        on = wf.get(True) if isinstance(wf, dict) else None
        if set([on] if isinstance(on, str) else on or []) - {"workflow_call"}:
            workflow = wf.get("name", "")
            for name, c, ok in test_jobs(file):
                out[td_enabled_workflow(workflow) and ok].add((workflow, name, c))
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
    if workflow is not None and not td_enabled_workflow(workflow):
        return False
    name, c = job.rsplit(" / ", 1)[0], config_of(job)
    jobs = ci_jobs()
    for norm in (lambda n: n, lambda n: VERSION.sub("#", MATRIX_SUFFIX.sub("", n))):
        td, other = ({(norm(n), k) for w, n, k in jobs[b] if workflow in (None, w)} for b in (True, False))  # fmt: skip
        if (key := (norm(name), c)) in td | other:
            return key in td and (workflow is None or key not in other)
    return not NO_TD_ENV.search(name) and runs_td(c)


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


def job_files(s: dict, seconds: Seconds) -> dict[str, set[str]]:
    """config -> the candidates of `s` in the test list of that config's job.

    That is the config's recorded files, the candidates no config records (CI
    amends to the job's whole selected_tests, so a newly added file is in every
    job), and the truth files that failed in it, which CI ran there. TD's cut
    (td_replay) and the hedge both work over these lists."""
    cands = set(s["candidates"])
    untimed = cands - set().union(*seconds.values())
    out = {c: (cands & fs.keys()) | untimed for c, fs in seconds.items()}
    for f in truth_files(s, strong=False):
        for c in {config_of(j) for _, j in truth_jobs(s, f)} or {"default"}:
            if c in out:
                out[c].add(f)
    return out


def hedge(
    sel: Selection,
    scored: list[str],
    seconds: dict[str, float],
    share: float,
    config: str,
    td_cut: Collection[str],
    untimed: Collection[str],
    previous: Collection[str] = (),
) -> set[str]:
    """What Selection.split adds to `sel` in a job whose test list is `seconds` and `untimed`, ranked by TD with `scored` positive, given a `share` hedge.

    That is the `previous` failures in TD's cut `td_cut` and TD's positive-score
    files that fit in `share` of `seconds`' total. It is `td_cut` where
    apply_selection keeps TD's split: when the split chooses nothing, or
    `seconds` sums to 0."""
    if not sum(seconds.values()):
        return set(td_cut)
    positive = set(scored)
    job = dict.fromkeys([*seconds, *untimed])
    ranked = [TestRun(f) for f in scored if f in job] + [TestRun(f) for f in job if f not in positive]  # fmt: skip
    split = sel.split(ranked, lambda x: float(x.test_file in positive), seconds, config, previous_failures=previous, td_include=td_cut, share=share)  # fmt: skip
    if split.chose_nothing:
        return set(td_cut)
    return split.previous_failures_added | split.hedged


grep_at = functools.cache(git_grep)


@functools.cache
def has_head(sha: str) -> bool:
    """Whether this clone has the commit `sha`, at which grep_at greps."""
    commit = f"{sha}^{{commit}}"
    p = subprocess.run(["git", "cat-file", "-e", commit], cwd=REPO, capture_output=True)
    return p.returncode == 0


def grep_head(s: dict, sid: str, failed: set[str]) -> Callable[[str], list[str]]:
    """grep_at at `s`'s head, finding nothing and adding `sid` to `failed` once git grep fails there."""

    def grep(pattern: str) -> list[str]:
        if sid not in failed:
            try:
                # A copy: selection.data_file_readers extends what grep returns.
                return list(grep_at(pattern, s["sha"]))
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
    """The files `name` runs on sample `s`, or None for td25 and final+tdN when
    `td` is None. A missing model result `r` counts as a failed call: run-all,
    or TD's cut under final+tdN.

    `grep` finds the importers of edited test files (see selection.importers),
    and `fallback` is fallback_seconds."""
    cands = set(s["candidates"])
    if name == "td25":
        if td is None:
            return None
        cut = {c: set(fs) for c, fs in td["top25_cfg"].items()}
        # TD runs only its cut, not even the files a selection always covers.
        return Choice(Selection(), cut, set(td["top25"]), {*cut, ""})
    if name not in POLICIES and not HEDGE.fullmatch(name):
        raise ValueError(f"unknown policy {name!r}")
    out = (r or {}).get("out")
    if m := HEDGE.fullmatch(name):
        if td is None:
            return None
        share = float(m.group(1)) / 100 if m.group(1) else LLM_SELECTOR_HEDGE
        sel = Selection(run_all=True) if unasked(s) else Selection.from_model_output(out, cands, s["changed_files"], grep, s.get("deleted", []))  # fmt: skip
        if sel.error is not None:
            # Not enforced, so apply_selection keeps TD's split.
            return policy("td25", s, r, td, seconds)
        times = {
            c: {f: cpu + gpu for f, (cpu, gpu) in fs.items() if f in cands}
            for c, fs in seconds.items()
        }
        cut, jobs = td["top25_cfg"], job_files(s, seconds)
        prev = set(s.get("previous_failures", ()))
        # Under run-all hedge adds nothing, but a config with no test time still keeps TD's cut.
        per_config = {c: hedge(sel, td["scored"], fs, share, c, cut.get(c, ()), jobs[c] - fs.keys(), prev) for c, fs in times.items()}  # fmt: skip
        # A config no job records is hedged over the fallback job's times, and its test list is every candidate.
        fb = {c: {f: t for f, t in fs.items() if f in cands} for c, fs in (fallback or {}).items()}  # fmt: skip
        for c in fb.keys() - times.keys() - {"default"}:
            per_config[c] = hedge(sel, td["scored"], fb[c], share, c, td["top25"], cands - set(fb[c]), prev)  # fmt: skip
            times[c] = fb[c]
        default = fb.get("default", {})
        other = hedge(sel, td["scored"], default, share, "", td["top25"], cands - set(default), prev)  # fmt: skip
        unenforced = {c for c, fs in [*times.items(), ("", default)] if not sum(fs.values())}  # fmt: skip
        return Choice(sel, per_config, other, unenforced)
    if unasked(s):
        return Choice(Selection(run_all=True))
    if name == "model":
        return Choice(
            Selection(run_all=True)
            if out is None
            else Selection.from_model_output(out, cands, [], grep)
        )
    if name == "final-hm":
        # A malformed answer is left for from_model_output to fall back on.
        with contextlib.suppress(ValueError):
            parse_model_output(out, cands, grep)
            if isinstance(out, dict):
                out = {**out, "tests": [t for t in out["tests"] if t.get("confidence") in ("high", "medium")]}  # fmt: skip
    return Choice(Selection.from_model_output(out, cands, s["changed_files"], grep, s.get("deleted", [])))  # fmt: skip


def unasked(s: dict) -> bool:
    """Whether select_tests.select runs everything on `s` without asking the model."""
    return run_all_reason(s["changed_files"]) is not None


def sample_prompt(sample: dict, minutes: dict) -> str:
    """The prompt select_tests.select builds for `sample`. Its added test
    files are the candidates the diff adds, as a new file or the new path of a
    moved one (the diff detects renames, while select diffs with --no-renames),
    except those already in the tree at main_base (default base), whose TESTS
    gave the candidates."""
    found = re.findall(r"^diff --git a/\S+ b/(\S+)\nnew file mode |^rename to (\S+)$", sample["diff"], re.MULTILINE)  # fmt: skip
    added = sorted({a or b for a, b in found} & {f"test/{c}.py" for c in sample["candidates"]})  # fmt: skip
    listed = git_paths("ls-tree", "--name-only", sample.get("main_base") or sample["base"], "--", *added) if added else []  # fmt: skip
    new = added_test_files(added, [p.removeprefix("test/").removesuffix(".py") for p in listed])  # fmt: skip
    return build_prompt(sample["title"], sample["body"] or "", sample["diffstat"], sample["diff"], sample["candidates"], minutes, new)  # fmt: skip


def prompt_sha(prompt: str, agentic: bool = False) -> str:
    """A hash of the text the model is sent: the system prompt, SCHEMA and `prompt` and, when `agentic`, AGENTIC_TOOLS and CHECKOUT_PREAMBLE. query_model's other CLI flags (permissions, --add-dir, --bare) are not hashed."""
    sent = [SYSTEM_PROMPT, SCHEMA, prompt]
    if agentic:
        sent += [AGENTIC_TOOLS, CHECKOUT_PREAMBLE]
    return hashlib.sha256(json.dumps(sent).encode()).hexdigest()


def replay_inputs(stats: str, times: bytes, sample: bytes) -> str:
    """The `inputs` hash of a TD replay of `sample` under `times` and the digest `stats` (see replay_stats)."""
    parts = [stats, *(hashlib.sha256(b).hexdigest() for b in (times, sample))]
    return hashlib.sha256(" ".join(parts).encode()).hexdigest()


def replay_stats(stats_dir: Path) -> str:
    """A hash of the STATS_ARTIFACTS in `stats_dir` and of the code a replay runs: TD_CODE, REPLAY_CODE and this module's functions td_replay calls, with the module constants they read."""
    paths = [stats_dir / a for a in STATS_ARTIFACTS] + sorted(TD_CODE.rglob("*.py")) + REPLAY_CODE  # fmt: skip
    parts = [f"{p.name} {hashlib.sha256(p.read_bytes()).hexdigest()}" for p in paths]
    # getclosurevars sees only the globals a function reads itself: a constant
    # read through a helper that is not in this tuple is missed.
    for f in (test_seconds, job_files, truth_files, truth_jobs, config_of):
        read = inspect.getclosurevars(inspect.unwrap(f)).globals
        consts = [f"{k} {v!r}" for k, v in sorted(read.items()) if k.isupper()]
        parts += [inspect.getsource(f), *consts]
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
    # The tag td is scored only as td25, on its replay.
    r = json.loads(rp.read_text()) if tag != "td" and rp.exists() else None
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
    """Rows of every tag and policy, all on the distinct ids that have a sample, a TD replay of the current inputs under `stats` (see replay_stats), git grep at their head and, unless `unasked`, a result for every tag but `td` of the prompt the sample gives now (see prompt_sha). Raises ValueError if a tag's results mix --model, --effort or --agentic."""
    ids = list(dict.fromkeys(ids))
    no_grep = "working git grep at their head (not in this clone?)"
    no_base = "main_base (default base) in this clone"
    settings: collections.defaultdict[str, set[tuple]] = collections.defaultdict(set)
    seconds = test_seconds(data / "test-times.json")
    fallback = fallback_seconds(data / "test-times.json")
    times = (data / "test-times.json").read_bytes()
    minutes = test_minutes(json.loads(times))
    rows = []
    dropped: collections.Counter[str] = collections.Counter()
    ungrepped: set[str] = set()
    no_workflow: set[str] = set()
    throttled: collections.Counter[str] = collections.Counter()
    for sid in ids:
        if not (samples / f"{sid}.json").exists():
            dropped["sample"] += 1
            continue
        loaded = {tag: load(data, tag, sid, samples, td_dir) for tag in tags}
        s, _, td = next(iter(loaded.values()))
        for tag, (_, r, _) in loaded.items():
            if r is not None:
                settings[tag].add((r.get("model"), r.get("effort"), r.get("agentic")))
        missing = [] if td else ["td replay"]
        if td and td.get("inputs") != replay_inputs(stats, times, (samples / f"{sid}.json").read_bytes()):  # fmt: skip
            missing = [
                "td replay of the current test-times.json, sample, STATS and TD code"
            ]
        if not unasked(s):
            results = {tag: r for tag, (_, r, _) in loaded.items() if tag != "td"}
            missing += [f"{tag} result" for tag, r in results.items() if r is None]
            try:
                prompt = sample_prompt(s, minutes)
            except subprocess.CalledProcessError:
                missing.append(no_base)
            else:
                missing += [f"{tag} result of the current sample, test-times.json, prompt and schema" for tag, r in results.items() if r is not None and r.get("prompt_sha") != prompt_sha(prompt, bool(r.get("agentic")))]  # fmt: skip
        # Probed for every policy: grep_head fails only where a policy greps.
        if not has_head(s["sha"]):
            missing.append(no_grep)
        dropped.update(missing)
        if missing:
            continue
        pairs, cands = truth_pairs(s, strong), set(s["candidates"])
        sample_rows, sample_throttled = [], []
        for tag, (_, r, td) in loaded.items():
            skipped = tag != "td" and unasked(s)
            error = not skipped and bool(r) and (bool(r["error"]) or Selection.from_model_output(r["out"], cands, [], grep_head(s, sid, ungrepped)).error is not None)  # fmt: skip
            if error and (r or {}).get("transient"):
                sample_throttled.append(tag)
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
                    "cost_usd": None if skipped else (r or {}).get("cost_usd") or 0.0,
                    "error": error,
                    "missed": [p for p, h in zip(pairs, hit) if not h],
                })  # fmt: skip
        if sid in ungrepped:
            dropped[no_grep] += 1
            continue
        rows += sample_rows
        throttled.update(sample_throttled)
        if any(wf is None for f in truth_files(s, strong) for wf, _ in truth_jobs(s, f)):  # fmt: skip
            no_workflow.add(sid)
    if mixed := sorted(t for t, v in settings.items() if len(v) > 1):
        raise ValueError(f"the results of {mixed} mix --model, --effort or --agentic")
    for why, n in sorted(dropped.items()):
        print(f"dropped {n} of {len(ids)} ids with no {why}", file=sys.stderr)
    for tag, n in sorted(throttled.items()):
        print(f"scored {n} {tag} results as errors that still had transient (throttled or 5xx) errors after run_eval's retries", file=sys.stderr)  # fmt: skip
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
        # Samples select_tests answers without the model have no cost.
        costs = [x["cost_usd"] for x in xs if x["cost_usd"] is not None] or [0.0]
        subset = [x["files"] for x in xs if not x["run_all"]] or [0]
        out.append({
            "tag": tag, "policy": pol, "kind": kind, "n": len(xs), "n_failing": len(failing),
            "caught_any": sum(x["hit"] > 0 for x in failing) / max(len(failing), 1),
            "caught_all": sum(x["hit"] == x["pairs"] for x in failing) / max(len(failing), 1),
            "pair_recall": sum(x["hit"] for x in failing) / max(sum(x["pairs"] for x in failing), 1),
            "cpu": st.mean(x["cpu"] for x in xs), "gpu": st.mean(x["gpu"] for x in xs),
            "run_all": sum(x["run_all"] for x in xs), "median_files": st.median(subset),
            "usd_per_call": st.mean(costs), "errors": sum(x["error"] for x in xs),
        })  # fmt: skip
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--ids", type=Path, required=True)
    p.add_argument("--td-dir", type=Path, help="td_replay.py output; default: DATA/td")
    p.add_argument("--stats", type=Path, default=REPO / ADDITIONAL_CI_FILES_FOLDER, help="td_replay.py's --stats")  # fmt: skip
    p.add_argument("--policies", default="model,final,final-hm,final+td")
    p.add_argument(
        "--all-truth",
        action="store_true",
        help="score PRs on every truth, not only reproduced ones",
    )
    p.add_argument("--json", type=Path, help="also write per-sample rows here")
    p.add_argument("tags", nargs="+")
    args = p.parse_args()
    policies = list(dict.fromkeys(args.policies.split(",")))
    if unknown := [x for x in policies if x not in POLICIES and not HEDGE.fullmatch(x)]:
        p.error(f"unknown --policies {unknown}; the tag td is always scored as td25")
    shares = [float(m.group(1)) / 100 if m.group(1) else LLM_SELECTOR_HEDGE for x in policies if (m := HEDGE.fullmatch(x))]  # fmt: skip
    if len(shares) != len(set(shares)):
        p.error(f"--policies {policies} has two hedge policies of the same share")
    if missing := [a for a in STATS_ARTIFACTS if not (args.stats / a).exists()]:
        p.error(f"{args.stats} lacks {missing}, which td_replay.py read")
    ids = args.ids.read_text().split()
    samples, td_dir = args.data / "samples", args.td_dir or args.data / "td"
    rows = score(args.data, ids, args.tags, policies, not args.all_truth, samples, td_dir, replay_stats(args.stats))  # fmt: skip
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
