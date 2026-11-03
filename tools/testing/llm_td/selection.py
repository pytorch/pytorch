"""The test selection an LLM makes for one change, and how CI applies it.

A selection names test files (paths under test/ without .py), whole test
directories, and wrapper sweeps (configs such as dynamo_wrapped that re-run
ordinary test files in a special mode). A test file is covered in a given CI
config when the selection names the file, names that config as a sweep, or is
run_all, and so is every test file the model was not shown. When run_test.py
enforces the selection, it runs the covered files plus, within a hedge budget,
the files that already failed on the PR and TD's top-scored files.
"""

from __future__ import annotations

import contextlib
import json
import re
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from tools.testing.target_determination.heuristics.edited_by_pr import (
    ADDITIONAL_MAPPINGS,
)
from tools.testing.target_determination.heuristics.previously_failed_in_pr import (
    get_previous_failures,
    read_additional_test_failures_file,
)
from tools.testing.target_determination.heuristics.utils import get_pr_number


if TYPE_CHECKING:
    from collections.abc import Callable, Collection
    from pathlib import Path

    from tools.testing.test_run import TestRun


SWEEPS = ("dynamo_wrapped", "crossref")
LLM_TD_SELECTION_FILE = "llm_td_selection.json"
# Longest chain of test modules importing each other that importers() follows.
MAX_IMPORT_HOPS = 8
# More edited failure-list classes than this are not looked up one by one.
MAX_FAILURE_LIST_CLASSES = 50
# When enforcing, run_test.py also runs, greedily in rank order, the files that
# already failed on the PR and then TD's positively scored files, as long as
# they have a known time and fit in this share (20%) of the config's test time
# (all shards together): TD and the model miss largely different failures.
LLM_TD_HEDGE = 0.20


# Tests that run the files a change matches although they do not import them:
# the drivers of the directories discover_tests.py blocklists (except
# torch_openreg/tests, whose driver test_openreg is never a candidate), TD's
# ADDITIONAL_MAPPINGS, and the compiled autograd rerun of test_autograd.
DRIVER_MAPPINGS = {
    **ADDITIONAL_MAPPINGS,
    "test_ao_sparsity": [r"^test/ao/"],
    "test_autograd": [r"^test/autograd/"],
    "test_fx": [r"^test/fx/"],
    "test_jit": [r"^test/jit/"],
    "test_package": [r"^test/package/"],
    "test_quantization": [r"^test/quantization/"],
    "inductor/test_compiled_autograd": [r"^test/compiled_autograd_skips/"],
}

# common_utils xfails or skips a test named in one of these lists, one file per
# test such as TestNNCPU.test_x or CPython313-test_set-TestSet.test_x, when it
# runs under dynamo (the dynamo_wrapped config), inductor or compiled autograd.
FAILURE_LIST_RE = re.compile(
    r"^test/(dynamo|inductor|compiled_autograd)_(?:expected_failures|skips)/(?:.*-)?(\w+)\.[^./]+$"
)
# instantiate_device_type_tests names TestFooCPU after TestFoo.
DEVICE_SUFFIX_RE = re.compile(r"(CPU|CUDA|MPS|XPU|HPU|META|LAZY|PRIVATEUSE1)$")

# Changes to these paths alter how every test is collected or run, so no
# subset is safe whatever the model says.
RUN_ALL_PATTERNS = [
    r"^test/run_test\.py$",
    r"^test/conftest\.py$",
    r"^pytest\.ini$",
    r"^torch/testing/_internal/common_utils\.py$",
    r"^torch/testing/_internal/common_device_type\.py$",
    r"^tools/testing/",
    r"^test/pytest_shard_custom\.py$",
    r"^tools/stats/(import_test_stats|upload_metrics)\.py$",
    r"^\.ci/pytorch/(test|multigpu-test|win-test|macos-test|macos-common|common|common-build|common_utils|torch_trace)\.sh$",
    r"^\.ci/pytorch/win-test-helpers/(setup_pytorch_env\.bat|test_python_shard\.bat|installation-helpers/activate_miniconda3\.bat)$",
    r"^\.github/workflows/_(linux|win|mac|rocm|xpu|s390x|tpu)-test(-no-dind)?\.yml$",
    # setup-linux is left out: the test jobs use it from main, not from the PR.
    r"^\.github/actions/(setup-rocm|setup-win|setup-xpu|filter-test-configs|download-td-artifacts)/",
    r"^\.github/scripts/filter_test_configs\.py$",
    # The pytest plugin every test job loads.
    r"^torch/testing/_internal/torchci/",
    # Installed by .ci/pytorch/common_utils.sh for the test jobs.
    r"^\.github/ci_commit_pins/(audio|vision)\.txt$",
    # The CI test images: a new package version or toolchain can break any test.
    r"^\.ci/docker/(build\.sh|requirements-ci\.txt|common/|ci_commit_pins/|ubuntu)",
]


def normalize_test_name(name: str, candidates: set[str]) -> str | None:
    """Map a model-written name such as 'test/dynamo/test_misc.py [3m]' onto a
    candidate, and a file that a driver runs, such as jit/test_freezing, onto the driver."""
    n = re.sub(r"\s*\[[^\]]*\]$", "", name.strip())
    n = n.split("::")[0].removeprefix("./").removeprefix("test/").removesuffix(".py")
    if n in candidates:
        return n
    n = n.replace(".", "/")
    if n in candidates:
        return n
    path = f"test/{n}.py"
    drivers = (t for t, regexes in DRIVER_MAPPINGS.items() if t in candidates and any(re.search(r, path) for r in regexes))  # fmt: skip
    return next(drivers, None)


def edited_tests(changed_files: list[str], candidates: set[str]) -> set[str]:
    """The candidates the change edits, and those that run an edited file, such as test_jit for test/jit/*."""
    edited = {
        n
        for f in changed_files
        if f.startswith("test/") and (n := normalize_test_name(f, candidates))
    }
    for test, regexes in DRIVER_MAPPINGS.items():
        if test in candidates and any(re.search(r, f) for r in regexes for f in changed_files):  # fmt: skip
            edited.add(test)
    return edited


def importers(
    sources: list[str], candidates: set[str], grep: Callable[[str], list[str]]
) -> set[str] | None:
    """The candidates among `sources` (paths under test/), the candidates that
    import one of them directly or through other modules under test/, and their
    drivers: test/jit/test_freezing.py selects test_jit, which runs it, and
    test_jit_legacy, which does `from test_jit import *`. `grep` lists the paths
    under test/ that match a `git grep -E` pattern at the change's head. None if
    the chain is longer than MAX_IMPORT_HOPS."""
    found: set[str] = set()
    seen: set[str] = set()
    frontier = {f for f in sources if f.startswith("test/") and f.endswith(".py")}
    for hop in range(MAX_IMPORT_HOPS + 1):
        tests = edited_tests(sorted(frontier), candidates)
        found |= tests
        frontier = (frontier | {f"test/{t}.py" for t in tests}) - seen
        if not frontier:
            return found
        if hop == MAX_IMPORT_HOPS:
            return None
        seen |= frontier
        mods = [f[5:-3] for f in frontier]
        names = sorted({n for m in mods for n in (m.replace("/", "."), m.rsplit("/", 1)[-1])})  # fmt: skip
        alt = "|".join(map(re.escape, names))
        pattern = rf"^[[:space:]]*((from|import)[[:space:]]+\.*({alt})|from[[:space:]]+\.+[[:space:]]+import[[:space:]]+({alt}))([^[:alnum:]_]|$)"  # fmt: skip
        frontier = {f for f in grep(pattern) if f.endswith(".py")}
    return None


def failure_list_sources(
    changed_files: list[str], grep: Callable[[str], list[str]] | None
) -> tuple[list[str], set[str]]:
    """The paths under test/ that define the classes named by the edited
    failure-list entries (see FAILURE_LIST_RE), and the lists ("dynamo",
    "inductor" or "compiled_autograd") naming a class that `grep` (see
    importers) does not find."""
    classes: dict[str, set[str]] = {}
    for f in changed_files:
        if m := FAILURE_LIST_RE.match(f):
            classes.setdefault(m[2], set()).add(m[1])
    if grep is None or len(classes) > MAX_FAILURE_LIST_CLASSES:
        return [], set().union(*classes.values())
    sources: list[str] = []
    missing: set[str] = set()
    for c, lists in sorted(classes.items()):
        alt = "|".join(sorted({c, DEVICE_SUFFIX_RE.sub("", c)}))
        found = grep(rf"^[[:space:]]*class[[:space:]]+({alt})([^[:alnum:]_]|$)")
        sources += found
        if not found:
            missing |= lists
    return sources, missing


def unmatched_tests(out: dict, candidates: set[str]) -> list[str]:
    """The `tests` names of a well-formed model answer that match no candidate."""
    return [t["file"] for t in out["tests"] if normalize_test_name(t["file"], candidates) is None]  # fmt: skip


def needs_run_all(changed_files: list[str]) -> bool:
    return any(re.search(p, f) for p in RUN_ALL_PATTERNS for f in changed_files)


def previous_failures() -> set[str]:
    """The test files that already failed on this PR, or none if the files are malformed."""
    try:
        with contextlib.redirect_stdout(sys.stderr):
            failed = get_previous_failures() | read_additional_test_failures_file()
        if not all(isinstance(f, str) for f in failed):
            raise TypeError(f"test files must be strings: {failed!r}")
        return failed
    except (OSError, ValueError, TypeError, AttributeError, RecursionError) as e:
        print(f"Ignoring the previous failures: {e!r}", file=sys.stderr)
        return set()


def parse_model_output(out: object, candidates: set[str]) -> tuple[set[str], set[str]]:
    """The files and sweeps a model answer names; ValueError if it is malformed.

    A `dirs` entry is a directory prefix such as "dynamo"; one that matches no
    candidate is malformed, and so is a non-empty `tests` list none of whose
    names matches a candidate, so misspelled names cannot select nothing."""
    if not isinstance(out, dict):
        raise ValueError("not an object")
    tests, dirs, sweeps = out.get("tests"), out.get("dirs", []), out.get("sweeps", [])
    if not isinstance(tests, list) or not all(isinstance(t, dict) and isinstance(t.get("file"), str) for t in tests):  # fmt: skip
        raise ValueError("tests is not a list of objects with a string file")
    for key, value in (("dirs", dirs), ("sweeps", sweeps)):
        if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
            raise ValueError(f"{key} is not a list of strings")
    if unknown := set(sweeps) - set(SWEEPS):
        raise ValueError(f"unknown sweeps {sorted(unknown)}")
    if not isinstance(out.get("run_all", False), bool):
        raise ValueError("run_all is not a boolean")
    files = {n for t in tests if (n := normalize_test_name(t["file"], candidates))}
    if tests and not files:
        raise ValueError("no tests entry matches a test file")
    for d in dirs:
        prefix = re.sub(r"(/\*\*?)?/*$", "", d.strip()).removeprefix("./").removeprefix("test/")  # fmt: skip
        in_dir = {c for c in candidates if c.startswith(prefix + "/")}
        if not in_dir:
            raise ValueError(f"dirs entry {d!r} matches no test file")
        files |= in_dir
    return files, set(sweeps)


@dataclass
class Selection:
    files: set[str] = field(default_factory=set)
    sweeps: set[str] = field(default_factory=set)
    run_all: bool = False
    reason: str = ""
    # Whether run_test.py should replace TD's split with Selection.split rather
    # than only reporting it. It does so only on PRs with TD enabled.
    enforce: bool = False
    # The test files the model chose from. Any other file, such as a custom
    # run_test.py handler or a C++ test, is always covered. None covers no extra file.
    candidates: set[str] | None = None

    @classmethod
    def from_model_output(
        cls,
        out: object,
        candidates: set[str],
        changed_files: list[str],
        grep: Callable[[str], list[str]] | None = None,
    ) -> Selection:
        """Turn the model's structured output into a Selection.

        A missing or malformed output and a change to test infrastructure
        all fall back to running everything. Test files the change edits and
        their drivers are always selected, and with `grep` so are the test
        files that import them (see importers) and those defining the tests
        that an edited failure list names (see failure_list_sources)."""
        if out is None:
            return cls(run_all=True, reason="no model output; running everything")
        if needs_run_all(changed_files):
            return cls(run_all=True, reason="change touches test infrastructure")
        try:
            files, sweeps = parse_model_output(out, candidates)
        except ValueError as e:
            return cls(run_all=True, reason=f"malformed model output ({e}); running everything")  # fmt: skip
        if not isinstance(out, dict):
            raise AssertionError("parse_model_output accepts only objects")
        sources, missing = failure_list_sources(changed_files, grep)
        if "inductor" in missing:
            return cls(run_all=True, reason="an edited inductor failure list names an unknown test class")  # fmt: skip
        # dynamo_wrapped also runs with compiled autograd on (see common_utils).
        if missing & {"dynamo", "compiled_autograd"}:
            sweeps.add("dynamo_wrapped")
        files |= edited_tests(changed_files, candidates)
        if grep is not None:
            found = importers(changed_files + sources, candidates, grep)
            if found is None:
                return cls(run_all=True, reason="test modules import the change through too many hops")  # fmt: skip
            files |= found
        run_all, reason = out.get("run_all", False), str(out.get("reason", ""))
        return cls(files, sweeps, run_all, reason, candidates=set(candidates))

    def covers(self, test_file: str, test_config: str) -> bool:
        if self.candidates is not None and test_file not in self.candidates:
            return True
        return self.run_all or test_config in self.sweeps or test_file in self.files

    def split(
        self,
        ranked: list[TestRun],
        score: Callable[[TestRun], float],
        file_time: dict[str, float],
        test_config: str,
        *,
        previously_failed: Collection[str] = (),
    ) -> Split:
        """Split TD's ranked TestRuns into the covered files, the
        `previously_failed` files (those that already failed on the PR), which
        always run whatever their size or TD score, and a hedge. The previous
        failures are charged first to a budget of LLM_TD_HEDGE (20%) of the
        config's test time; what is left (at least 0) is spent greedily in
        rank order on the files with a positively scored TestRun. One positive
        class-level run hedges its whole file at the file's full time. Files
        without a time are never hedged. Partial TestRuns of one file share
        its time, so a file is charged once and lands in one batch."""
        files = dict.fromkeys(x.test_file for x in ranked)
        covered = {f for f in files if self.covers(f, test_config)}
        prev = {f for f in files if f in previously_failed} - covered
        budget = LLM_TD_HEDGE * sum(file_time.get(f, 0) for f in files)
        left = max(0.0, budget - sum(file_time.get(f, 0) for f in prev))
        hedged: set[str] = set()
        chosen = covered | prev
        for x in ranked:
            f, t = x.test_file, file_time.get(x.test_file, 0)
            if f not in chosen and score(x) > 0 and t and t <= left:
                left -= t
                hedged.add(f)
                chosen.add(f)
        include = [x for x in ranked if x.test_file in chosen]
        exclude = [x for x in ranked if x.test_file not in chosen]
        return Split(include, exclude, covered, prev, hedged, budget)

    def to_json(self) -> dict:
        return {
            "files": sorted(self.files),
            "sweeps": sorted(self.sweeps),
            "run_all": self.run_all,
            "reason": self.reason,
            "enforce": self.enforce,
            "candidates": None if self.candidates is None else sorted(self.candidates),
        }

    @classmethod
    def from_json(cls, d: dict) -> Selection:
        files, sweeps = d["files"], d["sweeps"]
        run_all, enforce = d["run_all"], d.get("enforce", False)
        candidates = d.get("candidates")
        if not all(
            isinstance(v, list) and all(isinstance(x, str) for x in v)
            for v in (files, sweeps) + (() if candidates is None else (candidates,))
        ):
            raise TypeError("files, sweeps and candidates must be lists of strings")
        if not isinstance(run_all, bool) or not isinstance(enforce, bool):
            raise TypeError("run_all and enforce must be booleans")
        return cls(
            set(files),
            set(sweeps),
            run_all,
            d.get("reason", ""),
            enforce,
            None if candidates is None else set(candidates),
        )

    @classmethod
    def load(cls, path: Path) -> Selection | None:
        if not path.exists():
            return None
        try:
            with open(path) as f:
                return cls.from_json(json.load(f))
        except (OSError, ValueError, KeyError, TypeError, RecursionError) as e:
            print(
                f"::warning::Ignoring the LLM selection in {path}: {e!r}",
                file=sys.stderr,
            )
            return None


@dataclass
class Split:
    """Selection.split's result: the TestRuns to run and to skip, and which
    files run because they are covered, already failed on the PR or are hedged
    (disjoint sets), and the hedge budget in seconds."""

    include: list[TestRun]
    exclude: list[TestRun]
    covered: set[str]
    previously_failed: set[str]
    hedged: set[str]
    budget: float


@dataclass
class AppliedSelection:
    """A Selection applied to TD's split of one CI config in run_test.py.

    include and exclude are the split to run: the selection's when enforced,
    TD's otherwise."""

    selection: Selection
    test_config: str
    td_enabled: bool
    enforced: bool
    include: list[TestRun]
    exclude: list[TestRun]
    td_include: set[str]
    split: Split
    selection_metric: dict[str, object]

    def failure_metric(self, test: TestRun) -> dict[str, object]:
        f = test.test_file
        return {
            "failure": str(test),
            "covered": self.selection.covers(f, self.test_config),
            "enforced": self.enforced,
            "td_enabled": self.td_enabled,
            "in_td_include": f in self.td_include,
            "hedged": f in self.split.hedged,
            "previously_failed": f in self.split.previously_failed,
        }


def apply_selection(
    selection: Selection,
    include: list[TestRun],
    exclude: list[TestRun],
    score: Callable[[TestRun], float],
    test_times: dict[str, float],
    *,
    test_config: str,
    td_enabled: bool,
) -> AppliedSelection:
    """Apply the selection to TD's (include, exclude) split.

    The selection replaces TD's split only on a PR with TD enabled and the
    selection's enforce set, and only if some of the files have a test time.
    run_test.py falls back to the default job's times for a config without its
    own, so in practice that fails only without test-times.json."""
    ranked = include + exclude
    td_include = {x.test_file for x in include}
    # Partial TestRuns of one file share its time, so account per file.
    files = list(dict.fromkeys(x.test_file for x in ranked))
    file_time = {f: test_times.get(f, 0) for f in files}
    time_total = sum(file_time.values())
    enforced = td_enabled and selection.enforce and get_pr_number() is not None
    if enforced and not time_total:
        enforced = False
        print("LLM selection: no test times, using TD", file=sys.stderr)
    split = selection.split(
        ranked, score, file_time, test_config, previously_failed=previous_failures()
    )
    if enforced:
        include, exclude = split.include, split.exclude
        print(
            f"LLM selection: running {len({x.test_file for x in include})} of {len(files)} test files ({len(split.covered)} selected; {len(split.previously_failed)} previously failed and {len(split.hedged)} hedged within {split.budget:.0f}s)",
            file=sys.stderr,
        )
    metric: dict[str, object] = {
        "enforced": enforced,
        "td_enabled": td_enabled,
        "run_all": selection.run_all,
        "num_tests": len(files),
        "num_covered": len(split.covered),
        "num_td_include": len(td_include),
        "num_hedged": len(split.hedged),
        "num_previously_failed": len(split.previously_failed),
        "time_total": time_total,
        "time_covered": sum(file_time[f] for f in split.covered),
        "time_hedged": sum(file_time[f] for f in split.hedged),
        "time_previously_failed": sum(file_time[f] for f in split.previously_failed),
    }
    return AppliedSelection(
        selection,
        test_config,
        td_enabled,
        enforced,
        include,
        exclude,
        td_include,
        split,
        metric,
    )
