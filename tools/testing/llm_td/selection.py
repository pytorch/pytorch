"""The test selection an LLM makes for one change, and how CI applies it.

A selection names test files (paths under test/ without .py), whole test
directories, and wrapper sweeps (configs such as dynamo_wrapped that re-run
ordinary test files in a special mode). A test file is covered in a given CI
config when the selection names the file, names that config as a sweep, or is
run_all. When run_test.py enforces the selection, it runs the covered files, the
files that already failed on the PR, and a hedge of TD's top-scored files.
"""

from __future__ import annotations

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


# Directories that discover_tests.py blocklists because a driver runs their files.
DRIVER_MAPPINGS = {
    **ADDITIONAL_MAPPINGS,
    "test_ao_sparsity": [r"^test/ao/"],
    "test_autograd": [r"^test/autograd/"],
    "test_fx": [r"^test/fx/"],
    "test_jit": [r"^test/jit/"],
    "test_package": [r"^test/package/"],
    "test_quantization": [r"^test/quantization/"],
}

if TYPE_CHECKING:
    from collections.abc import Callable, Collection
    from pathlib import Path

    from tools.testing.test_run import TestRun


SWEEPS = ("dynamo_wrapped", "crossref")
LLM_TD_SELECTION_FILE = "llm_td_selection.json"
# When enforcing, run_test.py also runs, greedily in TD's rank order, the
# positively scored files with a known time that fit in this share (20%) of the
# config's test time (all shards together): TD and the model miss largely
# different failures.
LLM_TD_HEDGE = 0.20

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
    r"^\.github/actions/(setup-linux|setup-rocm|setup-win|setup-xpu|filter-test-configs|download-td-artifacts)/",
    r"^\.github/scripts/filter_test_configs\.py$",
    # The CI test images: a new package version or toolchain can break any test.
    r"^\.ci/docker/(build\.sh|requirements-ci\.txt|common/|ci_commit_pins/|ubuntu)",
]


def normalize_test_name(name: str, candidates: set[str]) -> str | None:
    """Map a model-written name such as 'test/dynamo/test_misc.py [3m]' onto a candidate."""
    n = re.sub(r"\s*\[[^\]]*\]$", "", name.strip())
    n = n.split("::")[0].removeprefix("./").removeprefix("test/").removesuffix(".py")
    if n in candidates:
        return n
    n = n.replace(".", "/")
    return n if n in candidates else None


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
    changed_files: list[str], candidates: set[str], grep: Callable[[str], list[str]]
) -> set[str]:
    """Candidates that import a test module the change edits, such as the
    *_dynamic_shapes copies of inductor/test_torchinductor. `grep` lists the
    paths under test/ that match a `git grep -E` pattern at the change's head."""
    mods = [f[5:-3] for f in changed_files if f.startswith("test/") and f.endswith(".py")]  # fmt: skip
    names = sorted({n for m in mods for n in (m.replace("/", "."), m.rsplit("/", 1)[-1])})  # fmt: skip
    if not names:
        return set()
    alt = "|".join(map(re.escape, names))
    pattern = rf"^[[:space:]]*(from|import)[[:space:]]+({alt})([^[:alnum:]_]|$)"
    return {n for f in grep(pattern) if (n := normalize_test_name(f, candidates))}


def unmatched_tests(out: dict, candidates: set[str]) -> list[str]:
    """The `tests` names of a well-formed model answer that match no candidate."""
    return [t["file"] for t in out["tests"] if normalize_test_name(t["file"], candidates) is None]  # fmt: skip


def needs_run_all(changed_files: list[str]) -> bool:
    return any(re.search(p, f) for p in RUN_ALL_PATTERNS for f in changed_files)


def previous_failures() -> set[str]:
    """The test files that already failed on this PR, or none if the files are malformed."""
    try:
        return get_previous_failures() | read_additional_test_failures_file()
    except (OSError, ValueError, TypeError, RecursionError) as e:
        print(f"Ignoring the previous failures: {e!r}")
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
    return files, {s for s in sweeps if s in SWEEPS}


@dataclass
class Selection:
    files: set[str] = field(default_factory=set)
    sweeps: set[str] = field(default_factory=set)
    run_all: bool = False
    reason: str = ""
    # Whether run_test.py should replace TD's split with Selection.split rather
    # than only reporting it. It does so only on PRs with TD enabled.
    enforce: bool = False

    @classmethod
    def from_model_output(
        cls,
        out: dict | None,
        candidates: set[str],
        changed_files: list[str],
        grep: Callable[[str], list[str]] | None = None,
    ) -> Selection:
        """Turn the model's structured output into a Selection.

        A missing or malformed output and a change to test infrastructure
        all fall back to running everything. Test files the
        change edits are always selected, and with `grep` (see importers) so
        are the test files that import them."""
        if out is None:
            return cls(run_all=True, reason="no model output; running everything")
        if needs_run_all(changed_files):
            return cls(run_all=True, reason="change touches test infrastructure")
        try:
            files, sweeps = parse_model_output(out, candidates)
        except ValueError as e:
            return cls(run_all=True, reason=f"malformed model output ({e}); running everything")  # fmt: skip
        files |= edited_tests(changed_files, candidates)
        if grep is not None:
            files |= importers(changed_files, candidates, grep)
        return cls(files, sweeps, out.get("run_all", False), str(out.get("reason", "")))  # fmt: skip

    def covers(self, test_file: str, test_config: str) -> bool:
        return self.run_all or test_config in self.sweeps or test_file in self.files

    def split(
        self,
        ranked: list[TestRun],
        score: Callable[[TestRun], float],
        file_time: dict[str, float],
        test_config: str,
        always: Collection[str] = (),
    ) -> tuple[list[TestRun], list[TestRun], set[str]]:
        """Split TD's ranked TestRuns into (include, exclude, hedged files).

        Include holds the covered files, the `always` files (those that already
        failed on the PR) and a hedge: the files with a positively scored
        TestRun, greedily in rank order, that fit in LLM_TD_HEDGE (20%) of the
        config's test time. One positive class-level run hedges its whole file
        at the file's full time. Files without a time are not hedged. Partial
        TestRuns of one file share its time, so a file is charged once and
        lands in one batch."""
        files = dict.fromkeys(x.test_file for x in ranked)
        covered = {f for f in files if self.covers(f, test_config) or f in always}
        budget = LLM_TD_HEDGE * sum(file_time.get(f, 0) for f in files)
        hedged: set[str] = set()
        for x in ranked:
            f, t = x.test_file, file_time.get(x.test_file, 0)
            if f in covered or f in hedged or score(x) <= 0 or not t or t > budget:
                continue
            budget -= t
            hedged.add(f)
        chosen = covered | hedged
        include = [x for x in ranked if x.test_file in chosen]
        return include, [x for x in ranked if x.test_file not in chosen], hedged

    def to_json(self) -> dict:
        return {
            "files": sorted(self.files),
            "sweeps": sorted(self.sweeps),
            "run_all": self.run_all,
            "reason": self.reason,
            "enforce": self.enforce,
        }

    @classmethod
    def from_json(cls, d: dict) -> Selection:
        files, sweeps = d["files"], d["sweeps"]
        run_all, enforce = d["run_all"], d.get("enforce", False)
        if not all(
            isinstance(v, list) and all(isinstance(x, str) for x in v)
            for v in (files, sweeps)
        ):
            raise TypeError("files and sweeps must be lists of strings")
        if not isinstance(run_all, bool) or not isinstance(enforce, bool):
            raise TypeError("run_all and enforce must be booleans")
        return cls(set(files), set(sweeps), run_all, d.get("reason", ""), enforce)

    @classmethod
    def load(cls, path: Path) -> Selection | None:
        if not path.exists():
            return None
        try:
            with open(path) as f:
                return cls.from_json(json.load(f))
        except (OSError, ValueError, KeyError, TypeError, RecursionError) as e:
            print(f"Ignoring the LLM selection in {path}: {e!r}")
            return None


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
    hedged: set[str]
    previously_failed: set[str]
    selection_metric: dict[str, object]

    def failure_metric(self, test: TestRun) -> dict[str, object]:
        f = test.test_file
        return {
            "failure": str(test),
            "covered": self.selection.covers(f, self.test_config),
            "enforced": self.enforced,
            "td_enabled": self.td_enabled,
            "in_td_include": f in self.td_include,
            "hedged": f in self.hedged,
            "previously_failed": f in self.previously_failed,
        }


def apply_selection(
    selection: Selection,
    include: list[TestRun],
    exclude: list[TestRun],
    score: Callable[[TestRun], float],
    test_times: dict[str, float],
    test_config: str,
    td_enabled: bool,
) -> AppliedSelection:
    """Apply the selection to TD's (include, exclude) split.

    The selection replaces TD's split only on a PR with TD enabled and the
    selection's enforce set, and only if the config has test times to size
    the hedge with."""
    ranked = include + exclude
    td_include = {x.test_file for x in include}
    # Partial TestRuns of one file share its time, so account per file.
    files = list(dict.fromkeys(x.test_file for x in ranked))
    covered = {f for f in files if selection.covers(f, test_config)}
    file_time = {f: test_times.get(f, 0) for f in files}
    time_total = sum(file_time.values())
    prev_failed = previous_failures()
    enforced = td_enabled and selection.enforce and get_pr_number() is not None
    if enforced and not time_total:
        enforced = False
        print("LLM selection: no test times for this config, using TD", file=sys.stderr)
    llm_include, llm_exclude, hedged = selection.split(
        ranked, score, file_time, test_config, prev_failed
    )
    if enforced:
        include, exclude = llm_include, llm_exclude
        num_run = len({x.test_file for x in include})
        budget = LLM_TD_HEDGE * time_total
        print(
            f"LLM selection: running {num_run} of {len(files)} test files ({len(covered)} selected, {len(hedged)} hedged within {budget:.0f}s)",
            file=sys.stderr,
        )
    metric: dict[str, object] = {
        "enforced": enforced,
        "td_enabled": td_enabled,
        "run_all": selection.run_all,
        "num_tests": len(files),
        "num_covered": len(covered),
        "num_td_include": len(td_include),
        "num_hedged": len(hedged),
        "num_previously_failed": len(prev_failed & set(files)),
        "time_total": time_total,
        "time_covered": sum(file_time[f] for f in covered),
        "time_hedged": sum(file_time[f] for f in hedged),
    }
    return AppliedSelection(
        selection,
        test_config,
        td_enabled,
        enforced,
        include,
        exclude,
        td_include,
        hedged,
        prev_failed,
        metric,
    )
