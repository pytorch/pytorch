"""The test selection an LLM makes for one change, and how CI applies it.

A selection names test files (paths under test/ without .py), whole test
directories, and wrapper sweeps (configs such as dynamo_wrapped that re-run
ordinary test files in a special mode). A test file runs in a given CI config
when the selection names the file, names that config as a sweep, or is run_all.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from tools.testing.target_determination.heuristics.edited_by_pr import (
    ADDITIONAL_MAPPINGS,
)


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
    from collections.abc import Callable
    from pathlib import Path


SWEEPS = ("dynamo_wrapped", "crossref")
LLM_TD_SELECTION_FILE = "llm_td_selection.json"
# When enforcing, run_test.py also runs TD's top-ranked files up to this share of
# the config's test time (all shards together): TD and the model miss largely different failures.
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
    r"^tools/stats/import_test_stats\.py$",
    r"^\.ci/pytorch/(test|win-test|macos-test|common|common-build|common_utils|torch_trace)\.sh$",
    r"^\.github/workflows/_(linux|win|mac|rocm|xpu)-test(-no-dind)?\.yml$",
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


def needs_run_all(changed_files: list[str]) -> bool:
    return any(re.search(p, f) for p in RUN_ALL_PATTERNS for f in changed_files)


def parse_model_output(out: object, candidates: set[str]) -> tuple[set[str], set[str]]:
    """The files and sweeps a model answer names; ValueError if it is malformed.

    A `dirs` entry is a directory prefix such as "dynamo"; one that matches no
    candidate is malformed, so a misspelled directory cannot select nothing."""
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
    # Whether run_test.py should skip the files the selection does not cover,
    # rather than only reporting what it would have skipped.
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
        return cls(
            set(d["files"]),
            set(d["sweeps"]),
            d["run_all"],
            d.get("reason", ""),
            d.get("enforce", False),
        )

    @classmethod
    def load(cls, path: Path) -> Selection | None:
        try:
            with open(path) as f:
                return cls.from_json(json.load(f))
        except (OSError, ValueError, KeyError, TypeError):
            return None
