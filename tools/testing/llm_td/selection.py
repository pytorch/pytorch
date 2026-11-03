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


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from tools.testing.test_run import TestRun


SWEEPS = ("dynamo_wrapped", "crossref")
LLM_TD_SELECTION_FILE = "llm_td_selection.json"
# When enforcing, run_test.py also runs TD's top-ranked files up to this share of
# the config's test time (all shards together): TD and the model miss largely different failures.
LLM_TD_HEDGE = 0.05

# Changes to these paths alter how every test is collected or run, so no
# subset is safe whatever the model says.
RUN_ALL_PATTERNS = [
    r"^test/run_test\.py$",
    r"^test/conftest\.py$",
    r"^pytest\.ini$",
    r"^torch/testing/_internal/common_utils\.py$",
    r"^torch/testing/_internal/common_device_type\.py$",
    r"^tools/testing/",
    r"^\.ci/pytorch/(test|win-test|macos-test|common_utils)\.sh$",
    r"^\.github/workflows/_(linux|win|mac)-test\.yml$",
    r"^\.github/actions/(linux-test|setup-linux|download-td-artifacts)/",
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
    return {
        n
        for f in changed_files
        if f.startswith("test/") and (n := normalize_test_name(f, candidates))
    }


def needs_run_all(changed_files: list[str]) -> bool:
    return any(re.search(p, f) for p in RUN_ALL_PATTERNS for f in changed_files)


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
        cls, out: dict | None, candidates: set[str], changed_files: list[str]
    ) -> Selection:
        """Turn the model's structured output into a Selection.

        A missing output (the model call failed) and a change to test
        infrastructure both fall back to running everything. Test files the
        change edits are always selected."""
        if out is None:
            return cls(run_all=True, reason="no model output; running everything")
        if needs_run_all(changed_files):
            return cls(run_all=True, reason="change touches test infrastructure")
        files = {
            n
            for t in out.get("tests") or []
            if isinstance(t, dict)
            and isinstance(f := t.get("file"), str)
            and (n := normalize_test_name(f, candidates))
        }
        for d in out.get("dirs") or []:
            prefix = d.strip().removeprefix("test/").strip("/") + "/"
            files |= {c for c in candidates if c.startswith(prefix)}
        files |= edited_tests(changed_files, candidates)
        sweeps = {s for s in out.get("sweeps") or [] if s in SWEEPS}
        return cls(files, sweeps, bool(out.get("run_all")), out.get("reason", ""))

    def covers(self, test_file: str, test_config: str) -> bool:
        return self.run_all or test_config in self.sweeps or test_file in self.files

    def split(
        self,
        ranked: list[TestRun],
        score: Callable[[TestRun], float],
        file_time: dict[str, float],
        test_config: str,
    ) -> tuple[list[TestRun], list[TestRun], set[str]]:
        """Split TD's ranked TestRuns into (include, exclude, hedged files).

        Include holds the covered files plus a hedge: TD's positively scored
        files, greedily in rank order, that fit in LLM_TD_HEDGE of the config's
        test time. Files without a time are not hedged. Partial TestRuns of one
        file share its time, so a file is charged once and lands in one batch."""
        files = dict.fromkeys(x.test_file for x in ranked)
        covered = {f for f in files if self.covers(f, test_config)}
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
