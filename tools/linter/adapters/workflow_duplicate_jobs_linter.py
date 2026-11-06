# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pyyaml==6.0.2",
# ]
# ///
"""Flags test configs that run in more than one workflow against main.

Workflows that run on a push to main or on a schedule all test main. A test
config with the same build-environment, config name and runner in two of them,
including in the reusable workflows they call, runs twice for the same commit.
The runner is part of the key so that the same config on different hardware,
like trunk's Graviton3 and pull's Graviton4 aarch64 jobs, is not a duplicate.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS_DIR = Path(".github/workflows")
LINTER_CODE = "WORKFLOWDUP"

# filter_test_configs.py turns this schedule into mem_leak_check and
# rerun_disabled_tests runs, which do not repeat the regular configs.
MEM_LEAK_CHECK_CRON = "29 8 * * *"

# (build-environment, config) pairs allowed to run in more than one workflow.
ALLOWLIST = {
    # periodic.yml repeats trunk's CUDA 13.2 default and distributed configs on a schedule.
    ("linux-jammy-cuda13.2-py3.11-gcc11", "default"),
    ("linux-jammy-cuda13.2-py3.11-gcc11", "distributed"),
}

MATRIX_REF = re.compile(r"\$\{\{\s*matrix\.([\w-]+)\s*\}\}")
# One test-matrix entry per line, e.g.
# { config: "default", shard: 1, num_shards: 5, runner: "${{ ... }}l-x86iavx512-8-64" },
ENTRY = re.compile(r'config:\s*"([^"]+)".*runner:\s*"(?:\$\{\{.*?\}\})?([^"]+)"')


class LintSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"


class LintMessage(NamedTuple):
    path: str | None
    line: int | None
    char: int | None
    code: str
    severity: LintSeverity
    name: str
    original: str | None
    replacement: str | None
    description: str | None


class TestConfig(NamedTuple):
    build_environment: str
    config: str
    runner: str


class Occurrence(NamedTuple):
    # The workflow that triggers the run, and the one that defines the job
    root: Path
    path: Path
    job_id: str


def tests_main(workflow: dict[Any, Any]) -> bool:
    # PyYAML reads the bare `on` key as True
    on = workflow.get(True, workflow.get("on")) or {}
    if not isinstance(on, dict):
        return False
    push = on.get("push") or {}
    crons = {s.get("cron") for s in on.get("schedule") or []}
    return bool(crons - {MEM_LEAK_CHECK_CRON}) or "main" in (push.get("branches") or [])


def matrix_combos(job: dict[str, Any]) -> list[dict[str, Any]]:
    matrix = (job.get("strategy") or {}).get("matrix") or {}
    if not isinstance(matrix, dict):
        return [{}]
    keys = [
        k
        for k, v in matrix.items()
        if isinstance(v, list) and k not in ("include", "exclude")
    ]
    return [dict(zip(keys, v)) for v in itertools.product(*(matrix[k] for k in keys))]


def substitute(text: str, combo: dict[str, Any]) -> str:
    return MATRIX_REF.sub(lambda m: str(combo.get(m[1], m[0])), text)


def test_configs(
    workflows: dict[Path, dict[Any, Any]],
    root: Path,
    path: Path,
    seen: tuple[Path, ...] = (),
) -> list[tuple[TestConfig, Occurrence]]:
    """Test configs that a workflow and the reusable workflows it calls run.

    A build job's test-matrix counts only when a test job in the same workflow
    reads it, since a test-matrix without a test job runs nothing."""
    workflow = workflows.get(path)
    if workflow is None or path in seen:
        return []
    jobs = workflow.get("jobs") or {}
    jobs_text = json.dumps(jobs)
    found = []
    for job_id, job in jobs.items():
        uses = job.get("uses", "")
        called = Path(uses)
        if uses.startswith("./") and not called.name.startswith("_"):
            found += test_configs(workflows, root, called, seen + (path,))
            continue
        inputs = job.get("with") or {}
        test_matrix = inputs.get("test-matrix")
        build_environment = inputs.get("build-environment")
        if not isinstance(test_matrix, str) or not isinstance(build_environment, str):
            continue
        if f"needs.{job_id}.outputs.test-matrix" not in jobs_text:
            continue
        for combo in matrix_combos(job):
            env = substitute(build_environment, combo)
            for line in substitute(test_matrix, combo).splitlines():
                if m := ENTRY.search(line):
                    occurrence = Occurrence(root, path, job_id)
                    found.append((TestConfig(env, m[1], m[2]), occurrence))
    return found


def find_duplicates(
    workflows: dict[Path, dict[Any, Any]],
) -> dict[TestConfig, set[Occurrence]]:
    where = defaultdict(set)
    for path, workflow in workflows.items():
        if path.name.startswith("_") or not tests_main(workflow):
            continue
        for key, occurrence in test_configs(workflows, path, path):
            where[key].add(occurrence)
    return {
        key: occurrences
        for key, occurrences in where.items()
        if len({o.root for o in occurrences}) > 1
        and (key.build_environment, key.config) not in ALLOWLIST
    }


def job_line(path: Path, job_id: str) -> int | None:
    lines = (REPO_ROOT / path).read_text().splitlines()
    start = next((i for i, line in enumerate(lines) if line == "jobs:"), 0)
    for i in range(start, len(lines)):
        if lines[i].strip() == f"{job_id}:":
            return i + 1
    return None


def load_workflows() -> dict[Path, dict[Any, Any]]:
    return {
        path.relative_to(REPO_ROOT): yaml.load(path.read_text(), Loader=yaml.SafeLoader)
        for path in sorted((REPO_ROOT / WORKFLOWS_DIR).glob("*.y*ml"))
    }


def lint(
    workflows: dict[Path, dict[Any, Any]], filenames: list[str]
) -> list[LintMessage]:
    # lintrunner passes absolute paths; workflows are keyed by repo-relative ones
    linted = {Path(f).resolve().relative_to(REPO_ROOT): f for f in filenames}
    messages = []
    for key, occurrences in find_duplicates(workflows).items():
        roots = {o.root for o in occurrences}
        for o in occurrences:
            if o.path not in linted:
                continue
            others = ", ".join(sorted(str(r) for r in roots - {o.root}))
            description = (
                f'{key.build_environment} config "{key.config}" on {key.runner} also runs from '
                f"{others}, so it runs twice for the same commit on main. Remove one, or add "
                "it to ALLOWLIST in tools/linter/adapters/workflow_duplicate_jobs_linter.py "
                "with the reason."
            )
            messages.append(
                LintMessage(
                    path=linted[o.path],
                    line=job_line(o.path, o.job_id),
                    char=None,
                    code=LINTER_CODE,
                    severity=LintSeverity.ERROR,
                    name="duplicate-test-config",
                    original=None,
                    replacement=None,
                    description=description,
                )
            )
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, fromfile_prefix_chars="@")
    parser.add_argument("filenames", nargs="+", help="paths to lint")
    args = parser.parse_args()
    for message in lint(load_workflows(), args.filenames):
        print(json.dumps(message._asdict()), flush=True)


if __name__ == "__main__":
    main()
