from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from tools.linter.adapters.workflow_duplicate_jobs_linter import (
    find_duplicates,
    lint,
    load_workflows,
    MEM_LEAK_CHECK_CRON,
    WORKFLOWS_DIR,
)


ON_MAIN = {"push": {"branches": ["main"]}}
ON_SCHEDULE = {"schedule": [{"cron": "0 0 * * *"}]}
ON_MEM_LEAK_CHECK = {"schedule": [{"cron": MEM_LEAK_CHECK_CRON}]}
ON_PR = {"pull_request": None}
ON_CALL = {"workflow_call": None}
RUNNER = "${{ needs.get-label-type.outputs.label-type }}l-x86iavx512-8-64"


def workflow(on: dict[str, Any], jobs: dict[str, Any]) -> dict[Any, Any]:
    return {True: on, "jobs": jobs}


def build_and_test(
    env: str, config: str, runner: str = RUNNER, with_test: bool = True
) -> dict[str, Any]:
    entry = f'{{ config: "{config}", shard: 1, num_shards: 1, runner: "{runner}" }},'
    inputs = {"build-environment": env, "test-matrix": f"{{ include: [\n{entry}\n]}}"}
    jobs: dict[str, Any] = {"build": {"uses": "./_linux-build.yml", "with": inputs}}
    if with_test:
        test_matrix = "${{ needs.build.outputs.test-matrix }}"
        jobs["test"] = {
            "uses": "./_linux-test.yml",
            "with": {"test-matrix": test_matrix},
        }
    return jobs


def duplicated(workflows: dict[str, dict[Any, Any]]) -> set[tuple[str, str]]:
    found = find_duplicates({Path(f"{name}.yml"): w for name, w in workflows.items()})
    return {(key.build_environment, key.config) for key in found}


class TestWorkflowDuplicateJobsLinter(unittest.TestCase):
    def test_same_config_in_called_workflow(self) -> None:
        dup = ("linux-py3.11", "dynamo_wrapped")
        workflows = {
            "pull": workflow(ON_MAIN, build_and_test(*dup)),
            "trunk": workflow(ON_MAIN, {"dynamo": {"uses": "./dynamo.yml"}}),
            "dynamo": workflow(ON_CALL, build_and_test(*dup)),
        }
        self.assertEqual(duplicated(workflows), {dup})

    def test_called_workflow_runs_from_two_workflows(self) -> None:
        workflows = {
            "pull": workflow(ON_MAIN, {"dynamo": {"uses": "./dynamo.yml"}}),
            "trunk": workflow(ON_MAIN, {"dynamo": {"uses": "./dynamo.yml"}}),
            "dynamo": workflow(ON_CALL, build_and_test("linux-py3.11", "default")),
        }
        self.assertEqual(duplicated(workflows), {("linux-py3.11", "default")})

    def test_matrix_expands_versions(self) -> None:
        matrix_jobs = build_and_test("linux-py${{ matrix.version }}", "dynamo_wrapped")
        matrix_jobs["build"]["strategy"] = {"matrix": {"version": ["3.12", "3.13"]}}
        workflows = {
            "pull": workflow(ON_MAIN, build_and_test("linux-py3.13", "dynamo_wrapped")),
            "trunk": workflow(ON_MAIN, matrix_jobs),
        }
        self.assertEqual(duplicated(workflows), {("linux-py3.13", "dynamo_wrapped")})

    def test_schedule_counts_as_testing_main(self) -> None:
        workflows = {
            "trunk": workflow(ON_MAIN, build_and_test("linux-cuda", "default")),
            "periodic": workflow(ON_SCHEDULE, build_and_test("linux-cuda", "default")),
        }
        self.assertEqual(duplicated(workflows), {("linux-cuda", "default")})

    def test_not_duplicates(self) -> None:
        cases = {
            "different runner": build_and_test("linux-arm", "default", runner="l-arm"),
            "no test job": build_and_test("linux-arm", "default", with_test=False),
        }
        for name, jobs in cases.items():
            with self.subTest(name):
                workflows = {
                    "pull": workflow(ON_MAIN, build_and_test("linux-arm", "default")),
                    "trunk": workflow(ON_MAIN, jobs),
                }
                self.assertEqual(duplicated(workflows), set())

    def test_workflows_not_testing_main_are_ignored(self) -> None:
        for name, on in {
            "pull request": ON_PR,
            "mem leak check": ON_MEM_LEAK_CHECK,
        }.items():
            with self.subTest(name):
                workflows = {
                    "pull": workflow(
                        ON_MAIN, build_and_test("linux-py3.11", "default")
                    ),
                    "other": workflow(on, build_and_test("linux-py3.11", "default")),
                }
                self.assertEqual(duplicated(workflows), set())

    def test_lint_takes_absolute_paths(self) -> None:
        pull, trunk = WORKFLOWS_DIR / "pull.yml", WORKFLOWS_DIR / "trunk.yml"
        workflows = {
            pull: workflow(ON_MAIN, build_and_test("linux-py3.11", "default")),
            trunk: workflow(ON_MAIN, build_and_test("linux-py3.11", "default")),
        }
        path = str(REPO_ROOT / pull)
        self.assertEqual([m.path for m in lint(workflows, [path])], [path])

    def test_repo_has_no_duplicates(self) -> None:
        self.assertEqual(find_duplicates(load_workflows()), {})


if __name__ == "__main__":
    unittest.main()
