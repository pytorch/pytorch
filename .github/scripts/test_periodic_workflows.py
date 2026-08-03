#!/usr/bin/env python3
"""Checks that periodic CI workflows enable tests marked with @periodic.

torch.testing._internal.common_utils.periodic skips tests unless
PYTORCH_TEST_WITH_PERIODIC is set, so every periodic-cadence workflow must
pass enable-periodic-tests: true to its test jobs, and every reusable test
workflow they use must forward the flag as PYTORCH_TEST_WITH_PERIODIC.
Behavioral tests for the decorator live in test/test_testing.py.
"""

import os
import re
from typing import Any
from unittest import main, TestCase

import yaml


WORKFLOWS_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "workflows"
)
# unstable-periodic.yml hosts experimental/flaky jobs parked from any
# workflow, so the periodic-tests invariant does not apply to it.
EXEMPT_WORKFLOWS = {"unstable-periodic.yml"}


def load_workflow(name: str) -> dict[str, Any]:
    with open(os.path.join(WORKFLOWS_DIR, name)) as f:
        return yaml.safe_load(f)


def periodic_workflows() -> list[tuple[str, dict[str, Any]]]:
    names = sorted(
        name
        for name in os.listdir(WORKFLOWS_DIR)
        if "periodic" in name
        and name.endswith((".yml", ".yaml"))
        and name not in EXEMPT_WORKFLOWS
    )
    return [(name, load_workflow(name)) for name in names]


def reusable_test_jobs(workflow: dict[str, Any]) -> dict[str, Any]:
    """Jobs delegating to a reusable test workflow (_*-test*.yml)."""
    jobs = {}
    for job_name, job in workflow.get("jobs", {}).items():
        base = os.path.basename(job.get("uses", ""))
        if base.startswith("_") and "-test" in base:
            jobs[job_name] = job
    return jobs


class TestPeriodicWorkflows(TestCase):
    def test_periodic_workflows_enable_periodic_tests(self) -> None:
        workflows = periodic_workflows()
        self.assertIn("periodic.yml", [name for name, _ in workflows])
        for workflow_name, workflow in workflows:
            test_jobs = reusable_test_jobs(workflow)
            self.assertTrue(
                test_jobs,
                msg=f"no test jobs found in {workflow_name}; if intentional, exempt it in this test",
            )
            for job_name, job in test_jobs.items():
                self.assertIs(
                    job.get("with", {}).get("enable-periodic-tests"),
                    True,
                    msg=f"{workflow_name}:{job_name} must pass enable-periodic-tests: true to {job['uses']}",
                )

    def test_test_workflows_forward_periodic_mode(self) -> None:
        used = sorted(
            {
                os.path.basename(job["uses"])
                for _, workflow in periodic_workflows()
                for job in reusable_test_jobs(workflow).values()
            }
        )
        self.assertIn("_linux-test.yml", used)
        for reusable_name in used:
            workflow = load_workflow(reusable_name)
            test_steps = [
                step
                for job in workflow["jobs"].values()
                for step in job.get("steps", [])
                if step.get("name") == "Test"
            ]
            self.assertTrue(test_steps, msg=f"no step named Test in {reusable_name}")
            for step in test_steps:
                self.assertEqual(
                    step.get("env", {}).get("PYTORCH_TEST_WITH_PERIODIC"),
                    "${{ inputs.enable-periodic-tests && '1' || '0' }}",
                    msg=f"{reusable_name} Test step must derive PYTORCH_TEST_WITH_PERIODIC from enable-periodic-tests",
                )
                run = step.get("run", "")
                if re.search(r"-e PYTORCH_TEST_\w+", run):
                    self.assertIn(
                        "-e PYTORCH_TEST_WITH_PERIODIC",
                        run,
                        msg=f"{reusable_name} forwards other PYTORCH_TEST_* vars via -e but not PYTORCH_TEST_WITH_PERIODIC",
                    )


if __name__ == "__main__":
    main()
