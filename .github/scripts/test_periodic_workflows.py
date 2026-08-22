#!/usr/bin/env python3
"""Checks the CI wiring that runs tests marked @periodic.

torch.testing._internal.common_utils.periodic gates tests behind
PYTORCH_TEST_WITH_PERIODIC. They run only in the periodic-strict workflow,
whose test jobs use the "periodic" test config: .ci/pytorch/test.sh maps that
config to PYTORCH_TEST_WITH_PERIODIC=1 plus PYTORCH_TEST_SKIP_NON_PERIODIC=1,
which skips every test not marked @periodic, plus PYTORCH_TEST_WITH_SLOW=1 so
slow gating cannot block a @periodic test. Other periodic-cadence workflows
are unaffected. Behavioral tests for the decorator live in test/test_testing.py.
"""

import os
import re
from typing import Any
from unittest import main, TestCase

import yaml


REPO_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")
WORKFLOWS_DIR = os.path.join(REPO_ROOT, ".github", "workflows")
TEST_SH = os.path.join(REPO_ROOT, ".ci", "pytorch", "test.sh")
STRICT_WORKFLOW = "periodic-strict.yml"


def literal_test_matrices(workflow: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Literal test-matrix include lists by job name (skips ${{ ... }} references)."""
    matrices = {}
    for job_name, job in workflow.get("jobs", {}).items():
        matrix = job.get("with", {}).get("test-matrix", "")
        if not isinstance(matrix, str) or not matrix or matrix.startswith("${{"):
            continue
        matrices[job_name] = yaml.safe_load(matrix)["include"]
    return matrices


class TestPeriodicWorkflows(TestCase):
    def test_periodic_strict_runs_only_the_periodic_config(self) -> None:
        with open(os.path.join(WORKFLOWS_DIR, STRICT_WORKFLOW)) as f:
            workflow = yaml.safe_load(f)
        matrices = literal_test_matrices(workflow)
        self.assertTrue(matrices, msg=f"no test matrices found in {STRICT_WORKFLOW}")
        for job_name, entries in matrices.items():
            for entry in entries:
                self.assertEqual(
                    entry.get("config"),
                    "periodic",
                    msg=f"{STRICT_WORKFLOW}:{job_name} must use only the periodic test config, got {entry}",
                )

    def test_no_other_workflow_runs_the_periodic_config(self) -> None:
        for name in sorted(os.listdir(WORKFLOWS_DIR)):
            if name == STRICT_WORKFLOW or not name.endswith((".yml", ".yaml")):
                continue
            with open(os.path.join(WORKFLOWS_DIR, name)) as f:
                self.assertNotRegex(
                    f.read(),
                    r"config:\s*['\"]periodic['\"]",
                    msg=f"{name} runs the periodic test config; @periodic tests run only in {STRICT_WORKFLOW}",
                )

    def test_test_sh_maps_the_periodic_config_to_periodic_mode(self) -> None:
        with open(TEST_SH) as f:
            text = f.read()
        guard = r"if \[\[ \"\$TEST_CONFIG\" == 'periodic' \]\]; then\n(?P<body>(  .*\n)+)fi\n"
        match = re.search(guard, text)
        self.assertIsNotNone(match, msg="test.sh lacks a TEST_CONFIG periodic block")
        exports = (
            "PYTORCH_TEST_WITH_PERIODIC",
            "PYTORCH_TEST_SKIP_NON_PERIODIC",
            "PYTORCH_TEST_WITH_SLOW",
        )
        for var in exports:
            self.assertIn(
                f"export {var}=1",
                match["body"],
                msg=f"the periodic config in test.sh must export {var}=1",
            )
        # Without a dedicated dispatch branch the periodic config falls through
        # to the default shard branches, which also run C++ suites that ignore
        # PYTORCH_TEST_SKIP_NON_PERIODIC.
        dispatch = 'elif [[ "${TEST_CONFIG}" == periodic ]]; then'
        self.assertIn(dispatch, text, msg="test.sh has no periodic dispatch branch")


if __name__ == "__main__":
    main()
