# For testing specific heuristics
from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT))

from tools.testing.target_determination.do_td_with_job_info import (
    get_job_info_from_workflow_file,
)


sys.path.remove(str(REPO_ROOT))


class TestGetJobInfoFromWorkflowFile(unittest.TestCase):
    def test_parsing(self):
        workflow_file = (
            "pytorch/pytorch/.github/workflows/pull.yml@refs/pull/165793/merge"
        )
        # Just some sanity checks on the output.  Don't check against a fixed
        # value since pull.yml changes often.
        job_info = get_job_info_from_workflow_file(workflow_file)
        self.assertGreater(len(job_info), 5)
        unique_jobs = {job["job_name"] for jobs in job_info for job in jobs}
        unique_configs = {job["config"] for jobs in job_info for job in jobs}
        self.assertGreater(len(unique_jobs), 5)
        self.assertGreater(len(unique_configs), 5)
        self.assertIn("default", unique_configs)
        self.assertIn("crossref", unique_configs)


if __name__ == "__main__":
    unittest.main()
