from typing import Any, Dict, List
from unittest import main, mock, TestCase

from fetch_latest_green_commit import isGreen, WorkflowCheck

workflowNames = [
    "pull",
    "trunk",
    "Lint",
    "linux-binary-libtorch-pre-cxx11",
    "android-tests",
    "windows-binary-wheel",
    "periodic",
    "docker-release-builds",
    "nightly",
    "pr-labels",
    "Close stale pull requests",
    "Update S3 HTML indices for download.pytorch.org",
    "Create Release",
]


def set_workflow_job_status(
    workflow: List[Dict[str, Any]], name: str, status: str
) -> List[Dict[str, Any]]:
    for check in workflow:
        if check["workflowName"] == name:
            check["conclusion"] = status
    return workflow


class TestChecks:
    def make_test_checks(self) -> List[Dict[str, Any]]:
        workflow_checks = []
        for i in range(len(workflowNames)):
            workflow_checks.append(
                WorkflowCheck(
                    workflowName=workflowNames[i],
                    name="test/job",
                    jobName="job",
                    conclusion="success",
                )._asdict()
            )
        return workflow_checks


class TestPrintCommits(TestCase):
    @mock.patch(
        "fetch_latest_green_commit.get_commit_results",
        return_value=TestChecks().make_test_checks(),
    )
    def test_all_successful(self, mock_get_commit_results: Any) -> None:
        "Test with workflows are successful"
        workflow_checks = mock_get_commit_results()
        self.assertTrue(isGreen("sha", workflow_checks)[0])

    @mock.patch(
        "fetch_latest_green_commit.get_commit_results",
        return_value=TestChecks().make_test_checks(),
    )
    def test_necessary_successful(self, mock_get_commit_results: Any) -> None:
        "Test with necessary workflows are successful"
        workflow_checks = mock_get_commit_results()
        workflow_checks = set_workflow_job_status(
            workflow_checks, workflowNames[8], "failed"
        )
        workflow_checks = set_workflow_job_status(
            workflow_checks, workflowNames[9], "failed"
        )
        workflow_checks = set_workflow_job_status(
            workflow_checks, workflowNames[10], "failed"
        )
        workflow_checks = set_workflow_job_status(
            workflow_checks, workflowNames[11], "failed"
        )
        workflow_checks = set_workflow_job_status(
            workflow_checks, workflowNames[12], "failed"
        )
        self.assertTrue(isGreen("sha", workflow_checks)[0])

    @mock.patch(
        "fetch_latest_green_commit.get_commit_results",
        return_value=TestChecks().make_test_checks(),
    )
    def test_necessary_skipped(self, mock_get_commit_results: Any) -> None:
        "Test with necessary job (ex: pull) skipped"
        workflow_checks = mock_get_commit_results()
        workflow_checks = set_workflow_job_status(workflow_checks, "pull", "skipped")
        result = isGreen("sha", workflow_checks)
        self.assertTrue(result[0])

    @mock.patch(
        "fetch_latest_green_commit.get_commit_results",
        return_value=TestChecks().make_test_checks(),
    )
    def test_skippable_skipped(self, mock_get_commit_results: Any) -> None:
        "Test with skippable jobs (periodic and docker-release-builds skipped"
        workflow_checks = mock_get_commit_results()
        workflow_checks = set_workflow_job_status(
            workflow_checks, "periodic", "skipped"
        )
        workflow_checks = set_workflow_job_status(
            workflow_checks, "docker-release-builds", "skipped"
        )
        self.assertTrue(isGreen("sha", workflow_checks))

    @mock.patch(
        "fetch_latest_green_commit.get_commit_results",
        return_value=TestChecks().make_test_checks(),
    )
    def test_necessary_failed(self, mock_get_commit_results: Any) -> None:
        "Test with necessary job (ex: Lint) failed"
        workflow_checks = mock_get_commit_results()
        workflow_checks = set_workflow_job_status(workflow_checks, "Lint", "failed")
        result = isGreen("sha", workflow_checks)
        self.assertFalse(result[0])
        self.assertEqual(result[1], "Lint checks were not successful")

    @mock.patch(
        "fetch_latest_green_commit.get_commit_results",
        return_value=TestChecks().make_test_checks(),
    )
    def test_skippable_failed(self, mock_get_commit_results: Any) -> None:
        "Test with failing skippable jobs (ex: docker-release-builds) should pass"
        workflow_checks = mock_get_commit_results()
        workflow_checks = set_workflow_job_status(
            workflow_checks, "periodic", "skipped"
        )
        workflow_checks = set_workflow_job_status(
            workflow_checks, "docker-release-builds", "failed"
        )
        result = isGreen("sha", workflow_checks)
        self.assertTrue(result[0])

    @mock.patch("fetch_latest_green_commit.get_commit_results", return_value={})
    def test_no_workflows(self, mock_get_commit_results: Any) -> None:
        "Test with missing workflows"
        workflow_checks = mock_get_commit_results()
        result = isGreen("sha", workflow_checks)
        self.assertFalse(result[0])
        self.assertEqual(
            result[1],
            "missing required workflows: pull, trunk, lint, linux-binary",
        )


if __name__ == "__main__":
    main()
