from unittest import TestCase, main, mock
from typing import Any, List, Dict
from print_latest_commits import isGreen, WorkflowCheck

class TestChecks:
    def make_test_checks(self) -> List[Dict[str, Any]]:
        workflow_checks = []
        for i in range(20):
            workflow_checks.append(WorkflowCheck(
                workflowName="test",
                name="test/job",
                jobName="job",
                conclusion="success",
            )._asdict())
        return workflow_checks

class TestPrintCommits(TestCase):
    @mock.patch('print_latest_commits.get_commit_results', return_value=TestChecks().make_test_checks())
    def test_match_rules(self, mock_get_commit_results: Any) -> None:
        "Test that passes all conditions for promote-able"
        workflow_checks = mock_get_commit_results()
        self.assertTrue(isGreen(workflow_checks))

    @mock.patch('print_latest_commits.get_commit_results', return_value=TestChecks().make_test_checks())
    def test_jobs_failing(self, mock_get_commit_results: Any) -> None:
        "Test with one job failing, no pending jobs, at least 20 jobs run"
        workflow_checks = mock_get_commit_results()
        workflow_checks[0]['conclusion'] = 'failed'
        self.assertFalse(isGreen(workflow_checks))

    @mock.patch('print_latest_commits.get_commit_results', return_value=TestChecks().make_test_checks())
    def test_pending_jobs(self, mock_get_commit_results: Any) -> None:
        "Test with pending jobs, all jobs passing, at least 20 jobs run"
        workflow_checks = mock_get_commit_results()
        workflow_checks[0]['conclusion'] = 'pending'
        self.assertFalse(isGreen(workflow_checks))

    @mock.patch('print_latest_commits.get_commit_results', return_value=TestChecks().make_test_checks())
    def test_jobs_not_run(self, mock_get_commit_results: Any) -> None:
        "Test with all jobs passing, no jobs pending, less than 20 jobs run"
        workflow_checks = mock_get_commit_results()
        workflow_checks.pop(0)
        self.assertFalse(isGreen(workflow_checks))

    # this may need to change, depending on the necessary specs for isGreen

if __name__ == "__main__":
    main()
