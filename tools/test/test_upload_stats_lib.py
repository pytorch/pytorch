import decimal
import inspect
import unittest
from typing import Any
from unittest import mock

from tools.stats.upload_stats_lib import emit_metric

# default values
REPO = "some/repo"
WORKFLOW = "some-workflow"
JOB = "some-job"
WORKFLOW_RUN_NUMBER = 123
WORKFLOW_RUN_ATTEMPT = 3


class TestUploadStats(unittest.TestCase):
    # Before each test, set the env vars to their default values
    def setUp(self) -> None:
        mock.patch.dict(
            "os.environ",
            {
                "CI": "true",
                "GITHUB_REPOSITORY": REPO,
                "GITHUB_WORKFLOW": WORKFLOW,
                "GITHUB_JOB": JOB,
                "GITHUB_RUN_NUMBER": str(WORKFLOW_RUN_NUMBER),
                "GITHUB_RUN_ATTEMPT": str(WORKFLOW_RUN_ATTEMPT),
            },
        ).start()

    @mock.patch("boto3.Session.resource")
    def test_emit_metric(self, mock_resource: Any) -> None:
        metric = {
            "some_number": 123,
            "float_number": 32.34,
        }

        # Querying for this instead of hard coding it b/c this will change
        # based on whether we run this test directly from python or from
        # pytest
        current_module = inspect.getmodule(inspect.currentframe()).__name__  # type: ignore[union-attr]

        expected_emit = {
            "dynamo_key": f"{REPO}/metric_name/{WORKFLOW}/{JOB}/{WORKFLOW_RUN_NUMBER}/{WORKFLOW_RUN_ATTEMPT}",
            "metric_name": "metric_name",
            "calling_file": "test_upload_stats_lib.py",
            "calling_module": current_module,
            "calling_function": "test_emit_metric",
            "repo": REPO,
            "workflow": WORKFLOW,
            "job": JOB,
            "workflow_run_number": WORKFLOW_RUN_NUMBER,
            "workflow_run_attempt": WORKFLOW_RUN_ATTEMPT,
            "some_number": 123,
            "float_number": decimal.Decimal(str(32.34)),
        }

        emit_metric("metric_name", metric)

        mock_table = mock_resource.return_value.Table.return_value
        mock_table.put_item.assert_called_once_with(Item=expected_emit)

    @mock.patch("boto3.resource")
    def test_blocks_emission_if_reserved_keyword_used(self, mock_resource: Any) -> None:
        metric = {"repo": "awesome/repo"}

        with self.assertRaises(ValueError):
            emit_metric("metric_name", metric)

    @mock.patch("boto3.resource")
    def test_no_metrics_emitted_if_env_var_not_set(self, mock_resource: Any) -> None:
        metric = {"some_number": 123}

        mock.patch.dict(
            "os.environ",
            {
                "GITHUB_REPOSITORY": "",  # unset this env var
            },
        ).start()

        emit_metric("metric_name", metric)

        mock_table = mock_resource.return_value.Table.return_value
        mock_table.put_item.assert_not_called()


if __name__ == "__main__":
    unittest.main()
