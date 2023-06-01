from typing import Any
import unittest
from unittest import mock
import boto3

from tools.stats.upload_stats_lib import emit_metric

# default values
REPO = "some/repo"
WORKFLOW_RUN_NUMBER = 123
WORKFLOW_RUN_ATTEMPT = 3

class TestUploadStats(unittest.TestCase):
    @mock.patch('boto3.resource')
    def test_emit_metric(self, mock_resource) -> None:
        metric = {"some_number" : 123}

        expected_emit = {
            'dynamo_key': f"{REPO}/metric_name/{WORKFLOW_RUN_NUMBER}/{WORKFLOW_RUN_ATTEMPT}",
            'metric_name': 'metric_name',
            'repo': REPO,
            'workflow_run_number': WORKFLOW_RUN_NUMBER,
            'workflow_run_attempt': WORKFLOW_RUN_ATTEMPT,
            **metric
        }

        emit_metric("metric_name", metric, REPO, WORKFLOW_RUN_NUMBER, WORKFLOW_RUN_ATTEMPT)

        mock_table = mock_resource.return_value.Table.return_value
        mock_table.put_item.assert_called_once_with(Item=expected_emit)

    @mock.patch('boto3.resource')
    def test_blocks_emission_if_reserved_keyword_used(self, mock_resource) -> None:
        metric = {"repo": "awesome/repo"}

        with self.assertRaises(ValueError):
            emit_metric("metric_name", metric, REPO, WORKFLOW_RUN_NUMBER, WORKFLOW_RUN_ATTEMPT)


if __name__ == "__main__":
    unittest.main()
