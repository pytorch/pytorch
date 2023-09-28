import decimal
import inspect
import unittest
from typing import Any, Dict
from unittest import mock

from tools.stats.upload_metrics import emit_metric

from tools.stats.upload_stats_lib import BATCH_SIZE, upload_to_rockset


# default values
REPO = "some/repo"
BUILD_ENV = "cuda-10.2"
TEST_CONFIG = "test-config"
WORKFLOW = "some-workflow"
JOB = "some-job"
RUN_ID = 56
RUN_NUMBER = 123
RUN_ATTEMPT = 3


class TestUploadStats(unittest.TestCase):
    # Before each test, set the env vars to their default values
    def setUp(self) -> None:
        mock.patch.dict(
            "os.environ",
            {
                "CI": "true",
                "BUILD_ENVIRONMENT": BUILD_ENV,
                "TEST_CONFIG": TEST_CONFIG,
                "GITHUB_REPOSITORY": REPO,
                "GITHUB_WORKFLOW": WORKFLOW,
                "GITHUB_JOB": JOB,
                "GITHUB_RUN_ID": str(RUN_ID),
                "GITHUB_RUN_NUMBER": str(RUN_NUMBER),
                "GITHUB_RUN_ATTEMPT": str(RUN_ATTEMPT),
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

        emit_should_include = {
            "metric_name": "metric_name",
            "calling_file": "test_upload_stats_lib.py",
            "calling_module": current_module,
            "calling_function": "test_emit_metric",
            "repo": REPO,
            "workflow": WORKFLOW,
            "build_environment": BUILD_ENV,
            "job": JOB,
            "test_config": TEST_CONFIG,
            "run_id": RUN_ID,
            "run_number": RUN_NUMBER,
            "run_attempt": RUN_ATTEMPT,
            "some_number": 123,
            "float_number": decimal.Decimal(str(32.34)),
        }

        # Preserve the metric emitted
        emitted_metric: Dict[str, Any] = {}

        def mock_put_item(Item: Dict[str, Any]) -> None:
            nonlocal emitted_metric
            emitted_metric = Item

        mock_resource.return_value.Table.return_value.put_item = mock_put_item

        emit_metric("metric_name", metric)

        self.assertDictContainsSubset(emit_should_include, emitted_metric)

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
                "CI": "true",
                "BUILD_ENVIRONMENT": BUILD_ENV,
            },
            clear=True,
        ).start()

        put_item_invoked = False

        def mock_put_item(Item: Dict[str, Any]) -> None:
            nonlocal put_item_invoked
            put_item_invoked = True

        mock_resource.return_value.Table.return_value.put_item = mock_put_item

        emit_metric("metric_name", metric)

        self.assertFalse(put_item_invoked)

    def test_upload_to_rockset_batch_size(self) -> None:
        cases = [
            {
                "batch_size": BATCH_SIZE - 1,
                "expected_number_of_requests": 1,
            },
            {
                "batch_size": BATCH_SIZE,
                "expected_number_of_requests": 1,
            },
            {
                "batch_size": BATCH_SIZE + 1,
                "expected_number_of_requests": 2,
            },
        ]

        for case in cases:
            mock_client = mock.Mock()
            mock_client.Documents.add_documents.return_value = "OK"

            batch_size = case["batch_size"]
            expected_number_of_requests = case["expected_number_of_requests"]

            docs = list(range(batch_size))
            upload_to_rockset(
                collection="test", docs=docs, workspace="commons", client=mock_client
            )
            self.assertEqual(
                mock_client.Documents.add_documents.call_count,
                expected_number_of_requests,
            )


if __name__ == "__main__":
    unittest.main()
