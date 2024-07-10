from __future__ import annotations

import decimal
import inspect
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from tools.stats.upload_metrics import add_global_metric, emit_metric

from tools.stats.upload_stats_lib import BATCH_SIZE, upload_to_rockset

sys.path.remove(str(REPO_ROOT))

# default values
REPO = "some/repo"
BUILD_ENV = "cuda-10.2"
TEST_CONFIG = "test-config"
WORKFLOW = "some-workflow"
JOB = "some-job"
RUN_ID = 56
RUN_NUMBER = 123
RUN_ATTEMPT = 3
PR_NUMBER = 6789
JOB_ID = 234
JOB_NAME = "some-job-name"


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
                "JOB_ID": str(JOB_ID),
                "JOB_NAME": str(JOB_NAME),
            },
            clear=True,  # Don't read any preset env vars
        ).start()

    @mock.patch("boto3.Session.resource")
    def test_emits_default_and_given_metrics(self, mock_resource: Any) -> None:
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
            "calling_function": "test_emits_default_and_given_metrics",
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
            "job_id": JOB_ID,
            "job_name": JOB_NAME,
        }

        # Preserve the metric emitted
        emitted_metric: dict[str, Any] = {}

        def mock_put_item(Item: dict[str, Any]) -> None:
            nonlocal emitted_metric
            emitted_metric = Item

        mock_resource.return_value.Table.return_value.put_item = mock_put_item

        emit_metric("metric_name", metric)

        self.assertEqual(
            emitted_metric,
            {**emit_should_include, **emitted_metric},
        )

    @mock.patch("boto3.Session.resource")
    def test_when_global_metric_specified_then_it_emits_it(
        self, mock_resource: Any
    ) -> None:
        metric = {
            "some_number": 123,
        }

        global_metric_name = "global_metric"
        global_metric_value = "global_value"

        add_global_metric(global_metric_name, global_metric_value)

        emit_should_include = {
            **metric,
            global_metric_name: global_metric_value,
        }

        # Preserve the metric emitted
        emitted_metric: dict[str, Any] = {}

        def mock_put_item(Item: dict[str, Any]) -> None:
            nonlocal emitted_metric
            emitted_metric = Item

        mock_resource.return_value.Table.return_value.put_item = mock_put_item

        emit_metric("metric_name", metric)

        self.assertEqual(
            emitted_metric,
            {**emitted_metric, **emit_should_include},
        )

    @mock.patch("boto3.Session.resource")
    def test_when_local_and_global_metric_specified_then_global_is_overridden(
        self, mock_resource: Any
    ) -> None:
        global_metric_name = "global_metric"
        global_metric_value = "global_value"
        local_override = "local_override"

        add_global_metric(global_metric_name, global_metric_value)

        metric = {
            "some_number": 123,
            global_metric_name: local_override,
        }

        emit_should_include = {
            **metric,
            global_metric_name: local_override,
        }

        # Preserve the metric emitted
        emitted_metric: dict[str, Any] = {}

        def mock_put_item(Item: dict[str, Any]) -> None:
            nonlocal emitted_metric
            emitted_metric = Item

        mock_resource.return_value.Table.return_value.put_item = mock_put_item

        emit_metric("metric_name", metric)

        self.assertEqual(
            emitted_metric,
            {**emitted_metric, **emit_should_include},
        )

    @mock.patch("boto3.Session.resource")
    def test_when_optional_envvar_set_to_actual_value_then_emit_vars_emits_it(
        self, mock_resource: Any
    ) -> None:
        metric = {
            "some_number": 123,
        }

        emit_should_include = {
            **metric,
            "pr_number": PR_NUMBER,
        }

        mock.patch.dict(
            "os.environ",
            {
                "PR_NUMBER": str(PR_NUMBER),
            },
        ).start()

        # Preserve the metric emitted
        emitted_metric: dict[str, Any] = {}

        def mock_put_item(Item: dict[str, Any]) -> None:
            nonlocal emitted_metric
            emitted_metric = Item

        mock_resource.return_value.Table.return_value.put_item = mock_put_item

        emit_metric("metric_name", metric)

        self.assertEqual(
            emitted_metric,
            {**emit_should_include, **emitted_metric},
        )

    @mock.patch("boto3.Session.resource")
    def test_when_optional_envvar_set_to_a_empty_str_then_emit_vars_ignores_it(
        self, mock_resource: Any
    ) -> None:
        metric = {"some_number": 123}

        emit_should_include: dict[str, Any] = metric.copy()

        # Github Actions defaults some env vars to an empty string
        default_val = ""
        mock.patch.dict(
            "os.environ",
            {
                "PR_NUMBER": default_val,
            },
        ).start()

        # Preserve the metric emitted
        emitted_metric: dict[str, Any] = {}

        def mock_put_item(Item: dict[str, Any]) -> None:
            nonlocal emitted_metric
            emitted_metric = Item

        mock_resource.return_value.Table.return_value.put_item = mock_put_item

        emit_metric("metric_name", metric)

        self.assertEqual(
            emitted_metric,
            {**emit_should_include, **emitted_metric},
            f"Metrics should be emitted when an option parameter is set to '{default_val}'",
        )
        self.assertFalse(
            emitted_metric.get("pr_number"),
            f"Metrics should not include optional item 'pr_number' when it's envvar is set to '{default_val}'",
        )

    @mock.patch("boto3.Session.resource")
    def test_blocks_emission_if_reserved_keyword_used(self, mock_resource: Any) -> None:
        metric = {"repo": "awesome/repo"}

        with self.assertRaises(ValueError):
            emit_metric("metric_name", metric)

    @mock.patch("boto3.Session.resource")
    def test_no_metrics_emitted_if_required_env_var_not_set(
        self, mock_resource: Any
    ) -> None:
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

        def mock_put_item(Item: dict[str, Any]) -> None:
            nonlocal put_item_invoked
            put_item_invoked = True

        mock_resource.return_value.Table.return_value.put_item = mock_put_item

        emit_metric("metric_name", metric)

        self.assertFalse(put_item_invoked)

    @mock.patch("boto3.Session.resource")
    def test_no_metrics_emitted_if_required_env_var_set_to_empty_string(
        self, mock_resource: Any
    ) -> None:
        metric = {"some_number": 123}

        mock.patch.dict(
            "os.environ",
            {
                "GITHUB_JOB": "",
            },
        ).start()

        put_item_invoked = False

        def mock_put_item(Item: dict[str, Any]) -> None:
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
