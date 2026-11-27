from __future__ import annotations

import gzip
import inspect
import json
import os
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import requests


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.stats import upload_stats_lib
from tools.stats.upload_metrics import add_global_metric, emit_metric, global_metrics
from tools.stats.upload_stats_lib import get_s3_resource, remove_nan_inf


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


@mock.patch("boto3.resource")
class TestUploadStats(unittest.TestCase):
    emitted_metric: dict[str, Any] = {"did_not_emit": True}

    def mock_put_item(self, **kwargs: Any) -> None:
        # Utility for mocking putting items into s3.  THis will save the emitted
        # metric so tests can check it
        self.emitted_metric = json.loads(
            gzip.decompress(kwargs["Body"]).decode("utf-8")
        )

    # Before each test, set the env vars to their default values
    def setUp(self) -> None:
        get_s3_resource.cache_clear()
        global_metrics.clear()

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
            "job_id": JOB_ID,
            "job_name": JOB_NAME,
            "info": metric,
        }

        mock_resource.return_value.Object.return_value.put = self.mock_put_item

        emit_metric("metric_name", metric)

        self.assertEqual(
            self.emitted_metric,
            {**self.emitted_metric, **emit_should_include},
        )

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

        mock_resource.return_value.Object.return_value.put = self.mock_put_item

        emit_metric("metric_name", metric)

        self.assertEqual(
            self.emitted_metric,
            {**self.emitted_metric, "info": emit_should_include},
        )

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

        mock_resource.return_value.Object.return_value.put = self.mock_put_item

        emit_metric("metric_name", metric)

        self.assertEqual(
            self.emitted_metric,
            {**self.emitted_metric, "info": emit_should_include},
        )

    def test_when_optional_envvar_set_to_actual_value_then_emit_vars_emits_it(
        self, mock_resource: Any
    ) -> None:
        metric = {
            "some_number": 123,
        }

        emit_should_include = {
            "info": {**metric},
            "pr_number": PR_NUMBER,
        }

        mock.patch.dict(
            "os.environ",
            {
                "PR_NUMBER": str(PR_NUMBER),
            },
        ).start()

        mock_resource.return_value.Object.return_value.put = self.mock_put_item

        emit_metric("metric_name", metric)

        self.assertEqual(
            self.emitted_metric,
            {**self.emitted_metric, **emit_should_include},
        )

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

        mock_resource.return_value.Object.return_value.put = self.mock_put_item

        emit_metric("metric_name", metric)

        self.assertEqual(
            self.emitted_metric,
            {**self.emitted_metric, "info": emit_should_include},
            f"Metrics should be emitted when an option parameter is set to '{default_val}'",
        )
        self.assertFalse(
            self.emitted_metric.get("pr_number"),
            f"Metrics should not include optional item 'pr_number' when it's envvar is set to '{default_val}'",
        )

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

        mock_resource.return_value.Object.return_value.put = self.mock_put_item

        emit_metric("metric_name", metric)

        self.assertTrue(self.emitted_metric["did_not_emit"])

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

        mock_resource.return_value.Object.return_value.put = self.mock_put_item

        emit_metric("metric_name", metric)

        self.assertTrue(self.emitted_metric["did_not_emit"])

    def test_remove_nan_inf(self, _mocked_resource: Any) -> None:
        checks = [
            (float("inf"), '"inf"', "Infinity"),
            (float("nan"), '"nan"', "NaN"),
            ({1: float("inf")}, '{"1": "inf"}', '{"1": Infinity}'),
            ([float("nan")], '["nan"]', "[NaN]"),
            ({1: [float("nan")]}, '{"1": ["nan"]}', '{"1": [NaN]}'),
        ]

        for input, clean, unclean in checks:
            clean_output = json.dumps(remove_nan_inf(input))
            unclean_output = json.dumps(input)
            self.assertEqual(
                clean_output,
                clean,
                f"Expected {clean} when input is {unclean}, got {clean_output}",
            )
            self.assertEqual(
                unclean_output,
                unclean,
                f"Expected {unclean} when input is {unclean}, got {unclean_output}",
            )


def _response(status: int, headers: dict[str, str] | None = None) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response.headers.update(headers or {})
    return response


class TestGithubGet(unittest.TestCase):
    def _get(self, *responses: requests.Response) -> tuple[int, list[float]]:
        """The final status and the sleeps, at time 1000."""
        with (
            mock.patch.dict(os.environ, {"GITHUB_TOKEN": "token"}),
            mock.patch.object(requests, "get", side_effect=responses),
            mock.patch.object(upload_stats_lib.time, "time", return_value=1000),
            mock.patch.object(upload_stats_lib.time, "sleep") as sleep,
        ):
            response = upload_stats_lib._github_get("url")
        return response.status_code, [call.args[0] for call in sleep.call_args_list]

    def test_waits_for_a_used_up_quota_to_reset(self) -> None:
        used_up = {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1060"}
        result = self._get(_response(403, used_up), _response(200))
        self.assertEqual(result, (200, [65]))

    def test_waits_once(self) -> None:
        used_up = {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1060"}
        result = self._get(_response(403, used_up), _response(403, used_up))
        self.assertEqual(result, (403, [65]))

    def test_other_errors_return_at_once(self) -> None:
        forbidden = {"X-RateLimit-Remaining": "4999"}
        self.assertEqual(self._get(_response(403, forbidden)), (403, []))

    def test_ignores_a_malformed_reset(self) -> None:
        malformed = {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "soon"}
        self.assertEqual(self._get(_response(403, malformed)), (403, []))

    def test_ignores_a_reset_past_the_deadline(self) -> None:
        bogus = {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "9999999999"}
        self.assertEqual(self._get(_response(403, bogus)), (403, []))


if __name__ == "__main__":
    unittest.main()
