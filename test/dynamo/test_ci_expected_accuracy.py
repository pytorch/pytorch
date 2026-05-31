# Owner(s): ["module: dynamo"]

import io
import os
import pathlib
import sys
from unittest import mock


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from benchmarks.dynamo.ci_expected_accuracy import update_expected


sys.path.remove(str(REPO_ROOT))

from torch._dynamo.test_case import run_tests, TestCase


class FakeResponse:
    def __init__(self, payload, links=None, headers=None, status_code=200):
        self.payload = payload
        self.links = links or {}
        self.headers = headers or {}
        self.status_code = status_code

    def json(self):
        return self.payload

    def raise_for_status(self):
        pass

    def close(self):
        pass


class TestUpdateExpected(TestCase):
    def test_query_job_sha_without_clickhouse_credentials_uses_github(self):
        github_results = [{"workflowName": "inductor"}]
        with (
            mock.patch.dict(os.environ, {"CH_KEY_ID": "", "CH_KEY_SECRET": ""}),
            mock.patch.object(
                update_expected,
                "query_job_sha_from_github",
                return_value=github_results,
            ) as query_github,
            mock.patch.object(
                update_expected, "query_job_sha_from_clickhouse"
            ) as query_clickhouse,
            mock.patch("sys.stderr", new_callable=io.StringIO),
        ):
            self.assertEqual(
                update_expected.query_job_sha(
                    "pytorch/pytorch",
                    "49ff884b1edc3b872eeb2387ec60ef230cae7f24",
                ),
                github_results,
            )

        query_github.assert_called_once_with(
            "pytorch/pytorch", "49ff884b1edc3b872eeb2387ec60ef230cae7f24"
        )
        query_clickhouse.assert_not_called()

    def test_github_fallback_pagination_and_artifact_urls(self):
        sha = "49ff884b1edc3b872eeb2387ec60ef230cae7f24"
        runs_url = (
            f"{update_expected.GITHUB_API_URL}/repos/pytorch/pytorch/actions/runs"
        )
        runs_next_url = f"{runs_url}?page=2"
        workflow_id = 17154115801
        jobs_url = (
            f"{update_expected.GITHUB_API_URL}/repos/pytorch/pytorch"
            f"/actions/runs/{workflow_id}/jobs"
        )
        jobs_next_url = f"{jobs_url}?page=2"
        calls = []
        artifact_url = (
            f"{update_expected.S3_BASE_URL}/pytorch/pytorch/"  # @lint-ignore
            "17154115801/3/artifact/"
            "test-reports-test-inductor_torchbench-1-2-"
            "linux.8xlarge.amx_48676215560.zip"
        )

        runs_page_1 = {
            "workflow_runs": [
                {
                    "id": 456,
                    "name": "inductor",
                    "event": "workflow_run",
                    "head_sha": sha,
                    "run_attempt": 1,
                    "created_at": "2025-08-22T11:00:00Z",
                }
            ]
        }
        runs_page_2 = {
            "workflow_runs": [
                {
                    "id": workflow_id,
                    "name": "inductor",
                    "event": "push",
                    "head_sha": sha,
                    "run_attempt": 3,
                    "created_at": "2025-08-22T11:31:16Z",
                },
                {
                    "id": 789,
                    "name": "pull",
                    "event": "push",
                    "head_sha": sha,
                    "run_attempt": 3,
                    "created_at": "2025-08-22T11:40:00Z",
                },
            ]
        }
        jobs_page_1 = {
            "jobs": [
                {
                    "id": 48676215561,
                    "name": "generate-test-matrix",
                    "head_sha": sha,
                    "run_attempt": 3,
                    "created_at": "2025-08-22T12:00:00Z",
                }
            ]
        }
        jobs_page_2 = {
            "jobs": [
                {
                    "id": 48676215560,
                    "name": (
                        "linux-jammy-cpu-py3.9-gcc11-inductor / "
                        "test (inductor_torchbench, 1, 2, linux.8xlarge.amx)"
                    ),
                    "head_sha": sha,
                    "run_attempt": 3,
                    "created_at": "2025-08-22T12:05:00Z",
                },
                {
                    "id": 48676215562,
                    "name": (
                        "linux-jammy-cpu-py3.9-gcc11-inductor / "
                        "test (inductor_timm, 1, 2, linux.8xlarge.amx)"
                    ),
                    "head_sha": "different-sha",
                    "run_attempt": 3,
                    "created_at": "2025-08-22T12:07:00Z",
                },
            ]
        }

        def fake_get(url, params=None, headers=None, timeout=None):
            calls.append(
                {
                    "url": url,
                    "params": params,
                    "headers": headers,
                    "timeout": timeout,
                }
            )
            if url == runs_url:
                return FakeResponse(runs_page_1, links={"next": {"url": runs_next_url}})
            if url == runs_next_url:
                return FakeResponse(runs_page_2)
            if url == jobs_url:
                return FakeResponse(jobs_page_1, links={"next": {"url": jobs_next_url}})
            if url == jobs_next_url:
                return FakeResponse(jobs_page_2)
            raise AssertionError(f"unexpected URL: {url}")

        with (
            mock.patch.dict(
                os.environ,
                {"CH_KEY_ID": "", "CH_KEY_SECRET": "", "GITHUB_TOKEN": ""},
            ),
            mock.patch.object(update_expected.requests, "get", side_effect=fake_get),
        ):
            results = update_expected.query_job_sha_from_github("pytorch/pytorch", sha)

        self.assertEqual(
            results,
            [
                {
                    "workflowName": "inductor",
                    "jobName": (
                        "linux-jammy-cpu-py3.9-gcc11-inductor / "
                        "test (inductor_torchbench, 1, 2, linux.8xlarge.amx)"
                    ),
                    "id": "48676215560",
                    "runAttempt": 3,
                    "workflowId": str(workflow_id),
                    "time": "2025-08-22T12:05:00Z",
                }
            ],
        )
        self.assertEqual(
            update_expected.get_artifacts_urls(
                "pytorch/pytorch",
                results,
                {"inductor_torchbench"},
                is_rocm=False,
            ),
            {("inductor_torchbench", 1): [artifact_url]},
        )
        self.assertEqual(
            [call["url"] for call in calls],
            [runs_url, runs_next_url, jobs_url, jobs_next_url],
        )
        self.assertEqual(calls[0]["params"], {"head_sha": sha, "per_page": 100})
        self.assertIsNone(calls[1]["params"])
        self.assertEqual(calls[2]["params"], {"filter": "all", "per_page": 100})
        self.assertIsNone(calls[3]["params"])
        self.assertTrue(all(call["timeout"] == 30 for call in calls))
        self.assertTrue(all("Authorization" not in call["headers"] for call in calls))


if __name__ == "__main__":
    run_tests()
