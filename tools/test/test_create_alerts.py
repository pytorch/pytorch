import json
from datetime import datetime
from unittest import main, TestCase
from unittest.mock import patch

from tools.alerts.create_alerts import (
    fetch_alerts,
    filter_job_names,
    gen_update_comment,
    generate_no_flaky_tests_issue,
    handle_flaky_tests_alert,
    PYTORCH_ALERT_LABEL,
    TEST_INFRA_REPO_NAME,
)


JOB_NAME = "periodic / linux-xenial-cuda10.2-py3-gcc7-slow-gradcheck / test (default, 2, 2, linux.4xlarge.nvidia.gpu)"
DISABLED_JOB_NAMES = [
    "linux-focal-rocm5.3-py3.8-slow / test (slow, 1, 1, linux.rocm.gpu, rerun_disabled_tests)",
    "unstable / linux-bionic-py3_7-clang8-xla / test (xla, 1, 1, linux.4xlarge)",
]
MOCK_TEST_DATA = [
    {
        "sha": "f02f3046571d21b48af3067e308a1e0f29b43af9",
        "id": 7819529276,
        "conclusion": "failure",
        "htmlUrl": "https://github.com/pytorch/pytorch/runs/7819529276?check_suite_focus=true",
        "logUrl": "https://ossci-raw-job-status.s3.amazonaws.com/log/7819529276",
        "durationS": 14876,
        "failureLine": "##[error]The action has timed out.",
        "failureContext": "",
        "failureCaptures": ["##[error]The action has timed out."],
        "failureLineNumber": 83818,
        "repo": "pytorch/pytorch",
    },
    {
        "sha": "d0d6b1f2222bf90f478796d84a525869898f55b6",
        "id": 7818399623,
        "conclusion": "failure",
        "htmlUrl": "https://github.com/pytorch/pytorch/runs/7818399623?check_suite_focus=true",
        "logUrl": "https://ossci-raw-job-status.s3.amazonaws.com/log/7818399623",
        "durationS": 14882,
        "failureLine": "##[error]The action has timed out.",
        "failureContext": "",
        "failureCaptures": ["##[error]The action has timed out."],
        "failureLineNumber": 72821,
        "repo": "pytorch/pytorch",
    },
]
ANOTHER_MOCK_TEST_DATA = [
    {
        "sha": "2936c8b9ce4ef4d81cc3fe6e43531cb440209c61",
        "id": 4364234624,
        "conclusion": "failure",
        "htmlUrl": "https://github.com/pytorch/pytorch/runs/4364234624?check_suite_focus=true",
        "logUrl": "https://ossci-raw-job-status.s3.amazonaws.com/log/4364234624",
        "durationS": 14342,
        "failureLine": "##[error]An unique error here.",
        "failureContext": "",
        "failureCaptures": ["##[error]An unique error here."],
        "failureLineNumber": 12345,
        "repo": "pytorch/pytorch",
    },
]


def mock_fetch_alerts(*args, **kwargs):
    """
    Return the mock JSON response when trying to fetch all existing alerts
    """

    class MockResponse:
        def __init__(self, json_data, status_code):
            self.text = json_data
            self.status_code = status_code

        def raise_for_status(self):
            pass

    response = {
        "data": {
            "repository": {
                "issues": {
                    "nodes": [
                        {
                            "title": "[Pytorch] There are 3 Recurrently Failing Jobs on pytorch/pytorch main",
                            "closed": False,
                            "number": 3763,
                            "body": "",
                            "comments": {"nodes": []},
                        },
                        {
                            "title": "[Pytorch] There are 3 Recurrently Failing Jobs on pytorch/pytorch nightly",
                            "closed": False,
                            "number": 3764,
                            "body": "",
                            "comments": {"nodes": []},
                        },
                    ]
                }
            }
        }
    }
    return MockResponse(json.dumps(response), 200)


class TestGitHubPR(TestCase):
    # Should fail when jobs are ? ? Fail Fail
    def test_alert(self) -> None:
        status = JobStatus(JOB_NAME, [{}] + [{}] + MOCK_TEST_DATA)
        self.assertTrue(status.should_alert())

    # Shouldn't alert when a newer job has already succeeded
    def test_no_alert_when_cleared(self) -> None:
        cases = [
            JobStatus(JOB_NAME, [{"conclusion": "success"}] + [{}] + MOCK_TEST_DATA),
            JobStatus(
                JOB_NAME,
                [{"conclusion": "pending"}]
                + [{"conclusion": "success"}]
                + MOCK_TEST_DATA,
            ),
        ]
        for case in cases:
            self.assertFalse(case.should_alert())

    # Shouldn't alert when jobs are Fail Success Fail
    def test_no_alert_when_not_consecutive(self) -> None:
        status = JobStatus(
            JOB_NAME,
            [MOCK_TEST_DATA[0]] + [{"conclusion": "success"}] + [MOCK_TEST_DATA[1]],
        )
        self.assertFalse(status.should_alert())

    # Shouldn't alert when the middle job is not yet done Fail ? Fail
    def test_no_alert_when_pending_job(self) -> None:
        status = JobStatus(
            JOB_NAME,
            [MOCK_TEST_DATA[0]] + [{"conclusion": "pending"}] + [MOCK_TEST_DATA[1]],
        )
        self.assertFalse(status.should_alert())

    # Shouldn't alert when failures are different ? Fail (1) Fail (2)
    def test_no_alert_when_different_failures(self) -> None:
        status = JobStatus(
            JOB_NAME, [{}] + [MOCK_TEST_DATA[0]] + ANOTHER_MOCK_TEST_DATA
        )
        self.assertFalse(status.should_alert())

    # No need to send alerts for some jobs
    def test_disabled_alert(self) -> None:
        for job_name in DISABLED_JOB_NAMES:
            status = JobStatus(job_name, [{}] + [{}] + MOCK_TEST_DATA)
            self.assertFalse(status.should_alert())

    def test_update_comment_empty(self):
        jobs = [JobStatus("job1", [{}]), JobStatus("job2", [{}])]
        body = (
            "- [job1](a) failed consecutively starting with commit []()\n"
            "- [job2](a) failed consecutively starting with commit []()"
        )
        update_comment = gen_update_comment(body, jobs)
        self.assertFalse(update_comment)

        jobs = [JobStatus("job1", [{}]), JobStatus("job2", [{}])]
        body = "- [job1](a) failed consecutively starting with commit []()"
        update_comment = gen_update_comment(body, jobs)
        self.assertTrue("started failing" in update_comment)
        self.assertTrue("job2" in update_comment)

        jobs = [JobStatus("job1", [{}])]
        body = (
            "- [job1](a) failed consecutively starting with commit []()\n"
            "- [job2](a) failed consecutively starting with commit []()"
        )
        update_comment = gen_update_comment(body, jobs)
        self.assertTrue("stopped failing" in update_comment)
        self.assertTrue("job2" in update_comment)

    def test_generate_no_flaky_tests_issue(self):
        issue = generate_no_flaky_tests_issue()
        self.assertListEqual(issue["labels"], ["no-flaky-tests-alert"])

    @patch("check_alerts.create_issue")
    @patch("check_alerts.datetime")
    @patch("check_alerts.get_num_issues_with_label")
    def test_handle_flaky_tests_alert(
        self, mock_get_num_issues_with_label, mock_date, mock_create_issue
    ):
        mock_issue = {
            "title": "dummy-title",
            "labels": ["dummy-label"],
        }
        mock_create_issue.return_value = mock_issue
        mock_date.today.return_value = datetime(2022, 10, 10)
        mock_get_num_issues_with_label.return_value = 5

        res = handle_flaky_tests_alert([])
        self.assertIsNone(res)

        existing_alerts = [
            {"createdAt": "2022-10-10T13:41:09Z"},
            {"createdAt": "2022-10-08T14:41:09Z"},
        ]
        res = handle_flaky_tests_alert(existing_alerts)
        self.assertIsNone(res)

        existing_alerts = [
            {"createdAt": "2022-10-09T13:41:09Z"},
            {"createdAt": "2022-10-08T14:41:09Z"},
        ]
        res = handle_flaky_tests_alert(existing_alerts)
        self.assertIsNone(res)

        mock_get_num_issues_with_label.return_value = 0
        res = handle_flaky_tests_alert(existing_alerts)
        self.assertDictEqual(res, mock_issue)

    # test filter job names
    def test_job_filter(self):
        job_names = [
            "pytorch_linux_xenial_py3_6_gcc5_4_test",
            "pytorch_linux_xenial_py3_6_gcc5_4_test2",
        ]
        self.assertListEqual(
            filter_job_names(job_names, ""),
            job_names,
            "empty regex should match all jobs",
        )
        self.assertListEqual(filter_job_names(job_names, ".*"), job_names)
        self.assertListEqual(filter_job_names(job_names, ".*xenial.*"), job_names)
        self.assertListEqual(
            filter_job_names(job_names, ".*xenial.*test2"),
            ["pytorch_linux_xenial_py3_6_gcc5_4_test2"],
        )
        self.assertListEqual(filter_job_names(job_names, ".*xenial.*test3"), [])
        self.assertRaises(
            Exception,
            lambda: filter_job_names(job_names, "["),
            msg="malformed regex should throw exception",
        )

    def test_builder_job_filter(self):
        job_names = [
            "cron / nightly / win / wheel-py3_8-cuda11_8 / wheel-py3_8-cuda11_8",
            "cron / release / linux / conda-py3_10-cpu / conda-py3_10-cpu",
            "Validate Nightly PyPI Wheel Binary Size / nightly-pypi-binary-size-validation",
            "Build libtorch docker images / build-docker-cuda (11.8)"
            "Validate binaries / linux",
        ]
        self.assertListEqual(
            filter_job_names(
                job_names, ".*nightly.pypi.binary.size.validation|cron / release /"
            ),
            [
                "cron / release / linux / conda-py3_10-cpu / conda-py3_10-cpu",
                "Validate Nightly PyPI Wheel Binary Size / nightly-pypi-binary-size-validation",
            ],
        )

    @patch("requests.post", side_effect=mock_fetch_alerts)
    def test_fetch_alert(self, mocked_alerts):
        cases = [
            {
                "repo": "pytorch/builder",
                "branch": "main",
                "expected": [],
            },
            {
                "repo": "pytorch/pytorch",
                "branch": "main",
                "expected": [
                    {
                        "title": "[Pytorch] There are 3 Recurrently Failing Jobs on pytorch/pytorch main",
                        "closed": False,
                        "number": 3763,
                        "body": "",
                        "comments": {"nodes": []},
                    },
                ],
            },
            {
                "repo": "pytorch/pytorch",
                "branch": "nightly",
                "expected": [
                    {
                        "title": "[Pytorch] There are 3 Recurrently Failing Jobs on pytorch/pytorch nightly",
                        "closed": False,
                        "number": 3764,
                        "body": "",
                        "comments": {"nodes": []},
                    },
                ],
            },
        ]

        for case in cases:
            alerts = fetch_alerts(
                repo=case["repo"],
                branch=case["branch"],
                alert_repo=TEST_INFRA_REPO_NAME,
                labels=PYTORCH_ALERT_LABEL,
            )
            self.assertListEqual(alerts, case["expected"])


if __name__ == "__main__":
    main()