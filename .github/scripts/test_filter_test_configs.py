#!/usr/bin/env python3

import json
import os
import tempfile
from typing import Any, Dict, List
from unittest import main, mock, TestCase

import yaml
from filter_test_configs import (
    filter,
    filter_selected_test_configs,
    get_labels,
    mark_unstable_jobs,
    parse_reenabled_issues,
    perform_misc_tasks,
    PREFIX,
    remove_disabled_jobs,
    set_periodic_modes,
    SUPPORTED_PERIODICAL_MODES,
)


MOCKED_DISABLED_UNSTABLE_JOBS = {
    "pull / mock-platform-1": [
        "pytorchbot",
        "1",
        "https://github.com/pytorch/pytorch/issues/1",
        "pull",
        "mock-platform-1",
        "",
    ],
    "trunk / mock-platform-2 / build": [
        "pytorchbot",
        "2",
        "https://github.com/pytorch/pytorch/issues/2",
        "trunk",
        "mock-platform-2",
        "build",
    ],
    "periodic / mock-platform-3 / test": [
        "pytorchbot",
        "3",
        "https://github.com/pytorch/pytorch/issues/3",
        "periodic",
        "mock-platform-3",
        "test",
    ],
    "pull / mock-platform-4 / build-and-test": [
        "pytorchbot",
        "4",
        "https://github.com/pytorch/pytorch/issues/4",
        "pull",
        "mock-platform-4",
        "build-and-test",
    ],
    "trunk / mock-platform-5 / test (backward_compat)": [
        "pytorchbot",
        "5",
        "https://github.com/pytorch/pytorch/issues/5",
        "trunk",
        "mock-platform-5",
        "test (backward_compat)",
    ],
    "periodic / mock-platform-6 / build-and-test (default)": [
        "pytorchbot",
        "6",
        "https://github.com/pytorch/pytorch/issues/6",
        "periodic",
        "mock-platform-6",
        "build-and-test (default)",
    ],
    "pull / mock-platform-7 / test [invalid syntax]": [
        "pytorchbot",
        "7",
        "https://github.com/pytorch/pytorch/issues/7",
        "pull",
        "mock-platform-7",
        "test [invalid syntax]",
    ],
    "trunk / mock-platform-8 / build (dynamo)": [
        "pytorchbot",
        "8",
        "https://github.com/pytorch/pytorch/issues/8",
        "trunk",
        "mock-platform-8",
        "build (dynamo)",
    ],
    "linux-binary-libtorch-cxx11-abi / libtorch-cpu-shared-with-deps-cxx11-abi-test / test": [
        "pytorchbot",
        "9",
        "https://github.com/pytorch/pytorch/issues/9",
        "linux-binary-libtorch-cxx11-abi",
        "libtorch-cpu-shared-with-deps-cxx11-abi-test",
        "test",
    ],
    "linux-binary-manywheel / manywheel-py3_8-cuda11_8-build": [
        "pytorchbot",
        "10",
        "https://github.com/pytorch/pytorch/issues/10",
        "linux-binary-manywheel",
        "manywheel-py3_8-cuda11_8-build",
        "",
    ],
    "inductor / cuda12.1-py3.10-gcc9-sm86 / test (inductor)": [
        "pytorchbot",
        "107079",
        "https://github.com/pytorch/pytorch/issues/107079",
        "inductor",
        "cuda12.1-py3.10-gcc9-sm86",
        "test (inductor)",
    ],
    "inductor / cuda12.1-py3.10-gcc9-sm86 / test (inductor_huggingface)": [
        "pytorchbot",
        "109153",
        "https://github.com/pytorch/pytorch/issues/109153",
        "inductor",
        "cuda12.1-py3.10-gcc9-sm86",
        "test (inductor_huggingface)",
    ],
    "inductor / cuda12.1-py3.10-gcc9-sm86 / test (inductor_huggingface_dynamic)": [
        "pytorchbot",
        "109154",
        "https://github.com/pytorch/pytorch/issues/109154",
        "inductor",
        "cuda12.1-py3.10-gcc9-sm86",
        "test (inductor_huggingface_dynamic)",
    ],
}

MOCKED_PR_INFO = {
    "url": "https://api.github.com/repos/pytorch/pytorch/issues/10338",
    "repository_url": "https://api.github.com/repos/pytorch/pytorch",
    "labels_url": "https://api.github.com/repos/pytorch/pytorch/issues/10338/labels{/name}",
    "comments_url": "https://api.github.com/repos/pytorch/pytorch/issues/10338/comments",
    "events_url": "https://api.github.com/repos/pytorch/pytorch/issues/10338/events",
    "html_url": "https://github.com/pytorch/pytorch/pull/10338",
    "id": 348543815,
    "node_id": "MDExOlB1bGxSZXF1ZXN0MjA2ODcwMTUy",
    "number": 10338,
    "title": "Add matrix_rank",
    "user": {
        "login": "vishwakftw",
        "id": 23639302,
        "node_id": "MDQ6VXNlcjIzNjM5MzAy",
        "avatar_url": "https://avatars.githubusercontent.com/u/23639302?v=4",
        "gravatar_id": "",
        "url": "https://api.github.com/users/vishwakftw",
        "html_url": "https://github.com/vishwakftw",
        "followers_url": "https://api.github.com/users/vishwakftw/followers",
        "following_url": "https://api.github.com/users/vishwakftw/following{/other_user}",
        "gists_url": "https://api.github.com/users/vishwakftw/gists{/gist_id}",
        "starred_url": "https://api.github.com/users/vishwakftw/starred{/owner}{/repo}",
        "subscriptions_url": "https://api.github.com/users/vishwakftw/subscriptions",
        "organizations_url": "https://api.github.com/users/vishwakftw/orgs",
        "repos_url": "https://api.github.com/users/vishwakftw/repos",
        "events_url": "https://api.github.com/users/vishwakftw/events{/privacy}",
        "received_events_url": "https://api.github.com/users/vishwakftw/received_events",
        "type": "User",
        "site_admin": False,
    },
    "labels": [
        {
            "id": 1392590051,
            "node_id": "MDU6TGFiZWwxMzkyNTkwMDUx",
            "url": "https://api.github.com/repos/pytorch/pytorch/labels/open%20source",
            "name": "open source",
            "color": "ededed",
            "default": False,
            "description": None,
        },
        {
            "id": 1392590051,
            "node_id": "MDU6TGFiZWwxMzkyNTkwMDUx",
            "url": "https://api.github.com/repos/pytorch/pytorch/labels/open%20source",
            "name": "foo",
            "color": "ededed",
            "default": False,
            "description": None,
        },
        {
            "id": 1392590051,
            "node_id": "MDU6TGFiZWwxMzkyNTkwMDUx",
            "url": "https://api.github.com/repos/pytorch/pytorch/labels/open%20source",
            "name": "",
            "color": "ededed",
            "default": False,
            "description": None,
        },
    ],
    "state": "closed",
    "locked": False,
    "assignee": None,
    "assignees": [],
    "milestone": None,
    "comments": 9,
    "created_at": "2018-08-08T01:39:20Z",
    "updated_at": "2019-06-24T21:05:45Z",
    "closed_at": "2018-08-23T01:58:38Z",
    "author_association": "CONTRIBUTOR",
    "active_lock_reason": None,
    "draft": False,
    "pull_request": {
        "url": "https://api.github.com/repos/pytorch/pytorch/pulls/10338",
        "html_url": "https://github.com/pytorch/pytorch/pull/10338",
        "diff_url": "https://github.com/pytorch/pytorch/pull/10338.diff",
        "patch_url": "https://github.com/pytorch/pytorch/pull/10338.patch",
        "merged_at": None,
    },
    "body": "- Similar functionality as NumPy\r\n- Added doc string\r\n- Added tests\r\n\r\ncc: @SsnL \r\n\r\nCloses #10292 ",
    "closed_by": {
        "login": "vishwakftw",
        "id": 23639302,
        "node_id": "MDQ6VXNlcjIzNjM5MzAy",
        "avatar_url": "https://avatars.githubusercontent.com/u/23639302?v=4",
        "gravatar_id": "",
        "url": "https://api.github.com/users/vishwakftw",
        "html_url": "https://github.com/vishwakftw",
        "followers_url": "https://api.github.com/users/vishwakftw/followers",
        "following_url": "https://api.github.com/users/vishwakftw/following{/other_user}",
        "gists_url": "https://api.github.com/users/vishwakftw/gists{/gist_id}",
        "starred_url": "https://api.github.com/users/vishwakftw/starred{/owner}{/repo}",
        "subscriptions_url": "https://api.github.com/users/vishwakftw/subscriptions",
        "organizations_url": "https://api.github.com/users/vishwakftw/orgs",
        "repos_url": "https://api.github.com/users/vishwakftw/repos",
        "events_url": "https://api.github.com/users/vishwakftw/events{/privacy}",
        "received_events_url": "https://api.github.com/users/vishwakftw/received_events",
        "type": "User",
        "site_admin": False,
    },
    "reactions": {
        "url": "https://api.github.com/repos/pytorch/pytorch/issues/10338/reactions",
        "total_count": 2,
        "+1": 2,
        "-1": 0,
        "laugh": 0,
        "hooray": 0,
        "confused": 0,
        "heart": 0,
        "rocket": 0,
        "eyes": 0,
    },
    "timeline_url": "https://api.github.com/repos/pytorch/pytorch/issues/10338/timeline",
    "performed_via_github_app": None,
    "state_reason": None,
}


class TestConfigFilter(TestCase):
    def setUp(self) -> None:
        os.environ["GITHUB_TOKEN"] = "GITHUB_TOKEN"
        if os.getenv("GITHUB_OUTPUT"):
            del os.environ["GITHUB_OUTPUT"]

    def tearDown(self) -> None:
        if os.getenv("GITHUB_OUTPUT"):
            os.remove(str(os.getenv("GITHUB_OUTPUT")))

    @mock.patch("filter_test_configs.download_json")
    def test_get_labels(self, mock_download_json: Any) -> None:
        mock_download_json.return_value = MOCKED_PR_INFO
        labels = get_labels(pr_number=12345)
        self.assertSetEqual({"open source", "foo"}, labels)

    @mock.patch("filter_test_configs.download_json")
    def test_get_labels_failed(self, mock_download_json: Any) -> None:
        mock_download_json.return_value = {}
        labels = get_labels(pr_number=54321)
        self.assertFalse(labels)

    def test_filter(self) -> None:
        mocked_labels = {f"{PREFIX}cfg", "ciflow/trunk", "plain-cfg"}
        testcases = [
            {
                "test_matrix": '{include: [{config: "default", runner: "linux"}]}',
                "expected": '{"include": []}',
                "description": "Request test-config/cfg but the test matrix doesn't have it",
            },
            {
                "test_matrix": '{include: [{config: "default", runner: "linux"}, {config: "plain-cfg"}]}',
                "expected": '{"include": []}',
                "description": "A valid test config label needs to start with test-config/",
            },
            {
                "test_matrix": '{include: [{config: "default", runner: "linux"}, {config: "cfg", shard: 1}]}',
                "expected": '{"include": [{"config": "cfg", "shard": 1}]}',
                "description": "Found a match, only keep that",
            },
        ]

        for case in testcases:
            filtered_test_matrix = filter(
                yaml.safe_load(case["test_matrix"]), mocked_labels
            )
            self.assertEqual(case["expected"], json.dumps(filtered_test_matrix))

    def test_filter_with_test_config_label(self) -> None:
        mocked_labels = {f"{PREFIX}cfg", "ciflow/trunk"}

        testcases = [
            {
                "test_matrix": '{include: [{config: "default", runner: "linux"}]}',
                "expected": '{"include": []}',
                "description": "Found a valid label in the PR body, return the filtered test matrix",
            },
            {
                "test_matrix": '{include: [{config: "default", runner: "linux"}, {config: "cfg", shard: 1}]}',
                "expected": '{"include": [{"config": "cfg", "shard": 1}]}',
                "description": "Found a match, only keep that",
            },
        ]

        for case in testcases:
            filtered_test_matrix = filter(
                yaml.safe_load(case["test_matrix"]), mocked_labels
            )
            self.assertEqual(case["expected"], json.dumps(filtered_test_matrix))

    def test_filter_selected_test_configs(self) -> None:
        testcases = [
            {
                "test_matrix": '{include: [{config: "default"}]}',
                "selected_test_configs": "",
                "expected": '{"include": [{"config": "default"}]}',
                "description": "No selected test configs",
            },
            {
                "test_matrix": '{include: [{config: "default"}]}',
                "selected_test_configs": "foo",
                "expected": '{"include": []}',
                "description": "A different test config is selected",
            },
            {
                "test_matrix": '{include: [{config: "default"}]}',
                "selected_test_configs": "foo, bar",
                "expected": '{"include": []}',
                "description": "A different set of test configs is selected",
            },
            {
                "test_matrix": '{include: [{config: "default"}]}',
                "selected_test_configs": "foo, bar,default",
                "expected": '{"include": [{"config": "default"}]}',
                "description": "One of the test config is selected",
            },
            {
                "test_matrix": '{include: [{config: "default"}, {config: "bar"}]}',
                "selected_test_configs": "foo, bar,Default",
                "expected": '{"include": [{"config": "default"}, {"config": "bar"}]}',
                "description": "Several test configs are selected",
            },
        ]

        for case in testcases:
            selected_test_configs = {
                v.strip().lower()
                for v in case["selected_test_configs"].split(",")
                if v.strip()
            }
            filtered_test_matrix = filter_selected_test_configs(
                yaml.safe_load(case["test_matrix"]), selected_test_configs
            )
            self.assertEqual(case["expected"], json.dumps(filtered_test_matrix))

    def test_set_periodic_modes(self) -> None:
        testcases: List[Dict[str, str]] = [
            {
                "job_name": "a CI job",
                "test_matrix": "{include: []}",
                "description": "Empty test matrix",
            },
            {
                "job_name": "a-ci-job",
                "test_matrix": '{include: [{config: "default", runner: "linux"}, {config: "cfg", runner: "macos"}]}',
                "descripion": "Replicate each periodic mode in a different config",
            },
            {
                "job_name": "a-ci-cuda11.8-job",
                "test_matrix": '{include: [{config: "default", runner: "linux"}, {config: "cfg", runner: "macos"}]}',
                "descripion": "Replicate each periodic mode in a different config for a CUDA job",
            },
            {
                "job_name": "a-ci-rocm-job",
                "test_matrix": '{include: [{config: "default", runner: "linux"}, {config: "cfg", runner: "macos"}]}',
                "descripion": "Replicate each periodic mode in a different config for a ROCm job",
            },
            {
                "job_name": "",
                "test_matrix": '{include: [{config: "default", runner: "linux"}, {config: "cfg", runner: "macos"}]}',
                "descripion": "Empty job name",
            },
            {
                "test_matrix": '{include: [{config: "default", runner: "linux"}, {config: "cfg", runner: "macos"}]}',
                "descripion": "Missing job name",
            },
        ]

        for case in testcases:
            job_name = case.get("job_name", None)
            test_matrix = yaml.safe_load(case["test_matrix"])
            scheduled_test_matrix = set_periodic_modes(test_matrix, job_name)

            expected_modes = [
                m for m, c in SUPPORTED_PERIODICAL_MODES.items() if c(job_name)
            ]
            self.assertEqual(
                len(test_matrix["include"]) * len(expected_modes),
                len(scheduled_test_matrix["include"]),
            )

    @mock.patch("filter_test_configs.download_json")
    def test_remove_disabled_jobs(self, mock_download_json: Any) -> None:
        mock_download_json.return_value = MOCKED_DISABLED_UNSTABLE_JOBS

        testcases = [
            {
                "workflow": "pull",
                "job_name": "invalid job name",
                "test_matrix": '{include: [{config: "default"}]}',
                "expected": '{"include": [{"config": "default"}]}',
                "description": "invalid job name",
            },
            {
                "workflow": "pull",
                "job_name": "mock-platform-1 / build",
                "test_matrix": '{include: [{config: "default"}]}',
                "expected": '{"include": []}',
                "description": "disable build and test jobs",
            },
            {
                "workflow": "trunk",
                "job_name": "mock-platform-2 / build",
                "test_matrix": '{include: [{config: "default"}]}',
                "expected": '{"include": []}',
                "description": "disable build job",
            },
            {
                "workflow": "periodic",
                "job_name": "mock-platform-3 / test",
                "test_matrix": '{include: [{config: "default"}]}',
                "expected": '{"include": []}',
                "description": "disable test job",
            },
            {
                "workflow": "pull",
                "job_name": "mock-platform-4 / build-and-test",
                "test_matrix": '{include: [{config: "default"}]}',
                "expected": '{"include": []}',
                "description": "disable build-and-test job",
            },
            {
                "workflow": "trunk",
                "job_name": "mock-platform-5 / test",
                "test_matrix": '{include: [{config: "default", runner: "linux"}, {config: "backward_compat"}]}',
                "expected": '{"include": [{"config": "default", "runner": "linux"}]}',
                "description": "disable a test config",
            },
            {
                "workflow": "periodic",
                "job_name": "mock-platform-6 / build-and-test",
                "test_matrix": '{include: [{config: "default", runner: "linux"}, {config: "backward_compat"}]}',
                "expected": '{"include": [{"config": "backward_compat"}]}',
                "description": "disable a build-and-test config",
            },
            {
                "workflow": "pull",
                "job_name": "mock-platform-7 / test",
                "test_matrix": '{include: [{config: "default"}, {config: "backward_compat"}]}',
                "expected": '{"include": [{"config": "default"}, {"config": "backward_compat"}]}',
                "description": "include an invalid job name in the disabled issue",
            },
            {
                "workflow": "trunk",
                "job_name": "mock-platform-8 / build",
                "test_matrix": '{include: [{config: "default"}, {config: "backward_compat"}]}',
                "expected": '{"include": [{"config": "default"}, {"config": "backward_compat"}]}',
                "description": "include an invalid combination of build and test config",
            },
            {
                "workflow": "inductor",
                "job_name": "mock-platform-8 / build",
                "test_matrix": '{include: [{config: "default"}, {config: "backward_compat"}]}',
                "expected": '{"include": [{"config": "default"}, {"config": "backward_compat"}]}',
                "description": "not disabled on this workflow",
            },
            {
                "workflow": "pull",
                "job_name": "mock-platform-9 / build",
                "test_matrix": '{include: [{config: "default"}, {config: "backward_compat"}]}',
                "expected": '{"include": [{"config": "default"}, {"config": "backward_compat"}]}',
                "description": "not disabled on this platform",
            },
            {
                "workflow": "linux-binary-libtorch-cxx11-abi",
                "job_name": "libtorch-cpu-shared-with-deps-cxx11-abi-build / build",
                "test_matrix": '{include: [{config: "default"}]}',
                "expected": '{"include": []}',
                "description": "Build job is not needed when test job has been disabled",
            },
            {
                "workflow": "linux-binary-libtorch-cxx11-abi",
                "job_name": "libtorch-cpu-shared-with-deps-cxx11-abi-test / test",
                "test_matrix": '{include: [{config: "default"}]}',
                "expected": '{"include": []}',
                "description": "The binary test job is disabled on this platform",
            },
            {
                "workflow": "linux-binary-manywheel",
                "job_name": "manywheel-py3_8-cuda11_8-build / build",
                "test_matrix": '{include: [{config: "default"}]}',
                "expected": '{"include": []}',
                "description": "Both binary build and test jobs are disabled",
            },
            {
                "workflow": "linux-binary-manywheel",
                "job_name": "manywheel-py3_8-cuda11_8-test / test",
                "test_matrix": '{include: [{config: "default"}]}',
                "expected": '{"include": []}',
                "description": "Both binary build and test jobs are disabled",
            },
        ]

        for case in testcases:
            workflow = case["workflow"]
            job_name = case["job_name"]
            test_matrix = yaml.safe_load(case["test_matrix"])

            filtered_test_matrix = remove_disabled_jobs(workflow, job_name, test_matrix)
            self.assertEqual(case["expected"], json.dumps(filtered_test_matrix))

    @mock.patch("filter_test_configs.download_json")
    def test_mark_unstable_jobs(self, mock_download_json: Any) -> None:
        mock_download_json.return_value = MOCKED_DISABLED_UNSTABLE_JOBS

        testcases = [
            {
                "workflow": "pull",
                "job_name": "invalid job name",
                "test_matrix": '{include: [{config: "default"}]}',
                "expected": '{"include": [{"config": "default"}]}',
                "description": "invalid job name",
            },
            {
                "workflow": "pull",
                "job_name": "mock-platform-1 / build",
                "test_matrix": '{include: [{config: "default"}]}',
                "expected": '{"include": [{"config": "default", "unstable": "unstable"}]}',
                "description": "mark build and test jobs as unstable",
            },
            {
                "workflow": "trunk",
                "job_name": "mock-platform-2 / build",
                "test_matrix": '{include: [{config: "default"}]}',
                "expected": '{"include": [{"config": "default", "unstable": "unstable"}]}',
                "description": "mark build job as unstable",
            },
            {
                "workflow": "periodic",
                "job_name": "mock-platform-3 / test",
                "test_matrix": '{include: [{config: "default"}]}',
                "expected": '{"include": [{"config": "default", "unstable": "unstable"}]}',
                "description": "mark test job as unstable",
            },
            {
                "workflow": "pull",
                "job_name": "mock-platform-4 / build-and-test",
                "test_matrix": '{include: [{config: "default"}]}',
                "expected": '{"include": [{"config": "default", "unstable": "unstable"}]}',
                "description": "mark build-and-test job as unstable",
            },
            {
                "workflow": "trunk",
                "job_name": "mock-platform-5 / test",
                "test_matrix": '{include: [{config: "default"}, {config: "backward_compat"}]}',
                "expected": '{"include": [{"config": "default"}, {"config": "backward_compat", "unstable": "unstable"}]}',
                "description": "mark a test config as unstable",
            },
            {
                "workflow": "periodic",
                "job_name": "mock-platform-6 / build-and-test",
                "test_matrix": '{include: [{config: "default"}, {config: "backward_compat"}]}',
                "expected": '{"include": [{"config": "default", "unstable": "unstable"}, {"config": "backward_compat"}]}',
                "description": "mark a build-and-test config as unstable",
            },
            {
                "workflow": "pull",
                "job_name": "mock-platform-7 / test",
                "test_matrix": '{include: [{config: "default"}, {config: "backward_compat"}]}',
                "expected": '{"include": [{"config": "default"}, {"config": "backward_compat"}]}',
                "description": "include an invalid job name in the unstable issue",
            },
            {
                "workflow": "trunk",
                "job_name": "mock-platform-8 / build",
                "test_matrix": '{include: [{config: "default"}, {config: "backward_compat"}]}',
                "expected": '{"include": [{"config": "default"}, {"config": "backward_compat"}]}',
                "description": "include an invalid combination of build and test config",
            },
            {
                "workflow": "inductor",
                "job_name": "mock-platform-8 / build",
                "test_matrix": '{include: [{config: "default"}, {config: "backward_compat"}]}',
                "expected": '{"include": [{"config": "default"}, {"config": "backward_compat"}]}',
                "description": "not target this workflow",
            },
            {
                "workflow": "pull",
                "job_name": "mock-platform-9 / build",
                "test_matrix": '{include: [{config: "default"}, {config: "backward_compat"}]}',
                "expected": '{"include": [{"config": "default"}, {"config": "backward_compat"}]}',
                "description": "not target this platform",
            },
            {
                "workflow": "linux-binary-libtorch-cxx11-abi",
                "job_name": "libtorch-cpu-shared-with-deps-cxx11-abi-build / build",
                "test_matrix": '{include: [{config: "default"}]}',
                "expected": '{"include": [{"config": "default", "unstable": "unstable"}]}',
                "description": "Unstable binary build job",
            },
            {
                "workflow": "linux-binary-libtorch-cxx11-abi",
                "job_name": "libtorch-cpu-shared-with-deps-cxx11-abi-test / test",
                "test_matrix": '{include: [{config: "default"}]}',
                "expected": '{"include": [{"config": "default", "unstable": "unstable"}]}',
                "description": "Unstable binary test job",
            },
            {
                "workflow": "linux-binary-manywheel",
                "job_name": "manywheel-py3_8-cuda11_8-build / build",
                "test_matrix": '{include: [{config: "default"}]}',
                "expected": '{"include": [{"config": "default", "unstable": "unstable"}]}',
                "description": "Both binary build and test jobs are unstable",
            },
            {
                "workflow": "linux-binary-manywheel",
                "job_name": "manywheel-py3_8-cuda11_8-test / test",
                "test_matrix": '{include: [{config: "default"}]}',
                "expected": '{"include": [{"config": "default", "unstable": "unstable"}]}',
                "description": "Both binary build and test jobs are unstable",
            },
            {
                "workflow": "inductor",
                "job_name": "cuda12.1-py3.10-gcc9-sm86 / build",
                "test_matrix": """
                    { include: [
                        { config: "inductor" },
                        { config: "inductor_huggingface", shard: 1 },
                        { config: "inductor_huggingface", shard: 2 },
                        { config: "inductor_timm", shard: 1 },
                        { config: "inductor_timm", shard: 2 },
                        { config: "inductor_torchbench" },
                        { config: "inductor_huggingface_dynamic" },
                        { config: "inductor_torchbench_dynamic" },
                        { config: "inductor_distributed" },
                    ]}
                """,
                "expected": """
                    { "include": [
                        { "config": "inductor", "unstable": "unstable" },
                        { "config": "inductor_huggingface", "shard": 1, "unstable": "unstable" },
                        { "config": "inductor_huggingface", "shard": 2, "unstable": "unstable" },
                        { "config": "inductor_timm", "shard": 1 },
                        { "config": "inductor_timm", "shard": 2 },
                        { "config": "inductor_torchbench" },
                        { "config": "inductor_huggingface_dynamic", "unstable": "unstable" },
                        { "config": "inductor_torchbench_dynamic" },
                        { "config": "inductor_distributed" }
                    ]}
                """,
                "description": "Marking multiple unstable configurations",
            },
        ]

        for case in testcases:
            workflow = case["workflow"]
            job_name = case["job_name"]
            test_matrix = yaml.safe_load(case["test_matrix"])

            filtered_test_matrix = mark_unstable_jobs(workflow, job_name, test_matrix)
            self.assertEqual(json.loads(case["expected"]), filtered_test_matrix)

    @mock.patch("subprocess.check_output")
    def test_perform_misc_tasks(self, mocked_subprocess: Any) -> None:
        def _gen_expected_string(
            keep_going: bool = False,
            ci_verbose_test_logs: bool = False,
            ci_no_test_timeout: bool = False,
            ci_no_td: bool = False,
            ci_td_distributed: bool = False,
            is_unstable: bool = False,
            reenabled_issues: str = "",
        ) -> str:
            return (
                f"keep-going={keep_going}\n"
                f"ci-verbose-test-logs={ci_verbose_test_logs}\n"
                f"ci-no-test-timeout={ci_no_test_timeout}\n"
                f"ci-no-td={ci_no_td}\n"
                f"ci-td-distributed={ci_td_distributed}\n"
                f"is-unstable={is_unstable}\n"
                f"reenabled-issues={reenabled_issues}\n"
            )

        mocked_subprocess.return_value = b""
        testcases: List[Dict[str, Any]] = [
            {
                "labels": {},
                "test_matrix": '{include: [{config: "default"}]}',
                "job_name": "A job name",
                "expected": _gen_expected_string(),
                "description": "No keep-going, no is-unstable",
            },
            {
                "labels": {"keep-going"},
                "test_matrix": '{include: [{config: "default"}]}',
                "job_name": "A job name",
                "expected": _gen_expected_string(keep_going=True),
                "description": "Has keep-going, no is-unstable",
            },
            {
                "labels": {},
                "test_matrix": '{include: [{config: "default"}]}',
                "job_name": "A job name",
                "pr_body": "[keep-going]",
                "expected": _gen_expected_string(keep_going=True),
                "description": "Keep-going in PR body",
            },
            {
                "labels": {"ci-verbose-test-logs"},
                "test_matrix": '{include: [{config: "default"}]}',
                "job_name": "A job name",
                "pr_body": "[ci-no-test-timeout]",
                "expected": _gen_expected_string(
                    ci_verbose_test_logs=True, ci_no_test_timeout=True
                ),
                "description": "No pipe logs label and no test timeout in PR body",
            },
            {
                "labels": {"ci-no-test-timeout"},
                "test_matrix": '{include: [{config: "default"}]}',
                "job_name": "A job name",
                "pr_body": "[ci-verbose-test-logs]",
                "expected": _gen_expected_string(
                    ci_verbose_test_logs=True, ci_no_test_timeout=True
                ),
                "description": "No pipe logs in PR body and no test timeout in label (same as the above but swapped)",
            },
            {
                "labels": {"ci-no-td"},
                "test_matrix": '{include: [{config: "default"}]}',
                "job_name": "A job name",
                "pr_body": "",
                "expected": _gen_expected_string(ci_no_td=True),
                "description": "No pipe logs in PR body and no test timeout in label (same as the above but swapped)",
            },
            {
                "labels": {},
                "test_matrix": '{include: [{config: "default"}]}',
                "job_name": None,
                "expected": _gen_expected_string(),
                "description": "No job name",
            },
            {
                "labels": {},
                "test_matrix": '{include: [{config: "default"}]}',
                "job_name": "macos-12-py3-arm64 / test (default, 1, 3, macos-m1-stable, unstable)",
                "expected": _gen_expected_string(is_unstable=True),
                "description": "Unstable job",
            },
            {
                "labels": {},
                "test_matrix": '{include: [{config: "default"}]}',
                "job_name": "macos-12-py3-arm64 / test (default, 1, 3, macos-m1-stable, unstable)",
                "expected": _gen_expected_string(is_unstable=True),
                "description": "Unstable job",
            },
            {
                "labels": {},
                "test_matrix": '{include: [{config: "1", unstable: "unstable"}, {config: "2", unstable: "unstable"}]}',
                "job_name": "macos-12-py3-arm64 / build",
                "expected": _gen_expected_string(is_unstable=True),
                "description": "All configs are unstable",
            },
            {
                "labels": {},
                "test_matrix": '{include: [{config: "1", unstable: "unstable"}, {config: "2"}]}',
                "job_name": "macos-12-py3-arm64 / build",
                "expected": _gen_expected_string(is_unstable=False),
                "description": "Only mark some configs as unstable",
            },
            {
                "labels": {},
                "test_matrix": '{include: [{config: "default"}]}',
                "job_name": "A job name",
                "pr_body": "resolves #123 fixes #234",
                "expected": _gen_expected_string(reenabled_issues="123,234"),
                "description": "Reenable some issues",
            },
        ]

        for case in testcases:
            labels = case["labels"]
            test_matrix = yaml.safe_load(case["test_matrix"])
            job_name = case["job_name"]
            pr_body = case.get("pr_body", "")

            with tempfile.NamedTemporaryFile(delete=False) as fp:
                os.environ["GITHUB_OUTPUT"] = fp.name
                perform_misc_tasks(labels, test_matrix, job_name, pr_body)

            with open(str(os.getenv("GITHUB_OUTPUT"))) as fp:
                self.assertEqual(case["expected"], fp.read())

    # test variations of close in PR_BODY
    def test_parse_reenabled_issues(self) -> None:
        pr_body = "closes #123 Close #143 ClOsE #345 closed #10283"
        self.assertEqual(
            parse_reenabled_issues(pr_body), ["123", "143", "345", "10283"]
        )

        # test variations of fix
        pr_body = "fix #123 FixEd #143 fixes #345 FiXeD #10283"
        self.assertEqual(
            parse_reenabled_issues(pr_body), ["123", "143", "345", "10283"]
        )

        # test variations of resolve
        pr_body = "resolve #123 resolveS #143 REsolved #345 RESOLVES #10283"
        self.assertEqual(
            parse_reenabled_issues(pr_body), ["123", "143", "345", "10283"]
        )

        # test links
        pr_body = "closes https://github.com/pytorch/pytorch/issues/75198 fixes https://github.com/pytorch/pytorch/issues/75123"
        self.assertEqual(parse_reenabled_issues(pr_body), ["75198", "75123"])

        # test strange spacing
        pr_body = (
            "resolve #123,resolveS #143Resolved #345\nRESOLVES #10283 "
            "Fixed #2348fixes https://github.com/pytorch/pytorch/issues/75123resolveS #2134"
        )
        self.assertEqual(
            parse_reenabled_issues(pr_body),
            ["123", "143", "345", "10283", "2348", "75123", "2134"],
        )

        # test bad things
        pr_body = (
            "fixes189 fixeshttps://github.com/pytorch/pytorch/issues/75123 "
            "closedhttps://githubcom/pytorch/pytorch/issues/75123"
            "fix 234, fixes # 45, fixing #123, close 234, closes#45, closing #123 resolve 234, "
            "resolves  #45, resolving #123"
        )
        self.assertEqual(parse_reenabled_issues(pr_body), [])

        pr_body = None
        self.assertEqual(parse_reenabled_issues(pr_body), [])


if __name__ == "__main__":
    main()
