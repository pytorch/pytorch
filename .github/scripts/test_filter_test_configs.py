#!/usr/bin/env python3

import json
import os
from typing import Any
from unittest import main, mock, TestCase

import yaml
from filter_test_configs import (
    filter,
    get_labels,
    PREFIX,
    remove_disabled_jobs,
    set_periodic_modes,
    SUPPORTED_PERIODICAL_MODES,
    VALID_TEST_CONFIG_LABELS,
)


MOCKED_DISABLED_JOBS = {
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
}
MOCKED_LABELS = [{"name": "foo"}, {"name": "bar"}, {}, {"name": ""}]


class TestConfigFilter(TestCase):
    def setUp(self) -> None:
        os.environ["GITHUB_TOKEN"] = "GITHUB_TOKEN"
        if os.getenv("GITHUB_OUTPUT"):
            del os.environ["GITHUB_OUTPUT"]

    @mock.patch("filter_test_configs.download_json")
    def test_get_labels(self, mock_download_json: Any) -> None:
        mock_download_json.return_value = MOCKED_LABELS
        labels = get_labels(pr_number=12345)
        self.assertSetEqual({"foo", "bar"}, labels)

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
                "expected": '{"include": [{"config": "default", "runner": "linux"}]}',
                "description": "No match, keep the same test matrix",
            },
            {
                "test_matrix": '{include: [{config: "default", runner: "linux"}, {config: "plain-cfg"}]}',
                "expected": '{"include": [{"config": "default", "runner": "linux"}, {"config": "plain-cfg"}]}',
                "description": "No match because there is no prefix or suffix, keep the same test matrix",
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

    def test_filter_with_valid_label(self) -> None:
        mocked_labels = {f"{PREFIX}cfg", "ciflow/trunk"}
        VALID_TEST_CONFIG_LABELS.add(f"{PREFIX}cfg")

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

    def test_set_periodic_modes(self) -> None:
        testcases = [
            {
                "test_matrix": "{include: []}",
                "description": "Empty test matrix",
            },
            {
                "test_matrix": '{include: [{config: "default", runner: "linux"}, {config: "cfg", runner: "macos"}]}',
                "descripion": "Replicate each periodic mode in a different config",
            },
        ]

        for case in testcases:
            test_matrix = yaml.safe_load(case["test_matrix"])
            scheduled_test_matrix = set_periodic_modes(test_matrix)
            self.assertEqual(
                len(test_matrix["include"]) * len(SUPPORTED_PERIODICAL_MODES),
                len(scheduled_test_matrix["include"]),
            )

    @mock.patch("filter_test_configs.download_json")
    def test_remove_disabled_jobs(self, mock_download_json: Any) -> None:
        mock_download_json.return_value = MOCKED_DISABLED_JOBS

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
        ]

        for case in testcases:
            workflow = case["workflow"]
            job_name = case["job_name"]
            test_matrix = yaml.safe_load(case["test_matrix"])

            filtered_test_matrix = remove_disabled_jobs(workflow, job_name, test_matrix)
            self.assertEqual(case["expected"], json.dumps(filtered_test_matrix))


if __name__ == "__main__":
    main()
