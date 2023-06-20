#!/usr/bin/env python3

import json
import os
import tempfile
from typing import Any, Dict, List
from unittest import main, mock, TestCase

import yaml
from filter_test_configs import (
    filter,
    get_labels,
    mark_unstable_jobs,
    perform_misc_tasks,
    PREFIX,
    remove_disabled_jobs,
    set_periodic_modes,
    SUPPORTED_PERIODICAL_MODES,
    VALID_TEST_CONFIG_LABELS,
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
}
MOCKED_LABELS = [{"name": "foo"}, {"name": "bar"}, {}, {"name": ""}]


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
        ]

        for case in testcases:
            workflow = case["workflow"]
            job_name = case["job_name"]
            test_matrix = yaml.safe_load(case["test_matrix"])

            filtered_test_matrix = mark_unstable_jobs(workflow, job_name, test_matrix)
            self.assertEqual(case["expected"], json.dumps(filtered_test_matrix))

    def test_perform_misc_tasks(self) -> None:
        testcases: List[Dict[str, Any]] = [
            {
                "labels": {},
                "test_matrix": '{include: [{config: "default"}]}',
                "job_name": "A job name",
                "expected": "keep-going=False\nis-unstable=False\n",
                "description": "No keep-going, no is-unstable",
            },
            {
                "labels": {"keep-going"},
                "test_matrix": '{include: [{config: "default"}]}',
                "job_name": "A job name",
                "expected": "keep-going=True\nis-unstable=False\n",
                "description": "Has keep-going, no is-unstable",
            },
            {
                "labels": {},
                "test_matrix": '{include: [{config: "default"}]}',
                "job_name": None,
                "expected": "keep-going=False\nis-unstable=False\n",
                "description": "No job name",
            },
            {
                "labels": {},
                "test_matrix": '{include: [{config: "default"}]}',
                "job_name": "macos-12-py3-arm64 / test (default, 1, 3, macos-m1-12, unstable)",
                "expected": "keep-going=False\nis-unstable=True\n",
                "description": "Unstable job",
            },
            {
                "labels": {},
                "test_matrix": '{include: [{config: "default"}]}',
                "job_name": "macos-12-py3-arm64 / test (default, 1, 3, macos-m1-12, unstable)",
                "expected": "keep-going=False\nis-unstable=True\n",
                "description": "Unstable job",
            },
            {
                "labels": {},
                "test_matrix": '{include: [{config: "1", unstable: "unstable"}, {config: "2", unstable: "unstable"}]}',
                "job_name": "macos-12-py3-arm64 / build",
                "expected": "keep-going=False\nis-unstable=True\n",
                "description": "All configs are unstable",
            },
            {
                "labels": {},
                "test_matrix": '{include: [{config: "1", unstable: "unstable"}, {config: "2"}]}',
                "job_name": "macos-12-py3-arm64 / build",
                "expected": "keep-going=False\nis-unstable=False\n",
                "description": "Only mark some configs as unstable",
            },
        ]

        for case in testcases:
            labels = case["labels"]
            test_matrix = yaml.safe_load(case["test_matrix"])
            job_name = case["job_name"]

            with tempfile.NamedTemporaryFile(delete=False) as fp:
                os.environ["GITHUB_OUTPUT"] = fp.name
                perform_misc_tasks(labels, test_matrix, job_name)

            with open(str(os.getenv("GITHUB_OUTPUT"))) as fp:
                self.assertEqual(case["expected"], fp.read())


if __name__ == "__main__":
    main()
