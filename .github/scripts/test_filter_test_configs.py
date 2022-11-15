#!/usr/bin/env python3

import os
import yaml
import json
from unittest import TestCase, main, mock
from filter_test_configs import (
    get_labels,
    filter,
    set_periodic_modes,
    PREFIX,
    VALID_TEST_CONFIG_LABELS,
    SUPPORTED_PERIODICAL_MODES
)
import requests
from requests.models import Response
from typing import Any, Dict


def mocked_gh_get_labels_failed(url: str, headers: Dict[str, str]) -> Response:
    mocked_response = Response()
    mocked_response.status_code = requests.codes.bad_request
    return mocked_response


def mocked_gh_get_labels(url: str, headers: Dict[str, str]) -> Response:
    mocked_response = Response()
    mocked_response.status_code = requests.codes.ok
    mocked_response._content = b'[{"name": "foo"}, {"name": "bar"}, {}, {"name": ""}]'
    return mocked_response


class TestConfigFilter(TestCase):

    def setUp(self) -> None:
        os.environ["GITHUB_TOKEN"] = "GITHUB_TOKEN"
        if os.getenv("GITHUB_OUTPUT"):
            del os.environ["GITHUB_OUTPUT"]

    @mock.patch("filter_test_configs.requests.get", side_effect=mocked_gh_get_labels)
    def test_get_labels(self, mocked_gh: Any) -> None:
        labels = get_labels(pr_number=12345)
        self.assertSetEqual({"foo", "bar"}, labels)

    @mock.patch("filter_test_configs.requests.get", side_effect=mocked_gh_get_labels_failed)
    def test_get_labels_failed(self, mocked_gh: Any) -> None:
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
            filtered_test_matrix = filter(yaml.safe_load(case["test_matrix"]), mocked_labels)
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
            filtered_test_matrix = filter(yaml.safe_load(case["test_matrix"]), mocked_labels)
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
                len(test_matrix["include"]) * (len(SUPPORTED_PERIODICAL_MODES) + 1),
                len(scheduled_test_matrix["include"])
            )


if __name__ == '__main__':
    main()
