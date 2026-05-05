#!/usr/bin/env python3

from unittest import main, TestCase

from match_reviewers import is_bot, is_wildcard_only, match_reviewers


SAMPLE_RULES = [
    {
        "name": "ONNX exporter",
        "patterns": ["torch/onnx/**", "test/onnx/**"],
        "approved_by": ["alice", "bob"],
    },
    {
        "name": "Distributed",
        "patterns": ["torch/distributed/**", "test/distributed/**"],
        "approved_by": ["carol", "pytorch/dist-team"],
    },
    {
        "name": "CI bot rule",
        "patterns": [".ci/docker/ci_commit_pins/triton.txt"],
        "approved_by": ["pytorchbot"],
    },
    {
        "name": "superuser",
        "patterns": ["*"],
        "approved_by": ["superadmin"],
    },
    {
        "name": "Core Maintainers",
        "patterns": ["**"],
        "approved_by": ["maintainer1"],
    },
    {
        "name": "ROCm",
        "patterns": ["**rocm**", "**hip**"],
        "approved_by": ["rocm-dev", "facebook-github-bot"],
    },
]


class TestIsWildcardOnly(TestCase):
    def test_single_star(self) -> None:
        self.assertTrue(is_wildcard_only(["*"]))

    def test_double_star(self) -> None:
        self.assertTrue(is_wildcard_only(["**"]))

    def test_mixed_wildcards(self) -> None:
        self.assertTrue(is_wildcard_only(["*", "**"]))

    def test_real_pattern(self) -> None:
        self.assertFalse(is_wildcard_only(["torch/onnx/**"]))

    def test_empty(self) -> None:
        # Empty pattern list matches nothing — not a superuser/wildcard rule.
        self.assertFalse(is_wildcard_only([]))


class TestIsBot(TestCase):
    def test_pytorchbot(self) -> None:
        self.assertTrue(is_bot("pytorchbot"))

    def test_facebook_github_bot(self) -> None:
        self.assertTrue(is_bot("facebook-github-bot"))

    def test_pytorchmergebot(self) -> None:
        self.assertTrue(is_bot("pytorchmergebot"))

    def test_regular_user(self) -> None:
        self.assertFalse(is_bot("alice"))

    def test_bot_in_middle(self) -> None:
        self.assertFalse(is_bot("bottleneck-dev"))

    def test_not_a_bot_suffix(self) -> None:
        self.assertFalse(is_bot("abbot"))


class TestMatchReviewers(TestCase):
    def test_onnx_files(self) -> None:
        reviewers, teams = match_reviewers(
            SAMPLE_RULES,
            ["torch/onnx/export.py", "torch/onnx/utils.py"],
            "someone",
        )
        self.assertEqual(reviewers, ["alice", "bob"])
        self.assertEqual(teams, [])

    def test_distributed_files_with_team(self) -> None:
        reviewers, teams = match_reviewers(
            SAMPLE_RULES,
            ["torch/distributed/rpc/api.py"],
            "someone",
        )
        self.assertEqual(reviewers, ["carol"])
        self.assertEqual(teams, ["dist-team"])

    def test_skips_wildcard_rules(self) -> None:
        reviewers, teams = match_reviewers(
            SAMPLE_RULES,
            ["torch/onnx/export.py"],
            "someone",
        )
        self.assertNotIn("superadmin", reviewers)
        self.assertNotIn("maintainer1", reviewers)

    def test_skips_bots(self) -> None:
        reviewers, teams = match_reviewers(
            SAMPLE_RULES,
            [".ci/docker/ci_commit_pins/triton.txt"],
            "someone",
        )
        self.assertNotIn("pytorchbot", reviewers)
        self.assertEqual(reviewers, [])

    def test_skips_pr_author(self) -> None:
        reviewers, teams = match_reviewers(
            SAMPLE_RULES,
            ["torch/onnx/export.py"],
            "Alice",  # case-insensitive
        )
        self.assertNotIn("alice", reviewers)
        self.assertEqual(reviewers, ["bob"])

    def test_no_matches(self) -> None:
        reviewers, teams = match_reviewers(
            SAMPLE_RULES,
            ["some/unknown/file.py"],
            "someone",
        )
        self.assertEqual(reviewers, [])
        self.assertEqual(teams, [])

    def test_multiple_rules_match(self) -> None:
        reviewers, teams = match_reviewers(
            SAMPLE_RULES,
            ["torch/onnx/export.py", "torch/distributed/rpc/api.py"],
            "someone",
        )
        self.assertEqual(reviewers, ["alice", "bob", "carol"])
        self.assertEqual(teams, ["dist-team"])

    def test_rocm_pattern_with_bot_filtering(self) -> None:
        reviewers, teams = match_reviewers(
            SAMPLE_RULES,
            ["aten/src/ATen/hip/impl.cpp"],
            "someone",
        )
        self.assertEqual(reviewers, ["rocm-dev"])
        # facebook-github-bot should be filtered out
        self.assertEqual(teams, [])

    def test_deduplication(self) -> None:
        rules = [
            {
                "name": "Rule A",
                "patterns": ["torch/foo/**"],
                "approved_by": ["alice"],
            },
            {
                "name": "Rule B",
                "patterns": ["torch/**"],
                "approved_by": ["alice"],
            },
        ]
        reviewers, teams = match_reviewers(rules, ["torch/foo/bar.py"], "someone")
        self.assertEqual(reviewers, ["alice"])


if __name__ == "__main__":
    main()
