"""test_check_labels.py"""

from typing import Any
from unittest import TestCase, mock, main

from trymerge import GitHubPR
from test_trymerge import mocked_gh_graphql
from check_labels import has_required_labels

release_notes_labels = [
    "release notes: AO frontend",
    "release notes: autograd",
    "release notes: benchmark",
    "release notes: build",
    "release notes: complex",
    "release notes: composability",
    "release notes: cpp",
    "release notes: cuda",
    "release notes: cudnn",
    "release notes: dataloader",
    "release notes: distributed (c10d)",
    "release notes: distributed (ddp)",
    "release notes: distributed (fsdp)",
    "release notes: distributed (pipeline)",
    "release notes: distributed (rpc)",
    "release notes: distributed (sharded)",
    "release notes: foreach_frontend",
    "release notes: functorch",
    "release notes: fx",
    "release notes: hub",
    "release notes: jit",
    "release notes: lazy",
    "release notes: linalg_frontend",
    "release notes: memory format",
    "release notes: Meta API",
    "release notes: mobile",
    "release notes: mps",
    "release notes: nested tensor",
    "release notes: nn",
    "release notes: onnx",
    "release notes: package/deploy",
    "release notes: performance_as_product",
    "release notes: profiler",
    "release notes: python_frontend",
    "release notes: quantization",
    "release notes: releng",
    "release notes: rocm",
    "release notes: sparse",
    "release notes: visualization",
    "release notes: vulkan",
]


class TestCheckLabels(TestCase):
    @mock.patch('trymerge.gh_graphql', side_effect=mocked_gh_graphql)
    @mock.patch('check_labels.get_release_notes_labels', return_value=release_notes_labels)
    def test_pr_with_missing_labels(self, mocked_rn_labels: Any, mocked_gql: Any) -> None:
        "Test PR with no 'release notes:' label or 'topic: not user facing' label"
        pr = GitHubPR("pytorch", "pytorch", 82169)
        self.assertFalse(has_required_labels(pr))

    @mock.patch('trymerge.gh_graphql', side_effect=mocked_gh_graphql)
    @mock.patch('check_labels.get_release_notes_labels', return_value=release_notes_labels)
    def test_pr_with_release_notes_label(self, mocked_rn_labels: Any, mocked_gql: Any) -> None:
        "Test PR with 'release notes: nn' label"
        pr = GitHubPR("pytorch", "pytorch", 71759)
        self.assertTrue(has_required_labels(pr))

    @mock.patch('trymerge.gh_graphql', side_effect=mocked_gh_graphql)
    @mock.patch('check_labels.get_release_notes_labels', return_value=release_notes_labels)
    def test_pr_with_not_user_facing_label(self, mocked_rn_labels: Any, mocked_gql: Any) -> None:
        "Test PR with 'topic: not user facing' label"
        pr = GitHubPR("pytorch", "pytorch", 75095)
        self.assertTrue(has_required_labels(pr))

if __name__ == "__main__":
    main()
