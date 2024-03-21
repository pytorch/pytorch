from typing import Any
from unittest import main, mock, TestCase

from label_utils import (
    get_last_page_num_from_header,
    gh_get_labels,
    has_required_labels,
)
from test_trymerge import mocked_gh_graphql
from trymerge import GitHubPR


release_notes_labels = [
    "release notes: nn",
]


class TestLabelUtils(TestCase):
    MOCK_HEADER_LINKS_TO_PAGE_NUMS = {
        1: {
            "link": "<https://api.github.com/dummy/labels?per_page=10&page=1>; rel='last'"
        },
        2: {"link": "<https://api.github.com/dummy/labels?per_page=1&page=2>;"},
        3: {"link": "<https://api.github.com/dummy/labels?per_page=1&page=2&page=3>;"},
    }

    def test_get_last_page_num_from_header(self) -> None:
        for (
            expected_page_num,
            mock_header,
        ) in self.MOCK_HEADER_LINKS_TO_PAGE_NUMS.items():
            self.assertEqual(
                get_last_page_num_from_header(mock_header), expected_page_num
            )

    MOCK_LABEL_INFO = '[{"name": "foo"}]'

    @mock.patch("label_utils.get_last_page_num_from_header", return_value=3)
    @mock.patch("label_utils.request_for_labels", return_value=(None, MOCK_LABEL_INFO))
    def test_gh_get_labels(
        self,
        mock_request_for_labels: Any,
        mock_get_last_page_num_from_header: Any,
    ) -> None:
        res = gh_get_labels("mock_org", "mock_repo")
        mock_get_last_page_num_from_header.assert_called_once()
        self.assertEqual(res, ["foo"] * 3)

    @mock.patch("label_utils.get_last_page_num_from_header", return_value=0)
    @mock.patch("label_utils.request_for_labels", return_value=(None, MOCK_LABEL_INFO))
    def test_gh_get_labels_raises_with_no_pages(
        self,
        mock_request_for_labels: Any,
        get_last_page_num_from_header: Any,
    ) -> None:
        with self.assertRaises(AssertionError) as err:
            gh_get_labels("foo", "bar")
        self.assertIn("number of pages of labels", str(err.exception))

    @mock.patch("trymerge.gh_graphql", side_effect=mocked_gh_graphql)
    @mock.patch(
        "label_utils.get_release_notes_labels", return_value=release_notes_labels
    )
    def test_pr_with_missing_labels(
        self, mocked_rn_labels: Any, mocked_gql: Any
    ) -> None:
        "Test PR with no 'release notes:' label or 'topic: not user facing' label"
        pr = GitHubPR("pytorch", "pytorch", 82169)
        self.assertFalse(has_required_labels(pr))

    @mock.patch("trymerge.gh_graphql", side_effect=mocked_gh_graphql)
    @mock.patch(
        "label_utils.get_release_notes_labels", return_value=release_notes_labels
    )
    def test_pr_with_release_notes_label(
        self, mocked_rn_labels: Any, mocked_gql: Any
    ) -> None:
        "Test PR with 'release notes: nn' label"
        pr = GitHubPR("pytorch", "pytorch", 71759)
        self.assertTrue(has_required_labels(pr))

    @mock.patch("trymerge.gh_graphql", side_effect=mocked_gh_graphql)
    @mock.patch(
        "label_utils.get_release_notes_labels", return_value=release_notes_labels
    )
    def test_pr_with_not_user_facing_label(
        self, mocked_rn_labels: Any, mocked_gql: Any
    ) -> None:
        "Test PR with 'topic: not user facing' label"
        pr = GitHubPR("pytorch", "pytorch", 75095)
        self.assertTrue(has_required_labels(pr))


if __name__ == "__main__":
    main()
