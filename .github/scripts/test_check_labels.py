"""test_check_labels.py"""

from typing import Any, TYPE_CHECKING
from unittest import TestCase, mock, main

from check_labels import main as check_labels_main
from label_utils import LABEL_ERR_MSG
from test_trymerge import mock_gh_get_info

# TODO: this is a temp workaround to avoid circular dependencies,
#       and should be removed once GitHubPR is refactored out of trymerge script.
if TYPE_CHECKING:
    from trymerge import GitHubPR

def mock_parse_args() -> object:
    class Object(object):
        def __init__(self) -> None:
            self.pr_num = 76123
    return Object()

def mock_add_label_err_comment(pr: "GitHubPR") -> None:
    pass

def mock_delete_all_label_err_comments(pr: "GitHubPR") -> None:
    pass


class TestCheckLabels(TestCase):
    @mock.patch('trymerge.gh_get_pr_info', return_value=mock_gh_get_info())
    @mock.patch('check_labels.parse_args', return_value=mock_parse_args())
    @mock.patch('check_labels.has_required_labels', return_value=False)
    @mock.patch('check_labels.delete_all_label_err_comments', side_effect=mock_delete_all_label_err_comments)
    @mock.patch('check_labels.add_label_err_comment', side_effect=mock_add_label_err_comment)
    def test_ci_fails_without_required_labels(
        self,
        mock_add_label_err_comment: Any,
        mock_delete_all_label_err_comments: Any,
        mock_has_required_labels: Any,
        mock_parse_args: Any,
        mock_gh_get_info: Any,
    ) -> None:
        with self.assertRaises(SystemExit) as err:
            check_labels_main()
            self.assertEqual(err.exception, LABEL_ERR_MSG)
            mock_add_label_err_comment.assert_called_once()
            mock_delete_all_label_err_comments.assert_not_called()

    @mock.patch('trymerge.gh_get_pr_info', return_value=mock_gh_get_info())
    @mock.patch('check_labels.parse_args', return_value=mock_parse_args())
    @mock.patch('check_labels.has_required_labels', return_value=True)
    @mock.patch('check_labels.delete_all_label_err_comments', side_effect=mock_delete_all_label_err_comments)
    @mock.patch('check_labels.add_label_err_comment', side_effect=mock_add_label_err_comment)
    def test_ci_success_with_required_labels(
        self,
        mock_add_label_err_comment: Any,
        mock_delete_all_label_err_comments: Any,
        mock_has_required_labels: Any,
        mock_parse_args: Any,
        mock_gh_get_info: Any,
    ) -> None:
        check_labels_main()
        mock_add_label_err_comment.assert_not_called()
        mock_delete_all_label_err_comments.assert_called_once()

if __name__ == "__main__":
    main()
