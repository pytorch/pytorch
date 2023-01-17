from typing import Any
from unittest import TestCase, mock, main

from check_labels import main as check_labels_main
from test_trymerge import (
    mock_gh_get_info,
    mock_add_label_err_comment,
    mock_delete_all_label_err_comments,
)

def mock_parse_args() -> object:
    class Object(object):
        def __init__(self) -> None:
            self.pr_num = 76123

    return Object()

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
        check_labels_main()
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
