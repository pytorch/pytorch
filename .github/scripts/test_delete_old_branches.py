import os
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch


os.environ["GITHUB_TOKEN"] = "test_token"

from delete_old_branches import (
    delete_old_tags,
    get_branches_with_magic_label_or_open_pr,
)


@patch("delete_old_branches.delete_branch")
@patch("gitutils.GitRepo._run_git")
class TestDeleteTag(unittest.TestCase):
    def test_delete_tag(
        self, mock_run_git: "MagicMock", mock_delete_tag: "MagicMock"
    ) -> None:
        for tag in [
            "ciflow/branch/12345",
            "ciflow/commitsha/1234567890abcdef1234567890abcdef12345678",
            "trunk/1234567890abcdef1234567890abcdef12345678",
        ]:
            mock_run_git.side_effect = [
                tag,
                str(int(datetime.now().timestamp() - 8 * 24 * 60 * 60)),  # 8 days ago
            ]
            delete_old_tags()
            mock_delete_tag.assert_called_once()
            mock_delete_tag.reset_mock()

            # Don't delete if the tag is not old enough
            mock_run_git.side_effect = [
                tag,
                str(int(datetime.now().timestamp() - 6 * 24 * 60 * 60)),  # 6 days ago
            ]
            delete_old_tags()
            mock_delete_tag.assert_not_called()

    def test_do_not_delete_tag(
        self, mock_run_git: "MagicMock", mock_delete_tag: "MagicMock"
    ) -> None:
        for tag in [
            "ciflow/doesntseemtomatch",
            "trunk/doesntseemtomatch",
            "doesntseemtomatch",
        ]:
            mock_run_git.side_effect = [
                tag,
                str(int(datetime.now().timestamp() - 8 * 24 * 60 * 60)),  # 8 days ago
            ]
            delete_old_tags()
            mock_delete_tag.assert_not_called()


class TestKeepBranches(unittest.TestCase):
    @patch("delete_old_branches.get_open_prs")
    @patch("delete_old_branches.paginate_graphql")
    def test_keeps_open_pr_heads_and_bases(
        self, mock_paginate: "MagicMock", mock_open_prs: "MagicMock"
    ) -> None:
        mock_paginate.return_value = [
            {"headRefName": "labeled", "baseRefName": "closed-base"},
        ]
        mock_open_prs.return_value = [
            {"headRefName": "alice/feat-2", "baseRefName": "alice/feat-1"},
            {"headRefName": "gh/bob/2/head", "baseRefName": "gh/bob/1/head"},
            {"headRefName": "alice/feat-1-fix", "baseRefName": "main"},
        ]
        self.assertEqual(
            get_branches_with_magic_label_or_open_pr(),
            {
                "labeled",
                "alice/feat-2",
                "alice/feat-1",
                "gh/bob/2",
                "gh/bob/1",
                "alice/feat-1-fix",
                "main",
            },
        )


if __name__ == "__main__":
    unittest.main()
