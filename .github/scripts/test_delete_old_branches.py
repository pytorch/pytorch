import os
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch


os.environ["GITHUB_TOKEN"] = "test_token"

from delete_old_branches import delete_old_tags


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


if __name__ == "__main__":
    unittest.main()
