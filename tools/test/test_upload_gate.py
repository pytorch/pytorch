import unittest

from tools.stats.upload_test_stats import should_upload_full_test_run


class TestUploadGate(unittest.TestCase):
    def test_main_branch_on_pytorch_repo(self) -> None:
        self.assertTrue(should_upload_full_test_run("main", "pytorch/pytorch"))

    def test_trunk_tag_valid_sha_on_pytorch_repo(self) -> None:
        sha = "a" * 40
        self.assertTrue(should_upload_full_test_run(f"trunk/{sha}", "pytorch/pytorch"))

    def test_trunk_tag_invalid_sha_on_pytorch_repo(self) -> None:
        # Not 40 hex chars
        self.assertFalse(should_upload_full_test_run("trunk/12345", "pytorch/pytorch"))

    def test_non_main_branch_on_pytorch_repo(self) -> None:
        self.assertFalse(
            should_upload_full_test_run("feature-branch", "pytorch/pytorch")
        )

    def test_main_branch_on_fork_repo(self) -> None:
        self.assertFalse(should_upload_full_test_run("main", "someone/fork"))


if __name__ == "__main__":
    unittest.main()
