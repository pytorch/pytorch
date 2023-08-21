import tempfile
import unittest

from commitlist import CommitList


class TestCommitList(unittest.TestCase):
    def test_create_new(self):
        with tempfile.TemporaryDirectory() as tempdir:
            commit_list_path = f"{tempdir}/commitlist.csv"
            commit_list = CommitList.create_new(
                commit_list_path, "v1.5.0", "6000dca5df"
            )
            self.assertEqual(len(commit_list.commits), 33)
            self.assertEqual(commit_list.commits[0].commit_hash, "7335f079ab")
            self.assertTrue(
                commit_list.commits[0].title.startswith("[pt][quant] qmul and qadd")
            )
            self.assertEqual(commit_list.commits[-1].commit_hash, "6000dca5df")
            self.assertTrue(
                commit_list.commits[-1].title.startswith(
                    "[nomnigraph] Copy device option when customize "
                )
            )

    def test_read_write(self):
        with tempfile.TemporaryDirectory() as tempdir:
            commit_list_path = f"{tempdir}/commitlist.csv"
            initial = CommitList.create_new(commit_list_path, "v1.5.0", "7543e7e558")
            initial.write_to_disk()

            expected = CommitList.from_existing(commit_list_path)
            expected.commits[-2].category = "foobar"
            expected.write_to_disk()

            commit_list = CommitList.from_existing(commit_list_path)
            for commit, expected in zip(commit_list.commits, expected.commits):
                self.assertEqual(commit, expected)

    def test_update_to(self):
        with tempfile.TemporaryDirectory() as tempdir:
            commit_list_path = f"{tempdir}/commitlist.csv"
            initial = CommitList.create_new(commit_list_path, "v1.5.0", "7543e7e558")
            initial.commits[-2].category = "foobar"
            self.assertEqual(len(initial.commits), 2143)
            initial.write_to_disk()

            commit_list = CommitList.from_existing(commit_list_path)
            commit_list.update_to("5702a28b26")
            self.assertEqual(len(commit_list.commits), 2143 + 4)
            self.assertEqual(commit_list.commits[-5], initial.commits[-1])


if __name__ == "__main__":
    unittest.main()
