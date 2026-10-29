#!/usr/bin/env python3
from pathlib import Path
from unittest import main, SkipTest, TestCase
from unittest.mock import call, MagicMock, patch

from gitutils import (
    _shasum,
    are_ghstack_branches_in_sync,
    GitRepo,
    patterns_to_regex,
    PeekableIterator,
    retries_decorator,
)


BASE_DIR = Path(__file__).parent


class TestPeekableIterator(TestCase):
    def test_iterator(self, input_: str = "abcdef") -> None:
        iter_ = PeekableIterator(input_)
        for idx, c in enumerate(iter_):
            self.assertEqual(c, input_[idx])

    def test_is_iterable(self) -> None:
        from collections.abc import Iterator

        iter_ = PeekableIterator("")
        self.assertTrue(isinstance(iter_, Iterator))

    def test_peek(self, input_: str = "abcdef") -> None:
        iter_ = PeekableIterator(input_)
        for idx, _c in enumerate(iter_):
            if idx + 1 < len(input_):
                self.assertEqual(iter_.peek(), input_[idx + 1])
            else:
                self.assertTrue(iter_.peek() is None)


class TestPattern(TestCase):
    def test_double_asterisks(self) -> None:
        allowed_patterns = [
            "aten/src/ATen/native/**LinearAlgebra*",
        ]
        patterns_re = patterns_to_regex(allowed_patterns)
        fnames = [
            "aten/src/ATen/native/LinearAlgebra.cpp",
            "aten/src/ATen/native/cpu/LinearAlgebraKernel.cpp",
        ]
        for filename in fnames:
            self.assertTrue(patterns_re.match(filename))


class TestRetriesDecorator(TestCase):
    def test_simple(self) -> None:
        @retries_decorator()
        def foo(x: int, y: int) -> int:
            return x + y

        self.assertEqual(foo(3, 4), 7)

    def test_fails(self) -> None:
        @retries_decorator(rc=0)
        def foo(x: int, y: int) -> int:
            return x + y

        self.assertEqual(foo("a", 4), 0)


class TestGitRepo(TestCase):
    def setUp(self) -> None:
        repo_dir = BASE_DIR.absolute().parent.parent
        if not (repo_dir / ".git").is_dir():
            raise SkipTest(
                "Can't find git directory, make sure to run this test on real repo checkout"
            )
        self.repo = GitRepo(str(repo_dir))

    def _skip_if_ref_does_not_exist(self, ref: str) -> None:
        """Skip test if ref is missing as stale branches are deleted with time"""
        try:
            self.repo.show_ref(ref)
        except RuntimeError as e:
            raise SkipTest(f"Can't find head ref {ref} due to {str(e)}") from e

    def test_compute_diff(self) -> None:
        diff = self.repo.diff("HEAD")
        sha = _shasum(diff)
        self.assertEqual(len(sha), 64)

    def test_ghstack_branches_in_sync(self) -> None:
        head_ref = "gh/SS-JIA/206/head"
        self._skip_if_ref_does_not_exist(head_ref)
        self.assertTrue(are_ghstack_branches_in_sync(self.repo, head_ref))

    def test_ghstack_branches_not_in_sync(self) -> None:
        head_ref = "gh/clee2000/1/head"
        self._skip_if_ref_does_not_exist(head_ref)
        self.assertFalse(are_ghstack_branches_in_sync(self.repo, head_ref))


@patch.object(GitRepo, "_run_git")
class TestGitRepoPush(TestCase):
    def setUp(self) -> None:
        self.repo = GitRepo("/nonexistent")

    def test_push_once_on_success(self, mock_run_git: MagicMock) -> None:
        self.repo.push("main", dry_run=False)
        mock_run_git.assert_called_once_with("push", "origin", "main")

    def test_push_dry_run(self, mock_run_git: MagicMock) -> None:
        self.repo.push("main", dry_run=True)
        mock_run_git.assert_called_once_with("push", "--dry-run", "origin", "main")

    def test_push_rebases_before_retry(self, mock_run_git: MagicMock) -> None:
        mock_run_git.side_effect = [RuntimeError("rejected"), "", "", ""]
        self.repo.push("main", dry_run=False)
        self.assertEqual(
            mock_run_git.call_args_list,
            [
                call("push", "origin", "main"),
                call("fetch", "origin"),
                call("rebase", "origin/main"),
                call("push", "origin", "main"),
            ],
        )

    def test_push_raises_after_last_attempt(self, mock_run_git: MagicMock) -> None:
        errors = [RuntimeError(f"attempt {i} rejected") for i in range(3)]
        mock_run_git.side_effect = [errors[0], "", "", errors[1], "", "", errors[2]]
        with self.assertRaises(RuntimeError) as ctx:
            self.repo.push("main", dry_run=False, retry=3)
        self.assertIs(ctx.exception, errors[2])
        self.assertEqual(mock_run_git.call_count, 7)


if __name__ == "__main__":
    main()
