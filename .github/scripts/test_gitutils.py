#!/usr/bin/env python3
from typing import Any
from gitutils import (
    PeekableIterator,
    patterns_to_regex,
    GitRepo,
    are_ghstack_branches_in_sync,
    _shasum,
    request_for_labels,
    get_last_page,
    gh_get_labels,
)
from pathlib import Path
from unittest import TestCase, main, mock, SkipTest

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
        for idx, c in enumerate(iter_):
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
            "aten/src/ATen/native/cpu/LinearAlgebraKernel.cpp"]
        for filename in fnames:
            self.assertTrue(patterns_re.match(filename))


class TestGitRepo(TestCase):
    def setUp(self) -> None:
        repo_dir = BASE_DIR.parent.parent.absolute()
        if not (repo_dir / ".git").is_dir():
            raise SkipTest("Can't find git directory, make sure to run this test on real repo checkout")
        self.repo = GitRepo(str(repo_dir))

    def _skip_if_ref_does_not_exist(self, ref: str) -> None:
        """ Skip test if ref is missing as stale branches are deleted with time """
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
class TestLabels(TestCase):
    MOCK_URL = "https://test.github.com/labels?"
    MOCK_HEADERS = [
        ({"link": "https://abc&page=123>;"}, 123),
        ({"link": "https://abc&page=1>;&page=2>;&page=3>;"}, 3),
    ]
    MOCK_INFO = "{"

    @mock.patch("gitutils.Request")
    @mock.patch("gitutils._read_url")
    def test_request_for_labels(self, mock__read_url: Any, mock_request: Any) -> None:
        request_for_labels(self.MOCK_URL)
        mock__read_url.assert_called_once()
        mock_request.assert_called_once_with(
            self.MOCK_URL,
            headers={'Accept': 'application/vnd.github.v3+json'}
        )

    def test_get_last_page(self) -> None:
        for header, expected_page_num in self.MOCK_HEADERS:
            self.assertEqual(get_last_page(header), expected_page_num)

    @mock.patch("gitutils.get_last_page", return_value=3)
    @mock.patch("gitutils.request_for_labels", return_value=("foo", "bar"))
    @mock.patch("gitutils.update_labels")
    def test_gh_get_labels(
        self, mock_update_labels: Any, mock_request_for_labels: Any, mock_get_last_page: Any
    ) -> None:
        gh_get_labels("foo", "bar")
        mock_get_last_page.assert_called_once()
        self.assertEqual(mock_update_labels.call_count, 3)
        self.assertEqual(mock_request_for_labels.call_count, 3)

    @mock.patch("gitutils.get_last_page", return_value=0)
    @mock.patch("gitutils.request_for_labels", return_value=("foo", "bar"))
    @mock.patch("gitutils.update_labels")
    def test_gh_get_labels_raises_with_no_pages(
        self, mock_update_labels: Any, mock_request_for_labels: Any, mock_get_last_page: Any
    ) -> None:
        with self.assertRaises(AssertionError):
            gh_get_labels("foo", "bar")
            mock_get_last_page.assert_called_once()
            mock_update_labels.assert_called_once()
            mock_request_for_labels.call_count.assert_called_once()


if __name__ == '__main__':
    main()
