#!/usr/bin/env python3
from gitutils import PeekableIterator
from unittest import TestCase, main

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



if __name__ == '__main__':
    main()
