from tools import trailing_newlines
import unittest
import tempfile


def correct_trailing_newlines(file_contents: str) -> bool:
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
        filename = tmp.name
        tmp.write(file_contents)
    return trailing_newlines.correct_trailing_newlines(filename)


class TestTrailingNewlines(unittest.TestCase):
    def test_empty(self) -> None:
        self.assertTrue(correct_trailing_newlines(''))

    def test_single_byte(self) -> None:
        self.assertFalse(correct_trailing_newlines('a'))

    def test_single_newline(self) -> None:
        self.assertFalse(correct_trailing_newlines('\n'))

    def test_two_newlines(self) -> None:
        self.assertFalse(correct_trailing_newlines('\n\n'))

    def test_three_newlines(self) -> None:
        self.assertFalse(correct_trailing_newlines('\n\n\n'))

    def test_hello_world(self) -> None:
        self.assertFalse(correct_trailing_newlines('hello world'))

    def test_hello_world_newline(self) -> None:
        self.assertTrue(correct_trailing_newlines('hello world\n'))

    def test_hello_world_two_newlines(self) -> None:
        self.assertFalse(correct_trailing_newlines('hello world\n\n'))

    def test_hello_world_three_newlines(self) -> None:
        self.assertFalse(correct_trailing_newlines('hello world\n\n\n'))

    def test_hello_world_multiline(self) -> None:
        self.assertFalse(correct_trailing_newlines('hello\nworld'))

    def test_hello_world_multiline_gap(self) -> None:
        self.assertTrue(correct_trailing_newlines('hello\n\nworld\n'))


if __name__ == '__main__':
    unittest.main()
