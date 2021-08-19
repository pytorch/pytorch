import unittest
from tools.linter.clang_tidy.max_tokens_pragma import (
    add_max_tokens_pragma,
    strip_max_tokens_pragmas,
)


def compare_code(a: str, b: str) -> bool:
    a_lines = [line.strip() for line in a.splitlines()]
    b_lines = [line.strip() for line in b.splitlines()]
    return a_lines == b_lines


class TestMaxTokensPragma(unittest.TestCase):
    def test_no_prior_pragmas(self) -> None:
        input = """\
        // File without any prior pragmas

        int main() {
          for (int i = 0; i < 10; i++);
          return 0;
        }
        """

        expected = """\
        #pragma clang max_tokens_total 42
        // File without any prior pragmas

        int main() {
          for (int i = 0; i < 10; i++);
          return 0;
        }
        """
        output = add_max_tokens_pragma(input, 42)
        self.assertTrue(compare_code(output, expected))

        output = strip_max_tokens_pragmas(output)
        self.assertTrue(compare_code(output, input))

    def test_single_prior_pragma(self) -> None:
        input = """\
        // File with prior pragmas

        #pragma clang max_tokens_total 1

        int main() {
          for (int i = 0; i < 10; i++);
          return 0;
        }
        """

        expected = """\
        // File with prior pragmas

        #pragma clang max_tokens_total 42

        int main() {
          for (int i = 0; i < 10; i++);
          return 0;
        }
        """
        stripped = """\
        // File with prior pragmas


        int main() {
          for (int i = 0; i < 10; i++);
          return 0;
        }
        """

        output = add_max_tokens_pragma(input, 42)
        self.assertTrue(compare_code(output, expected))

        output = strip_max_tokens_pragmas(output)
        self.assertTrue(compare_code(output, stripped))

    def test_multiple_prior_pragmas(self) -> None:
        input = """\
        // File with multiple prior pragmas

        #pragma clang max_tokens_total 1

        // Different pragma; script should ignore this
        #pragma clang max_tokens_here 20

        int main() {
          for (int i = 0; i < 10; i++);
          return 0;
        }

        #pragma clang max_tokens_total 1
        """

        expected = """\
        // File with multiple prior pragmas

        #pragma clang max_tokens_total 42

        // Different pragma; script should ignore this
        #pragma clang max_tokens_here 20

        int main() {
          for (int i = 0; i < 10; i++);
          return 0;
        }

        #pragma clang max_tokens_total 42
        """
        stripped = """\
        // File with multiple prior pragmas


        // Different pragma; script should ignore this
        #pragma clang max_tokens_here 20

        int main() {
          for (int i = 0; i < 10; i++);
          return 0;
        }

        """

        output = add_max_tokens_pragma(input, 42)
        self.assertTrue(compare_code(output, expected))

        output = strip_max_tokens_pragmas(output)
        self.assertTrue(compare_code(output, stripped))


if __name__ == "__main__":
    unittest.main()
