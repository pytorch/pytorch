import re
import unittest

from tools.linter.adapters.header_only_linter import (
    check_file,
    CPP_TEST_GLOBS,
    find_matched_symbols,
    LINTER_CODE,
    LintMessage,
    LintSeverity,
    REPO_ROOT,
)


class TestHeaderOnlyLinter(unittest.TestCase):
    """
    Test the header only linter functionality
    """

    def test_find_matched_symbols(self) -> None:
        sample_regex = re.compile("symDef|symD|symC|bbb|a")
        test_globs = ["tools/test/header_only_linter_testdata/*.cpp"]

        expected_matches = {"symDef", "symC", "a"}
        self.assertEqual(
            find_matched_symbols(sample_regex, test_globs), expected_matches
        )

    def test_find_matched_symbols_empty_regex(self) -> None:
        sample_regex = re.compile("")
        test_globs = ["tools/test/header_only_linter_testdata/*.cpp"]

        expected_matches: set[str] = set()
        self.assertEqual(
            find_matched_symbols(sample_regex, test_globs), expected_matches
        )

    def test_check_file_no_issues(self) -> None:
        sample_txt = str(REPO_ROOT / "tools/test/header_only_linter_testdata/good.txt")
        test_globs = ["tools/test/header_only_linter_testdata/*.cpp"]
        self.assertEqual(len(check_file(sample_txt, test_globs)), 0)

    def test_check_empty_file(self) -> None:
        sample_txt = str(REPO_ROOT / "tools/test/header_only_linter_testdata/empty.txt")
        test_globs = ["tools/test/header_only_linter_testdata/*.cpp"]
        self.assertEqual(len(check_file(sample_txt, test_globs)), 0)

    def test_check_file_with_untested_symbols(self) -> None:
        sample_txt = str(REPO_ROOT / "tools/test/header_only_linter_testdata/bad.txt")
        test_globs = ["tools/test/header_only_linter_testdata/*.cpp"]

        expected_msgs = [
            LintMessage(
                path=sample_txt,
                line=7,
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.ERROR,
                name="[untested-symbol]",
                original=None,
                replacement=None,
                description=(
                    f"bbb has been included as a header-only API "
                    "but is not tested in any of CPP_TEST_GLOBS, which "
                    f"contains {CPP_TEST_GLOBS}.\n"
                    "Please add a .cpp test using the symbol without "
                    "linking anything to verify that the symbol is in "
                    "fact header-only. If you already have a test but it's"
                    " not found, please add the .cpp file to CPP_TEST_GLOBS"
                    " in tools/linters/adapters/header_only_linter.py."
                ),
            ),
            LintMessage(
                path=sample_txt,
                line=8,
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.ERROR,
                name="[untested-symbol]",
                original=None,
                replacement=None,
                description=(
                    f"symD has been included as a header-only API "
                    "but is not tested in any of CPP_TEST_GLOBS, which "
                    f"contains {CPP_TEST_GLOBS}.\n"
                    "Please add a .cpp test using the symbol without "
                    "linking anything to verify that the symbol is in "
                    "fact header-only. If you already have a test but it's"
                    " not found, please add the .cpp file to CPP_TEST_GLOBS"
                    " in tools/linters/adapters/header_only_linter.py."
                ),
            ),
        ]
        self.assertEqual(set(check_file(sample_txt, test_globs)), set(expected_msgs))


if __name__ == "__main__":
    unittest.main()
