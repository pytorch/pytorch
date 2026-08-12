from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from tools.linter.adapters.raw_throw_linter import (
    check_source,
    find_throws,
    replacement_macro,
    scan_source,
)


PATH = "torch/csrc/example.cpp"


def code_of(source: str) -> str:
    return scan_source(source)[0]


def comments_of(source: str) -> str:
    return scan_source(source)[1]


def names(source: str, path: str = PATH) -> list[str]:
    return [m.name for m in check_source(path, source)]


def lines_flagged(source: str, path: str = PATH) -> list[int | None]:
    return [m.line for m in check_source(path, source) if m.name == "raw-throw"]


class TestScanSource(unittest.TestCase):
    def assertBlanked(self, source: str, expected_code: str) -> None:
        code = code_of(source)
        self.assertEqual(len(code), len(source))
        self.assertEqual(code, expected_code)

    def test_line_comment(self) -> None:
        self.assertBlanked("a; // throw\nb;\n", "a;         \nb;\n")

    def test_block_comment(self) -> None:
        self.assertBlanked("a; /* throw\n throw */ b;\n", "a;         \n          b;\n")

    def test_string_literal(self) -> None:
        self.assertBlanked('f("throw");\n', "f(       );\n")

    def test_string_with_escaped_quote(self) -> None:
        self.assertBlanked('f("a\\"throw");\n', "f(          );\n")

    def test_char_literal(self) -> None:
        self.assertBlanked("c = ';';\n", "c =    ;\n")

    def test_digit_separator_is_not_a_char_literal(self) -> None:
        # 1'000'000 must not be read as a char literal swallowing the throw.
        source = "n = 1'000'000;\nthrow std::runtime_error(x);\n"
        self.assertEqual(len(find_throws(code_of(source))), 1)

    def test_prefixed_char_literal_is_not_a_digit_separator(self) -> None:
        for prefix in ("L", "u", "U", "u8"):
            with self.subTest(prefix=prefix):
                source = f"wchar_t c = {prefix}'a';\nthrow std::runtime_error(x);\n"
                self.assertEqual([t.line for t in find_throws(code_of(source))], [2])

    def test_hex_digit_separator(self) -> None:
        source = "n = 0xFF'FF;\nthrow Foo();\n"
        self.assertEqual([t.line for t in find_throws(code_of(source))], [2])

    def test_raw_string(self) -> None:
        source = 'auto s = R"(throw "x" )";\nthrow Foo();\n'
        throws = find_throws(code_of(source))
        self.assertEqual([t.line for t in throws], [2])

    def test_raw_string_with_delimiter(self) -> None:
        source = 'auto s = R"py(throw)py";\nthrow Foo();\n'
        self.assertEqual([t.line for t in find_throws(code_of(source))], [2])

    def test_backslash_continued_line_comment(self) -> None:
        source = "// a \\\nthrow Foo();\nthrow Bar();\n"
        self.assertEqual([t.line for t in find_throws(code_of(source))], [3])

    def test_comment_view_keeps_only_comments(self) -> None:
        source = 'x; // @allow-raw-throw: ok\ny("@allow-raw-throw: no");\n'
        comments = comments_of(source)
        self.assertEqual(comments.splitlines()[0].strip(), "// @allow-raw-throw: ok")
        self.assertEqual(comments.splitlines()[1].strip(), "")

    def test_apostrophe_in_comment_does_not_swallow_code(self) -> None:
        source = "// don't do this\nthrow Foo();\n"
        self.assertEqual([t.line for t in find_throws(code_of(source))], [2])

    def test_slashes_inside_a_string_do_not_start_a_comment(self) -> None:
        source = 'auto sep = "a // b";\nthrow Foo();\n'
        self.assertEqual([t.line for t in find_throws(code_of(source))], [2])

    def test_unterminated_block_comment_swallows_the_rest(self) -> None:
        self.assertEqual(find_throws(code_of("/* x\nthrow Foo();\n")), [])

    def test_line_numbers_are_preserved(self) -> None:
        source = '/* a\n b */ "c\\n"\nthrow Foo();\n'
        self.assertEqual([t.line for t in find_throws(code_of(source))], [3])


class TestFindThrows(unittest.TestCase):
    def test_expression(self) -> None:
        throws = find_throws(code_of("throw std::runtime_error(msg);\n"))
        self.assertEqual([t.expression for t in throws], ["std::runtime_error(msg)"])

    def test_wrapped_throw_is_one_occurrence(self) -> None:
        source = "throw(\n    ErrorReport(loc)\n    << what);\n"
        throws = find_throws(code_of(source))
        self.assertEqual(len(throws), 1)
        self.assertEqual(throws[0].line, 1)
        self.assertEqual(throws[0].expression, "ErrorReport(loc) << what")

    def test_macro_body_without_a_semicolon_stops_at_the_newline(self) -> None:
        source = "#define SHAPE_ASSERT(c) if (!(c)) throw propagation_error()\nint x;\n"
        throws = find_throws(code_of(source))
        self.assertEqual([t.expression for t in throws], ["propagation_error()"])

    def test_a_continued_macro_line_keeps_going(self) -> None:
        source = (
            "#define C10_THROW(t, m) \\\n  throw ::c10::t( \\\n      {__func__}, m)\n"
        )
        throws = find_throws(code_of(source))
        self.assertEqual(throws[0].expression, "::c10::t( {__func__}, m)")

    def test_throw_ending_a_macro_is_not_taken_for_a_rethrow(self) -> None:
        # Only `throw;` counts as a rethrow. A macro that expands to a bare
        # `throw` runs on to the next line and is reported rather than allowed.
        source = "#define RETHROW throw\n#define OTHER 1\n"
        self.assertEqual(names(source), ["raw-throw"])

    def test_operand_on_the_next_line_is_not_a_bare_rethrow(self) -> None:
        for source in (
            "if (x)\n  throw\n      std::runtime_error(y);\n",
            "throw // NOLINT\n    std::runtime_error(y);\n",
        ):
            with self.subTest(source=source):
                throws = find_throws(code_of(source))
                self.assertEqual(
                    [t.expression for t in throws], ["std::runtime_error(y)"]
                )

    def test_bare_rethrow(self) -> None:
        throws = find_throws(code_of("throw;\n"))
        self.assertEqual([t.expression for t in throws], [""])

    def test_rethrow_word_is_not_a_throw(self) -> None:
        self.assertEqual(find_throws(code_of("std::rethrow_exception(p);\n")), [])

    def test_throw_specifier_in_identifier(self) -> None:
        self.assertEqual(find_throws(code_of("int throwing_count = 0;\n")), [])


class TestBuckets(unittest.TestCase):
    def test_flagged(self) -> None:
        for expression in (
            "std::runtime_error(x)",
            "std::invalid_argument(x)",
            "std::out_of_range(x)",
            "std::logic_error(x)",
            # python_error is typed, but TORCH_CHECK_PYTHON replaces it.
            "python_error()",
            # A torch error type is not automatically allowed either.
            "c10::Error(x, y)",
            # Not in ALLOWED_EXCEPTION_TYPES, so it must be justified first.
            "ErrorReport(loc) << x",
            "TypeError(x)",
            # Exactly TORCH_CHECK_TYPE / _VALUE / _INDEX, same Python type.
            "py::type_error(x)",
            "py::value_error(x)",
            "py::index_error(x)",
            # Throws a sliced copy rather than re-raising.
            "e",
            # Reads like a rethrow, but the operand is a fresh exception.
            "std::move(e)",
            # Not a rethrow either, and neither name is allowed.
            "MyError<int>(x)",
            "factory.make()",
        ):
            source = f"throw {expression};\n"
            with self.subTest(expression=expression):
                self.assertEqual(names(source), ["raw-throw"])

    def test_allowed(self) -> None:
        for expression in (
            "",  # bare `throw;`
            "py::cast_error(x)",
            "py::key_error(x)",
            "::py::stop_iteration(x)",
            "PythonError()",
            "WorkerException(e)",
            "c10::AcceleratorError(x, y, z)",
            "c10::AcceleratorError{x}",
        ):
            source = f"throw {expression};\n"
            with self.subTest(expression=expression):
                self.assertEqual(names(source), [])

    def test_a_variable_named_like_an_allowed_type_is_still_flagged(self) -> None:
        self.assertEqual(names("throw PythonError;\n"), ["raw-throw"])

    def test_a_namespace_ending_in_py_is_not_pybind(self) -> None:
        for expression in ("foo::py::cast_error(x)", "pyxx::cast_error(x)"):
            with self.subTest(expression=expression):
                self.assertEqual(names(f"throw {expression};\n"), ["raw-throw"])

    def test_sliced_rethrow_offers_both_routes(self) -> None:
        (message,) = check_source(PATH, "throw e;\n")
        self.assertIn("throw;", message.description or "")
        self.assertIn("TORCH_CHECK", message.description or "")

    def test_only_a_bare_variable_gets_the_slicing_advice(self) -> None:
        for expression in ("MyError<int>(x)", "factory.make()", "Foo()"):
            with self.subTest(expression=expression):
                (message,) = check_source(PATH, f"throw {expression};\n")
                self.assertNotIn("sliced", message.description or "")

    def test_arguments_are_elided_from_the_diagnostic(self) -> None:
        # By this point string literals have been blanked, so printing them back
        # would show `throw Foo( )`.
        (message,) = check_source(PATH, 'throw Foo("some message", x);\n')
        self.assertIn("`throw Foo(...)`", message.description or "")

    def test_word_throw_in_comment_or_string_is_not_flagged(self) -> None:
        source = (
            "// we throw here\n"
            "/* throw */\n"
            'TORCH_CHECK(false, "cannot throw");\n'
            "int throwaway = 0;\n"
        )
        self.assertEqual(names(source), [])


class TestAllowMarker(unittest.TestCase):
    def test_marker_on_preceding_line_allows_the_throw(self) -> None:
        source = "// @allow-raw-throw: PyErr is already set here\nthrow Foo();\n"
        self.assertEqual(names(source), [])

    def test_marker_without_reason_is_rejected(self) -> None:
        source = "// @allow-raw-throw\nthrow Foo();\n"
        self.assertEqual(names(source), ["allow-raw-throw-without-reason"])

    def test_marker_with_empty_reason_is_rejected(self) -> None:
        source = "// @allow-raw-throw:   \nthrow Foo();\n"
        self.assertEqual(names(source), ["allow-raw-throw-without-reason"])

    def test_marker_above_an_allowed_throw_is_redundant(self) -> None:
        source = "// @allow-raw-throw: reason\nthrow py::cast_error(x);\n"
        self.assertEqual(names(source), ["redundant-allow-raw-throw"])

    def test_redundant_marker_is_not_also_reported_as_reasonless(self) -> None:
        source = "// @allow-raw-throw\nthrow py::cast_error(x);\n"
        self.assertEqual(names(source), ["redundant-allow-raw-throw"])

    def test_trailing_marker_does_not_allow_the_throw(self) -> None:
        source = "throw Foo(); // @allow-raw-throw: nope\n"
        self.assertEqual(names(source), ["orphaned-allow-raw-throw", "raw-throw"])

    def test_file_level_marker_is_orphaned(self) -> None:
        source = "// @allow-raw-throw\n#include <x.h>\nthrow Foo();\n"
        self.assertEqual(names(source), ["orphaned-allow-raw-throw", "raw-throw"])

    def test_marker_separated_by_blank_line_is_orphaned(self) -> None:
        source = "// @allow-raw-throw: reason\n\nthrow Foo();\n"
        self.assertEqual(names(source), ["orphaned-allow-raw-throw", "raw-throw"])

    def test_prose_mentioning_the_marker_licenses_nothing(self) -> None:
        source = "// Use @allow-raw-throw: <reason> to keep a throw.\nthrow Foo();\n"
        self.assertEqual(names(source), ["raw-throw"])

    def test_marker_must_be_the_whole_comment(self) -> None:
        source = "int y; // see @allow-raw-throw: below\nthrow Foo();\n"
        self.assertEqual(names(source), ["raw-throw"])

    def test_a_longer_word_is_not_the_marker(self) -> None:
        source = "// @allow-raw-throwing nonsense\nthrow Foo();\n"
        self.assertEqual(names(source), ["raw-throw"])

    def test_block_comment_marker(self) -> None:
        source = "/* @allow-raw-throw: reason */\nthrow Foo();\n"
        self.assertEqual(names(source), [])

    def test_punctuation_is_not_a_reason(self) -> None:
        source = "// @allow-raw-throw: .\nthrow Foo();\n"
        self.assertEqual(names(source), ["allow-raw-throw-without-reason"])

    def test_marker_in_string_is_not_a_marker(self) -> None:
        source = 'const char* s = "@allow-raw-throw: x";\nthrow Foo();\n'
        self.assertEqual(names(source), ["raw-throw"])

    def test_marker_allows_only_the_next_throw(self) -> None:
        source = "// @allow-raw-throw: reason\nthrow Foo();\nthrow Bar();\n"
        self.assertEqual(lines_flagged(source), [3])

    def test_trailing_marker_does_not_license_the_next_line(self) -> None:
        source = "throw A(); // @allow-raw-throw: about A\nthrow B();\n"
        self.assertEqual(lines_flagged(source), [1, 2])

    def test_marker_licenses_only_one_throw_on_the_target_line(self) -> None:
        source = "// @allow-raw-throw: reason\nif (a) throw A(); else throw B();\n"
        self.assertEqual(names(source), ["raw-throw"])

    def test_form_feed_does_not_desync_marker_and_throw_lines(self) -> None:
        source = "\f\n// @allow-raw-throw: reason\nthrow Foo();\n"
        self.assertEqual(names(source), [])

    def test_marker_licenses_the_flagged_throw_not_the_allowed_one(self) -> None:
        allowed, marker = "throw py::cast_error(x);", "// @allow-raw-throw: reason"
        self.assertEqual(names(f"{allowed}\n{marker}\nthrow Foo();\n"), [])

    def test_marker_above_a_wrapped_throw(self) -> None:
        source = "// @allow-raw-throw: reason\nthrow(\n    Foo());\n"
        self.assertEqual(names(source), [])


class TestReplacementMacro(unittest.TestCase):
    def test_per_directory(self) -> None:
        cases = {
            "aten/src/ATen/Foo.cpp": "TORCH_CHECK",
            "c10/util/Foo.cpp": "TORCH_CHECK",
            "torch/headeronly/util/Foo.h": "STD_TORCH_CHECK",
            "torch/csrc/stable/foo.h": "STD_TORCH_CHECK",
            "torch/csrc/inductor/aoti_runtime/model.h": "AOTI_RUNTIME_CHECK",
            "torch/csrc/inductor/aoti_torch/foo.cpp": "TORCH_CHECK",
        }
        for path, macro in cases.items():
            with self.subTest(path=path):
                self.assertEqual(replacement_macro(path), macro)

    def test_description_names_the_macro(self) -> None:
        (message,) = check_source("torch/headeronly/util/Foo.h", "throw Foo();\n")
        self.assertIn("STD_TORCH_CHECK", message.description or "")


if __name__ == "__main__":
    unittest.main()
