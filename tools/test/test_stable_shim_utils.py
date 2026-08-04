import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))


import re

from tools.linter.adapters._stable_shim_utils import (
    DYNAMIC_VERSION_CALL_IDENTIFIER_MATCHER,
    FUNCTION_IDENTIFIER_MATCHER,
    IdentifierMatcher,
    IdentifierUse,
    MatcherAccumulator,
    MultilineMatcher,
    STRUCT_CLASS_IDENTIFIER_MATCHER,
    TYPEDEF_IDENTIFIER_MATCHER,
    USING_IDENTIFIER_MATCHER,
)


class TestStableShimUtils(unittest.TestCase):
    """Test stable shim utils functionality"""

    def _run_match_on_sample(
        self,
        sample: str,
        matcher: MatcherAccumulator,
        expected_version: tuple[int, int, int] | None,
    ) -> dict[int, list[str]]:
        # Run through the lines, collect all identifiers on the line and return.
        result = {}
        for li, line in enumerate(sample.split("\n"), 1):
            matcher.process_line(line)
            res = matcher.identifiers_used()
            if res:
                result[li] = [match.identifier for match in res]
                for match in res:
                    self.assertEqual(match.version, expected_version)
        return result

    def test_function_match_accumulator(self):
        matcher = MatcherAccumulator([FUNCTION_IDENTIFIER_MATCHER])
        expected_version = (2, 8, 0)
        matcher.set_scope_version(expected_version)

        sample = """
        // Simple version with just two function, checks that we don't
        // lose the line after the first match.
        AOTI_TORCH_EXPORT int primary_path(int arg);
        AOTI_TORCH_EXPORT int secondary_path(int arg);

        // Simple version over multiple lines.
        AOTI_TORCH_EXPORT int
        amazing_long_function_name_that_pushes_it_to_newline();

        // Situation where the args also flow over.
        AOTI_TORCH_EXPORT int
        amazing_long_function_name_with_args(
        int arg,
        void* foo,
        );

        // Situation where there's an end token ";" in a comment
        AOTI_TORCH_EXPORT int  // deprecated; was unnecessary
        amazing_long_function_name_with_end_in_comment(
        int arg,
        void* foo,
        );
        """

        expected = {
            4: ["primary_path"],
            5: ["secondary_path"],
            9: ["amazing_long_function_name_that_pushes_it_to_newline"],
            16: ["amazing_long_function_name_with_args"],
            23: ["amazing_long_function_name_with_end_in_comment"],
        }
        result = self._run_match_on_sample(sample, matcher, expected_version)

        self.assertEqual(result, expected)

    def test_typedef_match_accumulator(self):
        matcher = MatcherAccumulator([TYPEDEF_IDENTIFIER_MATCHER])
        expected_version = None
        matcher.set_scope_version(expected_version)

        sample = """
        typedef void (*legacy_callback)(int);
        typedef void
        (*more_lines)(int);
        """

        expected = {
            2: ["legacy_callback"],
            4: ["more_lines"],
        }
        result = self._run_match_on_sample(sample, matcher, expected_version)
        self.assertEqual(result, expected)

    def test_using_match_accumulator(self):
        matcher = MatcherAccumulator([USING_IDENTIFIER_MATCHER])

        expected_version = (2, 5, 0)
        matcher.set_scope_version(expected_version)

        sample = """
        using HandleType = OpaqueHandle*;
        """

        expected = {
            2: ["HandleType"],
        }
        result = self._run_match_on_sample(sample, matcher, expected_version)
        self.assertEqual(result, expected)

    def test_struct_class_match_accumulator(self):
        matcher = MatcherAccumulator([STRUCT_CLASS_IDENTIFIER_MATCHER])

        expected_version = (2, 5, 0)
        matcher.set_scope_version(expected_version)

        sample = """
        struct NewOpaqueStruct {
          int32_t type;
          void* buffer;
          size_t capacity;
        };

        class NewOpaqueClass {
         public:
          virtual ~NewOpaqueClass() = default;
          virtual void process() = 0;
        };
        """

        expected = {
            3: ["NewOpaqueStruct"],
            10: ["NewOpaqueClass"],
        }
        result = self._run_match_on_sample(sample, matcher, expected_version)
        self.assertEqual(result, expected)

    def test_future_multiple_identifier_accumulator(self):
        # This is a test that implements functionality that is not yet needed, but
        # ensures that the API can handle a single section returning multiple
        # identifiers at different versions.
        # In this case it parses a macro called `TO_BE_DETERMINED_MULTI_VERSION_MACRO`
        # with two arguments, where one argument gets the provided version, while the
        # other arguments is always set to the None version.
        def bespoke_macro_parser(
            buffer: str, current_version: tuple[int, int, int] | None
        ):
            pattern = r"TO_BE_DETERMINED_MULTI_VERSION_MACRO\(([^,]*),([^\)]*)\)"
            buffer_without_space = buffer.replace(" ", "").replace("\n", "")
            res = re.findall(pattern, buffer_without_space)
            return [
                IdentifierUse(identifier=res[0][0], version=None),
                IdentifierUse(identifier=res[0][1], version=current_version),
            ]

        matcher = MatcherAccumulator(
            [
                MultilineMatcher(
                    start_pattern=r"\s*TO_BE_DETERMINED_MULTI_VERSION_MACRO",
                    end_pattern=";",
                    handler=bespoke_macro_parser,
                )
            ]
        )

        expected_version = (2, 5, 0)
        matcher.set_scope_version(expected_version)

        sample = """
        const char* simple = TO_BE_DETERMINED_MULTI_VERSION_MACRO(A, B);
        //  Not simple;
        const char* thing = TO_BE_DETERMINED_MULTI_VERSION_MACRO(
        // Old thing always exists.
        old_thing,
        // New thing if we have it.
        new_thing);
        """
        expected = {
            2: [
                IdentifierUse(identifier="A", version=None),
                IdentifierUse(identifier="B", version=(2, 5, 0)),
            ],
            8: [
                IdentifierUse(identifier="old_thing", version=None),
                IdentifierUse(identifier="new_thing", version=(2, 5, 0)),
            ],
        }
        result = {}
        for li, line in enumerate(sample.split("\n"), 1):
            matcher.process_line(line)
            res = matcher.identifiers_used()
            if res:
                result[li] = res

        self.assertEqual(result, expected)

    def test_identifier_detection(self):
        """
        This unit test verifies arbitrary_identifier_matcher, which is used
        to generate matches for all the relevant function names and other
        identifiers by stable_shim_usage_linter's check_file.
        """
        matcher = MatcherAccumulator(
            [
                IdentifierMatcher.word("primary_path"),
                IdentifierMatcher.word("secondary_path"),
                IdentifierMatcher.word("short1"),
                IdentifierMatcher.word("short2"),
            ]
        )
        expected_version = (2, 8, 0)
        matcher.set_scope_version(expected_version)

        sample = """
        // Simple version with just two function, checks that we don't
        // lose the line after the first match.
        AOTI_TORCH_EXPORT int primary_path(int arg);
        AOTI_TORCH_EXPORT int secondary_path(int arg);

        // But what about two identifiers on a line?
        short1(3) + short2(5)
        """

        expected = {
            4: ["primary_path"],
            5: ["secondary_path"],
            8: ["short1", "short2"],
        }
        result = self._run_match_on_sample(sample, matcher, expected_version)

        self.assertEqual(result, expected)

    def test_dynamic_version_call_identifier(self):
        """
        This unit tests confirms the dynamic version call macro is parsed correctly and that its versioning guarantees
        are correct.
        """
        matcher = MatcherAccumulator([DYNAMIC_VERSION_CALL_IDENTIFIER_MATCHER])
        expected_version = (2, 13, 7)
        matcher.set_scope_version(expected_version)

        sample = """
        const auto& error_msg = TORCH_DYNAMIC_VERSION_CALL_2_13_0(
            torch_exception_get_what, torch_shim_bc_const_char_ptr);

        const auto& error_msg = TORCH_DYNAMIC_VERSION_CALL_2_10_0(
            something_from_2_10, super_old_fallback);
        """

        expected = {
            3: [
                IdentifierUse(
                    identifier="torch_exception_get_what", version=(2, 13, 0)
                ),
                IdentifierUse(
                    identifier="torch_shim_bc_const_char_ptr", version=(2, 13, 7)
                ),
            ],
            6: [
                IdentifierUse(identifier="something_from_2_10", version=(2, 10, 0)),
                IdentifierUse(identifier="super_old_fallback", version=(2, 13, 7)),
            ],
        }
        result = {}
        for li, line in enumerate(sample.split("\n"), 1):
            matcher.process_line(line)
            res = matcher.identifiers_used()
            if res:
                result[li] = res

        self.assertEqual(result, expected)

    def test_dynamic_version_call_with_trailing_args(self):
        """
        The dynamic version call macro is variadic: it forwards trailing args to
        the shim/fallback. The parser must still pick out only the shim and
        fallback identifiers, even when those trailing args contain commas or
        nested parens.
        """
        matcher = MatcherAccumulator([DYNAMIC_VERSION_CALL_IDENTIFIER_MATCHER])
        matcher.set_scope_version((2, 13, 0))

        sample = """
        auto a = TORCH_DYNAMIC_VERSION_CALL_2_13_0(
            shim_with_args, bc_with_args, self.get(), other.get(), alpha);

        auto b = TORCH_DYNAMIC_VERSION_CALL_2_10_0(no_args_shim, no_args_fallback);
        """

        expected = {
            3: [
                IdentifierUse(identifier="shim_with_args", version=(2, 13, 0)),
                IdentifierUse(identifier="bc_with_args", version=(2, 13, 0)),
            ],
            5: [
                IdentifierUse(identifier="no_args_shim", version=(2, 10, 0)),
                IdentifierUse(identifier="no_args_fallback", version=(2, 13, 0)),
            ],
        }
        result = {}
        for li, line in enumerate(sample.split("\n"), 1):
            matcher.process_line(line)
            res = matcher.identifiers_used()
            if res:
                result[li] = res

        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
