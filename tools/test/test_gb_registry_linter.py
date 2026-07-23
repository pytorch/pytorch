# mypy: ignore-errors
import json
import shutil
import unittest
from pathlib import Path

from tools.linter.adapters.gb_registry_linter import (
    check_registry_sync,
    LINTER_CODE,
    LintMessage,
    LintSeverity,
)


class TestGraphBreakRegistryLinter(unittest.TestCase):
    """
    Test the graph break registry linter functionality
    """

    def setUp(self):
        script_dir = Path(__file__).resolve()
        self.test_data_dir = script_dir.parent / "graph_break_registry_linter_testdata"
        self.test_data_dir.mkdir(parents=True, exist_ok=True)

        self.registry_path = self.test_data_dir / "graph_break_test_registry.json"
        with open(self.registry_path, "w") as f:
            json.dump({}, f)

        self.callsite_file = self.test_data_dir / "callsite_test.py"
        callsite_content = """from torch._dynamo.exc import unimplemented

def test(self):
    unimplemented(
        gb_type="testing",
        context="testing",
        explanation="testing",
        hints=["testing"],
    )
"""
        with open(self.callsite_file, "w") as f:
            f.write(callsite_content)

    def tearDown(self):
        if self.test_data_dir.exists():
            shutil.rmtree(self.test_data_dir)

    def test_case1_new_gb_type(self):
        """Test Case 1: Adding a completely new gb_type to an empty registry."""
        with open(self.registry_path) as f:
            original_content = f.read()

        messages = check_registry_sync(self.test_data_dir, self.registry_path)

        # Parse the replacement to get the actual GB ID that was generated
        self.assertEqual(len(messages), 1)
        replacement_registry = json.loads(messages[0].replacement)
        gb_id = next(iter(replacement_registry.keys()))

        expected_registry = {
            gb_id: [
                {
                    "Gb_type": "testing",
                    "Context": "testing",
                    "Explanation": "testing",
                    "Hints": ["testing"],
                }
            ]
        }
        expected_replacement = (
            json.dumps(expected_registry, indent=2, ensure_ascii=False) + "\n"
        )
        expected_msg = LintMessage(
            path=str(self.registry_path),
            line=None,
            char=None,
            code=LINTER_CODE,
            severity=LintSeverity.WARNING,
            name="Registry sync needed",
            original=original_content,
            replacement=expected_replacement,
            description="Registry sync needed (added 1 new gb_types). Run `lintrunner -a` to apply changes.",
        )
        self.assertEqual(messages, [expected_msg])

        if messages and messages[0].replacement:
            with open(self.registry_path, "w") as f:
                f.write(messages[0].replacement)

        messages_after_fix = check_registry_sync(self.test_data_dir, self.registry_path)
        self.assertEqual(
            len(messages_after_fix), 0, "Should have no messages after applying the fix"
        )

    def test_case2_rename_gb_type(self):
        """Test Case 2: Renaming a gb_type while keeping other content the same."""
        registry_data = {
            "GB0000": [
                {
                    "Gb_type": "testing",
                    "Context": "testing",
                    "Explanation": "testing",
                    "Hints": ["testing"],
                }
            ]
        }
        with open(self.registry_path, "w") as f:
            json.dump(registry_data, f, indent=2)

        renamed_callsite_content = """from torch._dynamo.exc import unimplemented
def test(self):
    unimplemented(gb_type="renamed_testing", context="testing", explanation="testing", hints=["testing"])
"""
        with open(self.callsite_file, "w") as f:
            f.write(renamed_callsite_content)

        with open(self.registry_path) as f:
            original_content = f.read()

        messages = check_registry_sync(self.test_data_dir, self.registry_path)
        expected_registry = {
            "GB0000": [
                {
                    "Gb_type": "renamed_testing",
                    "Context": "testing",
                    "Explanation": "testing",
                    "Hints": ["testing"],
                },
                {
                    "Gb_type": "testing",
                    "Context": "testing",
                    "Explanation": "testing",
                    "Hints": ["testing"],
                },
            ]
        }
        expected_replacement = (
            json.dumps(expected_registry, indent=2, ensure_ascii=False) + "\n"
        )
        expected_msg = LintMessage(
            path=str(self.registry_path),
            line=None,
            char=None,
            code=LINTER_CODE,
            severity=LintSeverity.WARNING,
            name="Registry sync needed",
            original=original_content,
            replacement=expected_replacement,
            description="Registry sync needed (renamed 'testing' → 'renamed_testing'). Run `lintrunner -a` to apply changes.",
        )
        self.assertEqual(messages, [expected_msg])

        if messages and messages[0].replacement:
            with open(self.registry_path, "w") as f:
                f.write(messages[0].replacement)

        messages_after_fix = check_registry_sync(self.test_data_dir, self.registry_path)
        self.assertEqual(
            len(messages_after_fix), 0, "Should have no messages after applying the fix"
        )

    def test_case3_content_change(self):
        """Test Case 3: Changing the content of an existing gb_type."""
        registry_data = {
            "GB0000": [
                {
                    "Gb_type": "testing",
                    "Context": "old_context",
                    "Explanation": "old_explanation",
                    "Hints": ["old_hint"],
                }
            ]
        }
        with open(self.registry_path, "w") as f:
            json.dump(registry_data, f, indent=2)

        updated_callsite_content = """from torch._dynamo.exc import unimplemented
def test(self):
    unimplemented(gb_type="testing", context="new_context", explanation="new_explanation", hints=["new_hint"])
"""
        with open(self.callsite_file, "w") as f:
            f.write(updated_callsite_content)

        with open(self.registry_path) as f:
            original_content = f.read()

        messages = check_registry_sync(self.test_data_dir, self.registry_path)
        expected_registry = {
            "GB0000": [
                {
                    "Gb_type": "testing",
                    "Context": "new_context",
                    "Explanation": "new_explanation",
                    "Hints": ["new_hint"],
                },
                {
                    "Gb_type": "testing",
                    "Context": "old_context",
                    "Explanation": "old_explanation",
                    "Hints": ["old_hint"],
                },
            ]
        }
        expected_replacement = (
            json.dumps(expected_registry, indent=2, ensure_ascii=False) + "\n"
        )
        expected_msg = LintMessage(
            path=str(self.registry_path),
            line=None,
            char=None,
            code=LINTER_CODE,
            severity=LintSeverity.WARNING,
            name="Registry sync needed",
            original=original_content,
            replacement=expected_replacement,
            description="Registry sync needed (). Run `lintrunner -a` to apply changes.",
        )
        self.assertEqual(messages, [expected_msg])

        if messages and messages[0].replacement:
            with open(self.registry_path, "w") as f:
                f.write(messages[0].replacement)

        messages_after_fix = check_registry_sync(self.test_data_dir, self.registry_path)
        self.assertEqual(
            len(messages_after_fix), 0, "Should have no messages after applying the fix"
        )

    def test_case4_no_changes(self):
        """Test Case 4: Ensuring no message is produced when the registry is in sync."""
        registry_data = {
            "GB0000": [
                {
                    "Gb_type": "testing",
                    "Context": "testing",
                    "Explanation": "testing",
                    "Hints": ["testing"],
                }
            ]
        }
        with open(self.registry_path, "w") as f:
            json.dump(registry_data, f, indent=2)

        messages = check_registry_sync(self.test_data_dir, self.registry_path)
        self.assertEqual(
            len(messages), 0, "Should have no messages when registry is already in sync"
        )

    def test_case5_new_gbid_on_full_change(self):
        """Test Case 5: A completely new entry should get a new GB ID."""
        registry_data = {
            "GB0000": [
                {
                    "Gb_type": "original_testing",
                    "Context": "original_context",
                    "Explanation": "original_explanation",
                    "Hints": ["original_hint"],
                }
            ]
        }
        with open(self.registry_path, "w") as f:
            json.dump(registry_data, f, indent=2)

        new_callsite_content = """from torch._dynamo.exc import unimplemented
def test(self):
    unimplemented(
        gb_type="completely_new_testing",
        context="completely_new_context",
        explanation="completely_new_explanation",
        hints=["completely_new_hint"],
    )
"""
        with open(self.callsite_file, "w") as f:
            f.write(new_callsite_content)

        with open(self.registry_path) as f:
            original_content = f.read()

        messages = check_registry_sync(self.test_data_dir, self.registry_path)

        # Parse the replacement to get the actual GB ID that was generated
        self.assertEqual(len(messages), 1)
        replacement_registry = json.loads(messages[0].replacement)

        # Build expected_registry in the same order as replacement_registry
        # since random insertion means order is not deterministic
        expected_registry = {}
        for gb_id in replacement_registry:
            if gb_id == "GB0000":
                expected_registry[gb_id] = [
                    {
                        "Gb_type": "original_testing",
                        "Context": "original_context",
                        "Explanation": "original_explanation",
                        "Hints": ["original_hint"],
                    }
                ]
            else:
                expected_registry[gb_id] = [
                    {
                        "Gb_type": "completely_new_testing",
                        "Context": "completely_new_context",
                        "Explanation": "completely_new_explanation",
                        "Hints": ["completely_new_hint"],
                    }
                ]

        expected_replacement = (
            json.dumps(expected_registry, indent=2, ensure_ascii=False) + "\n"
        )
        expected_msg = LintMessage(
            path=str(self.registry_path),
            line=None,
            char=None,
            code=LINTER_CODE,
            severity=LintSeverity.WARNING,
            name="Registry sync needed",
            original=original_content,
            replacement=expected_replacement,
            description="Registry sync needed (added 1 new gb_types). Run `lintrunner -a` to apply changes.",
        )
        self.assertEqual(messages, [expected_msg])

        # Apply the fix and verify the file's final state
        if messages and messages[0].replacement:
            with open(self.registry_path, "w") as f:
                f.write(messages[0].replacement)

        messages_after_fix = check_registry_sync(self.test_data_dir, self.registry_path)
        self.assertEqual(
            len(messages_after_fix), 0, "Should have no messages after applying the fix"
        )

    def test_case6_dynamic_hints_from_variable(self):
        """Test Case 6: Verifies hints can be unpacked from an imported variable."""
        mock_hints_file = self.test_data_dir / "graph_break_hints.py"
        init_py = self.test_data_dir / "__init__.py"
        try:
            supportable_string = (
                "It may be possible to write Dynamo tracing rules for this code. Please report an issue to PyTorch if you "
                "encounter this graph break often and it is causing performance issues."
            )
            mock_hints_content = f'SUPPORTABLE = ["{supportable_string}"]'
            with open(mock_hints_file, "w") as f:
                f.write(mock_hints_content)

            init_py.touch()

            dynamic_hints_callsite = """from torch._dynamo.exc import unimplemented
from torch._dynamo import graph_break_hints

def test(self):
    unimplemented(
        gb_type="testing_with_graph_break_hints",
        context="testing_with_graph_break_hints",
        explanation="testing_with_graph_break_hints",
        hints=[*graph_break_hints.SUPPORTABLE],
    )
    """
            with open(self.callsite_file, "w") as f:
                f.write(dynamic_hints_callsite)

            with open(self.registry_path) as f:
                original_content = f.read()

            messages = check_registry_sync(self.test_data_dir, self.registry_path)

            # Parse the replacement to get the actual GB ID that was generated
            self.assertEqual(len(messages), 1)
            replacement_registry = json.loads(messages[0].replacement)
            gb_id = next(iter(replacement_registry.keys()))

            expected_registry = {
                gb_id: [
                    {
                        "Gb_type": "testing_with_graph_break_hints",
                        "Context": "testing_with_graph_break_hints",
                        "Explanation": "testing_with_graph_break_hints",
                        "Hints": [supportable_string],
                    }
                ]
            }
            expected_replacement = (
                json.dumps(expected_registry, indent=2, ensure_ascii=False) + "\n"
            )
            expected_msg = LintMessage(
                path=str(self.registry_path),
                line=None,
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.WARNING,
                name="Registry sync needed",
                original=original_content,
                replacement=expected_replacement,
                description="Registry sync needed (added 1 new gb_types). Run `lintrunner -a` to apply changes.",
            )

            self.assertEqual(messages, [expected_msg])

            if messages and messages[0].replacement:
                with open(self.registry_path, "w") as f:
                    f.write(messages[0].replacement)

            messages_after_fix = check_registry_sync(
                self.test_data_dir, self.registry_path
            )
            self.assertEqual(
                len(messages_after_fix),
                0,
                "Should have no messages after applying the fix",
            )
        finally:
            mock_hints_file.unlink()
            init_py.unlink()

    def test_case7_duplicate_gb_type_in_registry(self):
        """Test Case 7: Detecting duplicate gb_types across different GB IDs in the registry."""
        registry_data = {
            "GB0000": [
                {
                    "Gb_type": "duplicate_type",
                    "Context": "context1",
                    "Explanation": "explanation1",
                    "Hints": ["hint1"],
                }
            ],
            "GB0042": [
                {
                    "Gb_type": "duplicate_type",
                    "Context": "context2",
                    "Explanation": "explanation2",
                    "Hints": ["hint2"],
                }
            ],
        }
        with open(self.registry_path, "w") as f:
            json.dump(registry_data, f, indent=2)

        # Create a callsite with one of the duplicate types
        callsite_content = """from torch._dynamo.exc import unimplemented
def test(self):
    unimplemented(gb_type="duplicate_type", context="context1", explanation="explanation1", hints=["hint1"])
"""
        with open(self.callsite_file, "w") as f:
            f.write(callsite_content)

        messages = check_registry_sync(self.test_data_dir, self.registry_path)

        expected_msg = LintMessage(
            path=str(self.registry_path),
            line=None,
            char=None,
            code=LINTER_CODE,
            severity=LintSeverity.ERROR,
            name="Duplicate gb_type in registry",
            original=None,
            replacement=None,
            description=(
                "The gb_type 'duplicate_type' appears in multiple GB IDs: GB0000, GB0042. "
                "Each gb_type must map to exactly one GB ID. Please manually fix the registry."
            ),
        )
        self.assertEqual(messages, [expected_msg])

    def test_case8_forbid_direct_raise_unsupported(self):
        callsite_content = """from torch._dynamo.exc import Unsupported

def test(self):
    raise Unsupported("testing")
"""
        with open(self.callsite_file, "w") as f:
            f.write(callsite_content)

        messages = check_registry_sync(self.test_data_dir, self.registry_path)
        self.assertEqual(len(messages), 1)

        message = messages[0]
        self.assertEqual(message.path, str(self.callsite_file))
        self.assertEqual(message.line, 4)
        self.assertEqual(message.code, LINTER_CODE)
        self.assertEqual(message.severity, LintSeverity.ERROR)
        self.assertEqual(message.name, "Direct raise Unsupported")
        self.assertIn("Use `unimplemented(...)`", message.description)

    def test_case9_forbid_raise_exc_unsupported(self):
        callsite_content = """from torch._dynamo import exc

def test(self):
    raise exc.Unsupported("testing")
"""
        with open(self.callsite_file, "w") as f:
            f.write(callsite_content)

        messages = check_registry_sync(self.test_data_dir, self.registry_path)
        self.assertEqual(len(messages), 1)

        message = messages[0]
        self.assertEqual(message.path, str(self.callsite_file))
        self.assertEqual(message.line, 4)
        self.assertEqual(message.code, LINTER_CODE)
        self.assertEqual(message.severity, LintSeverity.ERROR)
        self.assertEqual(message.name, "Direct raise Unsupported")

    def test_case10_allow_noqa_for_direct_raise_unsupported(self):
        callsite_content = """from torch._dynamo.exc import Unsupported

def test(self):
    raise Unsupported("testing")  # noqa: GB_REGISTRY
"""
        with open(self.callsite_file, "w") as f:
            f.write(callsite_content)

        messages = check_registry_sync(self.test_data_dir, self.registry_path)
        self.assertEqual(messages, [])

    def test_case10_allow_previous_line_noqa_for_direct_raise_unsupported(self):
        callsite_content = """from torch._dynamo.exc import Unsupported

def test(self):
    # noqa: GB_REGISTRY
    raise Unsupported("testing")
"""
        with open(self.callsite_file, "w") as f:
            f.write(callsite_content)

        messages = check_registry_sync(self.test_data_dir, self.registry_path)
        self.assertEqual(messages, [])

    def test_case11_allow_noqa_in_exc_py(self):
        callsite_content = """def test(self):
    pass
"""
        with open(self.callsite_file, "w") as f:
            f.write(callsite_content)

        exc_file = self.test_data_dir / "exc.py"
        exc_content = """class Unsupported(Exception):
    pass

def unimplemented():
    raise Unsupported("testing")  # noqa: GB_REGISTRY

def other():
    raise Unsupported("not allowed")  # noqa: GB_REGISTRY
"""
        with open(exc_file, "w") as f:
            f.write(exc_content)

        messages = check_registry_sync(self.test_data_dir, self.registry_path)
        self.assertEqual(messages, [])


if __name__ == "__main__":
    unittest.main()
