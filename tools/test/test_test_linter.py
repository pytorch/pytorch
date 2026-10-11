"""Tests for test_linter."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from tools.linter.adapters import test_linter
from tools.linter.adapters.test_linter import (
    check_file,
    error_msg,
    HardwareClassification,
    LintMessage,
    LintSeverity,
    main,
    REPO_ROOT,
    RuleId,
)


HC = HardwareClassification


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


class TestHwClassificationLinter(unittest.TestCase):
    def _run(self, content: str) -> list[LintMessage]:
        # The temp dir must live under test/ so the file passes _is_test_file's
        # location check and the linter actually runs on it.
        with tempfile.TemporaryDirectory(dir=str(REPO_ROOT / "test")) as td:
            root = Path(td)
            test_file = root / "test_sample.py"
            _write(test_file, textwrap.dedent(content))
            return check_file(str(test_file))

    # --- file parse errors ---

    def test_syntax_error_in_file(self) -> None:
        """A file that does not parse is reported, not silently skipped."""
        msgs = self._run("this is not valid python @@@")
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].name, test_linter.PARSE_ERROR_NAME)
        self.assertEqual(msgs[0].severity, LintSeverity.ERROR)
        self.assertEqual(msgs[0].line, 1)
        self.assertEqual(
            msgs[0].description, "Failed to parse the file: invalid syntax"
        )

    def test_unreadable_file_raises(self) -> None:
        with self.assertRaises(OSError):
            check_file(str(REPO_ROOT / "test" / "test_does_not_exist.py"))

    def test_error_msg_defaults(self) -> None:
        """Pin error_msg defaults so tests don't silently inherit a wrong severity/code."""
        msg = error_msg(name="[test]", path="some.py", line=None, description="boom")
        self.assertEqual(msg.severity, LintSeverity.ERROR)
        self.assertEqual(msg.code, "TEST_LINTER")

    # --- allowlist ---

    def test_allowlisted_file_skipped(self) -> None:
        """Files in the allowlist are skipped silently."""
        src = """\
            from torch.testing._internal.common_utils import TestCase
            class TestFoo(TestCase):
                def test_x(self): pass
        """
        with tempfile.TemporaryDirectory(dir=str(REPO_ROOT / "test")) as td:
            root = Path(td)
            test_file = root / "test_sample.py"
            _write(test_file, textwrap.dedent(src))
            rel_path = os.path.relpath(test_file, REPO_ROOT).replace("\\", "/")
            test_linter._allowlist.add(rel_path)
            try:
                self.assertEqual(check_file(str(test_file)), [])
            finally:
                test_linter._allowlist.discard(rel_path)

    # --- missing / invalid hw_classification ---

    def test_missing_or_invalid_hw_classification(self) -> None:
        """Classes without a valid hw_classification are flagged, including
        classes nested in control flow and camelCase/async-only test classes."""
        variants = [
            # (source, line of the class definition)
            (
                """\
                from torch.testing._internal.common_utils import TestCase
                class TestFoo(TestCase):
                    def test_x(self): pass
            """,
                2,
            ),
            (
                """\
                from torch.testing._internal.common_utils import TestCase
                if True:
                    class TestFoo(TestCase):
                        def test_x(self): pass
            """,
                3,
            ),
            (
                """\
                from torch.testing._internal.common_utils import TestCase
                try:
                    import missing_module
                except ImportError:
                    class TestFoo(TestCase):
                        def test_x(self): pass
            """,
                5,
            ),
            (
                """\
                from torch.testing._internal.common_utils import TestCase
                class TestFoo(TestCase):
                    hw_classification = "GENERIC"
                    def test_x(self): pass
            """,
                2,
            ),
            (
                """\
                from torch.testing._internal.common_utils import HardwareClassification, TestCase
                class TestFoo(TestCase):
                    hw_classification: HardwareClassification
                    def test_x(self): pass
            """,
                2,
            ),
            (
                """\
                from torch.testing._internal.common_utils import TestCase
                class TestFoo(TestCase):
                    def testBar(self): pass
            """,
                2,
            ),
            (
                """\
                from torch.testing._internal.common_utils import TestCase
                class TestFoo(TestCase):
                    async def test_x(self): pass
            """,
                2,
            ),
        ]
        description = (
            "Test class 'TestFoo': missing or invalid hw_classification."
            "\nSee the 'hw_classification' rule summary in tools/linter/adapters/test_linter.py for details."
        )
        for src, line in variants:
            msgs = self._run(src)
            self.assertEqual(len(msgs), 1, src)
            self.assertEqual(
                msgs[0],
                error_msg(
                    name="[hw_classification]",
                    path=msgs[0].path,
                    line=line,
                    description=description,
                ),
                src,
            )

    def test_missing_hw_classification_instantiated_class(self) -> None:
        """An instantiated class gets the same unified message; the linter does
        not guess the classification from instantiation."""
        src = """\
            from torch.testing._internal.common_utils import TestCase, instantiate_device_type_tests
            class TestFoo(TestCase):
                def test_x(self, device): pass
            instantiate_device_type_tests(TestFoo, globals())
        """
        msgs = self._run(src)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(
            msgs[0],
            error_msg(
                name="[hw_classification]",
                path=msgs[0].path,
                line=2,
                description=(
                    "Test class 'TestFoo': missing or invalid hw_classification."
                    "\nSee the 'hw_classification' rule summary in tools/linter/adapters/test_linter.py for details."
                ),
            ),
        )

    # --- non-test classes ---

    def test_no_test_methods_class_not_flagged(self) -> None:
        """A class with no test_* methods is not a test class and should be ignored."""
        src = """\
            from torch.testing._internal.common_utils import TestCase
            class TestMixin(TestCase):
                def helper(self): pass
        """
        self.assertEqual(self._run(src), [])

    # --- Test method shape: camelCase / async / except-handler scanning

    def test_mixed_snake_camel_methods_device_param_checked(self) -> None:
        """camelCase test methods in a mixed class are still checked per-method."""
        src = """\
            from torch.testing._internal.common_utils import HardwareClassification, TestCase
            class TestFoo(TestCase):
                hw_classification = HardwareClassification.GENERIC
                def test_snake(self): pass
                def testCamel(self, device): pass
        """
        msgs = self._run(src)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(
            msgs[0],
            error_msg(
                name="[device_param]",
                path=msgs[0].path,
                line=5,
                description=(
                    f"Test method 'TestFoo.testCamel' ({HC.GENERIC.value}): "
                    f"must not accept a 'device' or 'devices' parameter."
                    f"\nSee the 'device_param' rule summary in tools/linter/adapters/test_linter.py for details."
                ),
            ),
        )

    # --- GENERIC

    def test_valid_generic_classification(self) -> None:
        src = """\
            from torch.testing._internal.common_utils import HardwareClassification, TestCase
            class TestFoo(TestCase):
                hw_classification = HardwareClassification.GENERIC
                def test_x(self): pass
        """
        self.assertEqual(self._run(src), [])

    def test_valid_annotated_hw_classification(self) -> None:
        """The annotated form hw_classification: HardwareClassification = ... is valid."""
        src = """\
            from torch.testing._internal.common_utils import HardwareClassification, TestCase
            class TestFoo(TestCase):
                hw_classification: HardwareClassification = HardwareClassification.GENERIC
                def test_x(self): pass
        """
        self.assertEqual(self._run(src), [])

    def test_generic_classification_with_device_param(self) -> None:
        for param in ("device", "devices"):
            src = f"""\
                from torch.testing._internal.common_utils import HardwareClassification, TestCase
                class TestFoo(TestCase):
                    hw_classification = HardwareClassification.GENERIC
                    def test_x(self, {param}): pass
            """
            msgs = self._run(src)
            self.assertEqual(len(msgs), 1, f"failed with {param}")
            self.assertEqual(
                msgs[0],
                error_msg(
                    name="[device_param]",
                    path=msgs[0].path,
                    line=4,
                    description=(
                        f"Test method 'TestFoo.test_x' ({HC.GENERIC.value}): "
                        f"must not accept a 'device' or 'devices' parameter."
                        f"\nSee the 'device_param' rule summary in tools/linter/adapters/test_linter.py for details."
                    ),
                ),
            )

    def test_generic_classification_with_instantiated(self) -> None:
        src = """\
            from torch.testing._internal.common_device_type import instantiate_device_type_tests
            from torch.testing._internal.common_utils import HardwareClassification, TestCase
            class TestFoo(TestCase):
                hw_classification = HardwareClassification.GENERIC
                def test_x(self): pass
            instantiate_device_type_tests(TestFoo, globals())
        """
        msgs = self._run(src)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(
            msgs[0],
            error_msg(
                name="[instantiation]",
                path=msgs[0].path,
                line=3,
                description=(
                    f"Test class 'TestFoo' ({HC.GENERIC.value}): must not be used "
                    f"with instantiate_device_type_tests."
                    f"\nSee the 'instantiation' rule summary in tools/linter/adapters/test_linter.py for details."
                ),
            ),
        )

    def test_generic_forbidden_accelerator_api(self) -> None:
        for check in (
            "torch.cuda.is_available()",
            "torch.backends.mps.is_available()",
            "torch.cuda.synchronize()",
            "torch.cuda.device_count()",
            "torch.xpu.empty_cache()",
        ):
            src = f"""\
                from torch.testing._internal.common_utils import HardwareClassification, TestCase
                class TestFoo(TestCase):
                    hw_classification = HardwareClassification.GENERIC
                    def test_x(self):
                        {check}
            """
            msgs = self._run(src)
            self.assertEqual(len(msgs), 1, f"failed for {check}")
            self.assertEqual(
                msgs[0],
                error_msg(
                    name="[accelerator_api]",
                    path=msgs[0].path,
                    line=5,
                    description=(
                        f"Test class 'TestFoo' ({HC.GENERIC.value}): must not use "
                        f"accelerator API '{check}' in 'TestFoo.test_x'."
                        f"\nSee the 'accelerator_api' rule summary in "
                        f"tools/linter/adapters/test_linter.py for details."
                    ),
                ),
            )

    def test_generic_accelerator_api_in_setup(self) -> None:
        """The whole class body is scanned, and non-test methods are named."""
        src = """\
            from torch.testing._internal.common_utils import HardwareClassification, TestCase
            class TestFoo(TestCase):
                hw_classification = HardwareClassification.GENERIC
                def setUp(self):
                    if not torch.cuda.is_available():
                        self.skipTest("no cuda")
                def test_x(self): pass
        """
        msgs = self._run(src)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(
            msgs[0],
            error_msg(
                name="[accelerator_api]",
                path=msgs[0].path,
                line=5,
                description=(
                    f"Test class 'TestFoo' ({HC.GENERIC.value}): must not use "
                    f"accelerator API 'torch.cuda.is_available()' in 'TestFoo.setUp'."
                    f"\nSee the 'accelerator_api' rule summary in "
                    f"tools/linter/adapters/test_linter.py for details."
                ),
            ),
        )

    def test_generic_accelerator_api_in_class_decorator(self) -> None:
        """Class-level decorators are scanned too, not just the class body."""
        src = """\
            from torch.testing._internal.common_utils import HardwareClassification, TestCase
            import unittest
            @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
            class TestFoo(TestCase):
                hw_classification = HardwareClassification.GENERIC
                def test_x(self): pass
        """
        msgs = self._run(src)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(
            msgs[0],
            error_msg(
                name="[accelerator_api]",
                path=msgs[0].path,
                line=3,
                description=(
                    f"Test class 'TestFoo' ({HC.GENERIC.value}): must not use "
                    f"accelerator API 'torch.cuda.is_available()' in 'TestFoo'."
                    f"\nSee the 'accelerator_api' rule summary in "
                    f"tools/linter/adapters/test_linter.py for details."
                ),
            ),
        )

    # --- ACCELERATOR

    def test_valid_accelerator_basic(self) -> None:
        for param in ("device", "devices"):
            src = f"""\
                from torch.testing._internal.common_device_type import instantiate_device_type_tests
                from torch.testing._internal.common_utils import HardwareClassification, TestCase
                class TestFoo(TestCase):
                    hw_classification = HardwareClassification.ACCELERATOR
                    def test_x(self, {param}): pass
                instantiate_device_type_tests(TestFoo, globals())
            """
            self.assertEqual(self._run(src), [], f"failed with {param}")

    def test_valid_accelerator_only_accelerator_decorator(self) -> None:
        src = """\
            from torch.testing._internal.common_device_type import instantiate_device_type_tests, onlyAccelerator
            from torch.testing._internal.common_utils import HardwareClassification, TestCase
            class TestFoo(TestCase):
                hw_classification = HardwareClassification.ACCELERATOR
                @onlyAccelerator
                def test_x(self, device): pass
            instantiate_device_type_tests(TestFoo, globals())
        """
        self.assertEqual(self._run(src), [])

    def test_accelerator_missing_device(self) -> None:
        src = """\
            from torch.testing._internal.common_device_type import instantiate_device_type_tests
            from torch.testing._internal.common_utils import HardwareClassification, TestCase
            class TestFoo(TestCase):
                hw_classification = HardwareClassification.ACCELERATOR
                def test_x(self): pass
            instantiate_device_type_tests(TestFoo, globals())
        """
        msgs = self._run(src)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(
            msgs[0],
            error_msg(
                name="[device_param]",
                path=msgs[0].path,
                line=5,
                description=(
                    f"Test method 'TestFoo.test_x' ({HC.ACCELERATOR.value}): "
                    f"must accept a 'device' or 'devices' parameter."
                    f"\nSee the 'device_param' rule summary in tools/linter/adapters/test_linter.py for details."
                ),
            ),
        )

    def test_accelerator_not_instantiated(self) -> None:
        src = """\
            from torch.testing._internal.common_utils import HardwareClassification, TestCase
            class TestFoo(TestCase):
                hw_classification = HardwareClassification.ACCELERATOR
                def test_x(self, device): pass
        """
        msgs = self._run(src)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(
            msgs[0],
            error_msg(
                name="[instantiation]",
                path=msgs[0].path,
                line=2,
                description=(
                    f"Test class 'TestFoo' ({HC.ACCELERATOR.value}): "
                    f"must be used with instantiate_device_type_tests."
                    f"\nSee the 'instantiation' rule summary in tools/linter/adapters/test_linter.py for details."
                ),
            ),
        )

    def test_accelerator_forbidden_decorator(self) -> None:
        for bad_dec in (
            "onlyCPU",
            "onlyCUDA",
            "onlyMPS",
            "onlyXPU",
            "onlyHPU",
            "onlyPRIVATEUSE1",
            "onlyCUDAAndPRIVATEUSE1",
            "onlyNativeDeviceTypes",
        ):
            src = f"""\
                from torch.testing._internal.common_device_type import instantiate_device_type_tests, {bad_dec}
                from torch.testing._internal.common_utils import HardwareClassification, TestCase
                class TestFoo(TestCase):
                    hw_classification = HardwareClassification.ACCELERATOR
                    @{bad_dec}
                    def test_x(self, device): pass
                instantiate_device_type_tests(TestFoo, globals())
            """
            msgs = self._run(src)
            self.assertEqual(len(msgs), 1, f"failed for {bad_dec}")
            self.assertEqual(
                msgs[0],
                error_msg(
                    name="[decorator]",
                    path=msgs[0].path,
                    line=6,
                    description=(
                        f"Test method 'TestFoo.test_x' ({HC.ACCELERATOR.value}): "
                        f"must not use '@{bad_dec}'; only '@onlyAccelerator' is allowed."
                        f"\nSee the 'decorator' rule summary in tools/linter/adapters/test_linter.py for details."
                    ),
                ),
            )

    def test_accelerator_forbidden_call_decorator(self) -> None:
        """Call-form only* decorators (ast.Call nodes) must be detected."""
        for dec_name, dec_args in [
            ("onlyOn", '["cuda", "cpu"]'),
            ("onlyNativeDeviceTypesAnd", '["hpu"]'),
        ]:
            src = f"""\
                from torch.testing._internal.common_device_type import instantiate_device_type_tests, {dec_name}
                from torch.testing._internal.common_utils import HardwareClassification, TestCase
                class TestFoo(TestCase):
                    hw_classification = HardwareClassification.ACCELERATOR
                    @{dec_name}({dec_args})
                    def test_x(self, device): pass
                instantiate_device_type_tests(TestFoo, globals())
            """
            msgs = self._run(src)
            self.assertEqual(len(msgs), 1, f"failed for {dec_name}")
            self.assertEqual(
                msgs[0],
                error_msg(
                    name="[decorator]",
                    path=msgs[0].path,
                    line=6,
                    description=(
                        f"Test method 'TestFoo.test_x' ({HC.ACCELERATOR.value}): "
                        f"must not use '@{dec_name}'; only '@onlyAccelerator' is allowed."
                        f"\nSee the 'decorator' rule summary in tools/linter/adapters/test_linter.py for details."
                    ),
                ),
            )

    def test_accelerator_allowed_non_device_decorators(self) -> None:
        """Non-device decorators (dtypes, skipIfTorchDynamo) survive the deny-set check."""
        src = """\
            from torch.testing._internal.common_device_type import instantiate_device_type_tests
            from torch.testing._internal.common_utils import HardwareClassification, TestCase, dtypes, skipIfTorchDynamo
            class TestFoo(TestCase):
                hw_classification = HardwareClassification.ACCELERATOR
                @dtypes(torch.float32, torch.float64)
                @skipIfTorchDynamo
                def test_x(self, device): pass
            instantiate_device_type_tests(TestFoo, globals())
        """
        self.assertEqual(self._run(src), [])

    def test_accelerator_uses_only_for(self) -> None:
        src = """\
            from torch.testing._internal.common_device_type import instantiate_device_type_tests
            from torch.testing._internal.common_utils import HardwareClassification, TestCase
            class TestFoo(TestCase):
                hw_classification = HardwareClassification.ACCELERATOR
                def test_x(self, device): pass
            instantiate_device_type_tests(TestFoo, globals(), only_for='cuda')
        """
        msgs = self._run(src)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(
            msgs[0],
            error_msg(
                name="[only_for]",
                path=msgs[0].path,
                line=6,
                description=(
                    f"Test class 'TestFoo' ({HC.ACCELERATOR.value}): "
                    f"must not use only_for in instantiate_device_type_tests; use except_for instead."
                    f"\nSee the 'only_for' rule summary in tools/linter/adapters/test_linter.py for details."
                ),
            ),
        )

    # --- rule registry ---

    def test_rule_registry(self) -> None:
        """Every RuleId has exactly one rule, each with a summary.

        HwClassificationRule is the gate and runs before dispatch, so it is
        not in the dispatch table; include it explicitly.
        """
        all_rules = {rule for group in test_linter.rules.values() for rule in group}
        all_rules.add(test_linter.HwClassificationRule)
        self.assertEqual(len(all_rules), len(RuleId))
        self.assertEqual({rule.id for rule in all_rules}, set(RuleId))
        for rule in all_rules:
            self.assertTrue(rule.summary, rule.id)

    def test_all_message_names_come_from_registry(self) -> None:
        """Every message check_file() emits uses a registered rule name, and the
        kitchen-sink file exercises all check_file categories."""
        src = """\
            from torch.testing._internal.common_device_type import instantiate_device_type_tests, onlyCUDA
            from torch.testing._internal.common_utils import HardwareClassification, TestCase
            class TestMissingHw(TestCase):
                def test_x(self): pass
            class TestGeneric(TestCase):
                hw_classification = HardwareClassification.GENERIC
                def test_x(self, device): pass
                def test_y(self):
                    if not torch.cuda.is_available():
                        self.skipTest("no cuda")
            class TestAccel(TestCase):
                hw_classification = HardwareClassification.ACCELERATOR
                def test_x(self): pass
            class TestAccelDecorator(TestCase):
                hw_classification = HardwareClassification.ACCELERATOR
                @onlyCUDA
                def test_x(self, device): pass
            instantiate_device_type_tests(TestAccelDecorator, globals(), only_for='cuda')
            instantiate_device_type_tests(TestGeneric, globals())
        """
        registered = {
            rule.id.value for group in test_linter.rules.values() for rule in group
        }
        registered.add(test_linter.HwClassificationRule.id.value)
        msgs = self._run(src)
        emitted = {msg.name.removeprefix("[").removesuffix("]") for msg in msgs}
        self.assertEqual(
            emitted,
            {
                "hw_classification",
                "device_param",
                "instantiation",
                "accelerator_api",
                "decorator",
                "only_for",
            },
        )
        self.assertTrue(emitted <= registered)

    # --- Allowlist regeneration (--regenerate)

    def test_is_test_file(self) -> None:
        """Only files matching [linter.TEST_LINTER]'s patterns are linted."""

        def path_in_repo(name: str) -> str:
            return str(REPO_ROOT / name)

        for name in (
            "test/test_foo.py",
            "test/nested/test_foo.py",
            "test/foo_test.py",
        ):
            self.assertTrue(test_linter._is_test_file(path_in_repo(name)), name)
        for name in (
            "test/util.py",
            "torch/testing/test_foo.py",
            "tools/test/test_test_linter.py",
            "test/cpython/test_foo.py",
            "test/cpp_extensions/open_registration_extension/test_foo.py",
            "test/package/test_trace_dep/__init__.py",
        ):
            self.assertFalse(test_linter._is_test_file(path_in_repo(name)), name)

    def test_is_test_file_agrees_with_discovery(self) -> None:
        """Every file _discover_files() returns must satisfy _is_test_file().

        Otherwise a file is linted in normal runs but invisible to
        --regenerate, so it can neither be allowlisted nor kept out of it.
        """
        discovered = {
            path.relative_to(REPO_ROOT).as_posix()
            for path in test_linter._discover_files()
        }
        self.assertNotIn("test/package/test_trace_dep/__init__.py", discovered)
        for rel_path in discovered:
            self.assertTrue(
                test_linter._is_test_file(str(REPO_ROOT / rel_path)), rel_path
            )

    def test_regenerate_writes_only_failing_files(self) -> None:
        """regenerate_allowlist rebuilds the file from actual lint results.

        It clears the module-global _allowlist and writes ALLOWLIST_PATH, so
        both are patched: otherwise the real allowlist file gets overwritten
        and the global stays empty for the rest of the run.
        """
        with tempfile.TemporaryDirectory(dir=str(REPO_ROOT / "test")) as td:
            root = Path(td)
            passing = root / "test_regenerate_passing.py"
            failing = root / "test_regenerate_failing.py"
            # Only the AST is inspected, so the source needs no imports.
            _write(
                passing,
                textwrap.dedent(
                    """\
                    class TestPassing:
                        hw_classification = HardwareClassification.GENERIC
                        def test_x(self): pass
                    """
                ),
            )
            _write(failing, "class TestFailing:\n    def test_x(self): pass\n")

            allowlist_path = root / "allowlist.json"
            with (
                mock.patch.object(test_linter, "ALLOWLIST_PATH", allowlist_path),
                mock.patch.object(
                    test_linter, "_discover_files", return_value=[passing, failing]
                ),
                mock.patch.object(test_linter, "_allowlist", {"test/stale.py"}),
                mock.patch("builtins.print"),
            ):
                test_linter.regenerate_allowlist()

            # The passing file is excluded, the failing one is listed, and the
            # pre-existing stale entry is not carried over.
            self.assertEqual(
                json.loads(allowlist_path.read_text(encoding="utf-8")),
                [failing.relative_to(REPO_ROOT).as_posix()],
            )

    def test_regenerate_rejects_filenames(self) -> None:
        """--regenerate discovers test files itself, so filenames are rejected."""
        with mock.patch.object(
            sys, "argv", ["test_linter.py", "--regenerate", "test/foo_test.py"]
        ):
            with self.assertRaises(SystemExit) as cm:
                main()
        self.assertEqual(cm.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
