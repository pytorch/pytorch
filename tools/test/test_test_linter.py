"""Tests for test_linter."""

from __future__ import annotations

import os
import tempfile
import textwrap
import unittest
from pathlib import Path

from tools.linter.adapters import test_linter
from tools.linter.adapters.test_linter import (
    check_file,
    DEVICE_SPECIFIC_CLASSIFICATIONS,
    error_msg,
    HardwareClassification,
    LintMessage,
    LintSeverity,
    REPO_ROOT,
)


HC = HardwareClassification


def _allow_kwarg(cls: HardwareClassification) -> str:
    if cls is HardwareClassification.MPS:
        return ", allow_mps=True"
    if cls is HardwareClassification.XPU:
        return ", allow_xpu=True"
    return ""


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


class TestHwClassificationLinter(unittest.TestCase):
    def _run(self, content: str) -> list[LintMessage]:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            test_file = root / "test_sample.py"
            _write(test_file, textwrap.dedent(content))
            return check_file(str(test_file))

    # --- file parse errors ---

    def test_syntax_error_in_file(self) -> None:
        """A file with invalid syntax should produce a parse error, not crash."""
        src = "this is not valid python @@@"
        msgs = self._run(src)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].name, "[parse_error]")
        self.assertIsNotNone(msgs[0].description)
        self.assertIn("Failed to parse", msgs[0].description)

    def test_error_msg_defaults(self) -> None:
        """Pin error_msg defaults so tests don't silently inherit a wrong severity/code."""
        msg = error_msg(name="[test]", path="some.py", line=None, description="boom")
        self.assertEqual(msg.severity, LintSeverity.ERROR)
        self.assertEqual(msg.code, "TEST_LINTER")

    # --- allowlist / non-test files ---

    def test_non_test_file_skipped(self) -> None:
        """Non-test files (not test_*.py or *_test.py) return no messages."""
        msgs = check_file("some_util.py")
        self.assertEqual(msgs, [])

    def test_allowlisted_file_skipped(self) -> None:
        """Files in the allowlist are skipped silently."""
        src = """\
            from torch.testing._internal.common_utils import TestCase
            class TestFoo(TestCase):
                def test_x(self): pass
        """
        with tempfile.TemporaryDirectory() as td:
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

    def test_missing_classification(self) -> None:
        src = """\
            from torch.testing._internal.common_utils import TestCase
            class TestFoo(TestCase):
                def test_x(self): pass
        """
        msgs = self._run(src)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(
            msgs[0],
            error_msg(
                name="[hw_classification]",
                path=msgs[0].path,
                line=2,
                description="Test class 'TestFoo' is missing or has an invalid "
                "hw_classification. Only the exact forms below are accepted "
                "(aliased imports are not recognized):\n"
                "    hw_classification = HardwareClassification.<MEMBER>\n"
                "    hw_classification: HardwareClassification = HardwareClassification.<MEMBER>",
            ),
        )

    def test_class_under_if_guard_missing_classification(self) -> None:
        """Conditionally defined test classes are still scanned for classification."""
        src = """\
            from torch.testing._internal.common_utils import TestCase
            if True:
                class TestFoo(TestCase):
                    def test_x(self): pass
        """
        msgs = self._run(src)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(
            msgs[0],
            error_msg(
                name="[hw_classification]",
                path=msgs[0].path,
                line=3,
                description="Test class 'TestFoo' is missing or has an invalid "
                "hw_classification. Only the exact forms below are accepted "
                "(aliased imports are not recognized):\n"
                "    hw_classification = HardwareClassification.<MEMBER>\n"
                "    hw_classification: HardwareClassification = HardwareClassification.<MEMBER>",
            ),
        )

    def test_class_under_nested_if_guard_missing_classification(self) -> None:
        """Test classes inside nested if bodies are still scanned for classification."""
        src = """\
            from torch.testing._internal.common_utils import TestCase
            if True:
                if True:
                    class TestFoo(TestCase):
                        def test_x(self): pass
        """
        msgs = self._run(src)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(
            msgs[0],
            error_msg(
                name="[hw_classification]",
                path=msgs[0].path,
                line=4,
                description="Test class 'TestFoo' is missing or has an invalid "
                "hw_classification. Only the exact forms below are accepted "
                "(aliased imports are not recognized):\n"
                "    hw_classification = HardwareClassification.<MEMBER>\n"
                "    hw_classification: HardwareClassification = HardwareClassification.<MEMBER>",
            ),
        )

    def test_class_under_try_guard_missing_classification(self) -> None:
        """Test classes inside a try body are still scanned for classification."""
        src = """\
            from torch.testing._internal.common_utils import TestCase
            try:
                class TestFoo(TestCase):
                    def test_x(self): pass
            except Exception:
                pass
        """
        msgs = self._run(src)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(
            msgs[0],
            error_msg(
                name="[hw_classification]",
                path=msgs[0].path,
                line=3,
                description="Test class 'TestFoo' is missing or has an invalid "
                "hw_classification. Only the exact forms below are accepted "
                "(aliased imports are not recognized):\n"
                "    hw_classification = HardwareClassification.<MEMBER>\n"
                "    hw_classification: HardwareClassification = HardwareClassification.<MEMBER>",
            ),
        )

    def test_invalid_enum_value(self) -> None:
        src = """\
            from torch.testing._internal.common_utils import TestCase
            class TestFoo(TestCase):
                hw_classification = "GENERIC"
                def test_x(self): pass
        """
        msgs = self._run(src)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(
            msgs[0],
            error_msg(
                name="[hw_classification]",
                path=msgs[0].path,
                line=2,
                description="Test class 'TestFoo' is missing or has an invalid "
                "hw_classification. Only the exact forms below are accepted "
                "(aliased imports are not recognized):\n"
                "    hw_classification = HardwareClassification.<MEMBER>\n"
                "    hw_classification: HardwareClassification = HardwareClassification.<MEMBER>",
            ),
        )

    def test_annotation_without_value(self) -> None:
        src = """\
            from torch.testing._internal.common_utils import HardwareClassification, TestCase
            class TestFoo(TestCase):
                hw_classification: HardwareClassification
                def test_x(self): pass
        """
        msgs = self._run(src)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(
            msgs[0],
            error_msg(
                name="[hw_classification]",
                path=msgs[0].path,
                line=2,
                description="Test class 'TestFoo' is missing or has an invalid "
                "hw_classification. Only the exact forms below are accepted "
                "(aliased imports are not recognized):\n"
                "    hw_classification = HardwareClassification.<MEMBER>\n"
                "    hw_classification: HardwareClassification = HardwareClassification.<MEMBER>",
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

    # ==================================================================
    # Test method shape: camelCase / async / except-handler scanning
    # ==================================================================

    def test_camel_case_only_class_missing_classification(self) -> None:
        """A class whose tests are all camelCase (testFoo) is still a test class."""
        src = """\
            from torch.testing._internal.common_utils import TestCase
            class TestFoo(TestCase):
                def testBar(self): pass
        """
        msgs = self._run(src)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(
            msgs[0],
            error_msg(
                name="[hw_classification]",
                path=msgs[0].path,
                line=2,
                description="Test class 'TestFoo' is missing or has an invalid "
                "hw_classification. Only the exact forms below are accepted "
                "(aliased imports are not recognized):\n"
                "    hw_classification = HardwareClassification.<MEMBER>\n"
                "    hw_classification: HardwareClassification = HardwareClassification.<MEMBER>",
            ),
        )

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
                description=f"{HC.GENERIC.value} test method 'TestFoo.testCamel' "
                f"must not accept a 'device' or 'devices' parameter.",
            ),
        )

    def test_class_under_except_handler_scanned(self) -> None:
        """Test classes inside an except handler are still scanned."""
        src = """\
            from torch.testing._internal.common_utils import TestCase
            try:
                import missing_module
            except ImportError:
                class TestFoo(TestCase):
                    def test_x(self): pass
        """
        msgs = self._run(src)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(
            msgs[0],
            error_msg(
                name="[hw_classification]",
                path=msgs[0].path,
                line=5,
                description="Test class 'TestFoo' is missing or has an invalid "
                "hw_classification. Only the exact forms below are accepted "
                "(aliased imports are not recognized):\n"
                "    hw_classification = HardwareClassification.<MEMBER>\n"
                "    hw_classification: HardwareClassification = HardwareClassification.<MEMBER>",
            ),
        )

    def test_async_test_method_classified(self) -> None:
        """async def test_* methods make a class a test class."""
        src = """\
            from torch.testing._internal.common_utils import TestCase
            class TestFoo(TestCase):
                async def test_x(self): pass
        """
        msgs = self._run(src)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(
            msgs[0],
            error_msg(
                name="[hw_classification]",
                path=msgs[0].path,
                line=2,
                description="Test class 'TestFoo' is missing or has an invalid "
                "hw_classification. Only the exact forms below are accepted "
                "(aliased imports are not recognized):\n"
                "    hw_classification = HardwareClassification.<MEMBER>\n"
                "    hw_classification: HardwareClassification = HardwareClassification.<MEMBER>",
            ),
        )

    # ==================================================================
    # GENERIC
    # ==================================================================

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
                    description=f"{HC.GENERIC.value} test method 'TestFoo.test_x' "
                    f"must not accept a 'device' or 'devices' parameter.",
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
                description=f"{HC.GENERIC.value} class 'TestFoo' must not be "
                f"instantiated via 'instantiate_device_type_tests'.",
            ),
        )

    def test_generic_classification_instantiated_and_with_device_param(self) -> None:
        src = """\
            from torch.testing._internal.common_device_type import instantiate_device_type_tests
            from torch.testing._internal.common_utils import HardwareClassification, TestCase
            class TestFoo(TestCase):
                hw_classification = HardwareClassification.GENERIC
                def test_x(self, device): pass
            instantiate_device_type_tests(TestFoo, globals())
        """
        msgs = self._run(src)
        self.assertEqual(len(msgs), 2)
        self.assertEqual(
            msgs[0],
            error_msg(
                name="[instantiation]",
                path=msgs[0].path,
                line=3,
                description=f"{HC.GENERIC.value} class 'TestFoo' must not be "
                f"instantiated via 'instantiate_device_type_tests'.",
            ),
        )
        self.assertEqual(
            msgs[1],
            error_msg(
                name="[device_param]",
                path=msgs[1].path,
                line=5,
                description=f"{HC.GENERIC.value} test method 'TestFoo.test_x' "
                f"must not accept a 'device' or 'devices' parameter.",
            ),
        )

    # ==================================================================
    # SPECIFIC (CPU, CUDA, MPS, XPU)
    # ==================================================================

    def test_valid_device_specific_classification(self) -> None:
        for cls in DEVICE_SPECIFIC_CLASSIFICATIONS:
            allow = _allow_kwarg(cls)
            src = f"""\
                from torch.testing._internal.common_device_type import instantiate_device_type_tests
                from torch.testing._internal.common_utils import HardwareClassification, TestCase
                class TestFoo(TestCase):
                    hw_classification = HardwareClassification.{cls.name}
                    def test_x(self, device): pass
                instantiate_device_type_tests(TestFoo, globals(), only_for='{cls.value.lower()}'{allow})
            """
            self.assertEqual(self._run(src), [], f"failed for {cls}")

    def test_valid_device_specific_classification_tuple_only_for(self) -> None:
        """only_for as a tuple is accepted like a string."""
        for cls in DEVICE_SPECIFIC_CLASSIFICATIONS:
            allow = _allow_kwarg(cls)
            src = f"""\
                from torch.testing._internal.common_device_type import instantiate_device_type_tests
                from torch.testing._internal.common_utils import HardwareClassification, TestCase
                class TestFoo(TestCase):
                    hw_classification = HardwareClassification.{cls.name}
                    def test_x(self, device): pass
                instantiate_device_type_tests(TestFoo, globals(), only_for=('{cls.value.lower()}',){allow})
            """
            self.assertEqual(self._run(src), [], f"failed for {cls}")

    def test_device_specific_missing_device(self) -> None:
        for cls in DEVICE_SPECIFIC_CLASSIFICATIONS:
            allow = _allow_kwarg(cls)
            src = f"""\
                from torch.testing._internal.common_device_type import instantiate_device_type_tests
                from torch.testing._internal.common_utils import HardwareClassification, TestCase
                class TestFoo(TestCase):
                    hw_classification = HardwareClassification.{cls.name}
                    def test_x(self): pass
                instantiate_device_type_tests(TestFoo, globals(), only_for='{cls.value.lower()}'{allow})
            """
            msgs = self._run(src)
            self.assertEqual(len(msgs), 1, f"failed for {cls}")
            self.assertEqual(
                msgs[0],
                error_msg(
                    name="[device_param]",
                    path=msgs[0].path,
                    line=5,
                    description=f"{cls.value} test method 'TestFoo.test_x' "
                    f"must accept a 'device' or 'devices' parameter.",
                ),
            )

    def test_device_specific_classification_not_instantiated(self) -> None:
        for cls in DEVICE_SPECIFIC_CLASSIFICATIONS:
            src = f"""\
                from torch.testing._internal.common_utils import HardwareClassification, TestCase
                class TestFoo(TestCase):
                    hw_classification = HardwareClassification.{cls.name}
                    def test_x(self, device): pass
            """
            msgs = self._run(src)
            self.assertEqual(len(msgs), 1, f"failed for {cls}")
            self.assertEqual(
                msgs[0],
                error_msg(
                    name="[instantiation]",
                    path=msgs[0].path,
                    line=2,
                    description=f"{cls.value} class 'TestFoo' must be "
                    f"instantiated via 'instantiate_device_type_tests'.",
                ),
            )

    def test_device_specific_instantiation_attribute_form(self) -> None:
        """Attribute-form instantiate_device_type_tests (module.func) is recognized."""
        for cls in DEVICE_SPECIFIC_CLASSIFICATIONS:
            allow = _allow_kwarg(cls)
            src = f"""\
                from torch.testing._internal import common_device_type
                from torch.testing._internal.common_utils import HardwareClassification, TestCase
                class TestFoo(TestCase):
                    hw_classification = HardwareClassification.{cls.name}
                    def test_x(self, device): pass
                common_device_type.instantiate_device_type_tests(TestFoo, globals(), only_for='{cls.value.lower()}'{allow})
            """
            self.assertEqual(self._run(src), [], f"failed for {cls}")

    def test_device_specific_instantiation_under_if_guard(self) -> None:
        """Guarded instantiation calls are found, so no false 'must be instantiated'."""
        for cls in DEVICE_SPECIFIC_CLASSIFICATIONS:
            allow = _allow_kwarg(cls)
            src = f"""\
                from torch.testing._internal.common_device_type import instantiate_device_type_tests
                from torch.testing._internal.common_utils import HardwareClassification, TestCase
                class TestFoo(TestCase):
                    hw_classification = HardwareClassification.{cls.name}
                    def test_x(self, device): pass
                if True:
                    instantiate_device_type_tests(TestFoo, globals(), only_for='{cls.value.lower()}'{allow})
            """
            self.assertEqual(self._run(src), [], f"failed for {cls}")

    def test_device_specific_instantiation_under_with_guard(self) -> None:
        """Instantiation calls inside a with body are also found."""
        for cls in DEVICE_SPECIFIC_CLASSIFICATIONS:
            allow = _allow_kwarg(cls)
            src = f"""\
                from torch.testing._internal.common_device_type import instantiate_device_type_tests
                from torch.testing._internal.common_utils import HardwareClassification, TestCase
                class TestFoo(TestCase):
                    hw_classification = HardwareClassification.{cls.name}
                    def test_x(self, device): pass
                with contextlib.suppress():
                    instantiate_device_type_tests(TestFoo, globals(), only_for='{cls.value.lower()}'{allow})
            """
            self.assertEqual(self._run(src), [], f"failed for {cls}")

    def test_device_specific_instantiation_under_nested_if_guard(self) -> None:
        """Instantiation calls under nested if guards are also found."""
        for cls in DEVICE_SPECIFIC_CLASSIFICATIONS:
            allow = _allow_kwarg(cls)
            src = f"""\
                from torch.testing._internal.common_device_type import instantiate_device_type_tests
                from torch.testing._internal.common_utils import HardwareClassification, TestCase
                class TestFoo(TestCase):
                    hw_classification = HardwareClassification.{cls.name}
                    def test_x(self, device): pass
                if True:
                    if True:
                        instantiate_device_type_tests(TestFoo, globals(), only_for='{cls.value.lower()}'{allow})
            """
            self.assertEqual(self._run(src), [], f"failed for {cls}")

    def test_device_specific_instantiation_under_try_guard(self) -> None:
        """Instantiation calls inside a try body are also found."""
        for cls in DEVICE_SPECIFIC_CLASSIFICATIONS:
            allow = _allow_kwarg(cls)
            src = f"""\
                from torch.testing._internal.common_device_type import instantiate_device_type_tests
                from torch.testing._internal.common_utils import HardwareClassification, TestCase
                class TestFoo(TestCase):
                    hw_classification = HardwareClassification.{cls.name}
                    def test_x(self, device): pass
                try:
                    instantiate_device_type_tests(TestFoo, globals(), only_for='{cls.value.lower()}'{allow})
                except Exception:
                    pass
            """
            self.assertEqual(self._run(src), [], f"failed for {cls}")

    def test_device_specific_not_instantiated_and_missing_device(self) -> None:
        for cls in DEVICE_SPECIFIC_CLASSIFICATIONS:
            src = f"""\
                from torch.testing._internal.common_utils import HardwareClassification, TestCase
                class TestFoo(TestCase):
                    hw_classification = HardwareClassification.{cls.name}
                    def test_x(self): pass
            """
            msgs = self._run(src)
            self.assertEqual(len(msgs), 2, f"failed for {cls}")
            self.assertEqual(
                msgs[0],
                error_msg(
                    name="[instantiation]",
                    path=msgs[0].path,
                    line=2,
                    description=f"{cls.value} class 'TestFoo' must be "
                    f"instantiated via 'instantiate_device_type_tests'.",
                ),
            )
            self.assertEqual(
                msgs[1],
                error_msg(
                    name="[device_param]",
                    path=msgs[1].path,
                    line=4,
                    description=f"{cls.value} test method 'TestFoo.test_x' "
                    f"must accept a 'device' or 'devices' parameter.",
                ),
            )

    def test_device_specific_missing_only_for(self) -> None:
        for cls in DEVICE_SPECIFIC_CLASSIFICATIONS:
            allow = _allow_kwarg(cls)
            src = f"""\
                from torch.testing._internal.common_device_type import instantiate_device_type_tests
                from torch.testing._internal.common_utils import HardwareClassification, TestCase
                class TestFoo(TestCase):
                    hw_classification = HardwareClassification.{cls.name}
                    def test_x(self, device): pass
                instantiate_device_type_tests(TestFoo, globals(){allow})
            """
            msgs = self._run(src)
            self.assertEqual(len(msgs), 1, f"failed for {cls}")
            self.assertEqual(
                msgs[0],
                error_msg(
                    name="[only_for]",
                    path=msgs[0].path,
                    line=6,
                    description=f"{cls.value} class 'TestFoo' must use "
                    f"only_for='{cls.value.lower()}' in instantiate_device_type_tests.",
                ),
            )

    def test_device_specific_wrong_only_for(self) -> None:
        for cls in DEVICE_SPECIFIC_CLASSIFICATIONS:
            allow = _allow_kwarg(cls)
            wrong = "cpu" if cls.value.lower() != "cpu" else "cuda"
            src = f"""\
                from torch.testing._internal.common_device_type import instantiate_device_type_tests
                from torch.testing._internal.common_utils import HardwareClassification, TestCase
                class TestFoo(TestCase):
                    hw_classification = HardwareClassification.{cls.name}
                    def test_x(self, device): pass
                instantiate_device_type_tests(TestFoo, globals(), only_for='{wrong}'{allow})
            """
            msgs = self._run(src)
            self.assertEqual(len(msgs), 1, f"failed for {cls}")
            self.assertEqual(
                msgs[0],
                error_msg(
                    name="[only_for]",
                    path=msgs[0].path,
                    line=6,
                    description=f"{cls.value} class 'TestFoo' "
                    f"has only_for values ['{wrong}'], "
                    f"but must be exactly ['{cls.value.lower()}'].",
                ),
            )

    def test_device_specific_non_literal_only_for(self) -> None:
        """only_for with a non-literal value reports a static-resolution error,
        not a misleading 'must use only_for=' message."""
        for cls in DEVICE_SPECIFIC_CLASSIFICATIONS:
            allow = _allow_kwarg(cls)
            src = f"""\
                from torch.testing._internal.common_device_type import instantiate_device_type_tests
                from torch.testing._internal.common_utils import HardwareClassification, TestCase
                MY_CONST = '{cls.value.lower()}'
                class TestFoo(TestCase):
                    hw_classification = HardwareClassification.{cls.name}
                    def test_x(self, device): pass
                instantiate_device_type_tests(TestFoo, globals(), only_for=MY_CONST{allow})
            """
            msgs = self._run(src)
            self.assertEqual(len(msgs), 1, f"failed for {cls}")
            self.assertEqual(
                msgs[0],
                error_msg(
                    name="[only_for]",
                    path=msgs[0].path,
                    line=7,
                    description=f"{cls.value} class 'TestFoo' "
                    f"has a non-literal only_for in instantiate_device_type_tests "
                    f"that could not be resolved statically; "
                    f"use a literal only_for='{cls.value.lower()}'.",
                ),
            )

    def test_device_specific_only_for_none(self) -> None:
        """only_for=None is treated as absent and triggers missing-only_for error."""
        for cls in DEVICE_SPECIFIC_CLASSIFICATIONS:
            allow = _allow_kwarg(cls)
            src = f"""\
                from torch.testing._internal.common_device_type import instantiate_device_type_tests
                from torch.testing._internal.common_utils import HardwareClassification, TestCase
                class TestFoo(TestCase):
                    hw_classification = HardwareClassification.{cls.name}
                    def test_x(self, device): pass
                instantiate_device_type_tests(TestFoo, globals(), only_for=None{allow})
            """
            msgs = self._run(src)
            self.assertEqual(len(msgs), 1, f"failed for {cls}")
            self.assertEqual(
                msgs[0],
                error_msg(
                    name="[only_for]",
                    path=msgs[0].path,
                    line=6,
                    description=f"{cls.value} class 'TestFoo' "
                    f"must use only_for='{cls.value.lower()}' "
                    f"in instantiate_device_type_tests.",
                ),
            )

    def test_device_specific_uses_except_for(self) -> None:
        for cls in DEVICE_SPECIFIC_CLASSIFICATIONS:
            allow = _allow_kwarg(cls)
            src = f"""\
                from torch.testing._internal.common_device_type import instantiate_device_type_tests
                from torch.testing._internal.common_utils import HardwareClassification, TestCase
                class TestFoo(TestCase):
                    hw_classification = HardwareClassification.{cls.name}
                    def test_x(self, device): pass
                instantiate_device_type_tests(TestFoo, globals(),
                    only_for='{cls.value.lower()}'{allow}, except_for='hpu')
            """
            msgs = self._run(src)
            self.assertEqual(len(msgs), 1, f"failed for {cls}")
            self.assertEqual(
                msgs[0],
                error_msg(
                    name="[except_for]",
                    path=msgs[0].path,
                    line=6,
                    description=f"{cls.value} class 'TestFoo' "
                    f"must not use except_for in instantiate_device_type_tests.",
                ),
            )

    def test_device_specific_except_for_none_treated_as_absent(self) -> None:
        """except_for=None must not trigger the except_for rule."""
        for cls in DEVICE_SPECIFIC_CLASSIFICATIONS:
            allow = _allow_kwarg(cls)
            src = f"""\
                from torch.testing._internal.common_device_type import instantiate_device_type_tests
                from torch.testing._internal.common_utils import HardwareClassification, TestCase
                class TestFoo(TestCase):
                    hw_classification = HardwareClassification.{cls.name}
                    def test_x(self, device): pass
                instantiate_device_type_tests(TestFoo, globals(),
                    only_for='{cls.value.lower()}'{allow}, except_for=None)
            """
            self.assertEqual(self._run(src), [], f"failed for {cls}")

    def test_mps_requires_allow_mps(self) -> None:
        """MPS classes must pass allow_mps=True, otherwise zero tests are generated."""
        src = """\
            from torch.testing._internal.common_device_type import instantiate_device_type_tests
            from torch.testing._internal.common_utils import HardwareClassification, TestCase
            class TestFoo(TestCase):
                hw_classification = HardwareClassification.MPS
                def test_x(self, device): pass
            instantiate_device_type_tests(TestFoo, globals(), only_for='mps')
        """
        msgs = self._run(src)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(
            msgs[0],
            error_msg(
                name="[allow_mps]",
                path=msgs[0].path,
                line=6,
                description=f"{HC.MPS.value} class 'TestFoo' "
                f"must use allow_mps=True in instantiate_device_type_tests, "
                f"otherwise no tests are generated.",
            ),
        )

    def test_mps_non_literal_allow_mps_skipped(self) -> None:
        """A non-literal allow_mps= must not produce a spurious [allow_mps] error."""
        src = """\
            from torch.testing._internal.common_device_type import instantiate_device_type_tests
            from torch.testing._internal.common_utils import HardwareClassification, TestCase
            ALLOW = True
            class TestFoo(TestCase):
                hw_classification = HardwareClassification.MPS
                def test_x(self, device): pass
            instantiate_device_type_tests(TestFoo, globals(), only_for='mps', allow_mps=ALLOW)
        """
        self.assertEqual(self._run(src), [])

    def test_mps_with_allow_mps(self) -> None:
        src = """\
            from torch.testing._internal.common_device_type import instantiate_device_type_tests
            from torch.testing._internal.common_utils import HardwareClassification, TestCase
            class TestFoo(TestCase):
                hw_classification = HardwareClassification.MPS
                def test_x(self, device): pass
            instantiate_device_type_tests(TestFoo, globals(), only_for='mps', allow_mps=True)
        """
        self.assertEqual(self._run(src), [])

    def test_xpu_requires_allow_xpu(self) -> None:
        """XPU classes must pass allow_xpu=True, otherwise zero tests are generated."""
        src = """\
            from torch.testing._internal.common_device_type import instantiate_device_type_tests
            from torch.testing._internal.common_utils import HardwareClassification, TestCase
            class TestFoo(TestCase):
                hw_classification = HardwareClassification.XPU
                def test_x(self, device): pass
            instantiate_device_type_tests(TestFoo, globals(), only_for='xpu')
        """
        msgs = self._run(src)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(
            msgs[0],
            error_msg(
                name="[allow_xpu]",
                path=msgs[0].path,
                line=6,
                description=f"{HC.XPU.value} class 'TestFoo' "
                f"must use allow_xpu=True in instantiate_device_type_tests, "
                f"otherwise no tests are generated.",
            ),
        )

    def test_xpu_with_allow_xpu(self) -> None:
        src = """\
            from torch.testing._internal.common_device_type import instantiate_device_type_tests
            from torch.testing._internal.common_utils import HardwareClassification, TestCase
            class TestFoo(TestCase):
                hw_classification = HardwareClassification.XPU
                def test_x(self, device): pass
            instantiate_device_type_tests(TestFoo, globals(), only_for='xpu', allow_xpu=True)
        """
        self.assertEqual(self._run(src), [])

    # ==================================================================
    # ACCELERATOR
    # ==================================================================

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

    def test_valid_accelerator_with_except_for(self) -> None:
        src = """\
            from torch.testing._internal.common_device_type import instantiate_device_type_tests
            from torch.testing._internal.common_utils import HardwareClassification, TestCase
            class TestFoo(TestCase):
                hw_classification = HardwareClassification.ACCELERATOR
                def test_x(self, device): pass
            instantiate_device_type_tests(TestFoo, globals(), except_for='hpu')
        """
        self.assertEqual(self._run(src), [])

    def test_accelerator_except_for_none_treated_as_absent(self) -> None:
        """except_for=None is semantically identical to omitting the kwarg."""
        src = """\
            from torch.testing._internal.common_device_type import instantiate_device_type_tests
            from torch.testing._internal.common_utils import HardwareClassification, TestCase
            class TestFoo(TestCase):
                hw_classification = HardwareClassification.ACCELERATOR
                def test_x(self, device): pass
            instantiate_device_type_tests(TestFoo, globals(), except_for=None)
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
                description=f"{HC.ACCELERATOR.value} test method 'TestFoo.test_x' "
                f"must accept a 'device' or 'devices' parameter.",
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
                description=f"{HC.ACCELERATOR.value} class 'TestFoo' must be "
                f"instantiated via 'instantiate_device_type_tests'.",
            ),
        )

    def test_accelerator_not_instantiated_and_missing_device(self) -> None:
        src = """\
            from torch.testing._internal.common_utils import HardwareClassification, TestCase
            class TestFoo(TestCase):
                hw_classification = HardwareClassification.ACCELERATOR
                def test_x(self): pass
        """
        msgs = self._run(src)
        self.assertEqual(len(msgs), 2)
        self.assertEqual(
            msgs[0],
            error_msg(
                name="[instantiation]",
                path=msgs[0].path,
                line=2,
                description=f"{HC.ACCELERATOR.value} class 'TestFoo' must be "
                f"instantiated via 'instantiate_device_type_tests'.",
            ),
        )
        self.assertEqual(
            msgs[1],
            error_msg(
                name="[device_param]",
                path=msgs[1].path,
                line=4,
                description=f"{HC.ACCELERATOR.value} test method 'TestFoo.test_x' "
                f"must accept a 'device' or 'devices' parameter.",
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
                    description=f"{HC.ACCELERATOR.value} test method 'TestFoo.test_x' "
                    f"must not use '@{bad_dec}' decorators except onlyAccelerator",
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
                    description=f"{HC.ACCELERATOR.value} test method 'TestFoo.test_x' "
                    f"must not use '@{dec_name}' decorators except onlyAccelerator",
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
                description=f"{HC.ACCELERATOR.value} class 'TestFoo' "
                f"must not use only_for in instantiate_device_type_tests. "
                f"Use except_for instead (blacklist approach).",
            ),
        )

    # --- duplicate class / instantiation detection ---

    def test_duplicate_class_definition(self) -> None:
        """Defining the same test class twice (under different guards) reports
        [duplicate_class]."""
        src = """\
            from torch.testing._internal.common_utils import HardwareClassification, TestCase
            if USE_FAST_PATH:
                class TestFoo(TestCase):
                    hw_classification = HardwareClassification.GENERIC
                    def test_x(self): pass
            if USE_SLOW_PATH:
                class TestFoo(TestCase):
                    hw_classification = HardwareClassification.GENERIC
                    def test_y(self): pass
        """
        msgs = self._run(src)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(
            msgs[0],
            error_msg(
                name="[duplicate_class]",
                path=msgs[0].path,
                line=7,
                description="Test class 'TestFoo' is defined more than once; "
                "only the last definition is linted.",
            ),
        )

    def test_duplicate_instantiation(self) -> None:
        """Calling instantiate_device_type_tests twice for one class reports
        [duplicate_instantiation]."""
        src = """\
            from torch.testing._internal.common_device_type import instantiate_device_type_tests
            from torch.testing._internal.common_utils import HardwareClassification, TestCase
            class TestFoo(TestCase):
                hw_classification = HardwareClassification.CUDA
                def test_x(self, device): pass
            instantiate_device_type_tests(TestFoo, globals(), only_for='cuda')
            instantiate_device_type_tests(TestFoo, globals(), only_for='cuda')
        """
        msgs = self._run(src)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(
            msgs[0],
            error_msg(
                name="[duplicate_instantiation]",
                path=msgs[0].path,
                line=7,
                description="Class 'TestFoo' is passed to "
                "instantiate_device_type_tests more than once; "
                "only the last call is linted.",
            ),
        )


if __name__ == "__main__":
    unittest.main()
