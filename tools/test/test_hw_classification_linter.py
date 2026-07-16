"""Tests for hw_classification_linter."""

from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from tools.linter.adapters.hw_classification_linter import (
    _check_file,
    HW_CLASSIFICATION_ATTR,
    HW_CLASSIFICATION_ENUM_CLASS,
    LINTER_CODE,
    LintMessage,
    LintSeverity,
)


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


class TestHwClassificationLinter(unittest.TestCase):
    def _run(self, content: str) -> list[LintMessage]:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            test_file = root / "test_sample.py"
            _write(test_file, textwrap.dedent(content))
            return _check_file(str(test_file))

    def _msg(self, path: str, line: int, desc: str) -> LintMessage:
        return LintMessage(
            path=path,
            line=line,
            char=None,
            code=LINTER_CODE,
            severity=LintSeverity.ERROR,
            name=f"[{HW_CLASSIFICATION_ATTR}]",
            original=None,
            replacement=None,
            description=desc,
        )

    # case1: missing hw_classification attribute
    def test_missing_classification(self) -> None:
        src = """\
            from torch.testing._internal.common_utils import TestCase
            class TestFoo(TestCase):
                def test_x(self): pass
        """
        msgs = self._run(src)
        self.assertEqual(len(msgs), 1)
        path = msgs[0].path
        self.assertIsNotNone(path)
        self.assertEqual(
            msgs[0],
            self._msg(
                path,
                2,
                f"Test class 'TestFoo' must declare {HW_CLASSIFICATION_ATTR} = {HW_CLASSIFICATION_ENUM_CLASS}.<MEMBER>.",
            ),
        )

    # case2: GENERIC
    def test_valid_generic(self) -> None:
        src = f"""\
            from torch.testing._internal.common_utils import HardwareClassification, TestCase
            class TestFoo(TestCase):
                {HW_CLASSIFICATION_ATTR} = {HW_CLASSIFICATION_ENUM_CLASS}.GENERIC
                def test_x(self): pass
        """
        msgs = self._run(src)
        self.assertEqual(msgs, [])

    # case3: CUDA
    def test_valid_cuda(self) -> None:
        src = f"""\
            from torch.testing._internal.common_utils import HardwareClassification, TestCase
            class TestFoo(TestCase):
                {HW_CLASSIFICATION_ATTR} = {HW_CLASSIFICATION_ENUM_CLASS}.CUDA
                def test_x(self): pass
        """
        msgs = self._run(src)
        self.assertEqual(msgs, [])

    # case4: ACCELERATOR with device param + instantiate
    def test_valid_accelerator(self) -> None:
        src = f"""\
            from torch.testing._internal.common_device_type import instantiate_device_type_tests
            from torch.testing._internal.common_utils import HardwareClassification, TestCase
            class TestFoo(TestCase):
                {HW_CLASSIFICATION_ATTR} = {HW_CLASSIFICATION_ENUM_CLASS}.ACCELERATOR
                def test_x(self, device): pass
            instantiate_device_type_tests(TestFoo, globals())
        """
        msgs = self._run(src)
        self.assertEqual(msgs, [])

    def test_accelerator_with_devices_param(self) -> None:
        src = f"""\
            from torch.testing._internal.common_device_type import instantiate_device_type_tests
            from torch.testing._internal.common_utils import HardwareClassification, TestCase
            class TestFoo(TestCase):
                {HW_CLASSIFICATION_ATTR} = {HW_CLASSIFICATION_ENUM_CLASS}.ACCELERATOR
                def test_x(self, devices): pass
            instantiate_device_type_tests(TestFoo, globals())
        """
        msgs = self._run(src)
        self.assertEqual(msgs, [])

    # case5: ACCELERATOR missing device param
    def test_accelerator_missing_device(self) -> None:
        src = f"""\
            from torch.testing._internal.common_device_type import instantiate_device_type_tests
            from torch.testing._internal.common_utils import HardwareClassification, TestCase
            class TestFoo(TestCase):
                {HW_CLASSIFICATION_ATTR} = {HW_CLASSIFICATION_ENUM_CLASS}.ACCELERATOR
                def test_x(self): pass
            instantiate_device_type_tests(TestFoo, globals())
        """
        msgs = self._run(src)
        self.assertEqual(len(msgs), 1)
        path = msgs[0].path
        self.assertIsNotNone(path)
        self.assertEqual(
            msgs[0],
            self._msg(
                path,
                5,
                "ACCELERATOR test method 'TestFoo.test_x' must accept a 'device' or 'devices' parameter.",
            ),
        )

    # case6: ACCELERATOR not instantiated
    def test_accelerator_not_instantiated(self) -> None:
        src = f"""\
            from torch.testing._internal.common_utils import HardwareClassification, TestCase
            class TestFoo(TestCase):
                {HW_CLASSIFICATION_ATTR} = {HW_CLASSIFICATION_ENUM_CLASS}.ACCELERATOR
                def test_x(self, device): pass
        """
        msgs = self._run(src)
        self.assertEqual(len(msgs), 1)
        path = msgs[0].path
        self.assertIsNotNone(path)
        self.assertEqual(
            msgs[0],
            self._msg(
                path,
                2,
                "ACCELERATOR class 'TestFoo' must be instantiated via 'instantiate_device_type_tests'.",
            ),
        )

    # case7: invalid enum value
    def test_invalid_enum_value(self) -> None:
        src = f"""\
            from torch.testing._internal.common_utils import TestCase
            class TestFoo(TestCase):
                {HW_CLASSIFICATION_ATTR} = "GENERIC"
                def test_x(self): pass
        """
        msgs = self._run(src)
        self.assertEqual(len(msgs), 1)
        path = msgs[0].path
        self.assertIsNotNone(path)
        self.assertEqual(
            msgs[0],
            self._msg(
                path,
                3,
                f"Could not determine {HW_CLASSIFICATION_ATTR} value for class 'TestFoo'. "
                f"Use '{HW_CLASSIFICATION_ENUM_CLASS}.<MEMBER>'.",
            ),
        )


if __name__ == "__main__":
    unittest.main()
