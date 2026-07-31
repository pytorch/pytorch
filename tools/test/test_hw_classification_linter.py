"""Tests for hw_classification_linter."""

from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from tools.linter.adapters.hw_classification_linter import (
    check_file,
    CPU,
    create_error_msg,
    CUDA,
    GENERIC,
    HW_CLASSIFICATION_ATTR,
    HW_CLASSIFICATION_ENUM_CLASS,
    LintMessage,
    MPS,
    XPU,
)


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


class TestHwClassificationLinter(unittest.TestCase):
    def _run(self, content: str) -> list[LintMessage]:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            test_file = root / "test_sample.py"
            _write(test_file, textwrap.dedent(content))
            return check_file(str(test_file))

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
            create_error_msg(
                msgs[0].path,
                2,
                f"Test class 'TestFoo' must declare {HW_CLASSIFICATION_ATTR} = {HW_CLASSIFICATION_ENUM_CLASS}.<MEMBER>.",
            ),
        )

    def test_invalid_enum_value(self) -> None:
        src = f"""\
            from torch.testing._internal.common_utils import TestCase
            class TestFoo(TestCase):
                {HW_CLASSIFICATION_ATTR} = "GENERIC"
                def test_x(self): pass
        """
        msgs = self._run(src)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(
            msgs[0],
            create_error_msg(
                msgs[0].path,
                3,
                f"Could not determine {HW_CLASSIFICATION_ATTR} value for class 'TestFoo'. "
                f"Use '{HW_CLASSIFICATION_ENUM_CLASS}.<MEMBER>'.",
            ),
        )

    # ==================================================================
    # GENERIC
    # ==================================================================

    def test_valid_generic_classification(self) -> None:
        src = f"""\
            from torch.testing._internal.common_utils import HardwareClassification, TestCase
            class TestFoo(TestCase):
                {HW_CLASSIFICATION_ATTR} = {HW_CLASSIFICATION_ENUM_CLASS}.{GENERIC}
                def test_x(self): pass
        """
        self.assertEqual(self._run(src), [])

    def test_generic_classification_with_device_param(self) -> None:
        for param in ("device", "devices"):
            src = f"""\
                from torch.testing._internal.common_utils import HardwareClassification, TestCase
                class TestFoo(TestCase):
                    {HW_CLASSIFICATION_ATTR} = {HW_CLASSIFICATION_ENUM_CLASS}.{GENERIC}
                    def test_x(self, {param}): pass
            """
            msgs = self._run(src)
            self.assertEqual(len(msgs), 1, f"failed with {param}")
            self.assertEqual(
                msgs[0],
                create_error_msg(
                    msgs[0].path,
                    4,
                    f"{GENERIC} test method 'TestFoo.test_x' must not accept a 'device' or 'devices' parameter.",
                ),
            )

    def test_generic_classification_with_instantiated(self) -> None:
        src = f"""\
            from torch.testing._internal.common_device_type import instantiate_device_type_tests
            from torch.testing._internal.common_utils import HardwareClassification, TestCase
            class TestFoo(TestCase):
                {HW_CLASSIFICATION_ATTR} = {HW_CLASSIFICATION_ENUM_CLASS}.{GENERIC}
                def test_x(self): pass
            instantiate_device_type_tests(TestFoo, globals())
        """
        msgs = self._run(src)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(
            msgs[0],
            create_error_msg(
                msgs[0].path,
                3,
                f"{GENERIC} class 'TestFoo' must not be instantiated via 'instantiate_device_type_tests'.",
            ),
        )

    def test_generic_classification_instantiated_and_with_device_param(self) -> None:
        src = f"""\
            from torch.testing._internal.common_device_type import instantiate_device_type_tests
            from torch.testing._internal.common_utils import HardwareClassification, TestCase
            class TestFoo(TestCase):
                {HW_CLASSIFICATION_ATTR} = {HW_CLASSIFICATION_ENUM_CLASS}.{GENERIC}
                def test_x(self, device): pass
            instantiate_device_type_tests(TestFoo, globals())
        """
        msgs = self._run(src)
        self.assertEqual(len(msgs), 2)

        self.assertEqual(
            msgs[0],
            create_error_msg(
                msgs[0].path,
                3,
                f"{GENERIC} class 'TestFoo' must not be instantiated via 'instantiate_device_type_tests'.",
            ),
        )
        self.assertEqual(
            msgs[1],
            create_error_msg(
                msgs[1].path,
                5,
                f"{GENERIC} test method 'TestFoo.test_x' must not accept a 'device' or 'devices' parameter.",
            ),
        )

    # ==================================================================
    # SPECIFIC (CPU, CUDA, MPS, XPU)
    # ==================================================================

    def test_valid_device_specific_classification(self) -> None:
        for classification in (CPU, CUDA, MPS, XPU):
            src = f"""\
                from torch.testing._internal.common_device_type import instantiate_device_type_tests
                from torch.testing._internal.common_utils import HardwareClassification, TestCase
                class TestFoo(TestCase):
                    {HW_CLASSIFICATION_ATTR} = {HW_CLASSIFICATION_ENUM_CLASS}.{classification}
                    def test_x(self, device): pass
                instantiate_device_type_tests(TestFoo, globals(), only_for='{classification.lower()}')
            """
            self.assertEqual(self._run(src), [], f"failed for {classification}")

    def test_device_specific_missing_device(self) -> None:
        for classification in (CPU, CUDA, MPS, XPU):
            src = f"""\
                from torch.testing._internal.common_device_type import instantiate_device_type_tests
                from torch.testing._internal.common_utils import HardwareClassification, TestCase
                class TestFoo(TestCase):
                    {HW_CLASSIFICATION_ATTR} = {HW_CLASSIFICATION_ENUM_CLASS}.{classification}
                    def test_x(self): pass
                instantiate_device_type_tests(TestFoo, globals(), only_for='{classification.lower()}')
            """
            msgs = self._run(src)
            self.assertEqual(len(msgs), 1, f"failed for {classification}")
            self.assertEqual(
                msgs[0],
                create_error_msg(
                    msgs[0].path,
                    5,
                    f"{classification} test method 'TestFoo.test_x' must accept a 'device' or 'devices' parameter.",
                ),
            )

    def test_device_specific_classification_not_instantiated(self) -> None:
        for classification in (CPU, CUDA, MPS, XPU):
            src = f"""\
                from torch.testing._internal.common_utils import HardwareClassification, TestCase
                class TestFoo(TestCase):
                    {HW_CLASSIFICATION_ATTR} = {HW_CLASSIFICATION_ENUM_CLASS}.{classification}
                    def test_x(self, device): pass
            """
            msgs = self._run(src)
            self.assertEqual(len(msgs), 1, f"failed for {classification}")
            self.assertEqual(
                msgs[0],
                create_error_msg(
                    msgs[0].path,
                    2,
                    f"{classification} class 'TestFoo' must be instantiated via 'instantiate_device_type_tests'.",
                ),
            )

    def test_device_specific_not_instantiated_and_missing_device(self) -> None:
        for classification in (CPU, CUDA, MPS, XPU):
            src = f"""\
                from torch.testing._internal.common_utils import HardwareClassification, TestCase
                class TestFoo(TestCase):
                    {HW_CLASSIFICATION_ATTR} = {HW_CLASSIFICATION_ENUM_CLASS}.{classification}
                    def test_x(self): pass
            """
            msgs = self._run(src)
            self.assertEqual(len(msgs), 2, f"failed for {classification}")
            self.assertEqual(
                msgs[0],
                create_error_msg(
                    msgs[0].path,
                    2,
                    f"{classification} class 'TestFoo' must be instantiated via 'instantiate_device_type_tests'.",
                ),
            )
            self.assertEqual(
                msgs[1],
                create_error_msg(
                    msgs[1].path,
                    4,
                    f"{classification} test method 'TestFoo.test_x' must accept a 'device' or 'devices' parameter.",
                ),
            )

    def test_device_specific_missing_only_for(self) -> None:
        for classification in (CPU, CUDA, MPS, XPU):
            src = f"""\
                from torch.testing._internal.common_device_type import instantiate_device_type_tests
                from torch.testing._internal.common_utils import HardwareClassification, TestCase
                class TestFoo(TestCase):
                    {HW_CLASSIFICATION_ATTR} = {HW_CLASSIFICATION_ENUM_CLASS}.{classification}
                    def test_x(self, device): pass
                instantiate_device_type_tests(TestFoo, globals())
            """
            msgs = self._run(src)
            self.assertEqual(len(msgs), 1, f"failed for {classification}")
            self.assertEqual(
                msgs[0],
                create_error_msg(
                    msgs[0].path,
                    6,
                    f"{classification} class 'TestFoo' must use only_for='{classification.lower()}' "
                    f"in instantiate_device_type_tests.",
                ),
            )

    def test_device_specific_wrong_only_for(self) -> None:
        for classification in (CPU, CUDA, MPS, XPU):
            wrong = "cpu" if classification.lower() != "cpu" else "cuda"
            src = f"""\
                from torch.testing._internal.common_device_type import instantiate_device_type_tests
                from torch.testing._internal.common_utils import HardwareClassification, TestCase
                class TestFoo(TestCase):
                    {HW_CLASSIFICATION_ATTR} = {HW_CLASSIFICATION_ENUM_CLASS}.{classification}
                    def test_x(self, device): pass
                instantiate_device_type_tests(TestFoo, globals(), only_for='{wrong}')
            """
            msgs = self._run(src)
            self.assertEqual(len(msgs), 1, f"failed for {classification}")
            self.assertEqual(
                msgs[0],
                create_error_msg(
                    msgs[0].path,
                    6,
                    f"{classification} class 'TestFoo' "
                    f"has only_for values ['{wrong}'], "
                    f"but must be exactly ['{classification.lower()}'].",
                ),
            )

    def test_device_specific_uses_except_for(self) -> None:
        for classification in (CPU, CUDA, MPS, XPU):
            src = f"""\
                from torch.testing._internal.common_device_type import instantiate_device_type_tests
                from torch.testing._internal.common_utils import HardwareClassification, TestCase
                class TestFoo(TestCase):
                    {HW_CLASSIFICATION_ATTR} = {HW_CLASSIFICATION_ENUM_CLASS}.{classification}
                    def test_x(self, device): pass
                instantiate_device_type_tests(TestFoo, globals(), only_for='{classification.lower()}', except_for='hpu')
            """
            msgs = self._run(src)
            self.assertEqual(len(msgs), 1, f"failed for {classification}")
            self.assertEqual(
                msgs[0],
                create_error_msg(
                    msgs[0].path,
                    6,
                    f"{classification} class 'TestFoo' must not use except_for in instantiate_device_type_tests.",
                ),
            )

    # ==================================================================
    # ACCELERATOR
    # ==================================================================

    # --- valid ---

    def test_valid_accelerator_basic(self) -> None:
        for param in ("device", "devices"):
            src = f"""\
                from torch.testing._internal.common_device_type import instantiate_device_type_tests
                from torch.testing._internal.common_utils import HardwareClassification, TestCase
                class TestFoo(TestCase):
                    {HW_CLASSIFICATION_ATTR} = {HW_CLASSIFICATION_ENUM_CLASS}.ACCELERATOR
                    def test_x(self, {param}): pass
                instantiate_device_type_tests(TestFoo, globals())
            """
            self.assertEqual(self._run(src), [], f"failed with {param}")

    def test_valid_accelerator_only_accelerator_decorator(self) -> None:
        src = f"""\
            from torch.testing._internal.common_device_type import instantiate_device_type_tests, onlyAccelerator
            from torch.testing._internal.common_utils import HardwareClassification, TestCase
            class TestFoo(TestCase):
                {HW_CLASSIFICATION_ATTR} = {HW_CLASSIFICATION_ENUM_CLASS}.ACCELERATOR
                @onlyAccelerator
                def test_x(self, device): pass
            instantiate_device_type_tests(TestFoo, globals())
        """
        self.assertEqual(self._run(src), [])

    def test_valid_accelerator_with_except_for(self) -> None:
        src = f"""\
            from torch.testing._internal.common_device_type import instantiate_device_type_tests
            from torch.testing._internal.common_utils import HardwareClassification, TestCase
            class TestFoo(TestCase):
                {HW_CLASSIFICATION_ATTR} = {HW_CLASSIFICATION_ENUM_CLASS}.ACCELERATOR
                def test_x(self, device): pass
            instantiate_device_type_tests(TestFoo, globals(), except_for='hpu')
        """
        self.assertEqual(self._run(src), [])

    # --- invalid ---

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
        self.assertEqual(
            msgs[0],
            create_error_msg(
                msgs[0].path,
                5,
                "ACCELERATOR test method 'TestFoo.test_x' must accept a 'device' or 'devices' parameter.",
            ),
        )

    def test_accelerator_not_instantiated(self) -> None:
        src = f"""\
            from torch.testing._internal.common_utils import HardwareClassification, TestCase
            class TestFoo(TestCase):
                {HW_CLASSIFICATION_ATTR} = {HW_CLASSIFICATION_ENUM_CLASS}.ACCELERATOR
                def test_x(self, device): pass
        """
        msgs = self._run(src)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(
            msgs[0],
            create_error_msg(
                msgs[0].path,
                2,
                "ACCELERATOR class 'TestFoo' must be instantiated via 'instantiate_device_type_tests'.",
            ),
        )

    def test_accelerator_not_instantiated_and_missing_device(self) -> None:
        src = f"""\
            from torch.testing._internal.common_utils import HardwareClassification, TestCase
            class TestFoo(TestCase):
                {HW_CLASSIFICATION_ATTR} = {HW_CLASSIFICATION_ENUM_CLASS}.ACCELERATOR
                def test_x(self): pass
        """
        msgs = self._run(src)
        self.assertEqual(len(msgs), 2)
        self.assertEqual(
            msgs[0],
            create_error_msg(
                msgs[0].path,
                2,
                "ACCELERATOR class 'TestFoo' must be instantiated via 'instantiate_device_type_tests'.",
            ),
        )
        self.assertEqual(
            msgs[1],
            create_error_msg(
                msgs[1].path,
                4,
                "ACCELERATOR test method 'TestFoo.test_x' must accept a 'device' or 'devices' parameter.",
            ),
        )

    def test_accelerator_forbidden_decorator(self) -> None:
        for bad_dec in ("onlyCPU", "onlyCUDA", "onlyMPS", "onlyXPU"):
            src = f"""\
                from torch.testing._internal.common_device_type import instantiate_device_type_tests, {bad_dec}
                from torch.testing._internal.common_utils import HardwareClassification, TestCase
                class TestFoo(TestCase):
                    {HW_CLASSIFICATION_ATTR} = {HW_CLASSIFICATION_ENUM_CLASS}.ACCELERATOR
                    @{bad_dec}
                    def test_x(self, device): pass
                instantiate_device_type_tests(TestFoo, globals())
            """
            msgs = self._run(src)
            self.assertEqual(len(msgs), 1, f"failed for {bad_dec}")
            self.assertEqual(
                msgs[0],
                create_error_msg(
                    msgs[0].path,
                    6,
                    f"ACCELERATOR test method 'TestFoo.test_x' must not use '@{bad_dec}' decorators except onlyAccelerator",
                ),
            )

    def test_accelerator_uses_only_for(self) -> None:
        src = f"""\
            from torch.testing._internal.common_device_type import instantiate_device_type_tests
            from torch.testing._internal.common_utils import HardwareClassification, TestCase
            class TestFoo(TestCase):
                {HW_CLASSIFICATION_ATTR} = {HW_CLASSIFICATION_ENUM_CLASS}.ACCELERATOR
                def test_x(self, device): pass
            instantiate_device_type_tests(TestFoo, globals(), only_for='cuda')
        """
        msgs = self._run(src)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(
            msgs[0],
            create_error_msg(
                msgs[0].path,
                6,
                "ACCELERATOR class 'TestFoo' must not use only_for in instantiate_device_type_tests. "
                "Use except_for instead (blacklist approach).",
            ),
        )


if __name__ == "__main__":
    unittest.main()
