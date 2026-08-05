"""Tests for test_linter."""

from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from tools.linter.adapters.test_linter import (
    HardwareClassification,
    LintMessage,
    error_msg,
    check_file,
)

HC = HardwareClassification


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
            error_msg(
                name="[hw_classification]",
                path=msgs[0].path,
                line=2,
                description="Test class 'TestFoo' is missing or has an invalid "
                "hw_classification. Valid declarations:\n"
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
                "hw_classification. Valid declarations:\n"
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
        for cls in (HC.CPU, HC.CUDA, HC.MPS, HC.XPU):
            src = f"""\
                from torch.testing._internal.common_device_type import instantiate_device_type_tests
                from torch.testing._internal.common_utils import HardwareClassification, TestCase
                class TestFoo(TestCase):
                    hw_classification = HardwareClassification.{cls.name}
                    def test_x(self, device): pass
                instantiate_device_type_tests(TestFoo, globals(), only_for='{cls.value.lower()}')
            """
            self.assertEqual(self._run(src), [], f"failed for {cls}")

    def test_device_specific_missing_device(self) -> None:
        for cls in (HC.CPU, HC.CUDA, HC.MPS, HC.XPU):
            src = f"""\
                from torch.testing._internal.common_device_type import instantiate_device_type_tests
                from torch.testing._internal.common_utils import HardwareClassification, TestCase
                class TestFoo(TestCase):
                    hw_classification = HardwareClassification.{cls.name}
                    def test_x(self): pass
                instantiate_device_type_tests(TestFoo, globals(), only_for='{cls.value.lower()}')
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
        for cls in (HC.CPU, HC.CUDA, HC.MPS, HC.XPU):
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

    def test_device_specific_not_instantiated_and_missing_device(self) -> None:
        for cls in (HC.CPU, HC.CUDA, HC.MPS, HC.XPU):
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
        for cls in (HC.CPU, HC.CUDA, HC.MPS, HC.XPU):
            src = f"""\
                from torch.testing._internal.common_device_type import instantiate_device_type_tests
                from torch.testing._internal.common_utils import HardwareClassification, TestCase
                class TestFoo(TestCase):
                    hw_classification = HardwareClassification.{cls.name}
                    def test_x(self, device): pass
                instantiate_device_type_tests(TestFoo, globals())
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
        for cls in (HC.CPU, HC.CUDA, HC.MPS, HC.XPU):
            wrong = "cpu" if cls.value.lower() != "cpu" else "cuda"
            src = f"""\
                from torch.testing._internal.common_device_type import instantiate_device_type_tests
                from torch.testing._internal.common_utils import HardwareClassification, TestCase
                class TestFoo(TestCase):
                    hw_classification = HardwareClassification.{cls.name}
                    def test_x(self, device): pass
                instantiate_device_type_tests(TestFoo, globals(), only_for='{wrong}')
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

    def test_device_specific_uses_except_for(self) -> None:
        for cls in (HC.CPU, HC.CUDA, HC.MPS, HC.XPU):
            src = f"""\
                from torch.testing._internal.common_device_type import instantiate_device_type_tests
                from torch.testing._internal.common_utils import HardwareClassification, TestCase
                class TestFoo(TestCase):
                    hw_classification = HardwareClassification.{cls.name}
                    def test_x(self, device): pass
                instantiate_device_type_tests(TestFoo, globals(),
                    only_for='{cls.value.lower()}', except_for='hpu')
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
        for bad_dec in ("onlyCPU", "onlyCUDA", "onlyMPS", "onlyXPU"):
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


if __name__ == "__main__":
    unittest.main()
