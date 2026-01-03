# Owner(s): ["oncall: jit"]

import os
import sys
import warnings
from typing import Dict, List, Optional

import torch


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


class TestScriptModuleInstanceAttributeTypeAnnotation(JitTestCase):
    # NB: There are no tests for `Tuple` or `NamedTuple` here. In fact,
    # reassigning a non-empty Tuple to an attribute previously typed
    # as containing an empty Tuple SHOULD fail. See note in `_check.py`

    def test_annotated_falsy_base_type(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.x: int = 0

            def forward(self, x: int):
                self.x = x
                return 1

        with warnings.catch_warnings(record=True) as w:
            self.checkModule(M(), (1,))
        assert len(w) == 0

    def test_annotated_nonempty_container(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.x: List[int] = [1, 2, 3]

            def forward(self, x: List[int]):
                self.x = x
                return 1

        with warnings.catch_warnings(record=True) as w:
            self.checkModule(M(), ([1, 2, 3],))
        assert len(w) == 0

    def test_annotated_empty_tensor(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.x: torch.Tensor = torch.empty(0)

            def forward(self, x: torch.Tensor):
                self.x = x
                return self.x

        with warnings.catch_warnings(record=True) as w:
            self.checkModule(M(), (torch.rand(2, 3),))
        assert len(w) == 0

    def test_annotated_with_jit_attribute(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.x = torch.jit.Attribute([], List[int])

            def forward(self, x: List[int]):
                self.x = x
                return self.x

        with warnings.catch_warnings(record=True) as w:
            self.checkModule(M(), ([1, 2, 3],))
        assert len(w) == 0

    def test_annotated_class_level_annotation_only(self):
        class M(torch.nn.Module):
            x: List[int]

            def __init__(self) -> None:
                super().__init__()
                self.x = []

            def forward(self, y: List[int]):
                self.x = y
                return self.x

        with warnings.catch_warnings(record=True) as w:
            self.checkModule(M(), ([1, 2, 3],))
        assert len(w) == 0

    def test_annotated_class_level_annotation_and_init_annotation(self):
        class M(torch.nn.Module):
            x: List[int]

            def __init__(self) -> None:
                super().__init__()
                self.x: List[int] = []

            def forward(self, y: List[int]):
                self.x = y
                return self.x

        with warnings.catch_warnings(record=True) as w:
            self.checkModule(M(), ([1, 2, 3],))
        assert len(w) == 0

    def test_annotated_class_level_jit_annotation(self):
        class M(torch.nn.Module):
            x: List[int]

            def __init__(self) -> None:
                super().__init__()
                self.x: List[int] = torch.jit.annotate(List[int], [])

            def forward(self, y: List[int]):
                self.x = y
                return self.x

        with warnings.catch_warnings(record=True) as w:
            self.checkModule(M(), ([1, 2, 3],))
        assert len(w) == 0

    def test_annotated_empty_list(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.x: List[int] = []

            def forward(self, x: List[int]):
                self.x = x
                return 1

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Tried to set nonexistent attribute", "self.x = x"
        ):
            with self.assertWarnsRegex(
                UserWarning,
                "doesn't support instance-level annotations on empty non-base types",
            ):
                torch.jit.script(M())

    def test_annotated_empty_list_lowercase(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.x: list[int] = []

            def forward(self, x: list[int]):
                self.x = x
                return 1

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Tried to set nonexistent attribute", "self.x = x"
        ):
            with self.assertWarnsRegex(
                UserWarning,
                "doesn't support instance-level annotations on empty non-base types",
            ):
                torch.jit.script(M())

    def test_annotated_empty_dict(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.x: Dict[str, int] = {}

            def forward(self, x: Dict[str, int]):
                self.x = x
                return 1

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Tried to set nonexistent attribute", "self.x = x"
        ):
            with self.assertWarnsRegex(
                UserWarning,
                "doesn't support instance-level annotations on empty non-base types",
            ):
                torch.jit.script(M())

    def test_annotated_empty_dict_lowercase(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.x: dict[str, int] = {}

            def forward(self, x: dict[str, int]):
                self.x = x
                return 1

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Tried to set nonexistent attribute", "self.x = x"
        ):
            with self.assertWarnsRegex(
                UserWarning,
                "doesn't support instance-level annotations on empty non-base types",
            ):
                torch.jit.script(M())

    def test_annotated_empty_optional(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.x: Optional[str] = None

            def forward(self, x: Optional[str]):
                self.x = x
                return 1

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Wrong type for attribute assignment", "self.x = x"
        ):
            with self.assertWarnsRegex(
                UserWarning,
                "doesn't support instance-level annotations on empty non-base types",
            ):
                torch.jit.script(M())

    def test_annotated_with_jit_empty_list(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.x = torch.jit.annotate(List[int], [])

            def forward(self, x: List[int]):
                self.x = x
                return 1

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Tried to set nonexistent attribute", "self.x = x"
        ):
            with self.assertWarnsRegex(
                UserWarning,
                "doesn't support instance-level annotations on empty non-base types",
            ):
                torch.jit.script(M())

    def test_annotated_with_jit_empty_list_lowercase(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.x = torch.jit.annotate(list[int], [])

            def forward(self, x: list[int]):
                self.x = x
                return 1

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Tried to set nonexistent attribute", "self.x = x"
        ):
            with self.assertWarnsRegex(
                UserWarning,
                "doesn't support instance-level annotations on empty non-base types",
            ):
                torch.jit.script(M())

    def test_annotated_with_jit_empty_dict(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.x = torch.jit.annotate(Dict[str, int], {})

            def forward(self, x: Dict[str, int]):
                self.x = x
                return 1

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Tried to set nonexistent attribute", "self.x = x"
        ):
            with self.assertWarnsRegex(
                UserWarning,
                "doesn't support instance-level annotations on empty non-base types",
            ):
                torch.jit.script(M())

    def test_annotated_with_jit_empty_dict_lowercase(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.x = torch.jit.annotate(dict[str, int], {})

            def forward(self, x: dict[str, int]):
                self.x = x
                return 1

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Tried to set nonexistent attribute", "self.x = x"
        ):
            with self.assertWarnsRegex(
                UserWarning,
                "doesn't support instance-level annotations on empty non-base types",
            ):
                torch.jit.script(M())

    def test_annotated_with_jit_empty_optional(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.x = torch.jit.annotate(Optional[str], None)

            def forward(self, x: Optional[str]):
                self.x = x
                return 1

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Wrong type for attribute assignment", "self.x = x"
        ):
            with self.assertWarnsRegex(
                UserWarning,
                "doesn't support instance-level annotations on empty non-base types",
            ):
                torch.jit.script(M())

    def test_annotated_with_torch_jit_import(self):
        from torch import jit

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.x = jit.annotate(Optional[str], None)

            def forward(self, x: Optional[str]):
                self.x = x
                return 1

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Wrong type for attribute assignment", "self.x = x"
        ):
            with self.assertWarnsRegex(
                UserWarning,
                "doesn't support instance-level annotations on empty non-base types",
            ):
                torch.jit.script(M())


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
