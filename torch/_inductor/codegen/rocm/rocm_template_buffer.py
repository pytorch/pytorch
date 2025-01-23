from collections.abc import Sequence
from typing import Callable, TypeVar
from typing_extensions import ParamSpec

from ...ir import Buffer, Layout, TemplateBuffer


_P = ParamSpec("_P")
_T = TypeVar("_T")


class ROCmTemplateBuffer(TemplateBuffer):
    def __init__(
        self,
        layout: Layout,
        inputs: Sequence[Buffer],
        make_kernel_render: Callable[_P, _T],
        workspace_size: int,
        template: "ROCmTemplate",  # type: ignore[name-defined]  # noqa: F821
    ) -> None:
        super().__init__(layout, inputs, make_kernel_render)
        # Global memory (in bytes) needed for this template.
        self.workspace_size = workspace_size
        self.template = template

    def get_workspace_size(self) -> int:
        return self.workspace_size if self.workspace_size is not None else 0
