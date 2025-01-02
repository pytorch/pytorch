from typing import Callable, Optional, Sequence
from typing_extensions import Self

from ...ir import IRNode, Layout, TemplateBuffer


class ROCmTemplateBuffer(TemplateBuffer):
    def __init__(
        self,
        layout: Layout,
        inputs: Sequence[IRNode],
        make_kernel_render: Callable[[Self, Optional[Sequence[IRNode]]], str],
        workspace_size: int,
        template: "ROCmTemplate",  # type: ignore[name-defined]  # noqa: F821
    ) -> None:
        super().__init__(layout, inputs, make_kernel_render)
        # Global memory (in bytes) needed for this template.
        self.workspace_size = workspace_size
        self.template = template

    def get_workspace_size(self) -> int:
        return self.workspace_size if self.workspace_size is not None else 0
