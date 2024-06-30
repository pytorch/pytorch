# mypy: allow-untyped-defs
from ...ir import TemplateBuffer


class ROCmTemplateBuffer(TemplateBuffer):
    def __init__(
        self,
        layout,
        inputs,
        make_kernel_render,
        workspace_size: int,
        template: "ROCmTemplate",  # type: ignore[name-defined]  # noqa: F821
    ):
        super().__init__(layout, inputs, make_kernel_render)
        # Global memory (in bytes) needed for this template.
        self.workspace_size = workspace_size
        self.template = template

    def get_workspace_size(self):
        return self.workspace_size if self.workspace_size is not None else 0
