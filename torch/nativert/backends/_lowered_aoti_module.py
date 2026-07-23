import torch
from torch.export import ExportedProgram


class LoweredBackendModule(torch.nn.Module):
    def __init__(
        self,
        original_exported_program: ExportedProgram,
        backend_id: str,
        *,
        module_name: str | None = None,
    ) -> None:
        super().__init__()
        self._backend_id = backend_id
        self._module_name = module_name
        self._original_exported_program = original_exported_program

    @property
    def backend_id(self) -> str:
        return self._backend_id

    @property
    def module_name(self) -> str | None:
        return self._module_name

    @property
    def original_module(self) -> ExportedProgram:
        return self._original_exported_program

    def forward(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return torch._higher_order_ops.executorch_call_delegate(self, *args, **kwargs)
