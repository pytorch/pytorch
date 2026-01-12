import types

import torch
import torch.utils._pytree as pytree
from torch.export import ExportedProgram
from torch.export.pt2_archive._package import AOTI_FILES, package_pt2
from torch.types import FileLike
from ._lowered_aoti_module import LoweredBackendModule


def get_new_ep_with_flat_inputs_outputs(ep: ExportedProgram) -> ExportedProgram:
    class FlattenedModule(torch.nn.Module):
        def __init__(
            self,
            original_module: torch.fx.GraphModule,
            in_spec: pytree.TreeSpec,
            out_spec: pytree.TreeSpec,
        ) -> None:
            super().__init__()
            self.original_module = original_module
            self.in_spec = in_spec
            self.out_spec = out_spec

        def forward(self, *flat_inputs):  # type: ignore[no-untyped-def]
            # Unflatten inputs to original structure
            inputs = pytree.tree_unflatten(flat_inputs, self.in_spec)
            args, kwargs = inputs
            outputs = self.original_module(*args, **kwargs)
            # Flatten outputs
            flat_outputs, _ = pytree.tree_flatten(outputs)
            return tuple(flat_outputs)

    flattened_module = FlattenedModule(
        ep.module(), ep.call_spec.in_spec, ep.call_spec.out_spec
    )
    args, kwargs = ep.example_inputs
    flat_inputs, _ = pytree.tree_flatten((args, kwargs))
    flat_ep = torch.export.export(flattened_module, tuple(flat_inputs))

    return flat_ep


def lower_exported_program(
    exported_program: ExportedProgram, model_name: str, backend_id: str
) -> tuple[ExportedProgram, AOTI_FILES]:
    """
    Lower an exported program to AOTInductor and return a delegate ExportedProgram
    with the `executorch_call_delegate` HOP
    """
    args, kwargs = exported_program.example_inputs
    out_spec = exported_program.call_spec.out_spec
    flat_ep = get_new_ep_with_flat_inputs_outputs(exported_program)
    flat_inputs, _ = pytree.tree_flatten((args, kwargs))

    aoti_files = torch._inductor.aot_compile(
        flat_ep.module(), tuple(flat_inputs), options={"aot_inductor.package": True}
    )
    if not isinstance(aoti_files, list):
        raise AssertionError(
            f"aoti_files must be a list, got {type(aoti_files).__name__}"
        )

    lowered_aoti_module = LoweredBackendModule(
        flat_ep, backend_id, module_name=model_name
    )

    def patched_forward(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        flat_inputs, _ = pytree.tree_flatten((args, kwargs))
        flat_outputs = torch._higher_order_ops.executorch_call_delegate(
            self, *flat_inputs
        )
        if out_spec is not None and flat_outputs is not None:
            return pytree.tree_unflatten(flat_outputs, out_spec)
        else:
            return flat_outputs

    lowered_aoti_module.forward = types.MethodType(patched_forward, lowered_aoti_module)  # type: ignore[method-assign]

    aoti_delegate_ep = torch.export.export(lowered_aoti_module, args, kwargs)

    return aoti_delegate_ep, aoti_files


def package_nativert_with_aoti_delegate(
    f: FileLike,
    model_name: str,
    backend_id: str,
    original_ep: ExportedProgram,
    delegate_ep: ExportedProgram,
    delegate_files: AOTI_FILES,
) -> None:
    """
    Package a pt2 archive file that can be consumed by NativeRT with AOTI Delegate
    """
    package_pt2(
        f,
        exported_programs={
            model_name: original_ep,
            f"{model_name}-{backend_id}": delegate_ep,
        },
        aoti_files={f"{model_name}-{backend_id}": delegate_files},  # type: ignore[dict-item]
    )
    return
