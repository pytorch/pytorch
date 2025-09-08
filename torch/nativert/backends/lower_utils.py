import torch
from torch.export import ExportedProgram
from torch.export.pt2_archive._package import AOTI_FILES, package_pt2
from torch.types import FileLike

from .lowered_aoti_module import LoweredBackendModule


def lower_exported_program(
    exported_program: ExportedProgram, model_name: str, backend_id: str
) -> tuple[ExportedProgram, AOTI_FILES]:
    """
    Lower an exported program to AOTInductor and return a delegate ExportedProgram
    with the `executorch_call_delegate` HOP
    """
    args, kwargs = exported_program.example_inputs
    aoti_files = torch._inductor.aot_compile(
        exported_program.module(), args, kwargs, options={"aot_inductor.package": True}
    )
    assert isinstance(aoti_files, list)

    lowered_aoti_module = LoweredBackendModule(
        exported_program, backend_id, module_name=model_name
    )

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
