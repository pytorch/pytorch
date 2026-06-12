from __future__ import annotations

import torchgen.api.meta as meta
import torchgen.api.structured as structured
from torchgen.api.types import kernel_signature
from torchgen.context import with_native_function_and_index
from torchgen.model import (
    BackendIndex,
    DispatchKey,
    is_cuda_dispatch_key,
    is_xpu_dispatch_key,
    NativeFunction,
    NativeFunctionsGroup,
)
from torchgen.utils import mapMaybe


# native_functions.yaml tag: see tags.yaml ("cpu_dll_cuda_kernel").
_CPU_DLL_CUDA_KERNEL_TAG = "cpu_dll_cuda_kernel"
# QuantizedCUDA dispatch implemented in torch_cpu-only TU; tag applies only to
# QuantizedCUDA forward decl macro selection (see dll_export_macro_for_kernel).
_CPU_DLL_QUANTIZED_CUDA_KERNEL_TAG = "cpu_dll_quantized_cuda_kernel"


# Dispatch key → TORCH_* for generated forward decls (Windows). Also used from register_dispatch_key.
def torch_api_key_word_prefix(backend_index: BackendIndex) -> str:
    if backend_index.external:
        return ""

    # Use DispatchKey predicates, not only the bare "CUDA"/"XPU" enum names: NestedTensorCUDA,
    # SparseCUDA, etc. still use the CUDA component DLL and need TORCH_CUDA_CPP_API on Windows.
    # A name-only map left those on TORCH_API and caused -Winconsistent-dllimport vs HIP/CUDA TUs.
    dk = backend_index.dispatch_key
    if is_cuda_dispatch_key(dk):
        return "TORCH_CUDA_CPP_API"
    if is_xpu_dispatch_key(dk):
        return "TORCH_XPU_API"
    return "TORCH_API"


# See tags.yaml for cpu_dll_* tag semantics; shared-name kernels are merged in get_ns_grouped_kernels.
def dll_export_macro_for_kernel(
    backend_index: BackendIndex,
    g: NativeFunction | NativeFunctionsGroup | None,
) -> str:
    macro = torch_api_key_word_prefix(backend_index)
    if macro == "TORCH_CUDA_CPP_API" and g is not None:
        tags = (
            g.tags
            if isinstance(g, NativeFunction)
            else set().union(*(f.tags for f in g.functions()))
        )
        if _CPU_DLL_CUDA_KERNEL_TAG in tags:
            return "TORCH_API"
        if (
            _CPU_DLL_QUANTIZED_CUDA_KERNEL_TAG in tags
            and backend_index.dispatch_key == DispatchKey.QuantizedCUDA
        ):
            return "TORCH_API"
    return macro


@with_native_function_and_index
def gen_unstructured(f: NativeFunction, backend_index: BackendIndex) -> str | None:
    sig = kernel_signature(f, backend_index)
    metadata = backend_index.get_kernel(f)
    if metadata is None:
        return None
    if "legacy::" in metadata.kernel:
        return None
    else:
        prefix = (
            "static"
            if backend_index.external
            else dll_export_macro_for_kernel(backend_index, f)
        )
        return f"{prefix} {sig.decl(name=metadata.kernel)};"


@with_native_function_and_index
def gen_structured(g: NativeFunctionsGroup, backend_index: BackendIndex) -> list[str]:
    meta_name = meta.name(g)
    out_args = structured.impl_arguments(g)
    metadata = backend_index.get_kernel(g)
    if metadata is None:
        return []
    macro = dll_export_macro_for_kernel(backend_index, g)
    prefix = f"{macro} " if macro else ""
    return [
        f"""\
struct {prefix}structured_{metadata.kernel} : public at::meta::structured_{meta_name} {{
void impl({", ".join(a.decl() for a in out_args)});
}};
"""
    ]


# Generates NativeFunctions.h, a list of forward declarations of all
# actual kernel definitions we keep in aten/src/ATen/native/
@with_native_function_and_index
def compute_native_function_declaration(
    g: NativeFunctionsGroup | NativeFunction, backend_index: BackendIndex
) -> list[str]:
    metadata = backend_index.get_kernel(g)
    if isinstance(g, NativeFunctionsGroup):
        if metadata is not None and metadata.structured:
            if backend_index.external:
                # Structured hasn't been tested with external backends yet.
                raise AssertionError(
                    "Structured external backend functions are not implemented yet."
                )
            else:
                return gen_structured(g, backend_index)
        else:
            return list(
                mapMaybe(lambda f: gen_unstructured(f, backend_index), g.functions())
            )
    else:
        x = gen_unstructured(g, backend_index)
        return [] if x is None else [x]
