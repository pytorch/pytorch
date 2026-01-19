from __future__ import annotations

import torchgen.api.meta as meta
import torchgen.api.structured as structured
from torchgen.api.types import kernel_signature
from torchgen.context import with_native_function_and_index
from torchgen.model import BackendIndex, NativeFunction, NativeFunctionsGroup
from torchgen.utils import mapMaybe


def torch_api_key_word_prefix(bankend_index: BackendIndex) -> str:
    if bankend_index.external:
        return ""

    # Although Intel GPU ATen library is out-of-tree, it still utilizes torchgen to produce structured
    # kernels. Regarding these produced structured kernels, they should be visible for the Intel GPU ATen
    # library. Therefore, we need to add "TORCH_XPU_API" prefix to these structured kernels,
    # rather than "TORCH_API". Because the semantic of "TORCH_API" is "hidden" for out-of-tree backends.
    # For other in-tree backends like cpu and cuda, they still use "TORCH_API" prefix with "visible" semantic.
    device_torch_api_key_word_mapping = {
        "XPU": "TORCH_XPU_API",
    }

    return (
        device_torch_api_key_word_mapping.get(
            bankend_index.dispatch_key.name, "TORCH_API"
        )
        + " "
    )


@with_native_function_and_index
def gen_unstructured(f: NativeFunction, backend_index: BackendIndex) -> str | None:
    sig = kernel_signature(f, backend_index)
    metadata = backend_index.get_kernel(f)
    if metadata is None:
        return None
    if "legacy::" in metadata.kernel:
        return None
    else:
        prefix = "static" if backend_index.external else "TORCH_API"
        return f"{prefix} {sig.decl(name=metadata.kernel)};"


@with_native_function_and_index
def gen_structured(g: NativeFunctionsGroup, backend_index: BackendIndex) -> list[str]:
    meta_name = meta.name(g)
    out_args = structured.impl_arguments(g)
    metadata = backend_index.get_kernel(g)
    if metadata is None:
        return []
    prefix = torch_api_key_word_prefix(backend_index)
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
