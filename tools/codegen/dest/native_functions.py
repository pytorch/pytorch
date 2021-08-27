from typing import List, Union, Optional

from tools.codegen.context import with_native_function_and_index
from tools.codegen.utils import mapMaybe
from tools.codegen.model import NativeFunction, NativeFunctionsGroup, BackendIndex
from tools.codegen.api.types import kernel_signature
import tools.codegen.api.meta as meta
import tools.codegen.api.structured as structured
from .lazy_ir import process_ir_types
@with_native_function_and_index
def gen_unstructured(f: NativeFunction, backend_index: BackendIndex) -> Optional[str]:
    sig = kernel_signature(f, backend_index)
    metadata = backend_index.get_kernel(f)
    if metadata is None:
        return None
    if "legacy::" in metadata.kernel:
        return None
    else:
        prefix = 'static' if backend_index.external else 'TORCH_API'
        return f"{prefix} {sig.decl(name=metadata.kernel)};"

@with_native_function_and_index
def gen_structured(g: NativeFunctionsGroup, backend_index: BackendIndex) -> List[str]:
    meta_name = meta.name(g)
    out_args = structured.impl_arguments(g)
    metadata = backend_index.get_kernel(g)
    if metadata is None:
        return []
    prefix = '' if backend_index.external else 'TORCH_API '
    return [f"""\
struct {prefix}structured_{metadata.kernel} : public at::meta::structured_{meta_name} {{
void impl({', '.join(a.decl() for a in out_args)});
}};
"""]

# Generates NativeFunctions.h, a list of forward declarations of all
# actual kernel definitions we keep in aten/src/ATen/native/
@with_native_function_and_index
def compute_native_function_declaration(
        g: Union[NativeFunctionsGroup, NativeFunction],
        backend_index: BackendIndex
) -> List[str]:
    metadata = backend_index.get_kernel(g)
    if isinstance(g, NativeFunctionsGroup):
        if metadata is not None and metadata.structured:
            if backend_index.external:
                # Structured hasn't been tested with external backends yet.
                raise AssertionError("Structured external backend functions are not implemented yet.")
            else:
                return gen_structured(g, backend_index)
        else:
            return list(mapMaybe(lambda f: gen_unstructured(f, backend_index), g.functions()))
    else:
        x = gen_unstructured(g, backend_index)
        return [] if x is None else [x]


@with_native_function_and_index
def gen_unstructured_lazy_definition(f: NativeFunction, backend_index: BackendIndex) -> Optional[str]:
    sig = kernel_signature(f, backend_index)
    metadata = backend_index.get_kernel(f)
    
    # Lazy IR stuff
    all_types, value_types, scalar_types = process_ir_types(f.func)

    lazy_tensors = "\n    ".join([f"LazyTensor l_{t.name} = bridge::GetLtcTensor({t.name});" for t in value_types])
    node_ctor_inputs = ", ".join([f"l_{t.name}" for t in value_types] +
                                 [s.name for s in scalar_types])
    assert len(value_types) > 0, f"Only supporting tensor ops so far, none found in {sig}"
    first_tensor = value_types[0]

    if metadata is None:
        return None
    if "legacy::" in metadata.kernel:
        return None
    else:
        return f"""\
{sig.decl(name=metadata.kernel)} {{
    {lazy_tensors}
    return bridge::AtenFromLtcTensor(l_{first_tensor.name}.CreateFrom(
        ir::MakeNode<ir::ops::{metadata.kernel}>({node_ctor_inputs})));
}};
"""

@with_native_function_and_index
def compute_lazy_native_function_definition(
        g: Union[NativeFunctionsGroup, NativeFunction],
        backend_index: BackendIndex
) -> List[str]:
    metadata = backend_index.get_kernel(g)
    if isinstance(g, NativeFunctionsGroup):
        if metadata is not None and metadata.structured:
            raise AssertionError("Structured lazy functions are not implemented yet.")
        else:
            # return list(mapMaybe(lambda f: gen_unstructured_lazy_definition(f, backend_index), g.functions()))
            return []
    else:
        if metadata is not None:
            x = gen_unstructured_lazy_definition(g, backend_index)
            return [] if x is None else [x]
        return []