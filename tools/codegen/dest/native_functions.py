from typing import List, Union, Optional

from tools.codegen.context import with_native_function_and_index
from tools.codegen.utils import mapMaybe
from tools.codegen.model import NativeFunction, NativeFunctionsGroup, BackendIndex
from tools.codegen.api.types import kernel_signature, BaseCType, OptionalCType
import tools.codegen.api.meta as meta
import tools.codegen.api.structured as structured
from .lazy_ir import process_ir_types, ir_node_name
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

    lazy_tensor_decls = []
    node_ctor_values = []
    for t in value_types:
        if isinstance(t.type.elem, BaseCType):
            lazy_tensor_decls.append(f"LazyTensor l_{t.name} = bridge::GetLtcTensor({t.name});")
            node_ctor_values.append(f"l_{t.name}.GetIrValue()")
        elif isinstance(t.type.elem, OptionalCType):
            lazy_tensor_decls.append(f"c10::optional<LazyTensor> l_{t.name} =  {t.name}.has_value() ? c10::make_optional(bridge::GetLtcTensor({t.name}.value())) : c10::nullopt;")
            node_ctor_values.append(f"l_{t.name}.has_value() ? l_{t.name}.value().GetIrValue() : torch_lazy_tensors::ir::ops::kNullValue")
        else:
            assert False, ""
    lazy_tensor_decls = "\n    ".join(lazy_tensor_decls)

    node_ctor_scalars = [] 
    for t in scalar_types:
        # import ipdb; ipdb.set_trace()
        if isinstance(t.type, BaseCType) and t.type.type.name == "vector<int64_t>":
            node_ctor_scalars.append(f"std::vector<int64_t>({t.name}.begin(), {t.name}.end())")
        else:
            node_ctor_scalars.append(t.name)

    node_ctor_inputs = ",\n                              ".join(node_ctor_values + node_ctor_scalars)
    assert len(value_types) > 0, f"Only supporting tensor ops so far, none found in {sig}"
    first_tensor = value_types[0]

    if metadata is None:
        return None
    if "legacy::" in metadata.kernel:
        return None
    else:
        return f"""\
{sig.decl(name=metadata.kernel)} {{
    {lazy_tensor_decls}
    return bridge::AtenFromLtcTensor(l_{first_tensor.name}.CreateFrom(
        ir::MakeNode<ir::ops::{ir_node_name(f.func)}>({node_ctor_inputs})));
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