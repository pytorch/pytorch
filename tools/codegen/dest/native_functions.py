from typing import List, Union, Optional

from tools.codegen.context import with_native_function_and_index
from tools.codegen.utils import mapMaybe
from tools.codegen.model import FunctionSchema, Return, NativeFunction, NativeFunctionsGroup, BackendIndex, OperatorName
from tools.codegen.api.types import kernel_signature, BaseCType, OptionalCType, scalarTypeT, DispatcherSignature
import tools.codegen.api.meta as meta
import tools.codegen.api.structured as structured
from .lazy_ir import update_schema_for_lazy_ir, separate_value_scalar_types, ir_node_name, node_ctor_inputs, intArrayT


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


def lazy_tensor_decls(value_types):
    lazy_tensor_decls = []
    for t in value_types:
        if isinstance(t.type, BaseCType):
            lazy_tensor_decls.append(f"LazyTensor l_{t.name} = bridge::GetLtcTensor({t.name});")
        elif isinstance(t.type, OptionalCType):
            lazy_tensor_decls.append(
                f"c10::optional<LazyTensor> l_{t.name} =  {t.name}.has_value() ? c10::make_optional(bridge::GetLtcTensor({t.name}.value())) : c10::nullopt;")
        else:
            assert False, ""
    lazy_tensor_decls = "\n    ".join(lazy_tensor_decls)
    return lazy_tensor_decls


def gen_unstructured_lazy_definition(f: NativeFunction, backend_index: BackendIndex, codegen: List[OperatorName], class_method_name: str) -> Optional[str]:
    sig = kernel_signature(f, backend_index)
    metadata = backend_index.get_kernel(f)
    if f.func.name not in codegen:
        return None
    if metadata is None:
        return None
    if "legacy::" in metadata.kernel:
        return None

    # Lazy IR stuff
    schema = update_schema_for_lazy_ir(f.func)
    all_types, value_types, scalar_types = separate_value_scalar_types(schema)
    lazy_tensor_decls_str = lazy_tensor_decls(value_types)
    node_ctor_input_str = node_ctor_inputs(value_types, scalar_types)

    # call the meta kernel if it exists, to compute output shape/dtype for our IR
    if f.structured or f.structured_delegate != None:
        meta_args = ", ".join([f"{t.name}.to(c10::kMeta)" for t in value_types] + [t.name for t in scalar_types])
        meta_str = f"""auto out_meta = at::meta::{metadata.kernel}({meta_args});
    auto _out_shape = out_meta.sizes().vec();
    auto _out_dtype = out_meta.scalar_type();"""
    else:
        meta_str = f"""auto _out_shape = torch_lazy_tensors::ir::ops::compute_shape_{metadata.kernel}({", ".join([t.name for t in all_types])});
    auto _out_dtype = torch_lazy_tensors::ir::ops::compute_dtype_{metadata.kernel}({", ".join([t.name for t in all_types])});"""

    assert len(value_types) > 0, f"Only supporting tensor ops so far, none found in {sig}"
    first_tensor = value_types[0]

    return f"""\
{sig.decl(name=f"{class_method_name}::{metadata.kernel}")} {{
    {lazy_tensor_decls_str}
    {meta_str}
    return bridge::AtenFromLtcTensor(l_{first_tensor.name}.CreateFrom(
        ir::MakeNode<ir::ops::{ir_node_name(f.func)}>({node_ctor_input_str}, _out_dtype, _out_shape)));
}};
"""


def compute_lazy_native_function_definition(
        g: Union[NativeFunctionsGroup, NativeFunction],
        backend_index: BackendIndex,
        codegen: List[OperatorName],
        class_method_name: str,
) -> List[str]:

    metadata = backend_index.get_kernel(g)
    if isinstance(g, NativeFunctionsGroup):
        if metadata is not None and metadata.structured:
            raise AssertionError("Structured lazy functions are not implemented yet.")
        else:
            return list(mapMaybe(lambda f: gen_unstructured_lazy_definition(f, backend_index, codegen, class_method_name), g.functions()))
    else:
        x = gen_unstructured_lazy_definition(g, backend_index, codegen, class_method_name)
        return [] if x is None else [x]


def gen_lazy_shape_dtype_decl(f: NativeFunction, backend_index: BackendIndex, codegen: List[OperatorName]) -> Optional[str]:
    sig = kernel_signature(f, backend_index)
    metadata = backend_index.get_kernel(f)
    if f.func.name not in codegen:
        return None
    if metadata is None:
        return None
    if "legacy::" in metadata.kernel:
        return None

    # Lazy IR stuff
    schema = update_schema_for_lazy_ir(f.func)
    all_types, value_types, scalar_types = separate_value_scalar_types(schema)
    lazy_tensor_decls_str = lazy_tensor_decls(value_types)
    node_ctor_input_str = node_ctor_inputs(value_types, scalar_types)

    # Only generate shape/dtype fn for non-structured kernels,
    # since we just use the meta function for structured kernels
    if not f.structured and f.structured_delegate is None:
        shapeFn = DispatcherSignature.from_schema(FunctionSchema(f.func.name,
                                                                 f.func.arguments,
                                                                 returns=(Return(name=None,
                                                                                 type=intArrayT,
                                                                                 annotation=None),)),
                                                  prefix='compute_shape_')
        dtypeFn = DispatcherSignature.from_schema(FunctionSchema(f.func.name,
                                                                 f.func.arguments,
                                                                 returns=(Return(name=None,
                                                                                 type=scalarTypeT,
                                                                                 annotation=None),)),
                                                  prefix='compute_dtype_')

        return "\n".join([f"std::vector<int64_t> compute_shape_{metadata.kernel}({', '.join([a.decl() for a in shapeFn.arguments()])});",
                f"c10::ScalarType compute_dtype_{metadata.kernel}({', '.join([a.decl() for a in dtypeFn.arguments()])});"])
    else:
        return None


def compute_lazy_shape_dtype_decl(
        g: Union[NativeFunctionsGroup, NativeFunction],
        backend_index: BackendIndex,
        codegen: List[OperatorName]
) -> List[str]:

    metadata = backend_index.get_kernel(g)
    if isinstance(g, NativeFunctionsGroup):
        if metadata is not None and metadata.structured:
            raise AssertionError("Structured lazy functions are not implemented yet.")
        else:
            return list(mapMaybe(lambda f: gen_lazy_shape_dtype_decl(f, backend_index, codegen), g.functions()))
    else:
        x = gen_lazy_shape_dtype_decl(g, backend_index, codegen)
        return [] if x is None else [x]
