from typing import List, Union
from dataclasses import dataclass
from tools.codegen.context import method_with_native_function
from tools.codegen.model import (BackendIndex, NativeFunction,
                                 NativeFunctionsGroup)
from tools.codegen.api.types import (BaseCType, OptionalCType, NamedCType,
                                     VectorCType, kernel_signature)
import tools.codegen.api.dispatcher as dispatcher
from tools.codegen.api.lazy import LazyIrSchema, isValueType


def node_ctor_inputs(func: LazyIrSchema) -> str:
    """
    Produce a formatted string with the arguments as passed into the constructor of a node class.
    """
    node_ctor_values = []
    for arg in func.filtered_types():
        if isValueType(arg.type):
            if isinstance(arg.type, BaseCType):
                node_ctor_values.append(f"l_{arg.name}.GetIrValue()")
            elif isinstance(arg.type, OptionalCType):
                node_ctor_values.append(
                    f"l_{arg.name} ? "
                    f"c10::make_optional(l_{arg.name}->GetIrValue()) : "
                    "c10::nullopt")
            else:
                raise AssertionError("TODO not sure if there are other valid types to handle here")
        else:
            if isinstance(arg.type, VectorCType) and isinstance(arg.type.elem, BaseCType):
                node_ctor_values.append(f"lazy_tensors::util::ToVector<{arg.type.elem.type}>({arg.name})")
            else:
                node_ctor_values.append(f"{arg.name}")

    node_ctor_inputs_str = ",\n                              ".join(node_ctor_values)
    return node_ctor_inputs_str


@dataclass(frozen=True)
class LazyIR:
    backend_index: BackendIndex
    node_base: str

    @method_with_native_function
    def __call__(self, f: Union[NativeFunctionsGroup, NativeFunction]) -> List[str]:
        func = f.functional.func if isinstance(f, NativeFunctionsGroup) else f.func
        return self.gen(f)

    def gen(self, f: Union[NativeFunctionsGroup, NativeFunction]) -> List[str]:
        # for now, we just want one IR class decl and soon after also the method defs
        # and we use the functional version not out/inplace.
        func = f.functional.func if isinstance(f, NativeFunctionsGroup) else f.func
        schema = LazyIrSchema(func)
        all_types = schema.filtered_types()
        value_types = schema.filtered_types(values=True, scalars=False)
        scalar_types = schema.filtered_types(values=False, scalars=True)

        node_ctor_args = ", ".join([f"const {i.cpp_type()}& {i.name}" for i in all_types])
        scalar_initializers = ",\n        ".join([f"{t.name}_({t.name})" for t in scalar_types])
        comma_if_scalar_initializers = ",\n" if len(scalar_initializers) else ""
        scalar_decls = "\n  ".join([f"{t.cpp_type()} {t.name}_;" for t in scalar_types])
        scalar_hashes = ", ".join([f"{f.name}" for f in scalar_types])
        base_ctor_value_args_list = []
        optional_values = []
        for t in value_types:
            if isinstance(t.type, BaseCType):
                base_ctor_value_args_list.append(f"{t.name}")
            elif isinstance(t.type, OptionalCType):
                base_ctor_value_args_list.append(f"{t.name}.value_or(kNullValue)")
                optional_values.append(t.name)
            else:
                raise AssertionError("TODO not sure if there are other valid types to handle here")
        base_ctor_value_args = ", ".join(base_ctor_value_args_list)
        has_optional_decls = "\n  ".join([f"bool has_{value}: 1;" for value in optional_values])
        has_optional_defs = "\n    ".join([f"has_{value} = !!{value};" for value in optional_values])
        members_to_string = "\n    ".join([f'lazy_tensors::ToString("{t.name}", {t.name}_, ss);' for t in scalar_types])

        clone_impl_args = []
        for value in all_types:
            if isValueType(value.type):
                if isinstance(value.type, OptionalCType):
                    clone_impl_args.append(f"(has_{value.name} ? c10::make_optional(operands.at(i++)) : c10::nullopt)")
                    continue
                clone_impl_args.append("operands.at(i++)")
                continue
            clone_impl_args.append(f"{value.name}_")
        clone_impl_args.extend(["at_dtypes_", "at_shapes_"])

        clone_impl_args_str = ", ".join(clone_impl_args)
        clone_impl = f"ir::MakeNode<ir::ops::{schema.node_name}>({clone_impl_args_str});"

        return [f"""\
// TODO(alanwaketan): Public members don't need to have _ suffix.
class {schema.node_name} : public {self.node_base} {{
 public:
  {schema.node_name}({node_ctor_args}, const std::vector<at::ScalarType>& out_dtypes,
      const std::vector<std::vector<int64_t>>& out_shapes)
      : {self.node_base}(ir::OpKind(at::aten::{func.name.name}),
              {{{base_ctor_value_args}}},
              convertShape(out_dtypes, out_shapes),
              /* num_outputs */ {len(func.returns)},
              torch::lazy::MHash({scalar_hashes})),
        at_dtypes_(out_dtypes),
        at_shapes_(out_shapes){comma_if_scalar_initializers}
        {scalar_initializers}

  {{
    {has_optional_defs}
  }}

  std::string ToString() const override {{
    std::stringstream ss;
    ss << Node::ToString();
    {members_to_string}
    return ss.str();
  }}

  NodePtr Clone(OpList operands) const override {{
      size_t i = 0;
      {clone_impl}
  }}
  
  // TODO(whc) prefer to move these shapes to TsNode, but need to find a way to populate
  // them consistently from non-codegen TsNode classes first.
  // outer vector is for multiple tensors from an operation
  std::vector<at::ScalarType> at_dtypes_;
  std::vector<std::vector<int64_t>> at_shapes_;
  {scalar_decls}
  {has_optional_decls}

}};

""", ]


def lazy_tensor_decls(value_types: List[NamedCType]) -> str:
    lazy_tensor_decls: List[str] = []
    for t in value_types:
        if isinstance(t.type, BaseCType):
            lazy_tensor_decls.append(f"LazyTensor l_{t.name} = bridge::GetLtcTensor({t.name});")
        elif isinstance(t.type, OptionalCType):
            lazy_tensor_decls.append(
                f"c10::optional<LazyTensor> l_{t.name} =  "
                f"{t.name} ? "
                f"bridge::TryGetLtcTensor(*{t.name}) : "
                f"c10::nullopt;")
        else:
            raise AssertionError("TODO not sure if there are other valid types to handle here")
    return "\n    ".join(lazy_tensor_decls)


def gen_lazy_nativefunc_definition(func: NativeFunction, backend_index: BackendIndex,
                                   class_method_name: str, node_base: str) -> List[str]:
    sig = kernel_signature(func, backend_index)

    # Lazy IR stuff
    schema = LazyIrSchema(func.func)
    all_types = schema.filtered_types()
    value_types = schema.filtered_types(values=True, scalars=False)
    scalar_types = schema.filtered_types(values=False, scalars=True)
    lazy_tensor_decls_str = lazy_tensor_decls(value_types)
    node_ctor_input_str = node_ctor_inputs(schema)
    returns_length = len(schema.returns)

    # call the meta kernel if it exists, to compute output shape/dtype for our IR
    if func.structured or func.structured_delegate is not None:
        meta_out = """std::vector<std::vector<int64_t>> out_shape{out_meta.sizes().vec()};
    std::vector<c10::ScalarType> out_dtype{out_meta.scalar_type()};"""
        if returns_length > 1:
            meta_out = """auto out_shape = CreateComputationShapeFromMetaTensors(out_meta);
    auto out_dtype = CreateDTypeFromMetaTensors(out_meta);"""

        meta_args = ", ".join([f"{t.name}.to(c10::kMeta)" for t in value_types] +
                              [f"{t.name}" for t in scalar_types])
        meta_str = f"""auto out_meta = at::meta::{schema.aten_name}({meta_args});
    {meta_out}"""
    else:
        meta_args = ", ".join([f"{t.name}" for t in all_types])
        meta_str = f"""
    auto out_shape = torch_lazy_tensors::ir::ops::compute_shape_{schema.aten_name}({meta_args});
    auto out_dtype = torch_lazy_tensors::ir::ops::compute_dtype_{schema.aten_name}({meta_args});"""

    node_str = f"""auto node = ir::MakeNode<ir::ops::{schema.node_name}>({node_ctor_input_str}, out_dtype, out_shape);"""

    assert len(value_types) > 0, f"Only supporting tensor ops so far, none found in {sig}"
    first_tensor = value_types[0]
    bridge_str = f"""auto result = bridge::AtenFromLtcTensor(l_{first_tensor.name}.CreateFrom(node,
        // TODO (@wconstab): experiment on dtype
        // try always overriding output dtype to match the one ATen says our op should produce.
        // this diverges from most of the handwritten methods, which often do not override and
        // rely on other behavior in the lowering or copy process to make this correct.
        // (1) evaluate design goal: to always pick the IR's dtype in one place (here)
        // (2) rationalize this with Google's design, it may be a problem
        // (3) evaluate perf impact: make sure we're not actually doing casts becuase of this override
        out_dtype.front()));"""
    if returns_length > 1:
        bridge_str = f"""std::vector<LazyTensor> lazy_tensors;
    for (int i = 0; i < {returns_length}; i++) {{
        lazy_tensors.push_back(l_{first_tensor.name}.CreateFrom(ir::Value(node, i), out_dtype[i]));
    }}
    auto result = bridge::TupleAtenFromLtcTensors<{returns_length}>(lazy_tensors);"""

    return [f"""\
// TODO(alanwaketan): Quite a lot inefficient copy-by-value there. Let's optimize it.
{sig.decl(name=f"{class_method_name}::{schema.aten_name}")} {{
    LTC_FN_COUNTER("lazy::");
    {lazy_tensor_decls_str}
    {meta_str}
    {node_str}
    {bridge_str}
    return result;
}};\n
"""]


def gen_lazy_shape_dtype_decl(f: NativeFunction, backend_index: BackendIndex) -> List[str]:
    sig = kernel_signature(f, backend_index)

    # Lazy IR stuff
    schema = LazyIrSchema(f.func)
    value_types = schema.filtered_types(values=True, scalars=False)
    lazy_tensor_decls_str = lazy_tensor_decls(value_types)
    node_ctor_input_str = node_ctor_inputs(schema)

    # Only generate shape/dtype fn for non-structured kernels,
    # since we just use the meta function for structured kernels
    if not f.structured and f.structured_delegate is None:
        dispatch_args = ', '.join([a.decl() for a in dispatcher.arguments(f.func)])
        return ["\n".join([f"std::vector<std::vector<int64_t>> compute_shape_{schema.aten_name}({dispatch_args});",
                           f"std::vector<c10::ScalarType> compute_dtype_{schema.aten_name}({dispatch_args});"])]
    else:
        return []
