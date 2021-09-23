from typing import List, Union
from dataclasses import dataclass
from tools.codegen.context import method_with_native_function
from tools.codegen.model import (BackendIndex, NativeFunction,
                                 NativeFunctionsGroup)
from tools.codegen.api.types import (BaseCType, OptionalCType,
                                     NamedCType, kernel_signature)
import tools.codegen.api.dispatcher as dispatcher
from tools.codegen.api.lazy import LazyIrSchema, isValueType


def node_ctor_inputs(func: LazyIrSchema) -> str:
    """
    Produce a formatted string with the arguments as passed into the constructor of a node class.
    """
    node_ctor_values = []
    node_ctor_scalars = []
    for arg in func.filtered_types():
        if isValueType(arg.type):
            if isinstance(arg.type, BaseCType):
                node_ctor_values.append(f"l_{arg.name}.GetIrValue()")
            elif isinstance(arg.type, OptionalCType):
                node_ctor_values.append(
                    f"l_{arg.name}.has_value() ? "
                    f"l_{arg.name}.value().GetIrValue() : "
                    f"torch_lazy_tensors::ir::ops::kNullValue")
            else:
                raise AssertionError("TODO not sure if there are other valid types to handle here")
        else:
            if isinstance(arg.type, BaseCType) and arg.type.type.name == "vector<int64_t>":
                node_ctor_scalars.append(f"std::vector<int64_t>({arg.name}.begin(), {arg.name}.end())")
            else:
                node_ctor_scalars.append(f"{arg.name}")

    node_ctor_inputs_str = ",\n                              ".join(node_ctor_values + node_ctor_scalars)
    return node_ctor_inputs_str


@dataclass(frozen=True)
class LazyIR:
    backend_index: BackendIndex

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

        node_ctor_args = ", ".join([f"{i.cpp_type()} {i.name}" for i in all_types])
        scalar_initializers = ",\n        ".join([f"{t.name}_({t.name})" for t in scalar_types])
        comma_if_scalar_initializers = ",\n" if len(scalar_initializers) else ""
        scalar_decls = "\n  ".join([f"{t.cpp_type()} {t.name}_;" for t in scalar_types])
        scalar_hashes = ", ".join([f"{f.name}" for f in scalar_types])
        base_ctor_value_args_list = []
        for t in value_types:
            if isinstance(t.type, BaseCType):
                base_ctor_value_args_list.append(f"{t.name}")
            elif isinstance(t.type, OptionalCType):
                base_ctor_value_args_list.append(f"{t.name}.has_value() ? {t.name}.value() : kNullValue")
            else:
                raise AssertionError("TODO not sure if there are other valid types to handle here")
        base_ctor_value_args = ", ".join(base_ctor_value_args_list)
        members_to_string = "\n    ".join([f'lazy_tensors::ToString("{t.name}", {t.name}_, ss);' for t in scalar_types])

        # clone needs hand-overrides for cases where there are optional Tensor? args,
        # unless we clean up the OpList API to deal unambiguously with optionals.
        clone_impl_args = ",".join(
            [f"operands.at({i})" for i in range(len(value_types))] +
            [f"{s.name}_" for s in scalar_types] +
            ["out_dtype_", "out_shape_"])
        if any([isinstance(t.type, OptionalCType) for t in value_types]):
            scalar_args = ",".join([f"{s.name}_" for s in scalar_types])
            clone_impl = f"return Clone{schema.node_name}(operands, {scalar_args});"
            clone_decl_args = ", ".join([f"{i.cpp_type()} {i.name}" for i in scalar_types])
            clone_handcoded_decl = f"NodePtr Clone{schema.node_name}(OpList operands, {clone_decl_args});"
        else:
            clone_impl = f"ir::MakeNode<ir::ops::{schema.node_name}>({clone_impl_args});"
            clone_handcoded_decl = ""

        return [f"""\
{clone_handcoded_decl}
class {schema.node_name} : public Node {{
 public:
  {schema.node_name}({node_ctor_args}, at::ScalarType out_dtype, std::vector<int64_t> out_shape)
      : Node(ir::OpKind(at::aten::{func.name.name}),
              {{{base_ctor_value_args}}},
              /*shape=*/lazy_tensors::Shape(out_dtype, out_shape),
              /*num_outputs=*/{len(func.returns)},
              lazy_tensors::util::MHash({scalar_hashes})),
        out_dtype_(out_dtype),
        out_shape_(out_shape){comma_if_scalar_initializers}
        {scalar_initializers}

  {{
    //  throw std::runtime_error("need to hash scalars properly");
  }}

  std::string ToString() const override {{
    std::stringstream ss;
    ss << Node::ToString();
    {members_to_string}
    return ss.str();
  }}

  NodePtr Clone(OpList operands) const override {{
      {clone_impl}
  }}

  c10::ScalarType out_dtype_;
  std::vector<int64_t> out_shape_;
  {scalar_decls}
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
                "{t.name}.has_value() ? "
                "c10::make_optional(bridge::GetLtcTensor({t.name}.value())) : "
                "c10::nullopt;")
        else:
            raise AssertionError("TODO not sure if there are other valid types to handle here")
    return "\n    ".join(lazy_tensor_decls)


def gen_lazy_nativefunc_definition(f: NativeFunction, backend_index: BackendIndex,
                                   class_method_name: str) -> List[str]:
    sig = kernel_signature(f, backend_index)

    # Lazy IR stuff
    schema = LazyIrSchema(f.func)
    all_types = schema.filtered_types()
    value_types = schema.filtered_types(values=True, scalars=False)
    scalar_types = schema.filtered_types(values=False, scalars=True)
    lazy_tensor_decls_str = lazy_tensor_decls(value_types)
    node_ctor_input_str = node_ctor_inputs(schema)

    # call the meta kernel if it exists, to compute output shape/dtype for our IR
    if f.structured or f.structured_delegate is not None:
        meta_args = ", ".join([f"{t.name}.to(c10::kMeta)" for t in value_types] +
                              [f"{t.name}" for t in scalar_types])
        meta_str = f"""auto out_meta = at::meta::{schema.aten_name}({meta_args});
    auto _out_shape = out_meta.sizes().vec();
    auto _out_dtype = out_meta.scalar_type();"""
    else:
        meta_args = ", ".join([f"{t.name}" for t in all_types])
        meta_str = f"""
    auto _out_shape = torch_lazy_tensors::ir::ops::compute_shape_{schema.aten_name}({meta_args});
    auto _out_dtype = torch_lazy_tensors::ir::ops::compute_dtype_{schema.aten_name}({meta_args});"""

    assert len(value_types) > 0, f"Only supporting tensor ops so far, none found in {sig}"
    first_tensor = value_types[0]

    return [f"""\
{sig.decl(name=f"{class_method_name}::{schema.aten_name}")} {{
    LTC_FN_COUNTER("lazy::");
    {lazy_tensor_decls_str}
    {meta_str}
    return bridge::AtenFromLtcTensor(l_{first_tensor.name}.CreateFrom(
        ir::MakeNode<ir::ops::{schema.node_name}>({node_ctor_input_str}, _out_dtype, _out_shape),

        // (whc): experiment on dtype
        // try always overriding output dtype to match the one ATen says our op should produce.
        // this diverges from most of the handwritten methods, which often do not override and
        // rely on other behavior in the lowering or copy process to make this correct.
        // (1) evaluate design goal: to always pick the IR's dtype in one place (here)
        // (2) rationalize this with Google's design, it may be a problem
        // (3) evaluate perf impact: make sure we're not actually doing casts becuase of this override
        _out_dtype));
}};
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
        return ["\n".join([f"std::vector<int64_t> compute_shape_{schema.aten_name}({dispatch_args});",
                           f"c10::ScalarType compute_dtype_{schema.aten_name}({dispatch_args});"])]
    else:
        return []
