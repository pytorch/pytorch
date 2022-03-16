from abc import ABC, abstractmethod
from typing import List, Union
from dataclasses import dataclass
from tools.codegen.context import method_with_native_function
from tools.codegen.model import (BackendIndex, NativeFunction,
                                 NativeFunctionsGroup)
from tools.codegen.api.types import (BaseCType, OptionalCType,
                                     VectorCType, kernel_signature)
import tools.codegen.api.dispatcher as dispatcher
from tools.codegen.api.lazy import LazyIrSchema, LazyArgument, isValueType, tensorListValueT
from tools.codegen.dest.lazy_ts_lowering import ts_lowering_body

def node_ctor_arg_rvalue_string(arg: LazyArgument) -> str:
    """
    Given a LazyArgument,
    generate a c++ string for materializing an rvalue of that arg for passing into
    a lazy Node constructor.
    """

    if isValueType(arg.lazy_type):
        if isinstance(arg.lazy_type, BaseCType):
            if arg.is_wrapped_scalar:
                return f"torch::lazy::LazyGraphExecutor::Get()->GetIrValueForScalarFromCodegen({arg.name})"
            elif arg.lazy_type.type is tensorListValueT:
                return f"lazy_{arg.name}_tensorlist"
            return f"lazy_{arg.name}->GetIrValue()"
        elif isinstance(arg.lazy_type, OptionalCType):
            if arg.is_wrapped_scalar:
                return f"{arg.name} ? " \
                    f"c10::make_optional(torch::lazy::LazyGraphExecutor::Get()->GetIrValueForScalarFromCodegen(*{arg.name})) : " \
                    "c10::nullopt"
            return f"lazy_{arg.name} ? " \
                   f"c10::make_optional(lazy_{arg.name}->GetIrValue()) : " \
                   "c10::nullopt"
        else:
            raise AssertionError(f"TODO not sure if there are other valid types to handle here ({arg.lazy_type})")
    else:
        if isinstance(arg.lazy_type, VectorCType) and isinstance(arg.lazy_type.elem, BaseCType):
            return f"std::vector<{arg.lazy_type.elem.type}>({arg.name}.begin(), {arg.name}.end())"
        elif (isinstance(arg.lazy_type, OptionalCType) and
                isinstance(arg.lazy_type.elem, VectorCType) and
                isinstance(arg.lazy_type.elem.elem, BaseCType)):
            return f"torch::lazy::ToOptionalVector<{arg.lazy_type.elem.elem.type}>({arg.name})"
        else:
            return f"{arg.name}"

def node_ctor_inputs(schema: LazyIrSchema) -> str:
    """
    Produce a formatted string with the arguments as passed into the constructor of a node class.
    """
    node_ctor_values = [node_ctor_arg_rvalue_string(arg) for arg in schema.filtered_args()]
    return ",\n                              ".join(node_ctor_values)

def gen_fallback_code(schema: LazyIrSchema, overload_name: str) -> str:
    """
    Generate code that falls back to eager conditioned on a predicate
    """
    fallback_args = ",\n                ".join([str(arg.name) for arg in schema.filtered_args(generator=True)])
    if len(overload_name):
        aten_op_str = f"ATEN_OP2({schema.aten_name}, {overload_name})"
    else:
        aten_op_str = f"ATEN_OP({schema.aten_name})"
    or_has_generator = ""
    if schema.generator_arg:
        # generators are always optional and there is never more than one, at least currently
        or_has_generator = f" || ({schema.generator_arg.name}.has_value() && {schema.generator_arg.name}->defined())"
    return f"""
        if (force_eager_fallback({aten_symbol(schema)}){or_has_generator}) {{
            return at::native::call_fallback_fn<&ltc_eager_fallback, {aten_op_str}>::call(
                {fallback_args}
            );
        }}
"""

def aten_symbol(schema: LazyIrSchema) -> str:
    missing_interned_strings = {
        'sigmoid_backward',
    }
    if schema.aten_name in missing_interned_strings:
        return f'c10::Symbol::fromQualString("aten::{schema.aten_name}")'
    return f'at::aten::{schema.aten_name}'

@dataclass(frozen=True)
class LazyIR(ABC):
    backend_index: BackendIndex
    node_base: str
    lowering_function_type: str = ""
    lowering_context_type: str = ""
    lowering_return_type: str = ""

    @method_with_native_function
    def __call__(self, f: Union[NativeFunctionsGroup, NativeFunction]) -> List[str]:
        func = f.functional.func if isinstance(f, NativeFunctionsGroup) else f.func
        return self.gen(f)

    @abstractmethod
    def lowering_body(self, f: Union[NativeFunctionsGroup, NativeFunction]) -> str:
        pass

    def gen(self, f: Union[NativeFunctionsGroup, NativeFunction]) -> List[str]:
        # for now, we just want one IR class decl and soon after also the method defs
        # and we use the functional version not out/inplace.
        func = f.functional.func if isinstance(f, NativeFunctionsGroup) else f.func
        schema = LazyIrSchema(func)
        all_args = schema.filtered_args()
        value_args = schema.filtered_args(values=True, scalars=False)
        scalar_args = schema.filtered_args(values=False, scalars=True)

        node_ctor_args = ", ".join([f"const {i.lazy_type.cpp_type()}& {i.name}" for i in all_args])
        scalar_initializers = ",\n        ".join([f"{a.name}({a.name})" for a in scalar_args])
        comma_if_scalar_initializers = ",\n" if len(scalar_initializers) else ""
        scalar_decls = "\n  ".join([f"std::string {a.name};" if a.lazy_type.cpp_type() == "c10::string_view"
                                    else f"{a.lazy_type.cpp_type()} {a.name};"
                                    for a in scalar_args])
        scalar_hashes = ", ".join([f"{a.name}" for a in scalar_args])
        base_ctor_value_args_list = []
        optional_values = []
        for arg in value_args:
            if isinstance(arg.lazy_type, BaseCType) or isinstance(arg.lazy_type, VectorCType):
                base_ctor_value_args_list.append(f"{arg.name}")
            elif isinstance(arg.lazy_type, OptionalCType):
                base_ctor_value_args_list.append(f"{arg.name}.value_or(kNullValue)")
                optional_values.append(arg.name)
            else:
                raise AssertionError(f"TODO not sure if there are other valid types to handle here ({arg.lazy_type})")
        base_ctor_value_args = ", ".join(base_ctor_value_args_list)
        has_optional_decls = "\n  ".join([f"bool has_{value}: 1;" for value in optional_values])
        has_optional_defs = "\n    ".join([f"has_{value} = !!{value};" for value in optional_values])
        members_to_string = []
        for arg in scalar_args:
            if isinstance(arg.lazy_type, OptionalCType):
                members_to_string.append(f"""if ({arg.name}.has_value()) {{
    ss << ", {arg.name}=" << {arg.name}.value();
}} else {{
    ss << ", {arg.name}=null";
}}""")
            else:
                members_to_string.append(f'ss << ", {arg.name}=" << {arg.name};')
        members_to_string_str = "\n    ".join(members_to_string)

        return [f"""\
class {schema.node_name} : public {self.node_base} {{
 public:
  {schema.node_name}({node_ctor_args}, std::vector<Shape>&& shapes)
      : {self.node_base}(torch::lazy::OpKind({aten_symbol(schema)}),
              {{{base_ctor_value_args}}}, std::move(shapes),
              /* num_outputs */ {len(func.returns)},
              torch::lazy::MHash({scalar_hashes})){comma_if_scalar_initializers}
        {scalar_initializers}

  {{
    {has_optional_defs}
  }}

  std::string ToString() const override {{
    std::stringstream ss;
    ss << {self.node_base}::ToString();
    {members_to_string_str}
    return ss.str();
  }}

  {self.lowering_return_type} Lower({self.lowering_function_type} function,
                   {self.lowering_context_type} loctx) const override {{
    {self.lowering_body(f)}
  }}

  {scalar_decls}
  {has_optional_decls}

}};

""", ]


@dataclass(frozen=True)
class TSLazyIR(LazyIR):
    lowering_function_type: str = "std::shared_ptr<torch::jit::GraphFunction>"
    lowering_context_type: str = "torch::lazy::TSLoweringContext*"
    lowering_return_type: str = "torch::lazy::TSOpVector"

    def lowering_body(self, f: Union[NativeFunctionsGroup, NativeFunction]) -> str:
        return ts_lowering_body(f)


def lazy_tensor_decls(value_args: List[LazyArgument], tensor_class: str) -> str:
    lazy_tensor_decls: List[str] = []
    for arg in value_args:
        if arg.is_wrapped_scalar:
            # no lazy tensor wrapper for scalars that are promoted to IR values
            continue
        elif isinstance(arg.lazy_type, BaseCType):
            if arg.lazy_type.type is tensorListValueT:
                lazy_tensor_decls.append(
                    f"auto lazy_{arg.name}_tensorlist = torch::lazy::GetTensorList({arg.name});")
            else:
                lazy_tensor_decls.append(
                    f"{tensor_class}Ptr lazy_{arg.name} = "
                    f"torch::lazy::GetLtcTensorOrCreateForWrappedNumber({arg.name}, *common_device);")
        elif isinstance(arg.lazy_type, OptionalCType):
            # TODO(alanwaketan): Maybe we want to apply GetLtcTensorOrCreateForWrappedNumber here, but hold it
            # until we encounter a real world example.
            lazy_tensor_decls.append(
                f"    {tensor_class}Ptr lazy_{arg.name} = torch::lazy::TryGetLtcTensor({arg.name}.value_or(at::Tensor()));")
        else:
            raise AssertionError(f"TODO not sure if there are other valid types to handle here ({arg.lazy_type})")
    return ("\n        ").join(lazy_tensor_decls)

@dataclass(frozen=True)
class GenLazyNativeFuncDefinition:
    class_method_name: str
    backend_index: BackendIndex
    tensor_class: str

    @method_with_native_function
    def __call__(self, func: NativeFunction) -> List[str]:
        sig = kernel_signature(func, self.backend_index)
        metadata = self.backend_index.get_kernel(func)
        assert metadata is not None
        schema = LazyIrSchema(func.func)
        all_args = schema.filtered_args()
        value_args = schema.filtered_args(values=True, scalars=False)
        returns_length = len(schema.returns)

        fallback_str = gen_fallback_code(schema, overload_name=func.func.name.overload_name)

        value_types_names = [f"{a.name}" for a in value_args if not a.is_wrapped_scalar]
        assert len(value_types_names) > 0, "Code below assumes there is at least one tensor arg"
        get_device_str = f"""auto common_device = torch::lazy::GetBackendDevice({', '.join(value_types_names)});
        TORCH_INTERNAL_ASSERT(common_device);
        """

        lazy_tensor_decls_str = lazy_tensor_decls(value_args, self.tensor_class)
        node_ctor_input_str = node_ctor_inputs(schema)

        # call the meta kernel if it exists, to compute output shape/dtype for our IR
        if func.structured or func.structured_delegate is not None:
            meta_out = """std::vector<Shape> shapes{Shape(out_meta.scalar_type(), out_meta.sizes().vec())};"""
            if returns_length > 1:
                def this_shape(i: int) -> str:
                    return f"Shape(std::get<{i}>(out_meta).scalar_type(), std::get<{i}>(out_meta).sizes().vec())"
                shapes_str = ','.join([this_shape(i) for i in range(returns_length)])
                meta_out = "std::vector<Shape> shapes{" + shapes_str + "};"

            meta_str = f"""auto out_meta = at::meta::{schema.aten_name}({', '.join(str(a.name) for a in all_args)});
        {meta_out}"""
        else:
            shape_sig = ComputeShapeSignature(metadata.kernel, func)
            meta_str = f"""
        auto shapes = {shape_sig.shape_call};"""

        meta_str += f"""
        TORCH_INTERNAL_ASSERT(shapes.size() == {returns_length});"""

        node_str = f"""auto node = torch::lazy::MakeNode<ir::ops::{schema.node_name}>({node_ctor_input_str},
                                                                                      std::move(shapes));"""
        first_tensor_name = value_types_names[0]
        bridge_str = """auto result = torch::lazy::CreateAtenFromLtcTensor(
                torch::lazy::LazyTensor::Create(std::move(node), *common_device));"""

        if returns_length > 1:
            bridge_str = f"""std::vector<{self.tensor_class}Ptr> lazy_tensors;
        for (int i = 0; i < {returns_length}; i++) {{
            lazy_tensors.push_back(torch::lazy::LazyTensor::Create(torch::lazy::Value(node, i), *common_device));
        }}
        auto result = torch::lazy::TupleAtenFromLtcTensors<{returns_length}>(lazy_tensors);"""

        if schema.name.name.inplace or func.func.is_out_fn():
            assert returns_length == 1, "We assumed there was no such case where an op is an in-place variant " \
                                        f"and has tuple outputs, but got tuple of len {returns_length}."
            bridge_str = f"""lazy_{first_tensor_name}->SetInPlaceIrValue(node);
        auto& result = {first_tensor_name};"""


        return [f"""\
    {sig.decl(name=f"{self.class_method_name}::{metadata.kernel}")} {{
        {fallback_str}
        TORCH_LAZY_FN_COUNTER("lazy::");
        {get_device_str}
        {lazy_tensor_decls_str}
        {meta_str}
        {node_str}
        {bridge_str}
        return result;
    }};\n
    """]

class ComputeShapeSignature:
    """
    Here we use the base name as the suffix of the signature to avoid generating for in-place variants.
    """
    def __init__(self, kernel_name: str, f: NativeFunction):
        self.__schema = LazyIrSchema(f.func)
        self.__dispatch_args = ', '.join([a.decl() for a in dispatcher.arguments(f.func)])
        self.__call_args = ", ".join([f"{arg.name}" for arg in self.__schema.filtered_args(generator=True)])
        self.__kernel_name = kernel_name

    def __decl_suffix(self) -> str:
        return f"{self.__kernel_name}({self.__dispatch_args})"

    def __call_suffix(self) -> str:
        return f"{self.__kernel_name}({self.__call_args})"

    @property
    def shape_decl(self) -> str:
        return f"TORCH_API std::vector<Shape> compute_shape_{self.__decl_suffix()}"

    @property
    def shape_call(self) -> str:
        return f"torch::lazy::compute_shape_{self.__call_suffix()}"


@dataclass(frozen=True)
class GenLazyShapeInferenceDefinition:
    backend_index: BackendIndex
    tensor_class: str

    @method_with_native_function
    # def gen_lazy_shape_inference_decl(f: NativeFunction, backend_index: BackendIndex, tensor_class: str) -> List[str]:
    def __call__(self, f: NativeFunction) -> List[str]:
        sig = kernel_signature(f, self.backend_index)
        metadata = self.backend_index.get_kernel(f)
        assert metadata is not None
        schema = LazyIrSchema(f.func)
        value_args = schema.filtered_args(values=True, scalars=False)
        lazy_tensor_decls_str = lazy_tensor_decls(value_args, self.tensor_class)
        node_ctor_input_str = node_ctor_inputs(schema)

        # Only generate shape/dtype fn for non-structured kernels,
        # since we just use the meta function for structured kernels
        if not f.structured and f.structured_delegate is None:
            shape_sig = ComputeShapeSignature(metadata.kernel, f)
            return ["\n".join([f"{shape_sig.shape_decl};"])]
        else:
            return []
