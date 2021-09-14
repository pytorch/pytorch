from typing import List, Optional, Union, Tuple
import itertools
from typing_extensions import Literal
from dataclasses import dataclass
import textwrap
from tools.codegen import local
from tools.codegen.context import method_with_native_function, native_function_manager
from tools.codegen.utils import Target, mapMaybe
from tools.codegen.model import (BaseType, OptionalType, DispatchKey, NativeFunction,
                                 NativeFunctionsGroup, SchemaKind, FunctionSchema,
                                 TensorOptionsArguments, ListType,
                                 DeviceCheckType, Argument, assert_never,
                                 is_cuda_dispatch_key, BackendIndex,
                                 gets_generated_out_inplace_wrapper, OperatorName,
                                 Arguments, SelfArgument,)
from tools.codegen.api.types import (BaseTy, BaseCppType, BaseCType, OptionalCType,
                                     Binding, ConstRefCType, NamedCType,
                                     CppSignature, CppSignatureGroup,
                                     Expr, MutRefCType, kernel_signature,
                                     DispatcherSignature, VectorCType, intT, ListCType,
                                     scalarT, scalarTypeT)
import tools.codegen.api.meta as meta
import tools.codegen.api.cpp as cpp
import tools.codegen.api.structured as structured
from tools.codegen.api.translate import translate
from tools.codegen.selective_build.selector import SelectiveBuilder

# Generates {backend}_lazy_ir.h and .cpp
#
valueListT = BaseCppType('at', 'ValueList')  # TODO, not sure this type exists
valueT = BaseCppType('ir', 'Value')
intArrayT = BaseCppType('std', 'vector<int64_t>')  # TODO this should probably be different

def process_ir_type(typ: BaseTy, mutable: bool, binds: str) -> NamedCType:
    """
    This function takes a type from NativeFunctions and converts it for use with
    lazy tensor codegen.  Currently its output is used in several places, and so far
    it has been possible for them to all use the same conversions, but that may not be
    optimal or possible in the finished system.

    Type conversion for lazy currently consists of
     (1) changing Tensor-like things into Value-like things
     (2) wrapping everythign in a BaseCType
     (3) wrapping everything further in a NamedCType

    (1) converts Tensors to Values since Values are how Lazy IR represents tensors.  There
    is special handling for Optional[Tensor] or List[Tensor], etc- hence 'tensor-like'

    This is incomplete- there are assertions in places that it's expected to need to add
    more types as the codegen is used with more operators.
    """
    if isinstance(typ, BaseType):
        if typ.name == BaseTy.Tensor:
            return NamedCType(binds, BaseCType(valueT))
        elif typ.name == BaseTy.Scalar:
            return NamedCType(binds, BaseCType(scalarT))
        else:
            raise AssertionError(f"TODO add support for type {repr(typ)}")
    elif isinstance(typ, OptionalType):
        if str(typ.elem) == 'Tensor':
            return NamedCType(binds, OptionalCType(BaseCType(valueT)))
        elif str(typ.elem) == 'ScalarType':
            return NamedCType(binds, OptionalCType(BaseCType(scalarTypeT)))
        else:
            raise AssertionError(f"TODO add support for type {repr(typ)}")
    elif isinstance(typ, ListType):
        if str(typ.elem) == 'Tensor':
            return NamedCType(binds, BaseCType(valueListT))
        elif str(typ.elem) == 'Tensor?':
            return NamedCType(binds, ListCType(OptionalCType(BaseCType(valueT))))
        elif str(typ.elem) == 'int':
            return NamedCType(binds, BaseCType(intArrayT))
        else:
            raise AssertionError(f"TODO add support for type {repr(typ)}")
    else:
        raise AssertionError(f"unrecognized type {repr(typ)}")


def isValueType(typ: NamedCType) -> bool:
    """
    Given a type, determine if it is a Value-like type.  This is equivalent to
    being Tensor-like, but assumes the type has already been transformed.
    """
    if isinstance(typ.type, BaseCType):
        return typ.type.type.ns == valueT.ns and typ.type.type.name == valueT.name
    elif isinstance(typ.type, OptionalCType):
        return isValueType(typ.type.elem)
    elif isinstance(typ.type, ListCType):
        return isValueType(typ.type.elem)
    else:
        return False


def update_schema_for_lazy_ir(func: FunctionSchema) -> FunctionSchema:
    """
    We handle Tensor arguments specially as 'IR Values'.

    This function mainly deals with the lists of types buried in named fields of `Arguments`,
    delegating the actual type conversion to `process_ir_types`
    """
    new_args_fields = {}
    for arg_field in ["pre_self_positional",
                      "self_arg",
                      "post_self_positional",
                      "pre_tensor_options_kwarg_only",
                      "tensor_options",
                      "post_tensor_options_kwarg_only",
                      "out"]:
        if arg_field == "self_arg" and getattr(func.arguments, "self_arg") is not None:
            arg = getattr(func.arguments, "self_arg").argument
            new_args_fields[arg_field] = SelfArgument(Argument(arg.name, process_ir_type(
                arg.type, arg.is_write, arg.name), arg.default, arg.annotation))
        elif getattr(func.arguments, arg_field) is not None:
            new_args_fields[arg_field] = [
                Argument(
                    arg.name,
                    process_ir_type(arg.type, arg.is_write, arg.name), arg.default, arg.annotation) for arg in getattr(func.arguments, arg_field)]
        else:
            new_args_fields[arg_field] = None
    new_args = Arguments(**new_args_fields)
    return FunctionSchema(func.name, new_args, func.returns)


def separate_value_scalar_types(func: FunctionSchema) -> Tuple[List[NamedCType], List[NamedCType], List[NamedCType]]:
    """
    Since there are significant differences in how Values and 'everything else' are handled in the IR node class,
    it's useful to have a way to get a list of just the values, just the scalars, or all of them.
    """
    types = [arg.type for arg in func.schema_order_arguments()]
    value_types = [t for t in types if isValueType(t)]
    scalar_types = [t for t in types if not isValueType(t)]
    return types, value_types, scalar_types


def node_ctor_inputs(value_types: List[NamedCType], scalar_types: List[NamedCType]) -> str:
    """
    Produce a formatted string with the arguments as passed into the constructor of a node class.
    """
    node_ctor_values = []
    for t in value_types:
        if isinstance(t.type, BaseCType):
            node_ctor_values.append(f"l_{t.name}.GetIrValue()")
        elif isinstance(t.type, OptionalCType):
            node_ctor_values.append(
                f"l_{t.name}.has_value() ? l_{t.name}.value().GetIrValue() : torch_lazy_tensors::ir::ops::kNullValue")
        else:
            assert False, ""

    node_ctor_scalars = []
    for t in scalar_types:
        if isinstance(t.type, BaseCType) and t.type.type.name == "vector<int64_t>":
            node_ctor_scalars.append(f"std::vector<int64_t>({t.name}.begin(), {t.name}.end())")
        else:
            node_ctor_scalars.append(t.name)

    node_ctor_inputs = ",\n                              ".join(node_ctor_values + node_ctor_scalars)
    return node_ctor_inputs


def ir_node_name(func: FunctionSchema):
    return str(func.name).lower().capitalize()


@dataclass(frozen=True)
class LazyIR:
    backend_index: BackendIndex

    # Names of operators we want to codegen for, a subset of backend_index
    codegen: List[OperatorName]

    @method_with_native_function
    def __call__(self, f: Union[NativeFunctionsGroup, NativeFunction]) -> List[str]:
        func = f.functional.func if isinstance(f, NativeFunctionsGroup) else f.func
        if func.name in self.codegen:
            return self.gen(f)
        else:
            return []

    def gen(self, f: Union[NativeFunctionsGroup, NativeFunction]) -> List[str]:
        # for now, we just want one IR class decl and soon after also the method defs
        # and we use the functional version not out/inplace.
        func = f.functional.func if isinstance(f, NativeFunctionsGroup) else f.func
        class_name = ir_node_name(func)

        schema = update_schema_for_lazy_ir(func)
        all_types, value_types, scalar_types = separate_value_scalar_types(schema)

        node_ctor_args = ", ".join([f"{i.cpp_type()} {i.name}" for i in all_types])
        scalar_initializers = ",\n        ".join([f"{t.name}_({t.name})" for t in scalar_types])
        comma_if_scalar_initializers = ",\n" if len(scalar_initializers) else ""
        scalar_decls = "\n  ".join([f"{t.cpp_type()} {t.name}_;" for t in scalar_types])
        scalar_hashes = ", ".join([f.name for f in scalar_types])
        base_ctor_value_args = []
        for t in value_types:
            if isinstance(t.type, BaseCType):
                base_ctor_value_args.append(f"{t.name}")
            elif isinstance(t.type, OptionalCType):
                base_ctor_value_args.append(f"{t.name}.has_value() ? {t.name}.value() : kNullValue")
            else:
                assert False, ""
        base_ctor_value_args = ", ".join(base_ctor_value_args)
        members_to_string = "\n    ".join([f'lazy_tensors::ToString("{t.name}", {t.name}_, ss);' for t in scalar_types])

        # clone needs hand-overrides for cases where there are optional Tensor? args,
        # unless we clean up the OpList API to deal unambiguously with optionals.
        clone_impl_args = ",".join(
            [f"operands.at({i})" for i in range(len(value_types))] +
            [f"{s.name}_" for s in scalar_types] +
            ["out_dtype_", "out_shape_"])
        if any([isinstance(t.type, OptionalCType) for t in value_types]):
            scalar_args = ",".join([f"{s.name}_" for s in scalar_types])
            clone_impl = f"return Clone{class_name}(operands, {scalar_args});"
            clone_decl_args = ", ".join([f"{i.cpp_type()} {i.name}" for i in scalar_types])
            clone_handcoded_decl = f"NodePtr Clone{class_name}(OpList operands, {clone_decl_args});"
        else:
            clone_impl = f"ir::MakeNode<ir::ops::{ir_node_name(func)}>({clone_impl_args});"
            clone_handcoded_decl = ""
        

        return [f"""\
{clone_handcoded_decl}
class {class_name} : public Node {{
 public:
  {class_name}({node_ctor_args}, at::ScalarType out_dtype, std::vector<int64_t> out_shape)
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
