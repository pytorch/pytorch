from typing import List, Optional, Union, Tuple
import itertools
from typing_extensions import Literal
from dataclasses import dataclass
import textwrap
from tools.codegen import local
from tools.codegen.context import method_with_native_function, native_function_manager
from tools.codegen.utils import Target, mapMaybe
from tools.codegen.model import (Type, BaseTy, BaseType, OptionalType, DispatchKey, NativeFunction,
                                 NativeFunctionsGroup, SchemaKind, FunctionSchema,
                                 TensorOptionsArguments, ListType,
                                 DeviceCheckType, Argument, assert_never,
                                 is_cuda_dispatch_key, BackendIndex,
                                 gets_generated_out_inplace_wrapper, OperatorName,
                                 Arguments, SelfArgument, Return)
from tools.codegen.api.types import (BaseCppType, BaseCType, OptionalCType,
                                     Binding, ConstRefCType, NamedCType,
                                     CppSignature, CppSignatureGroup,
                                     Expr, MutRefCType, kernel_signature,
                                     DispatcherSignature, VectorCType, intT, ListCType,
                                     scalarT, scalarTypeT, ArrayRefCType, ArrayCType, TupleCType)
import tools.codegen.api.dispatcher as dispatcher
import tools.codegen.api.meta as meta
import tools.codegen.api.cpp as cpp
import tools.codegen.api.structured as structured
from tools.codegen.api.translate import translate
from tools.codegen.selective_build.selector import SelectiveBuilder

valueT = BaseCppType('ir', 'Value')

def process_ir_type(typ: Type) -> Union[BaseCType, OptionalCType, ListCType]:
    """
    This function takes a type from NativeFunctions and converts it for use with
    lazy tensor codegen.  Currently its output is used in several places, and so far
    it has been possible for them to all use the same conversions, but that may not be
    optimal or possible in the finished system.

    Type conversion for lazy currently consists of
     (1) changing Tensor-like things into Value-like things
     (2) wrapping everythign in a BaseCType

    (1) converts Tensors to Values since Values are how Lazy IR represents tensors.  There
    is special handling for Optional[Tensor] or List[Tensor], etc- hence 'tensor-like'

    This is incomplete- there are assertions in places that it's expected to need to add
    more types as the codegen is used with more operators.
    """
    if isinstance(typ, BaseType):
        if typ.name == BaseTy.Tensor:
            return BaseCType(valueT)
        elif typ.name == BaseTy.Scalar:
            return BaseCType(scalarT)
        else:
            raise AssertionError(f"TODO add support for type {repr(typ)}")
    elif isinstance(typ, OptionalType):
        if str(typ.elem) == 'Tensor':
            return OptionalCType(BaseCType(valueT))
        elif str(typ.elem) == 'ScalarType':
            return OptionalCType(BaseCType(scalarTypeT))
        else:
            raise AssertionError(f"TODO add support for type {repr(typ)}")
    elif isinstance(typ, ListType):
        if str(typ.elem) == 'Tensor':
            return BaseCType(VectorCType(BaseCType(valueT)))
        elif str(typ.elem) == 'Tensor?':
            # TODO(whc) is this actually correct? or should it use a Vector like above
            return ListCType(OptionalCType(BaseCType(valueT)))
        elif str(typ.elem) == 'int':
            return BaseCType(VectorCType(BaseCType(intT)))
        else:
            raise AssertionError(f"TODO add support for type {repr(typ)}")
    else:
        raise AssertionError(f"unrecognized type {repr(typ)}")


def isValueType(typ: Union[Type, BaseCType, OptionalCType, ConstRefCType, MutRefCType, ListCType, ArrayRefCType, ArrayCType, VectorCType, TupleCType]) -> bool:
    """
    Given a type, determine if it is a Value-like type.  This is equivalent to
    being Tensor-like, but assumes the type has already been transformed.
    """
    if isinstance(typ, BaseCType):
        return typ.type.ns == valueT.ns and typ.type.name == valueT.name
    elif isinstance(typ, OptionalCType):
        return isValueType(typ.elem)
    elif isinstance(typ, ListCType):
        return isValueType(typ.elem)
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
                arg.type), arg.default, arg.annotation))
        elif getattr(func.arguments, arg_field) is not None:
            new_args_fields[arg_field] = [
                Argument(
                    arg.name,
                    process_ir_type(arg.type), arg.default, arg.annotation) for arg in getattr(func.arguments, arg_field)]
        else:
            new_args_fields[arg_field] = None
    new_args = Arguments(**new_args_fields)
    return FunctionSchema(func.name, new_args, func.returns)


def separate_value_scalar_types(func: FunctionSchema) -> Tuple[List[NamedCType], List[NamedCType], List[NamedCType]]:
    """
    Since there are significant differences in how Values and 'everything else' are handled in the IR node class,
    it's useful to have a way to get a list of just the values, just the scalars, or all of them.
    """
    types = [NamedCType(arg.name, arg.type) for arg in func.schema_order_arguments()]
    value_types = [NamedCType(arg.name, arg.type) for arg in func.schema_order_arguments() if isValueType(arg.type)]
    scalar_types = [NamedCType(arg.name, arg.type) for arg in func.schema_order_arguments() if not isValueType(arg.type)]
    return types, value_types, scalar_types


def ir_node_name(func: FunctionSchema) -> str:
    return str(func.name).lower().capitalize()
