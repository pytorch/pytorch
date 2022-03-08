from typing import List, Union, Tuple
from tools.codegen.model import (Type, BaseTy, BaseType, OptionalType,
                                 ListType, OperatorName, FunctionSchema,
                                 Return, TensorOptionsArguments)
from tools.codegen.api.types import (CType, BaseCppType, BaseCType, OptionalCType,
                                     NamedCType, deviceT, layoutT,
                                     VectorCType, boolT, longT, doubleT, ListCType, stringT,
                                     scalarT, scalarTypeT, memoryFormatT)

valueT = BaseCppType('torch::lazy', 'Value')


def process_ir_type(typ: Type) -> Union[BaseCType, VectorCType, OptionalCType, ListCType]:
    """
    This function takes a type from NativeFunctions and converts it for use with
    lazy tensor codegen.  Currently its output is used in several places, and so far
    it has been possible for them to all use the same conversions, but that may not be
    optimal or possible in the finished system.

    Type conversion for lazy currently consists of
     (1) changing Tensor-like things into Value-like things
     (2) wrapping everything in a BaseCType
     (3) making reference types into values (e.g. vector instead of IntArrayRef)

    (1) converts Tensors to Values since Values are how Lazy IR represents tensors.  There
    is special handling for Optional[Tensor] or List[Tensor], etc- hence 'tensor-like'

    This is incomplete- there are assertions in places that it's expected to need to add
    more types as the codegen is used with more operators.
    """
    if isinstance(typ, BaseType):
        if typ.name == BaseTy.Tensor:
            return BaseCType(valueT)
        elif typ.name == BaseTy.Scalar:
            # at::scalar has special handling,
            # and is wrapped in an IR value just like at::tensor
            return BaseCType(valueT)
        elif typ.name == BaseTy.ScalarType:
            return BaseCType(scalarTypeT)
        elif typ.name == BaseTy.int:
            return BaseCType(longT)
        elif typ.name == BaseTy.bool:
            return BaseCType(boolT)
        elif typ.name == BaseTy.float:
            return BaseCType(doubleT)
        elif typ.name == BaseTy.str:
            return BaseCType(stringT)
        elif typ.name == BaseTy.Device:
            return BaseCType(deviceT)
        elif typ.name == BaseTy.Layout:
            return BaseCType(layoutT)
        elif typ.name == BaseTy.MemoryFormat:
            return BaseCType(memoryFormatT)
        else:
            raise AssertionError(f"TODO add support for type {repr(typ)}")
    elif isinstance(typ, OptionalType):
        return OptionalCType(process_ir_type(typ.elem))
    elif isinstance(typ, ListType):
        if str(typ.elem) == 'Tensor?':
            # TODO(whc) is this actually correct? or should it use a Vector like above
            return ListCType(OptionalCType(BaseCType(valueT)))
        else:
            return VectorCType(process_ir_type(typ.elem))
    else:
        raise AssertionError(f"unrecognized type {repr(typ)}")


def isValueType(typ: CType) -> bool:
    """
    Given a type, determine if it is a Value-like type.  This is equivalent to
    being Tensor-like, but assumes the type has already been transformed.
    """
    if isinstance(typ, BaseCType):
        # I am regretting my naming conventions, but now we are wrapping at::scalar in
        # lazy value, while preserving other 'scalar' types as scalars in the IR
        return typ.type == valueT or typ.type == scalarT
    elif isinstance(typ, (OptionalCType, ListCType, VectorCType)):
        return isValueType(typ.elem)
    else:
        return False

def isWrappedScalarType(typ: Type) -> bool:
    """
    Given a type, determine if it is a c10::scalar which we will wrap in a lazy Value.
    Since we literally change the type from scalarT to valueT, information is lost.
    This function helps build a list of wrapped scalars to save that information
    """
    if isinstance(typ, BaseType):
        # I am regretting my naming conventions, but now we are wrapping at::scalar in
        # lazy value, while preserving other 'scalar' types as scalars in the IR
        return typ.name == BaseTy.Scalar
    elif isinstance(typ, (OptionalType, ListType)):
        return isWrappedScalarType(typ.elem)
    else:
        return False


# Inspired by a FunctionSchema object, a LazyIrSchema holds the schema of a Lazy IR node.
# Unlike a FunctionSchema, it has no round-trippable string form (relating to the YAML),
# but carries type information from a native FunctionSchema modified for use with IR nodes,
# and preserving original argument names.


class LazyIrSchema:
    # The name of the operator this function schema describes.
    name: 'OperatorName'

    positional_arg_types: Tuple[NamedCType, ...]
    keyword_arg_types: Tuple[NamedCType, ...]

    # TODO: Need to handle collisions with argument names at some point
    returns: Tuple['Return', ...]

    wrapped_scalar_names: List[str]

    def __init__(self, func: FunctionSchema):

        positional_arg_types = []
        for arg_field in ["pre_self_positional",
                          "self_arg",
                          "post_self_positional"]:
            if arg_field == "self_arg" and func.arguments.self_arg is not None:
                arg = getattr(func.arguments, "self_arg").argument
                positional_arg_types.append(NamedCType(arg.name, process_ir_type(arg.type)))
            elif getattr(func.arguments, arg_field) is not None:
                positional_arg_types.extend([
                    NamedCType(
                        arg.name,
                        process_ir_type(arg.type)) for arg in getattr(func.arguments, arg_field)])
        self.positional_arg_types = tuple(positional_arg_types)

        keyword_arg_types = []
        for arg_field in ["pre_tensor_options_kwarg_only",
                          "tensor_options",
                          "post_tensor_options_kwarg_only",
                          "out"]:
            curr_args = getattr(func.arguments, arg_field)
            if curr_args is not None:
                if isinstance(curr_args, TensorOptionsArguments):
                    curr_args = curr_args.all()
                keyword_arg_types.extend([NamedCType(arg.name, process_ir_type(arg.type)) for arg in curr_args])
        self.keyword_arg_types = tuple(keyword_arg_types)
        self.name = func.name
        self.returns = func.returns
        self.wrapped_scalar_names = [arg.name for arg in func.schema_order_arguments() if isWrappedScalarType(arg.type)]

    @property
    def node_name(self) -> str:
        """
        Return camel-case version of op in node.

        Note: This function also appends any `overload_name` in the operation.
        For example, if the op is `bitwise_and.Tensor`, the returned name
        will be `BitwiseAndTensor`.
        """
        op_name = f"{self.name.name}_{self.name.overload_name}".lower()
        return "".join(word.capitalize() or "" for word in op_name.split("_"))

    @property
    def aten_name(self) -> str:
        return f"{self.name.name}"

    @property
    def base_name(self) -> str:
        return f"{self.name.name.base}"

    def filtered_types(self, positional: bool = True, keyword: bool = True,
                       values: bool = True, scalars: bool = True) -> List[NamedCType]:
        types: List[NamedCType] = []
        if positional:
            types.extend(self.positional_arg_types)
        if keyword:
            types.extend(self.keyword_arg_types)

        if values and scalars:
            return types

        if values:
            return [t for t in types if isValueType(t.type)]
        elif scalars:
            return [t for t in types if not isValueType(t.type)]

        return []

    @property
    def positional_values(self) -> List[NamedCType]:
        return self.filtered_types(positional=True, keyword=False, values=True, scalars=False)

    @property
    def positional_scalars(self) -> List[NamedCType]:
        return self.filtered_types(positional=True, keyword=False, values=False, scalars=True)

    @property
    def keyword_values(self) -> List[NamedCType]:
        return self.filtered_types(positional=False, keyword=True, values=True, scalars=False)

    @property
    def keyword_scalars(self) -> List[NamedCType]:
        return self.filtered_types(positional=False, keyword=True, values=False, scalars=True)
