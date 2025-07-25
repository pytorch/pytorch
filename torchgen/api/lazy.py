from __future__ import annotations

from typing import Any

from torchgen.api.types import (
    BaseCppType,
    BaseCType,
    boolT,
    CType,
    deviceT,
    doubleT,
    generatorT,
    layoutT,
    ListCType,
    longT,
    memoryFormatT,
    NamedCType,
    OptionalCType,
    scalarT,
    scalarTypeT,
    stringT,
    SymIntT,
    VectorCType,
)
from torchgen.model import (
    Argument,
    BaseTy,
    BaseType,
    FunctionSchema,
    ListType,
    OperatorName,
    OptionalType,
    Return,
    TensorOptionsArguments,
    Type,
)


_valueT: BaseCppType | None = None


# A ValueT is an IR type which represents the computation of a Tensor.  In other
# words, a PyTorch user will do operations on lazy tensors, and each output lazy
# tensor internally tracks a ValueT representing the IR node that would have
# actually produced the value of this tensor for real.
#
# This is configurable because different lazy tensor backends (LTC vs XLA) will
# have different IR representations.  (Though, arguably, after unification they
# shouldn't!)
def getValueT() -> BaseCppType:
    if not _valueT:
        raise NotImplementedError(
            "The value type needs to be set with setValueT() in run_gen_lazy_tensor()"
        )

    return _valueT


def setValueT(val: BaseCppType) -> None:
    global _valueT
    _valueT = val


# this is a bad hack. I need to refactor the data model to represent each arg in the schema as an object,
# making it easier to represent special properties of an arg.
tensorListValueT = BaseCppType("torch::lazy", "Value")


def process_ir_type(
    typ: Type, properties: LazyIrProperties, *, symint: bool
) -> BaseCType | VectorCType | OptionalCType | ListCType:
    """
    This function takes a type from NativeFunctions and converts it for use with
    lazy tensor codegen.

    Type conversion for lazy currently consists of
     (1) changing at::Tensors into lazy::Values
     (2) wrapping everything in a BaseCType
     (3) making cpp-reference types into cpp-value types (e.g. vector instead of IntArrayRef)

    (1) converts at::Tensors to lazy::Values (which wrap lazy::Nodes, with which Lazy IR represents tensors.)
    There is special handling for Optional[Tensor] or list[Tensor], etc- hence 'tensor-like'

    This is incomplete- there are assertions in places that it's expected to need to add
    more types as the codegen is used with more operators.
    """
    if isinstance(typ, BaseType):
        if typ.name == BaseTy.Tensor:
            return BaseCType(getValueT())
        elif typ.name == BaseTy.Scalar:
            if properties.TreatScalarsAsConstants:
                return BaseCType(scalarT)
            # at::scalar has special handling,
            # and is wrapped in an lazy::Value just like at::tensor
            return BaseCType(getValueT())
        elif typ.name == BaseTy.ScalarType:
            return BaseCType(scalarTypeT)
        elif typ.name == BaseTy.int:
            return BaseCType(longT)
        elif typ.name == BaseTy.SymInt:
            if symint:
                return BaseCType(getValueT())
            else:
                return BaseCType(longT)
        elif typ.name == BaseTy.bool:
            return BaseCType(boolT)
        elif typ.name == BaseTy.float:
            return BaseCType(doubleT)
        elif typ.name == BaseTy.str:
            return BaseCType(stringT)
        elif typ.name == BaseTy.Device:
            return BaseCType(deviceT)
        elif typ.name == BaseTy.Generator:
            return BaseCType(generatorT)
        elif typ.name == BaseTy.Layout:
            return BaseCType(layoutT)
        elif typ.name == BaseTy.MemoryFormat:
            return BaseCType(memoryFormatT)
        else:
            raise AssertionError(f"TODO add support for type {repr(typ)}")
    elif isinstance(typ, OptionalType):
        return OptionalCType(process_ir_type(typ.elem, properties, symint=symint))
    elif isinstance(typ, ListType):
        if str(typ.elem) == "Tensor?":
            # TODO(whc) is this actually correct? or should it use a Vector like above
            return ListCType(OptionalCType(BaseCType(getValueT())))
        elif str(typ.elem) == "Tensor":
            # this is a TensorList which comes in from GetTensorList as a Value
            return BaseCType(tensorListValueT)
        elif typ.elem == BaseType(BaseTy.SymInt):
            # TODO: return a value type.  The problem here is analogous to
            # the problem with tensorListValueT: if you have SymInt[] you
            # cannot conveniently save the list of Value directly, as nodes
            # expect to save values as a vector for ALL arguments.  So you
            # need a separate IR node that represents all of the size nodes
            # assembled into a list.  I'm not an LTC dev so I don't want to
            # figure it out right now.  Y'all figure it out...
            return VectorCType(BaseCType(longT))

        else:
            return VectorCType(process_ir_type(typ.elem, properties, symint=symint))
    else:
        raise AssertionError(f"unrecognized type {repr(typ)}")


# TODO: Determining this based off of CType is bad; this should be computed
# from Type directly; then the same logic as process_ir_type can be used
#
# Invariant: passed typ should be an *owning* CType (e.g., we will report
# that ArrayRef<Value> is NOT a value type)
def isValueType(typ: CType, properties: LazyIrProperties | None = None) -> bool:
    """
    Given a type, determine if it is a Value-like type.  This is equivalent to
    being Tensor-like, but assumes the type has already been transformed.
    """
    if isinstance(typ, BaseCType):
        # I am regretting my naming conventions, but now we are wrapping at::scalar in
        # lazy value, while preserving other 'scalar' types as scalars in the IR
        treat_scalars_as_constants = properties and properties.TreatScalarsAsConstants
        return (
            typ.type == getValueT()
            or (typ.type == scalarT and not treat_scalars_as_constants)
            or typ.type == SymIntT
        )
    elif typ == VectorCType(BaseCType(SymIntT)):
        # TODO: report True for this
        return False
    elif isinstance(typ, (OptionalCType, ListCType, VectorCType)):
        return isValueType(typ.elem, properties)
    return False


def isSymIntType(typ: Type) -> bool:
    return isinstance(typ, BaseType) and typ.name == BaseTy.SymInt


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
    return False


# TODO: dedupe with Type.is_generator_like
def isGeneratorType(typ: Type) -> bool:
    if isinstance(typ, BaseType):
        return typ.name == BaseTy.Generator
    elif isinstance(typ, (OptionalType)):
        return isGeneratorType(typ.elem)
    return False


# This class caches a few derived properties computed from an Argument
# and LazyIrProperties
class LazyArgument:
    name: str
    orig_type: Type
    lazy_type_: CType | None
    is_wrapped_scalar: bool
    is_generator: bool
    # TODO: this is lies, it is false for symint list
    is_symint_or_list: bool

    # Whether or not we are treating this as symint or not
    symint: bool

    # true if this argument is or contains a lazy IR value
    is_lazy_value: bool

    def __init__(
        self, arg: Argument, properties: LazyIrProperties, *, symint: bool
    ) -> None:
        self.name = arg.name
        self.orig_type = arg.type
        self.symint = symint
        self.is_optional = isinstance(arg.type, OptionalType)
        self.is_generator = isGeneratorType(arg.type)
        self.lazy_type_ = process_ir_type(arg.type, properties, symint=symint)
        self.is_wrapped_scalar = isWrappedScalarType(arg.type)
        self.is_symint_or_list = symint and (
            isSymIntType(arg.type)
            or (isinstance(arg.type, OptionalType) and isSymIntType(arg.type.elem))
            # TODO: lists of symints are not currently treated as value types
            # or (isinstance(arg.type, ListType) and isSymIntType(arg.type.elem))
        )

        self.is_lazy_value = isValueType(self.lazy_type, properties)

    @property
    def lazy_type(self) -> CType:
        assert self.lazy_type_ is not None, (
            f"Attempted to access lazy_type for invalid argument {self.name}"
        )
        return self.lazy_type_


class LazyIrProperties:
    """Collection of properties for an IR node

    The property groups are listed below. Each group is mutually
    exclusive, meaning that only one property from each group can be True
    at any one time. The properties can be accessed as if they were normal
    attributes. The mutual exclusivity is automatically handled.
    """

    Properties: tuple[tuple[str, ...], ...] = (
        (
            "ShapePrecompute",  # Assume shape has been precomputed
            "ShapeCompute",  # Need to compute the shape on construction
            "ShapeCache",  # Utilize the shape cache to defer computation
        ),
        (
            "Lower",  # Codegen full lower function
            "LowerDeclOnly",  # Codegen only lower function declaration
        ),
        (
            "CanBeReused",  # Codegen full reuse function
            "CanBeReusedDeclOnly",  # Codegen only reuse function declaration
        ),
        (
            "CreateFn",  # Codegen full create function
            "CreateFnDeclOnly",  # Codegen only create function declaration
        ),
        (
            "TreatScalarsAsConstants",  # Treat Scalars as constants instead of handling like values
        ),
    )

    def __init__(self, *default_properties: str) -> None:
        properties: dict[tuple[str, ...], str | None] = dict.fromkeys(
            LazyIrProperties.Properties
        )
        self.__dict__["properties"] = properties
        for p in default_properties:
            setattr(self, p, True)

    def __getattr__(self, key: str) -> Any:
        properties = self.__dict__["properties"]
        for values in LazyIrProperties.Properties:
            if key in values:
                return properties[values] == key

        return self.__getattribute__(key)

    def __setattr__(self, key: str, value: Any) -> Any:
        properties = self.__dict__["properties"]
        for values in LazyIrProperties.Properties:
            if key in values:
                properties[values] = key if value else None
                return value

        raise KeyError(f"Invalid property: {key}")


# Inspired by a FunctionSchema object, a LazyIrSchema holds the schema of a Lazy IR node.
# Unlike a FunctionSchema, it has no round-trippable string form (relating to the YAML),
# but carries type information from a native FunctionSchema modified for use with IR nodes,
# and preserving original argument names.
#
# TODO: This is not idiomatic with how other torchgen APIs transform on schema.
class LazyIrSchema:
    # The name of the operator this function schema describes.
    name: OperatorName

    positional_args: tuple[LazyArgument, ...]
    keyword_args: tuple[LazyArgument, ...]

    # TODO: Need to handle collisions with argument names at some point
    returns: tuple[Return, ...]

    # if this schema has a Generator arg, list its orig ctype/name but don't
    # build a LazyArgument since lazy IR doesn't support it
    generator_arg: NamedCType | None = None

    # original function schema
    func: FunctionSchema

    # Whether or not we are code-genning for SymInt or not
    symint: bool

    properties: LazyIrProperties = LazyIrProperties(
        # default properties
        "ShapePrecompute",
        "Lower",
        "CanBeReused",
    )
    opkind: str | None = None

    def __init__(
        self,
        func: FunctionSchema,
        properties: LazyIrProperties | None = None,
        *,
        symint: bool,
    ) -> None:
        if properties:
            self.properties = properties

        self.func = func
        self.symint = symint
        positional_args: list[LazyArgument] = []
        for arg_field in ["pre_self_positional", "self_arg", "post_self_positional"]:
            if arg_field == "self_arg" and func.arguments.self_arg is not None:
                arg = func.arguments.self_arg.argument
                positional_args.append(
                    LazyArgument(arg, self.properties, symint=symint)
                )
            elif getattr(func.arguments, arg_field) is not None:
                positional_args.extend(
                    LazyArgument(arg, self.properties, symint=symint)
                    for arg in getattr(func.arguments, arg_field)
                )
        self.positional_args = tuple(positional_args)

        keyword_args: list[LazyArgument] = []
        for arg_field in [
            "pre_tensor_options_kwarg_only",
            "tensor_options",
            "post_tensor_options_kwarg_only",
            "out",
        ]:
            curr_args = getattr(func.arguments, arg_field)
            if curr_args is not None:
                if isinstance(curr_args, TensorOptionsArguments):
                    curr_args = curr_args.all()
                for arg in curr_args:
                    if isGeneratorType(arg.type):
                        assert self.generator_arg is None, (
                            "We expect there is only one generator arg"
                        )
                        self.generator_arg = NamedCType(
                            arg.name,
                            arg.type,  # type:ignore[arg-type]
                        )
                keyword_args.extend(
                    LazyArgument(arg, self.properties, symint=symint)
                    for arg in curr_args
                )
        self.keyword_args = tuple(keyword_args)
        self.name = func.name
        self.returns = func.returns

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
        return str(self.name.name)

    @property
    def base_name(self) -> str:
        return f"{self.name.name.base}"

    def filtered_args(
        self,
        positional: bool = True,
        keyword: bool = True,
        values: bool = True,
        scalars: bool = True,
        generator: bool = True,
    ) -> list[LazyArgument]:
        # This function maintains the sorted order of arguments but provides different filtered views.
        # Some parts of the code care about kwargs vs args (TS lowerings),
        # other parts care about whether they need to wrap the arg in a lazy value or leave it alone.
        # Generators are special cased, as they are needed for fallback/shape-inference but not supported
        # in TS lowerings and therefore also omitted from lazy IR.
        args: list[LazyArgument] = []
        if positional:
            args.extend(self.positional_args)
        if keyword:
            args.extend(self.keyword_args)

        if values and scalars and generator:
            return args
        elif values and scalars:
            return [a for a in args if not a.is_generator]
        elif values:
            return [a for a in args if a.is_lazy_value]
        elif scalars:
            return [
                a
                for a in args
                if not a.is_lazy_value and (generator or not a.is_generator)
            ]

        return []

    @property
    def positional_values(self) -> list[LazyArgument]:
        return self.filtered_args(
            positional=True, keyword=False, values=True, scalars=False
        )

    @property
    def positional_scalars(self) -> list[LazyArgument]:
        return self.filtered_args(
            positional=True, keyword=False, values=False, scalars=True
        )

    @property
    def keyword_values(self) -> list[LazyArgument]:
        return self.filtered_args(
            positional=False, keyword=True, values=True, scalars=False
        )

    @property
    def keyword_scalars(self) -> list[LazyArgument]:
        return self.filtered_args(
            positional=False, keyword=True, values=False, scalars=True
        )
