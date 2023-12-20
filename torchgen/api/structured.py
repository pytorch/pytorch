from typing import List, Union

from torchgen.api import cpp

from torchgen.api.types import (
    ArgName,
    ArrayRefCType,
    BaseCType,
    Binding,
    ConstRefCType,
    dimnameListT,
    intArrayRefT,
    iOptTensorListRefT,
    iTensorListRefT,
    NamedCType,
    OptionalCType,
    optionalIntArrayRefT,
    optionalScalarRefT,
    optionalTensorRefT,
    scalarT,
    tensorT,
)
from torchgen.model import (
    Argument,
    BaseTy,
    BaseType,
    ListType,
    NativeFunctionsGroup,
    OptionalType,
    SelfArgument,
    TensorOptionsArguments,
    Type,
)
from torchgen.utils import assert_never

# This file describes the translation of JIT schema to the structured functions API.
# This is similar to native API, but a number of historical problems with native
# API have been fixed.


# Translation of types occurring in JIT arguments to a C++ argument type.
# NB: For now, mutable doesn't do anything; but it could if we make
# some more nominal types
def argumenttype_type(t: Type, *, mutable: bool, binds: ArgName) -> NamedCType:
    # If it's a value type, do the value type translation
    # NB: structured kernels ALWAYS have symint off, since they involve actual
    # kernels that require real ints.  The one exception is the
    # CompositeExplicitAutograd and the meta function (which could
    # hypothetically be SymInt), but for simplicity we plan for these to just
    # be handled in Python
    r = cpp.valuetype_type(t, symint=False, binds=binds)
    if r is not None:
        return r

    if isinstance(t, BaseType):
        if t.name == BaseTy.Tensor:
            return NamedCType(binds, ConstRefCType(BaseCType(tensorT)))
        elif t.name == BaseTy.Scalar:
            return NamedCType(binds, ConstRefCType(BaseCType(scalarT)))
        else:
            raise AssertionError(f"base type should have been value type {t}")
    elif isinstance(t, OptionalType):
        if t.elem == BaseType(BaseTy.Tensor):
            return NamedCType(binds, BaseCType(optionalTensorRefT))
        elif t.elem == BaseType(BaseTy.Scalar):
            return NamedCType(binds, BaseCType(optionalScalarRefT))
        elif isinstance(t.elem, ListType) and str(t.elem.elem) == "int":
            return NamedCType(binds, BaseCType(optionalIntArrayRefT))
        elem = argumenttype_type(t.elem, mutable=mutable, binds=binds)
        return NamedCType(binds, OptionalCType(elem.type))
    elif isinstance(t, ListType):
        if t.elem == BaseType(BaseTy.Tensor):
            return NamedCType(binds, ConstRefCType(BaseCType(iTensorListRefT)))
        elif t.elem == OptionalType(BaseType(BaseTy.Tensor)):
            return NamedCType(binds, BaseCType(iOptTensorListRefT))
        # TODO: delete these special cases; see torchgen.api.cpp--these
        # must be changed in tandem, but there are problems; see
        # https://github.com/pytorch/pytorch/pull/51485
        elif str(t.elem) == "int":
            return NamedCType(binds, BaseCType(intArrayRefT))
        elif str(t.elem) == "Dimname":
            return NamedCType(binds, BaseCType(dimnameListT))
        elem = argumenttype_type(t.elem, mutable=mutable, binds=binds)
        return NamedCType(binds, ArrayRefCType(elem.type))
    else:
        raise AssertionError(f"unrecognized type {repr(t)}")


def argument_type(a: Argument, *, binds: ArgName) -> NamedCType:
    return argumenttype_type(a.type, mutable=a.is_write, binds=binds)


# returns_type intentionally omitted, because structured kernels never "return";
# instead, they always indirectly report their outputs (in the case of a meta
# function, by calling set_output; in the case of an impl function, by writing
# directly into the provided out argument).


# Structured kernels are never defaulted
def argument(a: Union[Argument, SelfArgument, TensorOptionsArguments]) -> List[Binding]:
    if isinstance(a, Argument):
        return [
            Binding(
                nctype=argument_type(a, binds=a.name),
                name=a.name,
                default=None,
                argument=a,
            )
        ]
    elif isinstance(a, SelfArgument):
        return argument(a.argument)
    elif isinstance(a, TensorOptionsArguments):
        raise AssertionError("structured kernels don't support TensorOptions yet")
    else:
        assert_never(a)


def impl_arguments(g: NativeFunctionsGroup) -> List[Binding]:
    args: List[Union[Argument, TensorOptionsArguments, SelfArgument]] = []

    if g.out.precomputed:
        # A list of parameters for the impl function with
        # certain parameters replaced with precomputed counterparts
        # as specified in native_functions.yaml.
        non_out_args_replaced: List[
            Union[Argument, TensorOptionsArguments, SelfArgument]
        ] = []
        for a in g.out.func.arguments.non_out:
            if isinstance(a, Argument) and a.name in g.out.precomputed.replace:
                # If a is in precompute.replace, append the parameters
                # that should replace it onto non_out_args_replaced.
                non_out_args_replaced.extend(g.out.precomputed.replace[a.name])
            else:
                # If not, push a as it is.
                non_out_args_replaced.append(a)

        args.extend(non_out_args_replaced)
        # g.out.precomputed.add is the list of parameters that are added
        # without replacement after the non out args and just before the out args
        args.extend(g.out.precomputed.add)
    else:
        args.extend(g.out.func.arguments.non_out)

    args.extend(g.out.func.arguments.out)
    return [r for arg in args for r in argument(arg)]


def meta_arguments(g: NativeFunctionsGroup) -> List[Binding]:
    args: List[Union[Argument, TensorOptionsArguments, SelfArgument]] = []
    args.extend(g.functional.func.arguments.non_out)
    return [r for arg in args for r in argument(arg)]


def out_arguments(g: NativeFunctionsGroup) -> List[Binding]:
    args: List[Union[Argument, TensorOptionsArguments, SelfArgument]] = []
    args.extend(g.out.func.arguments.out)
    return [r for arg in args for r in argument(arg)]
