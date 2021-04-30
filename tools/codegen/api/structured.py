from tools.codegen.model import (Argument, BaseTy, BaseType, ListType,
                                 NativeFunctionsGroup, OptionalType,
                                 SelfArgument, TensorOptionsArguments, Type,
                                 assert_never)

from tools.codegen.api.types import (ArgName, BaseCType, Binding, ArrayRefCType,
                                     ConstRefCType, OptionalCType, NamedCType,
                                     tensorT, scalarT, intArrayRefT, dimnameListT)
from tools.codegen.api import cpp

from typing import Union, List

# This file describes the translation of JIT schema to the structured functions API.
# This is similar to native API, but a number of historical problems with native
# API have been fixed.

# Translation of types occuring in JIT arguments to a C++ argument type.
# NB: For now, mutable doesn't do anything; but it could if we make
# some more nominal types
def argumenttype_type(t: Type, *, mutable: bool, binds: ArgName) -> NamedCType:
    # If it's a value type, do the value type translation
    r = cpp.valuetype_type(t, binds=binds)
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
            raise AssertionError(
                "optional tensor not supported by structured yet; to implement this "
                "add OptionalTensor c.f. https://github.com/pytorch/pytorch/issues/51456"
            )
        elif t.elem == BaseType(BaseTy.Scalar):
            raise AssertionError(
                "optional scalar not supported by structured yet"
            )
        elem = argumenttype_type(t.elem, mutable=mutable, binds=binds)
        return NamedCType(binds, OptionalCType(elem.type))
    elif isinstance(t, ListType):
        if t.elem == BaseType(BaseTy.Tensor):
            raise AssertionError(
                "list of tensor not supported by structured yet; to implement this "
                "resolve torch::List issue, see "
                "https://fb.workplace.com/groups/894363187646754/permalink/1149276442155426"
            )
        # TODO: delete these special cases; see tools.codegen.api.cpp--these
        # must be changed in tandem, but there are problems; see
        # https://github.com/pytorch/pytorch/pull/51485
        elif str(t.elem) == 'int':
            return NamedCType(binds, BaseCType(intArrayRefT))
        elif str(t.elem) == 'Dimname':
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
        return [Binding(
            nctype=argument_type(a, binds=a.name),
            name=a.name,
            default=None,
            argument=a,
        )]
    elif isinstance(a, SelfArgument):
        return argument(a.argument)
    elif isinstance(a, TensorOptionsArguments):
        raise AssertionError("structured kernels don't support TensorOptions yet")
    else:
        assert_never(a)

def impl_arguments(g: NativeFunctionsGroup) -> List[Binding]:
    args: List[Union[Argument, TensorOptionsArguments, SelfArgument]] = []
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
