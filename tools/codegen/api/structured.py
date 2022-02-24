from tools.codegen.model import (Argument, BaseTy, BaseType, ListType,
                                 NativeFunctionsGroup, OptionalType,
                                 SelfArgument, TensorOptionsArguments, Type)

from tools.codegen.api.types import (ArgName, BaseCType, Binding, ArrayRefCType,
                                     ConstRefCType, OptionalCType, NamedCType,
                                     tensorT, scalarT, intArrayRefT, dimnameListT,
                                     optionalTensorRefT, optionalScalarRefT,
                                     iTensorListT, iOptTensorRefListT)

from tools.codegen.api import cpp
from tools.codegen.utils import assert_never

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
            return NamedCType(binds, BaseCType(optionalTensorRefT))
        elif t.elem == BaseType(BaseTy.Scalar):
            return NamedCType(binds, BaseCType(optionalScalarRefT))
        elem = argumenttype_type(t.elem, mutable=mutable, binds=binds)
        return NamedCType(binds, OptionalCType(elem.type))
    elif isinstance(t, ListType):
        if t.elem == BaseType(BaseTy.Tensor):
            return NamedCType(binds, ConstRefCType(BaseCType(iTensorListT)))
        elif t.elem == OptionalType(BaseType(BaseTy.Tensor)):
            return NamedCType(binds, BaseCType(iOptTensorRefListT))
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

    if g.out.precomputed:
        # A list of parameters for the impl function with
        # certain parameters replaced with precomputed counterparts
        # as specified in native_functions.yaml.
        non_out_args_replaced: List[Union[Argument, TensorOptionsArguments, SelfArgument]] = []

        for a in g.out.func.arguments.non_out:
            if isinstance(a, Argument) and a.name in g.out.precomputed.replace:
                # If a is in precompute.replace, append the parameters
                # that should replace it onto non_out_args_replaced.
                for replacement in g.out.precomputed.replace[a.name]:
                    non_out_args_replaced.append(replacement)
            else:
                # If not, push a as it is.
                non_out_args_replaced.append(a)

        args.extend(non_out_args_replaced)
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
