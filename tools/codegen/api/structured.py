from tools.codegen.model import *

from tools.codegen.api.types import *
import tools.codegen.api.cpp as cpp

from typing import Union, List

# This file describes the translation of JIT schema to the structured functions API.
# This is similar to native API, but a number of historical problems with native
# API have been fixed.

def name(func: FunctionSchema) -> str:
    name = str(func.name.name)
    if func.name.overload_name:
        name += f'_{func.name.overload_name}'
    return name

# Translation of types occuring in JIT arguments to a C++ argument type.
# NB: For now, mutable doesn't do anything; but it could if we make
# some more nominal types
def argumenttype_type(t: Type, *, mutable: bool, binds: ArgName) -> CType:
    # If it's a value type, do the value type translation
    r = cpp.valuetype_type(t, binds=binds)
    if r is not None:
        return r

    if isinstance(t, BaseType):
        if t.name == BaseTy.Tensor:
            return ConstRefCType(BaseCType('Tensor', binds))
        else:
            raise AssertionError(f"base type should have been value type {t}")
    elif isinstance(t, OptionalType):
        if t.elem == BaseType(BaseTy.Tensor):
            raise AssertionError(
                "optional tensor not supported by structured yet; to implement this "
                "add OptionalTensor c.f. https://github.com/pytorch/pytorch/issues/51456"
            )
        elem = argumenttype_type(t.elem, mutable=mutable, binds=binds)
        return OptionalCType(elem)
    elif isinstance(t, ListType):
        if t.elem == BaseType(BaseTy.Tensor):
            raise AssertionError(
                "list of tensor not supported by structured yet; to implement this "
                "resolve torch::List issue, see "
                "https://fb.workplace.com/groups/894363187646754/permalink/1149276442155426"
            )
        elem = argumenttype_type(t.elem, mutable=mutable, binds=binds)
        return BaseCType(f"ArrayRef<{elem.cpp_type()}>", binds)
    else:
        raise AssertionError(f"unrecognized type {repr(t)}")

def argument_type(a: Argument, *, binds: ArgName) -> CType:
    return argumenttype_type(a.type, mutable=a.is_write, binds=binds)

# returns_type intentionally omitted

# Structured kernels are never defaulted
def argument(a: Union[Argument, SelfArgument, TensorOptionsArguments]) -> List[Binding]:
    if isinstance(a, Argument):
        return [Binding(
            ctype=argument_type(a, binds=a.name),
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

def impl_arguments(g: StructuredNativeFunctions) -> List[Binding]:
    args: List[Union[Argument, TensorOptionsArguments, SelfArgument]] = []
    args.extend(g.out.func.arguments.non_out)
    args.extend(g.out.func.arguments.out)
    return [r for arg in args for r in argument(arg)]

def meta_arguments(g: StructuredNativeFunctions) -> List[Binding]:
    args: List[Union[Argument, TensorOptionsArguments, SelfArgument]] = []
    args.extend(g.functional.func.arguments.non_out)
    return [r for arg in args for r in argument(arg)]

def out_arguments(g: StructuredNativeFunctions) -> List[Binding]:
    args: List[Union[Argument, TensorOptionsArguments, SelfArgument]] = []
    args.extend(g.out.func.arguments.out)
    return [r for arg in args for r in argument(arg)]
