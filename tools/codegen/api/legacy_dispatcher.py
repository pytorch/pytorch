from tools.codegen.model import *

from tools.codegen.api.types import TensorOptionsArguments, LegacyDispatcherArgument, ThisArgument
import tools.codegen.api.cpp as cpp

from typing import Union, Sequence

# This file describes the translation of JIT schema to the legacy
# dispatcher API.  This looks a lot like the C++ API (which
# makes historical sense, because historically the dispatcher API
# and the C++ API exactly matched), but over time we have
# evolved the C++ API without actually changing our native::
# kernels.  To be deleted eventually.  Dispatcher calls use
# this when you are not use_c10_dispatcher: full.

def name(func: FunctionSchema) -> str:
    name = str(func.name.name)
    # TODO: delete this!
    if func.is_out_fn():
        name += '_out'
    if func.name.overload_name:
        name += f'_{func.name.overload_name}'
    return name

def argumenttype_type(t: Type, *, mutable: bool) -> str:
    if str(t) == 'Tensor?':
        if mutable:
            return 'Tensor &'
        else:
            return 'const Tensor &'
    elif str(t) == 'Tensor?[]':
        return 'TensorList'
    return cpp.argumenttype_type(t, mutable=mutable)

def returns_type(rs: Sequence[Return]) -> str:
    return cpp.returns_type(rs)

def argument_type(a: Argument) -> str:
    return argumenttype_type(a.type, mutable=a.is_write)

def argument(a: Union[Argument, ThisArgument, TensorOptionsArguments]) -> LegacyDispatcherArgument:
    if isinstance(a, Argument):
        return LegacyDispatcherArgument(
            type=argument_type(a),
            name=a.name,
            default=cpp.default_expr(a.default, a.type) if a.default is not None else None,
            argument=a,
        )
    elif isinstance(a, ThisArgument):
        # Erase ThisArgument from the distinction
        return LegacyDispatcherArgument(
            type=argument_type(a.argument),
            name=a.argument.name,
            default=None,
            argument=a.argument,
        )
    elif isinstance(a, TensorOptionsArguments):
        # TODO: expunge this logic entirely
        default = None
        if all(x.default == "None" for x in a.all()):
            default = '{}'
        elif a.dtype.default == "long":
            default = 'at::kLong'  # TODO: this is wrong
        return LegacyDispatcherArgument(
            type='const TensorOptions &',
            name='options',
            default=default,
            argument=a,
        )
    else:
        assert_never(a)

def arguments(func: FunctionSchema) -> Sequence[LegacyDispatcherArgument]:
    return list(map(argument, cpp.group_arguments(func)))
