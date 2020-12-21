from tools.codegen.model import *

from tools.codegen.api.types import *
import tools.codegen.api.cpp as cpp
from tools.codegen import local

from typing import Union, Sequence, List

# This file describes the translation of JIT schema to the native functions API.
# This looks a lot like the C++ API (which makes historical sense, because the
# idea was you wrote native functions to implement functions in the C++ API),
# but over time we have evolved the C++ API without actually changing our
# native:: kernels.  The intention is to make native API and dispatcher API
# line up as closely as possible, since this results in the least overhead
# (no translation is needed from dispatcher API to native API).
#
# When a function is not use_c10_dispatcher: full, the dispatcher API actually
# coincides with the native:: API (e.g., we do as dumb as pass through as
# possible).

def name(func: FunctionSchema) -> str:
    name = str(func.name.name)
    # TODO: delete this!
    if func.is_out_fn():
        name += '_out'
    if func.name.overload_name:
        name += f'_{func.name.overload_name}'
    return name

def argumenttype_type(t: Type, *, mutable: bool, binds: ArgName) -> CType:
    if str(t) == 'Tensor?':
        if mutable:
            return MutRefCType(BaseCType('Tensor', binds))
        else:
            return ConstRefCType(BaseCType('Tensor', binds))
    elif str(t) == 'Tensor?[]':
        return BaseCType('TensorList', binds)
    return cpp.argumenttype_type(t, mutable=mutable, binds=binds)

def returns_type(rs: Sequence[Return]) -> str:
    return cpp.returns_type(rs)

def argument_type(a: Argument, *, binds: ArgName) -> CType:
    return argumenttype_type(a.type, mutable=a.is_write, binds=binds)

def argument(a: Union[Argument, SelfArgument, TensorOptionsArguments]) -> List[Binding]:
    if isinstance(a, Argument):
        return [Binding(
            ctype=argument_type(a, binds=a.name),
            name=a.name,
            default=cpp.default_expr(a.default, a.type) if a.default is not None else None,
            argument=a,
        )]
    elif isinstance(a, SelfArgument):
        # Erase SelfArgument from the distinction
        return argument(a.argument)
    elif isinstance(a, TensorOptionsArguments):
        if local.use_c10_dispatcher() in [UseC10Dispatcher.hacky_wrapper_for_legacy_signatures,
                                          UseC10Dispatcher.with_codegenerated_unboxing_wrapper]:
            # TODO: expunge this logic entirely
            default = None
            if all(x.default == "None" for x in a.all()):
                default = '{}'
            elif a.dtype.default == "long":
                default = 'at::kLong'  # TODO: this is wrong
            return [Binding(
                ctype=ConstRefCType(BaseCType('TensorOptions', 'options')),
                name='options',
                default=default,
                argument=a,
            )]
        else:
            assert local.use_c10_dispatcher() == UseC10Dispatcher.full
            return [
                Binding(
                    ctype=OptionalCType(BaseCType('ScalarType', 'dtype')),
                    name='dtype',
                    default='{}',
                    argument=a,
                ),
                Binding(
                    ctype=OptionalCType(BaseCType('Layout', 'layout')),
                    name='layout',
                    default='{}',
                    argument=a,
                ),
                Binding(
                    ctype=OptionalCType(BaseCType('Device', 'device')),
                    name='device',
                    default='{}',
                    argument=a,
                ),
                Binding(
                    ctype=OptionalCType(BaseCType('bool', 'pin_memory')),
                    name='pin_memory',
                    default='{}',
                    argument=a,
                )]
    else:
        assert_never(a)

def arguments(func: FunctionSchema) -> List[Binding]:
    args: List[Union[Argument, TensorOptionsArguments, SelfArgument]] = []
    if local.use_c10_dispatcher() is UseC10Dispatcher.full:
        args.extend(func.arguments.non_out)
        args.extend(func.arguments.out)
    else:
        args.extend(func.arguments.out)
        args.extend(func.arguments.non_out)
    return [r for arg in args for r in argument(arg)]
