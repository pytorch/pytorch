from tools.codegen.model import *

from tools.codegen.api.types import TensorOptionsArguments, NativeArgument, ThisArgument
import tools.codegen.api.cpp as cpp
from tools.codegen import local

from typing import Union, Sequence, Tuple

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

def argument(a: Union[Argument, ThisArgument, TensorOptionsArguments], is_out_argument: bool) -> Sequence[NativeArgument]:
    if isinstance(a, Argument):
        return [NativeArgument(
            type=argument_type(a),
            name=a.name,
            default=cpp.default_expr(a.default, a.type) if a.default is not None else None,
            argument=a,
            is_out_argument=is_out_argument,
        )]
    elif isinstance(a, ThisArgument):
        assert not is_out_argument
        # Erase ThisArgument from the distinction
        return [NativeArgument(
            type=argument_type(a.argument),
            name=a.argument.name,
            default=None,
            argument=a.argument,
            is_out_argument=is_out_argument,
        )]
    elif isinstance(a, TensorOptionsArguments):
        assert not is_out_argument
        if local.use_c10_dispatcher() in [UseC10Dispatcher.hacky_wrapper_for_legacy_signatures,
                                          UseC10Dispatcher.with_codegenerated_unboxing_wrapper]:
            # TODO: expunge this logic entirely
            default = None
            if all(x.default == "None" for x in a.all()):
                default = '{}'
            elif a.dtype.default == "long":
                default = 'at::kLong'  # TODO: this is wrong
            return [NativeArgument(
                type='const TensorOptions &',
                name='options',
                default=default,
                argument=a,
                is_out_argument=is_out_argument,
            )]
        else:
            assert local.use_c10_dispatcher() == UseC10Dispatcher.full
            return [
                NativeArgument(
                    type='c10::optional<ScalarType>',
                    name='dtype',
                    default='{}',
                    argument=a,
                    is_out_argument=is_out_argument,
                ),
                NativeArgument(
                    type='c10::optional<Layout>',
                    name='layout',
                    default='{}',
                    argument=a,
                    is_out_argument=is_out_argument,
                ),
                NativeArgument(
                    type='c10::optional<Device>',
                    name='device',
                    default='{}',
                    argument=a,
                    is_out_argument=is_out_argument,
                ),
                NativeArgument(
                    type='c10::optional<bool>',
                    name='pin_memory',
                    default='{}',
                    argument=a,
                    is_out_argument=is_out_argument,
                )]
    else:
        assert_never(a)

def arguments(func: FunctionSchema) -> Tuple[NativeArgument, ...]:
    (args, out_args) = cpp.group_arguments(func, method=False)
    native_args = tuple(i for arg in args for i in argument(arg, is_out_argument=False))
    native_out_args = tuple(i for arg in out_args for i in argument(arg, is_out_argument=True))
    if local.use_c10_dispatcher().dispatcher_uses_new_style():
        return native_args + native_out_args
    else:
        return native_out_args + native_args
