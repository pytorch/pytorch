from tools.codegen.model import *

from tools.codegen.api.types import *
import tools.codegen.api.cpp as cpp
import tools.codegen.api.native as native
import tools.codegen.local as local

import itertools
from typing import Sequence, Optional, Tuple, List, Union

# This file describes the translation of JIT schema to the dispatcher
# API, the *unboxed* calling convention by which invocations through
# the dispatcher are made.  Historically, the dispatcher API matched
# the C++ API, but with the establishment of the boxed API, we've
# made changes to the dispatcher API to so that the unboxed API
# better aligns with the boxed API.  The dispatcher API hooks heavily
# into our template based boxing/unboxing machinery, so changes
# to this convention will usually need template updates too.
#
# Prominent characteristics of the dispatcher API:
#
#   - 'use_c10_dispatcher: full' controls whether or not we actually
#     use the modern calling convention or not.  When use_c10_dispatcher
#     is not enabled, we don't use the template machinery.
#
#   - dtype, layout, device and pin_memory are represented as separate
#     arguments.
#

def argumenttype_type(t: Type, *, mutable: bool) -> str:
    if local.use_c10_dispatcher().dispatcher_uses_new_style():
        # This is a faux amis.  If it makes sense in the future to add
        # more special cases here, or invert things so cpp.argument_type
        # calls this, or just completely inline the function, please do
        # it.
        return cpp.argumenttype_type(t, mutable=mutable)
    else:
        # This is real sharing.  If you're modifying this path, ask
        # yourself why you are changing the native functions protocol
        # here and not in native.
        return native.argumenttype_type(t, mutable=mutable)

def argument_type(a: Argument) -> str:
    return argumenttype_type(a.type, mutable=a.is_write)

def returns_type(rs: Sequence[Return]) -> str:
    # At present, there is no difference. But there could be!
    return cpp.returns_type(rs)

def argument(a: Argument) -> DispatcherArgument:
    if local.use_c10_dispatcher().dispatcher_uses_new_style():
        return DispatcherArgument(
            type=argument_type(a),
            name=a.name,
            argument=a,
        )
    else:
        la = native.argument(a)
        assert len(la) == 1, "Operators using the legacy signature in the dispatcher don't scatter TensorOptions."
        return DispatcherArgument(
            type=la[0].type,
            name=la[0].name,
            argument=la[0].argument,
        )

def name(func: FunctionSchema) -> str:
    return cpp.name(func)

def arguments(func: FunctionSchema) -> Tuple[DispatcherArgument, ...]:
    if local.use_c10_dispatcher().dispatcher_uses_new_style():
        return tuple(map(argument, itertools.chain(
            func.arguments.flat_positional,
            func.arguments.flat_kwarg_only,
            func.arguments.out
        )))
    else:
        return tuple(
            DispatcherArgument(type=la.type, name=la.name, argument=la.argument)
            for la in native.arguments(func)
        )

# Given a set of CppArguments in scope, return a sequence of dispatcher
# expressions that translate the cpp API into dispatcher API
#
# WARNING: This is unsound if you pass it CppArgument when you were
# supposed to pass it CppTensorOptionsArguments, it will directly
# translate device to device, which will give you the wrong signature
# for dispatcher.  If Argument "knew" that it was part of a
# TensorOptions that would help us dynamically test for this case
def cppargument_exprs(
    a: CppArgumentPack,
    *, tensor_options: Optional[CppArgument]
) -> Sequence[DispatcherExpr]:
    if isinstance(a, CppSingleArgumentPack):
        if isinstance(a.this.argument, TensorOptionsArguments):
            if local.use_c10_dispatcher().dispatcher_uses_new_style():
                # Scatter
                ta = a.this.argument
                name = a.this.name
                return [
                    DispatcherExpr(type=argument_type(ta.dtype), expr=f'optTypeMetaToScalarType({name}.dtype_opt())'),
                    DispatcherExpr(type=argument_type(ta.layout), expr=f'{name}.layout_opt()'),
                    DispatcherExpr(type=argument_type(ta.device), expr=f'{name}.device_opt()'),
                    DispatcherExpr(type=argument_type(ta.pin_memory), expr=f'{name}.pinned_memory_opt()'),  # weird discrep
                ]
            else:
                # No-op
                return [DispatcherExpr(type='const TensorOptions &', expr=a.this.name)]
        elif isinstance(a.this.argument, Argument):
            if a.this.name == 'memory_format' and \
                    tensor_options is not None and \
                    local.use_c10_dispatcher().dispatcher_uses_new_style():
                return [DispatcherExpr(
                    type=argument_type(a.this.argument),
                    expr=f'c10::impl::check_tensor_options_and_extract_memory_format({tensor_options.name}, {a.this.name})')
                ]
            else:
                return [DispatcherExpr(type=argument_type(a.this.argument), expr=a.this.name)]
        else:
            assert_never(a.this.argument)
    elif isinstance(a, CppTensorOptionsArgumentPack):
        if local.use_c10_dispatcher().dispatcher_uses_new_style():
            # No-op
            return [
                expr
                for sub_a in a.explicit_arguments()  # NB: don't really care about explicitness here
                for expr in cppargument_exprs(CppSingleArgumentPack(sub_a), tensor_options=tensor_options)
            ]
        else:
            # Gather
            return [DispatcherExpr(
                type='const TensorOptions &',
                expr=f'TensorOptions().dtype({a.dtype.name}).layout({a.layout.name})'
                     f'.device({a.device.name}).pinned_memory({a.pin_memory.name})',
            )]
    elif isinstance(a, CppThisArgumentPack):
        return [DispatcherExpr(
            type=a.type,
            expr='const_cast<Tensor&>(*this)',
        )]
    else:
        assert_never(a)

def cpparguments_exprs(func: FunctionSchema, * , method: bool, api_is_faithful: bool) -> Sequence[DispatcherExpr]:
    dispatcher_is_faithful = local.use_c10_dispatcher().dispatcher_uses_new_style()

    arguments: List[Union[Argument, TensorOptionsArguments, SelfArgument]] = []
    if dispatcher_is_faithful:
        arguments.extend(func.arguments.non_out)
        arguments.extend(func.arguments.out)
    else:
        arguments.extend(func.arguments.out)
        arguments.extend(func.arguments.non_out)

    if api_is_faithful:
        argument_packs = tuple(
            cpp.argument_faithful(a, method=method) for a in arguments
        )
    else:
        argument_packs = tuple(
            cpp.argument(a, method=method) for a in arguments
        )

    return _cpparguments_exprs(argument_packs)

def _cpparguments_exprs(args: Sequence[CppArgumentPack]) -> Sequence[DispatcherExpr]:
    tensor_options = next(
        (a.this for a in args if isinstance(a, CppSingleArgumentPack) and
            isinstance(a.this.argument, TensorOptionsArguments)),
        None
    )
    return [r for a in args for r in cppargument_exprs(a, tensor_options=tensor_options)]

# I don't think this is entirely sound, but it should be reasonably
# close
def nativearguments_exprs(args: Sequence[NativeArgument]) -> Sequence[DispatcherExpr]:
    return _cpparguments_exprs([
        CppSingleArgumentPack(CppArgument(type=a.type, name=a.name, default=None, argument=a.argument))
        for a in args
    ])

def exprs(args: Sequence[DispatcherArgument]) -> Sequence[DispatcherExpr]:
    return _cpparguments_exprs([
        CppSingleArgumentPack(CppArgument(type=a.type, name=a.name, default=None, argument=a.argument))
        for a in args
    ])
