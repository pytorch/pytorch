from tools.codegen.model import *

from tools.codegen.api.types import CppArgument, DispatcherExpr, TensorOptionsArguments, \
    DispatcherArgument, ThisArgument, LegacyDispatcherArgument
from tools.codegen.api import cpp
import tools.codegen.api.legacy_dispatcher as legacy_dispatcher
import tools.codegen.local as local
from enum import Enum
import itertools
from typing import Sequence, Optional

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
        # yourself why you are changing the legacy dispatcher protocol
        # here and not in legacy_dispatcher.
        return legacy_dispatcher.argumenttype_type(t, mutable=mutable)

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
        la = legacy_dispatcher.argument(a)
        assert len(la) == 1, "Operators using the legacy signature in the dispatcher don't scatter TensorOptions."
        return DispatcherArgument(
            type=la[0].type,
            name=la[0].name,
            argument=la[0].argument,
        )

def name(func: FunctionSchema) -> str:
    return cpp.name(func)

def arguments(func: FunctionSchema) -> Sequence[DispatcherArgument]:
    if local.use_c10_dispatcher().dispatcher_uses_new_style():
        return list(map(argument, itertools.chain(func.out_arguments, func.arguments, func.kwarg_only_arguments)))
    else:
        return [
            DispatcherArgument(type=la.type, name=la.name, argument=la.argument)
            for la in legacy_dispatcher.arguments(func)
        ]

# TODO GATHER is only needed for non-c10-full ops, remove later.
ProcessTensoroptions = Enum('ProcessTensoroptions', ('GATHER', 'SCATTER', 'PASS_THROUGH'))


# Given a set of CppArguments in scope, return a sequence of dispatcher
# expressions that translate the cpp API into dispatcher API
def cppargument_exprs(a: CppArgument,
                      *,
                      tensor_options: Optional[CppArgument],
                      process_tensoroptions: ProcessTensoroptions = ProcessTensoroptions.PASS_THROUGH
                      ) -> Sequence[DispatcherExpr]:
    if isinstance(a.argument, TensorOptionsArguments):
        if process_tensoroptions == ProcessTensoroptions.SCATTER:
            ta = a.argument
            return [
                DispatcherExpr(type=argument_type(ta.dtype), expr=f'optTypeMetaToScalarType({a.name}.dtype_opt())'),
                DispatcherExpr(type=argument_type(ta.layout), expr=f'{a.name}.layout_opt()'),
                DispatcherExpr(type=argument_type(ta.device), expr=f'{a.name}.device_opt()'),
                DispatcherExpr(type=argument_type(ta.pin_memory), expr=f'{a.name}.pinned_memory_opt()'),  # weird discrep
            ]
        elif process_tensoroptions == ProcessTensoroptions.GATHER:
            return [
                DispatcherExpr(
                    type='const TensorOptions &',
                    expr="TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory)")]
        else:
            assert process_tensoroptions == ProcessTensoroptions.PASS_THROUGH
            return [DispatcherExpr(type='const TensorOptions &', expr=a.name)]
    elif isinstance(a.argument, ThisArgument):
        return [DispatcherExpr(type=argument_type(a.argument.argument), expr=a.name)]
    elif isinstance(a.argument, Argument):
        if a.name == 'memory_format' and tensor_options is not None and \
                local.use_c10_dispatcher().dispatcher_uses_new_style():
            return [DispatcherExpr(
                type=argument_type(a.argument),
                expr=f'c10::impl::check_tensor_options_and_extract_memory_format({tensor_options.name}, {a.name})')
            ]
        else:
            return [DispatcherExpr(type=argument_type(a.argument), expr=a.name)]
    else:
        assert_never(a.argument)

def cpparguments_exprs(args: Sequence[CppArgument], process_tensoroptions: ProcessTensoroptions) -> Sequence[DispatcherExpr]:
    tensor_options = next((a for a in args if isinstance(a.argument, TensorOptionsArguments)), None)
    return [r for a in args for r in cppargument_exprs(a,
                                                       tensor_options=tensor_options,
                                                       process_tensoroptions=process_tensoroptions)]

# I don't think this is entirely sound, but it should be reasonably
# close
def legacydispatcherarguments_exprs(args: Sequence[LegacyDispatcherArgument]) -> Sequence[DispatcherExpr]:
    if local.use_c10_dispatcher().dispatcher_uses_new_style():
        process_tensoroptions = ProcessTensoroptions.SCATTER
    else:
        process_tensoroptions = ProcessTensoroptions.PASS_THROUGH
    return cpparguments_exprs([CppArgument(type=a.type,
                                           name=a.name,
                                           default=None,
                                           argument=a.argument) for a in args],
                              process_tensoroptions=process_tensoroptions)

def exprs(args: Sequence[DispatcherArgument]) -> Sequence[DispatcherExpr]:
    if local.use_c10_dispatcher().dispatcher_uses_new_style():
        process_tensoroptions = ProcessTensoroptions.SCATTER
    else:
        process_tensoroptions = ProcessTensoroptions.PASS_THROUGH
    return cpparguments_exprs([CppArgument(type=a.type,
                                           name=a.name,
                                           default=None,
                                           argument=a.argument) for a in args],
                              process_tensoroptions=process_tensoroptions)
