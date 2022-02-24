from tools.codegen.model import (Argument, FunctionSchema, Return,
                                 SelfArgument, TensorOptionsArguments, Type)

from tools.codegen.api.types import ArgName, Binding, NamedCType, CType
from tools.codegen.api import cpp
from tools.codegen.utils import concatMap, assert_never

import itertools
from typing import Sequence, List, Union

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
#   - dtype, layout, device and pin_memory are represented as separate
#     arguments.
#

def name(func: FunctionSchema) -> str:
    return cpp.name(func)

def argumenttype_type(
        t: Type,
        *,
        mutable: bool,
        binds: ArgName,
        remove_non_owning_ref_types: bool = False,
        structured_type_override: bool
) -> NamedCType:
    # This is a faux amis.  If it makes sense in the future to add
    # more special cases here, or invert things so cpp.argument_type
    # calls this, or just completely inline the function, please do
    # it.
    return cpp.argumenttype_type(
        t,
        mutable=mutable,
        binds=binds,
        remove_non_owning_ref_types=remove_non_owning_ref_types,
        structured_type_override=structured_type_override)

def argument_type(
        a: Argument,
        *,
        binds: ArgName,
        remove_non_owning_ref_types: bool = False,
        structured_type_override: bool
) -> NamedCType:
    return argumenttype_type(
        a.type,
        mutable=a.is_write,
        binds=binds,
        remove_non_owning_ref_types=remove_non_owning_ref_types,
        structured_type_override=structured_type_override)

def returns_type(rs: Sequence[Return]) -> CType:
    # At present, there is no difference. But there could be!
    return cpp.returns_type(rs)

def jit_arguments(func: FunctionSchema) -> List[Argument]:
    def to_argument(a: Union[Argument, TensorOptionsArguments, SelfArgument]) -> List[Argument]:
        if isinstance(a, Argument):
            return [a]
        elif isinstance(a, SelfArgument):
            return [a.argument]
        elif isinstance(a, TensorOptionsArguments):
            return [a.dtype, a.layout, a.device, a.pin_memory]
        else:
            assert_never(a)
    return list(concatMap(to_argument, itertools.chain(
        func.arguments.positional,
        func.arguments.kwarg_only,
        func.arguments.out)))

def argument(a: Argument, *, remove_non_owning_ref_types: bool = False, structured_type_override: bool) -> Binding:
    return Binding(
        nctype=argument_type(
            a,
            binds=a.name,
            remove_non_owning_ref_types=remove_non_owning_ref_types,
            structured_type_override=structured_type_override),
        name=a.name,
        argument=a
    )

def arguments(func: FunctionSchema, *, structured_type_override: bool) -> List[Binding]:
    return [argument(a, structured_type_override=structured_type_override) for a in jit_arguments(func)]
