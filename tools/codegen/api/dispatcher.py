from tools.codegen.model import *

from tools.codegen.api.types import *
import tools.codegen.api.cpp as cpp

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
#   - 'use_c10_dispatcher: full' controls whether or not we actually
#     use the modern calling convention or not.  When use_c10_dispatcher
#     is not enabled, we don't use the template machinery.
#
#   - dtype, layout, device and pin_memory are represented as separate
#     arguments.
#

def name(func: FunctionSchema) -> str:
    return cpp.name(func)

def argumenttype_type(t: Type, *, mutable: bool, binds: ArgName) -> CType:
    # This is a faux amis.  If it makes sense in the future to add
    # more special cases here, or invert things so cpp.argument_type
    # calls this, or just completely inline the function, please do
    # it.
    return cpp.argumenttype_type(t, mutable=mutable, binds=binds)

def argument_type(a: Argument, *, binds: ArgName) -> CType:
    return argumenttype_type(a.type, mutable=a.is_write, binds=binds)

def returns_type(rs: Sequence[Return]) -> str:
    # At present, there is no difference. But there could be!
    return cpp.returns_type(rs)

def argument(
    a: Union[Argument, TensorOptionsArguments, SelfArgument]
) -> List[Binding]:
    if isinstance(a, Argument):
        return [Binding(
            ctype=argument_type(a, binds=a.name),
            name=a.name,
            argument=a,
        )]
    elif isinstance(a, SelfArgument):
        return argument(a.argument)
    elif isinstance(a, TensorOptionsArguments):
        return argument(a.dtype) + argument(a.layout) + argument(a.device) + argument(a.pin_memory)
    else:
        assert_never(a)

def arguments(func: FunctionSchema) -> List[Binding]:
    return [
        r for a in itertools.chain(
            func.arguments.positional,
            func.arguments.kwarg_only,
            func.arguments.out
        ) for r in argument(a)
    ]
