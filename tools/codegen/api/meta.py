from tools.codegen.model import *
from tools.codegen.api.types import MetaArgument

import tools.codegen.api.cpp as cpp
import tools.codegen.api.dispatcher as dispatcher

from typing import Sequence
import itertools

# Follows dispatcher calling convention, but:
#   - Mutable arguments not allowed.  Meta functions are always
#     written in functional form.  Look at FunctionSchema.signature()
#   - No tensor returns; instead we return a TensorMeta describing
#     the tensor in question

def name(g: StructuredNativeFunctions) -> str:
    # use the overload name from the functional version
    return str(g.functional.func.name).replace('.', '_')

def argument_type(a: Argument) -> str:
    assert not a.is_write
    return dispatcher.argumenttype_type(a.type, mutable=False)

def returntype_type(t: Type) -> str:
    r = cpp.valuetype_type(t)
    if r is not None:
        return r

    if isinstance(t, BaseType):
        if t.name == BaseTy.Tensor:
            return 'TensorMeta'
    elif isinstance(t, ListType):
        raise NotImplementedError("list returns not supported yet")

    raise AssertionError(f"unrecognized return type {t}")

def return_type(r: Return) -> str:
    assert not r.is_write
    return returntype_type(r.type)

def returns_type(rs: Sequence[Return]) -> str:
    if len(rs) == 0:
        return 'void'
    elif len(rs) == 1:
        return return_type(rs[0])
    else:
        args = ','.join(map(return_type, rs))
        return f'std::tuple<{args}>'

def argument(a: Argument) -> MetaArgument:
    return MetaArgument(
        type=argument_type(a),
        name=a.name,
        argument=a,
    )

def arguments(func: FunctionSchema) -> Sequence[MetaArgument]:
    assert not func.arguments.out
    return list(map(argument, itertools.chain(func.arguments.positional, func.arguments.kwarg_only)))
