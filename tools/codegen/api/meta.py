from tools.codegen.model import *
from tools.codegen.api.types import MetaArgument

import tools.codegen.api.dispatcher as dispatcher

from typing import Sequence

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

def argument(a: Argument) -> MetaArgument:
    return MetaArgument(
        type=argument_type(a),
        name=a.name,
        argument=a,
    )

def arguments(func: FunctionSchema) -> Sequence[MetaArgument]:
    assert not func.arguments.out
    return list(map(argument, func.arguments.flat_non_out))
