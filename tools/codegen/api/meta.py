from tools.codegen.model import NativeFunctionsGroup

# Follows dispatcher calling convention, but:
#   - Mutable arguments not allowed.  Meta functions are always
#     written in functional form.  Look at FunctionSchema.signature()
#   - No tensor returns; instead we return a TensorMeta describing
#     the tensor in question

def name(g: NativeFunctionsGroup) -> str:
    # use the overload name from the functional version
    return str(g.functional.func.name).replace('.', '_')
