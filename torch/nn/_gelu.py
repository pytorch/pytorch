# Keep this file in sync with enums in aten/src/ATen/core/Gelu.h

def get_enum(gelu_approximation: str) -> int:
    if gelu_approximation == 'none':
        ret = 0
    elif gelu_approximation == 'tanh':
        ret = 1
    else:
        ret = -1  # TODO: remove once JIT exceptions support control flow
        raise ValueError("{} is not a valid value for gelu approximation".format(gelu_approximation))
    return ret
