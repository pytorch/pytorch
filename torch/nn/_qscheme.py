from .._jit_internal import weak_script

# NB: Keep this file in sync with enums in c10/core/QScheme.h


@weak_script
def get_enum(qscheme):
    # type: (str) -> int
    if qscheme == 'none':
        ret = 0
    elif qscheme == 'per_tensor_affine':
        ret = 1
    elif qscheme == 'per_channel_affine':
        ret = 2
    elif qscheme == 'per_tensor_symmetric':
        ret = 3
    elif qscheme == 'per_channel_symmetric':
        ret = 4
    else:
        raise ValueError(qscheme + " is not a valid value for qscheme")
    return ret
