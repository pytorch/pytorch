from ..._jit_internal import weak_script

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
        raise ValueError(qscheme + " is not a valid str value for qscheme")
    return ret

def get_string(enum):
    # type: (int) -> str
    if enum == 0:
        ret = 'none'
    elif enum == 1:
        ret = 'per_tensor_affine'
    elif enum == 2:
        ret = 'per_channel_affine'
    elif enum == 3:
        ret = 'per_tensor_symmetric'
    elif enum == 4:
        ret = 'per_channel_symmetric'
    else:
        raise ValueError(enum + " is not a valid enum value for qscheme")
    return ret
