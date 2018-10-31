# NB: Keep this file in sync with enums in THNN/Reduction.h
import warnings

from .._jit_internal import weak_script

@weak_script
def get_enum(reduction):
    # type: (str) -> int
    if reduction == 'none':
        ret = 0
    elif reduction == 'elementwise_mean':
        ret = 1
    elif reduction == 'sum':
        ret = 2
    else:
        # TODO: remove this when jit support control flow
        ret = -1
        raise ValueError(reduction + " is not a valid value for reduction")
    return ret

# In order to support previous versions, accept boolean size_average and reduce
# and convert them into the new constants for now


# We use these functions in torch/legacy as well, in which case we'll silence the warning
def legacy_get_string(size_average, reduce, emit_warning=True):
    warning = "size_average and reduce args will be deprecated, please use reduction='{}' instead."

    if size_average is None:
        size_average = True
    if reduce is None:
        reduce = True

    if size_average and reduce:
        ret = 'elementwise_mean'
    elif reduce:
        ret = 'sum'
    else:
        ret = 'none'
    if emit_warning:
        warnings.warn(warning.format(ret))
    return ret


def legacy_get_enum(size_average, reduce, emit_warning=True):
    return get_enum(legacy_get_string(size_average, reduce, emit_warning))
