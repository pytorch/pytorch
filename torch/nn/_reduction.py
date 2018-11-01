import warnings
from .._jit_internal import weak_script

# NB: Keep this class in sync with enums in aten/src/ATen/core/Reduction.h


@weak_script
def get_enum(reduction):
    if reduction == 'none':
        return 0
    elif reduction == 'mean':
        return 1
    elif reduction == 'elementwise_mean':
        warnings.warn("reduction='elementwise_mean' is deprecated, please use reduction='mean' instead.")
        return 1
    elif reduction == 'sum':
        return 2
    raise ValueError(reduction + " is not a valid value for reduction")

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
        ret = 'mean'
    elif reduce:
        ret = 'sum'
    else:
        ret = 'none'
    if emit_warning:
        warnings.warn(warning.format(ret))
    return ret


def legacy_get_enum(size_average, reduce, emit_warning=True):
    return _Reduction.get_enum(_Reduction.legacy_get_string(size_average, reduce, emit_warning))
