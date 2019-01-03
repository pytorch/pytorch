import warnings
from .._jit_internal import weak_script
import torch

# NB: Keep this file in sync with enums in aten/src/ATen/core/Reduction.h


@weak_script
def get_enum(reduction):
    # type: (str) -> int
    if reduction == 'none':
        ret = 0
    elif reduction == 'mean':
        ret = 1
    elif reduction == 'elementwise_mean':
        warnings.warn("reduction='elementwise_mean' is deprecated, please use reduction='mean' instead.")
        ret = 1
    elif reduction == 'sum':
        ret = 2
    else:
        ret = -1  # TODO: remove once JIT exceptions support control flow
        raise ValueError(reduction + " is not a valid value for reduction")
    return ret

# In order to support previous versions, accept boolean size_average and reduce
# and convert them into the new constants for now


# We use these functions in torch/legacy as well, in which case we'll silence the warning
@weak_script
def legacy_get_string(size_average, reduce, emit_warning=True):
    # type: (Optional[bool], Optional[bool], bool) -> str
    warning = "size_average and reduce args will be deprecated, please use reduction='{}' instead."

    if size_average is None:
        size_average = True
    if reduce is None:
        reduce = True

    size_average = torch.jit._unwrap_optional(size_average)
    reduce = torch.jit._unwrap_optional(reduce)
    if size_average and reduce:
        ret = 'mean'
    elif reduce:
        ret = 'sum'
    else:
        ret = 'none'
    if emit_warning:
        warnings.warn(warning.format(ret))
    return ret


@weak_script
def legacy_get_enum(size_average, reduce, emit_warning=True):
    # type: (Optional[bool], Optional[bool], bool) -> int
    return get_enum(legacy_get_string(size_average, reduce, emit_warning))
