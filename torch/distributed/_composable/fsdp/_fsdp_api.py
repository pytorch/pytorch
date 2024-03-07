from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import torch


ShardedParamType = torch.Tensor
UnshardedParamType = torch.Tensor
AllGatherInputsType = Tuple[torch.Tensor, ...]
AllGatherOutputsType = Tuple[torch.Tensor, ...]
MetadataType = Any


@dataclass(frozen=True)
class MixedPrecisionPolicy:
    """
    This configures FSDP's mixed precision. Unlike autocast, this applies mixed
    precision at the module level, not op level, which means low-precision
    activations are saved for backward and high-to-low-precision casts are
    incurred only at module boundaries.

    FSDP works well with module-level mixed precision since it keeps the
    high-precision sharded parameters in memory anyway. In other words, FSDP
    does not require any extra memory to keep a high-precision copy of the
    parameters for the optimizer step.

    Attributes:
        param_dtype (Optional[torch.dtype]): This specifies the dtype for
            the unsharded parameter and hence the dtype for forward/backward
            computation and the parameter all-gather. If this is ``None``, then
            the unsharded parameter uses the original dtype. The optimizer step
            uses the sharded parameter in the original dtype. (Default:
            ``None``)
        reduce_dtype (Optional[torch.dtype]): This specifies the dtype for
            gradient reduction (i.e. reduce-scatter or all-reduce). If this is
            ``None`` but ``param_dtype`` is not ``None``, then the reduction
            uses the compute dtype. This can be used to run gradient reduction
            in full precision while using low precision for compute. (Default:
            ``None``)
        output_dtype (Optional[torch.dtype]): This specifies the dtype for
            casting floating-point forward outputs. This can be used to
            help implement cases where different modules have different mixed
            precision policies. (Default: ``None``)
        cast_forward_inputs (bool): This specifies whether FSDP should cast the
            forward's floating-point input tensors to ``param_dtype`` or not.
    """

    param_dtype: Optional[torch.dtype] = None
    reduce_dtype: Optional[torch.dtype] = None
    output_dtype: Optional[torch.dtype] = None
    cast_forward_inputs: bool = True

    def __post_init__(self):
        # Clamp `reduce_dtype` to `None` if no casting is required: since
        # gradients are computed in `param_dtype`, if `reduce_dtype` matches,
        # then we do not need extra casting
        if self.param_dtype == self.reduce_dtype:
            # Bypass the frozen dataclass checks
            object.__setattr__(self, "reduce_dtype", None)


@dataclass
class OffloadPolicy:
    """
    Attributes:
        offload_type (Optional[str]): This specifies the type of offloading.
            Currently, only CPU offloading is suppored by passing ``"cpu"``.
            Sharded parameters are offloaded to CPU and copied host-to-device
            as needed before all-gather. The all-gathered parameters are freed
            according to ``reshard_after_forward``. Sharded gradients are
            copied device-to-host, and the optimizer step runs on CPU with CPU
            optimizer states. (Default: ``None``)
    """

    # Only support "cpu" for now but can add NVMe in the future, in which case
    # we need to add a directory field
    offload_type: Optional[str] = None


@dataclass(frozen=True)
class FSDPTensorExtensions:
    # fsdp_pre_all_gather(sharded_param) -> (all_gather_inputs, metadata)
    fsdp_pre_all_gather: Callable[
        [ShardedParamType], Tuple[AllGatherInputsType, MetadataType]
    ]
    # fsdp_post_all_gather(all_gather_outputs, metadata, param_dtype, *, out) -> unsharded_param
    # `param_dtype` is the mixed precision policy's `param_dtype` if specified
    # or the parameter's original dtype otherwise. This can be useful to set as
    # a wrapper subclass's dtype for gradient dtypes to match.
    fsdp_post_all_gather: Callable[
        [AllGatherOutputsType, MetadataType, torch.dtype, Optional[UnshardedParamType]],
        UnshardedParamType,
    ]
