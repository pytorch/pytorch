from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn


@dataclass
class InitPolicy:
    """
    Attributes:
        param_init_fn (Optional[Callable[[nn.Module], None]]): This function
            is called on managed meta-device modules to materialize and
            initialize them. This should only modify parameters/buffers for the
            passed-in module and not any of its children since FSDP will call
            it separately for the children.
        sync_module_states (bool): Whether to broadcast managed parameters and
            buffers from rank 0 to all ranks before sharding them.
    """

    param_init_fn: Optional[Callable[[nn.Module], None]] = None
    sync_module_states: bool = False


@dataclass
class MixedPrecisionPolicy:
    """
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
    buffer_dtype: Optional[torch.dtype] = None  # placeholder
    cast_forward_inputs: bool = True


@dataclass
class OffloadPolicy:
    """
    Attributes:
        offload_type (Optional[str]): This specifies the type of offloading for
            parameters and gradients. Currently, only CPU offloading is
            suppored by passing ``"cpu"``. (Default: ``None``)
    """

    # Only support "cpu" for now but can add NVMe in the future, in which case
    # we need to add a directory field
    offload_type: Optional[str] = None


@dataclass
class CommPolicy:
    """
    Attributes:
        forward_prefetch_limit (int): Number of all-gathers to prefetch in
            forward. Setting this to greater than one increases memory usage
            but *may* improve overlap in some workloads (e.g. CPU-bound).
            (Default: 1)
        backward_prefetch_limit (int): Number of all-gathers to prefetch in
            backward. Setting this to greater than one increases memory usage
            but *may* improve overlap in some workloads (e.g. CPU-bound).
            (Default: 1)

    .. note:: The prefetch limits should only be used in special cases, where
        the profiler trace has shown that additional prefetching can improve
        communication and computation overlap at the cost of increased memory.
    """

    forward_prefetch_limit: int = 1
    backward_prefetch_limit: int = 1
