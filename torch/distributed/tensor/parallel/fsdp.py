import warnings
from typing import Any, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed._shard.sharded_tensor import Shard
from torch.distributed.fsdp._fsdp_extensions import _set_fsdp_extensions, FSDPExtensions
from torch.distributed.tensor.parallel._data_parallel_utils import (
    _chunk_tensor,
    _flatten_tensor,
    _pre_load_state_dict,
    _unflatten_tensor,
)

__all__ = ["enable_2d_with_fsdp", "DTensorExtensions"]


class DTensorExtensions(FSDPExtensions):
    """
    DTensorExtension is the TensorFlattener extension needed for 2D FSDP + TP.
    This is the implementation for FSDPExtensions defined in
    https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/_fsdp_extensions.py
    """

    def pre_flatten_transform(
        self,
        tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        return _flatten_tensor(tensor)

    def post_unflatten_transform(
        self, tensor: torch.Tensor, param_extension: Any
    ) -> torch.Tensor:
        return _unflatten_tensor(tensor, param_extension)

    def chunk_tensor(
        self,
        tensor: torch.Tensor,
        rank: int,
        world_size: int,
        num_devices_per_node: int,
        pg: dist.ProcessGroup,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        return _chunk_tensor(tensor, rank, world_size, num_devices_per_node, pg)

    def pre_load_state_dict_transform(
        self,
        tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[Shard]]:
        return _pre_load_state_dict(tensor)


# TODO: remove enable_2d_with_fsdp() once we roll out the new 2D flow.
def enable_2d_with_fsdp() -> bool:
    """
    The API registers the extension which is needed for Tensor Parallelism (TP)
    to work with FullyShardedDataParallel (FSDP). We first parallelize parameters
    within one module or sub_modules based on a parallelize_plan and will let FSDP
    reshard the local tensor of distributed parameter which is essentially a DTensor.

    Return:
        A `bool` indicated whether extension registration succeeds or not.
    """

    torch._C._log_api_usage_once(
        "torch.distributed.tensor.parallel.enable_2d_with_fsdp"
    )

    try:
        _set_fsdp_extensions(DTensorExtensions())
        return True

    except BaseException as e:
        warnings.warn(
            "PyTorch doesn't have TensorFlattener extension point available"
            "2D parallelism won't work with FSDP"
            f"exception: {e}"
        )
        return False
