import collections
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp.flat_param import FlatParamHandle, HandleTrainingState
from torch.utils.hooks import RemovableHandle
from torch.distributed.fsdp.fully_sharded_data_parallel import TrainingState_


class ShardedDataParallelState:
    ...


def sharded_data_parallel(
    *modules: Tuple[nn.Module, ...],
    process_group: Optional[dist.ProcessGroup] = None,

):
    """"""
    # Initialize state
    state = ShardedDataParallelState()
    state.process_group = process_group
    state.rank = process_group.rank()
    state.world_size = process_group.size()

    assert torch.cuda.is_available()
    state.device = torch.cuda.current_device()
    state.streams: Dict[str, torch.cuda.Stream] = {}
    state.training_state = TrainingState_.IDLE

    state.handles: List[FlatParamHandle] = []
    state.pre_forward_handles: List[RemovableHandle] = []
    state.post_forward_handles: List[RemovableHandle] = []

    state.module_to_handles: Dict[nn.Module, List[FlatParamHandle]] = (
        collections.defaultdict(list)
    )

    # Construct `FlatParamHandle`s -- auto wrap with always wrap policy



def _register_pre_forward_hooks(

) -> None:
    """"""
    ...


def _register_post_forward_hooks(

) -> None:
    """"""
    ...

def _register_pre_backward_hooks(
    state: ShardedDataParallelState,
    output: Any,
    handles: List[FlatParamHandle],
) -> None:
    """"""
    ...

def _register_post_backward_hooks(
    state: ShardedDataParallelState,
    handles: List[FlatParamHandle],
) -> None:
    """"""
    ...

def _register_post_backward_callback(

) -> None:
    """"""
    ...


def _root_pre_forward(
    state: ShardedDataParallelState,
    *args,
    **kwargs,
):
    args, kwargs = _cast_forward_inputs(*args, **kwargs)
    return args, kwargs


def _pre_forward(
    state: ShardedDataParallelState,
    handles: List[FlatParamHandle],
    unshard_fn: Callable,
    module: nn.Module,
    input: Any,
):
    """
    """
    state.training_state = TrainingState_.FORWARD
    for handle in handles:
        handle._training_state = HandleTrainingState.FORWARD
    unshard_fn()
    _register_post_backward_hooks(state, handles)


def _post_forward(
    state: ShardedDataParallelState,
    handles: List[FlatParamHandle],
    reshard_fn: Callable,
    module: nn.Module,
    input: Any,
    output: Any,
) -> Any:
    reshard_fn()
    output = _register_pre_backward_hooks(state, output, handles)
    state.training_state = TrainingState_.IDLE
    for handle in handles:
        handle._training_state = HandleTrainingState.IDLE
    return output

