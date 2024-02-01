from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from torch.distributed.fsdp._common_utils import _named_parameters_with_duplicates
from ._fsdp_collectives import (
    AllGatherResult,
    AllGatherState,
    AllGatherStateHolder,
    foreach_all_gather,
    foreach_all_gather_copy_out,
)
from ._fsdp_common import FSDPMeshInfo, HSDPMeshInfo, TrainingState
from ._fsdp_param import FSDPParam, ParamModuleInfo, ShardedState


class FSDPParamGroup:
    """This class represents a parameter group to communicate together."""

    def __init__(
        self,
        params: List[nn.Parameter],
        module: nn.Module,
        mesh_info: FSDPMeshInfo,
        device: torch.device,
    ):
        self.module = module  # permit ref cycle because 1:1 lifetime
        param_module_infos = _get_param_module_infos(params, module)
        self.fsdp_params = [
            FSDPParam(param, module_info, mesh_info, device)
            for param, module_info in zip(params, param_module_infos)
        ]
        self.mesh_info = mesh_info
        self.device = device
        self._training_state = TrainingState.IDLE
        # Group's sharded state always matches its parameters' sharded states
        self._sharded_state = ShardedState.SHARDED
        self._init_mp_dtypes()
        self._module_fqn: Optional[str] = None  # prefixed from root module

        # - Communication and communication/computation overlap
        default_stream = torch.cuda.current_stream()
        self.default_stream: torch.cuda.Stream = default_stream
        self.all_gather_copy_in_stream: torch.cuda.Stream = default_stream
        self.all_gather_stream: torch.cuda.Stream = default_stream
        self.all_gather_state = AllGatherStateHolder()
        self._init_grad_divide_factors()

        # - CUDA events for stream synchronization
        # Holds the all-gather output buffer, sync objects, and metadata
        self._all_gather_result: Optional[AllGatherResult] = None

    # Initialization #
    def _init_mp_dtypes(self) -> None:
        orig_dtypes = {fsdp_param.orig_dtype for fsdp_param in self.fsdp_params}
        if len(orig_dtypes) != 1:
            # This can be relaxed if we copy-out for the reduce-scatter
            raise AssertionError(
                f"FSDP expects uniform original parameter dtype but got {orig_dtypes}"
            )
        self._orig_dtype = next(iter(orig_dtypes))
        self._param_dtype = self._orig_dtype

    def _init_grad_divide_factors(self):
        """
        For N data parallel workers, each worker computes g_i, and they
        collectively reduce to compute (g_1 + ... + g_N) / N. To avoid overflow
        and underflow, we divide by ~sqrt(N) before and after the reduction.
        """
        data_parallel_world_size = 1
        data_parallel_world_size *= self.mesh_info.shard_mesh_size
        if isinstance(self.mesh_info, HSDPMeshInfo):
            data_parallel_world_size *= self.mesh_info.replicate_mesh_size
        factor: int = 1
        while (
            data_parallel_world_size % factor == 0
            and data_parallel_world_size / factor > factor
        ):
            factor *= 2
        self._grad_predivide_factor: float = float(factor)
        self._grad_postdivide_factor: float = (
            data_parallel_world_size / self._grad_predivide_factor
        )

    # Runtime #
    def unshard(self, async_op: bool = False):
        if self._all_gather_result is not None:  # already called, pending wait
            return
        if self._sharded_state == ShardedState.UNSHARDED:
            return  # no-op
        self._all_gather_result = foreach_all_gather(
            self.fsdp_params,
            self._all_gather_process_group,
            async_op,
            self._all_gather_copy_in_stream_for_unshard,
            self._all_gather_stream_for_unshard,
            self.device,
            self._param_dtype,
        )

    def wait_for_unshard(self):
        """
        1. In forward with implict prefetching, to overlap the current copy-out
        with the next all-gather, we save a reference to the current all-gather
        result to free after the next copy-out.
        2. Otherwise (explicit prefetching or in backward), we free the
        all-gather result immediately after the current copy-out since we can
        already overlap the current copy-out with the previous reduce-scatter.
        """
        if not self._all_gather_result:
            return  # no preceding unshard
        if self._training_state == TrainingState.FORWARD:  # implicit prefetch
            if prev_all_gather_state := self.all_gather_state.pop():
                self._wait_all_gather_streams_on_event(prev_all_gather_state.event)
                del prev_all_gather_state  # free
        foreach_all_gather_copy_out(
            self._all_gather_result, self.fsdp_params, self._all_gather_process_group
        )
        for fsdp_param in self.fsdp_params:
            fsdp_param.init_unsharded_param()  # no-op after 1st call
        self._to_unsharded()
        all_gather_copy_out_event = torch.cuda.Event()
        all_gather_copy_out_event.record()
        if self._training_state == TrainingState.FORWARD:
            self.all_gather_state.put(
                AllGatherState(self._all_gather_result, all_gather_copy_out_event)
            )
        else:
            self._wait_all_gather_streams_on_event(all_gather_copy_out_event)
        self._all_gather_result = None  # free unless saved in `all_gather_state`

    def _wait_all_gather_streams_on_event(self, event: torch.cuda.Event):
        self.all_gather_copy_in_stream.wait_event(event)
        self.all_gather_stream.wait_event(event)

    def reshard(self):
        self._to_sharded()

    def pre_forward(
        self, module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        with torch.profiler.record_function("FSDP::pre_forward"):
            self._training_state = TrainingState.FORWARD
            self.unshard()
            self.wait_for_unshard()
            return args, kwargs

    def post_forward(self, module: nn.Module, input: Any, output: Any):
        with torch.profiler.record_function("FSDP::post_forward"):
            self.reshard()
            self._training_state = TrainingState.IDLE
            return output

    # Utilities #
    def _to_sharded(self):
        if self._sharded_state != ShardedState.SHARDED:
            for fsdp_param in self.fsdp_params:
                fsdp_param.to_sharded()
            self._sharded_state = ShardedState.SHARDED

    def _to_unsharded(self):
        if self._sharded_state != ShardedState.UNSHARDED:
            for fsdp_param in self.fsdp_params:
                fsdp_param.to_unsharded()
            self._sharded_state = ShardedState.UNSHARDED

    # Properties #
    @property
    def _all_gather_process_group(self) -> dist.ProcessGroup:
        mesh_info = self.mesh_info
        assert isinstance(mesh_info, FSDPMeshInfo)
        return mesh_info.shard_process_group

    @property
    def _use_all_gather_stream(self) -> bool:
        return self._training_state in (
            TrainingState.FORWARD,
            TrainingState.PRE_BACKWARD,
        )

    @property
    def _all_gather_copy_in_stream_for_unshard(self) -> torch.cuda.Stream:
        if self._use_all_gather_stream:
            return self.all_gather_copy_in_stream
        return self.default_stream

    @property
    def _all_gather_stream_for_unshard(self) -> torch.cuda.Stream:
        if self._use_all_gather_stream:
            return self.all_gather_stream
        return self.default_stream


def _get_param_module_infos(
    params: List[nn.Parameter], module: nn.Module
) -> List[ParamModuleInfo]:
    """
    Shared parameter:
        lin1.weight = lin2.weight
    Shared module:
        mlp.lin1 = mlp.lin2
    We do not remove duplicates when traversing both modules and parameters to
    find shared modules' parameters and shared parameters within a module.
    """
    params_set = set(params)
    param_to_module_info: Dict[nn.Parameter, ParamModuleInfo] = {}
    for _, submodule in module.named_modules(remove_duplicate=False):
        for param_name, param in _named_parameters_with_duplicates(
            submodule, recurse=False
        ):
            if param in params_set:
                if param not in param_to_module_info:
                    param_to_module_info[param] = ParamModuleInfo(submodule, param_name)
                else:
                    param_to_module_info[param].shared_modules.append(submodule)
                    param_to_module_info[param].shared_param_names.append(param_name)
    if len(param_to_module_info) != len(params):
        raise AssertionError(f"Some parameters are not in the module tree of {module}")
    return [param_to_module_info[param] for param in params]
