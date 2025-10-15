# mypy: allow-untyped-defs
import logging
from collections.abc import Callable
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.distributed.device_mesh import _get_device_handle
from torch.distributed.fsdp._fully_shard._fsdp_api import (
    MixedPrecisionPolicy,
    OffloadPolicy,
)
from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
    AllGather,
    AllGatherResult,
    DefaultAllGather,
    DefaultReduceScatter,
    foreach_all_gather,
    foreach_all_gather_copy_out,
    ReduceScatter,
)
from torch.distributed.fsdp._fully_shard._fsdp_common import (
    compiled_autograd_enabled,
    FSDPMeshInfo,
    TrainingState,
)
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam, ShardedState
from torch.distributed.fsdp._fully_shard._fsdp_param_group import (
    _get_param_module_infos,
    AllGatherState,
    AllReduceState,
    FSDPCommContext,
    FSDPParamGroup,
    ReduceScatterState,
)
from torch.distributed.tensor import Shard
from torch.profiler import record_function
from torch.utils.hooks import RemovableHandle

from .replicate_collective import foreach_reduce
from .replicate_param import alloc_storage, ReplicateParam


logger = logging.getLogger("torch.distributed.fsdp.fully_shard")

_ModuleToHandleDict = dict[nn.Module, RemovableHandle]  # for state dict


"""
[Note: Overlapping all-gather copy-in and all-gather]
For implicit forward prefetching, we want to overlap the next copy-in with the
current all-gather. We do so using a separate copy-in stream. However, since
we have the all-gather input as a view into the output, we must make sure to
copy into different memory from the current all-gather's output. Thus, we keep
a reference to the current all-gather's output and have the next FSDP parameter
group free it after its copy-in. Finally, we have the last FSDP state flush the
reference to avoid holding onto memory after forward.
"""


class ReplicateParamGroup(FSDPParamGroup):
    """This class represents a parameter group to communicate together."""

    _orig_dtype: Optional[torch.dtype]
    _reduce_dtype: Optional[torch.dtype]

    def __init__(
        self,
        params: list[nn.Parameter],
        modules: tuple[nn.Module, ...],
        mesh_info: FSDPMeshInfo,
        post_forward_mesh_info: Optional[FSDPMeshInfo],
        device: torch.device,
        shard_placement_fn: Optional[Callable[[nn.Parameter], Optional[Shard]]],
        mp_policy: MixedPrecisionPolicy,
        offload_policy: OffloadPolicy,
    ):
        self.modules = modules  # permit ref cycle because 1:1 lifetime
        param_module_infos = _get_param_module_infos(params, modules)

        self.fsdp_params = [
            ReplicateParam(
                param,
                module_info,  # type: ignore[arg-type]
                mesh_info,
                post_forward_mesh_info,
                device,
                shard_placement_fn,
                mp_policy,
                offload_policy,
            )
            for param, module_info in zip(params, param_module_infos)
        ]
        self.mesh_info = mesh_info
        self.post_forward_mesh_info = post_forward_mesh_info
        # pyrefly: ignore  # read-only
        self.device = device
        self.device_handle = _get_device_handle(device.type)
        self.mp_policy = mp_policy
        self.offload_policy = offload_policy
        self._training_state = TrainingState.IDLE
        # Group's sharded state always matches its parameters' sharded states
        self._sharded_state = ShardedState.SHARDED
        self._module_fqn: Optional[str] = None  # prefixed from root module
        # Only consider resetting sharded parameters once in lazy init since it
        # can incur nontrivial overhead to reset them
        self._reset_sharded_params: bool = False

        # - Hook state
        self._module_to_pre_save_state_dict_hook_handle: _ModuleToHandleDict = {}
        self._module_to_pre_load_state_dict_hook_handle: _ModuleToHandleDict = {}
        self._all_reduce_hook: Optional[Callable[[torch.Tensor], None]] = None
        self._all_gather_comm: AllGather = DefaultAllGather()
        self._all_gather_output = torch.empty(0, device=self.device)
        self._reduce_scatter_comm: ReduceScatter = DefaultReduceScatter()
        # Optional stream to run the user-defined all-reduce hook in
        # Saved here and not in the comm. context because we allow the user to
        # specify it, possibly at construction time before lazy init
        self._all_reduce_hook_stream: Optional[torch.cuda.Stream] = None

        # - Communication and communication/computation overlap
        self.comm_ctx = FSDPCommContext()
        # Group's indices in the shared post-forward order
        self._post_forward_indices: list[int] = []
        # Whether to reduce gradients at all (whether for FSDP or HSDP)
        self.reduce_grads: bool = True
        # Whether to all-reduce gradients for HSDP; only used if
        # `self.reduce_grads` is true, in which case setting this to false
        # means reduce-scatter but no all-reduce
        self.all_reduce_grads: bool = True
        # Whether to reshard parameters after backward (only useful for
        # gradient accumulation)
        self.reshard_after_backward: bool = True
        # Optional custom factor for the gradient reduction op (e.g. to divide
        # by a factor other than the world size)
        self.gradient_divide_factor: Optional[float] = None
        # Whether reduce-scatter and all-reduce should be issued using only
        # summations, potentially with separate pre-/post-scaling.
        self.force_sum_reduction_for_comms: bool = False
        # `async_op` arg used for pre-forward/pre-backward unshard; can be
        # overridden to only do explicit prefetching and avoid inter-stream
        # fragmentation from using separate unshard streams
        self.unshard_async_op: bool = False
        # Whether to unshard in backward: can be overridden by the user if the
        # parameters in this group are not needed for backward (e.g. embedding)
        self.unshard_in_backward: bool = True

        # - CUDA events for stream synchronization
        # Holds the all-gather output buffer, sync objects, and metadata
        self._all_gather_result: Optional[AllGatherResult] = None
        # Holds the reduce-scatter/all-reduce view-out CUDA event that marks the end of
        # the group's post-backward (e.g. reduce-scatter, all-reduce and div), which
        # should be waited on at the end of backward
        self._post_reduce_event: Optional[torch.Event] = None
        # Holds the reshard-after-forward CUDA event when resharding to a
        # different world size, which should be waited on in the next unshard
        self._reshard_after_forward_event: Optional[torch.Event] = None

        # Only for HSDP, if accumulating gradients without all-reduce, save the
        # partial reduce output (only reduce-scattered but not all-reduced)
        self._partial_reduce_output: Optional[torch.Tensor] = None
        # Holds the all-reduce input and all-reduce event to keep it alive
        # until the end of backward (critical when doing bf16 reduction with
        # fp32 parameters since the all-reduce input is allocated in the RS
        # stream and will have no refs to it after being upcast to fp32)
        self._all_reduce_state: Optional[AllReduceState] = None

    # Runtime #
    def unshard(self, async_op: bool = False):
        if self._all_gather_result is not None:  # already called, pending wait
            return
        if self.is_unsharded:
            return  # no-op
        if (
            not self.unshard_in_backward
            and self._training_state == TrainingState.PRE_BACKWARD
        ):
            return
        if self._reshard_after_forward_event is not None:
            # Resharded parameter data is allocated in the default stream and
            # used in the all-gather streams
            self._wait_all_gather_streams_on_event(self._reshard_after_forward_event)
            self._reshard_after_forward_event = None

        world_size = 1
        if world_size == 1:
            # can't skip due to early return in wait_for_unshard if
            # no self._all_gather_result
            self._all_gather_result = AllGatherResult(
                all_gather_output=self._all_gather_output,
                all_gather_event=self.device_handle.Event().record(),
                all_gather_work=None,
                param_all_gather_input_dtypes=[],
                param_all_gather_input_numels=[],
                all_gather_input_split_sizes=[],
            )

            return

        with record_function(self._with_fqn("FSDP::all_gather")):
            self._all_gather_result = foreach_all_gather(
                self.fsdp_params,  # type: ignore[arg-type]
                self._all_gather_process_group,
                async_op,
                *self.comm_ctx.get_all_gather_streams(async_op, self._training_state),
                self.device,
                self._all_gather_comm,
            )

    def wait_for_unshard(self):
        """
        1. In forward with implicit prefetching, to overlap the current copy-out
        with the next all-gather, we save a reference to the current all-gather
        result to free after the next copy-out.
        2. Otherwise (explicit prefetching or in backward), we free the
        all-gather result immediately after the current copy-out since we can
        already overlap the current copy-out with the previous reduce-scatter.
        """
        if not self._all_gather_result:
            return  # no preceding unshard
        async_op = self._all_gather_result.all_gather_work is not None
        if self._training_state == TrainingState.FORWARD:  # implicit prefetch
            if prev_all_gather_state := self.comm_ctx.all_gather_state:
                self._wait_all_gather_streams_on_event(prev_all_gather_state.event)
                self.comm_ctx.all_gather_state = None  # free the all-gather result
        world_size = 1
        if world_size == 1:
            # directly initialize unsharded parameters from sharded parameters

            for fsdp_param in self.fsdp_params:
                # Use all_gather_inputs which already handles conversion to param_dtype
                # This is consistent with the world_size > 1 path
                all_gather_input = fsdp_param.all_gather_inputs[0]

                # Make sure the all_gather_outputs has proper storage size before using it
                # First ensure we have at least one tensor in all_gather_outputs
                fsdp_param.init_all_gather_outputs(
                    [all_gather_input.numel()],
                    [all_gather_input.dtype],
                    world_size,
                    self.device,
                    force_recreate=False,
                )

                tensor = fsdp_param.all_gather_outputs[0]
                alloc_storage(tensor)

                # find alternative way to check if tensor.is_inference
                with torch.autograd._unsafe_preserve_version_counter(tensor):
                    tensor.copy_(all_gather_input)

        else:
            with record_function(self._with_fqn("FSDP::all_gather_copy_out")):
                foreach_all_gather_copy_out(
                    self._all_gather_result,
                    self.fsdp_params,  # type: ignore[arg-type]
                    self._all_gather_process_group,
                )

        for fsdp_param in self.fsdp_params:
            fsdp_param.init_unsharded_param()

        self._to_unsharded()
        all_gather_copy_out_event = self.device_handle.Event()
        all_gather_copy_out_event.record()

        if (
            not async_op
            and self._training_state == TrainingState.FORWARD
            and world_size > 1
        ):
            # Defer free to allow for overlap of this copy-out with next
            # all-gather collective
            self.comm_ctx.all_gather_state = AllGatherState(
                self._all_gather_result, all_gather_copy_out_event
            )
        else:
            self._wait_all_gather_streams_on_event(all_gather_copy_out_event)

        self._all_gather_result = None  # free unless saved in `all_gather_state`

    def post_backward(self, *unused: Any):
        # This method should be idempotent and safe to call even when this
        # FSDP parameter group was not used in backward (should be a no-op)
        if not compiled_autograd_enabled():
            logger.debug("%s", self._with_fqn("FSDP::post_backward"))
        self._training_state = TrainingState.POST_BACKWARD
        with record_function(self._with_fqn("FSDP::post_backward_accumulate")):
            for fsdp_param in self.fsdp_params:
                fsdp_param.accumulate_unsharded_grad_if_needed()
        with record_function(self._with_fqn("FSDP::post_backward_reshard")):
            if not self.reduce_grads:
                if self.reshard_after_backward:
                    self.reshard()
                for fsdp_param in self.fsdp_params:
                    fsdp_param.to_accumulated_grad_if_needed()
                return
            # Save the autograd-computed gradients before resharding to only
            # access the unsharded parameters when their data is present
            fsdp_params_with_grad: list[FSDPParam] = []
            unsharded_grads: list[torch.Tensor] = []
            for fsdp_param in self.fsdp_params:
                if not hasattr(fsdp_param, "_unsharded_param"):
                    continue
                # May have an accumulated gradient of the reduce dtype if the
                # previous backward did not reduce-scatter
                if fsdp_param.unsharded_accumulated_grad is not None:
                    fsdp_params_with_grad.append(fsdp_param)
                    unsharded_grads.append(fsdp_param.unsharded_accumulated_grad_data)
                    fsdp_param.unsharded_accumulated_grad = None
                elif fsdp_param.unsharded_param.grad is not None:
                    fsdp_params_with_grad.append(fsdp_param)
                    unsharded_grads.append(fsdp_param.unsharded_grad_data)
                    fsdp_param.unsharded_param.grad = None
            if self.reshard_after_backward:
                self.reshard()
        if len(fsdp_params_with_grad) == 0:
            return
        with record_function(self._with_fqn("FSDP::post_backward_reduce")):
            if (
                self.comm_ctx.reduce_scatter_state is not None
                and self.comm_ctx.reduce_scatter_state.event is not None
            ):
                self.device_handle.current_stream().wait_event(
                    self.comm_ctx.reduce_scatter_state.event
                )
            self.comm_ctx.reduce_scatter_state = None
            all_reduce_pg = self._all_reduce_process_group if self._is_hsdp else None
            all_reduce_stream: torch.cuda.Stream
            if all_reduce_pg is None and self._all_reduce_hook_stream is not None:
                # this means the native HSDP is not enabled,
                # but user may want to have a custom HSDP setup
                assert self._all_reduce_hook is not None, (
                    "all reduce hook stream is specified but hook itself is missing."
                )
                all_reduce_stream = self._all_reduce_hook_stream
            else:
                all_reduce_stream = self.comm_ctx.all_reduce_stream

            self._wait_for_post_backward()
            (
                reduce_scatter_input,
                reduce_scatter_event,
                self._post_reduce_event,
                all_reduce_input,
                all_reduce_event,
                self._partial_reduce_output,
            ) = foreach_reduce(
                fsdp_params_with_grad,  # type: ignore[arg-type]
                unsharded_grads,
                self._reduce_scatter_process_group,
                self.comm_ctx.reduce_scatter_stream,
                self._reduce_scatter_comm,
                self._orig_dtype,
                self._reduce_dtype,
                self.device,
                self.gradient_divide_factor,
                self._all_reduce_process_group if self._is_hsdp else None,
                all_reduce_stream,
                self.all_reduce_grads,
                self._partial_reduce_output,
                self._all_reduce_hook,
                self.force_sum_reduction_for_comms,
            )
            self.comm_ctx.reduce_scatter_state = ReduceScatterState(
                reduce_scatter_input, reduce_scatter_event
            )
            if all_reduce_input is not None:
                if self.device.type != "cpu":
                    assert all_reduce_event is not None
                self._all_reduce_state = AllReduceState(
                    all_reduce_input, all_reduce_event
                )
