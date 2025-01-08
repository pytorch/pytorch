# mypy: allow-untyped-defs
import contextlib
import logging
from typing import Any, Callable, cast, Dict, List, NamedTuple, Optional, Set, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import _get_device_handle
from torch.distributed.fsdp._common_utils import _named_parameters_with_duplicates
from torch.distributed.tensor import Shard
from torch.profiler import record_function
from torch.utils._pytree import tree_flatten, tree_unflatten
from torch.utils.hooks import RemovableHandle

from ._fsdp_api import CPUOffloadPolicy, MixedPrecisionPolicy, OffloadPolicy
from ._fsdp_collectives import (
    AllGatherResult,
    foreach_all_gather,
    foreach_all_gather_copy_out,
    foreach_reduce,
)
from ._fsdp_common import (
    compiled_autograd_enabled,
    FSDPMeshInfo,
    HSDPMeshInfo,
    TrainingState,
)
from ._fsdp_param import FSDPParam, ParamModuleInfo, ShardedState


logger = logging.getLogger("torch.distributed.fsdp.fully_shard")

_ModuleToHandleDict = Dict[nn.Module, RemovableHandle]  # for state dict


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


class FSDPCommContext:
    """This has the communication state shared across FSDP states/parameter groups."""

    def lazy_init(self, device: torch.device):
        self.device_handle = _get_device_handle(device.type)
        # Setting the all-gather/reduce-scatter streams to be higher priority
        # can help avoid some issues where their copies in/out are delayed and
        # block computation (this is different from high-pri NCCL streams)
        high_priority = -1
        # All-gather state and copy-in stream allow overlapping the next
        # copy-in with the current all-gather in forward; copy-in overlaps with
        # reduce-scatter in backward without the separate copy-in stream
        self.all_gather_copy_in_stream = self.device_handle.Stream(
            priority=high_priority
        )
        # All-gather stream allows overlapping next all-gather with current
        # forward compute
        self.all_gather_stream = self.device_handle.Stream(priority=high_priority)
        # Reduce-scatter stream gives separate execution "thread" for post-
        # backward logic like pre/post-gradient division and reduce-scatter
        self.reduce_scatter_stream = self.device_handle.Stream(priority=high_priority)
        # Run the HSDP all-reduces concurrently with all-gather/reduce-scatter
        # since collectives use different network resources and can overlap
        # in the typical intra-node sharding / inter-node replication case
        self.all_reduce_stream = self.device_handle.Stream()
        # All-gather/reduce-scatter states keep references to collective
        # tensors produced in one stream and used in another and accompanying
        # CUDA events for synchronization
        self.all_gather_state: Optional[AllGatherState] = None
        self.reduce_scatter_state: Optional[ReduceScatterState] = None
        # Post-forward order for explicit backward prefetching
        self.post_forward_order: List[FSDPParamGroup] = []  # will cause ref cycles

    def get_all_gather_streams(
        self, async_op: bool, training_state: TrainingState
    ) -> Tuple[torch.Stream, torch.Stream]:
        if not async_op and training_state in (
            TrainingState.FORWARD,
            TrainingState.PRE_BACKWARD,
        ):
            # Use separate streams for implicit prefetching
            return self.all_gather_copy_in_stream, self.all_gather_stream
        current_stream = self.device_handle.current_stream()
        return current_stream, current_stream


# See [Note: Overlapping all-gather copy-in and all-gather]
class AllGatherState(NamedTuple):
    all_gather_result: AllGatherResult
    event: torch.Event  # all-gather copy-out


class ReduceScatterState(NamedTuple):
    reduce_scatter_input: torch.Tensor
    event: torch.Event  # reduce-scatter event


class AllReduceState(NamedTuple):
    all_reduce_input: torch.Tensor
    event: torch.Event  # all-reduce event


class FSDPParamGroup:
    """This class represents a parameter group to communicate together."""

    _orig_dtype: torch.dtype
    _reduce_dtype: Optional[torch.dtype]

    def __init__(
        self,
        params: List[nn.Parameter],
        modules: Tuple[nn.Module, ...],
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
            FSDPParam(
                param,
                module_info,
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

        # - Communication and communication/computation overlap
        self.comm_ctx = FSDPCommContext()
        # Group's indices in the shared post-forward order
        self._post_forward_indices: List[int] = []
        # Whether to reduce gradients at all (whether for FSDP or HSDP)
        self.reduce_grads: bool = True
        # Whether to all-reduce gradients for HSDP; only used if
        # `self.reduce_grads` is true, in which case setting this to false
        # means reduce-scatter but no all-reduce
        self.all_reduce_grads: bool = True
        # Whether to reshard parameters after backward (only useful for
        # gradient accumulation)
        self.reshard_after_backward: bool = True
        # Optional custom reduce-scatter reduce op (e.g. to divide by a
        # factor other than the shard world size)
        self.reduce_scatter_reduce_op: Optional[dist.ReduceOp] = None
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

    # Initialization #
    def _init_mp_dtypes(self) -> None:
        for fsdp_param in self.fsdp_params:
            fsdp_param.init_dtype_attrs(self.mp_policy)
        orig_dtypes = {fsdp_param.orig_dtype for fsdp_param in self.fsdp_params}
        if len(orig_dtypes) != 1:
            # This can be relaxed if we copy-out for the reduce-scatter
            raise AssertionError(
                f"FSDP expects uniform original parameter dtype but got {orig_dtypes}"
            )
        self._orig_dtype = next(iter(orig_dtypes))
        reduce_dtypes = {fsdp_param.reduce_dtype for fsdp_param in self.fsdp_params}
        if len(reduce_dtypes) != 1:
            # This can be relaxed if we issue one reduce-scatter per reduce
            # dtype (but we would need a way for users to specify multiple
            # reduce dtypes)
            raise AssertionError(
                f"FSDP expects uniform reduce dtype but got {reduce_dtypes}"
            )
        self._reduce_dtype = next(iter(reduce_dtypes))

    def lazy_init(self):
        # Lazy init should be idempotent
        # Users may change or register parameters after construction time.
        # For example, DoRA (https://arxiv.org/abs/2402.09353) initializes linear magnitudes based on
        # other parameters (e.g. loaded from the state dict).
        if not hasattr(self.comm_ctx, "device_handle"):
            self.comm_ctx.device_handle = _get_device_handle(self.device.type)
        if self.is_sharded and not self._reset_sharded_params:
            for fsdp_param in self.fsdp_params:
                fsdp_param.reset_sharded_param()
                fsdp_param._init_extensions()  # allow monkey patch after init
            self._reset_sharded_params = True
        self._validate_no_meta_params()
        self._validate_cpu_offload_params()
        # Initialize mixed precision attributes lazily in case the user changes
        # the parameter dtypes after construction time but before forward
        self._init_mp_dtypes()
        self._register_state_dict_hooks()

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
        with record_function(self._with_fqn("FSDP::all_gather")):
            self._all_gather_result = foreach_all_gather(
                self.fsdp_params,
                self._all_gather_process_group,
                async_op,
                *self.comm_ctx.get_all_gather_streams(async_op, self._training_state),
                self.device,
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
        async_op = self._all_gather_result.all_gather_work is not None
        if self._training_state == TrainingState.FORWARD:  # implicit prefetch
            if prev_all_gather_state := self.comm_ctx.all_gather_state:
                self._wait_all_gather_streams_on_event(prev_all_gather_state.event)
                self.comm_ctx.all_gather_state = None  # free the all-gather result
        with record_function(self._with_fqn("FSDP::all_gather_copy_out")):
            foreach_all_gather_copy_out(
                self._all_gather_result,
                self.fsdp_params,
                self._all_gather_process_group,
            )
        for fsdp_param in self.fsdp_params:
            fsdp_param.init_unsharded_param()
        self._to_unsharded()
        all_gather_copy_out_event = self.device_handle.Event()
        all_gather_copy_out_event.record()
        if not async_op and self._training_state == TrainingState.FORWARD:
            # Defer free to allow for overlap of this copy-out with next
            # all-gather collective
            self.comm_ctx.all_gather_state = AllGatherState(
                self._all_gather_result, all_gather_copy_out_event
            )
        else:
            self._wait_all_gather_streams_on_event(all_gather_copy_out_event)
        self._all_gather_result = None  # free unless saved in `all_gather_state`

    def _wait_all_gather_streams_on_event(self, event: torch.Event):
        # Calling `unshard` before lazy init means streams are not initialized
        if hasattr(self.comm_ctx, "all_gather_copy_in_stream"):
            self.comm_ctx.all_gather_copy_in_stream.wait_event(event)
        if hasattr(self.comm_ctx, "all_gather_stream"):
            self.comm_ctx.all_gather_stream.wait_event(event)

    def reshard(self):
        if self._training_state == TrainingState.FORWARD:
            if not self._reshard_after_forward:
                return
            if self._use_post_forward_mesh:
                self._to_sharded_post_forward()
                self._reshard_after_forward_event = self.device_handle.Event()
                if self._reshard_after_forward_event is not None:
                    self._reshard_after_forward_event.record()
                return
        self._to_sharded()

    def pre_forward(
        self, module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        if not compiled_autograd_enabled():
            logger.debug("%s", self._with_fqn("FSDP::pre_forward"))
        with record_function(self._with_fqn("FSDP::pre_forward")):
            self._training_state = TrainingState.FORWARD
            self.unshard(self.unshard_async_op)
            self.wait_for_unshard()
            args, kwargs = self._register_post_backward_hook(args, kwargs)
            return args, kwargs

    def post_forward(self, module: nn.Module, input: Any, output: Any):
        if not compiled_autograd_enabled():
            logger.debug("%s", self._with_fqn("FSDP::post_forward"))
        with record_function(self._with_fqn("FSDP::post_forward")):
            self.reshard()
            self._record_post_forward()
            self._training_state = TrainingState.IDLE
            return output

    def _record_post_forward(self) -> None:
        # Since a group has one pre-backward unshard for each forward call
        # before the backward, we record each usage (with multiplicity)
        post_forward_index = len(self.comm_ctx.post_forward_order)
        self.comm_ctx.post_forward_order.append(self)
        self._post_forward_indices.append(post_forward_index)

    def pre_backward(self, default_prefetch: bool, *unused: Any):
        if (
            compiled_autograd_enabled()
            and self._training_state == TrainingState.PRE_BACKWARD
        ):
            # Traceable FSDP2 cannot trigger the param group's `post_backward` immediately after param usage;
            # instead it relies on this to trigger the previously unexecuted `post_backward`.
            self.post_backward()
        if self._training_state == TrainingState.PRE_BACKWARD:
            return
        if not compiled_autograd_enabled():
            logger.debug("%s", self._with_fqn("FSDP::pre_backward"))
        with record_function(self._with_fqn("FSDP::pre_backward")):
            self._training_state = TrainingState.PRE_BACKWARD
            self.unshard(self.unshard_async_op)  # no-op if prefetched
            self.wait_for_unshard()
            if default_prefetch and not compiled_autograd_enabled():
                self._backward_prefetch()

    def post_backward(self, *unused: Any):
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
            fsdp_params_with_grad: List[FSDPParam] = []
            unsharded_grads: List[torch.Tensor] = []
            for fsdp_param in self.fsdp_params:
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
            if self.comm_ctx.reduce_scatter_state is not None:
                self.device_handle.current_stream().wait_event(
                    self.comm_ctx.reduce_scatter_state.event
                )
                self.comm_ctx.reduce_scatter_state = None
            self._wait_for_post_backward()
            (
                reduce_scatter_input,
                reduce_scatter_event,
                self._post_reduce_event,
                all_reduce_input,
                all_reduce_event,
                self._partial_reduce_output,
            ) = foreach_reduce(
                fsdp_params_with_grad,
                unsharded_grads,
                self._reduce_scatter_process_group,
                self.comm_ctx.reduce_scatter_stream,
                self._orig_dtype,
                self._reduce_dtype,
                self.device,
                self.reduce_scatter_reduce_op,
                self._all_reduce_process_group if self._is_hsdp else None,
                self.comm_ctx.all_reduce_stream,
                self.all_reduce_grads,
                self._partial_reduce_output,
            )
            self.comm_ctx.reduce_scatter_state = ReduceScatterState(
                reduce_scatter_input, reduce_scatter_event
            )
            if all_reduce_input is not None:
                assert all_reduce_event is not None
                self._all_reduce_state = AllReduceState(
                    all_reduce_input, all_reduce_event
                )

    def finalize_backward(self):
        self._wait_for_post_backward()
        for fsdp_param in self.fsdp_params:
            if fsdp_param.grad_offload_event is not None:
                fsdp_param.grad_offload_event.synchronize()
                fsdp_param.grad_offload_event = None
        if self._all_gather_result is not None:
            # If there was a mistargeted unshard without a corresponding wait,
            # then we wait here and clear the unshard
            if (event := self._all_gather_result.all_gather_event) is not None:
                torch.cuda.current_stream().wait_event(event)
            work = self._all_gather_result.all_gather_work
            if isinstance(work, dist.distributed_c10d.Work):
                work.wait()
            self._all_gather_result = None
        self._post_forward_indices.clear()

    def _wait_for_post_backward(self):
        if self._post_reduce_event is not None:
            self.device_handle.current_stream().wait_event(self._post_reduce_event)
            self._post_reduce_event = None
        if self._all_reduce_state is not None:
            self.device_handle.current_stream().wait_event(self._all_reduce_state.event)
            self._all_reduce_state = None

    def _backward_prefetch(self) -> None:
        if self._training_state == TrainingState.PRE_BACKWARD:
            if not self._post_forward_indices:
                # Can be cleared if running multiple `backward`s
                return
            curr_index = self._post_forward_indices.pop()
            if (target_index := curr_index - 1) < 0:
                return
            # Prefetch naively using the reverse post-forward order, which may
            # have mistargeted prefetches if not all modules used in forward
            # are used in this backward
            target_fsdp_param_group = self.comm_ctx.post_forward_order[target_index]
            self._prefetch_unshard(target_fsdp_param_group, "backward")

    @staticmethod
    def _prefetch_unshard(
        target_fsdp_param_group: "FSDPParamGroup", pass_type: str
    ) -> None:
        if pass_type == "backward":
            training_state = TrainingState.PRE_BACKWARD
        elif pass_type == "forward":
            training_state = TrainingState.FORWARD
        else:
            raise ValueError(f"Unknown pass type: {pass_type}")
        target_fqn = target_fsdp_param_group._module_fqn
        with record_function(
            f"FSDP::{pass_type}_prefetch for {target_fqn}"
        ), target_fsdp_param_group.use_training_state(training_state):
            async_op = target_fsdp_param_group.unshard_async_op
            target_fsdp_param_group.unshard(async_op)

    # Utilities #
    def _to_sharded(self):
        if not self.is_sharded:
            for fsdp_param in self.fsdp_params:
                fsdp_param.to_sharded()
            self._sharded_state = ShardedState.SHARDED

    def _to_sharded_post_forward(self):
        if not self.is_sharded_post_forward:
            for fsdp_param in self.fsdp_params:
                fsdp_param.to_sharded_post_forward()
            self._sharded_state = ShardedState.SHARDED_POST_FORWARD

    def _to_unsharded(self):
        if not self.is_unsharded:
            for fsdp_param in self.fsdp_params:
                fsdp_param.to_unsharded()
            self._sharded_state = ShardedState.UNSHARDED

    @property
    def is_sharded(self) -> bool:
        return self._sharded_state == ShardedState.SHARDED

    @property
    def is_sharded_post_forward(self) -> bool:
        return self._sharded_state == ShardedState.SHARDED_POST_FORWARD

    @property
    def is_unsharded(self) -> bool:
        return self._sharded_state == ShardedState.UNSHARDED

    @contextlib.contextmanager
    def use_training_state(self, training_state: TrainingState):
        old_training_state = self._training_state
        self._training_state = training_state
        try:
            yield
        finally:
            self._training_state = old_training_state

    # Hook Registration #
    def _register_post_backward_hook(
        self, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        # Traceable FSDP2 relies on `root_post_backward_callback` to call each
        # `FSDPParamGroup.post_backward`
        if (not torch._dynamo.config.skip_fsdp_hooks) or compiled_autograd_enabled():
            return args, kwargs
        if not torch.is_grad_enabled():
            return args, kwargs
        args_list, args_spec = tree_flatten(args)
        kwargs_list, kwargs_spec = tree_flatten(kwargs)
        args_kwargs_list = list(args_list) + list(kwargs_list)
        inp_tensor_indices: List[int] = []
        inp_tensors: List[torch.Tensor] = []
        for i, obj in enumerate(args_kwargs_list):
            if torch.is_tensor(obj) and obj.requires_grad:
                inp_tensor_indices.append(i)
                inp_tensors.append(obj)
        if len(inp_tensors) == 0:
            return args, kwargs  # no tensors that require gradients
        inp_tensors = RegisterPostBackwardFunction.apply(self, *inp_tensors)
        for inp_tensor_idx, inp_tensor in zip(inp_tensor_indices, inp_tensors):
            args_kwargs_list[inp_tensor_idx] = inp_tensor
        args_list = args_kwargs_list[: len(args_list)]
        kwargs_list = args_kwargs_list[len(args_list) :]
        args = tree_unflatten(args_list, args_spec)
        kwargs = tree_unflatten(kwargs_list, kwargs_spec)
        return args, kwargs

    def _register_state_dict_hooks(self) -> None:
        num_pre_save_hooks = len(self._module_to_pre_save_state_dict_hook_handle)
        num_pre_load_hooks = len(self._module_to_pre_load_state_dict_hook_handle)
        assert (
            num_pre_save_hooks == num_pre_load_hooks
        ), f"Pre-save: {num_pre_save_hooks} pre-load: {num_pre_load_hooks}"
        if num_pre_save_hooks > 0:
            return  # already registered
        modules_with_fsdp_params: Set[nn.Module] = {
            fsdp_param._module_info.module for fsdp_param in self.fsdp_params
        }

        def to_sharded_hook(*args: Any, **kwargs: Any) -> None:
            self._to_sharded()

        for module in modules_with_fsdp_params:
            self._module_to_pre_save_state_dict_hook_handle[
                module
            ] = module.register_state_dict_pre_hook(to_sharded_hook)
            self._module_to_pre_load_state_dict_hook_handle[
                module
            ] = module._register_load_state_dict_pre_hook(to_sharded_hook)

    # Properties #
    @property
    def _reshard_after_forward(self) -> bool:
        return self.post_forward_mesh_info is not None

    @property
    def _use_post_forward_mesh(self) -> bool:
        return (
            self._reshard_after_forward
            and self.mesh_info != self.post_forward_mesh_info
        )

    @property
    def _is_hsdp(self) -> bool:
        return isinstance(self.mesh_info, HSDPMeshInfo)

    @property
    def _all_gather_process_group(self) -> dist.ProcessGroup:
        mesh_info = (
            cast(FSDPMeshInfo, self.post_forward_mesh_info)
            if self.is_sharded_post_forward
            else self.mesh_info
        )
        assert isinstance(mesh_info, FSDPMeshInfo)
        return mesh_info.shard_process_group

    @property
    def _reduce_scatter_process_group(self) -> dist.ProcessGroup:
        assert isinstance(self.mesh_info, FSDPMeshInfo)
        return self.mesh_info.shard_process_group

    @property
    def _all_reduce_process_group(self) -> dist.ProcessGroup:
        assert isinstance(self.mesh_info, HSDPMeshInfo)
        return self.mesh_info.replicate_process_group

    def _with_fqn(self, label: str) -> str:
        if self._module_fqn:
            return f"{label} ({self._module_fqn})"
        return label

    def __repr__(self):
        return f"FSDPParamGroup(fqn={self._module_fqn})"

    def _validate_no_meta_params(self):
        param_names_on_meta = [
            fsdp_param._param_fqn
            for fsdp_param in self.fsdp_params
            if fsdp_param.sharded_param.device.type == "meta"
        ]
        if param_names_on_meta:
            raise RuntimeError(
                "FSDP parameters should be materialized from meta device before training, "
                f"but the following were still on meta device: {param_names_on_meta}\n"
                "For example, call module.to_empty(device) to materialize to device and "
                "call module.reset_parameters() on each module to initialize values."
            )

    def _validate_cpu_offload_params(self):
        if not isinstance(self.offload_policy, CPUOffloadPolicy):
            return
        fsdp_params_not_on_cpu = [
            fsdp_param
            for fsdp_param in self.fsdp_params
            if fsdp_param.sharded_param.device.type != "cpu"
        ]
        if fsdp_params_not_on_cpu:
            raise RuntimeError(
                "FSDP parameters should be materialized on CPU when enabling CPU offloading. "
                'For example, load a CPU state dict or call module.to_empty(device="cpu"). '
                "Found following parameters on non-CPU device: "
                f"{[(fsdp_param._param_fqn, fsdp_param.sharded_param.device) for fsdp_param in fsdp_params_not_on_cpu]}\n"
            )


def _get_param_module_infos(
    params: List[nn.Parameter], modules: Tuple[nn.Module, ...]
) -> List[ParamModuleInfo]:
    """
    Shared parameter: lin1.weight = lin2.weight
    Shared module: mlp.lin1 = mlp.lin2
    We do not remove duplicates when traversing both modules and parameters to
    find shared modules' parameters and shared parameters within a module.
    """
    params_set = set(params)
    param_to_module_info: Dict[nn.Parameter, ParamModuleInfo] = {}
    for module in modules:
        for _, submodule in module.named_modules(remove_duplicate=False):
            for param_name, param in _named_parameters_with_duplicates(
                submodule, recurse=False
            ):
                if param in params_set:
                    if param not in param_to_module_info:
                        param_to_module_info[param] = ParamModuleInfo(
                            submodule, param_name
                        )
                    else:
                        param_to_module_info[param].shared_modules.append(submodule)
                        param_to_module_info[param].shared_param_names.append(
                            param_name
                        )
    if len(param_to_module_info) != len(params):
        raise AssertionError(f"Some parameters are not in the module tree of {module}")
    return [param_to_module_info[param] for param in params]


class RegisterPostBackwardFunction(torch.autograd.Function):
    @staticmethod
    def _assert_not_tracing_fsdp():
        if compiled_autograd_enabled():
            # TODO: Find a way to print the offending FSDP2 module.
            msg = """\
When Traceable FSDP2 is enabled, we should not be calling into `RegisterPostBackwardFunction`.
Instead, we rely on the param group's next `pre_backward` hook to trigger its previously unexecuted
`post_backward`, and we rely on FSDPState's `root_post_backward_callback` to trigger the resharding
of any leftover unsharded param groups.
If you are here, it means the forward part of this FSDP2 instance is not compiled, and you must also
compile the forward part if you want to use Traceable FSDP2."""
            torch._dynamo.comptime.comptime.print(msg)
            raise RuntimeError(msg)

    @staticmethod
    def forward(ctx, param_group: FSDPParamGroup, *inputs: torch.Tensor):
        # All tensors in `inputs` should require gradient
        RegisterPostBackwardFunction._assert_not_tracing_fsdp()
        ctx.param_group = param_group
        return inputs

    @staticmethod
    def backward(ctx, *grads: torch.Tensor):
        RegisterPostBackwardFunction._assert_not_tracing_fsdp()
        ctx.param_group.post_backward()
        return (None,) + grads
