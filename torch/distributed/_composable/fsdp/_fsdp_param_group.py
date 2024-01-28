import contextlib
import functools

from typing import Any, cast, Dict, List, Optional, Set, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from torch.distributed.fsdp._common_utils import _named_parameters_with_duplicates
from torch.utils._pytree import tree_flatten, tree_unflatten
from torch.utils.hooks import RemovableHandle
from ._fsdp_api import MixedPrecisionPolicy
from ._fsdp_collectives import (
    AllGatherResult,
    AllGatherState,
    AllGatherStateHolder,
    foreach_all_gather,
    foreach_all_gather_copy_out,
    foreach_reduce_scatter,
)
from ._fsdp_common import (
    _raise_assert_with_print,
    FSDPMeshInfo,
    HSDPMeshInfo,
    TrainingState,
)
from ._fsdp_param import FSDPParam, ParamModuleInfo, ShardedState

_ModuleToHandleDict = Dict[nn.Module, RemovableHandle]  # for state dict


class FSDPCommContext:
    """This has the communication state shared across FSDP states/parameter groups."""

    default_stream: torch.cuda.Stream
    all_gather_copy_in_stream: torch.cuda.Stream
    all_gather_stream: torch.cuda.Stream
    reduce_scatter_stream: torch.cuda.Stream
    all_gather_state: AllGatherStateHolder

    def init(self):
        # Setting the all-gather/reduce-scatter streams to be higher priority
        # can help avoid some issues where their copies in/out are delayed and
        # block computation
        high_priority = -1
        self.default_stream = torch.cuda.current_stream()
        self.all_gather_copy_in_stream = torch.cuda.Stream(priority=high_priority)
        self.all_gather_stream = torch.cuda.Stream(priority=high_priority)
        self.reduce_scatter_stream = torch.cuda.Stream(priority=high_priority)
        self.all_gather_state = AllGatherStateHolder()
        self.post_forward_order: List[FSDPParamGroup] = []


class FSDPParamGroup:
    """This class represents a parameter group to communicate together."""

    _orig_dtype: torch.dtype
    _reduce_dtype: Optional[torch.dtype]
    _param_dtype: torch.dtype

    def __init__(
        self,
        params: List[nn.Parameter],
        module: nn.Module,
        mesh_info: FSDPMeshInfo,
        post_forward_mesh_info: Optional[FSDPMeshInfo],
        device: torch.device,
        mp_policy: MixedPrecisionPolicy,
    ):
        self.module = module  # permit ref cycle because 1:1 lifetime
        param_module_infos = _get_param_module_infos(params, module)
        self.fsdp_params = [
            FSDPParam(
                param, module_info, mesh_info, post_forward_mesh_info, device, mp_policy
            )
            for param, module_info in zip(params, param_module_infos)
        ]
        self.mesh_info = mesh_info
        self.post_forward_mesh_info = post_forward_mesh_info
        self.device = device
        self._training_state = TrainingState.IDLE
        # Group's sharded state always matches its parameters' sharded states
        self._sharded_state = ShardedState.SHARDED
        self._init_mp_dtypes()
        self._module_fqn: Optional[str] = None  # prefixed from root module

        # - Hook state
        self._module_to_pre_save_state_dict_hook_handle: _ModuleToHandleDict = {}
        self._module_to_pre_load_state_dict_hook_handle: _ModuleToHandleDict = {}
        self._register_state_dict_hooks()

        # - Communication and communication/computation overlap
        self.comm_ctx = FSDPCommContext()
        self._post_forward_indices: List[int] = []
        # Used to avoid mistargeted backward prefetches in the case that some
        # module is used in forward but not in backward
        self.expected_backward_unshard_count: int = 0
        # Whether to reduce-scatter or all-reduce gradients, respectively
        # (can be set to false to save communication during gradient
        # accumulation); all-reducing without reduce-scatter is disallowed
        self.reduce_scatter_grads: bool = True
        self.all_reduce_grads: bool = True
        self._init_grad_divide_factors()

        # - CUDA events for stream synchronization
        # Holds the all-gather output buffer, sync objects, and metadata
        self._all_gather_result: Optional[AllGatherResult] = None
        # Holds the reduce-scatter view-out CUDA event that marks the end of
        # the group's post-backward (e.g. reduce-scatter and div), which should
        # be waited on at the end of backward
        self._reduce_scatter_view_out_event: Optional[torch.cuda.Event] = None
        # Holds the reshard-after-forward CUDA event when resharding to a
        # different world size, which should be waited on in the next unshard
        self._reshard_after_forward_event: Optional[torch.cuda.Event] = None

    # Initialization #
    def _init_mp_dtypes(self) -> None:
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
        param_dtypes = {fsdp_param.param_dtype for fsdp_param in self.fsdp_params}
        assert len(param_dtypes) == 1, "FSDPParam.param_dtype is not set correctly"
        self._param_dtype = next(iter(param_dtypes)) or self._orig_dtype

    def _init_grad_divide_factors(self):
        """
        For N data parallel workers, each worker computes g_i, and they
        collectively reduce to compute (g_1 + ... + g_N) / N. To avoid overflow
        and underflow, we divide by sqrt(N) before and after the reduction.
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
        if self._reshard_after_forward_event is not None:
            # Resharded parameter data is allocated in the default stream and
            # used in the all-gather streams
            self._wait_all_gather_streams_on_event(self._reshard_after_forward_event)
            self._reshard_after_forward_event = None
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
            if prev_all_gather_state := self.comm_ctx.all_gather_state.pop():
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
            self.comm_ctx.all_gather_state.put(
                AllGatherState(self._all_gather_result, all_gather_copy_out_event)
            )
        else:
            self._wait_all_gather_streams_on_event(all_gather_copy_out_event)
        self._all_gather_result = None  # free unless saved in `all_gather_state`

    def _wait_all_gather_streams_on_event(self, event: torch.cuda.Event):
        self.comm_ctx.all_gather_copy_in_stream.wait_event(event)
        self.comm_ctx.all_gather_stream.wait_event(event)

    def reshard(self):
        if self._training_state == TrainingState.FORWARD:
            if not self._reshard_after_forward:
                return
            if self._use_post_forward_mesh:
                self._to_sharded_post_forward()
                self._reshard_after_forward_event = torch.cuda.Event()
                self._reshard_after_forward_event.record()
                return
        self._to_sharded()

    def pre_forward(
        self, module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        with torch.profiler.record_function("FSDP::pre_forward"):
            self._training_state = TrainingState.FORWARD
            self.unshard()
            self.wait_for_unshard()
            args, kwargs = self._register_post_backward_hook(args, kwargs)
            return args, kwargs

    def post_forward(self, module: nn.Module, input: Any, output: Any):
        with torch.profiler.record_function("FSDP::post_forward"):
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

    def pre_backward(self, *unused: Any):
        with torch.profiler.record_function("FSDP::pre_backward"):
            self._training_state = TrainingState.PRE_BACKWARD
            self.unshard()  # no-op if prefetched
            self.wait_for_unshard()
            self.expected_backward_unshard_count -= 1
            self._prefetch_unshard()

    def _post_backward(self, *unused: Any):
        self._training_state = TrainingState.POST_BACKWARD
        with torch.profiler.record_function("FSDP::post_backward_reshard"):
            if not self.reduce_scatter_grads:
                self.reshard()
                return
            # Save the autograd-computed gradients before resharding to only
            # access the unsharded parameters when their data is present
            fsdp_params_with_grad: List[FSDPParam] = []
            unsharded_grads: List[torch.Tensor] = []
            for fsdp_param in self.fsdp_params:
                if fsdp_param.unsharded_param.grad is not None:
                    fsdp_params_with_grad.append(fsdp_param)
                    unsharded_grads.append(fsdp_param.unsharded_grad_data)
                    fsdp_param.unsharded_param.grad = None
            self.reshard()
        if len(fsdp_params_with_grad) == 0:
            return
        with torch.profiler.record_function("FSDP::post_backward_reduce"):
            self._reduce_scatter_view_out_event = foreach_reduce_scatter(
                fsdp_params_with_grad,
                unsharded_grads,
                self._reduce_scatter_process_group,
                self.comm_ctx.reduce_scatter_stream,
                self._orig_dtype,
                self._reduce_dtype,
                self.device,
                self._grad_predivide_factor,
                self._grad_postdivide_factor,
            )

    def finalize_backward(self):
        if self._sharded_state == ShardedState.UNSHARDED:
            # Run post-backward here since the forward inputs did not require
            # gradient, so the post-backward hook did not run
            self._post_backward()
        if self._reduce_scatter_view_out_event is not None:
            torch.cuda.current_stream().wait_event(self._reduce_scatter_view_out_event)
            self._reduce_scatter_view_out_event = None
        self._training_state = TrainingState.IDLE
        self._post_forward_indices.clear()
        self.expected_backward_unshard_count = 0

    def _prefetch_unshard(self):
        if self._training_state == TrainingState.PRE_BACKWARD:
            if not self._post_forward_indices:
                msg = "Unexpected backward prefetching without running forward"
                _raise_assert_with_print(msg)
            curr_index = self._post_forward_indices.pop()
            if (target_index := curr_index - 1) < 0:
                return
            target_fsdp_param_group = self.comm_ctx.post_forward_order[target_index]
            if target_fsdp_param_group.expected_backward_unshard_count > 0:
                with torch.profiler.record_function(
                    "FSDP::backward_prefetch"
                ), target_fsdp_param_group.use_training_state(
                    TrainingState.PRE_BACKWARD
                ):
                    target_fsdp_param_group.unshard()

    # State Dict #
    def _pre_save_state_dict_hook(self, module: nn.Module, *args, **kwargs):
        self._to_sharded()

    def _pre_load_state_dict_hook(self, module: nn.Module, *args, **kwargs):
        self._to_sharded()

    # Utilities #
    def _to_sharded(self):
        if self._sharded_state != ShardedState.SHARDED:
            for fsdp_param in self.fsdp_params:
                fsdp_param.to_sharded()
            self._sharded_state = ShardedState.SHARDED

    def _to_sharded_post_forward(self):
        if self._sharded_state != ShardedState.SHARDED_POST_FORWARD:
            for fsdp_param in self.fsdp_params:
                fsdp_param.to_sharded_post_forward()
            self._sharded_state = ShardedState.SHARDED_POST_FORWARD

    def _to_unsharded(self):
        if self._sharded_state != ShardedState.UNSHARDED:
            for fsdp_param in self.fsdp_params:
                fsdp_param.to_unsharded()
            self._sharded_state = ShardedState.UNSHARDED

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
        if not torch.is_grad_enabled():
            return args, kwargs
        args_list, args_spec = tree_flatten(args)
        kwargs_list, kwargs_spec = tree_flatten(kwargs)
        args_kwargs_list = list(args_list) + list(kwargs_list)
        inp_tensor_indices: List[int] = []
        inp_tensors: List[torch.Tensor] = []
        for i, obj in enumerate(args_kwargs_list):
            if not torch.is_tensor(obj) or not obj.requires_grad:
                continue
            inp_tensor_indices.append(i)
            inp_tensors.append(obj)
        if len(inp_tensors) == 0:
            return args, kwargs  # no tensors that require gradients
        inp_tensors = RegisterPostBackwardHook.apply(self, *inp_tensors)
        for inp_tensor_idx, inp_tensor in zip(inp_tensor_indices, inp_tensors):
            args_kwargs_list[inp_tensor_idx] = inp_tensor
        args_list = args_kwargs_list[: len(args_list)]
        kwargs_list = args_kwargs_list[len(args_list) :]
        args = tree_unflatten(args_list, args_spec)
        kwargs = tree_unflatten(kwargs_list, kwargs_spec)
        return args, kwargs

    def _register_state_dict_hooks(self) -> None:
        assert len(self._module_to_pre_save_state_dict_hook_handle) == 0
        assert len(self._module_to_pre_load_state_dict_hook_handle) == 0
        modules_with_fsdp_params: Set[nn.Module] = {
            fsdp_param._module_info.module for fsdp_param in self.fsdp_params
        }
        for module in modules_with_fsdp_params:
            self._module_to_pre_save_state_dict_hook_handle[
                module
            ] = module.register_state_dict_pre_hook(
                functools.partial(self._pre_save_state_dict_hook, module)
            )
            self._module_to_pre_load_state_dict_hook_handle[
                module
            ] = module._register_load_state_dict_pre_hook(
                functools.partial(self._pre_load_state_dict_hook, module)
            )

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
    def _all_gather_process_group(self) -> dist.ProcessGroup:
        mesh_info = (
            cast(FSDPMeshInfo, self.post_forward_mesh_info)
            if self._sharded_state == ShardedState.SHARDED_POST_FORWARD
            else self.mesh_info
        )
        assert isinstance(mesh_info, FSDPMeshInfo)
        return mesh_info.shard_process_group

    @property
    def _reduce_scatter_process_group(self) -> dist.ProcessGroup:
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
            return self.comm_ctx.all_gather_copy_in_stream
        return self.comm_ctx.default_stream

    @property
    def _all_gather_stream_for_unshard(self) -> torch.cuda.Stream:
        if self._use_all_gather_stream:
            return self.comm_ctx.all_gather_stream
        return self.comm_ctx.default_stream


def _get_param_module_infos(
    params: List[nn.Parameter], module: nn.Module
) -> List[ParamModuleInfo]:
    """
    Shared parameter: lin1.weight = lin2.weight
    Shared module: mlp.lin1 = mlp.lin2
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


class RegisterPostBackwardHook(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        param_group,
        *inputs,
    ):
        # All tensors in `inputs` should require gradient
        ctx.param_group = param_group
        return inputs

    @staticmethod
    def backward(ctx, *grads):
        ctx.param_group._post_backward()
        return (None,) + grads
