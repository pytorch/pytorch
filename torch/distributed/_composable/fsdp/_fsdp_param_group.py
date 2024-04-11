import contextlib

from typing import Any, cast, Dict, List, NamedTuple, Optional, Set, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn

from torch.autograd.graph import Node
from torch.distributed.fsdp._common_utils import _named_parameters_with_duplicates
from torch.utils._pytree import tree_flatten, tree_unflatten
from torch.utils.hooks import RemovableHandle
from ._fsdp_api import MixedPrecisionPolicy
from ._fsdp_collectives import (
    AllGatherResult,
    foreach_all_gather,
    foreach_all_gather_copy_out,
    foreach_reduce,
)
from ._fsdp_common import FSDPMeshInfo, HSDPMeshInfo, TrainingState
from ._fsdp_param import FSDPParam, ParamModuleInfo, ShardedState

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

    def init(self):
        # Setting the all-gather/reduce-scatter streams to be higher priority
        # can help avoid some issues where their copies in/out are delayed and
        # block computation
        high_priority = -1
        # All-gather state and copy-in stream allow overlapping the next
        # copy-in with the current all-gather in forward; copy-in overlaps with
        # reduce-scatter in backward without the separate copy-in stream
        self.all_gather_copy_in_stream = torch.cuda.Stream(priority=high_priority)
        self.all_gather_state: Optional[AllGatherState] = None
        # All-gather stream allows overlapping next all-gather with current
        # forward compute
        self.all_gather_stream = torch.cuda.Stream(priority=high_priority)
        # Reduce-scatter stream gives separate execution "thread" for post-
        # backward logic like pre/post-gradient division and reduce-scatter
        self.reduce_scatter_stream = torch.cuda.Stream(priority=high_priority)
        # Run the HSDP all-reduces concurrently with all-gather/reduce-scatter
        # since collectives use different network resources and can overlap
        # in the typical intra-node sharding / inter-node replication case
        self.all_reduce_stream = torch.cuda.Stream()
        # Post-forward order for explicit backward prefetching
        self.post_forward_order: List[FSDPParamGroup] = []  # will cause ref cycles

    def get_all_gather_streams(
        self, training_state: TrainingState
    ) -> Tuple[torch.cuda.Stream, torch.cuda.Stream]:
        if training_state in (TrainingState.FORWARD, TrainingState.PRE_BACKWARD):
            # Use separate streams for implicit prefetching
            return self.all_gather_copy_in_stream, self.all_gather_stream
        current_stream = torch.cuda.current_stream()
        return current_stream, current_stream


# See [Note: Overlapping all-gather copy-in and all-gather]
class AllGatherState(NamedTuple):
    all_gather_result: AllGatherResult
    event: torch.cuda.Event  # all-gather copy-out


class FSDPParamGroup:
    """This class represents a parameter group to communicate together."""

    _orig_dtype: torch.dtype
    _reduce_dtype: Optional[torch.dtype]

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
        self.mp_policy = mp_policy
        self._training_state = TrainingState.IDLE
        # Group's sharded state always matches its parameters' sharded states
        self._sharded_state = ShardedState.SHARDED
        self._module_fqn: Optional[str] = None  # prefixed from root module

        # - Hook state
        self._module_to_pre_save_state_dict_hook_handle: _ModuleToHandleDict = {}
        self._module_to_pre_load_state_dict_hook_handle: _ModuleToHandleDict = {}

        # - Communication and communication/computation overlap
        self.comm_ctx = FSDPCommContext()
        # Group's indices in the shared post-forward order
        self._post_forward_indices: List[int] = []
        # Used to avoid mistargeted backward prefetches when the module is used
        # in forward but not in backward: for each forward, we record a tuple
        # of the output's grad fns and later query the autograd engine whether
        # any grad fn will execute in the current backward to know to prefetch.
        self.all_forward_output_grad_fns: Set[Tuple[Node, ...]] = set()
        # Whether to reduce gradients at all (whether for FSDP or HSDP)
        self.reduce_grads: bool = True
        # Whether to all-reduce gradients for HSDP; only used if
        # `self.reduce_grads` is true, in which case setting this to false
        # means reduce-scatter but no all-reduce
        self.all_reduce_grads: bool = True

        # - CUDA events for stream synchronization
        # Holds the all-gather output buffer, sync objects, and metadata
        self._all_gather_result: Optional[AllGatherResult] = None
        # Holds the reduce-scatter/all-reduce view-out CUDA event that marks the end of
        # the group's post-backward (e.g. reduce-scatter, all-reduce and div), which
        # should be waited on at the end of backward
        self._post_reduce_view_out_event: Optional[torch.cuda.Event] = None
        # Holds the reshard-after-forward CUDA event when resharding to a
        # different world size, which should be waited on in the next unshard
        self._reshard_after_forward_event: Optional[torch.cuda.Event] = None

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

    def _init_grad_divide_factors(self):
        data_parallel_world_size = 1
        data_parallel_world_size *= self.mesh_info.shard_mesh_size
        if isinstance(self.mesh_info, HSDPMeshInfo):
            data_parallel_world_size *= self.mesh_info.replicate_mesh_size
        if self._reduce_dtype in (torch.float32, torch.bfloat16):
            # Use NCCL's AVG op to divide after reduction since it is more
            # performant and fp32 has sufficient precision
            self._grad_divide_factors: Union[Tuple[None, None], Tuple[float, float]] = (
                None,
                None,
            )
            return
        # Since fp16 has smaller dynamic range than fp32/bf16, we want to avoid
        # overflow/underflow. For N data parallel workers, each worker computes
        # g_i, and they collectively reduce (g_1 + ... + g_N) / N. To avoid
        # overflow/underflow, we divide by ~sqrt(N) before/after the reduction.
        factor: int = 1
        while (
            data_parallel_world_size % factor == 0
            and data_parallel_world_size / factor > factor
        ):
            factor *= 2
        factor = float(factor)
        self._grad_divide_factors = (factor, data_parallel_world_size / factor)

    def lazy_init(self):
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
        # Initialize mixed precision attributes lazily in case the user changes
        # the parameter dtypes after construction time but before forward
        self._init_mp_dtypes()
        self._init_grad_divide_factors()
        self._register_state_dict_hooks()

    # Runtime #
    def unshard(self, async_op: bool = False):
        if self._all_gather_result is not None:  # already called, pending wait
            return
        if self.is_unsharded:
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
            *self.comm_ctx.get_all_gather_streams(self._training_state),
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
        if self._training_state == TrainingState.FORWARD:  # implicit prefetch
            if prev_all_gather_state := self.comm_ctx.all_gather_state:
                self._wait_all_gather_streams_on_event(prev_all_gather_state.event)
                self.comm_ctx.all_gather_state = None  # free the all-gather result
        foreach_all_gather_copy_out(
            self._all_gather_result, self.fsdp_params, self._all_gather_process_group
        )
        for fsdp_param in self.fsdp_params:
            fsdp_param.init_unsharded_param()  # no-op after 1st call
        self._to_unsharded()
        all_gather_copy_out_event = torch.cuda.Event()
        all_gather_copy_out_event.record()
        if self._training_state == TrainingState.FORWARD:
            self.comm_ctx.all_gather_state = AllGatherState(
                self._all_gather_result, all_gather_copy_out_event
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

    def pre_backward(self, forward_grad_fns: Tuple[Any, ...], *unused: Any):
        with torch.profiler.record_function("FSDP::pre_backward"):
            self._training_state = TrainingState.PRE_BACKWARD
            self.unshard()  # no-op if prefetched
            self.wait_for_unshard()
            # Can be already removed if running multiple `backward`s
            if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
                self.all_forward_output_grad_fns.discard(forward_grad_fns)
            self._prefetch_unshard()

    def post_backward(self, *unused: Any):
        self._training_state = TrainingState.POST_BACKWARD
        with torch.profiler.record_function("FSDP::post_backward_reshard"):
            if not self.reduce_grads:
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
            self._post_reduce_view_out_event = foreach_reduce(
                fsdp_params_with_grad,
                unsharded_grads,
                self._reduce_scatter_process_group,
                self.comm_ctx.reduce_scatter_stream,
                self._orig_dtype,
                self._reduce_dtype,
                self.device,
                self._grad_divide_factors,
                self._all_reduce_process_group
                if self._should_all_reduce_grads()
                else None,
                self.comm_ctx.all_reduce_stream,
            )

    def finalize_backward(self):
        if self._post_reduce_view_out_event is not None:
            torch.cuda.current_stream().wait_event(self._post_reduce_view_out_event)
            self._post_reduce_view_out_event = None
        self._training_state = TrainingState.IDLE
        self._post_forward_indices.clear()
        if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
            self.all_forward_output_grad_fns.clear()

    def _prefetch_unshard(self):
        if self._training_state == TrainingState.PRE_BACKWARD:
            if not self._post_forward_indices:
                # Can be cleared if running multiple `backward`s
                return
            curr_index = self._post_forward_indices.pop()
            if (target_index := curr_index - 1) < 0:
                return
            target_fsdp_param_group = self.comm_ctx.post_forward_order[target_index]
            # NOTE(yf225): since compile doesn't support `t.grad_fn` access or `torch._C._will_engine_execute_node()` yet,
            # when compile, we always do unshard and rely on Inductor DCE to remove the unnecessary ops
            # NOTE(yf225): unfortunately we can't use `.use_training_state()` ctx manager because Dynamo doesn't support it yet.
            if torch.distributed._functional_collectives.is_torchdynamo_compiling() or \
            any(
                torch._C._will_engine_execute_node(grad_fn)  # type: ignore[attr-defined]
                for grad_fns in target_fsdp_param_group.all_forward_output_grad_fns
                for grad_fn in grad_fns
            ):
                old_training_state = target_fsdp_param_group._training_state
                target_fsdp_param_group._training_state = TrainingState.PRE_BACKWARD
                with torch.profiler.record_function(
                    "FSDP::backward_prefetch"
                ):
                    target_fsdp_param_group.unshard()
                target_fsdp_param_group._training_state = old_training_state

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
        if torch.distributed._functional_collectives.is_torchdynamo_compiling():
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
        assert len(self._module_to_pre_save_state_dict_hook_handle) == 0
        assert len(self._module_to_pre_load_state_dict_hook_handle) == 0
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
        mesh_info = self.mesh_info
        assert isinstance(mesh_info, FSDPMeshInfo)
        return mesh_info.shard_process_group

    @property
    def _all_reduce_process_group(self) -> dist.ProcessGroup:
        mesh_info = self.mesh_info
        assert isinstance(mesh_info, HSDPMeshInfo)
        return mesh_info.replicate_process_group

    def _should_all_reduce_grads(self) -> bool:
        return isinstance(self.mesh_info, HSDPMeshInfo) and self.all_reduce_grads


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


class RegisterPostBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, param_group: FSDPParamGroup, *inputs: torch.Tensor):
        # All tensors in `inputs` should require gradient
        ctx.param_group = param_group
        return inputs

    @staticmethod
    def backward(ctx, *grads: torch.Tensor):
        ctx.param_group.post_backward()
        return (None,) + grads
