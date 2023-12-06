import collections
import contextlib
import functools
import logging
import weakref
from typing import Any, cast, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from torch.distributed.fsdp._common_utils import _named_parameters_with_duplicates
from torch.utils._pytree import tree_flatten, tree_unflatten
from torch.utils.hooks import RemovableHandle

from ._fsdp_api import MixedPrecisionPolicy, OffloadPolicy
from ._fsdp_collectives import (
    AllGatherResult,
    AllGatherState,
    AllGatherStateHolder,
    foreach_all_gather,
    foreach_all_gather_copy_out,
    foreach_reduce_scatter,
)
from ._fsdp_common import (
    FSDP_ENABLE_LOGGING,
    FSDPInternalError,
    FSDPMeshInfo,
    HSDPMeshInfo,
    ParamModuleInfo,
    print_and_raise_internal,
    TrainingState,
)
from ._fsdp_param import FSDPParam, ShardedState


log = logging.getLogger(__name__)
if not FSDP_ENABLE_LOGGING:
    log.disabled = True


_ModuleToHandleDict = Dict[nn.Module, RemovableHandle]  # for state dict


class FSDPParamGroup:
    """
    This class represents a parameter group to communicate together.
    """

    def __init__(
        self,
        params: List[nn.Parameter],
        module: nn.Module,
        mesh_info: FSDPMeshInfo,
        post_forward_mesh_info: Optional[FSDPMeshInfo],
        device: torch.device,
        mp_policy: MixedPrecisionPolicy,
        offload_policy: OffloadPolicy,
    ):
        if (
            post_forward_mesh_info is not None
            and post_forward_mesh_info.mesh.size() == 1
        ):
            # Clamp to `None` since size-1 mesh is equivalent to not resharding
            post_forward_mesh_info = None
        self.module = module
        param_module_infos = self._get_param_module_infos(params, module)
        self.fsdp_params = [
            FSDPParam(
                param,
                module_info,
                mesh_info,
                post_forward_mesh_info,
                device,
                mp_policy,
                offload_policy,
            )
            for param, module_info in zip(params, param_module_infos)
        ]
        self._module_to_fsdp_param_refs = self._get_module_to_fsdp_params(
            self.fsdp_params
        )  # used for state dict
        self.mesh_info = mesh_info
        self.post_forward_mesh_info = post_forward_mesh_info
        self._device: torch.device = device
        self._training_state = TrainingState.IDLE

        # - Mixed precision
        self._orig_dtype, self._reduce_dtype = self._get_mp_dtypes()
        # TODO: Only support fp8/non-fp8 mixing for now.
        self._use_uint8_all_gather = any(
            fsdp_param._is_float8tensor for fsdp_param in self.fsdp_params
        )
        if not self._use_uint8_all_gather:
            unsharded_param_dtypes = {
                fsdp_param.unsharded_param_data_dtype for fsdp_param in self.fsdp_params
            }
            if len(unsharded_param_dtypes) != 1:
                raise AssertionError(
                    f"FSDP only supports uniform unsharded parameter dtype but got {unsharded_param_dtypes}"
                )

        # - Hook state
        self._pre_forward_hook_handle: Optional[RemovableHandle] = None
        self._post_forward_hook_handle: Optional[RemovableHandle] = None
        self._post_backward_reshard_hook_handle: Optional[RemovableHandle] = None
        self._module_to_pre_save_state_dict_hook_handle: _ModuleToHandleDict = {}
        self._module_to_pre_load_state_dict_hook_handle: _ModuleToHandleDict = {}
        self._register_state_dict_hooks()

        # - Communication and communication/computation overlap
        current_stream = torch.cuda.current_stream()
        self.default_stream: torch.cuda.Stream = current_stream
        self.all_gather_copy_in_stream: torch.cuda.Stream = current_stream
        self.all_gather_stream: torch.cuda.Stream = current_stream
        self.reduce_scatter_stream: torch.cuda.Stream = current_stream
        self.all_gather_state = AllGatherStateHolder()
        self.post_forward_order: List[weakref.ref[FSDPParamGroup]] = []
        self._post_forward_index: Optional[int] = None
        self._prefetched: bool = False
        # Used to avoid mistargeted backward prefetches in the case that some
        # module is used in forward but not in backward
        self.expected_backward_unshard_count: int = 0
        # Used to ensure that we only run the pre-backward hook once per
        # (pre-backward, post-backward) interval in case there are multiple
        # forward output tensors that require gradient
        self.ran_pre_backward: bool = False
        # Whether to reduce-scatter or all-reduce gradients, respectively
        # (defaulted to true and can be set to false to save communication
        # during gradient accumulation)
        self._reduce_scatter_grads: bool = True
        self._all_reduce_grads: bool = True
        # Whether to wait for the gradient sync at the end of backward, which
        # can be disabled if microbatching the backward
        self._wait_for_grad_sync: bool = True
        self._init_gradient_divide_factors()

        # - CUDA events for stream synchronization
        # Holds the all-gather output buffer as a list of per-rank shards
        # and the all-gather CUDA event
        self._all_gather_result: Optional[AllGatherResult] = None
        # Holds the reshard-after-forward CUDA event when resharding to a
        # different world size, which should be waited on in the next unshard
        self._reshard_after_forward_event: Optional[torch.cuda.Event] = None
        # Holds the reduce-scatter view-out CUDA event that marks the end of
        # the group's post-backward (e.g. reduce-scatter and div), which should
        # be waited on at the end of backward
        self._reduce_scatter_view_out_event: Optional[torch.cuda.Event] = None

        # - Debuggability
        self._module_fqn: Optional[str] = None  # prefixed from root module

    # Initialization #
    def _get_param_module_infos(
        self, params: List[nn.Parameter], module: nn.Module
    ) -> List[ParamModuleInfo]:
        params_set = set(params)
        param_to_module_info = {}
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
            raise FSDPInternalError(
                f"Some parameters are not in the module tree of {self.module}"
            )
        return [param_to_module_info[param] for param in params]

    def _get_module_to_fsdp_params(
        self, fsdp_params: List[FSDPParam]
    ) -> Dict[nn.Module, List[weakref.ReferenceType]]:
        module_to_fsdp_params: Dict[nn.Module, List] = collections.defaultdict(list)
        for fsdp_param in fsdp_params:
            module = fsdp_param._module_info.module
            module_to_fsdp_params[module].append(weakref.ref(fsdp_param))
        return module_to_fsdp_params

    def _init_gradient_divide_factors(self):
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
        self._gradient_predivide_factor: float = float(factor)
        self._gradient_postdivide_factor: float = (
            data_parallel_world_size / self._gradient_predivide_factor
        )

    def _get_mp_dtypes(self) -> Tuple[torch.dtype, Optional[torch.dtype]]:
        orig_dtypes = {fsdp_param.orig_dtype for fsdp_param in self.fsdp_params}
        if len(orig_dtypes) != 1:
            # This can be relaxed if we copy-out for the reduce-scatter
            raise AssertionError(
                f"FSDP expects uniform original parameter dtype but got {orig_dtypes}"
            )
        orig_dtype = next(iter(orig_dtypes))
        reduce_dtypes = {fsdp_param.reduce_dtype for fsdp_param in self.fsdp_params}
        if len(reduce_dtypes) != 1:
            # This can be relaxed if we issue one reduce-scatter per reduce
            # dtype (but we would need a way for users to specify multiple
            # reduce dtypes)
            raise AssertionError(
                f"FSDP expects uniform reduce dtype but got {reduce_dtypes}"
            )
        reduce_dtype = next(iter(reduce_dtypes))
        return orig_dtype, reduce_dtype

    # Runtime #
    def _unshard(self, async_op: bool = False):
        if self._all_gather_result is not None:  # manually unsharded
            return
        log.info("_unshard for %s", self._module_fqn)
        if self._reshard_after_forward_event is not None:
            # If previously resharded to a different world size, the resharded
            # parameter data are allocated in the default stream and used in
            # the all-gather streams.
            self.all_gather_copy_in_stream.wait_event(self._reshard_after_forward_event)
            self.all_gather_stream.wait_event(self._reshard_after_forward_event)
            self._reshard_after_forward_event = None
        self._all_gather_result = foreach_all_gather(
            self.fsdp_params,
            self._all_gather_process_group,
            async_op,
            self._all_gather_copy_in_stream_for_unshard,
            self._all_gather_stream_for_unshard,
            self._use_uint8_all_gather,
            self._device,
        )

    def _wait_for_unshard(self):
        """
        1. In forward with implict prefetching, to overlap the current copy-out
        with the next all-gather, we save a reference to the current all-gather
        result to free after the next copy-out.
        2. Otherwise (explicit prefetching or in backward), we free the
        all-gather result immediately after the current copy-out.
        """
        if not self._all_gather_result:
            return  # no preceding `_unshard()`
        if (event := self._all_gather_result.all_gather_event) is not None:  # sync
            torch.cuda.current_stream().wait_event(event)
        if (work := self._all_gather_result.all_gather_work) is not None:  # async
            work.wait()
        if self._training_state == TrainingState.FORWARD:  # implicit prefetch
            if prev_all_gather_state := self.all_gather_state.pop():
                self._wait_all_gather_streams_on_event(prev_all_gather_state.event)
                del prev_all_gather_state  # free
        foreach_all_gather_copy_out(
            self._all_gather_result.all_gather_output,
            self.fsdp_params,
            self._all_gather_process_group,
            self._use_uint8_all_gather,
        )
        for fsdp_param in self.fsdp_params:
            fsdp_param.to_unsharded()
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

    def _reshard(self):
        log.info("_reshard for %s", self._module_fqn)
        if self._training_state == TrainingState.FORWARD:
            if not self._reshard_after_forward:
                return
            if self._use_post_forward_mesh:
                self._prefetched = False
                for fsdp_param in self.fsdp_params:
                    fsdp_param.to_sharded_post_forward()
                self._reshard_after_forward_event = torch.cuda.Event()
                self._reshard_after_forward_event.record()
                return
        self._prefetched = False
        for fsdp_param in self.fsdp_params:
            fsdp_param.to_sharded()

    def pre_forward(
        self, module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        log.info("pre-forward for %s", self._module_fqn)
        with torch.profiler.record_function("FSDP::pre_forward"):
            self._training_state = TrainingState.FORWARD
            self._unshard()
            self._wait_for_unshard()
            args, kwargs = self._register_post_backward_hook(args, kwargs)
            return args, kwargs

    def post_forward(self, module: nn.Module, input: Any, output: Any):
        log.info("post-forward for %s", self._module_fqn)
        with torch.profiler.record_function("FSDP::post_forward"):
            self._record_post_forward()
            self._reshard()
            self._training_state = TrainingState.IDLE
            return output

    def _record_post_forward(self) -> None:
        # Since we use a custom autograd function for the post-backward hook,
        # which can run multiple times per backward, we prefer to always record
        # here instead of only recording the first usage
        post_forward_index = len(self.post_forward_order)
        self.post_forward_order.append(weakref.ref(self))
        # TODO: We probably need to append here and clear at the end of
        # backward (when `wait_for_grad_sync=True`). Then, we should consume
        # the index one at a time.
        self._post_forward_index = post_forward_index

    def _pre_backward(self, *unused: Any):
        if self.ran_pre_backward:
            return
        log.info("pre-backward for %s", self._module_fqn)
        with torch.profiler.record_function("FSDP::pre_backward"):
            self._training_state = TrainingState.PRE_BACKWARD
            if not self._prefetched:
                self._unshard()
            self._wait_for_unshard()
            self._prefetch_unshard()
            self.expected_backward_unshard_count -= 1
            self.ran_pre_backward = True

    def _post_backward(self, *unused: Any):
        log.info("post-backward for %s", self._module_fqn)
        # Reset this flag to enable the next (pre-backward, post-backward)
        # interval to handle the case of multiple forwards
        self.ran_pre_backward = False
        self._training_state = TrainingState.POST_BACKWARD
        for fsdp_param in self.fsdp_params:
            fsdp_param.accumulate_unsharded_grad_if_needed()
        if not self._reduce_scatter_grads:
            with torch.profiler.record_function("FSDP::post_backward_reshard"):
                # Reshard parameters before casting to the accumulated
                # gradient (of higher precision) to minimize peak memory
                self._reshard()
                for fsdp_param in self.fsdp_params:
                    fsdp_param.to_accumulated_grad_if_needed()
            return
        with torch.profiler.record_function("FSDP::post_backward_reshard"):
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
            self._reshard()
        if len(fsdp_params_with_grad) == 0:
            return
        with torch.profiler.record_function("FSDP::post_backward_reduce"):
            self._reduce_scatter_view_out_event = foreach_reduce_scatter(
                fsdp_params_with_grad,
                unsharded_grads,
                self._reduce_scatter_process_group,
                self.reduce_scatter_stream,
                self._orig_dtype,
                self._reduce_dtype,
                self._device,
                self._gradient_predivide_factor,
                self._gradient_postdivide_factor,
            )

    def finalize_backward(self):
        log.info("finalizing backward for %s", self._module_fqn)
        current_stream = torch.cuda.current_stream()
        for fsdp_param in self.fsdp_params:
            # Reshard any unsharded parameters, which should mainly happen for
            # the root's parameters since its inputs may not require gradient
            if fsdp_param.state != ShardedState.SHARDED:
                log.info(
                    "running post-backward hook in finalize for %s", self._module_fqn
                )
                self._post_backward()
            if self._wait_for_grad_sync and fsdp_param._grad_offload_event is not None:
                fsdp_param._grad_offload_event.synchronize()
                fsdp_param._grad_offload_event = None
        if self._wait_for_grad_sync and self._reduce_scatter_view_out_event is not None:
            current_stream.wait_event(self._reduce_scatter_view_out_event)
            self._reduce_scatter_view_out_event = None
        self._training_state = TrainingState.IDLE
        self._post_forward_index = None
        self.expected_backward_unshard_count = 0

    def _prefetch_unshard(self):
        if self._training_state in (
            TrainingState.PRE_BACKWARD,
            TrainingState.POST_BACKWARD,
        ):
            if self._post_forward_index is None:
                return
            target_index = self._post_forward_index - 1
            if target_index < 0:
                return
            target_fsdp_param_group = self.post_forward_order[target_index]()
            assert target_fsdp_param_group is not None, "Weakref deallocated"
            if (
                target_fsdp_param_group.expected_backward_unshard_count > 0
                and not target_fsdp_param_group._prefetched
            ):
                with torch.profiler.record_function("FSDP::backward_prefetch"):
                    with target_fsdp_param_group.use_training_state(
                        TrainingState.PRE_BACKWARD
                    ):
                        target_fsdp_param_group._unshard()
                target_fsdp_param_group._prefetched = True

    # State Dict #
    def _pre_save_state_dict_hook(
        self, module: nn.Module, *unused_args, **unused_kwargs
    ):
        if module not in self._module_to_fsdp_param_refs:
            raise FSDPInternalError(
                f"Module not found in module to FSDP parameter mapping: {module}"
            )
        module_fsdp_params = self._module_to_fsdp_param_refs[module]
        for fsdp_param_ref in module_fsdp_params:
            fsdp_param = fsdp_param_ref()
            assert fsdp_param is not None, "FSDPParam deallocated"
            fsdp_param.to_sharded()

    def _pre_load_state_dict_hook(
        self, module: nn.Module, *unused_args, **unused_kwargs
    ):
        if module not in self._module_to_fsdp_param_refs:
            raise FSDPInternalError(
                f"Module not found in module to FSDP parameter mapping: {module}"
            )
        module_fsdp_params = self._module_to_fsdp_param_refs[module]
        for fsdp_param_ref in module_fsdp_params:
            fsdp_param = fsdp_param_ref()
            assert fsdp_param is not None, "FSDPParam deallocated"
            fsdp_param.to_sharded()

    # Utilities #
    @contextlib.contextmanager
    def use_training_state(self, training_state: TrainingState):
        old_training_state = self._training_state
        self._training_state = training_state
        try:
            yield
        finally:
            self._training_state = old_training_state

    def set_reduce_scatter_grads(self, reduce_scatter_grads: bool):
        self._reduce_scatter_grads = reduce_scatter_grads

    def set_all_reduce_grads(self, all_reduce_grads: bool):
        self._all_reduce_grads = all_reduce_grads

    def set_wait_for_grad_sync(self, wait_for_grad_sync: bool):
        self._wait_for_grad_sync = wait_for_grad_sync

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
        for module in self._module_to_fsdp_param_refs:
            for module_to_hook_handle in (
                self._module_to_pre_save_state_dict_hook_handle,
                self._module_to_pre_load_state_dict_hook_handle,
            ):
                if module in module_to_hook_handle:
                    module_to_hook_handle[module].remove()
                    del module_to_hook_handle[module]
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
    def _sharded_state(self) -> ShardedState:
        state = self.fsdp_params[0].state
        for fsdp_param in self.fsdp_params[1:]:
            if state != fsdp_param.state:
                print_and_raise_internal(
                    "Parameters in the same group should be in the same "
                    f"sharded state but got {state} and {fsdp_param.state}"
                )
        return state

    @property
    def _all_gather_process_group(self) -> dist.ProcessGroup:
        mesh_info = (
            cast(FSDPMeshInfo, self.post_forward_mesh_info)
            if self._sharded_state == ShardedState.SHARDED_POST_FORWARD
            else self.mesh_info
        )
        group = mesh_info.shard_process_group
        if group is None:
            print_and_raise_internal(
                f"Mesh without a shard mesh dim does not need all-gather: {mesh_info.mesh}"
            )
        return group

    @property
    def _reduce_scatter_process_group(self) -> dist.ProcessGroup:
        group = self.mesh_info.shard_process_group
        if group is None:
            print_and_raise_internal(
                f"Mesh without a shard dim does not need reduce-scatter: {self.mesh_info.mesh}"
            )
        return group

    @property
    def _all_reduce_process_group(self) -> dist.ProcessGroup:
        mesh_info = cast(HSDPMeshInfo, self.mesh_info)
        group = mesh_info.replicate_process_group
        if group is None:
            print_and_raise_internal(
                f"Mesh without a replicate dim does not need all-reduce: {mesh_info.mesh}"
            )
        return group

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
