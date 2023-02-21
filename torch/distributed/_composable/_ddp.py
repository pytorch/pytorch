import copy
import inspect
import itertools
import logging
import os
import sys
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from enum import auto, Enum
from typing import Any, Callable, Optional, Type

import torch
import torch.distributed as dist
from torch.autograd import Function, Variable
from torch.utils._pytree import tree_flatten, tree_unflatten

if dist.is_available():
    from torch.distributed.distributed_c10d import _get_default_group, ReduceOp
    from torch.distributed.utils import (
        _sync_module_states,
        _to_kwargs,
        _verify_param_shape_across_processes,
    )

from torch._utils import _get_device_index

from torch.nn.modules import Module
from torch.nn.parallel.scatter_gather import gather, scatter_kwargs

__all__ = ["DistributedDataParallel"]

logger = logging.getLogger(__name__)


def _find_tensors(obj):
    r"""
    Recursively find all tensors contained in the specified object.
    """
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return itertools.chain(*map(_find_tensors, obj))
    if isinstance(obj, dict):
        return itertools.chain(*map(_find_tensors, obj.values()))
    return []


class _BufferCommHookLocation(Enum):
    PRE_FORWARD = auto()
    POST_FORWARD = auto()


@dataclass
class _BufferCommHook:
    buffer_comm_hook: Callable
    buffer_comm_hook_state: Any
    buffer_comm_hook_location: _BufferCommHookLocation


# Add a DDPSink to run various functions when backwards starts, such as
# queueing call back of out-most backward/graph task,
# this helps call back is fired after all gradients' calculation
# is completed.
class _DDPSink(Function):
    @staticmethod
    def forward(ctx, reducer, state_dict, *inputs):
        # set_materialize_grads(False) will ensure that None gradients stay as
        # None and are not filled with zeros.
        ctx.set_materialize_grads(False)
        ctx.reducer = reducer
        ctx.state_dict = state_dict
        ret = tuple(
            inp.clone() if isinstance(inp, torch.Tensor) else inp for inp in inputs
        )
        return ret

    @staticmethod
    def backward(ctx, *grad_outputs):
        state_dict = ctx.state_dict
        # Enqueue delay allreduce for static graph training on the first
        # iteration.
        if state_dict["static_graph"] and state_dict["num_iterations"] == 1:
            Variable._execution_engine.queue_callback(ctx.reducer._delay_all_reduce)  # type: ignore[call-arg,misc]

        return (None, None, *grad_outputs)


class DistributedDataParallel(Module):
    # used to track whether the given thread is inside ddp forward for torchdynamo purposes
    _active_ddp_module = None

    def __init__(
        self,
        module,
        device_ids=None,
        output_device=None,
        dim=0,
        broadcast_buffers=True,
        process_group=None,
        bucket_cap_mb=25,
        find_unused_parameters=False,
        gradient_as_bucket_view=False,
        static_graph=False,
    ):

        super().__init__()
        self.logger: Optional[dist.Logger] = None
        if not any((p.requires_grad for p in module.parameters())):
            self._log_and_throw(
                RuntimeError,
                "DistributedDataParallel is not needed when a module "
                "doesn't have any parameter that requires a gradient.",
            )

        if device_ids is not None and len(device_ids) > 1:
            self._log_and_throw(
                ValueError,
                "device_ids can only be None or contain a single element.",
            )

        self.is_multi_device_module = len({p.device for p in module.parameters()}) > 1
        distinct_device_types = {p.device.type for p in module.parameters()}
        if len(distinct_device_types) != 1:
            self._log_and_throw(
                ValueError,
                "DistributedDataParallel's input module must be on "
                "the same type of devices, but input module parameters locate in {}.".format(
                    distinct_device_types
                ),
            )

        self.device_type = list(distinct_device_types)[0]

        if (
            device_ids is None
            or len(device_ids) == 0  # For backward compatibility.
            or self.device_type == "cpu"
            or self.is_multi_device_module
        ):
            if device_ids or output_device:
                self._log_and_throw(
                    ValueError,
                    "DistributedDataParallel device_ids and output_device arguments "
                    "only work with single-device/multiple-device GPU modules or CPU modules, "
                    "but got device_ids {}, output_device {}, and module parameters {}.".format(
                        device_ids,
                        output_device,
                        {p.device for p in module.parameters()},
                    ),
                )

            self.device_ids = None
            self.output_device = None
        else:
            self.device_ids = [_get_device_index(x, True) for x in device_ids]

            if output_device is None:
                output_device = device_ids[0]

            self.output_device = _get_device_index(output_device, True)

        if process_group is None:
            self.process_group = _get_default_group()
        else:
            self.process_group = process_group

        self.static_graph = False
        self.dim = dim
        self.module = module
        self.device = list(self.module.parameters())[0].device
        self.broadcast_buffers = broadcast_buffers
        self.find_unused_parameters = find_unused_parameters
        self.require_backward_grad_sync = True
        self.require_forward_param_sync = True
        self.gradient_as_bucket_view = gradient_as_bucket_view
        if hasattr(module, "_ddp_params_and_buffers_to_ignore"):
            self.parameters_to_ignore = module._ddp_params_and_buffers_to_ignore
        else:
            self.parameters_to_ignore = []

        # Check that a module does not have Uninitialized parameters
        for param in module.parameters():
            if isinstance(param, torch.nn.parameter.UninitializedParameter):
                self._log_and_throw(
                    RuntimeError,
                    "Modules with uninitialized parameters can't be used with `DistributedDataParallel`. "
                    "Run a dummy forward pass to correctly initialize the modules",
                )
        # used for intra-node param sync and inter-node sync as well
        self.broadcast_bucket_size = int(250 * 1024 * 1024)

        # reduction bucket size
        self.bucket_bytes_cap = int(bucket_cap_mb * 1024 * 1024)
        # Whether to perform input tensor CPU to GPU copies on a side-stream
        self.use_side_stream_for_tensor_copies = (
            os.environ.get("PYTORCH_DDP_USE_SIDE_STREAM", "1") == "1"
        )

        # Build parameters for reducer.
        parameters, expect_sparse_gradient = self._build_params_for_reducer()
        # Verify model equivalence.
        _verify_param_shape_across_processes(self.process_group, parameters)
        # Sync params and buffers. Ensures all DDP models start off at the same value.
        _sync_module_states(
            module=self.module,
            process_group=self.process_group,
            broadcast_bucket_size=self.broadcast_bucket_size,
            src=0,
            params_and_buffers_to_ignore=self.parameters_to_ignore,
        )
        # In debug mode, build a mapping of parameter index -> parameter.
        param_to_name_mapping = self._build_debug_param_to_name_mapping(parameters)
        # Builds reducer.
        self._ddp_init_helper(
            parameters,
            expect_sparse_gradient,
            param_to_name_mapping,
            static_graph,
        )
        self._has_rebuilt_buckets = False

        if static_graph:
            self._set_static_graph()

    def _log_and_throw(self, err_type, err_msg):
        if self.logger is not None:
            self.logger.set_error_and_log(f"{str(err_type)}: {err_msg}")
        raise err_type(err_msg)

    def _ddp_init_helper(
        self,
        parameters,
        expect_sparse_gradient,
        param_to_name_mapping,
        static_graph,
    ):
        """
        Initialization helper function that does the following:
        (1) bucketing the parameters for reductions
        (2) resetting the bucketing states
        (3) registering the grad hooks
        (4) Logging construction-time DDP logging data
        (5) passing a handle of DDP to SyncBatchNorm Layer
        """
        self.num_iterations = 0
        # Notice, the parameters order is not in the order in which they are used,
        # especially in models with control flow.
        #
        # Alongside parameters are not presented in the real execution order,
        # if a certain model happens to also
        #   1) have other collectives comm ops in its backward graph.
        #   2) have unused parameter in subset ranks of the whole world.
        # bucketing could insert ALL-REDUCE comm op too early on the rank with unused parameter,
        # matching up with other collectives comm ops on other ranks unexpectedly.
        #
        # In order to handle this corner case, when the parameters are not in the real execution order,
        # we don't do bucketing, thus only one ALL-REDUCE is inserted after all the gradients
        # of the whole graph are computed.
        #
        # Notice, here we only disable bucketing for the first iteration.
        # After the first iteration, it's OK to rebuild buckets,
        # because "bucket rebuild" bucketizes parameters based on its real execution order in backward graph.

        # Can remove this branching once #73732 is landed.
        if static_graph is True or self.find_unused_parameters is False:
            bucket_size_limits = [sys.maxsize]
        else:
            bucket_size_limits = [
                dist._DEFAULT_FIRST_BUCKET_BYTES,
                self.bucket_bytes_cap,
            ]
        (
            bucket_indices,
            per_bucket_size_limits,
        ) = dist._compute_bucket_assignment_by_size(
            parameters,
            bucket_size_limits,
            expect_sparse_gradient,
        )

        # Note: reverse list of buckets because we want to approximate the
        # order in which their gradients are produced, and assume they
        # are used in the forward pass in the order they are defined.
        self.reducer = dist.Reducer(
            parameters,
            list(reversed(bucket_indices)),
            list(reversed(per_bucket_size_limits)),
            self.process_group,
            expect_sparse_gradient,
            # The bucket size limit is specified in the constructor.
            # Additionally, we allow for a single small bucket for parameters
            # that are defined first, such that their gradients don't spill into
            # a much larger bucket, adding unnecessary latency after gradient
            # computation finishes. Experiments showed 1MB is a reasonable value.
            self.bucket_bytes_cap,
            self.find_unused_parameters,
            self.gradient_as_bucket_view,
            param_to_name_mapping,
            # User can set dist._DEFAULT_FIRST_BUCKET_BYTES to tune DDP first
            # bucket.
            dist._DEFAULT_FIRST_BUCKET_BYTES,
        )

        self.logger = dist.Logger(self.reducer)
        # Set as a weak reference to avoid reference cycle between
        # logger and reducer.
        self.reducer.set_logger(self.logger)

        has_sync_bn = False
        for submodule in self.module.modules():
            if isinstance(submodule, torch.nn.SyncBatchNorm):
                has_sync_bn = True
                break

        # Set logging data that can be got during construction time.
        self.logger.set_construction_data_and_log(
            self.module.__class__.__name__,
            [] if self.device_ids is None else self.device_ids,
            -1 if self.output_device is None else self.output_device,
            self.broadcast_buffers,
            has_sync_bn,
            static_graph,
        )

        # passing a handle to torch.nn.SyncBatchNorm layer
        self._passing_sync_batchnorm_handle(self.module)

    def __getstate__(self):
        self._check_default_group()
        attrs = copy.copy(self.__dict__)
        del attrs["process_group"]
        del attrs["reducer"]
        del attrs["logger"]
        return attrs

    def __setstate__(self, state):
        # If serializable, then the process group should be the default one
        self.process_group = _get_default_group()
        super().__setstate__(state)
        self.__dict__.setdefault("require_forward_param_sync", True)
        self.__dict__.setdefault("require_backward_grad_sync", True)
        parameters, expect_sparse_gradient = self._build_params_for_reducer()
        # In debug mode, build a mapping of parameter index -> parameter.
        param_to_name_mapping = self._build_debug_param_to_name_mapping(parameters)
        # Builds reducer.
        self._ddp_init_helper(
            parameters,
            expect_sparse_gradient,
            param_to_name_mapping,
            self.static_graph,
        )
        if self.static_graph:
            self.reducer._set_static_graph()
            assert self.logger is not None
            self.logger._set_static_graph()

    def _build_params_for_reducer(self):
        # Build tuple of (module, parameter) for all parameters that require grads.
        modules_and_parameters = [
            (module, parameter)
            for module_name, module in self.module.named_modules()
            for parameter in [
                param
                # Note that we access module.named_parameters instead of
                # parameters(module). parameters(module) is only needed in the
                # single-process multi device case, where it accesses replicated
                # parameters through _former_parameters.
                for param_name, param in module.named_parameters(recurse=False)
                if param.requires_grad
                and f"{module_name}.{param_name}" not in self.parameters_to_ignore
            ]
        ]

        # Deduplicate any parameters that might be shared across child modules.
        memo = set()
        modules_and_parameters = [
            # "p not in memo" is the deduplication check.
            # "not memo.add(p)" is always True, and it's only there to cause "add(p)" if needed.
            (m, p)
            for m, p in modules_and_parameters
            if p not in memo and not memo.add(p)  # type: ignore[func-returns-value]
        ]

        # Build list of parameters.
        parameters = [parameter for _, parameter in modules_and_parameters]

        # Checks if a module will produce a sparse gradient.
        def produces_sparse_gradient(module):
            if isinstance(module, (torch.nn.Embedding, torch.nn.EmbeddingBag)):
                return module.sparse
            return False

        # Build list of booleans indicating whether or not to expect sparse
        # gradients for the corresponding parameters.
        expect_sparse_gradient = [
            produces_sparse_gradient(module) for module, _ in modules_and_parameters
        ]

        self._assign_modules_buffers()

        return parameters, expect_sparse_gradient

    def _assign_modules_buffers(self):
        """
        Assigns module buffers to self.modules_buffers which are then used to
        broadcast across ranks when broadcast_buffers=True. Note that this
        must be called every time buffers need to be synced because buffers can
        be reassigned by user module,
        see https://github.com/pytorch/pytorch/issues/63916.
        """
        # Collect buffers for modules, filtering out buffers that should be ignored.
        named_module_buffers = [
            (buffer, buffer_name)
            for buffer_name, buffer in self.module.named_buffers()
            if buffer_name not in self.parameters_to_ignore
        ]
        self.modules_buffers = [
            buffer for (buffer, buffer_name) in named_module_buffers
        ]
        # Dict[str, tensor] representing module buffers not ignored by DDP.
        self.named_module_buffers = {
            buffer_name: buffer for (buffer, buffer_name) in named_module_buffers
        }

    def _build_debug_param_to_name_mapping(self, parameters):
        if dist.get_debug_level() == dist.DebugLevel.OFF:
            return {}

        param_to_param_index = {parameters[i]: i for i in range(len(parameters))}
        param_set = set(parameters)
        param_index_to_param_fqn = {}
        for module_name, module in self.module.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fqn = f"{module_name}.{param_name}"
                # Bypass ignored parameters since those are not reduced by DDP
                # to begin with.
                if fqn not in self.parameters_to_ignore and param.requires_grad:
                    if param not in param_set:
                        self._log_and_throw(
                            ValueError,
                            f"Param with name {fqn} found in module parameters, but not DDP parameters."
                            " This indicates a bug in DDP, please report an issue to PyTorch.",
                        )
                    param_index = param_to_param_index[param]
                    param_index_to_param_fqn[param_index] = fqn

        # Ensure we covered all parameters
        if len(param_set) != len(param_index_to_param_fqn):
            self._log_and_throw(
                ValueError,
                (
                    "Expected param to name mapping to cover all parameters, but"
                    f" got conflicting lengths: {len(param_set)} vs "
                    f"{len(param_index_to_param_fqn)}. This indicates a bug in DDP"
                    ", please report an issue to PyTorch."
                ),
            )

        return param_index_to_param_fqn

    def _get_parameters(self, m, recurse=True):
        """
        Returns a generator of module parameters
        """

        def model_parameters(m):
            ps = (
                m._former_parameters.values()
                if hasattr(m, "_former_parameters")
                else m.parameters(recurse=False)
            )
            yield from ps

        for m in m.modules() if recurse else [m]:
            for p in model_parameters(m):
                yield p

    def _check_default_group(self):
        pickle_not_supported = False
        try:
            if self.process_group != _get_default_group():
                pickle_not_supported = True
        except RuntimeError:
            pickle_not_supported = True

        if pickle_not_supported:
            self._log_and_throw(
                RuntimeError,
                "DDP Pickling/Unpickling are only supported "
                "when using DDP with the default process "
                "group. That is, when you have called "
                "init_process_group and have not passed "
                "process_group argument to DDP constructor",
            )

    @contextmanager
    def no_sync(self):
        r"""
        A context manager to disable gradient synchronizations across DDP
        processes. Within this context, gradients will be accumulated on module
        variables, which will later be synchronized in the first
        forward-backward pass exiting the context.

        Example::

            >>> # xdoctest: +SKIP("undefined variables")
            >>> ddp = torch.nn.parallel.DistributedDataParallel(model, pg)
            >>> with ddp.no_sync():
            >>>   for input in inputs:
            >>>     ddp(input).backward()  # no synchronization, accumulate grads
            >>> ddp(another_input).backward()  # synchronize grads
        """
        old_require_backward_grad_sync = self.require_backward_grad_sync
        self.require_backward_grad_sync = False
        try:
            yield
        finally:
            self.require_backward_grad_sync = old_require_backward_grad_sync

    @classmethod
    def _get_active_ddp_module(cls):
        """
        TorchDynamo needs to know whether DDP is currently active, and access the DDP module in order to cooperatively optimize it.
        """
        return cls._active_ddp_module

    # note, this ctxmgr function is marked 'skip' in torchdynamo, so dynamo only kicks in
    # for the 'module_to_run' underneath
    # see torchdynamo/eval_frame.py TorchPatcher.patch for more details
    @contextmanager
    def _inside_ddp_forward(self):
        DistributedDataParallel._active_ddp_module = self
        try:
            yield
        except Exception:
            raise
        finally:
            DistributedDataParallel._active_ddp_module = None

    def pre_forward(self):
        with torch.autograd.profiler.record_function(
            "DistributedDataParallel.pre_forward"
        ):
            if torch.is_grad_enabled() and self.require_backward_grad_sync:
                assert self.logger is not None
                self.logger.set_runtime_stats_and_log()
                self.num_iterations += 1
                self.reducer.prepare_for_forward()

            # Calling _rebuild_buckets before forward computation,
            # It may allocate new buckets before deallocating old buckets
            # inside _rebuild_buckets. To save peak memory usage,
            # call _rebuild_buckets before the peak memory usage increases
            # during forward computation.
            # This should be called only once during whole training period.
            if torch.is_grad_enabled() and self.reducer._rebuild_buckets():
                logger.info("Reducer buckets have been rebuilt in this iteration.")
                self._has_rebuilt_buckets = True

            # sync params according to location (before/after forward) user
            # specified as part of hook, if hook was specified.
            if self._check_sync_bufs_pre_fwd():
                self._sync_buffers()

    def post_forward(self, output):
        with torch.autograd.profiler.record_function(
            "DistributedDataParallel.post_forward"
        ):
            # sync params according to location (before/after forward) user
            # specified as part of hook, if hook was specified.
            if self._check_sync_bufs_post_fwd():
                self._sync_buffers()

            if torch.is_grad_enabled() and self.require_backward_grad_sync:
                self.require_forward_param_sync = True
                # We'll return the output object verbatim since it is a freeform
                # object. We need to find any tensors in this object, though,
                # because we need to figure out which parameters were used during
                # this forward pass, to ensure we short circuit reduction for any
                # unused parameters. Only if `find_unused_parameters` is set.
                if self.find_unused_parameters and not self.static_graph:
                    # Do not need to populate this for static graph.
                    self.reducer.prepare_for_backward(list(_find_tensors(output)))
                else:
                    self.reducer.prepare_for_backward([])
            else:
                self.require_forward_param_sync = False

        # TODO: DDPSink is currently enabled for unused parameter detection and
        # static graph training for first iteration.
        if (self.find_unused_parameters and not self.static_graph) or (
            self.static_graph and self.num_iterations == 1
        ):
            state_dict = {
                "static_graph": self.static_graph,
                "num_iterations": self.num_iterations,
            }

            output_tensor_list, treespec = tree_flatten(output)
            output_placeholders = [None for _ in range(len(output_tensor_list))]
            # Do not touch tensors that have no grad_fn, which can cause issues
            # such as https://github.com/pytorch/pytorch/issues/60733
            for i, output in enumerate(output_tensor_list):
                if torch.is_tensor(output) and output.grad_fn is None:
                    output_placeholders[i] = output

            # When find_unused_parameters=True, makes tensors which require grad
            # run through the DDPSink backward pass. When not all outputs are
            # used in loss, this makes those corresponding tensors receive
            # undefined gradient which the reducer then handles to ensure
            # param.grad field is not touched and we don't error out.
            passthrough_tensor_list = _DDPSink.apply(
                self.reducer,
                state_dict,
                *output_tensor_list,
            )
            for i in range(len(output_placeholders)):
                if output_placeholders[i] is None:
                    output_placeholders[i] = passthrough_tensor_list[i]

            # Reconstruct output data structure.
            output = tree_unflatten(output_placeholders, treespec)
        return output

    def forward(self, *inputs, **kwargs):
        self.pre_forward(*inputs, **kwargs)
        with torch.autograd.profiler.record_function("DistributedDataParallel.forward"):
            if self.device_ids:
                inputs, kwargs = _to_kwargs(
                    inputs,
                    kwargs,
                    self.device_ids[0],
                    self.use_side_stream_for_tensor_copies,
                )
                with self._inside_ddp_forward():
                    output = self.module(*inputs[0], **kwargs[0])  # type: ignore[index]
            else:
                with self._inside_ddp_forward():
                    output = self.module(*inputs, **kwargs)

        output = self.post_forward(output)
        return output

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def to_kwargs(self, inputs, kwargs, device_id):
        # Kept for BC
        return _to_kwargs(
            inputs, kwargs, device_id, self.use_side_stream_for_tensor_copies
        )

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)

    def train(self, mode=True):
        super().train(mode)
        return self

    # When running in join mode, schedules an allreduce to notify joined ranks
    # of whether backwards pass synchronization will run this iteration or not.
    def _check_global_requires_backward_grad_sync(self, is_joined_rank):
        if not is_joined_rank and self.require_backward_grad_sync:
            requires_sync_tensor = torch.ones(1, device=self.device)
        else:
            requires_sync_tensor = torch.zeros(1, device=self.device)

        work = dist.all_reduce(
            requires_sync_tensor, group=self.process_group, async_op=True
        )
        return work

    # When running in join mode, checks and performs sync of module buffers if
    # the models have buffers that should be synchronized in the forward pass.
    def _check_and_sync_module_buffers(self):
        if self._check_sync_bufs_pre_fwd():
            authoritative_rank = self._find_common_rank(self._distributed_rank, False)
            self._sync_module_buffers(authoritative_rank)

    # When running in join model, agrees upon a common rank and broadcast model
    # parameters to all other ranks.
    def _sync_final_model(self, is_last_joiner):
        # Agree upon the process that will be the authoritative model copy.
        # The current rank is a candidate for being the authoritative copy if
        # is_last_joiner=True. We break ties via picking the larger rank.
        self._authoritative_rank = self._find_common_rank(
            self._distributed_rank, is_last_joiner
        )
        _sync_module_states(
            module=self.module,
            process_group=self.process_group,
            broadcast_bucket_size=self.broadcast_bucket_size,
            src=self._authoritative_rank,
            params_and_buffers_to_ignore=self.parameters_to_ignore,
        )

    # Schedule comm ops to match those scheduled in the reducer's backward
    # pass.
    def _match_all_reduce_for_bwd_pass(self):
        comm_work = []
        # Schedule comm in the same order as Reducer schedules them, i.e.
        # the order of the buckets. Retrieving the bucket order from the reducer
        # ensures that we keep the same order in join mode, such as when bucket
        # order is rebuilt dynamically.

        # Returns grad_buckets in order, but real tensors are substituted with
        # zero tensors of the same shape.
        grad_buckets = self.reducer._get_zeros_like_grad_buckets()
        for grad_bucket in grad_buckets:
            # Joined processes contribute zero gradient. In the case that
            # divide_by_initial_world_size=True, we divide grads by the static
            # world size, if not, the dividing factor is reduced by the number
            # of joined processes.
            work = self.reducer._run_comm_hook(grad_bucket)
            comm_work.append(work)
        for work in comm_work:
            work.wait()

    # Allreduces the used parameter mapping across ranks.
    def _match_unused_params_allreduce(self):
        locally_used_param_map = self.reducer._get_local_used_map()
        self.process_group.allreduce(locally_used_param_map)

    def _register_buffer_comm_hook(
        self,
        state,
        hook: Callable,
        comm_hook_location=_BufferCommHookLocation.POST_FORWARD,
    ):
        r"""
        Allows custom registration of hooks that define how buffer are
        synchronized across ranks. The hook takes in an optional state
        and is passed in a Dict[str, Tensor] corresponding to buffer names
        and the buffers, and can run arbitrary reductions on buffers as
        opposed to DDP's default broadcast from rank 0. This is useful for
        example if a counter needs to be summed or averaged across ranks
        every iteration.

        Args:
            state (Any): Optional state that is passed to the hook.
            hook (Callable): Callable with the following signature:
                         ``hook(state: object, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]``:
            comm_hook_location (_BufferCommHookLocation): Enum value indicating
                            where to run the hook.
                            _BufferCommHookLocation.PRE_FORWARD means that the
                            hook will run _before_ the forward pass, and
                            _BufferCommHookLocation.POST_FORWARD means that the
                            hook will run _after_ the forward pass.

            NOTE: To maximize performance, users can return a
                List[torch.futures.Future] from their hook, and DDP will
                install and await these hooks appropriately at the end of
                the backward pass. This will ensure all buffers are
                synchronized by the end of the backward pass. If this
                setting is used, it is recommended to pass
                comm_hook_location=_BufferCommHookLocation.POST_FORWARD,
                which will trigger the hook after the forward pass.
                If _BufferCommHookLocation.PRE_FORWARD is used, users must
                ensure appropriate synchronization when manipulating GPU
                buffers in the forward pass.
        """
        assert callable(hook)
        self.buffer_hook = _BufferCommHook(
            buffer_comm_hook=hook,
            buffer_comm_hook_state=state,
            buffer_comm_hook_location=comm_hook_location,
        )

    def register_comm_hook(self, state: object, hook: Callable):
        r"""
        Registers a communication hook which is an enhancement that provides a
        flexible hook to users where they can specify how DDP aggregates gradients
        across multiple workers.

        This hook would be very useful for researchers to try out new ideas. For
        example, this hook can be used to implement several algorithms like GossipGrad
        and gradient compression which involve different communication strategies for
        parameter syncs while running Distributed DataParallel training.

        Args:
            state (object): Passed to the hook to maintain any state information during the training process.
                            Examples include error feedback in gradient compression,
                            peers to communicate with next in GossipGrad, etc.

                            It is locally stored by each worker
                            and shared by all the gradient tensors on the worker.
            hook (Callable): Callable with the following signature:
                             ``hook(state: object, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]``:

                             This function is called once the bucket is ready. The
                             hook can perform whatever processing is needed and return
                             a Future indicating completion of any async work (ex: allreduce).
                             If the hook doesn't perform any communication, it still
                             must return a completed Future. The Future should hold the
                             new value of grad bucket's tensors. Once a bucket is ready,
                             c10d reducer would call this hook and use the tensors returned
                             by the Future and copy grads to individual parameters.
                             Note that the future's return type must be a single tensor.

                             We also provide an API called ``get_future`` to retrieve a
                             Future associated with the completion of ``c10d.ProcessGroup.Work``.
                             ``get_future`` is currently supported for NCCL and also supported for most
                             operations on GLOO and MPI, except for peer to peer operations (send/recv).

        .. warning ::
            Grad bucket's tensors will not be predivided by world_size. User is responsible
            to divide by the world_size in case of operations like allreduce.

        .. warning ::
            DDP communication hook can only be registered once and should be registered
            before calling backward.

        .. warning ::
            The Future object that hook returns should contain a single tensor
            that has the same shape with the tensors inside grad bucket.

        .. warning ::
            ``get_future`` API supports NCCL, and partially GLOO and MPI backends (no support
            for peer-to-peer operations like send/recv) and will return a ``torch.futures.Future``.

        Example::
            Below is an example of a noop hook that returns the same tensor.

            >>> # xdoctest: +REQUIRES(module:torch._C._distributed_c10d)
            >>> def noop(state: object, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
            >>>     fut = torch.futures.Future()
            >>>     fut.set_result(bucket.buffer())
            >>>     return fut

            >>> # xdoctest: +SKIP('undefined name')
            >>> ddp.register_comm_hook(state=None, hook=noop)

        Example::
            Below is an example of a Parallel SGD algorithm where gradients are encoded before
            allreduce, and then decoded after allreduce.

            >>> # xdoctest: +REQUIRES(module:torch._C._distributed_c10d)
            >>> def encode_and_decode(state: object, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
            >>>     encoded_tensor = encode(bucket.buffer()) # encode gradients
            >>>     fut = torch.distributed.all_reduce(encoded_tensor).get_future()
            >>>     # Define the then callback to decode.
            >>>     def decode(fut):
            >>>         decoded_tensor = decode(fut.value()[0]) # decode gradients
            >>>         return decoded_tensor
            >>>     return fut.then(decode)

            >>> # xdoctest: +SKIP('undefined name')
            >>> ddp.register_comm_hook(state=None, hook=encode_and_decode)
        """
        self._check_comm_hook(hook)
        assert self.logger is not None
        self.logger._set_comm_hook_name(hook.__qualname__)
        dist._register_comm_hook(self.reducer, state, hook)

    def _register_builtin_comm_hook(self, comm_hook_type):
        r"""
        Registers a built-in communication hook that specifies how DDP
        aggregates gradients across multiple workers.
        The built-in hooks aim to provide efficient C++ implementations for certain hooks,
        which might not be as efficient if implemented in Python using a Python communication hook.

        Args:
            comm_hook_type (dist.BuiltinCommHookType): type of communication hook, such as ALLREDUCE, FP16_COMPRESS, etc.

        .. warning ::
            DDP communication hook can only be registered once and should be registered
            before calling backward.

        Example::
            Below is an example of a FP16 compression where gradients are
            compressed into 16-bit floating-point numbers before allreduce, and
            then decompressed after allreduce.

            >>> # xdoctest: +SKIP('undefined name')
            >>> ddp._register_builtin_comm_hook(dist.BuiltinCommHookType.FP16_COMPRESS)

        """
        assert self.logger is not None
        self.logger._set_comm_hook_name(str(comm_hook_type))
        dist._register_builtin_comm_hook(self.reducer, comm_hook_type)

    def _register_fused_optim(self, optim: Type, *args, optim_params=None, **kwargs):
        r"""
            Registers an optimizer with DDP such that the optimization for a
            parameter will run immediately when that parameter's gradient is
            finished with reduction, instead of waiting for all parameters'
            gradients to finish reduction. This can result in a training speedup
            depending on your workload since the optimizer can run while gradient
            reduction for other parameters are still ongoing. In addition, this has
            the potential to reduce peak memory consumption during training, as it
            only needs to load the per-parameter optimizer states of a single
            parameter at a time, instead of loading all per-parameter optimizer
            states at once.

            Args:
                optim_cls (Type): a ``torch.optim.Optimizer`` class to be registered
                as a fused optimizer.
                *args (Sequence[Any]): Arguments to forward to `optim_cls`.
                optim_params (Optional[Iterable[torch.Tensor]]): Set of parameters
                to optimize, similar to `params` argument of traditional `torch.optim`
                Optimizers. If this is omitted, all DDP model parameters will be
                optimized.
                **kwargs: (Dict[str, Any]): Keyword arguments to forward to `optim_cls`.

        .. warning ::
            _register_fused_optim should only be called once on a DDP instance,
            and registering multiple fused optimizers for the same DDP model
            is not currently supported. Please ping
            https://github.com/pytorch/pytorch/issues/71595 if this is necessary
            for your use case.

        .. warning ::
            _register_fused_optim and register_comm_hook currently do not
            compose together, meaning that custom DDP communication hooks are
            not supported with overlapped optimizers. Please ping
            https://github.com/pytorch/pytorch/issues/71595 if this is necessary
            for your use case.

        .. warning ::
            Gradient accumulation and DDP `no_sync` are currently not supported
            with overlapped optimizer. Please ping
            https://github.com/pytorch/pytorch/issues/71595 if this is necessary
            for your use case.

        Example::

            >>> # xdoctest: +SKIP("No rendezvous handler")
            >>> torch.distributed.init_process_group(backend='nccl', world_size=4, init_method='...')
            >>> net = torch.nn.parallel.DistributedDataParallel(model, pg)
            >>> lr = 1e-2
            >>> betas = (0.9, 0.99)
            >>> eps = 1e-6
            >>> net._register_fused_optim(torch.optim.Adam, lr, betas=betas, eps=eps)
            >>> # Example with subset of parameters
            >>> params_to_opt = [list(net.parameters())[0]]
            >>> net._register_fused_optim(
            ...   torch.optim.Adam, lr, optim_params=params_to_opt,  betas=betas, eps=eps
            ... )
        """
        # Note: importing in function, otherwise this will cause a circular
        # import as optimizer_overlap module needs to import DistributedDataParallel.
        from torch.distributed.algorithms._optimizer_overlap import _as_overlapped_optim

        overlapped_optim = _as_overlapped_optim(optim, optim_params, *args, **kwargs)
        try:
            overlapped_optim.register_ddp(self)
        except NotImplementedError as e:
            raise RuntimeError(
                f"{optim} does not support overlapped DDP. Please file an issue to PyTorch or the respective owner of {optim}."
            ) from e

    def _distributed_broadcast_coalesced(
        self, tensors, buffer_size, authoritative_rank=0
    ):
        dist._broadcast_coalesced(
            self.process_group, tensors, buffer_size, authoritative_rank
        )

    def _check_sync_bufs_post_fwd(self):
        return (
            self.will_sync_module_buffers()
            and hasattr(self, "buffer_hook")
            and self.buffer_hook.buffer_comm_hook_location
            == _BufferCommHookLocation.POST_FORWARD
        )

    def _check_sync_bufs_pre_fwd(self):
        return self.will_sync_module_buffers() and (
            not hasattr(self, "buffer_hook")
            or self.buffer_hook.buffer_comm_hook_location
            == _BufferCommHookLocation.PRE_FORWARD
        )

    def will_sync_module_buffers(self):
        return (
            self.require_forward_param_sync
            and self.broadcast_buffers
            and len(self.modules_buffers) > 0
        )

    def _find_common_rank(self, input_rank, rank_cond):
        # -1 indicates that this rank is not under consideration to be the
        # common_rank
        rank_to_use = torch.tensor(
            [input_rank if rank_cond else -1],
            device=self.device,
        )
        dist.all_reduce(rank_to_use, op=ReduceOp.MAX, group=self.process_group)
        if rank_to_use.item() == -1:
            self._log_and_throw(
                ValueError,
                "BUG! Expected rank_cond to be true for at least one process."
                " This indicates a bug in PyTorch, please report an issue.",
            )
        return rank_to_use.item()

    def _sync_buffers(self):
        with torch.no_grad():
            # module buffer sync
            # Synchronize buffers across processes.
            # The process with rank 0 is considered the authoritative copy.
            authoritative_rank = 0
            # Update self.modules_buffers incase any buffers were
            # reassigned.
            self._assign_modules_buffers()
            self._sync_module_buffers(authoritative_rank)

    def _sync_module_buffers(self, authoritative_rank):
        if not hasattr(self, "buffer_hook"):
            self._default_broadcast_coalesced(authoritative_rank=authoritative_rank)
        else:
            hook = self.buffer_hook.buffer_comm_hook
            state = self.buffer_hook.buffer_comm_hook_state
            futs = hook(state, self.named_module_buffers)
            if futs is not None:
                self.reducer._install_post_backward_futures(futs)

    def _default_broadcast_coalesced(
        self, bufs=None, bucket_size=None, authoritative_rank=0
    ):
        """
        Broadcasts buffers from rank 0 to rest of workers. If bufs, bucket_size
        are None, default values self.modules_buffers and
        self.broadcast_bucket_size are used instead.
        """
        if bufs is None:
            bufs = self.modules_buffers
        if bucket_size is None:
            bucket_size = self.broadcast_bucket_size

        self._distributed_broadcast_coalesced(bufs, bucket_size, authoritative_rank)

    def _passing_sync_batchnorm_handle(self, module):
        for layer in module.modules():
            if isinstance(layer, torch.nn.modules.SyncBatchNorm):
                if self.device_type == "cpu":
                    self._log_and_throw(
                        ValueError,
                        "SyncBatchNorm layers only work with GPU modules",
                    )

    def _check_comm_hook(self, hook):
        if not callable(hook):
            self._log_and_throw(TypeError, "Communication hook must be callable.")

        sig = inspect.signature(hook)
        if (
            sig.parameters["bucket"].annotation != inspect._empty
            and sig.parameters["bucket"].annotation != dist.GradBucket
        ):
            self._log_and_throw(
                ValueError,
                "Communication hook: bucket annotation should be dist.GradBucket.",
            )

        if (
            sig.return_annotation != inspect._empty
            and sig.return_annotation != torch.futures.Future[torch.Tensor]
        ):
            self._log_and_throw(
                ValueError,
                "Communication hook: return annotation should be torch.futures.Future[torch.Tensor].",
            )

        if hook.__name__ in ["bf16_compress_hook", "bf16_compress_wrapper_hook"] and (
            (torch.version.cuda is None and torch.version.hip is None)
            or (
                torch.version.cuda is not None
                and int(torch.version.cuda.split(".")[0]) < 11
            )
            or not dist.is_available()
            or not dist.is_nccl_available()
            or torch.cuda.nccl.version() < (2, 10)
        ):
            self._log_and_throw(
                TypeError,
                "BF16 all reduce communication hook required CUDA 11+ and NCCL 2.10+.",
            )

    @property
    def _distributed_rank(self):
        return dist.get_rank(self.process_group)

    @staticmethod
    def _set_params_and_buffers_to_ignore_for_model(
        module, params_and_buffers_to_ignore
    ):
        """
        Sets parameters and buffers to be ignored by DDP. Expected format for
        parameters is the fully qualified name: {module_name}.{param_name}, and
        similarly, {module_name}.{buffer_name} for buffers. For example:
        params_to_ignore = []
        # NB: model here is vanilla PyTorch module, not yet wrapped with DDP.
        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if should_ignore(param):
                    # Create expected format
                    fqn = f"{module_name}.{param_name}"
                    params_to_ignore.append(fqn)
        torch.nn.parallel.DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(
            model,
            params_to_ignore
        )
        """
        # This is a workaround to set parameters and buffers DDP should ignore
        # during synchronization. It will be removed when the API is finalized
        # as part of addressing https://github.com/pytorch/pytorch/issues/43690.
        module._ddp_params_and_buffers_to_ignore = params_and_buffers_to_ignore

    def _get_ddp_logging_data(self):
        r"""
        This interface can be called after DistributedDataParallel() is
        constructed. It returns a dictionary of logging data. It could help
        for debugging and analysis. The logging data includes DistributedDataParallel
        constructor input parameters, some internal states of DistributedDataParallel
        and performance metrics. Simply print the dictionary and see what
        these metrics are.
        This is a prototype interface and subject to change in the future.
        """
        assert self.logger is not None
        ddp_logging_data = self.logger._get_ddp_logging_data()
        return {**ddp_logging_data.strs_map, **ddp_logging_data.ints_map}

    def _set_ddp_runtime_logging_sample_rate(self, sample_rate):
        r"""
        This interface allows users to set sample_rate of collecting
        runtime stats. The runtime stats will be recorded for the
        first 10 iterations, after 10 iterations runtime stats will be
        recorded once every "sample_rate" training iterations. In
        default, runtime stats are recorded for the first 10 iterations,
        after 10 iterations runtime stats are recorded once every
        "kDDPRuntimeLoggingSampleRate=100" training iterations.
        This is a prototype interface and subject to change in the future.
        """
        if sample_rate < 1:
            self._log_and_throw(
                ValueError,
                "DDP runtime logging sample rate should be equal or greater than 1",
            )
        self.reducer._set_ddp_runtime_logging_sample_rate(sample_rate)

    def _set_static_graph(self):
        """
        It is recommended to set static graph in the DDP constructor, which will
        call this private API internally.
        """
        # If self.static_graph has been set, no need to set it again
        if self.static_graph:
            warnings.warn(
                "You've set static_graph to be True, no need to set it again."
            )
            return
        self.static_graph = True
        self.reducer._set_static_graph()
        assert self.logger is not None
        self.logger._set_static_graph()
        if self.find_unused_parameters:
            warnings.warn(
                "You passed find_unused_parameters=true to DistributedDataParallel, "
                "`_set_static_graph` will detect unused parameters automatically, so "
                "you do not need to set find_unused_parameters=true, just be sure these "
                "unused parameters will not change during training loop while calling "
                "`_set_static_graph`."
            )
