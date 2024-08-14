# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
import operator
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.fx as fx
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensor
from torch.distributed._composable.fsdp.fully_shard import FSDPModule
from torch.fx.node import map_aggregate
from torch.nn.parallel import DistributedDataParallel

from ._backward import stage_backward, stage_backward_input, stage_backward_weight
from ._debug import map_debug_info
from ._utils import flatten_args, PipeInfo, validate_tensors_metadata


__all__ = [
    "PipelineStage",
    "build_stage",
]

logger = logging.getLogger(__name__)


class _RootArgPlaceholder:
    """
    Placeholder for model-level inputs.
    """

    def __init__(self, tensor):
        self.meta = tensor.to("meta")


class _RecvInfo:
    """
    Represents a stage input.
    """

    def __init__(
        self,
        input_name: str,
        source: int,
        buffer: torch.Tensor,
    ):
        # Name of this input
        self.input_name = input_name
        # Stage index of the source of this input
        self.source = source
        # Buffer to receive the input into.
        self.buffer = buffer

    def __repr__(self):
        return f"_RecvInfo(input={self.input_name}, source={self.source}, shape={self.buffer.size()})"


# An input can be either a received activation or a model input
InputInfo = Union[_RecvInfo, _RootArgPlaceholder]


def _make_tensor_from_meta(
    example: Union[torch.Tensor, FakeTensor],
    device: torch.device,
) -> torch.Tensor:
    """
    Create a real tensor from a tensor.
    """
    return torch.empty(
        example.size(),
        dtype=example.dtype,
        layout=example.layout,
        device=device,
    )


class _PipelineStageBase(ABC):
    """
    Base class for pipeline stages.
    Defines or implements common methods used by the `_PipelineStage` used by
    the tracing frontend and `PipelineStage` used by manual frontend.
    """

    def __init__(
        self,
        submodule: torch.nn.Module,
        stage_index: int,
        num_stages: int,
        device: torch.device,
        group: Optional[dist.ProcessGroup] = None,
        dw_builder: Optional[Callable[[], Callable[..., None]]] = None,
    ):
        """
        Args:
            submodule (torch.nn.Module): The module to be executed in this stage.
            stage_index (int): The index of this stage.
            num_stages (int): The total number of stages in this pipeline.
            device (torch.device): The device to run this stage on.
            group (Optional[dist.ProcessGroup]): The process group to use for communication.
                If `None`, the default process group will be used.
                Default: `None`.
            dw_builder (Optional[Callable[[], Callable[..., None]]): If provided, dw_runner is a builder function
                that will build a new dw_runner function that will run parts of module backward that were intentionally
                skipped during the module's actual backward pass.  builder must be invoked by stage after stage runs
                model backwards, and stage should save the latest dw_runner to run during weight pass.
                When used with schedules that only have F and B steps, the fresh dw_runner function will be called as
                part of B.
                When used with F,B,W schedules, the dw_runner function implements 'W'.
        """
        super().__init__()
        if stage_index >= num_stages:
            raise ValueError(
                f"Stage index {stage_index} is out of range of {num_stages}"
            )

        self.submod = submodule
        self.stage_index = stage_index
        self.num_stages = num_stages
        self.device = device
        self.group = group

        self.dw_builder = dw_builder

        # backward state
        self.backward_state: Dict[int, Tuple[Any, ...]] = {}

        # store dw_runner per microbatch_id
        self.dw_runner: Dict[int, Callable[..., None]] = {}

        # `group_rank` is rank in process group `group`.
        self.group_rank = dist.get_rank(self.group)
        self.group_size = dist.get_world_size(self.group)
        if self.group_size > self.num_stages:
            raise RuntimeError(
                f"Pipeline group size {self.group_size} cannot be larger than number of stages {self.num_stages}"
            )

        # Run time states
        self._outputs_meta: Optional[Tuple[torch.Tensor, ...]] = None
        # map microbatch ID to list of forward tensor args
        self.fwd_cache: Dict[int, Tuple[Any, List[torch.Tensor]]] = {}
        # Caching chunk outputs for final output merge or reduction
        self.output_chunks: List[Any] = []

        # Initialize has_backward to false; this will be set to true if loss
        # function is passed to pipeline schedule
        self.has_backward = False
        # Log prefix
        self.log_prefix = f"[Stage {self.stage_index}]"

        # Forward infra
        self.args_recv_info: Dict[int, Tuple[InputInfo, ...]] = {}
        self.set_requires_grad: Dict[int, bool] = {}
        self.act_send_info: Dict[int, List] = {}

        # Backward infra will created lazily
        self.grad_recv_info: Dict = {}
        self.grad_send_info: Optional[List] = None

        # Number of backward chunks seen. This is used to determine when to do
        # grad reduction in DDP or FSDP.
        self._seen_bwd_chunks = 0

        # To be populated later by the Schedule
        self.chunks: Optional[int] = None
        self.stage_index_to_group_rank: Dict[int, int] = {
            i: i % self.group_size for i in range(self.num_stages)
        }

    @property
    def has_backward(self) -> bool:
        """
        Returns true if this stage has a backward pass.
        """
        return self._has_backward

    @has_backward.setter
    def has_backward(self, has_backward: bool):
        self._has_backward = has_backward

    @property
    def is_first(self):
        """
        Returns true if this stage is the first stage in the pipeline.
        """
        return self.stage_index == 0

    @property
    def is_last(self):
        """
        Returns true if this stage is the last stage in the pipeline.
        """
        return self.stage_index == self.num_stages - 1

    def _check_chunk_id(self, chunk_id: int):
        if self.chunks is None:
            raise RuntimeError(
                "Attempted to access chunk_id before chunks have been configured."
            )
        if chunk_id >= self.chunks:
            raise RuntimeError(
                f"Chunk id {chunk_id} is out of range [0, {self.chunks})"
            )

    def _configure_outputs_meta(self, outputs_meta: Tuple[torch.Tensor, ...]):
        """
        Track the output shapes/dtype of this stage since they determine the send operation(s) which must match
        recv operations of the next stage.  The next stage _will_ be freezing its recv buffers based on its initial
        configuration, so it's important to also freeze/validate the output side to avoid any send/recv mismatches
        which could show up as hangs, silent corruption, or other errors.
        """
        assert (
            self._outputs_meta is None
        ), "Attempting to reconfigure output_meta, which is not supported"
        self._outputs_meta = tuple(outputs_meta)  # type: ignore[assignment]

    def get_outputs_meta(self) -> Tuple[torch.Tensor, ...]:
        """Get the output metadata (meta tensors) reprensenting the outputs of this stage"""
        assert (
            self._outputs_meta is not None
        ), "Attempted to get_outputs_meta() without configuring output meta"
        return self._outputs_meta

    def _create_grad_send_info(
        self,
        args_recv_info: Tuple,
    ) -> List[Optional[int]]:
        """
        Create a list of stage indices to send gradients to.
        """
        grad_send_info: List[Optional[int]] = []

        def map_recv_to_send(a):
            # Note: we send gradients back to previous stage as long as in
            # forward it is a received input, regardless of whether it requires
            # grad. It is up to the previous stage to disgard this gradient.
            if isinstance(a, _RecvInfo):
                grad_send_info.append(a.source)
                return a.source
            else:
                grad_send_info.append(None)
                return None

        map_aggregate(args_recv_info, map_recv_to_send)

        logger.debug("%s Grad send info: %s", self.log_prefix, grad_send_info)
        return grad_send_info

    @abstractmethod
    def _prepare_forward_infra(self, num_microbatches: int):
        raise NotImplementedError

    def _prepare_backward_infra(self, num_microbatches: int):
        # TODO: this is needed for backward_maybe_with_nosync
        self.chunks = num_microbatches

        for mb_index in range(num_microbatches):
            # `grad_recv_info` is a mirror of `act_send_info`
            self.grad_recv_info[mb_index] = self._create_grad_recv_info(
                self.act_send_info
            )

    @abstractmethod
    def _create_grad_recv_info(
        self,
        act_send_info: Dict,
    ) -> Tuple[_RecvInfo, ...]:
        raise NotImplementedError

    def _get_recv_ops(
        self,
        recv_infos: Tuple[InputInfo, ...],
    ) -> List[dist.P2POp]:
        """
        Helper function shared by `get_fwd_recv_ops` and `get_bwd_recv_ops`.
        Returns a list of ops that correspond to the recv infos.
        """
        ops: List[dist.P2POp] = []
        for info in recv_infos:
            if not isinstance(info, _RecvInfo):
                continue

            peer_rank = self.stage_index_to_group_rank[info.source]
            peer_global_rank = (
                peer_rank
                if self.group is None
                else dist.get_global_rank(self.group, peer_rank)
            )  # TODO
            ops.append(
                dist.P2POp(dist.irecv, info.buffer, peer_global_rank, self.group)
            )

        return ops

    def get_fwd_recv_ops(self, fwd_chunk_id: int) -> List[dist.P2POp]:
        """
        Returns a list of ops that are needed to receive the input arguments
        for this stage.
        """
        recv_infos: Tuple[InputInfo, ...] = self.args_recv_info[fwd_chunk_id]

        # In case there is backward pass, set requires_grad for receive buffers
        # before first forward
        if self.has_backward and not self.set_requires_grad[fwd_chunk_id]:
            for a in recv_infos:
                if isinstance(a, _RecvInfo):
                    a.buffer.requires_grad_(True)

        return self._get_recv_ops(recv_infos)

    def get_bwd_recv_ops(self, bwd_chunk_id: int) -> List[dist.P2POp]:
        """
        Returns a list of ops that are needed to receive the gradients
        for this stage.
        """
        if not self.has_backward or self.is_last:
            return []

        recv_infos = self.grad_recv_info[bwd_chunk_id]
        return self._get_recv_ops(recv_infos)

    def get_fwd_send_ops(self, fwd_chunk_id: int) -> List[dist.P2POp]:
        """
        Get the activation send ops for current stage's forward.
        """
        output = self.output_chunks[fwd_chunk_id]
        # Unify output form to tuple for easy correspondance with
        # `act_send_info`
        output_tuple = output if type(output) is tuple else (output,)

        ops: List[dist.P2POp] = []

        for idx, out in enumerate(output_tuple):
            dst_stages = self.act_send_info[idx]
            for dst in dst_stages:
                if dst is None:
                    continue
                logger.debug(
                    "%s Sending tensor to Stage %s: %s",
                    self.log_prefix,
                    dst,
                    out.size(),
                )
                peer_rank = self.stage_index_to_group_rank[dst]
                peer_global_rank = (
                    peer_rank
                    if self.group is None
                    else dist.get_global_rank(self.group, peer_rank)
                )  # TODO
                ops.append(dist.P2POp(dist.isend, out, peer_global_rank, self.group))

        return ops

    def get_bwd_send_ops(self, bwd_chunk_id: int) -> List[dist.P2POp]:
        """
        Get the gradient send ops for current stage's backward.
        """
        self._check_chunk_id(bwd_chunk_id)

        if not self.has_backward or self.is_first:
            return []

        # Create bwd send infra lazily
        if self.grad_send_info is None:
            # Send info for input grads during backward:
            # List of destinations corresponding to input grads
            # Can be None if an input has no grad
            # `grad_send_info` is a mirror of `args_recv_info`
            self.grad_send_info = self._create_grad_send_info(self.args_recv_info[0])

        ops: List[dist.P2POp] = []
        for grad, grad_recv_stage in zip(self.grads_input, self.grad_send_info):
            if isinstance(grad, torch.Tensor) and grad_recv_stage is not None:
                logger.debug(
                    "%s Sending gradient to Stage %s: %s",
                    self.log_prefix,
                    grad_recv_stage,
                    grad.size(),
                )
                peer_rank = self.stage_index_to_group_rank[grad_recv_stage]
                peer_global_rank = (
                    peer_rank
                    if self.group is None
                    else dist.get_global_rank(self.group, peer_rank)
                )  # TODO
                ops.append(dist.P2POp(dist.isend, grad, peer_global_rank, self.group))
            else:
                if not (grad is None and grad_recv_stage is None):
                    raise RuntimeError(
                        f"[{self.stage_index}] for chunk {bwd_chunk_id} has gradients {grad} "
                        f"and is expecting to send gradients to stage {grad_recv_stage}"
                    )
        return ops

    def clear_runtime_states(self) -> None:
        """
        Clear runtime states of the stage.
        """
        # map microbatch ID to list of forward tensor args
        self.fwd_cache.clear()
        # Caching chunk outputs for final output merge or reduction
        self.output_chunks.clear()
        # Reset bwd chunk counter
        self._seen_bwd_chunks = 0

        # Clear grad of input buffers in between schedule steps. This is because
        # `torch.autograd.backward()` will accumulate gradients into leaf
        # tensors by default. For gradients to pass back to previous stages, we
        # don't want such accumulation.
        for recv_tuple in self.args_recv_info.values():  # iterate over all chunks
            for a in recv_tuple:  # iterate over all input args
                if isinstance(a, _RecvInfo):
                    # Set to None is the newer and recommended way to clear grads, compared to `zero_()`.
                    # See https://github.com/pytorch/pytorch/pull/92731
                    a.buffer.grad = None

    def _map_tensor_from_recv_info(
        self,
        recv_infos: Tuple[InputInfo, ...],
    ):
        """
        Map tensors from recv infos to a list.
        """

        def get_recv_tensor(info):
            if isinstance(info, _RecvInfo):
                return info.buffer
            else:
                raise AssertionError(f"Expected _RecvInfo but got {type(info)}")

        tensors = map_aggregate(
            recv_infos,
            get_recv_tensor,
        )

        return tensors

    def _retrieve_recv_activations(self, fwd_chunk_id: int):
        """
        Retrieve the activations received for the current stage during forward.
        """
        recv_infos = self.args_recv_info[fwd_chunk_id]
        activations = self._map_tensor_from_recv_info(recv_infos)
        return activations

    def _retrieve_recv_grads(
        self,
        bwd_chunk_id: int,
    ):
        """
        Retrieve the gradients received for the current stage during backward.
        """
        recv_infos = self.grad_recv_info[bwd_chunk_id]
        grads = self._map_tensor_from_recv_info(recv_infos)
        return grads

    def forward_maybe_with_nosync(self, *args, **kwargs):
        # If submod is wrapped with DDP, we use the `no_sync` context manager to
        # avoid gradient all-reduce per microbatch
        if isinstance(self.submod, DistributedDataParallel):
            with self.submod.no_sync():  # type: ignore[operator]
                out_val = self.submod(*args, **kwargs)
        else:
            out_val = self.submod(*args, **kwargs)
        return out_val

    def backward_maybe_with_nosync(self, bwd_kwargs: Dict):
        """
        Whether using PP with FSDP or DDP, there are some runtime differences between the last backward step and the
        other steps.  Namely, we need to accumulate gradients on previous steps and reduce them on the last step, but
        there are additional state-variables and performance considerations depending on the data parallelism used.
        This helper should adapt any pipeline parallel schedule to work with common/supported data parallel libraries.
        """
        last_backward = self._seen_bwd_chunks == self.chunks - 1  # type: ignore[operator]

        # If submod is wrapped by DDP
        if isinstance(self.submod, DistributedDataParallel):
            if last_backward:
                # Last chunk, prepare for gradient reduction
                # HACK: reaching into DDP implementation details here. Is there a better way?
                self.submod.reducer.prepare_for_backward(  # type: ignore[union-attr, operator]
                    list(
                        torch.nn.parallel.distributed._find_tensors(  # type: ignore[attr-defined]
                            bwd_kwargs["stage_output"]
                        )
                    )
                )
                grads_input = stage_backward(**bwd_kwargs)
            else:
                with self.submod.no_sync():  # type: ignore[operator]
                    grads_input = stage_backward(**bwd_kwargs)
        # If submod is a FSDP module
        elif isinstance(self.submod, FSDPModule):
            self.submod.set_is_last_backward(last_backward)
            self.submod.set_requires_gradient_sync(last_backward)
            grads_input = stage_backward(**bwd_kwargs)
        else:
            # Non-DP submodule, regular backward
            grads_input = stage_backward(**bwd_kwargs)

        self._seen_bwd_chunks += 1
        return grads_input

    def forward_one_chunk(
        self,
        fwd_chunk_id: int,
        args: Tuple[Any, ...],
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Perform forward pass on the stage with one microbatch.
        `args` and `kwargs` are the inputs from *external* to this stage. They
        applies only to the first stage in most cases.
        """

        if self.is_first:
            # First stage doesn't need to receive anything
            composite_args = args
            composite_kwargs = kwargs or {}
        else:
            # Receive activations for this chunk
            # Activations only come in args form
            composite_args = self._retrieve_recv_activations(fwd_chunk_id)
            composite_kwargs = {}

        self._validate_fwd_input(args, kwargs)

        # Compute forward
        try:
            output = self.forward_maybe_with_nosync(*composite_args, **composite_kwargs)

        except Exception as e:
            exc_msg = f"""
            {self.log_prefix} failed to run forward:
            args: {map_debug_info(composite_args)}
            kwargs: {map_debug_info(composite_kwargs)}
            """
            raise RuntimeError(exc_msg) from e

        if type(output) is list:
            # HACK: this is a hacky workaround for the fact that export creates
            # output in list format
            output = tuple(output)

        # Unify output form to tuple for easy correspondance with
        # `act_send_info`
        output_tuple = output if type(output) is tuple else (output,)
        # Prepare for final output merge or reduction
        self.output_chunks.append(output)

        # Save activations and inputs for backward
        flat_args = flatten_args(composite_args)
        flat_kwargs = flatten_args(composite_kwargs)
        flatten_input_tensors = flat_args + flat_kwargs
        self.fwd_cache[fwd_chunk_id] = (
            output_tuple,  # stage_output
            flatten_input_tensors,  # input_values
        )

        logger.debug(
            "%s Forwarded chunk %s, outputs: %s",
            self.log_prefix,
            fwd_chunk_id,
            map_debug_info(output),
        )
        self._validate_fwd_outputs(output_tuple)
        return output

    def backward_one_chunk(
        self, bwd_chunk_id: int, loss=None, full_backward: bool = True
    ):
        """
        Perform backward pass on the module.
        This should only be called once per microbatch.

        If full_backward is True (the default), the full backward pass including weight and input gradients will be run,
        and it is an error to call `backward_weight_one_chunk` for this bwd_chunk_id.

        If full_backward is False, it is optional that `dw_runner` was provided to the PipelineStage at __init__ time,
        and a subsequent call to `backward_weight_one_chunk` is required to invoke dw_runner and complete the backward.
        """
        self._check_chunk_id(bwd_chunk_id)

        (
            stage_output,
            input_values,
        ) = self.fwd_cache.pop(bwd_chunk_id)

        # Compute backward
        if self.is_last:
            # Last stage computes gradients from loss and has no gradients from
            # next stage
            bwd_kwargs = {
                "stage_output": loss,
                "output_grads": None,
                "input_values": input_values,
            }
        else:
            # Otherwise, receive gradients from next stage
            grads_output = self._retrieve_recv_grads(bwd_chunk_id)
            # If an input to the pipeline requires gradient,
            # `torch.autograd.backward` will accumulate the gradient into the
            # `.grad` field of such input
            bwd_kwargs = {
                "stage_output": stage_output,
                "output_grads": grads_output,
                "input_values": input_values,
            }

        # Custom backward function
        if self.dw_builder:
            self.grads_input = self.backward_maybe_with_nosync(bwd_kwargs)
            if full_backward:
                self.dw_builder()()
            else:
                self.dw_runner[bwd_chunk_id] = self.dw_builder()
        else:
            if full_backward:
                self.grads_input = self.backward_maybe_with_nosync(bwd_kwargs)
            else:
                # perform the partial backwards for the inputs with a custom backward function
                # when the "stage_ouput" is a loss, then it is a tensor, otherwise it is a tuple of tensors
                if isinstance(bwd_kwargs["stage_output"], torch.Tensor):
                    bwd_kwargs["stage_output"] = (bwd_kwargs["stage_output"],)

                dinputs, param_groups = stage_backward_input(
                    bwd_kwargs["stage_output"],
                    bwd_kwargs["output_grads"],
                    bwd_kwargs["input_values"],
                    self.submod.parameters(),
                )
                # TODO: we dont need to save this, add to dw_runner?
                self.backward_state[bwd_chunk_id] = (
                    dinputs,
                    input_values,
                    param_groups,
                    bwd_kwargs["stage_output"],
                    bwd_kwargs["output_grads"],
                )
                self.grads_input = dinputs
                # save a dw_runner with the `stage_backward_weight` function
                self.dw_runner[bwd_chunk_id] = stage_backward_weight
        logger.debug("%s Backwarded chunk %s", self.log_prefix, bwd_chunk_id)

    def backward_weight_one_chunk(self, bwd_chunk_id: int):
        if self.dw_builder is not None:
            assert bwd_chunk_id in self.dw_runner, (
                f"{self.log_prefix} Attempted to run backward_weight_one_chunk for chunk {bwd_chunk_id}"
                " without first calling `backward_one_chunk(full_backward=False)`"
            )
            self.dw_runner.pop(bwd_chunk_id)()
        else:
            (
                dinputs,
                input_values,
                param_groups,
                stage_output,
                output_grads,
            ) = self.backward_state.pop(bwd_chunk_id)
            if self.stage_index != 0:
                dweights = self.dw_runner.pop(bwd_chunk_id)(
                    self.submod.parameters(), param_groups
                )
            else:
                # TODO: figure out a better way to do this:
                # if inputs does not require gradient,
                # then the parameter group will not be fully captured during stage_backward_weight
                # in this case, we need call grad directly on the parameters
                torch.autograd.backward(stage_output, grad_tensors=output_grads)

    def _validate_fwd_input(self, args, kwargs):
        """Raises a RuntimeError if shapes of input args/kwargs do not match the shapes configured for this stage."""

        if self.is_first:
            # TODO why is there a separate recv_info for each pipeline chunk?
            # kwen2501: to avoid passing a `fwd_chunk_id` to this function, we
            # check all chunks against args_recv_info[0]
            expected_args = self.args_recv_info[0]
        else:
            # We don't check inputs for non-0 stages assuming they don't accept
            # user inputs in canonical pipeline scenarios
            return

        if len(kwargs):
            # TODO- need a mapping of kwarg to position in self.args_recv_info
            # without it, we just validate shapes for args and ignore kwargs
            expected_args = expected_args[: len(expected_args) - len(kwargs)]

        # TODO- need a mapping of kwarg to position in self.args_recv_info
        # maybe it's impossible to tell whether the len mismatches because
        # (a) the user passed an extra arg or missed an arg
        # (b) the user did not pass a kwarg, which has a default value baked into expected_args
        expected_tensors_meta = [
            e.meta if isinstance(e, _RootArgPlaceholder) else e.buffer
            for e in expected_args
        ]
        validate_tensors_metadata(
            f"Stage {self.stage_index} forward inputs", expected_tensors_meta, args
        )

    def _validate_fwd_outputs(self, outputs: Tuple[torch.Tensor, ...]):
        """Raises a RuntimeError if this stage produces an output of unexpected shape/dtype.
        Most likely, this could be cause either by incorrect user specification of output shapes, or becuase
        shape inference was done on the original model but then at runtime the model is wrapped with something like
        mixed precision which changes output dtype.
        """
        expected_tensors_meta = self.get_outputs_meta()
        validate_tensors_metadata(
            f"Stage {self.stage_index} forward outputs", expected_tensors_meta, outputs
        )


class _PipelineStage(_PipelineStageBase):
    def __init__(
        self,
        stage_module: torch.nn.Module,
        stage_index: int,
        pipe_info: PipeInfo,
        device: torch.device,
        group: Optional[dist.ProcessGroup] = None,
    ):
        """
        Create a pipeline stage given a stage_module to be wrapped by this stage
        and a `pipe_info` describing the stage relationship of the pipeline.

        Args:
            stage_module (torch.nn.Module): the module to be wrapped by this stage
            stage_index (int): the index of this stage in the pipeline
            pipe_info (PipeInfo): information about the pipeline, can be retrieved by `pipe.info()`
            device (torch.device): the device to be used by this stage
            group (Optional[dist.ProcessGroup]): the process group to be used by this stage
        """
        _PipelineStageBase.__init__(
            self,
            stage_module,
            stage_index,
            pipe_info.num_stages,
            device,
            group,
        )
        self.pipe_info = pipe_info

        # Find stage nodes in graph
        submod_nodes = [
            node for node in pipe_info.graph.nodes if node.op == "call_module"
        ]
        if len(submod_nodes) != self.num_stages:
            raise AssertionError(
                f"Number of submodules in pipe graph {len(submod_nodes)} does not match number of stages {self.num_stages}"
            )

        # Find my stage node in graph
        self.node = submod_nodes[self.stage_index]
        self.name = self.node.name
        logger.info(
            "[%s] Creating PipelineStage %s for %s",
            self.group_rank,
            stage_index,
            self.name,
        )

        # Create mapping from stage name to stage index
        self.submod_to_stage_index: Dict[str, int] = {}
        for i, node in enumerate(submod_nodes):
            self.submod_to_stage_index.setdefault(node.name, i)

        # Cast submodule to device
        self._move_submod_to_device()

    def _move_submod_to_device(self):
        # Move submodule to indicated device if possible
        # Note: we cannot move meta module to real devices because meta tensors
        # do not support to() method. One needs to do an in-place tensor swap in
        # that case.
        has_meta_param = any(
            isinstance(p, FakeTensor) or p.is_meta for p in self.submod.parameters()
        )
        if has_meta_param:
            logger.debug("%s Found meta parameters!", self.log_prefix)
        else:
            self.submod.to(self.device)

    def _prepare_forward_infra(self, num_microbatches: int):
        """
        Create send/recv infrastructures for activations (during forward)
        """
        # Flag per chunk to keep track of whether we have set `requires_grad`
        # for receive buffers. Format: {chunk : Boolean}
        for chunk in range(num_microbatches):
            self.args_recv_info[chunk] = self._create_act_recv_info()
            self.set_requires_grad[chunk] = False

        # Send info during forward for each activation
        self.act_send_info = self._create_act_send_info()

    def get_stage_index_of_submod(
        self,
        submod_name: str,
    ):
        """
        Given a submodule name, return the stage index of the submodule.
        """
        if submod_name not in self.submod_to_stage_index:
            raise AssertionError(f"Stage id of {submod_name} not found")

        return self.submod_to_stage_index[submod_name]

    def _create_act_recv_info(
        self,
    ):
        """
        Create a tuple of `_RecvInfo` for inputs to the stage.
        """

        def create_recv_tensor(placeholder, arg_node):
            """
            Create a receive buffer for a placeholder.
            """
            example_value = placeholder.meta["val"]
            if arg_node.op == "placeholder":
                # This is a root level placeholder, thus an input argument to the entire model.
                # We are likely at stage 0, hence no need to create a receive buffer.
                return _RootArgPlaceholder(example_value)

            # Figure out the source stage of this input
            while arg_node.target is operator.getitem:
                # If the input is a getitem, we need to go deeper
                arg_node = arg_node.args[0]

            assert (
                arg_node.op == "call_module"
            ), f"Expecting call_module, got {arg_node.op}"
            src_stage = self.get_stage_index_of_submod(arg_node.name)

            # Create a receive buffer for this placeholder
            logger.debug(
                "%s Creating recv buffer for input '%s' : %s, %s",
                self.log_prefix,
                placeholder.name,
                example_value.shape,
                example_value.dtype,
            )
            buffer = _make_tensor_from_meta(example_value, self.device)

            return _RecvInfo(
                arg_node.name,
                src_stage,
                buffer,
            )

        args_recv_info: List[InputInfo] = []
        # Filter out placeholder nodes from `self.submod` (a GraphModule)
        placeholders = filter(
            lambda node: node.op == "placeholder", self.submod.graph.nodes
        )
        # `placeholders` are nodes internal to submod.
        # `self.node.args` are dependency nodes in the outer graph.
        # The two are 1:1.
        for placeholder, arg_node in zip(placeholders, self.node.args):
            # Create a receive buffer for this placeholder
            recv_info = create_recv_tensor(placeholder, arg_node)
            args_recv_info.append(recv_info)

        logger.debug(
            "%s Activation recv / args info: %s", self.log_prefix, args_recv_info
        )
        # `args` is a Tuple, hence we will return a Tuple[InputInfo]
        return tuple(args_recv_info)

    def find_dst_rank(
        self,
        user: fx.Node,
    ) -> Optional[int]:
        """
        Find the destination rank of a `user` node.
        If the `user` is not a submod, `None` may be returned.
        """
        if user.op == "call_module":
            # User is a stage (`call_module`)
            return self.get_stage_index_of_submod(user.name)
        else:
            # - If user.op == "output":
            #   No need to send back to rank 0
            # - If user.target is stage_backward:
            #   No need to send assuming submod output is stored locally or
            #   should be re-calucated in case of activation checkpointing
            return None

    def _create_act_send_info(self):
        """
        Create a dict of send info for activations.
        The dict is of the form:
        {
            output_index: [dst_rank_0, dst_rank_1, ...],
            ...
        }
        where the list of `dst_rank`s covers the case where an output value may
        be consumed by multiple stages.
        """
        # Output index: List of receiver ranks
        act_send_info: Dict[int, List] = {}
        out_idx = 0

        for user in self.node.users:
            if user.target is operator.getitem:
                # Recursively find the real destination
                gi_dsts = act_send_info.setdefault(out_idx, [])
                for gi_user in user.users:
                    dst_rank = self.find_dst_rank(gi_user)
                    if dst_rank is not None:
                        gi_dsts.append(dst_rank)
                # Next `getitem` will point to the next output index
                out_idx += 1
            else:
                # In case of single output value, `out_idx` will not increase
                dsts = act_send_info.setdefault(out_idx, [])
                dst_rank = self.find_dst_rank(user)
                if dst_rank is not None:
                    dsts.append(dst_rank)

        output_node = self._get_output_node()
        output_vals: Tuple[torch.Tensor] = tuple(
            v.meta["val"] for v in flatten_args(output_node.args)
        )
        self._configure_outputs_meta(output_vals)

        logger.debug("%s Send info: %s", self.log_prefix, act_send_info)
        return act_send_info

    def _get_output_node(self):
        output_nodes = [node for node in self.submod.graph.nodes if node.op == "output"]
        assert len(output_nodes) == 1
        output_node = output_nodes[0]
        return output_node

    def _create_grad_recv_info(
        self,
        act_send_info: Dict,
    ) -> Tuple[_RecvInfo, ...]:
        """
        Create a tuple of `_RecvInfo` for gradients.
        """
        # Dict[output_index, _RecvInfo]
        grad_recv_info: Dict[int, _RecvInfo] = {}
        output_node = self._get_output_node()

        # The output node may take multiple args, meaning the submod having multiple output values.
        output_vals = flatten_args(output_node.args)

        for out_idx, dst_list in act_send_info.items():
            if not dst_list:
                # No actual receiver for activation so no grad coming back
                continue

            output = output_vals[out_idx]
            example_value = output.meta["val"]
            logger.debug(
                f"{self.log_prefix} Creating grad recv buffer for output {output.name} "  # noqa: G004
                f": {example_value.shape}, {example_value.dtype}"
            )

            # TODO: otherwise needs grad accumulation
            assert len(dst_list) == 1, "Backward of skip connections not supported yet"
            grad_src = dst_list[0]
            grad_recv_info[out_idx] = _RecvInfo(
                f"{grad_src}",  # noqa: G004
                grad_src,
                _make_tensor_from_meta(example_value, self.device),
            )

        # Convert to tuple for convenience in get_ops and retrieve tensor
        grad_recv_info_tuple = tuple(grad_recv_info.values())
        logger.debug("%s Grad recv info: %s", self.log_prefix, grad_recv_info_tuple)
        return grad_recv_info_tuple


# A helper function to create a pipeline stage based on traced pipeline information
def build_stage(
    stage_module: torch.nn.Module,
    stage_index: int,
    pipe_info: PipeInfo,
    device: torch.device,
    group: Optional[dist.ProcessGroup] = None,
) -> _PipelineStage:
    """
    Create a pipeline stage given a stage_module to be wrapped by this stage
    and pipeline information.

    Args:
        stage_module (torch.nn.Module): the module to be wrapped by this stage
        stage_index (int): the index of this stage in the pipeline
        pipe_info (PipeInfo): information about the pipeline, can be retrieved by `pipe.info()`
        device (torch.device): the device to be used by this stage
        group (Optional[dist.ProcessGroup]): the process group to be used by this stage

    Returns:
        _PipelineStage: a pipeline stage that can run with `PipelineSchedules`.
    """
    return _PipelineStage(
        stage_module,
        stage_index,
        pipe_info,
        device,
        group,
    )


# Manual PipelineStage functions and definition

METADATA_TENSOR_LEN = 100
PLACEHOLDER_VAL = -1


def _create_empty_tensors(
    tensor: Union[torch.Tensor, Iterable[torch.Tensor]], device: torch.device
) -> List[torch.Tensor]:
    """
    Creates a list of empty tensors with the same properties (like shape and dtype) as the input tensor(s),
    and places them on the specified device.
    Args:
        tensor (Union[torch.Tensor, List[torch.tensor]]): The input tensor(s).
        device (torch.device): The device where the new tensors will be placed.
    Returns:
        List[torch.Tensor]: A list of empty tensors with the same properties as the input tensor(s).
    """
    if isinstance(tensor, torch.Tensor):
        return [torch.empty_like(tensor, device=device)]
    elif isinstance(tensor, (list, tuple)):
        return [torch.empty_like(t, device=device) for t in tensor]
    raise TypeError(f"Unsupported type {type(tensor)} cannot create empty tensors")


def _create_metadata_tensor(
    tensors: Optional[List[torch.Tensor]] = None,
    device: Optional[torch.device] = torch.device("cpu"),
) -> torch.Tensor:
    """
    Create a metadata tensor that can be sent over the wire.
    This tensor contains the number of dimensions and the shape of each tensor being sent.

    The data is of format [num_dims, dim1, dim2, ...].
    If the tensor is None, a tensor of only placeholder values will be returned.

    Inputs:
        tensors: A list of tensors, the tensors will converted into its shape dimensions and
                 these dimensions will be concatenated.
        device: The device where the metadata tensor will be created.
    If the tensor is None, then this tensor will contain PLACEHOLDER_VALs.

    """
    metadata_tensor = torch.full(
        (METADATA_TENSOR_LEN,),
        PLACEHOLDER_VAL,
        dtype=torch.int32,
        device=device,
    )
    if tensors:
        # Create a list of tensors containing the number of dimensions and the shape of each tensor
        data = [
            # data is of format [num_dims, dim1, dim2, ...]
            torch.tensor(
                [len(tensor.shape)] + list(tensor.shape),
                dtype=torch.int32,
                device=device,
            )
            for tensor in tensors
        ]
        # Concatenate the data into a single tensor
        data_tensor = torch.cat(data)
        dt_shape = data_tensor.shape[0]
        if dt_shape > METADATA_TENSOR_LEN:
            raise ValueError(
                f"Metadata tensor size ({dt_shape}) exceeds maximum allowed length ({METADATA_TENSOR_LEN})."
            )
        metadata_tensor[:dt_shape] = data_tensor
    return metadata_tensor


def _extract_metadata_from_tensor(tensor: torch.Tensor) -> List[torch.Size]:
    """
    Extract the number of dimensions and the shape of each tensor from a metadata tensor.
    """
    metadata: List[torch.Size] = []
    i = 0
    while i < len(tensor) and tensor[i] != PLACEHOLDER_VAL:
        num_dims = int(tensor[i].item())
        shape = torch.Size(tensor[i + 1 : i + 1 + num_dims].tolist())
        metadata.append(shape)
        i += num_dims + 1
    return metadata


def _get_stage_shapes(
    stage_modules: List[nn.Module],
    stage_ids: List[int],
    num_stages: int,
    rank: int,
    world_size: int,
    device: torch.device,
    microbatch: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
):
    """
    Performs a dry run through all the pipeline stages (a rank can have multiple pipeline stages in the case of
    virtual pipelining) and returns the shape of the inputs and outputs of the module.
    Only the first stage must pass in a microbatch.

    Each rank must call _get_stage_shapes or the program will hang.

    Args:
        stage_modules: The chunks assigned to this rank. Rhe length should be 1 for any
                non-interleaved schedules and >1 for any interleaved schedules.
        stage_ids: The id of the stages assigned to this rank.
        num_stages: Total number of stages.
        rank: Rank of the current process.
        world_size: Number of processes participating in the pipeline.
        device: Device where the tensors are allocated.

    Returns a dictionary containing the following keys:
        "inputs": Shape of the inputs to the module
        "outputs": Shape of the outputs of the module
    """

    stage_id_to_shapes: Dict[int, Dict[str, list[torch.Size]]] = {}
    for stage_id, model in zip(stage_ids, stage_modules):
        input_shape_metadata_tensor = _create_metadata_tensor(device=device)
        # TODO: Assumes prev_stage == rank - 1 and next_stage == rank + 1
        prev_rank = (rank - 1) % world_size
        next_rank = (rank + 1) % world_size
        shapes = {}

        # first stage doesn't receive anything and uses a microbatch
        if stage_id == 0:
            if microbatch is None:
                raise RuntimeError("Microbatch is required for first stage")
            example_fwd_inputs = microbatch
            if isinstance(example_fwd_inputs, torch.Tensor):
                example_fwd_inputs = [example_fwd_inputs]
        else:
            # other stages must receive shape information
            # TODO: send/recv should take a group, rather than use the default group
            dist.recv(input_shape_metadata_tensor, prev_rank)
            metadata = _extract_metadata_from_tensor(input_shape_metadata_tensor)
            example_fwd_inputs = [
                torch.empty(shape_list, device=device) for shape_list in metadata
            ]
        shapes["inputs"] = [fwd_input.shape for fwd_input in example_fwd_inputs]

        # perform forward
        # TODO: if forward fails raise a more descriptive error explaining which stage failed
        fwd_outputs = model(*example_fwd_inputs)
        fwd_outputs = _create_empty_tensors(fwd_outputs, device)
        shapes["outputs"] = [fwd_output.shape for fwd_output in fwd_outputs]

        # send shape dims
        if stage_id != num_stages - 1:
            output_shape_metadata_tensor = _create_metadata_tensor(
                fwd_outputs, device=device
            )
            dist.send(output_shape_metadata_tensor, next_rank)
        stage_id_to_shapes[stage_id] = shapes
    logger.info(stage_id_to_shapes)
    return stage_id_to_shapes


class PipelineStage(_PipelineStageBase):
    """
    A class representing a pipeline stage in a pipeline parallelism setup.
    This class is created manually by providing a example input (and optionally output)
    as opposed to the PipelineStage class that is outputed from pipeline().
    This class extends the `_PipelineStageBase` class and can similarly be used
    in `PipelineScheule`.

    Args:
        submodule (nn.Module): The PyTorch module wrapped by this stage.
        stage_index (int): The ID of this stage.
        num_stages (int): The total number of stages.
        device (torch.device): The device where this stage is located.
        input_args (Union[torch.Tensor, Tuple[torch.tensor]], optional): The input arguments for the submodule.
        output_args (Union[torch.Tensor, Tuple[torch.tensor]], optional): The output arguments for the submodule.
        group (dist.ProcessGroup, optional): The process group for distributed training. If None, default group.
        dw_builder: TODO clean up comments
    """

    def __init__(
        self,
        submodule: nn.Module,
        stage_index: int,
        num_stages: int,
        device: torch.device,
        input_args: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        output_args: Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...]]] = None,
        group: Optional[dist.ProcessGroup] = None,
        dw_builder: Optional[Callable[[], Callable[..., None]]] = None,
    ):
        super().__init__(submodule, stage_index, num_stages, device, group, dw_builder)
        self.submod.to(self.device)
        # When we materialize the model partition on cuda, we call reset_parameters() if it is available
        self.inputs: List[torch.Tensor] = []
        self.outputs: List[torch.Tensor] = []

        self.inputs = _create_empty_tensors(input_args, device)

        if output_args is None:
            logger.info("output_args not provided, performing forward using input_args")
            self.outputs = self.submod(*self.inputs)
            # create buffers for the output so that the data is in the correct
            # shape in order to use in p2p op (send)
            self.outputs = _create_empty_tensors(self.outputs, device)
        else:
            self.outputs = _create_empty_tensors(output_args, device)

        self._configure_outputs_meta(tuple(self.outputs))

        # these are the buffers used in backwards send/recv, they are allocated later
        self.outputs_grad: List[torch.Tensor] = []

        def stage_global_rank(peer_rank):
            return (
                peer_rank
                if self.group is None
                else dist.get_global_rank(self.group, peer_rank)
            )

        self.prev_stage = stage_global_rank((self.group_rank - 1) % self.group_size)
        self.next_stage = stage_global_rank((self.group_rank + 1) % self.group_size)

        logger.debug(
            f"finished pipeline stage init, {self.stage_index=}, {self.is_first=}, "  # noqa: G004
            f"{self.is_last=}, {self.num_stages=}, "
            f"inputs: {[inp.shape for inp in self.inputs]}, "
            f"output: {[output.shape for output in self.outputs]}"
        )

    def _prepare_forward_infra(self, num_microbatches: int) -> None:
        # Receive info during forward
        # TODO: create args_recv_info lazily? (same needed for PipelineStage)
        for chunk_id in range(num_microbatches):
            self.set_requires_grad[chunk_id] = False
            if not self.is_first:
                # We assume that we always receive from stage - 1
                recv_infos = tuple(
                    [
                        _RecvInfo(
                            f"recv_for_{self.stage_index}_from_{self.stage_index - 1}",
                            self.stage_index - 1,
                            _make_tensor_from_meta(inp, self.device),
                        )
                        for inp in self.inputs
                    ]
                )

                self.args_recv_info[chunk_id] = recv_infos
            else:
                self.args_recv_info[chunk_id] = tuple(
                    [_RootArgPlaceholder(i) for i in self.inputs]
                )

        # Send info during forward for each activation
        # only need the rank that is being sent to
        self.act_send_info: Dict[int, List] = {}
        for idx in range(len(self.outputs)):
            # We assume we always send to stage + 1
            if not self.is_last:
                self.act_send_info[idx] = [self.stage_index + 1]
            else:
                self.act_send_info[idx] = []

    def _create_grad_recv_info(
        self,
        act_send_info: Dict,
    ) -> Tuple[_RecvInfo, ...]:
        grad_recv_info: Tuple[_RecvInfo, ...] = ()
        if not self.is_last:
            # Receiving gradients from multiple sources is not supported
            # hence we only take the first destination
            grad_recv_info = tuple(
                [
                    _RecvInfo(
                        f"recv_grad_for_{self.stage_index}_from_{dst_list[0]}",
                        dst_list[0],
                        _make_tensor_from_meta(self.outputs[idx], self.device),
                    )
                    for idx, dst_list in act_send_info.items()
                ]
            )
        return grad_recv_info

    def _init_p2p_neighbors(self):
        """
        Set up p2p communitors between previous and next stages
        by sending a dummy tensor.

        If this is used, must be called for all pipeline stages.
        """
        ops = []
        recv_tensor = torch.zeros(1, device="cuda")
        send_tensor = torch.ones(1, device="cuda")
        # forward
        if not self.is_first:
            ops.append(dist.P2POp(dist.irecv, recv_tensor, self.prev_stage, self.group))
        if not self.is_last:
            ops.append(dist.P2POp(dist.isend, send_tensor, self.next_stage, self.group))

        # backward
        if not self.is_first:
            ops.append(dist.P2POp(dist.isend, send_tensor, self.prev_stage, self.group))
        if not self.is_last:
            ops.append(dist.P2POp(dist.irecv, recv_tensor, self.next_stage, self.group))

        return True


def _validate_stage_shapes(pipeline_stages: List[PipelineStage]):
    """
    Check that the buffer shapes match between stages was expected by performing an all_gather between
    all stages.
    """
    if len(pipeline_stages) == 0:
        raise ValueError("No pipeline stages provided.")

    virtual_pipeline_size = len(pipeline_stages)
    all_inputs = []
    all_outputs = []
    world_size = pipeline_stages[0].group_size
    num_stages = pipeline_stages[0].num_stages

    # perform all gathers between all stages
    for virtual_id, stage in enumerate(pipeline_stages):
        world_size = stage.group_size
        stage_id: int = stage.stage_index
        rank = stage.group_rank
        # check that world_size and num_stages are consistent across all stages
        if stage.group_size != world_size:
            raise ValueError(
                f"Stage id {stage_id} has world size ({stage.group_size}) \
                which does not match world size ({world_size}) of other stages."
            )
        if stage.num_stages != num_stages:
            raise ValueError(
                f"Stage id {stage_id} has num stages ({stage.num_stages}) \
                which does not match num stages ({num_stages}) of other stages."
            )

        pg_rank = dist.get_rank(stage.group)
        if rank != pg_rank:
            raise ValueError(
                f"Rank {rank} is not equal to process group rank {pg_rank}"
            )

        if (num_stages := stage.num_stages) % world_size != 0:
            raise ValueError(
                f"Number of stages ({num_stages}) must be a multiple of the world_size ({world_size})"
            )

        # all gather each ranks inputs
        tensor_list = [
            _create_metadata_tensor(device=stage.device)
            for _ in range(stage.group_size)
        ]
        expected_inputs = stage.inputs
        stage_input = _create_metadata_tensor(expected_inputs, device=stage.device)
        dist.all_gather(tensor_list, stage_input)
        stage_input_shapes = [
            _extract_metadata_from_tensor(tensor) for tensor in tensor_list
        ]

        # all gather each ranks outputs
        tensor_list = [
            _create_metadata_tensor(device=stage.device)
            for _ in range(stage.group_size)
        ]
        expected_outputs = stage.outputs
        stage_output = _create_metadata_tensor(expected_outputs, device=stage.device)
        dist.all_gather(tensor_list, stage_output)
        stage_output_shapes = [
            _extract_metadata_from_tensor(tensor) for tensor in tensor_list
        ]

        logger.debug(
            f"Rank: {pg_rank}"  # noqa: G004
            f"Stage id: {stage_id}"
            f"Stage num stages: {stage.num_stages}"
            f"Stage rank: {rank}"
            f"Stage world size: {world_size}"
            f"Stage {virtual_id * world_size}-{(virtual_id + 1) * world_size - 1} input shapes: {stage_input_shapes}"  # noqa: G003
            f"Stage {virtual_id * world_size}-{(virtual_id + 1) * world_size - 1} output shapes: {stage_output_shapes}"  # noqa: G003
        )

        all_inputs.extend(stage_input_shapes)
        all_outputs.extend(stage_output_shapes)

        # log only rank 0's view, they will all be equivalent
        if pg_rank == 0:
            logger.info(
                "all stage inputs: %s \n all stage outputs: %s", all_inputs, all_outputs
            )

    # Check if the output for stage 0 matches the input at stage 1, and so forth
    for i in range(virtual_pipeline_size * world_size - 1):
        if (out := all_outputs[i]) != (inp := all_inputs[i + 1]):
            raise ValueError(
                f"Stage_id {i} output shape {out} at does not match stage_id {i + 1} input shape {inp}."
            )
