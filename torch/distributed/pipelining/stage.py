# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
import operator
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Union

import torch
import torch.distributed as dist
import torch.fx as fx
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensor
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.fx.node import map_aggregate
from torch.nn.parallel import DistributedDataParallel
from torch.utils._pytree import tree_map_only

from ._backward import stage_backward, stage_backward_input, stage_backward_weight
from ._debug import map_debug_info
from ._utils import flatten_args, PipeInfo, validate_tensors_metadata


__all__ = [
    "PipelineStage",
    "build_stage",
]

logger = logging.getLogger(__name__)


def _normalize_model_output_as_tuple(output: Any) -> tuple[Any]:
    """[Note: pipeline model output type]

    The output of the model passed to pipelining can be any type, controlled by the user.

    However, there are 2 API surfaces that complicate this.
    (1) the outputs of intermediate stages are passed via Send/Recv ops to subsequent stages. The implicit assumption
    is that each element of the outputs is a tensor.  Otherwise, Send/Recv would not be supported.  The exception
    is the last layer of the model, which can output anything any which won't be communicated via Send/Recv.
    (2) the outputs of the last layer of the model are returned to the user, or, passed to the loss function.
    The loss function can be written in any way, such that its inputs match the outputs of the model.

    It would be convenient if we could strictly type the output signature of the pipeline stage wrapping the model,
    but we do not want to impose an unnecessary constraint on user provided models.

    Currently, we let user provided models return either a Tensor or a tuple of Tensors from each stage. Due to
    torch.export tracing, compiled models may also return a list instead of a Tuple, which we will normalize back to a
    tuple for consistency.

    TODO: should we be stricter about asserting that stage modules (intermediate and output) all return only Tensor
    values?
    """
    if type(output) is list:
        # HACK: this is a hacky workaround for the fact that export creates
        # output in list format
        output = tuple(output)

    # Unify output form to tuple for easy correspondance with
    # `act_send_info`
    output_tuple = output if type(output) is tuple else (output,)
    return output_tuple


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
                skipped during the module's actual backward pass. The builder must be invoked by stage after stage runs
                model backwards, and stage should save the latest dw_runner to run during weight pass.
                If not provided, a dw_runner will be generated automatically by traversing the autograd graph.
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
        self.backward_state: dict[int, tuple[Any, ...]] = {}

        # store dw_runner per microbatch_id
        self.dw_runner: dict[int, Callable[..., None]] = {}

        # `group_rank` is rank in process group `group`.
        self.group_rank = dist.get_rank(self.group)
        self.group_size = dist.get_world_size(self.group)
        if self.group_size > self.num_stages:
            raise RuntimeError(
                f"Pipeline group size {self.group_size} cannot be larger than number of stages {self.num_stages}"
            )

        # Run time states
        self._outputs_meta: Optional[tuple[torch.Tensor, ...]] = None
        # map microbatch ID to list of forward tensor args
        self.fwd_cache: dict[int, tuple[Any, list[torch.Tensor]]] = {}
        # map microbatch ID to list of backward grad tensor args
        self.bwd_cache: dict[int, tuple[Optional[torch.Tensor], ...]] = {}
        # Caching chunk outputs for final output merge or reduction
        self.output_chunks: list[Any] = []

        # Initialize has_backward to false; this will be set to true if loss
        # function is passed to pipeline schedule
        self.has_backward = False
        # Log prefix
        self.log_prefix = f"[Stage {self.stage_index}]"

        # Forward infra
        self.args_recv_info: dict[int, tuple[InputInfo, ...]] = {}
        self.act_send_info: dict[int, list] = {}

        # Backward infra will created lazily
        self.grad_recv_info: dict = {}
        self.grad_send_info: Optional[list] = None

        # To be populated later by the Schedule
        self.chunks: Optional[int] = None
        self.stage_index_to_group_rank: dict[int, int] = {
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

    def _configure_outputs_meta(self, outputs_meta: tuple[torch.Tensor, ...]):
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

    def get_outputs_meta(self) -> tuple[torch.Tensor, ...]:
        """Get the output metadata (meta tensors) reprensenting the outputs of this stage"""
        assert (
            self._outputs_meta is not None
        ), "Attempted to get_outputs_meta() without configuring output meta"
        return self._outputs_meta

    def _create_grad_send_info(
        self,
        args_recv_info: tuple,
    ) -> list[Optional[int]]:
        """
        Create a list of stage indices to send gradients to.
        """
        grad_send_info: list[Optional[int]] = []

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
    def _prepare_forward_infra(
        self,
        num_microbatches: int,
        args: tuple[Any, ...],
        kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[Any, ...]:
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
        act_send_info: dict,
    ) -> tuple[_RecvInfo, ...]:
        raise NotImplementedError

    def _get_recv_ops(
        self,
        recv_infos: tuple[InputInfo, ...],
    ) -> list[dist.P2POp]:
        """
        Helper function shared by `get_fwd_recv_ops` and `get_bwd_recv_ops`.
        Returns a list of ops that correspond to the recv infos.
        """
        ops: list[dist.P2POp] = []
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

    """[Note: V-schedule special case]

    V-Schedules have a special case where 2 stages with adjacent stage_id are on the same rank.

    ex: 2 ranks, 4 stages forms a simple V:
    rank0:  stage 0                   stage 3
    rank1:          stage 1  stage 2

    stage 0,1 and 2,3 communicate activations using send/recv as usual, but stage 1,2 do not need to
    use communication ops.  Instead, they should pass tensor data directly via function call.

    set_local_fwd_input and (get_local_bwd_output + set_local_bwd_input) facilitate this optimization, and
    should be called at the appropriate time during the pipeline schedule (after forward or backward execution).
    """

    def set_local_fwd_input(self, prev_stage_outputs: Any, mb_index: int) -> None:
        """
        Moves 'prev_stage_outputs' from another stage on the same rank into place as inputs for this stage. Avoids
        copying tensor data or using send/recv op.  Detaches original tensor and sets requires_grad so the
        tensor can serve as a leaf for autograd and gradients can be collected from it during backward.
        """
        recv_infos: tuple[InputInfo, ...] = self.args_recv_info[mb_index]

        # See [Note: pipeline model output type]
        prev_stage_outputs = _normalize_model_output_as_tuple(prev_stage_outputs)

        for info, tensor in zip(recv_infos, prev_stage_outputs):
            assert isinstance(
                tensor, torch.Tensor
            ), f"expected tensor values as outputs from prev stage, got {type(tensor)}"
            assert isinstance(
                info, _RecvInfo
            ), "set_local_Fwd_input should only be called on non-first stage, which should always have RecvInfo"

            # We don't need to do a data copy here, since we can directly pass the activation tensor reference from
            # one stage to the next.  However, we do need to mark the activation as a leaf tensor since it will serve
            # as the input tensor for a fresh autograd graph, not part of the previous stage's autograd graph.
            # TODO: confirm, do we use this activation as the root of the backward call for the previous stage? does
            # detach have any affect on that?
            info.buffer = tensor.detach().requires_grad_(True)

    def get_local_bwd_output(self, mb_index):
        """
        Returns the input grad tensors for this stage, which correspond to the stage inputs during forward.
        """
        assert (
            self.has_backward
        ), "can't steal_bwd_input if this stage doesn't have backward"
        assert not self.is_first, "can't get bwd output if this stage is first"

        self._check_chunk_id(mb_index)
        return self.bwd_cache.pop(mb_index)

    def set_local_bwd_input(
        self, next_stage_bwd_outputs: tuple[Optional[torch.Tensor], ...], mb_index: int
    ) -> None:
        """
        Moves 'grad input' tensors from the next stage to 'grad_output' on this stage, avoiding a copy or send/recv.
        Does not detach or set '_requires_grad'.
        """
        assert isinstance(
            next_stage_bwd_outputs, tuple
        ), f"Expected tuple, got {type(next_stage_bwd_outputs)}"

        assert (
            self.has_backward
        ), "can't set bwd input if this stage doesn't have backward"
        assert not self.is_last, "can't set bwd input if this stage is last"
        recv_infos = self.grad_recv_info[mb_index]
        for info, tensor in zip(recv_infos, next_stage_bwd_outputs):
            assert isinstance(
                tensor, torch.Tensor
            ), f"expected tensor values as outputs from prev stage, got {type(tensor)}"
            assert isinstance(
                info, _RecvInfo
            ), f"Expected a recv info, got {type(info)}"
            info.buffer = tensor

    def get_fwd_recv_ops(self, fwd_chunk_id: int) -> list[dist.P2POp]:
        """
        Returns a list of ops that are needed to receive the input arguments
        for this stage.
        """
        recv_infos: tuple[InputInfo, ...] = self.args_recv_info[fwd_chunk_id]

        return self._get_recv_ops(recv_infos)

    def get_bwd_recv_ops(self, bwd_chunk_id: int) -> list[dist.P2POp]:
        """
        Returns a list of ops that are needed to receive the gradients
        for this stage.
        """
        if not self.has_backward or self.is_last:
            return []

        recv_infos = self.grad_recv_info[bwd_chunk_id]
        return self._get_recv_ops(recv_infos)

    def get_fwd_send_ops(self, fwd_chunk_id: int) -> list[dist.P2POp]:
        """
        Get the activation send ops for current stage's forward.
        """
        output = self.output_chunks[fwd_chunk_id]
        # Unify output form to tuple for easy correspondance with
        # `act_send_info`
        output_tuple = output if type(output) is tuple else (output,)

        ops: list[dist.P2POp] = []

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

    def get_bwd_send_ops(self, bwd_chunk_id: int) -> list[dist.P2POp]:
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

        ops: list[dist.P2POp] = []
        grads_input = self.bwd_cache.pop(bwd_chunk_id)
        for grad, grad_recv_stage in zip(grads_input, self.grad_send_info):
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
        recv_infos: tuple[InputInfo, ...],
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
            recv_infos,  # type: ignore[arg-type]
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

    def scale_grads(self, grad_scale_factor: int) -> None:
        """Scale gradients model gradients by `grad_scale_factor`, which should be specified in coordination with the
        loss function used with pipelining.  For loss functions which perform 'mean' loss reduction, `grad_scale_factor`
        should be set to num_microbatches.  For loss functions that use `sum` reduction, `grad_scale_factor` should
        be set to 1.

        Should only be called once per pipeline schedule step, after all backwards passes have completed.
        """

        # PP scales only for its own contribution (microbatches), but relies on DP to scale further
        # for DP degree.
        if grad_scale_factor != 1:
            for p in self.submod.parameters():
                if p.grad is not None:
                    p.grad.div_(grad_scale_factor)

    def backward_maybe_with_nosync(
        self,
        backward_type,
        bwd_kwargs: dict,
        last_backward: bool = False,
    ) -> tuple[tuple[Optional[torch.Tensor], ...], Optional[list[dict[str, Any]]]]:
        """
        Whether using PP with FSDP or DDP, there are some runtime differences between the last backward step and the
        other steps.  Namely, we need to accumulate gradients on previous steps and reduce them on the last step, but
        there are additional state-variables and performance considerations depending on the data parallelism used.
        This helper should adapt any pipeline parallel schedule to work with common/supported data parallel libraries.
        """

        def perform_backward(
            backward_type,
        ) -> Callable[
            [],
            tuple[tuple[Optional[torch.Tensor], ...], Optional[list[dict[str, Any]]]],
        ]:
            if backward_type == "full":
                return lambda: (
                    stage_backward(
                        bwd_kwargs["stage_output"],
                        bwd_kwargs["output_grads"],
                        bwd_kwargs["input_values"],
                    ),
                    None,
                )
            elif backward_type == "input":
                return lambda: stage_backward_input(
                    bwd_kwargs["stage_output"],
                    bwd_kwargs["output_grads"],
                    bwd_kwargs["input_values"],
                    self.submod.parameters(),
                )
            elif backward_type == "weight":
                return lambda: (
                    stage_backward_weight(
                        self.submod.parameters(), bwd_kwargs["param_groups"]
                    ),
                    None,
                )
            else:
                raise RuntimeError(f"Unknown backward type: {backward_type}")

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
                result = perform_backward(backward_type)()
            else:
                with self.submod.no_sync():  # type: ignore[operator]
                    result = perform_backward(backward_type)()
        # If submod is a FSDP module
        elif isinstance(self.submod, FSDPModule):
            self.submod.set_is_last_backward(False)
            self.submod.set_reshard_after_backward(False)
            self.submod.set_requires_gradient_sync(False)
            result = perform_backward(backward_type)()
            if last_backward:
                # Manually call post backward for FSDP
                def run_post_backward(fsdp_module: FSDPModule) -> None:
                    fsdp_module.set_is_last_backward(True)
                    fsdp_module.set_reshard_after_backward(True)
                    fsdp_module.set_requires_gradient_sync(True)
                    fsdp_state = fully_shard.state(fsdp_module)  # type: ignore[arg-type]
                    for state in fsdp_state._state_ctx.all_states:
                        if state._fsdp_param_group:
                            state._fsdp_param_group.post_backward()

                    # it would be much better if pipelining backward invoked .backward so autograd hooks
                    # worked and modules like DDP/FSDP behaved as expected.  Working around this for the time being,
                    # we need to call this too to ensure FSDP syncs its grad reduction ops back to the default stream.
                    fsdp_state._root_post_backward_final_callback()

                run_post_backward(self.submod)

        else:
            # Non-DP submodule, regular backward
            result = perform_backward(backward_type)()

        grads, param_groups = result
        return grads, param_groups

    def forward_one_chunk(
        self,
        fwd_chunk_id: int,
        args: tuple[Any, ...],
        kwargs: Optional[dict[str, Any]] = None,
    ):
        """
        Perform forward pass on the stage with one microbatch.
        `args` and `kwargs` are the inputs from *external* to this stage.
        As of Sept 2024:
        - `args` applies to the first stage only, other stages receives args
          through activation transmission.
        - `kwargs` can be passed to all stages via respective `step` calls.
        """

        if self.is_first:
            # First stage doesn't need to receive anything
            composite_args = args
        else:
            # Receive activations for this chunk
            # Activations only come in args form
            composite_args = self._retrieve_recv_activations(fwd_chunk_id)

        composite_kwargs = kwargs or {}

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

        # See [Note: pipeline model output type]
        output_tuple = _normalize_model_output_as_tuple(output)

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

        # We return the original user-provied output, not normalized to tuple.
        # See [Note: pipeline model output type]
        return output

    def backward_one_chunk(
        self,
        bwd_chunk_id: int,
        loss=None,
        full_backward: bool = True,
        last_backward=False,
    ):
        """
        Perform backward pass on the module.
        This should only be called once per microbatch.

        If full_backward is True (the default), the full backward pass including weight and input gradients will be run,
        and it is an error to call `backward_weight_one_chunk` for this bwd_chunk_id.

        If full_backward is False, it is optional that `dw_runner` was provided to the PipelineStage at __init__ time,
        and a subsequent call to `backward_weight_one_chunk` is required to invoke dw_runner and complete the backward.

        last_backward is controlled by the schedule and signals synchronization of gradients across DP groups
        after the last backward.
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

        grads_input: tuple[Optional[torch.Tensor], ...] = ()

        # Custom backward function
        if self.dw_builder:
            # TODO: We may want to change our semantics so we are allowed to ignore
            # the 'dw_builder' and call full_backward directly when it is a full_backward op.
            grads_input, _ = self.backward_maybe_with_nosync(
                "full",
                bwd_kwargs,
                last_backward=last_backward,
            )
            if full_backward:
                self.dw_builder()()
            else:
                self.dw_runner[bwd_chunk_id] = self.dw_builder()
        else:
            if full_backward:
                grads_input, _ = self.backward_maybe_with_nosync(
                    "full", bwd_kwargs, last_backward=last_backward
                )
            else:
                param_groups: list[dict[str, Any]] | None = None
                # Skip the backward for the first stage since we will perform the weight update with
                # autograd.backward in backward_weight_one_chunk
                if not self.is_first:
                    if isinstance(bwd_kwargs["stage_output"], torch.Tensor):
                        bwd_kwargs["stage_output"] = (bwd_kwargs["stage_output"],)

                    # perform the partial backwards for the inputs with a custom backward function
                    # when the "stage_ouput" is a loss, then it is a tensor, otherwise it is a tuple of tensors
                    grads_input, param_groups = self.backward_maybe_with_nosync(
                        "input", bwd_kwargs, last_backward=last_backward
                    )

                # TODO: we dont need to save this, add to dw_runner?
                self.backward_state[bwd_chunk_id] = (
                    bwd_kwargs["input_values"],
                    param_groups,
                    bwd_kwargs["stage_output"],
                    bwd_kwargs["output_grads"],
                )
                # Save a placeholder for the dw_runner
                self.dw_runner[bwd_chunk_id] = lambda: None

        self.bwd_cache[bwd_chunk_id] = grads_input

        if self.is_last and not self.is_first:
            # Autograd dependencies:
            #    rest_of_autograd_graph -> stage_output -> loss
            # stage_output is no longer used in the last stage for backward and only needed
            # to return to the user in merge_output_chunks, therefore
            # this should be detached to release autograd graph context and free memory earlier
            for t in stage_output:
                if not t._is_view():  # views are not detachable in-place
                    t.detach_()

        logger.debug("%s Backwarded chunk %s", self.log_prefix, bwd_chunk_id)

    def backward_weight_one_chunk(self, bwd_chunk_id: int, last_backward=False):
        assert bwd_chunk_id in self.dw_runner, (
            f"{self.log_prefix} Attempted to run backward_weight_one_chunk for chunk {bwd_chunk_id}"
            " without first calling `backward_one_chunk(full_backward=False)`"
        )

        if self.dw_builder is not None:
            self.dw_runner.pop(bwd_chunk_id)()
        else:
            (
                input_values,
                param_groups,
                stage_output,
                output_grads,
            ) = self.backward_state.pop(bwd_chunk_id)

            if self.stage_index != 0:
                bwd_kwargs = {
                    "stage_output": stage_output,
                    "param_groups": param_groups,
                }
                self.backward_maybe_with_nosync(
                    "weight", bwd_kwargs, last_backward=last_backward
                )
            else:
                # TODO: figure out a better way to do this:
                # if inputs does not require gradient,
                # then the parameter group will not be fully captured during stage_backward_input
                # in this case, we need call grad directly on the parameters
                # To solve: make input fn do the intersect compute and then finish it off during W
                bwd_kwargs = {
                    "stage_output": stage_output,
                    "output_grads": output_grads,
                    "input_values": input_values,
                }
                self.backward_maybe_with_nosync(
                    "full", bwd_kwargs, last_backward=last_backward
                )

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
            # Without it, we are not 100% sure how to match the args and
            # expected_args.
            return

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

    def _validate_fwd_outputs(self, outputs: tuple[torch.Tensor, ...]):
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
        self.submod_to_stage_index: dict[str, int] = {}
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

    def _prepare_forward_infra(
        self,
        num_microbatches: int,
        args: tuple[Any, ...],
        kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[Any, ...]:
        """
        Create send/recv infrastructures for activations (during forward)
        """
        # TODO(whc)
        # this method should be deleted once lazy buffer allocation is implemented
        # for now, it ignores args/kwargs becuase it should not need to do shape inference
        for chunk in range(num_microbatches):
            self.args_recv_info[chunk] = self._create_act_recv_info()

        # Send info during forward for each activation
        self.act_send_info = self._create_act_send_info()
        return tuple()

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
            # In case there is backward pass, set requires_grad for receive buffers
            # before first forward
            if self.has_backward:
                buffer.requires_grad_(True)

            return _RecvInfo(
                arg_node.name,
                src_stage,
                buffer,
            )

        args_recv_info: list[InputInfo] = []
        # Filter out placeholder nodes from `self.submod` (a GraphModule)
        placeholders = filter(  # type: ignore[var-annotated]
            lambda node: node.op == "placeholder", self.submod.graph.nodes  # type: ignore[arg-type, union-attr]
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
        act_send_info: dict[int, list] = {}
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
        output_vals: tuple[torch.Tensor] = tuple(
            v.meta["val"] for v in flatten_args(output_node.args)
        )
        self._configure_outputs_meta(output_vals)

        logger.debug("%s Send info: %s", self.log_prefix, act_send_info)
        return act_send_info

    def _get_output_node(self):
        output_nodes = [node for node in self.submod.graph.nodes if node.op == "output"]  # type: ignore[union-attr]
        assert len(output_nodes) == 1
        output_node = output_nodes[0]
        return output_node

    def _create_grad_recv_info(
        self,
        act_send_info: dict,
    ) -> tuple[_RecvInfo, ...]:
        """
        Create a tuple of `_RecvInfo` for gradients.
        """
        # Dict[output_index, _RecvInfo]
        grad_recv_info: dict[int, _RecvInfo] = {}
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


class PipelineStage(_PipelineStageBase):
    """
    A class representing a pipeline stage in a pipeline parallelism setup.

    PipelineStage assumes sequential partitioning of the model, i.e. the model is split into chunks where outputs from
    one chunk feed into inputs of the next chunk, with no skip connections.

    PipelineStage performs runtime shape/dtype inference automatically by propagating the outputs from stage0 to
    stage1 and so forth, in linear order.  To bypass shape inference, pass the `input_args` and `output_args` to each
    PipelineStage instance.

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
        input_args: Optional[Union[torch.Tensor, tuple[torch.Tensor, ...]]] = None,
        output_args: Optional[Union[torch.Tensor, tuple[torch.Tensor, ...]]] = None,
        group: Optional[dist.ProcessGroup] = None,
        dw_builder: Optional[Callable[[], Callable[..., None]]] = None,
    ):
        super().__init__(submodule, stage_index, num_stages, device, group, dw_builder)
        self.inputs: Optional[list[torch.Tensor]] = None
        self.inputs_meta: Optional[tuple[torch.Tensor, ...]] = None
        # Note: inputs and submod should ideally be on meta device. We decided not to assert this (yet) becuase it
        # might be breaking for existing users.
        if input_args is None:
            assert output_args is None, (
                "If specifying output_args, input_args must also be specified. "
                "Otherwise, shape inference will be performed at runtime"
            )
        else:
            self.inputs_meta = (
                (input_args,) if isinstance(input_args, torch.Tensor) else input_args
            )
            if output_args is None:
                logger.warning(
                    "Deprecation warning: passing input_args and performing init-time shape inference is deprecated. "
                    "PipelineStage now supports runtime shape inference using the real inputs provided to schedule step(). "
                    "Either delete `input_args` arg to `PipelineStage` to opt-into runtime shape inference, "
                    "or additionally pass `output_args` to `PipelineStage` to fully override shape inference. "
                )
                try:
                    with torch.no_grad():
                        output_args = submodule(*self.inputs_meta)
                    output_args = tree_map_only(
                        torch.Tensor, lambda x: x.to("meta"), output_args
                    )
                except Exception as e:
                    raise RuntimeError(
                        "Failed to perform pipeline shape inference- are your inputs on the same device as your module?"
                    ) from e
            assert (
                output_args is not None
            ), "If passing input_args, also pass output_args to override shape inference"
            self._configure_outputs_meta(
                (output_args,) if isinstance(output_args, torch.Tensor) else output_args
            )

        # these are the buffers used in backwards send/recv, they are allocated later
        self.outputs_grad: list[torch.Tensor] = []

        def stage_global_rank(peer_rank):
            return (
                peer_rank
                if self.group is None
                else dist.get_global_rank(self.group, peer_rank)
            )

        self.prev_rank = stage_global_rank((self.group_rank - 1) % self.group_size)
        self.next_rank = stage_global_rank((self.group_rank + 1) % self.group_size)

        dbg_str = (
            f"Finished pipeline stage init, {self.stage_index=}, {self.is_first=}, "  # noqa: G004
            f"{self.is_last=}, {self.num_stages=}, "
        )
        if self.inputs_meta is not None:
            dbg_str += (
                f"inputs: {[inp.shape for inp in self.inputs_meta]}, "
                f"output: {[output.shape for output in self.get_outputs_meta()]}"
            )
        else:
            dbg_str += " running shape-inference at runtime"

        logger.debug(dbg_str)

    def _shape_inference(
        self,
        args: tuple[Any, ...],
        kwargs: Optional[dict[str, Any]] = None,
    ):
        if kwargs is None:
            kwargs = {}
        assert args is not None, "Args may be an empty tuple but not None"

        # We skip recv communication if we're the first stage, but also if the previous stage is on the same rank
        # and can pass its output shapes in as args instead of using send/recv.
        if (
            self.is_first
            # if not first stage, then check if prev stage is on the same rank
            or self.stage_index_to_group_rank[self.stage_index - 1] == self.group_rank
        ):
            logger.debug(
                "Shape inference: stage %s skipping recv, because shape info passed in via `args`",
                self.stage_index,
            )
            args = tree_map_only(torch.Tensor, lambda x: x.to("meta"), args)
        else:
            assert (
                len(args) == 0
            ), "Can't supply input args for shape inference on non-first stage"
            objects = [None]
            logger.debug(
                "Shape inference: stage %s receiving from stage %s",
                self.stage_index,
                self.stage_index - 1,
            )
            dist.recv_object_list(
                objects,
                src=dist.get_global_rank(
                    self.group or dist.distributed_c10d._get_default_group(),
                    self.stage_index_to_group_rank[self.stage_index - 1],
                ),
                group=self.group,
                device=self.device,
            )
            recv_args = objects[0]
            assert isinstance(recv_args, tuple), type(recv_args)
            args = recv_args

        # cache input shapes for use during recv buffer allocation
        self.inputs_meta = args
        args = tree_map_only(
            torch.Tensor, lambda x: torch.zeros_like(x, device=self.device), args
        )

        # set attributes needed for forward
        with torch.no_grad():
            outputs = self.submod(*args, **kwargs)

        # if single tensor, convert so it is always a list
        if isinstance(outputs, torch.Tensor):
            outputs = [outputs]

        # communicate meta outputs not real outputs for two reasons
        # 1 - its faster (esp. since obj coll pickles tensor data!)
        # 2 - avoid activating a cuda context for the src rank when unpickling on the recv end!
        outputs_meta = tuple(
            tree_map_only(torch.Tensor, lambda x: x.to("meta"), outputs)
        )
        logger.debug(
            "Shape inference: stage %s inputs %s, outputs %s",
            self.stage_index,
            self.inputs_meta,
            outputs_meta,
        )
        self._configure_outputs_meta(outputs_meta)

        # Passing outputs to the next stage:
        # two cases-
        # 1. Usually: use send/recv communication to pass the output
        # 2. Special case: for V-schedules, 2 'adjacent' stages (e.g. stage 3, 4 in an 8-stage 4-rank V)
        #    pass their shape info via return value and function args rather than send/recv.
        if (
            self.is_last
            # if not last stage, then check if next stage is on the same rank
            or self.stage_index_to_group_rank[self.stage_index + 1] == self.group_rank
        ):
            # Case (2) above: pass shape info via return value and caller passes it as args to next stage's
            # _shape_inference call
            logger.debug(
                "Shape inference: stage %s skipping send to next stage",
                self.stage_index,
            )

        else:
            # Case (1): send shapes via send operation, and ensure not to return it to the caller
            logger.debug(
                "Shape inference: stage %s sending to stage %s",
                self.stage_index,
                self.stage_index + 1,
            )
            dist.send_object_list(
                [outputs_meta],
                dst=dist.get_global_rank(
                    self.group or dist.distributed_c10d._get_default_group(),
                    self.stage_index_to_group_rank[self.stage_index + 1],
                ),
                group=self.group,
                device=self.device,
            )
            outputs_meta = tuple()

        return outputs_meta

    def _prepare_forward_infra(
        self,
        num_microbatches: int,
        args: tuple[Any, ...],
        kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[Any, ...]:
        # TODO move self.device to an argument from step API (from its input tensors)?
        assert num_microbatches is not None, "TODO fix num_microbatches"

        outputs: tuple[Any, ...] = tuple()
        if self.inputs_meta is None:
            outputs = self._shape_inference(args, kwargs)

        assert self.inputs_meta is not None
        # Receive info during forward
        # TODO: create args_recv_info lazily? (same needed for PipelineStage)
        for chunk_id in range(num_microbatches):
            if not self.is_first:
                # We assume that we always receive from stage - 1
                recv_infos = tuple(
                    [
                        _RecvInfo(
                            f"recv_for_{self.stage_index}_from_{self.stage_index - 1}",
                            self.stage_index - 1,
                            _make_tensor_from_meta(inp, self.device),
                        )
                        for inp in self.inputs_meta
                    ]
                )
                # In case there is backward pass, set requires_grad for receive buffers
                if self.has_backward:
                    for r in recv_infos:
                        r.buffer.requires_grad_(True)

                self.args_recv_info[chunk_id] = recv_infos
            else:
                self.args_recv_info[chunk_id] = tuple(
                    [_RootArgPlaceholder(i) for i in self.inputs_meta]
                )

        # Send info during forward for each activation
        # only need the rank that is being sent to
        self.act_send_info: dict[int, list] = {}

        for idx in range(len(self.get_outputs_meta())):
            # We assume we always send to stage + 1
            if not self.is_last:
                self.act_send_info[idx] = [self.stage_index + 1]
            else:
                self.act_send_info[idx] = []

        return outputs

    def _create_grad_recv_info(
        self,
        act_send_info: dict,
    ) -> tuple[_RecvInfo, ...]:
        grad_recv_info: tuple[_RecvInfo, ...] = ()
        if not self.is_last:
            # Receiving gradients from multiple sources is not supported
            # hence we only take the first destination
            grad_recv_info = tuple(
                [
                    _RecvInfo(
                        f"recv_grad_for_{self.stage_index}_from_{dst_list[0]}",
                        dst_list[0],
                        _make_tensor_from_meta(
                            self.get_outputs_meta()[idx], self.device
                        ),
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
            ops.append(dist.P2POp(dist.irecv, recv_tensor, self.prev_rank, self.group))
        if not self.is_last:
            ops.append(dist.P2POp(dist.isend, send_tensor, self.next_rank, self.group))

        # backward
        if not self.is_first:
            ops.append(dist.P2POp(dist.isend, send_tensor, self.prev_rank, self.group))
        if not self.is_last:
            ops.append(dist.P2POp(dist.irecv, recv_tensor, self.next_rank, self.group))

        return True
