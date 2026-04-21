# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
import operator
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.fx as fx
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensor
from torch.distributed._composable.replicate_with_fsdp import replicate, ReplicateModule
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.distributed.pipelining._utils import (
    _derive_grad_metas,
    _DTensorMeta,
    _make_tensor_from_meta,
    _MeshCache,
    _StageBackwardMeta,
    _StageForwardMeta,
    _StageMeta,
    _TensorMeta,
    extract_tensor_meta,
    extract_tensor_metas,
    flatten_args,
    GetMeshCallback,
    InferenceMode,
    PipeInfo,
    PipeliningMetadataError,
    TensorMeta,
    to_local_if_dtensor,
    validate_and_normalize_to_tuple,
    validate_static_arg_grad_correspondence,
    validate_tensors_metadata,
)
from torch.distributed.tensor import DTensor
from torch.fx.node import Argument, map_aggregate
from torch.nn.parallel import DistributedDataParallel

from ._backward import (
    _autograd_grad_for_inputs,
    stage_backward,
    stage_backward_input,
    stage_backward_weight,
)
from ._debug import map_debug_info


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

    # Unify output form to tuple for easy correspondence with
    # `act_send_info`
    output_tuple = output if type(output) is tuple else (output,)
    return output_tuple


class _RecvInfo:
    """Input tensor descriptor for a pipeline stage.

    Handles both received activations from a previous stage
    (``is_root_arg=False``) and root-level model inputs provided
    by the user (``is_root_arg=True``).
    """

    def __init__(
        self,
        input_name: str,
        source: int | None,
        buffer: torch.Tensor | None,
        tensor_meta: TensorMeta | None,
        *,
        is_root_arg: bool = False,
    ):
        # Name of this input
        self.input_name = input_name
        # Stage index of the source of this input (None for root args)
        self.source = source
        # Buffer to receive the input into (None for root args)
        self.buffer = buffer
        # Tensor metadata for validation and DTensor reconstruction
        self.tensor_meta = tensor_meta
        # Whether this is a root-level model input (no recv needed)
        self.is_root_arg = is_root_arg

    def __repr__(self):
        if self.is_root_arg:
            return f"_RecvInfo(input={self.input_name}, root_arg=True)"
        meta_type = type(self.tensor_meta).__name__ if self.tensor_meta else "None"
        buffer_shape = self.buffer.size() if self.buffer is not None else "None"
        return f"_RecvInfo(input={self.input_name}, source={self.source}, shape={buffer_shape}, meta={meta_type})"


class _PipelineStageBase(ABC):
    """Base class for pipeline stages.

    Defines common methods used by ``_PipelineStage`` (tracing frontend)
    and ``PipelineStage`` (manual frontend).
    """

    def __init__(
        self,
        submodule: torch.nn.Module,
        stage_index: int,
        num_stages: int,
        device: torch.device,
        group: dist.ProcessGroup | None = None,
        dw_builder: Callable[[], Callable[..., None]] | None = None,
    ):
        """
        Args:
            submodule: The module to be executed in this stage.
            stage_index: The index of this stage.
            num_stages: The total number of stages in this pipeline.
            device: The device to run this stage on.
            group: Process group for communication. Defaults to the
                default process group if ``None``.
            dw_builder: Builder function that produces a ``dw_runner``
                for deferred weight updates in F/I/W zero-bubble
                schedules. If ``None``, a runner is generated
                automatically via autograd graph traversal.
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
        # map microbatch ID to list of forward tensor args
        self.fwd_cache: dict[int, tuple[Any, list[torch.Tensor]]] = {}
        # map microbatch ID to list of backward grad tensor args
        self.bwd_cache: dict[int, tuple[torch.Tensor | None, ...]] = {}
        # Caching chunk outputs for final output merge or reduction
        self.output_chunks: list[Any] = []

        # Initialize has_backward to false; this will be set to true if loss
        # function is passed to pipeline schedule
        self.has_backward = False
        # Log prefix
        self.log_prefix = f"[Stage {self.stage_index}]"

        # Forward infra
        self.args_recv_info: dict[int, tuple[_RecvInfo, ...]] = {}
        self.act_send_info: dict[int, list] = {}

        # Backward infra will created lazily
        self.grad_recv_info: dict = {}
        self.grad_send_info: list | None = None

        # To be populated later by the Schedule
        self.chunks: int | None = None
        self.stage_index_to_group_rank: dict[int, int] = {
            i: i % self.group_size for i in range(self.num_stages)
        }

        # DTensor support: mesh cache for looking up DeviceMesh by (dim_names, layout)
        self._mesh_cache = _MeshCache()

        # Per-chunk runtime validation is expensive; only enable under
        # TORCH_DISTRIBUTED_DEBUG=DETAIL for debugging shape/dtype mismatches.
        self._runtime_validate = dist.get_debug_level() == dist.DebugLevel.DETAIL

        # DTensor support: consolidated stage metadata container
        # Contains inputs, outputs, input_grads, output_grads metadata
        self._stage_meta = _StageMeta()

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

    def _validate_stage_tensors(
        self,
        desc: str,
        expected: tuple[TensorMeta | None, ...] | None,
        actual: tuple[torch.Tensor | None, ...],
    ) -> None:
        """Validate actual tensors against expected metadata.

        Raises:
            PipeliningMetadataError: If metadata is missing or mismatched.
        """
        if expected is None:
            raise PipeliningMetadataError(f"{desc}: no metadata available")
        validate_tensors_metadata(desc, expected, actual)

    def _check_chunk_id(self, chunk_id: int):
        if self.chunks is None:
            raise RuntimeError(
                "Attempted to access chunk_id before chunks have been configured."
            )
        if chunk_id >= self.chunks:
            raise RuntimeError(
                f"Chunk id {chunk_id} is out of range [0, {self.chunks})"
            )

    def _create_grad_send_info(
        self,
        args_recv_info: tuple,
    ) -> list[int | None]:
        """
        Create a list of stage indices to send gradients to.
        """
        grad_send_info: list[int | None] = []

        def map_recv_to_send(a):
            # Note: we send gradients back to previous stage as long as in
            # forward it is a received input, regardless of whether it requires
            # grad. It is up to the previous stage to discard this gradient.
            if a.is_root_arg:
                # Root args don't have a source stage to send gradients to
                grad_send_info.append(None)
                return None
            else:
                grad_send_info.append(a.source)
                return a.source

        map_aggregate(args_recv_info, map_recv_to_send)

        logger.debug("%s Grad send info: %s", self.log_prefix, grad_send_info)
        return grad_send_info

    @abstractmethod
    def _prepare_forward_infra(
        self,
        num_microbatches: int,
        args: tuple[Any, ...] | _StageForwardMeta | None,
        kwargs: dict[str, Any] | None = None,
        has_backward: bool = False,
    ) -> _StageForwardMeta | None:
        raise NotImplementedError

    @abstractmethod
    def _prepare_backward_infra(
        self,
        num_microbatches: int,
        loss_fn: Callable[..., torch.Tensor] | None = None,
        target: torch.Tensor | None = None,
        received_grad_meta: _StageBackwardMeta | None = None,
    ) -> _StageBackwardMeta | None:
        raise NotImplementedError

    def _setup_backward_recv_info(self, num_microbatches: int):
        # TODO: this is needed for backward_maybe_with_nosync
        self.chunks = num_microbatches

        # IMPORTANT: _create_grad_recv_info reads self._stage_meta.output_grads
        # to attach DTensor metadata to _RecvInfo objects. The clear below MUST
        # happen after all _create_grad_recv_info calls complete.
        for mb_index in range(num_microbatches):
            self.grad_recv_info[mb_index] = self._create_grad_recv_info(
                self.act_send_info
            )

    @abstractmethod
    def _create_grad_recv_info(
        self,
        act_send_info: dict,
    ) -> tuple[_RecvInfo, ...]:
        raise NotImplementedError

    def _resolve_peer_global_rank(self, stage_idx: int) -> int:
        """Map a pipeline stage index to the corresponding global rank for P2P communication."""
        peer_rank = self.stage_index_to_group_rank[stage_idx]
        return dist.get_global_rank(
            self.group or dist.distributed_c10d._get_default_group(),
            peer_rank,
        )

    def _get_recv_ops(
        self,
        recv_infos: tuple[_RecvInfo, ...],
    ) -> list[dist.P2POp]:
        """
        Helper function shared by `get_fwd_recv_ops` and `get_bwd_recv_ops`.
        Returns a list of ops that correspond to the recv infos.
        """
        ops: list[dist.P2POp] = []
        for info in recv_infos:
            if info.is_root_arg:
                # Root args don't need recv operations
                continue
            # Skip entries with None buffer (None gradients)
            if info.buffer is None:
                assert info.tensor_meta is None  # noqa: S101
                continue
            # At this point, source and buffer are guaranteed non-None
            assert info.source is not None  # noqa: S101
            peer_global_rank = self._resolve_peer_global_rank(info.source)
            ops.append(
                dist.P2POp(dist.irecv, info.buffer, peer_global_rank, self.group)
            )

        return ops

    """[Note: V-schedule special case]

    V-Schedules have a special case where 2 stages with adjacent stage_id
    are on the same rank.

    Example: 2 ranks, 4 stages forms a simple V::

        rank0:  stage 0                   stage 3
        rank1:          stage 1  stage 2

    Stages 0/1 and 2/3 communicate via send/recv, but stages 1/2 pass
    tensors directly via function call, avoiding communication ops.
    """

    def set_local_fwd_input(self, prev_stage_outputs: Any, mb_index: int) -> None:
        """Pass outputs from a same-rank stage as forward inputs (V-schedule).

        Detaches tensors and sets ``requires_grad`` so they serve as autograd
        leaves. Handles DTensor activations transparently.
        """
        recv_infos: tuple[_RecvInfo, ...] = self.args_recv_info[mb_index]

        # See [Note: pipeline model output type]
        prev_stage_outputs = _normalize_model_output_as_tuple(prev_stage_outputs)

        for info, tensor in zip(recv_infos, prev_stage_outputs, strict=True):
            if not isinstance(tensor, torch.Tensor):
                raise AssertionError(
                    f"expected tensor values as outputs from prev stage, got {type(tensor)}"
                )
            if info.is_root_arg:
                raise AssertionError(
                    "set_local_fwd_input should only be called on non-first stage, which should always have non-root RecvInfo"
                )

            # Pass the activation tensor directly (same rank for local execution).
            # Detach to create a new autograd leaf for the fresh autograd graph.
            info.buffer = to_local_if_dtensor(tensor, detach=True)

    def get_local_bwd_output(self, mb_index):
        """
        Returns the input grad tensors for this stage, which correspond to the stage inputs during forward.
        """
        if not self.has_backward:
            raise AssertionError(
                "can't steal_bwd_input if this stage doesn't have backward"
            )
        if self.is_first:
            raise AssertionError("can't get bwd output if this stage is first")

        self._check_chunk_id(mb_index)
        return self.bwd_cache.pop(mb_index)

    def set_local_bwd_input(
        self, next_stage_bwd_outputs: tuple[torch.Tensor | None, ...], mb_index: int
    ) -> None:
        """
        Moves 'grad input' tensors from the next stage to 'grad_output' on this stage, avoiding a copy or send/recv.
        Does not detach or set '_requires_grad'.
        Handles DTensor gradients for V-schedule local passing.
        """
        if not isinstance(next_stage_bwd_outputs, tuple):
            raise AssertionError(f"Expected tuple, got {type(next_stage_bwd_outputs)}")

        if not self.has_backward:
            raise AssertionError(
                "can't set bwd input if this stage doesn't have backward"
            )
        if self.is_last:
            raise AssertionError("can't set bwd input if this stage is last")
        recv_infos = self.grad_recv_info[mb_index]
        for info, tensor in zip(recv_infos, next_stage_bwd_outputs, strict=True):
            if tensor is None:
                continue
            if not isinstance(tensor, torch.Tensor):
                raise AssertionError(
                    f"expected tensor values as outputs from prev stage, got {type(tensor)}"
                )
            if info.is_root_arg:
                raise AssertionError(
                    "set_local_bwd_input should only be called with non-root RecvInfo"
                )

            # Extract local tensor for the buffer (handles DTensor or plain tensor)
            info.buffer = to_local_if_dtensor(tensor)

    def get_fwd_recv_ops(self, fwd_chunk_id: int) -> list[dist.P2POp]:
        """
        Returns a list of ops that are needed to receive the input arguments
        for this stage.
        """
        recv_infos: tuple[_RecvInfo, ...] = self.args_recv_info[fwd_chunk_id]

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
        Handles DTensor outputs by extracting local tensors.
        """
        output_tuple, _ = self.fwd_cache[fwd_chunk_id]

        ops: list[dist.P2POp] = []

        for idx, out in enumerate(output_tuple):
            dst_stages = self.act_send_info[idx]
            for dst in dst_stages:
                if dst is None:
                    continue
                # Extract local tensor if DTensor
                send_tensor = to_local_if_dtensor(out, detach=True)
                logger.debug(
                    "%s Sending tensor to Stage %s: %s",
                    self.log_prefix,
                    dst,
                    send_tensor.size(),
                )
                peer_global_rank = self._resolve_peer_global_rank(dst)
                ops.append(
                    dist.P2POp(dist.isend, send_tensor, peer_global_rank, self.group)
                )

        return ops

    def get_bwd_send_ops(self, bwd_chunk_id: int) -> list[dist.P2POp]:
        """
        Get the gradient send ops for current stage's backward.
        Handles DTensor gradients by extracting local tensors.
        """
        if not self.has_backward or self.is_first:
            return []

        self._check_chunk_id(bwd_chunk_id)
        # Create bwd send infra lazily
        if self.grad_send_info is None:
            # Send info for input grads during backward:
            # List of destinations corresponding to input grads
            # Can be None if an input has no grad
            # `grad_send_info` is a mirror of `args_recv_info`
            self.grad_send_info = self._create_grad_send_info(self.args_recv_info[0])

        ops: list[dist.P2POp] = []
        grads_input = self.bwd_cache.pop(bwd_chunk_id)

        for grad, grad_recv_stage in zip(grads_input, self.grad_send_info, strict=True):
            if isinstance(grad, torch.Tensor) and grad_recv_stage is not None:
                # Extract local tensor if DTensor
                send_tensor = to_local_if_dtensor(grad)
                logger.debug(
                    "%s Sending gradient to Stage %s: %s",
                    self.log_prefix,
                    grad_recv_stage,
                    send_tensor.size(),
                )
                peer_global_rank = self._resolve_peer_global_rank(grad_recv_stage)
                ops.append(
                    dist.P2POp(dist.isend, send_tensor, peer_global_rank, self.group)
                )
            else:
                if grad is not None or grad_recv_stage is not None:
                    raise PipeliningMetadataError(
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
                if not a.is_root_arg and a.buffer is not None:
                    # Set to None is the newer and recommended way to clear grads, compared to `zero_()`.
                    # See https://github.com/pytorch/pytorch/pull/92731
                    a.buffer.grad = None

    def _map_tensor_from_recv_info(
        self,
        recv_infos: tuple[_RecvInfo, ...],
    ):
        """
        Map tensors from recv infos to a list.
        """

        def get_recv_tensor(info):
            if info.is_root_arg:
                raise PipeliningMetadataError("Cannot get recv tensor from root arg")
            return info.buffer

        return map_aggregate(cast(Argument, recv_infos), get_recv_tensor)

    def _retrieve_recv_activations(
        self,
        fwd_chunk_id: int,
    ):
        """
        Retrieve the activations received for the current stage during forward.
        Reconstructs DTensors if the inputs were DTensors.
        Also validates DTensor metadata against expected values.
        """
        recv_infos = self.args_recv_info[fwd_chunk_id]

        activations = []
        for i, info in enumerate(recv_infos):
            if not info.is_root_arg:
                # Non-root args have valid buffer and tensor_meta
                if info.buffer is None or info.tensor_meta is None:
                    raise PipeliningMetadataError(
                        f"Non-root arg '{info.input_name}' has None buffer or tensor_meta"
                    )
                # Effective requires_grad: metadata captures what the model
                # produced, but the runtime context (has_backward, grad mode)
                # determines whether we actually need gradients.
                effective_requires_grad = (
                    info.tensor_meta.requires_grad
                    and self.has_backward
                    and torch.is_grad_enabled()
                )
                if isinstance(info.tensor_meta, _DTensorMeta):
                    # Buffer must not require grad so from_local stays out
                    # of the autograd graph (no grad_placements needed).
                    if info.buffer.requires_grad:
                        raise PipeliningMetadataError(
                            f"Stage {self.stage_index}: recv buffer "
                            f"'{info.input_name}' unexpectedly requires grad "
                            f"before DTensor reconstruction"
                        )
                    mesh = self._mesh_cache.get_mesh(info.tensor_meta.mesh_cache_key)
                    activation = DTensor.from_local(
                        info.buffer,
                        device_mesh=mesh,
                        placements=info.tensor_meta.placements,
                        shape=info.tensor_meta.global_shape,
                        stride=info.tensor_meta.global_stride,
                        run_check=False,
                    ).requires_grad_(effective_requires_grad)
                else:
                    activation = info.buffer.requires_grad_(effective_requires_grad)
                # Activation must be a leaf so backward terminates here.
                if effective_requires_grad and not activation.is_leaf:
                    warnings.warn(
                        f"Stage {self.stage_index}: activation "
                        f"'{info.input_name}' is not a leaf "
                        f"(grad_fn={activation.grad_fn}); using "
                        f"retain_grad() as fallback",
                        stacklevel=2,
                    )
                    activation.retain_grad()
                activations.append(activation)
            else:
                raise PipeliningMetadataError(
                    f"_retrieve_recv_activations expected non-root _RecvInfo but got root arg at index {i}"
                )

        return tuple(activations)

    def _retrieve_recv_grads(
        self,
        bwd_chunk_id: int,
    ):
        """
        Retrieve the gradients received for the current stage during backward.

        Handles None gradients gracefully (for inputs that don't require grad).
        """
        recv_infos = self.grad_recv_info[bwd_chunk_id]

        grads: list[torch.Tensor | None] = []
        for i, info in enumerate(recv_infos):
            if not isinstance(info, _RecvInfo):
                raise PipeliningMetadataError(
                    f"Expected _RecvInfo but got {type(info)}"
                )
            if not info.is_root_arg:
                # Gradients can be None for non-differentiable outputs
                if info.buffer is None:
                    if info.tensor_meta is not None:
                        raise PipeliningMetadataError(
                            f"Grad recv '{info.input_name}': buffer is None but tensor_meta is not None"
                        )
                    grads.append(None)
                    continue
                if info.tensor_meta is None:
                    raise PipeliningMetadataError(
                        f"Grad recv '{info.input_name}': buffer is not None but tensor_meta is None"
                    )
                if isinstance(info.tensor_meta, _DTensorMeta):
                    # Reconstruct DTensor gradient from local tensor + metadata
                    mesh = self._mesh_cache.get_mesh(info.tensor_meta.mesh_cache_key)
                    grad = DTensor.from_local(
                        info.buffer,
                        device_mesh=mesh,
                        placements=info.tensor_meta.placements,
                        shape=info.tensor_meta.global_shape,
                        stride=info.tensor_meta.global_stride,
                        run_check=False,
                    )
                else:
                    grad = info.buffer
                grads.append(grad)
            else:
                raise PipeliningMetadataError(
                    f"grad_recv_info should not contain root args, but found one at index {i}"
                )

        return tuple(grads)

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
    ) -> tuple[tuple[torch.Tensor | None, ...], list[dict[str, Any]] | None]:
        """
        Whether using PP with FSDP, DDP, or replicate there are some runtime differences between the last backward step and the
        other steps.  Namely, we need to accumulate gradients on previous steps and reduce them on the last step, but
        there are additional state-variables and performance considerations depending on the data parallelism used.
        This helper should adapt any pipeline parallel schedule to work with common/supported data parallel libraries.
        """

        def perform_backward(
            backward_type,
        ) -> Callable[
            [],
            tuple[tuple[torch.Tensor | None, ...], list[dict[str, Any]] | None],
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

        # If submod is a FSDP or replicate module
        elif isinstance(self.submod, FSDPModule):
            self.submod.set_is_last_backward(False)
            self.submod.set_reshard_after_backward(False)
            self.submod.set_requires_gradient_sync(False)
            result = perform_backward(backward_type)()

        else:
            # Non-DP submodule, regular backward
            result = perform_backward(backward_type)()

        grads, param_groups = result
        return grads, param_groups

    def forward_one_chunk(
        self,
        fwd_chunk_id: int,
        args: tuple[Any, ...],
        kwargs: dict[str, Any] | None = None,
        save_forward_output: bool = True,
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

        if self._runtime_validate:
            self._validate_stage_tensors(
                f"Stage {self.stage_index} forward inputs",
                self._stage_meta.inputs,
                composite_args,
            )

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
        # Output chunks is only used for the last stage since we only merge the output of the last stage
        if self.is_last and save_forward_output:
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
        # Validate outputs before P2P send; skipped for last stage (outputs
        # go to loss/user, not via send/recv).
        if self._runtime_validate and not self.is_last:
            self._validate_stage_tensors(
                f"Stage {self.stage_index} forward outputs",
                self._stage_meta.outputs,
                output_tuple,
            )

        # We return the original user-provided output, not normalized to tuple.
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
        # skip backward computation if backward is not enabled
        if not self.has_backward:
            return

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
            if self._runtime_validate:
                # Validate backward input (output gradients) for DTensor metadata
                self._validate_stage_tensors(
                    f"Stage {self.stage_index} backward input (output_grads)",
                    self._stage_meta.output_grads,
                    grads_output,
                )
            # If an input to the pipeline requires gradient,
            # `torch.autograd.backward` will accumulate the gradient into the
            # `.grad` field of such input
            bwd_kwargs = {
                "stage_output": stage_output,
                "output_grads": grads_output,
                "input_values": input_values,
            }

        grads_input: tuple[torch.Tensor | None, ...] = ()

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
        # Note: grads_input may contain gradients for both args and kwargs (from fwd_cache),
        # Kwargs are local to each stage and don't need gradient transmission.
        # Validate backward output (input gradients) for DTensor metadata
        assert self._stage_meta.inputs is not None  # noqa: S101
        num_fwd_args = len(self._stage_meta.inputs)
        if self._runtime_validate and not self.is_first:
            self._validate_stage_tensors(
                f"Stage {self.stage_index} backward output (input_grads)",
                self._stage_meta.input_grads,
                grads_input[:num_fwd_args],
            )
        self.bwd_cache[bwd_chunk_id] = grads_input[:num_fwd_args]

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
        # skip backward computation if backward is not enabled
        if not self.has_backward:
            return

        if bwd_chunk_id not in self.dw_runner:
            raise AssertionError(
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

    def _get_init_p2p_neighbors_ops(self) -> list[dist.P2POp]:
        """
        Get the operations to initialize the p2p communicators between previous and next stages.
        This is done so by creating a dummy tensor and sending it to the next stage and receiving
        from the previous stage.
        """
        ops: list[dist.P2POp] = []
        next_stage_peer_rank = self.stage_index_to_group_rank.get(self.stage_index + 1)
        prev_stage_peer_rank = self.stage_index_to_group_rank.get(self.stage_index - 1)

        recv_tensor = torch.zeros(1, device=self.device, dtype=torch.float32)
        send_tensor = torch.tensor(
            self.stage_index, device=self.device, dtype=torch.float32
        )
        # forward
        if not self.is_first:
            ops.append(
                dist.P2POp(
                    dist.irecv,
                    recv_tensor,
                    group_peer=prev_stage_peer_rank,
                    group=self.group,
                )
            )
        if not self.is_last:
            ops.append(
                dist.P2POp(
                    dist.isend,
                    send_tensor,
                    group_peer=next_stage_peer_rank,
                    group=self.group,
                )
            )

        # backward
        if not self.is_first:
            ops.append(
                dist.P2POp(
                    dist.isend,
                    send_tensor,
                    group_peer=prev_stage_peer_rank,
                    group=self.group,
                )
            )
        if not self.is_last:
            ops.append(
                dist.P2POp(
                    dist.irecv,
                    recv_tensor,
                    group_peer=next_stage_peer_rank,
                    group=self.group,
                )
            )

        return ops

    def perform_reduce_grad(self, grad_scale_factor: int):
        """
        Called as a part of schedule IR.
        REDUCE_GRAD action is scheduled after all microbatches W, B actions.

        Currently contains "post_backward" functionality for FSDP.
        We can try to extract post_backward in a separate IR action in future.
        """
        # Manually call post backward for FSDP
        if isinstance(self.submod, FSDPModule):
            fsdp_module = self.submod
            fsdp_module.set_is_last_backward(True)
            fsdp_module.set_reshard_after_backward(True)
            fsdp_module.set_requires_gradient_sync(True)

            if isinstance(fsdp_module, ReplicateModule):
                distributed_state = replicate.state(fsdp_module)  # type: ignore[arg-type]
            else:
                distributed_state = fully_shard.state(fsdp_module)  # type: ignore[attr-defined]

            for state in distributed_state._state_ctx.all_states:
                for fsdp_param_group in state._fsdp_param_groups:
                    fsdp_param_group.post_backward()

            # it would be much better if pipelining backward invoked .backward so autograd hooks
            # worked and modules like DDP/FSDP behaved as expected.  Working around this for the time being,
            # we need to call this too to ensure FSDP syncs its grad reduction ops back to the default stream.
            distributed_state._root_post_backward_final_callback()
        # Call gradient scaling at the end of the backward pass
        # NOTE: this must happen after FSDP post_backward is FSDP is enabled
        if grad_scale_factor != 1:
            self.scale_grads(grad_scale_factor)


class _PipelineStage(_PipelineStageBase):
    def __init__(
        self,
        stage_module: torch.nn.Module,
        stage_index: int,
        pipe_info: PipeInfo,
        device: torch.device,
        group: dist.ProcessGroup | None = None,
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
            raise PipeliningMetadataError(
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
        args: tuple[Any, ...] | _StageForwardMeta | None,
        kwargs: dict[str, Any] | None = None,
        has_backward: bool = False,
    ) -> _StageForwardMeta | None:
        """
        Prepare forward infrastructure for traced pipeline.

        Metadata is created directly from graph placeholders with correct
        ``requires_grad`` — received activations get ``requires_grad=True``
        when ``has_backward`` is set, fixing the fact that ``torch.export``
        traces under ``no_grad()``.

        ``_stage_meta.inputs`` is derived from recv infos and aligned with
        ``forward_one_chunk``'s ``composite_args``: positional root inputs
        on the first stage, received activations only on subsequent stages.
        """
        # Step 1: Create recv info for each microbatch.
        # _create_act_recv_info is self-contained: it creates _TensorMeta
        # directly from graph placeholder values with correct requires_grad.
        for chunk in range(num_microbatches):
            self.args_recv_info[chunk] = self._create_act_recv_info()

        # Step 2: Derive _stage_meta.inputs from recv infos.
        # forward_one_chunk builds composite_args as:
        #   - First stage:     args (positional root inputs, excludes kwargs)
        #   - Non-first stages: received activations only (no root kwargs)
        # _stage_meta.inputs must match composite_args for validation.
        recv_infos = self.args_recv_info[0]
        if self.is_first:
            # All placeholders are root args.  Only the first len(args)
            # correspond to positional inputs (composite_args); the rest
            # are kwargs passed separately via composite_kwargs.
            # First stage always receives real tensor args, never _StageForwardMeta.
            if not isinstance(args, tuple):
                raise AssertionError("First stage requires real tensor args")
            n_positional = len(args)
            self._stage_meta.inputs = tuple(
                info.tensor_meta  # type: ignore[misc]
                for info in recv_infos[:n_positional]
            )
        else:
            self._stage_meta.inputs = tuple(
                info.tensor_meta  # type: ignore[misc]
                for info in recv_infos
                if not info.is_root_arg
            )

        # Step 3: Create send info and output metadata.
        self.act_send_info = self._create_act_send_info()

        return None

    def _prepare_backward_infra(
        self,
        num_microbatches: int,
        loss_fn: Callable[..., torch.Tensor] | None = None,
        target: torch.Tensor | None = None,
        received_grad_meta: _StageBackwardMeta | None = None,
    ) -> _StageBackwardMeta | None:
        """
        Prepare backward infrastructure for traced pipeline.
        Derives input_grads metadata from inputs (plain tensors only).

        Note: DTensors are NOT supported in the traced frontend.
        """
        # Derive input_grads from inputs (for plain tensors, grad shape == input shape)
        if self._stage_meta.inputs is None:
            raise PipeliningMetadataError(
                f"Stage {self.stage_index}: inputs metadata required for backward inference."
            )

        self._stage_meta.input_grads = _derive_grad_metas(self._stage_meta.inputs)

        # Setup backward recv info (calls _create_grad_recv_info which sets output_grads).
        # Note: grad_send_info is created lazily in get_bwd_send_ops() since
        # it mirrors args_recv_info (already populated during forward).
        self._setup_backward_recv_info(num_microbatches)

        return None

    def get_stage_index_of_submod(
        self,
        submod_name: str,
    ):
        """
        Given a submodule name, return the stage index of the submodule.
        """
        if submod_name not in self.submod_to_stage_index:
            raise PipeliningMetadataError(f"Stage id of {submod_name} not found")

        return self.submod_to_stage_index[submod_name]

    def _create_act_recv_info(
        self,
    ):
        """
        Create a tuple of `_RecvInfo` for inputs to the stage.

        Self-contained: creates ``_TensorMeta`` directly from graph
        placeholder values with correct ``requires_grad``.
        ``torch.export`` traces under ``no_grad()`` so traced metadata
        always has ``requires_grad=False``; for received activations we
        set ``requires_grad=True`` when ``has_backward`` is set.

        Note: DTensors are NOT supported in the traced frontend.
        """

        def create_recv_tensor(placeholder, arg_node):
            example_value = placeholder.meta["val"]

            # Reject DTensors in traced frontend
            if isinstance(example_value, DTensor):
                raise PipeliningMetadataError(
                    f"{self.log_prefix} DTensor detected in traced pipeline input "
                    f"'{placeholder.name}'. DTensor metadata propagation is NOT "
                    f"supported for the traced frontend (_PipelineStage). "
                    f"Use the manual PipelineStage frontend for full DTensor support."
                )

            if arg_node.op == "placeholder":
                # Root-level placeholder: an input argument to the entire
                # model.  Keep original metadata from the trace.
                return _RecvInfo(
                    input_name=f"root_input_{placeholder.name}",
                    source=None,
                    buffer=None,
                    tensor_meta=_TensorMeta.from_tensor(example_value),
                    is_root_arg=True,
                )

            # Received activation from a previous stage.
            while arg_node.target is operator.getitem:
                arg_node = arg_node.args[0]

            if arg_node.op != "call_module":
                raise PipeliningMetadataError(
                    f"Expecting call_module, got {arg_node.op}"
                )
            src_stage = self.get_stage_index_of_submod(arg_node.name)

            # Create metadata directly with correct requires_grad.
            tensor_meta = _TensorMeta(
                shape=example_value.shape,
                stride=example_value.stride(),
                dtype=example_value.dtype,
                requires_grad=self.has_backward,
            )

            logger.debug(
                "%s Creating recv buffer for input '%s' : %s, %s",
                self.log_prefix,
                placeholder.name,
                tensor_meta.shape,
                tensor_meta.dtype,
            )
            buffer = _make_tensor_from_meta(tensor_meta, self.device)
            if self.has_backward:
                buffer.requires_grad_(True)

            return _RecvInfo(
                arg_node.name,
                src_stage,
                buffer,
                tensor_meta,
            )

        args_recv_info: list[_RecvInfo] = []
        placeholders = filter(  # type: ignore[var-annotated]
            lambda node: node.op == "placeholder",  # type: ignore[arg-type]
            self.submod.graph.nodes,  # type: ignore[arg-type,union-attr]
        )
        # `placeholders` are nodes internal to submod.
        # `self.node.args` are dependency nodes in the outer graph.
        # The two are 1:1.
        for placeholder, arg_node in zip(placeholders, self.node.args, strict=True):
            args_recv_info.append(create_recv_tensor(placeholder, arg_node))

        logger.debug(
            "%s Activation recv / args info: %s", self.log_prefix, args_recv_info
        )
        return tuple(args_recv_info)

    def find_dst_rank(
        self,
        user: fx.Node,
    ) -> int | None:
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
            #   should be re-calculated in case of activation checkpointing
            return None

    def _create_act_send_info(self):
        """
        Create a dict of send info for activations and output metadata.

        Output metadata is created directly with correct ``requires_grad``
        (``torch.export`` traces under ``no_grad()``, so traced values
        always have ``requires_grad=False``; at runtime, stage outputs
        carry ``requires_grad=True`` when training).
        """
        # Output index: List of receiver ranks
        act_send_info: dict[int, list] = {}
        out_idx = 0

        for user in self.node.users:
            if user.target is operator.getitem:
                gi_dsts = act_send_info.setdefault(out_idx, [])
                for gi_user in user.users:
                    dst_rank = self.find_dst_rank(gi_user)
                    if dst_rank is not None:
                        gi_dsts.append(dst_rank)
                out_idx += 1
            else:
                dsts = act_send_info.setdefault(out_idx, [])
                dst_rank = self.find_dst_rank(user)
                if dst_rank is not None:
                    dsts.append(dst_rank)

        output_node = self._get_output_node()
        output_vals: tuple[torch.Tensor] = tuple(
            v.meta["val"] for v in flatten_args(output_node.args)
        )
        # Reject DTensors and create output metadata directly with
        # correct requires_grad.
        output_metas: list[_TensorMeta] = []
        for i, val in enumerate(output_vals):
            if isinstance(val, DTensor):
                raise PipeliningMetadataError(
                    f"{self.log_prefix} DTensor detected in traced pipeline output index {i}. "
                    f"DTensor metadata propagation is NOT supported for the traced frontend "
                    f"(_PipelineStage). Use the manual PipelineStage frontend for full DTensor support."
                )
            output_metas.append(
                _TensorMeta(
                    shape=val.shape,
                    stride=val.stride(),
                    dtype=val.dtype,
                    requires_grad=self.has_backward,
                )
            )
        self._stage_meta.outputs = tuple(output_metas)

        logger.debug("%s Send info: %s", self.log_prefix, act_send_info)
        return act_send_info

    def _get_output_node(self):
        output_nodes = [node for node in self.submod.graph.nodes if node.op == "output"]  # type: ignore[union-attr]
        if len(output_nodes) != 1:
            raise PipeliningMetadataError(
                f"Expected 1 output node, got {len(output_nodes)}"
            )
        output_node = output_nodes[0]
        return output_node

    def _create_grad_recv_info(self, act_send_info: dict) -> tuple[_RecvInfo, ...]:
        """
        Create a tuple of `_RecvInfo` for gradients.
        Reuses output metadata from _stage_meta.outputs (populated by _create_act_send_info).
        """
        if self._stage_meta.outputs is None:
            raise PipeliningMetadataError(
                f"Stage {self.stage_index}: outputs metadata required for grad recv info. "
                f"Ensure _create_act_send_info is called first."
            )

        outputs_meta = self._stage_meta.outputs
        output_grads_metas: list[TensorMeta | None] = []
        grad_recv_infos: list[_RecvInfo] = []

        for out_idx, out_meta in enumerate(outputs_meta):
            dst_list = act_send_info.get(out_idx, [])

            # Determine the source stage for gradients
            grad_src = dst_list[0] if dst_list else self.stage_index + 1

            # Check if this output needs gradients
            if not dst_list or not out_meta.requires_grad:
                output_grads_metas.append(None)
                grad_recv_infos.append(
                    _RecvInfo(
                        input_name=f"recv_grad_for_{self.stage_index}_none_{out_idx}",
                        source=grad_src,
                        buffer=None,
                        tensor_meta=None,
                    )
                )
            else:
                # Derive grad metadata from output metadata (same shape, requires_grad=False)
                grad_meta = _TensorMeta(
                    shape=out_meta.shape,
                    stride=out_meta.stride,
                    dtype=out_meta.dtype,
                    requires_grad=False,
                )
                output_grads_metas.append(grad_meta)

                if len(dst_list) != 1:
                    raise PipeliningMetadataError(
                        "Backward of skip connections not supported yet"
                    )

                logger.debug(
                    "%s Creating grad recv buffer for output %s : %s, %s",
                    self.log_prefix,
                    out_idx,
                    grad_meta.shape,
                    grad_meta.dtype,
                )

                grad_recv_infos.append(
                    _RecvInfo(
                        input_name=f"recv_grad_for_{self.stage_index}_from_{grad_src}",
                        source=grad_src,
                        buffer=_make_tensor_from_meta(grad_meta, self.device),
                        tensor_meta=grad_meta,
                    )
                )

        self._stage_meta.output_grads = tuple(output_grads_metas)
        logger.debug("%s Grad recv info: %s", self.log_prefix, grad_recv_infos)
        return tuple(grad_recv_infos)


# A helper function to create a pipeline stage based on traced pipeline information
def build_stage(
    stage_module: torch.nn.Module,
    stage_index: int,
    pipe_info: PipeInfo,
    device: torch.device,
    group: dist.ProcessGroup | None = None,
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
    """A pipeline stage for pipeline parallelism with sequential model partitioning.

    Supports both **static** and **dynamic** metadata inference:

    Static mode:
        All of ``input_args``, ``output_args`` (and ``input_grads``/``output_grads``
        when DTensors are present) are provided at construction time.

    Dynamic mode:
        Metadata is inferred from the first microbatch at runtime; any
        statically provided args are used for validation only.

    Args:
        submodule: The ``nn.Module`` wrapped by this stage.
        stage_index: Zero-based stage ID.
        num_stages: Total number of stages in the pipeline.
        device: Device this stage runs on.
        input_args: Example input tensors (single tensor or tuple). Optional.
        output_args: Example output tensors. Optional.
        output_grads: Example output gradients (received from next stage). Optional.
        input_grads: Example input gradients (sent to previous stage). Optional.
        group: Process group for P2P communication. Defaults to the
            world process group.
        dw_builder: Builder for deferred weight-update runners used by
            zero-bubble (F/I/W) schedules.
        get_mesh: `GetMeshCallback` used during
            dynamic DTensor inference. Ignored in fully static DTensor mode.
    """

    def __init__(
        self,
        submodule: nn.Module,
        stage_index: int,
        num_stages: int,
        device: torch.device,
        input_args: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
        output_args: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
        output_grads: torch.Tensor | tuple[torch.Tensor | None, ...] | None = None,
        input_grads: torch.Tensor | tuple[torch.Tensor | None, ...] | None = None,
        group: dist.ProcessGroup | None = None,
        dw_builder: Callable[[], Callable[..., None]] | None = None,
        get_mesh: GetMeshCallback | None = None,
    ):
        super().__init__(submodule, stage_index, num_stages, device, group, dw_builder)

        self._mesh_cache = _MeshCache(get_mesh_cb=get_mesh)
        self._inference_mode: InferenceMode | None = None
        self._fwd_outputs_for_bwd_meta: tuple[torch.Tensor, ...] | None = None
        self._fwd_inputs_for_bwd_meta: tuple[torch.Tensor, ...] | None = None
        self._fwd_kwargs_tensors_for_bwd_meta: tuple[torch.Tensor, ...] | None = None

        # Validate and normalize args to tuples
        inputs = validate_and_normalize_to_tuple(input_args)
        outputs = validate_and_normalize_to_tuple(output_args)
        in_grads = validate_and_normalize_to_tuple(input_grads, allow_none=True)
        out_grads = validate_and_normalize_to_tuple(output_grads, allow_none=True)

        self._user_meta = _StageMeta(
            inputs=extract_tensor_metas(inputs),
            outputs=extract_tensor_metas(outputs),
            input_grads=extract_tensor_metas(in_grads, allow_none=True),
            output_grads=extract_tensor_metas(out_grads, allow_none=True),
        )

        # Cache meshes from user-provided DTensors
        for args in (inputs, outputs, in_grads, out_grads):
            if args is not None:
                self._mesh_cache.update_from_tensors(args)

        # Validate DTensor↔grad correspondence independently for inputs and outputs
        if self._user_meta.has_dtensors():
            if inputs and in_grads:
                validate_static_arg_grad_correspondence(
                    self.stage_index, inputs, in_grads, is_input=True
                )
            if outputs and out_grads:
                validate_static_arg_grad_correspondence(
                    self.stage_index, outputs, out_grads, is_input=False
                )

    def _recv_meta(self, src_stage: int) -> Any:
        """Receive metadata object from a stage on a different rank via P2P."""
        objects: list[Any] = [None]
        dist.recv_object_list(
            objects,
            src=self._resolve_peer_global_rank(src_stage),
            group=self.group,
            device=self.device,
            use_batch=True,
        )
        if len(objects) != 1:
            raise PipeliningMetadataError(
                f"Expected exactly one object to be received but got: {len(objects)}"
            )
        return objects[0]

    def _send_meta(self, meta: Any, dst_stage: int) -> None:
        """Send metadata object to a stage on a different rank via P2P."""
        dist.send_object_list(
            [meta],
            dst=self._resolve_peer_global_rank(dst_stage),
            group=self.group,
            device=self.device,
            use_batch=True,
        )

    def _is_same_rank(self, other_stage: int) -> bool:
        """Check if another stage is on the same rank as this stage."""
        return self.stage_index_to_group_rank[other_stage] == self.group_rank

    def _warmup_forward_vote(
        self, has_backward: bool, received_acc: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward phase of the warm-up vote protocol (stage 0 → N−1).

        Each stage computes a vote (1 = STATIC, 0 = DYNAMIC) based on
        ``InferenceMode.needs_dynamic``, multiplies it with the accumulated
        product from the previous stage, and forwards the result to the next
        stage.  The final product at stage N−1 is 1 iff *every* stage voted
        STATIC.

        Args:
            has_backward: Whether the schedule includes a backward pass.
            received_acc: Accumulated product tensor from the previous
                same-rank stage (V-schedule), or ``None`` for the first
                stage / cross-rank.

        Returns:
            The accumulated product tensor after this stage's vote.
        """
        my_vote = 0 if InferenceMode.needs_dynamic(self._user_meta, has_backward) else 1

        my_vote_t = torch.tensor([my_vote], dtype=torch.int32, device=self.device)

        if self.is_first:
            acc = my_vote_t
        elif self._is_same_rank(self.stage_index - 1):
            assert received_acc is not None  # noqa: S101
            acc = received_acc * my_vote_t
        else:
            peer_global = self._resolve_peer_global_rank(self.stage_index - 1)
            acc = torch.zeros(1, dtype=torch.int32, device=self.device)
            dist.recv(acc, src=peer_global, group=self.group)
            acc = acc * my_vote_t

        if not self.is_last and not self._is_same_rank(self.stage_index + 1):
            peer_global = self._resolve_peer_global_rank(self.stage_index + 1)
            dist.send(acc, dst=peer_global, group=self.group)

        return acc

    def _warmup_backward_result(
        self, received_result: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Backward phase of the warm-up vote protocol (stage N−1 → 0).

        Propagates the final accumulated product (computed in the forward
        phase) back through the pipeline so every stage learns the global
        inference mode.

        Args:
            received_result: Result tensor from the next same-rank stage
                (V-schedule), or ``None`` for the last stage / cross-rank.

        Returns:
            The global vote result tensor for this stage.
        """
        if self.is_last or self._is_same_rank(self.stage_index + 1):
            assert received_result is not None  # noqa: S101
            result = received_result
        else:
            peer_global = self._resolve_peer_global_rank(self.stage_index + 1)
            result = torch.zeros(1, dtype=torch.int32, device=self.device)
            dist.recv(result, src=peer_global, group=self.group)

        if not self.is_first and not self._is_same_rank(self.stage_index - 1):
            peer_global = self._resolve_peer_global_rank(self.stage_index - 1)
            dist.send(result, dst=peer_global, group=self.group)

        return result

    def _compute_outputs(
        self,
        *args: torch.Tensor,
        module: torch.nn.Module,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor] | None:
        """Compute outputs of the submodule."""
        return module(*args, **kwargs)

    def _compute_input_grads(
        self,
        outputs: list[torch.Tensor],
        all_fwd_inputs: list[torch.Tensor],
        grad_outputs: list[torch.Tensor | None] | None = None,
    ) -> tuple[torch.Tensor | None, ...]:
        """Compute input gradients via :func:`_autograd_grad_for_inputs`."""
        return _autograd_grad_for_inputs(
            outputs,
            all_fwd_inputs,
            grad_outputs,
        )

    def _to_tensor(self, arg: torch.Tensor | TensorMeta) -> torch.Tensor:
        """Convert a tensor or metadata to a real tensor on ``self.device``.

        Real tensors are detached and re-set requires_grad to create a fresh
        autograd leaf, isolating metadata inference from the user's graph.
        TensorMeta is materialized as an empty tensor (or DTensor via mesh cache).
        """
        if isinstance(arg, torch.Tensor):
            return arg.detach().requires_grad_(arg.requires_grad)
        elif isinstance(arg, TensorMeta):
            if isinstance(arg, _DTensorMeta):
                mesh = self._mesh_cache.get_mesh(arg.mesh_cache_key)
                return arg.to_dtensor(self.device, mesh)
            else:
                return arg.to_tensor(self.device)
        else:
            raise PipeliningMetadataError(
                f"Unsupported type {type(arg)} for _to_tensor: {arg}"
            )

    def _ones_from_metadata(self, meta: TensorMeta) -> torch.Tensor:
        """Create a ones tensor from metadata for backward inference grad_outputs."""
        local_ones = torch.ones(
            meta.shape,
            dtype=meta.dtype,
            device=self.device,
        )
        if isinstance(meta, _DTensorMeta):
            mesh = self._mesh_cache.get_mesh(meta.mesh_cache_key)
            return DTensor.from_local(
                local_ones,
                device_mesh=mesh,
                placements=meta.placements,
                shape=meta.global_shape,
                stride=meta.global_stride,
                run_check=False,
            )
        return local_ones

    def _forward_metadata_inference(
        self,
        args: tuple[torch.Tensor, ...] | _StageForwardMeta | None,
        kwargs: dict[str, Any] | None = None,
        has_backward: bool = False,
    ) -> _StageForwardMeta | None:
        """Run forward metadata inference (Stage 0 → N).

        Args:
            args: Real tensors (first stage), ``_StageForwardMeta``
                (same-rank), or ``None`` (cross-rank P2P).
            kwargs: Keyword arguments forwarded to the submodule.
            has_backward: Whether backward inference follows.

        Returns:
            ``_StageForwardMeta`` for the next stage, or ``None`` if sent via P2P.
        """
        kwargs = kwargs or {}

        # === RECEIVE: Get input metadata and create meta tensors ===
        if self.is_first:
            # First stage: extract metadata from real tensors
            if args is None or isinstance(args, _StageForwardMeta):
                raise PipeliningMetadataError(
                    f"Stage {self.stage_index}: First stage requires real tensors, "
                    f"got {type(args).__name__}."
                )
            tensor_args = validate_and_normalize_to_tuple(args)
            assert tensor_args is not None  # noqa: S101
            self._stage_meta.inputs = extract_tensor_metas(tensor_args)
            inference_args = tuple(self._to_tensor(a) for a in tensor_args)
        elif self._is_same_rank(self.stage_index - 1):
            # Same-rank: _StageForwardMeta passed via argument
            if not isinstance(args, _StageForwardMeta):
                raise PipeliningMetadataError(
                    f"Stage {self.stage_index}: Expected _StageForwardMeta from same-rank "
                    f"previous stage, got {type(args).__name__}."
                )
            self._stage_meta.inputs = args.forward_metas
            inference_args = tuple(self._to_tensor(m) for m in args.forward_metas)
        else:
            # Cross-rank: receive _StageForwardMeta via P2P
            recv_meta = self._recv_meta(self.stage_index - 1)
            if not isinstance(recv_meta, _StageForwardMeta):
                raise PipeliningMetadataError(
                    f"Stage {self.stage_index}: Expected _StageForwardMeta from P2P, "
                    f"got {type(recv_meta).__name__}."
                )
            self._stage_meta.inputs = recv_meta.forward_metas
            inference_args = tuple(self._to_tensor(m) for m in recv_meta.forward_metas)

        inference_kwargs = {
            k: self._to_tensor(v) if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
        }

        # Isolate metadata inference from user's grad context.
        # has_backward → enable_grad() so backward tracing sees grad_fn;
        # no backward → no_grad() for cross-rank consistency.
        ctx = torch.enable_grad() if has_backward else torch.no_grad()
        with ctx:
            outputs = self._compute_outputs(
                *inference_args, module=self.submod, **inference_kwargs
            )

        # Normalize outputs to tuple
        outputs = validate_and_normalize_to_tuple(outputs)

        self._stage_meta.outputs = extract_tensor_metas(outputs)

        # Store for backward metadata inference (always, even during eval)
        fwd_kwargs_tensors = tuple(
            v for v in flatten_args(inference_kwargs) if isinstance(v, torch.Tensor)
        )
        self._fwd_outputs_for_bwd_meta = outputs
        self._fwd_inputs_for_bwd_meta = inference_args
        self._fwd_kwargs_tensors_for_bwd_meta = fwd_kwargs_tensors

        # === SEND: Pass output metadata to next stage ===
        if self._stage_meta.outputs is None:
            raise PipeliningMetadataError(
                f"Stage {self.stage_index}: output metadata is required for forward inference."
            )
        fwd_meta = _StageForwardMeta(forward_metas=self._stage_meta.outputs)

        if self.is_last or self._is_same_rank(self.stage_index + 1):
            # Same-rank or last: return for caller to pass
            return fwd_meta
        else:
            # Cross-rank: send via P2P
            self._send_meta(fwd_meta, self.stage_index + 1)
            return None

    def _backward_metadata_inference(
        self,
        loss_fn: Callable[..., torch.Tensor] | None = None,
        target: torch.Tensor | None = None,
        received_grad_meta: _StageBackwardMeta | None = None,
        loss_kwargs: dict[str, Any] | None = None,
    ) -> _StageBackwardMeta | None:
        """Run backward metadata inference (Stage N → 0).

        Args:
            loss_fn: Loss function (required for the last stage).
            target: Target tensor (required for the last stage).
            received_grad_meta: Grad metadata from next same-rank stage
                (V-schedule only).
            loss_kwargs: Extra keyword arguments forwarded to the loss
                function (e.g. ``global_valid_tokens``).

        Returns:
            ``_StageBackwardMeta`` for the previous stage, or ``None`` if sent via P2P.
        """
        fwd_outputs = self._fwd_outputs_for_bwd_meta
        fwd_inputs = self._fwd_inputs_for_bwd_meta
        if fwd_outputs is None or fwd_inputs is None:
            raise PipeliningMetadataError(
                "Backward metadata inference requires forward metadata inference to run first"
            )
        kwargs_tensors = self._fwd_kwargs_tensors_for_bwd_meta or ()
        all_fwd_inputs = list(fwd_inputs) + list(kwargs_tensors)
        # Clear temporary storage early — local refs are sufficient from here
        self._fwd_outputs_for_bwd_meta = None
        self._fwd_inputs_for_bwd_meta = None
        self._fwd_kwargs_tensors_for_bwd_meta = None
        # === RECEIVE: Get output grad metadata (except last stage) ===
        if self.is_last:
            if loss_fn is None or target is None:
                raise PipeliningMetadataError(
                    f"Stage {self.stage_index}: loss_fn and target required for last stage"
                )
            inference_target = self._to_tensor(target)
            loss = loss_fn(
                fwd_outputs[0] if len(fwd_outputs) == 1 else fwd_outputs,
                inference_target,
                **(loss_kwargs or {}),
            )
            self._stage_meta.output_grads = None
            all_input_grads = self._compute_input_grads(
                [loss],
                all_fwd_inputs,
            )
        else:
            # Non-last stage: receive grad metadata from next stage
            if self._is_same_rank(self.stage_index + 1):
                # Same-rank: _StageBackwardMeta passed via argument
                if not isinstance(received_grad_meta, _StageBackwardMeta):
                    raise PipeliningMetadataError(
                        f"Stage {self.stage_index}: Expected _StageBackwardMeta from same-rank "
                        f"next stage, got {type(received_grad_meta).__name__}."
                    )
                self._stage_meta.output_grads = received_grad_meta.backward_metas
            else:
                # Cross-rank: receive _StageBackwardMeta via P2P
                recv_meta = self._recv_meta(self.stage_index + 1)
                if not isinstance(recv_meta, _StageBackwardMeta):
                    raise PipeliningMetadataError(
                        f"Stage {self.stage_index}: Expected _StageBackwardMeta from P2P, "
                        f"got {type(recv_meta).__name__}."
                    )
                self._stage_meta.output_grads = recv_meta.backward_metas

            # === COMPUTE: Build grad_outputs and compute input grads ===
            # Extract output tensors and corresponding grad_outputs from metadata
            # Must iterate together to maintain alignment
            if self._stage_meta.output_grads is None:
                raise PipeliningMetadataError(
                    f"Stage {self.stage_index}: output_grads metadata is required for backward inference."
                )
            stage_output_grad_metas = self._stage_meta.output_grads

            filtered_fwd_outputs: list[torch.Tensor] = []
            filtered_output_grads: list[torch.Tensor | None] = []

            for idx, (fwd_out, grad_meta) in enumerate(
                zip(fwd_outputs, stage_output_grad_metas, strict=True)
            ):
                # Match _backward.py behavior: skip if output doesn't require grad AND has no grad_fn
                if not fwd_out.requires_grad:
                    if grad_meta is not None:
                        raise PipeliningMetadataError(
                            f"Stage {self.stage_index}: output {idx} requires_grad=False, "
                            f"but output_grads metadata is provided: {grad_meta}."
                        )
                    continue
                filtered_fwd_outputs.append(fwd_out)
                # For outputs that require grad, include them even if grad_meta is None
                # (runtime passes None grad_outputs to autograd.backward in this case)
                filtered_output_grads.append(
                    self._ones_from_metadata(grad_meta) if grad_meta else None
                )

            if filtered_fwd_outputs:
                all_input_grads = self._compute_input_grads(
                    filtered_fwd_outputs, all_fwd_inputs, filtered_output_grads
                )
                # Free intermediate references early
                filtered_fwd_outputs.clear()
                filtered_output_grads.clear()
                all_fwd_inputs.clear()
                # Only positional input grads flow to previous stage
            else:
                all_input_grads = tuple(None for _ in range(len(all_fwd_inputs)))

        input_grads = all_input_grads[: len(fwd_inputs)]
        self._stage_meta.input_grads = tuple(
            extract_tensor_meta(g) if isinstance(g, torch.Tensor) else None
            for g in input_grads
        )

        # === SEND: Pass input grad metadata to previous stage ===
        bwd_meta = _StageBackwardMeta(backward_metas=self._stage_meta.input_grads)

        if self.is_first or self._is_same_rank(self.stage_index - 1):
            # First rank or Same-rank: return for caller to pass
            return bwd_meta
        else:
            # Cross-rank: send via P2P
            self._send_meta(bwd_meta, self.stage_index - 1)
            return None

    def _post_metadata_inference_cleanup(self) -> None:
        """Clean up FSDP side effects (unsharded params, stale grads, stored
        tensors) after metadata inference with real tensors.
        """
        # Clear stored inference tensors (frees autograd graph + activations)
        self._fwd_outputs_for_bwd_meta = None
        self._fwd_inputs_for_bwd_meta = None
        self._fwd_kwargs_tensors_for_bwd_meta = None

        # Metadata inference runs real fwd/bwd, which unshards FSDP params and
        # accumulates grads.  Reshard to free memory and clear stale grads.
        for module in self.submod.modules():
            if isinstance(module, FSDPModule):
                module.reshard()
                for param in module.parameters():
                    param.grad = None

    def _prepare_backward_infra(
        self,
        num_microbatches: int,
        loss_fn: Callable[..., torch.Tensor] | None = None,
        target: torch.Tensor | None = None,
        received_grad_meta: "_StageBackwardMeta | None" = None,
        loss_kwargs: dict[str, Any] | None = None,
    ) -> "_StageBackwardMeta | None":
        """Run backward metadata inference and prepare backward infrastructure.

        Returns:
            ``_StageBackwardMeta`` for the previous same-rank stage, or ``None``.
        """
        grad_meta_result: _StageBackwardMeta | None = None
        if self._inference_mode == InferenceMode.DYNAMIC:
            # DYNAMIC mode: run backward metadata inference
            # received_grad_meta is used for same-rank V-schedule stages
            grad_meta_result = self._backward_metadata_inference(
                loss_fn=loss_fn,
                target=target,
                received_grad_meta=received_grad_meta,
                loss_kwargs=loss_kwargs,
            )
            # Validate dynamically inferred metadata against user-provided metadata
            self._validate_inferred_metadata()
        else:
            # STATIC mode: metadata comes from user inputs, no validation needed
            self._stage_meta.input_grads = self._user_meta.input_grads
            self._stage_meta.output_grads = self._user_meta.output_grads
            # For STATIC mode with plain tensors, if output_grads is not set but
            # we have outputs, derive output_grads from outputs.
            # (gradient shape == output shape, but requires_grad=False for gradients)
            if self._stage_meta.output_grads is None:
                if self._stage_meta.outputs is None:
                    raise PipeliningMetadataError(
                        f"Stage {self.stage_index}: output metadata is required for backward inference."
                    )
                self._stage_meta.output_grads = _derive_grad_metas(
                    self._stage_meta.outputs
                )
            # Similarly, derive input_grads from inputs if not provided
            if self._stage_meta.input_grads is None:
                if self._stage_meta.inputs is None:
                    raise PipeliningMetadataError(
                        f"Stage {self.stage_index}: input metadata is required for backward inference."
                    )
                self._stage_meta.input_grads = _derive_grad_metas(
                    self._stage_meta.inputs
                )

        # Note: grad_send_info is created lazily in get_bwd_send_ops() since
        # it mirrors args_recv_info (already populated during forward).
        self._setup_backward_recv_info(num_microbatches)
        return grad_meta_result

    def _validate_inferred_metadata(self) -> None:
        """Validate dynamically inferred metadata against user-provided metadata."""
        pairs = [
            (self._user_meta.inputs, self._stage_meta.inputs, "input"),
            (self._user_meta.outputs, self._stage_meta.outputs, "output"),
            (self._user_meta.input_grads, self._stage_meta.input_grads, "input_grad"),
            (
                self._user_meta.output_grads,
                self._stage_meta.output_grads,
                "output_grad",
            ),
        ]
        for user_val, stage_val, label in pairs:
            if user_val and stage_val:
                validate_tensors_metadata(
                    f"Stage {self.stage_index} {label}",
                    user_val,
                    stage_val,
                    warn_on_mismatch=True,
                )

    def _prepare_forward_infra(
        self,
        num_microbatches: int,
        args: tuple[Any, ...] | _StageForwardMeta | None,
        kwargs: dict[str, Any] | None = None,
        has_backward: bool = False,
    ) -> _StageForwardMeta | None:
        """Prepare the stage infrastructure for forward pass.

        Returns:
            ``_StageForwardMeta`` for next stage (same-rank), or ``None`` if sent via P2P.
        """
        if self._inference_mode is None:
            raise PipeliningMetadataError(
                f"Stage {self.stage_index}: inference mode not set. "
                f"Run warmup vote protocol first."
            )

        fwd_meta_output: _StageForwardMeta | None = None

        if self._inference_mode == InferenceMode.DYNAMIC:
            # DYNAMIC mode: run forward metadata inference
            # args may be _StageForwardMeta for same-rank V-schedule stages
            fwd_meta_output = self._forward_metadata_inference(
                args, kwargs, has_backward
            )
            # Validate dynamically inferred metadata against user-provided metadata
            self._validate_inferred_metadata()
        # STATIC mode: metadata comes from user inputs, no validation needed
        else:
            self._stage_meta.inputs = self._user_meta.inputs
            self._stage_meta.outputs = self._user_meta.outputs

        # Setup recv and send info
        self._setup_forward_recv_info(num_microbatches, has_backward)
        self._setup_forward_send_info()

        return fwd_meta_output

    def _setup_forward_recv_info(
        self, num_microbatches: int, has_backward: bool
    ) -> None:
        """Setup receive info for forward pass."""
        if self._stage_meta.inputs is None:
            raise PipeliningMetadataError(
                f"Stage {self.stage_index}: inputs metadata required for recv info."
            )
        for chunk_id in range(num_microbatches):
            if self.is_first:
                # First stage: all inputs are root arguments (no recv needed)
                self.args_recv_info[chunk_id] = tuple(
                    _RecvInfo(
                        input_name=f"root_input_{idx}",
                        source=None,
                        buffer=None,
                        tensor_meta=meta,
                        is_root_arg=True,
                    )
                    for idx, meta in enumerate(self._stage_meta.inputs)
                )
            else:
                # Non-first stages: receive from previous stage
                self.args_recv_info[chunk_id] = tuple(
                    _RecvInfo(
                        input_name=f"recv_for_{self.stage_index}_from_{self.stage_index - 1}",
                        source=self.stage_index - 1,
                        buffer=_make_tensor_from_meta(meta, self.device),
                        tensor_meta=meta,
                    )
                    for meta in self._stage_meta.inputs
                )

    def _setup_forward_send_info(self) -> None:
        """Setup send info for forward pass."""
        self.act_send_info: dict[int, list] = {}
        if self._stage_meta.outputs is None:
            raise PipeliningMetadataError(
                f"Stage {self.stage_index}: outputs metadata required for recv info."
            )
        for idx in range(len(self._stage_meta.outputs)):
            self.act_send_info[idx] = [self.stage_index + 1] if not self.is_last else []

    def _create_grad_recv_info(
        self,
        act_send_info: dict,
    ) -> tuple[_RecvInfo, ...]:
        grad_recv_infos: list[_RecvInfo] = []
        if not self.is_last:
            # Ensure output_grads metadata is available
            if self._stage_meta.output_grads is None:
                raise PipeliningMetadataError(
                    f"Stage {self.stage_index}: output_grads metadata is required for "
                    f"creating grad recv info. Ensure backward metadata is populated."
                )

            # Receiving gradients from multiple sources is not supported
            # hence we only take the first destination
            # Use a helper function to safely extract the metadata
            output_grads = self._stage_meta.output_grads
            for idx, dst_list in act_send_info.items():
                if dst_list is None:
                    raise PipeliningMetadataError(
                        f"Stage {self.stage_index}: output {idx} is not sent to any stage."
                    )
                src = dst_list[0]
                grad_meta = output_grads[idx]
                grad_recv_infos.append(
                    _RecvInfo(
                        input_name=f"recv_grad_for_{self.stage_index}_from_{src}",
                        source=src,
                        buffer=_make_tensor_from_meta(grad_meta, self.device)
                        if grad_meta
                        else None,
                        tensor_meta=grad_meta,
                    )
                )
        return tuple(grad_recv_infos)
