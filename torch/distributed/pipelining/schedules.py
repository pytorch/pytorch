# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

import copy
import csv
import itertools
import logging
import re
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Callable
from enum import Enum
from functools import lru_cache
from typing import Any, NamedTuple

import torch
import torch.distributed as dist
from torch._dynamo import OptimizedModule
from torch.nn.modules.loss import _Loss
from torch.profiler import record_function

from ._utils import generate_rank_to_stage_mapping, generate_stage_to_rank_mapping
from .microbatch import merge_chunks, split_args_kwargs_into_chunks, TensorChunkSpec
from .stage import _PipelineStageBase


__all__ = [
    "get_schedule_class",
    "PipelineScheduleSingle",
    "PipelineScheduleMulti",
    "Schedule1F1B",
    "ScheduleGPipe",
    "ScheduleInterleaved1F1B",
    "ScheduleLoopedBFS",
    "ScheduleInterleavedZeroBubble",
    "ScheduleZBVZeroBubble",
    "ScheduleDualPipeV",
]

logger = logging.getLogger(__name__)


class _ComputationType(str, Enum):
    # TODO(whc) rename to _ActType?
    FORWARD = "F"
    BACKWARD_INPUT = "I"
    BACKWARD_WEIGHT = "W"
    UNSHARD = "UNSHARD"
    RESHARD = "RESHARD"
    SEND_F = "SEND_F"
    RECV_F = "RECV_F"
    SEND_B = "SEND_B"
    RECV_B = "RECV_B"
    FULL_BACKWARD = "B"
    OVERLAP_F_B = "OVERLAP_F_B"
    REDUCE_GRAD = "REDUCE_GRAD"

    @staticmethod
    def from_str(action: str) -> "_ComputationType":
        try:
            return _ComputationType(action)
        except ValueError as exc:
            raise RuntimeError(f"Invalid computation type {action}") from exc


FORWARD = _ComputationType.FORWARD
BACKWARD_INPUT = _ComputationType.BACKWARD_INPUT
BACKWARD_WEIGHT = _ComputationType.BACKWARD_WEIGHT
UNSHARD = _ComputationType.UNSHARD
RESHARD = _ComputationType.RESHARD
SEND_F = _ComputationType.SEND_F
RECV_F = _ComputationType.RECV_F
SEND_B = _ComputationType.SEND_B
RECV_B = _ComputationType.RECV_B
FULL_BACKWARD = _ComputationType.FULL_BACKWARD
OVERLAP_F_B = _ComputationType.OVERLAP_F_B
REDUCE_GRAD = _ComputationType.REDUCE_GRAD

# Convenience shorthand for compute actions only since they are used in 'simple schedule format'
F = FORWARD
I = BACKWARD_INPUT
W = BACKWARD_WEIGHT
B = FULL_BACKWARD

# Helper to parse an action string like 1F0 into a tuple of (stage_index, computation_type, microbatch_index)
_action_regex = re.compile(
    r"(\d+)(F|I|B|W|UNSHARD|RESHARD|REDUCE_GRAD|SEND_F|RECV_F|SEND_B|RECV_B)(\d*)"
)


class _Action(NamedTuple):
    stage_index: int
    computation_type: _ComputationType
    microbatch_index: int | None = None
    sub_actions: tuple["_Action", ...] | None = None

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        if self.sub_actions is not None:
            # Use recursive repr for sub_actions
            sub_action_reprs = [repr(sub_action) for sub_action in self.sub_actions]
            return f"({';'.join(sub_action_reprs)}){self.computation_type.value}"
        else:
            repr_str = str(self.stage_index)
            # Use .value to get the short string (e.g., "F", "B") instead of the full enum name
            repr_str += self.computation_type.value
            if self.microbatch_index is not None:
                repr_str += str(self.microbatch_index)
            return repr_str

    @property
    def is_compute_op(self) -> bool:
        return self.computation_type in (
            FORWARD,
            FULL_BACKWARD,
            BACKWARD_INPUT,
            BACKWARD_WEIGHT,
            OVERLAP_F_B,
        )

    @staticmethod
    def from_str(action_string: str):
        """
        Reverse of __repr__

        String should be formatted as [stage][action type][(microbatch)]
            e.g. `2F0`, `1UNSHARD`, `3SEND_F1`
        """
        action_string = action_string.strip()
        if action_string == "":
            return None

        # Check for sub_actions format: [sub_action1;sub_action2;...]ComputationType
        if action_string.startswith("(") and ")" in action_string:
            # Find the closing bracket to separate sub_actions from computation type
            bracket_end = action_string.find(")")
            sub_part = action_string[
                1:bracket_end
            ]  # Remove '[' and get content before ']'
            computation_type_part = action_string[
                bracket_end + 1 :
            ]  # Get part after ']'

            # Parse sub_actions
            sub_actions = []
            if sub_part.strip():
                for sub_str in sub_part.split(";"):
                    sub_action = _Action.from_str(sub_str.strip())
                    if sub_action is not None:
                        sub_actions.append(sub_action)

            # For sub_actions format, we create an action with just the computation type
            # The stage_index and microbatch_index are not meaningful for the container action
            return _Action(
                stage_index=-1,  # Placeholder, not meaningful for sub_actions container
                computation_type=_ComputationType.from_str(computation_type_part),
                microbatch_index=None,
                sub_actions=tuple(sub_actions) if sub_actions else None,
            )

        # Handle regular single action format
        if match := _action_regex.match(action_string):
            stage_index, computation_type, microbatch_index = match.groups()
            return _Action(
                int(stage_index),
                _ComputationType.from_str(computation_type),
                int(microbatch_index) if len(microbatch_index) else None,
            )
        elif action_string == "":
            return None
        raise RuntimeError(
            f"Invalid action string: {action_string}, should be formatted as [stage][action type][(microbatch)] e.g. 2F0"
        )


@lru_cache
def _get_profiler_function_name(action: _Action) -> str:
    return f"PP:{str(action)}"


def _format_pipeline_order(
    pipeline_order: dict[int, list[_Action | None]],
    error_step_number: int | None = None,
) -> str:
    """
    Formats the pipeline order in a timestep (row) x rank (column) grid of actions
    and returns the formatted string.

    If `error_step_number` is passed in, an additional label will be added to signify which step
    that it is erroring on.
    """

    # don't mutate the original
    pipeline_order = copy.deepcopy(pipeline_order)

    # Replace None with ""
    for rank in pipeline_order:
        for i in range(len(pipeline_order[rank])):
            if pipeline_order[rank][i] is None:
                # TODO make a real 'None action' that prints as empty string and make mypy happy
                pipeline_order[rank][i] = ""  # type: ignore[call-overload]

    # Calculate the maximum number of steps across all ranks
    num_steps = max(len(actions) for actions in pipeline_order.values())
    step_labels = [
        "Step " + str(i).zfill(len(str(num_steps - 1))) for i in range(num_steps)
    ]
    # Sorting the dictionary by keys and retrieving values in that order
    rank_actions = [
        pipeline_order.get(key, [""] * num_steps) for key in sorted(pipeline_order)
    ]
    # Transpose the list of lists (rows to columns)
    # pyrefly: ignore [no-matching-overload]
    transposed_actions = list(itertools.zip_longest(*rank_actions, fillvalue=""))
    # Generate column labels for ranks
    num_ranks = len(pipeline_order)
    rank_labels = ["Rank " + str(i) for i in range(num_ranks)]
    # Calculate the maximum length of each column, considering labels
    max_lengths = [
        max(len(str(item)) if item is not None else 0 for item in col)
        for col in zip(step_labels, *transposed_actions)
    ]
    # Format the header row with rank labels
    header_row = " " * (len(step_labels[0]) + 2) + " ".join(
        f"{label:<{max_lengths[i]}}" for i, label in enumerate(rank_labels)
    )
    # Format each row with its corresponding label
    formatted_rows = [
        f"{label}: "
        + " ".join(f"{str(item):<{max_lengths[i]}}" for i, item in enumerate(row))
        + (
            " <-- ERROR HERE"
            if error_step_number is not None
            and int(label.split()[1]) == error_step_number
            else ""
        )
        for label, row in zip(step_labels, transposed_actions)
    ]
    # Join the rows into a single string
    formatted_table = header_row + "\n" + "\n".join(formatted_rows) + "\n"
    return formatted_table


class _PipelineSchedule(ABC):
    def __init__(
        self,
        n_microbatches: int,
        loss_fn: Callable[..., torch.Tensor] | None = None,
        args_chunk_spec: tuple[TensorChunkSpec, ...] | None = None,
        kwargs_chunk_spec: dict[str, TensorChunkSpec] | None = None,
        output_merge_spec: dict[str, Any] | tuple[Any] | None = None,
        scale_grads: bool = True,
    ):
        # From arguments
        self._n_microbatches = n_microbatches
        self._loss_fn = loss_fn

        # See documentation in `PipelineScheduleSingle` / `PipelineScheduleMulti`
        self.scale_grads = scale_grads

        # Chunking specification for positional inputs. (default: `None`)
        self._args_chunk_spec = args_chunk_spec
        # Chunking specification for keyword inputs. (default: `None`)
        self._kwargs_chunk_spec = kwargs_chunk_spec
        self._output_merge_spec = output_merge_spec
        """
        # args_chunk_spec and kwargs_chunk_spec specify how to chunk inputs.
        # They are used to convert batch to microbatches in `step(x)`.  See
        # `TensorChunkSpec` for helper methods for creating them.
        """

        # Derived
        self._has_backward = self._loss_fn is not None

        # Holds the losses for each microbatch.
        self._internal_losses: list[torch.Tensor] = []
        logger.info("Using %s", self.__class__.__name__)

    def _maybe_compute_loss(self, stage, output, target_mbs, mb_index):
        if stage.is_last and self._loss_fn is not None:
            loss = self._compute_loss(output, target_mbs[mb_index])  # type: ignore[index]
            self._internal_losses.append(loss)

    def _maybe_get_loss(self, stage, mb_index):
        valid_index = 0 <= mb_index < len(self._internal_losses)
        if stage.is_last and self._loss_fn is not None and valid_index:
            return self._internal_losses[mb_index]
        elif len(self._internal_losses) != 0 and not valid_index:
            raise RuntimeError(
                f"Loss for microbatch {mb_index} is not available. "
                f"Available losses for microbatches: {self._internal_losses}"
            )
        else:
            return None

    def _update_losses(self, stages, losses):
        """
        Update the losses to those in the internal state
        """
        # if stages not a list turn into a list
        if not isinstance(stages, list):
            stages = [stages]
        contains_last_stage = any(stage.is_last for stage in stages)

        # Return losses if there is a container passed in
        if contains_last_stage and losses is not None:
            if len(self._internal_losses) != self._n_microbatches:
                raise RuntimeError(
                    f"Expecting {self._n_microbatches} losses but got {len(self._internal_losses)}"
                )

            # Clean external container first
            losses.clear()
            # Copy internal losses to external container
            losses.extend(self._internal_losses)

        self._internal_losses.clear()

    @abstractmethod
    def _step_microbatches(
        self,
        arg_mbs: list | None = None,
        kwarg_mbs: list | None = None,
        target_mbs: list | None = None,
        losses: list | None = None,
        return_outputs: bool = True,
    ):
        """
        Run one iteration of the pipeline schedule with list of microbatches.
        Will go through all the microbatches according to the schedule
        implementation.

        Args:
            microbatches: list of microbatch args.
            return_outputs: whether to return the outputs from the last stage.
        """
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        *args,
        target=None,
        losses: list | None = None,
        return_outputs=True,
        **kwargs,
    ):
        """
        Run one iteration of the pipeline schedule with *whole-batch* input.
        Will chunk the input into microbatches automatically, and go through the
        microbatches according to the schedule implementation.

        args: positional arguments to the model (as in non-pipeline case).
        kwargs: keyword arguments to the model (as in non-pipeline case).
        target: target for the loss function.
        losses: a list to store the losses for each microbatch.
        return_outputs: whether to return the outputs from the last stage.
        """
        raise NotImplementedError

    def eval(self, *args, target=None, losses: list | None = None, **kwargs):
        """
        Run one iteration of the pipeline schedule with *whole-batch* input.
        Will chunk the input into microbatches automatically, and go through the
        microbatches, calling forward only.

        args: positional arguments to the model (as in non-pipeline case).
        kwargs: keyword arguments to the model (as in non-pipeline case).
        target: target values for the loss function.
        losses: a list to store the losses for each microbatch.
        """
        # Save the original has_backward state
        original_has_backward = self._has_backward
        try:
            self._has_backward = False
            return self.step(*args, target=target, losses=losses, **kwargs)
        finally:
            # Restore the original state
            self._has_backward = original_has_backward

    def _check_inputs(
        self,
        arg_mbs: list | None = None,
        kwarg_mbs: list | None = None,
        target_mbs: list | None = None,
        losses: list | None = None,
    ) -> tuple[list, list]:
        """
        Pre-process/check inputs
        """

        def check_type_and_len(mbs, name: str):
            if not isinstance(mbs, list):
                raise TypeError(f"{name} must be a list but got a {type(mbs)}")
            if len(mbs) != self._n_microbatches:
                raise ValueError(
                    f"Expecting {self._n_microbatches} {name} but got {len(mbs)}"
                )

        if arg_mbs is not None:
            check_type_and_len(arg_mbs, "arg_mbs")
        else:
            arg_mbs = [()] * self._n_microbatches

        if kwarg_mbs is not None:
            check_type_and_len(kwarg_mbs, "kwarg_mbs")
        else:
            kwarg_mbs = [{}] * self._n_microbatches

        if target_mbs is not None:
            check_type_and_len(target_mbs, "target_mbs")

        if losses is not None:
            if not isinstance(losses, list):
                raise TypeError(f"losses must be a list but got a {type(losses)}")

        return arg_mbs, kwarg_mbs

    def _compute_loss(self, output, target):
        return self._loss_fn(output, target)  # type: ignore[misc]

    def _split_inputs(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any] | None = None,
    ):
        """
        Splits a full-batch input into chunks (i.e. microbatches) and returns
        the chunks
        """
        if args or kwargs:
            args_split, kwargs_split = split_args_kwargs_into_chunks(
                args,
                kwargs,
                self._n_microbatches,
                self._args_chunk_spec,
                self._kwargs_chunk_spec,
            )
            return args_split, kwargs_split
        else:
            # Empty inputs (e.g. when called on middle stages)
            # Return a list of empty tuples/dicts with matching length as chunks
            return [()] * self._n_microbatches, [{}] * self._n_microbatches

    def _merge_outputs(self, output_chunks: list[Any]) -> Any:
        """
        Merge output chunks back to a batch state.
        If output_merge_spec is None, the utility will merge output chunks by dimension 0 (batch dim).
        """
        return merge_chunks(
            output_chunks,
            self._output_merge_spec,
        )


def _batch_p2p(p2p_ops: list[dist.P2POp], desc: str | None = None) -> list[dist.Work]:
    """
    Simple wrapper over batch_isend_irecv from torch.distributed, which just adds a descriptive logger on top.
    """
    if len(p2p_ops) == 0:
        return []
    desc_str = f"{desc}, " if desc else ""
    logger.debug("batch_p2p %s%s", desc_str, p2p_ops)
    return dist.batch_isend_irecv(p2p_ops)


def _sorted_batch_p2p(
    p2p_ops: list[dist.P2POp], desc: str | None = None
) -> dict[int, list[dist.Work]]:
    """
    Sorts the list of P2P ops by the peer rank, and then calls
    batch_isend_irecv. Return a dictionary of works by peer rank. This function
    helps us avoid hangs in case of skip connections.
    """
    # Arrange p2p_ops by peer rank:
    #   int is the peer rank;
    #   List is the list of ops towards the peer
    ops_by_peer: dict[int, list[dist.P2POp]] = defaultdict(list)
    work_by_peer: dict[int, list[dist.Work]] = {}
    if len(p2p_ops) == 0:
        return work_by_peer

    # Classify the ops by peer rank
    for op in p2p_ops:
        ops_by_peer[op.peer].append(op)

    # Call batch_isend_irecv per peer, in sorted order of the peers (to avoid hangs)
    for peer, ops in sorted(ops_by_peer.items()):
        work_by_peer[peer] = _batch_p2p(ops, desc=desc)

    return work_by_peer


def _wait_batch_p2p(work: list[dist.Work]):
    """
    Waits for a list of dist.Work (typically from _batch_p2p / _sorted_batch_p2p).
    """
    for w in work:
        w.wait()


class PipelineScheduleSingle(_PipelineSchedule):
    """
    Base class for single-stage schedules.
    Implements the `step` method.
    Derived classes should implement `_step_microbatches`.

    Gradients are scaled by num_microbatches depending on the `scale_grads` argument, defaulting to True.  This setting
    should match the configuration of your loss_fn, which may either average losses (scale_grads=True)
    or sum losses (scale_grads=False).
    """

    def __init__(
        self,
        stage: _PipelineStageBase,
        n_microbatches: int,
        loss_fn: Callable | None = None,
        args_chunk_spec: tuple[TensorChunkSpec, ...] | None = None,
        kwargs_chunk_spec: dict[str, TensorChunkSpec] | None = None,
        output_merge_spec: dict[str, Any] | tuple[Any] | None = None,
        scale_grads: bool = True,
    ):
        # Init parent
        super().__init__(
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            args_chunk_spec=args_chunk_spec,
            kwargs_chunk_spec=kwargs_chunk_spec,
            output_merge_spec=output_merge_spec,
            scale_grads=scale_grads,
        )
        # Self attributes
        self._stage = stage
        self._num_stages = stage.num_stages
        self._stage_forward_initialized = False
        self._stage_backward_initialized = False

        if n_microbatches < self._num_stages:
            raise ValueError(
                f"Number of microbatches ({n_microbatches}) must be greater than \
or equal to the number of stages ({self._num_stages})."
            )

        self.pipeline_order: dict[int, list[_Action | None]] | None = (
            self._get_pipeline_order()
        )

    def _initialize_stage(self, args, kwargs):
        if not self._stage_forward_initialized:
            # Prepare the communication needed for the pipeline schedule execution
            # This is needed because during execution we always perform a series of batch P2P ops
            # The first call of the batched P2P needs to involve the global group
            all_ops: list[dist.P2POp] = []
            all_ops.extend(self._stage._get_init_p2p_neighbors_ops())
            _wait_batch_p2p(_batch_p2p(all_ops))

            self._stage._prepare_forward_infra(self._n_microbatches, args, kwargs)
            self._stage_forward_initialized = True

        if self._has_backward and not self._stage_backward_initialized:
            self._stage._prepare_backward_infra(self._n_microbatches)
            self._stage_backward_initialized = True

    def step(
        self,
        *args,
        target=None,
        losses: list | None = None,
        return_outputs: bool = True,
        **kwargs,
    ):
        """
        Run one iteration of the pipeline schedule with *whole-batch* input.
        Will chunk the input into microbatches automatically, and go through the
        microbatches according to the schedule implementation.

        args: positional arguments to the model (as in non-pipeline case).
        kwargs: keyword arguments to the model (as in non-pipeline case).
        target: target for the loss function.
        losses: a list to store the losses for each microbatch.
        return_outputs: whether to return the outputs from the last stage.
        """
        if self._has_backward and not torch.is_grad_enabled():
            raise RuntimeError(
                "step() requires gradients to be enabled for backward computation; "
                "it should not be used under torch.no_grad() context. "
                "Please call eval() instead."
            )

        # Set the same has_backward flag for stage object
        self._stage.has_backward = self._has_backward

        # Clean per iteration
        self._stage.clear_runtime_states()

        # Split inputs into microbatches
        args_split, kwargs_split = self._split_inputs(args, kwargs)

        # Split target into microbatches
        if target is not None:
            targets_split = list(torch.tensor_split(target, self._n_microbatches))
        else:
            targets_split = None

        # Run microbatches
        self._step_microbatches(
            args_split, kwargs_split, targets_split, losses, return_outputs
        )

        # Return merged results per original format
        if self._stage.is_last and return_outputs:
            return self._merge_outputs(self._stage.output_chunks)
        else:
            return None

    def _get_pipeline_order(self) -> dict[int, list[_Action | None]] | None:
        """
        Returns the pipeline execution order as a schedule IR.

        The returned IR is a dictionary mapping rank IDs to lists of actions.
        Each action is either an _Action object representing computation to perform,
        or None representing a deliberate idle step.

        The None values are used to represent pipeline bubbles where a rank
        must wait for dependencies from other ranks before proceeding. However
        during execution, with  the _PipelineScheduleRuntime, these Nones are
        skipped since the relevant communication (send/recv) will be scheduled and waited on.

        Returns:
            A dictionary mapping rank -> list of actions
        """
        return None


class _ScheduleForwardOnly(PipelineScheduleSingle):
    """
    The forward-only schedule.
    Will go through all the microbatches and perform only the forward pass
    """

    def _step_microbatches(
        self,
        arg_mbs: list | None = None,
        kwarg_mbs: list | None = None,
        target_mbs: list | None = None,
        losses: list | None = None,
        return_outputs: bool = True,
    ):
        """
        Run one iteration of the pipeline schedule
        """
        if target_mbs is not None or losses is not None:
            raise RuntimeError(
                "Forward-only schedule does not support loss computation"
            )

        arg_mbs, kwarg_mbs = self._check_inputs(arg_mbs, kwarg_mbs, target_mbs, losses)
        self._initialize_stage(arg_mbs[0], kwarg_mbs[0])

        # Delay send waits
        fwd_sends_to_wait: list[list[dist.Work]] = []

        # Run microbatches
        for i in range(self._n_microbatches):
            with record_function(f"Forward {i}"):
                ops = self._stage.get_fwd_recv_ops(i)
                works = _sorted_batch_p2p(ops, desc="fwd_recv")
                for work in works.values():
                    _wait_batch_p2p(work)

                self._stage.forward_one_chunk(i, arg_mbs[i], kwarg_mbs[i])  # type: ignore[index]

                ops = self._stage.get_fwd_send_ops(i)
                works = _sorted_batch_p2p(ops, desc="fwd_send")
                fwd_sends_to_wait.extend(works.values())

            logger.debug("[%s] Forwarded microbatch %s", self._stage.stage_index, i)

        # Wait for all forward sends to finish
        # This should not have performance impact because by the time the first
        # backward arrives all the forward sends should have been finished.
        for work in fwd_sends_to_wait:
            _wait_batch_p2p(work)


class ScheduleGPipe(PipelineScheduleSingle):
    """
    The GPipe schedule.
    Will go through all the microbatches in a fill-drain manner.
    """

    def _step_microbatches(
        self,
        arg_mbs: list | None = None,
        kwarg_mbs: list | None = None,
        target_mbs: list | None = None,
        losses: list | None = None,
        return_outputs: bool = True,
    ):
        """
        Run one iteration of the pipeline schedule with list of microbatches.
        Will go through all the microbatches according to the GPipe schedule.

        Args:
            microbatches: list of microbatch args.
            return_outputs: whether to return the outputs from the last stage.
        """
        arg_mbs, kwarg_mbs = self._check_inputs(arg_mbs, kwarg_mbs, target_mbs, losses)
        self._initialize_stage(arg_mbs[0], kwarg_mbs[0])

        # Delay send waits
        fwd_sends_to_wait: list[list[dist.Work]] = []

        # Run microbatches
        for i in range(self._n_microbatches):
            with record_function(f"Forward {i}"):
                ops = self._stage.get_fwd_recv_ops(i)
                works = _sorted_batch_p2p(ops, desc="fwd_recv")
                for work in works.values():
                    _wait_batch_p2p(work)

                output = self._stage.forward_one_chunk(
                    i, arg_mbs[i], kwarg_mbs[i], save_forward_output=return_outputs
                )  # type: ignore[index]

                ops = self._stage.get_fwd_send_ops(i)
                works = _sorted_batch_p2p(ops, desc="fwd_send")
                fwd_sends_to_wait.extend(works.values())

            logger.debug("[%s] Forwarded microbatch %s", self._stage.stage_index, i)

            self._maybe_compute_loss(self._stage, output, target_mbs, i)

        # Wait for all forward sends to finish
        # This should not have performance impact because by the time the first
        # backward arrives all the forward sends should have been finished.
        for work in fwd_sends_to_wait:
            _wait_batch_p2p(work)

        # Run backward
        # Delay send waits
        bwd_sends_to_wait: list[list[dist.Work]] = []
        for i in range(self._n_microbatches):
            with record_function(f"Backward {i}"):
                ops = self._stage.get_bwd_recv_ops(i)
                works = _sorted_batch_p2p(ops, desc="bwd_recv")
                for work in works.values():
                    _wait_batch_p2p(work)

                loss = self._maybe_get_loss(self._stage, i)
                self._stage.backward_one_chunk(
                    i,
                    loss=loss,
                    last_backward=i == self._n_microbatches - 1,
                )

                ops = self._stage.get_bwd_send_ops(i)
                works = _sorted_batch_p2p(ops, desc="bwd_send")
                bwd_sends_to_wait.extend(works.values())

            logger.debug("[%s] Backwarded microbatch %s", self._stage.stage_index, i)

        # Wait for all backward sends to finish
        for work in bwd_sends_to_wait:
            _wait_batch_p2p(work)

        # Update losses if there is a container passed in
        self._update_losses(self._stage, losses)

        self._stage.perform_reduce_grad(self._n_microbatches if self.scale_grads else 1)

    def _get_pipeline_order(self) -> dict[int, list[_Action | None]] | None:
        """
        Returns the pipeline order for GPipe schedule.

        See base method in PipelineScheduleSingle for details on the schedule IR format.
        """
        pipeline_order = {}
        pp_group_size = self._num_stages

        for rank in range(pp_group_size):
            actions: list[_Action | None] = []

            # 1. Initial delay based on rank position
            warmup_delay = rank
            actions.extend([None] * warmup_delay)

            # 2. Forward passes for all microbatches
            for mb_idx in range(self._n_microbatches):
                actions.append(_Action(rank, _ComputationType.FORWARD, mb_idx))

            # 3. Wait period before backward passes can begin
            backward_delay = 3 * (pp_group_size - 1 - rank)
            actions.extend([None] * backward_delay)

            # 4. Backward passes for all microbatches
            for mb_idx in range(self._n_microbatches):
                actions.append(_Action(rank, _ComputationType.FULL_BACKWARD, mb_idx))

            pipeline_order[rank] = _add_reduce_grad(actions, self._n_microbatches)

        return pipeline_order  # type: ignore[return-value]


class Schedule1F1B(PipelineScheduleSingle):
    """
    The 1F1B schedule.
    Will perform one forward and one backward on the microbatches in steady state.
    """

    def _step_microbatches(
        self,
        arg_mbs: list | None = None,
        kwarg_mbs: list | None = None,
        target_mbs: list | None = None,
        losses: list | None = None,
        return_outputs: bool = True,
    ):
        """
        Run one iteration of the pipeline schedule with list of microbatches.
        Will go through all the microbatches according to the 1F1B schedule.

        Args:
            microbatches: list of microbatch args.
            return_outputs: whether to return the outputs from the last stage.
        """
        arg_mbs, kwarg_mbs = self._check_inputs(arg_mbs, kwarg_mbs, target_mbs, losses)
        self._initialize_stage(arg_mbs[0], kwarg_mbs[0])

        # Last stage has 1 warmup, second-to-last 2 warmups, ...
        # first stage `num_stages` warmups
        warmup_chunks = min(
            self._n_microbatches,
            self._num_stages - self._stage.stage_index,
        )

        # Chunk counters
        fwd_mb_index = 0
        bwd_mb_index = 0

        # Warmup phase
        send_work: list[dist.Work] = []
        fwd_sends = []
        for _ in range(warmup_chunks):
            # Receive activations
            fwd_recvs = self._stage.get_fwd_recv_ops(fwd_mb_index)
            _wait_batch_p2p(_batch_p2p(fwd_recvs, desc="fwd_recv"))

            # Compute
            output = self._stage.forward_one_chunk(
                fwd_mb_index,
                arg_mbs[fwd_mb_index],
                kwarg_mbs[fwd_mb_index],
                save_forward_output=return_outputs,
            )  # type: ignore[index]

            # Clear previous chunk's forward sends (hopefully they have well
            # finished, otherwise, we are heavily communication bound, in which
            # case it doesn't create a lot of benefit to compute next chunk
            # eagerly either)
            _wait_batch_p2p(send_work)

            # Send activations
            fwd_sends = self._stage.get_fwd_send_ops(fwd_mb_index)
            if fwd_mb_index != warmup_chunks - 1:
                # Safe to fire
                send_work = _batch_p2p(fwd_sends, desc="fwd_send")
            # otherwise:
            #   The last forward send is left for fuse with first 1B in 1B1F below

            # Compute loss
            self._maybe_compute_loss(self._stage, output, target_mbs, fwd_mb_index)
            fwd_mb_index += 1

        # Now we should have send ops left over, to be fused with first 1B of 1B1F phase below.

        # 1B1F phase
        while True:  # Don't worry, we have a break inside
            # We actually do 1B first as the `1B1F` name indicates, so prepare its recv ops
            bwd_recvs = self._stage.get_bwd_recv_ops(bwd_mb_index)

            # Now, we need to fire the fwd_sends and bwd_recvs together
            _wait_batch_p2p(_batch_p2p(fwd_sends + bwd_recvs, desc="fwd_send_bwd_recv"))

            # Backward one chunk
            loss = self._maybe_get_loss(self._stage, bwd_mb_index)
            self._stage.backward_one_chunk(
                bwd_mb_index,
                loss=loss,
                last_backward=bwd_mb_index == self._n_microbatches - 1,
            )

            # Get the bwd send ops, but don't fire, to be fused with the 1F below
            bwd_sends = self._stage.get_bwd_send_ops(bwd_mb_index)
            bwd_mb_index += 1

            if fwd_mb_index == self._n_microbatches:
                # We are done with 1B1F, so break with some left-over bwd_sends
                break

            # We prepare 1F of the `1B1F`
            fwd_recvs = self._stage.get_fwd_recv_ops(fwd_mb_index)

            # Fuse it with bwd_sends above
            _wait_batch_p2p(_batch_p2p(bwd_sends + fwd_recvs, desc="bwd_send_fwd_recv"))

            # Now do the fwd
            output = self._stage.forward_one_chunk(
                fwd_mb_index,
                arg_mbs[fwd_mb_index],
                kwarg_mbs[fwd_mb_index],
                save_forward_output=return_outputs,
            )  # type: ignore[index]

            # Compute loss
            self._maybe_compute_loss(self._stage, output, target_mbs, fwd_mb_index)

            # Get the fwd send ops, but don't fire, leave it for the next iter (wrap-around)
            fwd_sends = self._stage.get_fwd_send_ops(fwd_mb_index)
            fwd_mb_index += 1

        # Remember we still have some bwd_sends left over after the break? Now it is time to fire it
        send_work = _batch_p2p(bwd_sends, desc="bwd_send")

        # Cooldown
        while bwd_mb_index < self._n_microbatches:
            # prepare bwd recv ops
            bwd_recvs = self._stage.get_bwd_recv_ops(bwd_mb_index)
            _wait_batch_p2p(_batch_p2p(bwd_recvs, desc="bwd_recv"))

            # Backward one chunk
            loss = self._maybe_get_loss(self._stage, bwd_mb_index)
            self._stage.backward_one_chunk(
                bwd_mb_index,
                loss=loss,
                last_backward=bwd_mb_index == self._n_microbatches - 1,
            )

            # Clear previous chunk's backward sends (hopefully they have well finished)
            _wait_batch_p2p(send_work)

            # Get the bwd send ops, fire it
            bwd_sends = self._stage.get_bwd_send_ops(bwd_mb_index)
            send_work = _batch_p2p(bwd_sends, desc="bwd_send")
            bwd_mb_index += 1

        # Wait for the last backward send to finish
        _wait_batch_p2p(send_work)

        # Return losses if there is a container passed in
        self._update_losses(self._stage, losses)

        self._stage.perform_reduce_grad(self._n_microbatches if self.scale_grads else 1)

    def _get_pipeline_order(self) -> dict[int, list[_Action | None]] | None:
        """
        Returns the pipeline order for 1F1B schedule.

        See base method in PipelineScheduleSingle for details on the schedule IR format.
        """
        pipeline_order = {}
        pp_group_size = self._num_stages

        for rank in range(pp_group_size):
            actions: list[_Action | None] = []

            # 1. Warmup phase: initial delay based on rank
            actions.extend([None] * rank)

            # 2. Initial forward passes before 1F1B phase
            num_forward = (pp_group_size - 1) - rank
            forward_mb = 0
            for i in range(num_forward):
                actions.append(_Action(rank, _ComputationType.FORWARD, i))
                forward_mb = i

            # 3. Wait for backward to be ready
            wait_for_1f1b = max(0, 2 * (pp_group_size - 1 - rank))
            actions.extend([None] * wait_for_1f1b)

            # 4. 1F1B steady state phase
            backward_mb = 0
            remaining_forward = self._n_microbatches - num_forward

            while remaining_forward > 0:
                # One forward
                forward_mb += 1
                actions.append(_Action(rank, _ComputationType.FORWARD, forward_mb))
                remaining_forward -= 1

                # One backward
                actions.append(
                    _Action(rank, _ComputationType.FULL_BACKWARD, backward_mb)
                )
                backward_mb += 1

            # 5. Cooldown phase: remaining backward passes
            remaining_backward = self._n_microbatches - backward_mb

            while remaining_backward > 0:
                # Add None and backward actions in alternating pattern
                # based on distance from the last stage
                if (pp_group_size - rank) > 0:
                    actions.append(None)
                    # Decrement the wait counter only if we still have backward passes to do
                    if remaining_backward > 0:
                        actions.append(
                            _Action(rank, _ComputationType.FULL_BACKWARD, backward_mb)
                        )
                        backward_mb += 1
                        remaining_backward -= 1
                else:
                    # If we're at the last stage, just add backward actions without None
                    actions.append(
                        _Action(rank, _ComputationType.FULL_BACKWARD, backward_mb)
                    )
                    backward_mb += 1
                    remaining_backward -= 1

            pipeline_order[rank] = _add_reduce_grad(actions, self._n_microbatches)
        return pipeline_order


def _requires_reduce_grad(action_type: _ComputationType) -> bool:
    return action_type in (W, B)


def _add_reduce_grad(
    actions: list[_Action | None], n_microbatches: int
) -> list[_Action | None]:
    """
    REDUCE_GRAD refers to joint across minibatches grad reduction.
    reduce_grad frees memory and we want to schedule it just after the last "backward"-like stage.
    """
    actions_with_reduce_grad: list[_Action | None] = []
    cnt: dict[int, int] = defaultdict(int)

    def _leaf_action(a, to_schedule):
        if _requires_reduce_grad(a.computation_type):
            stage_index = a.stage_index
            cnt[stage_index] += 1
            if cnt[stage_index] == n_microbatches:
                to_schedule.append(stage_index)

    for a in actions:
        if a is None:
            continue
        actions_with_reduce_grad.append(a)
        schedule_reduce_grad_stage_idxs: list[int] = []
        if a.computation_type == OVERLAP_F_B and a.sub_actions is not None:
            for sub_action in a.sub_actions:
                _leaf_action(sub_action, schedule_reduce_grad_stage_idxs)
        else:
            _leaf_action(a, schedule_reduce_grad_stage_idxs)

        for stage_idx in schedule_reduce_grad_stage_idxs:
            actions_with_reduce_grad.append(_Action(stage_idx, REDUCE_GRAD, None))
    return actions_with_reduce_grad


# Import runtime classes (late import here to avoid circular dependency)


def _validate_schedule(
    actions: dict[int, list[_Action | None]],
    pp_group_size: int,
    num_stages: int,
    num_microbatches: int,
) -> dict[int, int]:
    if not (len(actions) == pp_group_size):
        raise AssertionError(
            f"Schedule has incorrect number of ranks - expected {pp_group_size}, actual {len(actions)}"
        )
    for rank in range(pp_group_size):
        if rank not in actions:
            raise AssertionError(f"Schedule is missing actions for rank {rank}")

    # We will count all the actions per stage and ensure they happen in a valid order
    # (e.g. F before (B, I) before W for a given microbatch)
    stage_actions: dict[int, dict[_ComputationType, set]] = {
        stage_id: {
            F: set(),
            B: set(),
            I: set(),
            W: set(),
        }
        for stage_id in range(num_stages)
    }
    stage_index_to_rank_mapping = {}

    def _process_action(action: _Action, rank: int, step: int):
        """Process a single action and update stage_actions and stage_index_to_rank_mapping"""
        s_id = action.stage_index
        ctype = action.computation_type
        mb_id = action.microbatch_index

        if ctype == F:
            stage_actions[s_id][F].add(mb_id)
        elif ctype == B:
            if mb_id not in stage_actions[s_id][F]:
                error_msg = (
                    f"Rank {rank}, step {step}: Running Full Backward for stage {s_id}, "
                    f"microbatch {mb_id} without first running Forward"
                )
                formatted_schedule = _format_pipeline_order(
                    actions, error_step_number=step
                )
                full_error_msg = (
                    f"{error_msg}\n\nFull pipeline schedule:\n{formatted_schedule}"
                )
                raise AssertionError(full_error_msg)
            stage_actions[s_id][B].add(mb_id)
        elif ctype == I:
            if mb_id not in stage_actions[s_id][F]:
                error_msg = (
                    f"Rank {rank}, step {step}: Running Backward Input for stage {s_id}, "
                    f"microbatch {mb_id} without first running Forward"
                )
                formatted_schedule = _format_pipeline_order(
                    actions, error_step_number=step
                )
                full_error_msg = (
                    f"{error_msg}\n\nFull pipeline schedule:\n{formatted_schedule}"
                )
                raise AssertionError(full_error_msg)
            stage_actions[s_id][I].add(mb_id)
        elif ctype == W:
            if mb_id not in stage_actions[s_id][I]:
                error_msg = (
                    f"Rank {rank}, step {step}: Running Backward Weight for stage {s_id}, "
                    f"microbatch {mb_id} without first running Backward Input"
                )
                formatted_schedule = _format_pipeline_order(
                    actions, error_step_number=step
                )
                full_error_msg = (
                    f"{error_msg}\n\nFull pipeline schedule:\n{formatted_schedule}"
                )
                raise AssertionError(full_error_msg)
            stage_actions[s_id][W].add(mb_id)

        if s_id not in stage_index_to_rank_mapping:
            stage_index_to_rank_mapping[s_id] = rank
        else:
            existing_rank = stage_index_to_rank_mapping[s_id]
            if not (rank == existing_rank):
                raise AssertionError(
                    f"Rank {rank}, step {step}: Stage {s_id} is assigned to both rank {rank} and rank {existing_rank}"
                )

    for rank in actions:
        for step, action in enumerate(actions[rank]):
            if action is None:
                continue
            if not isinstance(action, _Action):
                raise AssertionError(
                    f"Rank {rank}, step {step}: Got an invalid action: {action}, expected instance of _Action"
                )

            # Check if action has sub_actions
            if action.sub_actions is not None:
                # Process each sub_action instead of the main action
                for sub_action in action.sub_actions:
                    _process_action(sub_action, rank, step)
            else:
                # Process the main action normally
                _process_action(action, rank, step)

    for s_id in stage_actions:
        f_mb = len(stage_actions[s_id][F])
        b_mb = len(stage_actions[s_id][B])
        i_mb = len(stage_actions[s_id][I])
        w_mb = len(stage_actions[s_id][W])

        if not (f_mb == num_microbatches):
            raise AssertionError(
                f"Got {f_mb} {F} microbatches for stage {s_id}, expected {num_microbatches}"
            )

        if not (i_mb == w_mb):
            raise AssertionError(
                f"Invalid backward microbatches for stage {s_id}: I and W must have equal counts, \
            but got I={i_mb}, W={w_mb}"
            )

        if not (b_mb + (i_mb + w_mb) // 2 == num_microbatches):
            raise AssertionError(
                f"Invalid backward microbatches for stage {s_id}: expected {num_microbatches} total backwards, \
            but got B={b_mb}, I={i_mb}, W={w_mb}"
            )
    return stage_index_to_rank_mapping


class PipelineScheduleMulti(_PipelineSchedule):
    """
    Base class for multi-stage schedules.
    Implements the `step` method.

    Gradients are scaled by num_microbatches depending on the `scale_grads` argument, defaulting to True.  This setting
    should match the configuration of your loss_fn, which may either average losses (scale_grads=True)
    or sum losses (scale_grads=False).
    """

    def __init__(
        self,
        stages: list[_PipelineStageBase],
        n_microbatches: int,
        loss_fn: Callable | None = None,
        args_chunk_spec: tuple[TensorChunkSpec, ...] | None = None,
        kwargs_chunk_spec: dict[str, TensorChunkSpec] | None = None,
        output_merge_spec: dict[str, Any] | tuple[Any] | None = None,
        use_full_backward: bool | None = None,
        scale_grads: bool = True,
        backward_requires_autograd: bool = True,
    ):
        # Init parent
        super().__init__(
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            args_chunk_spec=args_chunk_spec,
            kwargs_chunk_spec=kwargs_chunk_spec,
            output_merge_spec=output_merge_spec,
            scale_grads=scale_grads,
        )
        # Self attributes
        self._stages = stages
        self._num_stages = stages[0].num_stages
        self.pp_group_size = stages[0].group_size
        self.rank = stages[0].group_rank
        # Set the pipeline stage states
        self.stage_index_to_group_rank = generate_stage_to_rank_mapping(
            self.pp_group_size, self._num_stages
        )
        for stage in self._stages:
            stage.stage_index_to_group_rank = self.stage_index_to_group_rank

        self._stages_forward_initialized = False
        self._stages_backward_initialized = False

        # avoid putting a reference to 'self' inside the lambda, it creates a ref cycle
        has_loss: bool = self._loss_fn is not None
        self._should_compute_loss = lambda stage: stage.is_last and has_loss

        # This will be set during init of derived schedules
        self.pipeline_order: dict[int, list[_Action | None]] = {}

        # When using a custom backward function, we may or may not need autograd to be used
        # for the backward pass. This flag is used to determine whether or torch.is_grad_enabled()
        # check should be performed before the step function.
        self._backward_requires_autograd = backward_requires_autograd

        if use_full_backward is not None:
            logger.warning(
                "Deprecation warning: 'use_full_backward' is no longer supported. "
                "Simply stop passing it, and everything should still work fine."
            )

    def _initialize_stages(self, args: tuple[Any, ...], kwargs):
        if not self._stages_forward_initialized:
            # Prepare the communication needed for the pipeline schedule execution
            # This is needed because during execution we always perform a series of batch P2P ops
            # The first call of the batched P2P needs to involve the global group
            all_ops: list[dist.P2POp] = []
            for stage in self._stages:
                all_ops.extend(stage._get_init_p2p_neighbors_ops())
            _wait_batch_p2p(_batch_p2p(all_ops))

            # may be 'none' value (if this stage sends its output shapes to the next stage via P2P)
            # or real value (if this stage and next stage are on the same device)
            next_stage_args: tuple[Any, ...] = tuple()
            for stage in self._stages:
                if stage.is_first:
                    next_stage_args = stage._prepare_forward_infra(
                        self._n_microbatches, args, kwargs
                    )
                else:
                    next_stage_args = stage._prepare_forward_infra(
                        self._n_microbatches, next_stage_args, kwargs
                    )
            self._stages_forward_initialized = True

        if self._has_backward and not self._stages_backward_initialized:
            for stage in self._stages:
                stage._prepare_backward_infra(self._n_microbatches)
            self._stages_backward_initialized = True

    def _validate_and_set_stage_mapping(
        self, actions: dict[int, list[_Action | None]]
    ) -> None:
        """
        Allocates the stage index to rank mapping which is needed for communication
        """
        self.stage_index_to_group_rank = _validate_schedule(
            actions,
            self.pp_group_size,
            self._num_stages,
            self._n_microbatches,
        )
        for stage in self._stages:
            stage.stage_index_to_group_rank = self.stage_index_to_group_rank

    def _dump_csv(self, filename):
        """Dump a CSV representation of the schedule into a file with the provided filename."""
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for rank in self.pipeline_order:
                writer.writerow(self.pipeline_order[rank])

    def _load_csv(self, filename, format="compute_only"):
        """Load a CSV representation of the schedule from a file with the provided filename.
        This API will most likely get renamed/refactored so is marked as internal for now.

        format must be "compute_only" for PipelineScheduleMulti.
        """
        if format != "compute_only":
            raise AssertionError(f'format must be "compute_only", got {format}')
        with open(filename, newline="") as csvfile:
            reader = csv.reader(csvfile)
            for rank, row in enumerate(reader):
                self.pipeline_order[rank] = [_Action.from_str(s) for s in row]

        # Validates the order of the pipeline actions and infers the stage_to_rank_mapping.
        # This will overwrite the default stage_to_rank_mapping created in the constructor
        self._validate_and_set_stage_mapping(self.pipeline_order)

    def step(
        self,
        *args,
        target=None,
        losses: list | None = None,
        return_outputs: bool = True,
        **kwargs,
    ):
        """
        Run one iteration of the pipeline schedule with *whole-batch* input.
        Will chunk the input into microbatches automatically, and go through the
        microbatches according to the schedule implementation.

        args: positional arguments to the model (as in non-pipeline case).
        kwargs: keyword arguments to the model (as in non-pipeline case).
        target: target for the loss function.
        losses: a list to store the losses for each microbatch.
        return_outputs: whether to return the outputs from the last stage.
        """
        if (
            self._has_backward
            and self._backward_requires_autograd
            and not torch.is_grad_enabled()
        ):
            raise RuntimeError(
                "step() requires gradients to be enabled for backward computation; "
                "it should not be used under torch.no_grad() context. "
                "Please call eval() instead."
            )

        # Set the same has_backward flag for stage object
        for stage in self._stages:
            stage.has_backward = self._has_backward

        # Clean per iteration
        for stage in self._stages:
            stage.clear_runtime_states()

        # Split inputs into microbatches
        args_split, kwargs_split = self._split_inputs(args, kwargs)

        # Split target into microbatches
        if target is not None:
            targets_split = list(torch.tensor_split(target, self._n_microbatches))
        else:
            targets_split = None

        # Run microbatches
        self._step_microbatches(
            args_split, kwargs_split, targets_split, losses, return_outputs
        )

        # Return merged results per original format
        for stage in self._stages:
            if stage.is_last and return_outputs:
                return self._merge_outputs(stage.output_chunks)
        # Does not contain the last stage or we do not return output chunks
        return None

    def _step_microbatches(
        self,
        arg_mbs: list | None = None,
        kwarg_mbs: list | None = None,
        target_mbs: list | None = None,
        losses: list | None = None,
        return_outputs: bool = True,
    ):
        """
        Operate on the microbatches for looped schedules (multiple stages on each rank).

        TODO: Does not use sorted_batch_isend_irecv(). As a result, this schedule does
        not support models with skip connections.
        """
        arg_mbs, kwarg_mbs = self._check_inputs(arg_mbs, kwarg_mbs, target_mbs, losses)

        self._initialize_stages(arg_mbs[0], kwarg_mbs[0])

        # Based on the plan in Step 1 created in __init__:
        # 2. Perform communication based on the pipeline_order
        stage_index_to_stage: dict[int, _PipelineStageBase] = {
            stage.stage_index: stage for stage in self._stages
        }

        # determine prev_rank and next_rank based on which ranks are next to
        # the stages in the pipeline_order
        all_prev_ranks: set[int] = set()
        all_next_ranks: set[int] = set()
        for stage_index in stage_index_to_stage:
            # TODO: assumption that stages only communicate from distances of +1/-1 (no skip connections)
            if stage_index > 0:
                all_prev_ranks.add(self.stage_index_to_group_rank[stage_index - 1])
            if stage_index < self._num_stages - 1:
                all_next_ranks.add(self.stage_index_to_group_rank[stage_index + 1])
        # count either full_backward or backward_weight together, to determine when to sync DP grads
        backward_counter: Counter[int] = Counter()
        for time_step, action in enumerate(self.pipeline_order[self.rank]):
            try:
                ops: list[dist.P2POp] = []
                if action is not None:
                    computation_type = action.computation_type
                    mb_index = action.microbatch_index
                    stage_index = action.stage_index
                    if mb_index is None:
                        raise AssertionError(
                            "All currently supported action types require valid microbatch_index"
                        )
                    if computation_type == _ComputationType.FORWARD:
                        # perform forward computation
                        stage = stage_index_to_stage[stage_index]
                        output = stage.forward_one_chunk(
                            mb_index,
                            arg_mbs[mb_index],
                            kwarg_mbs[mb_index],
                            save_forward_output=return_outputs,
                        )
                        self._maybe_compute_loss(stage, output, target_mbs, mb_index)
                        ops.extend(stage.get_fwd_send_ops(mb_index))
                    elif computation_type == _ComputationType.FULL_BACKWARD:
                        # perform backward computation
                        stage = stage_index_to_stage[stage_index]
                        loss = self._maybe_get_loss(stage, mb_index)
                        backward_counter[stage_index] += 1
                        last_backward = (
                            backward_counter[stage_index] == self._n_microbatches
                        )
                        grad_scale_factor = (
                            self._n_microbatches if self.scale_grads else 1
                        )
                        stage.backward_one_chunk(
                            mb_index,
                            loss=loss,
                            full_backward=True,
                            last_backward=last_backward,
                        )
                        if last_backward:
                            stage.scale_grads(grad_scale_factor)

                        ops.extend(stage.get_bwd_send_ops(mb_index))
                    elif computation_type == _ComputationType.BACKWARD_INPUT:
                        # perform backward computation
                        stage = stage_index_to_stage[stage_index]
                        loss = self._maybe_get_loss(stage, mb_index)
                        stage.backward_one_chunk(
                            mb_index,
                            loss=loss,
                            full_backward=False,
                            last_backward=False,
                        )
                        ops.extend(stage.get_bwd_send_ops(mb_index))
                    elif computation_type == _ComputationType.BACKWARD_WEIGHT:
                        # perform weight update
                        stage = stage_index_to_stage[stage_index]
                        backward_counter[stage_index] += 1
                        last_backward = (
                            backward_counter[stage_index] == self._n_microbatches
                        )
                        grad_scale_factor = (
                            self._n_microbatches if self.scale_grads else 1
                        )
                        stage.backward_weight_one_chunk(
                            mb_index,
                            last_backward=last_backward,
                        )
                        if last_backward:
                            stage.scale_grads(grad_scale_factor)
                    else:
                        raise ValueError(f"Unknown computation type {computation_type}")

                # Look at the neighboring ranks for this current timestep and determine whether
                # this current rank needs to do any recv communication
                for prev_rank in all_prev_ranks:
                    prev_rank_ops = self.pipeline_order[prev_rank]
                    prev_rank_action = None
                    if time_step < len(prev_rank_ops):
                        prev_rank_action = prev_rank_ops[time_step]
                    if prev_rank_action is not None:
                        computation_type = prev_rank_action.computation_type
                        mb_index = prev_rank_action.microbatch_index
                        stage_index = prev_rank_action.stage_index
                        if mb_index is None:
                            raise AssertionError(
                                "All currently supported action types require valid microbatch_index"
                            )
                        # Only handle sends for the forward from a previous rank
                        if computation_type == _ComputationType.FORWARD:
                            # If not the last stage, then receive fwd activations
                            if stage_index + 1 in stage_index_to_stage:
                                # TODO: We are assuming that stage will always receive from stage-1
                                # however that is not necessarily true of get_fwd_recv_ops
                                stage = stage_index_to_stage[stage_index + 1]
                                ops.extend(stage.get_fwd_recv_ops(mb_index))
                        elif computation_type in (
                            FULL_BACKWARD,
                            BACKWARD_INPUT,
                            BACKWARD_WEIGHT,
                        ):
                            # Previous rank doing backward has no influence for the current rank forward recv
                            pass
                        else:
                            raise ValueError(
                                f"Unknown computation type {computation_type}"
                            )
                for next_rank in all_next_ranks:
                    next_rank_ops = self.pipeline_order[next_rank]
                    next_rank_action = None
                    if time_step < len(next_rank_ops):
                        next_rank_action = next_rank_ops[time_step]
                    if next_rank_action is not None:
                        computation_type = next_rank_action.computation_type
                        mb_index = next_rank_action.microbatch_index
                        stage_index = next_rank_action.stage_index
                        if not (mb_index is not None):
                            raise AssertionError(
                                "All currently supported action types require valid microbatch_index"
                            )
                        # Only handle receives for the backwards from a next rank
                        if computation_type in (FORWARD, BACKWARD_WEIGHT):
                            # Next rank doing forward or weight update has no influence for the current rank backward recv
                            pass
                        elif computation_type in (BACKWARD_INPUT, FULL_BACKWARD):
                            # If not the first stage, then receive bwd gradients
                            if stage_index - 1 in stage_index_to_stage:
                                # TODO: We are assuming that stage will always receive from stage+1
                                # however that is not necessarily true of get_bwd_recv_ops
                                stage = stage_index_to_stage[stage_index - 1]
                                ops.extend(stage.get_bwd_recv_ops(mb_index))
                        else:
                            raise ValueError(
                                f"Unknown computation type {computation_type}"
                            )

                # do the communication
                _wait_batch_p2p(_batch_p2p(ops))
            except Exception as e:
                logger.error(  # noqa: G200
                    "[Rank %s] pipeline schedule %s caught the following exception '%s' \
at time_step %s when running action %s",
                    self.rank,
                    self.__class__.__name__,
                    str(e),
                    time_step,
                    action,
                )
                logger.error(
                    "%s",
                    _format_pipeline_order(
                        self.pipeline_order, error_step_number=time_step
                    ),
                )
                raise e
        # Return losses if there is a container passed in
        self._update_losses(self._stages, losses)


# Import runtime classes (late import here to avoid circular dependency)
from .runtime import _PipelineScheduleRuntime


class ScheduleLoopedBFS(_PipelineScheduleRuntime):
    """
    Breadth-First Pipeline Parallelism.
    See https://arxiv.org/abs/2211.05953 for details.
    Similar to Interleaved 1F1B, Looped BFS supports multiple stages per rank.
    What is different is that when microbatches are ready for multiple local
    stages, Loops BFS will prioritizes the earlier stage, running all available
    microbatches at once.
    """

    def __init__(
        self,
        stages: list[_PipelineStageBase],
        n_microbatches: int,
        loss_fn: Callable | _Loss | None = None,
        output_merge_spec: dict[str, Any] | tuple[Any] | None = None,
        scale_grads: bool = True,
        backward_requires_autograd: bool = True,
    ):
        super().__init__(
            stages=stages,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            output_merge_spec=output_merge_spec,
            scale_grads=scale_grads,
            backward_requires_autograd=backward_requires_autograd,
        )

        # 1. Create the pipeline_order (all ranks do this calculation)
        # This will be used to keep track of the current state of the entire pipeline
        # pipeline_order[rank] = [Action(computation_type, microbatch_index, stage_index), ...]
        self.pipeline_order: dict[int, list[_Action | None]] = {}
        # ========================================================================
        for rank in range(self.pp_group_size):
            rank_ops = self._calculate_single_rank_operations(rank)
            self.pipeline_order[rank] = rank_ops

        # Initialize the pipeline order with communication necessary to run with _PipelineScheduleRuntime
        self._prepare_schedule_with_comms(self.pipeline_order)

    def _calculate_single_rank_operations(self, rank):
        n_local_stages = len(self._stages)
        stage_indices = range(
            rank, self.pp_group_size * n_local_stages, self.pp_group_size
        )

        # Store the list of operations used for that rank
        # Pre-padding, rank starts with no-ops based on the warmup.
        rank_ops: list[_Action | None] = [None for _ in range(rank)]

        for stage_index in stage_indices:
            rank_ops.extend(
                _Action(stage_index, _ComputationType.FORWARD, mb_index)
                for mb_index in range(self._n_microbatches)
            )

        # wait for the first backward to trickle up
        # which is 2 for every hop away
        post_warmup_ops = 2 * (self.pp_group_size - 1 - rank)
        rank_ops.extend([None] * post_warmup_ops)

        for stage_index in reversed(stage_indices):
            rank_ops.extend(
                _Action(stage_index, _ComputationType.FULL_BACKWARD, mb_index)
                for mb_index in reversed(range(self._n_microbatches))
            )
        return rank_ops


def _get_1f1b_rank_ops(
    n_local_stages,
    pp_group_size,
    warmup_ops,
    fwd_bwd_ops,
    cooldown_ops,
    rank,
    forward_stage_index,
    backward_stage_index,
    num_1f1b_microbatches=0,
    enable_zero_bubble=False,
):
    # All stages start with handling microbatch 0
    fwd_stage_mb_index: dict[int, int] = defaultdict(int)
    bwd_stage_mb_index: dict[int, int] = defaultdict(int)
    weight_stage_mb_index: dict[int, int] = defaultdict(int)

    # Store the list of operations used for that rank
    # Pre-padding, rank starts with no-ops based on the warmup.
    rank_ops: list[_Action | None] = [None for _ in range(rank)]
    # These are used to calculate the number of slots to fill with no-ops, to account for the delay in warmup
    # when we want to wait for the backward to trickle back up and start 1f1b to align all ranks.
    # Formula:
    # pre-padding + warmup_ops + post_warmup_ops = earliest time step of first backward
    # post_warmup_ops = [earliest time step of first backward] - (warmup_ops + pre-padding)
    # earliest time step of first backward = [local_stages * group_size + 2 * (group_size - 1 - rank)]
    # warmup_ops = calculated above
    post_warmup_ops = (
        n_local_stages * pp_group_size + 2 * (pp_group_size - 1 - rank)
    ) - (warmup_ops + rank)

    if enable_zero_bubble:
        post_warmup_ops = pp_group_size - rank - 1

    total_ops = warmup_ops + fwd_bwd_ops + cooldown_ops

    backward_op_ids = []
    weight_op_count = 0

    FULL_BACKWARD_OR_BACKWARD_INPUT = (
        BACKWARD_INPUT if enable_zero_bubble else FULL_BACKWARD
    )

    for op in range(total_ops):
        # Warmup phase
        if op < warmup_ops:
            fwd_stage_index = forward_stage_index(op)
            # This will assign the current microbatch index and update it as well
            fwd_stage_mb_index[fwd_stage_index] = (
                mb_index := fwd_stage_mb_index[fwd_stage_index]
            ) + 1
            rank_ops.append(
                _Action(fwd_stage_index, _ComputationType.FORWARD, mb_index)
            )
            if op == warmup_ops - 1:
                # This is the last step in the warmup phase, so we need to wait for the backward to trickle back up
                rank_ops.extend([None] * post_warmup_ops)
        # 1F1B Phase (forward and backward)
        elif warmup_ops <= op < warmup_ops + fwd_bwd_ops:
            fwd_stage_index = forward_stage_index(op)
            fwd_stage_mb_index[fwd_stage_index] = (
                fwd_mb_index := fwd_stage_mb_index[fwd_stage_index]
            ) + 1
            rank_ops.append(
                _Action(fwd_stage_index, _ComputationType.FORWARD, fwd_mb_index)
            )
            bwd_stage_index = backward_stage_index(op)
            bwd_stage_mb_index[bwd_stage_index] = (
                bwd_mb_index := bwd_stage_mb_index[bwd_stage_index]
            ) + 1
            rank_ops.append(
                _Action(bwd_stage_index, FULL_BACKWARD_OR_BACKWARD_INPUT, bwd_mb_index)
            )
            backward_op_ids.append(op)

            if enable_zero_bubble and op - warmup_ops >= num_1f1b_microbatches:
                weight_stage_index = backward_stage_index(
                    backward_op_ids[weight_op_count]
                )
                weight_stage_mb_index[weight_stage_index] = (
                    weight_mb_index := weight_stage_mb_index[weight_stage_index]
                ) + 1
                rank_ops.append(
                    _Action(
                        weight_stage_index,
                        _ComputationType.BACKWARD_WEIGHT,
                        weight_mb_index,
                    )
                )
                weight_op_count += 1
        # Cooldown phase
        else:
            # During cooldown phase, we need steps to align with 1f1b happening in other ranks
            # TODO: we don't need to always append, after all 1f1b are finished we can stop appending None
            if not enable_zero_bubble:
                rank_ops.append(None)

            bwd_stage_index = backward_stage_index(op)
            bwd_stage_mb_index[bwd_stage_index] = (
                bwd_mb_index := bwd_stage_mb_index[bwd_stage_index]
            ) + 1
            rank_ops.append(
                _Action(bwd_stage_index, FULL_BACKWARD_OR_BACKWARD_INPUT, bwd_mb_index)
            )
            backward_op_ids.append(op)

            if enable_zero_bubble and op - warmup_ops >= num_1f1b_microbatches:
                weight_stage_index = backward_stage_index(
                    backward_op_ids[weight_op_count]
                )
                weight_stage_mb_index[weight_stage_index] = (
                    weight_mb_index := weight_stage_mb_index[weight_stage_index]
                ) + 1
                rank_ops.append(
                    _Action(
                        weight_stage_index,
                        _ComputationType.BACKWARD_WEIGHT,
                        weight_mb_index,
                    )
                )
                weight_op_count += 1

    while enable_zero_bubble and weight_op_count < len(backward_op_ids):
        weight_stage_index = backward_stage_index(backward_op_ids[weight_op_count])
        weight_stage_mb_index[weight_stage_index] = (
            weight_mb_index := weight_stage_mb_index[weight_stage_index]
        ) + 1
        rank_ops.append(
            _Action(
                weight_stage_index, _ComputationType.BACKWARD_WEIGHT, weight_mb_index
            )
        )
        weight_op_count += 1

    return rank_ops


def _get_warmup_ops(
    rank: int,
    n_local_stages: int,
    microbatches_per_round: int,
    pp_group_size: int,
    n_microbatches: int,
    multiply_factor: int = 2,
) -> int:
    """
    Calculate the number of warmup operations for interleaved schedules.
    """
    # Warmup operations for last stage
    warmups_ops_last_stage = (n_local_stages - 1) * microbatches_per_round
    # Increment warmup operations by multiply_factor for each hop away from the last stage
    warmup_ops = warmups_ops_last_stage + multiply_factor * ((pp_group_size - 1) - rank)
    # We cannot have more warmup operations than there are number of microbatches, so cap it there
    return min(warmup_ops, n_microbatches * n_local_stages)


class ScheduleInterleaved1F1B(_PipelineScheduleRuntime):
    """
    The Interleaved 1F1B schedule.
    See https://arxiv.org/pdf/2104.04473 for details.
    Will perform one forward and one backward on the microbatches in steady
    state and supports multiple stages per rank. When microbatches are ready for
    multiple local stages, Interleaved 1F1B prioritizes the earlier microbatch
    (also called "depth first").

    This schedule is mostly similar to the original paper.
    It differs by being relaxing the requirement of num_microbatch % pp_size == 0.
    Using the flex_pp schedule, we will have num_rounds = max(1, n_microbatches // pp_group_size) and
    it works as long as n_microbatches % num_rounds is 0. As a few examples, support

    1. pp_group_size = 4, n_microbatches = 10. We will have num_rounds = 2 and n_microbatches % 2 is 0.
    2. pp_group_size = 4, n_microbatches = 3. We will have num_rounds = 1 and n_microbatches % 1 is 0.
    """

    def __init__(
        self,
        stages: list[_PipelineStageBase],
        n_microbatches: int,
        loss_fn: Callable | None = None,
        args_chunk_spec: tuple[TensorChunkSpec, ...] | None = None,
        kwargs_chunk_spec: dict[str, TensorChunkSpec] | None = None,
        output_merge_spec: dict[str, Any] | tuple[Any] | None = None,
        scale_grads: bool = True,
        backward_requires_autograd: bool = True,
    ):
        self.pp_group_size = stages[0].group_size
        super().__init__(
            stages=stages,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            args_chunk_spec=args_chunk_spec,
            kwargs_chunk_spec=kwargs_chunk_spec,
            output_merge_spec=output_merge_spec,
            scale_grads=scale_grads,
            backward_requires_autograd=backward_requires_autograd,
        )
        self.n_local_stages = len(stages)
        self.rank = stages[0].group_rank
        self.number_of_rounds = max(1, n_microbatches // self.pp_group_size)
        self.microbatches_per_round = n_microbatches // self.number_of_rounds
        if n_microbatches % self.number_of_rounds != 0:
            raise ValueError(
                "Interleaved 1F1B requires the number of microbatches to be a "
                f"multiple of the number of rounds ({self.number_of_rounds}), "
                f"but got {n_microbatches}."
            )
        # 1. Create the pipeline_order (all ranks do this calculation)
        # This will be used to keep track of the current state of the entire pipeline
        # pipeline_order[rank] = [Action(computation_type, microbatch_index, stage_index), ...]
        self.pipeline_order: dict[int, list[_Action | None]] = {}
        for rank in range(self.pp_group_size):
            rank_ops = self._calculate_single_rank_operations(rank)
            self.pipeline_order[rank] = rank_ops

        # Initialize the pipeline order with communication necessary to run with _PipelineScheduleRuntime
        self._prepare_schedule_with_comms(self.pipeline_order)

    def _calculate_single_rank_operations(self, rank) -> list[_Action | None]:
        warmup_ops = _get_warmup_ops(
            rank,
            self.n_local_stages,
            self.microbatches_per_round,
            self.pp_group_size,
            self._n_microbatches,
            multiply_factor=2,
        )
        microbatch_ops = self.n_local_stages * self._n_microbatches
        # fwd_bwd_ops should encompass the remaining forwards
        fwd_bwd_ops = microbatch_ops - warmup_ops
        # cooldown_ops should encompass the remaining backwards
        cooldown_ops = microbatch_ops - fwd_bwd_ops
        # total ops encompass both forward and backward ops
        total_ops = warmup_ops + fwd_bwd_ops + cooldown_ops
        # warmup_ops + fwd_bwd_ops * 2 + cooldown_ops == microbatch_ops * 2
        logger.debug(
            "rank %s, warmup_ops %s, 1f1b %s, cooldown_ops %s total_ops %s",
            rank,
            warmup_ops,
            fwd_bwd_ops,
            cooldown_ops,
            total_ops,
        )

        # Calculates the stage index based on step and pp_group_size
        def forward_stage_index(step):
            # Get the local index from 0 to n_local_stages-1
            local_index = (step // self.microbatches_per_round) % self.n_local_stages
            return (local_index * self.pp_group_size) + rank

        def backward_stage_index(step):
            local_index = (
                self.n_local_stages
                - 1
                - ((step - warmup_ops) // self.microbatches_per_round)
                % self.n_local_stages
            )
            return (local_index * self.pp_group_size) + rank

        return _get_1f1b_rank_ops(
            self.n_local_stages,
            self.pp_group_size,
            warmup_ops,
            fwd_bwd_ops,
            cooldown_ops,
            rank,
            forward_stage_index,
            backward_stage_index,
        )


class ScheduleInterleavedZeroBubble(_PipelineScheduleRuntime):
    """
    The Interleaved Zero Bubble schedule.
    See https://arxiv.org/pdf/2401.10241 for details.
    Will perform one forward and one backward on inputs for the microbatches in steady
    state and supports multiple stages per rank. Uses the backward for weights to fill in
    the pipeline bubble.

    In particular this is implementing the ZB1P schedule in the paper.
    """

    def __init__(
        self,
        stages: list[_PipelineStageBase],
        n_microbatches: int,
        loss_fn: Callable | None = None,
        args_chunk_spec: tuple[TensorChunkSpec, ...] | None = None,
        kwargs_chunk_spec: dict[str, TensorChunkSpec] | None = None,
        output_merge_spec: dict[str, Any] | tuple[Any] | None = None,
        scale_grads: bool = True,
        backward_requires_autograd: bool = True,
    ):
        # TODO: we dont support input/weight backward split with torch.compile
        _check_torch_compile_compatibility(stages, self.__class__.__name__)
        self.pp_group_size = stages[0].group_size
        super().__init__(
            stages=stages,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            args_chunk_spec=args_chunk_spec,
            kwargs_chunk_spec=kwargs_chunk_spec,
            output_merge_spec=output_merge_spec,
            scale_grads=scale_grads,
            backward_requires_autograd=backward_requires_autograd,
        )
        self.n_local_stages = len(stages)
        self.rank = stages[0].group_rank
        self.number_of_rounds = max(1, n_microbatches // self.pp_group_size)
        self.microbatches_per_round = n_microbatches // self.number_of_rounds
        if n_microbatches % self.number_of_rounds != 0:
            raise ValueError(
                "Zero bubble requires the number of microbatches to be a "
                f"multiple of the number of rounds ({self.number_of_rounds}), "
                f"but got {n_microbatches}."
            )
        # 1. Create the pipeline_order (all ranks do this calculation)
        # This will be used to keep track of the current state of the entire pipeline
        # pipeline_order[rank] = [Action(computation_type, microbatch_index, stage_index), ...]
        self.pipeline_order: dict[int, list[_Action | None]] = {}
        for rank in range(self.pp_group_size):
            rank_ops = self._calculate_single_rank_operations(rank)
            self.pipeline_order[rank] = rank_ops

        # This function add bubbles to the generated schedule based on dependencies of actions
        # Note that the ZB1P schedule will not require bubbles to be manually added and it is
        # only useful when n_microbatches <= microbatches_per_round
        self.pipeline_order = self._add_bubbles_to_actions(
            self.n_local_stages * self.pp_group_size,
        )

        # Initialize the pipeline order with communication necessary to run with _PipelineScheduleRuntime
        self._prepare_schedule_with_comms(self.pipeline_order)

    def _calculate_single_rank_operations(self, rank) -> list[_Action | None]:
        warmup_ops = _get_warmup_ops(
            rank,
            self.n_local_stages,
            self.microbatches_per_round,
            self.pp_group_size,
            self._n_microbatches,
            multiply_factor=1,
        )
        microbatch_ops = self.n_local_stages * self._n_microbatches
        # fwd_bwd_ops should encompass the remaining forwards
        fwd_bwd_ops = microbatch_ops - warmup_ops
        # cooldown_ops should encompass the remaining backwards
        cooldown_ops = microbatch_ops - fwd_bwd_ops
        # total ops encompass both forward and backward ops
        total_ops = warmup_ops + fwd_bwd_ops + cooldown_ops
        # warmup_ops + fwd_bwd_ops * 2 + cooldown_ops == microbatch_ops * 2
        logger.debug(
            "rank %s, warmup_ops %s, 1f1b %s, cooldown_ops %s total_ops %s",
            rank,
            warmup_ops,
            fwd_bwd_ops,
            cooldown_ops,
            total_ops,
        )

        # Calculates the stage index based on step and pp_group_size

        def forward_stage_index(step):
            # Get the local index from 0 to n_local_stages-1
            local_index = (step // self.microbatches_per_round) % self.n_local_stages
            return (local_index * self.pp_group_size) + rank

        def backward_stage_index(step):
            local_index = (
                self.n_local_stages
                - 1
                - ((step - warmup_ops) // self.microbatches_per_round)
                % self.n_local_stages
            )
            return (local_index * self.pp_group_size) + rank

        num_1f1b_microbatches = rank

        return _get_1f1b_rank_ops(
            self.n_local_stages,
            self.pp_group_size,
            warmup_ops,
            fwd_bwd_ops,
            cooldown_ops,
            rank,
            forward_stage_index,
            backward_stage_index,
            num_1f1b_microbatches,
            enable_zero_bubble=True,
        )

    def _add_bubbles_to_actions(self, num_stages_global):
        actions = self.pipeline_order

        def need_bubble(stage, op, microbatch, num_stages_global, seen_ops):
            if op == _ComputationType.FORWARD:
                if stage != 0 and (stage - 1, op, microbatch) not in seen_ops:
                    return True
            elif op == _ComputationType.FULL_BACKWARD:
                if stage == num_stages_global - 1:
                    return (stage, _ComputationType.FORWARD, microbatch) not in seen_ops
                return (stage + 1, op, microbatch) not in seen_ops
            return False

        seen_ops: set[tuple[int, _ComputationType, int]] = set()
        result: dict[int, list[_Action | None]] = {}
        next_pointer: dict[int, int] = {}
        bubbles_added: dict[int, int] = {}
        total_bubbles_added = 0

        for rank in range(self.pp_group_size):
            result[rank] = []
            next_pointer[rank] = 0
            bubbles_added[rank] = 0

        while True:
            should_stop = True

            temp_seen_ops: set[tuple[int, _ComputationType, int]] = set()

            for rank in range(self.pp_group_size):
                timestamp = next_pointer[rank]
                if timestamp >= len(actions[rank]):
                    continue

                should_stop = False

                if actions[rank][timestamp] is not None:
                    temp_action = actions[rank][timestamp]
                    if temp_action is None:
                        raise AssertionError(
                            f"Expected temp_action to be not None, got {type(temp_action)}"
                        )
                    stage_index, op, microbatch, _ = temp_action
                    if not need_bubble(
                        stage_index, op, microbatch, num_stages_global, seen_ops
                    ):
                        result[rank].append(actions[rank][timestamp])
                        if microbatch is not None:
                            temp_seen_ops.add((stage_index, op, microbatch))
                        next_pointer[rank] += 1
                    else:
                        result[rank].append(None)
                        bubbles_added[rank] += 1
                else:
                    next_pointer[rank] += 1
                    result[rank].append(None)

            seen_ops.update(temp_seen_ops)
            if should_stop:
                break

        if total_bubbles_added > 0:
            logger.warning(
                "Non zero bubbles added: total_bubbles_added=%s bubbles_added=%s",
                total_bubbles_added,
                bubbles_added,
            )
        return result


class ScheduleZBVZeroBubble(_PipelineScheduleRuntime):
    """
    The Zero Bubble schedule (ZBV variant).
    See https://arxiv.org/pdf/2401.10241 Section 6 for details.

    This schedules requires exactly two stages per rank.

    This schedule will perform one forward and one backward on inputs for the microbatches in steady
    state and supports multiple stages per rank. Uses backward with respect to weights to fill in
    the pipeline bubble.

    This ZB-V schedule would have the "zero bubble" property only if time forward == time backward input == time backward weights.
    In practice, this is not likely true for real models so alternatively
    a greedy scheduler could be implemented for unequal/unbalanced time.
    """

    def __init__(
        self,
        stages: list[_PipelineStageBase],
        n_microbatches: int,
        loss_fn: Callable | None = None,
        args_chunk_spec: tuple[TensorChunkSpec, ...] | None = None,
        kwargs_chunk_spec: dict[str, TensorChunkSpec] | None = None,
        output_merge_spec: dict[str, Any] | tuple[Any] | None = None,
        scale_grads: bool = True,
        backward_requires_autograd: bool = True,
    ):
        # TODO: we dont support input/weight backward split with torch.compile
        _check_torch_compile_compatibility(stages, self.__class__.__name__)
        self.pp_group_size = stages[0].group_size
        super().__init__(
            stages=stages,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            args_chunk_spec=args_chunk_spec,
            kwargs_chunk_spec=kwargs_chunk_spec,
            output_merge_spec=output_merge_spec,
            scale_grads=scale_grads,
            backward_requires_autograd=backward_requires_autograd,
        )
        self.stage_index_to_group_rank = generate_stage_to_rank_mapping(
            self.pp_group_size, self._num_stages, style="v"
        )
        for stage in self._stages:
            stage.stage_index_to_group_rank = self.stage_index_to_group_rank

        self.n_local_stages = len(stages)
        if self.n_local_stages != 2:
            raise ValueError(
                "ZBV requires exactly 2 stages per rank, but got "
                f"{self.n_local_stages}."
            )

        self.rank = stages[0].group_rank
        self.num_stages = stages[0].num_stages

        # 1. Create the pipeline_order (all ranks do this calculation)
        # This will be used to keep track of the current state of the entire pipeline
        # pipeline_order[rank] = [Action(computation_type, microbatch_index, stage_index), ...]
        self.pipeline_order: dict[int, list[_Action | None]] = {}
        for rank in range(self.pp_group_size):
            rank_ops = self._calculate_single_rank_operations(rank)
            self.pipeline_order[rank] = rank_ops

        # Initialize the pipeline order with communication necessary to run with _PipelineScheduleRuntime
        self._prepare_schedule_with_comms(self.pipeline_order)

    def _calculate_single_rank_operations(self, rank) -> list[_Action | None]:
        # max(2 * self.pp_group_size - 1, ...) ensure the number of microbatches is at least
        # as large of the number of microbatches needed to fully utilize the pipeline
        n_micro = max(2 * self.pp_group_size - 1, self._n_microbatches)
        rank_ops: list[_Action | None] = [None for _ in range(rank)]

        # Forward and backward action counts for stage chunk 0 and chunk 1
        f0_cnt, f1_cnt, b0_cnt, b1_cnt = 0, 0, 0, 0
        # warm-up phase
        warmup_n1 = 2 * (self.pp_group_size - rank) - 1
        stage_id_chunk0 = rank
        stage_id_chunk1 = self.num_stages - 1 - rank

        for _ in range(warmup_n1):
            rank_ops.append(
                _Action(stage_id_chunk0, computation_type=F, microbatch_index=f0_cnt)
            )
            f0_cnt += 1
        warmup_n2 = rank
        for _ in range(warmup_n2):
            rank_ops.append(
                _Action(stage_id_chunk1, computation_type=F, microbatch_index=f1_cnt)
            )
            f1_cnt += 1
            rank_ops.append(
                _Action(stage_id_chunk0, computation_type=F, microbatch_index=f0_cnt)
            )
            f0_cnt += 1
        warmup_n3 = self.pp_group_size - rank
        for _ in range(warmup_n3):
            rank_ops.append(
                _Action(stage_id_chunk1, computation_type=F, microbatch_index=f1_cnt)
            )
            f1_cnt += 1
            rank_ops.append(
                _Action(stage_id_chunk1, computation_type=I, microbatch_index=b1_cnt)
            )
            rank_ops.append(
                _Action(stage_id_chunk1, computation_type=W, microbatch_index=b1_cnt)
            )
            b1_cnt += 1
        # stable phase
        while f1_cnt < f0_cnt or f0_cnt < n_micro:
            if f0_cnt < n_micro:
                rank_ops.append(
                    _Action(
                        stage_id_chunk0, computation_type=F, microbatch_index=f0_cnt
                    )
                )
                f0_cnt += 1
            rank_ops.append(
                _Action(stage_id_chunk0, computation_type=I, microbatch_index=b0_cnt)
            )
            rank_ops.append(
                _Action(stage_id_chunk0, computation_type=W, microbatch_index=b0_cnt)
            )
            b0_cnt += 1

            rank_ops.append(
                _Action(stage_id_chunk1, computation_type=F, microbatch_index=f1_cnt)
            )
            f1_cnt += 1
            rank_ops.append(
                _Action(stage_id_chunk1, computation_type=I, microbatch_index=b1_cnt)
            )
            rank_ops.append(
                _Action(stage_id_chunk1, computation_type=W, microbatch_index=b1_cnt)
            )
            b1_cnt += 1
        # cool-down phase
        w0_cnt, w1_cnt = b0_cnt, b1_cnt
        cooldown_n1 = rank
        for _ in range(cooldown_n1):
            rank_ops.append(
                _Action(stage_id_chunk0, computation_type=I, microbatch_index=b0_cnt)
            )
            b0_cnt += 1
            rank_ops.append(
                _Action(stage_id_chunk1, computation_type=I, microbatch_index=b1_cnt)
            )
            b1_cnt += 1
        cooldown_n2 = self.pp_group_size - rank
        for _ in range(cooldown_n2):
            rank_ops.append(
                _Action(stage_id_chunk0, computation_type=I, microbatch_index=b0_cnt)
            )
            b0_cnt += 1
            rank_ops.append(
                _Action(stage_id_chunk0, computation_type=W, microbatch_index=w0_cnt)
            )
            w0_cnt += 1
        while w1_cnt < b1_cnt:
            rank_ops.append(
                _Action(stage_id_chunk1, computation_type=W, microbatch_index=w1_cnt)
            )
            w1_cnt += 1
        while w0_cnt < b0_cnt:
            rank_ops.append(
                _Action(stage_id_chunk0, computation_type=W, microbatch_index=w0_cnt)
            )
            w0_cnt += 1

        if not (w0_cnt == b0_cnt and b0_cnt == f0_cnt):
            raise AssertionError(
                f"Expected w0_cnt == b0_cnt == f0_cnt, got w0_cnt={w0_cnt}, b0_cnt={b0_cnt}, f0_cnt={f0_cnt}"
            )
        if not (w1_cnt == b1_cnt and b1_cnt == f1_cnt):
            raise AssertionError(
                f"Expected w1_cnt == b1_cnt == f1_cnt, got w1_cnt={w1_cnt}, b1_cnt={b1_cnt}, f1_cnt={f1_cnt}"
            )
        # We use max() in the n_micro computation above, so we may need to
        # remove redundant microbatches
        rank_ops = [
            (
                action
                if action is not None
                and action.microbatch_index is not None
                and action.microbatch_index < self._n_microbatches
                else None
            )
            for action in rank_ops
        ]
        return rank_ops


class ScheduleDualPipeV(_PipelineScheduleRuntime):
    """
    The DualPipeV schedule. A more efficient schedule variant based on the
    DualPipe schedule introduced by DeepSeek in https://arxiv.org/pdf/2412.19437

    Based on the open sourced code from https://github.com/deepseek-ai/DualPipe
    """

    def __init__(
        self,
        stages: list[_PipelineStageBase],
        n_microbatches: int,
        loss_fn: Callable | None = None,
        args_chunk_spec: tuple[TensorChunkSpec, ...] | None = None,
        kwargs_chunk_spec: dict[str, TensorChunkSpec] | None = None,
        output_merge_spec: dict[str, Any] | tuple[Any] | None = None,
        scale_grads: bool = True,
        backward_requires_autograd: bool = True,
    ):
        # TODO: we dont support input/weight backward split with torch.compile
        _check_torch_compile_compatibility(stages, self.__class__.__name__)
        self.pp_group_size = stages[0].group_size
        super().__init__(
            stages=stages,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            args_chunk_spec=args_chunk_spec,
            kwargs_chunk_spec=kwargs_chunk_spec,
            output_merge_spec=output_merge_spec,
            scale_grads=scale_grads,
            backward_requires_autograd=backward_requires_autograd,
        )
        self.stage_index_to_group_rank = generate_stage_to_rank_mapping(
            self.pp_group_size, self._num_stages, style="v"
        )
        for stage in self._stages:
            stage.stage_index_to_group_rank = self.stage_index_to_group_rank

        self.n_local_stages = len(stages)
        if self.n_local_stages != 2:
            raise ValueError(
                "ZBV requires exactly 2 stages per rank, but got "
                f"{self.n_local_stages}."
            )
        if n_microbatches < self._num_stages:
            raise ValueError(
                "DualPipeV requires at least as many microbatches as stages, but got "
                f"{n_microbatches} microbatches and {self._num_stages} stages."
            )

        self.rank = stages[0].group_rank
        self.num_stages = stages[0].num_stages

        # 1. Create the pipeline_order (all ranks do this calculation)
        # This will be used to keep track of the current state of the entire pipeline
        # pipeline_order[rank] = [Action(computation_type, microbatch_index, stage_index), ...]
        self.pipeline_order: dict[int, list[_Action | None]] = {}
        for rank in range(self.pp_group_size):
            rank_ops = self._calculate_single_rank_operations(rank)
            self.pipeline_order[rank] = rank_ops

        # Initialize the pipeline order with communication necessary to run with _PipelineScheduleRuntime
        self._prepare_schedule_with_comms(self.pipeline_order)

    def _calculate_single_rank_operations(self, rank) -> list[_Action | None]:
        actions: list[_Action | None] = []
        counters: dict[
            tuple[int, _ComputationType], int
        ] = {}  # (stage_index, computation_type) -> mb_index
        weight_queue = []  # Queue of (stage_index, mb_index) for pending weight actions

        num_ranks = self.pp_group_size
        num_chunks = self._n_microbatches

        rank_to_stages = generate_rank_to_stage_mapping(
            num_ranks, num_ranks * 2, style="v"
        )
        stage0_index, stage1_index = rank_to_stages[rank]

        def increment_backward_counts(stage_index: int):
            """Helper method to increment BACKWARD_INPUT and BACKWARD_WEIGHT counters when FULL_BACKWARD is used."""
            input_key = (stage_index, BACKWARD_INPUT)
            weight_key = (stage_index, BACKWARD_WEIGHT)
            counters[input_key] = counters.get(input_key, 0) + 1
            counters[weight_key] = counters.get(weight_key, 0) + 1

        def add_overlap_f_b(
            actions: list,
            forward_stage: int,
            backward_stage: int,
        ):
            """Helper method to add an overlapped forward+backward action which tracks microbatch index."""
            # Create new overlapped forward+backward action with sub_actions
            forward_key = (forward_stage, FORWARD)
            backward_key = (backward_stage, BACKWARD_INPUT)

            forward_mb = counters.get(forward_key, 0)
            backward_mb = counters.get(backward_key, 0)

            sub_actions = (
                _Action(forward_stage, FORWARD, forward_mb),
                _Action(backward_stage, FULL_BACKWARD, backward_mb),
            )
            actions.append(_Action(-1, OVERLAP_F_B, None, sub_actions))

            # Update counters for sub_actions
            counters[forward_key] = forward_mb + 1
            increment_backward_counts(backward_stage)

        def add_action(
            actions: list,
            stage_index: int,
            computation_type: _ComputationType,
        ):
            # Regular single action, for FULL_BACKWARD we only use the BACKWARD_INPUT counter
            key = (
                (stage_index, computation_type)
                if computation_type != FULL_BACKWARD
                else (stage_index, BACKWARD_INPUT)
            )
            mb_index = counters.get(key, 0)
            actions.append(_Action(stage_index, computation_type, mb_index))

            # If FULL_BACKWARD is used, just increment the separate BACKWARD_INPUT and BACKWARD_WEIGHT counters
            if computation_type == FULL_BACKWARD:
                increment_backward_counts(stage_index)
            else:
                # If BACKWARD_INPUT is updated, add corresponding weight action to queue
                if computation_type == BACKWARD_INPUT:
                    # Add weight action to queue for later processing
                    weight_queue.append((stage_index, mb_index))
                counters[key] = mb_index + 1

        def add_weight_action_if_pending(actions: list):
            """Helper method to add a weight action from the queue."""
            if not weight_queue:
                return  # No pending weight actions, skip
            # Pop the oldest weight action from the queue
            actual_stage_index, weight_mb_index = weight_queue.pop(0)
            actions.append(
                _Action(
                    actual_stage_index,
                    BACKWARD_WEIGHT,
                    weight_mb_index,
                )
            )
            # Update the counter for the actual stage that was processed
            weight_key = (actual_stage_index, BACKWARD_WEIGHT)
            counters[weight_key] = counters.get(weight_key, 0) + 1

        # Step 1: F0
        step_1 = (num_ranks - rank - 1) * 2
        for _ in range(step_1):
            add_action(actions, stage0_index, FORWARD)

        # Step 2: F0F1
        step_2 = rank + 1
        for _ in range(step_2):
            add_action(actions, stage0_index, FORWARD)
            add_action(actions, stage1_index, FORWARD)

        # Step 3: I1W1F1 (Use zero bubble)
        step_3 = num_ranks - rank - 1
        for _ in range(step_3):
            add_action(actions, stage1_index, BACKWARD_INPUT)
            add_weight_action_if_pending(actions)
            add_action(actions, stage1_index, FORWARD)

        # Step 4 (Main step): F0B1-F1B0 (combined, overlapped forward+backward)
        step_4 = num_chunks - num_ranks * 2 + rank + 1
        for i in range(step_4):
            if i == 0 and rank == num_ranks - 1:
                # NOTE: We don't overlap these two chunks to further reduce bubble size.
                add_action(actions, stage0_index, FORWARD)
                add_action(actions, stage1_index, FULL_BACKWARD)
            else:
                add_overlap_f_b(
                    actions,
                    forward_stage=stage0_index,
                    backward_stage=stage1_index,
                )
            add_overlap_f_b(
                actions,
                forward_stage=stage1_index,
                backward_stage=stage0_index,
            )

        # Step 5: B1-F1B0
        step_5 = num_ranks - rank - 1
        for _ in range(step_5):
            add_action(actions, stage1_index, FULL_BACKWARD)
            add_overlap_f_b(
                actions,
                forward_stage=stage1_index,
                backward_stage=stage0_index,
            )

        # Step 6: B1B0 (The second half of the chunks use zero bubble)
        step_6 = rank + 1
        enable_zb = False
        for i in range(step_6):
            if i == step_6 // 2 and rank % 2 == 1:
                enable_zb = True
            comp_type = BACKWARD_INPUT if enable_zb else FULL_BACKWARD
            add_action(actions, stage1_index, comp_type)
            if i == step_6 // 2 and rank % 2 == 0:
                enable_zb = True
            comp_type = BACKWARD_INPUT if enable_zb else FULL_BACKWARD
            add_action(actions, stage0_index, comp_type)

        # Step 7: W0B0
        step_7 = num_ranks - rank - 1
        for _ in range(step_7):
            add_weight_action_if_pending(actions)
            comp_type = BACKWARD_INPUT if enable_zb else FULL_BACKWARD
            add_action(actions, stage0_index, comp_type)

        # Step 8: W0
        step_8 = rank + 1
        for _ in range(step_8):
            add_weight_action_if_pending(actions)

        return actions


def get_schedule_class(schedule_name: str):
    """
    Maps a schedule name (case insensitive) to its corresponding class object.

    Args:
        schedule_name (str): The name of the schedule.
    """
    schedule_map = {
        "1F1B": Schedule1F1B,
        "Interleaved1F1B": ScheduleInterleaved1F1B,
        "GPipe": ScheduleGPipe,
        "LoopedBFS": ScheduleLoopedBFS,
        "InterleavedZeroBubble": ScheduleInterleavedZeroBubble,
        "PipelineScheduleSingle": PipelineScheduleSingle,
        "PipelineScheduleMulti": PipelineScheduleMulti,
        "ZBVZeroBubble": ScheduleZBVZeroBubble,
        "DualPipeV": ScheduleDualPipeV,
    }
    lowercase_keys = {k.lower(): k for k in schedule_map}
    lowercase_schedule_name = schedule_name.lower()
    if lowercase_schedule_name not in lowercase_keys:
        raise ValueError(
            f"Unknown schedule name '{schedule_name}'. The valid options are {list(schedule_map.keys())}"
        )
    return schedule_map[lowercase_keys[lowercase_schedule_name]]


def _check_torch_compile_compatibility(
    stages: list[_PipelineStageBase], schedule_name: str
):
    """
    Check if the schedule is compatible with torch.compile.

    Args:
        stages: List of pipeline stages to check
        schedule_name: Name of the schedule for error message

    Raises:
        RuntimeError: If any stage uses torch.compile
    """
    for stage in stages:
        if not isinstance(stage.submod, torch.nn.Module):
            continue

        for module in stage.submod.modules():
            if isinstance(module, OptimizedModule):
                raise RuntimeError(
                    f"The {schedule_name} schedule is not supported with "
                    "stage modules that have used torch.compile. "
                    f"Found OptimizedModule in {type(module).__name__}"
                )
