# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

import copy
import csv
import itertools
import logging
import math
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    TYPE_CHECKING,
    Union,
)

import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp.fully_shard import FSDPModule, UnshardHandle

from torch.profiler import record_function

from .microbatch import merge_chunks, split_args_kwargs_into_chunks, TensorChunkSpec
from .stage import _PipelineStageBase

if TYPE_CHECKING:
    from torch.distributed import Work

__all__ = [
    "PipelineScheduleSingle",
    "PipelineScheduleMulti",
    "Schedule1F1B",
    "ScheduleFlexibleInterleaved1F1B",
    "ScheduleGPipe",
    "ScheduleInterleaved1F1B",
    "ScheduleLoopedBFS",
]

logger = logging.getLogger(__name__)


class _ComputationType(Enum):
    # TODO(whc) rename to _ActType?
    FORWARD = 1
    BACKWARD = 2
    WEIGHT = 3
    UNSHARD = 4
    RESHARD = 5
    SEND_F = 6
    RECV_F = 7
    SEND_B = 8
    RECV_B = 9
    SEND_F_RECV_B = 10
    SEND_B_RECV_F = 11
    # TODO- probably want to reconsider naming backward_input 'B' and having 'FULL_BACKWARD'.
    # instead, B = full backward, Bx, Bw are the partials?
    FULL_BACKWARD = 12

    def __str__(self):
        str_map = {
            _ComputationType.FORWARD: "F",
            _ComputationType.BACKWARD: "B",
            _ComputationType.WEIGHT: "W",
            _ComputationType.UNSHARD: "UNSHARD",
            _ComputationType.RESHARD: "RESHARD",
            _ComputationType.SEND_F: "SEND_F",
            _ComputationType.RECV_F: "RECV_F",
            _ComputationType.SEND_B: "SEND_B",
            _ComputationType.RECV_B: "RECV_B",
            _ComputationType.SEND_F_RECV_B: "SEND_F_RECV_B",
            _ComputationType.SEND_B_RECV_F: "SEND_B_RECV_F",
            _ComputationType.FULL_BACKWARD: "BW",
        }
        return str_map[self]

    @staticmethod
    def from_str(action):
        if action == "F":
            return _ComputationType.FORWARD
        elif action == "B":
            return _ComputationType.BACKWARD
        elif action == "W":
            return _ComputationType.WEIGHT
        elif action == "UNSHARD":
            return _ComputationType.UNSHARD
        elif action == "RESHARD":
            return _ComputationType.RESHARD
        elif action == "SEND_F":
            return _ComputationType.SEND_F
        elif action == "RECV_F":
            return _ComputationType.RECV_F
        elif action == "SEND_B":
            return _ComputationType.SEND_B
        elif action == "RECV_B":
            return _ComputationType.RECV_B
        elif action == "SEND_F_RECV_B":
            return _ComputationType.SEND_F_RECV_B
        elif action == "SEND_B_RECV_F":
            return _ComputationType.SEND_B_RECV_F
        elif action == "BW":
            return _ComputationType.FULL_BACKWARD
        else:
            raise RuntimeError(f"Invalid computation type {action}")


FORWARD = _ComputationType.FORWARD
BACKWARD = _ComputationType.BACKWARD
WEIGHT = _ComputationType.WEIGHT
UNSHARD = _ComputationType.UNSHARD
RESHARD = _ComputationType.RESHARD
SEND_F = _ComputationType.SEND_F
RECV_F = _ComputationType.RECV_F
SEND_B = _ComputationType.SEND_B
RECV_B = _ComputationType.RECV_B
SEND_F_RECV_B = _ComputationType.SEND_F_RECV_B
SEND_B_RECV_F = _ComputationType.SEND_B_RECV_F
FULL_BACKWARD = _ComputationType.FULL_BACKWARD

# Convenience shorthand for compute actions only since they are used in 'simple schedule format'
F = FORWARD
B = BACKWARD
W = WEIGHT
BW = FULL_BACKWARD

# Helper to parse an action string like 1F0 into a tuple of (stage_index, computation_type, microbatch_index)
_action_regex = re.compile(
    r"(\d+)(F|BW|B|W|UNSHARD|RESHARD|SEND_F|RECV_F|SEND_B|RECV_B){0,1}(\d*)(_(\d+)(RECV_B|RECV_F)(\d+)){0,1}"
)


class ZeroBubbleAlgorithm(Enum):
    ZB1P = 1
    ZB2P = 2
    ZBV = 3

    def __str__(self):
        str_map = {
            ZeroBubbleAlgorithm.ZB1P: "ZB1P",
            ZeroBubbleAlgorithm.ZB2P: "ZB2P",
            ZeroBubbleAlgorithm.ZBV: "ZBV",
        }
        return str_map[self]


class _Action(NamedTuple):
    stage_index: int
    computation_type: _ComputationType
    microbatch_index: Optional[int] = None
    # Used only for batched comms, for the second comm
    other_stage_index: Optional[int] = None
    other_microbatch_index: Optional[int] = None

    def __repr__(self):
        repr = str(self.stage_index)
        if self.computation_type == SEND_B_RECV_F:
            assert (
                self.microbatch_index is not None
            ), "SEND_B_RECV_F requires microbatch_index"
            assert (
                self.other_stage_index is not None
            ), "SEND_B_RECV_F requires other_stage_index"
            assert (
                self.other_microbatch_index is not None
            ), "SEND_B_RECV_F requires other_microbatch_index"
            repr += str(SEND_B) + str(self.microbatch_index)
            repr += "_" + str(self.other_stage_index)
            repr += str(RECV_F) + str(self.other_microbatch_index)
        elif self.computation_type == SEND_F_RECV_B:
            assert (
                self.microbatch_index is not None
            ), "SEND_F_RECV_B requires microbatch_index"
            assert (
                self.other_stage_index is not None
            ), "SEND_F_RECV_B requires other_stage_index"
            assert (
                self.other_microbatch_index is not None
            ), "SEND_F_RECV_B requires other_microbatch_index"
            repr += str(SEND_F) + str(self.microbatch_index)
            repr += "_" + str(self.other_stage_index)
            repr += str(RECV_B) + str(self.other_microbatch_index)
        else:
            repr += str(self.computation_type)
            if self.microbatch_index is not None:
                repr += str(self.microbatch_index)
        return repr

    @staticmethod
    def from_str(str):
        """
        Reverse of __repr__

        String should be formatted as [stage][action type][(microbatch)]
            e.g. `2F0`, `1UNSHARD`, `3SEND_F1`
        """
        if match := _action_regex.match(str):
            # the _ is for the combined group that captures the whole second action
            (
                stage_index,
                computation_type,
                microbatch_index,
                _,
                other_stage_index,
                other_computation_type,
                other_microbatch_index,
            ) = match.groups()
            if other_computation_type is not None:
                assert (
                    other_stage_index is not None and other_microbatch_index is not None
                )
                return _Action(
                    int(stage_index),
                    _ComputationType.from_str(
                        f"{computation_type}_{other_computation_type}"
                    ),
                    int(microbatch_index) if len(microbatch_index) else None,
                    int(other_stage_index),
                    int(other_microbatch_index),
                )
            return _Action(
                int(stage_index),
                _ComputationType.from_str(computation_type),
                int(microbatch_index) if len(microbatch_index) else None,
            )
        elif str == "" or str.isspace():
            return None
        raise RuntimeError(
            f"Invalid action string: {str}, should be formatted as [stage][action type][(microbatch)] e.g. 2F0"
        )


def _format_pipeline_order(pipeline_order: Dict[int, List[Optional[_Action]]]) -> str:
    """
    Formats the pipeline order in a timestep (row) x rank (column) grid of actions
    and returns the formatted string
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
        for label, row in zip(step_labels, transposed_actions)
    ]
    # Join the rows into a single string
    formatted_table = header_row + "\n" + "\n".join(formatted_rows) + "\n"
    return formatted_table


def _validate_pipeline_order(
    pipeline_order: Dict[int, List[Optional[_Action]]],
    num_microbatches: int,
    num_stages: int,
    enable_zero_bubble: bool = False,
):
    """
    pipeline_order[rank] = [(computation_type, microbatch_index, stage_index), ...]
    Validating that the pipeline order follows the rules:
    1. Forward action for a microbatch must be before the Backward action for that microbatch
    2. Recv for a microbatch must be before the send for that microbatch
    3. Microbatch index is handled in sequential order for each stage
    4. A later stage cannot operate on a microbatch before any of the previous stages have operated on it
    5. Same microbatch cannot be handled in the same time step across ranks
    """
    # microbatch_index: (current computation type, current stage)
    microbatch_process_info: Dict[int, Tuple[_ComputationType, int]] = {}
    max_timestep = max(len(rank_list) for rank_list in pipeline_order.values())
    for timestep in range(max_timestep):
        error_msg: List[str] = []
        current_timestep_actions = []
        for rank in range(len(pipeline_order)):
            action = (
                pipeline_order[rank][timestep]
                if timestep < len(pipeline_order[rank])
                else None
            )

            if action is not None:
                computation_type = action.computation_type
                if computation_type != _ComputationType.WEIGHT:
                    current_timestep_actions.append(action)

        # TODO: enable this
        # if len(current_timestep_actions) == 0:
        #     error_msg.append(
        #         "All actions were None, there is an unnecessary gap in the schedule"
        #     )

        # Ensure that no microbatch is operated on twice in current_timestep_actions
        unique_microbatch_indices = {
            action.microbatch_index for action in current_timestep_actions
        }
        if len(unique_microbatch_indices) != len(current_timestep_actions):
            error_msg.append(
                "Duplicate microbatch index found in current_timestep_actions"
            )

        for action in current_timestep_actions:
            stage_index = action.stage_index
            computation_type = action.computation_type
            mb_index = action.microbatch_index
            assert (
                mb_index is not None
            ), "All currently supported action types require valid microbatch_index"
            if mb_index >= num_microbatches:
                error_msg.append(f"Microbatch index {mb_index} out of range")

            # first microbatch
            if mb_index not in microbatch_process_info:
                if computation_type != _ComputationType.FORWARD or stage_index != 0:
                    error_msg.append(f"Incorrect start for microbatch {mb_index}")
                microbatch_process_info[mb_index] = (computation_type, stage_index)
            else:
                # if the microbatch is included, check that the current stage is right after prev
                prev_computation, prev_stage = microbatch_process_info[mb_index]

                if prev_computation == _ComputationType.FORWARD:
                    if prev_stage == num_stages - 1:
                        expected_stage = num_stages - 1
                        expected_computation = _ComputationType.BACKWARD
                    else:
                        expected_stage = prev_stage + 1
                        expected_computation = _ComputationType.FORWARD
                elif prev_computation == _ComputationType.BACKWARD:
                    if prev_stage == 0:
                        error_msg.append(
                            f"[{mb_index=}] already finished backward computation"
                        )
                        break
                    else:
                        expected_stage = prev_stage - 1
                        expected_computation = _ComputationType.BACKWARD
                else:
                    raise ValueError(
                        f"Computation type {prev_computation} not supported"
                    )

                if expected_computation is not None:
                    if expected_computation != computation_type:
                        error_msg.append(
                            f"[{mb_index=}] {expected_computation=} VS. actual {computation_type=}"
                        )

                if expected_stage != stage_index:
                    error_msg.append(
                        f"[{mb_index=}] {expected_stage=} VS. actual {stage_index=}"
                    )

                microbatch_process_info[mb_index] = (
                    expected_computation,
                    expected_stage,
                )

        if not enable_zero_bubble:
            if len(error_msg) != 0:
                raise RuntimeError(
                    f"Error at timestep {timestep}: " + ",".join(error_msg)
                )
            return

        for rank in range(len(pipeline_order)):
            backward_steps: Set[Tuple[int, int]] = set()
            weight_steps: Set[Tuple[int, int]] = set()

            for action in pipeline_order[rank]:
                if action is None:
                    continue

                stage_index = action.stage_index
                computation_type = action.computation_type
                mb_index = action.microbatch_index
                if computation_type == _ComputationType.BACKWARD:
                    if mb_index is not None:
                        backward_steps.add((mb_index, stage_index))
                elif computation_type == _ComputationType.WEIGHT:
                    if (mb_index, stage_index) not in backward_steps:
                        error_msg.append(
                            f"{mb_index=}, {stage_index=} Weight happened before bwd"
                        )
                    if (mb_index, stage_index) in weight_steps:
                        error_msg.append(
                            f"{mb_index=}, {stage_index=} Duplicated weight step"
                        )
                    if mb_index is not None:
                        weight_steps.add((mb_index, stage_index))

            if len(backward_steps) != len(weight_steps):
                error_msg.append("Length weight steps != Length bwd steps")

        if len(error_msg) != 0:
            raise RuntimeError(f"Error at timestep {timestep}: " + ",".join(error_msg))


class _PipelineSchedule(ABC):
    def __init__(
        self,
        n_microbatches: int,
        loss_fn: Optional[Callable[..., torch.Tensor]] = None,
        args_chunk_spec: Optional[Tuple[TensorChunkSpec, ...]] = None,
        kwargs_chunk_spec: Optional[Dict[str, TensorChunkSpec]] = None,
        output_merge_spec: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
    ):
        # From arguments
        self._n_microbatches = n_microbatches
        self._loss_fn = loss_fn
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
        self._internal_losses: List[torch.Tensor] = []
        logger.info(f"Using {self.__class__.__name__}")  # noqa: G004

    def _maybe_compute_loss(self, stage, output, target_mbs, mb_index):
        if stage.is_last and self._has_backward:
            loss = self._compute_loss(output, target_mbs[mb_index])  # type: ignore[index]
            self._internal_losses.append(loss)

    def _maybe_get_loss(self, stage, mb_index):
        valid_index = 0 <= mb_index < len(self._internal_losses)
        if stage.is_last and self._has_backward and valid_index:
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
        arg_mbs: Optional[List] = None,
        kwarg_mbs: Optional[List] = None,
        target_mbs: Optional[List] = None,
        losses: Optional[List] = None,
    ):
        """
        Run one iteration of the pipeline schedule with list of microbatches.
        Will go through all the microbatches according to the schedule
        implementation.

        Args:
            microbatches: list of microbatch args.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, *args, target=None, losses: Optional[List] = None, **kwargs):
        """
        Run one iteration of the pipeline schedule with *whole-batch* input.
        Will chunk the input into microbatches automatically, and go through the
        microbatches according to the schedule implementation.

        args: positional arguments to the model (as in non-pipeline case).
        kwargs: keyword arguments to the model (as in non-pipeline case).
        target: target for the loss function.
        losses: a list to store the losses for each microbatch.
        """
        raise NotImplementedError

    def _check_inputs(
        self,
        arg_mbs: Optional[List] = None,
        kwarg_mbs: Optional[List] = None,
        target_mbs: Optional[List] = None,
        losses: Optional[List] = None,
    ):
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
        args: Tuple[Any, ...],
        kwargs: Optional[Dict[str, Any]] = None,
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

    def _merge_outputs(self, output_chunks: List[Any]) -> Any:
        """
        Merge output chunks back to a batch state.
        If output_merge_spec is None, the utility will merge output chunks by dimension 0 (batch dim).
        """
        return merge_chunks(
            output_chunks,
            self._output_merge_spec,
        )


def _batch_p2p(p2p_ops: List[dist.P2POp], desc: Optional[str] = None):
    """
    Simple wrapper over batch_isend_irecv from torch.distributed, which just adds a descriptive logger on top.
    """
    if len(p2p_ops) == 0:
        return None
    desc_str = f"{desc}, " if desc else ""
    logger.debug(f"batch_p2p {desc_str}{p2p_ops}")  # noqa: G004
    return dist.batch_isend_irecv(p2p_ops).pop()


def _sorted_batch_p2p(
    p2p_ops: List[dist.P2POp], desc: Optional[str] = None
) -> Dict[int, dist.Work]:
    """
    Sorts the list of P2P ops by the peer rank, and then calls
    batch_isend_irecv. Return a dictionary of works by peer rank. This function
    helps us avoid hangs in case of skip connections.
    """
    # Arrange p2p_ops by peer rank:
    #   int is the peer rank;
    #   List is the list of ops towards the peer
    ops_by_peer: Dict[int, List[dist.P2POp]] = defaultdict(list)
    work_by_peer: Dict[int, dist.Work] = {}
    if len(p2p_ops) == 0:
        return work_by_peer

    # Classify the ops by peer rank
    for op in p2p_ops:
        ops_by_peer[op.peer].append(op)

    # Call batch_isend_irecv per peer, in sorted order of the peers (to avoid hangs)
    for peer, ops in sorted(ops_by_peer.items()):
        work_by_peer[peer] = _batch_p2p(ops, desc=desc)

    return work_by_peer


class PipelineScheduleSingle(_PipelineSchedule):
    """
    Base class for single-stage schedules.
    Implements the `step` method.
    Derived classes should implement `_step_microbatches`.
    """

    def __init__(
        self,
        stage: _PipelineStageBase,
        n_microbatches: int,
        loss_fn: Optional[Callable] = None,
        args_chunk_spec: Optional[Tuple[TensorChunkSpec, ...]] = None,
        kwargs_chunk_spec: Optional[Dict[str, TensorChunkSpec]] = None,
        output_merge_spec: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
    ):
        # Init parent
        super().__init__(
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            args_chunk_spec=args_chunk_spec,
            kwargs_chunk_spec=kwargs_chunk_spec,
            output_merge_spec=output_merge_spec,
        )
        # Self attributes
        self._stage = stage
        self._num_stages = stage.num_stages
        # Set the same has_backward flag for stage object
        self._stage.has_backward = self._has_backward

        # TODO: later replace this with lazy shape inference during forward
        # Prepare forward send/recv infrastructure for stage
        stage._prepare_forward_infra(n_microbatches)
        if self._has_backward:
            stage._prepare_backward_infra(n_microbatches)

    def step(self, *args, target=None, losses: Optional[List] = None, **kwargs):
        """
        Run one iteration of the pipeline schedule with *whole-batch* input.
        Will chunk the input into microbatches automatically, and go through the
        microbatches according to the schedule implementation.

        args: positional arguments to the model (as in non-pipeline case).
        kwargs: keyword arguments to the model (as in non-pipeline case).
        target: target for the loss function.
        losses: a list to store the losses for each microbatch.
        """

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
        self._step_microbatches(args_split, kwargs_split, targets_split, losses)

        # Return merged results per original format
        if self._stage.is_last:
            return self._merge_outputs(self._stage.output_chunks)
        else:
            return None


class ScheduleGPipe(PipelineScheduleSingle):
    """
    The GPipe schedule.
    Will go through all the microbatches in a fill-drain manner.
    """

    def _step_microbatches(
        self,
        arg_mbs: Optional[List] = None,
        kwarg_mbs: Optional[List] = None,
        target_mbs: Optional[List] = None,
        losses: Optional[List] = None,
    ):
        """
        Run one iteration of the pipeline schedule with list of microbatches.
        Will go through all the microbatches according to the GPipe schedule.

        Args:
            microbatches: list of microbatch args.
        """
        arg_mbs, kwarg_mbs = self._check_inputs(arg_mbs, kwarg_mbs, target_mbs, losses)

        # Delay send waits
        fwd_sends_to_wait: List[dist.Work] = []

        # Run microbatches
        for i in range(self._n_microbatches):
            with record_function(f"Forward {i}"):
                ops = self._stage.get_fwd_recv_ops(i)
                works = _sorted_batch_p2p(ops, desc="fwd_recv")
                for work in works.values():
                    work.wait()

                output = self._stage.forward_one_chunk(i, arg_mbs[i], kwarg_mbs[i])  # type: ignore[index]

                ops = self._stage.get_fwd_send_ops(i)
                works = _sorted_batch_p2p(ops, desc="fwd_send")
                fwd_sends_to_wait.extend(works.values())

            logger.debug(
                f"[{self._stage.stage_index}] Forwarded microbatch {i}"  # noqa: G004
            )

            self._maybe_compute_loss(self._stage, output, target_mbs, i)

        # Wait for all forward sends to finish
        # This should not have performance impact because by the time the first
        # backward arrives all the forward sends should have been finished.
        for work in fwd_sends_to_wait:
            work.wait()

        # No loss function, no need to run backward
        if not self._has_backward:
            return

        # Run backward
        # Delay send waits
        bwd_sends_to_wait: List[dist.Work] = []
        for i in range(self._n_microbatches):
            with record_function(f"Backward {i}"):
                ops = self._stage.get_bwd_recv_ops(i)
                works = _sorted_batch_p2p(ops, desc="bwd_recv")
                for work in works.values():
                    work.wait()

                loss = self._maybe_get_loss(self._stage, i)
                self._stage.backward_one_chunk(i, loss=loss)

                ops = self._stage.get_bwd_send_ops(i)
                works = _sorted_batch_p2p(ops, desc="bwd_send")
                bwd_sends_to_wait.extend(works.values())

            logger.debug(
                f"[{self._stage.stage_index}] Backwarded microbatch {i}"  # noqa: G004
            )

        # Return losses if there is a container passed in
        self._update_losses(self._stage, losses)

        # Wait for all backward sends to finish
        for work in bwd_sends_to_wait:
            work.wait()


class Schedule1F1B(PipelineScheduleSingle):
    """
    The 1F1B schedule.
    Will perform one forward and one backward on the microbatches in steady state.
    """

    def _step_microbatches(
        self,
        arg_mbs: Optional[List] = None,
        kwarg_mbs: Optional[List] = None,
        target_mbs: Optional[List] = None,
        losses: Optional[List] = None,
    ):
        """
        Run one iteration of the pipeline schedule with list of microbatches.
        Will go through all the microbatches according to the 1F1B schedule.

        Args:
            microbatches: list of microbatch args.
        """
        arg_mbs, kwarg_mbs = self._check_inputs(arg_mbs, kwarg_mbs, target_mbs, losses)

        # Last stage has 1 warmup, second-to-last 2 warmups, ...
        # first stage `num_stages` warmups
        warmup_chunks = min(
            self._n_microbatches,
            self._num_stages - self._stage.stage_index,
        )

        # Chunk counters
        fwd_mb_index = 0
        bwd_mb_index = 0
        weight_stage_mb_index = 0

        # Warmup phase
        send_work = None
        fwd_sends = []
        for _ in range(warmup_chunks):
            # Receive activations
            fwd_recvs = self._stage.get_fwd_recv_ops(fwd_mb_index)
            if recv_work := _batch_p2p(fwd_recvs, desc="fwd_recv"):
                recv_work.wait()

            # Compute
            output = self._stage.forward_one_chunk(fwd_mb_index, arg_mbs[fwd_mb_index], kwarg_mbs[fwd_mb_index])  # type: ignore[index]

            # Clear previous chunk's forward sends (hopefully they have well
            # finished, otherwise, we are heavily communication bound, in which
            # case it doesn't create a lot of benefit to compute next chunk
            # eagerly either)
            if send_work:
                send_work.wait()

            # Send activations
            fwd_sends = self._stage.get_fwd_send_ops(fwd_mb_index)
            if fwd_mb_index != warmup_chunks - 1:
                # Safe to fire
                send_work = _batch_p2p(fwd_sends, desc="fwd_send")
            # otherwise:
            #   The last foward send is left for fuse with first 1B in 1B1F below

            # Compute loss
            self._maybe_compute_loss(self._stage, output, target_mbs, fwd_mb_index)
            fwd_mb_index += 1

        # Now we should have send ops left over, to be fused with first 1B of 1B1F phase below.

        # 1B1F phase
        while True:  # Don't worry, we have a break inside
            # We actually do 1B first as the `1B1F` name indicates, so prepare its recv ops
            bwd_recvs = self._stage.get_bwd_recv_ops(bwd_mb_index)

            # Now, we need to fire the fwd_sends and bwd_recvs together
            if fuse_work := _batch_p2p(fwd_sends + bwd_recvs, desc="fwd_send_bwd_recv"):
                fuse_work.wait()

            # Backward one chunk
            loss = self._maybe_get_loss(self._stage, bwd_mb_index)
            self._stage.backward_one_chunk(bwd_mb_index, loss=loss)

            # Get the bwd send ops, but don't fire, to be fused with the 1F below
            bwd_sends = self._stage.get_bwd_send_ops(bwd_mb_index)
            bwd_mb_index += 1

            if fwd_mb_index == self._n_microbatches:
                # We are done with 1B1F, so break with some left-over bwd_sends
                break

            # We prepare 1F of the `1B1F`
            fwd_recvs = self._stage.get_fwd_recv_ops(fwd_mb_index)

            # Fuse it with bwd_sends above
            if fuse_work := _batch_p2p(bwd_sends + fwd_recvs, desc="bwd_send_fwd_recv"):
                fuse_work.wait()

            # Now do the fwd
            output = self._stage.forward_one_chunk(fwd_mb_index, arg_mbs[fwd_mb_index], kwarg_mbs[fwd_mb_index])  # type: ignore[index]

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
            if recv_work := _batch_p2p(bwd_recvs, desc="bwd_recv"):
                recv_work.wait()

            # Backward one chunk
            loss = self._maybe_get_loss(self._stage, bwd_mb_index)
            self._stage.backward_one_chunk(bwd_mb_index, loss=loss)

            # Clear previous chunk's backward sends (hopefully they have well finished)
            if send_work:
                send_work.wait()

            # Get the bwd send ops, fire it
            bwd_sends = self._stage.get_bwd_send_ops(bwd_mb_index)
            send_work = _batch_p2p(bwd_sends, desc="bwd_send")
            bwd_mb_index += 1

        # Wait for the last backward send to finish
        if send_work:
            send_work.wait()

        # Return losses if there is a container passed in
        self._update_losses(self._stage, losses)


def _add_unshard_reshard(
    compute_actions: List[Optional[_Action]],
    max_active_stages: int = 3,
) -> List[_Action]:
    """Given a basic schedule involving only compute actions (F,B,W), add UNSHARD/RESHARD actions for FSDP.

    UNSHARD refers to fetching the full contents of an FSDP-sharded layer, requiring an all-gather operation.
    RESHARD does the opposite, releasing memory (but doing no commmunication)

    We abandon the "timestep lock"  during lowering

    max_active_stages controls how many prefetches we allow. It should be measured in mb and tuneable but in practice
    3 stages is probably the thing we want?
    (to account for having one f and one b active, and something else prefetching?)
    """

    def next_stage_indices(
        count: int, next_actions: List[Optional[_Action]]
    ) -> List[int]:
        """Remove duplicates (same stage, different microbatch), find next 'count' stages that will do compute."""
        seen: Set[int] = set()
        ret: List[int] = []

        for a in next_actions:
            if a is not None and a.stage_index not in seen:
                seen.add(a.stage_index)
                ret.append(a.stage_index)
                if len(ret) == count:
                    break
        return ret

    active_stages: Set[int] = set()
    fsdp_aware_actions: List[_Action] = []

    def _unshard(stage_index: int):
        active_stages.add(stage_index)
        fsdp_aware_actions.append(_Action(stage_index, UNSHARD, None))

    def _reshard(stage_index: int):
        active_stages.remove(stage_index)
        fsdp_aware_actions.append(_Action(stage_index, RESHARD, None))

    for i, action in enumerate(compute_actions):
        if action is None:
            continue

        # We prefetch the next N stages we'll see, dropping existing stages to make room
        next_n = next_stage_indices(max_active_stages, compute_actions[i:])
        # Fetch needs to be ordered correctly, so don't use a set
        fetch = list(filter(lambda s: s not in active_stages, next_n))
        # Unclear what the best policy is for eviction, but we can maintain order so we do
        evict = list(filter(lambda s: s not in next_n, active_stages))

        # logger.debug(
        #     "_add_unshard_reshard Step %d active: %s fetch %s, evict %s",
        #     i,
        #     active_stages,
        #     fetch,
        #     evict,
        # )

        for stage in evict:
            _reshard(stage)
        for stage in fetch:
            _unshard(stage)
        fsdp_aware_actions.append(action)

    return fsdp_aware_actions


def _merge_bw(
    compute_actions: List[Optional[_Action]],
) -> List[_Action]:
    """Given a basic schedule involving only compute actions (F,B,W), merge adjacent B and W ops into BW ops.

    BW refers to running the whole backward (not separating grad_input and grad_weight), which can be more efficient
    in some cases.
    """
    merged_actions = []
    while compute_actions:
        action = compute_actions.pop(0)
        if action is None:
            continue

        while len(compute_actions) and (next_action := compute_actions[0]) is None:
            # remove any None actions between 'action' and 'next_action'
            compute_actions.pop(0)

        if (
            action.computation_type == B
            and next_action is not None
            and next_action.computation_type == W
            and action.stage_index == next_action.stage_index
            and action.microbatch_index == next_action.microbatch_index
        ):
            merged_actions.append(
                _Action(action.stage_index, BW, action.microbatch_index)
            )
            compute_actions.pop(0)
        else:
            merged_actions.append(action)
    return merged_actions


def _batch_send_recv(ops, peer_ops):
    # we intentionally mutate ops, peer_ops so the caller knows we consumed them.  maybe i should revsit that.
    new_ops = []
    new_peer_ops = []
    send_f_i = [i for i, o in enumerate(ops) if o.computation_type == SEND_F]
    recv_b_i = [i for i, o in enumerate(ops) if o.computation_type == RECV_B]
    send_b_i = [i for i, o in enumerate(peer_ops) if o.computation_type == SEND_B]
    recv_f_i = [i for i, o in enumerate(peer_ops) if o.computation_type == RECV_F]
    if len(send_f_i):
        assert (
            len(send_f_i) == 1
        ), f"Expect at most one send_f per step, {len(send_f_i)}"

    if len(send_f_i) == 1 and len(recv_b_i):
        # we have a batch candidate
        assert (
            len(recv_b_i) == 1
        ), f"Expect at most one recv_b per step, {len(recv_b_i)}"
        assert len(send_b_i) == 1, "Expected matching send_b for recv_b"
        assert len(recv_f_i) == 1, "Expected matching recv_f for send_f"
        send_f = ops[send_f_i[0]]
        recv_b = ops[recv_b_i[0]]
        for idx in sorted([send_f_i[0], recv_b_i[0]], reverse=True):
            ops.pop(idx)
        new_ops.append(
            _Action(
                send_f.stage_index,
                SEND_F_RECV_B,
                send_f.microbatch_index,
                recv_b.stage_index,
                recv_b.microbatch_index,
            )
        )

        send_b = peer_ops[send_b_i[0]]
        recv_f = peer_ops[recv_f_i[0]]
        assert (
            send_f.stage_index + 1 == recv_f.stage_index
        ), "Expected matching stage indices for send_f and recv_f"
        assert (
            send_f.microbatch_index == recv_f.microbatch_index
        ), "Expected matching microbatch indices for send_f and recv_f"
        assert (
            send_b.stage_index - 1 == recv_b.stage_index
        ), "Expected matching stage indices for send_b and recv_b"
        assert (
            send_b.microbatch_index == recv_b.microbatch_index
        ), "Expected matching microbatch indices for send_b and recv_b"

        for idx in sorted([send_b_i[0], recv_f_i[0]], reverse=True):
            peer_ops.pop(idx)
        new_peer_ops.append(
            _Action(
                send_b.stage_index,
                SEND_B_RECV_F,
                send_b.microbatch_index,
                recv_f.stage_index,
                recv_f.microbatch_index,
            )
        )

    # no more batches. just deal with single ops now
    # TODO - this is bad, i need to match them so they are ordered safely
    while ops:
        new_ops.append(ops.pop())
    while peer_ops:
        new_peer_ops.append(peer_ops.pop())
    return new_ops, new_peer_ops


def _add_send_recv(
    compute_actions: Dict[int, List[_Action]],
    stage_to_rank: Callable[[int], int],
    num_stages: int,
) -> Dict[int, List[_Action]]:
    comm_actions: Dict[int, List[_Action]] = {rank: [] for rank in compute_actions}

    def _has_comms(action: _Action) -> bool:
        if action.computation_type == F:
            return action.stage_index != num_stages - 1
        elif action.computation_type in (B, BW):
            return action.stage_index != 0
        return False

    def _get_comms(action: _Action) -> Tuple[_Action, _Action]:
        assert _has_comms(action), f"{action} is not a valid comm action"
        stage_idx = action.stage_index
        ctype = action.computation_type
        mb_idx = action.microbatch_index
        send = _Action(stage_idx, SEND_F if ctype == F else SEND_B, mb_idx)
        recv_stage_idx = stage_idx + 1 if ctype == F else stage_idx - 1
        recv = _Action(recv_stage_idx, RECV_F if ctype == F else RECV_B, mb_idx)
        return send, recv

    def _peer_rank(action: _Action) -> int:
        # TODO asserts for invalid stage ids (RECV_F for stage 0)
        if action.computation_type == SEND_F:
            return stage_to_rank(action.stage_index + 1)
        elif action.computation_type == SEND_B:
            return stage_to_rank(action.stage_index - 1)
        elif action.computation_type == RECV_F:
            return stage_to_rank(action.stage_index - 1)
        elif action.computation_type == RECV_B:
            return stage_to_rank(action.stage_index + 1)
        else:
            raise ValueError("unsupported action for peer rank")

    def _ready_to_schedule(
        action: Optional[_Action], prev_actions: List[_Action]
    ) -> bool:
        """We don't put our own recv ops in the schedule, we let a sender on another rank put our recv ops in place.
        This helps ensure a sane (non-hanging) ordering of sends and recvs.
        But it also means we might not be able to schedule our next compute action yet.
        """
        if action is None:
            return True
        elif action.computation_type == F and not action.stage_index == 0:
            for p in prev_actions:
                if (
                    p.computation_type == RECV_F
                    and p.stage_index == action.stage_index
                    and p.microbatch_index == action.microbatch_index
                ):
                    return True
                elif (
                    p.computation_type == SEND_B_RECV_F
                    and p.other_stage_index == action.stage_index
                    and p.other_microbatch_index == action.microbatch_index
                ):
                    return True
            return False
        elif (
            action.computation_type in (B, BW)
            and not action.stage_index == num_stages - 1
        ):
            for p in prev_actions:
                if (
                    p.computation_type == RECV_B
                    and p.stage_index == action.stage_index
                    and p.microbatch_index == action.microbatch_index
                ):
                    return True
                elif (
                    p.computation_type == SEND_F_RECV_B
                    and p.other_stage_index == action.stage_index
                    and p.other_microbatch_index == action.microbatch_index
                ):
                    return True
            return False
        else:
            return True

    while compute_actions:
        progress = False
        # go in order of ranks even if dict keys aren't ordered
        new_comms: Dict[int, defaultdict[int, list]] = {
            rank: defaultdict(list) for rank in sorted(compute_actions)
        }
        for rank in sorted(compute_actions):
            # TODO delete me?
            if rank not in compute_actions:
                continue

            assert len(compute_actions[rank]) > 0
            action = compute_actions[rank][0]
            if not _ready_to_schedule(action, comm_actions[rank]):
                continue

            if action is not None:
                comm_actions[rank].append(action)
                if _has_comms(action):
                    send, recv = _get_comms(action)
                    # TODO we can avoid send/recv if the 2 stages are on the same rank.
                    # should we avoid that in the runtime or here?
                    new_comms[rank][_peer_rank(send)].append(send)
                    new_comms[stage_to_rank(recv.stage_index)][rank].append(recv)

            compute_actions[rank].pop(0)
            if len(compute_actions[rank]) == 0:
                del compute_actions[rank]
            progress = True

        if not progress:
            print("WIP comms schedule:\n", _format_pipeline_order(comm_actions))  # type: ignore[arg-type]
            print("remaining compute actions:\n", compute_actions)
        assert progress, "Malformed compute schedule, can't schedule sends/recvs"

        # comm batching needs to be done carefully to avoid reordering comms and causing a hang
        # algorithm:
        # Process sends/recvs in pairs.  Processing means consuming from 'new_comms' and adding the final schedule
        # processing batches is done the same way except 4 ops at a time are consumed and 2 are written
        # rules:
        # 1- if we batch ops for one rank, we also batch matching ops for another rank
        # 2- when we create a batch, we append the batches to both ranks' schedules at the same time
        # 3- we remove individual sends/recvs from 'new_comms' when we consume them in a batch
        # 4- append individual (unbatchable) sends/recvs
        for rank in new_comms:
            for peer in new_comms[rank]:
                if rank == peer:
                    continue
                # we batch and process all the operations between rank and peer.
                # this should symmetrically consume all actions from new_comms[rank][peer] and new_comms[peer][rank]
                ops = new_comms[rank][peer]
                peer_ops = new_comms[peer][rank]
                if len(ops) == 0:
                    assert (
                        len(peer_ops) == 0
                    ), f"ops was empty but peer_ops was not, {peer_ops}"

                # batched_ops lists include both batched ops and unbatchable ops
                batched_ops, batched_peer_ops = _batch_send_recv(ops, peer_ops)

                # now we have consumed ops from this rank and matching ops from peer.
                # peer will be empty and we will not do anything when we iterate to it
                assert (
                    len(ops) == 0 and len(peer_ops) == 0
                ), f"Expected to process all ops, {ops}, {peer_ops}"
                comm_actions[rank].extend(batched_ops)
                comm_actions[peer].extend(batched_peer_ops)

    return comm_actions


def _simulate_comms_compute(
    pipeline_order, stage_to_rank: Callable[[int], int], num_stages: int
):
    pipeline_order = {
        rank: [a for a in pipeline_order[rank] if a is not None]
        for rank in sorted(pipeline_order)
    }
    schedule: Dict[int, List[_Action | None]] = {
        rank: [] for rank in sorted(pipeline_order)
    }

    def _prev_ops(stage_idx):
        rank = stage_to_rank(stage_idx)
        ops = copy.deepcopy(schedule[rank])
        if len(pipeline_order[rank]):
            # batched comm ops may need to be jointly scheduled (e.g. send_f_recv_b depends on and is a dep of send_b_recv_f)
            # assuming we iterate in sorted rank order, peeking at the next unscheduled action for later ranks should unblock us
            ops.append(pipeline_order[rank][0])

        return ops

    def _ready_to_schedule(action: Optional[_Action]) -> bool:
        if action is None:
            return True

        stage_idx = action.stage_index
        if action.computation_type == F:
            if action.stage_index == 0:
                return True
            for p in _prev_ops(stage_idx):
                if p is None:
                    continue
                elif (
                    p.computation_type == RECV_F
                    and p.stage_index == action.stage_index
                    and p.microbatch_index == action.microbatch_index
                ):
                    return True
                elif (
                    p.computation_type == SEND_B_RECV_F
                    and p.other_stage_index == action.stage_index
                    and p.other_microbatch_index == action.microbatch_index
                ):
                    return True
            return False
        elif action.computation_type in (B, BW):
            if action.stage_index == num_stages - 1:
                return True
            for p in _prev_ops(stage_idx):
                if p is None:
                    continue
                elif (
                    p.computation_type == RECV_B
                    and p.stage_index == action.stage_index
                    and p.microbatch_index == action.microbatch_index
                ):
                    return True
                elif (
                    p.computation_type == SEND_F_RECV_B
                    and p.other_stage_index == action.stage_index
                    and p.other_microbatch_index == action.microbatch_index
                ):
                    return True
            return False
        elif action.computation_type == W:
            return True
        elif action.computation_type == SEND_F:
            expected_f = _Action(action.stage_index, F, action.microbatch_index)
            return expected_f in _prev_ops(stage_idx)
        elif action.computation_type == RECV_F:
            peer_stage_idx = stage_idx - 1
            expected_send = _Action(peer_stage_idx, SEND_F, action.microbatch_index)
            return expected_send in _prev_ops(peer_stage_idx)
        elif action.computation_type == SEND_B:
            expected_b = _Action(action.stage_index, B, action.microbatch_index)
            expected_bw = _Action(action.stage_index, BW, action.microbatch_index)
            return expected_b in _prev_ops(stage_idx) or expected_bw in _prev_ops(
                stage_idx
            )
        elif action.computation_type == RECV_B:
            peer_stage_idx = stage_idx + 1
            expected_send = _Action(peer_stage_idx, SEND_B, action.microbatch_index)
            return expected_send in _prev_ops(peer_stage_idx)
        elif action.computation_type == SEND_F_RECV_B:
            # though the stage_index may not be the same between the SEND and the RECV, the rank must be
            peer_stage_idx = stage_idx + 1
            for p in _prev_ops(peer_stage_idx):
                if p is None:
                    continue
                elif (
                    p.computation_type == SEND_B_RECV_F
                    and action.other_stage_index is not None
                    and p.stage_index == action.other_stage_index + 1
                    and p.other_stage_index is not None
                    and p.other_stage_index == action.stage_index + 1
                    and p.microbatch_index == action.other_microbatch_index
                    and p.other_microbatch_index == action.microbatch_index
                ):
                    return True
            return False
        elif action.computation_type == SEND_B_RECV_F:
            # though the stage_index may not be the same between the SEND and the RECV, the rank must be
            peer_stage_idx = action.stage_index - 1
            for p in _prev_ops(peer_stage_idx):
                if p is None:
                    continue
                elif (
                    p.computation_type == SEND_F_RECV_B
                    and p.stage_index + 1 == action.other_stage_index
                    and p.other_stage_index + 1 == action.stage_index
                    and p.microbatch_index == action.other_microbatch_index
                    and p.other_microbatch_index == action.microbatch_index
                ):
                    return True
            return False

        else:
            raise ValueError(f"Unsupported action type {action}")

    while pipeline_order:
        progress = False
        for rank in sorted(pipeline_order):
            if len(pipeline_order[rank]) == 0:
                continue

            action = pipeline_order[rank][0]
            if _ready_to_schedule(action):
                if action is not None:
                    schedule[rank].append(action)
                pipeline_order[rank].pop(0)
                progress = True
            else:
                schedule[rank].append(None)

        for i in sorted(pipeline_order, reverse=True):
            if len(pipeline_order[i]) == 0:
                del pipeline_order[i]

        # hacky, but do a second pass to replace any 'none' at this timestep with a real action, if it got unblocked
        # by one of the later ranks
        for rank in sorted(pipeline_order):
            if len(pipeline_order[rank]) == 0:
                continue

            if schedule[rank][-1] is not None:
                continue

            action = pipeline_order[rank][0]
            if _ready_to_schedule(action):
                if action is not None:
                    schedule[rank][-1] = action
                pipeline_order[rank].pop(0)

        for i in sorted(pipeline_order, reverse=True):
            if len(pipeline_order[i]) == 0:
                del pipeline_order[i]

        if not progress:
            print("WIP comms schedule:\n", _format_pipeline_order(schedule))
            for rank in pipeline_order:
                print(f"{rank=} next action= {pipeline_order[rank][0]}")
            raise ValueError("Schedule is not progressing")

    return schedule


def _dump_chrometrace(schedule, filename):
    events = []
    for rank in sorted(schedule):
        for timestep, action in enumerate(schedule[rank]):
            if action is None:
                continue
            events.append(
                {
                    "name": str(action),
                    "cat": "computation"
                    if action.computation_type in (F, B, W)
                    else "communication",
                    "ph": "X",
                    "pid": rank,
                    "tid": rank,
                    "ts": timestep,
                    "dur": 1,
                }
            )
    import json

    with open(filename, "w") as f:
        json.dump({"traceEvents": events}, f)


class PipelineScheduleMulti(_PipelineSchedule):
    """
    Base class for multi-stage schedules.
    Implements the `step` method.
    """

    def __init__(
        self,
        stages: List[_PipelineStageBase],
        n_microbatches: int,
        loss_fn: Optional[Callable] = None,
        args_chunk_spec: Optional[Tuple[TensorChunkSpec, ...]] = None,
        kwargs_chunk_spec: Optional[Dict[str, TensorChunkSpec]] = None,
        output_merge_spec: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
        stage_index_to_group_rank: Optional[Dict[int, int]] = None,
        use_full_backward: bool = True,
    ):
        if len(stages) <= 1:
            raise ValueError(
                f"Multi-stage schedule expects at least two stages but got {len(stages)}"
            )
        # Init parent
        super().__init__(
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            args_chunk_spec=args_chunk_spec,
            kwargs_chunk_spec=kwargs_chunk_spec,
            output_merge_spec=output_merge_spec,
        )
        # Self attributes
        self._stages = stages
        self._num_stages = stages[0].num_stages
        self.pp_group_size = stages[0].group_size
        self.rank = stages[0].group_rank
        # Set the pipeline stage states
        if stage_index_to_group_rank is not None:
            for stage in self._stages:
                stage.stage_index_to_group_rank = stage_index_to_group_rank
        self.stage_index_to_group_rank = stages[0].stage_index_to_group_rank

        # Set the same has_backward flag for stage object
        for stage in self._stages:
            stage.has_backward = self._has_backward

        self._should_compute_loss = (
            lambda stage: stage.is_last and self._loss_fn is not None
        )

        # This will be set during init of derived schedules
        self.pipeline_order: Dict[int, List[Optional[_Action]]] = {}
        self.use_full_backward = use_full_backward

        # TODO: later replace this with lazy shape inference during forward
        # Prepare forward send/recv infrastructure for stage
        for stage in self._stages:
            stage._prepare_forward_infra(n_microbatches)
            if self._has_backward:
                stage._prepare_backward_infra(n_microbatches)

    def _dump_csv(self, filename):
        """Dump a CSV representation of the schedule into a file with the provided filename."""
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for rank in self.pipeline_order:
                writer.writerow(self.pipeline_order[rank])

    def _validate_schedule(self):
        # TODO(whc) this should be merged with the logic in test_schedule.py#L453-L554
        def _validate_rank_actions(
            actions: Dict[int, List[_Action | None]],
            num_stages: int,
            num_microbatches: int,
        ):
            # We will count all the actions per stage and ensure they happen in a valid order
            # (e.g. F before B before W for a given microbatch)
            stage_actions: Dict[int, Dict[_ComputationType, Set]] = {
                stage_id: {
                    F: set(),
                    B: set(),
                    W: set(),
                }
                for stage_id in range(num_stages)
            }
            for rank in actions:
                for action in actions[rank]:
                    if action is None:
                        continue
                    assert isinstance(
                        action, _Action
                    ), f"Got an invalid action: {action}, expected instance of _Action"
                    s_id = action.stage_index
                    ctype = action.computation_type
                    mb_id = action.microbatch_index
                    if ctype == F:
                        stage_actions[s_id][F].add(mb_id)
                    elif ctype == B:
                        assert (
                            mb_id in stage_actions[s_id][F]
                        ), f"Running Backward for stage {s_id}, microbatch {mb_id} without first running Forward"
                        stage_actions[s_id][B].add(mb_id)
                    elif ctype == W:
                        assert (
                            not self.use_full_backward
                        ), "Schedule contains 'W' actions, but is configured to use full backward"
                        assert (
                            mb_id in stage_actions[s_id][B]
                        ), f"Running Weight for stage {s_id}, microbatch {mb_id} without first running Backward"
                        stage_actions[s_id][W].add(mb_id)

            for s_id in stage_actions:
                for ctype in (F, B, W):
                    stage_mb = len(stage_actions[s_id][ctype])
                    assert (
                        stage_mb == num_microbatches
                    ), f"Got {stage_mb} {ctype} microbatches for stage {s_id}, expected {num_microbatches}"

        assert (
            len(self.pipeline_order) == self.pp_group_size
        ), f"Schedule has incorrect number of ranks - expected {self.pp_group_size}, actual {len(self.pipeline_order)}"
        for rank in range(self.pp_group_size):
            assert (
                rank in self.pipeline_order
            ), f"Schedule is missing actions for rank {rank}"
        _validate_rank_actions(
            self.pipeline_order,
            self._num_stages,
            self._n_microbatches,
        )

    def _load_csv(self, filename, format="compute_only"):
        """Load a CSV representation of the schedule from a file with the provided filename.
        This API will most likely get renamed/refactored so is marked as internal for now.

        format must be "compute_only" for PipelineScheduleMulti
        """
        assert format == "compute_only"
        with open(filename, newline="") as csvfile:
            reader = csv.reader(csvfile)
            for rank, row in enumerate(reader):
                self.pipeline_order[rank] = [_Action.from_str(s) for s in row]
        self._validate_schedule()

    def step(self, *args, target=None, losses: Optional[List] = None, **kwargs):
        """
        Run one iteration of the pipeline schedule with *whole-batch* input.
        Will chunk the input into microbatches automatically, and go through the
        microbatches according to the schedule implementation.

        args: positional arguments to the model (as in non-pipeline case).
        kwargs: keyword arguments to the model (as in non-pipeline case).
        target: target for the loss function.
        losses: a list to store the losses for each microbatch.
        """

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
        self._step_microbatches(args_split, kwargs_split, targets_split, losses)

        # Return merged results per original format
        for stage in self._stages:
            if stage.is_last:
                return self._merge_outputs(stage.output_chunks)
        # Does not contain the last stage
        return None

    def _step_microbatches(
        self,
        arg_mbs: Optional[List] = None,
        kwarg_mbs: Optional[List] = None,
        target_mbs: Optional[List] = None,
        losses: Optional[List] = None,
    ):
        """
        Operate on the microbatches for looped schedules (multiple stages on each rank).

        TODO: Does not use sorted_batch_isend_irecv(). As a result, this schedule does
        not support models with skip connections.
        """
        arg_mbs, kwarg_mbs = self._check_inputs(arg_mbs, kwarg_mbs, target_mbs, losses)

        # Based on the plan in Step 1 created in __init__:
        # 2. Perform communication based on the pipeline_order
        stage_index_to_stage: Dict[int, _PipelineStageBase] = {
            stage.stage_index: stage for stage in self._stages
        }

        # determine prev_rank and next_rank based on which ranks are next to
        # the stages in the pipeline_order
        all_prev_ranks: Set[int] = set()
        all_next_ranks: Set[int] = set()
        for stage_index in stage_index_to_stage.keys():
            # TODO: assumption that stages only communicate from distances of +1/-1 (no skip connections)
            if stage_index > 0:
                all_prev_ranks.add(self.stage_index_to_group_rank[stage_index - 1])
            if stage_index < self._num_stages - 1:
                all_next_ranks.add(self.stage_index_to_group_rank[stage_index + 1])

        for time_step, action in enumerate(self.pipeline_order[self.rank]):
            try:
                ops: List[dist.P2POp] = []
                if action is not None:
                    computation_type = action.computation_type
                    mb_index = action.microbatch_index
                    stage_index = action.stage_index
                    assert (
                        mb_index is not None
                    ), "All currently supported action types require valid microbatch_index"
                    if computation_type == _ComputationType.FORWARD:
                        # perform forward computation
                        stage = stage_index_to_stage[stage_index]
                        output = stage.forward_one_chunk(
                            mb_index, arg_mbs[mb_index], kwarg_mbs[mb_index]
                        )
                        self._maybe_compute_loss(stage, output, target_mbs, mb_index)
                        ops.extend(stage.get_fwd_send_ops(mb_index))
                    elif computation_type == _ComputationType.BACKWARD:
                        # perform backward computation
                        stage = stage_index_to_stage[stage_index]
                        loss = self._maybe_get_loss(stage, mb_index)
                        stage.backward_one_chunk(
                            mb_index, loss=loss, full_backward=self.use_full_backward
                        )
                        ops.extend(stage.get_bwd_send_ops(mb_index))
                    elif computation_type == _ComputationType.WEIGHT:
                        # perform weight update
                        if self.use_full_backward:
                            raise ValueError(
                                f"We detected a weight update in the pipeline schedule, but \
                                {self.use_full_backward=}"
                            )
                        stage = stage_index_to_stage[stage_index]
                        stage.backward_weight_one_chunk(mb_index)
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
                        assert (
                            mb_index is not None
                        ), "All currently supported action types require valid microbatch_index"
                        # Only handle sends for the forward from a previous rank
                        if computation_type == _ComputationType.FORWARD:
                            # If not the last stage, then receive fwd activations
                            if stage_index + 1 in stage_index_to_stage:
                                # TODO: We are assuming that stage will always receive from stage-1
                                # however that is not necessarily true of get_fwd_recv_ops
                                stage = stage_index_to_stage[stage_index + 1]
                                ops.extend(stage.get_fwd_recv_ops(mb_index))
                        elif (
                            computation_type == _ComputationType.BACKWARD
                            or computation_type == _ComputationType.WEIGHT
                        ):
                            # Previous rank doing backward or weight update has no influence for the current rank forward recv
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
                        assert (
                            mb_index is not None
                        ), "All currently supported action types require valid microbatch_index"
                        # Only handle receives for the backwards from a next rank
                        if (
                            computation_type == _ComputationType.FORWARD
                            or computation_type == _ComputationType.WEIGHT
                        ):
                            # Next rank doing forward or weight update has no influence for the current rank backward recv
                            pass
                        elif computation_type == _ComputationType.BACKWARD:
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
                if ops:
                    _batch_p2p(ops).wait()
            except Exception as e:
                logger.error(
                    "[Rank %s] pipeline schedule %s caught the following exception \
                     at time_step %s when running action %s",
                    self.rank,
                    self.__class__.__name__,
                    time_step,
                    action,
                )
                logger.error("%s", _format_pipeline_order(self.pipeline_order))
                raise e
        # Return losses if there is a container passed in
        self._update_losses(self._stages, losses)


class _PipelineScheduleRuntime(PipelineScheduleMulti):
    """
    Provides a simple runtime that requires a 'schedule IR' including specified communication operations.

    Can be instantiated directly by creating _PipelineScheduleRuntime and calling load_csv, or can be
    subclassed and the subclass can be responsible for creating a schedule IR.
    """

    def _load_actions(
        self,
        actions: Dict[int, List[Optional[_Action]]],
        format: str = "compute_only",
    ):
        """
        Given an in-memory representation for a simple compute-only schedule, lower it to a complex schedule including
        communication actions.  Stores the schedule in self, and must be called before running step_mo()
        """
        assert (
            self.stage_index_to_group_rank is not None
        ), "stage_index_to_group_rank is required for PipelineScheduleRuntime"
        self.pipeline_order_with_comms: Dict[int, List[_Action]] = {}
        if format == "compute_comms":
            for rank in actions:
                self.pipeline_order_with_comms[rank] = []
                for action in actions[rank]:
                    assert action is not None
                    self.pipeline_order_with_comms[rank].append(action)
            # TODO what level of validation should we offer for compute+comms schedule?
        elif format == "compute_only":
            # Perform schedule lowering
            for rank in actions:
                self.pipeline_order_with_comms[rank] = _add_unshard_reshard(
                    actions[rank]
                )

            self.pipeline_order_with_comms = _add_send_recv(
                self.pipeline_order_with_comms,
                stage_to_rank=lambda s: self.stage_index_to_group_rank[s],
                num_stages=self._num_stages,
            )
        else:
            raise NotImplementedError(f"{format=} is not implemented")

    def _load_csv(self, filename: str, format: str = "compute_only"):
        """Loads a csv in simple format and then lowers it to include comunication actions

        format must be either "compute_only" or "compute_comms".  If compute_only, the lowering passes
        will automatically be run to generate a compute_comms schedule.
        """
        if format == "compute_only":
            # this will populate self.pipeline_order
            super()._load_csv(filename)
            # this will populate self.pipeline_order_with_comms
            self._load_actions(self.pipeline_order)
        elif format == "compute_comms":
            actions = {}
            with open(filename, newline="") as csvfile:
                reader = csv.reader(csvfile)
                for rank, row in enumerate(reader):
                    actions[rank] = [_Action.from_str(s) for s in row]
                self._load_actions(actions, format=format)
        else:
            raise NotImplementedError(f"{format=} is not implemented")

    def _dump_csv(self, filename: str):
        """Dump a CSV representation of the compute + comms schedule into a file with the provided filename."""
        # TODO should there be an option to dump the compute_only schedule from PipelineScheduleRuntime? It's possible
        # that it does not exist if it was created from a compute_comms schedule.
        assert (
            self.pipeline_order_with_comms is not None
        ), "Must initialize compute_comms schedule before dump_csv"
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for rank in self.pipeline_order_with_comms:
                writer.writerow(self.pipeline_order_with_comms[rank])

    def _simulate(self):
        return _simulate_comms_compute(
            self.pipeline_order_with_comms,
            lambda s: self.stage_index_to_group_rank[s],
            self._num_stages,
        )

    def _step_microbatches(
        self,
        arg_mbs: Optional[List] = None,
        kwarg_mbs: Optional[List] = None,
        target_mbs: Optional[List] = None,
        losses: Optional[List] = None,
    ):
        """
        Operate on the microbatches for looped schedules (multiple stages on each rank).

        TODO: Does not use sorted_batch_isend_irecv(). As a result, this schedule does
        not support models with skip connections.
        """
        arg_mbs, kwarg_mbs = self._check_inputs(arg_mbs, kwarg_mbs, target_mbs, losses)

        # Based on the plan in Step 1 created in __init__:
        # 2. Perform communication based on the pipeline_order
        stage_index_to_stage: Dict[int, _PipelineStageBase] = {
            stage.stage_index: stage for stage in self._stages
        }

        assert (
            self.pipeline_order_with_comms is not None
        ), "Must call _load_actions() before calling _step_microbatches()"

        # recv ops indexed by (stage_idx, mb_idx) need to be waited on before use
        bwd_recv_ops: Dict[Tuple[int, int], Work] = {}
        fwd_recv_ops: Dict[Tuple[int, int], Work] = {}

        # send ops should be waited on before step() exists, mainly for hygeine
        send_ops: List[Work] = []

        # we track which stages are 'active' when used with FSDP, and wait on unshard ops before computing on stages
        unshard_ops: Dict[int, UnshardHandle] = {}
        unsharded_stages = set()

        def _assert_unsharded(stage_idx: int):
            """If an unshard is active for `stage_idx`, wait() it and mark `stage_idx` unshared."""
            if stage_idx in unshard_ops:
                unshard_ops[stage_idx].wait()
                del unshard_ops[stage_idx]
                unsharded_stages.add(stage_idx)
            assert (
                stage_idx in unsharded_stages
            ), f"Attempted to compute on sharded {stage_idx=}"

        for time_step, action in enumerate(self.pipeline_order_with_comms[self.rank]):
            try:
                comp_type = action.computation_type
                mb_index: int = (
                    action.microbatch_index
                    if action.microbatch_index is not None
                    else -1
                )
                assert mb_index >= 0 or comp_type in (
                    UNSHARD,
                    RESHARD,
                ), f"{action=} missing mb_index"
                stage_idx = action.stage_index
                stage = stage_index_to_stage[stage_idx]
                stage_uses_fsdp = isinstance(stage.submod, FSDPModule)

                logger.debug(
                    "_PipelineScheduleRuntime running time_step %d, action %s",
                    time_step,
                    action,
                )

                # TODO(whc) it's not actually safe to use _batch_p2p here in the uncommon case the model has skip-connections,
                # since we do not want to batch up ops between more than a pair of ranks.  _sorted_batch_p2p would be
                # safe to use instead.
                # However, I was wondering if I should avoid calling batched operators at all in the case that there is
                # only one operator per batch.  I could iterate through the 'fwd_send_ops' one by one and run them.
                if comp_type == SEND_F:
                    send_ops.append(_batch_p2p(stage.get_fwd_send_ops(mb_index)))
                elif comp_type == SEND_B:
                    send_ops.append(_batch_p2p(stage.get_bwd_send_ops(mb_index)))
                elif comp_type == RECV_F:
                    assert (
                        stage_idx,
                        mb_index,
                    ) not in fwd_recv_ops, "Recv twice for {stage_idx=} {mb_index=} without executing forward"
                    fwd_recv_ops[(stage_idx, mb_index)] = _batch_p2p(
                        stage.get_fwd_recv_ops(mb_index)
                    )
                elif comp_type == RECV_B:
                    assert (
                        stage_idx,
                        mb_index,
                    ) not in bwd_recv_ops, "Recv twice for {stage_idx=} {mb_index=} without executing backward"
                    bwd_recv_ops[(stage_idx, mb_index)] = _batch_p2p(
                        stage.get_bwd_recv_ops(mb_index)
                    )
                elif comp_type == UNSHARD:
                    if stage_uses_fsdp:
                        assert (
                            stage_idx not in unsharded_stages
                            and stage_idx not in unshard_ops
                        ), f"Unsharding the same {stage_idx=} twice"
                        unshard_ops[stage_idx] = stage.submod.unshard(async_op=True)
                elif comp_type == RESHARD:
                    if stage_uses_fsdp:
                        assert (
                            stage_idx in unsharded_stages
                        ), f"Resharding {stage_idx=} without unsharding"
                        assert (
                            stage_idx not in unshard_ops
                        ), f"Resharding {stage_idx=} before finishing unshard"
                        stage.submod.reshard()
                elif comp_type == FORWARD:
                    if stage_uses_fsdp:
                        _assert_unsharded(stage_idx)

                    if not stage.is_first:
                        assert (
                            stage_idx,
                            mb_index,
                        ) in fwd_recv_ops, f"Computing {action=} before receiving input"
                        fwd_recv_ops.pop((stage_idx, mb_index)).wait()
                    output = stage.forward_one_chunk(
                        mb_index, arg_mbs[mb_index], kwarg_mbs[mb_index]
                    )
                    self._maybe_compute_loss(stage, output, target_mbs, mb_index)
                elif comp_type == BACKWARD:
                    if stage_uses_fsdp:
                        _assert_unsharded(stage_idx)

                    if not stage.is_last:
                        assert (
                            stage_idx,
                            mb_index,
                        ) in bwd_recv_ops, (
                            f"Attempted to run compute {action=} before receiving input"
                        )
                        bwd_recv_ops.pop((stage_idx, mb_index)).wait()
                    loss = self._maybe_get_loss(stage, mb_index)
                    stage.backward_one_chunk(
                        mb_index, loss=loss, full_backward=self.use_full_backward
                    )
                elif comp_type == WEIGHT:
                    if stage_uses_fsdp:
                        _assert_unsharded(stage_idx)

                    if self.use_full_backward:
                        raise ValueError(
                            f"We detected a weight update in the pipeline schedule, but \
                            {self.use_full_backward=}"
                        )
                    stage.backward_weight_one_chunk(mb_index)
                else:
                    raise ValueError(f"{action=} is unknown or unsupported")
            except Exception as e:
                logger.error(
                    "_PipelineScheduleRuntime caught exception at step %s when running action %s.  Full Schedule:",
                    time_step,
                    action,
                )
                # TODO(whc) what is the best practice for printing a multiline log?
                # logger will split it into multiple log lines, but this makes it hard to read (too wide)
                print(_format_pipeline_order(self.pipeline_order_with_comms))  # type: ignore[arg-type]
                raise e

        # Mostly these operations should have finished long ago, but there isn't an obvious time when to wait for them
        while len(send_ops):
            send_ops.pop().wait()

        assert len(unshard_ops) == 0, "Unused unshard operations"

        # Return losses if there is a container passed in
        self._update_losses(self._stages, losses)


class ScheduleLoopedBFS(PipelineScheduleMulti):
    """
    Breadth-First Pipeline Parallelism.
    See https://arxiv.org/abs/2211.05953 for details.
    Simliar to Interleaved 1F1B, Looped BFS supports multiple stages per rank.
    What is different is that when microbatches are ready for multiple local
    stages, Loops BFS will prioritizes the earlier stage, running all available
    microbatches at once.
    """

    def __init__(
        self,
        stages: List[_PipelineStageBase],
        n_microbatches: int,
        loss_fn: Optional[Callable] = None,
        output_merge_spec: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
    ):
        super().__init__(
            stages=stages,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            output_merge_spec=output_merge_spec,
        )

        # 1. Create the pipeline_order (all ranks do this calculation)
        # This will be used to keep track of the current state of the entire pipeline
        # pipeline_order[rank] = [Action(computation_type, microbatch_index, stage_index), ...]
        self.pipeline_order: Dict[int, List[Optional[_Action]]] = {}
        # ========================================================================
        for rank in range(self.pp_group_size):
            rank_ops = self._calculate_single_rank_operations(rank)
            self.pipeline_order[rank] = rank_ops

    def _calculate_single_rank_operations(self, rank):
        n_local_stages = len(self._stages)
        stage_indices = range(
            rank, self.pp_group_size * n_local_stages, self.pp_group_size
        )

        # Store the list of operations used for that rank
        rank_ops: List[Optional[_Action]] = []
        # Pre-padding, rank starts with no-ops based on the warmup.
        for _ in range(rank):
            rank_ops.append(None)

        for stage_index in stage_indices:
            for mb_index in range(self._n_microbatches):
                rank_ops.append(
                    _Action(stage_index, _ComputationType.FORWARD, mb_index)
                )

        # wait for the first backward to trickle up
        # which is 2 for every hop away
        post_warmup_ops = 2 * (self.pp_group_size - 1 - rank)
        rank_ops.extend([None] * post_warmup_ops)

        for stage_index in reversed(stage_indices):
            for mb_index in reversed(range(self._n_microbatches)):
                rank_ops.append(
                    _Action(stage_index, _ComputationType.BACKWARD, mb_index)
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
):
    # All stages start with handling microbatch 0
    fwd_stage_mb_index: Dict[int, int] = defaultdict(int)
    bwd_stage_mb_index: Dict[int, int] = defaultdict(int)
    # Store the list of operations used for that rank
    rank_ops: List[Optional[_Action]] = []
    # Pre-padding, rank starts with no-ops based on the warmup.
    for _ in range(rank):
        rank_ops.append(None)
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

    total_ops = warmup_ops + fwd_bwd_ops + cooldown_ops

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
                _Action(bwd_stage_index, _ComputationType.BACKWARD, bwd_mb_index)
            )
        # Cooldown phase
        else:
            # During cooldown phase, we need steps to align with 1f1b happening in other ranks
            # TODO: we don't need to always append, after all 1f1b are finished we can stop appending None
            rank_ops.append(None)
            bwd_stage_index = backward_stage_index(op)
            bwd_stage_mb_index[bwd_stage_index] = (
                bwd_mb_index := bwd_stage_mb_index[bwd_stage_index]
            ) + 1
            rank_ops.append(
                _Action(bwd_stage_index, _ComputationType.BACKWARD, bwd_mb_index)
            )
    return rank_ops


class ScheduleInterleaved1F1B(PipelineScheduleMulti):
    """
    The Interleaved 1F1B schedule.
    See https://arxiv.org/pdf/2104.04473 for details.
    Will perform one forward and one backward on the microbatches in steady
    state and supports multiple stages per rank. When microbatches are ready for
    multiple local stages, Interleaved 1F1B prioritizes the earlier microbatch
    (also called "depth first").
    """

    def __init__(
        self,
        stages: List[_PipelineStageBase],
        n_microbatches: int,
        loss_fn: Optional[Callable] = None,
        args_chunk_spec: Optional[Tuple[TensorChunkSpec, ...]] = None,
        kwargs_chunk_spec: Optional[Dict[str, TensorChunkSpec]] = None,
        output_merge_spec: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
    ):
        self.pp_group_size = stages[0].group_size
        # TODO: is this limitation a must?
        if n_microbatches % self.pp_group_size != 0:
            raise ValueError(
                f"Interleaved 1F1B schedule requires the number of microbatches ({n_microbatches}) \
                to be a multiple of the number of pipeline ranks ({self.pp_group_size})."
            )

        super().__init__(
            stages=stages,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            args_chunk_spec=args_chunk_spec,
            kwargs_chunk_spec=kwargs_chunk_spec,
            output_merge_spec=output_merge_spec,
        )

        self.n_local_stages = len(stages)
        self.rank = stages[0].group_rank
        self.group = stages[0].group

        # 1. Create the pipeline_order (all ranks do this calculation)
        # This will be used to keep track of the current state of the entire pipeline
        # pipeline_order[rank] = [Action(computation_type, microbatch_index, stage_index), ...]
        self.pipeline_order: Dict[int, List[Optional[_Action]]] = {}

        for rank in range(self.pp_group_size):
            rank_ops = self._calculate_single_rank_operations(rank)
            self.pipeline_order[rank] = rank_ops

    def _calculate_single_rank_operations(self, rank) -> List[Optional[_Action]]:
        def get_rank_warmup_ops(rank):
            # Warms up operations for last stage
            warmups_ops_last_stage = (self.n_local_stages - 1) * self.pp_group_size
            # Increment warmup operations by 2 for each hop away from the last stage
            warmup_ops = warmups_ops_last_stage + 2 * ((self.pp_group_size - 1) - rank)
            # We cannot have more warmup operations than there are number of microbatches, so cap it there
            return min(warmup_ops, self._n_microbatches * self.n_local_stages)

        warmup_ops = get_rank_warmup_ops(rank)
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
            local_index = (step // self.pp_group_size) % self.n_local_stages
            return (local_index * self.pp_group_size) + rank

        def backward_stage_index(step):
            local_index = (
                self.n_local_stages
                - 1
                - ((step - warmup_ops) // self.pp_group_size) % self.n_local_stages
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


class ScheduleFlexibleInterleaved1F1B(PipelineScheduleMulti):
    """
    The Flexible Interleaved 1F1B schedule.

    This schedule is mostly similar to the interleaved 1F1B schedule.
    It differs by being relaxing the requirement of num_microbatch % pp_size == 0.
    Using the flex_pp schedule, we will have num_rounds = max(1, n_microbatches // pp_group_size) and
    it works as long as n_microbatches % num_rounds is 0. As a few examples, support

    1. pp_group_size = 4, n_microbatches = 10. We will have num_rounds = 2 and n_microbatches % 2 is 0.
    2. pp_group_size = 4, n_microbatches = 3. We will have num_rounds = 1 and n_microbatches % 1 is 0.

    When zero_bubble_algorithm is passed in, we will use the corresponding schedule in
    https://openreview.net/pdf?id=tuzTN0eIO5
    """

    def __init__(
        self,
        stages: List[_PipelineStageBase],
        n_microbatches: int,
        loss_fn: Optional[Callable] = None,
        args_chunk_spec: Optional[Tuple[TensorChunkSpec, ...]] = None,
        kwargs_chunk_spec: Optional[Dict[str, TensorChunkSpec]] = None,
        output_merge_spec: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
        zero_bubble_algorithm: Optional[ZeroBubbleAlgorithm] = None,
    ):
        self.pp_group_size = stages[0].group_size
        super().__init__(
            stages=stages,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            args_chunk_spec=args_chunk_spec,
            kwargs_chunk_spec=kwargs_chunk_spec,
            output_merge_spec=output_merge_spec,
            use_full_backward=not zero_bubble_algorithm,
        )
        self.n_local_stages = len(stages)
        self.rank = stages[0].group_rank
        self.number_of_rounds = max(1, n_microbatches // self.pp_group_size)
        self.microbatches_per_round = n_microbatches // self.number_of_rounds
        self.zero_bubble_algorithm = zero_bubble_algorithm
        if self.zero_bubble_algorithm is ZeroBubbleAlgorithm.ZBV:
            raise ValueError("ZBV is not yet supported")

        if n_microbatches % self.number_of_rounds != 0:
            raise ValueError(
                "Flexible Interleaved 1F1B requires the number of microbatches to be a "
                f"multiple of the number of rounds ({self.number_of_rounds}), "
                f"but got {n_microbatches}."
            )
        # 1. Create the pipeline_order (all ranks do this calculation)
        # This will be used to keep track of the current state of the entire pipeline
        # pipeline_order[rank] = [Action(computation_type, microbatch_index, stage_index), ...]
        self.pipeline_order: Dict[int, List[Optional[_Action]]] = {}
        for rank in range(self.pp_group_size):
            rank_ops = self._calculate_single_rank_operations(rank)
            self.pipeline_order[rank] = rank_ops

        # This function add bubbles to the generated schedule based on dependencies of actions
        # Note that the ZB1P schedule will not require bubbles to be manually added and it is
        # only useful when n_microbatches <= microbatches_per_round
        self.pipeline_order = self._add_bubbles_to_actions(
            self.n_local_stages * self.pp_group_size,
        )

    def _calculate_single_rank_operations(self, rank) -> List[Optional[_Action]]:
        def get_rank_warmup_ops(rank):
            # Warms up operations for last stage
            warmups_ops_last_stage = (
                self.n_local_stages - 1
            ) * self.microbatches_per_round
            # Increment warmup operations by 2 for each hop away from the last stage
            multiply_factor = 1
            if self.zero_bubble_algorithm is ZeroBubbleAlgorithm.ZB2P:
                multiply_factor = 2
            warmup_ops = warmups_ops_last_stage + multiply_factor * (
                (self.pp_group_size - 1) - rank
            )

            # We cannot have more warmup operations than there are number of microbatches, so cap it there
            return min(warmup_ops, self._n_microbatches * self.n_local_stages)

        warmup_ops = get_rank_warmup_ops(rank)
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

        if self.zero_bubble_algorithm:
            num_1f1b_microbatches = rank
            if self.zero_bubble_algorithm is ZeroBubbleAlgorithm.ZB2P:
                num_1f1b_microbatches = 2 * rank

            return self._get_1f1b_rank_ops_zero_bubble(
                self.n_local_stages,
                self.pp_group_size,
                warmup_ops,
                fwd_bwd_ops,
                cooldown_ops,
                rank,
                forward_stage_index,
                backward_stage_index,
                num_1f1b_microbatches,
                zero_bubble_algorithm=self.zero_bubble_algorithm,
                forward_local_stage_one_index=self.pp_group_size + rank,
            )

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

    def _add_bubbles_to_actions(self, num_stages_global):
        actions = self.pipeline_order
        if not self.zero_bubble_algorithm:
            return actions

        def need_bubble(stage, op, microbatch, num_stages_global, seen_ops):
            if op == _ComputationType.FORWARD:
                if stage != 0 and (stage - 1, op, microbatch) not in seen_ops:
                    return True
            elif op == _ComputationType.BACKWARD:
                if stage == num_stages_global - 1:
                    return (stage, _ComputationType.FORWARD, microbatch) not in seen_ops
                return (stage + 1, op, microbatch) not in seen_ops
            return False

        seen_ops: Set[Tuple[int, _ComputationType, int]] = set()
        result: Dict[int, List[Optional[_Action]]] = {}
        next_pointer: Dict[int, int] = {}
        bubbles_added: Dict[int, int] = {}
        total_bubbles_added = 0

        for rank in range(self.pp_group_size):
            result[rank] = []
            next_pointer[rank] = 0
            bubbles_added[rank] = 0

        while True:
            should_stop = True

            temp_seen_ops: Set[Tuple[int, _ComputationType, int]] = set()

            for rank in range(self.pp_group_size):
                timestamp = next_pointer[rank]
                if timestamp >= len(actions[rank]):
                    continue

                should_stop = False

                if actions[rank][timestamp] is not None:
                    temp_action = actions[rank][timestamp]
                    assert temp_action is not None
                    stage_index = temp_action.stage_index
                    op = temp_action.computation_type
                    microbatch = temp_action.microbatch_index
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
                f"Non zero bubbles added: {total_bubbles_added=} {bubbles_added=}"  # noqa: G004
            )
        return result

    def _get_1f1b_rank_ops_zero_bubble(
        self,
        n_local_stages,
        pp_group_size,
        warmup_ops,
        fwd_bwd_ops,
        cooldown_ops,
        rank,
        forward_stage_index,
        backward_stage_index,
        num_1f1b_microbatches,
        zero_bubble_algorithm,
        forward_local_stage_one_index,
    ):
        # All stages start with handling microbatch 0
        fwd_stage_mb_index: Dict[int, int] = defaultdict(int)
        bwd_stage_mb_index: Dict[int, int] = defaultdict(int)
        weight_stage_mb_index: Dict[int, int] = defaultdict(int)

        # Store the list of operations used for that rank
        rank_ops: List[Optional[_Action]] = []
        # Pre-padding, rank starts with no-ops based on the warmup.
        for _ in range(rank):
            rank_ops.append(None)
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

        if zero_bubble_algorithm is ZeroBubbleAlgorithm.ZB1P:
            post_warmup_ops = pp_group_size - rank - 1
        elif zero_bubble_algorithm is ZeroBubbleAlgorithm.ZB2P:
            post_warmup_ops = 0

        total_ops = warmup_ops + fwd_bwd_ops + cooldown_ops

        prefill_steps_1b1w = 0
        if zero_bubble_algorithm is ZeroBubbleAlgorithm.ZB2P:
            prefill_steps_1b1w = max(0, math.ceil((pp_group_size - 4) / 2) - rank)

        backward_op_ids = []
        weight_op_count = 0
        forward_op_id = 0
        backward_op_id = warmup_ops
        has_backfilled = False

        for op in range(total_ops - prefill_steps_1b1w):
            # Warmup phase
            if op < warmup_ops:
                fwd_stage_index = forward_stage_index(forward_op_id)
                # This will assign the current microbatch index and update it as well
                fwd_stage_mb_index[fwd_stage_index] = (
                    mb_index := fwd_stage_mb_index[fwd_stage_index]
                ) + 1
                rank_ops.append(
                    _Action(fwd_stage_index, _ComputationType.FORWARD, mb_index)
                )
                if forward_op_id == warmup_ops - 1:
                    # This is the last step in the warmup phase, so we need to wait for the backward to trickle back up
                    rank_ops.extend([None] * post_warmup_ops)
                forward_op_id += 1

            # 1F1B Phase (forward and backward)
            elif warmup_ops <= op < warmup_ops + fwd_bwd_ops:
                fwd_stage_index = forward_stage_index(forward_op_id)
                if (
                    fwd_stage_index == forward_local_stage_one_index
                    and not has_backfilled
                ):
                    has_backfilled = True
                    for _ in range(prefill_steps_1b1w):
                        bwd_stage_index = backward_stage_index(backward_op_id)
                        bwd_stage_mb_index[bwd_stage_index] = (
                            bwd_mb_index := bwd_stage_mb_index[bwd_stage_index]
                        ) + 1
                        rank_ops.append(
                            _Action(
                                bwd_stage_index, _ComputationType.BACKWARD, bwd_mb_index
                            )
                        )
                        backward_op_ids.append(backward_op_id)
                        backward_op_id += 1
                        weight_stage_index = backward_stage_index(
                            backward_op_ids[weight_op_count]
                        )
                        weight_stage_mb_index[weight_stage_index] = (
                            weight_mb_index := weight_stage_mb_index[weight_stage_index]
                        ) + 1
                        rank_ops.append(
                            _Action(
                                weight_stage_index,
                                _ComputationType.WEIGHT,
                                weight_mb_index,
                            )
                        )
                        weight_op_count += 1

                fwd_stage_mb_index[fwd_stage_index] = (
                    fwd_mb_index := fwd_stage_mb_index[fwd_stage_index]
                ) + 1
                rank_ops.append(
                    _Action(fwd_stage_index, _ComputationType.FORWARD, fwd_mb_index)
                )
                bwd_stage_index = backward_stage_index(backward_op_id)
                bwd_stage_mb_index[bwd_stage_index] = (
                    bwd_mb_index := bwd_stage_mb_index[bwd_stage_index]
                ) + 1
                rank_ops.append(
                    _Action(bwd_stage_index, _ComputationType.BACKWARD, bwd_mb_index)
                )
                backward_op_ids.append(backward_op_id)
                forward_op_id += 1
                backward_op_id += 1

                if op - warmup_ops >= num_1f1b_microbatches:
                    weight_stage_index = backward_stage_index(
                        backward_op_ids[weight_op_count]
                    )
                    weight_stage_mb_index[weight_stage_index] = (
                        weight_mb_index := weight_stage_mb_index[weight_stage_index]
                    ) + 1
                    rank_ops.append(
                        _Action(
                            weight_stage_index, _ComputationType.WEIGHT, weight_mb_index
                        )
                    )
                    weight_op_count += 1
            # Cooldown phase
            else:
                bwd_stage_index = backward_stage_index(backward_op_id)
                bwd_stage_mb_index[bwd_stage_index] = (
                    bwd_mb_index := bwd_stage_mb_index[bwd_stage_index]
                ) + 1
                rank_ops.append(
                    _Action(bwd_stage_index, _ComputationType.BACKWARD, bwd_mb_index)
                )
                backward_op_ids.append(backward_op_id)
                backward_op_id += 1

                if zero_bubble_algorithm and op - warmup_ops >= num_1f1b_microbatches:
                    weight_stage_index = backward_stage_index(
                        backward_op_ids[weight_op_count]
                    )
                    weight_stage_mb_index[weight_stage_index] = (
                        weight_mb_index := weight_stage_mb_index[weight_stage_index]
                    ) + 1
                    rank_ops.append(
                        _Action(
                            weight_stage_index, _ComputationType.WEIGHT, weight_mb_index
                        )
                    )
                    weight_op_count += 1

        while weight_op_count < len(backward_op_ids):
            weight_stage_index = backward_stage_index(backward_op_ids[weight_op_count])
            weight_stage_mb_index[weight_stage_index] = (
                weight_mb_index := weight_stage_mb_index[weight_stage_index]
            ) + 1
            rank_ops.append(
                _Action(weight_stage_index, _ComputationType.WEIGHT, weight_mb_index)
            )
            weight_op_count += 1

        return rank_ops
