# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch.profiler import record_function

from .microbatch import merge_chunks, split_args_kwargs_into_chunks, TensorChunkSpec
from .stage import _PipelineStageBase


__all__ = [
    "PipelineScheduleSingle",
    "PipelineScheduleMulti",
    "Schedule1F1B",
    "ScheduleGPipe",
    "ScheduleInterleaved1F1B",
    "ScheduleLoopedBFS",
]

logger = logging.getLogger(__name__)


class _ComputationType(Enum):
    FORWARD = 1
    BACKWARD = 2

    def __str__(self):
        if self == _ComputationType.FORWARD:
            return "F"
        else:
            return "B"


class _Action(NamedTuple):
    computation_type: _ComputationType
    microbatch_index: int
    stage_index: int

    def __repr__(self):
        return f"{self.computation_type}{self.microbatch_index}_s{self.stage_index}"


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
        # Set the same has_backward flag for stage object
        for stage in self._stages:
            stage.has_backward = self._has_backward

        self._should_compute_loss = (
            lambda stage: stage.is_last and self._loss_fn is not None
        )

        # This will be set during init of derived schedules
        self.pipeline_order: Dict[int, List[Optional[_Action]]] = {}

        # TODO: later replace this with lazy shape inference during forward
        # Prepare forward send/recv infrastructure for stage
        for stage in self._stages:
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
        prev_rank: int = (self.rank - 1) % self.pp_group_size
        next_rank: int = (self.rank + 1) % self.pp_group_size

        for time_step, action in enumerate(self.pipeline_order[self.rank]):
            prev_rank_ops = self.pipeline_order[prev_rank]
            next_rank_ops = self.pipeline_order[next_rank]
            ops: List[dist.P2POp] = []
            if action is not None:
                computation_type, mb_index, stage_index = action
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
                    stage.backward_one_chunk(mb_index, loss=loss)
                    ops.extend(stage.get_bwd_send_ops(mb_index))
                else:
                    raise ValueError(f"Unknown computation type {computation_type}")

            # Look at the neighboring ranks for this current timestep and determine whether
            # this current rank needs to do any recv communication
            prev_rank_action = None
            if time_step < len(prev_rank_ops):
                prev_rank_action = prev_rank_ops[time_step]
            if prev_rank_action is not None:
                computation_type, mb_index, stage_index = prev_rank_action
                # Only handle sends for the forward from a previous rank
                if computation_type == _ComputationType.FORWARD:
                    # If not the last stage, then receive fwd activations
                    if stage_index != self._num_stages - 1:
                        # TODO: We are assuming that stage will always receive from stage-1
                        # however that is not necessarily true of get_fwd_recv_ops
                        stage = stage_index_to_stage[stage_index + 1]
                        ops.extend(stage.get_fwd_recv_ops(mb_index))
                elif computation_type == _ComputationType.BACKWARD:
                    # Previous rank doing backward has no influence for the current rank forward recv
                    pass
                else:
                    raise ValueError(f"Unknown computation type {computation_type}")

            next_rank_action = None
            if time_step < len(next_rank_ops):
                next_rank_action = next_rank_ops[time_step]
            if next_rank_action is not None:
                computation_type, mb_index, stage_index = next_rank_action
                # Only handle receives for the backwards from a next rank
                if computation_type == _ComputationType.FORWARD:
                    # Next rank doing forward has no influence for the current rank backward recv
                    pass
                elif computation_type == _ComputationType.BACKWARD:
                    # If not the first stage, then receive bwd gradients
                    if stage_index != 0:
                        # TODO: We are assuming that stage will always receive from stage+1
                        # however that is not necessarily true of get_bwd_recv_ops
                        stage = stage_index_to_stage[stage_index - 1]
                        ops.extend(stage.get_bwd_recv_ops(mb_index))
                else:
                    raise ValueError(f"Unknown computation type {computation_type}")

            # do the communication
            if ops:
                _batch_p2p(ops).wait()
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
                    _Action(_ComputationType.FORWARD, mb_index, stage_index)
                )

        # wait for the first backward to trickle up
        # which is 2 for every hop away
        post_warmup_ops = 2 * (self.pp_group_size - 1 - rank)
        rank_ops.extend([None] * post_warmup_ops)

        for stage_index in reversed(stage_indices):
            for mb_index in reversed(range(self._n_microbatches)):
                rank_ops.append(
                    _Action(_ComputationType.BACKWARD, mb_index, stage_index)
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

        # Dictionary for tracking {stage index : current microbatch index}
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
            self.n_local_stages * self.pp_group_size
            + 2 * (self.pp_group_size - 1 - rank)
        ) - (warmup_ops + rank)

        for op in range(total_ops):
            # Warmup phase
            if op < warmup_ops:
                fwd_stage_index = forward_stage_index(op)
                # This will assign the current microbatch index and update it as well
                fwd_stage_mb_index[fwd_stage_index] = (
                    mb_index := fwd_stage_mb_index[fwd_stage_index]
                ) + 1
                rank_ops.append(
                    _Action(_ComputationType.FORWARD, mb_index, fwd_stage_index)
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
                    _Action(_ComputationType.FORWARD, fwd_mb_index, fwd_stage_index)
                )

                bwd_stage_index = backward_stage_index(op)
                bwd_stage_mb_index[bwd_stage_index] = (
                    bwd_mb_index := bwd_stage_mb_index[bwd_stage_index]
                ) + 1
                rank_ops.append(
                    _Action(_ComputationType.BACKWARD, bwd_mb_index, bwd_stage_index)
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
                    _Action(_ComputationType.BACKWARD, bwd_mb_index, bwd_stage_index)
                )

        # Post padding
        for _ in range(self.pp_group_size - rank - 1):
            rank_ops.append(None)
        return rank_ops
