# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

import csv
import logging
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule, UnshardHandle
from torch.profiler import record_function

# Import from schedules.py (actions and base classes)
from .schedules import (
    _Action,
    _add_reduce_grad,
    _batch_p2p,
    _ComputationType,
    _format_pipeline_order,
    _get_profiler_function_name,
    _PipelineSchedule,
    _wait_batch_p2p,
    B,
    BACKWARD_INPUT,
    BACKWARD_WEIGHT,
    F,
    FORWARD,
    FULL_BACKWARD,
    I,
    OVERLAP_F_B,
    RECV_B,
    RECV_F,
    REDUCE_GRAD,
    RESHARD,
    SEND_B,
    SEND_F,
    UNSHARD,
    W,
)
from ._utils import generate_stage_to_rank_mapping
from .microbatch import TensorChunkSpec
from .stage import _PipelineStageBase


__all__ = [
    "PipelineScheduleMulti",
    "_PipelineScheduleRuntime",
    "_PipelineContext",
    "_validate_schedule",
    "_add_unshard_reshard",
    "_merge_bw",
    "_add_send_recv",
    "_simulate_comms_compute",
]

logger = logging.getLogger(__name__)

def _add_unshard_reshard(
    compute_actions: list[_Action | None],
    max_active_stages: int = 3,
) -> list[_Action]:
    """Given a basic schedule involving only compute actions (F,B,W,OVERLAP_F_B), add UNSHARD/RESHARD actions for FSDP.

    UNSHARD refers to fetching the full contents of an FSDP-sharded layer, requiring an all-gather operation.
    RESHARD does the opposite, releasing memory (but doing no communication)

    We abandon the "timestep lock"  during lowering

    max_active_stages controls how many prefetches we allow. It should be measured in mb and tuneable but in practice
    3 stages is probably the thing we want?
    (to account for having one f and one b active, and something else prefetching?)
    """

    def next_stage_indices(count: int, next_actions: list[_Action | None]) -> list[int]:
        """Remove duplicates (same stage, different microbatch), find next 'count' stages that will do compute."""
        seen: set[int] = set()
        ret: list[int] = []

        for a in next_actions:
            if a is not None:
                # Handle OVERLAP_F_B actions by checking their sub_actions
                if a.computation_type == OVERLAP_F_B and a.sub_actions is not None:
                    for sub_action in a.sub_actions:
                        if sub_action.stage_index not in seen:
                            seen.add(sub_action.stage_index)
                            ret.append(sub_action.stage_index)
                    if len(ret) >= count:
                        break
                else:
                    # Regular action
                    if a.stage_index not in seen:
                        seen.add(a.stage_index)
                        ret.append(a.stage_index)
                        if len(ret) == count:
                            break
        return ret

    active_stages: set[int] = set()
    fsdp_aware_actions: list[_Action] = []

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

    # Reshard all remaining active stages after processing all operations
    for stage in list(active_stages):
        _reshard(stage)

    return fsdp_aware_actions

def _merge_bw(
    compute_actions: list[_Action | None],
) -> list[_Action]:
    """Given a basic schedule involving only compute actions (F,I,W), merge adjacent I and W ops into B ops.
    (note: I = BACKWARD_INPUT, W = BACKWARD_WEIGHT, B = FULL_BACKWARD)

    B refers to running the whole backward (not separating grad_input and grad_weight), which can be more efficient
    in some cases.
    """
    merged_actions = []
    while compute_actions:
        action = compute_actions.pop(0)
        if action is None:
            continue

        # Remove any None actions and find the next non-None action
        while len(compute_actions) and compute_actions[0] is None:
            compute_actions.pop(0)

        # Get the next action if it exists
        next_action = compute_actions[0] if len(compute_actions) > 0 else None

        if (
            action.computation_type == BACKWARD_INPUT
            and next_action is not None
            and next_action.computation_type == BACKWARD_WEIGHT
            and action.stage_index == next_action.stage_index
            and action.microbatch_index == next_action.microbatch_index
        ):
            merged_actions.append(
                _Action(action.stage_index, FULL_BACKWARD, action.microbatch_index)
            )
            compute_actions.pop(0)
        else:
            merged_actions.append(action)
    return merged_actions

def _add_send_recv(
    compute_actions: dict[int, list[_Action]],
    stage_to_rank: Callable[[int], int],
    num_stages: int,
) -> dict[int, list[_Action]]:
    """
    Transforms a compute-only schedule into a complete schedule with communication actions.

    For actions with sub-actions (OVERLAP_F_B) we ensure that all the subactions have been
    computed and the communication is ready
    """
    comm_actions: dict[int, list[_Action]] = {rank: [] for rank in compute_actions}
    prev_actions: dict[int, set[_Action]] = {rank: set() for rank in compute_actions}

    def _has_comms(action: _Action) -> bool:
        if action.computation_type == F:
            return action.stage_index != num_stages - 1 and stage_to_rank(
                action.stage_index + 1
            ) != stage_to_rank(action.stage_index)
        elif action.computation_type in (BACKWARD_INPUT, FULL_BACKWARD):
            return action.stage_index != 0 and stage_to_rank(
                action.stage_index - 1
            ) != stage_to_rank(action.stage_index)
        return False

    def _get_comms(action: _Action) -> tuple[_Action, _Action]:
        if not _has_comms(action):
            raise AssertionError(f"{action} is not a valid comm action")
        stage_idx = action.stage_index
        ctype = action.computation_type
        mb_idx = action.microbatch_index
        send = _Action(stage_idx, SEND_F if ctype == F else SEND_B, mb_idx)
        recv_stage_idx = stage_idx + 1 if ctype == F else stage_idx - 1
        recv = _Action(recv_stage_idx, RECV_F if ctype == F else RECV_B, mb_idx)
        return send, recv

    def _ready_to_schedule(action: _Action | None, prev_actions: set[_Action]) -> bool:
        """We don't put our own recv ops in the schedule, we let a sender on another rank put our recv ops in place.
        This helps ensure a sane (non-hanging) ordering of sends and recvs.
        But it also means we might not be able to schedule our next compute action yet.
        """
        if action is None:
            return True
        elif action.computation_type == F and action.stage_index != 0:
            if (
                _Action(action.stage_index, RECV_F, action.microbatch_index)
                in prev_actions
            ):
                return True
            elif (
                _Action(action.stage_index - 1, F, action.microbatch_index)
                in prev_actions
            ):
                return True
            return False
        elif (
            action.computation_type in (BACKWARD_INPUT, FULL_BACKWARD)
            and action.stage_index != num_stages - 1
        ):
            if (
                _Action(action.stage_index, RECV_B, action.microbatch_index)
                in prev_actions
            ):
                return True
            elif (
                _Action(action.stage_index + 1, BACKWARD_INPUT, action.microbatch_index)
                in prev_actions
            ):
                return True
            elif (
                _Action(action.stage_index + 1, FULL_BACKWARD, action.microbatch_index)
                in prev_actions
            ):
                return True
            return False
        else:
            return True

    while compute_actions:
        progress = False
        # go in order of ranks even if dict keys aren't ordered
        for rank in sorted(compute_actions):
            if not (len(compute_actions[rank]) > 0):
                raise AssertionError(f"{rank=}, {len(compute_actions[rank])=}")
            action = compute_actions[rank][0]
            # handle case where parent action (e.g. OVERLAP_F_B) can be comprised of subactions
            if action is not None and action.sub_actions is not None:
                all_actions = action.sub_actions
            else:
                all_actions = (action,)

            if not all(_ready_to_schedule(a, prev_actions[rank]) for a in all_actions):
                continue

            # The action's dependencies are satisfied, so add to schedule
            if action is not None:
                comm_actions[rank].append(action)
                for a in all_actions:
                    prev_actions[rank].add(a)
                    if _has_comms(a):
                        send, recv = _get_comms(a)
                        # TODO we can avoid send/recv if the 2 stages are on the same rank.
                        # should we avoid that in the runtime or here?
                        comm_actions[rank].append(send)
                        prev_actions[rank].add(send)
                        comm_actions[stage_to_rank(recv.stage_index)].append(recv)
                        prev_actions[stage_to_rank(recv.stage_index)].add(recv)

            compute_actions[rank].pop(0)
            if len(compute_actions[rank]) == 0:
                del compute_actions[rank]
            progress = True
        if not progress:
            raise AssertionError(
                "Malformed compute schedule, can't schedule sends/recvs"
            )
    return comm_actions

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


@dataclass
class _PipelineContext:
    """Container of the pipeline context passed for custom functions."""

    schedule_ref: _PipelineSchedule
    arg_mbs: list[tuple] | None = None
    kwarg_mbs: list[dict] | None = None
    target_mbs: list | None = None
    losses: list | None = None


class _CustomFunctionProtocol(Protocol):
    def __call__(self, action: _Action, ctx: _PipelineContext) -> None: ...


class _PipelineScheduleRuntime(PipelineScheduleMulti):
    """
    Provides a simple runtime that requires a 'schedule IR' including specified communication operations.

    Can be instantiated directly by creating _PipelineScheduleRuntime and calling load_csv, or can be
    subclassed and the subclass can be responsible for creating a schedule IR.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Action to custom function mapping
        self._comp_type_to_function_map: dict[_ComputationType, Callable] = {}
        # count either full_backward or backward_weight together, to determine when to sync DP grads
        self.backward_counter: Counter[int] = Counter()

        # recv ops indexed by (stage_idx, mb_idx) need to be waited on before use
        self.bwd_recv_ops: dict[tuple[int, int], list[dist.Work]] = {}
        self.fwd_recv_ops: dict[tuple[int, int], list[dist.Work]] = {}

        # we track which stages are 'active' when used with FSDP, and wait on unshard ops before computing on stages
        self.unshard_ops: dict[int, list[UnshardHandle]] = defaultdict(list)
        self.unsharded_stages = set()

    def register_custom_function(
        self,
        computation_type: _ComputationType,
        custom_function: _CustomFunctionProtocol,
    ) -> None:
        """
        Register a custom function to be executed for a specific computation type.

        Args:
            computation_type: The computation type for which to register the custom function
            custom_function: The function to execute when this computation type is encountered.
                Must have signature: (action: _Action, ctx: _PipelineContext) -> None
        """
        # Ensure that the computation type is valid
        if computation_type not in (
            FORWARD,
            FULL_BACKWARD,
            BACKWARD_INPUT,
            BACKWARD_WEIGHT,
            OVERLAP_F_B,
            UNSHARD,
            RESHARD,
            REDUCE_GRAD,
        ):
            raise ValueError(
                f"Invalid computation type {computation_type}. Only FORWARD, FULL_BACKWARD, \
                BACKWARD_INPUT, BACKWARD_WEIGHT, OVERLAP_F_B, UNSHARD, RESHARD and REDUCE_GRAD are supported."
            )

        # Check if computation_type is already registered
        if computation_type in self._comp_type_to_function_map:
            logger.warning(
                "Computation type %s is already registered. "
                "Overwriting the existing custom function.",
                computation_type,
            )

        self._comp_type_to_function_map[computation_type] = custom_function

    def _prepare_schedule_with_comms(
        self,
        actions: dict[int, list[_Action | None]],
        format: str = "compute_only",
    ):
        """
        Given an in-memory representation for a simple compute-only schedule, lower it to a complex schedule including
        communication actions.  Stores the schedule in self, and must be called before running step_mo()
        """
        # validate the provided actions are valid and overrides the default stage_index_to_group_rank
        super()._validate_and_set_stage_mapping(actions)

        self.pipeline_order_with_comms: dict[int, list[_Action]] = {}
        if format == "compute_comms":
            for rank in actions:
                self.pipeline_order_with_comms[rank] = []
                for action in actions[rank]:
                    if action is None:
                        raise AssertionError(
                            f"Expected action to be not None, got {type(action)}"
                        )
                    self.pipeline_order_with_comms[rank].append(action)
            # TODO what level of validation should we offer for compute+comms schedule?
        elif format == "compute_only":
            # Validate that the schedule does not have comms already added to it
            for rank, action_list in actions.items():
                for i, action in enumerate(action_list):
                    if action is not None:
                        if not action.is_compute_op:
                            raise ValueError(
                                f"Expected compute-only schedule but found communication action "
                                f"'{action}' at rank {rank}, position {i}. "
                                f"Communication actions (e.g. SEND_F, RECV_F, etc.) "
                                f"should not be present when format='compute_only'."
                            )

            # Perform schedule lowering
            for rank in actions:
                self.pipeline_order_with_comms[rank] = _add_unshard_reshard(
                    actions[rank]
                )
                self.pipeline_order_with_comms[rank] = _add_reduce_grad(  # type: ignore[assignment]
                    self.pipeline_order_with_comms[rank],  # type: ignore[arg-type]
                    self._n_microbatches,
                )

            self.pipeline_order_with_comms = _add_send_recv(
                self.pipeline_order_with_comms,
                stage_to_rank=lambda s: self.stage_index_to_group_rank[s],
                num_stages=self._num_stages,
            )
        else:
            raise NotImplementedError(f"{format=} is not implemented")

    def _load_csv(self, filename: str, format: str = "compute_only"):
        """Loads a csv in simple format and then lowers it to include communication actions

        format must be either "compute_only" or "compute_comms".  If compute_only, the lowering passes
        will automatically be run to generate a compute_comms schedule.
        """
        if format == "compute_only":
            # this will populate self.pipeline_order
            super()._load_csv(filename)
            # this will populate self.pipeline_order_with_comms
            self._prepare_schedule_with_comms(self.pipeline_order)
        elif format == "compute_comms":
            actions = {}
            with open(filename, newline="") as csvfile:
                reader = csv.reader(csvfile)
                for rank, row in enumerate(reader):
                    actions[rank] = [_Action.from_str(s) for s in row]
                self._prepare_schedule_with_comms(actions, format=format)
        else:
            raise NotImplementedError(f"{format=} is not implemented")

    def _dump_csv(self, filename: str, format: str = "compute_comms"):
        """Dump a CSV representation of the schedule into a file with the provided filename."""
        if format == "compute_only":
            if self.pipeline_order is None:
                raise AssertionError("Compute only schedule must be available")
            with open(filename, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                for rank in self.pipeline_order:
                    writer.writerow(self.pipeline_order[rank])
        elif format == "compute_comms":
            if self.pipeline_order_with_comms is None:
                raise AssertionError(
                    "Must initialize compute_comms schedule before dump_csv"
                )
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

    def _assert_unsharded(self, stage: _PipelineStageBase):
        """If an unshard is active for `stage_idx`, wait() it and mark `stage_idx` unshared."""
        stage_uses_fsdp = isinstance(stage.submod, FSDPModule)
        if stage_uses_fsdp:
            stage_idx = stage.stage_index
            if stage_idx in self.unshard_ops:
                for op in self.unshard_ops[stage_idx]:
                    op.wait()
                del self.unshard_ops[stage_idx]
                self.unsharded_stages.add(stage_idx)
            if stage_idx not in self.unsharded_stages:
                raise AssertionError(f"Attempted to compute on sharded {stage_idx=}")

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

        if self.pipeline_order_with_comms is None:
            raise AssertionError(
                "Must call _prepare_schedule_with_comms() before calling _step_microbatches()"
            )

        # send ops should be waited on before step() exists, mainly for hygiene
        send_ops: list[list[dist.Work]] = []

        def _perform_action(action: _Action) -> None:
            comp_type = action.computation_type
            mb_index: int = (
                action.microbatch_index if action.microbatch_index is not None else -1
            )
            if not (
                mb_index >= 0
                or comp_type
                in (
                    UNSHARD,
                    RESHARD,
                    REDUCE_GRAD,
                )
            ):
                raise AssertionError(f"{action=} missing mb_index")
            stage_idx = action.stage_index
            stage = stage_index_to_stage[stage_idx]
            stage_uses_fsdp = isinstance(stage.submod, FSDPModule)
            # see [Note: V-schedule special case]
            is_next_stage_on_this_rank = stage_idx + 1 in stage_index_to_stage
            is_prev_stage_on_this_rank = stage_idx - 1 in stage_index_to_stage

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
                if (stage_idx, mb_index) in self.fwd_recv_ops:
                    raise AssertionError(
                        f"Recv twice for {stage_idx=} {mb_index=} without executing forward"
                    )
                self.fwd_recv_ops[(stage_idx, mb_index)] = _batch_p2p(
                    stage.get_fwd_recv_ops(mb_index)
                )
            elif comp_type == RECV_B:
                if (stage_idx, mb_index) in self.bwd_recv_ops:
                    raise AssertionError(
                        f"Recv twice for {stage_idx=} {mb_index=} without executing backward"
                    )
                self.bwd_recv_ops[(stage_idx, mb_index)] = _batch_p2p(
                    stage.get_bwd_recv_ops(mb_index)
                )
            elif comp_type == UNSHARD:
                if stage_uses_fsdp:
                    if not (
                        stage_idx not in self.unsharded_stages
                        and stage_idx not in self.unshard_ops
                    ):
                        raise AssertionError(f"Unsharding the same {stage_idx=} twice")
                    for submodule in stage.submod.modules():
                        if not isinstance(submodule, FSDPModule):
                            continue
                        handle = cast(UnshardHandle, submodule.unshard(async_op=True))
                        self.unshard_ops[stage_idx].append(handle)
            elif comp_type == RESHARD:
                if stage_uses_fsdp:
                    if stage_idx not in self.unsharded_stages:
                        raise AssertionError(
                            f"Resharding {stage_idx=} without unsharding"
                        )
                    if stage_idx in self.unshard_ops:
                        raise AssertionError(
                            f"Resharding {stage_idx=} before finishing unshard"
                        )
                    for submodule in stage.submod.modules():
                        if not isinstance(submodule, FSDPModule):
                            continue
                        submodule.reshard()
                    self.unsharded_stages.remove(stage_idx)
            elif comp_type == FORWARD:
                self._assert_unsharded(stage)

                if (
                    not stage.is_first
                    # no recv op expected for V-schedule special case (see [Note: V-schedule special case])
                    and not is_prev_stage_on_this_rank
                ):
                    if (stage_idx, mb_index) not in self.fwd_recv_ops:
                        raise AssertionError(
                            f"Computing {action=} before receiving input"
                        )
                    _wait_batch_p2p(self.fwd_recv_ops.pop((stage_idx, mb_index)))

                output = stage.forward_one_chunk(
                    mb_index,
                    arg_mbs[mb_index],  # type: ignore[index]
                    kwarg_mbs[mb_index],  # type: ignore[index]
                    save_forward_output=return_outputs,
                )
                self._maybe_compute_loss(stage, output, target_mbs, mb_index)

                # SEND/RECV op are avoided for special case with 2 adjacent stages on same rank
                # see [Note: V-schedule special case]
                if is_next_stage_on_this_rank:
                    stage_index_to_stage[stage_idx + 1].set_local_fwd_input(
                        output, mb_index
                    )

            elif comp_type == FULL_BACKWARD:
                self._assert_unsharded(stage)

                if (
                    not stage.is_last
                    # no recv op expected for V-schedule special case (see [Note: V-schedule special case])
                    and not is_next_stage_on_this_rank
                ):
                    if (stage_idx, mb_index) not in self.bwd_recv_ops:
                        raise AssertionError(
                            f"Attempted to run compute {action=} before receiving input"
                        )
                    _wait_batch_p2p(self.bwd_recv_ops.pop((stage_idx, mb_index)))
                loss = self._maybe_get_loss(stage, mb_index)
                self.backward_counter[stage_idx] += 1
                last_backward = self.backward_counter[stage_idx] == self._n_microbatches
                stage.backward_one_chunk(
                    mb_index,
                    loss=loss,
                    full_backward=True,
                    last_backward=last_backward,
                )
                # SEND/RECV op are avoided for special case with 2 adjacent stages on same rank
                # see [Note: V-schedule special case]
                if is_prev_stage_on_this_rank:
                    stage_index_to_stage[stage_idx - 1].set_local_bwd_input(
                        stage.get_local_bwd_output(mb_index), mb_index
                    )
            elif comp_type == BACKWARD_INPUT:
                self._assert_unsharded(stage)

                if not stage.is_last and not is_next_stage_on_this_rank:
                    if (stage_idx, mb_index) not in self.bwd_recv_ops:
                        raise AssertionError(
                            f"Attempted to run compute {action=} before receiving input"
                        )
                    _wait_batch_p2p(self.bwd_recv_ops.pop((stage_idx, mb_index)))
                loss = self._maybe_get_loss(stage, mb_index)
                stage.backward_one_chunk(
                    mb_index,
                    loss=loss,
                    full_backward=False,
                    last_backward=False,
                )
                # SEND/RECV op are avoided for special case with 2 adjacent stages on same rank
                # see [Note: V-schedule special case]
                if is_prev_stage_on_this_rank:
                    stage_index_to_stage[stage_idx - 1].set_local_bwd_input(
                        stage.get_local_bwd_output(mb_index), mb_index
                    )
            elif comp_type == BACKWARD_WEIGHT:
                self._assert_unsharded(stage)
                self.backward_counter[stage_idx] += 1
                last_backward = self.backward_counter[stage_idx] == self._n_microbatches
                stage.backward_weight_one_chunk(
                    mb_index,
                    last_backward=last_backward,
                )
            elif comp_type == REDUCE_GRAD:
                grad_scale_factor = self._n_microbatches if self.scale_grads else 1
                stage.perform_reduce_grad(grad_scale_factor)
            else:
                raise ValueError(f"{action=} is unknown or unsupported")

        # count either full_backward or backward_weight together, to determine when to sync DP grads
        self.backward_counter.clear()
        for time_step, action in enumerate(self.pipeline_order_with_comms[self.rank]):
            logger.debug(
                "_PipelineScheduleRuntime running time_step %d, action %s",
                time_step,
                action,
            )
            try:
                with record_function(_get_profiler_function_name(action)):
                    if action.computation_type in self._comp_type_to_function_map:
                        ctx = _PipelineContext(
                            self,
                            arg_mbs,
                            kwarg_mbs,
                            target_mbs,
                            losses,
                        )
                        self._comp_type_to_function_map[action.computation_type](
                            action, ctx
                        )
                    elif action.computation_type == OVERLAP_F_B:
                        if action.sub_actions is None:
                            raise AssertionError("sub_actions must be set")
                        for sub_a in action.sub_actions:
                            _perform_action(sub_a)
                    else:
                        _perform_action(action)
            except Exception as e:
                logger.error(
                    "_PipelineScheduleRuntime caught exception at step %s when running action %s.  Full Schedule:",
                    time_step,
                    action,
                )
                logger.error(
                    _format_pipeline_order(
                        self.pipeline_order_with_comms,  # type: ignore[arg-type]
                        error_step_number=time_step,
                    )
                )
                raise e

        # Mostly these operations should have finished long ago, but there isn't an obvious time when to wait for them
        while send_ops:
            _wait_batch_p2p(send_ops.pop())

        if len(self.unshard_ops) != 0:
            raise AssertionError("Unused unshard operations")

        # Return losses if there is a container passed in
        self._update_losses(self._stages, losses)


def _simulate_comms_compute(
    pipeline_order, stage_to_rank: Callable[[int], int], num_stages: int
):
    """This function dry-run simulates the actions in the schedule from the perspective of all ranks, and flags
    any deadlocks caused by missing or misordered communications.  It also simulates any bubbles in time where a rank
    can not execute any action due to waiting for unmet dependencies.  The total number of simulator steps can be used
    as a metric for unit tests involving IR optimization passes as reordering and merging of IR can reduce the number
    of simulated steps.

    The simulation is not high-fidelity and does not model overlapping of compute and communication, or cuda streams.
    Future work may be to enhance this and model the compute time, comms overlap, and even memory.
    """
    pipeline_order = {
        rank: [a for a in pipeline_order[rank] if a is not None]
        for rank in sorted(pipeline_order)
    }
    _schedule: dict[int, list[_Action | None]] = {
        rank: [] for rank in sorted(pipeline_order)
    }

    _prev_ops_rank: dict[int, set[_Action]] = {rank: set() for rank in _schedule}

    def add_to_schedule(rank: int, action: _Action | None):
        _schedule[rank].append(action)
        if action is not None:
            _prev_ops_rank[rank].add(action)

    def _ready_to_schedule(action: _Action | None) -> bool:
        if action is None:
            return True

        stage_idx = action.stage_index
        prev_ops = _prev_ops_rank[stage_to_rank(stage_idx)]
        if action.computation_type == F:
            if action.stage_index == 0:
                return True
            elif (
                _Action(action.stage_index, RECV_F, action.microbatch_index) in prev_ops
            ):
                return True
            elif (
                _Action(action.stage_index - 1, F, action.microbatch_index) in prev_ops
            ):
                return True
            return False
        elif action.computation_type in (BACKWARD_INPUT, FULL_BACKWARD):
            if action.stage_index == num_stages - 1:
                return True
            if _Action(action.stage_index, RECV_B, action.microbatch_index) in prev_ops:
                return True
            if (
                _Action(action.stage_index + 1, BACKWARD_INPUT, action.microbatch_index)
                in prev_ops
            ):
                return True
            if (
                _Action(action.stage_index + 1, FULL_BACKWARD, action.microbatch_index)
                in prev_ops
            ):
                return True
            return False
        elif action.computation_type == BACKWARD_WEIGHT:
            return True
        elif action.computation_type == SEND_F:
            expected_f = _Action(action.stage_index, F, action.microbatch_index)
            return expected_f in prev_ops
        elif action.computation_type == RECV_F:
            peer_stage_idx = stage_idx - 1
            expected_send = _Action(peer_stage_idx, SEND_F, action.microbatch_index)
            return expected_send in _prev_ops_rank[stage_to_rank(peer_stage_idx)]
        elif action.computation_type == SEND_B:
            expected_b = _Action(
                action.stage_index, BACKWARD_INPUT, action.microbatch_index
            )
            expected_bw = _Action(
                action.stage_index, FULL_BACKWARD, action.microbatch_index
            )
            return expected_b in prev_ops or expected_bw in prev_ops
        elif action.computation_type == RECV_B:
            peer_stage_idx = stage_idx + 1
            expected_send = _Action(peer_stage_idx, SEND_B, action.microbatch_index)
            return expected_send in _prev_ops_rank[stage_to_rank(peer_stage_idx)]
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
                    add_to_schedule(rank, action)
                pipeline_order[rank].pop(0)
                progress = True
            else:
                add_to_schedule(rank, None)

        for i in sorted(pipeline_order, reverse=True):
            if len(pipeline_order[i]) == 0:
                del pipeline_order[i]

        # hacky, but do a second pass to replace any 'none' at this timestep with a real action, if it got unblocked
        # by one of the later ranks
        for rank in sorted(pipeline_order):
            if len(pipeline_order[rank]) == 0:
                continue

            if _schedule[rank][-1] is not None:
                continue

            action = pipeline_order[rank][0]
            if _ready_to_schedule(action):
                if action is not None:
                    _schedule[rank][-1] = action
                    _prev_ops_rank[rank].add(action)
                pipeline_order[rank].pop(0)

        for i in sorted(pipeline_order, reverse=True):
            if len(pipeline_order[i]) == 0:
                del pipeline_order[i]

        if not progress:
            print("WIP comms schedule:\n", _format_pipeline_order(_schedule))
            for rank in pipeline_order:
                print(f"{rank=} next action= {pipeline_order[rank][0]}")
            raise ValueError("Schedule is not progressing")

    return _schedule



def _dump_chrometrace(schedule, filename):
    """
    This function dumps a schedule IR into a chrometrace format so it can be visualized.

    It is currently very basic and only serves as a graphical alternative to dumping the schedule IR as text.

    As future work we may extend this to include more accurate heuristics for durations, or let users input durations,
    add 'flow events' to let the UI show the connection between sends and recvs, and model cuda streams for comm/compute
    as separate streams on the chrometrace view.
    """
    events = []
    for rank in sorted(schedule):
        for timestep, action in enumerate(schedule[rank]):
            if action is None:
                continue
            events.append(
                {
                    "name": str(action),
                    "cat": (
                        "computation"
                        if action.computation_type in (F, B, W)
                        else "communication"
                    ),
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
