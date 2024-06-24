# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import itertools
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import fx

from .schedules import _Action, _ComputationType


logger = logging.getLogger(__name__)


def flatten_args_detach(args):
    """
    Flatten the args into a list form and detach the tensors from computational graph.
    """
    flat_detached_args = []

    def extract_tensor_args(a):
        nonlocal flat_detached_args
        if isinstance(a, torch.Tensor):
            val = a.detach().requires_grad_(a.requires_grad)
            flat_detached_args.append(val)
            return val
        else:
            flat_detached_args.append(a)
            return a

    new_args = fx.node.map_aggregate(
        args,
        extract_tensor_args,
    )

    return new_args, flat_detached_args


def flatten_args(args):
    """
    Flatten the args into a list form.
    """
    flat_args = []

    def extract_tensor_args(a):
        nonlocal flat_args
        flat_args.append(a)
        return a

    fx.node.map_aggregate(
        args,
        extract_tensor_args,
    )

    return flat_args


class PipeliningShapeError(RuntimeError):
    """Shape mismatch between configured and runtime values."""


def validate_tensor_metadata(desc, expected, given):
    if not expected.shape == given.shape:
        raise PipeliningShapeError(
            f"{desc} has a shape mismatch: expected {expected.shape} actual {given.shape}"
        )
    if not expected.dtype == given.dtype:
        raise PipeliningShapeError(
            f"{desc} has a dtype mismatch: expected {expected.dtype} actual {given.dtype}"
        )
    if not expected.stride() == given.stride():
        raise PipeliningShapeError(
            f"{desc} has a stride mismatch: expected {expected.stride()} actual {given.stride()}"
        )


def validate_tensors_metadata(
    desc,
    expected_tensors: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]],
    actual_tensors: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]],
):
    if len(expected_tensors) != len(actual_tensors):
        raise PipeliningShapeError(
            f"{desc}: Number of values ({len(actual_tensors)}) does not match expected number ({len(expected_tensors)})"
        )
    for i in range(len(expected_tensors)):
        validate_tensor_metadata(
            f"{desc}: value {i}", expected_tensors[i], actual_tensors[i]
        )


@dataclass
class PipeInfo:
    """
    Captures information for a pipeline (`Pipe` object).
    """

    graph: fx.Graph
    num_stages: int
    has_loss_and_backward: bool


def format_pipeline_order(pipeline_order: Dict[int, List[Optional[_Action]]]) -> str:
    """
    Formats the pipeline order in a timestep (row) x rank (column) grid of actions
    and returns the formatted string
    """

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
    formatted_table = (
        "=========== ALL_RANK_ACTIONS ===========\n"
        + header_row
        + "\n"
        + "\n".join(formatted_rows)
        + "\n"
    )
    return formatted_table


def validate_pipeline_order(
    self,
    pipeline_order: Dict[int, List[Optional[_Action]]],
    num_microbatches: int,
    num_stages: int,
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
                current_timestep_actions.append(action)

        # TODO: enable this
        # if len(current_timestep_actions) == 0:
        #     error_msg.append(
        #         "All actions were None, there is an unnecessary gap in the schedule"
        #     )

        # Ensure that no microbatch is operated on twice in current_timestep_actions
        unique_microbatch_indices = {action[1] for action in current_timestep_actions}
        if len(unique_microbatch_indices) != len(current_timestep_actions):
            error_msg.append(
                "Duplicate microbatch index found in current_timestep_actions"
            )

        for action in current_timestep_actions:
            computation_type, mb_index, stage_index = action

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

        if len(error_msg) != 0:
            self.fail(f"Error at timestep {timestep}: " + ",".join(error_msg))
