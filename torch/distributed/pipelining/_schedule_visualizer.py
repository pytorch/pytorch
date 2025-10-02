# Copyright (c) Meta Platforms, Inc. and affiliates

"""
This visualizer requires matplotlib to be installed.

Example usage:

ops = get_schedule_ops("InterleavedZeroBubble", 4, 8)
visualize_schedule(ops, "test.png")
"""

import collections
from typing import NamedTuple, Optional, Union
from unittest import mock

from torch.distributed.pipelining.schedules import (
    _Action,
    _ComputationType,
    _PipelineSchedule,
    _PipelineScheduleRuntime,
    get_schedule_class,
    PipelineScheduleMulti,
    PipelineScheduleSingle,
)
from torch.distributed.pipelining.stage import PipelineStage


class OpKey(NamedTuple):
    stage_index: int
    computation_type: _ComputationType
    microbatch_index: int


def get_schedule_ops(
    schedule: Union[str, type[_PipelineSchedule]],
    pp_degree: int,
    num_microbatches: int,
    num_stages_per_rank: Optional[int] = None,
    add_spacing: bool = False,
    with_comms: bool = False,
) -> list[list[Optional[_Action]]]:
    """
    Get all actions for a given schedule, pp_degree, and num_microbatches. The actions are returned in a list of lists
    where each inner list represents a rank and each element in the inner list represents an action.

    The schedule can be specified as a string which is passed into get_schedule_class() or a _PipelineSchedule instance.
    """
    if add_spacing and with_comms:
        raise ValueError("Cannot add spacing and view comms at the same time")

    if isinstance(schedule, str):
        schedule_class = get_schedule_class(schedule)
    elif issubclass(schedule, _PipelineSchedule):
        schedule_class = schedule
    else:
        raise ValueError(f"Invalid schedule: {schedule}")

    # Create a mock of the PipelineStage class
    mock_pipeline_stage = mock.create_autospec(PipelineStage, instance=True)
    # Set the return values for group_rank and group_size methods
    mock_pipeline_stage.group_rank = 0
    mock_pipeline_stage.group_size = pp_degree
    mock_pipeline_stage.submod = None

    # Check num_stages_per_rank is valid
    if issubclass(schedule_class, PipelineScheduleSingle):
        if num_stages_per_rank is None:
            num_stages_per_rank = 1
        assert num_stages_per_rank == 1
        stages = mock_pipeline_stage
        stages.num_stages = num_stages_per_rank * pp_degree
    elif issubclass(schedule_class, PipelineScheduleMulti):
        if num_stages_per_rank is None:
            num_stages_per_rank = 2
        assert num_stages_per_rank >= 2
        stages = [mock_pipeline_stage for _ in range(num_stages_per_rank)]
        for stage in stages:
            stage.num_stages = num_stages_per_rank * pp_degree

    else:
        raise ValueError(f"Invalid schedule: {schedule_class}")

    # Instantiate the schedule class
    schedule_instance = schedule_class(stages, num_microbatches)
    assert schedule_instance.pipeline_order is not None

    # Convert to List[List[_Action]]
    all_actions: list[list[Optional[_Action]]] = []
    if with_comms:
        runtime = _PipelineScheduleRuntime(stages, num_microbatches)
        runtime._prepare_schedule_with_comms(schedule_instance.pipeline_order)
        for rank in range(pp_degree):
            all_actions.append(list(runtime.pipeline_order_with_comms[rank]))
    else:
        for rank in range(pp_degree):
            all_actions.append(schedule_instance.pipeline_order[rank])

    # Add spacing
    if add_spacing:
        # remove all Nones, then respace
        # TODO: later we can change this at the schedule creation level to not use Nones
        all_actions = [
            [action for action in rank if action is not None] for rank in all_actions
        ]
        all_actions = add_schedule_op_spacing(all_actions)

    # Return the pipeline order
    return all_actions


class _ComputationTypeVisual:
    def __init__(
        self,
        color: str,
        text: str = "",
        width: int = 1,
    ):
        self.color = color
        self.width = width
        self.text = text


# Update the mapping to use _ComputationTypeVisual instances
action_type_to_color_mapping = {
    _ComputationType.FORWARD: _ComputationTypeVisual("blue", "Forward"),
    _ComputationType.BACKWARD_INPUT: _ComputationTypeVisual("teal", "Backward Input"),
    _ComputationType.BACKWARD_WEIGHT: _ComputationTypeVisual(
        "green", "Backward Weight"
    ),
    _ComputationType.FULL_BACKWARD: _ComputationTypeVisual(
        "orange", "Full Backward", 2
    ),
    _ComputationType.OVERLAP_F_B: _ComputationTypeVisual("purple", "Overlap F+B", 3),
}


def add_schedule_op_spacing(
    schedule: list[list[Optional[_Action]]],
) -> list[list[Optional[_Action]]]:
    """
    Add spacing to the schedule based on dependencies between ranks.

    Before adding an operation to the list, this function checks if there are
    dependencies from other ranks. If there are dependencies (other ranks have
    not finished processing the required microbatch), it adds None instead.

    For example, Forward microbatch 0 on rank 1 depends on rank 0 processing
    Forward microbatch 0 first.

    Args:
        schedule: The original schedule as a list of lists where each inner list
                 represents a rank and each element represents an action.

    Returns:
        A new schedule with proper spacing based on dependencies.
    """
    if not schedule:
        return schedule

    num_stages = (
        max(
            action.stage_index
            for rank_actions in schedule
            for action in rank_actions
            if action is not None
        )
        + 1
    )

    num_ranks = len(schedule)
    spaced_schedule: list[list[Optional[_Action]]] = [[] for _ in range(num_ranks)]
    rank_ops = [collections.deque(ops) for ops in schedule]

    # Track completion times: (stage_index, action_type, microbatch_index) -> completion_time
    scheduled_ops: dict[OpKey, int] = {}

    def is_dependency_ready(dependency_key: OpKey, timestep: int) -> bool:
        """Check if a dependency operation has completed by the given timestep."""
        return (
            dependency_key in scheduled_ops
            and timestep >= scheduled_ops[dependency_key]
        )

    def get_dependencies(action: _Action) -> list[OpKey]:
        """Get the list of dependencies for an action."""
        stage_idx = action.stage_index
        comp_type = action.computation_type
        mb_idx = action.microbatch_index

        # Ensure mb_idx is not None for dependency tracking
        assert mb_idx is not None, f"Action {action} has None microbatch_index"

        # First stage forward has no dependencies
        if stage_idx == 0 and comp_type == _ComputationType.FORWARD:
            return []

        # Last stage backward depends on forward from previous stage
        if stage_idx == num_stages - 1 and comp_type in (
            _ComputationType.FULL_BACKWARD,
            _ComputationType.BACKWARD_INPUT,
        ):
            return [OpKey(stage_idx - 1, _ComputationType.FORWARD, mb_idx)]

        # Forward depends on previous stage forward
        if comp_type == _ComputationType.FORWARD:
            return [OpKey(stage_idx - 1, _ComputationType.FORWARD, mb_idx)]

        # Backward depends on next stage backward
        if comp_type in (
            _ComputationType.FULL_BACKWARD,
            _ComputationType.BACKWARD_INPUT,
        ):
            return [
                OpKey(stage_idx + 1, _ComputationType.FULL_BACKWARD, mb_idx),
                OpKey(stage_idx + 1, _ComputationType.BACKWARD_INPUT, mb_idx),
            ]

        # Weight backward depends on input backward
        if comp_type == _ComputationType.BACKWARD_WEIGHT:
            return [OpKey(stage_idx, _ComputationType.BACKWARD_INPUT, mb_idx)]

        raise RuntimeError(f"Unknown computation type: {comp_type}")

    def is_action_ready(action: _Action, timestep: int) -> bool:
        """Check if an action is ready to be scheduled at the given timestep."""
        # For OR dependencies (like backward), check if any dependency is satisfied
        if action.computation_type in (
            _ComputationType.FULL_BACKWARD,
            _ComputationType.BACKWARD_INPUT,
            _ComputationType.BACKWARD_WEIGHT,
        ):
            dependencies = get_dependencies(action)
            return any(is_dependency_ready(dep, timestep) for dep in dependencies)
        # For AND dependencies, all must be satisfied
        elif action.computation_type == _ComputationType.FORWARD:
            dependencies = get_dependencies(action)
            return all(is_dependency_ready(dep, timestep) for dep in dependencies)
        elif action.computation_type == _ComputationType.OVERLAP_F_B:
            assert action.sub_actions is not None, (
                f"OVERLAP_F_B action {action} has None sub_actions"
            )
            dep_list: list[bool] = []
            for sub_action in action.sub_actions:
                dep_list.append(is_action_ready(sub_action, timestep))
            return all(dep_list)
        else:
            raise RuntimeError(f"Unknown computation type: {action.computation_type}")

    def schedule_action(action: _Action, rank: int, timestep: int) -> int:
        """Schedule an action and return completion time."""
        spaced_schedule[rank].append(action)
        comp_type = action.computation_type
        comp_time = action_type_to_color_mapping[comp_type].width
        completion_time = timestep + comp_time

        if comp_type == _ComputationType.OVERLAP_F_B:
            # For overlap actions, schedule each sub-action with cumulative timing
            assert action.sub_actions is not None, (
                f"OVERLAP_F_B action {action} has None sub_actions"
            )
            cumulative_time = 0
            for sub_action in action.sub_actions:
                assert sub_action.microbatch_index is not None, (
                    f"Sub-action {sub_action} has None microbatch_index"
                )
                sub_comp_time = action_type_to_color_mapping[
                    sub_action.computation_type
                ].width
                cumulative_time += sub_comp_time
                scheduled_ops[
                    OpKey(
                        sub_action.stage_index,
                        sub_action.computation_type,
                        sub_action.microbatch_index,
                    )
                ] = timestep + cumulative_time
        else:
            assert action.microbatch_index is not None, (
                f"Action {action} has None microbatch_index"
            )
            scheduled_ops[
                OpKey(action.stage_index, comp_type, action.microbatch_index)
            ] = completion_time

        return completion_time

    # Main scheduling loop
    current_timestep = 0
    timesteps_without_progress = 0
    rank_completion_times = dict.fromkeys(range(num_ranks), 0)
    while rank_ops:
        print(f"Current timestep: {current_timestep}")
        # Process all operations during timestep until we run out of ready operations
        for rank, op_queue in enumerate(rank_ops):
            if not op_queue:
                continue

            op_queue = rank_ops[rank]
            action = op_queue[0]
            print(f"Rank: {rank}, {action=}")
            if action is None:
                spaced_schedule[rank].append(None)
                op_queue.popleft()
                timesteps_without_progress = 0
            elif current_timestep >= rank_completion_times[rank] and is_action_ready(
                action, current_timestep
            ):
                rank_completion_times[rank] = schedule_action(
                    action, rank, current_timestep
                )
                op_queue.popleft()
                timesteps_without_progress = 0

        # Add None for ranks that are waiting
        for rank in range(num_ranks):
            if current_timestep >= rank_completion_times[rank]:
                spaced_schedule[rank].append(None)

        # Remove empty queues and advance timestep
        rank_ops = [op_queue for op_queue in rank_ops if op_queue]
        current_timestep += 1
        timesteps_without_progress += 1

        if timesteps_without_progress > max(
            visual.width for visual in action_type_to_color_mapping.values()
        ):
            raise RuntimeError("No progress made in scheduling - possible deadlock")

    return spaced_schedule


def visualize_schedule(
    schedule: list[list[Optional[_Action]]],
    filename: Optional[str] = None,
) -> None:
    """
    Visualize the schedule using matplotlib.
    The schedule is a list of lists where each inner list represents a rank and each element in the inner list represents an action.
    The actions are represented as rectangles with different colors based on their computation type.
    The filename is optional and if provided, the plot will be saved to that file.

    Args:
        schedule: The schedule to visualize.
        filename: The filename to save the plot to. If not provided, the plot will be displayed.
        add_schedule_spacing: If True, add spacing to the schedule based on dependencies between ranks.

    """

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.rcParams["font.family"] = (
        "DejaVu Sans"  # or any other font available on your system
    )
    num_ranks = len(schedule)
    max_actions = max(len(rank) for rank in schedule)

    # Increase the figure size to provide more space for the legend
    fig, ax = plt.subplots(figsize=(max_actions + 2, num_ranks + 2))
    max_draw_position = -1
    # Calculate dynamic font size based on figure size
    font_size = min(max_actions, num_ranks) + 4
    used_computation = set()
    for rank_idx, actions in enumerate(schedule):
        draw_position = 0  # Initialize drawing position for each rank
        for action in actions:
            if action is not None:
                comp_type_color = action_type_to_color_mapping.get(
                    action.computation_type, _ComputationTypeVisual("black")
                )
                used_computation.add(action.computation_type)
                color = comp_type_color.color
                width = comp_type_color.width

                # Check if action has sub_actions to determine styling
                if action.sub_actions is not None:
                    linewidth = 2  # Thicker border for compound actions
                    text_weight = "normal"  # Bold text for compound actions
                else:
                    linewidth = 1  # Default linewidth for regular actions
                    text_weight = "normal"  # Default text weight

                # Draw the rectangle to represent the action duration
                rect = Rectangle(
                    (draw_position, num_ranks - rank_idx - 1),
                    width,
                    1,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=linewidth,
                )
                ax.add_patch(rect)

                # Draw the text centered within the rectangle
                ax.text(
                    draw_position + width / 2,
                    num_ranks - rank_idx - 1 + 0.5,
                    str(action),
                    ha="center",
                    va="center",
                    fontsize=font_size,
                    color="white",
                    weight=text_weight,
                )

                draw_position += width
            else:
                draw_position += 1  # Move to the next
            max_draw_position = max(max_draw_position, draw_position)
    ax.set_xlim(-0.5, max_draw_position + 1)
    ax.set_ylim(-0.5, num_ranks + 0.5)  # Add extra space at the top
    # Set y-ticks to be in the middle of each rank's row
    ax.set_yticks([num_ranks - rank_idx - 0.5 for rank_idx in range(num_ranks)])
    ax.set_yticklabels([f"Rank {i}" for i in range(num_ranks)], fontsize=font_size)
    ax.set_xticklabels([])

    # Remove grid lines and ticks
    ax.grid(False)
    # Add legend with larger font size
    legend_elements = [
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor=action_type_to_color_mapping[comp_type].color,
            edgecolor="black",
            label=action_type_to_color_mapping[comp_type].text,
        )
        for comp_type in used_computation
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=font_size)
    # Save to file if filename is provided, otherwise display the plot
    if filename:
        plt.savefig(filename, bbox_inches="tight")
    else:
        plt.show()
