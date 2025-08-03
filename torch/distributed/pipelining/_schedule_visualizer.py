# Copyright (c) Meta Platforms, Inc. and affiliates

"""
This visualizer requires matplotlib to be installed.

Example usage:

ops = get_schedule_ops("InterleavedZeroBubble", 4, 8)
visualize_schedule(ops, "test.png")
"""

from typing import Optional, Union
from unittest import mock

from torch.distributed.pipelining.schedules import (
    _Action,
    _ComputationType,
    _PipelineSchedule,
    get_schedule_class,
    PipelineScheduleMulti,
    PipelineScheduleSingle,
)
from torch.distributed.pipelining.stage import PipelineStage


def get_schedule_ops(
    schedule: Union[str, _PipelineSchedule],
    pp_degree: int,
    num_microbatches: int,
    num_stages_per_rank: Optional[int] = None,
) -> list[list[Optional[_Action]]]:
    """
    Get all actions for a given schedule, pp_degree, and num_microbatches. The actions are returned in a list of lists
    where each inner list represents a rank and each element in the inner list represents an action.

    The schedule can be specified as a string which is passed into get_schedule_class() or a _PipelineSchedule instance.
    """

    if isinstance(schedule, str):
        schedule_class = get_schedule_class(schedule)
    elif type(schedule) == _PipelineSchedule:
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

    # Convert to List[List[_Action]]
    all_actions = []
    for rank in range(pp_degree):
        all_actions.append(schedule_instance.pipeline_order[rank])

    # Return the pipeline order
    return all_actions


class _ComputationTypeColor:
    def __init__(
        self,
        color: str,
        text: str = "",
        width: int = 1,
    ):
        self.color = color
        self.width = width
        self.text = text


# Update the mapping to use _ComputationTypeColor instances
action_type_to_color_mapping = {
    _ComputationType.FORWARD: _ComputationTypeColor("blue", "Forward"),
    _ComputationType.BACKWARD_INPUT: _ComputationTypeColor("teal", "Backward Input"),
    _ComputationType.BACKWARD_WEIGHT: _ComputationTypeColor("green", "Backward Weight"),
    _ComputationType.FULL_BACKWARD: _ComputationTypeColor("orange", "Full Backward", 2),
}


def visualize_schedule(
    schedule: list[list[Optional[_Action]]], filename: Optional[str] = None
) -> None:
    """
    Visualize the schedule using matplotlib.
    The schedule is a list of lists where each inner list represents a rank and each element in the inner list represents an action.
    The actions are represented as rectangles with different colors based on their computation type.
    The filename is optional and if provided, the plot will be saved to that file.
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
                    action.computation_type, _ComputationTypeColor("black")
                )
                used_computation.add(action.computation_type)
                color = comp_type_color.color
                width = comp_type_color.width
                # Draw the rectangle to represent the action duration
                rect = Rectangle(
                    (draw_position, num_ranks - rank_idx - 1),
                    width,
                    1,
                    facecolor=color,
                    edgecolor="black",
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
                )
                # Increment the drawing position by the width of the current action
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
