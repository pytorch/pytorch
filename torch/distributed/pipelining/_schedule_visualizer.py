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

import copy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


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
legend = {
    _ComputationType.FORWARD: _ComputationTypeColor("blue", "Forward"),
    _ComputationType.BACKWARD_INPUT: _ComputationTypeColor("teal", "Backward Input"),
    _ComputationType.BACKWARD_WEIGHT: _ComputationTypeColor("green", "Backward Weight"),
    _ComputationType.FULL_BACKWARD: _ComputationTypeColor("orange", "Full Backward", 2),
}


def visualize_schedule(
    schedule: list[list[Optional[_Action]]],
    filename: Optional[str] = None,
    legend: dict[
        _ComputationType, _ComputationTypeColor
    ] = legend,
    base_width: float = 1,
) -> tuple:
    """
    Visualize the schedule using matplotlib.
    The schedule is a list of lists where each inner list represents a rank and each element in the inner list represents an action.
    The actions are represented as rectangles with different colors based on their computation type.
    The filename is optional and if provided, the plot will be saved to that file.
    """

    plt.rcParams["font.family"] = (
        "DejaVu Sans"  # or any other font available on your system
    )
    num_ranks = len(schedule)
    max_actions = max(len(rank) for rank in schedule)

    # Increase the figure size to provide more space for the legend
    _, ax = plt.subplots(figsize=(max_actions + 2, num_ranks + 2))
    max_draw_position = -1
    # Calculate dynamic font size based on figure size
    font_size = min(max_actions, num_ranks) + 4
    used_computation = set()
    for rank_idx, actions in enumerate(schedule):
        draw_position = 0  # Initialize drawing position for each rank
        for action in actions:
            if action is not None:
                comp_type_color = legend.get(
                    action.computation_type, _ComputationTypeColor("black")
                )
                used_computation.add(action.computation_type)
                color = comp_type_color.color
                width = comp_type_color.width
                total_width = width * base_width
                # Draw the rectangle to represent the action duration
                rect = Rectangle(
                    (draw_position, num_ranks - rank_idx - 1),
                    total_width,
                    1,
                    facecolor=color,
                    edgecolor="black",
                )
                ax.add_patch(rect)
                # Draw the text centered within the rectangle
                ax.text(
                    draw_position + total_width / 2,
                    num_ranks - rank_idx - 1 + 0.5,
                    str(action),
                    ha="center",
                    va="center",
                    fontsize=font_size,
                    color="white",
                )
                # Increment the drawing position by the width of the current action
                draw_position += total_width
            else:
                draw_position += base_width  # Move to the next
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
            facecolor=legend[comp_type].color,
            edgecolor="black",
            label=legend[comp_type].text,
        )
        for comp_type in used_computation
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=font_size)
    # Save to file if filename is provided, otherwise display the plot
    if filename:
        plt.savefig(filename, bbox_inches="tight")
    else:
        plt.show()

    return ax

def plot_two_schedules(ax1, ax2, legend, title1: str = "Schedule 1", title2: str = "Schedule 2", filename: Optional[str] = None):
    # Create a new figure to combine the two plots
    fig, (ax_combined1, ax_combined2) = plt.subplots(2, 1, sharex=True, figsize=(12, 10))
    # Copy the patches from the first axis to the combined axis
    for patch in ax1.patches:
        new_patch = Rectangle(
            patch.get_xy(),
            patch.get_width(),
            patch.get_height(),
            facecolor=patch.get_facecolor(),
            edgecolor=patch.get_edgecolor()
        )
        ax_combined1.add_patch(new_patch)
    ax_combined1.set_ylim(ax1.get_ylim())
    ax_combined1.set_yticks(ax1.get_yticks())
    ax_combined1.set_yticklabels(ax1.get_yticklabels())
    ax_combined1.set_title(title1)
    ax_combined1.set_xticks([])  # Remove x-axis labels
    ax_combined1.grid(False)  # Remove grid marks

    # Copy the patches from the second axis to the combined axis
    for patch in ax2.patches:
        new_patch = Rectangle(
            patch.get_xy(),
            patch.get_width(),
            patch.get_height(),
            facecolor=patch.get_facecolor(),
            edgecolor=patch.get_edgecolor()
        )
        ax_combined2.add_patch(new_patch)
    ax_combined2.set_ylim(ax2.get_ylim())
    ax_combined2.set_yticks(ax2.get_yticks())
    ax_combined2.set_yticklabels(ax2.get_yticklabels())
    ax_combined2.set_title(title2)
    ax_combined2.set_xticks([])  # Remove x-axis labels
    ax_combined2.grid(False)  # Remove grid marks

    # Determine the maximum x-limit from both axes
    xlim1 = ax1.get_xlim()
    xlim2 = ax2.get_xlim()
    max_xlim = max(xlim1[1], xlim2[1])
    # Set the x-limits to the maximum value
    ax_combined1.set_xlim(0, max_xlim)
    ax_combined2.set_xlim(0, max_xlim)

    # Combine legends from both axes
    handles1 = ax1.get_legend().legend_handles
    labels1 = ax1.get_legend().get_texts()
    handles2 = ax2.get_legend().legend_handles
    labels2 = ax2.get_legend().get_texts()
    combined_handles = handles1 + handles2
    combined_labels = labels1 + labels2
    unique_legend = set()
    legend_elements = [
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor=legend[comp_type].color,
            edgecolor="black",
            label=legend[comp_type].text,
        )
        for comp_type in (_ComputationType.FORWARD, _ComputationType.FULL_BACKWARD)
    ]
    fig.legend(handles=legend_elements, loc="upper right")

    legend_elements = [
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor=legend[comp_type].color,
            edgecolor="black",
            label=legend[comp_type].text,
        )
        for comp_type in (_ComputationType.FORWARD, _ComputationType.BACKWARD_INPUT, _ComputationType.BACKWARD_WEIGHT)
    ]
    fig.legend(handles=legend_elements, loc="lower right")

    # Adjust the layout to prevent overlap
    plt.tight_layout()
    # Save to file if filename is provided, otherwise display the plot
    if filename:
        plt.savefig(filename, bbox_inches="tight")
    else:
        plt.show()


def remove_none_ops(schedule: list[list[Optional[_Action]]]) -> list[list[Optional[_Action]]]:
    """
    Only remove none ops after the last forward
    """
    new_schedule = copy.deepcopy(schedule)
    for rank in range(len(new_schedule)):
        # Iterate in reverse to find the last Forward action
        for i in range(len(new_schedule[rank]) - 1, -1, -1):
            action = new_schedule[rank][i]
            if action is not None and action.computation_type == _ComputationType.FORWARD:
                break
            if action is None:
                new_schedule[rank].pop(i)
    return new_schedule

if __name__ == "__main__":
    # single microbatch forward and backward schedule
    model_parallel = [
        [
            _Action(0, _ComputationType.FORWARD, 0),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            _Action(0, _ComputationType.FULL_BACKWARD, 0),
        ],
        [
            None,
            _Action(1, _ComputationType.FORWARD, 0),
            None,
            None,
            None,
            None,
            None,
            None,
            _Action(1, _ComputationType.FULL_BACKWARD, 0),
            None,
        ],
        [
            None,
            None,
            _Action(2, _ComputationType.FORWARD, 0),
            None,
            None,
            None,
            _Action(2, _ComputationType.FULL_BACKWARD, 0),
            None,
            None,
        ],
        [
            None,
            None,
            None,
            _Action(3, _ComputationType.FORWARD, 0),
            _Action(3, _ComputationType.FULL_BACKWARD, 0),
            None,
            None,
            None,
        ],
    ]


    # Example usage:
    ops1 = get_schedule_ops("interleaved1f1b", 4, 8)
    ops2 = get_schedule_ops("InterleavedZeroBubble", 4, 8)

    # remove all None ops after last forward 


    ax1 = visualize_schedule(ops1)
    ax2 = visualize_schedule(ops2)
    plot_two_schedules(ax1, ax2, legend, "Interleaved 1F1B", "Interleaved Zero Bubble", filename="compare2.png")
