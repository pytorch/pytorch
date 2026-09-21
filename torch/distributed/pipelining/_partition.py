from __future__ import annotations

import heapq
import math
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

from torch.distributed.pipelining._utils import (
    generate_rank_to_stage_mapping,
    generate_stage_to_rank_mapping,
)
from torch.distributed.pipelining.schedules import (
    _Action,
    _add_send_recv,
    _ComputationType,
    _get_1f1b_rank_ops,
    _get_warmup_ops,
    _simulate_comms_compute,
)

# This file is intentionally limited to the pieces needed for partition search
# and schedule review:
# - schedule-aware iteration-time simulation,
# - critical-path recovery,
# - compute-only text visualization,
# - DP baseline partitioning,
# - heuristic refinement on top of that baseline.
#
# The implementation is organized in the same reading order:
# 1. types
# 2. validation / normalization
# 3. partition construction
# 4. steady-phase analysis
# 5. heuristic search
# 6. schedule generation
# 7. timing DAG simulation
# 8. text visualization
# 9. public entry points

# Keep the public surface small and explicit so downstream scripts can tell at
# a glance which helpers are stable debugging / review entry points.
__all__ = [
    "auto_partition",
    "CriticalPathOperation",
    "PipelinePartitionPlan",
    "PipelinePartitionSearchResult",
    "PipelineScheduleType",
    "PipelineSimulationResult",
    "SteadyCriticalStageInfo",
    "aggregate_layer_values_by_partition",
    "generate_compute_schedule",
    "identify_steady_critical_stage",
    "optimize_partition_model_layers",
    "partition_model_layers",
    "render_compute_schedule_timeline",
    "render_rank_compute_schedule_timeline",
    "simulate_pipeline_schedule",
]


class PipelineScheduleType(str, Enum):
    ONE_F_ONE_B = "1F1B"
    INTERLEAVED_1F1B = "Interleaved1F1B"
    INTERLEAVED_ZERO_BUBBLE = "InterleavedZeroBubble"


@dataclass(frozen=True)
class CriticalPathOperation:
    rank: int
    stage: int
    microbatch: int
    op_type: str
    duration: float
    start_time: float
    end_time: float


@dataclass(frozen=True)
class PipelineSimulationResult:
    iteration_time: float
    critical_path: list[CriticalPathOperation]
    all_operations: list[CriticalPathOperation] = field(default_factory=list)


@dataclass(frozen=True)
class PipelinePartitionPlan:
    # The schedule decides how many logical pipeline stages are needed for a
    # fixed number of physical pipeline ranks.  For example:
    # - 1F1B uses one stage per rank.
    # - Interleaved schedules typically use multiple local stages per rank.
    schedule_type: str
    pp_group_size: int
    n_microbatches: int
    num_stages: int
    virtual_stages_per_rank: int

    # Contiguous layer assignment for each logical pipeline stage.
    stage_partitions: list[list[int]]

    # Logical stage placement used by the schedule runtime/simulator.
    stage_to_rank: list[int]
    rank_to_stages: dict[int, list[int]]

    # Aggregated stage-level inputs that can be fed directly into the schedule
    # simulator.
    stage_forward_flops: list[int]
    stage_backward_flops: list[int]
    stage_communication_volume: list[int]

    # The current partition objective only models compute time.  These values
    # make the chosen balance explicit for review.
    stage_compute_times: list[float]
    objective_value: float


@dataclass(frozen=True)
class PipelinePartitionSearchResult:
    base_plan: PipelinePartitionPlan
    best_plan: PipelinePartitionPlan
    base_iteration_time: float
    best_iteration_time: float
    best_critical_stage: int
    explored_candidates: int


@dataclass(frozen=True)
class SteadyCriticalStageInfo:
    # Inclusive timestep range of steady phase in the aligned schedule.
    steady_start_step: int
    steady_end_step: int

    # Accumulated critical-path duration on each stage within steady phase.
    stage_durations: dict[int, float]
    critical_stage: int


@dataclass(frozen=True)
class _ScheduledAction:
    rank: int
    action: _Action


@dataclass(frozen=True)
class _PartitionSearchState:
    plan: PipelinePartitionPlan
    result: PipelineSimulationResult
    critical_stage: int


ScheduleByRank = Dict[int, List[Optional[_Action]]]
ScheduleWithComms = Dict[int, List[_Action]]
ActionSignature = tuple[int, str, int]


# ---------------------------------------------------------------------------
# Input validation and hardware-aware normalization
# ---------------------------------------------------------------------------

def _ensure_non_negative_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got {type(value)}.")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}.")


def _ensure_positive_number(value: int | float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a positive number, got {type(value)}.")
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}.")


def _normalize_schedule_type(schedule_type: str | PipelineScheduleType) -> PipelineScheduleType:
    if isinstance(schedule_type, PipelineScheduleType):
        return schedule_type
    value = schedule_type.strip().lower()
    if value == "1f1b":
        return PipelineScheduleType.ONE_F_ONE_B
    if value == "interleaved1f1b":
        return PipelineScheduleType.INTERLEAVED_1F1B
    if value == "interleavedzerobubble":
        return PipelineScheduleType.INTERLEAVED_ZERO_BUBBLE
    raise ValueError(
        f"Unsupported schedule type '{schedule_type}'. "
        "Supported values: 1F1B, Interleaved1F1B, InterleavedZeroBubble."
    )


def _normalize_comm_times(
    communication_volume: int | list[int],
    num_stages: int,
    network_bandwidth: int | float,
) -> list[float]:
    # Communication time uses a simple bandwidth model:
    #   time_seconds = bytes / bytes_per_second
    _ensure_positive_number(network_bandwidth, "network_bandwidth")
    if num_stages <= 1:
        return []
    if isinstance(communication_volume, int):
        _ensure_non_negative_int(communication_volume, "communication_volume")
        return [float(communication_volume) / float(network_bandwidth)] * (num_stages - 1)

    if not isinstance(communication_volume, list):
        raise TypeError(
            "communication_volume must be an int or list[int], "
            f"got {type(communication_volume)}."
        )
    for idx, vol in enumerate(communication_volume):
        _ensure_non_negative_int(vol, f"communication_volume[{idx}]")
    comm = [float(t) / float(network_bandwidth) for t in communication_volume]
    if len(comm) != num_stages - 1:
        raise ValueError(
            "communication_volume must have length num_stages - 1, "
            f"but got {len(comm)} and num_stages={num_stages}."
        )
    return comm


def _normalize_compute_times(
    flops_by_stage: list[int] | list[float],
    device_compute_throughput: int | float,
) -> list[float]:
    # Compute time uses the same explicit hardware model:
    #   time_seconds = flops / flops_per_second
    _ensure_positive_number(device_compute_throughput, "device_compute_throughput")
    return [float(flops) / float(device_compute_throughput) for flops in flops_by_stage]


def _normalize_backward_split(
    backward_flops: list[int],
    backward_input_flops: list[int] | None,
    backward_weight_flops: list[int] | None,
) -> tuple[list[float], list[float]]:
    num_stages = len(backward_flops)
    if backward_input_flops is None and backward_weight_flops is None:
        # If no explicit split is provided, use an integer split:
        # I = floor(B/2), W = B - I.
        backward_input = [float(t // 2) for t in backward_flops]
        backward_weight = [float(t - int(bi)) for t, bi in zip(backward_flops, backward_input)]
        return backward_input, backward_weight

    if backward_input_flops is not None and len(backward_input_flops) != num_stages:
        raise ValueError(
            "backward_input_flops must have the same length as backward_flops, "
            f"but got {len(backward_input_flops)} and {num_stages}."
        )
    if backward_weight_flops is not None and len(backward_weight_flops) != num_stages:
        raise ValueError(
            "backward_weight_flops must have the same length as backward_flops, "
            f"but got {len(backward_weight_flops)} and {num_stages}."
        )

    if backward_input_flops is not None:
        for idx, flops in enumerate(backward_input_flops):
            _ensure_non_negative_int(flops, f"backward_input_flops[{idx}]")
    if backward_weight_flops is not None:
        for idx, flops in enumerate(backward_weight_flops):
            _ensure_non_negative_int(flops, f"backward_weight_flops[{idx}]")

    if backward_input_flops is None and backward_weight_flops is not None:
        backward_weight = [float(t) for t in backward_weight_flops]
        backward_input = [float(t - int(w)) for t, w in zip(backward_flops, backward_weight)]
    elif backward_weight_flops is None and backward_input_flops is not None:
        backward_input = [float(t) for t in backward_input_flops]
        backward_weight = [float(t - int(i)) for t, i in zip(backward_flops, backward_input)]
    else:
        backward_input = [float(t) for t in backward_input_flops or []]
        backward_weight = [float(t) for t in backward_weight_flops or []]

    for idx, (bi, bw) in enumerate(zip(backward_input, backward_weight)):
        if bi < 0 or bw < 0:
            raise ValueError(
                "Backward split FLOPs must be non-negative, "
                f"but stage {idx} has backward_input_flops={bi}, backward_weight_flops={bw}."
            )
    return backward_input, backward_weight


def _validate_partition_inputs(
    forward_flops: list[int],
    backward_flops: list[int],
    communication_volume: int | list[int],
    *,
    pp_group_size: int,
    device_compute_throughput: int | float,
    network_bandwidth: int | float,
    n_microbatches: int,
) -> None:
    if len(forward_flops) == 0 or len(backward_flops) == 0:
        raise ValueError("forward_flops and backward_flops cannot be empty.")
    if len(forward_flops) != len(backward_flops):
        raise ValueError(
            "forward_flops and backward_flops must have the same length, "
            f"but got {len(forward_flops)} and {len(backward_flops)}."
        )
    for idx, flops in enumerate(forward_flops):
        _ensure_non_negative_int(flops, f"forward_flops[{idx}]")
    for idx, flops in enumerate(backward_flops):
        _ensure_non_negative_int(flops, f"backward_flops[{idx}]")

    if isinstance(communication_volume, list):
        if len(communication_volume) != len(forward_flops) - 1:
            raise ValueError(
                "communication_volume must have length len(forward_flops) - 1, "
                f"but got {len(communication_volume)} and len(forward_flops)={len(forward_flops)}."
            )
        for idx, volume in enumerate(communication_volume):
            _ensure_non_negative_int(volume, f"communication_volume[{idx}]")
    else:
        _ensure_non_negative_int(communication_volume, "communication_volume")

    if n_microbatches <= 0:
        raise ValueError(f"n_microbatches must be positive, got {n_microbatches}.")
    if pp_group_size <= 0 or pp_group_size > len(forward_flops):
        raise ValueError(
            f"pp_group_size must be in [1, {len(forward_flops)}], but got {pp_group_size}."
        )
    _ensure_positive_number(device_compute_throughput, "device_compute_throughput")
    _ensure_positive_number(network_bandwidth, "network_bandwidth")


# ---------------------------------------------------------------------------
# DP baseline partitioning
# ---------------------------------------------------------------------------

def _default_virtual_stages_per_rank(
    schedule_type: PipelineScheduleType,
    *,
    num_layers: int,
    pp_group_size: int,
) -> int:
    if schedule_type == PipelineScheduleType.ONE_F_ONE_B:
        return 1

    # For interleaved schedules, two local stages per rank are the smallest
    # configuration that exposes actual interleaving.  If the model is too
    # small for that, gracefully fall back to one stage per rank.
    if num_layers >= 2 * pp_group_size:
        return 2
    return 1


def _compute_partition_stage_count(
    schedule_type: PipelineScheduleType,
    *,
    num_layers: int,
    pp_group_size: int,
    virtual_stages_per_rank: int | None,
) -> tuple[int, int]:
    local_stage_count = (
        _default_virtual_stages_per_rank(
            schedule_type,
            num_layers=num_layers,
            pp_group_size=pp_group_size,
        )
        if virtual_stages_per_rank is None
        else virtual_stages_per_rank
    )
    if local_stage_count <= 0:
        raise ValueError(
            "virtual_stages_per_rank must be positive, "
            f"but got {local_stage_count}."
        )
    if schedule_type == PipelineScheduleType.ONE_F_ONE_B and local_stage_count != 1:
        raise ValueError(
            "1F1B uses exactly one logical stage per rank, so "
            f"virtual_stages_per_rank must be 1, got {local_stage_count}."
        )

    num_stages = pp_group_size * local_stage_count
    if num_stages > num_layers:
        raise ValueError(
            "The requested logical stage count exceeds the number of model layers: "
            f"num_stages={num_stages}, num_layers={num_layers}."
        )
    if schedule_type != PipelineScheduleType.ONE_F_ONE_B and num_stages % pp_group_size != 0:
        raise ValueError(
            "Interleaved schedules require num_stages to be divisible by pp_group_size, "
            f"but got num_stages={num_stages}, pp_group_size={pp_group_size}."
        )
    return num_stages, local_stage_count


def _prefix_compute_times(compute_times: list[float]) -> list[float]:
    prefix = [0.0]
    for time in compute_times:
        prefix.append(prefix[-1] + time)
    return prefix


def _block_partition_by_compute_time(
    compute_times: list[float],
    num_stages: int,
) -> list[list[int]]:
    num_layers = len(compute_times)
    prefix = _prefix_compute_times(compute_times)

    dp = [[math.inf] * (num_stages + 1) for _ in range(num_layers + 1)]
    split = [[-1] * (num_stages + 1) for _ in range(num_layers + 1)]
    dp[0][0] = 0.0

    for end in range(1, num_layers + 1):
        max_parts = min(end, num_stages)
        for parts in range(1, max_parts + 1):
            best_cost = math.inf
            best_start = -1
            # Leave at least one layer for each previous partition.
            for start in range(parts - 1, end):
                stage_cost = prefix[end] - prefix[start]
                candidate = max(dp[start][parts - 1], stage_cost)
                if candidate < best_cost:
                    best_cost = candidate
                    best_start = start
            dp[end][parts] = best_cost
            split[end][parts] = best_start

    partitions_reversed: list[list[int]] = []
    remaining_layers = num_layers
    remaining_parts = num_stages
    while remaining_parts > 0:
        start = split[remaining_layers][remaining_parts]
        if start < 0:
            raise RuntimeError("Failed to reconstruct the dynamic-programming partition.")
        partitions_reversed.append(list(range(start, remaining_layers)))
        remaining_layers = start
        remaining_parts -= 1

    partitions_reversed.reverse()
    return partitions_reversed


def _aggregate_comm_volume_for_partition(
    communication_volume: int | list[int],
    stage_partitions: list[list[int]],
) -> list[int]:
    if len(stage_partitions) <= 1:
        return []

    if isinstance(communication_volume, int):
        return [communication_volume] * (len(stage_partitions) - 1)

    stage_comm: list[int] = []
    for stage_idx in range(len(stage_partitions) - 1):
        boundary_layer = stage_partitions[stage_idx][-1]
        stage_comm.append(communication_volume[boundary_layer])
    return stage_comm


def aggregate_layer_values_by_partition(
    stage_partitions: list[list[int]],
    values: list[int],
) -> list[int]:
    # Collapse layer-level metadata into stage-level metadata after a contiguous
    # partition has been chosen.
    if len(values) == 0 and len(stage_partitions) == 0:
        return []
    aggregated: list[int] = []
    for stage_idx, stage_layers in enumerate(stage_partitions):
        if len(stage_layers) == 0:
            raise ValueError(f"stage_partitions[{stage_idx}] cannot be empty.")
        stage_total = 0
        for layer in stage_layers:
            if layer < 0 or layer >= len(values):
                raise ValueError(
                    f"stage_partitions[{stage_idx}] contains out-of-range layer index {layer}."
                )
            stage_total += values[layer]
        aggregated.append(stage_total)
    return aggregated


def _build_partition_plan_from_stage_partitions(
    schedule_type: PipelineScheduleType,
    *,
    stage_partitions: list[list[int]],
    forward_flops: list[int],
    backward_flops: list[int],
    communication_volume: int | list[int],
    device_compute_throughput: int | float,
    pp_group_size: int,
    n_microbatches: int,
    virtual_stages_per_rank: int,
) -> PipelinePartitionPlan:
    stage_to_rank_mapping = generate_stage_to_rank_mapping(
        pp_size=pp_group_size,
        num_stages=len(stage_partitions),
        style="loop",
    )
    stage_to_rank = [stage_to_rank_mapping[stage_idx] for stage_idx in range(len(stage_partitions))]
    rank_to_stages = generate_rank_to_stage_mapping(
        pp_size=pp_group_size,
        num_stages=len(stage_partitions),
        style="loop",
    )

    forward_times = _normalize_compute_times(forward_flops, device_compute_throughput)
    backward_times = _normalize_compute_times(backward_flops, device_compute_throughput)
    layer_compute_times = [
        fwd_time + bwd_time for fwd_time, bwd_time in zip(forward_times, backward_times)
    ]

    stage_forward_flops = aggregate_layer_values_by_partition(stage_partitions, forward_flops)
    stage_backward_flops = aggregate_layer_values_by_partition(stage_partitions, backward_flops)
    stage_compute_times = [
        sum(layer_compute_times[layer] for layer in stage_layers)
        for stage_layers in stage_partitions
    ]
    stage_communication_volume = _aggregate_comm_volume_for_partition(
        communication_volume,
        stage_partitions,
    )

    return PipelinePartitionPlan(
        schedule_type=schedule_type.value,
        pp_group_size=pp_group_size,
        n_microbatches=n_microbatches,
        num_stages=len(stage_partitions),
        virtual_stages_per_rank=virtual_stages_per_rank,
        stage_partitions=stage_partitions,
        stage_to_rank=stage_to_rank,
        rank_to_stages=rank_to_stages,
        stage_forward_flops=stage_forward_flops,
        stage_backward_flops=stage_backward_flops,
        stage_communication_volume=stage_communication_volume,
        stage_compute_times=stage_compute_times,
        objective_value=max(stage_compute_times, default=0.0),
    )


def _aggregate_optional_partition_values(
    stage_partitions: list[list[int]],
    values: list[int] | None,
) -> list[int] | None:
    if values is None:
        return None
    return aggregate_layer_values_by_partition(stage_partitions, values)


def partition_model_layers(
    schedule_type: str | PipelineScheduleType,
    forward_flops: list[int],
    backward_flops: list[int],
    communication_volume: int | list[int],
    device_compute_throughput: int | float,
    network_bandwidth: int | float,
    pp_group_size: int,
    n_microbatches: int,
    *,
    virtual_stages_per_rank: int | None = None,
) -> PipelinePartitionPlan:
    # Public DP baseline:
    # choose contiguous stage partitions by balancing compute time only.  The
    # returned plan already contains stage-level inputs for the simulator.
    # Current partition objective:
    # - convert each layer's forward/backward FLOPs into compute time
    # - ignore communication when choosing the partition
    # - minimize the slowest logical stage by a contiguous block DP
    #
    # `communication_volume`, `network_bandwidth`, and `n_microbatches` are
    # still accepted and validated so this interface already matches the inputs
    # needed by the full simulator and can be extended later to communication-
    # aware partitioning without changing call sites.
    schedule_kind = _normalize_schedule_type(schedule_type)
    _validate_partition_inputs(
        forward_flops,
        backward_flops,
        communication_volume,
        pp_group_size=pp_group_size,
        device_compute_throughput=device_compute_throughput,
        network_bandwidth=network_bandwidth,
        n_microbatches=n_microbatches,
    )

    num_stages, local_stage_count = _compute_partition_stage_count(
        schedule_kind,
        num_layers=len(forward_flops),
        pp_group_size=pp_group_size,
        virtual_stages_per_rank=virtual_stages_per_rank,
    )

    forward_times = _normalize_compute_times(forward_flops, device_compute_throughput)
    backward_times = _normalize_compute_times(backward_flops, device_compute_throughput)
    layer_compute_times = [
        fwd_time + bwd_time for fwd_time, bwd_time in zip(forward_times, backward_times)
    ]

    stage_partitions = _block_partition_by_compute_time(layer_compute_times, num_stages)
    return _build_partition_plan_from_stage_partitions(
        schedule_kind,
        stage_partitions=stage_partitions,
        forward_flops=forward_flops,
        backward_flops=backward_flops,
        communication_volume=communication_volume,
        device_compute_throughput=device_compute_throughput,
        pp_group_size=pp_group_size,
        n_microbatches=n_microbatches,
        virtual_stages_per_rank=local_stage_count,
    )


# ---------------------------------------------------------------------------
# Steady-phase critical-stage analysis
# ---------------------------------------------------------------------------

def _partition_signature(stage_partitions: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(stage_layers) for stage_layers in stage_partitions)


def _action_signature(op: CriticalPathOperation) -> ActionSignature:
    return (op.stage, op.op_type, op.microbatch)


def _compute_steady_phase_steps(
    schedule_type: PipelineScheduleType,
    *,
    num_stages: int,
    pp_group_size: int,
    n_microbatches: int,
) -> tuple[int, int]:
    compute_schedule = _build_compute_schedule(
        schedule_type,
        num_stages=num_stages,
        pp_group_size=pp_group_size,
        n_microbatches=n_microbatches,
    )
    stage_to_rank = lambda stage: stage % pp_group_size
    simulated_communication_schedule = _simulate_comms_compute(
        _add_send_recv(compute_schedule, stage_to_rank, num_stages),
        stage_to_rank,
        num_stages,
    )

    first_backward_step: int | None = None
    last_forward_step: int | None = None
    total_steps = max((len(actions) for actions in simulated_communication_schedule.values()), default=0)

    for step in range(total_steps):
        for rank in sorted(simulated_communication_schedule):
            action = (
                simulated_communication_schedule[rank][step]
                if step < len(simulated_communication_schedule[rank])
                else None
            )
            if action is None:
                continue
            if (
                first_backward_step is None
                and action.computation_type
                in (_ComputationType.FULL_BACKWARD, _ComputationType.BACKWARD_INPUT)
            ):
                first_backward_step = step
            if action.computation_type == _ComputationType.FORWARD:
                last_forward_step = step

    if first_backward_step is None or last_forward_step is None:
        return (0, max(0, total_steps - 1))
    if first_backward_step > last_forward_step:
        return (first_backward_step, first_backward_step)
    return first_backward_step, last_forward_step


def _build_action_step_index(
    schedule_type: PipelineScheduleType,
    *,
    num_stages: int,
    pp_group_size: int,
    n_microbatches: int,
) -> dict[ActionSignature, int]:
    compute_schedule = _build_compute_schedule(
        schedule_type,
        num_stages=num_stages,
        pp_group_size=pp_group_size,
        n_microbatches=n_microbatches,
    )
    stage_to_rank = lambda stage: stage % pp_group_size
    simulated_communication_schedule = _simulate_comms_compute(
        _add_send_recv(compute_schedule, stage_to_rank, num_stages),
        stage_to_rank,
        num_stages,
    )

    action_to_step: dict[ActionSignature, int] = {}
    total_steps = max((len(actions) for actions in simulated_communication_schedule.values()), default=0)
    for step in range(total_steps):
        for rank in sorted(simulated_communication_schedule):
            action = (
                simulated_communication_schedule[rank][step]
                if step < len(simulated_communication_schedule[rank])
                else None
            )
            if action is None:
                continue
            action_to_step[(action.stage_index, action.computation_type.value, action.microbatch_index)] = step
    return action_to_step


def identify_steady_critical_stage(
    schedule_type: str | PipelineScheduleType,
    plan: PipelinePartitionPlan,
    result: PipelineSimulationResult,
) -> SteadyCriticalStageInfo:
    # Map the operation-level critical path back onto the aligned discrete
    # schedule, restrict the accounting to steady phase, and report which stage
    # contributes the most critical-path time there.
    schedule_kind = _normalize_schedule_type(schedule_type)
    steady_start, steady_end = _compute_steady_phase_steps(
        schedule_kind,
        num_stages=plan.num_stages,
        pp_group_size=plan.pp_group_size,
        n_microbatches=plan.n_microbatches,
    )
    action_to_step = _build_action_step_index(
        schedule_kind,
        num_stages=plan.num_stages,
        pp_group_size=plan.pp_group_size,
        n_microbatches=plan.n_microbatches,
    )

    stage_durations: dict[int, float] = defaultdict(float)
    for op in result.critical_path:
        step = action_to_step.get(_action_signature(op))
        if step is None:
            continue
        if steady_start <= step <= steady_end:
            stage_durations[op.stage] += op.duration

    if not stage_durations:
        for op in result.critical_path:
            stage_durations[op.stage] += op.duration

    critical_stage = max(sorted(stage_durations), key=lambda stage: stage_durations[stage])
    return SteadyCriticalStageInfo(
        steady_start_step=steady_start,
        steady_end_step=steady_end,
        stage_durations=dict(stage_durations),
        critical_stage=critical_stage,
    )


def _find_critical_stage_in_steady_phase(
    schedule_type: PipelineScheduleType,
    plan: PipelinePartitionPlan,
    result: PipelineSimulationResult,
) -> int:
    return identify_steady_critical_stage(
        schedule_type,
        plan,
        result,
    ).critical_stage


# ---------------------------------------------------------------------------
# Heuristic search around the DP baseline
# ---------------------------------------------------------------------------

def _simulate_partition_plan(
    schedule_type: PipelineScheduleType,
    plan: PipelinePartitionPlan,
    *,
    device_compute_throughput: int | float,
    network_bandwidth: int | float,
    backward_input_flops: list[int] | None,
    backward_weight_flops: list[int] | None,
) -> PipelineSimulationResult:
    stage_backward_input = _aggregate_optional_partition_values(
        plan.stage_partitions,
        backward_input_flops,
    )
    stage_backward_weight = _aggregate_optional_partition_values(
        plan.stage_partitions,
        backward_weight_flops,
    )
    return simulate_pipeline_schedule(
        schedule_type,
        plan.stage_forward_flops,
        plan.stage_backward_flops,
        plan.stage_communication_volume,
        plan.n_microbatches,
        pp_group_size=plan.pp_group_size,
        backward_input_flops=stage_backward_input,
        backward_weight_flops=stage_backward_weight,
        device_compute_throughput=device_compute_throughput,
        network_bandwidth=network_bandwidth,
    )


def _compute_total_layer_compute_times(
    forward_flops: list[int],
    backward_flops: list[int],
    *,
    device_compute_throughput: int | float,
) -> list[float]:
    # The partition search currently balances only compute time, so both the
    # DP baseline and the heuristic refinement use the same per-layer cost:
    #   layer_cost = forward_time + backward_time
    forward_times = _normalize_compute_times(forward_flops, device_compute_throughput)
    backward_times = _normalize_compute_times(backward_flops, device_compute_throughput)
    return [
        forward_time + backward_time
        for forward_time, backward_time in zip(forward_times, backward_times)
    ]


def _repartition_layer_range(
    layer_compute_times: list[float],
    *,
    start_layer: int,
    end_layer: int,
    num_stages: int,
) -> list[list[int]]:
    # Re-run the same contiguous DP partition, but only on a prefix window of
    # layers.  Returned layer ids stay in the original global numbering so the
    # result can be stitched back into a full pipeline partition directly.
    if num_stages <= 0:
        raise ValueError(f"num_stages must be positive, got {num_stages}.")
    local_compute_times = layer_compute_times[start_layer:end_layer]
    local_partitions = _block_partition_by_compute_time(local_compute_times, num_stages)
    return [
        [start_layer + local_layer for local_layer in stage_layers]
        for stage_layers in local_partitions
    ]


def _build_left_rebalanced_candidate(
    stage_partitions: list[list[int]],
    critical_stage: int,
    layer_compute_times: list[float],
) -> list[list[int]] | None:
    # Move the first layer of the critical stage to the previous stage, then
    # re-balance the whole prefix ending at that moved layer with the DP
    # partitioner.
    if critical_stage <= 0 or critical_stage >= len(stage_partitions):
        return None

    critical_layers = stage_partitions[critical_stage]
    if len(critical_layers) <= 1:
        return None

    moved_layer = critical_layers[0]
    prefix_partitions = _repartition_layer_range(
        layer_compute_times,
        start_layer=0,
        end_layer=moved_layer + 1,
        num_stages=critical_stage,
    )
    suffix_partitions = [list(stage_layers) for stage_layers in stage_partitions[critical_stage:]]
    suffix_partitions[0] = suffix_partitions[0][1:]
    return prefix_partitions + suffix_partitions


def _build_right_rebalanced_candidate(
    stage_partitions: list[list[int]],
    critical_stage: int,
    layer_compute_times: list[float],
) -> list[list[int]] | None:
    # Move the last layer of the critical stage to the next stage, then
    # re-balance the prefix ending at the remaining critical-stage boundary.
    if critical_stage < 0 or critical_stage >= len(stage_partitions) - 1:
        return None

    critical_layers = stage_partitions[critical_stage]
    if len(critical_layers) <= 1:
        return None

    moved_layer = critical_layers[-1]
    prefix_partitions = _repartition_layer_range(
        layer_compute_times,
        start_layer=0,
        end_layer=moved_layer,
        num_stages=critical_stage + 1,
    )
    suffix_partitions = [list(stage_layers) for stage_layers in stage_partitions[critical_stage + 1 :]]
    suffix_partitions[0] = [moved_layer] + suffix_partitions[0]
    return prefix_partitions + suffix_partitions


def _generate_rebalanced_candidates(
    stage_partitions: list[list[int]],
    critical_stage: int,
    layer_compute_times: list[float],
) -> list[list[list[int]]]:
    # The heuristic always tries the same two local edits around the current
    # critical stage:
    # 1. move the first layer left and rebalance the left prefix,
    # 2. move the last layer right and rebalance the left prefix.
    candidates: list[list[list[int]]] = []

    left_candidate = _build_left_rebalanced_candidate(
        stage_partitions,
        critical_stage,
        layer_compute_times,
    )
    if left_candidate is not None:
        candidates.append(left_candidate)

    right_candidate = _build_right_rebalanced_candidate(
        stage_partitions,
        critical_stage,
        layer_compute_times,
    )
    if right_candidate is not None:
        candidates.append(right_candidate)

    return candidates


def _evaluate_search_candidate(
    schedule_type: PipelineScheduleType,
    plan: PipelinePartitionPlan,
    *,
    device_compute_throughput: int | float,
    network_bandwidth: int | float,
    backward_input_flops: list[int] | None,
    backward_weight_flops: list[int] | None,
) -> _PartitionSearchState:
    # Bundle the two pieces of information the search needs for every visited
    # partition candidate:
    # - simulated iteration time / critical path,
    # - steady-phase critical stage derived from that simulation result.
    result = _simulate_partition_plan(
        schedule_type,
        plan,
        device_compute_throughput=device_compute_throughput,
        network_bandwidth=network_bandwidth,
        backward_input_flops=backward_input_flops,
        backward_weight_flops=backward_weight_flops,
    )
    critical_stage = _find_critical_stage_in_steady_phase(
        schedule_type,
        plan,
        result,
    )
    return _PartitionSearchState(
        plan=plan,
        result=result,
        critical_stage=critical_stage,
    )


def optimize_partition_model_layers(
    schedule_type: str | PipelineScheduleType,
    forward_flops: list[int],
    backward_flops: list[int],
    communication_volume: int | list[int],
    device_compute_throughput: int | float,
    network_bandwidth: int | float,
    pp_group_size: int,
    n_microbatches: int,
    *,
    virtual_stages_per_rank: int | None = None,
    backward_input_flops: list[int] | None = None,
    backward_weight_flops: list[int] | None = None,
) -> PipelinePartitionSearchResult:
    # Public heuristic search:
    # start from the DP baseline, simulate it, identify the steady-phase
    # critical stage, and try to lighten that stage by moving one boundary
    # layer to an adjacent stage.  After each move, re-run the DP partitioner
    # on the affected prefix so the candidate keeps the "contiguous and
    # compute-balanced" property on the side that was changed.
    # Search procedure:
    # 1. Build the DP-based baseline partition that balances stage compute time.
    # 2. Simulate it with the selected schedule to recover the iteration time
    #    and one critical path.
    # 3. If the critical stage is not the first stage, build up to two new
    #    candidates:
    #    - move the first layer left, then re-DP the left prefix;
    #    - move the last layer right, then re-DP the left prefix ending at the
    #      new critical-stage boundary.
    # 4. Simulate each candidate and continue exploring only when the new
    #    critical stage does not move to the right of the current one.
    # 5. Return the best iteration-time plan seen during the whole search.
    schedule_kind = _normalize_schedule_type(schedule_type)
    _validate_partition_inputs(
        forward_flops,
        backward_flops,
        communication_volume,
        pp_group_size=pp_group_size,
        device_compute_throughput=device_compute_throughput,
        network_bandwidth=network_bandwidth,
        n_microbatches=n_microbatches,
    )
    layer_compute_times = _compute_total_layer_compute_times(
        forward_flops,
        backward_flops,
        device_compute_throughput=device_compute_throughput,
    )
    base_plan = partition_model_layers(
        schedule_kind,
        forward_flops,
        backward_flops,
        communication_volume,
        device_compute_throughput,
        network_bandwidth,
        pp_group_size,
        n_microbatches,
        virtual_stages_per_rank=virtual_stages_per_rank,
    )
    base_state = _evaluate_search_candidate(
        schedule_kind,
        base_plan,
        device_compute_throughput=device_compute_throughput,
        network_bandwidth=network_bandwidth,
        backward_input_flops=backward_input_flops,
        backward_weight_flops=backward_weight_flops,
    )

    best_plan = base_plan
    best_result = base_state.result
    best_critical_stage = base_state.critical_stage
    explored_candidates = 1

    # Standard BFS-style worklist over candidate partitions:
    # - `visited` prevents revisiting the same contiguous partition,
    # - `queue` contains only candidates whose critical stage did not move to
    #   the right, matching the intended heuristic pruning rule.
    visited = {_partition_signature(base_plan.stage_partitions)}
    queue: list[_PartitionSearchState] = [base_state]

    while queue:
        current = queue.pop(0)
        if current.critical_stage == 0:
            continue

        for neighbor_partitions in _generate_rebalanced_candidates(
            current.plan.stage_partitions,
            current.critical_stage,
            layer_compute_times,
        ):
            signature = _partition_signature(neighbor_partitions)
            if signature in visited:
                continue
            visited.add(signature)

            neighbor_plan = _build_partition_plan_from_stage_partitions(
                schedule_kind,
                stage_partitions=neighbor_partitions,
                forward_flops=forward_flops,
                backward_flops=backward_flops,
                communication_volume=communication_volume,
                device_compute_throughput=device_compute_throughput,
                pp_group_size=pp_group_size,
                n_microbatches=n_microbatches,
                virtual_stages_per_rank=current.plan.virtual_stages_per_rank,
            )
            neighbor_state = _evaluate_search_candidate(
                schedule_kind,
                neighbor_plan,
                device_compute_throughput=device_compute_throughput,
                network_bandwidth=network_bandwidth,
                backward_input_flops=backward_input_flops,
                backward_weight_flops=backward_weight_flops,
            )
            explored_candidates += 1

            if neighbor_state.result.iteration_time < best_result.iteration_time:
                best_plan = neighbor_plan
                best_result = neighbor_state.result
                best_critical_stage = neighbor_state.critical_stage

            if neighbor_state.critical_stage <= current.critical_stage:
                queue.append(neighbor_state)

    return PipelinePartitionSearchResult(
        base_plan=base_plan,
        best_plan=best_plan,
        base_iteration_time=base_state.result.iteration_time,
        best_iteration_time=best_result.iteration_time,
        best_critical_stage=best_critical_stage,
        explored_candidates=explored_candidates,
    )


# ---------------------------------------------------------------------------
# Compute schedule generation
# ---------------------------------------------------------------------------


def _build_1f1b_compute_schedule(
    pp_group_size: int,
    n_microbatches: int,
) -> dict[int, list[_Action | None]]:
    if n_microbatches < pp_group_size:
        raise ValueError(
            "1F1B requires n_microbatches >= pp_group_size, "
            f"but got n_microbatches={n_microbatches}, pp_group_size={pp_group_size}."
        )

    schedule: dict[int, list[_Action | None]] = {}
    for rank in range(pp_group_size):
        # This discrete schedule is the unit-slot view used by the simulator and
        # textual visualizer.  The non-bubble operation order matches the 1F1B
        # runtime:
        # 1. warmup forwards,
        # 2. steady-state alternating forward/backward,
        # 3. cooldown backwards.
        #
        # The inserted `None` slots represent pipeline bubbles while a stage is
        # waiting for the first backward wave to arrive or for cooldown
        # backwards to become ready.
        actions: list[_Action | None] = [None] * rank

        # Before the first backward arrives, rank `r` can inject
        # `pp_group_size - 1 - r` forwards contiguously.  The next warmup
        # forward still belongs to the same warmup set, but it can only happen
        # after the backward wave has moved back through the downstream stages.
        num_forward = max(0, pp_group_size - 1 - rank)
        num_forward = min(num_forward, n_microbatches)
        for mb in range(num_forward):
            actions.append(_Action(rank, _ComputationType.FORWARD, mb))

        actions.extend([None] * max(0, pp_group_size - 1 - rank))

        backward_mb = 0
        forward_mb = num_forward - 1
        remaining_forward = n_microbatches - num_forward
        while remaining_forward > 0:
            forward_mb += 1
            actions.append(_Action(rank, _ComputationType.FORWARD, forward_mb))
            remaining_forward -= 1

            actions.append(_Action(rank, _ComputationType.FULL_BACKWARD, backward_mb))
            backward_mb += 1

        remaining_backward = n_microbatches - backward_mb
        while remaining_backward > 0:
            actions.append(None)
            actions.append(_Action(rank, _ComputationType.FULL_BACKWARD, backward_mb))
            backward_mb += 1
            remaining_backward -= 1

        schedule[rank] = actions
    return schedule


def _build_compute_schedule(
    schedule_type: PipelineScheduleType,
    *,
    num_stages: int,
    pp_group_size: int,
    n_microbatches: int,
) -> ScheduleByRank:
    # Keep schedule generation identical to the logic in schedules.py:
    # - 1F1B uses one stage per rank.
    # - Interleaved schedules use loop-style stage placement.
    if schedule_type == PipelineScheduleType.ONE_F_ONE_B:
        if pp_group_size != num_stages:
            raise ValueError(
                "1F1B simulation assumes one stage per rank, so pp_group_size must equal num_stages. "
                f"Got pp_group_size={pp_group_size}, num_stages={num_stages}."
            )
        return _build_1f1b_compute_schedule(pp_group_size, n_microbatches)
    if schedule_type == PipelineScheduleType.INTERLEAVED_1F1B:
        return _build_interleaved_compute_schedule(
            pp_group_size=pp_group_size,
            num_stages=num_stages,
            n_microbatches=n_microbatches,
            zero_bubble=False,
        )
    return _build_interleaved_compute_schedule(
        pp_group_size=pp_group_size,
        num_stages=num_stages,
        n_microbatches=n_microbatches,
        zero_bubble=True,
    )


def _build_interleaved_compute_schedule(
    pp_group_size: int,
    num_stages: int,
    n_microbatches: int,
    zero_bubble: bool,
) -> dict[int, list[_Action | None]]:
    if pp_group_size <= 0:
        raise ValueError(f"pp_group_size must be positive, got {pp_group_size}.")
    if num_stages % pp_group_size != 0:
        raise ValueError(
            "num_stages must be divisible by pp_group_size for interleaved schedules, "
            f"but got num_stages={num_stages}, pp_group_size={pp_group_size}."
        )
    n_local_stages = num_stages // pp_group_size
    if n_local_stages <= 0:
        raise ValueError(
            "n_local_stages must be positive after partitioning, "
            f"but got n_local_stages={n_local_stages}."
        )

    number_of_rounds = max(1, n_microbatches // pp_group_size)
    microbatches_per_round = n_microbatches // number_of_rounds
    if n_microbatches % number_of_rounds != 0:
        name = "InterleavedZeroBubble" if zero_bubble else "Interleaved1F1B"
        raise ValueError(
            f"{name} requires n_microbatches to be divisible by number_of_rounds "
            f"({number_of_rounds}), but got n_microbatches={n_microbatches}."
        )

    schedule: dict[int, list[_Action | None]] = {}
    for rank in range(pp_group_size):
        warmup_ops = _get_warmup_ops(
            rank=rank,
            n_local_stages=n_local_stages,
            microbatches_per_round=microbatches_per_round,
            pp_group_size=pp_group_size,
            n_microbatches=n_microbatches,
            multiply_factor=1 if zero_bubble else 2,
        )
        microbatch_ops = n_local_stages * n_microbatches
        fwd_bwd_ops = microbatch_ops - warmup_ops
        cooldown_ops = microbatch_ops - fwd_bwd_ops

        def forward_stage_index(step: int) -> int:
            local_index = (step // microbatches_per_round) % n_local_stages
            return (local_index * pp_group_size) + rank

        def backward_stage_index(step: int) -> int:
            local_index = (
                n_local_stages
                - 1
                - ((step - warmup_ops) // microbatches_per_round) % n_local_stages
            )
            return (local_index * pp_group_size) + rank

        schedule[rank] = _get_1f1b_rank_ops(
            n_local_stages=n_local_stages,
            pp_group_size=pp_group_size,
            warmup_ops=warmup_ops,
            fwd_bwd_ops=fwd_bwd_ops,
            cooldown_ops=cooldown_ops,
            rank=rank,
            forward_stage_index=forward_stage_index,
            backward_stage_index=backward_stage_index,
            num_1f1b_microbatches=rank if zero_bubble else 0,
            enable_zero_bubble=zero_bubble,
        )

    if zero_bubble:
        schedule = _add_bubbles_to_zero_bubble_actions(schedule, pp_group_size, num_stages)
    return schedule


def _add_bubbles_to_zero_bubble_actions(
    actions: dict[int, list[_Action | None]],
    pp_group_size: int,
    num_stages_global: int,
) -> dict[int, list[_Action | None]]:
    def need_bubble(
        stage: int,
        op: _ComputationType,
        microbatch: int,
        seen_ops: set[tuple[int, _ComputationType, int]],
    ) -> bool:
        if op == _ComputationType.FORWARD:
            return stage != 0 and (stage - 1, op, microbatch) not in seen_ops
        if op == _ComputationType.FULL_BACKWARD:
            if stage == num_stages_global - 1:
                return (stage, _ComputationType.FORWARD, microbatch) not in seen_ops
            return (stage + 1, op, microbatch) not in seen_ops
        return False

    seen_ops: set[tuple[int, _ComputationType, int]] = set()
    result: dict[int, list[_Action | None]] = {rank: [] for rank in range(pp_group_size)}
    next_pointer: dict[int, int] = {rank: 0 for rank in range(pp_group_size)}

    while True:
        should_stop = True
        temp_seen_ops: set[tuple[int, _ComputationType, int]] = set()

        for rank in range(pp_group_size):
            timestamp = next_pointer[rank]
            if timestamp >= len(actions[rank]):
                continue

            should_stop = False
            action = actions[rank][timestamp]
            if action is None:
                next_pointer[rank] += 1
                result[rank].append(None)
                continue

            stage_index = action.stage_index
            op = action.computation_type
            microbatch = action.microbatch_index
            if not need_bubble(stage_index, op, microbatch, seen_ops):
                result[rank].append(action)
                temp_seen_ops.add((stage_index, op, microbatch))
                next_pointer[rank] += 1
            else:
                result[rank].append(None)

        seen_ops.update(temp_seen_ops)
        if should_stop:
            break
    return result


# ---------------------------------------------------------------------------
# Timing DAG construction and critical-path recovery
# ---------------------------------------------------------------------------

def _simulate_action_graph(
    schedule: dict[int, list[_Action]],
    stage_to_rank: Callable[[int], int],
    num_stages: int,
    forward_times: list[float],
    backward_times: list[float],
    backward_input_times: list[float],
    backward_weight_times: list[float],
    communication_times: list[float],
) -> PipelineSimulationResult:
    # Build a global action list so we can model dependencies as a DAG.
    #
    # Resource constraints:
    # - Compute work on the same rank is serialized.  This approximates one
    #   logical compute stream per rank for forward / backward kernels.
    # - Communication work on the same rank is also serialized, but it is
    #   tracked separately from compute.  This matches schedules.py more
    #   closely than a single rank-local chain because both 1F1B and the
    #   interleaved schedules issue P2P ops asynchronously:
    #   * Schedule1F1B keeps outstanding send_work in warmup/cooldown and
    #     fuses send/recv batches in steady state.
    #   * Interleaved runtime schedules lower SEND_* / RECV_* to async P2P
    #     calls as well.
    # - A long SEND_* therefore does not automatically stall the next local
    #   compute op unless an explicit data dependency requires it.
    #
    # Data dependencies:
    # - Cross-rank data movement is still represented explicitly in the DAG via
    #   F -> SEND_F -> RECV_F -> next-stage F and
    #   B/I -> SEND_B -> RECV_B -> previous-stage B/I.
    # - Same-rank adjacent stages still depend directly on each other because
    #   they exchange activations / gradients through local memory without
    #   materializing SEND_* / RECV_* nodes.
    #
    # Critical-path recovery:
    # - After the DAG is built, we run a longest-path style earliest-finish
    #   pass in topological order.
    # - The predecessor that determines each node's earliest start is stored in
    #   `critical_pred`, then backtracked from the last-finishing node to
    #   recover one critical path for reporting.
    nodes: list[_ScheduledAction] = []
    node_id_by_rank_pos: dict[tuple[int, int], int] = {}
    for rank in sorted(schedule):
        for pos, action in enumerate(schedule[rank]):
            node_id = len(nodes)
            node_id_by_rank_pos[(rank, pos)] = node_id
            nodes.append(_ScheduledAction(rank=rank, action=action))

    num_nodes = len(nodes)
    preds: list[set[int]] = [set() for _ in range(num_nodes)]
    succs: list[set[int]] = [set() for _ in range(num_nodes)]

    def add_edge(src: int, dst: int) -> None:
        if src == dst:
            return
        if src not in preds[dst]:
            preds[dst].add(src)
            succs[src].add(dst)

    def is_compute_action(action: _Action) -> bool:
        return action.computation_type in (
            _ComputationType.FORWARD,
            _ComputationType.FULL_BACKWARD,
            _ComputationType.BACKWARD_INPUT,
            _ComputationType.BACKWARD_WEIGHT,
        )

    # Resource-constraint edges:
    # - compute ops serialize with compute ops on the same rank
    # - comm ops serialize with comm ops on the same rank
    # Cross-stream ordering is added later only when a true data dependency
    # exists.
    for rank in sorted(schedule):
        prev_compute: int | None = None
        prev_comm: int | None = None
        for pos, action in enumerate(schedule[rank]):
            node_id = node_id_by_rank_pos[(rank, pos)]
            if is_compute_action(action):
                if prev_compute is not None:
                    add_edge(prev_compute, node_id)
                prev_compute = node_id
            else:
                if prev_comm is not None:
                    add_edge(prev_comm, node_id)
                prev_comm = node_id

    send_f: dict[tuple[int, int], int] = {}
    recv_f: dict[tuple[int, int], int] = {}
    send_b: dict[tuple[int, int], int] = {}
    recv_b: dict[tuple[int, int], int] = {}
    fwd_nodes: dict[tuple[int, int], int] = {}
    bwd_nodes: dict[tuple[int, int], int] = {}
    bwd_input_nodes: dict[tuple[int, int], int] = {}
    weight_nodes: dict[tuple[int, int], int] = {}

    for node_id, item in enumerate(nodes):
        action = item.action
        key = (action.stage_index, action.microbatch_index)
        if action.computation_type == _ComputationType.SEND_F:
            send_f[key] = node_id
        elif action.computation_type == _ComputationType.RECV_F:
            recv_f[key] = node_id
        elif action.computation_type == _ComputationType.SEND_B:
            send_b[key] = node_id
        elif action.computation_type == _ComputationType.RECV_B:
            recv_b[key] = node_id
        elif action.computation_type == _ComputationType.FORWARD:
            fwd_nodes[key] = node_id
        elif action.computation_type == _ComputationType.FULL_BACKWARD:
            bwd_nodes[key] = node_id
        elif action.computation_type == _ComputationType.BACKWARD_INPUT:
            bwd_input_nodes[key] = node_id
        elif action.computation_type == _ComputationType.BACKWARD_WEIGHT:
            weight_nodes[key] = node_id

    for (stage, mb), recv_node in recv_f.items():
        send_node = send_f.get((stage - 1, mb))
        if send_node is not None:
            add_edge(send_node, recv_node)
    for (stage, mb), recv_node in recv_b.items():
        send_node = send_b.get((stage + 1, mb))
        if send_node is not None:
            add_edge(send_node, recv_node)

    # Data-dependency edges: a SEND node cannot start until the local producer
    # has generated the payload that will be transferred.
    for (stage, mb), send_node in send_f.items():
        pred = fwd_nodes.get((stage, mb))
        if pred is not None:
            add_edge(pred, send_node)
    for (stage, mb), send_node in send_b.items():
        pred = bwd_input_nodes.get((stage, mb), bwd_nodes.get((stage, mb)))
        if pred is not None:
            add_edge(pred, send_node)

    for (stage, mb), node_id in fwd_nodes.items():
        # Adjacent local stages on the same rank communicate through memory, so
        # we need an explicit dependency even when no SEND/RECV is inserted.
        if stage > 0 and stage_to_rank(stage - 1) == stage_to_rank(stage):
            pred = fwd_nodes.get((stage - 1, mb))
            if pred is not None:
                add_edge(pred, node_id)
        else:
            # When the producer is on another rank, the consumer forward is
            # gated by the matching RECV_F node instead of a direct local edge.
            pred = recv_f.get((stage, mb))
            if pred is not None:
                add_edge(pred, node_id)

    for (stage, mb), node_id in bwd_nodes.items():
        # Backward depends on the same stage's forward and, when colocated, on
        # the next stage's backward result.
        pred_fwd = fwd_nodes.get((stage, mb))
        if pred_fwd is not None:
            add_edge(pred_fwd, node_id)
        if stage < num_stages - 1 and stage_to_rank(stage + 1) == stage_to_rank(stage):
            pred = bwd_input_nodes.get((stage + 1, mb), bwd_nodes.get((stage + 1, mb)))
            if pred is not None:
                add_edge(pred, node_id)
        else:
            # Cross-rank backward dependencies are materialized through
            # SEND_B -> RECV_B rather than a direct stage-to-stage edge.
            pred = recv_b.get((stage, mb))
            if pred is not None:
                add_edge(pred, node_id)

    for (stage, mb), node_id in bwd_input_nodes.items():
        # Split backward-input follows the same dependency pattern as full backward.
        pred_fwd = fwd_nodes.get((stage, mb))
        if pred_fwd is not None:
            add_edge(pred_fwd, node_id)
        if stage < num_stages - 1 and stage_to_rank(stage + 1) == stage_to_rank(stage):
            pred = bwd_input_nodes.get((stage + 1, mb), bwd_nodes.get((stage + 1, mb)))
            if pred is not None:
                add_edge(pred, node_id)
        else:
            # BACKWARD_INPUT follows the same communication gating as a full
            # backward op when the gradient arrives from another rank.
            pred = recv_b.get((stage, mb))
            if pred is not None:
                add_edge(pred, node_id)

    for (stage, mb), node_id in weight_nodes.items():
        # Weight-gradient work can only start after backward-input for the same
        # stage and microbatch has completed.
        pred = bwd_input_nodes.get((stage, mb))
        if pred is not None:
            add_edge(pred, node_id)

    def duration(action: _Action) -> float:
        stage = action.stage_index
        op = action.computation_type
        if op == _ComputationType.FORWARD:
            return forward_times[stage]
        if op == _ComputationType.FULL_BACKWARD:
            return backward_times[stage]
        if op == _ComputationType.BACKWARD_INPUT:
            return backward_input_times[stage]
        if op == _ComputationType.BACKWARD_WEIGHT:
            return backward_weight_times[stage]
        if op == _ComputationType.SEND_F:
            return communication_times[stage]
        if op == _ComputationType.SEND_B:
            return communication_times[stage - 1]
        if op in (_ComputationType.RECV_F, _ComputationType.RECV_B):
            return 0.0
        raise RuntimeError(f"Unsupported action type {op}.")

    # Critical-path recovery: compute earliest start / finish times in
    # topological order, and remember which predecessor dominates each node.
    indegree = [len(pred_set) for pred_set in preds]
    ready: list[int] = [idx for idx, deg in enumerate(indegree) if deg == 0]
    heapq.heapify(ready)
    topo_order: list[int] = []
    while ready:
        node = heapq.heappop(ready)
        topo_order.append(node)
        for nxt in succs[node]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                heapq.heappush(ready, nxt)

    if len(topo_order) != num_nodes:
        raise RuntimeError("Action graph has a cycle. The generated schedule is invalid.")

    start_times = [0.0] * num_nodes
    end_times = [0.0] * num_nodes
    critical_pred: list[int | None] = [None] * num_nodes

    for node in topo_order:
        pred_list = list(preds[node])
        if len(pred_list) > 0:
            best_pred = max(pred_list, key=lambda p: (end_times[p], -p))
            critical_pred[node] = best_pred
            start_times[node] = end_times[best_pred]
        end_times[node] = start_times[node] + duration(nodes[node].action)

    end_node = max(range(num_nodes), key=lambda n: (end_times[n], -n))
    iteration_time = end_times[end_node]

    # Backtrack from the last finishing node to recover one critical path.
    critical_path_node_ids = []
    cur: int | None = end_node
    while cur is not None:
        critical_path_node_ids.append(cur)
        cur = critical_pred[cur]
    critical_path_node_ids.reverse()

    all_operations = [
        CriticalPathOperation(
            rank=nodes[node_id].rank,
            stage=nodes[node_id].action.stage_index,
            microbatch=nodes[node_id].action.microbatch_index,
            op_type=nodes[node_id].action.computation_type.value,
            duration=duration(nodes[node_id].action),
            start_time=start_times[node_id],
            end_time=end_times[node_id],
        )
        for node_id in range(num_nodes)
    ]

    critical_path = [
        CriticalPathOperation(
            rank=nodes[node_id].rank,
            stage=nodes[node_id].action.stage_index,
            microbatch=nodes[node_id].action.microbatch_index,
            op_type=nodes[node_id].action.computation_type.value,
            duration=duration(nodes[node_id].action),
            start_time=start_times[node_id],
            end_time=end_times[node_id],
        )
        for node_id in critical_path_node_ids
    ]
    return PipelineSimulationResult(
        iteration_time=iteration_time,
        critical_path=critical_path,
        all_operations=all_operations,
    )


# ---------------------------------------------------------------------------
# Compute-only text visualization
# ---------------------------------------------------------------------------

def _action_to_label(action: _Action | None) -> str:
    if action is None:
        return ""
    return f"{action.computation_type.value}{action.microbatch_index}"


def _action_to_rank_label(action: _Action | None) -> str:
    if action is None:
        return ""
    return f"s{action.stage_index}{action.computation_type.value}{action.microbatch_index}"


def _render_stage_timeline(
    schedule: ScheduleByRank,
    *,
    num_stages: int,
) -> str:
    # Convert a rank-indexed aligned schedule into stage rows so reviewers can
    # inspect the schedule directly against schedules.py.
    total_steps = max((len(actions) for actions in schedule.values()), default=0)
    stage_rows: dict[int, list[str]] = {
        stage: ["" for _ in range(total_steps)] for stage in range(num_stages)
    }

    for rank in sorted(schedule):
        for step in range(total_steps):
            action = schedule[rank][step] if step < len(schedule[rank]) else None
            if action is None:
                continue
            stage_rows[action.stage_index][step] = _action_to_label(action)

    labels = [label for row in stage_rows.values() for label in row if label]
    cell_width = max([1] + [len(label) for label in labels])
    bubble = " " * cell_width
    stage_label_width = max(len(f"stage{stage}") for stage in range(num_stages))

    lines = []
    for stage in range(num_stages):
        padded = [
            (label if label else bubble).ljust(cell_width)
            for label in stage_rows[stage]
        ]
        stage_label = f"stage{stage}".ljust(stage_label_width)
        # Keep a trailing separator so the text view mirrors the clocked
        # schedule tables commonly used in pipeline papers and bug reports.
        lines.append(f"{stage_label}: " + "|".join(padded) + "|")
    return "\n".join(lines)


def _render_rank_timeline(
    schedule: ScheduleByRank,
) -> str:
    # This complementary view keeps rows in physical-rank order.  Unlike the
    # stage view, each non-bubble cell embeds the stage id (for example `s2F3`)
    # so interleaving across multiple local stages is immediately visible.
    total_steps = max((len(actions) for actions in schedule.values()), default=0)
    rank_rows: dict[int, list[str]] = {}
    for rank in sorted(schedule):
        row = ["" for _ in range(total_steps)]
        for step in range(total_steps):
            action = schedule[rank][step] if step < len(schedule[rank]) else None
            row[step] = _action_to_rank_label(action)
        rank_rows[rank] = row

    labels = [label for row in rank_rows.values() for label in row if label]
    cell_width = max([1] + [len(label) for label in labels])
    bubble = " " * cell_width
    rank_label_width = max(len(f"rank{rank}") for rank in rank_rows) if rank_rows else len("rank0")

    lines = []
    for rank in sorted(rank_rows):
        padded = [
            (label if label else bubble).ljust(cell_width)
            for label in rank_rows[rank]
        ]
        rank_label = f"rank{rank}".ljust(rank_label_width)
        lines.append(f"{rank_label}: " + "|".join(padded) + "|")
    return "\n".join(lines)


def _simulate_compute_only(
    pipeline_order: ScheduleByRank,
    *,
    num_stages: int,
) -> ScheduleByRank:
    # Dry-run the compute actions only.  This is the rank-view counterpart to
    # `_simulate_comms_compute`: we keep the per-rank local action order, then
    # advance each rank as soon as the data dependency of its next compute op is
    # satisfied. 
    pending: ScheduleByRank = {
        rank: list(actions) for rank, actions in sorted(pipeline_order.items())
    }
    simulated: ScheduleByRank = {rank: [] for rank in pending}

    seen_ops: set[_Action] = set()

    def add_to_schedule(rank: int, action: Optional[_Action]) -> None:
        simulated[rank].append(action)

    def ready_to_schedule(action: Optional[_Action], ready_snapshot: set[_Action]) -> bool:
        if action is None:
            return True

        stage = action.stage_index
        op = action.computation_type
        mb = action.microbatch_index

        if op == _ComputationType.FORWARD:
            return (
                stage == 0
                or _Action(stage - 1, _ComputationType.FORWARD, mb) in ready_snapshot
            )

        if op in (_ComputationType.FULL_BACKWARD, _ComputationType.BACKWARD_INPUT):
            if _Action(stage, _ComputationType.FORWARD, mb) not in ready_snapshot:
                return False
            if stage == num_stages - 1:
                return True
            return (
                _Action(stage + 1, _ComputationType.FULL_BACKWARD, mb) in ready_snapshot
                or _Action(stage + 1, _ComputationType.BACKWARD_INPUT, mb) in ready_snapshot
            )

        if op == _ComputationType.BACKWARD_WEIGHT:
            return _Action(stage, _ComputationType.BACKWARD_INPUT, mb) in ready_snapshot

        raise ValueError(f"Unsupported compute action {action}.")

    while pending:
        progress = False
        # A timestep is evaluated against the state from the previous timestep.
        # Ops that execute earlier in the same clock slot should not
        # immediately unblock later ranks in that same slot.
        ready_snapshot = set(seen_ops)
        executed_this_step: list[_Action] = []

        for rank in sorted(list(pending.keys())):
            if not pending[rank]:
                del pending[rank]
                continue

            action = pending[rank][0]
            if action is None:
                add_to_schedule(rank, action)
                pending[rank].pop(0)
                progress = True
                continue

            if ready_to_schedule(action, ready_snapshot):
                add_to_schedule(rank, action)
                executed_this_step.append(action)
                pending[rank].pop(0)
                progress = True
            else:
                add_to_schedule(rank, None)

        if not progress and pending:
            raise RuntimeError("Compute-only action simulation stalled. The schedule is invalid.")

        seen_ops.update(executed_this_step)

    return simulated


def _build_rank_view_compute_schedule(
    schedule_type: PipelineScheduleType,
    *,
    num_stages: int,
    pp_group_size: int,
    n_microbatches: int,
    compute_schedule: ScheduleByRank,
) -> ScheduleByRank:
    if schedule_type != PipelineScheduleType.INTERLEAVED_1F1B:
        return compute_schedule

    # For rank view we want a tighter, execution-oriented picture: keep
    # the local op order, preserve only the warmup hold needed before entering
    # 1F1B steady state, and then let a compute-only simulator place the real
    # bubbles induced by dependencies.
    if num_stages % pp_group_size != 0:
        raise ValueError(
            "num_stages must be divisible by pp_group_size for interleaved schedules, "
            f"but got num_stages={num_stages}, pp_group_size={pp_group_size}."
        )

    n_local_stages = num_stages // pp_group_size
    number_of_rounds = max(1, n_microbatches // pp_group_size)
    microbatches_per_round = n_microbatches // number_of_rounds

    compact_schedule: ScheduleByRank = {}
    for rank in range(pp_group_size):
        warmup_ops = _get_warmup_ops(
            rank=rank,
            n_local_stages=n_local_stages,
            microbatches_per_round=microbatches_per_round,
            pp_group_size=pp_group_size,
            n_microbatches=n_microbatches,
            multiply_factor=2,
        )
        microbatch_ops = n_local_stages * n_microbatches
        fwd_bwd_ops = microbatch_ops - warmup_ops

        non_none_actions = [action for action in compute_schedule[rank] if action is not None]
        warmup_actions = non_none_actions[:warmup_ops]
        steady_state_actions = non_none_actions[warmup_ops : warmup_ops + (2 * fwd_bwd_ops)]
        cooldown_actions = non_none_actions[warmup_ops + (2 * fwd_bwd_ops) :]

        # Only keep the warmup-to-steady-state hold that is intrinsic to the
        # local 1F1B transition. 
        hold_slots = max(0, pp_group_size - 1 - rank)
        compact_actions = warmup_actions + ([None] * hold_slots) + steady_state_actions

        # Cooldown also has a phase boundary.  Keeping just this one local
        # transition hold matches the actual rank-level execution order without
        # reintroducing the padded per-backward bubbles from pipeline_order.
        if len(cooldown_actions) > 0:
            compact_actions.extend([None] * hold_slots)
            compact_actions.extend(cooldown_actions)

        compact_schedule[rank] = compact_actions

    return _simulate_compute_only(
        compact_schedule,
        num_stages=num_stages,
    )


# ---------------------------------------------------------------------------
# Public schedule inspection and simulation APIs
# ---------------------------------------------------------------------------

def auto_partition(
    forward_flops: list[int],
    backward_flops: list[int],
    num_stages: int,
    n_microbatches: int,
    *,
    schedule_type: str | PipelineScheduleType = PipelineScheduleType.ONE_F_ONE_B,
    communication_volume: int | list[int] | None = None,
    device_compute_throughput: int | float = 1.0,
    network_bandwidth: int | float = 1.0,
    virtual_stages_per_rank: int | None = None,
    backward_input_flops: list[int] | None = None,
    backward_weight_flops: list[int] | None = None,
) -> list[int]:
    # Default strategies:
    # - schedule_type defaults to 1F1B,
    # - communication defaults to zero,
    #   partitioning view more closely,
    # - device/network defaults to 1 so the numeric inputs can also be treated
    #   as already-normalized time costs if desired,
    # - num_stages is interpreted as the physical pipeline width (pp_group_size).
    schedule_kind = _normalize_schedule_type(schedule_type)
    resolved_communication = 0 if communication_volume is None else communication_volume
    search_result = optimize_partition_model_layers(
        schedule_kind,
        forward_flops,
        backward_flops,
        resolved_communication,
        device_compute_throughput,
        network_bandwidth,
        num_stages,
        n_microbatches,
        virtual_stages_per_rank=virtual_stages_per_rank,
        backward_input_flops=backward_input_flops,
        backward_weight_flops=backward_weight_flops,
    )
    return [stage_layers[0] for stage_layers in search_result.best_plan.stage_partitions]


def generate_compute_schedule(
    schedule_type: str | PipelineScheduleType,
    *,
    num_stages: int,
    n_microbatches: int,
    pp_group_size: int | None = None,
) -> ScheduleByRank:
    # Public helper used by tests and debugging utilities.
    schedule_kind = _normalize_schedule_type(schedule_type)
    pp_size = num_stages if pp_group_size is None else pp_group_size
    return _build_compute_schedule(
        schedule_kind,
        num_stages=num_stages,
        pp_group_size=pp_size,
        n_microbatches=n_microbatches,
    )


def render_compute_schedule_timeline(
    schedule_type: str | PipelineScheduleType,
    *,
    num_stages: int,
    n_microbatches: int,
    pp_group_size: int | None = None,
) -> str:
    # Stage-oriented compute-only view for comparing against textbook or paper
    # schedule diagrams.
    pp_size = num_stages if pp_group_size is None else pp_group_size
    compute_schedule = generate_compute_schedule(
        schedule_type,
        num_stages=num_stages,
        n_microbatches=n_microbatches,
        pp_group_size=pp_size,
    )
    return _render_stage_timeline(compute_schedule, num_stages=num_stages)


def render_rank_compute_schedule_timeline(
    schedule_type: str | PipelineScheduleType,
    *,
    num_stages: int,
    n_microbatches: int,
    pp_group_size: int | None = None,
) -> str:
    # Rank-oriented compute-only view.  Each token embeds the logical stage id
    # so local-stage interleaving is visible directly in one row.
    schedule_kind = _normalize_schedule_type(schedule_type)
    pp_size = num_stages if pp_group_size is None else pp_group_size
    compute_schedule = generate_compute_schedule(
        schedule_type,
        num_stages=num_stages,
        n_microbatches=n_microbatches,
        pp_group_size=pp_size,
    )
    rank_view_schedule = _build_rank_view_compute_schedule(
        schedule_kind,
        num_stages=num_stages,
        pp_group_size=pp_size,
        n_microbatches=n_microbatches,
        compute_schedule=compute_schedule,
    )
    return _render_rank_timeline(rank_view_schedule)


def simulate_pipeline_schedule(
    schedule_type: str | PipelineScheduleType,
    forward_flops: list[int],
    backward_flops: list[int],
    communication_volume: int | list[int],
    n_microbatches: int,
    *,
    pp_group_size: int | None = None,
    backward_input_flops: list[int] | None = None,
    backward_weight_flops: list[int] | None = None,
    device_compute_throughput: int | float = 1.0,
    network_bandwidth: int | float = 1.0,
) -> PipelineSimulationResult:
    # Main public simulator entry point.
    #
    # Inputs stay in hardware-meaningful units:
    # - computation is expressed in FLOPs,
    # - communication is expressed in bytes,
    # - hardware is expressed as FLOP/s and bytes/s.
    #
    # The simulator then:
    # 1. converts FLOPs / bytes into seconds with the supplied hardware model,
    # 2. generates the selected pipeline schedule,
    # 3. inserts the required SEND/RECV actions,
    # 4. builds the timing DAG and recovers one critical path.
    if n_microbatches <= 0:
        raise ValueError(f"n_microbatches must be positive, got {n_microbatches}.")
    if len(forward_flops) == 0 or len(backward_flops) == 0:
        raise ValueError("forward_flops and backward_flops cannot be empty.")
    if len(forward_flops) != len(backward_flops):
        raise ValueError(
            "forward_flops and backward_flops must have the same length, "
            f"but got {len(forward_flops)} and {len(backward_flops)}."
        )
    for idx, flops in enumerate(forward_flops):
        _ensure_non_negative_int(flops, f"forward_flops[{idx}]")
    for idx, flops in enumerate(backward_flops):
        _ensure_non_negative_int(flops, f"backward_flops[{idx}]")

    schedule_kind = _normalize_schedule_type(schedule_type)
    num_stages = len(forward_flops)
    pp_size = num_stages if pp_group_size is None else pp_group_size
    if pp_size <= 0 or pp_size > num_stages:
        raise ValueError(
            f"pp_group_size must be in [1, {num_stages}], but got {pp_size}."
        )

    _ensure_positive_number(device_compute_throughput, "device_compute_throughput")
    _ensure_positive_number(network_bandwidth, "network_bandwidth")

    forward = _normalize_compute_times(
        forward_flops,
        device_compute_throughput,
    )
    backward = _normalize_compute_times(
        backward_flops,
        device_compute_throughput,
    )
    comm = _normalize_comm_times(
        communication_volume,
        num_stages,
        network_bandwidth,
    )
    bwd_input, bwd_weight = _normalize_backward_split(
        backward_flops,
        backward_input_flops,
        backward_weight_flops,
    )
    bwd_input = _normalize_compute_times(
        bwd_input,
        device_compute_throughput,
    )
    bwd_weight = _normalize_compute_times(
        bwd_weight,
        device_compute_throughput,
    )

    compute_schedule = _build_compute_schedule(
        schedule_kind,
        num_stages=num_stages,
        pp_group_size=pp_size,
        n_microbatches=n_microbatches,
    )

    stage_to_rank = lambda stage: stage % pp_size
    comm_schedule = _add_send_recv(compute_schedule, stage_to_rank, num_stages)

    return _simulate_action_graph(
        schedule=comm_schedule,
        stage_to_rank=stage_to_rank,
        num_stages=num_stages,
        forward_times=forward,
        backward_times=backward,
        backward_input_times=bwd_input,
        backward_weight_times=bwd_weight,
        communication_times=comm,
    )

