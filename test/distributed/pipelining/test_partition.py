# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
from torch.distributed.pipelining._partition import (
    auto_partition,
    generate_compute_schedule,
    optimize_partition_model_layers,
    partition_model_layers,
    PipelineScheduleType,
    render_rank_compute_schedule_timeline,
    simulate_pipeline_schedule,
)
from torch.distributed.pipelining.schedules import _ComputationType
from torch.testing._internal.common_utils import run_tests, TestCase


def _double_flops(values):
    # Keep test workloads aligned with the common theoretical assumption that
    # backward compute is 2x the corresponding forward compute.
    return [2 * value for value in values]


def _half_even_flops(values):
    # Zero-bubble tests split the total backward compute evenly into input-
    # gradient work and weight-gradient work.
    halves = []
    for value in values:
        if value % 2 != 0:
            raise ValueError(f"Expected an even FLOP count, got {value}.")
        halves.append(value // 2)
    return halves


class PartitionSimulationTests(TestCase):
    TIME_DISPLAY_SCALE_MS = 1_000.0
    # Use one shared 8-layer Transformer-like workload across all schedule
    # tests so their printed iteration time / critical path / rank timeline can
    # be compared directly on the same 4-device budget.
    #
    # Mapping:
    # - Interleaved schedules: 8 stages over 4 ranks, so each rank owns
    #   2 local stages.
    # - 1F1B: collapse every 2 adjacent layers into 1 pipeline stage so the
    #   schedule still runs on the same 4 ranks.
    #
    # The magnitude is chosen to resemble one modern dense Transformer decoder
    # layer with hidden size around 4096, feed-forward size around 14336, and a
    # 4096-token microbatch.  The numbers are rounded, stage-homogeneous
    # approximations so schedule comparisons stay easy to read.
    COMMON_FORWARD_FLOPS = [2_250_000_000_000] * 8
    COMMON_BACKWARD_FLOPS = [4_500_000_000_000] * 8

    # Approximate activation traffic for one layer boundary:
    # 4096 tokens * 4096 hidden * 2 bytes (bf16) ~= 32 MiB.
    COMMON_COMMUNICATION_VOLUME = [33_554_432] * 7
    COMMON_N_MICROBATCHES = 8

    # Under the standard zero-bubble abstraction, backward-input work and
    # backward-weight work evenly split the total backward cost.
    COMMON_BACKWARD_INPUT_FLOPS = [2_250_000_000_000] * 8
    COMMON_BACKWARD_WEIGHT_FLOPS = [2_250_000_000_000] * 8

    # Shared hardware assumptions for turning FLOPs / bytes into seconds:
    # - 312 TFLOP/s models one A100-class bf16 device.
    # - 50 GB/s models one 400 Gb/s interconnect path.
    COMMON_DEVICE_COMPUTE_THROUGHPUT = 312_000_000_000_000
    COMMON_NETWORK_BANDWIDTH = 50_000_000_000
    COMMON_PP_GROUP_SIZE = 4

    PARTITION_TEST_FORWARD_FLOPS = [
        1_000,
        2_000,
        3_000,
        4_000,
        5_000,
        6_000,
        7_000,
        8_000,
    ]
    PARTITION_TEST_BACKWARD_FLOPS = _double_flops(PARTITION_TEST_FORWARD_FLOPS)
    PARTITION_TEST_COMMUNICATION_VOLUME = [128] * 7
    HEURISTIC_1F1B_FORWARD_FLOPS = [1170, 2846, 1146, 4254, 2199, 1289, 6888, 2312]
    HEURISTIC_1F1B_BACKWARD_FLOPS = _double_flops(HEURISTIC_1F1B_FORWARD_FLOPS)
    HEURISTIC_1F1B_COMMUNICATION_VOLUME = [128] * 7

    HEURISTIC_INTERLEAVED_FORWARD_FLOPS = [
        3_424_614_582_864,
        6_186_943_846_406,
        6_099_927_589_441,
        4_961_141_868_353,
        6_808_911_722_949,
        1_723_089_387_674,
        6_387_281_953_781,
        5_278_282_850_207,
        6_543_029_278_476,
        3_135_416_417_038,
        2_027_661_282_943,
        2_940_059_633_160,
    ]
    HEURISTIC_INTERLEAVED_BACKWARD_FLOPS = _double_flops(
        HEURISTIC_INTERLEAVED_FORWARD_FLOPS
    )
    HEURISTIC_INTERLEAVED_COMMUNICATION_VOLUME = [
        43_405_107,
        180_456_491,
        60_605_618,
        172_762_728,
        164_606_193,
        42_130_509,
        115_014_477,
        34_570_395,
        109_358_683,
        41_150_899,
        19_768_913,
    ]

    HEURISTIC_ZERO_BUBBLE_FORWARD_FLOPS = [
        3084,
        4757,
        4139,
        1960,
        6283,
        2830,
        3507,
        6252,
        6451,
        2029,
        5175,
        5108,
    ]
    HEURISTIC_ZERO_BUBBLE_BACKWARD_FLOPS = _double_flops(
        HEURISTIC_ZERO_BUBBLE_FORWARD_FLOPS
    )
    HEURISTIC_ZERO_BUBBLE_COMMUNICATION_VOLUME = [128] * 11

    def _format_critical_path(self, result):
        chunks = []
        for op in result.critical_path:
            chunks.append(
                f"{op.op_type}(s{op.stage},mb{op.microbatch},"
                f"{op.start_time * self.TIME_DISPLAY_SCALE_MS:.3f}ms->"
                f"{op.end_time * self.TIME_DISPLAY_SCALE_MS:.3f}ms)"
            )
        return " -> ".join(chunks)

    def _report(self, title, result):
        print(
            f"\n[{title}] iteration_time="
            f"{result.iteration_time * self.TIME_DISPLAY_SCALE_MS:.3f} ms"
        )
        print(f"[{title}] critical_path={self._format_critical_path(result)}")

    def _collapse_adjacent_layers(self, values, layers_per_stage):
        # 1F1B uses one logical stage per rank in these tests, so the shared
        # 8-layer model is collapsed into 4 stage-level inputs for fair
        # schedule comparisons on the same 4-device budget.
        if len(values) % layers_per_stage != 0:
            raise ValueError(
                f"Cannot collapse {len(values)} values with layers_per_stage={layers_per_stage}."
            )
        collapsed = []
        for start in range(0, len(values), layers_per_stage):
            collapsed.append(sum(values[start : start + layers_per_stage]))
        return collapsed

    def _stage_inputs_for_schedule(self, schedule_type):
        # Build schedule-specific simulator inputs from the shared model-level
        # workload definition above.
        if schedule_type == PipelineScheduleType.ONE_F_ONE_B:
            layers_per_stage = len(self.COMMON_FORWARD_FLOPS) // self.COMMON_PP_GROUP_SIZE
            forward_flops = self._collapse_adjacent_layers(
                self.COMMON_FORWARD_FLOPS, layers_per_stage
            )
            backward_flops = self._collapse_adjacent_layers(
                self.COMMON_BACKWARD_FLOPS, layers_per_stage
            )
            # Communication happens only at stage boundaries after collapsing
            # adjacent layers into a single 1F1B stage.
            communication_volume = [
                self.COMMON_COMMUNICATION_VOLUME[(idx + 1) * layers_per_stage - 1]
                for idx in range(self.COMMON_PP_GROUP_SIZE - 1)
            ]
            return (
                forward_flops,
                backward_flops,
                communication_volume,
                None,
                None,
            )

        if schedule_type == PipelineScheduleType.INTERLEAVED_ZERO_BUBBLE:
            return (
                self.COMMON_FORWARD_FLOPS,
                self.COMMON_BACKWARD_FLOPS,
                self.COMMON_COMMUNICATION_VOLUME,
                self.COMMON_BACKWARD_INPUT_FLOPS,
                self.COMMON_BACKWARD_WEIGHT_FLOPS,
            )

        return (
            self.COMMON_FORWARD_FLOPS,
            self.COMMON_BACKWARD_FLOPS,
            self.COMMON_COMMUNICATION_VOLUME,
            None,
            None,
        )

    def _simulate_with_common_hardware(
        self,
        schedule_type,
        forward_flops,
        backward_flops,
        communication_volume,
        n_microbatches,
        *,
        pp_group_size,
        backward_input_flops=None,
        backward_weight_flops=None,
    ):
        # Most tests exercise schedule behavior, not hardware sensitivity, so
        # use one shared hardware configuration helper to keep the call sites
        # short and easy to review.
        return simulate_pipeline_schedule(
            schedule_type,
            forward_flops,
            backward_flops,
            communication_volume,
            n_microbatches,
            pp_group_size=pp_group_size,
            backward_input_flops=backward_input_flops,
            backward_weight_flops=backward_weight_flops,
            device_compute_throughput=self.COMMON_DEVICE_COMPUTE_THROUGHPUT,
            network_bandwidth=self.COMMON_NETWORK_BANDWIDTH,
        )

    def _partition_with_common_hardware(
        self,
        schedule_type,
        forward_flops,
        backward_flops,
        communication_volume,
        *,
        pp_group_size,
        n_microbatches,
    ):
        return partition_model_layers(
            schedule_type,
            forward_flops,
            backward_flops,
            communication_volume,
            self.COMMON_DEVICE_COMPUTE_THROUGHPUT,
            self.COMMON_NETWORK_BANDWIDTH,
            pp_group_size=pp_group_size,
            n_microbatches=n_microbatches,
        )

    def _optimize_with_common_hardware(
        self,
        schedule_type,
        forward_flops,
        backward_flops,
        communication_volume,
        *,
        pp_group_size,
        n_microbatches,
        backward_input_flops=None,
        backward_weight_flops=None,
    ):
        # Mirror `_partition_with_common_hardware()` for the heuristic search
        # entry point so optimization tests can focus on expected partitions.
        return optimize_partition_model_layers(
            schedule_type,
            forward_flops,
            backward_flops,
            communication_volume,
            self.COMMON_DEVICE_COMPUTE_THROUGHPUT,
            self.COMMON_NETWORK_BANDWIDTH,
            pp_group_size=pp_group_size,
            n_microbatches=n_microbatches,
            backward_input_flops=backward_input_flops,
            backward_weight_flops=backward_weight_flops,
        )

    def _assert_simulation_result_valid(self, result):
        self.assertGreater(result.iteration_time, 0.0)
        self.assertGreater(len(result.critical_path), 0)
        self.assertGreater(len(result.all_operations), 0)
        self.assertEqual(result.critical_path[-1].end_time, result.iteration_time)
        self.assertEqual(result.critical_path[0].start_time, 0.0)
        for i in range(1, len(result.critical_path)):
            self.assertEqual(
                result.critical_path[i - 1].end_time,
                result.critical_path[i].start_time,
            )

    def _assert_rank_is_interleaved(self, compute_schedule, rank):
        # Interleaved schedules should make a physical rank execute more than
        # one local stage, rather than degenerating to a plain 1F1B stage order.
        stage_sequence = [
            action.stage_index
            for action in compute_schedule[rank]
            if action is not None
            and action.computation_type
            in (
                _ComputationType.FORWARD,
                _ComputationType.FULL_BACKWARD,
                _ComputationType.BACKWARD_INPUT,
                _ComputationType.BACKWARD_WEIGHT,
            )
        ]
        self.assertGreater(len(set(stage_sequence)), 1)

    def _assert_partition_plan_valid(self, plan, expected_num_ranks):
        self.assertEqual(len(plan.rank_to_stages), expected_num_ranks)
        self.assertEqual(len(plan.stage_partitions), plan.num_stages)
        self.assertEqual(len(plan.stage_to_rank), plan.num_stages)
        self.assertEqual(len(plan.stage_forward_flops), plan.num_stages)
        self.assertEqual(len(plan.stage_backward_flops), plan.num_stages)
        self.assertEqual(len(plan.stage_compute_times), plan.num_stages)
        self.assertEqual(
            len(plan.stage_communication_volume),
            max(0, plan.num_stages - 1),
        )
        self.assertEqual(plan.objective_value, max(plan.stage_compute_times))

        flattened = []
        for stage_layers in plan.stage_partitions:
            self.assertGreater(len(stage_layers), 0)
            self.assertEqual(stage_layers, list(range(stage_layers[0], stage_layers[-1] + 1)))
            flattened.extend(stage_layers)
        self.assertEqual(flattened, list(range(len(flattened))))

        for stage_idx, rank in enumerate(plan.stage_to_rank):
            self.assertIn(stage_idx, plan.rank_to_stages[rank])

    def _assert_backward_flops_convention(
        self,
        forward_flops,
        backward_flops,
        backward_input_flops=None,
        backward_weight_flops=None,
    ):
        # The tests intentionally follow the common theoretical assumption:
        # backward = 2 * forward, and zero-bubble splits backward evenly into
        # input-gradient work and weight-gradient work.
        self.assertEqual(backward_flops, _double_flops(forward_flops))
        if backward_input_flops is not None:
            self.assertEqual(backward_input_flops, _half_even_flops(backward_flops))
        if backward_weight_flops is not None:
            self.assertEqual(backward_weight_flops, _half_even_flops(backward_flops))

    def _assert_partition_search_result_valid(self, search_result):
        self._assert_partition_plan_valid(
            search_result.base_plan,
            expected_num_ranks=search_result.base_plan.pp_group_size,
        )
        self._assert_partition_plan_valid(
            search_result.best_plan,
            expected_num_ranks=search_result.best_plan.pp_group_size,
        )
        self.assertGreater(search_result.explored_candidates, 0)
        self.assertGreaterEqual(
            search_result.base_iteration_time,
            search_result.best_iteration_time,
        )
        self.assertGreaterEqual(search_result.best_critical_stage, 0)
        self.assertLess(search_result.best_critical_stage, search_result.best_plan.num_stages)

    def _run_schedule_case(
        self,
        title,
        schedule_type,
        forward_flops,
        backward_flops,
        communication_volume,
        n_microbatches,
        *,
        pp_group_size,
        backward_input_flops=None,
        backward_weight_flops=None,
    ):
        # The schedule smoke tests intentionally print only the three most
        # review-friendly artifacts: iteration time, recovered critical path,
        # and rank-level compute-only timeline.
        num_stages = len(forward_flops)
        result = self._simulate_with_common_hardware(
            schedule_type,
            forward_flops,
            backward_flops,
            communication_volume,
            n_microbatches,
            pp_group_size=pp_group_size,
            backward_input_flops=backward_input_flops,
            backward_weight_flops=backward_weight_flops,
        )
        self._report(title, result)
        self._assert_simulation_result_valid(result)

        rank_compute_timeline = render_rank_compute_schedule_timeline(
            schedule_type,
            num_stages=num_stages,
            n_microbatches=n_microbatches,
            pp_group_size=pp_group_size,
        )

        print(f"[{title}] rank compute-only timeline:\n{rank_compute_timeline}")

        return result, rank_compute_timeline

    def test_flops_conventions(self):
        self._assert_backward_flops_convention(
            self.COMMON_FORWARD_FLOPS,
            self.COMMON_BACKWARD_FLOPS,
            self.COMMON_BACKWARD_INPUT_FLOPS,
            self.COMMON_BACKWARD_WEIGHT_FLOPS,
        )
        self._assert_backward_flops_convention(
            self.PARTITION_TEST_FORWARD_FLOPS,
            self.PARTITION_TEST_BACKWARD_FLOPS,
        )
        self._assert_backward_flops_convention(
            self.HEURISTIC_1F1B_FORWARD_FLOPS,
            self.HEURISTIC_1F1B_BACKWARD_FLOPS,
        )
        self._assert_backward_flops_convention(
            self.HEURISTIC_INTERLEAVED_FORWARD_FLOPS,
            self.HEURISTIC_INTERLEAVED_BACKWARD_FLOPS,
        )
        self._assert_backward_flops_convention(
            self.HEURISTIC_ZERO_BUBBLE_FORWARD_FLOPS,
            self.HEURISTIC_ZERO_BUBBLE_BACKWARD_FLOPS,
        )

    def test_simulate_1f1b(self):
        (
            forward_flops,
            backward_flops,
            communication_volume,
            _,
            _,
        ) = self._stage_inputs_for_schedule(PipelineScheduleType.ONE_F_ONE_B)
        _, rank_compute_timeline = self._run_schedule_case(
            "1F1B",
            PipelineScheduleType.ONE_F_ONE_B,
            forward_flops=forward_flops,
            backward_flops=backward_flops,
            communication_volume=communication_volume,
            n_microbatches=self.COMMON_N_MICROBATCHES,
            pp_group_size=self.COMMON_PP_GROUP_SIZE,
        )

    def test_simulate_interleaved_1f1b(self):
        (
            forward_flops,
            backward_flops,
            communication_volume,
            _,
            _,
        ) = self._stage_inputs_for_schedule(
            PipelineScheduleType.INTERLEAVED_1F1B
        )
        _, rank_compute_timeline = self._run_schedule_case(
            "Interleaved1F1B",
            PipelineScheduleType.INTERLEAVED_1F1B,
            forward_flops=forward_flops,
            backward_flops=backward_flops,
            communication_volume=communication_volume,
            n_microbatches=self.COMMON_N_MICROBATCHES,
            pp_group_size=self.COMMON_PP_GROUP_SIZE,
        )
        compute_schedule = generate_compute_schedule(
            PipelineScheduleType.INTERLEAVED_1F1B,
            num_stages=len(forward_flops),
            n_microbatches=self.COMMON_N_MICROBATCHES,
            pp_group_size=self.COMMON_PP_GROUP_SIZE,
        )
        for rank in range(self.COMMON_PP_GROUP_SIZE):
            self._assert_rank_is_interleaved(compute_schedule, rank=rank)

    def test_simulate_interleaved_zero_bubble(self):
        (
            forward_flops,
            backward_flops,
            communication_volume,
            backward_input_flops,
            backward_weight_flops,
        ) = self._stage_inputs_for_schedule(
            PipelineScheduleType.INTERLEAVED_ZERO_BUBBLE
        )
        _, rank_compute_timeline = self._run_schedule_case(
            "InterleavedZeroBubble",
            PipelineScheduleType.INTERLEAVED_ZERO_BUBBLE,
            forward_flops=forward_flops,
            backward_flops=backward_flops,
            communication_volume=communication_volume,
            n_microbatches=self.COMMON_N_MICROBATCHES,
            pp_group_size=self.COMMON_PP_GROUP_SIZE,
            backward_input_flops=backward_input_flops,
            backward_weight_flops=backward_weight_flops,
        )
        compute_schedule = generate_compute_schedule(
            PipelineScheduleType.INTERLEAVED_ZERO_BUBBLE,
            num_stages=len(forward_flops),
            n_microbatches=self.COMMON_N_MICROBATCHES,
            pp_group_size=self.COMMON_PP_GROUP_SIZE,
        )
        for rank in range(self.COMMON_PP_GROUP_SIZE):
            self._assert_rank_is_interleaved(compute_schedule, rank=rank)

    def test_partition_model_layers_for_1f1b(self):
        plan = self._partition_with_common_hardware(
            PipelineScheduleType.ONE_F_ONE_B,
            self.PARTITION_TEST_FORWARD_FLOPS,
            self.PARTITION_TEST_BACKWARD_FLOPS,
            self.PARTITION_TEST_COMMUNICATION_VOLUME,
            pp_group_size=4,
            n_microbatches=8,
        )
        self._assert_partition_plan_valid(plan, expected_num_ranks=4)
        self.assertEqual(plan.num_stages, 4)
        self.assertEqual(plan.virtual_stages_per_rank, 1)
        self.assertEqual(plan.stage_to_rank, [0, 1, 2, 3])
        self.assertEqual(plan.stage_partitions, [[0, 1, 2, 3], [4, 5], [6], [7]])

    def test_partition_model_layers_for_interleaved_1f1b(self):
        plan = self._partition_with_common_hardware(
            PipelineScheduleType.INTERLEAVED_1F1B,
            self.COMMON_FORWARD_FLOPS,
            self.COMMON_BACKWARD_FLOPS,
            self.COMMON_COMMUNICATION_VOLUME,
            pp_group_size=self.COMMON_PP_GROUP_SIZE,
            n_microbatches=self.COMMON_N_MICROBATCHES,
        )
        self._assert_partition_plan_valid(plan, expected_num_ranks=self.COMMON_PP_GROUP_SIZE)
        self.assertEqual(plan.num_stages, 8)
        self.assertEqual(plan.virtual_stages_per_rank, 2)
        self.assertEqual(plan.stage_to_rank, [0, 1, 2, 3, 0, 1, 2, 3])

    def test_partition_model_layers_for_interleaved_zero_bubble(self):
        plan = self._partition_with_common_hardware(
            PipelineScheduleType.INTERLEAVED_ZERO_BUBBLE,
            self.COMMON_FORWARD_FLOPS,
            self.COMMON_BACKWARD_FLOPS,
            self.COMMON_COMMUNICATION_VOLUME,
            pp_group_size=self.COMMON_PP_GROUP_SIZE,
            n_microbatches=self.COMMON_N_MICROBATCHES,
        )
        self._assert_partition_plan_valid(plan, expected_num_ranks=self.COMMON_PP_GROUP_SIZE)
        self.assertEqual(plan.num_stages, 8)
        self.assertEqual(plan.virtual_stages_per_rank, 2)
        self.assertEqual(plan.stage_to_rank, [0, 1, 2, 3, 0, 1, 2, 3])

    def test_optimize_partition_model_layers_rebalances_prefix_after_shift(self):
        # This regression workload ensures the heuristic does not stop at the
        # original DP partition when a prefix re-balance can still help.
        forward_flops = [8, 11, 2, 1, 18, 5, 4, 11]
        backward_flops = _double_flops(forward_flops)
        communication_volume = [0] * (len(forward_flops) - 1)

        search_result = optimize_partition_model_layers(
            PipelineScheduleType.ONE_F_ONE_B,
            forward_flops,
            backward_flops,
            communication_volume,
            device_compute_throughput=1_000,
            network_bandwidth=1_000,
            pp_group_size=4,
            n_microbatches=8,
        )

        self.assertEqual(
            search_result.base_plan.stage_partitions,
            [[0], [1, 2, 3], [4], [5, 6, 7]],
        )
        self.assertEqual(
            search_result.best_plan.stage_partitions,
            [[0, 1, 2], [3, 4], [5], [6, 7]],
        )
        self.assertLess(
            search_result.best_iteration_time,
            search_result.base_iteration_time,
        )

    def test_auto_partition_matches_best_plan_starts(self):
        search_result = self._optimize_with_common_hardware(
            PipelineScheduleType.INTERLEAVED_1F1B,
            self.HEURISTIC_INTERLEAVED_FORWARD_FLOPS,
            self.HEURISTIC_INTERLEAVED_BACKWARD_FLOPS,
            self.HEURISTIC_INTERLEAVED_COMMUNICATION_VOLUME,
            pp_group_size=4,
            n_microbatches=8,
        )
        split_points = auto_partition(
            self.HEURISTIC_INTERLEAVED_FORWARD_FLOPS,
            self.HEURISTIC_INTERLEAVED_BACKWARD_FLOPS,
            4,
            n_microbatches=8,
            schedule_type=PipelineScheduleType.INTERLEAVED_1F1B,
            communication_volume=self.HEURISTIC_INTERLEAVED_COMMUNICATION_VOLUME,
            device_compute_throughput=self.COMMON_DEVICE_COMPUTE_THROUGHPUT,
            network_bandwidth=self.COMMON_NETWORK_BANDWIDTH,
        )

        self.assertEqual(
            split_points,
            [stage_layers[0] for stage_layers in search_result.best_plan.stage_partitions],
        )

    def test_optimize_partition_model_layers_for_1f1b(self):
        search_result = self._optimize_with_common_hardware(
            PipelineScheduleType.ONE_F_ONE_B,
            self.HEURISTIC_1F1B_FORWARD_FLOPS,
            self.HEURISTIC_1F1B_BACKWARD_FLOPS,
            self.HEURISTIC_1F1B_COMMUNICATION_VOLUME,
            pp_group_size=4,
            n_microbatches=8,
        )
        self._assert_partition_search_result_valid(search_result)
        self.assertEqual(
            search_result.base_plan.stage_partitions,
            [[0, 1, 2], [3, 4, 5], [6], [7]],
        )
        self.assertEqual(
            search_result.best_plan.stage_partitions,
            [[0, 1, 2, 3], [4, 5], [6], [7]],
        )
        self.assertLess(
            search_result.best_iteration_time,
            search_result.base_iteration_time,
        )

    def test_optimize_partition_model_layers_for_interleaved_1f1b(self):
        search_result = self._optimize_with_common_hardware(
            PipelineScheduleType.INTERLEAVED_1F1B,
            self.HEURISTIC_INTERLEAVED_FORWARD_FLOPS,
            self.HEURISTIC_INTERLEAVED_BACKWARD_FLOPS,
            self.HEURISTIC_INTERLEAVED_COMMUNICATION_VOLUME,
            pp_group_size=4,
            n_microbatches=8,
        )
        self._assert_partition_search_result_valid(search_result)
        self.assertEqual(
            search_result.base_plan.stage_partitions,
            [[0, 1], [2], [3], [4], [5, 6], [7], [8], [9, 10, 11]],
        )
        self.assertEqual(
            search_result.best_plan.stage_partitions,
            [[0, 1], [2], [3], [4], [5, 6], [7], [8, 9], [10, 11]],
        )
        self.assertLess(
            search_result.best_iteration_time,
            search_result.base_iteration_time,
        )

    def test_optimize_partition_model_layers_for_interleaved_zero_bubble(self):
        search_result = self._optimize_with_common_hardware(
            PipelineScheduleType.INTERLEAVED_ZERO_BUBBLE,
            self.HEURISTIC_ZERO_BUBBLE_FORWARD_FLOPS,
            self.HEURISTIC_ZERO_BUBBLE_BACKWARD_FLOPS,
            self.HEURISTIC_ZERO_BUBBLE_COMMUNICATION_VOLUME,
            pp_group_size=4,
            n_microbatches=8,
        )
        self._assert_partition_search_result_valid(search_result)
        self.assertEqual(
            search_result.base_plan.stage_partitions,
            [[0, 1], [2, 3], [4], [5, 6], [7], [8], [9, 10], [11]],
        )
        self.assertEqual(
            search_result.best_plan.stage_partitions,
            [[0, 1], [2, 3], [4], [5, 6], [7], [8], [9], [10, 11]],
        )
        self.assertLess(
            search_result.best_iteration_time,
            search_result.base_iteration_time,
        )


if __name__ == "__main__":
    run_tests()
