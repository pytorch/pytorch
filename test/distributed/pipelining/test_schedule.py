# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
import copy
import csv
import logging
import os
from datetime import timedelta
from unittest.mock import MagicMock, patch

from model_registry import MultiMLP

import torch
import torch.distributed.config as dist_config
from torch._dynamo import OptimizedModule
from torch.distributed.pipelining import (
    analyze_pipeline_activation_liveness,
    PipelineActivationLiveness,
    Schedule1F1B,
    ScheduleDualPipeV,
    ScheduleGPipe,
    ScheduleInterleaved1F1B,
    ScheduleInterleavedZeroBubble,
    ScheduleLoopedBFS,
    ScheduleZBVZeroBubble,
)
from torch.distributed.pipelining._p2p import (
    _build_p2p_edge_groups,
    _directed_edge_split_rounds,
    _logical_edge_preconnect_rounds,
    _PP_EDGE_GROUP_CACHE,
    _preconnect_shared_p2p_edges,
    _stage_rank_assignment,
)
from torch.distributed.pipelining._recv_buffers import _RecvInfo
from torch.distributed.pipelining._utils import (
    _TensorMeta,
    generate_stage_to_rank_mapping,
    InferenceMode,
    PipeliningMetadataError,
)
from torch.distributed.pipelining.schedules import (
    _Action,
    _add_reduce_grad,
    _add_send_recv,
    _add_unshard_reshard,
    _batch_p2p,
    _build_recv_ops,
    _defer_recv_ops,
    _format_pipeline_order,
    _merge_bw,
    _PipelineSchedule,
    _PipelineScheduleRuntime,
    _simulate_comms_compute,
    _validate_schedule,
    B,
    F,
    get_schedule_class,
    I,
    OVERLAP_F_B,
    PipelineScheduleMulti,
    PipelineScheduleSingle,
    RECV_B,
    RECV_F,
    RESHARD,
    SEND_B,
    UNSHARD,
    W,
)
from torch.distributed.pipelining.stage import _PipelineStageBase, PipelineStage
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    check_leaked_tensors,
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    subtest,
    TestCase,
)
from torch.testing._internal.distributed.fake_pg import FakeStore


ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

logger = logging.getLogger(__name__)
torch.manual_seed(0)


class P2PInitializationTest(TestCase):
    def test_schedule_builds_and_preconnects_one_shared_edge_group_map(self):
        parent = MagicMock()
        assignment = dict(enumerate((0, 1, 2, 3, 0, 1, 2, 3)))
        stages = []
        for stage_index in (0, 4):
            stage = MockPipelineStage(
                num_stages=8,
                group_size=4,
                group_rank=0,
                group=parent,
            )
            stage.stage_index = stage_index
            stage.device = torch.device("cpu")
            stage.p2p_per_edge = True
            stage.stage_index_to_group_rank = assignment
            stages.append(stage)

        groups = {(0, 1): MagicMock(), (1, 0): MagicMock()}
        split_rounds = (((0, 1),), ((1, 0),))
        with (
            patch(
                "torch.distributed.pipelining.schedules.dist.get_backend",
                return_value="gloo",
            ),
            patch(
                "torch.distributed.pipelining.schedules.dist.all_reduce"
            ) as initialize_parent,
            patch(
                "torch.distributed.pipelining.schedules._build_p2p_edge_groups",
                return_value=(groups, split_rounds),
            ) as build_groups,
            patch(
                "torch.distributed.pipelining.schedules._preconnect_p2p_edge_groups"
            ) as preconnect_groups,
        ):
            _PipelineSchedule._initialize_pipeline_distributed_state(
                MagicMock(), stages, has_backward=True, initialize_p2p=True
            )

        initialize_parent.assert_called_once()
        build_groups.assert_called_once_with(parent, assignment, torch.device("cpu"))
        preconnect_groups.assert_called_once_with(
            parent, groups, split_rounds, torch.device("cpu")
        )
        self.assertTrue(all(stage._p2p_edge_groups is groups for stage in stages))

    def test_per_edge_setup_requires_consistent_local_stage_metadata(self):
        parent = MagicMock()
        assignment = {0: 0, 1: 1}

        def make_stage(stage_index: int) -> MockPipelineStage:
            stage = MockPipelineStage(
                num_stages=2,
                group_size=2,
                group_rank=0,
                group=parent,
            )
            stage.stage_index = stage_index
            stage.device = torch.device("cpu")
            stage.p2p_per_edge = True
            stage.stage_index_to_group_rank = assignment
            return stage

        stages = [make_stage(0), make_stage(1)]
        stages[1].stage_index_to_group_rank = {0: 1, 1: 0}
        with self.assertRaisesRegex(ValueError, "stage-to-rank assignment"):
            _PipelineSchedule._initialize_pipeline_distributed_state(
                MagicMock(), stages, has_backward=True, initialize_p2p=True
            )

        stages = [make_stage(0), make_stage(1)]
        stages[1].device = torch.device("cuda", 1)
        with self.assertRaisesRegex(ValueError, "use one device"):
            _PipelineSchedule._initialize_pipeline_distributed_state(
                MagicMock(), stages, has_backward=True, initialize_p2p=True
            )

    def test_p2p_initialization_state_is_independent_from_metadata(self):
        stage = MockPipelineStage(num_stages=1, group_size=1)
        stage.stage_index = 0
        stage.device = torch.device("cpu")
        stage._prepare_forward_infra = MagicMock(return_value=None)
        schedule = ScheduleGPipe(stage, n_microbatches=1)

        with patch.object(
            schedule,
            "_initialize_pipeline_distributed_state",
        ) as initialize_distributed_state:
            schedule._initialize_stage((), {})
            schedule._stage_forward_initialized = False
            schedule._initialize_stage((), {})

        self.assertEqual(
            [call.args[2] for call in initialize_distributed_state.call_args_list],
            [True, False],
        )

    def test_loop_assignment_uses_four_disjoint_split_rounds(self):
        assignment = _stage_rank_assignment(
            {stage: stage % 4 for stage in range(8)}, group_size=4
        )
        self.assertEqual(
            _directed_edge_split_rounds(assignment),
            (
                ((0, 1), (2, 3)),
                ((1, 0), (3, 2)),
                ((0, 3), (1, 2)),
                ((3, 0), (2, 1)),
            ),
        )

    def test_shared_preconnect_rounds_preserve_repeated_logical_edges(self):
        assignment = (0, 1, 2, 3, 0, 1, 2, 3)

        rounds = _logical_edge_preconnect_rounds(assignment)

        self.assertEqual(
            rounds,
            (
                ((0, 1), (2, 3)),
                ((1, 2), (3, 4)),
                ((4, 5), (6, 7)),
                ((5, 6),),
            ),
        )
        self.assertTrue(
            all(
                len({assignment[stage] for edge in round_edges for stage in edge})
                == 2 * len(round_edges)
                for round_edges in rounds
            )
        )

    def test_shared_preconnect_exercises_raw_path(self):
        parent = MagicMock()
        raw_work = MagicMock()
        with (
            patch(
                "torch.distributed.pipelining._p2p.dist.get_world_size",
                return_value=2,
            ),
            patch(
                "torch.distributed.pipelining._p2p.dist.get_rank",
                return_value=0,
            ),
            patch(
                "torch.distributed.pipelining._p2p.dist.isend",
                return_value=raw_work,
            ) as isend,
            patch("torch.distributed.pipelining._p2p.dist.all_reduce") as sync,
        ):
            _preconnect_shared_p2p_edges(
                parent,
                {0: 0, 1: 1},
                {0: torch.device("cpu")},
                torch.device("cpu"),
            )

        isend.assert_called_once()
        self.assertIs(isend.call_args.kwargs["group"], parent)
        self.assertEqual(isend.call_args.kwargs["group_dst"], 1)
        raw_work.wait.assert_called_once()
        sync.assert_called_once()

    def test_v_assignment_excludes_same_rank_turn_and_wraparound(self):
        assignment = _stage_rank_assignment(
            dict(enumerate((0, 1, 2, 3, 3, 2, 1, 0))),
            group_size=4,
        )

        rounds = _directed_edge_split_rounds(assignment)
        edges = {edge for round_edges in rounds for edge in round_edges}
        self.assertEqual(
            edges,
            {(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)},
        )
        self.assertNotIn((0, 3), edges)
        self.assertNotIn((3, 3), edges)
        self.assertTrue(
            all(
                len({rank for edge in round_edges for rank in edge})
                == 2 * len(round_edges)
                for round_edges in rounds
            )
        )

    def test_assignment_requires_contiguous_valid_stage_mapping(self):
        with self.assertRaisesRegex(ValueError, "contiguous indices"):
            _stage_rank_assignment({0: 0, 2: 1}, group_size=2)
        with self.assertRaisesRegex(ValueError, "outside"):
            _stage_rank_assignment({0: 0, 1: 2}, group_size=2)

    def test_edge_groups_inherit_timeout_and_filter_only_mixed_backends(self):
        cases = (
            ("gloo", torch.device("cpu"), None),
            ("cpu:gloo,cuda:nccl", torch.device("cuda"), "cuda:nccl"),
            ("cpu:gloo,cuda:nccl", torch.device("cpu"), None),
        )
        for backend_config, device, expected_filter in cases:
            with self.subTest(backend_config=backend_config, device=device):
                timeout = timedelta(seconds=17)
                store = FakeStore()
                torch.distributed.init_process_group(
                    backend="fake",
                    rank=0,
                    world_size=2,
                    store=store,
                    timeout=timeout,
                )
                parent = torch.distributed.distributed_c10d._get_default_group()
                backend = MagicMock()
                backend.supports_splitting = True
                backend.options._timeout = timeout
                try:
                    with (
                        patch.object(
                            torch.distributed, "get_backend", return_value="gloo"
                        ),
                        patch.object(
                            torch.distributed,
                            "get_backend_config",
                            return_value=backend_config,
                        ),
                        patch.object(
                            torch.distributed.ProcessGroup,
                            "_get_backend",
                            return_value=backend,
                        ),
                        patch.object(
                            torch.distributed, "split_group", return_value=parent
                        ) as split_group,
                        patch(
                            "torch.distributed.pipelining._p2p."
                            "_initialize_additional_parent_backends"
                        ) as initialize_additional,
                    ):
                        _build_p2p_edge_groups(parent, {0: 0, 1: 1}, device)

                    self.assertEqual(split_group.call_count, 2)
                    for call in split_group.call_args_list:
                        self.assertEqual(call.kwargs["backend"], expected_filter)
                        self.assertEqual(call.kwargs["timeout"], timeout)
                    if backend_config != "gloo" and device.type == "cpu":
                        initialize_additional.assert_called_once_with(
                            parent,
                            {"cpu": "gloo", "cuda": "nccl"},
                            "gloo",
                        )
                    else:
                        initialize_additional.assert_not_called()
                finally:
                    _PP_EDGE_GROUP_CACHE.pop(parent, None)
                    torch.distributed.destroy_process_group()

    def test_edge_groups_validate_split_support_before_reading_options(self):
        class UnsupportedBackend:
            supports_splitting = False

            @property
            def options(self):
                raise AssertionError("options must not be read")

        store = FakeStore()
        torch.distributed.init_process_group(
            backend="fake", rank=0, world_size=2, store=store
        )
        parent = torch.distributed.distributed_c10d._get_default_group()
        try:
            with (
                patch.object(torch.distributed, "get_backend", return_value="gloo"),
                patch.object(
                    torch.distributed,
                    "get_backend_config",
                    return_value="cpu:gloo",
                ),
                patch.object(
                    torch.distributed.ProcessGroup,
                    "_get_backend",
                    return_value=UnsupportedBackend(),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "support split_group"):
                    _build_p2p_edge_groups(parent, {0: 0, 1: 1}, torch.device("cpu"))
        finally:
            torch.distributed.destroy_process_group()

    def test_edge_group_cache_distinguishes_stage_device(self):
        store = FakeStore()
        torch.distributed.init_process_group(
            backend="fake", rank=0, world_size=2, store=store
        )
        parent = torch.distributed.distributed_c10d._get_default_group()
        backend = MagicMock()
        backend.supports_splitting = True
        backend.options._timeout = timedelta(seconds=17)
        try:
            with (
                patch.object(torch.distributed, "get_backend", return_value="gloo"),
                patch.object(
                    torch.distributed,
                    "get_backend_config",
                    return_value="cpu:gloo,cuda:nccl",
                ),
                patch.object(
                    torch.distributed.ProcessGroup,
                    "_get_backend",
                    return_value=backend,
                ),
                patch.object(
                    torch.distributed, "split_group", return_value=parent
                ) as split_group,
                patch(
                    "torch.distributed.pipelining._p2p."
                    "_initialize_additional_parent_backends"
                ),
            ):
                _build_p2p_edge_groups(parent, {0: 0, 1: 1}, torch.device("cpu"))
                _build_p2p_edge_groups(parent, {0: 0, 1: 1}, torch.device("cuda"))
                _build_p2p_edge_groups(parent, {0: 0, 1: 1}, torch.device("cpu"))

            self.assertEqual(
                [call.kwargs["backend"] for call in split_group.call_args_list],
                [None, None, "cuda:nccl", "cuda:nccl"],
            )
        finally:
            _PP_EDGE_GROUP_CACHE.pop(parent, None)
            torch.distributed.destroy_process_group()

    def test_fake_edge_groups_do_not_retain_their_parent(self):
        store = FakeStore()
        torch.distributed.init_process_group(
            backend="fake", rank=0, world_size=2, store=store
        )
        parent = torch.distributed.distributed_c10d._get_default_group()
        try:
            groups, _ = _build_p2p_edge_groups(
                parent, {0: 0, 1: 1}, torch.device("cpu")
            )
            self.assertEqual(set(groups), {(0, 1), (1, 0)})
            self.assertNotIn(parent, _PP_EDGE_GROUP_CACHE)
        finally:
            torch.distributed.destroy_process_group()

    def test_edge_groups_delegate_split_policy_to_torchcomms(self):
        store = FakeStore()
        torch.distributed.init_process_group(
            backend="fake", rank=0, world_size=2, store=store
        )
        parent = torch.distributed.distributed_c10d._get_default_group()
        try:
            with (
                patch.object(torch.distributed, "get_backend", return_value="gloo"),
                patch.object(
                    torch.distributed,
                    "get_backend_config",
                    return_value="gloo",
                ),
                patch.object(
                    torch.distributed.distributed_c10d,
                    "_use_torchcomms_enabled",
                    return_value=True,
                ),
                patch.object(
                    torch.distributed.ProcessGroup,
                    "_get_backend",
                    side_effect=AssertionError("native backend must not be read"),
                ),
                patch.object(
                    torch.distributed, "split_group", return_value=parent
                ) as split_group,
            ):
                _build_p2p_edge_groups(parent, {0: 0, 1: 1}, torch.device("cpu"))

            self.assertEqual(split_group.call_count, 2)
            for call in split_group.call_args_list:
                self.assertIsNone(call.kwargs["backend"])
                self.assertIsNone(call.kwargs["timeout"])
        finally:
            _PP_EDGE_GROUP_CACHE.pop(parent, None)
            torch.distributed.destroy_process_group()


class MockPipelineStage(_PipelineStageBase):
    def __init__(self, *args, **kwargs):
        # Mock the necessary attributes
        self.submod = None
        self.num_stages = kwargs.get("num_stages", 1)
        self.group_size = kwargs.get("group_size", 1)
        self.group_rank = kwargs.get("group_rank", 0)
        self.group = kwargs.get("group")
        self.p2p_per_edge = False

    def _create_grad_recv_info(self, *args, **kwargs):
        return None

    def _prepare_forward_infra(self, n_microbatches):
        pass

    def _prepare_backward_infra(self, n_microbatches):
        pass


def _make_adjacency_stage(
    stage_index, num_stages, group_rank, args_recv_info, act_send_info
):
    """Factory for a minimal mock stage used by adjacency-guard tests."""

    class _AdjacencyTestStage(MockPipelineStage):
        def __init__(self):
            super().__init__(
                num_stages=num_stages,
                group_size=num_stages,
                group_rank=group_rank,
            )
            self.stage_index = stage_index
            self.device = "cpu"

        def _prepare_forward_infra(self, *args, **kwargs):
            self.args_recv_info = args_recv_info
            self.act_send_info = act_send_info
            return tuple()

        def _prepare_backward_infra(self, *args, **kwargs):
            return None

        def clear_runtime_states(self):
            return None

        def get_fwd_recv_ops(self, mb_index):
            return []

        def get_fwd_send_ops(self, mb_index):
            return []

        def get_bwd_recv_ops(self, mb_index):
            return []

        def get_bwd_send_ops(self, mb_index):
            return []

    return _AdjacencyTestStage()


def _run_adjacency_validation(stage, num_stages):
    """Run one step of PipelineScheduleMulti to trigger adjacency validation."""
    schedule = PipelineScheduleMulti([stage], n_microbatches=1)
    schedule.pipeline_order = {i: [None] for i in range(num_stages)}
    schedule.step()


def _max_live_closed_intervals(intervals: list[tuple[int, int]]) -> int:
    """Return peak overlap for inclusive integer intervals."""
    events: list[tuple[int, int]] = []
    for start, release in intervals:
        events.extend(((start, 1), (release + 1, -1)))
    live = peak = 0
    for _, delta in sorted(events):
        live += delta
        peak = max(peak, live)
    return peak


def _activation_intervals(
    actions: list[_Action], stage_indices: tuple[int, ...]
) -> dict[tuple[int, int], tuple[int, int]]:
    """Derive activation lifetimes without using the production analyzer.

    This oracle independently checks the minimum slot count. It intentionally
    shares only the contract that forward starts an activation lifetime and
    full- or weight-backward releases it.
    """
    starts: dict[tuple[int, int], int] = {}
    releases: dict[tuple[int, int], int] = {}

    def visit(action: _Action, position: int) -> None:
        if action.sub_actions is not None:
            for sub_action in action.sub_actions:
                visit(sub_action, position)
            return
        if action.stage_index not in stage_indices or action.microbatch_index is None:
            return
        key = (action.stage_index, action.microbatch_index)
        if action.computation_type == F:
            starts[key] = position
        elif action.computation_type in (B, W):
            releases[key] = position

    for position, action in enumerate(actions):
        visit(action, position)
    return {key: (starts[key], releases[key]) for key in starts}


class ScheduleTest(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_stage_recv_buffer_allocation_and_consumption(self):
        stage = MockPipelineStage(num_stages=3, group_size=1, group_rank=0)
        stage.stage_index = 1
        stage.device = torch.device("cpu")
        info = _RecvInfo(
            "activation", source=0, tensor_meta=_TensorMeta.from_tensor(torch.ones(2))
        )

        with (
            patch.object(stage, "_resolve_peer_global_rank", return_value=0),
            patch("torch.distributed.pipelining.stage.dist.P2POp") as p2p,
        ):
            ops = stage._get_recv_ops((info,))

        self.assertEqual(len(ops), 1)
        self.assertIsNotNone(info.buffer)
        self.assertIs(p2p.call_args.args[1], info.buffer)

        allocated = info.take_buffer()
        self.assertIsNotNone(allocated)
        self.assertIsNone(info.buffer)

    def test_recv_info_rejects_invalid_state_transitions(self):
        info = _RecvInfo(
            "activation", source=0, tensor_meta=_TensorMeta.from_tensor(torch.ones(2))
        )
        info.allocate_buffer(torch.device("cpu"))
        with self.assertRaisesRegex(
            PipeliningMetadataError, "incomplete pipeline step"
        ):
            info.allocate_buffer(torch.device("cpu"))

        info.take_buffer()
        with self.assertRaisesRegex(PipeliningMetadataError, "has not been set"):
            info.take_buffer()

        info.set_buffer(torch.ones(2))
        with self.assertRaisesRegex(
            PipeliningMetadataError, "incomplete pipeline step"
        ):
            info.set_buffer(torch.ones(2))

        missing_grad = _RecvInfo("grad", source=2, tensor_meta=None)
        with self.assertRaisesRegex(PipeliningMetadataError, "no tensor metadata"):
            missing_grad.allocate_buffer(torch.device("cpu"))
        with self.assertRaisesRegex(PipeliningMetadataError, "expects no gradient"):
            missing_grad.set_buffer(torch.ones(2))

    def test_stage_recv_construction_is_transactional(self):
        stage = MockPipelineStage(num_stages=3, group_size=1, group_rank=0)
        stage.stage_index = 1
        stage.device = torch.device("cpu")
        meta = _TensorMeta.from_tensor(torch.ones(2))
        valid_before_invalid = _RecvInfo("valid", source=0, tensor_meta=meta)
        invalid_source = _RecvInfo("invalid", source=None, tensor_meta=meta)
        with (
            patch.object(stage, "_resolve_peer_global_rank", return_value=0),
            self.assertRaisesRegex(AssertionError, "info.source"),
        ):
            stage._get_recv_ops((valid_before_invalid, invalid_source))
        self.assertIsNone(valid_before_invalid.buffer)
        self.assertIsNone(invalid_source.buffer)

        first = _RecvInfo("first", source=0, tensor_meta=meta)
        second = _RecvInfo("second", source=0, tensor_meta=meta)
        with (
            patch.object(stage, "_resolve_peer_global_rank", return_value=0),
            patch(
                "torch.distributed.pipelining.stage.dist.P2POp",
                side_effect=[MagicMock(), RuntimeError("construction failed")],
            ),
            self.assertRaisesRegex(RuntimeError, "construction failed"),
        ):
            stage._get_recv_ops((first, second))
        self.assertIsNone(first.buffer)
        self.assertIsNone(second.buffer)

        with (
            patch.object(stage, "_resolve_peer_global_rank", return_value=0),
            patch(
                "torch.distributed.pipelining._recv_buffers._make_tensor_from_meta",
                side_effect=[torch.empty(2), RuntimeError("allocation failed")],
            ),
            patch("torch.distributed.pipelining.stage.dist.P2POp"),
            self.assertRaisesRegex(RuntimeError, "allocation failed"),
        ):
            stage._get_recv_ops((first, second))
        self.assertIsNone(first.buffer)
        self.assertIsNone(second.buffer)

        missing_edge = _RecvInfo("activation", source=0, tensor_meta=meta)
        stage.args_recv_info = {2: (missing_edge,)}
        stage.p2p_per_edge = True
        stage.stage_index_to_group_rank = {0: 0, 1: 1}
        stage._p2p_edge_groups = {}
        with (
            patch.object(stage, "_resolve_peer_global_rank", return_value=0),
            self.assertRaisesRegex(RuntimeError, "Missing directed pipeline"),
        ):
            stage.get_fwd_recv_ops(2)
        self.assertIsNone(missing_edge.buffer)

    def test_same_rank_recv_assignment_is_transactional(self):
        stage = MockPipelineStage(num_stages=3, group_size=1, group_rank=0)
        stage.stage_index = 1
        stage.device = torch.device("cpu")
        stage.has_backward = True
        meta = _TensorMeta.from_tensor(torch.ones(2))

        fwd_infos = tuple(
            _RecvInfo(name, source=0, tensor_meta=meta) for name in ("fwd_0", "fwd_1")
        )
        stage.args_recv_info = {0: fwd_infos}
        with self.assertRaisesRegex(AssertionError, "expected tensor values"):
            stage.set_local_fwd_input((torch.ones(2), "invalid"), 0)
        self.assertTrue(all(info.buffer is None for info in fwd_infos))

        occupied = torch.ones(2)
        fwd_infos[1].set_buffer(occupied)
        with self.assertRaisesRegex(
            PipeliningMetadataError, "incomplete pipeline step"
        ):
            stage.set_local_fwd_input((torch.ones(2), torch.ones(2)), 0)
        self.assertIsNone(fwd_infos[0].buffer)
        self.assertIs(fwd_infos[1].buffer, occupied)
        fwd_infos[1].take_buffer()

        bwd_infos = tuple(
            _RecvInfo(name, source=2, tensor_meta=meta) for name in ("bwd_0", "bwd_1")
        )
        stage.grad_recv_info = {0: bwd_infos}
        with self.assertRaisesRegex(AssertionError, "expected tensor values"):
            stage.set_local_bwd_input((None, "invalid"), 0)
        self.assertTrue(all(info.buffer is None for info in bwd_infos))

    def test_timestep_recv_validation_releases_prior_batches(self):
        meta = _TensorMeta.from_tensor(torch.ones(2))
        valid_stage = MockPipelineStage(num_stages=3, group_size=1, group_rank=0)
        valid_stage.stage_index = 1
        valid_stage.device = torch.device("cpu")
        valid_info = _RecvInfo("valid", source=0, tensor_meta=meta)
        valid_stage.args_recv_info = {0: (valid_info,)}

        invalid_stage = MockPipelineStage(num_stages=3, group_size=1, group_rank=0)
        invalid_stage.stage_index = 2
        invalid_stage.device = torch.device("cpu")
        invalid_info = _RecvInfo("invalid", source=None, tensor_meta=meta)
        invalid_stage.args_recv_info = {0: (invalid_info,)}

        with (
            patch.object(valid_stage, "_resolve_peer_global_rank", return_value=0),
            patch("torch.distributed.pipelining.stage.dist.P2POp"),
            self.assertRaisesRegex(AssertionError, "info.source"),
        ):
            _build_recv_ops([(valid_stage, True, 0), (invalid_stage, True, 0)])

        self.assertIsNone(valid_info.buffer)
        self.assertIsNone(invalid_info.buffer)

    def test_pipeline_activation_liveness_reuses_completed_slots(self):
        stage = MockPipelineStage(group_size=1, num_stages=1)
        stage.stage_index = 0
        schedule = PipelineScheduleMulti([stage], n_microbatches=4)
        # Input backward does not release microbatch 0's activation, while its
        # following weight backward does.
        schedule.pipeline_order = {
            0: [
                _Action(0, F, 0),
                _Action(0, F, 1),
                _Action(0, I, 0),
                _Action(0, F, 2),
                _Action(0, W, 0),
                _Action(0, F, 3),
                _Action(0, I, 1),
                _Action(0, W, 1),
                _Action(0, I, 2),
                _Action(0, W, 2),
                _Action(0, I, 3),
                _Action(0, W, 3),
            ]
        }

        plan = analyze_pipeline_activation_liveness(
            schedule,
            pp_rank=0,
            stage_indices=(0,),
            granularity="stage_microbatch",
        )

        self.assertIsInstance(plan, PipelineActivationLiveness)
        self.assertEqual(plan.pp_rank, 0)
        self.assertEqual(plan.num_slots, 3)
        self.assertEqual(
            plan.slot_by_stage_and_microbatch,
            {(0, 0): 0, (0, 1): 1, (0, 2): 2, (0, 3): 0},
        )
        self.assertEqual(plan.get_activation_lifetime(0, 0), (0, 4))
        self.assertEqual(plan.get_activation_lifetime(0, 2), (3, 9))
        with self.assertRaisesRegex(ValueError, "No activation slot exists"):
            plan.slot_for(1, 0)
        with self.assertRaisesRegex(ValueError, "No activation lifetime exists"):
            plan.get_activation_lifetime(1, 0)
        self.assertEqual(
            plan.num_slots,
            _max_live_closed_intervals([(0, 4), (1, 7), (3, 9), (5, 11)]),
        )
        self.assertEqual(
            plan,
            analyze_pipeline_activation_liveness(
                schedule,
                pp_rank=0,
                stage_indices=(0,),
                granularity="stage_microbatch",
            ),
        )

    def test_pipeline_activation_liveness_granularity(self):
        stages = [MockPipelineStage(group_size=1, num_stages=3) for _ in range(2)]
        stages[0].stage_index = 0
        stages[1].stage_index = 2
        schedule = PipelineScheduleMulti(stages, n_microbatches=3)
        schedule.pipeline_order = {
            0: [
                _Action(0, F, 0),
                _Action(2, F, 0),
                _Action(0, F, 1),
                _Action(2, B, 0),
                _Action(0, B, 0),
                _Action(2, F, 1),
                _Action(2, B, 1),
                _Action(0, B, 1),
                _Action(0, F, 2),
                _Action(2, F, 2),
                _Action(2, B, 2),
                _Action(0, B, 2),
            ]
        }

        stage_plan = analyze_pipeline_activation_liveness(
            schedule,
            pp_rank=0,
            stage_indices=(0, 2),
            granularity="stage_microbatch",
        )
        microbatch_plan = analyze_pipeline_activation_liveness(
            schedule,
            pp_rank=0,
            stage_indices=(0, 2),
            granularity="microbatch",
        )

        self.assertEqual(stage_plan.num_slots, 3)
        self.assertEqual(
            stage_plan.num_slots,
            _max_live_closed_intervals(
                [(0, 4), (1, 3), (2, 7), (5, 6), (8, 11), (9, 10)]
            ),
        )
        self.assertEqual(
            [
                stage_plan.slot_for(0, 0),
                stage_plan.slot_for(2, 0),
                stage_plan.slot_for(0, 1),
                stage_plan.slot_for(2, 1),
                stage_plan.slot_for(0, 2),
                stage_plan.slot_for(2, 2),
            ],
            [0, 1, 2, 0, 0, 1],
        )
        self.assertEqual(stage_plan.get_activation_lifetime(0, 0), (0, 4))
        self.assertEqual(stage_plan.get_activation_lifetime(2, 0), (1, 3))
        self.assertEqual(stage_plan.get_activation_lifetime(0, 1), (2, 7))
        self.assertEqual(stage_plan.get_activation_lifetime(2, 1), (5, 6))
        self.assertEqual(microbatch_plan.num_slots, 2)
        actions = schedule.pipeline_order[0]
        intervals_by_stage = {
            stage_index: _activation_intervals(actions, (stage_index,))
            for stage_index in (0, 2)
        }
        merged_intervals = [
            (
                min(
                    intervals_by_stage[stage][(stage, microbatch)][0]
                    for stage in (0, 2)
                ),
                max(
                    intervals_by_stage[stage][(stage, microbatch)][1]
                    for stage in (0, 2)
                ),
            )
            for microbatch in range(3)
        ]
        self.assertEqual(
            microbatch_plan.num_slots,
            _max_live_closed_intervals(merged_intervals),
        )
        self.assertEqual(microbatch_plan.slot_for(0, 0), 0)
        self.assertEqual(microbatch_plan.slot_for(2, 0), 0)
        self.assertEqual(microbatch_plan.slot_for(0, 1), 1)
        self.assertEqual(microbatch_plan.slot_for(2, 1), 1)
        self.assertEqual(microbatch_plan.slot_for(0, 2), 0)
        self.assertEqual(microbatch_plan.slot_for(2, 2), 0)
        self.assertEqual(microbatch_plan.get_activation_lifetime(0, 0), (0, 4))
        self.assertEqual(microbatch_plan.get_activation_lifetime(2, 0), (0, 4))
        self.assertEqual(microbatch_plan.get_activation_lifetime(0, 1), (2, 7))
        self.assertEqual(microbatch_plan.get_activation_lifetime(2, 1), (2, 7))

    def test_pipeline_activation_liveness_overlap_is_order_independent(self):
        stage = MockPipelineStage(group_size=1, num_stages=1)
        stage.stage_index = 0
        for sub_actions in (
            (_Action(0, I, 0), _Action(0, W, 0)),
            (_Action(0, W, 0), _Action(0, I, 0)),
        ):
            schedule = PipelineScheduleMulti([stage], n_microbatches=1)
            schedule.pipeline_order = {
                0: [
                    _Action(0, F, 0),
                    _Action(-1, OVERLAP_F_B, None, sub_actions),
                ]
            }

            plan = analyze_pipeline_activation_liveness(
                schedule,
                pp_rank=0,
                stage_indices=(0,),
                granularity="stage_microbatch",
            )
            self.assertEqual(plan.num_slots, 1)
            self.assertEqual(plan.slot_for(0, 0), 0)

        point_schedule = PipelineScheduleMulti([stage], n_microbatches=1)
        point_schedule.pipeline_order = {
            0: [
                _Action(
                    -1,
                    OVERLAP_F_B,
                    None,
                    (_Action(0, F, 0), _Action(0, B, 0)),
                )
            ]
        }
        point_plan = analyze_pipeline_activation_liveness(
            point_schedule,
            pp_rank=0,
            stage_indices=(0,),
            granularity="stage_microbatch",
        )
        self.assertEqual(point_plan.num_slots, 1)

    def test_pipeline_activation_liveness_with_dual_pipe_v(self):
        group_size, num_stages, num_microbatches = 2, 4, 4
        stages = [
            MockPipelineStage(group_size=group_size, num_stages=num_stages)
            for _ in range(2)
        ]
        stages[0].stage_index = 0
        stages[1].stage_index = 3
        schedule = ScheduleDualPipeV(stages, num_microbatches)
        for rank in range(group_size):
            stage_indices = tuple(
                stage_index
                for stage_index, stage_rank in schedule.stage_index_to_group_rank.items()
                if stage_rank == rank
            )
            plan = analyze_pipeline_activation_liveness(
                schedule,
                pp_rank=rank,
                stage_indices=stage_indices,
                granularity="stage_microbatch",
            )
            intervals = _activation_intervals(
                schedule.pipeline_order_with_comms[rank], stage_indices
            )
            self.assertEqual(
                plan.num_slots,
                _max_live_closed_intervals(list(intervals.values())),
            )
            self.assertEqual(
                set(plan.slot_by_stage_and_microbatch),
                {
                    (stage_index, microbatch_index)
                    for stage_index in stage_indices
                    for microbatch_index in range(num_microbatches)
                },
            )
            for key, lifetime in intervals.items():
                self.assertEqual(plan.get_activation_lifetime(*key), lifetime)
            interval_items = list(intervals.items())
            for index, (left_key, (left_start, left_release)) in enumerate(
                interval_items
            ):
                for right_key, (right_start, right_release) in interval_items[
                    index + 1 :
                ]:
                    if left_start <= right_release and right_start <= left_release:
                        self.assertNotEqual(
                            plan.slot_for(*left_key),
                            plan.slot_for(*right_key),
                        )

    @parametrize(
        "granularity,stage_indices,pp_rank,actions,error",
        [
            subtest(("layer", (0,), 0, (), "Unsupported"), name="granularity"),
            subtest(
                ("stage_microbatch", (), 0, (), "must not be empty"),
                name="empty_stages",
            ),
            subtest(
                ("stage_microbatch", (0, 0), 0, (), "must be unique"),
                name="duplicate_stages",
            ),
            subtest(
                ("stage_microbatch", (0,), 1, (), "not present"),
                name="missing_rank",
            ),
            subtest(
                ("stage_microbatch", (0,), 0, (_Action(0, F, None),), "no microbatch"),
                name="missing_microbatch",
            ),
            subtest(
                ("stage_microbatch", (0,), 0, (_Action(0, F, 1),), "outside"),
                name="out_of_range_microbatch",
            ),
            subtest(
                (
                    "stage_microbatch",
                    (0,),
                    0,
                    (_Action(0, F, 0), _Action(0, F, 0)),
                    "multiple forwards",
                ),
                name="duplicate_forward",
            ),
            subtest(
                (
                    "stage_microbatch",
                    (0,),
                    0,
                    (_Action(0, F, 0), _Action(0, I, 0), _Action(0, I, 0)),
                    "multiple input backwards",
                ),
                name="duplicate_input_backward",
            ),
            subtest(
                (
                    "stage_microbatch",
                    (0,),
                    0,
                    (_Action(0, F, 0), _Action(0, B, 0), _Action(0, W, 0)),
                    "multiple release actions",
                ),
                name="duplicate_release",
            ),
            subtest(
                ("stage_microbatch", (0,), 0, (_Action(0, B, 0),), "Forward actions"),
                name="missing_forward",
            ),
            subtest(
                (
                    "stage_microbatch",
                    (0,),
                    0,
                    (_Action(0, F, 0),),
                    "Backward release actions",
                ),
                name="missing_release",
            ),
            subtest(
                (
                    "stage_microbatch",
                    (0,),
                    0,
                    (_Action(0, F, 0), _Action(0, W, 0)),
                    "without input backward",
                ),
                name="weight_without_input_backward",
            ),
            subtest(
                (
                    "stage_microbatch",
                    (0,),
                    0,
                    (_Action(0, B, 0), _Action(0, F, 0)),
                    "ends before its forward",
                ),
                name="release_before_forward",
            ),
        ],
    )
    def test_pipeline_activation_liveness_validation(
        self, granularity, stage_indices, pp_rank, actions, error
    ):
        stage = MockPipelineStage(group_size=1, num_stages=1)
        stage.stage_index = 0
        schedule = PipelineScheduleMulti([stage], n_microbatches=1)
        schedule.pipeline_order = {0: list(actions)}

        with self.assertRaisesRegex(ValueError, error):
            analyze_pipeline_activation_liveness(
                schedule,
                pp_rank=pp_rank,
                stage_indices=stage_indices,
                granularity=granularity,
            )

    def test_get_schedule_class(self):
        # List of all expected schedule names
        schedule_names = [
            "1F1B",
            "1f1b",
            "Interleaved1F1B",
            "INTERLEAVED1F1B",
            "GPipe",
            "LoopedBFS",
            "PipelineScheduleSingle",
            "PipelineScheduleMulti",
        ]

        # Test each schedule name
        for name in schedule_names:
            with self.subTest(name=name):
                schedule_class = get_schedule_class(name)
                self.assertIsNotNone(
                    schedule_class,
                    lambda msg: f"{msg}\nClass for {name} should not be None",
                )
                self.assertTrue(
                    issubclass(schedule_class, _PipelineSchedule),
                    lambda msg: f"{msg}\n{name} should be a subclass of _PipelineSchedule",
                )

        error_case = ["ScheduleThatDoesNotExist"]
        for name in error_case:
            # Test that the original name is included in the error message
            with self.assertRaisesRegex(ValueError, f"{name}"):
                get_schedule_class(name)

    def test_adjacency_guard_rejects_nonadjacent_send(self):
        # Stage 0 sending to stage 2 (skip connection)
        stage = _make_adjacency_stage(0, 3, 0, {0: tuple()}, {0: [2]})
        with self.assertRaisesRegex(RuntimeError, "adjacent-stage communication"):
            _run_adjacency_validation(stage, 3)

    def test_adjacency_guard_rejects_nonadjacent_recv(self):
        # Stage 3 receiving from stage 0 (non-adjacent)
        stage = _make_adjacency_stage(
            3,
            4,
            3,
            {0: (_RecvInfo("x", source=0, tensor_meta=None),)},
            {},
        )
        with self.assertRaisesRegex(RuntimeError, "adjacent-stage communication"):
            _run_adjacency_validation(stage, 4)

    def test_adjacency_guard_allows_adjacent_send(self):
        # Stage 0 -> stage 1 is valid
        stage = _make_adjacency_stage(0, 3, 0, {0: tuple()}, {0: [1]})
        _run_adjacency_validation(stage, 3)

    def test_adjacency_guard_allows_empty_recv_middle_stage(self):
        # Middle stage with no recv is fine (no non-adjacent peers)
        stage = _make_adjacency_stage(1, 3, 1, {0: tuple()}, {0: [2]})
        _run_adjacency_validation(stage, 3)

    @parametrize(
        "ScheduleClass",
        [
            Schedule1F1B,
            ScheduleGPipe,
            ScheduleInterleaved1F1B,
            ScheduleInterleavedZeroBubble,
            ScheduleLoopedBFS,
        ],
    )
    def test_schedule_with_single_stage(self, ScheduleClass):
        """
        Test that schedules with only a single stage work as expected for all schedules.
        """
        store = FakeStore()
        torch.distributed.init_process_group(
            backend="fake", rank=0, world_size=1, store=store
        )
        try:
            d_hid, batch_size = 512, 256
            n_stages = 1
            device = "cpu"
            full_mod = MultiMLP(d_hid, n_layers=n_stages)
            full_mod.to(device)

            x = torch.randn(batch_size, d_hid, device=device)
            ref_mod = copy.deepcopy(full_mod)
            with torch.no_grad():
                y = ref_mod(x)
                # Add a small perturbation
                target = y + torch.randn(batch_size, d_hid, device=device)

            def loss_fn(y, target):
                return torch.nn.functional.cross_entropy(y, target)

            # Run reference
            for _ in range(2):
                ref_mod.zero_grad()
                ref_out = ref_mod(x)
                ref_loss = loss_fn(ref_out, target)
                ref_loss.backward()

            submod_name = "layers.0"
            stage_module = full_mod.get_submodule(submod_name)

            # Create a pipeline stage to wrap that submodule
            num_microbatches = 2
            stages = [
                PipelineStage(
                    stage_module,
                    0,
                    n_stages,
                    device,
                )
            ]

            if issubclass(ScheduleClass, PipelineScheduleSingle):
                stages = stages[0]

            # Attach to a schedule
            schedule = ScheduleClass(
                stages,
                num_microbatches,
                loss_fn=loss_fn,
            )
            # Run
            for _ in range(2):
                # Zero gradients
                stage_module.zero_grad()
                losses = []
                out = schedule.step(x, target=target, losses=losses)

            # Check output
            torch.testing.assert_close(out, ref_out)
            # Check loss
            # Since the reduction used in the loss function above is "mean", we use
            # "mean" here to reduce microbatch losses into a single value too.
            pipe_loss = torch.stack(losses).mean()
            torch.testing.assert_close(pipe_loss, ref_loss)

            # Check gradients
            # Get corresponding submodule from reference model
            ref_submod = ref_mod.get_submodule(submod_name)
            # Check gradients per parameter
            for name, p in stage_module.named_parameters():
                ref_p = ref_submod.get_parameter(name)
                try:
                    torch.testing.assert_close(p.grad, ref_p.grad, rtol=1e-5, atol=4e-5)
                except AssertionError:
                    print(f"Gradient test failed for {name}: {p.grad} vs {ref_p.grad}")
                    raise

        finally:
            torch.distributed.destroy_process_group()

    @parametrize(
        "ScheduleClass",
        [
            Schedule1F1B,
            ScheduleGPipe,
            ScheduleInterleaved1F1B,
            ScheduleInterleavedZeroBubble,
            ScheduleLoopedBFS,
        ],
    )
    def test_schedule_with_pre_split_inputs(self, ScheduleClass):
        """
        Test that schedules can consume pre-split microbatch args, kwargs, and target.
        """
        store = FakeStore()
        torch.distributed.init_process_group(
            backend="fake", rank=0, world_size=1, store=store
        )
        try:
            d_hid, batch_size = 16, 8
            n_stages = 1
            num_microbatches = 2
            device = "cpu"

            class KwargModule(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = torch.nn.Linear(d_hid, d_hid)

                def forward(self, x, y):
                    return self.linear(torch.relu(x + y))

            auto_mod = KwargModule().to(device)
            pre_split_mod = copy.deepcopy(auto_mod)

            x = torch.randn(batch_size, d_hid, device=device)
            y = torch.randn(batch_size, d_hid, device=device)
            target = torch.randn(batch_size, d_hid, device=device)
            loss_fn = torch.nn.MSELoss(reduction="sum")

            def make_schedule(mod):
                stage = PipelineStage(mod, 0, n_stages, device)
                if issubclass(ScheduleClass, PipelineScheduleSingle):
                    stages = stage
                else:
                    stages = [stage]
                return ScheduleClass(
                    stages,
                    num_microbatches,
                    loss_fn=loss_fn,
                    scale_grads=False,
                )

            auto_schedule = make_schedule(auto_mod)
            pre_split_schedule = make_schedule(pre_split_mod)

            auto_losses = []
            auto_out = auto_schedule.step(x, y=y, target=target, losses=auto_losses)

            arg_mbs = [(x_mb,) for x_mb in torch.tensor_split(x, num_microbatches)]
            kwarg_mbs = [
                {"y": y_mb} for y_mb in torch.tensor_split(y, num_microbatches)
            ]
            target_mbs = list(torch.tensor_split(target, num_microbatches))
            pre_split_losses = []
            pre_split_out = pre_split_schedule.step(
                arg_mbs=arg_mbs,
                kwarg_mbs=kwarg_mbs,
                target_mbs=target_mbs,
                losses=pre_split_losses,
            )

            self.assertEqual(pre_split_out, auto_out)
            self.assertEqual(torch.stack(pre_split_losses), torch.stack(auto_losses))

            for (name, pre_split_param), (auto_name, auto_param) in zip(
                pre_split_mod.named_parameters(),
                auto_mod.named_parameters(),
                strict=True,
            ):
                self.assertEqual(name, auto_name)
                self.assertEqual(
                    pre_split_param.grad,
                    auto_param.grad,
                    msg=f"Gradient mismatch for {name}",
                )
        finally:
            torch.distributed.destroy_process_group()

    @parametrize("enabled", [False, True])
    def test_schedule_passes_stage_and_microbatch_indices(self, enabled):
        class IndexModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls: list[tuple[int | None, int | None]] = []

            def forward(self, x, *, scale, stage_idx=None, mb_idx=None):
                self.calls.append((stage_idx, mb_idx))
                return x * scale

        store = FakeStore()
        torch.distributed.init_process_group(
            backend="fake", rank=0, world_size=1, store=store
        )
        try:
            module = IndexModule()
            stage = PipelineStage(module, 0, 1, torch.device("cpu"))
            schedule_kwargs = (
                {"pass_stage_and_microbatch_indices": True} if enabled else {}
            )
            schedule = ScheduleGPipe(
                stage,
                2,
                loss_fn=lambda output, target: (output - target).square().sum(),
                scale_grads=False,
                **schedule_kwargs,
            )
            x = torch.ones(2, requires_grad=True)
            scale = torch.full((2,), 3.0, requires_grad=True)

            self.assertEqual(
                schedule.step(x, scale=scale, target=torch.zeros(2)), x * scale
            )
            expected = [(0, 0), (0, 0), (0, 1)] if enabled else [(None, None)] * 3
            self.assertEqual(module.calls, expected)
            self.assertEqual(x.grad, torch.full_like(x, 18))
            self.assertEqual(scale.grad, torch.full_like(scale, 6))
        finally:
            torch.distributed.destroy_process_group()

    def test_stage_and_microbatch_indices_require_manual_stages(self):
        stage = MockPipelineStage(num_stages=1)
        with self.assertRaisesRegex(ValueError, "manually constructed PipelineStage"):
            ScheduleGPipe(
                stage,
                1,
                pass_stage_and_microbatch_indices=True,
            )

    def test_schedule_pre_split_validation(self):
        store = FakeStore()
        torch.distributed.init_process_group(
            backend="fake", rank=0, world_size=1, store=store
        )
        try:
            d_hid = 4
            device = "cpu"
            x0 = torch.randn(2, d_hid, device=device)
            x1 = torch.randn(2, d_hid, device=device)
            stage = PipelineStage(torch.nn.Identity(), 0, 1, device)
            schedule = ScheduleGPipe(stage, 2)

            with self.assertRaisesRegex(
                ValueError,
                "pass pre-split positional inputs through arg_mbs",
            ):
                schedule.step(x0, arg_mbs=[(x0,), (x1,)])

            with self.assertRaisesRegex(
                ValueError,
                "Unexpected keyword arguments with pre-split inputs: y",
            ):
                schedule.step(y=x0, kwarg_mbs=[{}, {}])

            with self.assertRaisesRegex(
                ValueError,
                "pass pre-split targets through target_mbs",
            ):
                schedule.step(target=x0, target_mbs=[x0, x1])

            with self.assertRaisesRegex(TypeError, "arg_mbs must be a list"):
                schedule.step(arg_mbs=(x0,))

            with self.assertRaisesRegex(ValueError, "Expecting 2 arg_mbs"):
                schedule.step(arg_mbs=[])

            with self.assertRaisesRegex(TypeError, "kwarg_mbs must be a list"):
                schedule.step(
                    arg_mbs=[(x0,), (x1,)],
                    kwarg_mbs={"y": x0},
                )

            with self.assertRaisesRegex(TypeError, "arg_mbs must be a list of tuples"):
                schedule.step(arg_mbs=[x0, x1])

            with self.assertRaisesRegex(TypeError, "kwarg_mbs must be a list of dicts"):
                schedule.step(kwarg_mbs=[x0, x1])

            with self.assertRaisesRegex(ValueError, "Expecting 2 target_mbs"):
                schedule.step(
                    arg_mbs=[(x0,), (x1,)],
                    target_mbs=[x0],
                )

            indexed_schedule = ScheduleGPipe(
                stage,
                2,
                pass_stage_and_microbatch_indices=True,
            )
            for name in ("stage_idx", "mb_idx"):
                for pre_split in (False, True):
                    with (
                        self.subTest(name=name, pre_split=pre_split),
                        self.assertRaisesRegex(ValueError, f"reserves.*{name}"),
                    ):
                        if pre_split:
                            indexed_schedule.step(
                                arg_mbs=[(x0,), (x1,)],
                                kwarg_mbs=[{name: -1}, {}],
                            )
                        else:
                            indexed_schedule.step(x0, **{name: -1})
        finally:
            torch.distributed.destroy_process_group()

    @parametrize(
        "ScheduleClass",
        [
            Schedule1F1B,
            ScheduleGPipe,
            ScheduleInterleaved1F1B,
            ScheduleInterleavedZeroBubble,
            ScheduleLoopedBFS,
        ],
    )
    def test_schedule_eval_then_train(self, ScheduleClass):
        """Test full-batch and pre-split evaluation followed by training."""
        store = FakeStore()
        torch.distributed.init_process_group(
            backend="fake", rank=0, world_size=1, store=store
        )
        d_hid, batch_size = 512, 256
        n_stages = 1
        device = "cpu"
        full_mod = MultiMLP(d_hid, n_layers=n_stages)
        full_mod.to(device)

        x = torch.randn(batch_size, d_hid, device=device)
        target = torch.randn(batch_size, d_hid, device=device)

        def loss_fn(y, target):
            return torch.nn.functional.cross_entropy(y, target)

        submod_name = "layers.0"
        stage_module = full_mod.get_submodule(submod_name)

        # Create a pipeline stage to wrap that submodule
        num_microbatches = 2
        stages = [PipelineStage(stage_module, 0, n_stages, device)]

        if issubclass(ScheduleClass, PipelineScheduleSingle):
            stages = stages[0]

        # Attach to a schedule
        schedule = ScheduleClass(stages, num_microbatches, loss_fn=loss_fn)
        arg_mbs = [(x_mb,) for x_mb in torch.tensor_split(x, num_microbatches)]
        target_mbs = list(torch.tensor_split(target, num_microbatches))

        try:
            stage_module.zero_grad()
            full_losses = []
            full_out = schedule.eval(x, target=target, losses=full_losses)
            pre_split_losses = []
            pre_split_out = schedule.eval(
                arg_mbs=arg_mbs,
                target_mbs=target_mbs,
                losses=pre_split_losses,
            )

            self.assertEqual(pre_split_out, full_out)
            self.assertEqual(torch.stack(pre_split_losses), torch.stack(full_losses))
            for name, parameter in stage_module.named_parameters():
                self.assertIsNone(
                    parameter.grad,
                    msg=f"eval unexpectedly produced a gradient for {name}",
                )

            train_losses = []
            train_out = schedule.step(
                arg_mbs=arg_mbs,
                target_mbs=target_mbs,
                losses=train_losses,
            )
            self.assertEqual(train_out, full_out)
            self.assertEqual(torch.stack(train_losses), torch.stack(full_losses))
            for name, parameter in stage_module.named_parameters():
                self.assertIsNotNone(
                    parameter.grad,
                    msg=f"training did not produce a gradient for {name}",
                )
        finally:
            torch.distributed.destroy_process_group()

    @parametrize("rank", [0, 1])
    @parametrize("per_edge", [False, True])
    def test_fake_pg_cross_rank_uses_static_metadata(self, rank, per_edge):
        """
        With a fake process group, the cross-rank warm-up vote cannot exchange
        real data, so the schedule must infer the metadata mode locally:
        STATIC when complete metadata is supplied, and a clear error when
        dynamic inference would be required.
        """
        store = FakeStore()
        torch.distributed.init_process_group(
            backend="fake", rank=rank, world_size=2, store=store
        )
        d_hid, batch_size = 16, 8
        n_stages, num_microbatches = 2, 2
        device = torch.device("cpu")
        mod = MultiMLP(d_hid, n_layers=n_stages).get_submodule(f"layers.{rank}")

        x = torch.randn(batch_size, d_hid, device=device)
        mb = torch.randn(batch_size // num_microbatches, d_hid, device=device)
        try:
            with dist_config.patch(pipeline_per_edge_p2p=per_edge):
                stage = PipelineStage(
                    mod, rank, n_stages, device, input_args=mb, output_args=mod(mb)
                )
                schedule = ScheduleGPipe(stage, num_microbatches)
                schedule.step(x) if rank == 0 else schedule.step()
                self.assertEqual(stage._inference_mode, InferenceMode.STATIC)

                # Without static metadata, dynamic inference is required, which
                # cannot work over a fake group and must fail loudly.
                stage_dyn = PipelineStage(mod, rank, n_stages, device)
                schedule_dyn = ScheduleGPipe(stage_dyn, num_microbatches)
                with self.assertRaisesRegex(RuntimeError, "fake process group"):
                    schedule_dyn.step(x) if rank == 0 else schedule_dyn.step()
        finally:
            torch.distributed.destroy_process_group()

    @parametrize(
        "ScheduleClass",
        [
            ScheduleInterleavedZeroBubble,
            ScheduleZBVZeroBubble,
            ScheduleDualPipeV,
        ],
    )
    def test_zero_bubble_schedule_errors_with_compile(self, ScheduleClass):
        """
        Test that zero bubble schedules raise an error when used with torch.compile.
        """
        store = FakeStore()
        torch.distributed.init_process_group(
            backend="fake", rank=0, world_size=1, store=store
        )
        n_stages = 1
        device = torch.device("cpu")
        model = MultiMLP(8, n_layers=n_stages)
        # full_mod
        compiled_model = torch.compile(model)
        self.assertTrue(isinstance(compiled_model, OptimizedModule))
        stage = PipelineStage(
            compiled_model,
            0,
            n_stages,
            device,
        )
        try:
            with self.assertRaises(RuntimeError):
                ScheduleClass([stage], 2)
        finally:
            torch.distributed.destroy_process_group()


instantiate_parametrized_tests(ScheduleTest)


class TestSchedulePlan(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def setUp(self):
        super().setUp()
        # Define a list of test cases with varying num_local_stages, num_microbatches, and group_size
        # These should succeed since num_microbatches % group_size == 0
        self.test_cases = [
            # small number of stages
            (2, 2, 2),
            (2, 4, 4),
            (2, 8, 2),
            (2, 8, 4),
            (2, 8, 8),
            (4, 4, 4),
            (4, 8, 4),
            (4, 8, 8),
            # large microbatches
            (4, 16, 4),
            (4, 32, 4),
            (4, 64, 4),
            # large groups
            (4, 16, 16),
            (4, 32, 32),
            (4, 128, 64),
            # odd num pipeline stages
            (3, 2, 2),
            (3, 8, 2),
            (3, 12, 4),
            # odd group_sizes
            (4, 6, 3),
            (4, 10, 5),
            # n_mb non divisible by group_size
            (2, 3, 4),
            (2, 4, 4),
            (2, 10, 4),
            (2, 15, 4),
        ]

    @parametrize(
        "ScheduleClass",
        [ScheduleInterleaved1F1B, ScheduleLoopedBFS],
    )
    def test_pipeline_order(self, ScheduleClass):
        for num_local_stages, num_microbatches, group_size in self.test_cases:
            with self.subTest(
                num_local_stages=num_local_stages,
                num_microbatches=num_microbatches,
                group_size=group_size,
            ):
                if num_microbatches % group_size != 0:
                    continue

                logger.info(
                    "num_local_stages=%d num_microbatches=%d group_size=%d",
                    num_local_stages,
                    num_microbatches,
                    group_size,
                )
                num_stages = num_local_stages * group_size
                stages = [
                    MockPipelineStage(group_size=group_size, num_stages=num_stages)
                    for i in range(num_local_stages)
                ]

                schedule = ScheduleClass(stages, num_microbatches)
                _formatted_pipeline_order = _format_pipeline_order(
                    schedule.pipeline_order
                )

                def stage_to_rank(stage):
                    return stage % group_size

                comms_sch = _add_send_recv(
                    schedule.pipeline_order,
                    stage_to_rank=stage_to_rank,
                    num_stages=num_stages,
                )
                _simulate_comms_compute(
                    comms_sch,
                    stage_to_rank=stage_to_rank,
                    num_stages=num_stages,
                )

    def test_max_active_stages_is_used_for_lowering(self):
        stages = [
            MockPipelineStage(group_size=4, group_rank=3, num_stages=16)
            for _ in range(4)
        ]

        default_schedule = ScheduleInterleaved1F1B(
            stages,
            n_microbatches=16,
        )
        retained_schedule = ScheduleInterleaved1F1B(
            stages,
            n_microbatches=16,
            max_active_stages=4,
        )

        def count_stage_15_unshards(schedule):
            return sum(
                action.stage_index == 15 and action.computation_type == UNSHARD
                for action in schedule.pipeline_order_with_comms[3]
            )

        self.assertEqual(count_stage_15_unshards(default_schedule), 4)
        self.assertEqual(count_stage_15_unshards(retained_schedule), 1)

    @parametrize(
        "ScheduleClass",
        [ScheduleInterleaved1F1B, ScheduleInterleavedZeroBubble],
    )
    def test_pipeline_order_flex_and_zero_bubble(self, ScheduleClass):
        for num_local_stages, num_microbatches, group_size in self.test_cases:
            with self.subTest(
                num_local_stages=num_local_stages,
                num_microbatches=num_microbatches,
                group_size=group_size,
            ):
                warmups_ops_last_stage = (num_local_stages - 1) * (
                    num_microbatches // max(1, num_microbatches // group_size)
                )
                warmup_ops = warmups_ops_last_stage + 2 * (group_size - 1)
                warmup_ops = min(warmup_ops, num_microbatches * num_local_stages)

                num_stages = num_local_stages * group_size
                stages = [
                    MockPipelineStage(group_size=group_size, num_stages=num_stages)
                    for i in range(num_local_stages)
                ]
                schedule = ScheduleClass(stages, num_microbatches)

                def stage_to_rank(stage):
                    return stage % group_size

                comms_sch = _add_send_recv(
                    schedule.pipeline_order,
                    stage_to_rank=stage_to_rank,
                    num_stages=num_stages,
                )
                # print(_format_pipeline_order(comms_sch))
                _simulate_comms_compute(
                    comms_sch,
                    stage_to_rank=stage_to_rank,
                    num_stages=num_stages,
                )

    @parametrize(
        "ScheduleClass",
        [ScheduleDualPipeV, ScheduleZBVZeroBubble],
    )
    def test_pipeline_order_for_v_schedules(self, ScheduleClass):
        for num_local_stages, num_microbatches, group_size in self.test_cases:
            with self.subTest(
                num_local_stages=num_local_stages,
                num_microbatches=num_microbatches,
                group_size=group_size,
            ):
                num_stages = num_local_stages * group_size
                stages = [
                    MockPipelineStage(group_size=group_size, num_stages=num_stages)
                    for i in range(num_local_stages)
                ]

                # V schedules only support 2 stages per rank so if num_local_stages is not 2, ensure an error is thrown
                if num_local_stages != 2:
                    with self.assertRaises(ValueError):
                        ScheduleClass(
                            stages,
                            num_microbatches,
                        )
                    continue

                # DualPipeV requires num_microbatches to be >= num_stages
                if ScheduleClass == ScheduleDualPipeV and num_microbatches < num_stages:
                    with self.assertRaises(ValueError):
                        ScheduleClass(
                            stages,
                            num_microbatches,
                        )
                    continue

                # Create schedule and validate it
                schedule = ScheduleClass(stages, num_microbatches)
                _validate_schedule(
                    schedule.pipeline_order, group_size, num_stages, num_microbatches
                )


instantiate_parametrized_tests(TestSchedulePlan)


class TestScheduleCsv(TestCase):
    hw_classification = HardwareClassification.GENERIC

    @parametrize(
        "ScheduleClass,csv_name",
        [
            (ScheduleDualPipeV, "dualpipev_4rank_10mb"),
        ],
    )
    def test_csv_compare(self, ScheduleClass, csv_name):
        """
        Test that schedules matches the expected CSV.  This is a regression test to ensure that the schedule
        is not changed unintentionally.
        """
        num_local_stages = 2
        group_size = 4
        num_stages = num_local_stages * group_size
        stages = [
            MockPipelineStage(group_size=group_size, num_stages=num_stages)
            for _ in range(num_local_stages)
        ]
        num_microbatches = 10
        schedule = ScheduleClass(stages, num_microbatches)
        comms_csv = os.path.join(ARTIFACTS_DIR, f"{csv_name}.csv")
        sch = schedule.pipeline_order

        # Uncomment to regenerate reference output
        # schedule._dump_csv("test.csv", "compute_only")

        sch_ref = {}
        with open(comms_csv, newline="") as ref:
            for rank, row in enumerate(csv.reader(ref)):
                sch_ref[rank] = [_Action.from_str(s) for s in row]

        for rank in sch_ref:
            for timestep, (a, b) in enumerate(zip(sch[rank], sch_ref[rank])):
                self.assertEqual(
                    a,
                    b,
                    lambda msg: f"{msg}\nMismatch at {timestep=}, {a=}, expected {b}",
                )


instantiate_parametrized_tests(TestScheduleCsv)


class ScheduleLoweringTestBase(TestCase):
    def _parse_actions(self, actions: list[str]) -> list[_Action]:
        return [_Action.from_str(s) for s in actions]


class TestScheduleLowering(ScheduleLoweringTestBase):
    """Tests lowering passes that convert simple compute-only (FBW) schedules into compute+comms schedules"""

    hw_classification = HardwareClassification.GENERIC

    @parametrize(
        "action_str_and_ref",
        [
            ("1F0", _Action(1, F, 0)),
            ("2I1", _Action(2, I, 1)),
            ("0W3", _Action(0, W, 3)),
            ("0B3", _Action(0, B, 3)),
            ("1UNSHARD", _Action(1, UNSHARD, None)),
            ("3RESHARD", _Action(3, RESHARD, None)),
            ("2SEND_B2", _Action(2, SEND_B, 2)),
            ("1RECV_F1", _Action(1, RECV_F, 1)),
        ],
    )
    def test_action_parse(self, action_str_and_ref):
        """Test that actions can be parsed from strings and round-tripped back to the same strings."""
        act_str, ref = action_str_and_ref
        act = _Action.from_str(act_str)
        self.assertEqual(act, ref)
        self.assertEqual(act_str, act.__repr__())

    @parametrize(
        "test_info",
        [
            {
                "compute": ["0F0", "0F1", "   ", "0B0", "0B1"],
                "comms": ["0UNSHARD", "0F0", "0F1", "0B0", "0B1", "0RESHARD"],
            },
            {
                "compute": ["0F0", "0F1", "1F0", "1F1", "1B0", "1B1", "0B0", "0B1"],
                "comms": [
                    "0UNSHARD",
                    "1UNSHARD",
                    "0F0",
                    "0F1",
                    "1F0",
                    "1F1",
                    "1B0",
                    "1B1",
                    "1RESHARD",
                    "0B0",
                    "0B1",
                    "0RESHARD",
                ],
            },
        ],
    )
    def test_unshard_reshard(self, test_info):
        """Test the lowering pass that takes a 'compute only' schedule (with only F,B,W ops) and adds
        FSDP unshard/reshard operations to the schedule.  This is just part of the process of adding communication
        ops and producing a complete schedule.
        """
        compute_sch = self._parse_actions(test_info["compute"])
        expected_comms_sch = self._parse_actions(test_info["comms"])

        comms_sch = _add_unshard_reshard(compute_sch)
        for expected, actual in zip(expected_comms_sch, comms_sch):
            self.assertEqual(
                expected,
                actual,
                (
                    lambda msg: f"{msg}\nMismatch: expected action {expected} but found {actual}."
                    f"\nWhole Schedule: {comms_sch}"
                ),
            )

    @parametrize(
        "test_info",
        [
            {
                "compute": ["0F0", "0F1", "   ", "0B0", "0B1"],
                "comms": ["0F0", "0F1", "0B0", "0B1", "0REDUCE_GRAD"],
            },
            {
                "compute": ["0F0", "0F1", "1F0", "1F1", "1B0", "1B1", "0B0", "0B1"],
                "comms": [
                    "0F0",
                    "0F1",
                    "1F0",
                    "1F1",
                    "1B0",
                    "1B1",
                    "1REDUCE_GRAD",
                    "0B0",
                    "0B1",
                    "0REDUCE_GRAD",
                ],
            },
        ],
    )
    def test_reduce_grad(self, test_info):
        compute_sch = self._parse_actions(test_info["compute"])
        expected_comms_sch = self._parse_actions(test_info["comms"])

        comms_sch = _add_reduce_grad(compute_sch, 2)
        for expected, actual in zip(expected_comms_sch, comms_sch, strict=True):
            self.assertEqual(
                expected,
                actual,
                (
                    lambda msg: f"{msg}\nMismatch: expected action {expected} but found {actual}."
                    f"\nWhole Schedule: {comms_sch}"
                ),
            )

    @parametrize(
        "test_info",
        [
            {
                "compute": [
                    "0F0",
                    "0F1",
                    "0F2",
                    "0I0",
                    "0I1",
                    "0W0",
                    "0I2",
                    "0W2",
                    "0W1",
                ],
                "comms": ["0F0", "0F1", "0F2", "0I0", "0I1", "0W0", "0B2", "0W1"],
            },
        ],
    )
    def test_merge_bw(self, test_info):
        """Test the pass that merges adjacent I and W operations into a B operation."""
        compute_sch = self._parse_actions(test_info["compute"])
        expected_merged_sch = self._parse_actions(test_info["comms"])

        merged_sch = _merge_bw(compute_sch)
        for expected, actual in zip(expected_merged_sch, merged_sch):
            self.assertEqual(
                expected,
                actual,
                (
                    lambda msg: f"{msg}\nMismatch: expected action {expected} but found {actual}."
                    f"\nWhole Schedule: {merged_sch}"
                ),
            )

    @parametrize(
        "test_info",
        [
            {
                "schedule": "simple_2_rank_2_stage",
                "compute": {
                    0: ["0F0", "0F1", "   ", "0B0", "   ", "0B1"],
                    1: ["   ", "1F0", "1B0", "1F1", "1B1", "   "],
                },
                "comms": {
                    0: [
                        "0F0",
                        "0SEND_F0",
                        "0F1",
                        "0SEND_F1",
                        "0RECV_B0",
                        "0B0",
                        "0RECV_B1",
                        "0B1",
                    ],
                    1: [
                        "1RECV_F0",
                        "1RECV_F1",
                        "1F0",
                        "1B0",
                        "1SEND_B0",
                        "1F1",
                        "1B1",
                        "1SEND_B1",
                    ],
                },
                "stage_to_rank": lambda stage_idx: stage_idx,
                "num_stages": 2,
                "simulated_steps": 11,
            },
            {
                "schedule": "v_2_rank_4_stage",
                "compute": {
                    0: [
                        "0F0",
                        "0F1",
                        "   ",
                        "3F0",
                        "3B0",
                        "3F1",
                        "3B1",
                        "0B0",
                        "3W0",
                        "0B1",
                        "3W1",
                        "0W0",
                        "0W1",
                    ],
                    1: [
                        "   ",
                        "1F0",
                        "2F0",
                        "1F1",
                        "2F1",
                        "2B0",
                        "1B0",
                        "2B1",
                        "1B1",
                        "2W0",
                        "2W1",
                        "1W0",
                        "1W1",
                    ],
                },
                "comms": {
                    0: [
                        "0F0",
                        "0SEND_F0",
                        "0F1",
                        "0SEND_F1",
                        "3RECV_F0",
                        "3F0",
                        "3B0",
                        "3SEND_B0",
                        "3RECV_F1",
                        "3F1",
                        "3B1",
                        "3SEND_B1",
                        "0RECV_B0",
                        "0B0",
                        "3W0",
                        "0RECV_B1",
                        "0B1",
                        "3W1",
                        "0W0",
                        "0W1",
                    ],
                    1: [
                        "1RECV_F0",
                        # interesting that this gets scheduled up front, is that expected?
                        "1RECV_F1",
                        "1F0",
                        "2F0",
                        "2SEND_F0",
                        "1F1",
                        # ditto
                        "2RECV_B0",
                        "2F1",
                        "2SEND_F1",
                        "2B0",
                        # ditto
                        "2RECV_B1",
                        "1B0",
                        "1SEND_B0",
                        "2B1",
                        "1B1",
                        "1SEND_B1",
                        "2W0",
                        "2W1",
                        "1W0",
                        "1W1",
                    ],
                },
                "stage_to_rank": lambda stage_idx: [0, 1, 1, 0][stage_idx],
                "num_stages": 4,
                "simulated_steps": 24,
            },
        ],
    )
    def test_send_recv(self, test_info):
        """Tests the lowering pass that adds send/recv ops to a compute-only schedule."""
        compute_sch = {
            rank: self._parse_actions(test_info["compute"][rank])
            for rank in test_info["compute"]
        }
        expected_comms_sch = {
            rank: self._parse_actions(test_info["comms"][rank])
            for rank in test_info["comms"]
        }

        comms_sch = _add_send_recv(
            compute_sch, test_info["stage_to_rank"], test_info["num_stages"]
        )
        for rank in expected_comms_sch:
            for i, (expected, actual) in enumerate(
                zip(expected_comms_sch[rank], comms_sch[rank])
            ):
                self.assertEqual(
                    expected,
                    actual,
                    (
                        lambda msg: f"{msg}\nMismatch on rank {rank} at position {i}."
                        f"\nExpected: {expected_comms_sch[rank]}"
                        f"\nActual:   {comms_sch[rank]}"
                    ),
                )
            self.assertEqual(len(comms_sch[rank]), len(expected_comms_sch[rank]))

        simulated_schedule = _simulate_comms_compute(
            comms_sch,
            stage_to_rank=test_info["stage_to_rank"],
            num_stages=test_info["num_stages"],
        )
        # _dump_chrometrace(simulated_schedule, "lowered_comms.json")
        # print(_format_pipeline_order(simulated_schedule))
        num_steps = max([len(simulated_schedule[rank]) for rank in simulated_schedule])
        self.assertEqual(num_steps, test_info["simulated_steps"])

    @parametrize(
        "test_info",
        [
            {
                "schedule": "simple_2_rank_2_stage_no_overlap",
                "input": {
                    0: [
                        "0F0",
                        "0SEND_F0",
                        "0F1",
                        "0SEND_F1",
                        "0RECV_B0",
                        "0B0",
                        "0RECV_B1",
                        "0B1",
                    ],
                    1: [
                        "1RECV_F0",
                        "1RECV_F1",
                        "1F0",
                        "1B0",
                        "1SEND_B0",
                        "1F1",
                        "1B1",
                        "1SEND_B1",
                    ],
                },
                "expected": {
                    0: [
                        "0F0",
                        "0SEND_F0",
                        "0F1",
                        "0SEND_F1",
                        "0RECV_B0",
                        "0B0",
                        "0RECV_B1",
                        "0B1",
                    ],
                    1: [
                        # Rank 1 > rank 0: SEND before RECV (no flush constraint)
                        # 1RECV_F0 deferred to before 1F0
                        # 1RECV_F1 deferred to before 1F1
                        "1RECV_F0",
                        "1F0",
                        "1B0",
                        "1SEND_B0",
                        "1RECV_F1",
                        "1F1",
                        "1B1",
                        "1SEND_B1",
                    ],
                },
                "stage_to_rank": lambda stage_idx: stage_idx,
                "num_stages": 2,
            },
            {
                "schedule": "v_2_rank_4_stage_no_overlap",
                "input": {
                    0: [
                        "0F0",
                        "0SEND_F0",
                        "0F1",
                        "0SEND_F1",
                        "3RECV_F0",
                        "3F0",
                        "3B0",
                        "3SEND_B0",
                        "3RECV_F1",
                        "3F1",
                        "3B1",
                        "3SEND_B1",
                        "0RECV_B0",
                        "0B0",
                        "3W0",
                        "0RECV_B1",
                        "0B1",
                        "3W1",
                        "0W0",
                        "0W1",
                    ],
                    1: [
                        "1RECV_F0",
                        "1RECV_F1",
                        "1F0",
                        "2F0",
                        "2SEND_F0",
                        "1F1",
                        "2RECV_B0",
                        "2F1",
                        "2SEND_F1",
                        "2B0",
                        "2RECV_B1",
                        "1B0",
                        "1SEND_B0",
                        "2B1",
                        "1B1",
                        "1SEND_B1",
                        "2W0",
                        "2W1",
                        "1W0",
                        "1W1",
                    ],
                },
                "expected": {
                    0: [
                        # Rank 0 < rank 1: RECV before SEND (flush constraint)
                        # RECVs already right before consumers, no change
                        "0F0",
                        "0SEND_F0",
                        "0F1",
                        "0SEND_F1",
                        "3RECV_F0",
                        "3F0",
                        "3B0",
                        "3SEND_B0",
                        "3RECV_F1",
                        "3F1",
                        "3B1",
                        "3SEND_B1",
                        "0RECV_B0",
                        "0B0",
                        "3W0",
                        "0RECV_B1",
                        "0B1",
                        "3W1",
                        "0W0",
                        "0W1",
                    ],
                    1: [
                        # Rank 1 > rank 0: SEND before RECV (no flush constraint)
                        # All RECVs deferred to right before their consumers
                        "1RECV_F0",
                        "1F0",
                        "2F0",
                        "2SEND_F0",
                        "1RECV_F1",
                        "1F1",
                        "2F1",
                        "2SEND_F1",
                        "2RECV_B0",
                        "2B0",
                        "1B0",
                        "1SEND_B0",
                        "2RECV_B1",
                        "2B1",
                        "1B1",
                        "1SEND_B1",
                        "2W0",
                        "2W1",
                        "1W0",
                        "1W1",
                    ],
                },
                "stage_to_rank": lambda stage_idx: [0, 1, 1, 0][stage_idx],
                "num_stages": 4,
            },
            {
                # Schedule using BACKWARD_INPUT (I) + BACKWARD_WEIGHT (W) split
                # instead of FULL_BACKWARD (B). Verifies that BACKWARD_INPUT
                # consumes RECV_B (like FULL_BACKWARD does) and that
                # BACKWARD_WEIGHT does NOT trigger any RECV flushing.
                "schedule": "simple_2_rank_2_stage_iw_split",
                "input": {
                    0: [
                        # 0RECV_B0 placed early; flush rule (rank 0 < rank 1)
                        # should move it to right before the next SEND_F to rank 1.
                        "0RECV_B0",
                        "0F0",
                        "0SEND_F0",
                        "0F1",
                        "0SEND_F1",
                        "0I0",
                        "0W0",
                        "0RECV_B1",
                        "0I1",
                        "0W1",
                    ],
                    1: [
                        "1RECV_F0",
                        "1RECV_F1",
                        "1F0",
                        "1I0",
                        "1SEND_B0",
                        "1W0",
                        "1F1",
                        "1I1",
                        "1SEND_B1",
                        "1W1",
                    ],
                },
                "expected": {
                    0: [
                        # 0RECV_B0 deferred until just before 0SEND_F0 (flush
                        # required: rank 0 < peer rank 1). 0RECV_B1 deferred
                        # to immediately before its consumer 0I1 (BACKWARD_INPUT
                        # consumes RECV_B).
                        "0F0",
                        "0RECV_B0",
                        "0SEND_F0",
                        "0F1",
                        "0SEND_F1",
                        "0I0",
                        "0W0",
                        "0RECV_B1",
                        "0I1",
                        "0W1",
                    ],
                    1: [
                        # Rank 1 > peer rank 0: no flush. RECV_Fs deferred to
                        # right before their forward consumers. BACKWARD_WEIGHT
                        # (1W0/1W1) does not consume any RECV.
                        "1RECV_F0",
                        "1F0",
                        "1I0",
                        "1SEND_B0",
                        "1W0",
                        "1RECV_F1",
                        "1F1",
                        "1I1",
                        "1SEND_B1",
                        "1W1",
                    ],
                },
                "stage_to_rank": lambda stage_idx: stage_idx,
                "num_stages": 2,
            },
            {
                # Schedule containing an OVERLAP_F_B compound action whose
                # sub_actions consume both a deferred RECV_F (forward sub) and
                # a deferred RECV_B (backward sub). Verifies that RECV
                # flushing iterates over sub_actions correctly.
                "schedule": "v_2_rank_4_stage_overlap_f_b",
                "input": {
                    0: [
                        "0F0",
                        "0SEND_F0",
                        "0F1",
                        "0SEND_F1",
                        "3RECV_F0",
                        "3F0",
                        "3B0",
                        "3SEND_B0",
                        "3RECV_F1",
                        "3F1",
                        "3B1",
                        "3SEND_B1",
                        "0RECV_B0",
                        "0B0",
                        "3W0",
                        "0RECV_B1",
                        "0B1",
                        "3W1",
                        "0W0",
                        "0W1",
                    ],
                    1: [
                        "1RECV_F0",
                        "1RECV_F1",
                        "2RECV_B0",
                        "1F0",
                        "2F0",
                        "2SEND_F0",
                        "(1F1;2B0)OVERLAP_F_B",
                        "2W0",
                        "1B0",
                        "1SEND_B0",
                        "1W0",
                    ],
                },
                "expected": {
                    0: [
                        # Rank 0 < peer rank 1, but every RECV is already
                        # adjacent to its consumer; no movement.
                        "0F0",
                        "0SEND_F0",
                        "0F1",
                        "0SEND_F1",
                        "3RECV_F0",
                        "3F0",
                        "3B0",
                        "3SEND_B0",
                        "3RECV_F1",
                        "3F1",
                        "3B1",
                        "3SEND_B1",
                        "0RECV_B0",
                        "0B0",
                        "3W0",
                        "0RECV_B1",
                        "0B1",
                        "3W1",
                        "0W0",
                        "0W1",
                    ],
                    1: [
                        # Rank 1 > peer rank 0, no flush. 1RECV_F0 moves to
                        # before 1F0; 1RECV_F1 and 2RECV_B0 are both flushed
                        # immediately before the OVERLAP_F_B that consumes
                        # them via its sub_actions.
                        "1RECV_F0",
                        "1F0",
                        "2F0",
                        "2SEND_F0",
                        "1RECV_F1",
                        "2RECV_B0",
                        "(1F1;2B0)OVERLAP_F_B",
                        "2W0",
                        "1B0",
                        "1SEND_B0",
                        "1W0",
                    ],
                },
                "stage_to_rank": lambda stage_idx: [0, 1, 1, 0][stage_idx],
                "num_stages": 4,
            },
        ],
    )
    def test_defer_recv_ops(self, test_info):
        """Tests that _defer_recv_ops defers RECVs with rank-parity ordering."""
        input_sch = {
            rank: self._parse_actions(test_info["input"][rank])
            for rank in test_info["input"]
        }
        expected_sch = {
            rank: self._parse_actions(test_info["expected"][rank])
            for rank in test_info["expected"]
        }

        result_sch = _defer_recv_ops(input_sch, test_info["stage_to_rank"])
        for rank in expected_sch:
            for i, (expected, actual) in enumerate(
                zip(expected_sch[rank], result_sch[rank])
            ):
                self.assertEqual(
                    expected,
                    actual,
                    (
                        lambda msg: f"{msg}\nMismatch on rank {rank} at position {i}."
                        f"\nExpected: {expected_sch[rank]}"
                        f"\nActual:   {result_sch[rank]}"
                    ),
                )
            self.assertEqual(len(result_sch[rank]), len(expected_sch[rank]))

    def test_defer_recv_ops_no_deadlock(self):
        """Tests the issue from pytorch/pytorch#172668: RECV_F for stage 2 should not
        be placed before unrelated compute op 0F1 on the same rank, and the deferred
        schedule must not introduce deadlocks.

        Deadlock safety is ensured via rank-parity P2P ordering:
        - Lower rank (rank < peer): RECV before SEND
        - Higher rank (rank > peer): SEND before RECV
        """
        from torch.distributed.pipelining._schedule_visualizer import get_schedule_ops

        schedule_ops_overlap = get_schedule_ops(
            schedule="Interleaved1F1B",
            pp_degree=2,
            num_microbatches=4,
            with_comms=True,
            defer_pp_recv=False,
        )
        schedule_ops_no_overlap = get_schedule_ops(
            schedule="Interleaved1F1B",
            pp_degree=2,
            num_microbatches=4,
            with_comms=True,
            defer_pp_recv=True,
        )

        rank0_overlap = schedule_ops_overlap[0]
        rank0_no_overlap = schedule_ops_no_overlap[0]

        def find_action(actions, stage_idx, comp_type, mb_idx):
            for i, a in enumerate(actions):
                if (
                    a is not None
                    and a.stage_index == stage_idx
                    and a.computation_type == comp_type
                    and a.microbatch_index == mb_idx
                ):
                    return i
            return -1

        # With overlap (default): 2RECV_F0 comes before 0F1
        recv_f0_pos_overlap = find_action(rank0_overlap, 2, RECV_F, 0)
        f1_pos_overlap = find_action(rank0_overlap, 0, F, 1)
        self.assertGreater(f1_pos_overlap, recv_f0_pos_overlap)

        # Without overlap (rank 0 < rank 1, so RECV before SEND applies):
        # RECVs are flushed before SENDs to the same peer, which may interpose
        # a SEND between the RECV and its compute consumer. So on rank 0 we
        # only assert that the RECV moved later than in the overlap=True
        # schedule (i.e. deferral happened), not that it lands immediately
        # before its consumer.
        recv_f0_pos_no_overlap = find_action(rank0_no_overlap, 2, RECV_F, 0)
        self.assertGreater(recv_f0_pos_no_overlap, recv_f0_pos_overlap)

        # On rank 1 (higher rank, no flush constraint) the strict invariant
        # holds: every standalone RECV is placed immediately before the
        # compute op that consumes it. Apply the strict
        # assertEqual(recv_pos, consumer_pos - 1) check that is the core
        # invariant _defer_recv_ops is supposed to guarantee.
        rank1_no_overlap = schedule_ops_no_overlap[1]
        rank1_overlap = schedule_ops_overlap[1]

        # Spot-check a representative pair using the literal find_action style.
        recv_f0_pos_rank1 = find_action(rank1_no_overlap, 1, RECV_F, 0)
        f0_pos_rank1 = find_action(rank1_no_overlap, 1, F, 0)
        self.assertEqual(recv_f0_pos_rank1, f0_pos_rank1 - 1)

        # Generalize: every RECV on rank 1 is immediately followed by its
        # consumer (handles compound actions like OVERLAP_F_B by inspecting
        # sub_actions).
        non_none = [a for a in rank1_no_overlap if a is not None]
        for i, a in enumerate(non_none):
            if a.computation_type not in (RECV_F, RECV_B):
                continue
            self.assertLess(
                i,
                len(non_none) - 1,
                lambda msg: f"{msg}\nRECV {a} on rank 1 has no following consumer",
            )
            consumer = non_none[i + 1]
            consumer_subs = (
                consumer.sub_actions
                if consumer.sub_actions is not None
                else (consumer,)
            )
            if a.computation_type == RECV_F:
                matched = any(
                    s.stage_index == a.stage_index
                    and s.computation_type == F
                    and s.microbatch_index == a.microbatch_index
                    for s in consumer_subs
                )
            else:
                matched = any(
                    s.stage_index == a.stage_index
                    and s.computation_type in (B, I)
                    and s.microbatch_index == a.microbatch_index
                    for s in consumer_subs
                )
            self.assertTrue(
                matched,
                lambda msg: f"{msg}\nRECV {a} on rank 1 is not immediately followed by its "
                f"consumer (next action: {consumer})",
            )

        # Sanity: total RECV count should be unchanged by deferral.
        recv_count_overlap = sum(
            1
            for a in rank1_overlap
            if a is not None and a.computation_type in (RECV_F, RECV_B)
        )
        recv_count_no_overlap = sum(
            1
            for a in rank1_no_overlap
            if a is not None and a.computation_type in (RECV_F, RECV_B)
        )
        self.assertEqual(recv_count_overlap, recv_count_no_overlap)

    @parametrize("csv_name", ["zb1p_2rank_2stagep"])
    def test_csv(self, csv_name):
        def _dump_csv(pipeline_order_with_comms, filename: str):
            """Dump a CSV representation of the compute + comms schedule into a file with the provided filename."""
            with open(filename, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                for rank in pipeline_order_with_comms:
                    writer.writerow(pipeline_order_with_comms[rank])

        compute_sch = {}
        with open(
            os.path.join(ARTIFACTS_DIR, f"{csv_name}_compute.csv"), newline=""
        ) as csvfile:
            for rank, row in enumerate(csv.reader(csvfile)):
                compute_sch[rank] = [_Action.from_str(s) for s in row]
        # print(_format_pipeline_order(compute_sch))
        num_model_chunks = 2
        pipeline_parallel_size = 2
        num_stages = num_model_chunks * pipeline_parallel_size

        for rank in compute_sch:
            compute_sch[rank] = _merge_bw(compute_sch[rank])

        comms_sch = _add_send_recv(
            compute_sch,
            stage_to_rank=lambda chunk_index: chunk_index % pipeline_parallel_size,
            num_stages=num_stages,
        )

        comms_csv = os.path.join(ARTIFACTS_DIR, f"{csv_name}_comms.csv")

        # Uncomment to regenerate reference output
        # _dump_csv(comms_sch, comms_csv)

        sch_ref = {}
        with open(comms_csv, newline="") as ref:
            for rank, row in enumerate(csv.reader(ref)):
                sch_ref[rank] = [_Action.from_str(s) for s in row]

        for rank in sch_ref:
            for timestep, (a, b) in enumerate(zip(comms_sch[rank], sch_ref[rank])):
                self.assertEqual(
                    a,
                    b,
                    lambda msg: f"{msg}\nMismatch at {timestep=}, {a=}, expected {b}",
                )

        simulated_schedule = _simulate_comms_compute(
            comms_sch,
            stage_to_rank=lambda s: s % pipeline_parallel_size,
            num_stages=num_stages,
        )

        num_steps = max([len(simulated_schedule[rank]) for rank in simulated_schedule])
        # print(_format_pipeline_order(simulated_schedule))
        self.assertEqual(num_steps, 113)


class TestScheduleLoweringDevice(ScheduleLoweringTestBase):
    hw_classification = HardwareClassification.ACCELERATOR

    def test_grad_with_v_schedule(self, device):
        """
        We have a special case for V schedules where 2 adjacent stages are on the same rank.
        E.g.
        rank0:  stage 0,                 stage3
        rank1:          stage 1, stage 2,

        The special case involves not using send/recv ops but directly passing tensors between colocated stages.

        This test runs on a single rank and just tests the 'stage1, stage2' portion for both F and B, comparing
        gradients to a reference model with 2 layers.
        """
        store = FakeStore()
        torch.distributed.init_process_group(
            backend="fake", rank=0, world_size=1, store=store
        )
        try:
            d_hid = 512
            batch_size = 256
            n_stages = 2
            full_mod = MultiMLP(d_hid, n_layers=n_stages)
            full_mod.to(device)

            ref_mod = copy.deepcopy(full_mod)
            x = torch.randn(batch_size, d_hid, device=device)
            with torch.no_grad():
                y = ref_mod(x)
                # Add a small perturbation
                target = y + torch.randn(batch_size, d_hid, device=device)

            loss_fn = torch.nn.MSELoss(reduction="sum")

            # Run reference
            for _ in range(2):
                ref_mod.zero_grad()
                ref_out = ref_mod(x)
                ref_loss = loss_fn(ref_out, target)
                ref_loss.backward()

            stage_indices = [0, 1]
            submod_names = [f"layers.{i}" for i in stage_indices]
            stage_modules = [
                full_mod.get_submodule(submod_name) for submod_name in submod_names
            ]
            # Create a pipeline stage to wrap that submodule
            num_microbatches = 2
            stages = [
                PipelineStage(
                    stage_module,
                    stage_idx,
                    n_stages,
                    device,
                )
                for stage_module, stage_idx in zip(stage_modules, stage_indices)
            ]

            # Attach to a schedule
            schedule = _PipelineScheduleRuntime(
                stages,
                num_microbatches,
                loss_fn=loss_fn,
                scale_grads=False,
            )
            schedule._prepare_schedule_with_comms(
                {
                    0: self._parse_actions(
                        [
                            "0F0",
                            "0F1",
                            "1F0",
                            "1F1",
                            "1B0",
                            "1B1",
                            "0B0",
                            "0B1",
                        ]
                    ),
                },
                format="compute_comms",
            )

            # Run
            with check_leaked_tensors() as garbage_tensors:
                for _ in range(2):
                    # Zero gradients
                    for stage_module in stage_modules:
                        stage_module.zero_grad()
                    losses = []
                    out = schedule.step(x, target=target, losses=losses)
            self.assertEqual(
                len(garbage_tensors),
                0,
                "Found leaked tensors, check logs above for debug info",
            )

            # Check output
            torch.testing.assert_close(out, ref_out)
            # Check loss
            # Since the reduction used in the loss function above is "sum", we use
            # "sum" here to reduce microbatch losses into a single value too.
            pipe_loss = sum(losses)
            torch.testing.assert_close(pipe_loss, ref_loss)

            # Check gradients
            for stage_module, submod_name in zip(stage_modules, submod_names):
                # Get corresponding submodule from reference model
                ref_submod = ref_mod.get_submodule(submod_name)
                # Check gradients per parameter
                for name, p in stage_module.named_parameters():
                    ref_p = ref_submod.get_parameter(name)
                    try:
                        torch.testing.assert_close(
                            p.grad, ref_p.grad, rtol=1e-5, atol=4e-5
                        )
                    except AssertionError:
                        print(
                            f"Gradient test failed for {name}: {p.grad} vs {ref_p.grad}"
                        )
                        raise

        finally:
            torch.distributed.destroy_process_group()

    def test_grad_with_split_b_w(self, device):
        """
        Ensure that separate dInput and dWeight computations are correctly executed.
        This test runs on a single rank and just tests a single stage with 2 microbatches with separate B, W operations.
        """
        store = FakeStore()
        torch.distributed.init_process_group(
            backend="fake", rank=0, world_size=1, store=store
        )
        try:
            d_hid = 512
            batch_size = 256
            n_stages = 1
            full_mod = MultiMLP(d_hid, n_layers=n_stages)
            full_mod.to(device)

            ref_mod = copy.deepcopy(full_mod)
            x = torch.randn(batch_size, d_hid, device=device)
            with torch.no_grad():
                y = ref_mod(x)
                # Add a small perturbation
                target = y + torch.randn(batch_size, d_hid, device=device)

            loss_fn = torch.nn.MSELoss(reduction="sum")

            # Run reference
            for _ in range(2):
                ref_mod.zero_grad()
                ref_out = ref_mod(x)
                ref_loss = loss_fn(ref_out, target)
                ref_loss.backward()

            stage_indices = [0]
            submod_names = [f"layers.{i}" for i in stage_indices]
            stage_modules = [
                full_mod.get_submodule(submod_name) for submod_name in submod_names
            ]
            # Create a pipeline stage to wrap that submodule
            num_microbatches = 2
            stages = [
                PipelineStage(
                    stage_module,
                    stage_idx,
                    n_stages,
                    device,
                )
                for stage_module, stage_idx in zip(stage_modules, stage_indices)
            ]

            # Attach to a schedule
            schedule = _PipelineScheduleRuntime(
                stages,
                num_microbatches,
                loss_fn=loss_fn,
                scale_grads=False,
            )
            schedule._prepare_schedule_with_comms(
                {
                    0: self._parse_actions(
                        [
                            "0F0",
                            "0F1",
                            "0I0",
                            "0I1",
                            "0W0",
                            "0W1",
                        ]
                    ),
                },
                format="compute_comms",
            )

            # Run
            with check_leaked_tensors() as garbage_tensors:
                for _ in range(2):
                    # Zero gradients
                    for stage_module in stage_modules:
                        stage_module.zero_grad()
                    losses = []
                    out = schedule.step(x, target=target, losses=losses)
            self.assertEqual(
                len(garbage_tensors),
                0,
                "Found leaked tensors, check logs above for debug info",
            )

            # Check output
            torch.testing.assert_close(out, ref_out)
            # Check loss
            # Since the reduction used in the loss function above is "sum", we use
            # "sum" here to reduce microbatch losses into a single value too.
            pipe_loss = sum(losses)
            torch.testing.assert_close(pipe_loss, ref_loss)

            # Check gradients
            for stage_module, submod_name in zip(stage_modules, submod_names):
                # Get corresponding submodule from reference model
                ref_submod = ref_mod.get_submodule(submod_name)
                # Check gradients per parameter
                for name, p in stage_module.named_parameters():
                    ref_p = ref_submod.get_parameter(name)
                    try:
                        torch.testing.assert_close(
                            p.grad, ref_p.grad, rtol=1e-5, atol=4e-5
                        )
                    except AssertionError:
                        print(
                            f"Gradient test failed for {name}: {p.grad} vs {ref_p.grad}"
                        )
                        raise

        finally:
            torch.distributed.destroy_process_group()


instantiate_device_type_tests(
    TestScheduleLoweringDevice,
    globals(),
    except_for=["cpu"],
    allow_xpu=True,
)


class TestValidateSchedule(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_valid_schedule(self):
        schedule_actions = [
            {
                0: [_Action(0, F, 0), _Action(0, B, 0)],
                1: [_Action(1, F, 0), _Action(1, B, 0)],
            },
            {
                0: [_Action(0, F, 0), _Action(0, I, 0), _Action(0, W, 0)],
                1: [_Action(1, F, 0), _Action(1, I, 0), _Action(1, W, 0)],
            },
        ]
        pp_group_size = 2
        num_stages = 2
        num_microbatches = 1
        for actions in schedule_actions:
            _validate_schedule(actions, pp_group_size, num_stages, num_microbatches)

    def test_invalid_schedule_missing_rank(self):
        actions = {
            0: [_Action(0, F, 0), _Action(0, B, 0)],
        }
        pp_group_size = 2
        num_stages = 2
        num_microbatches = 1
        with self.assertRaises(AssertionError):
            _validate_schedule(actions, pp_group_size, num_stages, num_microbatches)

    def test_invalid_schedule_missing_action(self):
        actions = {
            0: [_Action(0, F, 0)],
            1: [_Action(1, F, 0)],
        }
        pp_group_size = 2
        num_stages = 2
        num_microbatches = 1
        with self.assertRaises(AssertionError):
            _validate_schedule(actions, pp_group_size, num_stages, num_microbatches)


class ScheduleUtilTests(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_generate_stage_to_rank_mapping(self):
        stage_to_rank = generate_stage_to_rank_mapping(2, 2)
        self.assertEqual(
            stage_to_rank,
            {
                0: 0,
                1: 1,
            },
        )
        stage_to_rank = generate_stage_to_rank_mapping(2, 4)
        self.assertEqual(stage_to_rank, {0: 0, 1: 1, 2: 0, 3: 1})
        stage_to_rank = generate_stage_to_rank_mapping(4, 8)
        self.assertEqual(
            stage_to_rank, {0: 0, 1: 1, 2: 2, 3: 3, 4: 0, 5: 1, 6: 2, 7: 3}
        )
        stage_to_rank = generate_stage_to_rank_mapping(2, 4, style="v")
        self.assertEqual(
            stage_to_rank,
            {
                0: 0,
                1: 1,
                2: 1,
                3: 0,
            },
        )
        stage_to_rank = generate_stage_to_rank_mapping(4, 12, style="v")
        self.assertEqual(
            stage_to_rank,
            {
                0: 0,
                1: 1,
                2: 2,
                3: 3,
                4: 3,
                5: 2,
                6: 1,
                7: 0,
                8: 0,
                9: 1,
                10: 2,
                11: 3,
            },
        )
        stage_to_rank = generate_stage_to_rank_mapping(4, 16, style="v")
        self.assertEqual(
            stage_to_rank,
            {
                0: 0,
                1: 1,
                2: 2,
                3: 3,
                4: 3,
                5: 2,
                6: 1,
                7: 0,
                8: 0,
                9: 1,
                10: 2,
                11: 3,
                12: 3,
                13: 2,
                14: 1,
                15: 0,
            },
        )


instantiate_parametrized_tests(TestScheduleLowering)


class TestBatchP2P(TestCase):
    """Tests that _batch_p2p dispatches homogeneous ops individually to avoid
    head-of-line blocking, while still batching mixed ops for deadlock avoidance."""

    hw_classification = HardwareClassification.GENERIC

    def _make_p2p_op(self, op, group_peer=0, group=None):
        p = MagicMock()
        p.op = op
        p.tensor = torch.zeros(1)
        # Ops in a single _batch_p2p call normally share one group; tests pass an
        # explicit shared group. _batch_p2p splits a list spanning multiple groups
        # into one batch per communicator, covered separately.
        p.group = group if group is not None else MagicMock()
        # _batch_p2p groups/orders ops by group_name, so give each group a stable
        # string name (identity-derived) rather than a MagicMock attribute.
        p.group.group_name = f"pg_{id(p.group)}"
        p.tag = 0
        p.group_peer = group_peer
        return p

    def test_empty_ops(self):
        self.assertEqual(_batch_p2p([]), [])

    @patch("torch.distributed.pipelining.schedules.dist.batch_isend_irecv")
    @patch("torch.distributed.pipelining.schedules.dist.isend")
    def test_all_isend_dispatched_individually(self, mock_isend, mock_batch):
        mock_isend.return_value = MagicMock()
        group = MagicMock()
        ops = [
            self._make_p2p_op(mock_isend, group_peer=i, group=group) for i in range(3)
        ]

        result = _batch_p2p(ops)

        mock_batch.assert_not_called()
        self.assertEqual(len(result), 3)
        self.assertEqual(mock_isend.call_count, 3)
        for p in ops:
            mock_isend.assert_any_call(
                p.tensor, group=p.group, tag=p.tag, group_dst=p.group_peer
            )

    @patch("torch.distributed.pipelining.schedules.dist.batch_isend_irecv")
    @patch("torch.distributed.pipelining.schedules.dist.irecv")
    def test_all_irecv_dispatched_individually(self, mock_irecv, mock_batch):
        mock_irecv.return_value = MagicMock()
        group = MagicMock()
        ops = [
            self._make_p2p_op(mock_irecv, group_peer=i, group=group) for i in range(3)
        ]

        result = _batch_p2p(ops)

        mock_batch.assert_not_called()
        self.assertEqual(len(result), 3)
        self.assertEqual(mock_irecv.call_count, 3)
        for p in ops:
            mock_irecv.assert_any_call(
                p.tensor, group=p.group, tag=p.tag, group_src=p.group_peer
            )

    @patch("torch.distributed.pipelining.schedules.dist.batch_isend_irecv")
    @patch("torch.distributed.pipelining.schedules.dist.irecv")
    @patch("torch.distributed.pipelining.schedules.dist.isend")
    def test_mixed_ops_use_batch(self, mock_isend, mock_irecv, mock_batch):
        mock_batch.return_value = [MagicMock(), MagicMock()]
        group = MagicMock()
        ops = [
            self._make_p2p_op(mock_isend, group_peer=0, group=group),
            self._make_p2p_op(mock_irecv, group_peer=1, group=group),
        ]

        result = _batch_p2p(ops)

        mock_batch.assert_called_once_with(ops)
        mock_isend.assert_not_called()
        mock_irecv.assert_not_called()
        self.assertEqual(len(result), 2)

    @patch("torch.distributed.pipelining.schedules.dist.batch_isend_irecv")
    @patch("torch.distributed.pipelining.schedules.dist.irecv")
    @patch("torch.distributed.pipelining.schedules.dist.isend")
    def test_mixed_ops_split_per_group(self, mock_isend, mock_irecv, mock_batch):
        """Issue a mixed operation list once per communicator.

        Directed-edge P2P can place one fused schedule batch on several child
        groups. Splitting by group lets each edge use its own communicator
        instead of sharing one FIFO.
        """
        mock_batch.side_effect = lambda ops: [MagicMock() for _ in ops]
        g_fwd, g_bwd = MagicMock(), MagicMock()
        ops = [
            self._make_p2p_op(mock_isend, group_peer=1, group=g_fwd),
            self._make_p2p_op(mock_irecv, group_peer=1, group=g_fwd),
            self._make_p2p_op(mock_isend, group_peer=2, group=g_bwd),
            self._make_p2p_op(mock_irecv, group_peer=2, group=g_bwd),
        ]

        result = _batch_p2p(ops)

        self.assertEqual(mock_batch.call_count, 2)
        mock_batch.assert_any_call([ops[0], ops[1]])
        mock_batch.assert_any_call([ops[2], ops[3]])
        mock_isend.assert_not_called()
        mock_irecv.assert_not_called()
        self.assertEqual(len(result), 4)


if __name__ == "__main__":
    run_tests()
