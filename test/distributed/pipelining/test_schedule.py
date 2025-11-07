# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
import copy
import csv
import logging
import os

from model_registry import MultiMLP

import torch
from torch._dynamo import OptimizedModule
from torch.distributed.pipelining import (
    Schedule1F1B,
    ScheduleDualPipeV,
    ScheduleGPipe,
    ScheduleInterleaved1F1B,
    ScheduleInterleavedZeroBubble,
    ScheduleLoopedBFS,
    ScheduleZBVZeroBubble,
)
from torch.distributed.pipelining._utils import generate_stage_to_rank_mapping
from torch.distributed.pipelining.schedules import (
    _Action,
    _add_reduce_grad,
    _add_send_recv,
    _add_unshard_reshard,
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
    PipelineScheduleSingle,
    RECV_F,
    RESHARD,
    SEND_B,
    UNSHARD,
    W,
)
from torch.distributed.pipelining.stage import _PipelineStageBase, PipelineStage
from torch.testing._internal.common_distributed import requires_accelerator_dist_backend
from torch.testing._internal.common_utils import (
    check_leaked_tensors,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.testing._internal.distributed.fake_pg import FakeStore


ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

device = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
logger = logging.getLogger(__name__)
torch.manual_seed(0)


class MockPipelineStage(_PipelineStageBase):
    def __init__(self, *args, **kwargs):
        # Mock the necessary attributes
        self.submod = None
        self.num_stages = kwargs.get("num_stages", 1)
        self.group_size = kwargs.get("group_size", 1)
        self.group_rank = kwargs.get("group_rank", 0)
        self.group = kwargs.get("group")

    def _create_grad_recv_info(self, *args, **kwargs):
        return None

    def _prepare_forward_infra(self, n_microbatches):
        pass

    def _prepare_backward_infra(self, n_microbatches):
        pass


class ScheduleTest(TestCase):
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
                    schedule_class, f"Class for {name} should not be None"
                )
                self.assertTrue(
                    issubclass(schedule_class, _PipelineSchedule),
                    f"{name} should be a subclass of _PipelineSchedule",
                )

        error_case = ["ScheduleThatDoesNotExist"]
        for name in error_case:
            # Test that the original name is included in the error message
            with self.assertRaisesRegex(ValueError, f"{name}"):
                get_schedule_class(name)

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
        """
        Test that simply runs evaluation followed by training.
        """
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
        # Run eval
        for _ in range(2):
            # Zero gradients
            stage_module.zero_grad()
            losses = []
            schedule.eval(x, target=target, losses=losses)
        # Run training
        try:
            for _ in range(2):
                losses = []
                schedule.step(x, target=target, losses=losses)
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
    def setUp(self):
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
                _format_pipeline_order(schedule.pipeline_order)

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
                self.assertEqual(a, b, f"Mismatch at {timestep=}, {a=}, expected {b}")


instantiate_parametrized_tests(TestScheduleCsv)


class TestScheduleLowering(TestCase):
    """Tests lowering passes that convert simple compute-only (FBW) schedules into compute+comms schedules"""

    def _parse_actions(self, actions: list[str]) -> list[_Action]:
        return [_Action.from_str(s) for s in actions]

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
                    f"Mismatch: expected action {expected} but found {actual}."
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
                    f"Mismatch: expected action {expected} but found {actual}."
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
                    f"Mismatch: expected action {expected} but found {actual}."
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
                        f"Mismatch on rank {rank} at position {i}."
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
                self.assertEqual(a, b, f"Mismatch at {timestep=}, {a=}, expected {b}")

        simulated_schedule = _simulate_comms_compute(
            comms_sch,
            stage_to_rank=lambda s: s % pipeline_parallel_size,
            num_stages=num_stages,
        )

        num_steps = max([len(simulated_schedule[rank]) for rank in simulated_schedule])
        # print(_format_pipeline_order(simulated_schedule))
        self.assertEqual(num_steps, 113)

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    def test_grad_with_v_schedule(self):
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
                    torch.testing.assert_close(p.grad, ref_p.grad, rtol=1e-5, atol=4e-5)
                except AssertionError:
                    print(f"Gradient test failed for {name}: {p.grad} vs {ref_p.grad}")
                    raise

        torch.distributed.destroy_process_group()

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    def test_grad_with_split_b_w(self):
        """
        Ensure that separate dInput and dWeight computations are correctly executed.
        This test runs on a single rank and just tests a single stage with 2 microbatches with separate B, W operations.
        """
        store = FakeStore()
        torch.distributed.init_process_group(
            backend="fake", rank=0, world_size=1, store=store
        )
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
                    torch.testing.assert_close(p.grad, ref_p.grad, rtol=1e-5, atol=4e-5)
                except AssertionError:
                    print(f"Gradient test failed for {name}: {p.grad} vs {ref_p.grad}")
                    raise

        torch.distributed.destroy_process_group()


class TestValidateSchedule(TestCase):
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

if __name__ == "__main__":
    run_tests()
