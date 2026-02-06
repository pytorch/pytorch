# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
"""
Tests for DTensor support in Pipeline Parallelism stages.

Tests the following functionality:
1. test_dtensor_pipeline: Parameterized test covering multiple schedule types,
   sharded/replicated inputs, and static/dynamic metadata inference
2. test_dtensor_mesh_validation: Mesh validation when spmd_mesh doesn't match input DTensor
"""

import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining._utils import generate_rank_to_stage_mapping
from torch.distributed.pipelining.schedules import (
    ScheduleInterleaved1F1B,
    ScheduleZBVZeroBubble,
)
from torch.distributed.tensor import distribute_tensor, DTensor, Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    requires_accelerator_dist_backend,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skip_but_pass_in_sandcastle_if,
    TEST_MULTIACCELERATOR,
)


d_hid = 256
batch_size = 64
n_microbatches = 4
microbatch_size = batch_size // n_microbatches  # Size per microbatch for stage creation

device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
backend = dist.get_default_backend_for_device(device_type)


class MLPModule(torch.nn.Module):
    """Simple MLP layer for testing."""

    def __init__(self, d_hid: int):
        super().__init__()
        self.net1 = torch.nn.Linear(d_hid, d_hid)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x):
        x = self.net1(x)
        x = self.relu(x)
        x = self.net2(x)
        return x


class MultiMLP(torch.nn.Module):
    """Multi-layer MLP for pipeline testing."""

    def __init__(self, d_hid: int, n_layers: int = 2):
        super().__init__()
        self.layers = torch.nn.ModuleList([MLPModule(d_hid) for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def _dtensor_loss_fn(output, target):
    """Loss function that works with DTensor using cross_entropy.

    Cross entropy has a sharding strategy registered for DTensor.
    Target should be a tensor of class indices with shape (batch_size,).
    """
    return torch.nn.functional.cross_entropy(output, target, reduction="sum")


def _get_tp_parallelize_plan(sharded_input_output: bool = False):
    """Get the TP parallelization plan for MLPModule.

    Args:
        sharded_input_output: If True, configure the plan for Shard(1) input/output
            on the feature dimension. If False, use default Replicate input/output.

    Note: use_local_output=False keeps outputs as DTensors instead of
    converting them to local tensors. This is required for pipeline
    stages to properly track DTensor metadata.
    """
    if sharded_input_output:
        # For sharded inputs/outputs on feature dimension:
        # - ColwiseParallel: input_layouts=Shard(1) to accept sharded input,
        #   internally redistributes to Replicate, produces Shard(1) output
        # - RowwiseParallel: takes Shard(1) input, output_layouts=Shard(1)
        #   to produce sharded output (reduce-scatter instead of all-reduce)
        return {
            "net1": ColwiseParallel(
                input_layouts=Shard(1),
                use_local_output=False,
            ),
            "net2": RowwiseParallel(
                output_layouts=Shard(1),
                use_local_output=False,
            ),
        }
    else:
        return {
            "net1": ColwiseParallel(use_local_output=False),
            "net2": RowwiseParallel(use_local_output=False),
        }


class DTensorPipelineTest(MultiProcessTestCase):
    """Tests for DTensor support in Pipeline Parallelism."""

    @property
    def world_size(self) -> int:
        return min(4, torch.get_device_module(device_type).device_count())

    @property
    def device(self) -> torch.device:
        return torch.device(device_type, self.rank)

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def init_pg(self):
        store = dist.FileStore(self.file_name, self.world_size)
        if device_type == "cuda":
            torch.cuda.set_device(self.device)
        dist.init_process_group(
            backend=backend,
            store=store,
            rank=self.rank,
            world_size=self.world_size,
            device_id=self.device,
        )

    def _create_dtensor_input(self, mesh, placements, shape=None, use_microbatch=False):
        """Create a DTensor input with the given mesh and placements.

        Args:
            mesh: DeviceMesh to use for the DTensor
            placements: Placement tuple for the DTensor
            shape: Optional GLOBAL shape tuple. Defaults to (batch_size, d_hid)
            use_microbatch: If True, use microbatch_size instead of batch_size
        """
        if shape is None:
            bs = microbatch_size if use_microbatch else batch_size
            shape = (bs, d_hid)

        # Create tensor with global shape and distribute it
        global_tensor = torch.randn(*shape, device=self.device)
        return distribute_tensor(global_tensor, mesh, placements)

    def _create_target_indices(self, mesh, num_classes=d_hid):
        """Create target indices for cross_entropy loss as a DTensor.

        Args:
            mesh: DeviceMesh to use for the DTensor
            num_classes: Number of classes for cross_entropy (default: d_hid)

        Returns:
            DTensor of shape (batch_size,) with random class indices, Replicated
        """
        local_target = torch.randint(0, num_classes, (batch_size,), device=self.device)
        return DTensor.from_local(
            local_target,
            device_mesh=mesh,
            placements=(Replicate(),),
            run_check=False,
        )

    def _get_stage_module(self, full_mod, stage_idx):
        """Get the submodule for a given stage index."""
        return full_mod.get_submodule(f"layers.{stage_idx}")

    def _get_expected_local_shape(self, global_shape, placements, tp_size):
        """Calculate the expected local shape given global shape and placements."""
        local_shape = list(global_shape)
        for placement in placements:
            if isinstance(placement, Shard):
                dim = placement.dim
                local_shape[dim] = local_shape[dim] // tp_size
        return torch.Size(local_shape)

    def _get_expected_stride(self, shape):
        """Calculate the expected contiguous stride for a shape."""
        stride = []
        s = 1
        for dim in reversed(shape):
            stride.append(s)
            s *= dim
        return tuple(reversed(stride))

    def _assert_dtensor_metadata(
        self,
        stage,
        placements,
        global_shape,
        expected_global_stride,
        expected_local_shape=None,
    ):
        """Assert that stage has correct DTensor metadata for inputs and outputs.

        Args:
            stage: PipelineStage to check
            placements: Expected placements tuple
            global_shape: Expected global shape
            expected_global_stride: Expected global stride
            expected_local_shape: If provided, also verify inputs_meta local shape
        """
        self.assertIsNotNone(
            stage._inputs_dtensor_meta,
            f"Stage {stage.stage_index} has None _inputs_dtensor_meta",
        )
        self.assertIsNotNone(
            stage._outputs_dtensor_meta,
            f"Stage {stage.stage_index} has None _outputs_dtensor_meta",
        )

        # Verify input DTensor metadata
        input_meta = stage._inputs_dtensor_meta[0]
        self.assertIsNotNone(
            input_meta,
            f"Stage {stage.stage_index} has None input DTensor metadata",
        )
        self.assertEqual(input_meta.placements, placements)
        self.assertEqual(input_meta.global_shape, torch.Size(global_shape))
        self.assertEqual(input_meta.global_stride, expected_global_stride)
        self.assertEqual(input_meta.dtype, torch.float32)

        # Verify output DTensor metadata
        output_meta = stage._outputs_dtensor_meta[0]
        self.assertIsNotNone(
            output_meta,
            f"Stage {stage.stage_index} has None output DTensor metadata",
        )
        self.assertEqual(output_meta.placements, placements)
        self.assertEqual(output_meta.global_shape, torch.Size(global_shape))
        self.assertEqual(output_meta.global_stride, expected_global_stride)
        self.assertEqual(output_meta.dtype, torch.float32)

        # Optionally verify local shape in inputs_meta
        if expected_local_shape is not None:
            self.assertEqual(stage.inputs_meta[0].shape, expected_local_shape)

        # Verify spmd_mesh is set
        self.assertIsNotNone(stage.spmd_mesh)
        self.assertEqual(stage.spmd_mesh.ndim, 1)

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 4+ GPUs"
    )
    @parametrize(
        "ScheduleClass",
        [ScheduleInterleaved1F1B, ScheduleZBVZeroBubble],
    )
    @parametrize("sharded_input", [True, False])
    @parametrize("metadata_inference", ["static", "dynamic"])
    def test_dtensor_pipeline(self, ScheduleClass, sharded_input, metadata_inference):
        """
        Consolidated test for DTensor support in pipeline parallelism.

        Parameters:
            ScheduleClass: The schedule class to use (Interleaved1F1B or ZBVZeroBubble)
            sharded_input: If True, use Shard(1) on feature dim; if False, use Replicate
            metadata_inference: "static" for input/output args at stage creation,
                              "dynamic" for runtime inference

        This test always uses 2 virtual stages per PP rank (4 total stages).
        For Interleaved1F1B: stages are assigned in loop order (0,2 on rank0; 1,3 on rank1)
        For ZBVZeroBubble: stages are assigned in V order (0,3 on rank0; 1,2 on rank1)
        """
        self.init_pg()

        pp_size = 2
        tp_size = self.world_size // pp_size
        n_virtual_stages = 2
        total_stages = pp_size * n_virtual_stages

        mesh_2d = init_device_mesh(
            device_type, (pp_size, tp_size), mesh_dim_names=("pp", "tp")
        )
        tp_mesh = mesh_2d["tp"]
        pp_group = mesh_2d.get_group("pp")
        pp_rank = dist.get_rank(pp_group)

        # Determine schedule style for stage-to-rank mapping
        if ScheduleClass == ScheduleZBVZeroBubble:
            schedule_style = "v"
        else:
            schedule_style = "loop"

        # Get stage indices for this rank
        rank_to_stages = generate_rank_to_stage_mapping(
            pp_size, total_stages, style=schedule_style
        )
        my_stage_indices = rank_to_stages[pp_rank]

        # Determine placements based on sharded_input parameter
        if sharded_input:
            placements = (Shard(1),)
            parallelize_plan = _get_tp_parallelize_plan(sharded_input_output=True)
        else:
            placements = (Replicate(),)
            parallelize_plan = _get_tp_parallelize_plan(sharded_input_output=False)

        # Calculate expected shapes for assertions
        global_shape = (microbatch_size, d_hid)
        expected_local_shape = self._get_expected_local_shape(
            global_shape, placements, tp_size
        )
        expected_global_stride = self._get_expected_stride(global_shape)

        # Create model
        torch.manual_seed(0)
        full_mod = MultiMLP(d_hid, n_layers=total_stages)
        full_mod.to(self.device)

        # Create stages for this rank
        stages = []
        for stage_idx in my_stage_indices:
            stage_mod = self._get_stage_module(full_mod, stage_idx)

            # Apply tensor parallelism
            parallelize_module(stage_mod, tp_mesh, parallelize_plan)

            if metadata_inference == "static":
                # Static: provide input/output args at stage creation
                input_dtensor = self._create_dtensor_input(
                    tp_mesh, placements, use_microbatch=True
                )
                with torch.no_grad():
                    output_dtensor = stage_mod(input_dtensor)

                stage = PipelineStage(
                    stage_mod,
                    stage_idx,
                    total_stages,
                    self.device,
                    input_args=input_dtensor,
                    output_args=output_dtensor,
                    group=pp_group,
                )

                # Verify DTensor metadata is populated at creation time
                self._assert_dtensor_metadata(
                    stage,
                    placements,
                    global_shape,
                    expected_global_stride,
                    expected_local_shape,
                )

            else:
                # Dynamic: no input/output args, provide spmd_mesh for reconstruction
                stage = PipelineStage(
                    stage_mod,
                    stage_idx,
                    total_stages,
                    self.device,
                    group=pp_group,
                    spmd_mesh=tp_mesh,
                )

                # Initially, DTensor metadata should be None
                self.assertIsNone(stage._inputs_dtensor_meta)
                self.assertIsNone(stage._outputs_dtensor_meta)

            stages.append(stage)

        # Create schedule
        schedule = ScheduleClass(
            stages,
            n_microbatches,
            loss_fn=_dtensor_loss_fn,
        )

        # Create actual input for first stage
        x = self._create_dtensor_input(tp_mesh, placements)

        # Determine which rank has first and last stage
        stage_to_rank = {s: r for r, ss in rank_to_stages.items() for s in ss}
        first_stage_rank = stage_to_rank[0]
        last_stage_rank = stage_to_rank[total_stages - 1]

        # Run pipeline
        if pp_rank == first_stage_rank:
            if pp_rank == last_stage_rank:
                # Same rank has both first and last stage (e.g., V-schedule rank 0)
                target = self._create_target_indices(tp_mesh)
                schedule.step(x, target=target)
            else:
                schedule.step(x)
        elif pp_rank == last_stage_rank:
            target = self._create_target_indices(tp_mesh)
            schedule.step(target=target)
        else:
            schedule.step()

        # For dynamic inference, verify metadata was populated after first step
        if metadata_inference == "dynamic":
            for stage in stages:
                self._assert_dtensor_metadata(
                    stage, placements, global_shape, expected_global_stride
                )

        dist.barrier()
        dist.destroy_process_group()

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 4+ GPUs"
    )
    def test_dtensor_mesh_validation(self):
        """
        Test that mesh validation works correctly when spmd_mesh is provided
        and DTensor inputs have a different mesh.
        """
        self.init_pg()

        pp_size = 2
        tp_size = self.world_size // pp_size

        mesh_2d = init_device_mesh(
            device_type, (pp_size, tp_size), mesh_dim_names=("pp", "tp")
        )
        tp_mesh = mesh_2d["tp"]
        pp_group = mesh_2d.get_group("pp")
        pp_rank = dist.get_rank(pp_group)

        # Create a different mesh to test validation
        different_mesh = init_device_mesh(device_type, (tp_size,))

        torch.manual_seed(0)
        full_mod = MultiMLP(d_hid, n_layers=pp_size)
        full_mod.to(self.device)

        stage_mod = self._get_stage_module(full_mod, pp_rank)

        # Apply TP parallelization
        parallelize_plan = _get_tp_parallelize_plan()
        parallelize_module(stage_mod, tp_mesh, parallelize_plan)

        placements = (Replicate(),)
        # Create DTensor with different_mesh (not tp_mesh)
        input_dtensor = self._create_dtensor_input(different_mesh, placements)

        # This should raise an error because spmd_mesh doesn't match input DTensor's mesh
        with self.assertRaisesRegex(RuntimeError, "provided spmd_mesh does not match"):
            PipelineStage(
                stage_mod,
                pp_rank,
                pp_size,
                self.device,
                input_args=input_dtensor,
                group=pp_group,
                spmd_mesh=tp_mesh,  # Different from input_dtensor's mesh
            )

        dist.barrier()
        dist.destroy_process_group()


instantiate_parametrized_tests(DTensorPipelineTest)


if __name__ == "__main__":
    run_tests()
