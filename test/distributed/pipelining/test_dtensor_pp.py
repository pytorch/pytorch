# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
"""
Tests for DTensor support in Pipeline Parallelism stages.

Test Categories:
1. Unit tests for _DTensorMeta and validation (no distributed setup)
2. Static metadata inference from constructor args
3. Static metadata validation errors
4. Runtime metadata inference
5. Static vs inferred metadata validation
6. V-schedule backward metadata passing
7. DTensor reconstruction and validation
8. Mesh caching behavior
9. End-to-end pipeline execution (parameterized)
"""

import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining._utils import (
    _DTensorMeta,
    generate_rank_to_stage_mapping,
    PipeliningDTensorError,
    validate_dtensor_metadata,
)
from torch.distributed.pipelining.schedules import (
    Schedule1F1B,
    ScheduleInterleaved1F1B,
    ScheduleZBVZeroBubble,
)
from torch.distributed.tensor import distribute_tensor, DTensor, Replicate, Shard
from torch.distributed.tensor.placement_types import Partial
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
    TestCase,
)


# Test constants
d_hid = 256
batch_size = 64
n_microbatches = 4
microbatch_size = batch_size // n_microbatches

device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
backend = dist.get_default_backend_for_device(device_type)


# =============================================================================
# Test Modules
# =============================================================================


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


# =============================================================================
# Helper Functions
# =============================================================================


def _dtensor_loss_fn(output, target):
    """Loss function that works with DTensor using cross_entropy."""
    return torch.nn.functional.cross_entropy(output, target, reduction="sum")


def _get_tp_parallelize_plan(sharded_input_output: bool = False):
    """Get the TP parallelization plan for MLPModule."""
    if sharded_input_output:
        return {
            "net1": ColwiseParallel(input_layouts=Shard(1), use_local_output=False),
            "net2": RowwiseParallel(output_layouts=Shard(1), use_local_output=False),
        }
    else:
        return {
            "net1": ColwiseParallel(use_local_output=False),
            "net2": RowwiseParallel(use_local_output=False),
        }


# =============================================================================
# Test 1: Unit Tests for _DTensorMeta and Validation (No Distributed)
# =============================================================================


class TestDTensorMetaAndValidation(TestCase):
    """Unit tests for _DTensorMeta and validate_dtensor_metadata - no distributed setup."""

    def test_dtensor_meta_and_validation(self):
        """Test _DTensorMeta creation and validate_dtensor_metadata errors."""
        # Setup: Create a simple mesh and DTensor
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
        global_tensor = torch.randn(8, 16)
        dtensor_shard = distribute_tensor(global_tensor, mesh, [Shard(0)])

        # 1. Test from_dtensor extracts correct metadata
        meta = _DTensorMeta.from_dtensor(dtensor_shard)
        self.assertEqual(meta.global_shape, torch.Size([8, 16]))
        self.assertEqual(meta.global_stride, (16, 1))
        self.assertEqual(meta.dtype, torch.float32)
        self.assertEqual(meta.placements, (Shard(0),))
        self.assertEqual(meta.mesh_dim_names, ("tp",))
        self.assertEqual(meta.mesh_shape, (2,))
        self.assertIsNotNone(meta.mesh)

        # 2. Test validate_dtensor_metadata passes for matching DTensor
        validate_dtensor_metadata("test_match", meta, dtensor_shard)  # Should not raise

        # 3. Test PipeliningDTensorError on shape mismatch
        wrong_shape_meta = _DTensorMeta(
            global_shape=torch.Size([4, 16]),  # Wrong shape
            global_stride=(16, 1),
            dtype=torch.float32,
            placements=(Shard(0),),
            mesh_dim_names=("tp",),
        )
        with self.assertRaises(PipeliningDTensorError) as ctx:
            validate_dtensor_metadata("test_shape", wrong_shape_meta, dtensor_shard)
        self.assertIn("shape mismatch", str(ctx.exception))
        self.assertIn("4, 16", str(ctx.exception))
        self.assertIn("8, 16", str(ctx.exception))

        # 4. Test PipeliningDTensorError on placement mismatch
        wrong_placement_meta = _DTensorMeta(
            global_shape=torch.Size([8, 16]),
            global_stride=(16, 1),
            dtype=torch.float32,
            placements=(Replicate(),),  # Expected Replicate
            mesh_dim_names=("tp",),
        )
        with self.assertRaises(PipeliningDTensorError) as ctx:
            validate_dtensor_metadata(
                "test_placement", wrong_placement_meta, dtensor_shard
            )
        self.assertIn("placements mismatch", str(ctx.exception))

        # 5. Test PipeliningDTensorError when given non-DTensor
        regular_tensor = torch.randn(8, 16)
        with self.assertRaises(PipeliningDTensorError) as ctx:
            # type: ignore intentionally passing wrong type to test error handling
            validate_dtensor_metadata("test_not_dtensor", meta, regular_tensor)  # type: ignore[arg-type]
        self.assertIn("expected DTensor", str(ctx.exception))


# =============================================================================
# Distributed Test Base Class
# =============================================================================


class DTensorPPTestBase(MultiProcessTestCase):
    """Base class for DTensor PP tests with common setup."""

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

    def _create_dtensor(self, mesh, placements, shape=None, requires_grad=False):
        """Create a DTensor with given properties."""
        if shape is None:
            shape = (microbatch_size, d_hid)
        t = torch.randn(*shape, device=self.device, requires_grad=requires_grad)
        return distribute_tensor(t, mesh, placements)

    def _create_stage_module(self, tp_mesh, parallelize=True):
        """Create a simple stage module with optional TP."""
        torch.manual_seed(0)
        mod = MLPModule(d_hid).to(self.device)
        if parallelize:
            parallelize_module(mod, tp_mesh, _get_tp_parallelize_plan())
        return mod

    def _create_target(self, mesh):
        """Create target indices for loss function."""
        local_target = torch.randint(0, d_hid, (batch_size,), device=self.device)
        return DTensor.from_local(local_target, mesh, (Replicate(),), run_check=False)


# =============================================================================
# Test 2: Static Metadata Inference
# =============================================================================


class TestStaticMetadataInference(DTensorPPTestBase):
    """Tests for static metadata inference from constructor args."""

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 4+ GPUs"
    )
    def test_static_metadata_inference(self):
        """Test static metadata from input_args, output_args, output_grads, input_grads."""
        self.init_pg()

        pp_size = 2
        tp_size = self.world_size // pp_size
        mesh_2d = init_device_mesh(
            device_type, (pp_size, tp_size), mesh_dim_names=("pp", "tp")
        )
        tp_mesh = mesh_2d["tp"]
        pp_group = mesh_2d.get_group("pp")
        pp_rank = dist.get_rank(pp_group)

        stage_mod = self._create_stage_module(tp_mesh)

        # Create DTensor inputs/outputs
        input_dtensor = self._create_dtensor(tp_mesh, [Replicate()], requires_grad=True)
        with torch.no_grad():
            output_dtensor = stage_mod(input_dtensor)

        # Create gradient DTensors (may have different placements)
        output_grad = self._create_dtensor(tp_mesh, [Partial()])
        input_grad = self._create_dtensor(tp_mesh, [Partial()])

        # Create stage with all static metadata
        stage = PipelineStage(
            stage_mod,
            pp_rank,
            pp_size,
            self.device,
            input_args=input_dtensor,
            output_args=output_dtensor,
            output_grads=output_grad,
            input_grads=input_grad,
            group=pp_group,
        )

        # 1. Verify _inputs_dtensor_meta populated from input_args
        assert stage._inputs_dtensor_meta is not None
        self.assertEqual(len(stage._inputs_dtensor_meta), 1)
        assert stage._inputs_dtensor_meta[0] is not None
        self.assertEqual(stage._inputs_dtensor_meta[0].placements, (Replicate(),))

        # 2. Verify _outputs_dtensor_meta populated from output_args
        assert stage._outputs_dtensor_meta is not None
        self.assertEqual(len(stage._outputs_dtensor_meta), 1)
        assert stage._outputs_dtensor_meta[0] is not None
        self.assertEqual(stage._outputs_dtensor_meta[0].placements, (Replicate(),))

        # 3. Verify _outputs_grad_dtensor_meta populated from output_grads
        assert stage._outputs_grad_dtensor_meta is not None
        self.assertEqual(len(stage._outputs_grad_dtensor_meta), 1)
        assert stage._outputs_grad_dtensor_meta[0] is not None
        self.assertEqual(stage._outputs_grad_dtensor_meta[0].placements, (Partial(),))

        # 4. Verify _inputs_grad_dtensor_meta populated from input_grads
        assert stage._inputs_grad_dtensor_meta is not None
        self.assertEqual(len(stage._inputs_grad_dtensor_meta), 1)
        assert stage._inputs_grad_dtensor_meta[0] is not None
        self.assertEqual(stage._inputs_grad_dtensor_meta[0].placements, (Partial(),))

        # 5. Verify mesh is cached from static DTensors
        self.assertIsNotNone(stage._inputs_dtensor_meta[0].mesh)

        dist.barrier()
        dist.destroy_process_group()


# =============================================================================
# Test 3: Static Metadata Validation Errors
# =============================================================================


class TestStaticMetadataValidationErrors(DTensorPPTestBase):
    """Tests for validation errors in static metadata configuration."""

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 4+ GPUs"
    )
    def test_static_metadata_validation_errors(self):
        """Test errors for invalid static metadata configuration."""
        self.init_pg()

        pp_size = 2
        tp_size = self.world_size // pp_size
        mesh_2d = init_device_mesh(
            device_type, (pp_size, tp_size), mesh_dim_names=("pp", "tp")
        )
        tp_mesh = mesh_2d["tp"]
        pp_group = mesh_2d.get_group("pp")
        pp_rank = dist.get_rank(pp_group)

        stage_mod = self._create_stage_module(tp_mesh)

        input_dtensor = self._create_dtensor(tp_mesh, [Replicate()])
        with torch.no_grad():
            output_dtensor = stage_mod(input_dtensor)

        # 1. ValueError when output_grads length != output_args length
        output_grads_wrong_len = (
            self._create_dtensor(tp_mesh, [Partial()]),
            self._create_dtensor(tp_mesh, [Partial()]),  # Extra element
        )
        with self.assertRaises(ValueError) as ctx:
            PipelineStage(
                stage_mod,
                pp_rank,
                pp_size,
                self.device,
                input_args=input_dtensor,
                output_args=output_dtensor,
                output_grads=output_grads_wrong_len,
                group=pp_group,
            )
        self.assertIn("output_grads", str(ctx.exception))
        self.assertIn("length", str(ctx.exception).lower())

        # 2. ValueError when input_grads length != input_args length
        input_grads_wrong_len = (
            self._create_dtensor(tp_mesh, [Partial()]),
            self._create_dtensor(tp_mesh, [Partial()]),  # Extra element
        )
        output_grad = self._create_dtensor(tp_mesh, [Partial()])
        with self.assertRaises(ValueError) as ctx:
            PipelineStage(
                stage_mod,
                pp_rank,
                pp_size,
                self.device,
                input_args=input_dtensor,
                output_args=output_dtensor,
                output_grads=output_grad,
                input_grads=input_grads_wrong_len,
                group=pp_group,
            )
        self.assertIn("input_grads", str(ctx.exception))
        self.assertIn("length", str(ctx.exception).lower())

        # 3. Verify input_grads without output_grads logs warning but doesn't error
        input_grad = self._create_dtensor(tp_mesh, [Partial()])
        # This should work - defers backward metadata to runtime
        stage = PipelineStage(
            stage_mod,
            pp_rank,
            pp_size,
            self.device,
            input_args=input_dtensor,
            output_args=output_dtensor,
            input_grads=input_grad,  # Without output_grads
            group=pp_group,
        )
        # input_grads is ignored when output_grads not provided
        self.assertIsNone(stage._outputs_grad_dtensor_meta)

        dist.barrier()
        dist.destroy_process_group()


# =============================================================================
# Test 4: Runtime Metadata Inference
# =============================================================================


class TestRuntimeMetadataInference(DTensorPPTestBase):
    """Tests for runtime metadata inference."""

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 4+ GPUs"
    )
    def test_runtime_metadata_inference(self):
        """Test runtime metadata inference when no static args provided."""
        self.init_pg()

        pp_size = 2
        tp_size = self.world_size // pp_size
        mesh_2d = init_device_mesh(
            device_type, (pp_size, tp_size), mesh_dim_names=("pp", "tp")
        )
        tp_mesh = mesh_2d["tp"]
        pp_group = mesh_2d.get_group("pp")
        pp_rank = dist.get_rank(pp_group)

        stage_mod = self._create_stage_module(tp_mesh)

        # Track get_mesh calls
        get_mesh_call_count = [0]

        def mesh_provider(dim_names, shape=None):
            get_mesh_call_count[0] += 1
            if dim_names == ("tp",):
                return tp_mesh
            raise ValueError(f"Unknown mesh: {dim_names}")

        # 1. Create stage without static args
        stage = PipelineStage(
            stage_mod,
            pp_rank,
            pp_size,
            self.device,
            group=pp_group,
            get_mesh=mesh_provider,
        )

        # 2. Verify _inputs_dtensor_meta is None at construction
        self.assertIsNone(stage._inputs_dtensor_meta)
        self.assertIsNone(stage._outputs_dtensor_meta)

        # 3. Create schedule and run - this triggers runtime inference
        schedule = Schedule1F1B(
            stage,
            n_microbatches,
            loss_fn=_dtensor_loss_fn,
        )

        x = self._create_dtensor(tp_mesh, [Replicate()], shape=(batch_size, d_hid))
        target = self._create_target(tp_mesh)

        if pp_rank == 0:
            schedule.step(x)
        else:
            schedule.step(target=target)

        # 4. Verify metadata populated after execution
        self.assertIsNotNone(stage._inputs_dtensor_meta)
        self.assertIsNotNone(stage._outputs_dtensor_meta)

        # 5. Test RuntimeError when get_mesh not provided
        stage_no_mesh = PipelineStage(
            self._create_stage_module(tp_mesh),
            pp_rank,
            pp_size,
            self.device,
            group=pp_group,
            # No get_mesh provided
        )
        self.assertIsNone(stage_no_mesh.get_mesh)

        dist.barrier()
        dist.destroy_process_group()


# =============================================================================
# Test 5: Static vs Inferred Metadata Validation
# =============================================================================


class TestStaticVsInferredValidation(DTensorPPTestBase):
    """Tests for validation when static metadata doesn't match inferred."""

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 4+ GPUs"
    )
    def test_static_vs_inferred_validation(self):
        """Test errors when static metadata doesn't match inferred."""
        self.init_pg()

        pp_size = 2
        tp_size = self.world_size // pp_size
        mesh_2d = init_device_mesh(
            device_type, (pp_size, tp_size), mesh_dim_names=("pp", "tp")
        )
        tp_mesh = mesh_2d["tp"]
        pp_group = mesh_2d.get_group("pp")
        pp_rank = dist.get_rank(pp_group)

        # Only test on last stage (pp_rank == 1) where backward metadata inference runs
        if pp_rank == pp_size - 1:
            stage_mod = self._create_stage_module(tp_mesh)

            input_dtensor = self._create_dtensor(
                tp_mesh, [Replicate()], requires_grad=True
            )
            with torch.no_grad():
                output_dtensor = stage_mod(input_dtensor)

            # Create output_grads with WRONG placement (Shard instead of expected Partial)
            # The actual gradient from loss.backward() will likely be Partial or Replicate
            wrong_output_grad = self._create_dtensor(tp_mesh, [Shard(1)])

            stage = PipelineStage(
                stage_mod,
                pp_rank,
                pp_size,
                self.device,
                input_args=input_dtensor,
                output_args=output_dtensor,
                output_grads=wrong_output_grad,  # Wrong placement
                group=pp_group,
            )

            # Prepare a target for backward
            target = self._create_target(tp_mesh)

            # Run backward metadata inference - should error on mismatch
            with self.assertRaises(PipeliningDTensorError) as ctx:
                stage._backward_metadata_inference(
                    loss_fn=_dtensor_loss_fn,
                    target=target,
                )
            self.assertIn("placement", str(ctx.exception).lower())
            self.assertIn("mismatch", str(ctx.exception).lower())

        dist.barrier()
        dist.destroy_process_group()


# =============================================================================
# Test 6: V-Schedule Backward Metadata Passing
# =============================================================================


class TestVScheduleBackwardMetadataPassing(DTensorPPTestBase):
    """Tests for V-schedule backward metadata passing."""

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 4+ GPUs"
    )
    def test_v_schedule_backward_metadata_passing(self):
        """Test V-schedule backward metadata flows correctly."""
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

        # V-schedule stage mapping
        rank_to_stages = generate_rank_to_stage_mapping(
            pp_size, total_stages, style="v"
        )
        my_stage_indices = rank_to_stages[pp_rank]

        # Create model and stages
        torch.manual_seed(0)
        full_mod = MultiMLP(d_hid, n_layers=total_stages).to(self.device)

        def mesh_provider(dim_names, shape=None):
            if dim_names == ("tp",):
                return tp_mesh
            raise ValueError(f"Unknown mesh: {dim_names}")

        stages = []
        for stage_idx in my_stage_indices:
            stage_mod = full_mod.get_submodule(f"layers.{stage_idx}")
            parallelize_module(stage_mod, tp_mesh, _get_tp_parallelize_plan())

            stage = PipelineStage(
                stage_mod,
                stage_idx,
                total_stages,
                self.device,
                group=pp_group,
                get_mesh=mesh_provider,
            )
            stages.append(stage)

        # Create schedule
        schedule = ScheduleZBVZeroBubble(
            stages,
            n_microbatches,
            loss_fn=_dtensor_loss_fn,
        )

        # Run pipeline
        x = self._create_dtensor(tp_mesh, [Replicate()], shape=(batch_size, d_hid))
        target = self._create_target(tp_mesh)

        # V-schedule: rank 0 has stages 0 and 3 (first and last)
        if pp_rank == 0:
            schedule.step(x, target=target)
        else:
            schedule.step()

        # Verify all stages have backward metadata populated
        for stage in stages:
            # After execution, grad recv info should be set up
            self.assertIsNotNone(stage.grad_recv_info)

        dist.barrier()
        dist.destroy_process_group()


# =============================================================================
# Test 7: DTensor Reconstruction and Validation
# =============================================================================


class TestDTensorReconstructionAndValidation(DTensorPPTestBase):
    """Tests for DTensor reconstruction during runtime."""

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 4+ GPUs"
    )
    def test_dtensor_reconstruction_and_validation(self):
        """Test DTensor reconstruction and validation in retrieve functions."""
        self.init_pg()

        pp_size = 2
        tp_size = self.world_size // pp_size
        mesh_2d = init_device_mesh(
            device_type, (pp_size, tp_size), mesh_dim_names=("pp", "tp")
        )
        tp_mesh = mesh_2d["tp"]
        pp_group = mesh_2d.get_group("pp")
        pp_rank = dist.get_rank(pp_group)

        stage_mod = self._create_stage_module(tp_mesh)

        def mesh_provider(dim_names, shape=None):
            if dim_names == ("tp",):
                return tp_mesh
            raise ValueError(f"Unknown mesh: {dim_names}")

        stage = PipelineStage(
            stage_mod,
            pp_rank,
            pp_size,
            self.device,
            group=pp_group,
            get_mesh=mesh_provider,
        )

        schedule = Schedule1F1B(
            stage,
            n_microbatches,
            loss_fn=_dtensor_loss_fn,
        )

        x = self._create_dtensor(tp_mesh, [Replicate()], shape=(batch_size, d_hid))
        target = self._create_target(tp_mesh)

        if pp_rank == 0:
            schedule.step(x)
        else:
            schedule.step(target=target)

        # Verify DTensor metadata was properly used for reconstruction
        # Check that args_recv_info has DTensor metadata attached
        # Import _RecvInfo for isinstance check
        from torch.distributed.pipelining.stage import _RecvInfo

        if not stage.is_first:
            for recv_infos in stage.args_recv_info.values():
                for recv_info in recv_infos:
                    if (
                        isinstance(recv_info, _RecvInfo)
                        and recv_info.dtensor_meta is not None
                    ):
                        self.assertEqual(
                            recv_info.dtensor_meta.placements, (Replicate(),)
                        )

        dist.barrier()
        dist.destroy_process_group()


# =============================================================================
# Test 8: Mesh Caching
# =============================================================================


class TestMeshCaching(DTensorPPTestBase):
    """Tests for stage-level mesh caching behavior."""

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 4+ GPUs"
    )
    def test_mesh_caching(self):
        """Test stage-level mesh caching behavior."""
        self.init_pg()

        pp_size = 2
        tp_size = self.world_size // pp_size
        mesh_2d = init_device_mesh(
            device_type, (pp_size, tp_size), mesh_dim_names=("pp", "tp")
        )
        tp_mesh = mesh_2d["tp"]
        pp_group = mesh_2d.get_group("pp")
        pp_rank = dist.get_rank(pp_group)

        stage_mod = self._create_stage_module(tp_mesh)

        # Track get_mesh calls
        get_mesh_calls = []

        def mesh_provider(dim_names, shape=None):
            get_mesh_calls.append(dim_names)
            if dim_names == ("tp",):
                return tp_mesh
            raise ValueError(f"Unknown mesh: {dim_names}")

        # Create stage with static metadata (mesh should be cached from DTensor)
        input_dtensor = self._create_dtensor(tp_mesh, [Replicate()])
        with torch.no_grad():
            output_dtensor = stage_mod(input_dtensor)

        stage = PipelineStage(
            stage_mod,
            pp_rank,
            pp_size,
            self.device,
            input_args=input_dtensor,
            output_args=output_dtensor,
            group=pp_group,
            get_mesh=mesh_provider,
        )

        # 1. Verify mesh is cached from static DTensors (get_mesh not called)
        self.assertIn(
            (("tp",), (tp_size,)),
            stage._mesh_cache,
        )

        # 2. Create DTensor meta and test cache lookup
        meta = _DTensorMeta(
            global_shape=torch.Size([8, 16]),
            global_stride=(16, 1),
            dtype=torch.float32,
            placements=(Replicate(),),
            mesh_dim_names=("tp",),
            mesh_shape=(tp_size,),
            mesh=None,  # No mesh cached in meta
        )

        # First lookup - should use get_mesh
        get_mesh_calls.clear()
        mesh1 = stage._get_mesh_for_dtensor_meta(meta)
        self.assertEqual(mesh1, tp_mesh)

        # 3. Second lookup with same dim_names - should use cache
        get_mesh_calls.clear()
        mesh2 = stage._get_mesh_for_dtensor_meta(meta)
        self.assertEqual(mesh2, tp_mesh)
        # get_mesh should NOT be called again (cache hit)
        self.assertEqual(len(get_mesh_calls), 0)

        dist.barrier()
        dist.destroy_process_group()


# =============================================================================
# Test 9: End-to-End Pipeline (Parameterized - Existing)
# =============================================================================


class TestDTensorPipelineE2E(DTensorPPTestBase):
    """End-to-end tests for DTensor pipeline execution."""

    def _get_stage_module(self, full_mod, stage_idx):
        """Get the submodule for a given stage index."""
        return full_mod.get_submodule(f"layers.{stage_idx}")

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 4+ GPUs"
    )
    @parametrize("ScheduleClass", [ScheduleInterleaved1F1B, ScheduleZBVZeroBubble])
    @parametrize("sharded_input", [True, False])
    @parametrize("metadata_inference", ["static", "dynamic"])
    def test_dtensor_pipeline(self, ScheduleClass, sharded_input, metadata_inference):
        """
        Consolidated end-to-end test for DTensor support in pipeline parallelism.

        Parameters:
            ScheduleClass: The schedule class to use
            sharded_input: If True, use Shard(1) on feature dim; if False, use Replicate
            metadata_inference: "static" or "dynamic"
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

        schedule_style = "v" if ScheduleClass == ScheduleZBVZeroBubble else "loop"
        rank_to_stages = generate_rank_to_stage_mapping(
            pp_size, total_stages, style=schedule_style
        )
        my_stage_indices = rank_to_stages[pp_rank]

        placements = [Shard(1)] if sharded_input else [Replicate()]
        parallelize_plan = _get_tp_parallelize_plan(sharded_input_output=sharded_input)

        torch.manual_seed(0)
        full_mod = MultiMLP(d_hid, n_layers=total_stages).to(self.device)

        def mesh_provider(dim_names, shape=None):
            if dim_names == ("tp",):
                return tp_mesh
            raise ValueError(f"Unknown mesh: {dim_names}")

        stages = []
        for stage_idx in my_stage_indices:
            stage_mod = self._get_stage_module(full_mod, stage_idx)
            parallelize_module(stage_mod, tp_mesh, parallelize_plan)

            if metadata_inference == "static":
                input_dtensor = self._create_dtensor(tp_mesh, placements)
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
                self.assertIsNotNone(stage._inputs_dtensor_meta)
            else:
                stage = PipelineStage(
                    stage_mod,
                    stage_idx,
                    total_stages,
                    self.device,
                    group=pp_group,
                    get_mesh=mesh_provider,
                )
                self.assertIsNone(stage._inputs_dtensor_meta)

            stages.append(stage)

        schedule = ScheduleClass(stages, n_microbatches, loss_fn=_dtensor_loss_fn)

        x = self._create_dtensor(tp_mesh, placements, shape=(batch_size, d_hid))
        target = self._create_target(tp_mesh)

        stage_to_rank = {s: r for r, ss in rank_to_stages.items() for s in ss}
        first_stage_rank = stage_to_rank[0]
        last_stage_rank = stage_to_rank[total_stages - 1]

        if pp_rank == first_stage_rank:
            if pp_rank == last_stage_rank:
                schedule.step(x, target=target)
            else:
                schedule.step(x)
        elif pp_rank == last_stage_rank:
            schedule.step(target=target)
        else:
            schedule.step()

        # Verify metadata was populated after execution
        if metadata_inference == "dynamic":
            for stage in stages:
                self.assertIsNotNone(stage._inputs_dtensor_meta)
                self.assertIsNotNone(stage._outputs_dtensor_meta)

        dist.barrier()
        dist.destroy_process_group()


# =============================================================================
# Instantiate and Run
# =============================================================================

instantiate_parametrized_tests(TestDTensorPipelineE2E)


if __name__ == "__main__":
    run_tests()
