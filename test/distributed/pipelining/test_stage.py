# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import os

from model_registry import ExampleCode, ModelWithKwargs, MultiMLP

import torch
import torch.distributed as dist
from torch.distributed.pipelining import (
    build_stage,
    pipeline,
    PipelineStage,
    ScheduleGPipe,
)
from torch.distributed.pipelining._utils import PipeliningShapeError
from torch.testing._internal.common_cuda import TEST_MULTIGPU
from torch.testing._internal.common_distributed import (
    MultiProcContinousTest,
    MultiProcessTestCase,
    requires_nccl,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skip_but_pass_in_sandcastle,
    skip_but_pass_in_sandcastle_if,
)
from torch.utils._pytree import tree_map_only


d_hid = 512
batch_size = 256
chunks = 4

device_type = "cuda"

torch.manual_seed(0)


def get_dtype_change_hook(new_dtype):
    """A simple hook for simulating mixed precision"""

    def dtype_change_hook(module, input, output):
        def f(x):
            return x.to(new_dtype)

        return tree_map_only(torch.Tensor, f, output)

    return dtype_change_hook


def get_flatten_hook():
    """A simple hook for simulating wrong model output shape"""

    def flatten_hook(module, input, output):
        def f(x):
            return x.flatten()

        return tree_map_only(torch.Tensor, f, output)

    return flatten_hook


class StageTest(MultiProcContinousTest):
    @classmethod
    def backend_str(cls) -> str:
        # Testing with NCCL backend
        return "nccl"

    @classmethod
    def device_type(cls) -> str:
        return device_type

    @property
    def device(self) -> torch.device:
        return torch.device(device_type, self.rank)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize("ModelClass", [ExampleCode, MultiMLP])
    def test_tracer(self, ModelClass):
        mod = ModelClass(d_hid, self.world_size)
        mod.to(self.device)

        x = torch.randn(batch_size, d_hid, device=self.device)
        x_mb = x.chunk(chunks)[0]

        split_spec = mod.split_spec if hasattr(mod, "split_spec") else None
        pipe = pipeline(
            mod,
            mb_args=(x_mb,),
            split_spec=split_spec,
        )

        stage = pipe.build_stage(
            self.rank,
            self.device,
        )

        # Attach to a schedule
        schedule = ScheduleGPipe(stage, chunks)

        # Run
        def _run_step(x):
            if self.rank == 0:
                return schedule.step(x)
            else:
                return schedule.step()

        out = _run_step(x)
        # Last rank checks result
        if self.rank == self.world_size - 1:
            ref_out = mod(x)
            torch.testing.assert_close(out, ref_out, atol=1e-3, rtol=5e-2)

        # Test qualname mapping
        submod_keys = stage.submod.state_dict().keys()
        # Confirm keys are consistent with original model
        old_keys = mod.state_dict().keys()
        assert all(k in old_keys for k in submod_keys)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize("ModelClass", [ModelWithKwargs])
    def test_tracer_kwargs(self, ModelClass):
        mod = ModelClass(d_hid, self.world_size)
        mod.to(self.device)

        x = torch.randn(batch_size, d_hid, device=self.device)
        y = torch.randn(batch_size, d_hid, device=self.device)

        x_mb = x.chunk(chunks)[0]
        y_mb = y.chunk(chunks)[0]

        pipe = pipeline(
            mod,
            mb_args=(x_mb,),
            mb_kwargs={"y": y_mb},
        )

        stage_mod = pipe.get_stage_module(self.rank)

        # Test build_stage
        stage = build_stage(
            stage_mod,
            self.rank,
            pipe.info(),
            self.device,
        )

        # Attach to a schedule
        schedule = ScheduleGPipe(stage, chunks)

        # Run
        if self.rank == 0:
            out = schedule.step(x, y=y)
        else:
            out = schedule.step()

        # Last rank checks result
        if self.rank == self.world_size - 1:
            ref_out = mod(x, y=y)
            torch.testing.assert_close(out, ref_out, atol=1e-3, rtol=5e-2)

        # Test qualname mapping
        submod_keys = stage.submod.state_dict().keys()
        # Confirm keys are consistent with original model
        old_keys = mod.state_dict().keys()
        assert all(k in old_keys for k in submod_keys)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_manual(self):
        full_mod = MultiMLP(d_hid, n_layers=self.world_size)
        full_mod.to(self.device)
        stage_mod = full_mod.get_submodule(f"layers.{self.rank}")

        x = torch.randn(batch_size, d_hid, device=self.device)

        stage = PipelineStage(
            stage_mod,
            self.rank,
            self.world_size,
            self.device,
        )

        # Attach to a schedule
        schedule = ScheduleGPipe(stage, chunks)

        # Run
        def _run_step(x):
            if self.rank == 0:
                return schedule.step(x)
            else:
                return schedule.step()

        out = _run_step(x)
        # Last rank checks result
        if self.rank == self.world_size - 1:
            ref_out = full_mod(x)
            torch.testing.assert_close(out, ref_out)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_custom_dw_with_fb_schedule(self):
        """Tests that separate weight grad function 'dw_runner' gets run under a schedule that's only aware of F/B."""
        full_mod = MultiMLP(d_hid, n_layers=self.world_size)
        full_mod.to(self.device)
        stage_mod = full_mod.get_submodule(f"layers.{self.rank}")

        x = torch.randn(batch_size, d_hid, device=self.device)
        target = torch.randn(batch_size, d_hid, device=self.device)

        class CustomState:
            def __init__(self) -> None:
                self.i = 0

            def dw_builder(self):
                """This simulates a function attached to a model with a custom backward.
                Each call to builder gives a new dw_runner that has some updated state to compute the latest dw.
                """

                def dw_runner():
                    # This inner function would be called by PipelineStage during `backward_weight_one_chunk`
                    print(f"dw called {self.i}th time")
                    self.i += 1

                return dw_runner

        cs = CustomState()

        stage = PipelineStage(
            stage_mod,
            self.rank,
            self.world_size,
            self.device,
            dw_builder=cs.dw_builder,
        )

        # Attach to a schedule
        schedule = ScheduleGPipe(
            stage, chunks, loss_fn=torch.nn.MSELoss(reduction="sum")
        )

        # Run
        def _run_step(x):
            if self.rank == 0:
                return schedule.step(x)
            elif self.rank == self.world_size - 1:
                return schedule.step(target=target)
            else:
                return schedule.step()

        out = _run_step(x)

        self.assertEqual(cs.i, chunks)

        # Last rank checks result
        if self.rank == self.world_size - 1:
            ref_out = full_mod(x)
            torch.testing.assert_close(out, ref_out)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_output_chunks_memory_usage(self):
        """Test that output_chunks doesn't store memory for non-first stages."""
        full_mod = MultiMLP(d_hid, n_layers=self.world_size)
        full_mod.to(self.device)
        stage_mod = full_mod.get_submodule(f"layers.{self.rank}")
        x = torch.randn(batch_size, d_hid, device=self.device)
        target = torch.randn(batch_size, d_hid, device=self.device)
        stage = PipelineStage(
            stage_mod,
            self.rank,
            self.world_size,
            self.device,
        )
        self.assertEqual(
            len(stage.output_chunks), 0, "output_chunks should be empty initially"
        )

        schedule = ScheduleGPipe(
            stage, chunks, loss_fn=torch.nn.MSELoss(reduction="sum")
        )

        def _run_step(x):
            if self.rank == 0:
                return schedule.step(x)
            elif self.rank == self.world_size - 1:
                return schedule.step(target=target)
            else:
                return schedule.step()

        _run_step(x)

        # Verify fwd_cache is empty
        self.assertEqual(len(stage.fwd_cache), 0, "fwd_cache should be cleared")

        # Check output_chunks state after step
        if self.rank == self.world_size - 1:
            self.assertEqual(
                len(stage.output_chunks),
                chunks,
                "Last stage should store output chunks",
            )
        else:
            self.assertEqual(
                len(stage.output_chunks),
                0,
                f"Non-last stage (rank {self.rank}) should not store output chunks",
            )

        # Clear the schedule and stage caches
        stage.clear_runtime_states()
        if self.rank == self.world_size - 1:
            # Last stage should have output_chunks populated
            self.assertEqual(
                len(stage.output_chunks), 0, "Last stage should store output chunks"
            )


instantiate_parametrized_tests(StageTest)


class StageNegativeTest(MultiProcessTestCase):
    @property
    def world_size(self) -> int:
        return torch.get_device_module(device_type).device_count()

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
        dist.init_process_group(
            backend="nccl",
            store=store,
            rank=self.rank,
            world_size=self.world_size,
            device_id=self.device,
        )

    @requires_nccl()
    @skip_but_pass_in_sandcastle("Flaky in CI")
    def test_shape_prop_mismatch(self):
        """Tests shape prop errors are raised"""
        self.init_pg()

        full_mod = MultiMLP(d_hid, n_layers=self.world_size)
        full_mod.to(self.device)
        stage_mod = full_mod.get_submodule(f"layers.{self.rank}")

        x = torch.randn(batch_size, d_hid, device=self.device)

        stage = PipelineStage(
            stage_mod,
            self.rank,
            self.world_size,
            self.device,
        )

        # Attach to a schedule
        schedule = ScheduleGPipe(stage, chunks)

        # Run
        def _run_step(x):
            if self.rank == 0:
                return schedule.step(x)
            else:
                return schedule.step()

        _run_step(x)

        if self.rank == 0:
            with self.assertRaisesRegex(PipeliningShapeError, "shape mismatch"):
                _run_step(torch.randn(batch_size + 1, d_hid, device=self.device))

            with self.assertRaisesRegex(PipeliningShapeError, "dtype mismatch"):
                _run_step(x.to(torch.int32))

            # output of stage's mlp layer will be flattened by this hook, the stage should err
            handle = stage_mod.register_forward_hook(get_flatten_hook())
            with self.assertRaisesRegex(PipeliningShapeError, "shape mismatch"):
                _run_step(x)
            handle.remove()

            stage_mod.register_forward_hook(get_dtype_change_hook(torch.bfloat16))
            with self.assertRaisesRegex(PipeliningShapeError, "dtype mismatch"):
                _run_step(x)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_custom_dw_errors(self):
        """Tests expected errors are raised"""
        self.init_pg()

        full_mod = MultiMLP(d_hid, n_layers=self.world_size)
        full_mod.to(self.device)
        stage_mod = full_mod.get_submodule(f"layers.{self.rank}")

        stage_with_dw_builder = PipelineStage(
            stage_mod,
            self.rank,
            self.world_size,
            self.device,
            dw_builder=lambda: None,
        )
        stage_with_dw_builder._has_backward = True
        with self.assertRaisesRegex(AssertionError, "backward_one_chunk"):
            stage_with_dw_builder.backward_weight_one_chunk(bwd_chunk_id=0)


if __name__ == "__main__":
    run_tests()
