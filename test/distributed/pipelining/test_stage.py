# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import os
import tempfile
from unittest import mock

from model_registry import ExampleCode, ModelWithKwargs, MultiMLP

import torch
import torch.distributed as dist
from torch.autograd.graph import GradientEdge
from torch.distributed.pipelining import (
    build_stage,
    pipeline,
    PipelineStage,
    ScheduleGPipe,
)
from torch.distributed.pipelining._utils import (
    extract_tensor_meta,
    PipeliningMetadataError,
)
from torch.distributed.pipelining.stage import _early_send_release_default, _RecvInfo
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
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
from torch.utils._pytree import tree_map_only


d_hid = 512
batch_size = 256
chunks = 8

device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
backend = dist.get_default_backend_for_device(device_type)

torch.manual_seed(0)


class PipelineStageMetadataInferenceTest(TestCase):
    def test_dynamic_metadata_inference_restores_module_buffers(self):
        class BufferMutatingModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
                self.register_buffer("counter", torch.zeros(4))
                self.register_buffer("scale", torch.ones(4))

            def forward(self, x):
                with torch.no_grad():
                    self.counter.add_(1)
                out = self.linear(x) * self.scale
                if out.requires_grad:
                    out.register_hook(self._backward_hook)
                return out

            def _backward_hook(self, grad):
                with torch.no_grad():
                    self.counter.add_(10)
                return grad

        device = torch.device("cpu")
        init_pg = not dist.is_initialized()
        with tempfile.TemporaryDirectory() as tmpdir:
            if init_pg:
                dist.init_process_group(
                    "gloo",
                    init_method=f"file://{os.path.join(tmpdir, 'pg')}",
                    rank=0,
                    world_size=1,
                )
            try:
                mod = BufferMutatingModule().to(device)
                stage = PipelineStage(
                    mod,
                    stage_index=0,
                    num_stages=1,
                    device=device,
                )
                schedule = ScheduleGPipe(
                    stage,
                    n_microbatches=1,
                    loss_fn=lambda out, target: out.sum() + target.sum() * 0,
                )

                initial_counter = mod.counter.clone()
                initial_scale = mod.scale.clone()
                x = torch.randn(2, 4, device=device, requires_grad=True)
                target = torch.zeros((), device=device)

                # This exercises the full metadata-inference lifecycle. The
                # scale buffer is saved by autograd, so restoring buffers before
                # backward metadata inference would bump its version counter.
                schedule._initialize_stage((x,), {}, target=target)

                self.assertEqual(mod.counter, initial_counter)
                self.assertEqual(mod.scale, initial_scale)
            finally:
                if init_pg:
                    dist.destroy_process_group()


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


class OutputSlotReleaseTest(TestCase):
    """Tests forward-cache output release."""

    def setUp(self):
        super().setUp()
        self._tmpdir = tempfile.TemporaryDirectory()
        self._owns_pg = not dist.is_initialized()
        if self._owns_pg:
            dist.init_process_group(
                "gloo",
                init_method=f"file://{os.path.join(self._tmpdir.name, 'pg')}",
                rank=0,
                world_size=1,
            )

    def tearDown(self):
        if self._owns_pg and dist.is_initialized():
            dist.destroy_process_group()
        self._tmpdir.cleanup()
        super().tearDown()

    def _make_stage(self, *, stage_index=0, num_stages=2, dst_stages=None, **kwargs):
        stage = PipelineStage(
            torch.nn.Linear(d_hid, d_hid),
            stage_index,
            num_stages,
            torch.device("cpu"),
            **kwargs,
        )
        stage.has_backward = True
        # Explicit, so the test does not depend on the default.
        stage.early_send_release = True
        # Avoid peer-dependent forward setup.
        stage.act_send_info = {0: dst_stages or [stage_index + 1]}
        return stage

    def _forward_and_send(self, stage, mb_index=0):
        x = torch.randn(batch_size, d_hid)
        stage.forward_one_chunk(mb_index, (x,))
        return stage.get_fwd_send_ops(mb_index)

    def test_output_released_when_send_retires(self):
        stage = self._make_stage()
        ops = self._forward_and_send(stage)
        entry = stage.fwd_cache[0]

        self.assertEqual(len(ops), 1)
        self.assertEqual(entry.pending_consumers, [1])
        self.assertIsNotNone(entry.live_outputs[0])
        self.assertIsInstance(entry.backward_roots[0], GradientEdge)

        stage.retire_fwd_sends(0)

        self.assertEqual(entry.pending_consumers, [0])
        self.assertIsNone(entry.live_outputs[0])
        self.assertIs(entry.stage_output_for_backward()[0], entry.backward_roots[0])

    def test_output_kept_until_every_destination_retires(self):
        stage = self._make_stage(num_stages=3, dst_stages=[1, 2])
        ops = self._forward_and_send(stage)
        entry = stage.fwd_cache[0]

        # Destinations share one batched buffer lease.
        self.assertEqual(len(ops), 2)
        self.assertIs(ops[0].tensor, ops[1].tensor)
        self.assertEqual(entry.pending_consumers, [1])

        stage.retire_fwd_sends(0)
        self.assertIsNone(entry.live_outputs[0])

    def test_last_stage_output_never_released(self):
        stage = self._make_stage(stage_index=0, num_stages=1, dst_stages=[None])
        ops = self._forward_and_send(stage)
        entry = stage.fwd_cache[0]

        self.assertEqual(ops, [])
        self.assertEqual(entry.releasable, (False,))
        self.assertIsNone(entry.backward_roots[0])
        self.assertEqual(stage._retained_output_reason, "last-stage output")

        stage.retire_fwd_sends(0)
        self.assertIsNotNone(entry.live_outputs[0])

    def test_early_send_release_defaults_on(self):
        # Do not inherit the caller's opt-out.
        with mock.patch.dict(os.environ):
            os.environ.pop("TORCH_PIPELINING_EARLY_SEND_RELEASE", None)
            self.assertTrue(_early_send_release_default())
            stage = PipelineStage(
                torch.nn.Linear(d_hid, d_hid), 0, 2, torch.device("cpu")
            )
            self.assertTrue(stage.early_send_release)

    def test_early_send_release_env_opt_out(self):
        with mock.patch.dict(os.environ, {"TORCH_PIPELINING_EARLY_SEND_RELEASE": "0"}):
            self.assertFalse(_early_send_release_default())
            stage = PipelineStage(
                torch.nn.Linear(d_hid, d_hid), 0, 2, torch.device("cpu")
            )
            self.assertFalse(stage.early_send_release)

    def test_forward_only_keeps_no_backward_root(self):
        # An unused edge would retain the forward-only graph.
        stage = self._make_stage()
        stage.has_backward = False
        self._forward_and_send(stage)
        entry = stage.fwd_cache[0]

        self.assertEqual(entry.backward_roots, (None,))
        stage.retire_fwd_sends(0)
        self.assertIsNone(entry.live_outputs[0])

    def test_release_disabled_keeps_output(self):
        stage = self._make_stage()
        stage.early_send_release = False
        self._forward_and_send(stage)
        entry = stage.fwd_cache[0]

        self.assertEqual(entry.releasable, (False,))
        stage.retire_fwd_sends(0)
        self.assertIsNotNone(entry.live_outputs[0])
        self.assertIsInstance(entry.stage_output_for_backward()[0], torch.Tensor)

    def test_dw_builder_keeps_output(self):
        stage = self._make_stage(dw_builder=lambda: (lambda: None))
        self._forward_and_send(stage)
        entry = stage.fwd_cache[0]

        self.assertEqual(entry.releasable, (False,))
        stage.retire_fwd_sends(0)
        self.assertIsNotNone(entry.live_outputs[0])

    def test_tensor_subclass_output_kept(self):
        class TaggedTensor(torch.Tensor):
            pass

        class TaggedModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(d_hid, d_hid)

            def forward(self, x):
                return TaggedTensor(self.linear(x))

        stage = PipelineStage(TaggedModule(), 0, 2, torch.device("cpu"))
        stage.has_backward = True
        # Explicit, so the subclass check is what rejects the slot.
        stage.early_send_release = True
        stage.act_send_info = {0: [1]}
        stage.forward_one_chunk(0, (torch.randn(batch_size, d_hid),))
        entry = stage.fwd_cache[0]

        self.assertEqual(entry.releasable, (False,))
        self.assertIn("TaggedTensor", stage._retained_output_reason)

    def test_retire_after_backward_consumed_the_entry(self):
        # The final drain may run after backward pops the entry.
        stage = self._make_stage()
        self._forward_and_send(stage)
        stage.fwd_cache.pop(0)
        stage.retire_fwd_sends(0)

    def test_split_backward_drops_edge_roots(self):
        """Drop edge roots before split weight backward."""
        # The first stage keeps its roots for weight backward.
        stage = self._make_stage(stage_index=1, num_stages=3)
        stage.chunks = 1
        stage._stage_meta.inputs = (None,)

        act_buffer = torch.randn(batch_size, d_hid, requires_grad=True)
        stage.args_recv_info = {
            0: (_RecvInfo("act_recv", 0, act_buffer, extract_tensor_meta(act_buffer)),)
        }
        stage.forward_one_chunk(0, ())
        stage.get_fwd_send_ops(0)
        stage.retire_fwd_sends(0)
        self.assertIsNone(stage.fwd_cache[0].live_outputs[0])

        grad_buffer = torch.randn(batch_size, d_hid)
        stage.grad_recv_info = {
            0: (
                _RecvInfo(
                    "grad_recv", 1, grad_buffer, extract_tensor_meta(grad_buffer)
                ),
            )
        }
        stage.backward_one_chunk(0, full_backward=False)

        roots = stage.backward_state[0][2]
        self.assertTrue(all(not isinstance(root, GradientEdge) for root in roots))

    def test_send_after_release_is_rejected(self):
        stage = self._make_stage()
        self._forward_and_send(stage)
        stage.retire_fwd_sends(0)

        with self.assertRaisesRegex(AssertionError, "released before its send"):
            stage.get_fwd_send_ops(0)


class StageTest(MultiProcContinuousTest):
    @classmethod
    def backend_str(cls) -> str:
        # Testing with NCCL backend
        return backend

    @classmethod
    def device_type(cls) -> str:
        return device_type

    @property
    def device(self) -> torch.device:
        return torch.device(device_type, self.rank)

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 2+ GPUs"
    )
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
        if not all(k in old_keys for k in submod_keys):
            raise AssertionError(
                f"Some keys not found in old_keys: {[k for k in submod_keys if k not in old_keys]}"
            )

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 2+ GPUs"
    )
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
        if not all(k in old_keys for k in submod_keys):
            raise AssertionError(
                f"Some keys not found in old_keys: {[k for k in submod_keys if k not in old_keys]}"
            )

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 2+ GPUs"
    )
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

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 2+ GPUs"
    )
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

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 2+ GPUs"
    )
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
                lambda msg: f"{msg}\nNon-last stage (rank {self.rank}) should not store output chunks",
            )

        # Clear the schedule and stage caches
        stage.clear_runtime_states()
        if self.rank == self.world_size - 1:
            # Last stage should have output_chunks populated
            self.assertEqual(
                len(stage.output_chunks), 0, "Last stage should store output chunks"
            )


instantiate_parametrized_tests(StageTest)


class StageNegativeTest(MultiProcContinuousTest):
    @classmethod
    def backend_str(cls) -> str:
        return backend

    @classmethod
    def device_type(cls) -> str:
        return device_type

    @property
    def device(self) -> torch.device:
        return torch.device(device_type, self.rank)

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 2+ GPUs"
    )
    def test_shape_prop_mismatch(self):
        """Tests shape prop errors are raised"""
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
        stage._runtime_validate = True

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
            with self.assertRaisesRegex(PipeliningMetadataError, "shape mismatch"):
                _run_step(torch.randn(batch_size + 1, d_hid, device=self.device))

            with self.assertRaisesRegex(PipeliningMetadataError, "dtype mismatch"):
                _run_step(x.to(torch.int32))

            # output of stage's mlp layer will be flattened by this hook, the stage should err
            handle = stage_mod.register_forward_hook(get_flatten_hook())
            with self.assertRaisesRegex(PipeliningMetadataError, "shape mismatch"):
                _run_step(x)
            handle.remove()

            stage_mod.register_forward_hook(get_dtype_change_hook(torch.bfloat16))
            with self.assertRaisesRegex(PipeliningMetadataError, "dtype mismatch"):
                _run_step(x)

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 2+ GPUs"
    )
    def test_custom_dw_errors(self):
        """Tests expected errors are raised"""
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
