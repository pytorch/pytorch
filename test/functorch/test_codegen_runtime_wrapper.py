# Owner(s): ["oncall: pt2"]

"""
Tests for codegen'ing the RuntimeWrapper orchestration in aot_autograd.

The codegen'd runtime wrapper collapses _RuntimeCompiledFnInvoker.run,
_RuntimeForwardEpilogue.capture_orig_inputs, increment_mutation_versions,
and finalize into a single generated function with all branches resolved
at compile time: trace_joint, detach indices, epilogue_args_idx, number
of mutated inputs, output arity, and dynamic dims are all baked in.

Tests verify that a "runtime_wrapper_orchestration" artifact is emitted
via trace_structured.
"""

import warnings

from common_utils import capture_codegen_source

import torch
import torch._dynamo
import torch._functorch.config as functorch_config
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


class TestCodegenRuntimeWrapper(TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()

    def assertCallCompiledFnIsFirstStatement(self, source):
        # For runtime overhead, everything before the compiled function call must
        # stay in C++; FlexAttention cares about time to first kernel launch.
        lines = [line for line in source.splitlines() if line.strip()]
        self.assertGreaterEqual(len(lines), 2)
        self.assertEqual(lines[0], "def _runtime_wrapper(_compiled_fn_, args):")
        self.assertEqual(
            lines[1],
            "    orig_inputs, all_outs = _call_compiled_fn_(_call_spec_, _compiled_fn_, args)",
        )

    def test_inference_simple(self):
        """
        Simple inference: no mutations, no aliases. Generated code should
        use the inference path (grad disabled) with empty orig_inputs.
        """
        with capture_codegen_source("runtime_wrapper_orchestration") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                return x * 2

            x = torch.randn(4)
            out = f(x)

        self.assertEqual(out, x * 2)
        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertCallCompiledFnIsFirstStatement(source)
        self.assertIn("orig_inputs, all_outs = _call_compiled_fn_", source)
        self.assertIn("_call_spec_", source)
        self.assertIn("_call_compiled_fn_", source)
        self.assertNotIn("_force_view_tracking_", source)
        self.assertNotIn("_is_view_replay_enabled", source)
        self.assertNotIn("_set_view_replay_enabled", source)

    def test_training_simple(self):
        """
        Simple training path: no mutations. Generated code should use
        call_compiled_fn to prepare state and invoke the compiled function.
        """
        with capture_codegen_source("runtime_wrapper_orchestration") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                return x * 2

            x = torch.randn(4, requires_grad=True)
            out = f(x)

        self.assertEqual(out, x * 2)
        out.sum().backward()
        self.assertEqual(x.grad, torch.full((4,), 2.0))

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertCallCompiledFnIsFirstStatement(source)
        self.assertIn("_call_spec_", source)
        self.assertIn("_call_compiled_fn_", source)
        self.assertNotIn("torch.enable_grad()", source)

    def test_training_with_detach_indices(self):
        """
        Training path with a non-leaf input whose gradient is None in
        the backward graph. The input must be detached before calling
        the joint graph. Generated code should pass args to call_compiled_fn
        so it can copy and detach selected inputs.
        """
        with capture_codegen_source("runtime_wrapper_orchestration") as captured:
            y_base = torch.randn(4, requires_grad=True)
            y = y_base * 2  # non-leaf tensor with grad_fn

            @torch.compile(backend="aot_eager")
            def f(x, y):
                return x * y.detach()

            x = torch.randn(4, requires_grad=True)
            out = f(x, y)
            out.sum().backward()

        self.assertEqual(out, x * y)
        self.assertIsNotNone(x.grad)
        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("_call_spec_", source)
        self.assertIn(
            "orig_inputs, all_outs = _call_compiled_fn_(_call_spec_, _compiled_fn_, args)",
            source,
        )
        self.assertNotIn("args_", source)
        self.assertNotIn(".detach()", source)

    def test_inference_with_mutation(self):
        """
        Inference with input mutation. With keep_inference_input_mutations,
        mutations are kept in-graph so the runtime wrapper just increments
        versions (no runtime _apply_mutations_ needed).
        """
        with capture_codegen_source("runtime_wrapper_orchestration") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                x.add_(1)
                return x.clone()

            x = torch.randn(4)
            x_ref = x.clone()
            out = f(x)

        self.assertEqual(x, x_ref + 1)
        self.assertEqual(out, x_ref + 1)

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("_call_compiled_fn_", source)
        self.assertNotIn("_increment_version_", source)

    def test_inference_with_output_alias(self):
        """
        Inference with output aliased to input. Generated code should
        capture orig_inputs and call _replay_aliases_.
        """
        with capture_codegen_source("runtime_wrapper_orchestration") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                return x.view(-1)

            x = torch.randn(2, 3)
            out = f(x)

        self.assertEqual(out, x.view(-1))
        self.assertEqual(out.data_ptr(), x.data_ptr())

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("_replay_aliases_", source)
        self.assertIn("orig_inputs, all_outs = _call_compiled_fn_", source)

    def test_inference_with_mutation_and_alias(self):
        """
        Inference: input mutation + output alias. With
        keep_inference_input_mutations, mutations are in-graph. The
        runtime wrapper handles the alias replay.
        """
        with capture_codegen_source("runtime_wrapper_orchestration") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                x.add_(1)
                return x.view(-1)

            x = torch.randn(2, 3)
            x_ref = x.clone()
            out = f(x)

        self.assertEqual(x, x_ref + 1)
        self.assertEqual(out, (x_ref + 1).view(-1))

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("_replay_aliases_", source)

    def test_training_with_alias(self):
        """
        Training path with output alias and backward correctness.
        """
        with capture_codegen_source("runtime_wrapper_orchestration") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                return x * 2, x.view(-1)

            x = torch.randn(2, 3, requires_grad=True)
            out1, out2 = f(x)

        self.assertEqual(out1, x * 2)
        self.assertEqual(out2, x.view(-1))

        out1.sum().backward()
        self.assertEqual(x.grad, torch.full((2, 3), 2.0))

        self.assertEqual(len(captured), 1)

    def test_multiple_inputs_mutation_version_increment(self):
        """
        Multiple inputs with mutations. Generated code should increment
        versions for all mutated inputs.
        """
        with capture_codegen_source("runtime_wrapper_orchestration") as captured:

            @torch.compile(backend="aot_eager")
            def f(x, y):
                x.add_(1)
                y.mul_(2)
                return x + y

            x = torch.randn(4)
            y = torch.randn(4)
            x_ref, y_ref = x.clone(), y.clone()
            out = f(x, y)

        self.assertEqual(x, x_ref + 1)
        self.assertEqual(y, y_ref * 2)
        self.assertEqual(out, (x_ref + 1) + (y_ref * 2))

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("_call_compiled_fn_", source)
        self.assertNotIn("_increment_version_", source)

    def test_output_arity_validation_baked(self):
        """
        The expected output arity should be baked into the generated code
        as a constant, not computed at runtime.
        """
        with capture_codegen_source("runtime_wrapper_orchestration") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                return x + 1, x * 2, x - 1

            f(torch.randn(4))

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("if len(all_outs) != 3:", source)

    @skipIfTorchDynamo("dynamo handles mutations in-graph")
    def test_split_index_baked(self):
        """
        When there are mutated inputs that produce runtime mutation
        indices, the split index between updated_inputs and fw_outs
        should be baked as a constant. Uses aot_function directly to
        avoid keep_inference_input_mutations.
        """
        from functorch.compile import nop
        from torch._functorch.aot_autograd import aot_function

        with capture_codegen_source("runtime_wrapper_orchestration") as captured:

            def f(x, y):
                x.add_(1)
                return y * 2

            compiled_f = aot_function(f, nop, keep_inference_input_mutations=False)
            compiled_f(torch.randn(4), torch.randn(4))

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("updated_inputs = all_outs[:1]", source)
        self.assertIn("fw_outs = all_outs[1:]", source)
        self.assertIn("_apply_mutations_", source)

    @skipIfTorchDynamo("dynamo handles metadata mutations in-graph")
    def test_metadata_mutation(self):
        """
        Metadata-only mutation (transpose_). Verify the generated wrapper
        correctly applies metadata mutations via _apply_mutations_.
        """
        from functorch.compile import nop
        from torch._functorch.aot_autograd import aot_function

        with capture_codegen_source("runtime_wrapper_orchestration") as captured:

            def f(x):
                x.transpose_(1, 0)
                return x + 1

            x = torch.randn(3, 4).add(0)
            compiled_f = aot_function(f, nop)
            out = compiled_f(x)

        self.assertEqual(x.shape, (4, 3))
        self.assertEqual(out.shape, (4, 3))
        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("_apply_mutations_", source)

    def test_inference_disable_amp(self):
        """
        Inference path with autocast active at compile time. Generated code
        should use the call_compiled_fn path.
        """
        with capture_codegen_source("runtime_wrapper_orchestration") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                return x * 2

            with torch.autocast("cpu"):
                f(torch.randn(4))

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("_call_spec_", source)
        self.assertIn("_call_compiled_fn_", source)
        self.assertNotIn("_DisableAutocast_", source)
        self.assertNotIn("torch._C._set_grad_enabled(False)", source)

    def test_training_disable_amp(self):
        """
        Training path with autocast active at compile time. Generated code
        should use the same call_compiled_fn path as regular training.
        """
        with capture_codegen_source("runtime_wrapper_orchestration") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                return x * 2

            with torch.autocast("cpu"):
                x = torch.randn(4, requires_grad=True)
                out = f(x)
                out.sum().backward()

        self.assertEqual(x.grad, torch.full((4,), 2.0))
        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("_call_spec_", source)
        self.assertIn("_call_compiled_fn_", source)
        self.assertNotIn("_DisableAutocast_", source)

    def test_dynamic_dims(self):
        """
        With dynamic=True, output dimensions are symbolic. Generated code
        should call _maybe_mark_dynamic_helper_ for dynamic outputs.
        """
        with capture_codegen_source("runtime_wrapper_orchestration") as captured:

            @torch.compile(backend="aot_eager", dynamic=True)
            def f(x):
                return x * 2

            f(torch.randn(4))

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("_mark_dynamic_", source)

    @skipIfTorchDynamo("dynamo handles grad mode changes in-graph")
    def test_grad_enabled_mutation(self):
        """
        Function that mutates grad_enabled state. Generated code should
        replay the mutation via torch._C._set_grad_enabled at the end.
        """
        from functorch.compile import nop
        from torch._functorch.aot_autograd import aot_function

        with capture_codegen_source("runtime_wrapper_orchestration") as captured:

            def f(x):
                torch._C._set_grad_enabled(False)
                return x * 2

            compiled_f = aot_function(f, nop)
            prior = torch.is_grad_enabled()
            try:
                compiled_f(torch.randn(4))
                self.assertFalse(torch.is_grad_enabled())
            finally:
                torch.set_grad_enabled(prior)

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("torch._C._set_grad_enabled(False)", source)

    def test_many_mutations(self):
        """
        Five inputs all mutated. Generated code should increment versions
        for all of them.
        """
        with capture_codegen_source("runtime_wrapper_orchestration") as captured:

            @torch.compile(backend="aot_eager")
            def f(a, b, c, d, e):
                a.add_(1)
                b.add_(2)
                c.add_(3)
                d.add_(4)
                e.add_(5)
                return a + b + c + d + e

            tensors = [torch.randn(4) for _ in range(5)]
            refs = [t.clone() for t in tensors]
            f(*tensors)

        for i, (t, r) in enumerate(zip(tensors, refs)):
            self.assertEqual(t, r + (i + 1))

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("_call_compiled_fn_", source)
        self.assertNotIn("_increment_version_", source)

    def test_multiple_output_aliases_different_inputs(self):
        """
        Two outputs aliasing two different inputs. Generated code should
        capture both inputs in orig_inputs and call _replay_aliases_.
        """
        with capture_codegen_source("runtime_wrapper_orchestration") as captured:

            @torch.compile(backend="aot_eager")
            def f(x, y):
                return x.view(-1), y.view(-1)

            x = torch.randn(2, 3)
            y = torch.randn(3, 2)
            out1, out2 = f(x, y)

        self.assertEqual(out1, x.view(-1))
        self.assertEqual(out2, y.view(-1))
        self.assertEqual(out1.data_ptr(), x.data_ptr())
        self.assertEqual(out2.data_ptr(), y.data_ptr())

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("_replay_aliases_", source)
        self.assertIn("orig_inputs, all_outs = _call_compiled_fn_", source)

    def test_first_invocation_ctx_threaded_through_codegen(self):
        """
        runtime_wrapper owns _FirstInvocationContext and wraps the codegen'd
        wrapper in it on first call. This activates
        _AnalyzeCustomOpInputOutputMode, which checks custom op aliasing.
        Verify the mode fires on first call and not on second.
        """
        with torch.library._scoped_library("test_rw", "FRAGMENT") as lib:
            lib.define("alias_op(Tensor x) -> Tensor")
            lib.impl("alias_op", lambda x: x.view_as(x), "CompositeExplicitAutograd")
            lib.impl("alias_op", lambda x: x.view_as(x), "Meta")

            @torch.compile(backend="aot_eager")
            def f(x):
                return torch.ops.test_rw.alias_op(x) + 1

            x = torch.randn(4)
            with functorch_config.patch(
                check_custom_op_aliasing=True,
                error_on_custom_op_aliasing=False,
            ):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    f(x)
                    aliasing_warnings = [
                        x for x in w if "may not alias any inputs" in str(x.message)
                    ]
                    self.assertEqual(len(aliasing_warnings), 1)

                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    f(x)
                    aliasing_warnings = [
                        x for x in w if "may not alias any inputs" in str(x.message)
                    ]
                    self.assertEqual(len(aliasing_warnings), 0)

    @skipIfTorchDynamo("dynamo handles mutations in-graph")
    def test_leaf_no_grad_mutation_uses_copy(self):
        """
        Leaf input without requires_grad that is mutated via detach().
        The codegen should emit a plain copy_() (not detach().copy_()),
        matching the original _apply_input_mutations behavior.
        """
        from functorch.compile import nop
        from torch._functorch.aot_autograd import aot_function

        with capture_codegen_source("mutation_epilogue") as captured:

            def f(x):
                x.detach().mul_(2)
                return x + 1

            x = torch.randn(4)
            x_ref = x.clone()
            compiled_f = aot_function(f, nop, keep_inference_input_mutations=False)
            out = compiled_f(x)

        self.assertEqual(x, x_ref * 2)
        self.assertEqual(out, x_ref * 2 + 1)
        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn(".requires_grad", source)
        self.assertIn(".copy_(", source)


class TestAOTAutogradCallCompiledFn(TestCase):
    def test_direct_helper_handles_training_pre_call_work(self):
        call_compiled_fn = torch._C._aot_autograd_call_compiled_fn
        call_spec = torch._C._CompiledFnCallSpec((1,), True, False, False, (0,), (0,))

        x = torch.randn(2, requires_grad=True)
        y_base = torch.randn(2, requires_grad=True)
        y = y_base * 2
        args = [x, y]

        prev_view_replay_enabled = torch._C._is_view_replay_enabled()
        torch._C._set_view_replay_enabled(False)
        try:
            x_version = x._version
            seen = {}

            def compiled_fn(call_args):
                seen["same_args_list"] = call_args is args
                self.assertTrue(torch.is_grad_enabled())
                self.assertTrue(torch._C._is_view_replay_enabled())
                self.assertIs(call_args[0], x)
                self.assertIsNot(call_args[1], y)
                self.assertFalse(call_args[1].requires_grad)
                self.assertEqual(call_args[1], y)
                return [call_args[0] + 1]

            with torch.no_grad():
                orig_inputs, all_outs = call_compiled_fn(call_spec, compiled_fn, args)

            self.assertFalse(seen["same_args_list"])
            self.assertIs(orig_inputs[0], x)
            self.assertEqual(all_outs[0], x + 1)
            self.assertEqual(x._version, x_version + 1)
            self.assertFalse(torch._C._is_view_replay_enabled())
        finally:
            torch._C._set_view_replay_enabled(prev_view_replay_enabled)

    def test_direct_helper_handles_inference_pre_call_work(self):
        call_compiled_fn = torch._C._aot_autograd_call_compiled_fn
        call_spec = torch._C._CompiledFnCallSpec((), False, True, False, (), ())
        x = torch.randn(2)

        def compiled_fn(call_args):
            self.assertFalse(torch.is_grad_enabled())
            self.assertFalse(torch.is_autocast_enabled("cpu"))
            self.assertIs(call_args[0], x)
            return [call_args[0] + 1]

        with torch.enable_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            self.assertTrue(torch.is_grad_enabled())
            self.assertTrue(torch.is_autocast_enabled("cpu"))
            orig_inputs, all_outs = call_compiled_fn(call_spec, compiled_fn, [x])
            self.assertTrue(torch.is_grad_enabled())
            self.assertTrue(torch.is_autocast_enabled("cpu"))

        self.assertEqual(orig_inputs, {})
        self.assertEqual(all_outs[0], x + 1)


if __name__ == "__main__":
    run_tests()
