# Owner(s): ["module: functorch"]

"""
Tests for codegen'ing the output alias regeneration in
_create_runtime_wrapper.

The codegen'd output alias handler inlines each handler type's logic per
output as straight-line code: NoopAliasHandler becomes a direct fw_outs[i]
reference, IsInputHandler becomes orig_inputs[base_idx], and
AliasOfInput/IntermediateHandler become inline gen_alias_from_base calls
with baked-in indices and metadata.

Tests verify that an "output_alias_wrapper" artifact is emitted via
trace_structured.
"""

import logging
from contextlib import contextmanager
from unittest.mock import patch

import torch
import torch._functorch.config
from functorch.compile import nop
from torch._functorch.aot_autograd import aot_function
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


trace_log = logging.getLogger("torch.__trace")


class TestCodegenOutputAlias(TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()

    @contextmanager
    def _capture_codegen_source(self, artifact_name):
        """Capture codegen artifacts from the structured trace log."""
        captured: list[str] = []

        class _ArtifactHandler(logging.Handler):
            def emit(self, record):
                metadata = getattr(record, "metadata", {})
                if (
                    "artifact" in metadata
                    and metadata["artifact"].get("name") == artifact_name
                ):
                    payload = getattr(record, "payload", None)
                    if payload is not None:
                        captured.append(payload)

        handler = _ArtifactHandler()
        handler.setLevel(logging.DEBUG)
        old_level = trace_log.level
        trace_log.setLevel(logging.DEBUG)
        trace_log.addHandler(handler)
        try:
            yield captured
        finally:
            trace_log.removeHandler(handler)
            trace_log.setLevel(old_level)

    def test_output_is_view_of_input(self):
        """
        Output that is a view of an input (alias_of_input). Codegen should
        emit gen_alias_from_base(orig_inputs[i], ...) inline.
        """
        with self._capture_codegen_source("output_alias_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                return x.view(-1)

            x = torch.randn(2, 3)
            out = f(x)

        self.assertEqual(out, x.view(-1))
        self.assertEqual(out.data_ptr(), x.data_ptr())

        self.assertEqual(
            len(captured),
            1,
            "Expected output_alias_wrapper codegen artifact to be emitted",
        )

    def test_output_is_input(self):
        """
        Output that IS the input (is_input). Codegen should emit a direct
        reference to orig_inputs[i].
        """
        with self._capture_codegen_source("output_alias_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                x.mul_(2)
                return x

            x = torch.randn(4)
            x_ref = x.clone()
            out = f(x)

        self.assertEqual(x, x_ref * 2)
        self.assertIs(out, x)

        self.assertEqual(
            len(captured),
            1,
            "Expected output_alias_wrapper codegen artifact to be emitted",
        )

    def test_mixed_alias_and_non_alias_outputs(self):
        """
        Multiple outputs: one aliased, one not. Codegen should emit
        gen_alias_from_base for the alias and a noop for the non-alias.
        """
        with self._capture_codegen_source("output_alias_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                return x * 2, x.view(-1)

            x = torch.randn(2, 3)
            out1, out2 = f(x)

        self.assertEqual(out1, x * 2)
        self.assertEqual(out2, x.view(-1))
        self.assertEqual(out2.data_ptr(), x.data_ptr())

        self.assertEqual(
            len(captured),
            1,
            "Expected output_alias_wrapper codegen artifact to be emitted",
        )

    def test_output_alias_with_mutation(self):
        """
        Input is mutated AND output is a view of the input. Codegen should
        handle both mutation epilogue and alias regeneration.
        """
        with self._capture_codegen_source("output_alias_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                x.add_(1)
                return x.view(-1)

            x = torch.randn(2, 3)
            x_ref = x.clone()
            out = f(x)

        self.assertEqual(x, x_ref + 1)
        self.assertEqual(out, (x_ref + 1).view(-1))

        self.assertEqual(
            len(captured),
            1,
            "Expected output_alias_wrapper codegen artifact to be emitted",
        )

    def test_output_aliases_intermediate(self):
        """
        Output is a view of another output (intermediate), not of an input.
        Triggers AliasOfIntermediateHandler in the codegen.
        """
        with self._capture_codegen_source("output_alias_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                y = x + 1
                return y, y.view(-1)

            x = torch.randn(2, 3, requires_grad=True)
            out1, out2 = f(x)

        self.assertEqual(out1, x + 1)
        self.assertEqual(out2, (x + 1).view(-1))

        self.assertEqual(
            len(captured),
            1,
            "Expected output_alias_wrapper codegen artifact to be emitted",
        )

    def test_multiple_views_of_same_input(self):
        """
        Two outputs both alias the same input. Codegen should emit two
        separate gen_alias_from_base calls referencing the same base.
        """
        with self._capture_codegen_source("output_alias_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                return x.view(-1), x.reshape(6)

            x = torch.randn(2, 3)
            out1, out2 = f(x)

        self.assertEqual(out1, x.view(-1))
        self.assertEqual(out2, x.reshape(6))
        self.assertEqual(out1.data_ptr(), x.data_ptr())
        self.assertEqual(out2.data_ptr(), x.data_ptr())

        self.assertEqual(
            len(captured),
            1,
            "Expected output_alias_wrapper codegen artifact to be emitted",
        )

    def test_training_path_view_of_input(self):
        """
        Training path (trace_joint=True): output is a view of input with
        requires_grad=True. Codegen should use _unwrap_tensoralias in the
        alias function. Also verifies backward correctness.
        """
        with self._capture_codegen_source("output_alias_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                return x.view(-1)

            x = torch.randn(2, 3, requires_grad=True)
            out = f(x)

        self.assertEqual(out, x.view(-1))
        self.assertEqual(out.data_ptr(), x.data_ptr())

        out.sum().backward()
        self.assertEqual(x.grad, torch.ones(2, 3))

        self.assertEqual(
            len(captured),
            1,
            "Expected output_alias_wrapper codegen artifact to be emitted",
        )

    def test_training_path_mixed_requires_grad(self):
        """
        Training path with mixed differentiable and non-differentiable
        outputs. Exercises non-differentiable output collection in
        _transform_raw_returns codegen and backward correctness.
        """
        with (
            self._capture_codegen_source("compiled_fn_wrapper") as xform_captured,
            self._capture_codegen_source("output_alias_wrapper") as _alias_captured,
        ):

            @torch.compile(backend="aot_eager")
            def f(x, y):
                return x * 2, y.view(-1)

            x = torch.randn(2, 3, requires_grad=True)
            y = torch.randn(4)
            out1, out2 = f(x, y)

        self.assertEqual(out1, x * 2)
        self.assertEqual(out2, y.view(-1))
        self.assertTrue(out1.requires_grad)
        self.assertFalse(out2.requires_grad)

        out1.sum().backward()
        self.assertEqual(x.grad, torch.full((2, 3), 2.0))

        self.assertEqual(
            len(xform_captured),
            1,
            "Expected compiled_fn_wrapper codegen artifact to be emitted",
        )

    def test_training_path_mutation_and_alias(self):
        """
        Training path: input is mutated AND returned as a view. Exercises
        both the mutation epilogue and alias codegen with trace_joint=True.
        Uses a non-leaf tensor to allow in-place mutation with grad.
        """
        with self._capture_codegen_source("output_alias_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                x.add_(1)
                return x.view(-1)

            base = torch.randn(2, 3, requires_grad=True)
            x = base.clone()
            x_ref = x.detach().clone()
            out = f(x)

        self.assertEqual(x, x_ref + 1)
        self.assertEqual(out, (x_ref + 1).view(-1))

        self.assertEqual(
            len(captured),
            1,
            "Expected output_alias_wrapper codegen artifact to be emitted",
        )

    def test_training_path_is_input(self):
        """
        Training path: output IS the input (mutation + return identity).
        Exercises IsInputHandler with trace_joint=True. Uses a non-leaf
        tensor to allow in-place mutation with grad.
        """
        with self._capture_codegen_source("output_alias_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                x.mul_(2)
                return x

            base = torch.randn(4, requires_grad=True)
            x = base.clone()
            x_ref = x.detach().clone()
            out = f(x)

        self.assertEqual(x, x_ref * 2)
        self.assertIs(out, x)

        self.assertEqual(
            len(captured),
            1,
            "Expected output_alias_wrapper codegen artifact to be emitted",
        )

    def test_training_path_alias_of_intermediate_detach(self):
        """
        Training path: one output is a detached view of an intermediate,
        the other is a differentiable view. Exercises
        AliasOfIntermediateHandler with trace_joint=True and the
        base_is_user_output sub-path.
        """
        with self._capture_codegen_source("output_alias_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                y = x + 1
                return y.detach(), y.view(-1)

            x = torch.randn(3, 3, requires_grad=True)
            out_detach, out_view = f(x)

        self.assertEqual(out_detach, x + 1)
        self.assertEqual(out_view, (x + 1).view(-1))
        self.assertFalse(out_detach.requires_grad)
        self.assertTrue(out_view.requires_grad)
        self.assertEqual(out_detach.data_ptr(), out_view.data_ptr())

        out_view.sum().backward()
        self.assertEqual(x.grad, torch.ones(3, 3))

        self.assertEqual(
            len(captured),
            1,
            "Expected output_alias_wrapper codegen artifact to be emitted",
        )

    def test_view_replay_config_false(self):
        """
        Test that view_replay_for_aliased_outputs=False is correctly
        baked into the codegen'd alias function.
        """
        with patch(
            "torch._functorch.config.view_replay_for_aliased_outputs",
            False,
        ):
            with self._capture_codegen_source("output_alias_wrapper") as captured:

                @torch.compile(backend="aot_eager")
                def f(x):
                    return x.view(-1)

                x = torch.randn(2, 3)
                out = f(x)

            self.assertEqual(out, x.view(-1))
            self.assertEqual(out.data_ptr(), x.data_ptr())

            self.assertEqual(len(captured), 1)
            self.assertIn("replay_views=False", captured[0])

    def test_view_replay_config_true(self):
        """
        Test that view_replay_for_aliased_outputs=True (default) is
        correctly baked into the codegen'd alias function.
        """
        with patch(
            "torch._functorch.config.view_replay_for_aliased_outputs",
            True,
        ):
            with self._capture_codegen_source("output_alias_wrapper") as captured:

                @torch.compile(backend="aot_eager")
                def f(x):
                    return x.view(-1)

                x = torch.randn(2, 3)
                out = f(x)

            self.assertEqual(out, x.view(-1))
            self.assertEqual(out.data_ptr(), x.data_ptr())

            self.assertEqual(len(captured), 1)
            self.assertIn("replay_views=True", captured[0])

    def test_codegen_source_contains_gen_alias(self):
        """
        Verify the codegen'd source contains gen_alias_from_base for
        alias-of-input outputs and orig_inputs for is_input outputs.
        """
        with self._capture_codegen_source("output_alias_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                x.mul_(2)
                return x, x.view(-1)

            x = torch.randn(2, 3)
            f(x)

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("gen_alias_from_base", source)
        self.assertIn("orig_inputs[", source)

    def test_codegen_source_noop_handler(self):
        """
        Verify the codegen'd source contains fw_outs[i] for non-aliased
        (NoopAliasHandler) outputs in a mixed scenario.
        """
        with self._capture_codegen_source("output_alias_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                return x * 2, x.view(-1)

            x = torch.randn(2, 3)
            f(x)

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("fw_outs[", source)
        self.assertIn("gen_alias_from_base", source)

    def test_alias_of_intermediate_save_as_output(self):
        """
        Two outputs aliasing the same intermediate (not an input). When
        multiple outputs share the same intermediate base, the first triggers
        alias_of_intermediate_save_as_output and the second triggers
        alias_of_intermediate. Both use AliasOfIntermediateHandler in codegen.
        """
        with self._capture_codegen_source("output_alias_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                y = x + 1
                return y.view(-1), y.view(3, 3)

            x = torch.randn(3, 3, requires_grad=True)
            out1, out2 = f(x)

        expected = x + 1
        self.assertEqual(out1, expected.view(-1))
        self.assertEqual(out2, expected.view(3, 3))
        self.assertEqual(out1.data_ptr(), out2.data_ptr())

        out1.sum().backward()
        self.assertEqual(x.grad, torch.ones(3, 3))

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("gen_alias_from_base", source)

    def test_xform_unsafe_view_output(self):
        """
        _transform_raw_returns codegen: when an output is a view of an
        intermediate and is the only output aliasing that intermediate
        (unsafe_view_alias), the codegen emits an _unsafe_view call.
        """
        with self._capture_codegen_source("compiled_fn_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                return (x + 1).view(-1)

            x = torch.randn(2, 3, requires_grad=True)
            out = f(x)

        self.assertEqual(out, (x + 1).view(-1))
        self.assertTrue(out.requires_grad)

        out.sum().backward()
        self.assertEqual(x.grad, torch.ones(2, 3))

        self.assertEqual(len(captured), 1)
        self.assertIn("_unsafe_view", captured[0])

    @skipIfTorchDynamo("dynamo handles metadata mutations in-graph")
    def test_xform_metadata_only_mutation(self):
        """
        _transform_raw_returns codegen: when an input has a metadata-only
        mutation (mutates_metadata=True, mutates_data=False), the codegen
        wraps the corresponding mutated input return in TensorAlias.
        Uses aot_function directly because dynamo handles metadata
        mutations in-graph, so they never reach the _transform_raw_returns
        codegen path.
        """
        with self._capture_codegen_source("compiled_fn_wrapper") as captured:

            def f(a, b):
                a.transpose_(1, 0)
                return a + b

            a = torch.randn(3, 4, requires_grad=True).add(0)
            b = torch.randn(4, 3)
            compiled_f = aot_function(f, nop)
            out = compiled_f(a, b)

        self.assertEqual(a.shape, (4, 3))
        self.assertEqual(out.shape, (4, 3))

        self.assertEqual(len(captured), 1)
        self.assertIn("TensorAlias", captured[0])

    def test_cross_dtype_view_alias(self):
        """
        Output is a cross-dtype view of the input (view_as_real on a
        complex tensor). Exercises gen_alias_from_base's cross-dtype
        handling through the codegen path.
        """
        with self._capture_codegen_source("output_alias_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                return torch.view_as_real(x)

            x = torch.randn(4, dtype=torch.complex64)
            out = f(x)

        self.assertEqual(out, torch.view_as_real(x))
        self.assertEqual(out.shape, (4, 2))
        self.assertEqual(out.dtype, torch.float32)

        self.assertEqual(len(captured), 1)
        self.assertIn("gen_alias_from_base", captured[0])

    def test_xform_aliased_output_tensoralias_wrapping(self):
        """
        _transform_raw_returns codegen: aliased outputs get wrapped in
        TensorAlias so autograd.Function doesn't treat them as regular
        tensors. Verifies the TensorAlias wrapping path for aliased
        outputs (distinct from the metadata-only mutation wrapping).
        Needs a non-view computation (x * 2) to force the autograd
        factory path; a pure view like x.view(-1) alone bypasses it.
        """
        with self._capture_codegen_source("compiled_fn_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                return x * 2, x.view(-1)

            x = torch.randn(2, 3, requires_grad=True)
            out1, out2 = f(x)

        self.assertEqual(out1, x * 2)
        self.assertEqual(out2, x.view(-1))
        self.assertEqual(out2.data_ptr(), x.data_ptr())

        out1.sum().backward()
        self.assertEqual(x.grad, torch.full((2, 3), 2.0))

        self.assertEqual(len(captured), 1)
        self.assertIn("TensorAlias", captured[0])


if __name__ == "__main__":
    run_tests()
