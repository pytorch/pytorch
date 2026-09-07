# Owner(s): ["module: functorch"]

"""
Tests for codegen'ing the mutation epilogue in _create_runtime_wrapper.

The codegen'd mutation epilogue emits one of as_strided_(),
_replay_input_mutation(), or detach().copy_() per mutated input, with the
branch resolved at codegen time from each input's mutation metadata
(mutates_metadata, mutates_data, is_leaf).

Tests that exercise data-only mutations use torch.compile (dynamo handles
metadata mutations in-graph, so only data mutations reach the epilogue).

Tests that exercise metadata mutations (metadata-only, data+metadata)
use aot_function directly so metadata mutations flow through the epilogue.

Tests verify that a "mutation_epilogue" artifact is emitted via
trace_structured.
"""

import logging
import re
import warnings
from contextlib import contextmanager
from unittest import mock

import torch
import torch._functorch.config
from functorch.compile import nop
from torch._functorch._aot_autograd import runtime_wrappers as rw
from torch._functorch.aot_autograd import aot_function
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


trace_log = logging.getLogger("torch.__trace")


class TestCodegenMutationEpilogue(TestCase):
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

    @contextmanager
    def _capture_reference_epilogue(self):
        """Build the reference _RuntimeForwardEpilogue for each compiled wrapper."""
        epilogues: list[rw._RuntimeForwardEpilogue] = []
        orig_post_compile = rw.RuntimeWrapper.post_compile

        def spy(self, compiled_fn, aot_config, *, runtime_metadata):
            epilogues.append(
                rw._RuntimeForwardEpilogue(
                    runtime_metadata=runtime_metadata,
                    trace_joint=self.trace_joint,
                    keep_input_mutations=aot_config.keep_inference_input_mutations,
                )
            )
            return orig_post_compile(
                self, compiled_fn, aot_config, runtime_metadata=runtime_metadata
            )

        with mock.patch.object(rw.RuntimeWrapper, "post_compile", spy):
            yield epilogues

    @staticmethod
    def _custom_function_view(t):
        """A view stamped IN_CUSTOM_FUNCTION: an input a Function returned as-is."""

        class Identity(torch.autograd.Function):
            @staticmethod
            def forward(ctx, t):
                return t

            @staticmethod
            def backward(ctx, g):
                return g

        return Identity.apply(t)

    def test_single_data_mutation(self):
        """
        Single input data mutation via mul_. Codegen should emit a direct
        copy_() for this input.
        """
        with self._capture_codegen_source("mutation_epilogue") as captured:

            @torch.compile(backend="aot_eager")
            def f(x, y):
                x.mul_(2)
                return x + y

            x = torch.randn(4, requires_grad=True).clone()
            x.retain_grad()
            y = torch.randn(4)
            x_ref = x.detach().clone()
            y_ref = y.clone()
            out = f(x, y)

        self.assertEqual(x.detach(), x_ref * 2)
        self.assertEqual(out, x_ref * 2 + y_ref)

        self.assertEqual(
            len(captured),
            1,
            "Expected mutation_epilogue codegen artifact to be emitted",
        )
        # The write-back goes through _replay_input_mutation rather than a bare
        # copy_: a view created inside a custom autograd Function is written
        # under no_grad with its version counter preserved, which the helper
        # decides from the creation meta at runtime.
        self.assertIn("_replay_input_mutation", captured[0])

    def test_multiple_data_mutations(self):
        """
        Multiple inputs mutated. Codegen should emit one write-back per mutated
        input, with non-mutated inputs skipped entirely.
        """
        with self._capture_codegen_source("mutation_epilogue") as captured:

            @torch.compile(backend="aot_eager")
            def f(a, b, c):
                a.mul_(2)
                c.add_(1)
                return a + b + c

            a = torch.randn(4, requires_grad=True).clone()
            a.retain_grad()
            b = torch.randn(4)
            c = torch.randn(4, requires_grad=True).clone()
            c.retain_grad()
            a_ref, c_ref = a.detach().clone(), c.detach().clone()
            out = f(a, b, c)

        self.assertEqual(a.detach(), a_ref * 2)
        self.assertEqual(c.detach(), c_ref + 1)
        self.assertEqual(out, a_ref * 2 + b + c_ref + 1)

        self.assertEqual(
            len(captured),
            1,
            "Expected mutation_epilogue codegen artifact to be emitted",
        )
        self.assertIn("_replay_input_mutation", captured[0])

    def test_codegen_epilogue_matches_reference_on_custom_function_view(self):
        """
        The codegen'd epilogue and the reference _apply_input_mutations must
        hand _replay_input_mutation the same arguments: the mutated input's
        position among the compiled function's inputs (dynamo orders inputs by
        first use, so x is input 0 and the mutated y is input 1 in slot 0), the
        compile id the warning names, and the epilogue's own set of inputs
        already warned about, which makes the warning once per graph. On an
        IN_CUSTOM_FUNCTION view both write the values under no_grad with the
        version counter preserved.
        """
        with (
            self._capture_codegen_source("mutation_epilogue") as captured,
            self._capture_reference_epilogue() as epilogues,
        ):

            @torch.compile(backend="aot_eager")
            def f(x, y):
                out = x + 1
                y.mul_(2)
                return out + y

            # Tracing against the view itself fails in fake mode the way eager
            # does; the epilogue's per-call dispatch exists for a graph traced
            # against an ordinary tensor and handed the view later.
            x = torch.randn(4)
            f(x, torch.randn(4, requires_grad=True) * 1.0)
            data = torch.randn(4)
            y = self._custom_function_view(data.clone().requires_grad_() * 1.0)
            version = y._version
            pattern = "mutated input 1 of compiled graph"
            with self.assertWarnsRegex(UserWarning, pattern) as cm:
                f(x, y)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                f(x, y)

        self.assertEqual(y.detach(), data * 4)
        self.assertEqual(y._version, version)
        rewarned = [c for c in caught if "without autograd tracking" in str(c.message)]
        self.assertEqual(rewarned, [])

        (reference,) = epilogues
        cid = reference.runtime_metadata.compile_id_str
        self.assertIsNotNone(cid)
        self.assertIn(f"[{cid}]", str(cm.warning))
        self.assertEqual(len(captured), 1)
        calls = re.findall(r"_replay_input_mutation\((.*)\)", captured[0])
        expected = f"orig_inputs[1], updated_inputs[0], 1, {cid!r}, _warned_inputs, False, False"
        self.assertEqual(calls, [expected])

        y_ref = self._custom_function_view(data.clone().requires_grad_() * 1.0)
        pattern = rf"mutated input 1 of compiled graph \[{re.escape(cid)}\]"
        with self.assertWarnsRegex(UserWarning, pattern):
            reference._apply_input_mutations({0: x, 1: y_ref}, [y_ref.detach() * 2])
            reference._apply_input_mutations({0: x, 1: y_ref}, [y_ref.detach() * 2])
        self.assertEqual(y_ref.detach(), y.detach())
        self.assertEqual(y_ref._version, y._version)

    @skipIfTorchDynamo(
        "aot_function uses FX tracing which conflicts with dynamo wrapping"
    )
    def test_leaf_custom_function_view_without_grad_matches_reference(self):
        """
        A leaf input that does not require grad but is an IN_CUSTOM_FUNCTION
        view takes the else arm of the codegen'd leaf branch, which must be
        _replay_input_mutation like the reference. Eager accepts a tracked
        in-place op on such a view (and bumps its version), so the helper does
        too. Uses aot_function directly so the data mutation reaches the
        epilogue instead of staying in the graph.
        """
        eager = self._custom_function_view(torch.randn(4))
        eager.mul_(2)
        self.assertEqual(eager._version, 1)

        with (
            self._capture_codegen_source("mutation_epilogue") as captured,
            self._capture_reference_epilogue() as epilogues,
        ):

            def f(a):
                a.mul_(2)
                return a + 1

            data = torch.randn(4)
            a = self._custom_function_view(data.clone())
            self.assertTrue(a.is_leaf and not a.requires_grad)
            version = a._version
            out = aot_function(f, nop)(a)

        self.assertEqual(a, data * 2)
        self.assertEqual(out, data * 2 + 1)
        self.assertEqual(a._version, version + 1)
        self.assertEqual(len(captured), 1)
        calls = re.findall(r"else: _replay_input_mutation\((.*)\)", captured[0])
        expected = (
            "orig_inputs[0], updated_inputs[0], 0, None, _warned_inputs, False, False"
        )
        self.assertEqual(calls, [expected])

        (reference,) = epilogues
        a_ref = self._custom_function_view(data.clone())
        reference._apply_input_mutations({0: a_ref}, [a_ref * 2])
        self.assertEqual(a_ref, a)
        self.assertEqual(a_ref._version, a._version)

    def test_leaf_mutation_under_no_grad(self):
        """
        Leaf tensor mutated under no_grad (e.g. via detach().mul_()).
        Codegen should emit detach().copy_() for this case.
        """
        with self._capture_codegen_source("mutation_epilogue") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                x.detach().mul_(2)
                return x + 1

            x = torch.randn(4, requires_grad=True)
            x_ref = x.detach().clone()
            out = f(x)

        self.assertEqual(x.detach(), x_ref * 2)
        self.assertEqual(out, x_ref * 2 + 1)

        self.assertEqual(
            len(captured),
            1,
            "Expected mutation_epilogue codegen artifact to be emitted",
        )
        self.assertIn("detach().copy_", captured[0])

    @skipIfTorchDynamo(
        "aot_function uses FX tracing which conflicts with dynamo wrapping"
    )
    def test_metadata_only_mutation(self):
        """
        Metadata-only mutation via transpose_(). Codegen should emit
        as_strided_() without copy_(). Uses aot_function directly because
        dynamo handles metadata mutations in-graph.
        """
        with self._capture_codegen_source("mutation_epilogue") as captured:

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
        self.assertIn("as_strided_", captured[0])
        self.assertNotIn("copy_", captured[0])

    @skipIfTorchDynamo(
        "aot_function uses FX tracing which conflicts with dynamo wrapping"
    )
    def test_data_and_metadata_mutation(self):
        """
        Both data and metadata mutated (transpose_ then mul_). Codegen
        should emit as_strided_() followed by copy_(). Uses aot_function
        directly because dynamo handles metadata mutations in-graph.
        """
        with self._capture_codegen_source("mutation_epilogue") as captured:

            def f(a):
                a.transpose_(1, 0)
                a.mul_(2)
                return a + 1

            a = torch.randn(3, 4, requires_grad=True).add(0)
            a_ref = a.detach().clone()
            compiled_f = aot_function(f, nop)
            out = compiled_f(a)

        self.assertEqual(a.shape, (4, 3))
        self.assertEqual(a.detach(), a_ref.transpose(1, 0) * 2)
        self.assertEqual(out, a_ref.transpose(1, 0) * 2 + 1)

        self.assertEqual(len(captured), 1)
        self.assertIn("as_strided_", captured[0])
        self.assertIn("_replay_input_mutation", captured[0])

    def test_no_mutation_no_epilogue(self):
        """
        No mutations at all. No mutation_epilogue artifact should be
        emitted.
        """
        with self._capture_codegen_source("mutation_epilogue") as captured:

            @torch.compile(backend="aot_eager")
            def f(x, y):
                return x + y

            x = torch.randn(4, requires_grad=True)
            y = torch.randn(4)
            out = f(x, y)

        self.assertEqual(out, x + y)
        self.assertEqual(len(captured), 0)


if __name__ == "__main__":
    run_tests()
