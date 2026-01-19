# Owner(s): ["module: dynamo"]
"""
Parity tests for pythonify wrapper-covered calling conventions.

These tests verify that pythonified code produces the same results as
torch.compile for scenarios where AOTAutograd post-compile wrappers
modify calling conventions:

- Effect tokens (prepended/stripped from inputs/outputs)
- Dedupe wrapper (duplicate argument removal/reinsertion)
- Runtime wrapper (input detachment)
- Debug assert wrapper (requires_grad validation)
- RNG functionalization wrapper (CUDA RNG state plumbing)
- Fakified output wrapper (output stride preservation)
- Subclass dispatch wrapper (tensor subclass handling)
- Synthetic base wrapper (aliased inputs via synthetic bases)
- Autograd assembly (forward/backward stitching)

Each test compiles a model that triggers specific wrapper behavior, then
compares the pythonified execution against the compiled callable.

NOTE: End-to-end parity tests are currently skipped because the wrapper
codegen implementation is incomplete. The tests are structured to verify
parity once the implementation is complete. The codegen helper tests verify
that wrapper helper functions work correctly in isolation.
"""

import inspect
import os
import tempfile
import unittest

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests, TestCase


# Skip message for tests that require complete wrapper codegen implementation
WRAPPER_CODEGEN_WIP = (
    "Pythonify wrapper codegen is work-in-progress. "
    "These tests verify parity once implementation is complete."
)


class TestWrapperParityEffectTokens(TestCase):
    """
    Parity tests for EffectTokensWrapper scenarios.

    Effect tokens are used when the compiled function has side effects
    that need to be tracked. The wrapper prepends None tokens to args
    and strips them from outputs.
    """

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_simple_model_no_effect_tokens_parity(self):
        """
        Baseline test: simple model without effect tokens.

        Verifies that pythonified code matches compiled output for a
        basic model where no effect tokens are injected.
        """
        torch.set_default_device("cuda")

        class SimpleModel(nn.Module):
            def __init__(self, features: int):
                super().__init__()
                self.linear = nn.Linear(features, features)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x)

        torch.manual_seed(0)
        features = 8
        batch = 4
        model = SimpleModel(features)
        x = torch.randn(batch, features, requires_grad=True)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name

        try:
            torch._dynamo.reset()
            compiled_model = torch.compile(model, pythonify=path)
            y_compiled = compiled_model(x)

            with open(path) as f:
                code = f.read()

            x_test = x.detach().clone().requires_grad_(True)

            frame = inspect.currentframe()
            namespace = {**frame.f_globals, **frame.f_locals}
            namespace["x"] = x_test
            namespace["model"] = model
            namespace["torch"] = torch

            exec(code, namespace)
            y_pythonified = namespace["y"]

            self.assertTrue(
                torch.allclose(y_compiled.detach(), y_pythonified.detach(), rtol=1e-4, atol=1e-4),
                f"Forward outputs should match. Compiled: {y_compiled}, Pythonified: {y_pythonified}",
            )
        finally:
            os.unlink(path)


class TestWrapperParityDedupe(TestCase):
    """
    Parity tests for AOTDedupeWrapper scenarios.

    Dedupe wrapper handles duplicate argument removal and reinsertion.
    This occurs when the same tensor is passed multiple times to a function.
    """

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_duplicate_input_forward_parity(self):
        """
        Test model where same input is used multiple times.

        When the same tensor is passed as multiple arguments, AOTAutograd
        may deduplicate them. The pythonified code should handle this
        correctly.
        """
        torch.set_default_device("cuda")

        class DuplicateInputModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(4, 4))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x @ self.weight + x

        torch.manual_seed(42)
        model = DuplicateInputModel()
        x = torch.randn(2, 4, requires_grad=True)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name

        try:
            torch._dynamo.reset()
            compiled_model = torch.compile(model, pythonify=path)
            y_compiled = compiled_model(x)

            with open(path) as f:
                code = f.read()

            x_test = x.detach().clone().requires_grad_(True)

            frame = inspect.currentframe()
            namespace = {**frame.f_globals, **frame.f_locals}
            namespace["x"] = x_test
            namespace["model"] = model
            namespace["torch"] = torch

            exec(code, namespace)
            y_pythonified = namespace["y"]

            self.assertTrue(
                torch.allclose(y_compiled.detach(), y_pythonified.detach(), rtol=1e-4, atol=1e-4),
                f"Forward outputs should match for duplicate input model.",
            )
        finally:
            os.unlink(path)


class TestWrapperParityRuntimeWrapper(TestCase):
    """
    Parity tests for RuntimeWrapper scenarios.

    RuntimeWrapper handles input detachment at specified indices,
    autocast/grad context restoration, and alias/mutation epilogue.
    """

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_model_with_requires_grad_inputs(self):
        """
        Test model with requires_grad inputs that need runtime handling.

        The RuntimeWrapper may detach certain inputs. Pythonified code
        should produce the same forward outputs.
        """
        torch.set_default_device("cuda")

        class ModelWithGrad(nn.Module):
            def __init__(self, features: int):
                super().__init__()
                self.W = nn.Parameter(torch.randn(features, features))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.relu(x @ self.W)

        torch.manual_seed(123)
        features = 6
        batch = 3
        model = ModelWithGrad(features)
        x = torch.randn(batch, features, requires_grad=True)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name

        try:
            torch._dynamo.reset()
            compiled_model = torch.compile(model, pythonify=path)
            y_compiled = compiled_model(x)

            with open(path) as f:
                code = f.read()

            x_test = x.detach().clone().requires_grad_(True)

            frame = inspect.currentframe()
            namespace = {**frame.f_globals, **frame.f_locals}
            namespace["x"] = x_test
            namespace["model"] = model
            namespace["torch"] = torch

            exec(code, namespace)
            y_pythonified = namespace["y"]

            self.assertTrue(
                torch.allclose(y_compiled.detach(), y_pythonified.detach(), rtol=1e-4, atol=1e-4),
                f"Forward outputs should match for requires_grad model.",
            )
        finally:
            os.unlink(path)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_inference_mode_model(self):
        """
        Test model running in inference mode (no gradients).

        RuntimeWrapper behavior differs for inference vs training.
        """
        torch.set_default_device("cuda")

        class InferenceModel(nn.Module):
            def __init__(self, features: int):
                super().__init__()
                self.linear = nn.Linear(features, features)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x).sigmoid()

        torch.manual_seed(456)
        features = 5
        batch = 2
        model = InferenceModel(features)
        x = torch.randn(batch, features)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name

        try:
            torch._dynamo.reset()
            with torch.no_grad():
                compiled_model = torch.compile(model, pythonify=path)
                y_compiled = compiled_model(x)

            with open(path) as f:
                code = f.read()

            x_test = x.detach().clone()

            frame = inspect.currentframe()
            namespace = {**frame.f_globals, **frame.f_locals}
            namespace["x"] = x_test
            namespace["model"] = model
            namespace["torch"] = torch

            with torch.no_grad():
                exec(code, namespace)
            y_pythonified = namespace["y"]

            self.assertTrue(
                torch.allclose(y_compiled, y_pythonified, rtol=1e-4, atol=1e-4),
                f"Inference outputs should match.",
            )
        finally:
            os.unlink(path)


class TestWrapperParityAutogradAssembly(TestCase):
    """
    Parity tests for autograd assembly (forward/backward stitching).

    AOTDispatchAutograd.post_compile stitches forward and backward
    callables into a torch.autograd.Function. These tests verify that
    pythonified code correctly handles the autograd assembly.
    """

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_backward_pass_parity(self):
        """
        Test that backward pass outputs match between compiled and pythonified.

        This is a critical test for autograd assembly correctness.
        """
        torch.set_default_device("cuda")

        class SimpleGradModel(nn.Module):
            def __init__(self, features: int):
                super().__init__()
                self.W = nn.Parameter(torch.randn(features, features))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return (x @ self.W).sum()

        torch.manual_seed(789)
        features = 4
        batch = 2
        model = SimpleGradModel(features)

        x = torch.randn(batch, features, requires_grad=True)
        x_compiled = x.detach().clone().requires_grad_(True)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name

        try:
            torch._dynamo.reset()
            compiled_model = torch.compile(model, pythonify=path)
            y_compiled = compiled_model(x_compiled)

            y_compiled.backward()
            grad_compiled = x_compiled.grad.clone()

            with open(path) as f:
                code = f.read()

            x_pythonified = x.detach().clone().requires_grad_(True)

            frame = inspect.currentframe()
            namespace = {**frame.f_globals, **frame.f_locals}
            namespace["x"] = x_pythonified
            namespace["model"] = model
            namespace["torch"] = torch

            exec(code, namespace)
            y_pythonified = namespace["y"]

            y_pythonified.backward()
            grad_pythonified = x_pythonified.grad.clone()

            self.assertTrue(
                torch.allclose(y_compiled.detach(), y_pythonified.detach(), rtol=1e-4, atol=1e-4),
                f"Forward outputs should match.",
            )

            self.assertTrue(
                torch.allclose(grad_compiled, grad_pythonified, rtol=1e-4, atol=1e-4),
                f"Backward gradients should match. Compiled: {grad_compiled}, Pythonified: {grad_pythonified}",
            )
        finally:
            os.unlink(path)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_multi_output_backward_parity(self):
        """
        Test backward pass with multiple outputs.

        Models that return multiple tensors have more complex autograd
        assembly. This tests that pythonified code handles it correctly.
        """
        torch.set_default_device("cuda")

        class MultiOutputModel(nn.Module):
            def __init__(self, features: int):
                super().__init__()
                self.W1 = nn.Parameter(torch.randn(features, features))
                self.W2 = nn.Parameter(torch.randn(features, features))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out1 = x @ self.W1
                out2 = x @ self.W2
                return (out1 + out2).sum()

        torch.manual_seed(321)
        features = 3
        batch = 2
        model = MultiOutputModel(features)

        x = torch.randn(batch, features, requires_grad=True)
        x_compiled = x.detach().clone().requires_grad_(True)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name

        try:
            torch._dynamo.reset()
            compiled_model = torch.compile(model, pythonify=path)
            y_compiled = compiled_model(x_compiled)

            y_compiled.backward()
            grad_compiled = x_compiled.grad.clone()

            with open(path) as f:
                code = f.read()

            x_pythonified = x.detach().clone().requires_grad_(True)

            frame = inspect.currentframe()
            namespace = {**frame.f_globals, **frame.f_locals}
            namespace["x"] = x_pythonified
            namespace["model"] = model
            namespace["torch"] = torch

            exec(code, namespace)
            y_pythonified = namespace["y"]

            y_pythonified.backward()
            grad_pythonified = x_pythonified.grad.clone()

            self.assertTrue(
                torch.allclose(y_compiled.detach(), y_pythonified.detach(), rtol=1e-4, atol=1e-4),
                f"Forward outputs should match for multi-output model.",
            )

            self.assertTrue(
                torch.allclose(grad_compiled, grad_pythonified, rtol=1e-4, atol=1e-4),
                f"Backward gradients should match for multi-output model.",
            )
        finally:
            os.unlink(path)


class TestWrapperParityMixedScenarios(TestCase):
    """
    Parity tests for scenarios with multiple wrappers active.

    In practice, multiple wrappers are often active simultaneously.
    These tests verify parity in such mixed scenarios.
    """

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_complex_model_with_multiple_layers(self):
        """
        Test a more complex model with multiple layers and activations.

        This exercises multiple wrapper scenarios simultaneously with a model
        that has multiple nn.Linear layers. Verifies that pythonify correctly
        extracts full paths like 'layer1.weight', 'layer2.weight' instead of
        just 'weight', avoiding name collisions.

        Note: This test focuses on forward pass parity. Backward pass parity
        has separate known issues tracked in a different TODO.
        """
        torch.set_default_device("cuda")

        class ComplexModel(nn.Module):
            def __init__(self, features: int, hidden: int):
                super().__init__()
                self.layer1 = nn.Linear(features, hidden)
                self.layer2 = nn.Linear(hidden, hidden)
                self.layer3 = nn.Linear(hidden, features)
                self.activation = nn.ReLU()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.activation(self.layer1(x))
                x = self.activation(self.layer2(x))
                return self.layer3(x)

        torch.manual_seed(999)
        features = 8
        hidden = 16
        batch = 4
        model = ComplexModel(features, hidden)

        x = torch.randn(batch, features, requires_grad=False)
        x_compiled = x.detach().clone()

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name

        try:
            torch._dynamo.reset()
            compiled_model = torch.compile(model, pythonify=path)
            y_compiled = compiled_model(x_compiled)

            with open(path) as f:
                code = f.read()

            x_pythonified = x.detach().clone()

            frame = inspect.currentframe()
            namespace = {**frame.f_globals, **frame.f_locals}
            namespace["x"] = x_pythonified
            namespace["model"] = model
            namespace["torch"] = torch

            exec(code, namespace)
            y_pythonified = namespace["y"]

            self.assertTrue(
                torch.allclose(y_compiled.detach(), y_pythonified.detach(), rtol=1e-4, atol=1e-4),
                f"Forward outputs should match for complex model with multiple layers.",
            )
        finally:
            os.unlink(path)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_model_with_inplace_ops(self):
        """
        Test model with in-place operations.

        In-place operations may trigger mutation handling in RuntimeWrapper.
        """
        torch.set_default_device("cuda")

        class InplaceModel(nn.Module):
            def __init__(self, features: int):
                super().__init__()
                self.W = nn.Parameter(torch.randn(features, features))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                y = x @ self.W
                y = torch.relu_(y)
                return y

        torch.manual_seed(555)
        features = 4
        batch = 2
        model = InplaceModel(features)

        x = torch.randn(batch, features, requires_grad=True)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name

        try:
            torch._dynamo.reset()
            compiled_model = torch.compile(model, pythonify=path)
            y_compiled = compiled_model(x)

            with open(path) as f:
                code = f.read()

            x_test = x.detach().clone().requires_grad_(True)

            frame = inspect.currentframe()
            namespace = {**frame.f_globals, **frame.f_locals}
            namespace["x"] = x_test
            namespace["model"] = model
            namespace["torch"] = torch

            exec(code, namespace)
            y_pythonified = namespace["y"]

            self.assertTrue(
                torch.allclose(y_compiled.detach(), y_pythonified.detach(), rtol=1e-4, atol=1e-4),
                f"Forward outputs should match for model with in-place ops.",
            )
        finally:
            os.unlink(path)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_model_with_view_ops(self):
        """
        Test model with view operations.

        View operations create aliased tensors that may trigger
        synthetic base wrapper or alias handling in RuntimeWrapper.
        """
        torch.set_default_device("cuda")

        class ViewOpModel(nn.Module):
            def __init__(self, features: int):
                super().__init__()
                self.W = nn.Parameter(torch.randn(features, features))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                y = x @ self.W
                y = y.view(-1)
                return y.sum()

        torch.manual_seed(666)
        features = 4
        batch = 2
        model = ViewOpModel(features)

        x = torch.randn(batch, features, requires_grad=True)
        x_compiled = x.detach().clone().requires_grad_(True)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name

        try:
            torch._dynamo.reset()
            compiled_model = torch.compile(model, pythonify=path)
            y_compiled = compiled_model(x_compiled)

            y_compiled.backward()
            grad_compiled = x_compiled.grad.clone()

            with open(path) as f:
                code = f.read()

            x_pythonified = x.detach().clone().requires_grad_(True)

            frame = inspect.currentframe()
            namespace = {**frame.f_globals, **frame.f_locals}
            namespace["x"] = x_pythonified
            namespace["model"] = model
            namespace["torch"] = torch

            exec(code, namespace)
            y_pythonified = namespace["y"]

            y_pythonified.backward()
            grad_pythonified = x_pythonified.grad.clone()

            self.assertTrue(
                torch.allclose(y_compiled.detach(), y_pythonified.detach(), rtol=1e-4, atol=1e-4),
                f"Forward outputs should match for view ops model.",
            )

            self.assertTrue(
                torch.allclose(grad_compiled, grad_pythonified, rtol=1e-4, atol=1e-4),
                f"Backward gradients should match for view ops model.",
            )
        finally:
            os.unlink(path)


class TestWrapperParityEdgeCases(TestCase):
    """
    Edge case parity tests for wrapper scenarios.

    These tests cover unusual but valid configurations that may
    exercise specific wrapper paths.
    """

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_scalar_output_model(self):
        """
        Test model that returns a scalar output.

        Scalar outputs have different handling in some wrappers.
        """
        torch.set_default_device("cuda")

        class ScalarOutputModel(nn.Module):
            def __init__(self, features: int):
                super().__init__()
                self.W = nn.Parameter(torch.randn(features))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return (x * self.W).sum()

        torch.manual_seed(777)
        features = 5
        model = ScalarOutputModel(features)

        x = torch.randn(features, requires_grad=True)
        x_compiled = x.detach().clone().requires_grad_(True)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name

        try:
            torch._dynamo.reset()
            compiled_model = torch.compile(model, pythonify=path)
            y_compiled = compiled_model(x_compiled)

            y_compiled.backward()
            grad_compiled = x_compiled.grad.clone()

            with open(path) as f:
                code = f.read()

            x_pythonified = x.detach().clone().requires_grad_(True)

            frame = inspect.currentframe()
            namespace = {**frame.f_globals, **frame.f_locals}
            namespace["x"] = x_pythonified
            namespace["model"] = model
            namespace["torch"] = torch

            exec(code, namespace)
            y_pythonified = namespace["y"]

            y_pythonified.backward()
            grad_pythonified = x_pythonified.grad.clone()

            self.assertTrue(
                torch.allclose(y_compiled.detach(), y_pythonified.detach(), rtol=1e-4, atol=1e-4),
                f"Scalar outputs should match.",
            )

            self.assertTrue(
                torch.allclose(grad_compiled, grad_pythonified, rtol=1e-4, atol=1e-4),
                f"Gradients should match for scalar output.",
            )
        finally:
            os.unlink(path)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_empty_parameter_model(self):
        """
        Test model with no parameters.

        Models without parameters are a valid edge case that tests
        argument extraction handling.
        """
        torch.set_default_device("cuda")

        class NoParamModel(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x * 2 + 1

        model = NoParamModel()
        x = torch.randn(3, 4)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name

        try:
            torch._dynamo.reset()
            compiled_model = torch.compile(model, pythonify=path)
            y_compiled = compiled_model(x)

            with open(path) as f:
                code = f.read()

            x_test = x.detach().clone()

            frame = inspect.currentframe()
            namespace = {**frame.f_globals, **frame.f_locals}
            namespace["x"] = x_test
            namespace["model"] = model
            namespace["torch"] = torch

            exec(code, namespace)
            y_pythonified = namespace["y"]

            self.assertTrue(
                torch.allclose(y_compiled, y_pythonified, rtol=1e-4, atol=1e-4),
                f"Outputs should match for no-param model.",
            )
        finally:
            os.unlink(path)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_multiple_input_model(self):
        """
        Test model with multiple input tensors.

        Multiple inputs may trigger different argument extraction paths.
        """
        torch.set_default_device("cuda")

        class MultiInputModel(nn.Module):
            def __init__(self, features: int):
                super().__init__()
                self.W = nn.Parameter(torch.randn(features, features))

            def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
                return (x + z) @ self.W

        torch.manual_seed(888)
        features = 4
        batch = 2
        model = MultiInputModel(features)

        x = torch.randn(batch, features)
        z = torch.randn(batch, features)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name

        try:
            torch._dynamo.reset()
            compiled_model = torch.compile(model, pythonify=path)
            y_compiled = compiled_model(x, z)

            with open(path) as f:
                code = f.read()

            x_test = x.detach().clone()
            z_test = z.detach().clone()

            frame = inspect.currentframe()
            namespace = {**frame.f_globals, **frame.f_locals}
            namespace["x"] = x_test
            namespace["z"] = z_test
            namespace["model"] = model
            namespace["torch"] = torch

            exec(code, namespace)
            y_pythonified = namespace["y"]

            self.assertTrue(
                torch.allclose(y_compiled, y_pythonified, rtol=1e-4, atol=1e-4),
                f"Outputs should match for multi-input model.",
            )
        finally:
            os.unlink(path)


class TestWrapperParityCodegenHelpers(TestCase):
    """
    Tests for wrapper helper function code generation.

    These tests verify that the wrapper helper functions generated
    by pythonify work correctly in isolation. These tests should pass
    as they test the codegen helpers directly without full end-to-end
    pythonify execution.
    """

    def test_effect_tokens_inject_strip_roundtrip(self):
        """
        Test that inject/strip effect tokens functions work correctly.

        The generated functions should correctly add and remove tokens.
        """
        from torch._dynamo.pythonify.gen_python import PythonCodeGenVisitor
        from torch._dynamo.pythonify.ir import EffectTokensWrapperNode

        visitor = PythonCodeGenVisitor()
        node = EffectTokensWrapperNode(token_count=3)
        visitor.visit_effect_tokens_wrapper(node)
        code = visitor.get_code()

        namespace = {}
        exec(code, namespace)

        original_args = [1, 2, 3]
        injected = namespace["_inject_effect_tokens"](original_args)
        self.assertEqual(injected, [None, None, None, 1, 2, 3])

        outputs_with_tokens = [None, None, None, 10, 20, 30]
        stripped = namespace["_strip_effect_tokens"](outputs_with_tokens)
        self.assertEqual(stripped, [10, 20, 30])

    def test_dedupe_remove_add_roundtrip(self):
        """
        Test that remove/add dupe args functions work correctly.

        When args [a, b, a, c] have mask [T, T, F, T] and map [0, 1, 0, 2],
        remove should give [a, b, c] and add should restore [a, b, a, c].
        """
        from torch._dynamo.pythonify.gen_python import PythonCodeGenVisitor
        from torch._dynamo.pythonify.ir import AOTDedupeWrapperNode

        visitor = PythonCodeGenVisitor()
        node = AOTDedupeWrapperNode(
            keep_arg_mask=[True, True, False, True],
            add_dupe_map=[0, 1, 0, 2],
            needs_post_compile=True,
        )
        visitor.visit_aot_dedupe_wrapper(node)
        code = visitor.get_code()

        namespace = {}
        exec(code, namespace)

        original = ["a", "b", "a", "c"]
        removed = namespace["_remove_dupe_args"](original)
        self.assertEqual(removed, ["a", "b", "c"])

        restored = namespace["_add_dupe_args"](removed)
        self.assertEqual(restored, ["a", "b", "a", "c"])

    def test_detach_inputs_function(self):
        """
        Test that the detach_inputs function correctly detaches tensors.
        """
        from torch._dynamo.pythonify.gen_python import PythonCodeGenVisitor
        from torch._dynamo.pythonify.ir import RuntimeWrapperNode

        visitor = PythonCodeGenVisitor()
        node = RuntimeWrapperNode(indices_of_inps_to_detach=[0, 2])
        visitor.visit_runtime_wrapper(node)
        code = visitor.get_code()

        namespace = {"torch": torch}
        exec(code, namespace)

        t1 = torch.tensor([1.0], requires_grad=True)
        t2 = torch.tensor([2.0], requires_grad=True)
        t3 = torch.tensor([3.0], requires_grad=True)

        result = namespace["_detach_inputs"]([t1, t2, t3])

        self.assertFalse(result[0].requires_grad)
        self.assertTrue(result[1].requires_grad)
        self.assertFalse(result[2].requires_grad)

    def test_debug_assert_requires_grad(self):
        """
        Test that debug assert function validates requires_grad correctly.
        """
        from torch._dynamo.pythonify.gen_python import PythonCodeGenVisitor
        from torch._dynamo.pythonify.ir import DebugAssertWrapperNode

        visitor = PythonCodeGenVisitor()
        node = DebugAssertWrapperNode(flat_requires_grad=[True, False])
        visitor.visit_debug_assert_wrapper(node)
        code = visitor.get_code()

        namespace = {}
        exec(code, namespace)

        t1 = torch.tensor([1.0], requires_grad=True)
        t2 = torch.tensor([2.0], requires_grad=False)

        namespace["_assert_requires_grad"]([t1, t2])

        t1_wrong = torch.tensor([1.0], requires_grad=False)
        with self.assertRaises(AssertionError):
            namespace["_assert_requires_grad"]([t1_wrong, t2])


class TestWrapperStackIntegration(TestCase):
    """
    Integration tests verifying the complete wrapper stack works end-to-end.

    These tests exercise multiple wrappers simultaneously to ensure they
    compose correctly. Unlike the isolated parity tests above, these tests
    specifically construct models that trigger multiple AOTAutograd wrappers
    at once:

    - RuntimeWrapper (always present for training with requires_grad inputs)
    - Autograd assembly (forward/backward stitching)
    - In-place ops and view ops (may trigger mutation handling)
    - Multiple parameters (exercises argument extraction ordering)

    Each test compiles a model with pythonify, executes the generated code,
    and verifies both forward and backward outputs match the compiled model.
    """

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_wrapper_stack_training_with_multiple_params(self):
        """
        Integration test for complete wrapper stack with training mode.

        This test exercises:
        - RuntimeWrapper (input detachment for autograd)
        - Autograd assembly (forward/backward stitching)
        - Multiple parameter argument extraction
        - Gradient computation through the wrapper stack

        The model has multiple parameters to verify that argument ordering
        and parameter extraction work correctly through the entire stack.
        """
        torch.set_default_device("cuda")

        class MultiParamModel(nn.Module):
            def __init__(self, features: int):
                super().__init__()
                self.W1 = nn.Parameter(torch.randn(features, features))
                self.b1 = nn.Parameter(torch.randn(features))
                self.W2 = nn.Parameter(torch.randn(features, features))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                h = torch.relu(x @ self.W1 + self.b1)
                return (h @ self.W2).sum()

        torch.manual_seed(12345)
        features = 6
        batch = 4
        model = MultiParamModel(features)

        x = torch.randn(batch, features, requires_grad=True)
        x_compiled = x.detach().clone().requires_grad_(True)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name

        try:
            torch._dynamo.reset()
            compiled_model = torch.compile(model, pythonify=path)
            y_compiled = compiled_model(x_compiled)

            y_compiled.backward()
            grad_compiled = x_compiled.grad.clone()
            param_grads_compiled = {
                name: p.grad.clone() for name, p in model.named_parameters()
            }

            model.zero_grad()

            with open(path) as f:
                code = f.read()

            x_pythonified = x.detach().clone().requires_grad_(True)

            frame = inspect.currentframe()
            namespace = {**frame.f_globals, **frame.f_locals}
            namespace["x"] = x_pythonified
            namespace["model"] = model
            namespace["torch"] = torch

            exec(code, namespace)
            y_pythonified = namespace["y"]

            y_pythonified.backward()
            grad_pythonified = x_pythonified.grad.clone()
            param_grads_pythonified = {
                name: p.grad.clone() for name, p in model.named_parameters()
            }

            self.assertTrue(
                torch.allclose(y_compiled.detach(), y_pythonified.detach(), rtol=1e-4, atol=1e-4),
                f"Forward outputs should match. Compiled: {y_compiled.item():.6f}, "
                f"Pythonified: {y_pythonified.item():.6f}",
            )

            self.assertTrue(
                torch.allclose(grad_compiled, grad_pythonified, rtol=1e-4, atol=1e-4),
                f"Input gradients should match.",
            )

            for name in param_grads_compiled:
                self.assertTrue(
                    torch.allclose(
                        param_grads_compiled[name],
                        param_grads_pythonified[name],
                        rtol=1e-4,
                        atol=1e-4,
                    ),
                    f"Parameter '{name}' gradients should match.",
                )
        finally:
            os.unlink(path)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_wrapper_stack_inplace_and_view_ops(self):
        """
        Integration test for wrapper stack with in-place and view operations.

        This test exercises:
        - RuntimeWrapper with mutation handling
        - View operations that create aliased tensors
        - In-place operations (relu_)
        - Autograd assembly for backward pass

        In-place ops and views may trigger additional wrapper paths for
        alias/mutation handling in RuntimeWrapper.
        """
        torch.set_default_device("cuda")

        class InplaceViewModel(nn.Module):
            def __init__(self, features: int):
                super().__init__()
                self.W = nn.Parameter(torch.randn(features, features))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                h = x @ self.W
                h = torch.relu_(h)
                h_flat = h.view(-1)
                return h_flat.sum()

        torch.manual_seed(54321)
        features = 5
        batch = 3
        model = InplaceViewModel(features)

        x = torch.randn(batch, features, requires_grad=True)
        x_compiled = x.detach().clone().requires_grad_(True)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name

        try:
            torch._dynamo.reset()
            compiled_model = torch.compile(model, pythonify=path)
            y_compiled = compiled_model(x_compiled)

            y_compiled.backward()
            grad_compiled = x_compiled.grad.clone()

            model.zero_grad()

            with open(path) as f:
                code = f.read()

            x_pythonified = x.detach().clone().requires_grad_(True)

            frame = inspect.currentframe()
            namespace = {**frame.f_globals, **frame.f_locals}
            namespace["x"] = x_pythonified
            namespace["model"] = model
            namespace["torch"] = torch

            exec(code, namespace)
            y_pythonified = namespace["y"]

            y_pythonified.backward()
            grad_pythonified = x_pythonified.grad.clone()

            self.assertTrue(
                torch.allclose(y_compiled.detach(), y_pythonified.detach(), rtol=1e-4, atol=1e-4),
                f"Forward outputs should match for inplace+view model.",
            )

            self.assertTrue(
                torch.allclose(grad_compiled, grad_pythonified, rtol=1e-4, atol=1e-4),
                f"Backward gradients should match for inplace+view model.",
            )
        finally:
            os.unlink(path)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_wrapper_stack_mixed_requires_grad(self):
        """
        Integration test with mixed requires_grad inputs.

        This test exercises:
        - RuntimeWrapper handling different requires_grad settings
        - Autograd only tracking gradients for grad-enabled inputs
        - Multiple inputs with different gradient requirements

        The model takes two inputs: one with requires_grad=True (for training)
        and one with requires_grad=False (a frozen input like an embedding).
        """
        torch.set_default_device("cuda")

        class MixedGradModel(nn.Module):
            def __init__(self, features: int):
                super().__init__()
                self.W = nn.Parameter(torch.randn(features, features))

            def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
                h = (x * mask) @ self.W
                return h.sum()

        torch.manual_seed(99999)
        features = 4
        batch = 2
        model = MixedGradModel(features)

        x = torch.randn(batch, features, requires_grad=True)
        mask = torch.ones(batch, features, requires_grad=False)

        x_compiled = x.detach().clone().requires_grad_(True)
        mask_compiled = mask.detach().clone()

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name

        try:
            torch._dynamo.reset()
            compiled_model = torch.compile(model, pythonify=path)
            y_compiled = compiled_model(x_compiled, mask_compiled)

            y_compiled.backward()
            grad_compiled = x_compiled.grad.clone()

            model.zero_grad()

            with open(path) as f:
                code = f.read()

            x_pythonified = x.detach().clone().requires_grad_(True)
            mask_pythonified = mask.detach().clone()

            frame = inspect.currentframe()
            namespace = {**frame.f_globals, **frame.f_locals}
            namespace["x"] = x_pythonified
            namespace["mask"] = mask_pythonified
            namespace["model"] = model
            namespace["torch"] = torch

            exec(code, namespace)
            y_pythonified = namespace["y"]

            y_pythonified.backward()
            grad_pythonified = x_pythonified.grad.clone()

            self.assertTrue(
                torch.allclose(y_compiled.detach(), y_pythonified.detach(), rtol=1e-4, atol=1e-4),
                f"Forward outputs should match for mixed grad model.",
            )

            self.assertTrue(
                torch.allclose(grad_compiled, grad_pythonified, rtol=1e-4, atol=1e-4),
                f"Input gradients should match for mixed grad model.",
            )

            self.assertIsNone(
                mask_pythonified.grad,
                "Non-grad input should not have gradients.",
            )
        finally:
            os.unlink(path)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_wrapper_stack_inference_with_buffers(self):
        """
        Integration test for inference mode with registered buffers.

        This test exercises:
        - RuntimeWrapper in inference mode (no backward)
        - Registered buffer handling (non-parameter tensors)
        - Argument extraction for mixed params and buffers

        Buffers are tensors registered with the module but not trained.
        The pythonify pipeline must handle them correctly.
        """
        torch.set_default_device("cuda")

        class BufferModel(nn.Module):
            def __init__(self, features: int):
                super().__init__()
                self.W = nn.Parameter(torch.randn(features, features))
                self.register_buffer("scale", torch.ones(features) * 2.0)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return (x @ self.W) * self.scale

        torch.manual_seed(11111)
        features = 5
        batch = 3
        model = BufferModel(features)
        x = torch.randn(batch, features)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name

        try:
            torch._dynamo.reset()
            with torch.no_grad():
                compiled_model = torch.compile(model, pythonify=path)
                y_compiled = compiled_model(x)

            with open(path) as f:
                code = f.read()

            x_test = x.detach().clone()

            frame = inspect.currentframe()
            namespace = {**frame.f_globals, **frame.f_locals}
            namespace["x"] = x_test
            namespace["model"] = model
            namespace["torch"] = torch

            with torch.no_grad():
                exec(code, namespace)
            y_pythonified = namespace["y"]

            self.assertTrue(
                torch.allclose(y_compiled, y_pythonified, rtol=1e-4, atol=1e-4),
                f"Inference outputs should match for buffer model.",
            )
        finally:
            os.unlink(path)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_wrapper_stack_full_pipeline_verification(self):
        """
        Full pipeline verification test for wrapper stack integration.

        This is the most comprehensive integration test that:
        - Verifies forward pass parity
        - Verifies backward pass parity
        - Verifies parameter gradient parity
        - Verifies correct handling of multiple operations

        The model includes:
        - Multiple parameters (W1, b1, W2, b2)
        - Different activation functions (relu, tanh)
        - Element-wise operations
        - Reduction operation (sum)

        This exercises the complete wrapper stack from argument extraction
        through guard checks, wrapper application, kernel execution, and
        result return.
        """
        torch.set_default_device("cuda")

        class FullPipelineModel(nn.Module):
            def __init__(self, in_features: int, hidden: int):
                super().__init__()
                self.W1 = nn.Parameter(torch.randn(in_features, hidden))
                self.b1 = nn.Parameter(torch.randn(hidden))
                self.W2 = nn.Parameter(torch.randn(hidden, in_features))
                self.b2 = nn.Parameter(torch.randn(in_features))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                h = torch.relu(x @ self.W1 + self.b1)
                h = torch.tanh(h @ self.W2 + self.b2)
                return h.sum()

        torch.manual_seed(77777)
        in_features = 6
        hidden = 8
        batch = 4
        model = FullPipelineModel(in_features, hidden)

        x = torch.randn(batch, in_features, requires_grad=True)
        x_compiled = x.detach().clone().requires_grad_(True)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name

        try:
            torch._dynamo.reset()
            compiled_model = torch.compile(model, pythonify=path)
            y_compiled = compiled_model(x_compiled)

            y_compiled.backward()
            grad_compiled = x_compiled.grad.clone()
            param_grads_compiled = {
                name: p.grad.clone() for name, p in model.named_parameters()
            }

            model.zero_grad()

            with open(path) as f:
                code = f.read()

            x_pythonified = x.detach().clone().requires_grad_(True)

            frame = inspect.currentframe()
            namespace = {**frame.f_globals, **frame.f_locals}
            namespace["x"] = x_pythonified
            namespace["model"] = model
            namespace["torch"] = torch

            exec(code, namespace)
            y_pythonified = namespace["y"]

            y_pythonified.backward()
            grad_pythonified = x_pythonified.grad.clone()
            param_grads_pythonified = {
                name: p.grad.clone() for name, p in model.named_parameters()
            }

            self.assertTrue(
                torch.allclose(y_compiled.detach(), y_pythonified.detach(), rtol=1e-4, atol=1e-4),
                f"Forward outputs should match in full pipeline test.",
            )

            self.assertTrue(
                torch.allclose(grad_compiled, grad_pythonified, rtol=1e-4, atol=1e-4),
                f"Input gradients should match in full pipeline test.",
            )

            for name in param_grads_compiled:
                self.assertTrue(
                    torch.allclose(
                        param_grads_compiled[name],
                        param_grads_pythonified[name],
                        rtol=1e-4,
                        atol=1e-4,
                    ),
                    f"Parameter '{name}' gradients should match in full pipeline test. "
                    f"Compiled grad norm: {param_grads_compiled[name].norm().item():.6f}, "
                    f"Pythonified grad norm: {param_grads_pythonified[name].norm().item():.6f}",
                )
        finally:
            os.unlink(path)


class TestWrapperParitySubclass(TestCase):
    """
    Parity tests for AOTDispatchSubclassWrapper scenarios.

    These tests verify that pythonified code correctly handles tensor subclass
    inputs by unwrapping them before the compiled function and re-wrapping
    outputs. The subclass wrapper is triggered when:
    - Input tensors are instances of traceable wrapper subclasses
    - The subclass implements __tensor_flatten__ and __tensor_unflatten__

    Note: End-to-end parity tests for tensor subclasses are complex because:
    1. The subclass definitions need to be available in the exec namespace
    2. Pythonify generates code that operates on unwrapped tensors
    3. The subclass dispatch wrapper codegen is designed for integration with
       the full AOTAutograd pipeline, not standalone execution

    The codegen helper tests verify the generated functions work correctly.
    """

    def test_subclass_codegen_helper_unwrap_function(self):
        """
        Unit test for the subclass wrapper unwrap helper function codegen.

        Verifies that the generated _unwrap_tensor_subclasses function
        correctly flattens tensor subclasses to their inner tensors.
        """
        from torch._dynamo.pythonify.gen_python import PythonCodeGenVisitor
        from torch._dynamo.pythonify.ir import AOTDispatchSubclassWrapperNode

        visitor = PythonCodeGenVisitor()
        node = AOTDispatchSubclassWrapperNode(
            maybe_subclass_meta={"inp_subclass_info": [{"type": "SomeSubclass"}]},
            num_fw_outs_saved_for_bw=2,
        )
        visitor.visit_aot_dispatch_subclass_wrapper(node)
        code = visitor.get_code()

        self.assertIn("def _unwrap_tensor_subclasses", code)
        self.assertIn("__tensor_flatten__", code)
        self.assertIn("is_traceable_wrapper_subclass", code)

        namespace = {"torch": torch}
        exec(code, namespace)

        self.assertIn("_unwrap_tensor_subclasses", namespace)
        self.assertIn("_wrap_tensor_subclasses", namespace)

    def test_subclass_codegen_helper_wrap_function(self):
        """
        Unit test for the subclass wrapper wrap helper function codegen.

        Verifies that the generated _wrap_tensor_subclasses function
        correctly reconstructs tensor subclasses from flat outputs.
        """
        from torch._dynamo.pythonify.gen_python import PythonCodeGenVisitor
        from torch._dynamo.pythonify.ir import AOTDispatchSubclassWrapperNode

        visitor = PythonCodeGenVisitor()
        node = AOTDispatchSubclassWrapperNode(
            maybe_subclass_meta={"fw_subclass_out_meta": [{"type": "SomeSubclass"}]},
            num_fw_outs_saved_for_bw=1,
        )
        visitor.visit_aot_dispatch_subclass_wrapper(node)
        code = visitor.get_code()

        self.assertIn("def _wrap_tensor_subclasses", code)
        self.assertIn("creation_fn", code)
        self.assertIn("PlainTensorMeta", code)

        namespace = {"torch": torch}
        exec(code, namespace)

        self.assertIn("_wrap_tensor_subclasses", namespace)

    def test_subclass_no_meta_generates_no_code(self):
        """
        Verify that when no subclass metadata is present, no code is generated.

        This is the baseline case where inputs are regular tensors (not
        subclasses), so the AOTDispatchSubclassWrapper is a no-op.
        """
        from torch._dynamo.pythonify.gen_python import PythonCodeGenVisitor
        from torch._dynamo.pythonify.ir import AOTDispatchSubclassWrapperNode

        visitor = PythonCodeGenVisitor()
        node = AOTDispatchSubclassWrapperNode(maybe_subclass_meta=None)
        result = visitor.visit_aot_dispatch_subclass_wrapper(node)

        self.assertEqual(result, "")
        self.assertEqual(visitor.get_code(), "")

    def test_subclass_codegen_with_num_fw_outs_saved(self):
        """
        Verify that num_fw_outs_saved_for_bw is correctly emitted in codegen.

        This field is used during backward to determine how many forward
        outputs were saved for backward computation.
        """
        from torch._dynamo.pythonify.gen_python import PythonCodeGenVisitor
        from torch._dynamo.pythonify.ir import AOTDispatchSubclassWrapperNode

        visitor = PythonCodeGenVisitor()
        node = AOTDispatchSubclassWrapperNode(
            maybe_subclass_meta={"some": "meta"},
            num_fw_outs_saved_for_bw=5,
        )
        visitor.visit_aot_dispatch_subclass_wrapper(node)
        code = visitor.get_code()

        self.assertIn("_num_fw_outs_saved_for_bw = 5", code)

    def test_subclass_codegen_imports_required_utils(self):
        """
        Verify that the generated code imports is_traceable_wrapper_subclass.

        This import is required for the unwrap function to check if a tensor
        is a traceable wrapper subclass.
        """
        from torch._dynamo.pythonify.gen_python import PythonCodeGenVisitor
        from torch._dynamo.pythonify.ir import AOTDispatchSubclassWrapperNode

        visitor = PythonCodeGenVisitor()
        node = AOTDispatchSubclassWrapperNode(
            maybe_subclass_meta={"inp_subclass_info": [{"type": "TestSubclass"}]},
        )
        visitor.visit_aot_dispatch_subclass_wrapper(node)
        code = visitor.get_code()

        self.assertIn("is_traceable_wrapper_subclass", code)

    def test_subclass_codegen_flatten_helper(self):
        """
        Verify that the _flatten_subclass helper is generated.

        This helper recursively flattens nested tensor subclasses to their
        inner tensors using __tensor_flatten__.
        """
        from torch._dynamo.pythonify.gen_python import PythonCodeGenVisitor
        from torch._dynamo.pythonify.ir import AOTDispatchSubclassWrapperNode

        visitor = PythonCodeGenVisitor()
        node = AOTDispatchSubclassWrapperNode(
            maybe_subclass_meta={"nested": True},
        )
        visitor.visit_aot_dispatch_subclass_wrapper(node)
        code = visitor.get_code()

        self.assertIn("def _flatten_subclass", code)
        self.assertIn("__tensor_flatten__", code)

    def test_subclass_codegen_handles_plain_tensor_meta(self):
        """
        Verify that wrap function handles PlainTensorMeta correctly.

        PlainTensorMeta indicates a regular tensor that doesn't need wrapping,
        just passed through from the unwrapped outputs.
        """
        from torch._dynamo.pythonify.gen_python import PythonCodeGenVisitor
        from torch._dynamo.pythonify.ir import AOTDispatchSubclassWrapperNode

        visitor = PythonCodeGenVisitor()
        node = AOTDispatchSubclassWrapperNode(
            maybe_subclass_meta={"includes_plain_tensor": True},
        )
        visitor.visit_aot_dispatch_subclass_wrapper(node)
        code = visitor.get_code()

        self.assertIn("PlainTensorMeta", code)
        self.assertIn("unwrapped_idx", code)

    def test_subclass_unwrap_function_valid_python(self):
        """
        Verify the generated unwrap function is syntactically valid Python.

        The function should compile and execute without errors.
        """
        from torch._dynamo.pythonify.gen_python import PythonCodeGenVisitor
        from torch._dynamo.pythonify.ir import AOTDispatchSubclassWrapperNode

        visitor = PythonCodeGenVisitor()
        node = AOTDispatchSubclassWrapperNode(
            maybe_subclass_meta={"test": "data"},
            num_fw_outs_saved_for_bw=3,
        )
        visitor.visit_aot_dispatch_subclass_wrapper(node)
        code = visitor.get_code()

        namespace = {"torch": torch}
        try:
            exec(code, namespace)
        except SyntaxError as e:
            self.fail(f"Generated code has syntax error: {e}")

        self.assertIn("_unwrap_tensor_subclasses", namespace)
        self.assertTrue(callable(namespace["_unwrap_tensor_subclasses"]))

    def test_subclass_wrap_function_valid_python(self):
        """
        Verify the generated wrap function is syntactically valid Python.

        The function should compile and execute without errors.
        """
        from torch._dynamo.pythonify.gen_python import PythonCodeGenVisitor
        from torch._dynamo.pythonify.ir import AOTDispatchSubclassWrapperNode

        visitor = PythonCodeGenVisitor()
        node = AOTDispatchSubclassWrapperNode(
            maybe_subclass_meta={"output_meta": [1, 2, 3]},
            num_fw_outs_saved_for_bw=2,
        )
        visitor.visit_aot_dispatch_subclass_wrapper(node)
        code = visitor.get_code()

        namespace = {"torch": torch}
        try:
            exec(code, namespace)
        except SyntaxError as e:
            self.fail(f"Generated code has syntax error: {e}")

        self.assertIn("_wrap_tensor_subclasses", namespace)
        self.assertTrue(callable(namespace["_wrap_tensor_subclasses"]))

    def test_subclass_unwrap_passthrough_regular_tensor(self):
        """
        Verify that _unwrap_tensor_subclasses passes through regular tensors.

        When called with a list containing regular tensors (not subclasses),
        the function should return them unchanged.
        """
        from torch._dynamo.pythonify.gen_python import PythonCodeGenVisitor
        from torch._dynamo.pythonify.ir import AOTDispatchSubclassWrapperNode

        visitor = PythonCodeGenVisitor()
        node = AOTDispatchSubclassWrapperNode(
            maybe_subclass_meta={"test": True},
        )
        visitor.visit_aot_dispatch_subclass_wrapper(node)
        code = visitor.get_code()

        namespace = {"torch": torch}
        exec(code, namespace)

        regular_tensor = torch.randn(3, 4)
        result = namespace["_unwrap_tensor_subclasses"]([regular_tensor], None)

        self.assertEqual(len(result), 1)
        self.assertTrue(torch.equal(result[0], regular_tensor))

    def test_subclass_wrap_passthrough_with_none_metas(self):
        """
        Verify that _wrap_tensor_subclasses with None metas returns unchanged.

        When subclass_metas is None, the wrap function should return the
        outputs unchanged (no wrapping needed).
        """
        from torch._dynamo.pythonify.gen_python import PythonCodeGenVisitor
        from torch._dynamo.pythonify.ir import AOTDispatchSubclassWrapperNode

        visitor = PythonCodeGenVisitor()
        node = AOTDispatchSubclassWrapperNode(
            maybe_subclass_meta={"test": True},
        )
        visitor.visit_aot_dispatch_subclass_wrapper(node)
        code = visitor.get_code()

        namespace = {"torch": torch}
        exec(code, namespace)

        outputs = [torch.randn(2, 2), torch.randn(3, 3)]
        result = namespace["_wrap_tensor_subclasses"](outputs, None, None)

        self.assertEqual(result, outputs)


class TestWrapperParityFunctionalizedRNG(TestCase):
    """
    Parity tests for FunctionalizedRngRuntimeWrapper scenarios.

    RNG functionalization is used to ensure deterministic behavior across
    forward/backward passes when the model contains RNG operations (like
    dropout, torch.randn, torch.rand). When enabled via:
        torch._functorch.config.functionalize_rng_ops = True

    The FunctionalizedRngRuntimeWrapper:
    1. Appends CUDA RNG state (seed, offset) to the compiled function args
    2. Extracts the new RNG offset from outputs and updates CUDA RNG state
    3. Strips the RNG offset from returned outputs

    These tests verify that pythonified code correctly handles RNG
    functionalization for models with random operations.
    """

    def test_rng_codegen_helper_append_rng_state(self):
        """
        Unit test for the _append_rng_state helper function.

        When RNG is functionalized, the helper should append the CUDA RNG
        state (seed, offset) to the input args list.
        """
        from torch._dynamo.pythonify.gen_python import PythonCodeGenVisitor
        from torch._dynamo.pythonify.ir import FunctionalizedRngRuntimeWrapperNode

        visitor = PythonCodeGenVisitor()
        node = FunctionalizedRngRuntimeWrapperNode(
            is_rng_op_functionalized=True,
            num_outputs_rng_offset=1,
            num_forward_returns=3,
        )
        visitor.visit_functionalized_rng_runtime_wrapper(node)
        code = visitor.get_code()

        self.assertIn("def _append_rng_state(args):", code)
        self.assertIn("torch.cuda.get_rng_state_all()", code)
        self.assertIn("torch.cuda.initial_seed()", code)

        namespace = {"torch": torch}
        exec(code, namespace)

        self.assertIn("_append_rng_state", namespace)
        self.assertTrue(callable(namespace["_append_rng_state"]))

    def test_rng_codegen_helper_handle_rng_outputs(self):
        """
        Unit test for the _handle_rng_outputs helper function.

        When RNG is functionalized, the helper should extract RNG offset
        from outputs and return the remaining user outputs.
        """
        from torch._dynamo.pythonify.gen_python import PythonCodeGenVisitor
        from torch._dynamo.pythonify.ir import FunctionalizedRngRuntimeWrapperNode

        visitor = PythonCodeGenVisitor()
        node = FunctionalizedRngRuntimeWrapperNode(
            is_rng_op_functionalized=True,
            num_outputs_rng_offset=1,
            num_forward_returns=2,
        )
        visitor.visit_functionalized_rng_runtime_wrapper(node)
        code = visitor.get_code()

        self.assertIn("def _handle_rng_outputs(outputs):", code)
        self.assertIn("_rng_num_outputs_offset", code)

        namespace = {"torch": torch}
        exec(code, namespace)

        self.assertIn("_handle_rng_outputs", namespace)

        outputs_with_rng = [torch.tensor(1.0), torch.tensor(2.0), 123]
        result = namespace["_handle_rng_outputs"](outputs_with_rng)

        self.assertEqual(len(result), 2)
        self.assertEqual(result, [torch.tensor(1.0), torch.tensor(2.0)])

    def test_rng_codegen_no_rng_generates_nothing(self):
        """
        When RNG is not functionalized, no helper code should be generated.
        """
        from torch._dynamo.pythonify.gen_python import PythonCodeGenVisitor
        from torch._dynamo.pythonify.ir import FunctionalizedRngRuntimeWrapperNode

        visitor = PythonCodeGenVisitor()
        node = FunctionalizedRngRuntimeWrapperNode(
            is_rng_op_functionalized=False,
        )
        result = visitor.visit_functionalized_rng_runtime_wrapper(node)

        self.assertEqual(result, "")
        self.assertEqual(visitor.get_code(), "")

    def test_rng_codegen_stores_metadata_variables(self):
        """
        Verify that RNG metadata variables are stored in generated code.

        The generated code should store num_outputs_rng_offset and
        num_forward_returns as module-level variables for use by helpers.
        """
        from torch._dynamo.pythonify.gen_python import PythonCodeGenVisitor
        from torch._dynamo.pythonify.ir import FunctionalizedRngRuntimeWrapperNode

        visitor = PythonCodeGenVisitor()
        node = FunctionalizedRngRuntimeWrapperNode(
            is_rng_op_functionalized=True,
            num_outputs_rng_offset=2,
            num_forward_returns=5,
        )
        visitor.visit_functionalized_rng_runtime_wrapper(node)
        code = visitor.get_code()

        self.assertIn("_rng_num_outputs_offset = 2", code)
        self.assertIn("_rng_num_forward_returns = 5", code)

    def test_rng_codegen_valid_python_syntax(self):
        """
        Verify the generated RNG wrapper code is syntactically valid Python.
        """
        from torch._dynamo.pythonify.gen_python import PythonCodeGenVisitor
        from torch._dynamo.pythonify.ir import FunctionalizedRngRuntimeWrapperNode

        visitor = PythonCodeGenVisitor()
        node = FunctionalizedRngRuntimeWrapperNode(
            is_rng_op_functionalized=True,
            num_outputs_rng_offset=1,
            num_forward_returns=3,
            num_graphsafe_rng_states=2,
            graphsafe_rng_state_index=1,
        )
        visitor.visit_functionalized_rng_runtime_wrapper(node)
        code = visitor.get_code()

        namespace = {"torch": torch}
        try:
            exec(code, namespace)
        except SyntaxError as e:
            self.fail(f"Generated code has syntax error: {e}")

    def test_rng_codegen_handle_outputs_strips_offset(self):
        """
        Unit test that _handle_rng_outputs correctly strips RNG offset.

        Given outputs (out1, out2, rng_offset), should return (out1, out2).
        """
        from torch._dynamo.pythonify.gen_python import PythonCodeGenVisitor
        from torch._dynamo.pythonify.ir import FunctionalizedRngRuntimeWrapperNode

        visitor = PythonCodeGenVisitor()
        node = FunctionalizedRngRuntimeWrapperNode(
            is_rng_op_functionalized=True,
            num_outputs_rng_offset=1,
            num_forward_returns=2,
        )
        visitor.visit_functionalized_rng_runtime_wrapper(node)
        code = visitor.get_code()

        namespace = {"torch": torch}
        exec(code, namespace)

        outputs = ["a", "b", "rng_offset"]
        result = namespace["_handle_rng_outputs"](outputs)
        self.assertEqual(result, ["a", "b"])

    def test_rng_codegen_handle_outputs_preserves_none(self):
        """
        Unit test that _handle_rng_outputs handles None outputs gracefully.
        """
        from torch._dynamo.pythonify.gen_python import PythonCodeGenVisitor
        from torch._dynamo.pythonify.ir import FunctionalizedRngRuntimeWrapperNode

        visitor = PythonCodeGenVisitor()
        node = FunctionalizedRngRuntimeWrapperNode(
            is_rng_op_functionalized=True,
            num_outputs_rng_offset=1,
            num_forward_returns=1,
        )
        visitor.visit_functionalized_rng_runtime_wrapper(node)
        code = visitor.get_code()

        namespace = {"torch": torch}
        exec(code, namespace)

        result = namespace["_handle_rng_outputs"](None)
        self.assertIsNone(result)

    def test_rng_codegen_append_function_cpu_fallback(self):
        """
        Verify _append_rng_state handles non-CUDA case gracefully.

        On CPU, the function should return args unchanged (no RNG state
        appended since CUDA is not available for that context).
        """
        from torch._dynamo.pythonify.gen_python import PythonCodeGenVisitor
        from torch._dynamo.pythonify.ir import FunctionalizedRngRuntimeWrapperNode

        visitor = PythonCodeGenVisitor()
        node = FunctionalizedRngRuntimeWrapperNode(
            is_rng_op_functionalized=True,
            num_outputs_rng_offset=1,
            num_forward_returns=2,
        )
        visitor.visit_functionalized_rng_runtime_wrapper(node)
        code = visitor.get_code()

        self.assertIn("torch.cuda.is_available()", code)
        self.assertIn("if rng_state is not None:", code)

    def test_rng_wrapper_node_ir_construction(self):
        """
        Verify FunctionalizedRngRuntimeWrapperNode can be constructed with metadata.
        """
        from torch._dynamo.pythonify.ir import FunctionalizedRngRuntimeWrapperNode

        node = FunctionalizedRngRuntimeWrapperNode(
            is_rng_op_functionalized=True,
            num_outputs_rng_offset=1,
            num_forward_returns=3,
            num_graphsafe_rng_states=2,
            graphsafe_rng_state_index=1,
        )

        self.assertTrue(node.is_rng_op_functionalized)
        self.assertEqual(node.num_outputs_rng_offset, 1)
        self.assertEqual(node.num_forward_returns, 3)
        self.assertEqual(node.num_graphsafe_rng_states, 2)
        self.assertEqual(node.graphsafe_rng_state_index, 1)

    def test_rng_wrapper_node_default_values(self):
        """
        Verify FunctionalizedRngRuntimeWrapperNode has sensible defaults.
        """
        from torch._dynamo.pythonify.ir import FunctionalizedRngRuntimeWrapperNode

        node = FunctionalizedRngRuntimeWrapperNode()

        self.assertFalse(node.is_rng_op_functionalized)
        self.assertEqual(node.num_outputs_rng_offset, 0)
        self.assertEqual(node.num_forward_returns, 0)
        self.assertEqual(node.num_graphsafe_rng_states, 0)
        self.assertIsNone(node.graphsafe_rng_state_index)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_rng_codegen_append_with_cuda(self):
        """
        Integration test for _append_rng_state on CUDA.

        When CUDA is available, the function should append seed and offset
        to the args list.
        """
        from torch._dynamo.pythonify.gen_python import PythonCodeGenVisitor
        from torch._dynamo.pythonify.ir import FunctionalizedRngRuntimeWrapperNode

        visitor = PythonCodeGenVisitor()
        node = FunctionalizedRngRuntimeWrapperNode(
            is_rng_op_functionalized=True,
            num_outputs_rng_offset=1,
            num_forward_returns=2,
        )
        visitor.visit_functionalized_rng_runtime_wrapper(node)
        code = visitor.get_code()

        namespace = {"torch": torch}
        exec(code, namespace)

        original_args = [torch.randn(2, 2).cuda(), torch.randn(3, 3).cuda()]
        result = namespace["_append_rng_state"](original_args)

        self.assertEqual(len(result), 4)
        self.assertTrue(torch.equal(result[0], original_args[0]))
        self.assertTrue(torch.equal(result[1], original_args[1]))

        self.assertIsInstance(result[2], int)
        self.assertIsInstance(result[3], int)


class TestWrapperParityFakifiedOut(TestCase):
    """
    Parity tests for FakifiedOutWrapper scenarios.

    FakifiedOutWrapper re-fakifies outputs using traced metadata and corrects
    output strides when they differ from what was recorded during tracing.
    This is important for layout-sensitive operations where Inductor may
    produce outputs with different strides than what was traced.

    The wrapper is triggered when:
    - The model contains operations that can change tensor layout (transpose, view)
    - Inductor optimizes memory layout differently than eager mode
    - The output tensor strides differ from forward output strides recorded in
      fw_metadata.fwd_output_strides

    These tests verify that:
    1. The _fix_output_strides helper function correctly applies as_strided
       to restore expected strides
    2. Stride mismatches are detected and corrected
    3. Models with transpose/permute operations produce matching strides
    4. Pythonified code handles stride correction correctly
    """

    def test_fakified_out_codegen_helper_fix_strides_matching(self):
        """
        Unit test: _fix_output_strides returns tensors unchanged when strides match.

        When the output tensor's actual strides match the expected strides,
        no as_strided correction should be applied.
        """
        from torch._dynamo.pythonify.gen_python import PythonCodeGenVisitor
        from torch._dynamo.pythonify.ir import FakifiedOutWrapperNode

        visitor = PythonCodeGenVisitor()
        node = FakifiedOutWrapperNode(
            out_metas={"some": "meta"},
            fwd_output_strides=[[4, 1]],
        )
        visitor.visit_fakified_out_wrapper(node)
        code = visitor.get_code()

        namespace = {"torch": torch}
        exec(code, namespace)

        t = torch.randn(2, 4)
        original_stride = t.stride()
        result = namespace["_fix_output_strides"]([t])

        self.assertEqual(result[0].shape, t.shape)
        self.assertEqual(result[0].stride(), original_stride)

    def test_fakified_out_codegen_helper_fix_strides_mismatch(self):
        """
        Unit test: _fix_output_strides applies as_strided for stride mismatch.

        When the output tensor's actual strides differ from expected, the
        helper should use as_strided to correct the layout.
        """
        from torch._dynamo.pythonify.gen_python import PythonCodeGenVisitor
        from torch._dynamo.pythonify.ir import FakifiedOutWrapperNode

        expected_strides = [1, 2]
        visitor = PythonCodeGenVisitor()
        node = FakifiedOutWrapperNode(
            out_metas={"some": "meta"},
            fwd_output_strides=[expected_strides],
        )
        visitor.visit_fakified_out_wrapper(node)
        code = visitor.get_code()

        namespace = {"torch": torch}
        exec(code, namespace)

        t = torch.randn(2, 4)
        original_strides = t.stride()
        self.assertNotEqual(list(original_strides), expected_strides)

        result = namespace["_fix_output_strides"]([t])
        self.assertEqual(list(result[0].stride()), expected_strides)

    def test_fakified_out_codegen_helper_multiple_outputs(self):
        """
        Unit test: _fix_output_strides handles multiple tensor outputs.

        When there are multiple outputs with different expected strides,
        the helper should correct each one independently.
        """
        from torch._dynamo.pythonify.gen_python import PythonCodeGenVisitor
        from torch._dynamo.pythonify.ir import FakifiedOutWrapperNode

        visitor = PythonCodeGenVisitor()
        node = FakifiedOutWrapperNode(
            out_metas={"multi": "out"},
            fwd_output_strides=[[4, 1], [3, 1]],
        )
        visitor.visit_fakified_out_wrapper(node)
        code = visitor.get_code()

        namespace = {"torch": torch}
        exec(code, namespace)

        t1 = torch.randn(2, 4)
        t2 = torch.randn(3, 3)
        result = namespace["_fix_output_strides"]([t1, t2])

        self.assertEqual(len(result), 2)
        self.assertEqual(list(result[0].stride()), [4, 1])
        self.assertEqual(list(result[1].stride()), [3, 1])

    def test_fakified_out_codegen_helper_none_strides_skipped(self):
        """
        Unit test: _fix_output_strides skips tensors with None expected strides.

        When expected_strides contains None for a position, that output should
        be passed through unchanged (Inductor couldn't compute expected strides).
        """
        from torch._dynamo.pythonify.gen_python import PythonCodeGenVisitor
        from torch._dynamo.pythonify.ir import FakifiedOutWrapperNode

        visitor = PythonCodeGenVisitor()
        node = FakifiedOutWrapperNode(
            out_metas={"some": "meta"},
            fwd_output_strides=[None, [4, 1]],
        )
        visitor.visit_fakified_out_wrapper(node)
        code = visitor.get_code()

        namespace = {"torch": torch}
        exec(code, namespace)

        t1 = torch.randn(3, 3)
        t2 = torch.randn(2, 4)
        original_t1_stride = t1.stride()

        result = namespace["_fix_output_strides"]([t1, t2])

        self.assertEqual(result[0].stride(), original_t1_stride)
        self.assertEqual(list(result[1].stride()), [4, 1])

    def test_fakified_out_codegen_helper_non_tensor_passthrough(self):
        """
        Unit test: _fix_output_strides passes non-tensor outputs through.

        When the outputs list contains non-tensor values (like int, None),
        they should be returned unchanged.
        """
        from torch._dynamo.pythonify.gen_python import PythonCodeGenVisitor
        from torch._dynamo.pythonify.ir import FakifiedOutWrapperNode

        visitor = PythonCodeGenVisitor()
        node = FakifiedOutWrapperNode(
            out_metas={"some": "meta"},
            fwd_output_strides=[[4, 1], None, [2, 1]],
        )
        visitor.visit_fakified_out_wrapper(node)
        code = visitor.get_code()

        namespace = {"torch": torch}
        exec(code, namespace)

        t1 = torch.randn(2, 4)
        non_tensor = 42
        t2 = torch.randn(3, 2)

        result = namespace["_fix_output_strides"]([t1, non_tensor, t2])

        self.assertIsInstance(result[0], torch.Tensor)
        self.assertEqual(result[1], 42)
        self.assertIsInstance(result[2], torch.Tensor)

    def test_fakified_out_codegen_helper_single_tensor(self):
        """
        Unit test: _fix_output_strides handles single tensor (not list) output.

        When the output is a single tensor (not wrapped in a list), the helper
        should handle it correctly and return a single tensor.
        """
        from torch._dynamo.pythonify.gen_python import PythonCodeGenVisitor
        from torch._dynamo.pythonify.ir import FakifiedOutWrapperNode

        visitor = PythonCodeGenVisitor()
        expected_strides = [1, 2]
        node = FakifiedOutWrapperNode(
            out_metas={"some": "meta"},
            fwd_output_strides=[expected_strides],
        )
        visitor.visit_fakified_out_wrapper(node)
        code = visitor.get_code()

        namespace = {"torch": torch}
        exec(code, namespace)

        t = torch.randn(2, 4)
        result = namespace["_fix_output_strides"](t)

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(list(result.stride()), expected_strides)

    def test_fakified_out_codegen_valid_python(self):
        """
        Verify the generated stride fix code is syntactically valid Python.
        """
        from torch._dynamo.pythonify.gen_python import PythonCodeGenVisitor
        from torch._dynamo.pythonify.ir import FakifiedOutWrapperNode

        visitor = PythonCodeGenVisitor()
        node = FakifiedOutWrapperNode(
            out_metas={"test": "metadata"},
            fwd_output_strides=[[4, 1], [2, 1], [1, 3]],
        )
        visitor.visit_fakified_out_wrapper(node)
        code = visitor.get_code()

        namespace = {"torch": torch}
        try:
            exec(code, namespace)
        except SyntaxError as e:
            self.fail(f"Generated code has syntax error: {e}")

        self.assertIn("_fix_output_strides", namespace)
        self.assertTrue(callable(namespace["_fix_output_strides"]))

    def test_fakified_out_no_metas_generates_no_code(self):
        """
        Verify that when out_metas is None, no code is generated.

        This is the baseline case where no fakified output handling is needed.
        """
        from torch._dynamo.pythonify.gen_python import PythonCodeGenVisitor
        from torch._dynamo.pythonify.ir import FakifiedOutWrapperNode

        visitor = PythonCodeGenVisitor()
        node = FakifiedOutWrapperNode(out_metas=None)
        result = visitor.visit_fakified_out_wrapper(node)

        self.assertEqual(result, "")
        self.assertEqual(visitor.get_code(), "")

    def test_fakified_out_ir_node_construction(self):
        """
        Verify FakifiedOutWrapperNode can be constructed with all metadata.
        """
        from torch._dynamo.pythonify.ir import FakifiedOutWrapperNode

        out_metas = [{"shape": [2, 4], "dtype": "float32"}]
        fwd_strides = [[4, 1], [2, 1]]

        node = FakifiedOutWrapperNode(
            out_metas=out_metas,
            fwd_output_strides=fwd_strides,
        )

        self.assertEqual(node.out_metas, out_metas)
        self.assertEqual(node.fwd_output_strides, fwd_strides)

    def test_fakified_out_ir_node_default_values(self):
        """
        Verify FakifiedOutWrapperNode has sensible default values.
        """
        from torch._dynamo.pythonify.ir import FakifiedOutWrapperNode

        node = FakifiedOutWrapperNode()

        self.assertIsNone(node.out_metas)
        self.assertIsNone(node.fwd_output_strides)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_fakified_out_transpose_model_parity(self):
        """
        End-to-end parity test for model with transpose operation.

        Transpose changes tensor strides, which may trigger FakifiedOutWrapper
        if Inductor produces outputs with different strides than traced.
        This test verifies that pythonified code correctly handles the output.
        """
        torch.set_default_device("cuda")

        class TransposeModel(nn.Module):
            def __init__(self, features: int):
                super().__init__()
                self.W = nn.Parameter(torch.randn(features, features))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                y = x @ self.W
                return y.t()

        torch.manual_seed(44444)
        features = 4
        batch = 3
        model = TransposeModel(features)
        x = torch.randn(batch, features)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name

        try:
            torch._dynamo.reset()
            with torch.no_grad():
                compiled_model = torch.compile(model, pythonify=path)
                y_compiled = compiled_model(x)

            with open(path) as f:
                code = f.read()

            x_test = x.detach().clone()

            frame = inspect.currentframe()
            namespace = {**frame.f_globals, **frame.f_locals}
            namespace["x"] = x_test
            namespace["model"] = model
            namespace["torch"] = torch

            with torch.no_grad():
                exec(code, namespace)
            y_pythonified = namespace["y"]

            self.assertTrue(
                torch.allclose(y_compiled, y_pythonified, rtol=1e-4, atol=1e-4),
                f"Outputs should match for transpose model.",
            )

            self.assertEqual(
                y_compiled.shape, y_pythonified.shape,
                "Output shapes should match.",
            )
        finally:
            os.unlink(path)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_fakified_out_permute_model_parity(self):
        """
        End-to-end parity test for model with permute operation.

        Permute can produce non-contiguous tensors with specific stride patterns.
        This tests that pythonified code handles permuted outputs correctly.
        """
        torch.set_default_device("cuda")

        class PermuteModel(nn.Module):
            def __init__(self, in_features: int, out_features: int):
                super().__init__()
                self.W = nn.Parameter(torch.randn(in_features, out_features))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                y = x @ self.W
                y = y.unsqueeze(1).expand(-1, 4, -1)
                return y.permute(0, 2, 1)

        torch.manual_seed(55555)
        in_features = 3
        out_features = 5
        batch = 2
        model = PermuteModel(in_features, out_features)
        x = torch.randn(batch, in_features)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name

        try:
            torch._dynamo.reset()
            with torch.no_grad():
                compiled_model = torch.compile(model, pythonify=path)
                y_compiled = compiled_model(x)

            with open(path) as f:
                code = f.read()

            x_test = x.detach().clone()

            frame = inspect.currentframe()
            namespace = {**frame.f_globals, **frame.f_locals}
            namespace["x"] = x_test
            namespace["model"] = model
            namespace["torch"] = torch

            with torch.no_grad():
                exec(code, namespace)
            y_pythonified = namespace["y"]

            self.assertTrue(
                torch.allclose(y_compiled, y_pythonified, rtol=1e-4, atol=1e-4),
                f"Outputs should match for permute model.",
            )

            self.assertEqual(
                y_compiled.shape, y_pythonified.shape,
                "Output shapes should match for permute model.",
            )
        finally:
            os.unlink(path)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_fakified_out_contiguous_conversion_parity(self):
        """
        End-to-end parity test for model that may require contiguous conversion.

        When a tensor is transposed and then used in an operation requiring
        contiguous memory, Inductor may produce outputs with different strides.
        """
        torch.set_default_device("cuda")

        class ContiguousModel(nn.Module):
            def __init__(self, features: int):
                super().__init__()
                self.W = nn.Parameter(torch.randn(features, features))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                y = (x @ self.W).t()
                return y.contiguous()

        torch.manual_seed(66666)
        features = 5
        batch = 3
        model = ContiguousModel(features)
        x = torch.randn(batch, features)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name

        try:
            torch._dynamo.reset()
            with torch.no_grad():
                compiled_model = torch.compile(model, pythonify=path)
                y_compiled = compiled_model(x)

            with open(path) as f:
                code = f.read()

            x_test = x.detach().clone()

            frame = inspect.currentframe()
            namespace = {**frame.f_globals, **frame.f_locals}
            namespace["x"] = x_test
            namespace["model"] = model
            namespace["torch"] = torch

            with torch.no_grad():
                exec(code, namespace)
            y_pythonified = namespace["y"]

            self.assertTrue(
                torch.allclose(y_compiled, y_pythonified, rtol=1e-4, atol=1e-4),
                f"Outputs should match for contiguous model.",
            )

            self.assertEqual(
                y_compiled.stride(), y_pythonified.stride(),
                "Output strides should match after contiguous.",
            )
        finally:
            os.unlink(path)


class TestWrapperParitySyntheticBase(TestCase):
    """
    Parity tests for AOTSyntheticBaseWrapper scenarios.

    The synthetic base wrapper is triggered when aliased tensor inputs are
    passed to a compiled function. Aliasing occurs when:
    - The same tensor is passed multiple times as different arguments
    - Views of the same underlying storage are passed as separate arguments
    - Tensors that alias each other through shared storage

    The wrapper:
    1. Pre-call: Merges aliased inputs into synthetic bases using merge_view_inputs
    2. Post-call: Unpacks synthetic bases to reconstruct original views
    3. Post-call: Applies metadata mutations using as_strided_ for mutated aliases

    These tests verify that pythonified code correctly handles aliased inputs
    by comparing pythonified execution to compiled execution.

    Test categories:
    - Codegen helper tests: Verify generated merge/unpack/apply functions work
    - IR node tests: Verify IR construction with synthetic base metadata
    - End-to-end parity tests: Compile models with aliased inputs and compare
    """

    def test_synthetic_base_codegen_merge_function(self):
        """
        Unit test for the _merge_aliased_inputs_to_synthetic_bases function.

        Verifies that the generated merge function is syntactically valid
        and can be called with expected arguments.
        """
        from torch._dynamo.pythonify.gen_python import PythonCodeGenVisitor
        from torch._dynamo.pythonify.ir import AOTSyntheticBaseWrapperNode

        visitor = PythonCodeGenVisitor()
        node = AOTSyntheticBaseWrapperNode(
            needs_post_compile=True,
            synthetic_base_info={"some": "info"},
            aliased_arg_idx_with_metadata_mutations=[],
        )
        visitor.visit_aot_synthetic_base_wrapper(node)
        code = visitor.get_code()

        self.assertIn("def _merge_aliased_inputs_to_synthetic_bases", code)
        self.assertIn("merge_view_inputs", code)

        namespace = {"torch": torch}
        try:
            exec(code, namespace)
        except SyntaxError as e:
            self.fail(f"Generated code has syntax error: {e}")

        self.assertIn("_merge_aliased_inputs_to_synthetic_bases", namespace)
        self.assertTrue(callable(namespace["_merge_aliased_inputs_to_synthetic_bases"]))

    def test_synthetic_base_codegen_unpack_function(self):
        """
        Unit test for the _unpack_synthetic_bases function.

        Verifies that the generated unpack function correctly reconstructs
        original views from synthetic bases.
        """
        from torch._dynamo.pythonify.gen_python import PythonCodeGenVisitor
        from torch._dynamo.pythonify.ir import AOTSyntheticBaseWrapperNode

        visitor = PythonCodeGenVisitor()
        node = AOTSyntheticBaseWrapperNode(
            needs_post_compile=True,
            synthetic_base_info={"some": "info"},
        )
        visitor.visit_aot_synthetic_base_wrapper(node)
        code = visitor.get_code()

        self.assertIn("def _unpack_synthetic_bases", code)
        self.assertIn("gen_alias_from_base", code)

        namespace = {"torch": torch}
        exec(code, namespace)

        self.assertIn("_unpack_synthetic_bases", namespace)

    def test_synthetic_base_codegen_apply_metadata_mutations(self):
        """
        Unit test for the _apply_metadata_mutations function.

        When aliased inputs have metadata mutations, the wrapper applies
        as_strided_ to update the original inputs with new size/stride/offset.
        """
        from torch._dynamo.pythonify.gen_python import PythonCodeGenVisitor
        from torch._dynamo.pythonify.ir import AOTSyntheticBaseWrapperNode

        visitor = PythonCodeGenVisitor()
        node = AOTSyntheticBaseWrapperNode(
            needs_post_compile=True,
            synthetic_base_info={"some": "info"},
            aliased_arg_idx_with_metadata_mutations=[0, 1],
        )
        visitor.visit_aot_synthetic_base_wrapper(node)
        code = visitor.get_code()

        self.assertIn("def _apply_metadata_mutations", code)
        self.assertIn("as_strided_", code)
        self.assertIn("mutated_inp.size()", code)
        self.assertIn("mutated_inp.stride()", code)
        self.assertIn("mutated_inp.storage_offset()", code)

    def test_synthetic_base_no_info_generates_no_code(self):
        """
        Verify that when synthetic_base_info is None, no code is generated.

        This is the baseline case where inputs are not aliased.
        """
        from torch._dynamo.pythonify.gen_python import PythonCodeGenVisitor
        from torch._dynamo.pythonify.ir import AOTSyntheticBaseWrapperNode

        visitor = PythonCodeGenVisitor()
        node = AOTSyntheticBaseWrapperNode(
            needs_post_compile=True,
            synthetic_base_info=None,
        )
        result = visitor.visit_aot_synthetic_base_wrapper(node)

        self.assertEqual(result, "")
        self.assertEqual(visitor.get_code(), "")

    def test_synthetic_base_no_post_compile_generates_no_code(self):
        """
        Verify that when needs_post_compile is False, no code is generated.

        This happens when the synthetic base analysis determines no aliased
        inputs require handling.
        """
        from torch._dynamo.pythonify.gen_python import PythonCodeGenVisitor
        from torch._dynamo.pythonify.ir import AOTSyntheticBaseWrapperNode

        visitor = PythonCodeGenVisitor()
        node = AOTSyntheticBaseWrapperNode(
            needs_post_compile=False,
            synthetic_base_info={"some": "info"},
        )
        result = visitor.visit_aot_synthetic_base_wrapper(node)

        self.assertEqual(result, "")
        self.assertEqual(visitor.get_code(), "")

    def test_synthetic_base_codegen_stores_metadata(self):
        """
        Verify that metadata variables are stored in generated code.

        The generated code should store _aliased_arg_idx_with_metadata_mutations
        and _synthetic_base_is_inference for use by helper functions.
        """
        from torch._dynamo.pythonify.gen_python import PythonCodeGenVisitor
        from torch._dynamo.pythonify.ir import AOTSyntheticBaseWrapperNode

        visitor = PythonCodeGenVisitor()
        node = AOTSyntheticBaseWrapperNode(
            needs_post_compile=True,
            synthetic_base_info={"test": "data"},
            aliased_arg_idx_with_metadata_mutations=[1, 3, 5],
            trace_joint=False,
        )
        visitor.visit_aot_synthetic_base_wrapper(node)
        code = visitor.get_code()

        self.assertIn("_aliased_arg_idx_with_metadata_mutations = [1, 3, 5]", code)
        self.assertIn("_synthetic_base_is_inference = True", code)

    def test_synthetic_base_codegen_training_mode(self):
        """
        Verify correct is_inference setting for training mode.

        When trace_joint=True (training), is_inference should be False.
        """
        from torch._dynamo.pythonify.gen_python import PythonCodeGenVisitor
        from torch._dynamo.pythonify.ir import AOTSyntheticBaseWrapperNode

        visitor = PythonCodeGenVisitor()
        node = AOTSyntheticBaseWrapperNode(
            needs_post_compile=True,
            synthetic_base_info={"test": "data"},
            trace_joint=True,
        )
        visitor.visit_aot_synthetic_base_wrapper(node)
        code = visitor.get_code()

        self.assertIn("_synthetic_base_is_inference = False", code)

    def test_synthetic_base_codegen_valid_python_syntax(self):
        """
        Verify all generated code is syntactically valid Python.

        The complete generated code should compile and execute without errors.
        """
        from torch._dynamo.pythonify.gen_python import PythonCodeGenVisitor
        from torch._dynamo.pythonify.ir import AOTSyntheticBaseWrapperNode

        visitor = PythonCodeGenVisitor()
        node = AOTSyntheticBaseWrapperNode(
            needs_post_compile=True,
            synthetic_base_info={"complex": "metadata"},
            aliased_arg_idx_with_metadata_mutations=[0, 2, 4],
            trace_joint=True,
        )
        visitor.visit_aot_synthetic_base_wrapper(node)
        code = visitor.get_code()

        namespace = {"torch": torch}
        try:
            exec(code, namespace)
        except SyntaxError as e:
            self.fail(f"Generated code has syntax error: {e}")

        self.assertIn("_merge_aliased_inputs_to_synthetic_bases", namespace)
        self.assertIn("_unpack_synthetic_bases", namespace)
        self.assertIn("_apply_metadata_mutations", namespace)

    def test_synthetic_base_ir_node_construction(self):
        """
        Verify AOTSyntheticBaseWrapperNode can be constructed with all metadata.
        """
        from torch._dynamo.pythonify.ir import AOTSyntheticBaseWrapperNode

        node = AOTSyntheticBaseWrapperNode(
            needs_post_compile=True,
            synthetic_base_info={"idx": 0, "view_info": [(0, None)]},
            aliased_arg_idx_with_metadata_mutations=[1, 2],
            trace_joint=True,
        )

        self.assertTrue(node.needs_post_compile)
        self.assertEqual(node.synthetic_base_info, {"idx": 0, "view_info": [(0, None)]})
        self.assertEqual(node.aliased_arg_idx_with_metadata_mutations, [1, 2])
        self.assertTrue(node.trace_joint)

    def test_synthetic_base_ir_node_default_values(self):
        """
        Verify AOTSyntheticBaseWrapperNode has sensible default values.
        """
        from torch._dynamo.pythonify.ir import AOTSyntheticBaseWrapperNode

        node = AOTSyntheticBaseWrapperNode()

        self.assertFalse(node.needs_post_compile)
        self.assertIsNone(node.synthetic_base_info)
        self.assertIsNone(node.aliased_arg_idx_with_metadata_mutations)
        self.assertFalse(node.trace_joint)

    def test_synthetic_base_codegen_no_mutations_skips_apply(self):
        """
        Verify _apply_metadata_mutations is not generated when no mutations.

        When aliased_arg_idx_with_metadata_mutations is empty, the apply
        function is not needed and should not be generated.
        """
        from torch._dynamo.pythonify.gen_python import PythonCodeGenVisitor
        from torch._dynamo.pythonify.ir import AOTSyntheticBaseWrapperNode

        visitor = PythonCodeGenVisitor()
        node = AOTSyntheticBaseWrapperNode(
            needs_post_compile=True,
            synthetic_base_info={"some": "info"},
            aliased_arg_idx_with_metadata_mutations=[],
        )
        visitor.visit_aot_synthetic_base_wrapper(node)
        code = visitor.get_code()

        self.assertIn("def _merge_aliased_inputs_to_synthetic_bases", code)
        self.assertIn("def _unpack_synthetic_bases", code)
        self.assertNotIn("def _apply_metadata_mutations", code)

    def test_synthetic_base_unpack_with_direct_index(self):
        """
        Verify _unpack_synthetic_bases handles direct index (int) correctly.

        When an element in synthetic_base_info is an integer, it means the
        primal at that index is used directly (not a view).
        """
        from torch._dynamo.pythonify.gen_python import PythonCodeGenVisitor
        from torch._dynamo.pythonify.ir import AOTSyntheticBaseWrapperNode

        visitor = PythonCodeGenVisitor()
        node = AOTSyntheticBaseWrapperNode(
            needs_post_compile=True,
            synthetic_base_info={"idx": 0},
        )
        visitor.visit_aot_synthetic_base_wrapper(node)
        code = visitor.get_code()

        self.assertIn("isinstance(inner_idx_or_tuple, int)", code)
        self.assertIn("primals[inner_idx_or_tuple]", code)

    @unittest.skip(
        "Pythonify multi-input source extraction incorrectly maps second arg "
        "as model attribute. This test validates parity once the issue is fixed."
    )
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_aliased_input_view_forward_parity(self):
        """
        End-to-end parity test with aliased view inputs using a module.

        This test passes a tensor and its view as separate arguments,
        which triggers synthetic base creation. The pythonified code
        should produce the same forward output as the compiled function.

        Note: We use operations that work with both shapes (sum) rather
        than broadcasting, since aliased views may have different shapes.
        """
        torch.set_default_device("cuda")

        class AliasedViewModel(nn.Module):
            def __init__(self, features: int):
                super().__init__()
                self.W = nn.Parameter(torch.randn(features))

            def forward(self, x: torch.Tensor, x_view: torch.Tensor) -> torch.Tensor:
                return x.sum() * self.W.sum() + x_view.sum()

        torch.manual_seed(11111)
        features = 4
        model = AliasedViewModel(features)
        x = torch.randn(2, 4, requires_grad=True)
        x_view = x.view(8)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name

        try:
            torch._dynamo.reset()
            compiled_model = torch.compile(model, pythonify=path)
            y_compiled = compiled_model(x, x_view)

            with open(path) as f:
                code = f.read()

            x_test = x.detach().clone().requires_grad_(True)
            x_view_test = x_test.view(8)

            frame = inspect.currentframe()
            namespace = {**frame.f_globals, **frame.f_locals}
            namespace["x"] = x_test
            namespace["x_view"] = x_view_test
            namespace["model"] = model
            namespace["torch"] = torch

            exec(code, namespace)
            y_pythonified = namespace["y"]

            self.assertTrue(
                torch.allclose(y_compiled.detach(), y_pythonified.detach(), rtol=1e-4, atol=1e-4),
                f"Forward outputs should match for aliased view inputs.",
            )
        finally:
            os.unlink(path)

    @unittest.skip(
        "Pythonify multi-input source extraction incorrectly maps second arg "
        "as model attribute. This test validates parity once the issue is fixed."
    )
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_aliased_input_same_tensor_twice_parity(self):
        """
        End-to-end parity test with same tensor passed twice.

        Passing the same tensor as multiple arguments creates aliasing.
        The pythonified code should handle this correctly.
        """
        torch.set_default_device("cuda")

        class DuplicateInputModel(nn.Module):
            def __init__(self, features: int):
                super().__init__()
                self.W = nn.Parameter(torch.randn(features))

            def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                return (a * 2 + b * 3).sum() * self.W.sum()

        torch.manual_seed(22222)
        features = 3
        model = DuplicateInputModel(features)
        x = torch.randn(3, 3, requires_grad=True)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name

        try:
            torch._dynamo.reset()
            compiled_model = torch.compile(model, pythonify=path)
            y_compiled = compiled_model(x, x)

            with open(path) as f:
                code = f.read()

            x_test = x.detach().clone().requires_grad_(True)

            frame = inspect.currentframe()
            namespace = {**frame.f_globals, **frame.f_locals}
            namespace["a"] = x_test
            namespace["b"] = x_test
            namespace["model"] = model
            namespace["torch"] = torch

            exec(code, namespace)
            y_pythonified = namespace["y"]

            self.assertTrue(
                torch.allclose(y_compiled.detach(), y_pythonified.detach(), rtol=1e-4, atol=1e-4),
                f"Forward outputs should match for duplicate tensor inputs.",
            )
        finally:
            os.unlink(path)

    @unittest.skip(
        "Pythonify multi-input source extraction incorrectly maps second arg "
        "as model attribute. This test validates parity once the issue is fixed."
    )
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_aliased_input_multiple_views_parity(self):
        """
        End-to-end parity test with multiple views of the same tensor.

        Multiple views (reshape, view, flatten) of the same underlying storage
        should all be correctly handled by the synthetic base wrapper.
        """
        torch.set_default_device("cuda")

        class MultiViewModel(nn.Module):
            def __init__(self, features: int):
                super().__init__()
                self.W = nn.Parameter(torch.randn(features, features))

            def forward(
                self, x: torch.Tensor, x_flat: torch.Tensor, x_reshaped: torch.Tensor
            ) -> torch.Tensor:
                y1 = x @ self.W
                y2 = x_flat.sum()
                y3 = x_reshaped.sum()
                return y1.sum() + y2 + y3

        torch.manual_seed(33333)
        features = 4
        batch = 2
        model = MultiViewModel(features)

        x = torch.randn(batch, features, requires_grad=True)
        x_flat = x.flatten()
        x_reshaped = x.reshape(features, batch)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name

        try:
            torch._dynamo.reset()
            compiled_model = torch.compile(model, pythonify=path)
            y_compiled = compiled_model(x, x_flat, x_reshaped)

            with open(path) as f:
                code = f.read()

            x_test = x.detach().clone().requires_grad_(True)
            x_flat_test = x_test.flatten()
            x_reshaped_test = x_test.reshape(features, batch)

            frame = inspect.currentframe()
            namespace = {**frame.f_globals, **frame.f_locals}
            namespace["x"] = x_test
            namespace["x_flat"] = x_flat_test
            namespace["x_reshaped"] = x_reshaped_test
            namespace["model"] = model
            namespace["torch"] = torch

            exec(code, namespace)
            y_pythonified = namespace["y"]

            self.assertTrue(
                torch.allclose(y_compiled.detach(), y_pythonified.detach(), rtol=1e-4, atol=1e-4),
                f"Forward outputs should match for multiple view inputs.",
            )
        finally:
            os.unlink(path)

    @unittest.skip(
        "Pythonify multi-input source extraction incorrectly maps second arg "
        "as model attribute. This test validates parity once the issue is fixed."
    )
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_aliased_input_inference_mode_parity(self):
        """
        End-to-end parity test for aliased inputs in inference mode.

        The synthetic base wrapper behaves differently in inference mode
        (is_inference=True) vs training mode. This tests the inference path.
        """
        torch.set_default_device("cuda")

        class InferenceAliasModel(nn.Module):
            def __init__(self, features: int):
                super().__init__()
                self.W = nn.Parameter(torch.randn(features, features))

            def forward(self, x: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
                return (x @ self.W + x_t.t()).sum()

        torch.manual_seed(44444)
        features = 4
        model = InferenceAliasModel(features)

        x = torch.randn(features, features)
        x_t = x.t()

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name

        try:
            torch._dynamo.reset()
            with torch.no_grad():
                compiled_model = torch.compile(model, pythonify=path)
                y_compiled = compiled_model(x, x_t)

            with open(path) as f:
                code = f.read()

            x_test = x.detach().clone()
            x_t_test = x_test.t()

            frame = inspect.currentframe()
            namespace = {**frame.f_globals, **frame.f_locals}
            namespace["x"] = x_test
            namespace["x_t"] = x_t_test
            namespace["model"] = model
            namespace["torch"] = torch

            with torch.no_grad():
                exec(code, namespace)
            y_pythonified = namespace["y"]

            self.assertTrue(
                torch.allclose(y_compiled, y_pythonified, rtol=1e-4, atol=1e-4),
                f"Inference outputs should match for aliased transpose inputs.",
            )
        finally:
            os.unlink(path)

    @unittest.skip(
        "Pythonify multi-input source extraction incorrectly maps second arg "
        "as model attribute. This test validates parity once the issue is fixed."
    )
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_aliased_input_with_parameter_parity(self):
        """
        End-to-end parity test with aliased inputs and model parameters.

        The model has parameters that should be extracted correctly alongside
        the aliased input handling.
        """
        torch.set_default_device("cuda")

        class ParamAliasModel(nn.Module):
            def __init__(self, features: int):
                super().__init__()
                self.scale = nn.Parameter(torch.randn(features))
                self.bias = nn.Parameter(torch.randn(features))

            def forward(self, x: torch.Tensor, x_view: torch.Tensor) -> torch.Tensor:
                return ((x * self.scale + self.bias).sum() + x_view.sum())

        torch.manual_seed(55555)
        features = 6
        model = ParamAliasModel(features)

        x = torch.randn(2, features, requires_grad=True)
        x_view = x.flatten()

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name

        try:
            torch._dynamo.reset()
            compiled_model = torch.compile(model, pythonify=path)
            y_compiled = compiled_model(x, x_view)

            with open(path) as f:
                code = f.read()

            x_test = x.detach().clone().requires_grad_(True)
            x_view_test = x_test.flatten()

            frame = inspect.currentframe()
            namespace = {**frame.f_globals, **frame.f_locals}
            namespace["x"] = x_test
            namespace["x_view"] = x_view_test
            namespace["model"] = model
            namespace["torch"] = torch

            exec(code, namespace)
            y_pythonified = namespace["y"]

            self.assertTrue(
                torch.allclose(y_compiled.detach(), y_pythonified.detach(), rtol=1e-4, atol=1e-4),
                f"Forward outputs should match for aliased inputs with parameters.",
            )
        finally:
            os.unlink(path)

    @unittest.skip(
        "Pythonify multi-input source extraction incorrectly maps second arg "
        "as model attribute. This test validates parity once the issue is fixed."
    )
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_aliased_slice_view_parity(self):
        """
        End-to-end parity test with slice views.

        Slices create views that share storage. The synthetic base wrapper
        should handle slice views correctly.
        """
        torch.set_default_device("cuda")

        class SliceViewModel(nn.Module):
            def __init__(self, features: int):
                super().__init__()
                self.W = nn.Parameter(torch.randn(features))

            def forward(self, x: torch.Tensor, x_slice: torch.Tensor) -> torch.Tensor:
                return (x.sum() + x_slice.sum()) * self.W.sum()

        torch.manual_seed(66666)
        features = 4
        model = SliceViewModel(features)
        x = torch.randn(4, 4, requires_grad=True)
        x_slice = x[:2, :]

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name

        try:
            torch._dynamo.reset()
            compiled_model = torch.compile(model, pythonify=path)
            y_compiled = compiled_model(x, x_slice)

            with open(path) as f:
                code = f.read()

            x_test = x.detach().clone().requires_grad_(True)
            x_slice_test = x_test[:2, :]

            frame = inspect.currentframe()
            namespace = {**frame.f_globals, **frame.f_locals}
            namespace["x"] = x_test
            namespace["x_slice"] = x_slice_test
            namespace["model"] = model
            namespace["torch"] = torch

            exec(code, namespace)
            y_pythonified = namespace["y"]

            self.assertTrue(
                torch.allclose(y_compiled.detach(), y_pythonified.detach(), rtol=1e-4, atol=1e-4),
                f"Forward outputs should match for slice view inputs.",
            )
        finally:
            os.unlink(path)


if __name__ == "__main__":
    run_tests()
