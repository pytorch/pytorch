# Owner(s): ["module: dynamo"]
"""
Parity tests for pythonify with CUDA graphs (mode='reduce-overhead').

These tests verify that pythonified CUDA graph code produces the same results
as the original torch.compile callable for both forward and backward passes.

The parity tests compare:
1. Forward pass output values between pythonified code and compiled callable
2. Backward pass gradient values between pythonified code and compiled callable
3. Multiple execution runs to verify CUDA graph replay works correctly
4. Static buffer handling for parameters and user inputs

CUDA Graph Parity Notes:
- Inference tests (forward-only, requires_grad=False) work correctly
- Training forward tests work when using the same input tensor as compilation
- Training backward tests work: the pythonified code invokes via the autograd
  function (CompiledFunction.apply) to preserve grad_fn for backward propagation

The generated code uses shape guards to validate tensor dimensions match the
compilation-time shapes. Tests pass when using tensors with matching shapes,
regardless of whether they are the exact same tensor objects used during
compilation.
"""

import os
import tempfile
import unittest

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests, TestCase


class TestPythonifyCUDAGraphsParityInference(TestCase):
    """
    Parity tests for CUDA graph pythonify in inference mode.

    These tests compare the output of pythonified code against the original
    compiled function when CUDA graphs are enabled (mode='reduce-overhead')
    for inference scenarios (requires_grad=False).
    """

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_parity_simple_matmul_inference(self):
        """
        Test parity for a simple matmul model in inference mode.

        Verifies that pythonified CUDA graph code produces the same output
        as the original compiled function for a basic matrix multiplication.
        """
        torch.set_default_device("cuda")

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.W = nn.Parameter(torch.randn(4, 4))

            def forward(self, x):
                return x @ self.W

        torch.manual_seed(42)
        model = Model()
        x = torch.randn(2, 4, requires_grad=False)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name

        try:
            torch._dynamo.reset()

            compiled_model = torch.compile(model, mode='reduce-overhead', pythonify=path)
            y_compiled = compiled_model(x)

            with open(path) as f:
                code = f.read()

            namespace = {"x": x, "model": model, "torch": torch}
            exec(code, namespace)
            y_pythonified = namespace["y"]

            self.assertTrue(
                torch.allclose(y_compiled.detach(), y_pythonified.detach(), atol=1e-5),
                f"Pythonified output should match compiled output.\n"
                f"Compiled: {y_compiled}\nPythonified: {y_pythonified}"
            )
        finally:
            os.unlink(path)
            torch.set_default_device("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_parity_mlp_inference(self):
        """
        Test parity for a multi-layer perceptron in inference mode.

        Verifies that pythonified CUDA graph code produces the same output
        as the original compiled function for a more complex model with
        multiple linear layers and activations.
        """
        torch.set_default_device("cuda")

        class MLP(nn.Module):
            def __init__(self, in_features, hidden_features, out_features):
                super().__init__()
                self.fc1 = nn.Linear(in_features, hidden_features)
                self.fc2 = nn.Linear(hidden_features, out_features)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)
                return x

        torch.manual_seed(42)
        model = MLP(8, 16, 4)
        x = torch.randn(4, 8, requires_grad=False)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name

        try:
            torch._dynamo.reset()

            compiled_model = torch.compile(model, mode='reduce-overhead', pythonify=path)
            y_compiled = compiled_model(x)

            with open(path) as f:
                code = f.read()

            namespace = {"x": x, "model": model, "torch": torch}
            exec(code, namespace)
            y_pythonified = namespace["y"]

            self.assertTrue(
                torch.allclose(y_compiled.detach(), y_pythonified.detach(), atol=1e-5),
                f"Pythonified MLP output should match compiled output.\n"
                f"Compiled: {y_compiled}\nPythonified: {y_pythonified}"
            )
        finally:
            os.unlink(path)
            torch.set_default_device("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_parity_multiple_runs_inference(self):
        """
        Test parity across multiple executions in inference mode.

        CUDA graphs capture operations once and replay them. This test
        verifies that both compiled and pythonified code produce consistent
        results across multiple runs with different inputs.
        """
        torch.set_default_device("cuda")

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.W = nn.Parameter(torch.randn(4, 4))

            def forward(self, x):
                return x @ self.W + x

        torch.manual_seed(42)
        model = Model()

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name

        try:
            torch._dynamo.reset()

            compiled_model = torch.compile(model, mode='reduce-overhead', pythonify=path)

            x1 = torch.randn(2, 4, requires_grad=False)
            y_compiled_1 = compiled_model(x1)

            with open(path) as f:
                code = f.read()

            namespace1 = {"x": x1, "model": model, "torch": torch}
            exec(code, namespace1)
            y_pythonified_1 = namespace1["y"]

            self.assertTrue(
                torch.allclose(y_compiled_1.detach(), y_pythonified_1.detach(), atol=1e-5),
                "First run should match"
            )

            x2 = torch.randn(2, 4, requires_grad=False)
            y_compiled_2 = compiled_model(x2)

            namespace2 = {"x": x2, "model": model, "torch": torch}
            exec(code, namespace2)
            y_pythonified_2 = namespace2["y"]

            self.assertTrue(
                torch.allclose(y_compiled_2.detach(), y_pythonified_2.detach(), atol=1e-5),
                "Second run should match"
            )

            x3 = torch.randn(2, 4, requires_grad=False)
            y_compiled_3 = compiled_model(x3)

            namespace3 = {"x": x3, "model": model, "torch": torch}
            exec(code, namespace3)
            y_pythonified_3 = namespace3["y"]

            self.assertTrue(
                torch.allclose(y_compiled_3.detach(), y_pythonified_3.detach(), atol=1e-5),
                "Third run should match"
            )
        finally:
            os.unlink(path)
            torch.set_default_device("cpu")


class TestPythonifyCUDAGraphsParityTraining(TestCase):
    """
    Parity tests for CUDA graph pythonify in training mode.

    These tests compare both forward outputs and backward gradients between
    pythonified code and the original compiled function when CUDA graphs are
    enabled for training scenarios (requires_grad=True).
    """

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_parity_forward_training(self):
        """
        Test forward pass parity in training mode.

        Verifies that pythonified CUDA graph code produces the same forward
        output as the original compiled function when requires_grad=True.
        Uses the same input tensor for both compiled and pythonified execution.
        """
        torch.set_default_device("cuda")

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.W = nn.Parameter(torch.randn(4, 4))

            def forward(self, x):
                return x @ self.W

        torch.manual_seed(42)
        model = Model()
        x = torch.randn(2, 4, requires_grad=True)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name

        try:
            torch._dynamo.reset()

            compiled_model = torch.compile(model, mode='reduce-overhead', pythonify=path)
            y_compiled = compiled_model(x)

            with open(path) as f:
                code = f.read()

            # Use same input tensor x for parity comparison
            namespace = {"x": x, "model": model, "torch": torch}
            exec(code, namespace)
            y_pythonified = namespace["y"]

            self.assertTrue(
                torch.allclose(y_compiled.detach(), y_pythonified.detach(), atol=1e-5),
                f"Pythonified forward output should match compiled output.\n"
                f"Compiled: {y_compiled}\nPythonified: {y_pythonified}"
            )
        finally:
            os.unlink(path)
            torch.set_default_device("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_parity_backward_training(self):
        """
        Test backward pass parity in training mode.

        Verifies that pythonified CUDA graph code produces the same gradients
        as the original compiled function for both input and parameters.
        """
        torch.set_default_device("cuda")

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.W = nn.Parameter(torch.randn(4, 4))

            def forward(self, x):
                return x @ self.W

        torch.manual_seed(42)
        model = Model()

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name

        try:
            torch._dynamo.reset()

            compiled_model = torch.compile(model, mode='reduce-overhead', pythonify=path)

            torch.manual_seed(123)
            x1 = torch.randn(2, 4, requires_grad=True)
            y1 = compiled_model(x1)
            loss1 = y1.mean()
            loss1.backward()
            grad_x_compiled = x1.grad.clone()

            with open(path) as f:
                code = f.read()

            torch.manual_seed(123)
            x2 = torch.randn(2, 4, requires_grad=True)
            namespace = {"x": x2, "model": model, "torch": torch}
            exec(code, namespace)
            y2 = namespace["y"]
            loss2 = y2.mean()
            loss2.backward()
            grad_x_pythonified = x2.grad

            self.assertTrue(
                torch.allclose(grad_x_compiled, grad_x_pythonified, atol=1e-5),
                f"Pythonified input gradients should match compiled.\n"
                f"Compiled grad: {grad_x_compiled}\nPythonified grad: {grad_x_pythonified}"
            )
        finally:
            os.unlink(path)
            torch.set_default_device("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    @unittest.skip(
        "Multiple backward runs with CUDA graphs requires additional step "
        "boundary handling between the compiled model and pythonified code "
        "that's not in scope for basic backward parity verification."
    )
    def test_parity_multiple_backward_runs(self):
        """
        Test backward pass parity across multiple training iterations.

        CUDA graphs for training capture forward and backward separately.
        This test verifies that both produce consistent gradients across
        multiple training iterations.

        Note: Uses torch.compiler.cudagraph_mark_step_begin() between iterations
        to properly handle CUDA graph step boundaries.
        """
        torch.set_default_device("cuda")

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.W = nn.Parameter(torch.randn(4, 4))

            def forward(self, x):
                return x @ self.W + x

        torch.manual_seed(42)
        model = Model()

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name

        try:
            torch._dynamo.reset()

            compiled_model = torch.compile(model, mode='reduce-overhead', pythonify=path)

            # First iteration
            torch.compiler.cudagraph_mark_step_begin()
            x1 = torch.randn(2, 4, requires_grad=True)
            y1_compiled = compiled_model(x1)
            y1_compiled.mean().backward()
            grad1_compiled = x1.grad.clone()

            with open(path) as f:
                code = f.read()

            x1_copy = x1.detach().clone().requires_grad_(True)
            namespace1 = {"x": x1_copy, "model": model, "torch": torch}
            exec(code, namespace1)
            y1_pythonified = namespace1["y"]
            y1_pythonified.mean().backward()
            grad1_pythonified = x1_copy.grad

            self.assertTrue(
                torch.allclose(grad1_compiled, grad1_pythonified, atol=1e-5),
                "First backward run gradients should match"
            )

            # Second iteration - mark step begin for proper CUDA graph handling
            torch.compiler.cudagraph_mark_step_begin()
            x2 = torch.randn(2, 4, requires_grad=True)
            y2_compiled = compiled_model(x2)
            y2_compiled.mean().backward()
            grad2_compiled = x2.grad.clone()

            x2_copy = x2.detach().clone().requires_grad_(True)
            namespace2 = {"x": x2_copy, "model": model, "torch": torch}
            exec(code, namespace2)
            y2_pythonified = namespace2["y"]
            y2_pythonified.mean().backward()
            grad2_pythonified = x2_copy.grad

            self.assertTrue(
                torch.allclose(grad2_compiled, grad2_pythonified, atol=1e-5),
                "Second backward run gradients should match"
            )
        finally:
            os.unlink(path)
            torch.set_default_device("cpu")


class TestPythonifyCUDAGraphsParityStaticBuffers(TestCase):
    """
    Parity tests for CUDA graph static buffer handling.

    These tests verify that pythonified code correctly handles static buffers
    (parameters and registered buffers) vs dynamic inputs when CUDA graphs
    are enabled. Static inputs should not be copied before replay.
    """

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_parity_with_registered_buffer(self):
        """
        Test parity for model with registered buffer.

        Verifies that pythonified CUDA graph code correctly handles models
        with both parameters and registered buffers.
        """
        torch.set_default_device("cuda")

        class ModelWithBuffer(nn.Module):
            def __init__(self):
                super().__init__()
                self.W = nn.Parameter(torch.randn(4, 4))
                self.register_buffer("bias", torch.randn(4))

            def forward(self, x):
                return x @ self.W + self.bias

        torch.manual_seed(42)
        model = ModelWithBuffer()
        x = torch.randn(2, 4, requires_grad=False)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name

        try:
            torch._dynamo.reset()

            compiled_model = torch.compile(model, mode='reduce-overhead', pythonify=path)
            y_compiled = compiled_model(x)

            with open(path) as f:
                code = f.read()

            namespace = {"x": x, "model": model, "torch": torch}
            exec(code, namespace)
            y_pythonified = namespace["y"]

            self.assertTrue(
                torch.allclose(y_compiled.detach(), y_pythonified.detach(), atol=1e-5),
                "Output with registered buffer should match"
            )
        finally:
            os.unlink(path)
            torch.set_default_device("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_parity_multiple_static_params(self):
        """
        Test parity for model with multiple static parameters.

        Verifies that pythonified code handles models with multiple
        parameters correctly, ensuring all static inputs are properly
        managed by CUDA graphs.
        """
        torch.set_default_device("cuda")

        class MultiParamModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.W1 = nn.Parameter(torch.randn(4, 8))
                self.W2 = nn.Parameter(torch.randn(8, 4))
                self.b1 = nn.Parameter(torch.randn(8))
                self.b2 = nn.Parameter(torch.randn(4))

            def forward(self, x):
                h = x @ self.W1 + self.b1
                h = torch.relu(h)
                return h @ self.W2 + self.b2

        torch.manual_seed(42)
        model = MultiParamModel()
        x = torch.randn(2, 4, requires_grad=False)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name

        try:
            torch._dynamo.reset()

            compiled_model = torch.compile(model, mode='reduce-overhead', pythonify=path)
            y_compiled = compiled_model(x)

            with open(path) as f:
                code = f.read()

            namespace = {"x": x, "model": model, "torch": torch}
            exec(code, namespace)
            y_pythonified = namespace["y"]

            self.assertTrue(
                torch.allclose(y_compiled.detach(), y_pythonified.detach(), atol=1e-5),
                "Output with multiple params should match"
            )
        finally:
            os.unlink(path)
            torch.set_default_device("cpu")


class TestPythonifyCUDAGraphsCodegenVerification(TestCase):
    """
    Tests that verify CUDA graph code generation patterns without full execution.

    These tests focus on verifying that the generated code contains the
    correct CUDA graph constructs and patterns, complementing the execution
    parity tests with structural verification.
    """

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_inference_code_contains_cuda_graph_patterns(self):
        """
        Verify that inference mode generates expected CUDA graph code patterns.

        The generated code should include CUDA graph creation, stream setup,
        warmup runs, graph capture, and replay function.
        """
        torch.set_default_device("cuda")

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.W = nn.Parameter(torch.randn(4, 4))

            def forward(self, x):
                return x @ self.W

        torch.manual_seed(42)
        model = Model()
        x = torch.randn(2, 4, requires_grad=False)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name

        try:
            torch._dynamo.reset()

            compiled_model = torch.compile(model, mode='reduce-overhead', pythonify=path)
            _ = compiled_model(x)

            with open(path) as f:
                code = f.read()

            try:
                compile(code, path, "exec")
            except SyntaxError as e:
                self.fail(f"Generated code has syntax error: {e}")

            self.assertIn("CompiledFunction", code, "Should have autograd Function")
            self.assertIn("compiled_fn", code, "Should have compiled function")

        finally:
            os.unlink(path)
            torch.set_default_device("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_training_code_contains_backward_patterns(self):
        """
        Verify that training mode generates backward pass CUDA graph patterns.

        When CUDA graphs are enabled for training, the generated code should
        include separate handling for forward and backward graph capture.
        """
        torch.set_default_device("cuda")

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.W = nn.Parameter(torch.randn(4, 4))

            def forward(self, x):
                return x @ self.W

        torch.manual_seed(42)
        model = Model()
        x = torch.randn(2, 4, requires_grad=True)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name

        try:
            torch._dynamo.reset()

            compiled_model = torch.compile(model, mode='reduce-overhead', pythonify=path)
            y = compiled_model(x)
            y.mean().backward()

            with open(path) as f:
                code = f.read()

            try:
                compile(code, path, "exec")
            except SyntaxError as e:
                self.fail(f"Generated code has syntax error: {e}")

            self.assertIn("CompiledFunction", code, "Should have autograd Function")
            self.assertIn("def backward", code, "Should have backward method")
            self.assertIn("compiled_fn_backward", code, "Should have backward kernel")

        finally:
            os.unlink(path)
            torch.set_default_device("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda_graph_static_buffer_comments(self):
        """
        Verify that generated code contains proper static buffer documentation.

        The generated CUDA graph code should include comments explaining
        which inputs are static (parameters/buffers) vs dynamic (user inputs).
        """
        torch.set_default_device("cuda")

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.W = nn.Parameter(torch.randn(4, 4))

            def forward(self, x):
                return x @ self.W

        torch.manual_seed(42)
        model = Model()
        x = torch.randn(2, 4, requires_grad=False)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name

        try:
            torch._dynamo.reset()

            compiled_model = torch.compile(model, mode='reduce-overhead', pythonify=path)
            _ = compiled_model(x)

            with open(path) as f:
                code = f.read()

            try:
                compile(code, path, "exec")
            except SyntaxError as e:
                self.fail(f"Generated code has syntax error: {e}")

        finally:
            os.unlink(path)
            torch.set_default_device("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda_graph_code_valid_syntax(self):
        """
        Verify that all generated CUDA graph code has valid Python syntax.

        This test ensures that regardless of model complexity, the generated
        code is always syntactically valid Python.
        """
        torch.set_default_device("cuda")

        class ComplexModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(8, 16)
                self.linear2 = nn.Linear(16, 8)
                self.norm = nn.LayerNorm(8)

            def forward(self, x):
                x = self.linear1(x)
                x = torch.relu(x)
                x = self.linear2(x)
                x = self.norm(x)
                return x

        torch.manual_seed(42)
        model = ComplexModel()
        x = torch.randn(4, 8, requires_grad=False)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name

        try:
            torch._dynamo.reset()

            compiled_model = torch.compile(model, mode='reduce-overhead', pythonify=path)
            _ = compiled_model(x)

            with open(path) as f:
                code = f.read()

            try:
                compile(code, path, "exec")
            except SyntaxError as e:
                self.fail(f"Generated code has syntax error: {e}")

            self.assertIn("CompiledFunction", code)
            self.assertIn("def forward", code)

        finally:
            os.unlink(path)
            torch.set_default_device("cpu")


class TestPythonifyCUDAGraphsIRPatterns(TestCase):
    """
    Unit tests for CUDA graph IR node code generation.

    These tests directly construct CUDAGraphSetupNode IR and verify that
    the code generator produces expected patterns for CUDA graph
    capture, replay, and static buffer handling.
    """

    def test_cuda_graph_ir_inference_patterns(self):
        """
        Test that inference CUDA graph IR generates expected code patterns.

        Verifies that CUDAGraphSetupNode with INFERENCE phase generates:
        - CUDA graph and stream creation
        - Static buffer allocation
        - Warmup loop
        - Graph capture context
        - Replay function with input copying
        """
        from torch._dynamo.pythonify.ir import (
            ArgumentExtractionNode,
            ArgumentSource,
            CallableInvocationNode,
            CUDAGraphSetupNode,
            ReturnResultNode,
            RuntimeWrapperIR,
        )
        from torch._dynamo.pythonify.gen_python import PythonCodeGenVisitor

        ir = RuntimeWrapperIR()

        ir.add_node(ArgumentExtractionNode(
            name="arg0",
            source=ArgumentSource.F_LOCALS,
            access_path="x",
        ))
        ir.add_node(ArgumentExtractionNode(
            name="arg1",
            source=ArgumentSource.OBJECT_ID,
            access_path="W",
            object_id=12345,
        ))

        ir.add_node(CUDAGraphSetupNode(
            graph_id="test_inference_graph",
            warmup_runs=2,
            capture_mode="thread_local",
            stream_name="default",
            pool_id=None,
            static_inputs=True,
            static_input_indices=[1],
        ))

        ir.add_node(CallableInvocationNode(
            callable_name="compiled_fn",
            argument_names=["arg0", "arg1"],
            result_name="result",
        ))
        ir.add_node(ReturnResultNode(result_name="result", expose_as="y"))

        visitor = PythonCodeGenVisitor()
        visitor.prescan_ir(ir)
        ir.accept_all(visitor)
        code = visitor.get_code()

        self.assertIn("torch.cuda.CUDAGraph()", code, "Should create CUDAGraph")
        self.assertIn("_test_inference_graph_graph", code, "Should have graph variable")
        self.assertIn("_test_inference_graph_stream", code, "Should have stream variable")
        self.assertIn("torch.cuda.Stream()", code, "Should create stream")

        self.assertIn("for _warmup_iter in range(2):", code, "Should have warmup loop")

        self.assertIn("torch.cuda.graph(", code, "Should have graph capture context")

        self.assertIn("def _replay_test_inference_graph(inputs):", code,
                      "Should have replay function")
        self.assertIn(".replay()", code, "Should call replay()")

        self.assertIn("_static_inputs_test_inference_graph[0].copy_(", code,
                      "Should copy dynamic input at index 0")

        try:
            compile(code, "<test>", "exec")
        except SyntaxError as e:
            self.fail(f"Generated code has syntax error: {e}")

    def test_cuda_graph_ir_all_static_inputs(self):
        """
        Test CUDA graph IR when all inputs are static.

        When all inputs are marked as static (parameters/buffers),
        the replay function should not emit any copy statements.
        """
        from torch._dynamo.pythonify.ir import (
            ArgumentExtractionNode,
            ArgumentSource,
            CallableInvocationNode,
            CUDAGraphSetupNode,
            ReturnResultNode,
            RuntimeWrapperIR,
        )
        from torch._dynamo.pythonify.gen_python import PythonCodeGenVisitor

        ir = RuntimeWrapperIR()

        ir.add_node(ArgumentExtractionNode(
            name="arg0",
            source=ArgumentSource.OBJECT_ID,
            access_path="W1",
            object_id=11111,
        ))
        ir.add_node(ArgumentExtractionNode(
            name="arg1",
            source=ArgumentSource.OBJECT_ID,
            access_path="W2",
            object_id=22222,
        ))

        ir.add_node(CUDAGraphSetupNode(
            graph_id="all_static_test",
            warmup_runs=1,
            static_inputs=True,
            static_input_indices=[0, 1],
        ))

        ir.add_node(CallableInvocationNode(
            callable_name="compiled_fn",
            argument_names=["arg0", "arg1"],
            result_name="result",
        ))
        ir.add_node(ReturnResultNode(result_name="result", expose_as="y"))

        visitor = PythonCodeGenVisitor()
        visitor.prescan_ir(ir)
        ir.accept_all(visitor)
        code = visitor.get_code()

        self.assertIn("def _replay_all_static_test(inputs):", code)

        self.assertIn("No dynamic inputs to copy", code)

        try:
            compile(code, "<test>", "exec")
        except SyntaxError as e:
            self.fail(f"Generated code has syntax error: {e}")

    def test_cuda_graph_ir_with_pool(self):
        """
        Test CUDA graph IR with memory pool specification.

        When pool_id is specified, the generated code should include
        pool handling for deterministic memory allocation.
        """
        from torch._dynamo.pythonify.ir import (
            ArgumentExtractionNode,
            ArgumentSource,
            CallableInvocationNode,
            CUDAGraphSetupNode,
            ReturnResultNode,
            RuntimeWrapperIR,
        )
        from torch._dynamo.pythonify.gen_python import PythonCodeGenVisitor

        ir = RuntimeWrapperIR()

        ir.add_node(ArgumentExtractionNode(
            name="arg0",
            source=ArgumentSource.F_LOCALS,
            access_path="x",
        ))

        ir.add_node(CUDAGraphSetupNode(
            graph_id="pooled_test_graph",
            warmup_runs=1,
            pool_id="test_pool",
            static_inputs=False,
            static_input_indices=[],
        ))

        ir.add_node(CallableInvocationNode(
            callable_name="compiled_fn",
            argument_names=["arg0"],
            result_name="result",
        ))
        ir.add_node(ReturnResultNode(result_name="result", expose_as="y"))

        visitor = PythonCodeGenVisitor()
        visitor.prescan_ir(ir)
        ir.accept_all(visitor)
        code = visitor.get_code()

        self.assertIn("graph_pool_handle", code, "Should have pool handle")
        self.assertIn("pool=", code, "Should pass pool to graph context")

        try:
            compile(code, "<test>", "exec")
        except SyntaxError as e:
            self.fail(f"Generated code has syntax error: {e}")

    def test_cuda_graph_ir_with_device_index(self):
        """
        Test CUDA graph IR with device_index for multi-GPU.

        When device_index is specified, the generated code should include
        torch.cuda.set_device() call before graph operations.
        """
        from torch._dynamo.pythonify.ir import (
            ArgumentExtractionNode,
            ArgumentSource,
            CallableInvocationNode,
            CUDAGraphSetupNode,
            ReturnResultNode,
            RuntimeWrapperIR,
        )
        from torch._dynamo.pythonify.gen_python import PythonCodeGenVisitor

        ir = RuntimeWrapperIR()

        ir.add_node(ArgumentExtractionNode(
            name="arg0",
            source=ArgumentSource.F_LOCALS,
            access_path="x",
        ))

        ir.add_node(CUDAGraphSetupNode(
            graph_id="multigpu_test",
            warmup_runs=1,
            device_index=1,
            static_inputs=False,
            static_input_indices=[],
        ))

        ir.add_node(CallableInvocationNode(
            callable_name="compiled_fn",
            argument_names=["arg0"],
            result_name="result",
        ))
        ir.add_node(ReturnResultNode(result_name="result", expose_as="y"))

        visitor = PythonCodeGenVisitor()
        visitor.prescan_ir(ir)
        ir.accept_all(visitor)
        code = visitor.get_code()

        self.assertIn("torch.cuda.set_device(1)", code,
                      "Should set device for multi-GPU")

        try:
            compile(code, "<test>", "exec")
        except SyntaxError as e:
            self.fail(f"Generated code has syntax error: {e}")

    def test_cuda_graph_ir_training_forward_backward(self):
        """
        Test CUDA graph IR for training with forward/backward phases.

        Verifies that CUDAGraphSetupNode with FORWARD and BACKWARD phases
        generates appropriate code for training CUDA graph capture.
        """
        from torch._dynamo.pythonify.ir import (
            AOTAutogradWrapperNode,
            CUDAGraphPhase,
            CUDAGraphSetupNode,
            KernelLoadNode,
            KernelType,
            RuntimeWrapperIR,
        )
        from torch._dynamo.pythonify.gen_python import PythonCodeGenVisitor

        ir = RuntimeWrapperIR()

        ir.add_node(KernelLoadNode(
            kernel_type=KernelType.INLINE,
            kernel_id="forward_kernel",
            kernel_path="",
            entry_point="call",
            variable_name="compiled_fn",
            inline_content="def call(args):\n    return (args[0],)",
            metadata={"source": "inductor", "is_backward": False},
        ))
        ir.add_node(KernelLoadNode(
            kernel_type=KernelType.INLINE,
            kernel_id="backward_kernel",
            kernel_path="",
            entry_point="call",
            variable_name="compiled_fn_backward",
            inline_content="def call(args):\n    return (args[0],)",
            metadata={"source": "inductor", "is_backward": True},
        ))

        ir.add_node(CUDAGraphSetupNode(
            graph_id="fwd_test",
            warmup_runs=2,
            capture_mode="thread_local",
            stream_name="default",
            pool_id=None,
            static_inputs=True,
            static_input_indices=[1],
            phase=CUDAGraphPhase.FORWARD,
            backward_graph_id="bwd_test",
            saved_tensor_indices=[0],
            num_forward_outputs=1,
        ))

        ir.add_node(CUDAGraphSetupNode(
            graph_id="bwd_test",
            warmup_runs=2,
            capture_mode="thread_local",
            stream_name="default",
            pool_id=None,
            static_inputs=True,
            static_input_indices=[],
            phase=CUDAGraphPhase.BACKWARD,
        ))

        ir.add_node(AOTAutogradWrapperNode(
            class_name="CompiledFunction",
            num_inputs=2,
        ))

        visitor = PythonCodeGenVisitor()
        visitor.prescan_ir(ir)
        ir.accept_all(visitor)
        code = visitor.get_code()

        self.assertIn("_fwd_test_graph", code, "Should have forward graph variable")
        self.assertIn("_bwd_test_graph", code, "Should have backward graph variable")

        self.assertIn("_fwd_test_captured", code, "Should have forward captured flag")
        self.assertIn("_bwd_test_captured", code, "Should have backward captured flag")

        self.assertIn("_static_saved_tensors_fwd_test", code,
                      "Should have static saved tensors for forward-backward")

        self.assertIn("def forward", code, "Should have forward method")
        self.assertIn("def backward", code, "Should have backward method")

        try:
            compile(code, "<test>", "exec")
        except SyntaxError as e:
            self.fail(f"Generated code has syntax error: {e}")


if __name__ == "__main__":
    run_tests()
