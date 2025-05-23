# Owner(s): ["module: inductor"]
"""
Test the FX IR backend.
"""

import itertools
import operator
import unittest
from typing import Callable, Optional

import sympy

import torch
import torch._inductor.codegen.common as common
import torch.utils._pytree as pytree
from torch._dynamo.exc import BackendCompilerFailed
from torch._dynamo.utils import same
from torch._higher_order_ops.triton_kernel_wrap import triton_kernel_wrapper_mutation
from torch._inductor import config
from torch._inductor.codegen.common import register_backend_for_device
from torch._inductor.codegen.cpp import CppScheduling
from torch._inductor.codegen.triton import TritonScheduling
from torch._inductor.codegen.wrapper_fxir import FxConverter, WrapperFxCodegen
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.test_case import TestCase as InductorTestCase
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_GPU,
    requires_gpu,
    TRITON_HAS_CPU,
)


@requires_gpu()
@config.patch(
    compile_threads=1,
    alignment_asserts=False,
    size_asserts=False,
    scalar_asserts=False,
    nan_asserts=False,
)
class FxirTestCase(InductorTestCase):
    device = GPU_TYPE

    def _count_ops(self, gm: torch.fx.GraphModule, target: Callable) -> int:
        return len(gm.graph.find_nodes(op="call_function", target=target))

    def _run_and_capture_graphs(self, opt, args) -> torch.fx.GraphModule:
        gms = []

        orig_generate = FxConverter.generate

        def generate(self) -> torch.fx.GraphModule:
            nonlocal gms
            gm = orig_generate(self)
            gms.append(gm)
            return gm

        with unittest.mock.patch.object(
            torch._inductor.codegen.wrapper_fxir.FxConverter, "generate", generate
        ):
            opt(*args)

        return gms

    def _compile_and_check(
        self,
        func,
        args,
        expected_num_triton_kernels: int = 1,
        metadata_only: bool = False,
        compile_kwargs: Optional[dict] = None,
    ):
        if compile_kwargs is None:
            compile_kwargs = {}

        opt = torch.compile(func, **compile_kwargs)

        # Get the FX graph from the backend.
        gms = self._run_and_capture_graphs(opt, args)

        # Check the code for triton kernels.
        num_kernels = sum(
            self._count_ops(gm, triton_kernel_wrapper_mutation) for gm in gms
        )
        self.assertEqual(num_kernels, expected_num_triton_kernels)

        # Check accuracy.
        result = opt(*args)
        ref = func(*args)
        if metadata_only:
            # When we only want to check metadata, fill in zeros for tensor data.
            ref, result = tuple(
                pytree.tree_map(torch.zeros_like, x) for x in (ref, result)
            )

        self.assertTrue(same(ref, result))

        return gms

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        # Register the FX backend.
        register_backend_for_device(cls.device, TritonScheduling, WrapperFxCodegen)

    def test_basic(self):
        args = [torch.randn(8, device=self.device) for _ in range(2)]
        self._compile_and_check(torch.add, args)

    def test_multiple_kernels(self):
        def foo(x, y):
            return x.sum() + y.sum()

        args = [torch.randn(length, device=self.device) for length in [517, 1029]]
        self._compile_and_check(foo, args, expected_num_triton_kernels=2)

    def test_free(self):
        """
        Test a program that frees a buffer which is no longer in use.
        """

        def foo(x, y, z):
            w = x.sum() + y
            return z.sum() + w.sum()

        args = [torch.randn(length, device=self.device) for length in [517, 1029, 123]]
        (gm,) = self._compile_and_check(foo, args, expected_num_triton_kernels=3)

        # Check the generated code for frees.
        num_frees = gm.code.count("= None")
        self.assertGreater(num_frees, 0)

    def test_extern(self):
        """
        Test a program that calls an extern kernel.
        """

        def foo(x, y):
            return x @ y + y.sum()

        args = [
            torch.randn(size, device=self.device) for size in [(129, 129), (129, 1)]
        ]
        (gm,) = self._compile_and_check(foo, args, expected_num_triton_kernels=1)

        # Check for the extern kernel
        num_extern = self._count_ops(gm, extern_kernels.addmm)
        self.assertEqual(num_extern, 1)

    def test_fallback(self):
        """
        Test a program that calls an aten fallback.
        """

        length = 8

        def foo(x):
            return x + torch.randn(1, device=self.device)

        args = (torch.randn(length, device=self.device),)

        # Since the program has a random output, just check metadata.
        # Don't check for an exact value.
        (gm,) = self._compile_and_check(
            foo, args, expected_num_triton_kernels=2, metadata_only=True
        )

        # Check for the fallback kernel.
        num_fallback = self._count_ops(gm, torch.ops.aten.randint.low_out)
        self.assertEqual(num_fallback, 1)

    def test_cat_inputs(self):
        """
        Test concatenation of graph inputs.
        """

        def foo(x, y):
            return torch.cat((x, y)) + 1

        args = [torch.randn(8, device=self.device) for _ in range(2)]
        self._compile_and_check(foo, args, expected_num_triton_kernels=1)

    def test_cat_to_alloc(self):
        """
        Test concatenation that's optimized out to an allocation.
        """
        length = 8

        def foo(x):
            y, z = tuple(
                torch.arange(length // 2, device=self.device) for _ in range(2)
            )
            return x + torch.cat((y, z))

        args = [torch.randn(length, device=self.device)]
        (gm,) = self._compile_and_check(foo, args, expected_num_triton_kernels=1)

        # Expect a single allocation, even though eager mode would use 2.
        num_allocs = self._count_ops(gm, torch.empty_strided)
        self.assertEqual(num_allocs, 1)

    def test_cat_reinterpret_view(self):
        """
        Test torch.cat using ReinterpretView.
        """
        length = 8

        def foo(x):
            y, z = tuple(torch.randn(length // 2, device=self.device) for _ in range(2))
            return x + torch.cat((y, z))

        args = [torch.randn(length, device=self.device)]

        # Since this test generates random numbers, check metadata only.
        (gm,) = self._compile_and_check(
            foo, args, expected_num_triton_kernels=3, metadata_only=True
        )

        # Check for as_strided. We map ReinterpretView to this.
        num_as_strided = self._count_ops(gm, torch.as_strided)
        self.assertEqual(num_as_strided, 2)

    def test_reshape_output(self):
        """
        Test reshaping the output, which maps to a ReinterpretView.
        """

        def foo(x, y):
            return torch.reshape(x + y, (8,))

        args = [torch.randn((2, 4), device=self.device) for _ in range(2)]
        (gm,) = self._compile_and_check(foo, args, expected_num_triton_kernels=1)

        # Check for as_strided. We map ReinterpretView to this.
        num_as_strided = self._count_ops(gm, torch.as_strided)
        self.assertEqual(num_as_strided, 1)

    def test_extern_multi_output(self):
        """
        Test an extern kernel with multiple outputs.
        Also test a graph with multiple outputs.
        """

        def foo(x):
            top, idx = torch.topk(x, 2)
            return top + 1, idx * 2

        args = [torch.randn(8, device=self.device)]
        (gm,) = self._compile_and_check(foo, args, expected_num_triton_kernels=2)

        # Check for multiple kernel outputs via getitems.
        num_getitems = self._count_ops(gm, operator.getitem)
        self.assertEqual(num_getitems, 2)

        # Check for multiple graph outputs.
        output_node = gm.graph.find_nodes(op="output")[0]
        self.assertEqual(len(output_node.args[0]), 2)

    def test_duplicate_input(self):
        """
        Test duplicated inputs. This will collapse into a single input in the GM.
        """

        args = [torch.randn(4, device=self.device)] * 2
        (gm,) = self._compile_and_check(torch.add, args, expected_num_triton_kernels=1)

        num_placeholders = len(gm.graph.find_nodes(op="placeholder"))
        self.assertEqual(num_placeholders, 1)

    def test_backward(self):
        """
        Test a program with a backward pass.
        """

        x = torch.ones(5, device=self.device)  # input tensor
        y = torch.zeros(3, device=self.device)  # expected output
        w = torch.randn(5, 3, requires_grad=True, device=self.device)
        b = torch.randn(3, requires_grad=True, device=self.device)

        def foo(x, y):
            z = torch.matmul(x, w) + b
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
            loss.backward()
            return w.grad, b.grad

        # Expect separate forward and backward graphs.
        (forward_gm, backward_gm) = self._compile_and_check(
            foo, (x, y), expected_num_triton_kernels=3
        )

    def test_custom_compiler(self):
        """
        Test a derived backend with a custom compiler.
        """
        offset = 1

        class CustomWrapperCodegen(WrapperFxCodegen):
            def compile_graph(self, gm):
                def compiled_fn(*args):
                    # Adds an offset to the program's outputs.
                    outputs = gm(*args)
                    return pytree.tree_map(lambda x: x + 1, outputs)

                return compiled_fn

        args = [torch.randn(8, device=self.device) for _ in range(2)]
        custom_backend = common.DeviceCodegen(
            TritonScheduling, CustomWrapperCodegen, None
        )
        with unittest.mock.patch.dict(
            common.device_codegens, {self.device: custom_backend}
        ):
            func = torch.add
            opt = torch.compile(func)
            result = opt(*args)

        # Check the output is offset from eager mode.
        ref = func(*args)
        self.assertFalse(same(result, ref))
        self.assertNotEqual(offset, 0)
        self.assertTrue(same(result - offset, ref))

    def test_dynamic_shapes_and_strides(self):
        """
        Test a graph with dynamic shapes and strides.
        """

        static_dims = (8, 8)

        def get_input():
            full_size = (16, 8)
            full = torch.randn(full_size, device=self.device)
            view = torch.as_strided(full, static_dims, full.stride())
            return view

        func = torch.add
        args = [get_input() for _ in range(2)]
        (gm,) = self._compile_and_check(func, args, compile_kwargs={"dynamic": True})

        # Check for a symbolic output shape.
        (empty_strided,) = gm.graph.find_nodes(
            op="call_function", target=torch.empty_strided
        )
        example_tensor = empty_strided.meta["val"]
        symbolic_dims = example_tensor.shape
        self.assertEqual(len(symbolic_dims), len(static_dims))

        # Check for symbolic output strides.
        (stride, one) = example_tensor.stride()
        self.assertEqual(one, sympy.S.One)

        # Find the size symbols, and check for a corresponding placeholders defining them.
        for symbol in itertools.chain(symbolic_dims, [stride]):
            self.assertTrue(isinstance(symbol, torch.SymInt))
            (placeholder,) = [
                node
                for node in gm.graph.find_nodes(op="placeholder")
                if node.name == str(symbol)
            ]
            self.assertEqual(placeholder.meta["val"], symbol)

    @config.patch({"trace.enabled": True})
    @unittest.mock.patch("torch._inductor.debug.DebugFormatter.output_code")
    def test_debug(self, mock_output_code):
        # Compile in debug mode.
        args = [torch.randn(11, device=self.device) for _ in range(2)]
        self._compile_and_check(torch.sub, args)

        # Check the output code for a Triton kernel call.
        mock_output_code.assert_called_once()
        (output_filename,) = mock_output_code.call_args.args
        with open(output_filename) as f:
            output_code = f.read()
        self.assertIn("triton_kernel_wrapper_mutation", output_code)

    @torch._inductor.config.patch("graph_partition", True)
    def test_subgraph_raises(self):
        """
        Test a model with subgraphs. This is not yet supported, so check that we get the
        expected exception.
        """

        def foo(cond, x):
            return torch.cond(cond, torch.cos, torch.sin, [x])

        cond = torch.tensor([True], device=self.device)
        x = torch.ones([2, 3], device=self.device)

        with self.assertRaisesRegex(BackendCompilerFailed, "Subgraph"):
            self._compile_and_check(foo, [cond, x])

    def test_cpp_raises(self):
        """
        Test the C++ CPU backend. C++ kernels are not yet supported, so for now check
        that we get the expected exception.
        """

        def foo(x, y):
            return x + y * 5

        device = torch.device("cpu")
        args = [torch.randn(5, device=device) for _ in range(2)]

        cpp_backend = common.DeviceCodegen(CppScheduling, WrapperFxCodegen, None)
        with unittest.mock.patch.dict(
            common.device_codegens, {device.type: cpp_backend}
        ), self.assertRaisesRegex(BackendCompilerFailed, "Triton"):
            self._compile_and_check(foo, args)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_GPU or TRITON_HAS_CPU:
        run_tests(needs="filelock")
