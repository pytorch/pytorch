# Owner(s): ["module: inductor"]
"""
Test the FX IR backend.
"""

import operator
import unittest
from typing import Callable

import torch
from torch._higher_order_ops.triton_kernel_wrap import triton_kernel_wrapper_mutation
from torch._inductor import config
from torch._inductor.codegen.common import register_backend_for_device
from torch._inductor.codegen.triton import TritonScheduling
from torch._inductor.codegen.wrapper_fxir import delete, WrapperFxCodegen
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.test_case import TestCase as InductorTestCase
from torch.testing._internal.inductor_utils import (
    allclose_many,
    call_many,
    GPU_TYPE,
    HAS_GPU,
    requires_gpu,
    TRITON_HAS_CPU,
)


@requires_gpu()
class FxirTestCase(InductorTestCase):
    device = GPU_TYPE

    def _count_ops(self, gm: torch.fx.GraphModule, target: Callable) -> int:
        return len(gm.graph.find_nodes(op="call_function", target=target))

    @config.patch(
        compile_threads=1, size_asserts=False, scalar_asserts=False, nan_asserts=False
    )
    def _run_and_capture_graphs(self, opt, args) -> torch.fx.GraphModule:
        gms = []

        def generate(self, *args, **kwargs):
            nonlocal gms
            gms.append(self.gm)
            self._generate(*args, **kwargs)

        with unittest.mock.patch.object(
            torch._inductor.codegen.wrapper_fxir.WrapperFxCodegen, "generate", generate
        ):
            try:
                opt(*args)
            except torch._inductor.exc.InductorError:
                # Expect this exception, since the FX backend doesn't support Python codegen.
                pass

        return gms

    def _compile_and_check(
        self,
        func,
        args,
        expected_num_triton_kernels: int = 1,
        metadata_only: bool = False,
    ):
        opt = torch.compile(func, fullgraph=True)

        # Get the FX graph from the backend.
        (gm,) = self._run_and_capture_graphs(opt, args)

        # Check code
        num_kernels = self._count_ops(gm, triton_kernel_wrapper_mutation)
        self.assertEqual(num_kernels, expected_num_triton_kernels)

        # Check accuracy
        result = gm(*args)
        ref = func(*args)
        if metadata_only:

            def check_metadata(x, y):
                self.assertEqual(x.shape, y.shape)
                self.assertEqual(x.dtype, y.dtype)

            call_many(check_metadata, ref, result)
        else:
            allclose_many(self, ref, result)

        return gm

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        # Register the FX backend.
        register_backend_for_device(cls.device, TritonScheduling, WrapperFxCodegen)

    def test_basic(self):
        func = torch.add
        args = [torch.rand(8, device=self.device) for _ in range(2)]

        self._compile_and_check(func, args)

    def test_multiple_kernels(self):
        def foo(x, y):
            return x.sum() + y.sum()

        args = [torch.rand(length, device=self.device) for length in [517, 1029]]
        self._compile_and_check(foo, args, expected_num_triton_kernels=2)

    def test_free(self):
        """
        Test a program that frees a buffer which is no longer in use.
        """

        def foo(x, y, z):
            w = x.sum() + y
            return z.sum() + w.sum()

        args = [torch.rand(length, device=self.device) for length in [517, 1029, 123]]
        gm = self._compile_and_check(foo, args, expected_num_triton_kernels=3)

        # Check for frees
        num_frees = self._count_ops(gm, delete)
        self.assertGreater(num_frees, 0)

    def test_extern(self):
        """
        Test a program that calls an extern kernel.
        """

        def foo(x, y):
            return x @ y + y.sum()

        args = [torch.rand(size, device=self.device) for size in [(129, 129), (129, 1)]]
        gm = self._compile_and_check(foo, args, expected_num_triton_kernels=1)

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

        args = (torch.rand(length, device=self.device),)

        # Since the program has a random output, just check metadata.
        # Don't check for an exact value.
        gm = self._compile_and_check(
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

        args = [torch.rand(8, device=self.device) for _ in range(2)]
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

        args = [torch.rand(length, device=self.device)]
        gm = self._compile_and_check(foo, args, expected_num_triton_kernels=1)

        # Expect a single allocation, even though eager mode would use 2.
        num_allocs = self._count_ops(gm, torch.empty_strided)
        self.assertEqual(num_allocs, 1)

    def test_cat_reinterpret_view(self):
        """
        Test torch.cat using ReinterpretView.
        """
        length = 8

        def foo(x):
            y, z = tuple(torch.rand(length // 2, device=self.device) for _ in range(2))
            return x + torch.cat((y, z))

        args = [torch.rand(length, device=self.device)]

        # Since this test generates random numbers, check metadata only.
        gm = self._compile_and_check(
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

        args = [torch.rand((2, 4), device=self.device) for _ in range(2)]
        gm = self._compile_and_check(foo, args, expected_num_triton_kernels=1)

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

        args = [torch.rand(8, device=self.device)]
        gm = self._compile_and_check(foo, args, expected_num_triton_kernels=2)

        # Check for multiple kernel outputs via getitems.
        num_getitems = self._count_ops(gm, operator.getitem)
        self.assertEqual(num_getitems, 2)

        # Check for multiple graph outputs.
        output_node = gm.graph.find_nodes(op="output")[0]
        self.assertEqual(len(output_node.args[0]), 2)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_GPU or TRITON_HAS_CPU:
        run_tests(needs="filelock")
