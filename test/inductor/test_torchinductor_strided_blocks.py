# Owner(s): ["module: inductor"]
import contextlib
import importlib
import unittest
from typing import Any, Callable, Iterable, Optional, Tuple

import torch
import torch.utils._pytree as pytree
from torch._inductor import config
from torch._inductor.runtime.hints import TRITON_MAX_BLOCK
from torch._inductor.runtime.runtime_utils import is_power_of_2
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_GPU,
    requires_gpu,
    skip_windows_ci,
)


skip_windows_ci(__name__, __file__)

importlib.import_module("filelock")

max_block: int = TRITON_MAX_BLOCK["X"]


@requires_gpu()
@config.patch("triton.use_block_ptr", True)
@instantiate_parametrized_tests
class TritonBlockPointerTest(InductorTestCase):
    def count_block_pointers(self, code: Iterable[str]) -> int:
        return sum(prog.count("tl.make_block_ptr") for prog in code)

    def run_and_compare(
        self,
        func: Callable[..., Any],
        *args,
        compile_kwargs: Optional[dict] = None,
        expected_num_block_pointers: Optional[int] = None,
        expected_num_programs: int = 1,
        expected_num_triton_kernels: int = 1,
    ):
        """
        Runs the module through Inductor, comparing to eager reference.
        """
        if compile_kwargs is None:
            compile_kwargs = {}

        def flatten_tensors(tensors):
            flat, spec = pytree.tree_flatten(tensors)
            return flat

        compiled = torch.compile(func, backend="inductor", **compile_kwargs)

        ref_tensors = flatten_tensors(func(*args))
        result, code = run_and_get_code(compiled, *args)
        actual_tensors = flatten_tensors(result)

        # Check numerical accuracy
        for ref, actual in zip(ref_tensors, actual_tensors):
            self.assertTrue(torch.allclose(ref, actual))

        def count_code(substr: str, expected: Optional[int]):
            count = sum(prog.count(substr) for prog in code)
            if expected is not None:
                self.assertEqual(count, expected)

        # Check the code
        self.assertEqual(len(code), expected_num_programs)
        count_code("@triton.jit", expected_num_triton_kernels)
        count_code("tl.make_block_ptr", expected_num_block_pointers)

        return result, code

    @parametrize(
        "expected_num_block_pointers,raises",
        [
            (3, False),  # This should pass
            (9, True),  # This should fail
        ],
    )
    def test_expected_num_block_pointers(
        self, expected_num_block_pointers: int, raises: bool
    ):
        """
        Checks that the test harness verifies the number of block pointers correctly.
        """

        def foo(x, y):
            return x + y

        device = torch.device(GPU_TYPE)
        inputs = [torch.randn(8).to(device) for arg_idx in range(2)]

        # Expect failure for bad inputs
        with self.assertRaises(AssertionError) if raises else contextlib.nullcontext():
            # Expect 3 block pointers: 2 inputs 1 output
            self.run_and_compare(
                foo, *inputs, expected_num_block_pointers=expected_num_block_pointers
            )

    @parametrize(
        "full_size,view_size,stride,offset,require_block_ptr",
        [
            ((64, 32, 32), (32, 16, 8), None, None, True),
            ((16, 8, 8, 8), (8, 8, 4, 2), None, None, True),
            ((8, 8), (4, 4), None, 10, True),  # Storage offset
            ((8, 8), (4, 4), (16, 2), None, True),  # Non-default strides
            ((8, 8), (4, 4), (1, 8), None, True),  # Transposed strides
            ((15, 9), (8, 8), None, None, True),  # Non-power-of-2 full dims
            ((15, 9), (15, 3), None, None, False),  # Non-power-of-2 view dims
            ((1, 1, 1), (1, 1, 1), None, None, False),  # Scalar
            (
                (2, 4 * max_block),
                (2, 3 * max_block),
                None,
                None,
                True,
            ),  # Multiple of max_block
        ],
    )
    def test_strided_block_ptr(
        self,
        full_size: Tuple[int],
        view_size: Tuple[int],
        stride: Optional[Tuple[int]],
        offset: Optional[int],
        require_block_ptr: bool,
    ):
        """
        Test generating strided ND block pointers.

        If require_block_ptr is True, the generated code must contain block
        pointers. However, ND block pointers are not supported for all shapes. So
        we also test some odd shapes with require_block_ptr set to False, to ensure that
        block pointer analysis does not break these cases.
        """

        def view(full: torch.Tensor):
            # Use the original tensor's stride by default
            view_stride = full.stride() if stride is None else stride
            return torch.as_strided(full, view_size, view_stride, storage_offset=offset)

        def foo(x, y):
            x, y = tuple(view(tensor) for tensor in (x, y))
            return x + y

        device = torch.device(GPU_TYPE)
        args = [torch.randn(full_size).to(device) for arg_idx in range(2)]

        # Expect 3 block pointers: 2 inputs 1 output
        self.run_and_compare(
            foo, *args, expected_num_block_pointers=3 if require_block_ptr else None
        )

    def test_broadcast(self):
        """
        Test that we can generate strided block pointers when inputs have different
        shapes, and they are broadcast together.
        """

        def foo(x, y):
            a = x + 1
            b = y * 2
            return a + b

        device = torch.device(GPU_TYPE)
        full_size = (16, 16)
        full = torch.randn(full_size).to(device)
        x_size = (8, 8)
        y_size = (1, 8)
        x, y = tuple(
            torch.as_strided(full, size, full.stride()) for size in (x_size, y_size)
        )

        # Check that input sizes are not the same
        self.assertNotEqual(x.shape, y.shape)

        # Broadcast is not yet supported, so we only expect 2 block pointers: one input, and output
        self.run_and_compare(foo, x, y, expected_num_block_pointers=2)

    def test_reduction(self):
        """
        Tests a reduction kernel.
        """

        device = torch.device(GPU_TYPE)
        full_size = (15, 15)
        view_size = (8, 8)
        full = torch.randn(full_size).to(device)
        view = torch.as_strided(full, view_size, full.stride())

        # Expect 1 block pointer: input
        result, (code,) = self.run_and_compare(
            torch.sum, view, expected_num_block_pointers=1
        )

    def test_multiple_max_block_non_power_of_2(self):
        """
        Check that we support dims of size n * MAX_BLOCK, where n is any positive integer, not
        necessarily a power of 2.
        """

        def foo(x):
            return x - 1

        device = torch.device(GPU_TYPE)
        full_size = (3 * max_block, 3)
        view_size = (3 * max_block, 2)
        full = torch.randn(full_size).to(device)
        view = torch.as_strided(full, view_size, full.stride())

        # Check that we're using dims that aren't all powers of 2
        have_np2_dim = not all(is_power_of_2(dim) for dim in view_size)
        self.assertTrue(have_np2_dim)

        # Check that we need more than one stride to represent the tensor
        nontrivial_dims = [dim for dim in view_size if dim > 1]
        self.assertTrue(len(nontrivial_dims) > 1)

        # Expect 2 block pointers: input and output
        self.run_and_compare(foo, view, expected_num_block_pointers=2)

    def test_dynamic_shapes_generic(self):
        """
        Test a generic strided block with dynamic shapes. Block pointers are not
        expected. This only checks that the analysis doesn't break this case.
        """

        def foo(x, y):
            return x / y

        device = torch.device(GPU_TYPE)
        full_size = (8, 8)
        view_size = (4, 4)
        full = torch.randn(full_size).to(device)
        view = torch.as_strided(full, view_size, full.stride())

        self.run_and_compare(foo, view, view, compile_kwargs={"dynamic": True})

    @unittest.skip(reason="Dynamo tracing error")
    def test_dynamic_shapes_multiple_max_block(self):
        """
        Test dynamic shapes, where we know the shape is a multiple of the max block
        size. We should be able to generate a block pointer for this case.
        """

        def foo(x):
            tile_dims = (3 * max_block * x.shape[0], 3 * x.shape[1])
            view_size = (3 * max_block * x.shape[0], 2 * x.shape[1])
            full = x.tile(tile_dims)
            view = torch.as_strided(full, view_size, full.stride())
            return view + view

        device = torch.device(GPU_TYPE)
        x_size = (1, 1)
        x = torch.randn(x_size).to(device)

        # Expect 2 block pointers: input and output
        self.run_and_compare(
            x, compile_kwargs={"dynamic": True}, expected_num_block_pointers=2
        )


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_GPU:
        run_tests(needs="filelock")
