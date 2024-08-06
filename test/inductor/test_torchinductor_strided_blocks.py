# Owner(s): ["module: inductor"]
import contextlib
import importlib
import unittest
from typing import Any, Callable, Optional, Tuple

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
        result, code = run_and_get_code(compiled, *args)

        # Check numerical accuracy
        ref_tensors = flatten_tensors(func(*args))
        actual_tensors = flatten_tensors(result)
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
            ((8, 8, 8, 8), (4, 4, 4, 4), None, None, True),
            ((8, 8), (4, 4), None, 10, True),  # Storage offset
            ((8, 8), (4, 4), (16, 2), None, True),  # Non-default strides
            ((8, 8), (4, 4), (1, 8), None, True),  # Transposed strides
            (
                (5, 9),
                (5, 8),
                None,
                None,
                True,
            ),  # Non-power-of-2 leading dim: block ptr
            (
                (15, 9),
                (15, 3),
                None,
                None,
                False,
            ),  # Non-power-of-2 inner dims: non-block ptr
            ((1, 1, 1), (1, 1, 1), None, None, False),  # Scalar: non-block ptr
            (
                (2, 4 * max_block),
                (2, 3 * max_block),
                None,
                None,
                True,
            ),  # Inner dim multiple of max_block
        ],
    )
    def test_pointwise(
        self,
        full_size: Tuple[int],
        view_size: Tuple[int],
        stride: Optional[Tuple[int]],
        offset: Optional[int],
        require_block_ptr: bool,
    ):
        """
        Test generating strided ND block pointers for a pointwise kernel.

        If require_block_ptr is True, the generated code must contain block
        pointers. However, ND block pointers are not supported for all shapes. So
        we also test some odd shapes with require_block_ptr set to False, to ensure that
        block pointer analysis does not break these cases.
        """

        def get_input() -> torch.Tensor:
            device = torch.device(GPU_TYPE)
            full = torch.randn(full_size).to(device)

            # Use the original tensor's stride by default
            view_stride = full.stride() if stride is None else stride

            return torch.as_strided(full, view_size, view_stride, storage_offset=offset)

        args = [get_input() for arg_idx in range(2)]

        # Expect 3 block pointers: 2 inputs 1 output
        self.run_and_compare(
            torch.add,
            *args,
            expected_num_block_pointers=3 if require_block_ptr else None,
        )

    @parametrize(
        "x_size,y_size",
        [
            ((8, 8), (8, 1)),
            ((8, 8), (1, 8)),
            (
                (4, 1, 4),
                (1, 4, 1),
            ),  # Very important case: index variables are disjoint!
            (
                (1, 1, 1, 4),
                (4, 4, 4, 4),
            ),  # Unmatched dims for first operand.
        ],
    )
    def test_broadcast(self, x_size: Tuple[int], y_size: Tuple[int]):
        """
        Test that we can generate strided block pointers when inputs have different
        shapes, and they are broadcast together.
        """

        def foo(x, y):
            a = x + 1
            b = y * 2
            return a + b

        def get_input(view_size: Tuple[int]) -> torch.Tensor:
            device = torch.device(GPU_TYPE)
            full_size = tuple(2 * dim for dim in view_size)
            full = torch.randn(full_size).to(device)
            view = torch.as_strided(full, view_size, full.stride())
            return view

        x, y = (get_input(size) for size in (x_size, y_size))

        # Check that input sizes are not the same
        self.assertNotEqual(x.shape, y.shape)

        # Check that at least one dimension is a singleton
        all_dims = x.shape + y.shape
        self.assertIn(1, all_dims)

        # Expect 3 block pointers: 2 inputs one output
        self.run_and_compare(foo, x, y, expected_num_block_pointers=3)

    @parametrize(
        "view_size,num_block_pointers,num_triton_kernels",
        [
            ((4, 4), 1, 1),
            ((4, 4, 4), 1, 1),
            ((8, 8, 8), 1, 1),
            ((15, 15), 0, 1),  # Non-power of 2
            ((3 * max_block, 2), 3, 2),  # Multiple of max block. Uses loops.
            (
                (2, 3 * max_block),
                3,
                2,
            ),  # Multiple of max block. Uses loops.
            ((128, 128), 3, 2),  # Test a large size, with loops.
        ],
    )
    def test_reduction(
        self, view_size: Tuple[int], num_block_pointers: int, num_triton_kernels: int
    ):
        """
        Tests a reduction kernel.
        """

        device = torch.device(GPU_TYPE)
        full_size = tuple(2 * dim for dim in view_size)
        full = torch.randn(full_size).to(device)
        view = torch.as_strided(full, view_size, full.stride())

        # Expect at least 1 block pointer for the input.
        # Add 2 more if we generate 2 kernels.
        result, (code,) = self.run_and_compare(
            torch.sum,
            view,
            expected_num_block_pointers=num_block_pointers,
            expected_num_triton_kernels=num_triton_kernels,
        )

    @parametrize(
        "view_size,num_block_pointers,num_triton_kernels",
        [
            ((8, 8), 2, 1),  # No loops. Should be supported.
            (
                (128, 128),
                None,
                None,
            ),  # Looped reduction. Block pointers not yet supported.
        ],
    )
    def test_mixed_pointwise_reduction(
        self, view_size: Tuple[int], num_block_pointers: int, num_triton_kernels: int
    ):
        """
        Tests mixing pointwise with reduction ops.
        """

        def foo(x, y):
            return torch.sum(x + y)

        device = torch.device(GPU_TYPE)
        full_size = tuple(2 * dim for dim in view_size)

        def get_input() -> torch.Tensor:
            full = torch.randn(full_size).to(device)
            view = torch.as_strided(full, view_size, full.stride())
            return view

        inputs = [get_input() for input_idx in range(2)]

        # Expect 2 block pointers: inputs
        result, (code,) = self.run_and_compare(
            foo,
            *inputs,
            expected_num_block_pointers=num_block_pointers,
            expected_num_triton_kernels=num_triton_kernels,
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

        device = torch.device(GPU_TYPE)
        full_size = (8, 8)
        view_size = (4, 4)
        full = torch.randn(full_size).to(device)
        view = torch.as_strided(full, view_size, full.stride())

        self.run_and_compare(torch.div, view, view, compile_kwargs={"dynamic": True})

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
