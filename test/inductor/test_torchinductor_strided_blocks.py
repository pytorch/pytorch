# Owner(s): ["module: inductor"]
# ruff: noqa: F841
import contextlib
import importlib
import unittest
from typing import Any, Callable, Optional, Union

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
    skipIfXpu,
    subtest,
)
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_GPU,
    requires_gpu,
    skip_windows_ci,
    TRITON_HAS_CPU,
)


try:
    from . import test_torchinductor
except ImportError:
    import test_torchinductor


skip_windows_ci(__name__, __file__)

importlib.import_module("filelock")

max_block: int = TRITON_MAX_BLOCK["X"]

# Config shortcuts
tiled_reduction_config = {
    "triton.prefer_nd_tiling": True,
    "triton.tile_reductions": True,
}


def run_and_compare(
    self: InductorTestCase,
    func: Callable[..., Any],
    *args,
    compile_kwargs: Optional[dict] = None,
    expected_num_block_pointers: Optional[int] = None,
    expected_num_programs: int = 1,
    expected_num_triton_kernels: int = 1,
    config_patches: Optional[dict] = None,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
):
    """
    Runs the module through Inductor, comparing to eager reference.
    """
    if compile_kwargs is None:
        compile_kwargs = {}
    if config_patches is None:
        config_patches = {}

    def flatten_tensors(tensors):
        flat, spec = pytree.tree_flatten(tensors)
        return flat

    with config.patch(config_patches):
        compiled = torch.compile(func, backend="inductor", **compile_kwargs)
        result, code = run_and_get_code(compiled, *args)

    # Check numerical accuracy
    ref_tensors = flatten_tensors(func(*args))
    actual_tensors = flatten_tensors(result)
    for ref, actual in zip(ref_tensors, actual_tensors):
        # Don't clobber the default tolerance values
        tol = {t: v for t, v in {"rtol": rtol, "atol": atol}.items() if v is not None}
        self.assertTrue(torch.allclose(ref, actual, **tol))

    def count_code(substr: str, expected: Optional[int]):
        count = sum(prog.count(substr) for prog in code)
        if expected is not None:
            self.assertEqual(count, expected)

    # Check the code
    self.assertEqual(len(code), expected_num_programs)
    count_code("@triton.jit", expected_num_triton_kernels)
    count_code("tl.make_block_ptr", expected_num_block_pointers)

    return result, code


class BlockPointerTestBase(InductorTestCase):
    def _discontiguous_tensor(
        self, view_size: tuple[int, ...], device: Union[torch.device, str]
    ) -> torch.Tensor:
        """
        Create a padded tensor of the given size.
        The strides correspond to a tensor that is twice as large in each dimension.
        """
        if isinstance(device, str):
            device = torch.device(device)
        full_size = tuple(2 * dim for dim in view_size)
        full = torch.randn(full_size).to(device)
        view = torch.as_strided(full, view_size, full.stride())
        return view

    def _assert_pointwise_ndims(self, code, num_dims: int) -> None:
        pointwise_blocks = ["XBLOCK", "YBLOCK", "ZBLOCK"]
        return self._assert_tiling_ndims(code, pointwise_blocks, num_dims)

    def _assert_reduction_ndims(self, code, num_dims: int) -> None:
        reduction_blocks = ["R0_BLOCK", "R1_BLOCK"]
        return self._assert_tiling_ndims(code, reduction_blocks, num_dims)

    def _assert_tiling_ndims(self, code, blocks: list[str], num_dims: int) -> None:
        for expected_block in blocks[:num_dims]:
            self.assertIn(expected_block, code)
        for unexpected_block in blocks[num_dims:]:
            self.assertNotIn(unexpected_block, code)

    def _get_lines_containing_substr(self, code: str, substr: str) -> str:
        return "\n".join(line for line in code.split("\n") if substr in line)


@instantiate_parametrized_tests
class CommonTemplate:
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

        device = torch.device(self.device)
        inputs = [torch.randn(8).to(device) for arg_idx in range(2)]

        # Expect failure for bad inputs
        with self.assertRaises(AssertionError) if raises else contextlib.nullcontext():
            # Expect 3 block pointers: 2 inputs 1 output
            run_and_compare(
                self,
                foo,
                *inputs,
                expected_num_block_pointers=expected_num_block_pointers,
            )

    @parametrize("prefer_nd_tiling", [False, True])
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
            subtest(
                arg_values=(
                    (2, 4 * max_block),
                    (2, 3 * max_block),
                    None,
                    None,
                    True,
                ),  # Inner dim multiple of max_block
                decorators=[
                    test_torchinductor.skip_if_triton_cpu("Triton CPU: slow test")
                ],
            ),
        ],
    )
    def test_pointwise(
        self,
        full_size: tuple[int],
        view_size: tuple[int],
        stride: Optional[tuple[int]],
        offset: Optional[int],
        require_block_ptr: bool,
        prefer_nd_tiling: bool,
    ):
        """
        Test generating strided ND block pointers for a pointwise kernel.

        If require_block_ptr is True, the generated code must contain block
        pointers. However, ND block pointers are not supported for all shapes. So
        we also test some odd shapes with require_block_ptr set to False, to ensure that
        block pointer analysis does not break these cases.
        """

        def get_input() -> torch.Tensor:
            device = torch.device(self.device)
            full = torch.randn(full_size).to(device)

            # Use the original tensor's stride by default
            view_stride = full.stride() if stride is None else stride

            return torch.as_strided(full, view_size, view_stride, storage_offset=offset)

        args = [get_input() for arg_idx in range(2)]

        # Expect 3 block pointers: 2 inputs 1 output
        run_and_compare(
            self,
            torch.add,
            *args,
            expected_num_block_pointers=3 if require_block_ptr else None,
            config_patches={"triton.prefer_nd_tiling": prefer_nd_tiling},
        )

    @parametrize("prefer_nd_tiling", [False, True])
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
    def test_broadcast(
        self, x_size: tuple[int], y_size: tuple[int], prefer_nd_tiling: bool
    ):
        """
        Test that we can generate strided block pointers when inputs have different
        shapes, and they are broadcast together.
        """

        def foo(x, y):
            a = x + 1
            b = y * 2
            return a + b

        x, y = (
            self._discontiguous_tensor(size, self.device) for size in (x_size, y_size)
        )

        # Check that input sizes are not the same
        self.assertNotEqual(x.shape, y.shape)

        # Check that at least one dimension is a singleton
        all_dims = x.shape + y.shape
        self.assertIn(1, all_dims)

        # Expect 3 block pointers: 2 inputs one output
        run_and_compare(
            self,
            foo,
            x,
            y,
            expected_num_block_pointers=3,
            config_patches={"triton.prefer_nd_tiling": prefer_nd_tiling},
        )

    @parametrize(
        "x_size,y_size",
        [
            ((32, 1), (32, 32)),
            ((1, 8), (8, 8)),
            # ((4, 1, 3), (4, 5, 3)), # TODO: T207754224
            ((4, 1, 3), (4, 4, 3)),
            ((1, 5, 5), (5, 5, 5)),
            ((5, 5, 1), (5, 5, 5)),
            ((5, 1, 1), (5, 5, 5)),
            ((1, 1, 5), (5, 5, 5)),
            ((1, 5, 1), (5, 5, 5)),
            ((7, 1, 1, 4), (7, 3, 4, 4)),
            ((5, 6, 1, 1), (5, 6, 4, 3)),
        ],
    )
    def test_expand_broadcast(self, x_size: tuple[int], y_size: tuple[int]):
        """
        When the load and store have different shapes, we should use broadcast.
        """

        def foo(x, y_size):
            return x.expand(y_size).clone()

        def get_input(size: tuple[int]) -> torch.Tensor:
            device = torch.device(self.device)
            full = torch.randn(size).to(device)
            view = torch.as_strided(full, size, full.stride())
            return view

        x = get_input(x_size)
        y = y_size

        # Check that input sizes are not the same
        self.assertNotEqual(x_size, y_size)

        # Check that is valid broadcast
        self.assertEqual(len(x_size), len(y_size))
        for i, j in zip(x_size, y_size):
            if i != 1:
                self.assertEqual(i, j)

        result, (triton_code,) = run_and_compare(self, foo, x, y)

    @parametrize("prefer_nd_tiling", [False, True])
    @config.patch("triton.skip_l1_cache", False)
    def test_pointwise_broadcast_nonzero_strides(self, prefer_nd_tiling: bool):
        """
        Test that we emit tl.broadcast_to instead of using strides of 0.
        """

        full_shape = (8, 8)
        col_shape = (full_shape[1], 1)
        device = torch.device(self.device)
        full = torch.randn(full_shape).to(device)
        col = torch.as_strided(full, col_shape, full.stride())

        # Expect 3 block pointers: 2 inputs one output
        result, (triton_code,) = run_and_compare(
            self,
            torch.add,
            full,
            col,
            expected_num_block_pointers=3,
            config_patches={
                "triton.prefer_nd_tiling": prefer_nd_tiling,
            },
        )

        # Check the code for broadcasts.
        # We shouldn't see any strides of 0.
        load_lines, store_lines = tuple(
            self._get_lines_containing_substr(triton_code, substr)
            for substr in ("tl.load", "tl.store")
        )
        if prefer_nd_tiling:
            self.assertExpectedInline(
                load_lines,
                """\
    tmp0 = tl.load(tl.make_block_ptr(in_ptr0, shape=[8, 8], strides=[8, 1], block_shape=[YBLOCK, XBLOCK], order=[1, 0], offsets=[yoffset, xoffset]), boundary_check=[0, 1])
    tmp1 = tl.load(tl.make_block_ptr(in_ptr1, shape=[8], strides=[8], block_shape=[YBLOCK], order=[0], offsets=[yoffset]), boundary_check=[0], eviction_policy='evict_last')[:, None]""",  # noqa: B950
            )
            self.assertExpectedInline(
                store_lines,
                """    tl.store(tl.make_block_ptr(out_ptr0, shape=[8, 8], strides=[8, 1], block_shape=[YBLOCK, XBLOCK], order=[1, 0], offsets=[yoffset, xoffset]), tl.broadcast_to(tmp2, [YBLOCK, XBLOCK]).to(tl.float32), boundary_check=[0, 1])""",  # noqa: B950
            )
        else:
            self.assertExpectedInline(
                load_lines,
                """\
    tmp0 = tl.load(tl.make_block_ptr(in_ptr0, shape=[64], strides=[1], block_shape=[XBLOCK], order=[0], offsets=[xoffset]), boundary_check=[0])
    tmp1 = tl.reshape(tl.broadcast_to(tl.load(tl.make_block_ptr(in_ptr1, shape=[8], strides=[8], block_shape=[(7 + XBLOCK) // 8], order=[0], offsets=[xoffset // 8]), boundary_check=[0], eviction_policy='evict_last')[:, None, None], [(7 + XBLOCK) // 8, ((1) * ((1) <= ((7 + XBLOCK) // 8)) + ((7 + XBLOCK) // 8) * (((7 + XBLOCK) // 8) < (1))), ((8) * ((8) <= (XBLOCK)) + (XBLOCK) * ((XBLOCK) < (8)))]), [XBLOCK])""",  # noqa: B950
            )
            self.assertExpectedInline(
                store_lines,
                """    tl.store(tl.make_block_ptr(out_ptr0, shape=[64], strides=[1], block_shape=[XBLOCK], order=[0], offsets=[xoffset]), tl.broadcast_to(tmp2, [XBLOCK]).to(tl.float32), boundary_check=[0])""",  # noqa: B950
            )

    @parametrize("prefer_nd_tiling", [False, True])
    @parametrize(
        "view_size,num_block_pointers,num_triton_kernels",
        [
            ((4, 4), 1, 1),
            ((4, 4, 4), 1, 1),
            ((8, 8, 8), 1, 1),
            ((15, 15), None, 1),  # Non-power of 2
            # Multiple of max block. Uses loops.
            subtest(
                arg_values=((3 * max_block, 2), 3, 2),
                decorators=[
                    test_torchinductor.skip_if_triton_cpu("Triton CPU: slow test")
                ],
            ),
            (
                (2, 3 * max_block),
                2,
                2,
            ),  # Multiple of max block. Uses loops.
            ((128, 128), 3, 2),  # Test a large size, with loops.
        ],
    )
    def test_reduction(
        self,
        view_size: tuple[int],
        num_block_pointers: int,
        num_triton_kernels: int,
        prefer_nd_tiling: bool,
    ):
        """
        Tests a reduction kernel.
        """
        if self.device == "cpu" and all(
            # Multiple of max block. Uses loops.
            [
                view_size == (3 * max_block, 2),
                num_block_pointers == 3,
                num_triton_kernels == 2,
                prefer_nd_tiling is False,
            ]
        ):
            raise unittest.SkipTest(
                "Long test and raises BrokenProcessPool Error if triton CPU"
            )

        device = torch.device(self.device)

        view = self._discontiguous_tensor(view_size, self.device)

        if num_triton_kernels == 2 and config.triton.cooperative_reductions:
            # fewer kernels with cooperative reductions
            num_triton_kernels = 1
            num_block_pointers -= 2

        # Expect at least 1 block pointer for the input.
        # Add 2 more if we generate 2 kernels.
        result, (code,) = run_and_compare(
            self,
            torch.sum,
            view,
            expected_num_block_pointers=num_block_pointers,
            expected_num_triton_kernels=num_triton_kernels,
            config_patches={"triton.prefer_nd_tiling": prefer_nd_tiling},
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
        self, view_size: tuple[int], num_block_pointers: int, num_triton_kernels: int
    ):
        """
        Tests mixing pointwise with reduction ops.
        """

        def foo(x, y):
            return torch.sum(x + y)

        inputs = [
            self._discontiguous_tensor(view_size, self.device) for input_idx in range(2)
        ]

        # Expect 2 block pointers: inputs
        result, (code,) = run_and_compare(
            self,
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

        device = torch.device(self.device)
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
        run_and_compare(self, foo, view, expected_num_block_pointers=2)

    @parametrize(
        "nd_tiling,num_block_pointers",
        [
            (True, 2),  # With tiling, the index is affine.
            (False, 1),  # We can't infer that the load is a power of 2.
        ],
    )
    def test_dynamic_shapes_pointwise(self, nd_tiling: bool, num_block_pointers: int):
        """
        Test a pointwise kernel with dynamic shapes.
        """

        view_size = (4, 4)
        view = self._discontiguous_tensor(view_size, self.device)

        run_and_compare(
            self,
            torch.div,
            view,
            view,
            expected_num_block_pointers=num_block_pointers,
            config_patches={"triton.prefer_nd_tiling": nd_tiling},
            compile_kwargs={"dynamic": True},
        )

    @parametrize(
        "with_tiling,num_block_pointers",
        [
            (True, 1),  # With tiling, the index is affine.
            (False, 0),  # We can't infer that the load is a power of 2.
        ],
    )
    @skipIfXpu(msg="Remove this after Intel triton issue #4000 resolved.")
    def test_dynamic_shapes_reduction(self, with_tiling: bool, num_block_pointers: int):
        """
        Test a reduction kernel with dynamic shapes.
        """

        view_size = (4, 4)
        view = self._discontiguous_tensor(view_size, self.device)

        run_and_compare(
            self,
            torch.prod,
            view,
            expected_num_block_pointers=num_block_pointers,
            config_patches={
                "triton.prefer_nd_tiling": with_tiling,
                "triton.tile_reductions": with_tiling,
            },
            compile_kwargs={"dynamic": True},
        )

    @unittest.skip(reason="Dynamo tracing error")
    def test_dynamic_shapes_pointwise_multiple_max_block(self):
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

        device = torch.device(self.device)
        x_size = (1, 1)
        x = torch.randn(x_size).to(device)

        # Expect 2 block pointers: input and output
        run_and_compare(
            self, x, compile_kwargs={"dynamic": True}, expected_num_block_pointers=2
        )

    @parametrize(
        "full_size,view_size,num_block_pointers,num_tiles",
        [
            (
                (32, 32),
                (16, 32),
                3,
                1,
            ),  # Contiguous 2D tensor. Does not require tiling.
            ((5, 9), (3, 7), 3, 2),  # 2D tensor with 1 discontiguous dim.
            ((11, 13, 7), (9, 13, 5), 3, 2),  # 3D tensor with 1 discontiguous dim (2).
            subtest(
                arg_values=(
                    (3, 11, 13, 7),
                    (2, 9, 13, 7),
                    3,
                    2,
                ),
                decorators=[
                    test_torchinductor.skip_if_triton_cpu("Triton CPU: slow test")
                ],
            ),  # 4D tensor with 1 discontiguous dim (1).
            (
                (3, 11, 13, 7),
                (2, 11, 9, 7),
                3,
                2,
            ),
            # 4D tensor with 1 discontiguous dim (2).
            (
                (5, 5, 5, 5, 5),
                (3, 3, 5, 3, 5),
                1,
                2,
            ),  # 5D tensor with 2 discontiguous dims (3, 1). Block pointers unexpected.
        ],
    )
    def test_nd_tiling_odd_shapes_pointwise(
        self,
        full_size: tuple[int],
        view_size: tuple[int],
        num_block_pointers: int,
        num_tiles: int,
    ):
        """
        Test odd shapes with ND tiling enabled.
        Uses a pointwise op.
        """

        def get_input() -> torch.Tensor:
            device = torch.device(self.device)
            full = torch.randn(full_size).to(device)
            return torch.as_strided(full, view_size, full.stride())

        args = [get_input() for arg_idx in range(2)]

        # Expect up to 3 block pointers: 2 inputs 1 output.
        result, code = run_and_compare(
            self,
            torch.add,
            *args,
            expected_num_block_pointers=num_block_pointers,
            config_patches={
                "triton.prefer_nd_tiling": True,
            },
        )

        # Check the code for the expected tiling.
        all_tiles = ("XBLOCK", "YBLOCK", "ZBLOCK")
        expected_tiles = set(all_tiles[:num_tiles])
        for tile_name in all_tiles:
            for program in code:
                if tile_name in expected_tiles:
                    self.assertIn(tile_name, program)
                else:
                    self.assertNotIn(tile_name, program)

    @parametrize(
        "view_size,num_block_pointers,num_triton_kernels,reduction_op",
        [
            ((15, 15), 1, 1, torch.sum),  # Non-power-of 2 shapes.
            ((129, 129), 3, 2, torch.sum),  # Large size, with loops.
            ((3, 3), 1, 1, torch.argmax),
            ((129, 129), 1, 1, torch.argmax),
            ((5, 5), 1, 1, torch.var_mean),  # Reduction + pointwise fusion.
        ],
    )
    def test_2d_reduction_odd_shapes(
        self,
        view_size: tuple[int],
        num_block_pointers: int,
        num_triton_kernels: int,
        reduction_op: Callable,
    ):
        """
        Tests 2D reduction kernels. These arise from "odd" shapes which are not
        expressible with a 1D block pointer.
        """
        view = self._discontiguous_tensor(view_size, self.device)

        # Expect at least 1 block pointer for the input.
        # Add 2 more if we generate 2 kernels.
        result, (code,) = run_and_compare(
            self,
            reduction_op,
            view,
            expected_num_block_pointers=num_block_pointers,
            expected_num_triton_kernels=num_triton_kernels,
            config_patches=tiled_reduction_config,
        )

        # Check the code for multiple Rn_BLOCK's
        self._assert_reduction_ndims(code, 2)

    def test_2d_reduction_no_x_dim(self):
        """
        Tests a 2D reduction without an "x" dimension.
        """
        # We need a size to get no x dim.
        view = self._discontiguous_tensor((2, 346), self.device)

        # Expect 1 block pointer for the input.
        result, (code,) = run_and_compare(
            self,
            torch.prod,
            view,
            expected_num_block_pointers=1,
            expected_num_triton_kernels=1,
            config_patches=tiled_reduction_config,
        )

        # Check that there's no X dimension in the signature.
        (signature_line,) = (
            line for line in code.splitlines() if line.startswith("def triton")
        )
        self.assertNotIn("BLOCK", signature_line)

        # Check for 2 reduction dimensions in the body.
        self._assert_reduction_ndims(code, 2)

    @parametrize(
        "size,expected_num_block_pointers,expected_num_triton_kernels,expect_fallback",
        [
            ((8, 8), 1, 1, True),  # Persistent Welford fallback
            ((128, 128), 9, 2, False),  # Looped Welford reduction
        ],
    )
    def test_2d_welford_reduction(
        self,
        size: tuple[int],
        expected_num_block_pointers: int,
        expected_num_triton_kernels: int,
        expect_fallback: bool,
    ):
        """
        Tests a 2D welford reduction.

        NB: the input size should be "nice" in the sense that it's a multiple of the
        number of processors. Otherwise, we will get more complex indexing that
        doesn't generate a block pointer. Since tiling welford reductions depends on
        the block pointer analysis, those cases would fall back to 1D.
        """
        view = self._discontiguous_tensor(size, self.device)

        # We expect many block pointers for this one.
        result, (code,) = run_and_compare(
            self,
            torch.var_mean,
            view,
            expected_num_block_pointers=expected_num_block_pointers,
            expected_num_triton_kernels=expected_num_triton_kernels,
            config_patches=tiled_reduction_config,
        )

        # Check for a Welford reduction.
        self.assertEqual("welford" in code, not expect_fallback)

        # Check for 2 reduction dimensions.
        self._assert_reduction_ndims(code, 2)

    @test_torchinductor.skip_if_triton_cpu("Triton CPU: slow test")
    def test_welford_non_block_pointer(
        self,
    ):
        """
        Tests a welford reduction where block pointer analysis fails.
        The main loop will be a 1D reduction, instead of 2D.
        """
        # Use a "bad" size that's not evenly divisible by the launch grid.
        # This won't decompose into a block pointer.
        view = self._discontiguous_tensor((259, 311), self.device)

        # We expect many block pointers for this one.
        result, (code,) = run_and_compare(
            self,
            torch.var_mean,
            view,
            expected_num_block_pointers=6,
            expected_num_triton_kernels=2,
            config_patches={"triton.prefer_nd_tiling": True},
        )

        # Check for a Welford reduction.
        self.assertIn("welford", code)

        # Check for a single reduction dimension.
        self._assert_reduction_ndims(code, 1)

    def test_reduction_multiple_discontiguous_dims(self):
        """
        Test reducing a tensor with more than one discontiguous dimension. This case
        won't generate a block pointer, since we don'allow enough tiling dimensions.
        """
        # Use odd shapes to frustrate block pointer analysis.
        view = self._discontiguous_tensor((3, 7, 11), self.device)

        result, (code,) = run_and_compare(
            self,
            torch.sum,
            view,
            expected_num_block_pointers=0,
            expected_num_triton_kernels=1,
            config_patches=tiled_reduction_config,
        )

        # Check for 2 reduction dimensions.
        self._assert_reduction_ndims(code, 2)

    @test_torchinductor.skip_if_triton_cpu  # Illegal instruction  File; cannot xfail because it crashes process
    def test_2d_reduction_multi_kernel(self):
        """
        Test a 2D reduction in multi kernel mode.
        """
        view = self._discontiguous_tensor((2, 4, 1024), self.device)

        def foo(x):
            """
            Reshape to 2D and take the softmax of all trailing dims.
            """
            x = x.reshape(x.shape[0], -1)
            return torch.softmax(x, -1)

        result, (code,) = run_and_compare(
            self,
            foo,
            view,
            expected_num_block_pointers=6,
            expected_num_triton_kernels=2,
            config_patches={
                "triton.multi_kernel": True,
                **tiled_reduction_config,
            },
        )

        # Check for multi kernel mode.
        self.assertIn("multi_kernel", code)

        # Check for 2 reduction dimensions.
        self._assert_reduction_ndims(code, 2)

    def test_fused_2d_reduction(
        self,
    ):
        """
        Tests fusing multiple reductions on the same input, with 2D tiling.
        """

        def foo(x):
            return torch.sum(x) + torch.argmax(x)

        view_size = (5, 7)
        view = self._discontiguous_tensor(view_size, self.device)

        # Expect at least 1 block pointer for the input.
        result, (code,) = run_and_compare(
            self,
            foo,
            view,
            expected_num_block_pointers=1,
            expected_num_triton_kernels=1,
            config_patches=tiled_reduction_config,
        )

        # Check the code for multiple Rn_BLOCK's
        self._assert_reduction_ndims(code, 2)

    @parametrize("reduction_op", [torch.sum, torch.argmax])
    def test_2d_reductions_mixed_indexing(
        self,
        reduction_op: Callable,
    ):
        """
        Tests a program with multiple reductions using different strides.
        These might not be fused.
        """

        def foo(*args):
            return sum(reduction_op(arg) for arg in args)

        view_size = (5, 7)
        arg0 = self._discontiguous_tensor(view_size, self.device)
        arg1 = torch.empty(view_size)

        # No guarantees on the number of kernels or pointers.
        result, (code,) = run_and_compare(
            self,
            foo,
            arg0,
            arg1,
            config_patches=tiled_reduction_config,
        )

        # Check the code for multiple Rn_BLOCK's
        self._assert_reduction_ndims(code, 2)

    @parametrize(
        "tile_reductions",
        [False, True],
    )
    def test_enable_tiled_reductions(self, tile_reductions: bool):
        """
        Tests enabling and disabling tiled reductions.
        """
        view = self._discontiguous_tensor((9, 11), self.device)

        # If tiled, we expect 1 block pointer for the input.
        result, (code,) = run_and_compare(
            self,
            torch.sum,
            view,
            expected_num_block_pointers=1 if tile_reductions else 0,
            expected_num_triton_kernels=1,
            config_patches={
                "triton.prefer_nd_tiling": True,
                "triton.tile_reductions": tile_reductions,
            },
        )

        # Check the code for multiple Rn_BLOCK's
        self._assert_reduction_ndims(code, 2 if tile_reductions else 1)

    def test_complex_reshape_block_ptr(self):
        def func(x, y):
            add_ = x + y
            reshape_0 = add_.reshape([8, 16, 128])
            permute_0 = reshape_0.permute([0, 2, 1])
            reshape_1 = permute_0.reshape([1024, 16])
            clone_0 = reshape_1.clone(memory_format=torch.contiguous_format)
            permute_1 = clone_0.permute([1, 0])
            clone_1 = permute_1.clone(memory_format=torch.contiguous_format)

            return clone_0, clone_1

        inps = (torch.rand((8, 2048), device=self.device, dtype=torch.float32),) * 2
        result, code = run_and_compare(
            self,
            func,
            *inps,
            expected_num_triton_kernels=2,
            expected_num_block_pointers=4,
        )
        self.assertTrue("Min" not in code[0])

    @requires_gpu()  # FIXME this test failed on Triton-CPU
    def test_3d_permute_tiling(self):
        """
        Test 3D tiling with permute.
        """

        def foo(x, y, z):
            dims = [0, 2, 1]
            a = x.permute(dims=dims) + y
            b = (z + y).permute(dims=dims)
            return a + b

        inps = (torch.rand((51, 51, 51), device=self.device, dtype=torch.float32),) * 3
        result, (code,) = run_and_compare(
            self,
            foo,
            *inps,
            expected_num_triton_kernels=1,
            expected_num_block_pointers=3,
            config_patches={
                "triton.max_tiles": 3,
                "triton.prefer_nd_tiling": True,
            },
        )

        # Check for 3D tiling
        self._assert_pointwise_ndims(code, 3)

    @torch._dynamo.config.patch({"capture_scalar_outputs": True})
    @parametrize("num_tile_candidates", (1, 2))
    def test_unbacked_size_on_non_contig_dim(self, num_tile_candidates: int):
        # NUM_REPEAT should determine # of candidate_tilings.
        NUM_REPEAT = 2 if num_tile_candidates == 2 else 8

        def foo(x, length):
            unbacked = length.item()
            torch._check_is_size(unbacked)

            repeated = x.repeat(1, unbacked, NUM_REPEAT)
            # permute creates split in middle with unbacked symint is the first range
            # ranges: [33*unbacked, NUM_REPEAT, 64]
            permute120 = repeated.permute([1, 2, 0])
            return permute120.cos()

        inps = (
            torch.rand((64, 33, 1), device=self.device, dtype=torch.float32),
            torch.scalar_tensor(16, device=self.device, dtype=torch.int32),
        )

        with torch._dynamo.config.patch({"capture_scalar_outputs": True}):
            run_and_compare(
                self,
                foo,
                *inps,
                expected_num_triton_kernels=1,
                expected_num_block_pointers=0,
                config_patches={
                    "triton.max_tiles": 3,
                    "triton.prefer_nd_tiling": True,
                },
            )

    # block_ptr advancements should also be deferrered conditional
    # on the associated buffer not being removed
    # in this case the bernoulli operation is fused with the following sum
    # so an output buffer is not needed to store the immediate result of the
    # bernoulli operation
    # TODO: fails for triton CPU "Failed to convert to LLVM IR"
    @test_torchinductor.xfail_if_triton_cpu
    def test_removed_buffers(self):
        from torch.ops import aten

        def fn(a):
            return aten.bernoulli(a).sum() / torch.prod(torch.tensor(a.size()))

        p = 0.3
        result, code = run_and_compare(
            self,
            fn,
            *[torch.ones(200, 200, device=self.device) * p],
            expected_num_triton_kernels=2,
            expected_num_block_pointers=3,
            atol=p * 0.06,
            rtol=0.06,
        )

    def test_pointwise_index_order(self):
        """
        Test the order of indices in pointwise kernels. Expect Z to be the leading dim,
        then Y, then X.
        """

        inps = [
            self._discontiguous_tensor((5, 5, 5), device=self.device) for _ in range(2)
        ]

        result, (triton_code,) = run_and_compare(
            self,
            torch.add,
            *inps,
            expected_num_triton_kernels=1,
            expected_num_block_pointers=3,
            config_patches={
                "triton.max_tiles": 3,
                "triton.prefer_nd_tiling": True,
            },
        )

        # Check the load and store for block pointer strides.
        load_lines, store_lines, index_lines = tuple(
            self._get_lines_containing_substr(triton_code, substr)
            for substr in ("tl.load", "tl.store", "index =")
        )
        self.assertExpectedInline(
            load_lines,
            """\
    tmp0 = tl.load(tl.make_block_ptr(in_ptr0, shape=[5, 5, 5], strides=[100, 10, 1], block_shape=[ZBLOCK, YBLOCK, XBLOCK], order=[2, 1, 0], offsets=[zoffset, yoffset, xoffset]), boundary_check=[0, 1, 2])
    tmp1 = tl.load(tl.make_block_ptr(in_ptr1, shape=[5, 5, 5], strides=[100, 10, 1], block_shape=[ZBLOCK, YBLOCK, XBLOCK], order=[2, 1, 0], offsets=[zoffset, yoffset, xoffset]), boundary_check=[0, 1, 2])""",  # noqa: B950
        )

        self.assertExpectedInline(
            store_lines,
            """    tl.store(tl.make_block_ptr(out_ptr0, shape=[5, 5, 5], strides=[25, 5, 1], block_shape=[ZBLOCK, YBLOCK, XBLOCK], order=[2, 1, 0], offsets=[zoffset, yoffset, xoffset]), tl.broadcast_to(tmp2, [ZBLOCK, YBLOCK, XBLOCK]).to(tl.float32), boundary_check=[0, 1, 2])""",  # noqa: B950
        )

        # Check the indices. These are used for non-block pointers.
        self.assertExpectedInline(
            index_lines,
            """\
    zindex = zoffset + tl.arange(0, ZBLOCK)[:, None, None]
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :, None]
    xindex = xoffset + tl.arange(0, XBLOCK)[None, None, :]""",  # noqa: B950
        )

    def test_expand_clone_broadcast(self):
        """
        Test expand followed by clone. This uses an explicit Triton broadcast.
        """
        base_size = (1, 32)
        expanded_size = (32, 32)

        def foo(x):
            return x.expand(*expanded_size).clone()

        inps = [torch.randn(base_size, device=self.device)]
        result, (triton_code,) = run_and_compare(
            self,
            foo,
            *inps,
            expected_num_triton_kernels=1,
            expected_num_block_pointers=2,
            config_patches={
                "triton.max_tiles": 3,
                "triton.prefer_nd_tiling": True,
            },
        )

        # We should only need one broadcast.
        num_broadcasts = triton_code.count("tl.broadcast_to")
        self.assertEqual(num_broadcasts, 1)

    def test_mul_broadcast_multi_output(self):
        def foo(x, y, z):
            a = x * y
            b = 128.0
            c = a * b
            d = a * z
            e = x * z
            return a, c, d, e

        inps = [
            torch.randn((8, 11, 128), device=self.device),
            torch.randn((128,), device=self.device),
            torch.randn((8, 11, 128), device=self.device),
        ]
        result, (triton_code,) = run_and_compare(
            self,
            foo,
            *inps,
            expected_num_triton_kernels=1,
            expected_num_block_pointers=7,
            config_patches={
                "triton.max_tiles": 3,
                "triton.prefer_nd_tiling": True,
            },
        )

        # Check that the tiling is 2D, even though we allow up to 3D.
        # Singleton splits should be discarded.
        self._assert_pointwise_ndims(triton_code, 2)


@unittest.skipIf(not TRITON_HAS_CPU, "requires triton CPU backend")
@config.patch(cpu_backend="triton")
@config.patch("triton.use_block_ptr", True)
class TritonBlockPointerTestCPU(BlockPointerTestBase):
    device = "cpu"


test_torchinductor.copy_tests(
    CommonTemplate,
    TritonBlockPointerTestCPU,
    "cpu",
    xfail_prop="_expected_failure_triton_cpu",
)


@unittest.skipIf(not HAS_GPU, "requires triton GPU backend")
@config.patch("triton.use_block_ptr", True)
class TritonBlockPointerTestGPU(BlockPointerTestBase):
    device = GPU_TYPE


test_torchinductor.copy_tests(CommonTemplate, TritonBlockPointerTestGPU, GPU_TYPE)

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_GPU or TRITON_HAS_CPU:
        run_tests(needs="filelock")
