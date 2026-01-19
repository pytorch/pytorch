# Owner(s): ["module: inductor"]
# ruff: noqa: F841
import contextlib
import dataclasses
import importlib
import math
import unittest
from collections.abc import Callable
from typing import Any, Optional, Union

import torch
import torch.utils._pytree as pytree
from torch._dynamo.debug_utils import InputReader
from torch._inductor import config
from torch._inductor.choices import InductorChoices
from torch._inductor.codegen.triton import FixedTritonConfig
from torch._inductor.runtime.hints import TRITON_MAX_BLOCK
from torch._inductor.runtime.runtime_utils import get_max_y_grid, is_power_of_2
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch._inductor.virtualized import V
from torch.testing._internal.common_utils import (
    decorateIf,
    instantiate_parametrized_tests,
    parametrize,
    skipIfXpu,
    subtest,
)
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_CUDA_AND_TRITON,
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


# These xfails are due to the current restrictions with the TMA descriptor API.
# see Note: TMA API Restrictions. In some cases TMA descriptors cannot be generated, and so tests
# that assert on the expected number of descriptors (= equivalent block ptrs) will fail
def xfail_if_use_tensor_descriptor(fn):
    fn._expected_failure_use_tensor_descriptor = True
    return fn


TMA_XFAIL = test_torchinductor.TestFailure(GPU_TYPE, is_skip=False)
TMA_TEST_XFAIL = dict.fromkeys(
    (
        "test_pointwise_prefer_nd_tiling_False_full_size1_view_size1_stride1_offset1_require_block_ptr_True",
        "test_pointwise_prefer_nd_tiling_False_full_size4_view_size4_stride4_offset4_require_block_ptr_True",
        "test_pointwise_prefer_nd_tiling_False_full_size6_view_size6_stride6_offset6_require_block_ptr_True",
        "test_pointwise_prefer_nd_tiling_True_full_size1_view_size1_stride1_offset1_require_block_ptr_True",
        "test_pointwise_prefer_nd_tiling_True_full_size4_view_size4_stride4_offset4_require_block_ptr_True",
        "test_pointwise_prefer_nd_tiling_True_full_size6_view_size6_stride6_offset6_require_block_ptr_True",
        "test_reduction_prefer_nd_tiling_False_view_size4_num_block_pointers_3_num_triton_kernels_2",
        "test_reduction_prefer_nd_tiling_False_view_size6_num_block_pointers_3_num_triton_kernels_2",
        "test_reduction_prefer_nd_tiling_True_view_size4_num_block_pointers_3_num_triton_kernels_2",
        "test_reduction_prefer_nd_tiling_True_view_size6_num_block_pointers_3_num_triton_kernels_2",
        "test_2d_reduction_odd_shapes_view_size1_num_block_pointers_3_num_triton_kernels_2_reduction_op1",
        "test_broadcast_prefer_nd_tiling_False_x_size0_y_size0",
        "test_broadcast_prefer_nd_tiling_False_x_size2_y_size2",
        "test_broadcast_prefer_nd_tiling_True_x_size0_y_size0",
        "test_broadcast_prefer_nd_tiling_True_x_size2_y_size2",
    ),
    TMA_XFAIL,
)


class BlockDescriptorTestBase(InductorTestCase):
    block_descriptor_constructor_str = "tl.make_block_ptr"

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

    def _run_and_compare(
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
            tol = {
                t: v for t, v in {"rtol": rtol, "atol": atol}.items() if v is not None
            }
            self.assertTrue(torch.allclose(ref, actual, **tol))

        def count_code(substr: str, expected: Optional[int]):
            count = sum(prog.count(substr) for prog in code)
            if expected is not None:
                self.assertEqual(count, expected)

        # Check the code
        self.assertEqual(len(code), expected_num_programs)
        count_code("@triton.jit", expected_num_triton_kernels)
        count_code(self.block_descriptor_constructor_str, expected_num_block_pointers)

        return result, code


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
            self._run_and_compare(
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
        full_size: tuple[int, ...],
        view_size: tuple[int, ...],
        stride: Optional[tuple[int, ...]],
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
        self._run_and_compare(
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
        self, x_size: tuple[int, ...], y_size: tuple[int, ...], prefer_nd_tiling: bool
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
        self._run_and_compare(
            foo,
            x,
            y,
            expected_num_block_pointers=3,
            config_patches={"triton.prefer_nd_tiling": prefer_nd_tiling},
        )

    def test_broadcast_with_singleton_dims(self):
        # This tests the case when the input / output contains both zero strides
        # and singleton dimensions. In this case the broadcasting dimensions
        # generated for the descriptor need to ignore dimensions that have zero
        # strides with size 1

        # This is a minified repro based on HuggingFaceTB/SmolLM2-135M
        # original issue:
        # store index=x2 + 192*y0 + 64*y1
        # matched block params = BlockParameters(
        #     shape=[3, 4, 1, 1, 64],
        #     block_shape=[((YBLOCK + 3)//4), Min(4, YBLOCK), 1, 1, XBLOCK],
        #     strides=[64, 192, 0, 0, 1],
        #     offsets=[(yoffset//4), ModularIndexing(yoffset, 1, 4), 0, 0, xoffset]
        # )
        # broadcasting_dims=[False, False, True, True, False]
        # broadcast_shape=[((YBLOCK + 3)//4), Min(4, YBLOCK), XBLOCK]
        # error, len(broadcasting_dims) != broadcast_shape
        def forward(expand_4, permute_4, mul_7):
            clone = torch.ops.aten.clone.default(
                expand_4, memory_format=torch.contiguous_format
            )
            expand_4 = None
            view_4 = torch.ops.aten.view.default(clone, [1, 4, 64])
            clone = None
            cos = torch.ops.aten.cos.default(view_4)
            view_4 = None
            mul = torch.ops.aten.mul.Tensor(cos, 1.0)
            cos = None
            unsqueeze_4 = torch.ops.aten.unsqueeze.default(mul, 1)
            mul = None
            mul_6 = torch.ops.aten.mul.Tensor(permute_4, unsqueeze_4)
            permute_4 = unsqueeze_4 = None
            add_3 = torch.ops.aten.add.Tensor(mul_6, mul_7)
            mul_6 = mul_7 = None
            unsqueeze_6 = torch.ops.aten.unsqueeze.default(add_3, 2)
            add_3 = None
            return (unsqueeze_6,)

        def load_args(reader):
            buf0 = reader.storage(storage_hash=None, nbytes=512, device=self.device)
            reader.tensor(buf0, (1, 4, 2, 32), (128, 1, 0, 4), is_leaf=True)  # expand_4
            buf1 = reader.storage(storage_hash=None, nbytes=3072, device=self.device)
            reader.tensor(
                buf1, (1, 3, 4, 64), (768, 64, 192, 1), is_leaf=True
            )  # permute_4
            buf2 = reader.storage(storage_hash=None, nbytes=3072, device=self.device)
            reader.tensor(buf2, (1, 3, 4, 64), is_leaf=True)  # mul_7

        load_args._version = 0

        input_reader = InputReader()
        load_args(input_reader)
        args = input_reader.args
        if self.device == "xpu":
            atol = 1e-7
            rtol = 1e-5
        else:
            atol = None
            rtol = None

        self._run_and_compare(
            forward,
            *args,
            expected_num_block_pointers=4,
            atol=atol,
            rtol=rtol,
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
    def test_expand_broadcast(self, x_size: tuple[int, ...], y_size: tuple[int, ...]):
        """
        When the load and store have different shapes, we should use broadcast.
        """

        def foo(x, y_size):
            return x.expand(y_size).clone()

        def get_input(size: tuple[int, ...]) -> torch.Tensor:
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

        result, (triton_code,) = self._run_and_compare(foo, x, y)

    @xfail_if_use_tensor_descriptor
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
        result, (triton_code,) = self._run_and_compare(
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
                (4, 6 * max_block),
                2,
                2,
            ),  # Multiple of max block. Uses loops.
            ((128, 128), 3, 2),  # Test a large size, with loops.
        ],
    )
    def test_reduction(
        self,
        view_size: tuple[int, ...],
        num_block_pointers: int,
        num_triton_kernels: int,
        prefer_nd_tiling: bool,
    ):
        """
        Tests a reduction kernel.
        """
        if view_size ==(2, 3 * max_block) and torch.version.hip is not None:
            view_size = (4, 6 * max_block)

        if view_size == (128, 128) and torch.version.hip is not None:
            view_size = (256, 256)
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
        result, (code,) = self._run_and_compare(
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
        self,
        view_size: tuple[int, ...],
        num_block_pointers: int,
        num_triton_kernels: int,
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
        result, (code,) = self._run_and_compare(
            foo,
            *inputs,
            expected_num_block_pointers=num_block_pointers,
            expected_num_triton_kernels=num_triton_kernels,
        )

    @xfail_if_use_tensor_descriptor
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
        self._run_and_compare(foo, view, expected_num_block_pointers=2)

    @parametrize(
        "nd_tiling,num_block_pointers",
        [
            subtest(
                (True, 2), decorators=[xfail_if_use_tensor_descriptor]
            ),  # With tiling, the index is affine.
            (False, 1),  # We can't infer that the load is a power of 2.
        ],
    )
    def test_dynamic_shapes_pointwise(self, nd_tiling: bool, num_block_pointers: int):
        """
        Test a pointwise kernel with dynamic shapes.
        """

        view_size = (4, 4)
        view = self._discontiguous_tensor(view_size, self.device)

        self._run_and_compare(
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
            subtest(
                (True, 1), decorators=[xfail_if_use_tensor_descriptor]
            ),  # With tiling, the index is affine.
            (False, 0),  # We can't infer that the load is a power of 2.
        ],
    )
    def test_dynamic_shapes_reduction(self, with_tiling: bool, num_block_pointers: int):
        """
        Test a reduction kernel with dynamic shapes.
        """

        view_size = (4, 4)
        view = self._discontiguous_tensor(view_size, self.device)

        self._run_and_compare(
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
        self._run_and_compare(
            x, compile_kwargs={"dynamic": True}, expected_num_block_pointers=2
        )

    @decorateIf(
        xfail_if_use_tensor_descriptor,
        lambda param_kwargs: not (
            param_kwargs["num_block_pointers"] == 3 and param_kwargs["num_tiles"] == 1
        ),
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
        full_size: tuple[int, ...],
        view_size: tuple[int, ...],
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
        result, code = self._run_and_compare(
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

    @xfail_if_use_tensor_descriptor
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
        view_size: tuple[int, ...],
        num_block_pointers: int,
        num_triton_kernels: int,
        reduction_op: Callable,
    ):
        """
        Tests 2D reduction kernels. These arise from "odd" shapes which are not
        expressible with a 1D block pointer.
        """
        if reduction_op == torch.sum and torch.version.hip is not None:
            view_size = (513, 513) if view_size == (129, 129) else view_size
        view = self._discontiguous_tensor(view_size, self.device)

        # Expect at least 1 block pointer for the input.
        # Add 2 more if we generate 2 kernels.
        result, (code,) = self._run_and_compare(
            reduction_op,
            view,
            expected_num_block_pointers=num_block_pointers,
            expected_num_triton_kernels=num_triton_kernels,
            config_patches=tiled_reduction_config,
        )

        # Check the code for multiple Rn_BLOCK's
        self._assert_reduction_ndims(code, 2)

    @parametrize(
        "size,expected_num_block_pointers,expected_num_triton_kernels,expect_fallback",
        [
            ((8, 8), 1, 1, True),  # Persistent Welford fallback
            subtest(
                ((128, 128), 7, 2, False), decorators=[xfail_if_use_tensor_descriptor]
            ),  # Looped Welford reduction
        ],
    )
    def test_2d_welford_reduction(
        self,
        size: tuple[int, ...],
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
        if torch.version.hip is not None and expected_num_triton_kernels == 2:
            size = (256, 256) 
        view = self._discontiguous_tensor(size, self.device)

        # We expect many block pointers for this one.
        result, (code,) = self._run_and_compare(
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
        result, (code,) = self._run_and_compare(
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

        result, (code,) = self._run_and_compare(
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

        result, (code,) = self._run_and_compare(
            foo,
            view,
            expected_num_block_pointers=5,
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

    @xfail_if_use_tensor_descriptor
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
        result, (code,) = self._run_and_compare(
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
        result, (code,) = self._run_and_compare(
            foo,
            arg0,
            arg1,
            config_patches=tiled_reduction_config,
        )

        # Check the code for multiple Rn_BLOCK's
        self._assert_reduction_ndims(code, 2)

    @parametrize(
        "tile_reductions",
        [False, subtest(True, decorators=[xfail_if_use_tensor_descriptor])],
    )
    def test_enable_tiled_reductions(self, tile_reductions: bool):
        """
        Tests enabling and disabling tiled reductions.
        """
        view = self._discontiguous_tensor((9, 11), self.device)

        # If tiled, we expect 1 block pointer for the input.
        result, (code,) = self._run_and_compare(
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
        result, code = self._run_and_compare(
            func,
            *inps,
            expected_num_triton_kernels=2,
            expected_num_block_pointers=4,
        )
        self.assertTrue("Min" not in code[0])

    @xfail_if_use_tensor_descriptor
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
        result, (code,) = self._run_and_compare(
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
            self._run_and_compare(
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
    # Disable split_reductions on this test for now due to the interaction with LOAF
    @config.patch(split_reductions=False)
    def test_removed_buffers(self):
        from torch.ops import aten

        def fn(a):
            return aten.bernoulli(a).sum() / torch.prod(torch.tensor(a.size()))

        p = 0.3
        result, code = self._run_and_compare(
            fn,
            *[torch.ones(200, 200, device=self.device) * p],
            expected_num_triton_kernels=1,
            expected_num_block_pointers=1,
            atol=p * 0.06,
            rtol=0.06,
        )

    @xfail_if_use_tensor_descriptor
    def test_pointwise_index_order(self):
        """
        Test the order of indices in pointwise kernels. Expect Z to be the leading dim,
        then Y, then X.
        """

        inps = [
            self._discontiguous_tensor((5, 5, 5), device=self.device) for _ in range(2)
        ]

        result, (triton_code,) = self._run_and_compare(
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
        result, (triton_code,) = self._run_and_compare(
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
        result, (triton_code,) = self._run_and_compare(
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

    # Integration test to ensure that matched dims & strides from match_mod_div_expr
    # are unsigned and signed integers respectively. This test case has the following
    # index:=(ModularIndexing(xindex, 4, 4)) + 4*(ModularIndexing(xindex, 32, 2))
    # and the match below is a candidate that is invalid:
    # match={
    #   dim_mod4_: 32, dim_mod3_: 2, stride_mod3_: 4, dim_mod2_: 1/16,
    #   dim_mod1_: 4, stride_mod1_: 1, stride_mod4_: 0, stride_mod2_: 0, stride_mod0_: 0
    # }
    # This is now fixed by ensuring that that wild symbols only match integers
    @skipIfXpu(
        msg="Triton issue exposed by new driver, will be resolved after next triton update."
    )
    def test_ensure_integral_dims_and_strides(self):
        def model(data, *args):
            return torch.nn.functional.unfold(data, *args)

        data = torch.zeros(
            [2, 3, 5, 5], dtype=torch.float16, requires_grad=True, device=self.device
        )
        args = [2, 1, 0, 1]
        self._run_and_compare(
            model,
            data,
            *args,
            expected_num_triton_kernels=2,
            expected_num_block_pointers=4,
            compile_kwargs={"fullgraph": True},
        )

    # Integration test to test block analysis with index expressions using
    # negative strides.
    # This test case has the following index:
    # index_relative_to_xyr_index = -256*((xindex//64)) - (ModularIndexing(xindex, 1, 8))
    #    - 16*(ModularIndexing(xindex, 8, 8)) + 1911
    # subexpr = -256*((xindex//64)) - (ModularIndexing(xindex, 1, 8)) - 16*(ModularIndexing(xindex, 8, 8))
    # Block analysis should produce the following:
    # BlockParameters(
    #   shape=[8, 8, 8],
    #   block_shape=[((XBLOCK + 63)//64), Min(8, ((XBLOCK + 7)//8)), Min(8, XBLOCK) ],
    #   strides=[-256, -16, -1],
    #   offsets=[(xoffset//64), ModularIndexing(xoffset, 8, 8), ModularIndexing(xoffset, 1, 8)]
    #   )
    # constant_offset = 1911
    @xfail_if_use_tensor_descriptor
    def test_negative_strides(self):
        def model(x, y):
            # Slice in reverse order via a negative stride
            return torch.flip(x, [0, 1, 2]) + y

        x, y = (
            self._discontiguous_tensor((8, 8, 8), device=self.device) for _ in range(2)
        )
        self._run_and_compare(
            model,
            x,
            y,
            expected_num_triton_kernels=1,
            expected_num_block_pointers=3,
        )

    @config.patch("triton.prefer_nd_tiling", True)
    @config.patch("triton.max_tiles", 3)
    @parametrize(
        "block_multiple, ynumel_exceed_ygrid_size, include_z",
        [
            # No boundary check in all dimensions
            [True, False, True],
            # No xdim boundary check, ydim is checked since > max_ygrid
            # z dim can be used since its not included
            [True, True, False],
            # Boundary check in all dimensions
            # skip triton_cpu very slow test > 1000s
            subtest(
                [False, False, True], decorators=[test_torchinductor.skip_if_triton_cpu]
            ),
        ],
    )
    @xfail_if_use_tensor_descriptor
    def test_boundary_check(self, block_multiple, ynumel_exceed_ygrid_size, include_z):
        @dataclasses.dataclass
        class InputShape:
            x: int
            y: int
            z: Optional[int] = None

            def to_list(self):
                out = [self.y, self.x]
                if self.z is not None:
                    out.insert(0, self.z)
                return out

        BLOCK_SIZE = 8
        DIM_SIZE = BLOCK_SIZE if block_multiple else BLOCK_SIZE + 1
        shape = InputShape(DIM_SIZE, DIM_SIZE, DIM_SIZE if include_z else None)
        if ynumel_exceed_ygrid_size:
            shape.y = math.ceil(get_max_y_grid()) * shape.y + shape.y

        # Use fixed block sizes to avoid having to generate very large input tensors
        class FixedBlockSizeChoices(InductorChoices):
            def triton_kernel_kwargs(self, kernel_cls, features, groups, kernel_kwargs):
                block_sizes = {
                    f"{prefix.upper()}BLOCK": BLOCK_SIZE
                    for prefix, size in dataclasses.asdict(shape).items()
                    if size is not None
                }
                kernel_kwargs["fixed_config"] = FixedTritonConfig(block_sizes)
                return kernel_kwargs

        a = self._discontiguous_tensor(shape.to_list(), device=self.device)
        b_shape = shape.to_list()
        b_shape[-1] = 1
        b = self._discontiguous_tensor(b_shape, device=self.device)

        def func(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a + b

        with V.set_choices_handler(FixedBlockSizeChoices()):
            result, code = self._run_and_compare(
                func,
                a,
                b,
                expected_num_triton_kernels=1,
                expected_num_block_pointers=3,
            )

            code = code[0]
            if block_multiple:
                if ynumel_exceed_ygrid_size:
                    self.assertIn(
                        "yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK",
                        code,
                    )
                    # Only the y dimension should be boundary checked
                    # a, b, and output
                    self.assertEqual(code.count("boundary_check=[0]"), 3)
                else:
                    # No boundary checking
                    self.assertNotIn("boundary_check", code)
            else:
                # Loading a
                self.assertTrue("boundary_check=[0, 1, 2]" in code)
                # Loading b
                self.assertTrue("boundary_check=[0, 1]" in code)


@unittest.skipIf(not TRITON_HAS_CPU, "requires triton CPU backend")
@config.patch(cpu_backend="triton")
@config.patch("triton.use_block_ptr", True)
class TritonBlockPointerTestCPU(BlockDescriptorTestBase):
    device = "cpu"


test_torchinductor.copy_tests(
    CommonTemplate,
    TritonBlockPointerTestCPU,
    "cpu",
    xfail_prop="_expected_failure_triton_cpu",
)


@unittest.skipIf(not HAS_GPU, "requires triton GPU backend")
@config.patch("triton.use_block_ptr", True)
class TritonBlockPointerTestGPU(BlockDescriptorTestBase):
    device = GPU_TYPE


test_torchinductor.copy_tests(CommonTemplate, TritonBlockPointerTestGPU, GPU_TYPE)


@unittest.skipIf(
    not (HAS_CUDA_AND_TRITON and torch.cuda.get_device_capability()[0] >= 9)
    or torch.version.hip,
    "Requires Triton CUDA backend and CUDA compute capability >= 9.0. Not supported on ROCm",
    # ROCm triton doesn't support/generate "tl.make_tensor_descriptor" which is exactly what this unit test is about
)
@config.patch({"triton.use_tensor_descriptor": True, "assume_aligned_inputs": True})
@instantiate_parametrized_tests
class TritonTensorDescriptorTestCUDA(BlockDescriptorTestBase):
    block_descriptor_constructor_str = "tl.make_tensor_descriptor"
    device = GPU_TYPE

    @config.patch({"triton.transpose_discontiguous_tensor_descriptor": True})
    @parametrize(
        "view_size,permute_order,num_tensor_descriptors,expect_transpose",
        [
            ((128,), (0,), 3, False),
            ((128, 128), (0, 1), 3, False),
            ((128, 64), (1, 0), 3, True),
            ((256, 32, 16), (2, 0, 1), 3, True),
            ((16, 32, 256), (2, 0, 1), 3, True),
        ],
    )
    def test_match_with_transpose(
        self,
        view_size: tuple[int],
        permute_order: tuple[int],
        num_tensor_descriptors: int,
        expect_transpose: bool,
    ):
        a = self._discontiguous_tensor(view_size, self.device)
        pre_permute_size = [1] * len(view_size)
        for i, value in zip(permute_order, view_size):
            pre_permute_size[i] = value
        b = self._discontiguous_tensor(pre_permute_size, self.device)
        b = b.permute(permute_order)

        def fn(a, b):
            return a * b

        result, (code,) = self._run_and_compare(
            fn,
            a,
            b,
            expected_num_block_pointers=num_tensor_descriptors,
            expected_num_triton_kernels=1,
            config_patches=tiled_reduction_config,
        )

        transpose_count = code.count("tl.trans")
        self.assertEqual(transpose_count, 1 if expect_transpose else 0)

    def test_rms_norm_backward_does_not_crash_with_tma(self):
        B, S, D = 1, 1024, 40096
        with torch.device(self.device):
            x = torch.randn(B, S, D, dtype=torch.bfloat16, requires_grad=True)
            w = torch.randn(D, dtype=torch.bfloat16, requires_grad=True)

        def f(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            y = torch.rms_norm(x, (D,), w, 1e-5)
            return (y * 0.1).sum()

        compiled_f = torch.compile(f, backend="inductor", fullgraph=True)
        loss = compiled_f(x, w)
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(w.grad)


test_torchinductor.copy_tests(
    CommonTemplate,
    TritonTensorDescriptorTestCUDA,
    GPU_TYPE,
    xfail_prop="_expected_failure_use_tensor_descriptor",
    test_failures=TMA_TEST_XFAIL,
)


class TestTilingExtra(InductorTestCase):
    @requires_gpu()
    def test_tiling_split_valid(self):
        import torch.nn.functional as F

        class GraphModule(torch.nn.Module):
            def forward(
                self,
                sub_dense0_w,  # f16[64, 80]
                sub_dense0_b,  # f16[64]
                s25,  # Sym(s25) - batch size
                s70,  # Sym(s70) - sequence length
                input_feat,  # f16[s25, s70, 80]
                sub_conv0_w,  # f16[64, 64, 5]
                sub_conv0_b,  # f16[64]
                sub_conv1_w,  # f16[32, 64, 5]
                sub_conv1_b,  # f16[32]
                sub_dense1_w,  # f16[64, 32]
                sub_dense1_b,  # f16[64]
                rot_inv_freq,  # f32[8]
                rot_attn_scale,  # f64[] cpu
                l0_norm_ff1_w,  # f16[64]
                l0_ff1_lin1_w,  # f16[256, 64]
                l0_ff1_lin2_w,  # f16[64, 256]
                l0_ff_res0,  # f64[] cpu
                l0_ff_res1,  # f64[] cpu
                l0_norm_attn_w,  # f16[64]
                l0_q_w,  # f16[64, 64]
                l0_k_w,  # f16[64, 64]
                l0_v_w,  # f16[64, 64]
                l0_o_w,  # f16[64, 64]
                l0_norm_conv_w,  # f16[64]
                l0_pw_conv1_w,  # f16[128, 64, 1]
                l0_dw_conv_w,  # f16[64, 1, 8]
                l0_bn_mean,  # f16[64]
                l0_bn_var,  # f16[64]
                l0_bn_w,  # f16[64]
                l0_bn_b,  # f16[64]
                l0_pw_conv2_w,  # f16[64, 64, 1]
                l0_conv_res0,  # f64[] cpu
                l0_conv_res1,  # f64[] cpu
                l0_norm_ff2_w,  # f16[64]
                l0_ff2_lin1_w,  # f16[256, 64]
                l0_ff2_lin2_w,  # f16[64, 256]
                l0_norm_out_w,  # f16[64]
                l1_norm_ff1_w,  # f16[64]
                l1_ff1_lin1_w,  # f16[256, 64]
                l1_ff1_lin2_w,  # f16[64, 256]
                l1_norm_attn_w,  # f16[64]
                l1_q_w,  # f16[64, 64]
                l1_k_w,  # f16[64, 64]
                l1_v_w,  # f16[64, 64]
                l1_o_w,  # f16[64, 64]
                l1_norm_conv_w,  # f16[64]
                l1_pw_conv1_w,  # f16[128, 64, 1]
                l1_dw_conv_w,  # f16[64, 1, 8]
                l1_bn_mean,  # f16[64]
                l1_bn_var,  # f16[64]
                l1_bn_w,  # f16[64]
                l1_bn_b,  # f16[64]
                l1_pw_conv2_w,  # f16[64, 64, 1]
                l1_norm_ff2_w,  # f16[64]
                l1_ff2_lin1_w,  # f16[256, 64]
                l1_ff2_lin2_w,  # f16[64, 256]
                l1_norm_out_w,  # f16[64]
                out_norm_w,  # f16[64]
            ):
                # Subsampler: dense_0 + relu
                linear = torch._C._nn.linear(input_feat, sub_dense0_w, sub_dense0_b)
                hidden_states = F.relu(linear, inplace=False)

                # Transpose for conv
                hidden_states_1 = hidden_states.transpose(1, 2)

                # Conv_0 with stride=2
                conv1d = torch.conv1d(
                    hidden_states_1, sub_conv0_w, sub_conv0_b, (2,), (0,), (1,), 1
                )
                hidden_states_2 = F.relu(conv1d, inplace=False)

                # Conv_1 with stride=2
                conv1d_1 = torch.conv1d(
                    hidden_states_2, sub_conv1_w, sub_conv1_b, (2,), (0,), (1,), 1
                )
                hidden_states_3 = F.relu(conv1d_1, inplace=False)

                # Transpose back
                hidden_states_4 = hidden_states_3.transpose(1, 2)

                # Dense_1
                hidden_states_5 = torch._C._nn.linear(
                    hidden_states_4, sub_dense1_w, sub_dense1_b
                )

                # Compute output sequence length: ((s70 - 5) // 4) - 1
                # Note: In the dynamo graph, torch.sym_sum is used, but we use direct arithmetic here
                sym_sum = s70 - 5
                floordiv = sym_sum // 4
                sym_sum_1 = floordiv - 1

                # Create position ids
                arange = torch.arange(sym_sum_1, device=GPU_TYPE)
                unsqueeze = arange.unsqueeze(0)

                # Rotary embedding computation
                getitem_3 = rot_inv_freq[(None, slice(None, None, None), None)]
                float_1 = getitem_3.float()
                expand = float_1.expand(1, -1, 1)
                inv_freq_expanded = expand.to(torch.device(GPU_TYPE, index=0))

                getitem_6 = unsqueeze[
                    (slice(None, None, None), None, slice(None, None, None))
                ]
                position_ids_expanded = getitem_6.float()

                # Rotary embedding frequency computation (autocast removed for tracing)
                float_3 = inv_freq_expanded.float()
                float_4 = position_ids_expanded.float()
                matmul = float_3 @ float_4
                freqs = matmul.transpose(1, 2)

                emb = torch.cat((freqs, freqs), dim=-1)

                cos = emb.cos()
                item = rot_attn_scale.item()
                cos_1 = cos * item

                sin = emb.sin()
                sin_1 = sin * item

                cos_2 = cos_1.to(dtype=torch.float16)
                sin_2 = sin_1.to(dtype=torch.float16)

                # Dropout (no-op in eval mode)
                hidden_states_6 = F.dropout(hidden_states_5, p=0.1, training=False)
                cos_3 = F.dropout(cos_2, p=0.0, training=False)
                sin_3 = F.dropout(sin_2, p=0.0, training=False)

                # Create attention mask
                cache_position = torch.arange(
                    sym_sum_1, device=GPU_TYPE, dtype=torch.int64
                )
                arange_4 = torch.arange(sym_sum_1, device=GPU_TYPE)

                q_indices = cache_position[(None, None, slice(None, None, None), None)]
                attention_mask = q_indices >= 0
                attention_mask_1 = attention_mask.expand(s25, -1, sym_sum_1, sym_sum_1)

                # ============ LAYER 0 ============
                # Feed forward 1
                layer_norm = F.layer_norm(
                    hidden_states_6, (64,), l0_norm_ff1_w, None, 1e-06
                )
                linear_2 = torch._C._nn.linear(layer_norm, l0_ff1_lin1_w, None)
                hidden_states_7 = F.silu(linear_2)
                hidden_states_8 = F.dropout(hidden_states_7, p=0.1, training=False)
                hidden_states_9 = torch._C._nn.linear(
                    hidden_states_8, l0_ff1_lin2_w, None
                )

                # Residual connection with weights
                item_5 = l0_ff_res0.item()
                mul_2 = item_5 * hidden_states_6
                item_6 = l0_ff_res1.item()
                mul_3 = item_6 * hidden_states_9
                hidden_states_10 = mul_2 + mul_3

                # Self attention
                normalized_hidden_states = F.layer_norm(
                    hidden_states_10, (64,), l0_norm_attn_w, None, 1e-06
                )

                linear_4 = torch._C._nn.linear(normalized_hidden_states, l0_q_w, None)
                view = linear_4.view((s25, sym_sum_1, -1, 16))
                query_states = view.transpose(1, 2)

                linear_5 = torch._C._nn.linear(normalized_hidden_states, l0_k_w, None)
                view_1 = linear_5.view((s25, sym_sum_1, -1, 16))
                key_states = view_1.transpose(1, 2)

                linear_6 = torch._C._nn.linear(normalized_hidden_states, l0_v_w, None)
                view_2 = linear_6.view((s25, sym_sum_1, -1, 16))
                value_states = view_2.transpose(1, 2)

                # Apply rotary embeddings
                cos_4 = cos_3.unsqueeze(1)
                sin_4 = sin_3.unsqueeze(1)

                mul_4 = query_states * cos_4
                x1 = query_states[(Ellipsis, slice(None, 8, None))]
                x2 = query_states[(Ellipsis, slice(8, None, None))]
                neg = -x2
                cat_1 = torch.cat((neg, x1), dim=-1)
                mul_5 = cat_1 * sin_4
                q_embed = mul_4 + mul_5

                mul_6 = key_states * cos_4
                x1_1 = key_states[(Ellipsis, slice(None, 8, None))]
                x2_1 = key_states[(Ellipsis, slice(8, None, None))]
                neg_1 = -x2_1
                cat_2 = torch.cat((neg_1, x1_1), dim=-1)
                mul_7 = cat_2 * sin_4
                k_embed = mul_6 + mul_7

                # SDPA
                attn_output = torch._C._nn.scaled_dot_product_attention(
                    q_embed,
                    k_embed,
                    value_states,
                    attn_mask=attention_mask_1,
                    dropout_p=0.0,
                    scale=0.25,
                    is_causal=False,
                )

                transpose_6 = attn_output.transpose(1, 2)
                attn_output_1 = transpose_6.contiguous()
                reshape = attn_output_1.reshape(s25, sym_sum_1, -1)
                attn_output_2 = reshape.contiguous()
                attn_output_3 = torch._C._nn.linear(attn_output_2, l0_o_w, None)

                hidden_states_11 = hidden_states_10 + attn_output_3

                # Convolution module
                layer_norm_2 = F.layer_norm(
                    hidden_states_11, (64,), l0_norm_conv_w, None, 1e-06
                )
                hidden_states_12 = layer_norm_2.transpose(1, 2)
                hidden_states_13 = torch.conv1d(
                    hidden_states_12, l0_pw_conv1_w, None, (1,), (0,), (1,), 1
                )
                hidden_states_14 = F.glu(hidden_states_13, dim=1)

                invert = ~attention_mask_1
                all_masked_rows = torch.all(invert, dim=2)
                hidden_states_15 = hidden_states_14.masked_fill(all_masked_rows, 0.0)

                hidden_states_16 = torch.conv1d(
                    hidden_states_15, l0_dw_conv_w, None, (1,), "same", (1,), 64
                )
                hidden_states_17 = F.batch_norm(
                    hidden_states_16,
                    l0_bn_mean,
                    l0_bn_var,
                    l0_bn_w,
                    l0_bn_b,
                    False,
                    0.01,
                    1e-05,
                )
                hidden_states_18 = F.silu(hidden_states_17)
                hidden_states_19 = torch.conv1d(
                    hidden_states_18, l0_pw_conv2_w, None, (1,), (0,), (1,), 1
                )
                conv_output = hidden_states_19.transpose(1, 2)

                # Conv residual
                item_12 = l0_conv_res0.item()
                item_13 = l0_conv_res1.item()
                mul_8 = item_12 * hidden_states_11
                mul_9 = item_13 * conv_output
                hidden_states_20 = mul_8 + mul_9

                # Feed forward 2
                layer_norm_3 = F.layer_norm(
                    hidden_states_20, (64,), l0_norm_ff2_w, None, 1e-06
                )
                linear_8 = torch._C._nn.linear(layer_norm_3, l0_ff2_lin1_w, None)
                hidden_states_21 = F.silu(linear_8)
                hidden_states_22 = F.dropout(hidden_states_21, p=0.1, training=False)
                hidden_states_23 = torch._C._nn.linear(
                    hidden_states_22, l0_ff2_lin2_w, None
                )

                mul_10 = item_5 * hidden_states_20
                mul_11 = item_6 * hidden_states_23
                hidden_states_24 = mul_10 + mul_11

                hidden_states_25 = F.layer_norm(
                    hidden_states_24, (64,), l0_norm_out_w, None, 1e-06
                )

                # ============ LAYER 1 ============
                # Feed forward 1
                layer_norm_5 = F.layer_norm(
                    hidden_states_25, (64,), l1_norm_ff1_w, None, 1e-06
                )
                linear_10 = torch._C._nn.linear(layer_norm_5, l1_ff1_lin1_w, None)
                hidden_states_26 = F.silu(linear_10)
                hidden_states_27 = F.dropout(hidden_states_26, p=0.1, training=False)
                hidden_states_28 = torch._C._nn.linear(
                    hidden_states_27, l1_ff1_lin2_w, None
                )

                mul_12 = item_5 * hidden_states_25
                mul_13 = item_6 * hidden_states_28
                hidden_states_29 = mul_12 + mul_13

                # Self attention
                normalized_hidden_states_1 = F.layer_norm(
                    hidden_states_29, (64,), l1_norm_attn_w, None, 1e-06
                )

                linear_12 = torch._C._nn.linear(
                    normalized_hidden_states_1, l1_q_w, None
                )
                view_3 = linear_12.view((s25, sym_sum_1, -1, 16))
                query_states_1 = view_3.transpose(1, 2)

                linear_13 = torch._C._nn.linear(
                    normalized_hidden_states_1, l1_k_w, None
                )
                view_4 = linear_13.view((s25, sym_sum_1, -1, 16))
                key_states_1 = view_4.transpose(1, 2)

                linear_14 = torch._C._nn.linear(
                    normalized_hidden_states_1, l1_v_w, None
                )
                view_5 = linear_14.view((s25, sym_sum_1, -1, 16))
                value_states_1 = view_5.transpose(1, 2)

                # Apply rotary embeddings
                cos_5 = cos_3.unsqueeze(1)
                sin_5 = sin_3.unsqueeze(1)

                mul_14 = query_states_1 * cos_5
                x1_2 = query_states_1[(Ellipsis, slice(None, 8, None))]
                x2_2 = query_states_1[(Ellipsis, slice(8, None, None))]
                neg_2 = -x2_2
                cat_3 = torch.cat((neg_2, x1_2), dim=-1)
                mul_15 = cat_3 * sin_5
                q_embed_1 = mul_14 + mul_15

                mul_16 = key_states_1 * cos_5
                x1_3 = key_states_1[(Ellipsis, slice(None, 8, None))]
                x2_3 = key_states_1[(Ellipsis, slice(8, None, None))]
                neg_3 = -x2_3
                cat_4 = torch.cat((neg_3, x1_3), dim=-1)
                mul_17 = cat_4 * sin_5
                k_embed_1 = mul_16 + mul_17

                # SDPA
                attn_output_4 = torch._C._nn.scaled_dot_product_attention(
                    q_embed_1,
                    k_embed_1,
                    value_states_1,
                    attn_mask=attention_mask_1,
                    dropout_p=0.0,
                    scale=0.25,
                    is_causal=False,
                )

                transpose_12 = attn_output_4.transpose(1, 2)
                attn_output_5 = transpose_12.contiguous()
                reshape_1 = attn_output_5.reshape(s25, sym_sum_1, -1)
                attn_output_6 = reshape_1.contiguous()
                attn_output_7 = torch._C._nn.linear(attn_output_6, l1_o_w, None)

                hidden_states_30 = hidden_states_29 + attn_output_7

                # Convolution module
                layer_norm_7 = F.layer_norm(
                    hidden_states_30, (64,), l1_norm_conv_w, None, 1e-06
                )
                hidden_states_31 = layer_norm_7.transpose(1, 2)
                hidden_states_32 = torch.conv1d(
                    hidden_states_31, l1_pw_conv1_w, None, (1,), (0,), (1,), 1
                )
                hidden_states_33 = F.glu(hidden_states_32, dim=1)

                invert_1 = ~attention_mask_1
                all_masked_rows_1 = torch.all(invert_1, dim=2)
                hidden_states_34 = hidden_states_33.masked_fill(all_masked_rows_1, 0.0)

                hidden_states_35 = torch.conv1d(
                    hidden_states_34, l1_dw_conv_w, None, (1,), "same", (1,), 64
                )
                hidden_states_36 = F.batch_norm(
                    hidden_states_35,
                    l1_bn_mean,
                    l1_bn_var,
                    l1_bn_w,
                    l1_bn_b,
                    False,
                    0.01,
                    1e-05,
                )
                hidden_states_37 = F.silu(hidden_states_36)
                hidden_states_38 = torch.conv1d(
                    hidden_states_37, l1_pw_conv2_w, None, (1,), (0,), (1,), 1
                )
                conv_output_1 = hidden_states_38.transpose(1, 2)

                # Conv residual
                mul_18 = item_12 * hidden_states_30
                mul_19 = item_13 * conv_output_1
                hidden_states_39 = mul_18 + mul_19

                # Feed forward 2
                layer_norm_8 = F.layer_norm(
                    hidden_states_39, (64,), l1_norm_ff2_w, None, 1e-06
                )
                linear_16 = torch._C._nn.linear(layer_norm_8, l1_ff2_lin1_w, None)
                hidden_states_40 = F.silu(linear_16)
                hidden_states_41 = F.dropout(hidden_states_40, p=0.1, training=False)
                hidden_states_42 = torch._C._nn.linear(
                    hidden_states_41, l1_ff2_lin2_w, None
                )

                mul_20 = item_5 * hidden_states_39
                mul_21 = item_6 * hidden_states_42
                hidden_states_43 = mul_20 + mul_21

                hidden_states_44 = F.layer_norm(
                    hidden_states_43, (64,), l1_norm_out_w, None, 1e-06
                )

                # Final output norm
                hidden_states_45 = F.layer_norm(
                    hidden_states_44, (64,), out_norm_w, None, 1e-06
                )

                return (hidden_states_45,)

        def create_parameters(device="cuda", dtype=torch.float16):
            """Create all the parameters needed by the GraphModule."""
            params = {}

            # Subsampler parameters
            params["sub_dense0_w"] = torch.randn(64, 80, device=device, dtype=dtype)
            params["sub_dense0_b"] = torch.randn(64, device=device, dtype=dtype)
            params["sub_conv0_w"] = torch.randn(64, 64, 5, device=device, dtype=dtype)
            params["sub_conv0_b"] = torch.randn(64, device=device, dtype=dtype)
            params["sub_conv1_w"] = torch.randn(32, 64, 5, device=device, dtype=dtype)
            params["sub_conv1_b"] = torch.randn(32, device=device, dtype=dtype)
            params["sub_dense1_w"] = torch.randn(64, 32, device=device, dtype=dtype)
            params["sub_dense1_b"] = torch.randn(64, device=device, dtype=dtype)

            # Rotary embedding
            params["rot_inv_freq"] = torch.randn(8, device=device, dtype=torch.float32)
            params["rot_attn_scale"] = torch.tensor(
                1.0, device="cpu", dtype=torch.float64
            )

            # Layer 0 parameters
            params["l0_norm_ff1_w"] = torch.randn(64, device=device, dtype=dtype)
            params["l0_ff1_lin1_w"] = torch.randn(256, 64, device=device, dtype=dtype)
            params["l0_ff1_lin2_w"] = torch.randn(64, 256, device=device, dtype=dtype)
            params["l0_ff_res0"] = torch.tensor(0.5, device="cpu", dtype=torch.float64)
            params["l0_ff_res1"] = torch.tensor(0.5, device="cpu", dtype=torch.float64)
            params["l0_norm_attn_w"] = torch.randn(64, device=device, dtype=dtype)
            params["l0_q_w"] = torch.randn(64, 64, device=device, dtype=dtype)
            params["l0_k_w"] = torch.randn(64, 64, device=device, dtype=dtype)
            params["l0_v_w"] = torch.randn(64, 64, device=device, dtype=dtype)
            params["l0_o_w"] = torch.randn(64, 64, device=device, dtype=dtype)
            params["l0_norm_conv_w"] = torch.randn(64, device=device, dtype=dtype)
            params["l0_pw_conv1_w"] = torch.randn(
                128, 64, 1, device=device, dtype=dtype
            )
            params["l0_dw_conv_w"] = torch.randn(64, 1, 8, device=device, dtype=dtype)
            params["l0_bn_mean"] = torch.randn(64, device=device, dtype=dtype)
            params["l0_bn_var"] = (
                torch.abs(torch.randn(64, device=device, dtype=dtype)) + 0.1
            )
            params["l0_bn_w"] = torch.randn(64, device=device, dtype=dtype)
            params["l0_bn_b"] = torch.randn(64, device=device, dtype=dtype)
            params["l0_pw_conv2_w"] = torch.randn(64, 64, 1, device=device, dtype=dtype)
            params["l0_conv_res0"] = torch.tensor(
                0.5, device="cpu", dtype=torch.float64
            )
            params["l0_conv_res1"] = torch.tensor(
                0.5, device="cpu", dtype=torch.float64
            )
            params["l0_norm_ff2_w"] = torch.randn(64, device=device, dtype=dtype)
            params["l0_ff2_lin1_w"] = torch.randn(256, 64, device=device, dtype=dtype)
            params["l0_ff2_lin2_w"] = torch.randn(64, 256, device=device, dtype=dtype)
            params["l0_norm_out_w"] = torch.randn(64, device=device, dtype=dtype)

            # Layer 1 parameters
            params["l1_norm_ff1_w"] = torch.randn(64, device=device, dtype=dtype)
            params["l1_ff1_lin1_w"] = torch.randn(256, 64, device=device, dtype=dtype)
            params["l1_ff1_lin2_w"] = torch.randn(64, 256, device=device, dtype=dtype)
            params["l1_norm_attn_w"] = torch.randn(64, device=device, dtype=dtype)
            params["l1_q_w"] = torch.randn(64, 64, device=device, dtype=dtype)
            params["l1_k_w"] = torch.randn(64, 64, device=device, dtype=dtype)
            params["l1_v_w"] = torch.randn(64, 64, device=device, dtype=dtype)
            params["l1_o_w"] = torch.randn(64, 64, device=device, dtype=dtype)
            params["l1_norm_conv_w"] = torch.randn(64, device=device, dtype=dtype)
            params["l1_pw_conv1_w"] = torch.randn(
                128, 64, 1, device=device, dtype=dtype
            )
            params["l1_dw_conv_w"] = torch.randn(64, 1, 8, device=device, dtype=dtype)
            params["l1_bn_mean"] = torch.randn(64, device=device, dtype=dtype)
            params["l1_bn_var"] = (
                torch.abs(torch.randn(64, device=device, dtype=dtype)) + 0.1
            )
            params["l1_bn_w"] = torch.randn(64, device=device, dtype=dtype)
            params["l1_bn_b"] = torch.randn(64, device=device, dtype=dtype)
            params["l1_pw_conv2_w"] = torch.randn(64, 64, 1, device=device, dtype=dtype)
            params["l1_norm_ff2_w"] = torch.randn(64, device=device, dtype=dtype)
            params["l1_ff2_lin1_w"] = torch.randn(256, 64, device=device, dtype=dtype)
            params["l1_ff2_lin2_w"] = torch.randn(64, 256, device=device, dtype=dtype)
            params["l1_norm_out_w"] = torch.randn(64, device=device, dtype=dtype)

            # Output norm
            params["out_norm_w"] = torch.randn(64, device=device, dtype=dtype)

            return params

        torch.manual_seed(42)

        device = GPU_TYPE
        dtype = torch.float16

        # Create model and parameters
        model = GraphModule().eval()
        params = create_parameters(device, dtype)

        # Create example input
        batch_size = 13
        seq_len = 1024
        input_features = torch.randn(
            batch_size, seq_len, 80, device=device, dtype=dtype
        )
        compiled_model = torch.compile(model, fullgraph=True, dynamic=True)

        with torch.no_grad():
            eager_output = model(
                s25=batch_size,
                s70=seq_len,
                input_feat=input_features,
                **params,
            )
            compiled_output = compiled_model(
                s25=batch_size,
                s70=seq_len,
                input_feat=input_features,
                **params,
            )


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_GPU or TRITON_HAS_CPU:
        run_tests(needs="filelock")
