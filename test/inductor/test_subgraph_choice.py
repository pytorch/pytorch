# Owner(s): ["module: inductor"]
from unittest import mock
from unittest.mock import MagicMock

import torch
from torch._inductor.ir import Buffer, FixedLayout, FlexibleLayout
from torch._inductor.lowering import register_lowering
from torch._inductor.select_algorithm import autotune_select_algorithm
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import skipIfXpu
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CPU, HAS_GPU


def decomposeK(a, b, kPartitions):
    m = a.shape[0]
    n = b.shape[1]
    k = a.shape[1]

    B = k // kPartitions
    a_reshaped = torch.permute(a.reshape(m, B, kPartitions), (1, 0, 2))
    b_reshaped = b.reshape(B, kPartitions, n)
    result = torch.bmm(a_reshaped, b_reshaped, out_dtype=torch.float32)
    result_fp32 = result.to(torch.float32)
    reduced_buf = torch.sum(result_fp32, 0)
    return reduced_buf.to(a.dtype)


class TestSubgraphChoice(TestCase):
    def setUp(self):
        super().setUp()

    def _create_buffer(self, name, shape, dtype):
        return Buffer(
            name=name,
            layout=FixedLayout(torch.device(f"{GPU_TYPE}:0"), dtype=dtype, size=shape),
        )

    @skipIfXpu
    def test_subgraph_decompose_k(self):
        from torch._inductor.kernel.mm import aten_mm
        from torch._inductor.kernel.mm_common import mm_args

        mat1_shape, mat2_shape = (32, 4096), (4096, 32)

        @torch.library.custom_op("mylib::matmul_decompose", mutates_args={})
        def matmul_decompose(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a @ b

        @matmul_decompose.register_fake
        def _(a, b):
            return a @ b

        @register_lowering(torch.ops.mylib.matmul_decompose)
        def _(a, b):
            _, _, _, layout, mat1, mat2 = mm_args(a, b)

            choices = [aten_mm.bind((mat1, mat2), layout)]

            kPartitions = 256

            decompose_k_subgraph_template = (
                torch._inductor.kernel.mm.DecomposeKSugraphTemplate()
            )

            decompose_k_subgraph_template.maybe_append_choice(
                choices,
                k_split=kPartitions,
                input_nodes=(mat1, mat2),
                layout=layout,
            )

            # Test benchmarking against aten
            autotune_select_algorithm("test_subgraph_choice", choices, [a, b], layout)

            # Only return decomposeK case for codegen
            choices = [choices[1]]
            return autotune_select_algorithm(
                "test_subgraph_choice", choices, [a, b], layout
            )

        a_in = torch.randn(
            mat1_shape, dtype=torch.float16, device=torch.device(f"{GPU_TYPE}:0")
        )
        b_in = torch.randn(
            mat2_shape, dtype=torch.float16, device=torch.device(f"{GPU_TYPE}:0")
        )

        def func(mat1, mat2):
            return torch.ops.mylib.matmul_decompose(mat1, mat2)

        compiled_func = torch.compile(func, mode="max-autotune", dynamic=False)

        res = compiled_func(a_in, b_in)

        # Check same results of compiled result and regular torch.mm
        torch.testing.assert_close(res, a_in @ b_in, atol=1e-1, rtol=1e-1)

    @skipIfXpu
    def test_subgraph_freeze_layout(self):
        from torch._inductor.kernel.mm_common import mm_args

        M, N, K = (4, 128, 14240)
        a_in = torch.randn(
            (M, K), dtype=torch.bfloat16, device=torch.device(f"{GPU_TYPE}:0")
        )
        b_in = torch.randn(
            (K, N), dtype=torch.bfloat16, device=torch.device(f"{GPU_TYPE}:0")
        )

        @torch.library.custom_op("mylib::matmul_decompose_padding", mutates_args={})
        def matmul_decompose(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a @ b

        @matmul_decompose.register_fake
        def _(a, b):
            return a @ b

        @register_lowering(torch.ops.mylib.matmul_decompose_padding)
        def _(a, b):
            _, _, _, layout, mat1, mat2 = mm_args(a, b)
            mat1_layout = mat1.layout
            if not isinstance(mat1_layout, FlexibleLayout):
                raise AssertionError
            mat1_stride = mat1_layout.stride

            choices = []

            kPartitions = 2

            decompose_k_subgraph_template = (
                torch._inductor.kernel.mm.DecomposeKSugraphTemplate()
            )

            decompose_k_subgraph_template.maybe_append_choice(
                choices,
                k_split=kPartitions,
                input_nodes=(mat1, mat2),
                layout=layout,
            )

            choice = choices[0]
            if not isinstance(mat1.layout, FixedLayout):
                raise AssertionError

            # Creating the subgraph choice should have frozen the layout
            # We ensure padding so the stride should differ
            if mat1.layout.stride == mat1_stride:
                raise AssertionError

            for example_stride, layout_stride in zip(
                choice.example_inputs[0].stride(), mat1.layout.stride
            ):
                # Example inputs should have same stride as current layout
                if example_stride != layout_stride:
                    raise AssertionError

            return autotune_select_algorithm(
                "test_subgraph_choice", choices, [a, b], layout
            )

        def func(mat1, mat2):
            return torch.ops.mylib.matmul_decompose_padding((mat1 + 1.0), mat2)

        with mock.patch("torch._inductor.ir.V.get_current_node") as get_node_mock:
            node_mock = MagicMock()
            node_mock.meta = {"dislike_padding": False}
            get_node_mock.return_value = node_mock

            compiled_func = torch.compile(func, mode="max-autotune", dynamic=False)

            compiled_func(a_in, b_in)


if __name__ == "__main__":
    # Set env to make it work in CI.
    if HAS_GPU and HAS_CPU:
        run_tests()
