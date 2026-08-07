# Owner(s): ["module: inductor"]
import unittest
from unittest import mock
from unittest.mock import MagicMock

import torch
from torch._inductor.ir import FixedLayout, FlexibleLayout
from torch._inductor.lowering import register_lowering
from torch._inductor.select_algorithm import autotune_select_algorithm
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import HardwareClassification
from torch.utils._triton import has_triton


class TestSubgraphChoice(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    def setUp(self):
        super().setUp()

    @unittest.skipIf(not has_triton(), "requires triton")
    def test_subgraph_decompose_k(self, device):
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
            node, _ = autotune_select_algorithm(
                "test_subgraph_choice", choices, [a, b], layout
            )
            return node

        a_in = torch.randn(mat1_shape, dtype=torch.float16, device=device)
        b_in = torch.randn(mat2_shape, dtype=torch.float16, device=device)

        def func(mat1, mat2):
            return torch.ops.mylib.matmul_decompose(mat1, mat2)

        compiled_func = torch.compile(func, mode="max-autotune", dynamic=False)

        res = compiled_func(a_in, b_in)

        # Check same results of compiled result and regular torch.mm
        torch.testing.assert_close(res, a_in @ b_in, atol=1e-1, rtol=1e-1)

    @unittest.skipIf(not has_triton(), "requires triton")
    def test_subgraph_freeze_layout(self, device):
        from torch._inductor.kernel.mm_common import mm_args

        M, N, K = (4, 128, 14240)
        a_in = torch.randn((M, K), dtype=torch.bfloat16, device=device)
        b_in = torch.randn((K, N), dtype=torch.bfloat16, device=device)

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

            node, _ = autotune_select_algorithm(
                "test_subgraph_choice", choices, [a, b], layout
            )
            return node

        def func(mat1, mat2):
            return torch.ops.mylib.matmul_decompose_padding((mat1 + 1.0), mat2)

        with mock.patch("torch._inductor.ir.V.get_current_node") as get_node_mock:
            node_mock = MagicMock()
            node_mock.meta = {"dislike_padding": False}
            get_node_mock.return_value = node_mock

            compiled_func = torch.compile(func, mode="max-autotune", dynamic=False)

            compiled_func(a_in, b_in)


instantiate_device_type_tests(
    TestSubgraphChoice, globals(), except_for="cpu", allow_xpu=True
)


if __name__ == "__main__":
    run_tests()
