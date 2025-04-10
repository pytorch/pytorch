# Owner(s): ["module: inductor"]
import functools

import torch
from torch._dispatch.python import enable_python_dispatcher
from torch._inductor.codegen.subgraph import SubgraphTemplate
from torch._inductor.decomposition import select_decomp_table
from torch._inductor.ir import Buffer, FixedLayout
from torch._inductor.lowering import register_lowering
from torch._inductor.select_algorithm import (
    AlgorithmSelectorCache,
    autotune_select_algorithm,
)
from torch._inductor.test_case import run_tests, TestCase
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CPU, HAS_GPU


class TestSubgraphChoice(TestCase):
    def setUp(self):
        super().setUp()

    def _create_buffer(self, name, shape, dtype):
        return Buffer(
            name=name,
            layout=FixedLayout(torch.device(f"{GPU_TYPE}:0"), dtype=dtype, size=shape),
        )

    def test_subgraph_decompose_k(self):
        from torch._inductor.kernel.mm import aten_mm
        from torch._inductor.kernel.mm_common import mm_args

        @torch.library.custom_op("mylib::matmul_decompose", mutates_args={})
        def matmul_decompose(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a @ b

        @matmul_decompose.register_fake
        def _(a, b):
            return a @ b

        def decomposeK(a, b, kPartitions):
            m = a.shape[0]
            n = b.shape[1]
            k = a.shape[1]

            B = k // kPartitions
            a_reshaped = torch.permute(a.reshape(m, B, kPartitions), (1, 0, 2))
            b_reshaped = b.reshape(B, kPartitions, n)
            result = torch.bmm(a_reshaped, b_reshaped)
            result_fp32 = result.to(torch.float32)
            reduced_buf = torch.sum(result_fp32, 0)
            return reduced_buf.to(a.dtype)

        mat1_shape, mat2_shape = (32, 4096), (4096, 32)

        @register_lowering(torch.ops.mylib.matmul_decompose)
        def _(a, b):
            _, _, _, layout, mat1, mat2 = mm_args(a, b)

            choices = [aten_mm.bind((mat1, mat2), layout)]

            # TODO (PaulZhang12): Once decomposeK lands in Inductor, move this
            kPartitions = 256
            with enable_python_dispatcher():
                decompositions = select_decomp_table()

                decompose_k_subgraph_template = SubgraphTemplate(
                    name="decompose_k_mm",
                    make_fx_graph=make_fx(
                        functools.partial(decomposeK, kPartitions=kPartitions),
                        decompositions,
                        tracing_mode="real",
                    ),
                )

            mat1_tensor, mat2_tensor = (
                AlgorithmSelectorCache.benchmark_example_value(mat1),
                AlgorithmSelectorCache.benchmark_example_value(mat2),
            )
            decompose_k_subgraph_template.maybe_append_choice(
                choices,
                input_nodes=(mat1, mat2),
                layout=layout,
                example_inputs=[mat1_tensor, mat2_tensor],
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
        # Relax precision as decomposeK does first accumulation in fp16
        torch.testing.assert_close(res, a_in @ b_in, atol=1e-1, rtol=1e-1)


if __name__ == "__main__":
    # Set env to make it work in CI.
    if HAS_GPU and HAS_CPU:
        run_tests()
