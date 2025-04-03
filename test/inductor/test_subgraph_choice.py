from torch._inductor.test_case import TestCase
import torch
from torch._inductor.decomposition import select_decomp_table
from torch._dispatch.python import enable_python_dispatcher
from torch._inductor.codegen.subgraph import SubgraphTemplate, SubgraphChoiceCaller
from torch.fx.experimental.proxy_tensor import make_fx
import functools
from torch.testing._internal.inductor_utils import GPU_TYPE
from torch._inductor.ir import Buffer, FixedLayout
from torch._inductor.graph import GraphLowering
from torch._inductor.virtualized import V
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._guards import tracing, TracingContext
from unittest.mock import Mock

class TestFlexAttention(TestCase):
    def setUp(self):
        super().setUp()
    
    def _create_buffer(self, name, shape, dtype):
        return Buffer(
            name=name,
            layout=FixedLayout(
                torch.device(f"{GPU_TYPE}:0"), dtype=dtype, size=shape
            ),
        )

    def test_subgraph_decompose_k(self):
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
            return (reduced_buf.to(a.dtype), )

        a = torch.randn((32, 4096), dtype=torch.float16, device=torch.device(f"{GPU_TYPE}:0"))
        b = torch.randn((4096, 32), dtype=torch.float16, device=torch.device(f"{GPU_TYPE}:0"))
        gm = make_fx(
            lambda a, b: a @ b
        )(a, b)  # a dummy graph to construct the GraphLowering
        graph = GraphLowering(gm)

        # the graph handler is neede to create benchmark example value below
        fake_tensor_mode = FakeTensorMode()
        mock_debug_handler = Mock()

        with V.set_graph_handler(graph), V.set_fake_mode(fake_tensor_mode), tracing(TracingContext(fake_tensor_mode)), V.set_debug_handler(mock_debug_handler):
            kPartitions = 256
            with enable_python_dispatcher():
                decompositions = (
                    select_decomp_table()
                )

                decompose_k_subgraph_template = SubgraphTemplate(
                    name="decompose_k_mm",
                    make_fx_graph=make_fx(functools.partial(decomposeK, kPartitions=kPartitions), decompositions, tracing_mode="real"),
                )
            
            mat1 = self._create_buffer("mat1", (32, 4096), torch.float16)
            mat2 = self._create_buffer("mat2", (4096, 32), torch.float16)

            ##### Test Benchmarking
            choices = []

            layout=FixedLayout(
                torch.device(f"{GPU_TYPE}:0"), dtype=torch.float16, size=[32, 32]
            )
            decompose_k_subgraph_template.maybe_append_choice(
                choices,
                input_nodes=(mat1, mat2),
                layout=layout,
                example_inputs=[a, b],
            )

            self.assertEqual(len(choices), 1)
            self.assertTrue(isinstance(choices[0], SubgraphChoiceCaller))

            out = torch.zeros((32, 32), dtype=torch.float16, device=torch.device(f"{GPU_TYPE}:0"))
            subgraph_choice = choices[0]
            subgraph_choice.benchmark(*[a, b], out=out)

            torch.testing.assert_close(out, a @ b, atol=1e-1, rtol=0.5)


            ##### Test Codegen
            V.graph.init_wrapper_code()
            node = subgraph_choice.output_node().data.data
            node.codegen(V.graph.wrapper_code)
            
            # Check subgraph lines, stricter check for generated code?
            self.assertTrue(len(node.subgraph.wrapper_code.lines) > 0)

            # Check parent graph lines, stricter check for generated code?
            self.assertTrue(len(V.graph.wrapper_code.lines) > 0)
