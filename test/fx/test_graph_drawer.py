# Owner(s): ["oncall: fx"]

import os
import tempfile

import torch
import torch._dynamo as dynamo
from torch.fx.passes.graph_drawer import FxGraphDrawer
from torch.nn import LayerNorm
from torch.testing._internal.common_utils import run_tests, TestCase


class TestGraphDrawer(TestCase):
    def test_that_graph_with_subgraph_draws_successfully(self):
        # covering regression: https://github.com/pytorch/pytorch/issues/137499
        if os.environ.get("INSTALLED_GRAPHVIZ", "") == "yes":
            # temporary, to make sure one of the tests are triggering
            import sys

            sys.exit(1)
            batch_size = 32
            seq_length = 50
            hidden_size = 768
            layer_norm = LayerNorm(hidden_size)

            torch.set_grad_enabled(False)

            @torch.compile
            def fn(inp, weight):
                matmul_output = inp @ weight
                final_output = layer_norm(matmul_output)
                return final_output

            inp = torch.randn(batch_size, seq_length, hidden_size)
            weight = torch.randn(hidden_size, hidden_size)

            graph_module = dynamo.export(fn)(inp, weight)[0]

            g = FxGraphDrawer(graph_module, name="fn")
            dot = g.get_main_dot_graph()
            out_name = tempfile.NamedTemporaryFile(delete=True).name + ".svg"
            # This should succeed
            dot.write_svg(out_name)


if __name__ == "__main__":
    run_tests()
