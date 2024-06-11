# Owner(s): ["module: fx"]

import os
import tempfile

import torch
from torch.fx import subgraph_rewriter, symbolic_trace
from torch.fx.passes.graph_transform_observer import GraphTransformObserver

from torch.testing._internal.common_utils import TestCase


if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_fx.py TESTNAME\n\n"
        "instead."
    )


class TestGraphTransformObserver(TestCase):
    def test_graph_transform_observer(self):
        class M(torch.nn.Module):
            def forward(self, x):
                val = torch.neg(x)
                return torch.add(val, val)

        def pattern(x):
            return torch.neg(x)

        def replacement(x):
            return torch.relu(x)

        traced = symbolic_trace(M())

        log_url = tempfile.mkdtemp()

        with GraphTransformObserver(traced, "replace_neg_with_relu", log_url) as ob:
            subgraph_rewriter.replace_pattern(traced, pattern, replacement)

            self.assertTrue("relu" in ob.created_nodes)
            self.assertTrue("neg" in ob.erased_nodes)

        current_pass_count = GraphTransformObserver.get_current_pass_count()

        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    log_url,
                    f"pass_{current_pass_count}_replace_neg_with_relu_input_graph.svg",
                )
            )
        )
        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    log_url,
                    f"pass_{current_pass_count}_replace_neg_with_relu_output_graph.svg",
                )
            )
        )
