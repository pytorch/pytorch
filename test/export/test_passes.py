# Owner(s): ["module: dynamo"]
import unittest

import torch
import torch._dynamo as torchdynamo
from torch.testing._internal.common_utils import run_tests, TestCase
from torch._dynamo.eval_frame import is_dynamo_supported
from torch._export.passes import ReplaceBrokenOpsWithFunctionalOpsPass


class TestPasses(TestCase):
    @unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
    def test_replace_broken_ops(self) -> None:
        x = torch.randn([2, 3, 4, 5])
        model: torch.nn.Linear = torch.nn.Linear(5, 5)

        def f(inp: torch.Tensor) -> torch.Tensor:
            return model(inp)

        gm, _ = torchdynamo.export(f, x, aten_graph=True)

        new_gm = ReplaceBrokenOpsWithFunctionalOpsPass()(gm)
        self.assertIsNotNone(new_gm)
        new_gm = new_gm.graph_module

        count_after = 0
        for node in new_gm.graph.nodes:
            if node.target == torch.ops.aten.view.default:
                count_after += 1
        self.assertEqual(count_after, 0)
        self.assertTrue(torch.allclose(gm(x), f(x)))

if __name__ == '__main__':
    run_tests()
