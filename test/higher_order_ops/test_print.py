import unittest
import unittest.mock as mock
from unittest.mock import patch

import io

import torch
from torch._dynamo.utils import counters

from torch.testing._internal.common_utils import skipIfTorchDynamo, TestCase


class TestHopPrint(TestCase):
    def test_base_print(self):
        # class M(torch.nn.Module):
        #     def forward(self, x):
        #         torch._higher_order_ops.print("moo")
        #         res = x + x
        #         torch._higher_order_ops.print(("moo")
        #         return (res,)
        def f(x):
            x = x + x
            # change it to HOP it should still work
            # print("moo")
            torch._higher_order_ops.print("moo")
            x = x * x
            return x

        counters.clear()
        x = torch.randn(3, 3)
        opt_f = torch.compile(backend="eager")(f)
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            opt_out = opt_f(x)
            printed_output = mock_stdout.getvalue().strip()
            orig_out = f(x)

        self.assertTrue(same(orig_out, opt_out))
        self.assertEqual(printed_output, "moo")
        self.assertEqual(len(counters["graph_break"]), 1)
