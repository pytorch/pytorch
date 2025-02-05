# Owner(s): ["module: dynamo"]
# flake8: noqa: F841
import torch
import torch._dynamo.test_case
from torch._dynamo.exc import NoGraphError


class FullGraphConstraints(torch._dynamo.test_case.TestCase):
    def test_no_graph(self):
        def fn(x, y):
            return x + y

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaises(NoGraphError):
            opt_fn(1, 2)

    def test_no_graph2(self):
        def fn(x):
            d = {1: 2}
            return x

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaises(NoGraphError):
            opt_fn(torch.rand(4))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
