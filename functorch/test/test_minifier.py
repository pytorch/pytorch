# Owner(s): ["module: functorch"]

import torch
from functorch.compile import minifier
from functorch import make_fx
from torch.testing._internal.common_utils import TestCase, run_tests


class TestMinifier(TestCase):
    # https://github.com/pytorch/functorch/issues/913
    def test_has_mul_minifier(self):
        def failing_f(x, y):
            y = y / 3
            x = x + 3
            x = x * y
            return x + y
        inps = [torch.randn(3), torch.randn(3)]
        failing_f = make_fx(failing_f)(*inps)

        def pass_checker(fx_g, inps):
            return (torch.ops.aten.mul.Tensor in set([i.target for i in fx_g.graph.nodes]))

        min_f, inps = minifier(failing_f, inps, pass_checker)
        assert len(min_f.graph.nodes) == 4
        assert len(inps) == 2

    def test_has_add_mul(self):
        def failing_f(x):
            x = x * 3
            x = x + 5
            x = x.cos()
            zero = x - x
            result = zero / zero
            result = result + 3
            return (result * 2,)

        inps = [torch.randn(3)]
        failing_f = make_fx(failing_f)(*inps)

        def pass_checker(fx_g, inps):
            # Basically, make sure none of the inputs are nans
            for i in inps:
                if torch.isnan(i).any():
                    return False
            return torch.isnan(fx_g(*inps)[0]).any()

        min_f, inps = minifier(failing_f, inps, pass_checker)
        assert len(min_f.graph.nodes) == 3
        assert len(inps) == 1


if __name__ == "__main__":
    run_tests()
