# Owner(s): ["module: fx"]

import torch
from torch.fx import symbolic_trace
from torch.fx.experimental.proxy_tensor import make_fx
from torch._decomp import decomposition_table
from torch.fx.passes.dialect.aten.canonicalize import is_canonical

from torch.testing._internal.common_utils import TestCase, run_tests

class TestAtenDialectPasses(TestCase):

    def test_is_canonical_out_variant(self):

        def fn(x):
            y = torch.add(x, x, out=x)
            return y

        gm_aten = make_fx(fn)(torch.rand(3))
        self.assertFalse(is_canonical(gm_aten))

    def test_is_canonical_inplace_op(self):

        def fn(x):
            z = x.add_(1)
            return z

        gm_aten = make_fx(fn)(torch.rand(3))
        self.assertFalse(is_canonical(gm_aten))

    def test_is_canonical_true(self):
        def fn(x):
            y = torch.add(x, x)
            z = y.add(1)
            a = torch.split(z, 2)
            return a[0]

        gm_aten = make_fx(fn)(torch.rand(3))
        self.assertTrue(is_canonical(gm_aten))

        # prim graph is considered canonical
        gm_prims = make_fx(fn, decomposition_table=decomposition_table)(torch.rand(3))
        self.assertTrue(is_canonical(gm_prims))

        gm = symbolic_trace(fn)
        self.assertFalse(is_canonical(gm))


if __name__ == '__main__':
    run_tests()
