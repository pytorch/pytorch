# Owner(s): ["oncall: jit"]

import torch
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


class TestFuserCommon(JitTestCase):
    def test_autodiff_fallback(self):
        for rq in [True, False]:

            @torch.jit.script
            def fn(x):
                return torch.max(x**2.0, x**3.0)

            x = torch.randn(5, requires_grad=not rq)
            # cause optimization to be created
            for _ in range(5):
                fn(x)
            # test fallback when optimization is not applicable
            y = fn(torch.randn(5, requires_grad=rq))
            self.assertEqual(y.requires_grad, rq)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit_fuser_te.py")
