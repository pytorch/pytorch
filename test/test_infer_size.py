import torch
from torch.testing._internal.jit_utils import JitTestCase

class TestJit(JitTestCase):
    def test_infer_size(self):
        from torch._C import _infer_size

        def fn(x, y):
            # type: (Tensor, Tensor) -> List[int]
            return _infer_size(x.size(), y.size())

        self.checkScript(fn, (torch.ones(2, 4, 2), torch.ones(2, 4, 2)))


    def test_infer_size_bce(self):
        from torch._C import _infer_size

        for dimA, dimB in [((2,), (2,)),
                           ((2, 1), (2,)),
                           ((2,), (2, 1)),
                           ]:
            a, b = torch.empty(dimA).uniform_(), torch.empty(dimB).uniform_()
            print("Testing dimA {}, dimB {}".format(dimA, dimB))

            @torch.no_grad()
            def fn(x, y):
                return _infer_size(x.size(), y.size())

            x = torch.jit.trace(fn, [a, b])
            print(x.graph)
            # self.checkTrace(fn, (torch.ones(2, 4, 2), torch.ones(2, 4, 2)), inputs_require_grads=True)
            # out = fn(torch.ones(1), torch.ones(1))
            # print(out)
