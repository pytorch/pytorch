# Owner(s): ["module: functionalization"]
import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.fx.passes.reinplace import reinplace
from torch.fx.experimental.proxy_tensor import make_fx

class TestReinplacePass(TestCase):

    def test_reinplace_basic(self):
        # Basic test: the out-of-place add() call should be converted
        # into add_()
        def f(x):
            a = x.clone()
            b = a.view(-1)
            b_updated = b.add(1)
            a_updated = b_updated.view(a.shape)
            return a_updated

        inpt = torch.ones(2, device='meta')
        f2 = reinplace(make_fx(f)(inpt), inpt)
        self.assertExpectedInline(f2.code, """\



def forward(self, x_1):
    clone_default = torch.ops.aten.clone.default(x_1);  x_1 = None
    view_default = torch.ops.aten.view.default(clone_default, [-1]);  clone_default = None
    add_tensor = torch.ops.aten.add_.Tensor(view_default, 1)
    view_default_1 = torch.ops.aten.view.default(view_default, [2]);  view_default = None
    return view_default_1
    """)

    def test_reinplace_mutation_on_input(self):
        # We can't convert the first add() call into an inplace,
        # because it was performed on an input.
        # The second add() should be converted into add_() though.
        def f(x):
            a = x.add(1)
            b = a.add(1)
            return b

        inpt = torch.ones(2, device='meta')
        f2 = reinplace(make_fx(f)(inpt), inpt)
        self.assertExpectedInline(f2.code, """\



def forward(self, x_1):
    add_tensor = torch.ops.aten.add.Tensor(x_1, 1);  x_1 = None
    add_tensor_1 = torch.ops.aten.add_.Tensor(add_tensor, 1)
    return add_tensor
    """)

    def test_reinplace_mutation_on_input_alias(self):
        def f(x):
            # The first add() should be reinplaced, but not the second.
            # The first add() is performed on an alias of the input.
            a = x.view(-1)
            b = a.add(1)
            c = b.add(1)
            return c

        inpt = torch.ones(2, device='meta')
        f2 = reinplace(make_fx(f)(inpt), inpt)
        self.assertExpectedInline(f2.code, """\



def forward(self, x_1):
    view_default = torch.ops.aten.view.default(x_1, [-1]);  x_1 = None
    add_tensor = torch.ops.aten.add.Tensor(view_default, 1);  view_default = None
    add_tensor_1 = torch.ops.aten.add_.Tensor(add_tensor, 1)
    return add_tensor
    """)

if __name__ == '__main__':
    run_tests()
