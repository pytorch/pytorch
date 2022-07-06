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
            b = a.add(1)
            return b

        inpt = torch.ones(2, device='meta')
        f2 = reinplace(make_fx(f)(inpt), inpt)
        self.assertExpectedInline(f2.code, """\



def forward(self, x_1):
    clone_default = torch.ops.aten.clone.default(x_1);  x_1 = None
    add_tensor = torch.ops.aten.add_.Tensor(clone_default, 1)
    return clone_default
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

    def test_reinplace_with_view(self):
        def f(x):
            a = x.clone()
            a_view = a.view(-1)
            # We shouldn't re-inplace the first add(), because an alias of a is re-used later in the program
            b = a.add(1)
            # Second add() is fine to re-inplace
            c = a_view.add(1)
            return c

        inpt = torch.ones(2, device='meta')
        f2 = reinplace(make_fx(f)(inpt), inpt)
        self.assertExpectedInline(f2.code, """\



def forward(self, x_1):
    clone_default = torch.ops.aten.clone.default(x_1);  x_1 = None
    view_default = torch.ops.aten.view.default(clone_default, [-1])
    add_tensor = torch.ops.aten.add.Tensor(clone_default, 1);  clone_default = None
    add_tensor_1 = torch.ops.aten.add_.Tensor(view_default, 1)
    return view_default
    """)

    def test_reinplace_mutation_on_input_alias(self):
        def f(x):
            a = x.view(-1)
            # We can't reinplace the first add, since it was run on an alias
            # of an input.
            b = a.add(1)
            # Second add() is fine to re-inplace though.
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


    def test_reinplace_mutation_on_input_alias_with_overwritten_data(self):
        def f(x, y):
            # The first add() is safe to reinplace even though x is an input, because x is overwritten later.
            a = x.add(1)
            # The second add() is NOT safe to reinplace, because a aliases a_view,
            # and a is used as an input to the copy_() later.
            a_view = a.view(-1)
            b = a_view.add(1)
            x.copy_(a)
            # The second add() is NOT safe to reinplace, because y is an input that is not overwritten later.
            c = y.add(1)
            return c

        x = torch.ones(2, device='meta')
        y = torch.ones(2, device='meta')
        f2 = reinplace(make_fx(f)(x, y), x, y)
        self.assertExpectedInline(f2.code, """\



def forward(self, x_1, y_1):
    add_tensor = torch.ops.aten.add_.Tensor(x_1, 1)
    view_default = torch.ops.aten.view.default(x_1, [-1])
    add_tensor_1 = torch.ops.aten.add.Tensor(view_default, 1);  view_default = None
    copy__default = torch.ops.aten.copy_.default(x_1, x_1);  x_1 = None
    add_tensor_2 = torch.ops.aten.add.Tensor(y_1, 1);  y_1 = None
    return add_tensor_2
    """)

if __name__ == '__main__':
    run_tests()
