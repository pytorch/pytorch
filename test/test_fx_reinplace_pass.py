# Owner(s): ["module: functionalization"]
import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.fx.passes.reinplace import reinplace
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch._dynamo.source import ConstantSource
from torch.fx.experimental.sym_node import SymNode

try:
    from functorch.experimental import functionalize
    HAS_FUNCTIONALIZATION = True
except Exception:
    HAS_FUNCTIONALIZATION = False

class TestReinplacePass(TestCase):

    def test_reinplace_basic(self):
        # Basic test: the out-of-place add() call should be converted
        # into add_()
        def f(x):
            a = x.clone()
            b = a.add(1)
            return b

        inpt = torch.ones(2)
        f2 = reinplace(make_fx(f)(inpt), inpt)
        expected_out = f(inpt)
        actual_out = f2(inpt)
        self.assertEqual(actual_out, expected_out)
        self.assertExpectedInline(f2.code, """\



def forward(self, x_1):
    clone = torch.ops.aten.clone.default(x_1);  x_1 = None
    add = torch.ops.aten.add_.Tensor(clone, 1);  add = None
    return clone
    """)


    def test_reinplace_with_view(self):
        def f(x):
            a = x.clone()
            a_view = a.view(-1)
            # We shouldn't re-inplace the first add(), because an alias of a is reused later in the program
            b = a.add(1)  # noqa: F841

            # Second add() is fine to re-inplace
            c = a_view.add(1)
            return c

        inpt = torch.ones(2)
        f2 = reinplace(make_fx(f)(inpt), inpt)
        expected_out = f(inpt)
        actual_out = f2(inpt)
        self.assertEqual(actual_out, expected_out)
        self.assertExpectedInline(f2.code, """\



def forward(self, x_1):
    clone = torch.ops.aten.clone.default(x_1);  x_1 = None
    view = torch.ops.aten.view.default(clone, [-1])
    add = torch.ops.aten.add.Tensor(clone, 1);  clone = add = None
    add_1 = torch.ops.aten.add_.Tensor(view, 1);  add_1 = None
    return view
    """)

    def test_reinplace_different_metadata(self):
        def f(a_):
            a = a_.clone()
            b = a + 1
            # Naively, we shouldn't try to inplace the .ge() call,
            # because that would require resizing "b" (from a float to a bool tensor).
            c = torch.ge(b, a)
            return c
        inpt = torch.ones(4)
        f2 = reinplace(make_fx(f)(inpt), inpt)
        expected_out = f(inpt)
        actual_out = f2(inpt)
        self.assertEqual(actual_out, expected_out)
        # The .ge() should not be reinplaced.
        self.assertExpectedInline(f2.code, """\



def forward(self, a__1):
    clone = torch.ops.aten.clone.default(a__1);  a__1 = None
    add = torch.ops.aten.add.Tensor(clone, 1)
    ge = torch.ops.aten.ge.Tensor(add, clone);  add = clone = None
    return ge
    """)

    def test_reinplace_overlapping_memory(self):
        def f(a_):
            a = a_.clone()
            b = a.expand(4, 4)
            # Can't reinplace because b has overlapping memory.
            c = b.add(1)
            return c
        inpt = torch.ones(1)
        f2 = reinplace(make_fx(f)(inpt), inpt)
        expected_out = f(inpt)
        actual_out = f2(inpt)
        self.assertEqual(actual_out, expected_out)
        self.assertExpectedInline(f2.code, """\



def forward(self, a__1):
    clone = torch.ops.aten.clone.default(a__1);  a__1 = None
    expand = torch.ops.aten.expand.default(clone, [4, 4]);  clone = None
    add = torch.ops.aten.add.Tensor(expand, 1);  expand = None
    return add
    """)

    # This test won't actually run in CI, because it requires functionalize() from functorch.
    # I'm planning on testing more comprehensively with torchbench models,
    # but we can make this testing better once functorch moves into pytorch/pytorch.
    def test_reinplace_scatter_op(self):
        def f(a_):
            # for now, don't test mutations to inputs
            a = a_.clone()
            e = a.view(-1)
            b = a.view(-1)
            c = b[0]
            d = c.view(-1)
            d.add_(1)
            return a + e

        if not HAS_FUNCTIONALIZATION:
            return
        inpt = torch.ones(4)
        f2 = reinplace(make_fx(functionalize(f))(inpt), inpt)
        expected_out = f(inpt)
        actual_out = f2(inpt)
        self.assertEqual(actual_out, expected_out)
        # NOTE: one slight pessimization here is the fact that
        # there are a bunch of redundant views in the graph.
        # Technically, half of these views are duplicates that we could de-dup.
        # This shouldn't really hurt performance though, since creating an extra view
        # is effectively just moving some metadata around (and allocating a new TensorImpl).
        # We can/should update the pass in the future to clean this up.
        self.assertExpectedInline(f2.code, """\



def forward(self, a__1):
    clone = torch.ops.aten.clone.default(a__1);  a__1 = None
    view = torch.ops.aten.view.default(clone, [-1]);  view = None
    view_1 = torch.ops.aten.view.default(clone, [-1])
    select = torch.ops.aten.select.int(view_1, 0, 0);  view_1 = None
    view_2 = torch.ops.aten.view.default(select, [-1]);  select = None
    add = torch.ops.aten.add_.Tensor(view_2, 1);  add = None
    view_3 = torch.ops.aten.view.default(clone, [-1]);  clone = None
    select_1 = torch.ops.aten.select.int(view_3, 0, 0);  select_1 = None
    view_4 = torch.ops.aten.view.default(view_2, []);  view_2 = view_4 = None
    view_5 = torch.ops.aten.view.default(view_3, [4]);  view_3 = None
    view_6 = torch.ops.aten.view.default(view_5, [-1])
    select_2 = torch.ops.aten.select.int(view_6, 0, 0);  view_6 = None
    view_7 = torch.ops.aten.view.default(select_2, [-1]);  select_2 = view_7 = None
    view_8 = torch.ops.aten.view.default(view_5, [-1])
    add_1 = torch.ops.aten.add_.Tensor(view_5, view_8);  view_8 = add_1 = None
    return view_5
    """)

    def test_reinplace_scatter_twice(self):
        def f(a_):
            # for now, don't test mutations to inputs
            a = a_.clone()
            b = a[:, 1]
            c = b[1]
            c.add_(1)
            return a

        if not HAS_FUNCTIONALIZATION:
            return

        inpt = torch.ones(4, 4)
        f2 = reinplace(make_fx(functionalize(f))(inpt), inpt)
        expected_out = f(inpt)
        actual_out = f2(inpt)
        self.assertEqual(actual_out, expected_out)
        self.assertExpectedInline(f2.code, """\



def forward(self, a__1):
    clone = torch.ops.aten.clone.default(a__1);  a__1 = None
    select = torch.ops.aten.select.int(clone, 1, 1)
    select_1 = torch.ops.aten.select.int(select, 0, 1);  select = None
    add = torch.ops.aten.add_.Tensor(select_1, 1);  select_1 = add = None
    select_2 = torch.ops.aten.select.int(clone, 1, 1);  select_2 = None
    select_3 = torch.ops.aten.select.int(clone, 1, 1)
    select_4 = torch.ops.aten.select.int(select_3, 0, 1);  select_3 = select_4 = None
    return clone
    """)

    def test_reinplace_scatter_twice_with_different_view_op_valid(self):
        def f(a_):
            a = a_.clone()
            b = a[:, 1]
            c = b[1]
            c_updated = c.add(1)
            good_mirror_of_b = a.as_strided((4,), (4,), 1)
            # good_mirror_of_b points to the same region of memory as b.
            # and this scatter op below tries to scatter c_updated into the same region
            # that c currently takes up.
            # reinplacing logic checks this by confirming that:
            #   c_updated
            #   good_mirror_of_b.select(0, 1)
            # have the same size/stride/storage_offset.
            b_updated = torch.select_scatter(good_mirror_of_b, c_updated, 0, 1)
            return b_updated

        inpt = torch.ones(4, 4)
        f2 = reinplace(make_fx(f)(inpt), inpt)
        expected_out = f(inpt)
        actual_out = f2(inpt)
        self.assertEqual(actual_out, expected_out)
        self.assertExpectedInline(f2.code, """\



def forward(self, a__1):
    clone = torch.ops.aten.clone.default(a__1);  a__1 = None
    select = torch.ops.aten.select.int(clone, 1, 1)
    select_1 = torch.ops.aten.select.int(select, 0, 1);  select = None
    add = torch.ops.aten.add_.Tensor(select_1, 1);  select_1 = add = None
    as_strided = torch.ops.aten.as_strided.default(clone, [4], [4], 1);  clone = None
    return as_strided
    """)

    # Test example where we have a scatter op, where the base tensor
    # has the same size/stride/storage offset (even though it is a different view),
    # making it valid to re-inplace
    def test_reinplace_scatter_twice_with_different_view_op_invalid(self):
        def f(a_):
            a = a_.clone()
            b = a[:, 1]
            c = b[1]
            c_updated = c.add(1)
            good_mirror_of_b = a.as_strided((4,), (4,), 1)
            # The first arg to select_scatter is an equivalent view to b.
            # However, the select_scatter call below tries to put c_updated
            # into a different slice of "b" than what "c" currently occupies.
            #
            b_updated = torch.select_scatter(good_mirror_of_b, c_updated, 0, 0)
            return b_updated

        inpt = torch.ones(4, 4)
        f2 = reinplace(make_fx(f)(inpt), inpt)
        expected_out = f(inpt)
        actual_out = f2(inpt)
        self.assertEqual(actual_out, expected_out)
        self.assertExpectedInline(f2.code, """\



def forward(self, a__1):
    clone = torch.ops.aten.clone.default(a__1);  a__1 = None
    select = torch.ops.aten.select.int(clone, 1, 1)
    select_1 = torch.ops.aten.select.int(select, 0, 1);  select = None
    add = torch.ops.aten.add.Tensor(select_1, 1);  select_1 = None
    as_strided = torch.ops.aten.as_strided.default(clone, [4], [4], 1);  clone = None
    select_int = torch.ops.aten.select.int(as_strided, 0, 0)
    copy__default = torch.ops.aten.copy_.default(select_int, add);  select_int = add = copy__default = None
    return as_strided
    """)  # noqa: B950

    def test_reinplace_scatter_twice_with_different_view_op_invalid2(self):
        def f(a_):
            a = a_.clone()
            b = a[:, 1]
            c = b[1]
            c_updated = c.add(1)
            bad_mirror_of_b = a.as_strided((4,), (4,), 0)
            # The first arg to select_scatter points to a different than c's base.
            # This makes it invalid to re-inplace.
            b_updated = torch.select_scatter(bad_mirror_of_b, c_updated, 0, 1)
            return b_updated

        inpt = torch.ones(4, 4)
        f2 = reinplace(make_fx(f)(inpt), inpt)
        expected_out = f(inpt)  # noqa: F841
        actual_out = f2(inpt)  # noqa: F841
        # self.assertEqual(actual_out, expected_out)
        self.assertExpectedInline(f2.code, """\



def forward(self, a__1):
    clone = torch.ops.aten.clone.default(a__1);  a__1 = None
    select = torch.ops.aten.select.int(clone, 1, 1)
    select_1 = torch.ops.aten.select.int(select, 0, 1);  select = None
    add = torch.ops.aten.add.Tensor(select_1, 1);  select_1 = None
    as_strided = torch.ops.aten.as_strided.default(clone, [4], [4], 0);  clone = None
    select_int = torch.ops.aten.select.int(as_strided, 0, 1)
    copy__default = torch.ops.aten.copy_.default(select_int, add);  select_int = add = copy__default = None
    return as_strided
    """)  # noqa: B950


    def test_out_node_updated(self):
        def f():
            x = torch.zeros(2, 2)
            y = x.diagonal()
            y_updated = y.add(1)
            z = torch.diagonal_scatter(x, y_updated)
            # reinplace needs to know to replace output [z] with [x]
            return [z]

        if not HAS_FUNCTIONALIZATION:
            return
        f2 = reinplace(make_fx(functionalize(f))())
        expected_out = f()
        actual_out = f2()
        self.assertEqual(actual_out, expected_out)
        self.assertExpectedInline(f2.code, """\



def forward(self):
    zeros = torch.ops.aten.zeros.default([2, 2], device = device(type='cpu'), pin_memory = False)
    diagonal = torch.ops.aten.diagonal.default(zeros)
    add = torch.ops.aten.add_.Tensor(diagonal, 1);  diagonal = add = None
    return [zeros]
    """)

    def test_reinplace_index_mutation(self):
        def f():
            a = torch.zeros(4, 4, 4)
            a[:, 2:] = torch.ones(4, 2, 4)
            return a

        if not HAS_FUNCTIONALIZATION:
            return
        f2 = reinplace(make_fx(functionalize(f))())
        expected_out = f()
        actual_out = f2()
        self.assertEqual(actual_out, expected_out)
        self.assertExpectedInline(f2.code, """\



def forward(self):
    zeros = torch.ops.aten.zeros.default([4, 4, 4], device = device(type='cpu'), pin_memory = False)
    ones = torch.ops.aten.ones.default([4, 2, 4], device = device(type='cpu'), pin_memory = False)
    slice_1 = torch.ops.aten.slice.Tensor(zeros, 1, 2, 9223372036854775807)
    copy = torch.ops.aten.copy_.default(slice_1, ones);  slice_1 = ones = copy = None
    slice_2 = torch.ops.aten.slice.Tensor(zeros, 1, 2, 9223372036854775807);  slice_2 = None
    return zeros
    """)

    def test_reinplace_sym_input(self):
        # Symbolic input test: the out-of-place add() call should be converted
        # into add_(), and symbolic input won't cause any error.
        def f(x, index):
            a = torch.select(x, 0, index)
            clone = a.clone()
            b = clone.add(1)
            return b

        x = torch.randn((4, 8, 16, 16), requires_grad=False)
        index = 2
        shape_env = ShapeEnv()
        symbol = shape_env.create_symbol(index, source=ConstantSource(
            f"__testing_only{len(shape_env.backed_var_to_val)}"))
        sym_index = torch.SymInt(SymNode(symbol, shape_env, int, hint=index))

        inpt = [x, sym_index]
        f2 = reinplace(make_fx(f)(*inpt), *inpt)

        real_inpt = [x, index]
        expected_out = f(*real_inpt)
        actual_out = f2(*real_inpt)
        self.assertEqual(actual_out, expected_out)
        print(f2.code)
        self.assertExpectedInline(f2.code, """\



def forward(self, x_1, index_1):
    select = torch.ops.aten.select.int(x_1, 0, index_1);  x_1 = index_1 = None
    clone = torch.ops.aten.clone.default(select);  select = None
    add = torch.ops.aten.add_.Tensor(clone, 1);  add = None
    return clone
    """)


if __name__ == '__main__':
    run_tests()
