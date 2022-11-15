# Owner(s): ["module: codegen"]

import torch
from torch.testing._internal.common_utils import TestCase, run_tests, skipIfTorchDynamo, TEST_WITH_TORCHDYNAMO
from torch.testing._internal.logging_tensor import LoggingTensor, capture_logs
from torch.utils._pytree import tree_map
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.reinplace import reinplace

import unittest

def are_aliased(x, y):
    if x._base is None and y._base is None:
        return False
    if x._base is not None and y._base is None:
        return x._base is y
    if x._base is None and y._base is not None:
        return y._base is x
    return x._base is y._base

# We can unify testing and use functionalize() here instead
# if/when functorch moves into core.
# This is basically a crappy version of `functionalize()` for single-tensor-arg inputs.
def _functionalize(f, *, reapply_views: bool):
    def wrapped(a):
        input_functional = torch._to_functional_tensor(a)
        input_functional.requires_grad = a.requires_grad
        torch._enable_functionalization(reapply_views=reapply_views)
        try:
            out = f(input_functional)
        finally:
            torch._disable_functionalization()
        torch._sync(input_functional)
        inpt_new = torch._from_functional_tensor(input_functional)
        if inpt_new is not a:
            # Existing deficiency in functionalize():
            # we don't correctly mutate input metadata (yet?)
            if inpt_new.shape == a.shape:
                a.copy_(inpt_new)
        tree_map(torch._sync, out)
        out_unwrapped = tree_map(torch._from_functional_tensor, out)
        return out_unwrapped

    return wrapped

@unittest.skipIf(TEST_WITH_TORCHDYNAMO, "https://github.com/pytorch/pytorch/issues/81457")
class TestFunctionalization(TestCase):

    def get_logs(self, func, inpt, *, reapply_views=False, run_reinplace=False):
        inpt_clone = inpt.clone()
        traced_f = make_fx(_functionalize(func, reapply_views=reapply_views))(inpt)
        if run_reinplace:
            traced_f = reinplace(traced_f, inpt_clone)
        return traced_f.code

    def assert_functionalization(self, func, inpt, *, reapply_views=False, mutated_input_metadata=False):
        input_clone = inpt.clone()
        input_clone2 = inpt.clone()
        input_clone3 = inpt.clone()

        # Compare outputs (and mutated inputs), with and without functionalization.
        out_ref = func(inpt)
        out_functional = _functionalize(func, reapply_views=reapply_views)(input_clone)
        # The reinplacing pass is only valid to run with reapply_views=True.
        functional_func = make_fx(_functionalize(func, reapply_views=True))(input_clone2)
        reinplace_func = reinplace(make_fx(_functionalize(func, reapply_views=True))(input_clone2), input_clone2)

        # NOTE: for now, need to pass in fresh inputs here, because make_fx
        # will directly mutate the inputs that you trace with.
        # Once this is fixed we can clean this up.
        out_reinplace = reinplace_func(input_clone3)

        # functionalize() deficiency: input metadata mutations aren't propagated properly,
        # so we just need to skip checks here for the tests that exercise that.
        if not mutated_input_metadata:
            self.assertEqual(inpt, input_clone)  # input mutations should still occur
            self.assertEqual(inpt, input_clone3)

        # Handle tests with multi-tensor outputs
        if isinstance(out_ref, tuple):
            out_refs, out_functionals, out_reinplaces = list(out_ref), list(out_functional), list(out_reinplace)
        else:
            out_refs, out_functionals, out_reinplaces = [out_ref], [out_functional], [out_reinplace]

        for out_ref_, out_functional_, out_reinplace_ in zip(out_refs, out_functionals, out_reinplaces):
            self.assertEqual(out_ref_, out_functional_)
            self.assertEqual(out_ref_, out_reinplace_)

    def test_save_for_backwards_segfault(self):
        inp = torch._to_functional_tensor(LoggingTensor(torch.randn(2, 2))).requires_grad_(True)
        inp.exp()

    def test_multiple_views_of_same_base(self):
        def f(x):
            y = x.view(-1)
            z = x.view(-1)
            x.add_(1)
            # y should have been updated.
            y2 = y + 1
            # z should have been updated too.
            z2 = z + 1
            return z2
        self.assert_functionalization(f, torch.ones(4))

    def test_freeze(self):
        def f(x):
            y = x.clone()
            z = y[0]
            torch._freeze_functional_tensor(y)
            x.add_(1)
            self.assertRaises(RuntimeError, lambda: y.add_(1))
            self.assertRaises(RuntimeError, lambda: z.add_(1))
            return z

        _functionalize(f, reapply_views=True)(torch.ones(3, 3))

    def test_view_clone_view_inplace(self):
        def f(input):
            shape = [1, 1024, 128, 128]
            input_reshaped = input.view(shape)
            out = input_reshaped.clone()
            r = out.view(input.shape)
            r.relu_()
            return r

        def g(x):
            loss = f(x).sum()
            from functorch._src.aot_autograd import setup_stacktrace_preservation_hooks
            import torch.fx.traceback as fx_traceback
            setup_stacktrace_preservation_hooks([loss.grad_fn])
            with fx_traceback.override_stack_trace():
                loss.backward()
            return x.grad

        with torch.autograd.detect_anomaly(check_nan=False):
            logs = self.get_logs(g, torch.ones(16, 64, 128, 128, requires_grad=True))
        self.assertExpectedInline(logs, """\



def forward(self, a_1):
    view_copy = torch.ops.aten.view_copy.default(a_1, [1, 1024, 128, 128]);  a_1 = None
    clone = torch.ops.aten.clone.default(view_copy);  view_copy = None
    view_copy_1 = torch.ops.aten.view_copy.default(clone, [16, 64, 128, 128])
    relu = torch.ops.aten.relu.default(view_copy_1);  view_copy_1 = None
    view_copy_2 = torch.ops.aten.view_copy.default(clone, [16, 64, 128, 128]);  clone = None
    sum_1 = torch.ops.aten.sum.default(relu)
    ones_like = torch.ops.aten.ones_like.default(sum_1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False, memory_format = torch.preserve_format);  sum_1 = None
    expand_copy = torch.ops.aten.expand_copy.default(ones_like, [16, 64, 128, 128]);  ones_like = None
    _reshape_alias_copy = torch.ops.aten._reshape_alias_copy.default(expand_copy, [1, 1024, 128, 128], [16777216, 16384, 128, 1]);  expand_copy = None
    new_empty_strided = torch.ops.aten.new_empty_strided.default(_reshape_alias_copy, [1, 1024, 128, 128], [16777216, 16384, 128, 1])
    view_copy_3 = torch.ops.aten.view_copy.default(_reshape_alias_copy, [16, 64, 128, 128])
    view_copy_4 = torch.ops.aten.view_copy.default(_reshape_alias_copy, [16, 64, 128, 128])
    clone_1 = torch.ops.aten.clone.default(view_copy_4, memory_format = torch.contiguous_format);  view_copy_4 = None
    threshold_backward = torch.ops.aten.threshold_backward.default(clone_1, relu, 0);  clone_1 = relu = None
    _reshape_alias_copy_1 = torch.ops.aten._reshape_alias_copy.default(_reshape_alias_copy, [16, 64, 128, 128], [1048576, 16384, 128, 1]);  _reshape_alias_copy = None
    detach_copy = torch.ops.aten.detach_copy.default(_reshape_alias_copy_1);  _reshape_alias_copy_1 = None
    view_copy_5 = torch.ops.aten.view_copy.default(threshold_backward, [1, 1024, 128, 128]);  threshold_backward = None
    _reshape_alias_copy_2 = torch.ops.aten._reshape_alias_copy.default(view_copy_5, [16, 64, 128, 128], [1048576, 16384, 128, 1]);  view_copy_5 = None
    detach_copy_1 = torch.ops.aten.detach_copy.default(_reshape_alias_copy_2);  _reshape_alias_copy_2 = None
    return detach_copy_1
    """)  # noqa: B950

    def test_simple(self):
        def f(x):
            # simple test: 1 view op, 1 inplace op
            tmp = torch.ones(4, 2)
            y = x.view(4, 2)
            y.add_(tmp)
            z = x * x
            return y
        self.assert_functionalization(f, torch.ones(4, 2))
        logs = self.get_logs(f, torch.ones(4, 2))
        self.assertExpectedInline(logs, """\



def forward(self, a_1):
    ones = torch.ops.aten.ones.default([4, 2], device = device(type='cpu'), pin_memory = False)
    view_copy = torch.ops.aten.view_copy.default(a_1, [4, 2])
    add = torch.ops.aten.add.Tensor(view_copy, ones);  view_copy = ones = None
    view_copy_1 = torch.ops.aten.view_copy.default(add, [4, 2])
    mul = torch.ops.aten.mul.Tensor(view_copy_1, view_copy_1)
    copy_ = torch.ops.aten.copy_.default(a_1, view_copy_1);  a_1 = view_copy_1 = None
    return add
    """)

        reinplaced_logs = self.get_logs(f, torch.ones(4, 2), reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(reinplaced_logs, """\



def forward(self, a_1):
    ones = torch.ops.aten.ones.default([4, 2], device = device(type='cpu'), pin_memory = False)
    view = torch.ops.aten.view.default(a_1, [4, 2])
    add = torch.ops.aten.add.Tensor(view, ones);  view = ones = None
    view_1 = torch.ops.aten.view.default(add, [4, 2])
    mul = torch.ops.aten.mul.Tensor(view_1, view_1)
    copy_ = torch.ops.aten.copy_.default(a_1, view_1);  a_1 = view_1 = None
    return add
    """)

    def test_simple_out(self):
        def f(x):
            tmp = torch.ones(4, 2)
            y = x.view(4, 2)
            # the out= tensor will get resized, since it has size=0 to start.
            z = torch.empty(())
            torch.add(y, tmp, out=z)
            w = z * z
            return w
        self.assert_functionalization(f, torch.ones(4, 2))
        logs = self.get_logs(f, torch.ones(4, 2))
        self.assertExpectedInline(logs, """\



def forward(self, a_1):
    ones = torch.ops.aten.ones.default([4, 2], device = device(type='cpu'), pin_memory = False)
    view_copy = torch.ops.aten.view_copy.default(a_1, [4, 2]);  a_1 = None
    empty = torch.ops.aten.empty.memory_format([], device = device(type='cpu'), pin_memory = False)
    add = torch.ops.aten.add.Tensor(view_copy, ones);  view_copy = ones = None
    mul = torch.ops.aten.mul.Tensor(add, add);  add = None
    return mul
    """)

        reinplaced_logs = self.get_logs(f, torch.ones(4, 2), reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(reinplaced_logs, """\



def forward(self, a_1):
    ones = torch.ops.aten.ones.default([4, 2], device = device(type='cpu'), pin_memory = False)
    view = torch.ops.aten.view.default(a_1, [4, 2]);  a_1 = None
    empty = torch.ops.aten.empty.memory_format([], device = device(type='cpu'), pin_memory = False)
    add = torch.ops.aten.add.Tensor(view, ones);  view = ones = None
    mul = torch.ops.aten.mul.Tensor(add, add);  add = None
    return mul
    """)

    def test_multi_out(self):
        def f(x):
            # aminmax.out returns a tuple of tensors.
            # functionalization should properly handle the tuple.
            out_min = torch.empty(4)
            out_max = torch.empty(4)
            torch.aminmax(x, dim=0, out=(out_max, out_min))
            return out_max
        self.assert_functionalization(f, torch.arange(8, dtype=torch.float32))
        logs = self.get_logs(f, torch.arange(8, dtype=torch.float32))
        self.assertExpectedInline(logs, """\



def forward(self, a_1):
    empty = torch.ops.aten.empty.memory_format([4], device = device(type='cpu'), pin_memory = False)
    empty_1 = torch.ops.aten.empty.memory_format([4], device = device(type='cpu'), pin_memory = False)
    aminmax = torch.ops.aten.aminmax.default(a_1, dim = 0);  a_1 = None
    getitem = aminmax[0]
    getitem_1 = aminmax[1];  aminmax = None
    return getitem
    """)

        reinplaced_logs = self.get_logs(f, torch.arange(8, dtype=torch.float32), reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(reinplaced_logs, """\



def forward(self, a_1):
    empty = torch.ops.aten.empty.memory_format([4], device = device(type='cpu'), pin_memory = False)
    empty_1 = torch.ops.aten.empty.memory_format([4], device = device(type='cpu'), pin_memory = False)
    aminmax = torch.ops.aten.aminmax.default(a_1, dim = 0);  a_1 = None
    getitem = aminmax[0]
    getitem_1 = aminmax[1];  aminmax = None
    return getitem
    """)

    def test_tensor_ctr(self):
        def f(x):
            y = torch.tensor((1, 2, 3))
            z = y.view(-1)
            z.add_(1)
            return y

        inpt = torch.arange(3, dtype=torch.float32)
        self.assert_functionalization(f, inpt)

        logs = self.get_logs(f, inpt)
        self.assertExpectedInline(logs, """\



def forward(self, a_1):
    _tensor_constant0 = self._tensor_constant0
    lift_fresh_copy = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
    view_copy = torch.ops.aten.view_copy.default(lift_fresh_copy, [-1]);  lift_fresh_copy = None
    add = torch.ops.aten.add.Tensor(view_copy, 1);  view_copy = None
    view_copy_1 = torch.ops.aten.view_copy.default(add, [3]);  add = None
    return view_copy_1
    """)

        reinplaced_logs = self.get_logs(f, inpt, reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(reinplaced_logs, """\



def forward(self, a_1):
    _tensor_constant0 = self._tensor_constant0
    lift_fresh_copy = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
    view = torch.ops.aten.view.default(lift_fresh_copy, [-1]);  lift_fresh_copy = None
    add = torch.ops.aten.add_.Tensor(view, 1)
    view_1 = torch.ops.aten.view.default(view, [3]);  view = None
    return view_1
    """)


    def test_tensor_list_mixed_functional_nonfunctional(self):
        nonfunctional_tensor = torch.ones(2, dtype=torch.long)

        def f(x):
            # simple test: 1 view op, 1 inplace op
            functional_tensor = torch.ones(2, dtype=torch.long)
            out = x[functional_tensor, nonfunctional_tensor]
            return out
        out = f(torch.ones(2, 2))
        out_functional = _functionalize(f, reapply_views=True)(torch.ones(2, 2))
        self.assertEqual(out, out_functional)

    def test_inplace_on_non_view(self):
        def f(x):
            # test for the case where we functionalize an inplace op on the other tensor - not a view.
            # This is worth checking because the tensor will have an empty ViewMeta stack, which needs to be special cased.
            tmp = torch.ones(4, 2)
            y = x.view(4, 2)
            x.add_(tmp)
            return y
        self.assert_functionalization(f, torch.ones(4, 2))
        logs = self.get_logs(f, torch.ones(4, 2))
        self.assertExpectedInline(logs, """\



def forward(self, a_1):
    ones = torch.ops.aten.ones.default([4, 2], device = device(type='cpu'), pin_memory = False)
    view_copy = torch.ops.aten.view_copy.default(a_1, [4, 2])
    add = torch.ops.aten.add.Tensor(a_1, ones);  ones = None
    copy_ = torch.ops.aten.copy_.default(a_1, add);  a_1 = None
    view_copy_1 = torch.ops.aten.view_copy.default(add, [4, 2]);  add = None
    return view_copy_1
    """)

        reinplaced_logs = self.get_logs(f, torch.ones(4, 2), reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(reinplaced_logs, """\



def forward(self, a_1):
    ones = torch.ops.aten.ones.default([4, 2], device = device(type='cpu'), pin_memory = False)
    view = torch.ops.aten.view.default(a_1, [4, 2])
    add = torch.ops.aten.add.Tensor(a_1, ones);  ones = None
    copy_ = torch.ops.aten.copy_.default(a_1, add);  a_1 = None
    view_1 = torch.ops.aten.view.default(add, [4, 2]);  add = None
    return view_1
    """)

    # Some ops that are mutable are neither inplace nor out= ops.
    # They also need special handling.
    def test_mutable_op_not_inplace_or_other(self):
        def f(x):
            return torch._fused_moving_avg_obs_fq_helper(x, x, x, x, x, x, x, 1.0, 0, 1, 0)

        logs = self.get_logs(f, torch.ones(1))
        self.assertExpectedInline(logs, """\



def forward(self, a_1):
    _fused_moving_avg_obs_fq_helper_functional = torch.ops.aten._fused_moving_avg_obs_fq_helper_functional.default(a_1, a_1, a_1, a_1, a_1, a_1, a_1, 1.0, 0, 1, 0)
    getitem = _fused_moving_avg_obs_fq_helper_functional[0]
    getitem_1 = _fused_moving_avg_obs_fq_helper_functional[1]
    getitem_2 = _fused_moving_avg_obs_fq_helper_functional[2]
    getitem_3 = _fused_moving_avg_obs_fq_helper_functional[3]
    getitem_4 = _fused_moving_avg_obs_fq_helper_functional[4]
    getitem_5 = _fused_moving_avg_obs_fq_helper_functional[5];  _fused_moving_avg_obs_fq_helper_functional = None
    copy_ = torch.ops.aten.copy_.default(a_1, getitem_5);  a_1 = getitem_5 = None
    return (getitem, getitem_1)
    """)  # noqa: B950

    def test_as_strided(self):
        def f(x):
            y = x.as_strided((2,), (2,), 1)
            y.add_(1)
            return x
        self.assert_functionalization(f, torch.ones(9))
        logs = self.get_logs(f, torch.ones(9))
        self.assertExpectedInline(logs, """\



def forward(self, a_1):
    as_strided_copy = torch.ops.aten.as_strided_copy.default(a_1, [2], [2], 1)
    add = torch.ops.aten.add.Tensor(as_strided_copy, 1);  as_strided_copy = None
    as_strided_scatter = torch.ops.aten.as_strided_scatter.default(a_1, add, [2], [2], 1);  add = None
    copy_ = torch.ops.aten.copy_.default(a_1, as_strided_scatter);  a_1 = None
    return as_strided_scatter
    """)

    def test_tensor_list_composite(self):
        def f(x):
            # Test an op with TensorList input
            y = torch.block_diag(x, x)
            return y
        self.assert_functionalization(f, torch.ones(2, 2))
        logs = self.get_logs(f, torch.ones(2, 2))
        self.assertExpectedInline(logs, """\



def forward(self, a_1):
    block_diag = torch.ops.aten.block_diag.default([a_1, a_1]);  a_1 = None
    return block_diag
    """)

    def test_cat(self):
        def f(x):
            out = torch.empty(0)
            torch.cat((x,), out=out)
            return out
        self.assert_functionalization(f, torch.ones(2, 2))
        logs = self.get_logs(f, torch.ones(2, 2))
        self.assertExpectedInline(logs, """\



def forward(self, a_1):
    empty = torch.ops.aten.empty.memory_format([0], device = device(type='cpu'), pin_memory = False)
    cat = torch.ops.aten.cat.default([a_1]);  a_1 = None
    return cat
    """)

        reinplaced_logs = self.get_logs(f, torch.ones(2, 2), reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(reinplaced_logs, """\



def forward(self, a_1):
    empty = torch.ops.aten.empty.memory_format([0], device = device(type='cpu'), pin_memory = False)
    cat = torch.ops.aten.cat.default([a_1]);  a_1 = None
    return cat
    """)


    def test_diagonal(self):
        def f(x):
            # test: view ops that take a subset of the original tensor (select/diagonal)
            tmp = torch.ones(2)
            y = x.clone().diagonal()
            y.add_(tmp)
            z = x * x
            return z
        self.assert_functionalization(f, torch.ones(2, 2))
        logs = self.get_logs(f, torch.ones(2, 2))
        self.assertExpectedInline(logs, """\



def forward(self, a_1):
    ones = torch.ops.aten.ones.default([2], device = device(type='cpu'), pin_memory = False)
    clone = torch.ops.aten.clone.default(a_1)
    diagonal_copy = torch.ops.aten.diagonal_copy.default(clone);  clone = None
    add = torch.ops.aten.add.Tensor(diagonal_copy, ones);  diagonal_copy = ones = None
    mul = torch.ops.aten.mul.Tensor(a_1, a_1);  a_1 = None
    return mul
    """)

        reinplaced_logs = self.get_logs(f, torch.ones(2, 2), reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(reinplaced_logs, """\



def forward(self, a_1):
    ones = torch.ops.aten.ones.default([2], device = device(type='cpu'), pin_memory = False)
    clone = torch.ops.aten.clone.default(a_1)
    diagonal = torch.ops.aten.diagonal.default(clone);  clone = None
    add = torch.ops.aten.add_.Tensor(diagonal, ones);  diagonal = ones = None
    mul = torch.ops.aten.mul.Tensor(a_1, a_1);  a_1 = None
    return mul
    """)

    def test_diagonal_mutated_input(self):
        def f(x):
            # simple test: there are pending updates afterwards, which the test syncs manually
            tmp = torch.ones(2)
            y = x.diagonal()
            y.add_(tmp)
            return x
        x = torch.ones(2, 2)
        self.assert_functionalization(f, x)
        logs = self.get_logs(f, torch.ones(2, 2))
        self.assertExpectedInline(logs, """\



def forward(self, a_1):
    ones = torch.ops.aten.ones.default([2], device = device(type='cpu'), pin_memory = False)
    diagonal_copy = torch.ops.aten.diagonal_copy.default(a_1)
    add = torch.ops.aten.add.Tensor(diagonal_copy, ones);  diagonal_copy = ones = None
    diagonal_scatter = torch.ops.aten.diagonal_scatter.default(a_1, add);  add = None
    copy_ = torch.ops.aten.copy_.default(a_1, diagonal_scatter);  a_1 = None
    return diagonal_scatter
    """)

    def test_split(self):
        def f(x):
            # test: view ops that return multiple tensors (split)
            tmp = torch.ones(2)
            y1, y2 = x.split(2)
            y3 = y2.diagonal()
            y3.add_(tmp)
            z = x * x
            return y3
        self.assert_functionalization(f, torch.ones(4, 2))
        logs = self.get_logs(f, torch.ones(4, 2))
        self.assertExpectedInline(logs, """\



def forward(self, a_1):
    ones = torch.ops.aten.ones.default([2], device = device(type='cpu'), pin_memory = False)
    split_copy = torch.ops.aten.split_copy.Tensor(a_1, 2)
    getitem = split_copy[0]
    getitem_1 = split_copy[1];  split_copy = None
    diagonal_copy = torch.ops.aten.diagonal_copy.default(getitem_1);  getitem_1 = None
    add = torch.ops.aten.add.Tensor(diagonal_copy, ones);  diagonal_copy = ones = None
    split_copy_1 = torch.ops.aten.split_copy.Tensor(a_1, 2)
    getitem_2 = split_copy_1[0]
    getitem_3 = split_copy_1[1];  split_copy_1 = None
    diagonal_scatter = torch.ops.aten.diagonal_scatter.default(getitem_3, add);  getitem_3 = None
    slice_scatter = torch.ops.aten.slice_scatter.default(a_1, diagonal_scatter, 0, 2, 4);  diagonal_scatter = None
    mul = torch.ops.aten.mul.Tensor(slice_scatter, slice_scatter)
    copy_ = torch.ops.aten.copy_.default(a_1, slice_scatter);  a_1 = slice_scatter = None
    return add
    """)  # noqa: B950

    def test_view_inplace(self):
        def f(x):
            # test: view + inplace op (transpose_)
            tmp = torch.ones(4)
            x.transpose_(1, 0)
            y = x[0]
            y.add_(tmp)
            return x
        self.assert_functionalization(f, torch.ones(4, 2), mutated_input_metadata=True)
        logs = self.get_logs(f, torch.ones(4, 2))
        self.assertExpectedInline(logs, """\



def forward(self, a_1):
    ones = torch.ops.aten.ones.default([4], device = device(type='cpu'), pin_memory = False)
    transpose_copy = torch.ops.aten.transpose_copy.int(a_1, 1, 0)
    select_copy = torch.ops.aten.select_copy.int(transpose_copy, 0, 0);  transpose_copy = None
    add = torch.ops.aten.add.Tensor(select_copy, ones);  select_copy = ones = None
    transpose_copy_1 = torch.ops.aten.transpose_copy.int(a_1, 1, 0);  a_1 = None
    select_scatter = torch.ops.aten.select_scatter.default(transpose_copy_1, add, 0, 0);  transpose_copy_1 = add = None
    transpose_copy_2 = torch.ops.aten.transpose_copy.int(select_scatter, 1, 0);  select_scatter = None
    transpose_copy_3 = torch.ops.aten.transpose_copy.int(transpose_copy_2, 1, 0);  transpose_copy_2 = None
    return transpose_copy_3
    """)  # noqa: B950

    def test_optional_tensor_list(self):
        def f(x):
            # test: an operator that takes in a List[Optional[Tensor]] argument
            # (index_put)
            y = x.view(8)
            indices = torch.arange(4)
            values = torch.arange(4, dtype=y.dtype)
            y.index_put_((indices,), values, accumulate=False)
            return y
        self.assert_functionalization(f, torch.ones(4, 2))
        logs = self.get_logs(f, torch.ones(4, 2))
        self.assertExpectedInline(logs, """\



def forward(self, a_1):
    view_copy = torch.ops.aten.view_copy.default(a_1, [8])
    arange = torch.ops.aten.arange.default(4, device = device(type='cpu'), pin_memory = False)
    arange_1 = torch.ops.aten.arange.default(4, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    index_put = torch.ops.aten.index_put.default(view_copy, [arange], arange_1);  view_copy = arange = arange_1 = None
    view_copy_1 = torch.ops.aten.view_copy.default(index_put, [4, 2])
    copy_ = torch.ops.aten.copy_.default(a_1, view_copy_1);  a_1 = view_copy_1 = None
    return index_put
    """)  # noqa: B950

    def test_scalars(self):
        def f(x):
            # test: the pass can handle scalar inputs properly
            tmp = torch.ones(4, 2)
            y = x.view(4, 2)
            y.add_(1)
            z = 2 * y
            z.div_(1)
            return z
        self.assert_functionalization(f, torch.ones(4, 2))
        logs = self.get_logs(f, torch.ones(4, 2))
        self.assertExpectedInline(logs, """\



def forward(self, a_1):
    ones = torch.ops.aten.ones.default([4, 2], device = device(type='cpu'), pin_memory = False)
    view_copy = torch.ops.aten.view_copy.default(a_1, [4, 2])
    add = torch.ops.aten.add.Tensor(view_copy, 1);  view_copy = None
    mul = torch.ops.aten.mul.Tensor(add, 2)
    div = torch.ops.aten.div.Tensor(mul, 1);  mul = None
    view_copy_1 = torch.ops.aten.view_copy.default(add, [4, 2]);  add = None
    copy_ = torch.ops.aten.copy_.default(a_1, view_copy_1);  a_1 = view_copy_1 = None
    return div
    """)

    @skipIfTorchDynamo("Test does not work with TorchDynamo")
    def test_metadata_change(self):
        def f(x):
            # ops like ge_() are allowed to change the dtype of the input.
            # functionalization should pick up on that.
            y = x.clone()
            out = y.ge_(0)
            return out
        self.assert_functionalization(f, torch.ones(4, 2))
        logs = self.get_logs(f, torch.ones(4, 2))
        self.assertExpectedInline(logs, """\



def forward(self, a_1):
    clone = torch.ops.aten.clone.default(a_1);  a_1 = None
    ge = torch.ops.aten.ge.Scalar(clone, 0);  clone = None
    _to_copy = torch.ops.aten._to_copy.default(ge, dtype = torch.float32, layout = torch.strided);  ge = None
    return _to_copy
    """)

        reinplaced_logs = self.get_logs(f, torch.ones(2, 2), reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(reinplaced_logs, """\



def forward(self, a_1):
    clone = torch.ops.aten.clone.default(a_1);  a_1 = None
    ge = torch.ops.aten.ge.Scalar(clone, 0);  clone = None
    _to_copy = torch.ops.aten._to_copy.default(ge, dtype = torch.float32, layout = torch.strided);  ge = None
    return _to_copy
    """)  # noqa: B950

    @skipIfTorchDynamo("Test does not work with TorchDynamo")
    def test_metadata_change_out_op(self):
        def f(t, y):
            out_1 = torch.ones(1)
            return torch.add(t, y, out=out_1)

        inpt1, inpt2 = torch.tensor([1]), torch.tensor([1])
        inpt1_func, inpt2_func = torch._to_functional_tensor(inpt1), torch._to_functional_tensor(inpt2)

        out_ref = f(inpt1, inpt2)
        torch._enable_functionalization(reapply_views=True)
        try:
            out_functional = f(inpt1_func, inpt2_func)
        finally:
            torch._disable_functionalization()
        self.assertEqual(out_ref, torch._from_functional_tensor(out_functional))


    def test_only_one_view(self):
        def f(x):
            # This tests that we don't have any unnecessary views in the trace.
            # If the input wasn't mutated, we don't need to regenerate it,
            # so there should be a total of 1 op in the output trace.
            return x.view(4, 2)
        logs = self.get_logs(f, torch.ones(4, 2))
        self.assertExpectedInline(logs, """\



def forward(self, a_1):
    view_copy = torch.ops.aten.view_copy.default(a_1, [4, 2]);  a_1 = None
    return view_copy
    """)

    def test_everything(self):
        def f(x):
            # test: everything
            tmp = torch.ones(2, 2)
            x2 = x + x
            y = x2.view(8)
            z0 = y.reshape(2, 4)
            z1 = z0.transpose(1, 0)
            z1.unsqueeze_(0)
            z1.squeeze_()
            z2, z3 = z1.split(2)
            z2.add_(tmp)
            z4 = z0[0] + z2.reshape(4)
            return z2
        self.assert_functionalization(f, torch.ones(4, 2))
        logs = self.get_logs(f, torch.ones(4, 2))
        self.assertExpectedInline(logs, """\



def forward(self, a_1):
    ones = torch.ops.aten.ones.default([2, 2], device = device(type='cpu'), pin_memory = False)
    add = torch.ops.aten.add.Tensor(a_1, a_1);  a_1 = None
    view_copy = torch.ops.aten.view_copy.default(add, [8])
    _reshape_alias_copy = torch.ops.aten._reshape_alias_copy.default(view_copy, [2, 4], [4, 1]);  view_copy = None
    transpose_copy = torch.ops.aten.transpose_copy.int(_reshape_alias_copy, 1, 0)
    unsqueeze_copy = torch.ops.aten.unsqueeze_copy.default(transpose_copy, 0);  transpose_copy = None
    squeeze_copy = torch.ops.aten.squeeze_copy.default(unsqueeze_copy);  unsqueeze_copy = None
    split_copy = torch.ops.aten.split_copy.Tensor(squeeze_copy, 2);  squeeze_copy = None
    getitem = split_copy[0]
    getitem_1 = split_copy[1];  split_copy = None
    add_1 = torch.ops.aten.add.Tensor(getitem, ones);  getitem = ones = None
    select_copy = torch.ops.aten.select_copy.int(_reshape_alias_copy, 0, 0);  _reshape_alias_copy = None
    _reshape_alias_copy_1 = torch.ops.aten._reshape_alias_copy.default(add_1, [4], [1])
    view_copy_1 = torch.ops.aten.view_copy.default(add, [8]);  add = None
    _reshape_alias_copy_2 = torch.ops.aten._reshape_alias_copy.default(view_copy_1, [2, 4], [4, 1]);  view_copy_1 = None
    transpose_copy_1 = torch.ops.aten.transpose_copy.int(_reshape_alias_copy_2, 1, 0);  _reshape_alias_copy_2 = None
    unsqueeze_copy_1 = torch.ops.aten.unsqueeze_copy.default(transpose_copy_1, 0);  transpose_copy_1 = None
    squeeze_copy_1 = torch.ops.aten.squeeze_copy.default(unsqueeze_copy_1);  unsqueeze_copy_1 = None
    slice_scatter = torch.ops.aten.slice_scatter.default(squeeze_copy_1, add_1, 0, 0, 2);  squeeze_copy_1 = None
    unsqueeze_copy_2 = torch.ops.aten.unsqueeze_copy.default(slice_scatter, 0);  slice_scatter = None
    squeeze_copy_2 = torch.ops.aten.squeeze_copy.dim(unsqueeze_copy_2, 0);  unsqueeze_copy_2 = None
    transpose_copy_2 = torch.ops.aten.transpose_copy.int(squeeze_copy_2, 1, 0);  squeeze_copy_2 = None
    _reshape_alias_copy_3 = torch.ops.aten._reshape_alias_copy.default(transpose_copy_2, [8], [1]);  transpose_copy_2 = None
    view_copy_2 = torch.ops.aten.view_copy.default(_reshape_alias_copy_3, [4, 2]);  _reshape_alias_copy_3 = None
    view_copy_3 = torch.ops.aten.view_copy.default(view_copy_2, [8])
    _reshape_alias_copy_4 = torch.ops.aten._reshape_alias_copy.default(view_copy_3, [2, 4], [4, 1]);  view_copy_3 = None
    select_copy_1 = torch.ops.aten.select_copy.int(_reshape_alias_copy_4, 0, 0);  _reshape_alias_copy_4 = None
    view_copy_4 = torch.ops.aten.view_copy.default(view_copy_2, [8]);  view_copy_2 = None
    _reshape_alias_copy_5 = torch.ops.aten._reshape_alias_copy.default(view_copy_4, [2, 4], [4, 1]);  view_copy_4 = None
    transpose_copy_3 = torch.ops.aten.transpose_copy.int(_reshape_alias_copy_5, 1, 0);  _reshape_alias_copy_5 = None
    unsqueeze_copy_3 = torch.ops.aten.unsqueeze_copy.default(transpose_copy_3, 0);  transpose_copy_3 = None
    squeeze_copy_3 = torch.ops.aten.squeeze_copy.default(unsqueeze_copy_3);  unsqueeze_copy_3 = None
    split_copy_1 = torch.ops.aten.split_copy.Tensor(squeeze_copy_3, 2);  squeeze_copy_3 = None
    getitem_2 = split_copy_1[0]
    getitem_3 = split_copy_1[1];  split_copy_1 = None
    _reshape_alias_copy_6 = torch.ops.aten._reshape_alias_copy.default(getitem_2, [4], [1]);  getitem_2 = None
    add_2 = torch.ops.aten.add.Tensor(select_copy_1, _reshape_alias_copy_6);  select_copy_1 = _reshape_alias_copy_6 = None
    return add_1
    """)  # noqa: B950

        reinplaced_logs = self.get_logs(f, torch.ones(4, 2), reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(reinplaced_logs, """\



def forward(self, a_1):
    ones = torch.ops.aten.ones.default([2, 2], device = device(type='cpu'), pin_memory = False)
    add = torch.ops.aten.add.Tensor(a_1, a_1);  a_1 = None
    view = torch.ops.aten.view.default(add, [8])
    _reshape_alias = torch.ops.aten._reshape_alias.default(view, [2, 4], [4, 1]);  view = None
    transpose = torch.ops.aten.transpose.int(_reshape_alias, 1, 0)
    unsqueeze = torch.ops.aten.unsqueeze.default(transpose, 0);  transpose = None
    squeeze = torch.ops.aten.squeeze.default(unsqueeze);  unsqueeze = None
    split = torch.ops.aten.split.Tensor(squeeze, 2);  squeeze = None
    getitem = split[0]
    getitem_1 = split[1];  split = None
    add_1 = torch.ops.aten.add_.Tensor(getitem, ones);  ones = None
    select = torch.ops.aten.select.int(_reshape_alias, 0, 0);  _reshape_alias = None
    clone = torch.ops.aten.clone.default(getitem, memory_format = torch.contiguous_format)
    _unsafe_view = torch.ops.aten._unsafe_view.default(clone, [4]);  clone = None
    view_1 = torch.ops.aten.view.default(add, [8]);  add = None
    _reshape_alias_1 = torch.ops.aten._reshape_alias.default(view_1, [2, 4], [4, 1]);  view_1 = None
    transpose_1 = torch.ops.aten.transpose.int(_reshape_alias_1, 1, 0);  _reshape_alias_1 = None
    unsqueeze_1 = torch.ops.aten.unsqueeze.default(transpose_1, 0);  transpose_1 = None
    squeeze_1 = torch.ops.aten.squeeze.default(unsqueeze_1);  unsqueeze_1 = None
    unsqueeze_2 = torch.ops.aten.unsqueeze.default(squeeze_1, 0);  squeeze_1 = None
    squeeze_2 = torch.ops.aten.squeeze.dim(unsqueeze_2, 0);  unsqueeze_2 = None
    transpose_2 = torch.ops.aten.transpose.int(squeeze_2, 1, 0);  squeeze_2 = None
    _reshape_alias_2 = torch.ops.aten._reshape_alias.default(transpose_2, [8], [1]);  transpose_2 = None
    view_2 = torch.ops.aten.view.default(_reshape_alias_2, [4, 2]);  _reshape_alias_2 = None
    view_3 = torch.ops.aten.view.default(view_2, [8]);  view_2 = None
    _reshape_alias_3 = torch.ops.aten._reshape_alias.default(view_3, [2, 4], [4, 1]);  view_3 = None
    select_1 = torch.ops.aten.select.int(_reshape_alias_3, 0, 0);  _reshape_alias_3 = None
    add_2 = torch.ops.aten.add.Tensor(select_1, _unsafe_view);  select_1 = _unsafe_view = None
    return getitem
    """)

    def test_reapply_views_simple(self):
        def f(x):
            tmp = torch.ones(4, 2)
            y = x.view(4, 2)
            y.add_(tmp)
            z = x * x
            return y
        self.assert_functionalization(f, torch.ones(4, 2), reapply_views=True)
        logs = self.get_logs(f, torch.ones(4, 2), reapply_views=True)
        self.assertExpectedInline(logs, """\



def forward(self, a_1):
    ones = torch.ops.aten.ones.default([4, 2], device = device(type='cpu'), pin_memory = False)
    view = torch.ops.aten.view.default(a_1, [4, 2])
    add = torch.ops.aten.add.Tensor(view, ones);  view = ones = None
    view_1 = torch.ops.aten.view.default(add, [4, 2])
    mul = torch.ops.aten.mul.Tensor(view_1, view_1)
    copy_ = torch.ops.aten.copy_.default(a_1, view_1);  a_1 = view_1 = None
    return add
    """)

    def test_aliases_maintained_after_pass_when_reapplying_views(self):
        def f(x):
            tmp = torch.ones(4, 2)
            y = x.view(4, 2)
            z = x.view(4, 2)
            y.add_(tmp)
            return y, z

        input_functional = torch._to_functional_tensor(torch.ones(4, 2))
        torch._enable_functionalization(reapply_views=True)
        try:
            y, z = f(input_functional)
            torch._sync(y)
            torch._sync(z)
        finally:
            torch._disable_functionalization()

        # y and z are aliases inside of the function, and that aliasing relationship should be maintained.
        _y = torch._from_functional_tensor(y)
        _z = torch._from_functional_tensor(z)
        self.assertTrue(are_aliased(_y, _z))

    # copy_() gets its own test, because it is special cased in functionalization.
    # self.copy_(src) decomposes into src.to(self).expand_as(self).
    def test_copy_(self):
        def f(x):
            tmp = torch.zeros(2, 2)
            tmp_slice = tmp.diagonal()
            y = tmp_slice.copy_(x)
            z = y.add_(x)
            return z

        # Test 1: copy_() with same dtype and shape
        # to() is a composite op that noops when the dtype/shape match, so nothing gets logged.
        # self.assert_functionalization(f, torch.ones(2))
        logs = self.get_logs(f, torch.ones(2))
        self.assertExpectedInline(logs, """\



def forward(self, a_1):
    zeros = torch.ops.aten.zeros.default([2, 2], device = device(type='cpu'), pin_memory = False)
    diagonal_copy = torch.ops.aten.diagonal_copy.default(zeros);  zeros = None
    add = torch.ops.aten.add.Tensor(a_1, a_1);  a_1 = None
    return add
    """)

        reinplaced_logs = self.get_logs(f, torch.ones(2), reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(reinplaced_logs, """\



def forward(self, a_1):
    zeros = torch.ops.aten.zeros.default([2, 2], device = device(type='cpu'), pin_memory = False)
    diagonal = torch.ops.aten.diagonal.default(zeros);  zeros = None
    add = torch.ops.aten.add.Tensor(a_1, a_1);  a_1 = None
    return add
    """)

        # Test 2: copy_() with same dtype, different shape
        self.assert_functionalization(f, torch.ones(1))
        logs = self.get_logs(f, torch.ones(1))
        self.assertExpectedInline(logs, """\



def forward(self, a_1):
    zeros = torch.ops.aten.zeros.default([2, 2], device = device(type='cpu'), pin_memory = False)
    diagonal_copy = torch.ops.aten.diagonal_copy.default(zeros);  zeros = None
    expand_copy = torch.ops.aten.expand_copy.default(a_1, [2])
    add = torch.ops.aten.add.Tensor(expand_copy, a_1);  expand_copy = a_1 = None
    return add
    """)

        reinplaced_logs = self.get_logs(f, torch.ones(1), reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(reinplaced_logs, """\



def forward(self, a_1):
    zeros = torch.ops.aten.zeros.default([2, 2], device = device(type='cpu'), pin_memory = False)
    diagonal = torch.ops.aten.diagonal.default(zeros);  zeros = None
    expand_copy = torch.ops.aten.expand_copy.default(a_1, [2])
    add = torch.ops.aten.add_.Tensor(expand_copy, a_1);  a_1 = None
    return expand_copy
    """)

        # Test 3: copy_() with different dtype, same shape
        self.assert_functionalization(f, torch.ones(2, dtype=torch.long))
        logs = self.get_logs(f, torch.ones(2, dtype=torch.long))
        self.assertExpectedInline(logs, """\



def forward(self, a_1):
    zeros = torch.ops.aten.zeros.default([2, 2], device = device(type='cpu'), pin_memory = False)
    diagonal_copy = torch.ops.aten.diagonal_copy.default(zeros);  zeros = None
    _to_copy = torch.ops.aten._to_copy.default(a_1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    add = torch.ops.aten.add.Tensor(_to_copy, a_1);  _to_copy = a_1 = None
    return add
    """)  # noqa: B950

        reinplaced_logs = self.get_logs(f, torch.ones(2, dtype=torch.long), reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(reinplaced_logs, """\



def forward(self, a_1):
    zeros = torch.ops.aten.zeros.default([2, 2], device = device(type='cpu'), pin_memory = False)
    diagonal = torch.ops.aten.diagonal.default(zeros);  zeros = None
    _to_copy = torch.ops.aten._to_copy.default(a_1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    add = torch.ops.aten.add_.Tensor(_to_copy, a_1);  a_1 = None
    return _to_copy
    """)  # noqa: B950

        # Test 4: copy_() with different dtype, different shape
        self.assert_functionalization(f, torch.ones(1, dtype=torch.long))
        logs = self.get_logs(f, torch.ones(1, dtype=torch.long))
        self.assertExpectedInline(logs, """\



def forward(self, a_1):
    zeros = torch.ops.aten.zeros.default([2, 2], device = device(type='cpu'), pin_memory = False)
    diagonal_copy = torch.ops.aten.diagonal_copy.default(zeros);  zeros = None
    _to_copy = torch.ops.aten._to_copy.default(a_1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    expand_copy = torch.ops.aten.expand_copy.default(_to_copy, [2]);  _to_copy = None
    add = torch.ops.aten.add.Tensor(expand_copy, a_1);  expand_copy = a_1 = None
    return add
    """)  # noqa: B950

        reinplaced_logs = self.get_logs(f, torch.ones(1, dtype=torch.long), reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(reinplaced_logs, """\



def forward(self, a_1):
    zeros = torch.ops.aten.zeros.default([2, 2], device = device(type='cpu'), pin_memory = False)
    diagonal = torch.ops.aten.diagonal.default(zeros);  zeros = None
    _to_copy = torch.ops.aten._to_copy.default(a_1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    expand_copy = torch.ops.aten.expand_copy.default(_to_copy, [2]);  _to_copy = None
    add = torch.ops.aten.add_.Tensor(expand_copy, a_1);  a_1 = None
    return expand_copy
    """)  # noqa: B950

    def test_expand_symint(self):
        # Once some existing SymInt bugs are ironed out, we should update
        # this test to plumb FakeSymbolicTensors through it
        def f(x):
            return x.expand(x.size(0), x.size(1))

        self.assert_functionalization(f, torch.ones(2, 2))
        logs = self.get_logs(f, torch.ones(2, 2))
        self.assertExpectedInline(logs, """\



def forward(self, a_1):
    expand_copy = torch.ops.aten.expand_copy.default(a_1, [2, 2]);  a_1 = None
    return expand_copy
    """)

    def test_fill_(self):
        def f(x):
            y = x + x
            z = y.diagonal()
            z.fill_(0)
            return y

        self.assert_functionalization(f, torch.ones(2, 2))
        logs = self.get_logs(f, torch.ones(2, 2))
        self.assertExpectedInline(logs, """\



def forward(self, a_1):
    add = torch.ops.aten.add.Tensor(a_1, a_1);  a_1 = None
    diagonal_copy = torch.ops.aten.diagonal_copy.default(add)
    fill = torch.ops.aten.fill.Scalar(diagonal_copy, 0);  diagonal_copy = None
    diagonal_scatter = torch.ops.aten.diagonal_scatter.default(add, fill);  add = fill = None
    return diagonal_scatter
    """)

        reinplaced_logs = self.get_logs(f, torch.ones(2, 2), reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(reinplaced_logs, """\



def forward(self, a_1):
    add = torch.ops.aten.add.Tensor(a_1, a_1);  a_1 = None
    diagonal = torch.ops.aten.diagonal.default(add)
    fill = torch.ops.aten.fill_.Scalar(diagonal, 0);  diagonal = None
    return add
    """)

    def test_resize_smaller(self):
        def f(w):
            # Resizing to a smaller size doesn't affect storage
            x = w + 1
            y = x.view(4, 4)
            y.resize_(3, 3)
            y2 = y.view(-1)
            y2.add_(1)
            z = y + 1
            return z

        self.assert_functionalization(f, torch.ones(8, 2))
        logs = self.get_logs(f, torch.ones(8, 2))
        self.assertExpectedInline(logs, """\



def forward(self, a_1):
    add = torch.ops.aten.add.Tensor(a_1, 1);  a_1 = None
    view_copy = torch.ops.aten.view_copy.default(add, [4, 4])
    resize = torch.ops.aten.resize.default(view_copy, [3, 3])
    as_strided_copy = torch.ops.aten.as_strided_copy.default(view_copy, [3, 3], [3, 1]);  view_copy = None
    view_copy_1 = torch.ops.aten.view_copy.default(as_strided_copy, [-1]);  as_strided_copy = None
    add_1 = torch.ops.aten.add.Tensor(view_copy_1, 1);  view_copy_1 = None
    view_copy_2 = torch.ops.aten.view_copy.default(add, [4, 4]);  add = None
    as_strided_copy_1 = torch.ops.aten.as_strided_copy.default(view_copy_2, [3, 3], [3, 1])
    view_copy_3 = torch.ops.aten.view_copy.default(add_1, [3, 3]);  add_1 = None
    as_strided_scatter = torch.ops.aten.as_strided_scatter.default(view_copy_2, view_copy_3, [3, 3], [3, 1]);  view_copy_2 = view_copy_3 = None
    view_copy_4 = torch.ops.aten.view_copy.default(as_strided_scatter, [8, 2]);  as_strided_scatter = None
    view_copy_5 = torch.ops.aten.view_copy.default(view_copy_4, [4, 4]);  view_copy_4 = None
    as_strided_copy_2 = torch.ops.aten.as_strided_copy.default(view_copy_5, [3, 3], [3, 1]);  view_copy_5 = None
    add_2 = torch.ops.aten.add.Tensor(as_strided_copy_2, 1);  as_strided_copy_2 = None
    return add_2
    """)  # noqa: B950

        reinplaced_logs = self.get_logs(f, torch.ones(8, 2), reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(reinplaced_logs, """\



def forward(self, a_1):
    add = torch.ops.aten.add.Tensor(a_1, 1);  a_1 = None
    view = torch.ops.aten.view.default(add, [4, 4])
    resize = torch.ops.aten.resize.default(view, [3, 3])
    as_strided = torch.ops.aten.as_strided.default(view, [3, 3], [3, 1]);  view = None
    view_1 = torch.ops.aten.view.default(as_strided, [-1]);  as_strided = None
    add_1 = torch.ops.aten.add_.Tensor(view_1, 1)
    view_2 = torch.ops.aten.view.default(add, [4, 4]);  add = None
    as_strided_1 = torch.ops.aten.as_strided.default(view_2, [3, 3], [3, 1])
    view_3 = torch.ops.aten.view.default(view_1, [3, 3]);  view_1 = None
    view_4 = torch.ops.aten.view.default(view_2, [8, 2]);  view_2 = None
    view_5 = torch.ops.aten.view.default(view_4, [4, 4]);  view_4 = None
    as_strided_2 = torch.ops.aten.as_strided.default(view_5, [3, 3], [3, 1]);  view_5 = None
    add_2 = torch.ops.aten.add_.Tensor(as_strided_2, 1)
    return as_strided_2
    """)

    def test_resize_larger_valid(self):
        def f(x):
            y = x + 1
            # resizing a tensor to a larger size is only currently allowed
            # if the tensor-to-resize is not a view / has no outstanding views.
            # See Note [resize_() in functionalization pass]
            y.resize_(5, 5)
            y2 = y.view(25)
            # Do a mutation to ensure that aliases of the output of resize_()
            # propagate mutations correctly.
            # I'm using fill_ specifically because I want to guarantee that
            # none of the output has uninitialized memory at the end
            # (since these tests compare the data output against a reference impl)
            y2.fill_(1)
            out = y + 1
            return y, out

        self.assert_functionalization(f, torch.ones(8, 2))
        logs = self.get_logs(f, torch.ones(8, 2))
        self.assertExpectedInline(logs, """\



def forward(self, a_1):
    add = torch.ops.aten.add.Tensor(a_1, 1);  a_1 = None
    resize = torch.ops.aten.resize.default(add, [5, 5]);  add = None
    view_copy = torch.ops.aten.view_copy.default(resize, [25]);  resize = None
    fill = torch.ops.aten.fill.Scalar(view_copy, 1);  view_copy = None
    view_copy_1 = torch.ops.aten.view_copy.default(fill, [5, 5]);  fill = None
    add_1 = torch.ops.aten.add.Tensor(view_copy_1, 1)
    return (view_copy_1, add_1)
    """)

        reinplaced_logs = self.get_logs(f, torch.ones(8, 2), reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(reinplaced_logs, """\



def forward(self, a_1):
    add = torch.ops.aten.add.Tensor(a_1, 1);  a_1 = None
    resize = torch.ops.aten.resize_.default(add, [5, 5])
    view = torch.ops.aten.view.default(add, [25]);  add = None
    fill = torch.ops.aten.fill_.Scalar(view, 1)
    view_1 = torch.ops.aten.view.default(view, [5, 5]);  view = None
    add_1 = torch.ops.aten.add.Tensor(view_1, 1)
    return (view_1, add_1)
    """)

    def test_resize_larger_invalid(self):
        def f(x):
            y = x + 1
            z = y.view(4, 4)
            # resizing a tensor to a larger size is only currently allowed
            # if the tensor-to-resize is not a view / has no outstanding views.
            # See Note [resize_() in functionalization pass]
            # This should fail
            z.resize_(5, 5)
            z2 = z.view(25)
            z2.fill_(1)
            out = z + 1
            return y, out

        with self.assertRaisesRegex(
                RuntimeError,
                r'Attempted to resize a view tensor to a larger size. This is not allowed in the functionalization pass'):
            self.assert_functionalization(f, torch.ones(8, 2))

    def test_nested_functions_propagate_updates(self):
        def g(x):
            # Create a view of x
            y = x[0]
            y.add_(1)
            # The view, y, gets deallocated at the end of this function

        def f(x):
            # Calling g(x) should mutate x
            g(x)
            # We expect x to be synced here, even though the alias created in g() has been deallocated!
            y = x + x
            return y

        self.assert_functionalization(f, torch.ones(2, 2))

    def test_mixed_wrappers_valid(self):
        def f(x, y):
            z = x + y
            z.add_(1)
            return z

        x1_not_functional = LoggingTensor(torch.ones(4))
        x2_functional = torch._to_functional_tensor(LoggingTensor(torch.ones(4)))

        with capture_logs() as logs:
            y = f(x1_not_functional, x2_functional)

        # Make sure that functionalization ran the "+" kernel
        # with a functional + non-functional tensor, and wrapped the output appropriately.
        self.assertExpectedInline('\n'.join(logs), """\
$2 = torch._ops.aten.add.Tensor($0, $1)
$3 = torch._ops.aten.add.Tensor($2, 1)""")

    def test_mixed_wrappers_invalid(self):
        x1_not_functional = torch.ones(4)
        x2_functional = torch._to_functional_tensor(torch.ones(4))

        # When dealing with mixed functional + non functional tensors,
        # normal_tensor.add_(functional_tensor) is not valid
        # because normal_tensor would need to be "promoted" to a functional tensor.
        with self.assertRaises(RuntimeError):
            x1_not_functional.add_(x2_functional)

    def test_index_mutation_on_non_input(self):
        def f(x):
            tmp = torch.zeros(10)
            tmp[5].fill_(1)
            return tmp
        self.assert_functionalization(f, torch.ones(2))
        logs = self.get_logs(f, torch.ones(2))
        self.assertExpectedInline(logs, """\



def forward(self, a_1):
    zeros = torch.ops.aten.zeros.default([10], device = device(type='cpu'), pin_memory = False)
    select_copy = torch.ops.aten.select_copy.int(zeros, 0, 5)
    fill = torch.ops.aten.fill.Scalar(select_copy, 1);  select_copy = None
    select_scatter = torch.ops.aten.select_scatter.default(zeros, fill, 0, 5);  zeros = fill = None
    return select_scatter
    """)  # noqa: B950

        reinplaced_logs = self.get_logs(f, torch.ones(2), reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(reinplaced_logs, """\



def forward(self, a_1):
    zeros = torch.ops.aten.zeros.default([10], device = device(type='cpu'), pin_memory = False)
    select = torch.ops.aten.select.int(zeros, 0, 5)
    fill = torch.ops.aten.fill_.Scalar(select, 1);  select = None
    return zeros
    """)

if __name__ == '__main__':
    run_tests()
