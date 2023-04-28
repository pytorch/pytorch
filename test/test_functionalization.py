# Owner(s): ["module: codegen"]

import torch
from contextlib import nullcontext
from torch.testing._internal.common_utils import (
    TestCase, run_tests, skipIfTorchDynamo, TEST_WITH_TORCHDYNAMO, IS_WINDOWS,
    xfail_inherited_tests
)
from torch.testing._internal.logging_tensor import LoggingTensor, capture_logs
from torch.utils._pytree import tree_map, tree_map_only, tree_flatten
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.reinplace import reinplace
from torch._dispatch.python import enable_crossref_functionalize, enable_python_dispatcher
from torch.multiprocessing.reductions import StorageWeakRef

import unittest

def are_aliased(x, y):
    x_storage = StorageWeakRef(x.storage())
    y_storage = StorageWeakRef(y.storage())
    return x_storage == y_storage

# We can unify testing and use functionalize() here instead
# if/when functorch moves into core.
# This is basically a crappy version of `functionalize()`.
def _functionalize(f, *, reapply_views: bool, crossref: bool):
    def to_fun(t: torch.Tensor):
        func_t = torch._to_functional_tensor(t)
        func_t.requires_grad = t.requires_grad
        return func_t

    def wrapped(*inputs):
        ctx = nullcontext()
        if crossref:
            ctx = enable_crossref_functionalize()
        with ctx:
            inputs_functional = tree_map_only(torch.Tensor, to_fun, inputs)
            torch._enable_functionalization(reapply_views=reapply_views)
            try:
                out = f(*inputs_functional)
            finally:
                torch._disable_functionalization()
            flat_inputs, _ = tree_flatten(inputs)
            flat_inputs_functional, _ = tree_flatten(inputs_functional)
            for inpt, input_functional in zip(flat_inputs, flat_inputs_functional):
                torch._sync(input_functional)
                inpt_new = torch._from_functional_tensor(input_functional)
                if inpt_new is not inpt:
                    # Existing deficiency in functionalize():
                    # we don't correctly mutate input metadata (yet?)
                    if inpt_new.shape == inpt.shape:
                        inpt.copy_(inpt_new)
            tree_map(torch._sync, out)
            out_unwrapped = tree_map(torch._from_functional_tensor, out)
            return out_unwrapped

    return wrapped

@unittest.skipIf(TEST_WITH_TORCHDYNAMO, "https://github.com/pytorch/pytorch/issues/81457")
class TestFunctionalization(TestCase):

    crossref = False

    def get_logs(self, func, *inpts, reapply_views=False, run_reinplace=False):
        inpts_clone = tree_map_only(torch.Tensor, torch.clone, inpts)
        traced_f = make_fx(_functionalize(func, reapply_views=reapply_views, crossref=self.crossref))(*inpts)
        if run_reinplace:
            traced_f = reinplace(traced_f, *inpts_clone)
        return traced_f.code

    def assert_functionalization(self, func, *inpts, reapply_views=False, mutated_input_metadata=False):
        clones1 = tree_map_only(torch.Tensor, torch.clone, inpts)
        clones2 = tree_map_only(torch.Tensor, torch.clone, inpts)
        clones3 = tree_map_only(torch.Tensor, torch.clone, inpts)

        # Compare outputs (and mutated inputs), with and without functionalization.
        out_ref = func(*inpts)
        out_functional = _functionalize(func, reapply_views=reapply_views, crossref=self.crossref)(*clones1)

        # The reinplacing pass is only valid to run with reapply_views=True.
        functional_func = make_fx(_functionalize(func, reapply_views=True, crossref=self.crossref))(*clones2)
        reinplace_func = reinplace(functional_func, *clones2)

        # NOTE: for now, need to pass in fresh inputs here, because make_fx
        # will directly mutate the inputs that you trace with.
        # Once this is fixed we can clean this up.
        out_reinplace = reinplace_func(*clones3)

        # functionalize() deficiency: input metadata mutations aren't propagated properly,
        # so we just need to skip checks here for the tests that exercise that.
        if not mutated_input_metadata:
            flat_inpts, _ = tree_flatten(inpts)
            flat_clones1, _ = tree_flatten(clones1)
            flat_clones3, _ = tree_flatten(clones3)
            for inpt, input_clone, input_clone3 in zip(flat_inpts, flat_clones1, flat_clones3):
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

        _functionalize(f, reapply_views=True, crossref=self.crossref)(torch.ones(3, 3))

    def test_copy_stride_mismatch(self):
        def f(x):
            y = torch.empty_strided((2, 2), (5, 1))
            y.copy_(x)
            return y

        r = _functionalize(f, reapply_views=True, crossref=self.crossref)(torch.ones(2, 2))
        self.assertEqual(r.stride(), (5, 1))

    def test_set_(self):
        def f(x):
            y = torch.ones(2)
            y.set_(x.storage())
            return y

        # We should probaby get the crossref test to work,
        # but fixing it for Storage() objects is annoying.
        r = _functionalize(f, reapply_views=True, crossref=False)(torch.ones(2))
        self.assertEqual(str(r.device), 'cpu')

    def test_advanced_indexing(self):
        def f():
            x = torch.zeros(3, 3)
            idx = torch.tensor([0])
            val = torch.ones(3, 1)
            x[:, idx] = val
            return x

        self.assert_functionalization(f)

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
            from torch._functorch.aot_autograd import setup_stacktrace_preservation_hooks
            import torch.fx.traceback as fx_traceback
            setup_stacktrace_preservation_hooks([loss.grad_fn])
            with fx_traceback.preserve_node_meta():
                loss.backward()
            return x.grad

        with torch.autograd.detect_anomaly(check_nan=False):
            logs = self.get_logs(g, torch.ones(16, 64, 128, 128, requires_grad=True))
        self.assertExpectedInline(logs, """\



def forward(self, arg0_1):
    view_copy = torch.ops.aten.view_copy.default(arg0_1, [1, 1024, 128, 128]);  arg0_1 = None
    clone = torch.ops.aten.clone.default(view_copy);  view_copy = None
    view_copy_1 = torch.ops.aten.view_copy.default(clone, [16, 64, 128, 128])
    relu = torch.ops.aten.relu.default(view_copy_1);  view_copy_1 = None
    view_copy_2 = torch.ops.aten.view_copy.default(relu, [1, 1024, 128, 128]);  relu = None
    view_copy_3 = torch.ops.aten.view_copy.default(view_copy_2, [16, 64, 128, 128]);  view_copy_2 = None
    view_copy_4 = torch.ops.aten.view_copy.default(clone, [16, 64, 128, 128]);  clone = None
    sum_1 = torch.ops.aten.sum.default(view_copy_3)
    ones_like = torch.ops.aten.ones_like.default(sum_1, pin_memory = False, memory_format = torch.preserve_format);  sum_1 = None
    expand_copy = torch.ops.aten.expand_copy.default(ones_like, [16, 64, 128, 128]);  ones_like = None
    view_copy_5 = torch.ops.aten.view_copy.default(expand_copy, [1, 1024, 128, 128]);  expand_copy = None
    new_empty_strided = torch.ops.aten.new_empty_strided.default(view_copy_5, [1, 1024, 128, 128], [16777216, 16384, 128, 1])
    copy = torch.ops.aten.copy.default(new_empty_strided, view_copy_5);  new_empty_strided = view_copy_5 = None
    view_copy_6 = torch.ops.aten.view_copy.default(copy, [16, 64, 128, 128])
    view_copy_7 = torch.ops.aten.view_copy.default(copy, [16, 64, 128, 128])
    clone_1 = torch.ops.aten.clone.default(view_copy_7, memory_format = torch.contiguous_format)
    threshold_backward = torch.ops.aten.threshold_backward.default(clone_1, view_copy_3, 0);  clone_1 = view_copy_3 = None
    copy_1 = torch.ops.aten.copy.default(view_copy_7, threshold_backward);  view_copy_7 = threshold_backward = None
    view_copy_8 = torch.ops.aten.view_copy.default(copy_1, [1, 1024, 128, 128]);  copy_1 = None
    view_copy_9 = torch.ops.aten.view_copy.default(view_copy_8, [16, 64, 128, 128])
    view_copy_10 = torch.ops.aten.view_copy.default(copy, [16, 64, 128, 128]);  copy = None
    detach_copy = torch.ops.aten.detach_copy.default(view_copy_10);  view_copy_10 = None
    view_copy_11 = torch.ops.aten.view_copy.default(view_copy_8, [16, 64, 128, 128]);  view_copy_8 = None
    detach_copy_1 = torch.ops.aten.detach_copy.default(view_copy_11);  view_copy_11 = None
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



def forward(self, arg0_1):
    ones = torch.ops.aten.ones.default([4, 2], device = device(type='cpu'), pin_memory = False)
    view_copy = torch.ops.aten.view_copy.default(arg0_1, [4, 2])
    add = torch.ops.aten.add.Tensor(view_copy, ones);  view_copy = ones = None
    view_copy_1 = torch.ops.aten.view_copy.default(add, [4, 2]);  add = None
    view_copy_2 = torch.ops.aten.view_copy.default(view_copy_1, [4, 2])
    mul = torch.ops.aten.mul.Tensor(view_copy_1, view_copy_1)
    copy_ = torch.ops.aten.copy_.default(arg0_1, view_copy_1);  arg0_1 = view_copy_1 = None
    return view_copy_2
    """)

        reinplaced_logs = self.get_logs(f, torch.ones(4, 2), reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(reinplaced_logs, """\



def forward(self, arg0_1):
    ones = torch.ops.aten.ones.default([4, 2], device = device(type='cpu'), pin_memory = False)
    view = torch.ops.aten.view.default(arg0_1, [4, 2])
    add = torch.ops.aten.add.Tensor(view, ones);  view = ones = None
    view_1 = torch.ops.aten.view.default(add, [4, 2]);  add = None
    view_2 = torch.ops.aten.view.default(view_1, [4, 2])
    mul = torch.ops.aten.mul.Tensor(view_1, view_1)
    copy_ = torch.ops.aten.copy_.default(arg0_1, view_1);  arg0_1 = view_1 = None
    return view_2
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



def forward(self, arg0_1):
    ones = torch.ops.aten.ones.default([4, 2], device = device(type='cpu'), pin_memory = False)
    view_copy = torch.ops.aten.view_copy.default(arg0_1, [4, 2]);  arg0_1 = None
    empty = torch.ops.aten.empty.memory_format([], device = device(type='cpu'), pin_memory = False)
    add = torch.ops.aten.add.Tensor(view_copy, ones);  view_copy = ones = None
    mul = torch.ops.aten.mul.Tensor(add, add);  add = None
    return mul
    """)

        reinplaced_logs = self.get_logs(f, torch.ones(4, 2), reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(reinplaced_logs, """\



def forward(self, arg0_1):
    ones = torch.ops.aten.ones.default([4, 2], device = device(type='cpu'), pin_memory = False)
    view = torch.ops.aten.view.default(arg0_1, [4, 2]);  arg0_1 = None
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



def forward(self, arg0_1):
    empty = torch.ops.aten.empty.memory_format([4], device = device(type='cpu'), pin_memory = False)
    empty_1 = torch.ops.aten.empty.memory_format([4], device = device(type='cpu'), pin_memory = False)
    aminmax = torch.ops.aten.aminmax.default(arg0_1, dim = 0);  arg0_1 = None
    getitem = aminmax[0]
    getitem_1 = aminmax[1];  aminmax = None
    return getitem
    """)

        reinplaced_logs = self.get_logs(f, torch.arange(8, dtype=torch.float32), reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(reinplaced_logs, """\



def forward(self, arg0_1):
    empty = torch.ops.aten.empty.memory_format([4], device = device(type='cpu'), pin_memory = False)
    empty_1 = torch.ops.aten.empty.memory_format([4], device = device(type='cpu'), pin_memory = False)
    aminmax = torch.ops.aten.aminmax.default(arg0_1, dim = 0);  arg0_1 = None
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



def forward(self, arg0_1):
    _tensor_constant0 = self._tensor_constant0
    lift_fresh_copy = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
    view_copy = torch.ops.aten.view_copy.default(lift_fresh_copy, [-1]);  lift_fresh_copy = None
    add = torch.ops.aten.add.Tensor(view_copy, 1);  view_copy = None
    view_copy_1 = torch.ops.aten.view_copy.default(add, [3]);  add = None
    view_copy_2 = torch.ops.aten.view_copy.default(view_copy_1, [-1])
    return view_copy_1
    """)

        reinplaced_logs = self.get_logs(f, inpt, reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(reinplaced_logs, """\



def forward(self, arg0_1):
    _tensor_constant0 = self._tensor_constant0
    lift_fresh_copy = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
    view = torch.ops.aten.view.default(lift_fresh_copy, [-1]);  lift_fresh_copy = None
    add = torch.ops.aten.add_.Tensor(view, 1)
    view_1 = torch.ops.aten.view.default(view, [3]);  view = None
    view_2 = torch.ops.aten.view.default(view_1, [-1])
    return view_1
    """)

    def test_advanced_indexing_correct_strides(self):
        def f(a):
            # This test requires that *_scatter ops are able to return
            # non-contiguous tensors.
            b = a.clone()[:, 1]
            c = torch.ones_like(b, dtype=torch.bool)
            d = b.masked_fill_(c, 0)
            return d
        self.assert_functionalization(f, torch.ones(2, 2), reapply_views=True)

    def test_tensor_list_mixed_functional_nonfunctional(self):
        nonfunctional_tensor = torch.ones(2, dtype=torch.long)

        def f(x):
            # simple test: 1 view op, 1 inplace op
            functional_tensor = torch.ones(2, dtype=torch.long)
            out = x[functional_tensor, nonfunctional_tensor]
            return out
        out = f(torch.ones(2, 2))
        out_functional = _functionalize(f, reapply_views=True, crossref=self.crossref)(torch.ones(2, 2))
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



def forward(self, arg0_1):
    ones = torch.ops.aten.ones.default([4, 2], device = device(type='cpu'), pin_memory = False)
    view_copy = torch.ops.aten.view_copy.default(arg0_1, [4, 2])
    add = torch.ops.aten.add.Tensor(arg0_1, ones);  ones = None
    copy_ = torch.ops.aten.copy_.default(arg0_1, add);  arg0_1 = None
    view_copy_1 = torch.ops.aten.view_copy.default(add, [4, 2]);  add = None
    return view_copy_1
    """)

        reinplaced_logs = self.get_logs(f, torch.ones(4, 2), reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(reinplaced_logs, """\



def forward(self, arg0_1):
    ones = torch.ops.aten.ones.default([4, 2], device = device(type='cpu'), pin_memory = False)
    view = torch.ops.aten.view.default(arg0_1, [4, 2])
    add = torch.ops.aten.add.Tensor(arg0_1, ones);  ones = None
    copy_ = torch.ops.aten.copy_.default(arg0_1, add);  arg0_1 = None
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



def forward(self, arg0_1):
    _fused_moving_avg_obs_fq_helper_functional = torch.ops.aten._fused_moving_avg_obs_fq_helper_functional.default(arg0_1, arg0_1, arg0_1, arg0_1, arg0_1, arg0_1, arg0_1, 1.0, 0, 1, 0)
    getitem = _fused_moving_avg_obs_fq_helper_functional[0]
    getitem_1 = _fused_moving_avg_obs_fq_helper_functional[1]
    getitem_2 = _fused_moving_avg_obs_fq_helper_functional[2]
    getitem_3 = _fused_moving_avg_obs_fq_helper_functional[3]
    getitem_4 = _fused_moving_avg_obs_fq_helper_functional[4]
    getitem_5 = _fused_moving_avg_obs_fq_helper_functional[5];  _fused_moving_avg_obs_fq_helper_functional = None
    copy_ = torch.ops.aten.copy_.default(arg0_1, getitem_5);  arg0_1 = getitem_5 = None
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



def forward(self, arg0_1):
    as_strided_copy = torch.ops.aten.as_strided_copy.default(arg0_1, [2], [2], 1)
    add = torch.ops.aten.add.Tensor(as_strided_copy, 1);  as_strided_copy = None
    as_strided_scatter = torch.ops.aten.as_strided_scatter.default(arg0_1, add, [2], [2], 1);  add = None
    as_strided_copy_1 = torch.ops.aten.as_strided_copy.default(as_strided_scatter, [2], [2], 1)
    copy_ = torch.ops.aten.copy_.default(arg0_1, as_strided_scatter);  arg0_1 = None
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



def forward(self, arg0_1):
    block_diag = torch.ops.aten.block_diag.default([arg0_1, arg0_1]);  arg0_1 = None
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



def forward(self, arg0_1):
    empty = torch.ops.aten.empty.memory_format([0], device = device(type='cpu'), pin_memory = False)
    cat = torch.ops.aten.cat.default([arg0_1]);  arg0_1 = None
    return cat
    """)

        reinplaced_logs = self.get_logs(f, torch.ones(2, 2), reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(reinplaced_logs, """\



def forward(self, arg0_1):
    empty = torch.ops.aten.empty.memory_format([0], device = device(type='cpu'), pin_memory = False)
    cat = torch.ops.aten.cat.default([arg0_1]);  arg0_1 = None
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



def forward(self, arg0_1):
    ones = torch.ops.aten.ones.default([2], device = device(type='cpu'), pin_memory = False)
    clone = torch.ops.aten.clone.default(arg0_1)
    diagonal_copy = torch.ops.aten.diagonal_copy.default(clone)
    add = torch.ops.aten.add.Tensor(diagonal_copy, ones);  diagonal_copy = ones = None
    diagonal_scatter = torch.ops.aten.diagonal_scatter.default(clone, add);  clone = add = None
    diagonal_copy_1 = torch.ops.aten.diagonal_copy.default(diagonal_scatter);  diagonal_scatter = None
    mul = torch.ops.aten.mul.Tensor(arg0_1, arg0_1);  arg0_1 = None
    return mul
    """)

        reinplaced_logs = self.get_logs(f, torch.ones(2, 2), reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(reinplaced_logs, """\



def forward(self, arg0_1):
    ones = torch.ops.aten.ones.default([2], device = device(type='cpu'), pin_memory = False)
    clone = torch.ops.aten.clone.default(arg0_1)
    diagonal = torch.ops.aten.diagonal.default(clone)
    add = torch.ops.aten.add_.Tensor(diagonal, ones);  diagonal = ones = None
    diagonal_1 = torch.ops.aten.diagonal.default(clone);  clone = None
    mul = torch.ops.aten.mul.Tensor(arg0_1, arg0_1);  arg0_1 = None
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



def forward(self, arg0_1):
    ones = torch.ops.aten.ones.default([2], device = device(type='cpu'), pin_memory = False)
    diagonal_copy = torch.ops.aten.diagonal_copy.default(arg0_1)
    add = torch.ops.aten.add.Tensor(diagonal_copy, ones);  diagonal_copy = ones = None
    diagonal_scatter = torch.ops.aten.diagonal_scatter.default(arg0_1, add);  add = None
    diagonal_copy_1 = torch.ops.aten.diagonal_copy.default(diagonal_scatter)
    copy_ = torch.ops.aten.copy_.default(arg0_1, diagonal_scatter);  arg0_1 = None
    return diagonal_scatter
    """)

    def test_channels_last_contiguous(self):
        def f(x):
            return x.contiguous(memory_format=torch.channels_last)
            tmp = torch.ones(2)
            y = x.diagonal()
            y.add_(tmp)
            return x
        x = torch.randn(4, 8, 8, 3).permute(0, 3, 1, 2)
        self.assert_functionalization(f, x)
        logs = self.get_logs(f, x).strip()
        # There should be no clone in the graph
        self.assertExpectedInline(logs, """\
def forward(self, arg0_1):
    return arg0_1""")

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



def forward(self, arg0_1):
    ones = torch.ops.aten.ones.default([2], device = device(type='cpu'), pin_memory = False)
    split_copy = torch.ops.aten.split_copy.Tensor(arg0_1, 2)
    getitem = split_copy[0]
    getitem_1 = split_copy[1];  split_copy = None
    diagonal_copy = torch.ops.aten.diagonal_copy.default(getitem_1);  getitem_1 = None
    add = torch.ops.aten.add.Tensor(diagonal_copy, ones);  diagonal_copy = ones = None
    split_copy_1 = torch.ops.aten.split_copy.Tensor(arg0_1, 2)
    getitem_2 = split_copy_1[0]
    getitem_3 = split_copy_1[1];  split_copy_1 = None
    diagonal_scatter = torch.ops.aten.diagonal_scatter.default(getitem_3, add);  getitem_3 = add = None
    slice_scatter = torch.ops.aten.slice_scatter.default(arg0_1, diagonal_scatter, 0, 2, 4);  diagonal_scatter = None
    split_copy_2 = torch.ops.aten.split_copy.Tensor(slice_scatter, 2)
    getitem_4 = split_copy_2[0]
    getitem_5 = split_copy_2[1];  split_copy_2 = None
    diagonal_copy_1 = torch.ops.aten.diagonal_copy.default(getitem_5);  getitem_5 = None
    mul = torch.ops.aten.mul.Tensor(slice_scatter, slice_scatter)
    copy_ = torch.ops.aten.copy_.default(arg0_1, slice_scatter);  arg0_1 = slice_scatter = None
    return diagonal_copy_1
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



def forward(self, arg0_1):
    ones = torch.ops.aten.ones.default([4], device = device(type='cpu'), pin_memory = False)
    transpose_copy = torch.ops.aten.transpose_copy.int(arg0_1, 1, 0)
    select_copy = torch.ops.aten.select_copy.int(transpose_copy, 0, 0);  transpose_copy = None
    add = torch.ops.aten.add.Tensor(select_copy, ones);  select_copy = ones = None
    transpose_copy_1 = torch.ops.aten.transpose_copy.int(arg0_1, 1, 0);  arg0_1 = None
    select_scatter = torch.ops.aten.select_scatter.default(transpose_copy_1, add, 0, 0);  transpose_copy_1 = add = None
    transpose_copy_2 = torch.ops.aten.transpose_copy.int(select_scatter, 1, 0);  select_scatter = None
    transpose_copy_3 = torch.ops.aten.transpose_copy.int(transpose_copy_2, 1, 0)
    select_copy_1 = torch.ops.aten.select_copy.int(transpose_copy_3, 0, 0);  transpose_copy_3 = None
    transpose_copy_4 = torch.ops.aten.transpose_copy.int(transpose_copy_2, 1, 0);  transpose_copy_2 = None
    return transpose_copy_4
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



def forward(self, arg0_1):
    view_copy = torch.ops.aten.view_copy.default(arg0_1, [8])
    arange = torch.ops.aten.arange.default(4, device = device(type='cpu'), pin_memory = False)
    arange_1 = torch.ops.aten.arange.default(4, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    index_put = torch.ops.aten.index_put.default(view_copy, [arange], arange_1);  view_copy = arange = arange_1 = None
    view_copy_1 = torch.ops.aten.view_copy.default(index_put, [4, 2]);  index_put = None
    view_copy_2 = torch.ops.aten.view_copy.default(view_copy_1, [8])
    copy_ = torch.ops.aten.copy_.default(arg0_1, view_copy_1);  arg0_1 = view_copy_1 = None
    return view_copy_2
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



def forward(self, arg0_1):
    ones = torch.ops.aten.ones.default([4, 2], device = device(type='cpu'), pin_memory = False)
    view_copy = torch.ops.aten.view_copy.default(arg0_1, [4, 2])
    add = torch.ops.aten.add.Tensor(view_copy, 1);  view_copy = None
    view_copy_1 = torch.ops.aten.view_copy.default(add, [4, 2]);  add = None
    view_copy_2 = torch.ops.aten.view_copy.default(view_copy_1, [4, 2])
    mul = torch.ops.aten.mul.Tensor(view_copy_2, 2);  view_copy_2 = None
    div = torch.ops.aten.div.Tensor(mul, 1);  mul = None
    copy_ = torch.ops.aten.copy_.default(arg0_1, view_copy_1);  arg0_1 = view_copy_1 = None
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



def forward(self, arg0_1):
    clone = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None
    ge = torch.ops.aten.ge.Scalar(clone, 0);  clone = None
    _to_copy = torch.ops.aten._to_copy.default(ge, dtype = torch.float32, layout = torch.strided);  ge = None
    return _to_copy
    """)

        reinplaced_logs = self.get_logs(f, torch.ones(2, 2), reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(reinplaced_logs, """\



def forward(self, arg0_1):
    clone = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None
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



def forward(self, arg0_1):
    view_copy = torch.ops.aten.view_copy.default(arg0_1, [4, 2]);  arg0_1 = None
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



def forward(self, arg0_1):
    ones = torch.ops.aten.ones.default([2, 2], device = device(type='cpu'), pin_memory = False)
    add = torch.ops.aten.add.Tensor(arg0_1, arg0_1);  arg0_1 = None
    view_copy = torch.ops.aten.view_copy.default(add, [8])
    view_copy_1 = torch.ops.aten.view_copy.default(view_copy, [2, 4]);  view_copy = None
    transpose_copy = torch.ops.aten.transpose_copy.int(view_copy_1, 1, 0)
    unsqueeze_copy = torch.ops.aten.unsqueeze_copy.default(transpose_copy, 0);  transpose_copy = None
    squeeze_copy = torch.ops.aten.squeeze_copy.default(unsqueeze_copy);  unsqueeze_copy = None
    split_copy = torch.ops.aten.split_copy.Tensor(squeeze_copy, 2);  squeeze_copy = None
    getitem = split_copy[0]
    getitem_1 = split_copy[1];  split_copy = None
    add_1 = torch.ops.aten.add.Tensor(getitem, ones);  getitem = ones = None
    view_copy_2 = torch.ops.aten.view_copy.default(add, [8]);  add = None
    view_copy_3 = torch.ops.aten.view_copy.default(view_copy_2, [2, 4]);  view_copy_2 = None
    transpose_copy_1 = torch.ops.aten.transpose_copy.int(view_copy_3, 1, 0);  view_copy_3 = None
    unsqueeze_copy_1 = torch.ops.aten.unsqueeze_copy.default(transpose_copy_1, 0);  transpose_copy_1 = None
    squeeze_copy_1 = torch.ops.aten.squeeze_copy.default(unsqueeze_copy_1);  unsqueeze_copy_1 = None
    slice_scatter = torch.ops.aten.slice_scatter.default(squeeze_copy_1, add_1, 0, 0, 2);  squeeze_copy_1 = add_1 = None
    unsqueeze_copy_2 = torch.ops.aten.unsqueeze_copy.default(slice_scatter, 0);  slice_scatter = None
    squeeze_copy_2 = torch.ops.aten.squeeze_copy.dim(unsqueeze_copy_2, 0);  unsqueeze_copy_2 = None
    transpose_copy_2 = torch.ops.aten.transpose_copy.int(squeeze_copy_2, 1, 0);  squeeze_copy_2 = None
    view_copy_4 = torch.ops.aten.view_copy.default(transpose_copy_2, [8]);  transpose_copy_2 = None
    view_copy_5 = torch.ops.aten.view_copy.default(view_copy_4, [4, 2]);  view_copy_4 = None
    view_copy_6 = torch.ops.aten.view_copy.default(view_copy_5, [8])
    view_copy_7 = torch.ops.aten.view_copy.default(view_copy_6, [2, 4]);  view_copy_6 = None
    transpose_copy_3 = torch.ops.aten.transpose_copy.int(view_copy_7, 1, 0);  view_copy_7 = None
    unsqueeze_copy_3 = torch.ops.aten.unsqueeze_copy.default(transpose_copy_3, 0);  transpose_copy_3 = None
    squeeze_copy_3 = torch.ops.aten.squeeze_copy.default(unsqueeze_copy_3);  unsqueeze_copy_3 = None
    split_copy_1 = torch.ops.aten.split_copy.Tensor(squeeze_copy_3, 2);  squeeze_copy_3 = None
    getitem_2 = split_copy_1[0]
    getitem_3 = split_copy_1[1];  split_copy_1 = None
    select_copy = torch.ops.aten.select_copy.int(view_copy_1, 0, 0);  view_copy_1 = None
    view_copy_8 = torch.ops.aten.view_copy.default(getitem_2, [4])
    view_copy_9 = torch.ops.aten.view_copy.default(view_copy_5, [8])
    view_copy_10 = torch.ops.aten.view_copy.default(view_copy_9, [2, 4]);  view_copy_9 = None
    select_copy_1 = torch.ops.aten.select_copy.int(view_copy_10, 0, 0);  view_copy_10 = None
    view_copy_11 = torch.ops.aten.view_copy.default(view_copy_5, [8]);  view_copy_5 = None
    view_copy_12 = torch.ops.aten.view_copy.default(view_copy_11, [2, 4]);  view_copy_11 = None
    transpose_copy_4 = torch.ops.aten.transpose_copy.int(view_copy_12, 1, 0);  view_copy_12 = None
    unsqueeze_copy_4 = torch.ops.aten.unsqueeze_copy.default(transpose_copy_4, 0);  transpose_copy_4 = None
    squeeze_copy_4 = torch.ops.aten.squeeze_copy.default(unsqueeze_copy_4);  unsqueeze_copy_4 = None
    split_copy_2 = torch.ops.aten.split_copy.Tensor(squeeze_copy_4, 2);  squeeze_copy_4 = None
    getitem_4 = split_copy_2[0]
    getitem_5 = split_copy_2[1];  split_copy_2 = None
    view_copy_13 = torch.ops.aten.view_copy.default(getitem_4, [4]);  getitem_4 = None
    add_2 = torch.ops.aten.add.Tensor(select_copy_1, view_copy_13);  select_copy_1 = view_copy_13 = None
    return getitem_2
    """)  # noqa: B950

        reinplaced_logs = self.get_logs(f, torch.ones(4, 2), reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(reinplaced_logs, """\



def forward(self, arg0_1):
    ones = torch.ops.aten.ones.default([2, 2], device = device(type='cpu'), pin_memory = False)
    add = torch.ops.aten.add.Tensor(arg0_1, arg0_1);  arg0_1 = None
    view = torch.ops.aten.view.default(add, [8])
    view_1 = torch.ops.aten.view.default(view, [2, 4]);  view = None
    transpose = torch.ops.aten.transpose.int(view_1, 1, 0)
    unsqueeze = torch.ops.aten.unsqueeze.default(transpose, 0);  transpose = None
    squeeze = torch.ops.aten.squeeze.default(unsqueeze);  unsqueeze = None
    split = torch.ops.aten.split.Tensor(squeeze, 2);  squeeze = None
    getitem = split[0]
    getitem_1 = split[1];  split = None
    add_1 = torch.ops.aten.add_.Tensor(getitem, ones);  getitem = ones = None
    view_2 = torch.ops.aten.view.default(add, [8]);  add = None
    view_3 = torch.ops.aten.view.default(view_2, [2, 4]);  view_2 = None
    transpose_1 = torch.ops.aten.transpose.int(view_3, 1, 0);  view_3 = None
    unsqueeze_1 = torch.ops.aten.unsqueeze.default(transpose_1, 0);  transpose_1 = None
    squeeze_1 = torch.ops.aten.squeeze.default(unsqueeze_1);  unsqueeze_1 = None
    unsqueeze_2 = torch.ops.aten.unsqueeze.default(squeeze_1, 0);  squeeze_1 = None
    squeeze_2 = torch.ops.aten.squeeze.dim(unsqueeze_2, 0);  unsqueeze_2 = None
    transpose_2 = torch.ops.aten.transpose.int(squeeze_2, 1, 0);  squeeze_2 = None
    view_4 = torch.ops.aten.view.default(transpose_2, [8]);  transpose_2 = None
    view_5 = torch.ops.aten.view.default(view_4, [4, 2]);  view_4 = None
    view_6 = torch.ops.aten.view.default(view_5, [8])
    view_7 = torch.ops.aten.view.default(view_6, [2, 4]);  view_6 = None
    transpose_3 = torch.ops.aten.transpose.int(view_7, 1, 0);  view_7 = None
    unsqueeze_3 = torch.ops.aten.unsqueeze.default(transpose_3, 0);  transpose_3 = None
    squeeze_3 = torch.ops.aten.squeeze.default(unsqueeze_3);  unsqueeze_3 = None
    split_1 = torch.ops.aten.split.Tensor(squeeze_3, 2);  squeeze_3 = None
    getitem_2 = split_1[0]
    getitem_3 = split_1[1];  split_1 = None
    select = torch.ops.aten.select.int(view_1, 0, 0);  view_1 = None
    clone = torch.ops.aten.clone.default(getitem_2, memory_format = torch.contiguous_format)
    _unsafe_view = torch.ops.aten._unsafe_view.default(clone, [4]);  clone = None
    view_8 = torch.ops.aten.view.default(view_5, [8]);  view_5 = None
    view_9 = torch.ops.aten.view.default(view_8, [2, 4]);  view_8 = None
    select_1 = torch.ops.aten.select.int(view_9, 0, 0);  view_9 = None
    add_2 = torch.ops.aten.add.Tensor(select_1, _unsafe_view);  select_1 = _unsafe_view = None
    return getitem_2
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



def forward(self, arg0_1):
    ones = torch.ops.aten.ones.default([4, 2], device = device(type='cpu'), pin_memory = False)
    view = torch.ops.aten.view.default(arg0_1, [4, 2])
    add = torch.ops.aten.add.Tensor(view, ones);  view = ones = None
    view_1 = torch.ops.aten.view.default(add, [4, 2]);  add = None
    view_2 = torch.ops.aten.view.default(view_1, [4, 2])
    mul = torch.ops.aten.mul.Tensor(view_1, view_1)
    copy_ = torch.ops.aten.copy_.default(arg0_1, view_1);  arg0_1 = view_1 = None
    return view_2
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

    # copy_() gets its own test, because it used to be special cased in functionalization.
    # However, now it works pretty similar to other functional ops
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



def forward(self, arg0_1):
    zeros = torch.ops.aten.zeros.default([2, 2], device = device(type='cpu'), pin_memory = False)
    diagonal_copy = torch.ops.aten.diagonal_copy.default(zeros)
    copy = torch.ops.aten.copy.default(diagonal_copy, arg0_1);  diagonal_copy = None
    diagonal_scatter = torch.ops.aten.diagonal_scatter.default(zeros, copy);  zeros = copy = None
    diagonal_copy_1 = torch.ops.aten.diagonal_copy.default(diagonal_scatter)
    add = torch.ops.aten.add.Tensor(diagonal_copy_1, arg0_1);  diagonal_copy_1 = arg0_1 = None
    diagonal_scatter_1 = torch.ops.aten.diagonal_scatter.default(diagonal_scatter, add);  diagonal_scatter = add = None
    diagonal_copy_2 = torch.ops.aten.diagonal_copy.default(diagonal_scatter_1);  diagonal_scatter_1 = None
    return diagonal_copy_2
    """)

        reinplaced_logs = self.get_logs(f, torch.ones(2), reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(reinplaced_logs, """\



def forward(self, arg0_1):
    zeros = torch.ops.aten.zeros.default([2, 2], device = device(type='cpu'), pin_memory = False)
    diagonal = torch.ops.aten.diagonal.default(zeros)
    copy = torch.ops.aten.copy_.default(diagonal, arg0_1);  diagonal = None
    diagonal_1 = torch.ops.aten.diagonal.default(zeros)
    add = torch.ops.aten.add_.Tensor(diagonal_1, arg0_1);  diagonal_1 = arg0_1 = None
    diagonal_2 = torch.ops.aten.diagonal.default(zeros);  zeros = None
    return diagonal_2
    """)

        # Test 2: copy_() with same dtype, different shape
        self.assert_functionalization(f, torch.ones(1))
        logs = self.get_logs(f, torch.ones(1))
        self.assertExpectedInline(logs, """\



def forward(self, arg0_1):
    zeros = torch.ops.aten.zeros.default([2, 2], device = device(type='cpu'), pin_memory = False)
    diagonal_copy = torch.ops.aten.diagonal_copy.default(zeros)
    copy = torch.ops.aten.copy.default(diagonal_copy, arg0_1);  diagonal_copy = None
    diagonal_scatter = torch.ops.aten.diagonal_scatter.default(zeros, copy);  zeros = copy = None
    diagonal_copy_1 = torch.ops.aten.diagonal_copy.default(diagonal_scatter)
    add = torch.ops.aten.add.Tensor(diagonal_copy_1, arg0_1);  diagonal_copy_1 = arg0_1 = None
    diagonal_scatter_1 = torch.ops.aten.diagonal_scatter.default(diagonal_scatter, add);  diagonal_scatter = add = None
    diagonal_copy_2 = torch.ops.aten.diagonal_copy.default(diagonal_scatter_1);  diagonal_scatter_1 = None
    return diagonal_copy_2
    """)

        reinplaced_logs = self.get_logs(f, torch.ones(1), reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(reinplaced_logs, """\



def forward(self, arg0_1):
    zeros = torch.ops.aten.zeros.default([2, 2], device = device(type='cpu'), pin_memory = False)
    diagonal = torch.ops.aten.diagonal.default(zeros)
    copy = torch.ops.aten.copy_.default(diagonal, arg0_1);  diagonal = None
    diagonal_1 = torch.ops.aten.diagonal.default(zeros)
    add = torch.ops.aten.add_.Tensor(diagonal_1, arg0_1);  diagonal_1 = arg0_1 = None
    diagonal_2 = torch.ops.aten.diagonal.default(zeros);  zeros = None
    return diagonal_2
    """)

        # Test 3: copy_() with different dtype, same shape
        self.assert_functionalization(f, torch.ones(2, dtype=torch.long))
        logs = self.get_logs(f, torch.ones(2, dtype=torch.long))
        self.assertExpectedInline(logs, """\



def forward(self, arg0_1):
    zeros = torch.ops.aten.zeros.default([2, 2], device = device(type='cpu'), pin_memory = False)
    diagonal_copy = torch.ops.aten.diagonal_copy.default(zeros)
    copy = torch.ops.aten.copy.default(diagonal_copy, arg0_1);  diagonal_copy = None
    diagonal_scatter = torch.ops.aten.diagonal_scatter.default(zeros, copy);  zeros = copy = None
    diagonal_copy_1 = torch.ops.aten.diagonal_copy.default(diagonal_scatter)
    add = torch.ops.aten.add.Tensor(diagonal_copy_1, arg0_1);  diagonal_copy_1 = arg0_1 = None
    diagonal_scatter_1 = torch.ops.aten.diagonal_scatter.default(diagonal_scatter, add);  diagonal_scatter = add = None
    diagonal_copy_2 = torch.ops.aten.diagonal_copy.default(diagonal_scatter_1);  diagonal_scatter_1 = None
    return diagonal_copy_2
    """)  # noqa: B950

        reinplaced_logs = self.get_logs(f, torch.ones(2, dtype=torch.long), reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(reinplaced_logs, """\



def forward(self, arg0_1):
    zeros = torch.ops.aten.zeros.default([2, 2], device = device(type='cpu'), pin_memory = False)
    diagonal = torch.ops.aten.diagonal.default(zeros)
    copy = torch.ops.aten.copy_.default(diagonal, arg0_1);  diagonal = None
    diagonal_1 = torch.ops.aten.diagonal.default(zeros)
    add = torch.ops.aten.add_.Tensor(diagonal_1, arg0_1);  diagonal_1 = arg0_1 = None
    diagonal_2 = torch.ops.aten.diagonal.default(zeros);  zeros = None
    return diagonal_2
    """)  # noqa: B950

        # Test 4: copy_() with different dtype, different shape
        self.assert_functionalization(f, torch.ones(1, dtype=torch.long))
        logs = self.get_logs(f, torch.ones(1, dtype=torch.long))
        self.assertExpectedInline(logs, """\



def forward(self, arg0_1):
    zeros = torch.ops.aten.zeros.default([2, 2], device = device(type='cpu'), pin_memory = False)
    diagonal_copy = torch.ops.aten.diagonal_copy.default(zeros)
    copy = torch.ops.aten.copy.default(diagonal_copy, arg0_1);  diagonal_copy = None
    diagonal_scatter = torch.ops.aten.diagonal_scatter.default(zeros, copy);  zeros = copy = None
    diagonal_copy_1 = torch.ops.aten.diagonal_copy.default(diagonal_scatter)
    add = torch.ops.aten.add.Tensor(diagonal_copy_1, arg0_1);  diagonal_copy_1 = arg0_1 = None
    diagonal_scatter_1 = torch.ops.aten.diagonal_scatter.default(diagonal_scatter, add);  diagonal_scatter = add = None
    diagonal_copy_2 = torch.ops.aten.diagonal_copy.default(diagonal_scatter_1);  diagonal_scatter_1 = None
    return diagonal_copy_2
    """)  # noqa: B950

        reinplaced_logs = self.get_logs(f, torch.ones(1, dtype=torch.long), reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(reinplaced_logs, """\



def forward(self, arg0_1):
    zeros = torch.ops.aten.zeros.default([2, 2], device = device(type='cpu'), pin_memory = False)
    diagonal = torch.ops.aten.diagonal.default(zeros)
    copy = torch.ops.aten.copy_.default(diagonal, arg0_1);  diagonal = None
    diagonal_1 = torch.ops.aten.diagonal.default(zeros)
    add = torch.ops.aten.add_.Tensor(diagonal_1, arg0_1);  diagonal_1 = arg0_1 = None
    diagonal_2 = torch.ops.aten.diagonal.default(zeros);  zeros = None
    return diagonal_2
    """)  # noqa: B950

    def test_expand_symint(self):
        # Once some existing SymInt bugs are ironed out, we should update
        # this test to plumb FakeSymbolicTensors through it
        def f(x):
            return x.expand(x.size(0), x.size(1))

        self.assert_functionalization(f, torch.ones(2, 2))
        logs = self.get_logs(f, torch.ones(2, 2))
        self.assertExpectedInline(logs, """\



def forward(self, arg0_1):
    expand_copy = torch.ops.aten.expand_copy.default(arg0_1, [2, 2]);  arg0_1 = None
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



def forward(self, arg0_1):
    add = torch.ops.aten.add.Tensor(arg0_1, arg0_1);  arg0_1 = None
    diagonal_copy = torch.ops.aten.diagonal_copy.default(add)
    fill = torch.ops.aten.fill.Scalar(diagonal_copy, 0);  diagonal_copy = None
    diagonal_scatter = torch.ops.aten.diagonal_scatter.default(add, fill);  add = fill = None
    diagonal_copy_1 = torch.ops.aten.diagonal_copy.default(diagonal_scatter)
    return diagonal_scatter
    """)

        reinplaced_logs = self.get_logs(f, torch.ones(2, 2), reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(reinplaced_logs, """\



def forward(self, arg0_1):
    add = torch.ops.aten.add.Tensor(arg0_1, arg0_1);  arg0_1 = None
    diagonal = torch.ops.aten.diagonal.default(add)
    fill = torch.ops.aten.fill_.Scalar(diagonal, 0);  diagonal = None
    diagonal_1 = torch.ops.aten.diagonal.default(add)
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



def forward(self, arg0_1):
    add = torch.ops.aten.add.Tensor(arg0_1, 1);  arg0_1 = None
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
    view_copy_5 = torch.ops.aten.view_copy.default(view_copy_4, [4, 4])
    as_strided_copy_2 = torch.ops.aten.as_strided_copy.default(view_copy_5, [3, 3], [3, 1]);  view_copy_5 = None
    view_copy_6 = torch.ops.aten.view_copy.default(as_strided_copy_2, [-1]);  as_strided_copy_2 = None
    view_copy_7 = torch.ops.aten.view_copy.default(view_copy_4, [4, 4]);  view_copy_4 = None
    as_strided_copy_3 = torch.ops.aten.as_strided_copy.default(view_copy_7, [3, 3], [3, 1]);  view_copy_7 = None
    add_2 = torch.ops.aten.add.Tensor(as_strided_copy_3, 1);  as_strided_copy_3 = None
    return add_2
    """)  # noqa: B950

        reinplaced_logs = self.get_logs(f, torch.ones(8, 2), reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(reinplaced_logs, """\



def forward(self, arg0_1):
    add = torch.ops.aten.add.Tensor(arg0_1, 1);  arg0_1 = None
    view = torch.ops.aten.view.default(add, [4, 4])
    resize = torch.ops.aten.resize.default(view, [3, 3])
    as_strided = torch.ops.aten.as_strided.default(view, [3, 3], [3, 1]);  view = None
    view_1 = torch.ops.aten.view.default(as_strided, [-1]);  as_strided = None
    add_1 = torch.ops.aten.add_.Tensor(view_1, 1)
    view_2 = torch.ops.aten.view.default(add, [4, 4]);  add = None
    as_strided_1 = torch.ops.aten.as_strided.default(view_2, [3, 3], [3, 1])
    view_3 = torch.ops.aten.view.default(view_1, [3, 3]);  view_1 = None
    view_4 = torch.ops.aten.view.default(view_2, [8, 2]);  view_2 = None
    view_5 = torch.ops.aten.view.default(view_4, [4, 4])
    as_strided_2 = torch.ops.aten.as_strided.default(view_5, [3, 3], [3, 1]);  view_5 = None
    view_6 = torch.ops.aten.view.default(as_strided_2, [-1]);  as_strided_2 = None
    view_7 = torch.ops.aten.view.default(view_4, [4, 4]);  view_4 = None
    as_strided_3 = torch.ops.aten.as_strided.default(view_7, [3, 3], [3, 1]);  view_7 = None
    add_2 = torch.ops.aten.add_.Tensor(as_strided_3, 1)
    return as_strided_3
    """)

    def test_resize_same_size_diff_rank(self):
        def f(x):
            y = x.clone()
            y.resize_(25, 5)
            return y

        self.assert_functionalization(f, torch.ones(5, 5, 5))

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



def forward(self, arg0_1):
    add = torch.ops.aten.add.Tensor(arg0_1, 1);  arg0_1 = None
    resize = torch.ops.aten.resize.default(add, [5, 5]);  add = None
    view_copy = torch.ops.aten.view_copy.default(resize, [25]);  resize = None
    fill = torch.ops.aten.fill.Scalar(view_copy, 1);  view_copy = None
    view_copy_1 = torch.ops.aten.view_copy.default(fill, [5, 5]);  fill = None
    view_copy_2 = torch.ops.aten.view_copy.default(view_copy_1, [25])
    add_1 = torch.ops.aten.add.Tensor(view_copy_1, 1)
    return (view_copy_1, add_1)
    """)

        reinplaced_logs = self.get_logs(f, torch.ones(8, 2), reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(reinplaced_logs, """\



def forward(self, arg0_1):
    add = torch.ops.aten.add.Tensor(arg0_1, 1);  arg0_1 = None
    resize = torch.ops.aten.resize_.default(add, [5, 5])
    view = torch.ops.aten.view.default(add, [25]);  add = None
    fill = torch.ops.aten.fill_.Scalar(view, 1)
    view_1 = torch.ops.aten.view.default(view, [5, 5]);  view = None
    view_2 = torch.ops.aten.view.default(view_1, [25])
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



def forward(self, arg0_1):
    zeros = torch.ops.aten.zeros.default([10], device = device(type='cpu'), pin_memory = False)
    select_copy = torch.ops.aten.select_copy.int(zeros, 0, 5)
    fill = torch.ops.aten.fill.Scalar(select_copy, 1);  select_copy = None
    select_scatter = torch.ops.aten.select_scatter.default(zeros, fill, 0, 5);  zeros = fill = None
    select_copy_1 = torch.ops.aten.select_copy.int(select_scatter, 0, 5)
    return select_scatter
    """)  # noqa: B950

        reinplaced_logs = self.get_logs(f, torch.ones(2), reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(reinplaced_logs, """\



def forward(self, arg0_1):
    zeros = torch.ops.aten.zeros.default([10], device = device(type='cpu'), pin_memory = False)
    select = torch.ops.aten.select.int(zeros, 0, 5)
    fill = torch.ops.aten.fill_.Scalar(select, 1);  select = None
    select_1 = torch.ops.aten.select.int(zeros, 0, 5)
    return zeros
    """)


    def test_instance_norm(self):
        size = 100

        def f(x, running_mean, running_var):
            with enable_python_dispatcher():
                return torch.instance_norm(x, None, None, running_mean, running_var,
                                           use_input_stats=True, momentum=0.1, eps=1e-5, cudnn_enabled=False)
        self.assert_functionalization(f, torch.randn(20, size, 35, 45), torch.zeros(size), torch.ones(size))
        # On Windows, for instance_norm, the alias_copy's are reordered to come right before they need to be used
        # whereas on other platforms, the alias_copy's are before the view_copy's.
        # e.g., the alias_copy after the getitem_4 assignment would be moved to be right before the copy assignment.
        if not IS_WINDOWS:
            logs = self.get_logs(f, torch.randn(20, size, 35, 45), torch.zeros(size), torch.ones(size))
            self.assertExpectedInline(logs, """\



def forward(self, arg0_1, arg1_1, arg2_1):
    repeat = torch.ops.aten.repeat.default(arg1_1, [20])
    repeat_1 = torch.ops.aten.repeat.default(arg2_1, [20])
    view_copy = torch.ops.aten.view_copy.default(arg0_1, [1, 2000, 35, 45]);  arg0_1 = None
    empty = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
    _native_batch_norm_legit_functional = torch.ops.aten._native_batch_norm_legit_functional.default(view_copy, None, None, repeat, repeat_1, True, 0.1, 1e-05);  view_copy = repeat = repeat_1 = None
    getitem = _native_batch_norm_legit_functional[0]
    getitem_1 = _native_batch_norm_legit_functional[1]
    getitem_2 = _native_batch_norm_legit_functional[2]
    getitem_3 = _native_batch_norm_legit_functional[3]
    getitem_4 = _native_batch_norm_legit_functional[4];  _native_batch_norm_legit_functional = None
    alias_copy = torch.ops.aten.alias_copy.default(arg1_1)
    view_copy_1 = torch.ops.aten.view_copy.default(getitem_3, [20, 100])
    view_copy_2 = torch.ops.aten.view_copy.default(getitem_3, [20, 100]);  getitem_3 = None
    mean = torch.ops.aten.mean.dim(view_copy_2, [0]);  view_copy_2 = None
    copy = torch.ops.aten.copy.default(alias_copy, mean);  alias_copy = mean = None
    alias_copy_1 = torch.ops.aten.alias_copy.default(copy);  copy = None
    alias_copy_2 = torch.ops.aten.alias_copy.default(alias_copy_1)
    alias_copy_3 = torch.ops.aten.alias_copy.default(arg2_1)
    view_copy_3 = torch.ops.aten.view_copy.default(getitem_4, [20, 100])
    view_copy_4 = torch.ops.aten.view_copy.default(getitem_4, [20, 100]);  getitem_4 = None
    mean_1 = torch.ops.aten.mean.dim(view_copy_4, [0]);  view_copy_4 = None
    copy_1 = torch.ops.aten.copy.default(alias_copy_3, mean_1);  alias_copy_3 = mean_1 = None
    alias_copy_4 = torch.ops.aten.alias_copy.default(copy_1);  copy_1 = None
    alias_copy_5 = torch.ops.aten.alias_copy.default(alias_copy_4)
    view_copy_5 = torch.ops.aten.view_copy.default(getitem, [20, 100, 35, 45]);  getitem = None
    copy_ = torch.ops.aten.copy_.default(arg1_1, alias_copy_1);  arg1_1 = alias_copy_1 = None
    copy__1 = torch.ops.aten.copy_.default(arg2_1, alias_copy_4);  arg2_1 = alias_copy_4 = None
    return view_copy_5
    """)  # noqa: B950

            reinplaced_logs = self.get_logs(
                f, torch.randn(20, size, 35, 45), torch.zeros(size), torch.ones(size),
                reapply_views=True, run_reinplace=True
            )
            self.assertExpectedInline(reinplaced_logs, """\



def forward(self, arg0_1, arg1_1, arg2_1):
    repeat = torch.ops.aten.repeat.default(arg1_1, [20])
    repeat_1 = torch.ops.aten.repeat.default(arg2_1, [20])
    view = torch.ops.aten.view.default(arg0_1, [1, 2000, 35, 45]);  arg0_1 = None
    empty = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
    _native_batch_norm_legit_functional = torch.ops.aten._native_batch_norm_legit_functional.default(view, None, None, repeat, repeat_1, True, 0.1, 1e-05);  view = repeat = repeat_1 = None
    getitem = _native_batch_norm_legit_functional[0]
    getitem_1 = _native_batch_norm_legit_functional[1]
    getitem_2 = _native_batch_norm_legit_functional[2]
    getitem_3 = _native_batch_norm_legit_functional[3]
    getitem_4 = _native_batch_norm_legit_functional[4];  _native_batch_norm_legit_functional = None
    alias = torch.ops.aten.alias.default(arg1_1)
    view_1 = torch.ops.aten.view.default(getitem_3, [20, 100])
    view_2 = torch.ops.aten.view.default(getitem_3, [20, 100]);  getitem_3 = None
    mean = torch.ops.aten.mean.dim(view_2, [0]);  view_2 = None
    copy = torch.ops.aten.copy.default(alias, mean);  alias = mean = None
    alias_1 = torch.ops.aten.alias.default(copy);  copy = None
    alias_2 = torch.ops.aten.alias.default(alias_1)
    alias_3 = torch.ops.aten.alias.default(arg2_1)
    view_3 = torch.ops.aten.view.default(getitem_4, [20, 100])
    view_4 = torch.ops.aten.view.default(getitem_4, [20, 100]);  getitem_4 = None
    mean_1 = torch.ops.aten.mean.dim(view_4, [0]);  view_4 = None
    copy_1 = torch.ops.aten.copy.default(alias_3, mean_1);  alias_3 = mean_1 = None
    alias_4 = torch.ops.aten.alias.default(copy_1);  copy_1 = None
    alias_5 = torch.ops.aten.alias.default(alias_4)
    view_5 = torch.ops.aten.view.default(getitem, [20, 100, 35, 45]);  getitem = None
    copy_ = torch.ops.aten.copy_.default(arg1_1, alias_1);  arg1_1 = alias_1 = None
    copy__1 = torch.ops.aten.copy_.default(arg2_1, alias_4);  arg2_1 = alias_4 = None
    return view_5
    """)  # noqa: B950

    def test_mutation_overlapping_mem(self):
        def fn(x):
            # x: (1, 5)
            t1 = torch.add(x, x)
            t2 = t1.unfold(1, 3, 2)
            t3 = t2.abs_()
            return t3

        with self.assertRaisesRegex(RuntimeError, r'encountered a tensor being mutated that has internal overlap'):
            x = torch.ones(1, 5)
            out = _functionalize(fn, reapply_views=True, crossref=False)(x)


    def test_batch_norm(self):
        def f(x, running_mean, running_var):
            with enable_python_dispatcher():
                return torch.batch_norm(x, None, None, running_mean, running_var, True, 0.1, 1e-5, False)

        self.assert_functionalization(f, torch.randn(20, 100, 35, 45), torch.zeros(100), torch.ones(100))
        logs = self.get_logs(f, torch.randn(20, 100, 35, 45), torch.zeros(100), torch.ones(100))
        self.assertExpectedInline(logs, """\



def forward(self, arg0_1, arg1_1, arg2_1):
    empty = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
    _native_batch_norm_legit_functional = torch.ops.aten._native_batch_norm_legit_functional.default(arg0_1, None, None, arg1_1, arg2_1, True, 0.1, 1e-05);  arg0_1 = None
    getitem = _native_batch_norm_legit_functional[0]
    getitem_1 = _native_batch_norm_legit_functional[1]
    getitem_2 = _native_batch_norm_legit_functional[2]
    getitem_3 = _native_batch_norm_legit_functional[3]
    getitem_4 = _native_batch_norm_legit_functional[4];  _native_batch_norm_legit_functional = None
    copy_ = torch.ops.aten.copy_.default(arg1_1, getitem_3);  arg1_1 = getitem_3 = None
    copy__1 = torch.ops.aten.copy_.default(arg2_1, getitem_4);  arg2_1 = getitem_4 = None
    return getitem
    """)  # noqa: B950

        reinplaced_logs = self.get_logs(
            f, torch.randn(20, 100, 35, 45), torch.zeros(100), torch.ones(100), reapply_views=True, run_reinplace=True
        )
        self.assertExpectedInline(reinplaced_logs, """\



def forward(self, arg0_1, arg1_1, arg2_1):
    empty = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))
    _native_batch_norm_legit_functional = torch.ops.aten._native_batch_norm_legit_functional.default(arg0_1, None, None, arg1_1, arg2_1, True, 0.1, 1e-05);  arg0_1 = None
    getitem = _native_batch_norm_legit_functional[0]
    getitem_1 = _native_batch_norm_legit_functional[1]
    getitem_2 = _native_batch_norm_legit_functional[2]
    getitem_3 = _native_batch_norm_legit_functional[3]
    getitem_4 = _native_batch_norm_legit_functional[4];  _native_batch_norm_legit_functional = None
    copy_ = torch.ops.aten.copy_.default(arg1_1, getitem_3);  arg1_1 = getitem_3 = None
    copy__1 = torch.ops.aten.copy_.default(arg2_1, getitem_4);  arg2_1 = getitem_4 = None
    return getitem
    """)  # noqa: B950


@xfail_inherited_tests([
    "test_as_strided",
    "test_copy_",
    "test_diagonal",
    "test_diagonal_mutated_input",
    "test_everything",
    "test_fill_",
    "test_split",
    "test_view_clone_view_inplace",
    "test_view_inplace",
])
class TestCrossRefFunctionalization(TestFunctionalization):
    crossref = True

if __name__ == '__main__':
    run_tests()
