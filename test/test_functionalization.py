# Owner(s): ["module: codegen"]

import torch
from torch.testing._internal.common_utils import TestCase, run_tests, skipIfTorchDynamo, TEST_WITH_TORCHDYNAMO
from torch.testing._internal.logging_tensor import LoggingTensor, capture_logs
from torch.utils._pytree import tree_map
from torch.fx.experimental.proxy_tensor import make_fx

import unittest

def are_aliased(x, y):
    if x._base is None and y._base is None:
        return False
    if x._base is not None and y._base is None:
        return x._base is y
    if x._base is None and y._base is not None:
        return y._base is x
    return x._base is y._base


@unittest.skipIf(TEST_WITH_TORCHDYNAMO, "https://github.com/pytorch/pytorch/issues/81457")
class TestFunctionalization(TestCase):
    # We can unify testing and use functionalize() here instead
    # if/when functorch moves into core.
    def _functionalize(self, f, *, reapply_views: bool):
        def wrapped(a):
            input_functional = torch._to_functional_tensor(a)
            torch._enable_functionalization(reapply_views=reapply_views)
            try:
                out = f(input_functional)
            finally:
                torch._disable_functionalization()
            torch._sync(input_functional)
            tree_map(torch._sync, out)
            out_unwrapped = tree_map(torch._from_functional_tensor, out)
            return out_unwrapped

        return wrapped

    def get_logs(self, func, inpt, *, reapply_views=False):
        traced_f = make_fx(self._functionalize(func, reapply_views=reapply_views))(inpt)
        return traced_f.code

    def assert_functionalization(self, func, inpt, *, reapply_views=False):
        input_clone = inpt.clone()
        input_clone2 = inpt.clone()
        input_functional = torch._to_functional_tensor(input_clone2)

        # Compare outputs (and mutated inputs), with and without functionalization.
        out_ref = func(inpt)

        torch._enable_functionalization(reapply_views=reapply_views)
        try:
            out_functional = func(input_functional)
        finally:
            torch._disable_functionalization()

        # We need to sync the input tensors first, in case there are any queued mutations left.
        torch._sync(input_functional)
        self.assertEqual(inpt, torch._from_functional_tensor(input_functional))  # input mutations should still occur

        # Handle tests with multi-tensor outputs
        if isinstance(out_ref, tuple) and isinstance(out_functional, tuple):
            out_refs, out_functionals = list(out_ref), list(out_functional)
        else:
            out_refs, out_functionals = [out_ref], [out_functional]

        for out_ref_, out_functional_ in zip(out_refs, out_functionals):
            self.assertEqual(out_ref_.size(), out_functional_.size())
            torch._sync(out_functional_)
            out_functional_unwrapped = torch._from_functional_tensor(out_functional_)
            self.assertEqual(out_ref_, out_functional_unwrapped)

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
    empty = torch.ops.aten.empty.memory_format([4, 2], dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    fill_scalar = torch.ops.aten.fill.Scalar(empty, 1.0);  empty = None
    view_copy_default = torch.ops.aten.view_copy.default(a_1, [4, 2]);  a_1 = None
    add_tensor = torch.ops.aten.add.Tensor(view_copy_default, fill_scalar);  view_copy_default = fill_scalar = None
    view_copy_default_1 = torch.ops.aten.view_copy.default(add_tensor, [4, 2])
    mul_tensor = torch.ops.aten.mul.Tensor(view_copy_default_1, view_copy_default_1);  view_copy_default_1 = None
    return add_tensor
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
    empty = torch.ops.aten.empty.memory_format([4, 2], dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    fill_scalar = torch.ops.aten.fill.Scalar(empty, 1.0);  empty = None
    view_copy_default = torch.ops.aten.view_copy.default(a_1, [4, 2]);  a_1 = None
    empty_1 = torch.ops.aten.empty.memory_format([], dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    add_tensor = torch.ops.aten.add.Tensor(view_copy_default, fill_scalar);  view_copy_default = fill_scalar = None
    mul_tensor = torch.ops.aten.mul.Tensor(add_tensor, add_tensor);  add_tensor = None
    return mul_tensor
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
    empty = torch.ops.aten.empty.memory_format([4], dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    empty_1 = torch.ops.aten.empty.memory_format([4], dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    aminmax_default = torch.ops.aten.aminmax.default(a_1, dim = 0);  a_1 = None
    getitem = aminmax_default[0]
    getitem_1 = aminmax_default[1];  aminmax_default = None
    return getitem
    """)

    def test_tensor_ctr(self):
        def f(x):
            y = torch.tensor((1, 2, 3))
            z = y.view(-1)
            z.add_(1)
            return y
        self.assert_functionalization(f, torch.arange(3, dtype=torch.float32))

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
    empty = torch.ops.aten.empty.memory_format([4, 2], dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    fill_scalar = torch.ops.aten.fill.Scalar(empty, 1.0);  empty = None
    view_copy_default = torch.ops.aten.view_copy.default(a_1, [4, 2])
    add_tensor = torch.ops.aten.add.Tensor(a_1, fill_scalar);  a_1 = fill_scalar = None
    view_copy_default_1 = torch.ops.aten.view_copy.default(add_tensor, [4, 2]);  add_tensor = None
    return view_copy_default_1
    """)

    # Some ops that are mutable are neither inplace nor out= ops.
    # They also need special handling.
    def test_mutable_op_not_inplace_or_other(self):
        def f(x):
            return torch._fused_moving_avg_obs_fq_helper(x, x, x, x, x, x, x, 1.0, 0, 1, 0)

        logs = self.get_logs(f, torch.ones(1))
        self.assertExpectedInline(logs, """\



def forward(self, a_1):
    _fused_moving_avg_obs_fq_helper_functional_default = torch.ops.aten._fused_moving_avg_obs_fq_helper_functional.default(a_1, a_1, a_1, a_1, a_1, a_1, a_1, 1.0, 0, 1, 0);  a_1 = None
    getitem = _fused_moving_avg_obs_fq_helper_functional_default[0]
    getitem_1 = _fused_moving_avg_obs_fq_helper_functional_default[1]
    getitem_2 = _fused_moving_avg_obs_fq_helper_functional_default[2]
    getitem_3 = _fused_moving_avg_obs_fq_helper_functional_default[3]
    getitem_4 = _fused_moving_avg_obs_fq_helper_functional_default[4]
    getitem_5 = _fused_moving_avg_obs_fq_helper_functional_default[5];  _fused_moving_avg_obs_fq_helper_functional_default = None
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
    as_strided_copy_default = torch.ops.aten.as_strided_copy.default(a_1, [2], [2], 1)
    add_tensor = torch.ops.aten.add.Tensor(as_strided_copy_default, 1);  as_strided_copy_default = None
    as_strided_scatter_default = torch.ops.aten.as_strided_scatter.default(a_1, add_tensor, [2], [2], 1);  a_1 = add_tensor = None
    return as_strided_scatter_default
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
    block_diag_default = torch.ops.aten.block_diag.default([a_1, a_1]);  a_1 = None
    return block_diag_default
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
    empty = torch.ops.aten.empty.memory_format([0], dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    cat_default = torch.ops.aten.cat.default([a_1]);  a_1 = None
    return cat_default
    """)

    def test_diagonal(self):
        def f(x):
            # test: view ops that take a subset of the original tensor (select/diagonal)
            tmp = torch.ones(2)
            y = x.diagonal()
            y.add_(tmp)
            z = x * x
            return z
        self.assert_functionalization(f, torch.ones(2, 2))
        logs = self.get_logs(f, torch.ones(2, 2))
        self.assertExpectedInline(logs, """\



def forward(self, a_1):
    empty = torch.ops.aten.empty.memory_format([2], dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    fill_scalar = torch.ops.aten.fill.Scalar(empty, 1.0);  empty = None
    diagonal_copy_default = torch.ops.aten.diagonal_copy.default(a_1)
    add_tensor = torch.ops.aten.add.Tensor(diagonal_copy_default, fill_scalar);  diagonal_copy_default = fill_scalar = None
    diagonal_scatter_default = torch.ops.aten.diagonal_scatter.default(a_1, add_tensor);  a_1 = add_tensor = None
    mul_tensor = torch.ops.aten.mul.Tensor(diagonal_scatter_default, diagonal_scatter_default);  diagonal_scatter_default = None
    return mul_tensor
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
    empty = torch.ops.aten.empty.memory_format([2], dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    fill_scalar = torch.ops.aten.fill.Scalar(empty, 1.0);  empty = None
    split_copy_tensor = torch.ops.aten.split_copy.Tensor(a_1, 2)
    getitem = split_copy_tensor[0]
    getitem_1 = split_copy_tensor[1];  split_copy_tensor = None
    diagonal_copy_default = torch.ops.aten.diagonal_copy.default(getitem_1);  getitem_1 = None
    add_tensor = torch.ops.aten.add.Tensor(diagonal_copy_default, fill_scalar);  diagonal_copy_default = fill_scalar = None
    split_copy_tensor_1 = torch.ops.aten.split_copy.Tensor(a_1, 2)
    getitem_2 = split_copy_tensor_1[0]
    getitem_3 = split_copy_tensor_1[1];  split_copy_tensor_1 = None
    diagonal_scatter_default = torch.ops.aten.diagonal_scatter.default(getitem_3, add_tensor);  getitem_3 = None
    slice_scatter_default = torch.ops.aten.slice_scatter.default(a_1, diagonal_scatter_default, 0, 2, 4);  a_1 = diagonal_scatter_default = None
    mul_tensor = torch.ops.aten.mul.Tensor(slice_scatter_default, slice_scatter_default);  slice_scatter_default = None
    return add_tensor
    """)  # noqa: B950

    def test_view_inplace(self):
        def f(x):
            # test: view + inplace op (transpose_)
            tmp = torch.ones(4)
            x.transpose_(1, 0)
            y = x[0]
            y.add_(tmp)
            return x
        self.assert_functionalization(f, torch.ones(4, 2))
        logs = self.get_logs(f, torch.ones(4, 2))
        self.assertExpectedInline(logs, """\



def forward(self, a_1):
    empty = torch.ops.aten.empty.memory_format([4], dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    fill_scalar = torch.ops.aten.fill.Scalar(empty, 1.0);  empty = None
    transpose_copy_int = torch.ops.aten.transpose_copy.int(a_1, 1, 0)
    select_copy_int = torch.ops.aten.select_copy.int(transpose_copy_int, 0, 0);  transpose_copy_int = None
    add_tensor = torch.ops.aten.add.Tensor(select_copy_int, fill_scalar);  select_copy_int = fill_scalar = None
    transpose_copy_int_1 = torch.ops.aten.transpose_copy.int(a_1, 1, 0);  a_1 = None
    select_scatter_default = torch.ops.aten.select_scatter.default(transpose_copy_int_1, add_tensor, 0, 0);  transpose_copy_int_1 = add_tensor = None
    transpose_copy_int_2 = torch.ops.aten.transpose_copy.int(select_scatter_default, 1, 0);  select_scatter_default = None
    transpose_copy_int_3 = torch.ops.aten.transpose_copy.int(transpose_copy_int_2, 1, 0);  transpose_copy_int_2 = None
    return transpose_copy_int_3
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
    view_copy_default = torch.ops.aten.view_copy.default(a_1, [8]);  a_1 = None
    empty = torch.ops.aten.empty.memory_format([0], dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    arange = torch.ops.aten.arange.start_step(0, 4, 1, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'))
    empty_1 = torch.ops.aten.empty.memory_format([0], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    arange_1 = torch.ops.aten.arange.start_step(0, 4, 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    index_put_default = torch.ops.aten.index_put.default(view_copy_default, [arange], arange_1);  view_copy_default = arange = arange_1 = None
    view_copy_default_1 = torch.ops.aten.view_copy.default(index_put_default, [4, 2])
    return index_put_default
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
    empty = torch.ops.aten.empty.memory_format([4, 2], dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    fill_scalar = torch.ops.aten.fill.Scalar(empty, 1.0);  empty = None
    view_copy_default = torch.ops.aten.view_copy.default(a_1, [4, 2]);  a_1 = None
    add_tensor = torch.ops.aten.add.Tensor(view_copy_default, 1);  view_copy_default = None
    mul_tensor = torch.ops.aten.mul.Tensor(add_tensor, 2)
    div_tensor = torch.ops.aten.div.Tensor(mul_tensor, 1);  mul_tensor = None
    view_copy_default_1 = torch.ops.aten.view_copy.default(add_tensor, [4, 2]);  add_tensor = None
    return div_tensor
    """)

    @skipIfTorchDynamo("Test does not work with TorchDynamo")
    def test_metadata_change(self):
        def f(x):
            # ops like ge_() are allowed to change the dtype of the input.
            # functionalization should pick up on that.
            return x.ge_(0)
        self.assert_functionalization(f, torch.ones(4, 2))
        logs = self.get_logs(f, torch.ones(4, 2))
        self.assertExpectedInline(logs, """\



def forward(self, a_1):
    ge_scalar = torch.ops.aten.ge.Scalar(a_1, 0);  a_1 = None
    to_dtype_layout = torch.ops.aten.to.dtype_layout(ge_scalar, dtype = torch.float32, layout = torch.strided);  ge_scalar = None
    return to_dtype_layout
    """)

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
    view_copy_default = torch.ops.aten.view_copy.default(a_1, [4, 2]);  a_1 = None
    return view_copy_default
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
    empty = torch.ops.aten.empty.memory_format([2, 2], dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    fill_scalar = torch.ops.aten.fill.Scalar(empty, 1.0);  empty = None
    add_tensor = torch.ops.aten.add.Tensor(a_1, a_1);  a_1 = None
    view_copy_default = torch.ops.aten.view_copy.default(add_tensor, [8])
    _reshape_alias_copy_default = torch.ops.aten._reshape_alias_copy.default(view_copy_default, [2, 4], [4, 1]);  view_copy_default = None
    transpose_copy_int = torch.ops.aten.transpose_copy.int(_reshape_alias_copy_default, 1, 0)
    unsqueeze_copy_default = torch.ops.aten.unsqueeze_copy.default(transpose_copy_int, 0);  transpose_copy_int = None
    squeeze_copy_default = torch.ops.aten.squeeze_copy.default(unsqueeze_copy_default);  unsqueeze_copy_default = None
    split_copy_tensor = torch.ops.aten.split_copy.Tensor(squeeze_copy_default, 2);  squeeze_copy_default = None
    getitem = split_copy_tensor[0]
    getitem_1 = split_copy_tensor[1];  split_copy_tensor = None
    add_tensor_1 = torch.ops.aten.add.Tensor(getitem, fill_scalar);  getitem = fill_scalar = None
    select_copy_int = torch.ops.aten.select_copy.int(_reshape_alias_copy_default, 0, 0);  _reshape_alias_copy_default = None
    clone_default = torch.ops.aten.clone.default(add_tensor_1, memory_format = torch.contiguous_format)
    _unsafe_view_default = torch.ops.aten._unsafe_view.default(clone_default, [4]);  clone_default = None
    view_copy_default_1 = torch.ops.aten.view_copy.default(add_tensor, [8]);  add_tensor = None
    _reshape_alias_copy_default_1 = torch.ops.aten._reshape_alias_copy.default(view_copy_default_1, [2, 4], [4, 1]);  view_copy_default_1 = None
    transpose_copy_int_1 = torch.ops.aten.transpose_copy.int(_reshape_alias_copy_default_1, 1, 0);  _reshape_alias_copy_default_1 = None
    unsqueeze_copy_default_1 = torch.ops.aten.unsqueeze_copy.default(transpose_copy_int_1, 0);  transpose_copy_int_1 = None
    squeeze_copy_default_1 = torch.ops.aten.squeeze_copy.default(unsqueeze_copy_default_1);  unsqueeze_copy_default_1 = None
    slice_scatter_default = torch.ops.aten.slice_scatter.default(squeeze_copy_default_1, add_tensor_1, 0, 0, 2);  squeeze_copy_default_1 = None
    unsqueeze_copy_default_2 = torch.ops.aten.unsqueeze_copy.default(slice_scatter_default, 0);  slice_scatter_default = None
    squeeze_copy_dim = torch.ops.aten.squeeze_copy.dim(unsqueeze_copy_default_2, 0);  unsqueeze_copy_default_2 = None
    transpose_copy_int_2 = torch.ops.aten.transpose_copy.int(squeeze_copy_dim, 1, 0);  squeeze_copy_dim = None
    _reshape_alias_copy_default_2 = torch.ops.aten._reshape_alias_copy.default(transpose_copy_int_2, [8], [1]);  transpose_copy_int_2 = None
    view_copy_default_2 = torch.ops.aten.view_copy.default(_reshape_alias_copy_default_2, [4, 2]);  _reshape_alias_copy_default_2 = None
    view_copy_default_3 = torch.ops.aten.view_copy.default(view_copy_default_2, [8]);  view_copy_default_2 = None
    _reshape_alias_copy_default_3 = torch.ops.aten._reshape_alias_copy.default(view_copy_default_3, [2, 4], [4, 1]);  view_copy_default_3 = None
    select_copy_int_1 = torch.ops.aten.select_copy.int(_reshape_alias_copy_default_3, 0, 0);  _reshape_alias_copy_default_3 = None
    add_tensor_2 = torch.ops.aten.add.Tensor(select_copy_int_1, _unsafe_view_default);  select_copy_int_1 = _unsafe_view_default = None
    return add_tensor_1
    """)  # noqa: B950

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
    empty = torch.ops.aten.empty.memory_format([4, 2], dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    fill_scalar = torch.ops.aten.fill.Scalar(empty, 1.0);  empty = None
    view_default = torch.ops.aten.view.default(a_1, [4, 2]);  a_1 = None
    add_tensor = torch.ops.aten.add.Tensor(view_default, fill_scalar);  view_default = fill_scalar = None
    view_default_1 = torch.ops.aten.view.default(add_tensor, [4, 2])
    mul_tensor = torch.ops.aten.mul.Tensor(view_default_1, view_default_1);  view_default_1 = None
    return add_tensor
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
            # NOTE: LoggingTensor isn't a mode, which means that the diagonal call
            # will not be logged. This is fine for testing.
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
    empty = torch.ops.aten.empty.SymInt([2, 2], dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    zero_default = torch.ops.aten.zero.default(empty);  empty = None
    diagonal_copy_default = torch.ops.aten.diagonal_copy.default(zero_default)
    diagonal_copy_default_1 = torch.ops.aten.diagonal_copy.default(zero_default);  zero_default = None
    copy_default = torch.ops.aten.copy.default(diagonal_copy_default_1, a_1);  diagonal_copy_default_1 = None
    add_tensor = torch.ops.aten.add.Tensor(copy_default, a_1);  copy_default = a_1 = None
    return add_tensor
    """)

        # Test 2: copy_() with same dtype, different shape
        self.assert_functionalization(f, torch.ones(1))
        logs = self.get_logs(f, torch.ones(1))
        self.assertExpectedInline(logs, """\



def forward(self, a_1):
    empty = torch.ops.aten.empty.SymInt([2, 2], dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    zero_default = torch.ops.aten.zero.default(empty);  empty = None
    diagonal_copy_default = torch.ops.aten.diagonal_copy.default(zero_default)
    diagonal_copy_default_1 = torch.ops.aten.diagonal_copy.default(zero_default);  zero_default = None
    copy_default = torch.ops.aten.copy.default(diagonal_copy_default_1, a_1);  diagonal_copy_default_1 = None
    add_tensor = torch.ops.aten.add.Tensor(copy_default, a_1);  copy_default = a_1 = None
    return add_tensor
    """)

        # Test 3: copy_() with different dtype, same shape
        self.assert_functionalization(f, torch.ones(2, dtype=torch.long))
        logs = self.get_logs(f, torch.ones(2, dtype=torch.long))
        self.assertExpectedInline(logs, """\



def forward(self, a_1):
    empty = torch.ops.aten.empty.SymInt([2, 2], dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    zero_default = torch.ops.aten.zero.default(empty);  empty = None
    diagonal_copy_default = torch.ops.aten.diagonal_copy.default(zero_default)
    diagonal_copy_default_1 = torch.ops.aten.diagonal_copy.default(zero_default);  zero_default = None
    copy_default = torch.ops.aten.copy.default(diagonal_copy_default_1, a_1);  diagonal_copy_default_1 = None
    add_tensor = torch.ops.aten.add.Tensor(copy_default, a_1);  copy_default = a_1 = None
    return add_tensor
    """)

        # Test 4: copy_() with different dtype, different shape
        self.assert_functionalization(f, torch.ones(1, dtype=torch.long))
        logs = self.get_logs(f, torch.ones(1, dtype=torch.long))
        self.assertExpectedInline(logs, """\



def forward(self, a_1):
    empty = torch.ops.aten.empty.SymInt([2, 2], dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    zero_default = torch.ops.aten.zero.default(empty);  empty = None
    diagonal_copy_default = torch.ops.aten.diagonal_copy.default(zero_default)
    diagonal_copy_default_1 = torch.ops.aten.diagonal_copy.default(zero_default);  zero_default = None
    copy_default = torch.ops.aten.copy.default(diagonal_copy_default_1, a_1);  diagonal_copy_default_1 = None
    add_tensor = torch.ops.aten.add.Tensor(copy_default, a_1);  copy_default = a_1 = None
    return add_tensor
    """)

    def test_expand_symint(self):
        # Once some existing SymInt bugs are ironed out, we should update
        # this test to plumb FakeSymbolicTensors through it
        def f(x):
            return x.expand(x.size(0), x.size(1))

        self.assert_functionalization(f, torch.ones(2, 2))
        logs = self.get_logs(f, torch.ones(2, 2))
        self.assertExpectedInline(logs, """\



def forward(self, a_1):
    expand_copy_default = torch.ops.aten.expand_copy.default(a_1, [2, 2]);  a_1 = None
    return expand_copy_default
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
    add_tensor = torch.ops.aten.add.Tensor(a_1, a_1);  a_1 = None
    diagonal_copy_default = torch.ops.aten.diagonal_copy.default(add_tensor)
    fill_scalar = torch.ops.aten.fill.Scalar(diagonal_copy_default, 0);  diagonal_copy_default = None
    diagonal_scatter_default = torch.ops.aten.diagonal_scatter.default(add_tensor, fill_scalar);  add_tensor = fill_scalar = None
    return diagonal_scatter_default
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
    add_tensor = torch.ops.aten.add.Tensor(a_1, 1);  a_1 = None
    view_copy_default = torch.ops.aten.view_copy.default(add_tensor, [4, 4])
    resize_default = torch.ops.aten.resize.default(view_copy_default, [3, 3])
    as_strided_copy_default = torch.ops.aten.as_strided_copy.default(view_copy_default, [3, 3], [3, 1]);  view_copy_default = None
    view_copy_default_1 = torch.ops.aten.view_copy.default(as_strided_copy_default, [-1]);  as_strided_copy_default = None
    add_tensor_1 = torch.ops.aten.add.Tensor(view_copy_default_1, 1);  view_copy_default_1 = None
    view_copy_default_2 = torch.ops.aten.view_copy.default(add_tensor, [4, 4]);  add_tensor = None
    as_strided_copy_default_1 = torch.ops.aten.as_strided_copy.default(view_copy_default_2, [3, 3], [3, 1])
    view_copy_default_3 = torch.ops.aten.view_copy.default(add_tensor_1, [3, 3]);  add_tensor_1 = None
    as_strided_scatter_default = torch.ops.aten.as_strided_scatter.default(view_copy_default_2, view_copy_default_3, [3, 3], [3, 1]);  view_copy_default_2 = view_copy_default_3 = None
    view_copy_default_4 = torch.ops.aten.view_copy.default(as_strided_scatter_default, [8, 2]);  as_strided_scatter_default = None
    view_copy_default_5 = torch.ops.aten.view_copy.default(view_copy_default_4, [4, 4]);  view_copy_default_4 = None
    as_strided_copy_default_2 = torch.ops.aten.as_strided_copy.default(view_copy_default_5, [3, 3], [3, 1]);  view_copy_default_5 = None
    add_tensor_2 = torch.ops.aten.add.Tensor(as_strided_copy_default_2, 1);  as_strided_copy_default_2 = None
    return add_tensor_2
    """)  # noqa: B950

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
    add_tensor = torch.ops.aten.add.Tensor(a_1, 1);  a_1 = None
    resize_default = torch.ops.aten.resize.default(add_tensor, [5, 5]);  add_tensor = None
    view_copy_default = torch.ops.aten.view_copy.default(resize_default, [25]);  resize_default = None
    fill_scalar = torch.ops.aten.fill.Scalar(view_copy_default, 1);  view_copy_default = None
    view_copy_default_1 = torch.ops.aten.view_copy.default(fill_scalar, [5, 5]);  fill_scalar = None
    add_tensor_1 = torch.ops.aten.add.Tensor(view_copy_default_1, 1)
    return (view_copy_default_1, add_tensor_1)
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

if __name__ == '__main__':
    run_tests()
