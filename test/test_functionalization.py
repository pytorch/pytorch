# Owner(s): ["module: codegen"]

import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.logging_tensor import LoggingTensor, LoggingTensorReentrant, capture_logs, log_input
from torch.utils._pytree import tree_map

import logging

def are_aliased(x, y):
    if x._base is None and y._base is None:
        return False
    if x._base is not None and y._base is None:
        return x._base is y
    if x._base is None and y._base is not None:
        return y._base is x
    return x._base is y._base

# Just for testing: a logging tensor that also transforms out-of-place ops into inplace ops.
# That way even if the outer wrapper is functionalized, the inner wrapper will also need functionalization.
class InplaceLoggingTensor(LoggingTensorReentrant):
    @staticmethod
    def __new__(cls, e):
        r = torch.Tensor._make_wrapper_subclass(cls, e.shape, dtype=e.dtype, requires_grad=False)
        r.elem = e
        return r

    __torch_function__ = torch._C._disabled_torch_function_impl

    def __str__(self):
        return f'InplaceLoggingTensor({self.elem})'

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap(e):
            if isinstance(e, InplaceLoggingTensor):
                return e.elem
            else:
                return e

        def wrap(e):
            if isinstance(e, torch.Tensor):
                return InplaceLoggingTensor(e)
            else:
                return e
        f = func
        # this subclass converts all `add()` ops into `add_()` ops
        if f is torch.ops.aten.add.Tensor:
            f = torch.ops.aten.add_.Tensor

        with cls.context():
            rs = tree_map(wrap, f(*tree_map(unwrap, args), **tree_map(unwrap, kwargs)))
        # after running the (potentially transformed) op,
        # log the original op that we saw.
        logging.getLogger("LoggingTensor").info(f"{func.__module__}.{func.__name__}", args, kwargs, rs)
        return rs



class TestFunctionalization(TestCase):

    def get_logs(self, func, inpt, *, reapply_views=False):
        input_clone_logging = LoggingTensor(inpt.clone())
        input_functional_logging = torch._to_functional_tensor(input_clone_logging)

        with capture_logs() as logs:
            log_input("input", input_clone_logging)
            torch._enable_functionalization(reapply_views=reapply_views)
            try:
                func(input_functional_logging)
            finally:
                torch._disable_functionalization()
        return logs

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
        self.assertExpectedInline('\n'.join(logs), """\
$0 = input('input')
$1 = torch._ops.aten.view_copy.default($0, [4, 2])
$2 = torch._ops.aten.add.Tensor($1, tensor([[1., 1.],
        [1., 1.],
        [1., 1.],
        [1., 1.]]))
$3 = torch._ops.aten.view_copy.default($2, [4, 2])
$4 = torch._ops.aten.mul.Tensor($3, $3)""")

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
        self.assertExpectedInline('\n'.join(logs), """\
$0 = input('input')
$1 = torch._ops.aten.view_copy.default($0, [4, 2])
$2 = torch._ops.aten.add.Tensor($1, tensor([[1., 1.],
        [1., 1.],
        [1., 1.],
        [1., 1.]]))
$3 = torch._ops.aten.mul.Tensor($2, $2)""")

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
        self.assertExpectedInline('\n'.join(logs), """\
$0 = input('input')
$1, $2 = torch._ops.aten.aminmax.default($0, dim=0)""")

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
        self.assertExpectedInline('\n'.join(logs), """\
$0 = input('input')
$1 = torch._ops.aten.view_copy.default($0, [4, 2])
$2 = torch._ops.aten.add.Tensor($0, tensor([[1., 1.],
        [1., 1.],
        [1., 1.],
        [1., 1.]]))""")

    # Some ops that are mutable are neither inplace nor out= ops.
    # They also need special handling.
    def test_mutable_op_not_inplace_or_other(self):
        def f(x):
            return torch._fused_moving_avg_obs_fq_helper(x, x, x, x, x, x, x, 1.0, 0, 1, 0)

        logs = self.get_logs(f, torch.ones(1))
        self.assertExpectedInline('\n'.join(logs), """\
$0 = input('input')
$1, $2, $3, $4, $5, $6 = torch._ops.aten._fused_moving_avg_obs_fq_helper_functional.default($0, $0, $0, $0, $0, $0, $0, 1.0, 0, 1, 0)""")

    def test_as_strided(self):
        def f(x):
            y = x.as_strided((2,), (2,), 1)
            y.add_(1)
            return x
        self.assert_functionalization(f, torch.ones(9))
        logs = self.get_logs(f, torch.ones(9))
        self.assertExpectedInline('\n'.join(logs), """\
$0 = input('input')
$1 = torch._ops.aten.as_strided_copy.default($0, [2], [2], 1)
$2 = torch._ops.aten.add.Tensor($1, 1)""")

    def test_tensor_list_composite(self):
        def f(x):
            # Test an op with TensorList input
            y = torch.block_diag(x, x)
            return y
        self.assert_functionalization(f, torch.ones(2, 2))
        logs = self.get_logs(f, torch.ones(2, 2))
        self.assertExpectedInline('\n'.join(logs), """\
$0 = input('input')
$1 = torch._ops.aten.block_diag.default([LoggingTensor(tensor([[1., 1.],
        [1., 1.]])), LoggingTensor(tensor([[1., 1.],
        [1., 1.]]))])""")

    def test_cat(self):
        def f(x):
            out = torch.empty(0)
            torch.cat((x,), out=out)
            return out
        self.assert_functionalization(f, torch.ones(2, 2))
        logs = self.get_logs(f, torch.ones(2, 2))
        self.assertExpectedInline('\n'.join(logs), """\
$0 = input('input')
$1 = torch._ops.aten.cat.default([LoggingTensor(tensor([[1., 1.],
        [1., 1.]]))])""")

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
        self.assertExpectedInline('\n'.join(logs), """\
$0 = input('input')
$1 = torch._ops.aten.diagonal_copy.default($0)
$2 = torch._ops.aten.add.Tensor($1, tensor([1., 1.]))
$3 = torch._ops.aten.diagonal_scatter.default($0, $2)
$4 = torch._ops.aten.mul.Tensor($3, $3)""")

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
        self.assertExpectedInline('\n'.join(logs), """\
$0 = input('input')
$1, $2 = torch._ops.aten.split_copy.Tensor($0, 2)
$3 = torch._ops.aten.diagonal_copy.default($2)
$4 = torch._ops.aten.add.Tensor($3, tensor([1., 1.]))
$5, $6 = torch._ops.aten.split_copy.Tensor($0, 2)
$7 = torch._ops.aten.diagonal_scatter.default($6, $4)
$8 = torch._ops.aten.slice_scatter.default($0, $7, 0, 2, 4)
$9 = torch._ops.aten.mul.Tensor($8, $8)""")

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
        self.assertExpectedInline('\n'.join(logs), """\
$0 = input('input')
$1 = torch._ops.aten.transpose_copy.int($0, 1, 0)
$2 = torch._ops.aten.select_copy.int($1, 0, 0)
$3 = torch._ops.aten.add.Tensor($2, tensor([1., 1., 1., 1.]))""")

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
        self.assertExpectedInline('\n'.join(logs), """\
$0 = input('input')
$1 = torch._ops.aten.view_copy.default($0, [8])
$2 = torch._ops.aten.index_put.default($1, [tensor([0, 1, 2, 3])], tensor([0., 1., 2., 3.]))""")

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
        self.assertExpectedInline('\n'.join(logs), """\
$0 = input('input')
$1 = torch._ops.aten.view_copy.default($0, [4, 2])
$2 = torch._ops.aten.add.Tensor($1, 1)
$3 = torch._ops.aten.mul.Tensor($2, 2)
$4 = torch._ops.aten.div.Tensor($3, 1)""")

    def test_metadata_change(self):
        def f(x):
            # ops like ge_() are allowed to change the dtype of the input.
            # functionalization should pick up on that.
            return x.ge_(0)
        self.assert_functionalization(f, torch.ones(4, 2))
        logs = self.get_logs(f, torch.ones(4, 2))
        self.assertExpectedInline('\n'.join(logs), """\
$0 = input('input')
$1 = torch._ops.aten.ge.Scalar($0, 0)
$2 = torch._ops.aten._to_copy.default($1, dtype=torch.float32, layout=torch.strided)""")

    def test_only_one_view(self):
        def f(x):
            # This tests that we don't have any unnecessary views in the trace.
            # If the input wasn't mutated, we don't need to regenerate it,
            # so there should be a total of 1 op in the output trace.
            return x.view(4, 2)
        logs = self.get_logs(f, torch.ones(4, 2))
        self.assertExpectedInline('\n'.join(logs), """\
$0 = input('input')
$1 = torch._ops.aten.view_copy.default($0, [4, 2])""")

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
        self.assertExpectedInline('\n'.join(logs), """\
$0 = input('input')
$1 = torch._ops.aten.add.Tensor($0, $0)
$2 = torch._ops.aten.view_copy.default($1, [8])
$3 = torch._ops.aten._reshape_alias_copy.default($2, [2, 4], [4, 1])
$4 = torch._ops.aten.transpose_copy.int($3, 1, 0)
$5 = torch._ops.aten.unsqueeze_copy.default($4, 0)
$6 = torch._ops.aten.squeeze_copy.default($5)
$7, $8 = torch._ops.aten.split_copy.Tensor($6, 2)
$9 = torch._ops.aten.add.Tensor($7, tensor([[1., 1.],
        [1., 1.]]))
$10 = torch._ops.aten.select_copy.int($3, 0, 0)
$11 = torch._ops.aten.clone.default($9, memory_format=torch.contiguous_format)
$12 = torch._ops.aten._unsafe_view.default($11, [4])
$13 = torch._ops.aten.view_copy.default($1, [8])
$14 = torch._ops.aten._reshape_alias_copy.default($13, [2, 4], [4, 1])
$15 = torch._ops.aten.transpose_copy.int($14, 1, 0)
$16 = torch._ops.aten.unsqueeze_copy.default($15, 0)
$17 = torch._ops.aten.squeeze_copy.default($16)
$18 = torch._ops.aten.slice_scatter.default($17, $9, 0, 0, 2)
$19 = torch._ops.aten.unsqueeze_copy.default($18, 0)
$20 = torch._ops.aten.squeeze_copy.dim($19, 0)
$21 = torch._ops.aten.transpose_copy.int($20, 1, 0)
$22 = torch._ops.aten._reshape_alias_copy.default($21, [8], [1])
$23 = torch._ops.aten.view_copy.default($22, [4, 2])
$24 = torch._ops.aten.view_copy.default($23, [8])
$25 = torch._ops.aten._reshape_alias_copy.default($24, [2, 4], [4, 1])
$26 = torch._ops.aten.select_copy.int($25, 0, 0)
$27 = torch._ops.aten.add.Tensor($26, $12)""")

    def test_reapply_views_simple(self):
        def f(x):
            tmp = torch.ones(4, 2)
            y = x.view(4, 2)
            y.add_(tmp)
            z = x * x
            return y
        self.assert_functionalization(f, torch.ones(4, 2), reapply_views=True)
        logs = self.get_logs(f, torch.ones(4, 2), reapply_views=True)
        self.assertExpectedInline('\n'.join(logs), """\
$0 = input('input')
$1 = torch._ops.aten.view.default($0, [4, 2])
$2 = torch._ops.aten.add.Tensor($1, tensor([[1., 1.],
        [1., 1.],
        [1., 1.],
        [1., 1.]]))
$3 = torch._ops.aten.view.default($2, [4, 2])
$4 = torch._ops.aten.mul.Tensor($3, $3)""")

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
        self.assertExpectedInline('\n'.join(logs), """\
$0 = input('input')
$1 = torch._ops.aten.copy.default(tensor([0., 0.]), $0)
$2 = torch._ops.aten.add.Tensor($1, $0)""")

        # Test 2: copy_() with same dtype, different shape
        self.assert_functionalization(f, torch.ones(1))
        logs = self.get_logs(f, torch.ones(1))
        self.assertExpectedInline('\n'.join(logs), """\
$0 = input('input')
$1 = torch._ops.aten.copy.default(tensor([0., 0.]), $0)
$2 = torch._ops.aten.add.Tensor($1, $0)""")

        # Test 3: copy_() with different dtype, same shape
        self.assert_functionalization(f, torch.ones(2, dtype=torch.long))
        logs = self.get_logs(f, torch.ones(2, dtype=torch.long))
        self.assertExpectedInline('\n'.join(logs), """\
$0 = input('input')
$1 = torch._ops.aten.copy.default(tensor([0., 0.]), $0)
$2 = torch._ops.aten.add.Tensor($1, $0)""")

        # Test 4: copy_() with different dtype, different shape
        self.assert_functionalization(f, torch.ones(1, dtype=torch.long))
        logs = self.get_logs(f, torch.ones(1, dtype=torch.long))
        self.assertExpectedInline('\n'.join(logs), """\
$0 = input('input')
$1 = torch._ops.aten.copy.default(tensor([0., 0.]), $0)
$2 = torch._ops.aten.add.Tensor($1, $0)""")

    def test_fill_(self):
        def f(x):
            y = x + x
            z = y.diagonal()
            z.fill_(0)
            return y

        self.assert_functionalization(f, torch.ones(2, 2))
        logs = self.get_logs(f, torch.ones(2, 2))
        self.assertExpectedInline('\n'.join(logs), """\
$0 = input('input')
$1 = torch._ops.aten.add.Tensor($0, $0)
$2 = torch._ops.aten.diagonal_copy.default($1)
$3 = torch._ops.aten.fill.Scalar($2, 0)""")

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
        self.assertExpectedInline('\n'.join(logs), """\
$0 = input('input')
$1 = torch._ops.aten.add.Tensor($0, 1)
$2 = torch._ops.aten.view_copy.default($1, [4, 4])
$3 = torch._ops.aten.resize.default($2, [3, 3])
$4 = torch._ops.aten.as_strided_copy.default($2, [3, 3], [3, 1])
$5 = torch._ops.aten.view_copy.default($4, [-1])
$6 = torch._ops.aten.add.Tensor($5, 1)
$7 = torch._ops.aten.view_copy.default($1, [4, 4])
$8 = torch._ops.aten.as_strided_copy.default($7, [3, 3], [3, 1])
$9 = torch._ops.aten.view_copy.default($6, [3, 3])
$10 = torch._ops.aten.as_strided_scatter.default($7, $9, [3, 3], [3, 1])
$11 = torch._ops.aten.view_copy.default($10, [8, 2])
$12 = torch._ops.aten.view_copy.default($11, [4, 4])
$13 = torch._ops.aten.as_strided_copy.default($12, [3, 3], [3, 1])
$14 = torch._ops.aten.add.Tensor($13, 1)""")

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
        self.assertExpectedInline('\n'.join(logs), """\
$0 = input('input')
$1 = torch._ops.aten.add.Tensor($0, 1)
$2 = torch._ops.aten.resize.default($1, [5, 5])
$3 = torch._ops.aten.view_copy.default($2, [25])
$4 = torch._ops.aten.fill.Scalar($3, 1)
$5 = torch._ops.aten.view_copy.default($4, [5, 5])
$6 = torch._ops.aten.add.Tensor($5, 1)""")

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

    # This tests the behavior of functionalization with multiple layers of wrapped tensor subclasses.
    def test_multiple_levels_of_wrapping(self):
        def f(x):
            # call an inplace op and have it get logged twice (by the outer + inner wrapper)
            x.add_(1)

        # Test 1: both the inner and outer wrapper are "functionalized"
        x_inner_and_outer_functional = torch._to_functional_tensor(
            InplaceLoggingTensor(torch._to_functional_tensor(LoggingTensor(torch.ones(4)))))

        with capture_logs() as logs:
            f(x_inner_and_outer_functional)

        # Since both wrappers were unctionalized, they both log "add"
        self.assertExpectedInline('\n'.join(logs), """\
$1 = torch._ops.aten.add.Tensor($0, 1)
$3 = torch._ops.aten.add.Tensor($2, 1)""")

        # Test 2: only the inner wrapper is "functionalized"
        x_only_inner_functional = InplaceLoggingTensor(torch._to_functional_tensor(LoggingTensor(torch.ones(4))))

        with capture_logs() as logs:
            f(x_only_inner_functional)

        # Since only the inner wrapper is functionalized, then the inner (first) log is functionalized
        self.assertExpectedInline('\n'.join(logs), """\
$1 = torch._ops.aten.add.Tensor($0, 1)
$3 = torch._ops.aten.add_.Tensor($2, 1)""")

        # Test 3: only the inner wrapper is "functionalized"
        x_only_outer_functional = torch._to_functional_tensor(InplaceLoggingTensor(LoggingTensor(torch.ones(4))))

        with capture_logs() as logs:
            f(x_only_outer_functional)

        # Only the outer add_ is functionalized
        # Since only the outer wrapper is functionalized, then the outer (second) log is functionalized
        self.assertExpectedInline('\n'.join(logs), """\
$1 = torch._ops.aten.add_.Tensor($0, 1)
$3 = torch._ops.aten.add.Tensor($2, 1)""")

if __name__ == '__main__':
    run_tests()
