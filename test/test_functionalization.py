# Owner(s): ["module: codegen"]

import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.logging_tensor import LoggingTensor, capture_logs, log_input
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
class InplaceLoggingTensor(LoggingTensor):
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

        rs = tree_map(wrap, f(*tree_map(unwrap, args), **tree_map(unwrap, kwargs)))
        # after running the (potentially transformed) op,
        # log the original op that we saw.
        logging.getLogger("LoggingTensor").info(f"{func.__module__}.{func.__name__}", args, kwargs, rs)
        return rs



class TestFunctionalization(TestCase):

    def get_logs(self, func, inpt):
        input_clone_logging = LoggingTensor(inpt.clone())
        input_functional_logging = torch._to_functional_tensor(input_clone_logging)

        with capture_logs() as logs:
            log_input("input", input_clone_logging)
            torch._enable_functionalization()
            try:
                func(input_functional_logging)
            finally:
                torch._disable_functionalization()
        return logs

    def assert_functionalization(self, func, inpt):
        input_clone = inpt.clone()
        input_clone2 = inpt.clone()
        input_functional = torch._to_functional_tensor(input_clone2)

        # Compare outputs (and mutated inputs), with and without functionalization.
        out_ref = func(inpt)

        torch._enable_functionalization()
        try:
            out_functional = func(input_functional)
        finally:
            torch._disable_functionalization()

        # We need to sync the input tensors first, in case there are any queued mutations left.
        torch._sync(input_functional)
        torch._sync(out_functional)
        self.assertEqual(out_ref, torch._from_functional_tensor(out_functional))
        self.assertEqual(inpt, torch._from_functional_tensor(input_functional))  # input mutations should still occur

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
$1 = torch._ops.aten.view.default($0, [4, 2])
$2 = torch._ops.aten.add.Tensor($1, tensor([[1., 1.],
        [1., 1.],
        [1., 1.],
        [1., 1.]]))
$3 = torch._ops.aten.view.default($2, [4, 2])
$4 = torch._ops.aten.mul.Tensor($3, $3)""")

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
$1 = torch._ops.aten.view.default($0, [4, 2])
$2 = torch._ops.aten.add.Tensor($0, tensor([[1., 1.],
        [1., 1.],
        [1., 1.],
        [1., 1.]]))""")

    def test_tensor_list_composite(self):
        def f(x):
            # Test an op with TensorList input
            y = torch.block_diag(x, x)
            return y
        self.assert_functionalization(f, torch.ones(2, 2))
        logs = self.get_logs(f, torch.ones(2, 2))
        # Only seeing copy_() calls in the logs are actually expected:
        # - block_diag is a CompositeImplicitAutograd op, implemented in terms of copy_() and a few other ops.
        # - copy_() doesn't have an out-of-place variant, so the pass leaves it alone
        # - the other ops are all not called on the input tensor, which means that the LoggingTensor doesn't see them
        # We can update the output of this test if/when these tests eventually use LoggingTensor with PythonMode
        self.assertExpectedInline('\n'.join(logs), """\
$0 = input('input')
$1 = torch._ops.aten.copy_.default(tensor([[1., 1.],
        [1., 1.]]), $0)
$2 = torch._ops.aten.copy_.default(tensor([[1., 1.],
        [1., 1.]]), $0)""")

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
$1 = torch._ops.aten.diagonal.default($0)
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
$1, $2 = torch._ops.aten.split.Tensor($0, 2)
$3 = torch._ops.aten.diagonal.default($2)
$4 = torch._ops.aten.add.Tensor($3, tensor([1., 1.]))
$5, $6 = torch._ops.aten.split.Tensor($0, 2)
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
            return y
        self.assert_functionalization(f, torch.ones(4, 2))
        logs = self.get_logs(f, torch.ones(4, 2))
        self.assertExpectedInline('\n'.join(logs), """\
$0 = input('input')
$1 = torch._ops.aten.transpose.int($0, 1, 0)
$2 = torch._ops.aten.select.int($1, 0, 0)
$3 = torch._ops.aten.add.Tensor($2, tensor([1., 1., 1., 1.]))""")

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
$1 = torch._ops.aten.view.default($0, [4, 2])
$2 = torch._ops.aten.add.Tensor($1, tensor(1))
$3 = torch._ops.aten.mul.Tensor($2, tensor(2))
$4 = torch._ops.aten.div.Tensor($3, tensor(1))""")

    def test_everything(self):
        def f(x):
            # test: everything
            tmp = torch.ones(2, 2)
            y = x.view(8)
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
$1 = torch._ops.aten.view.default($0, [8])
$2 = torch._ops.aten._reshape_alias.default($1, [2, 4], [4, 1])
$3 = torch._ops.aten.transpose.int($2, 1, 0)
$4 = torch._ops.aten.view.default($0, [8])
$5 = torch._ops.aten._reshape_alias.default($4, [2, 4], [4, 1])
$6 = torch._ops.aten.transpose.int($5, 1, 0)
$7 = torch._ops.aten.unsqueeze.default($6, 0)
$8 = torch._ops.aten.view.default($0, [8])
$9 = torch._ops.aten._reshape_alias.default($8, [2, 4], [4, 1])
$10 = torch._ops.aten.transpose.int($9, 1, 0)
$11 = torch._ops.aten.unsqueeze.default($10, 0)
$12 = torch._ops.aten.squeeze.default($11)
$13, $14 = torch._ops.aten.split.Tensor($12, 2)
$15 = torch._ops.aten.add.Tensor($13, tensor([[1., 1.],
        [1., 1.]]))
$16 = torch._ops.aten.select.int($2, 0, 0)
$17 = torch._ops.aten.clone.default($15, memory_format=0)
$18 = torch._ops.aten._unsafe_view.default($17, [4])
$19 = torch._ops.aten.view.default($0, [8])
$20 = torch._ops.aten._reshape_alias.default($19, [2, 4], [4, 1])
$21 = torch._ops.aten.transpose.int($20, 1, 0)
$22 = torch._ops.aten.unsqueeze.default($21, 0)
$23 = torch._ops.aten.squeeze.default($22)
$24 = torch._ops.aten.slice_scatter.default($23, $15, 0, 0, 2)
$25 = torch._ops.aten.unsqueeze.default($24, 0)
$26 = torch._ops.aten.squeeze.dim($25, 0)
$27 = torch._ops.aten.transpose.int($26, 1, 0)
$28 = torch._ops.aten._reshape_alias.default($27, [8], [1])
$29 = torch._ops.aten.view.default($28, [4, 2])
$30 = torch._ops.aten.view.default($29, [8])
$31 = torch._ops.aten._reshape_alias.default($30, [2, 4], [4, 1])
$32 = torch._ops.aten.select.int($31, 0, 0)
$33 = torch._ops.aten.add.Tensor($32, $18)""")

    def test_aliases_maintained_after_pass(self):
        def f(x):
            tmp = torch.ones(4, 2)
            y = x.view(4, 2)
            z = x.view(4, 2)
            y.add_(tmp)
            return y, z

        input_functional = torch._to_functional_tensor(torch.ones(4, 2))
        torch._enable_functionalization()
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
        self.assert_functionalization(f, torch.ones(2))
        logs = self.get_logs(f, torch.ones(2))
        self.assertExpectedInline('\n'.join(logs), """\
$0 = input('input')
$1 = torch._ops.aten.expand.default($0, [2])
$2 = torch._ops.aten.add.Tensor($1, $0)""")

        # Test 2: copy_() with same dtype, different shape
        self.assert_functionalization(f, torch.ones(1))
        logs = self.get_logs(f, torch.ones(1))
        self.assertExpectedInline('\n'.join(logs), """\
$0 = input('input')
$1 = torch._ops.aten.expand.default($0, [2])
$2 = torch._ops.aten.add.Tensor($1, $0)""")

        # Test 3: copy_() with different dtype, same shape
        self.assert_functionalization(f, torch.ones(2, dtype=torch.long))
        logs = self.get_logs(f, torch.ones(2, dtype=torch.long))
        self.assertExpectedInline('\n'.join(logs), """\
$0 = input('input')
$1 = torch._ops.aten._to_copy.default($0, dtype=6, layout=0, device=device(type='cpu'), pin_memory=False)
$2 = torch._ops.aten.expand.default($1, [2])
$3 = torch._ops.aten.add.Tensor($2, $0)""")

        # Test 4: copy_() with different dtype, different shape
        self.assert_functionalization(f, torch.ones(1, dtype=torch.long))
        logs = self.get_logs(f, torch.ones(1, dtype=torch.long))
        self.assertExpectedInline('\n'.join(logs), """\
$0 = input('input')
$1 = torch._ops.aten._to_copy.default($0, dtype=6, layout=0, device=device(type='cpu'), pin_memory=False)
$2 = torch._ops.aten.expand.default($1, [2])
$3 = torch._ops.aten.add.Tensor($2, $0)""")

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

        # I think the alias trace is coming from the fact that x2 is technically *not*
        # a LoggingTensor (instead it *contains* a LoggingTensor), but x1 *is* a LoggingTensor.
        # The important thing here though is that functionalization ran the "+" kernel
        # with a functional + non-functional tensor, and wrapped the output appropriately.
        self.assertExpectedInline('\n'.join(logs), """\
$2 = torch._ops.aten.add.Tensor($0, $1)
$3 = torch._ops.aten.alias.default($2)
$4 = torch._ops.aten.add.Tensor($3, tensor(1))""")

    def test_mixed_wrappers_invalid(self):
        x1_not_functional = torch.ones(4)
        x2_functional = torch._to_functional_tensor(torch.ones(4))

        # When dealing with mixed functional + nonfunctional tensors,
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
$1 = torch._ops.aten.add.Tensor($0, tensor(1))
$3 = torch._ops.aten.add.Tensor($2, tensor(1))""")

        # Test 2: only the inner wrapper is "functionalized"
        x_only_inner_functional = InplaceLoggingTensor(torch._to_functional_tensor(LoggingTensor(torch.ones(4))))

        with capture_logs() as logs:
            f(x_only_inner_functional)

        # Since only the inner wrapper is functionalized, then the inner (first) log is functionalized
        self.assertExpectedInline('\n'.join(logs), """\
$1 = torch._ops.aten.add.Tensor($0, tensor(1))
$3 = torch._ops.aten.add_.Tensor($2, tensor(1))""")

        # Test 3: only the inner wrapper is "functionalized"
        x_only_outer_functional = torch._to_functional_tensor(InplaceLoggingTensor(LoggingTensor(torch.ones(4))))

        with capture_logs() as logs:
            f(x_only_outer_functional)

        # Only the outer add_ is functionalized
        # Since only the outer wrapper is functionalized, then the outer (second) log is functionalized
        self.assertExpectedInline('\n'.join(logs), """\
$1 = torch._ops.aten.add_.Tensor($0, tensor(1))
$3 = torch._ops.aten.add.Tensor($2, tensor(1))""")

if __name__ == '__main__':
    run_tests()
