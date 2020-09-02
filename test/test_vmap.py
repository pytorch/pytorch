from torch.testing._internal.common_utils import TestCase, run_tests
import torch
from torch import Tensor, vmap
import functools
import warnings
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import TEST_WITH_ROCM

class TestVmapAPI(TestCase):
    def test_non_tensor_output_raises(self):
        with self.assertRaisesRegex(ValueError, "got type <class 'float'> as the return"):
            output = vmap(lambda x: 3.14)(torch.ones(3))

        def multiple_outputs(x):
            return x, 3

        with self.assertRaisesRegex(ValueError, "got type <class 'int'> for return 1"):
            vmap(multiple_outputs)(torch.ones(3))

    def test_different_map_dim_size_raises(self):
        x = torch.randn(2)
        y = torch.randn(3)
        expected_msg = 'Expected all tensors to have the same size in the mapped dimension'
        with self.assertRaisesRegex(ValueError, expected_msg):
            vmap(torch.mul)(x, y)

    def test_func_with_no_inputs(self):
        expected_msg = 'got no inputs'

        def foo():
            return torch.randn(3)

        def bar(x):
            return torch.randn(3)

        with self.assertRaisesRegex(ValueError, expected_msg):
            vmap(foo)()

        with self.assertRaisesRegex(ValueError, expected_msg):
            vmap(bar)()

    def test_constant_function(self):
        output = vmap(lambda x: torch.tensor(3.14))(torch.ones(3))
        self.assertEqual(output, torch.tensor([3.14, 3.14, 3.14]))

    def test_single_input(self):
        x = torch.randn(2, 3)

        def square(x):
            return x * x

        output = vmap(square)(x)
        self.assertEqual(output, x * x)

    def test_multiple_inputs(self):
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        output = vmap(torch.mul)(x, y)
        self.assertEqual(output, x * y)

    def test_multiple_outputs(self):
        def foo(x):
            return x * x, x * x * x

        x = torch.randn(3)
        outputs = vmap(foo)(x)
        self.assertEqual(outputs[0], x * x)
        self.assertEqual(outputs[1], x * x * x)

    def test_multiple_outputs_error_cases(self):
        # This is the same thing as
        # def returns_tuple_of_tensors(x):
        #     return x, x
        def returns_tuple_of_tensors(x):
            return (x, x)

        def returns_list_of_two_tensors(x):
            return [x, x]

        def returns_list_of_one_tensor(x):
            return [x]

        x = torch.randn(3)

        # should not throw
        vmap(returns_tuple_of_tensors)(x)

        # jax supports these, but we don't yet
        msg = "must only return Tensors, got type <class 'list'>"
        with self.assertRaisesRegex(ValueError, msg):
            vmap(returns_list_of_two_tensors)(x)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(returns_list_of_one_tensor)(x)

    def test_nested_with_same_map_dim(self):
        x = torch.randn(2, 3, 5)
        y = torch.randn(2, 3, 5)
        output = vmap(vmap(torch.mul))(x, y)
        self.assertEqual(output, x * y)

        output = vmap(vmap(vmap(torch.mul)))(x, y)
        self.assertEqual(output, x * y)

    def test_nested_with_different_map_dim(self):
        x = torch.randn(2, 3)
        y = torch.randn(5, 3)
        output = vmap(lambda x: vmap(lambda y: x * y)(y))(x)
        self.assertEqual(output.shape, (2, 5, 3))
        self.assertEqual(output, x.view(2, 1, 3) * y)

        z = torch.randn(7, 3)
        output = vmap(lambda x: vmap(lambda y: vmap(lambda z: x * y * z)(z))(y))(x)
        self.assertEqual(output.shape, (2, 5, 7, 3))
        self.assertEqual(output, x.view(2, 1, 1, 3) * y.view(5, 1, 3) * z)

    def test_noop_in_inner_vmap(self):
        x = torch.randn(3)
        y = torch.randn(5)
        output = vmap(lambda x: vmap(lambda y: x)(y))(x)
        self.assertEqual(output, x.view(3, 1).expand(3, 5))

    def test_unsupported_op_err_msg(self):
        # Unsupported view op
        tensor = torch.randn(2, 3)
        with self.assertRaisesRegex(RuntimeError, "doesn't work on in-place or view ops"):
            vmap(torch.as_strided, (0, None, None))(tensor, [2, 3], [0, 0])

        # The fallback doesn't support TensorList
        with self.assertRaisesRegex(RuntimeError, 'Batching rule not implemented'):
            vmap(lambda t: torch.stack([t]))(tensor)

        # Don't support non-tensor returns. This is a limitation of vmap;
        # functions that don't return tensors must be special cased
        with self.assertRaisesRegex(RuntimeError, 'Batching rule not implemented'):
            vmap(torch.Tensor.item)(tensor)

    def test_unsupported_inplace_op_err_msg(self):
        def foo(x):
            return x.cos_()

        x = torch.randn(3)
        with self.assertRaisesRegex(
                RuntimeError, 'Batching rule not implemented'):
            vmap(foo)(x)

    def test_nonzero_out_dims(self):
        # Basic test
        tensor = torch.randn(2, 3)
        result = vmap(lambda x: x, out_dims=1)(tensor)
        self.assertEqual(result, tensor.permute(1, 0))
        self.assertEqual(result.data_ptr(), tensor.data_ptr())

        # Test that the batch dimension gets permuted to dim 2
        tensor = torch.randn(2, 3, 5, 7)
        result = vmap(lambda x: x, out_dims=2)(tensor)
        self.assertEqual(result, tensor.permute(1, 2, 0, 3))
        self.assertEqual(result.data_ptr(), tensor.data_ptr())

        # negative out_dim
        tensor = torch.randn(2, 3, 5, 7)
        result = vmap(lambda x: x, out_dims=-1)(tensor)
        self.assertEqual(result, tensor.permute(1, 2, 3, 0))
        self.assertEqual(result.data_ptr(), tensor.data_ptr())

        # check that out_dims works on ALL outputs
        tensor = torch.randn(2, 3, 5, 7)
        other = torch.randn(2, 3, 5, 7)
        result = vmap(lambda x, y: (x, y), out_dims=2)(tensor, other)
        self.assertEqual(result, (tensor.permute(1, 2, 0, 3), other.permute(1, 2, 0, 3)))

        # use out_dims with the maximum vmap-able tensor dims (64 dims)
        ndims = 64
        shape = [2] + [1] * (ndims - 1)
        expected_shape = [1, 1, 2] + [1] * (ndims - 3)
        tensor = torch.randn(shape)
        result = vmap(lambda x: x, out_dims=2)(tensor)
        self.assertEqual(result.shape, expected_shape)

        # test something that is not the identity function
        def foo(x, y):
            return x, x * y, x * y * y
        x = torch.randn(2, 3, 5)
        y = torch.randn(2, 3, 5)
        result = vmap(foo, out_dims=1)(x, y)
        self.assertEqual(
            result,
            (x.permute(1, 0, 2), (x * y).permute(1, 0, 2), (x * y * y).permute(1, 0, 2)))

    def test_multiple_out_dims(self):
        def foo(x):
            return x, x

        def bar(x, y):
            return x, x, x, x * y

        x = torch.randn(2, 3, 5)
        y = torch.randn(2, 3, 5)
        result = vmap(foo, out_dims=(0, 1))(x)
        self.assertEqual(result, (x, x.permute(1, 0, 2)))

        result = vmap(bar, out_dims=(-1, 0, 1, 2))(x, y)
        expected = (
            x.permute(1, 2, 0),
            x,
            x.permute(1, 0, 2),
            (x * y).permute(1, 2, 0),
        )
        self.assertEqual(result, expected)

    def test_nested_out_dims(self):
        y = torch.randn(2, 3, 5, 7)

        # Inner vmap has non-zero out_dim
        result = vmap(lambda y: vmap(lambda x: x, out_dims=1)(y))(y)
        self.assertEqual(result.shape, (2, 5, 3, 7))
        self.assertEqual(result, y.permute(0, 2, 1, 3))

        # all vmaps have non-zero out_dim
        result = vmap(lambda y: vmap(lambda x: x, out_dims=1)(y), out_dims=1)(y)
        self.assertEqual(result.shape, (5, 2, 3, 7))
        self.assertEqual(result, y.permute(2, 0, 1, 3))

        # throwing in some negative out_dims
        result = vmap(lambda y: vmap(lambda x: x, out_dims=-1)(y), out_dims=-1)(y)
        self.assertEqual(result.shape, (5, 7, 3, 2))
        self.assertEqual(result, y.permute(2, 3, 1, 0))

        # testing fn that isn't the identity
        x = torch.randn(2, 3)
        y = torch.randn(5, 3)
        result = vmap(lambda y: vmap(lambda x: x * y, out_dims=1)(x), out_dims=-1)(y)
        self.assertEqual(result.shape, (3, 2, 5))
        self.assertEqual(result, (y.view(5, 1, 3) * x).permute(2, 1, 0))

    def test_out_dims_edge_case(self):
        def foo(x):
            return x

        # Test that we accept out_dims=(1,) for a function with one output.
        tensor = torch.randn(2, 3)
        expected = vmap(foo, out_dims=1)(tensor)
        result = vmap(foo, out_dims=(1,))(tensor)
        self.assertEqual(result, expected)

    def test_out_dims_must_be_int_or_tuple_of_int_err_msg(self):
        msg = '`out_dims` must be an int or a tuple of int'
        tensor = torch.randn(2, 3)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: x, out_dims='lol')(tensor)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: x, out_dims=('lol',))(tensor)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: x, out_dims=None)(tensor)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: x, out_dims=(None,))(tensor)

    def test_out_dims_and_num_outputs_mismatch_err_msg(self):
        msg = '`out_dims` must have one dim per output'
        x = torch.randn(2, 3, 5)

        # Too many out_dims
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: x, out_dims=(0, 0))(x)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: (x, x, x), out_dims=(0, 0, 0, 0))(x)

        # Too few out_dims
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: (x, x), out_dims=(0,))(x)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: (x, x, x), out_dims=(0, 0))(x)

    def test_out_dim_out_of_bounds_err_msg(self):
        # TODO(rzou): This error message isn't that great. It comes straight
        # from maybe_wrap_dim. Consider doing a try-catch-(add some context) to
        # the error message in the future in C++
        msg = 'Dimension out of range'
        x = torch.randn(2, 3, 5)
        with self.assertRaisesRegex(IndexError, msg):
            vmap(lambda x: x, out_dims=3)(x)
        with self.assertRaisesRegex(IndexError, msg):
            vmap(lambda x: x, out_dims=-4)(x)

    def test_non_zero_in_dims(self):
        tensor = torch.randn(2, 3, 5)

        # Implicit out_dims = 0; vmap will move the batch dim to the front.
        output = vmap(lambda x: x, (1,))(tensor)
        self.assertEqual(output, tensor.permute(1, 0, 2))
        self.assertEqual(output.data_ptr(), tensor.data_ptr())

        x = torch.randn(2, 3)
        y = torch.randn(3, 2)
        output = vmap(torch.mul, (0, 1))(x, y)
        self.assertEqual(output, x * y.t())
        output = vmap(torch.mul, (1, 0))(x, y)
        self.assertEqual(output, x.t() * y)

    def test_none_in_dims(self):
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)

        # None in_dim for a Tensor means we don't map over it
        output = vmap(torch.mul, (0, None))(x, y)
        self.assertEqual(output.shape, (2, 2, 3))
        self.assertEqual(output, x.view(2, 1, 3) * y)

        # None in_dim for non-tensor arguments
        output = vmap(torch.mul, (0, None))(x, 2)
        self.assertEqual(output, x * 2)

    def test_nested_non_default_in_dims(self):
        x = torch.rand(5, 2, 3)
        y = torch.rand(3, 5, 2)
        result = vmap(vmap(vmap(torch.mul), (1, 0)), (1, 2))(x, y)
        self.assertEqual(result, x.permute(1, 2, 0) * y.permute(2, 0, 1))

    def test_non_default_in_dims_out_dims(self):
        x = torch.randn(2, 3, 5)

        # Same in_dim as out_dim, vmap over identity
        result = vmap(lambda x: x, in_dims=1, out_dims=1)(x)
        self.assertEqual(result, x)
        self.assertEqual(result.data_ptr(), x.data_ptr())

        # Different in_dim from out_dim, vmap over identity
        result = vmap(lambda x: x, in_dims=2, out_dims=1)(x)
        self.assertEqual(result.shape, (2, 5, 3))
        self.assertEqual(result, x.transpose(1, 2))
        self.assertEqual(result.data_ptr(), x.data_ptr())

        def foo(x):
            return x * 2

        # Same in_dim as out_dim, vmap over operation
        result = vmap(foo, in_dims=1, out_dims=1)(x)
        self.assertEqual(result, x * 2)

        # Different in_dim as out_dim, vmap over operation
        result = vmap(foo, in_dims=2, out_dims=1)(x)
        self.assertEqual(result.shape, (2, 5, 3))
        self.assertEqual(result, (x * 2).transpose(1, 2))

        # Basic nested test.
        result = vmap(vmap(foo, 1, 1), 1, 1)(x)
        self.assertEqual(result, x * 2)

    def test_in_dims_wrong_type_err_msg(self):
        x = torch.randn(3)
        y = torch.randn(3)
        msg = 'expected `in_dims` to be int or tuple'
        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.mul, [0, 0])(x, y)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.mul, set({0, 0}))(x, y)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.mul, 'lol')(x, y)
        # The following should not throw
        vmap(torch.mul, (0, 0))(x, y)

    def test_not_enough_in_dims_err_msg(self):
        x = torch.randn(3)
        y = torch.randn(3)
        msg = r'expected one `in_dim` per input \(got \w+ inputs\)'

        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.mul, (0,))(x, y)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.mul, (0, 0, 0))(x, y)
        # The following should not throw
        vmap(torch.mul, (0, 0))(x, y)

    def test_in_dims_must_be_flat_tuple_err_msg(self):
        msg = 'in_dims must be a flat tuple containing ints and/or Nones'

        x = torch.randn(3)
        y = torch.randn(3)
        z = torch.randn(3)

        def foo(xy):
            return xy[0] * xy[1]

        def bar(x, yz):
            return x * yz[0] * yz[1]

        # NB: jax supports all of the following, we don't yet.
        with self.assertRaisesRegex(ValueError, msg):
            vmap(foo, ((0, 0),))((x, y))
        with self.assertRaisesRegex(ValueError, msg):
            vmap(bar, (0, (0, 0)))(x, (y, z))
        with self.assertRaisesRegex(ValueError, msg):
            vmap(foo, ({0: 0, 1: 0},))({0: x, 1: y})

    def test_integer_in_dim_but_not_tensor_input_err_msg(self):
        def foo(xy):
            return xy[0] * xy[1]

        def bar(x, yz):
            return x * yz[0] * yz[1]

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)

        # jax supports these, we too can in the future.
        msg = 'Got in_dim=0 for input 0, but input 0 is not a Tensor'
        with self.assertRaisesRegex(ValueError, msg):
            vmap(foo)((x, y))
        with self.assertRaisesRegex(ValueError, msg):
            vmap(foo, (0,))((x, y))

        # jax supports these as well, we too can in the future.
        msg = 'Got in_dim=0 for input 1, but input 1 is not a Tensor'
        with self.assertRaisesRegex(ValueError, msg):
            vmap(foo)(x, (x, y))
        with self.assertRaisesRegex(ValueError, msg):
            vmap(foo, (0, 0))(x, (x, y))

        # the following are errors in jax (and will always be errors)
        msg = 'Got in_dim=0 for input 1, but input 1 is not a Tensor'
        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.sum)(x, 0)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.sum, (0, 0))(x, 0)
        # The following should not throw
        vmap(torch.sum, (0, None))(x, 0)

    def test_in_dim_not_in_tensor_err_msg(self):
        def foo(x):
            return x * x

        msg = r'Got in_dim=-?\w for input 0, but input 0 is a Tensor of dimensionality \w'
        with self.assertRaisesRegex(ValueError, msg):
            vmap(foo)(torch.randn([]))
        with self.assertRaisesRegex(ValueError, msg):
            vmap(foo, in_dims=(0,))(torch.randn([]))
        with self.assertRaisesRegex(ValueError, msg):
            vmap(foo, in_dims=(-1,))(torch.randn(2, 3))
        with self.assertRaisesRegex(ValueError, msg):
            vmap(foo, in_dims=(2,))(torch.randn(2, 3))
        # the following should not throw
        vmap(foo, in_dims=(0,))(torch.randn(2, 3))
        vmap(foo, in_dims=(1,))(torch.randn(2, 3))

    def _assert_uses_vmap_fallback(self, vmap_args, inputs):
        with warnings.catch_warnings(record=True) as wa:
            result = vmap(*vmap_args)(*inputs)
            self.assertEqual(len(wa), 2)
            self.assertRegex(str(wa[-1].message),
                             r'falling back to slow \(for loop and stack\) implementation')

    def test_fallback_atan2(self):
        # NB: One day we will implement a batching rule for torch.atan2.
        # If/when we do, this test should be replaced to test the fallback
        # path on another operator to avoid bitrot.
        op = torch.atan2

        x = torch.randn(5, 7, 11)
        y = torch.randn(5, 7, 11)

        self._assert_uses_vmap_fallback((op,), (x, y))

        # fallback on torch.sub
        x = torch.randn(7, 11, 5)
        y = torch.randn(5, 7, 11)
        result = vmap(op, (2, 0))(x, y)
        self.assertEqual(result, op(x.permute(2, 0, 1), y))

        # fallback on torch.sub, nested vmap
        x = torch.randn(7, 11, 5)
        y = torch.randn(5, 7, 11)
        result = vmap(vmap(op), (2, 0))(x, y)
        self.assertEqual(result, op(x.permute(2, 0, 1), y))

        # big batch size (total 10000)
        x = torch.randn(100, 10, 10, 5)
        y = torch.randn(100, 10, 10)
        result = vmap(vmap(vmap(op)))(x, y)
        self.assertEqual(result, op(x, y.view(100, 10, 10, 1)))

    def test_fallback_masked_fill(self):
        # NB: One day we will implement a batching rule for masked_fill
        # If/when we do, this test should be replaced to test the fallback
        # path on another operator to avoid bitrot.
        def run_test(batch_size):
            B0 = batch_size
            x = torch.randn(B0, 7, 11, 13)
            dim = 0
            index = torch.tensor([0, 4, 2])
            values = torch.randn(B0, 3, 13)

            self._assert_uses_vmap_fallback((torch.index_add, (0, None, None, 0)), (x, dim, index, values))

            result = vmap(torch.index_add, (0, None, None, 0))(x, dim, index, values)
            expected = torch.index_add(
                x, dim + 1, index, values.view(B0, 3, 1, 13))
            self.assertEqual(result, expected)

        run_test(batch_size=5)
        run_test(batch_size=1237)

    def test_fallback_multiple_returns(self):
        # NB: One day we will implement a batching rule for torch.var_mean
        # If/when we do, this test should be replaced to test the fallback
        # path on another operator to avoid bitrot.
        B0, B1, B2 = 2, 3, 1237
        tensor = torch.randn(B0, 10)

        self._assert_uses_vmap_fallback((torch.var_mean,), (tensor,))

        # fallback correctness on torch.var_mean
        result = vmap(torch.var_mean)(tensor)
        expected = torch.var_mean(tensor, dim=1)
        self.assertEqual(result, expected)

        # nested vmap
        tensor = torch.randn(B0, B1, 10)
        result = vmap(vmap(torch.var_mean))(tensor)
        expected = torch.var_mean(tensor, dim=2)
        self.assertEqual(result, expected)

        # big batch size, nested vmap
        tensor = torch.randn(B0, B1, B2, 10)
        result = vmap(vmap(vmap(torch.var_mean)))(tensor)
        expected = torch.var_mean(tensor, dim=3)
        self.assertEqual(result, expected)

    def test_backward_unsupported_interaction(self):
        x = torch.randn(3, requires_grad=True)
        y = torch.randn(5)
        grad = torch.randn_like(x)
        err_msg = r'backward\(\) called inside torch.vmap'

        def backward_on_vmapped_tensor(x):
            x.sum().backward()

        with self.assertRaisesRegex(RuntimeError, err_msg):
            vmap(backward_on_vmapped_tensor)(x)

        def backward_with_vmapped_grad(x, grad):
            x.backward(grad)

        with self.assertRaisesRegex(RuntimeError, err_msg):
            vmap(backward_with_vmapped_grad)(x, grad)

        def completely_unrelated_backward(y):
            x.sum().backward()

        with self.assertRaisesRegex(RuntimeError, err_msg):
            vmap(completely_unrelated_backward)(y)

    def test_grad_unsupported_interaction(self):
        input_tensor = torch.randn(3, requires_grad=True)
        err_msg = 'autograd.grad.* called inside torch.vmap'

        captured = torch.randn(3, requires_grad=True)

        def output_to_grad_is_vmapped(input_tensor):
            output = (captured * input_tensor).sum()
            return torch.autograd.grad([output], [captured])[0]

        with self.assertRaisesRegex(RuntimeError, err_msg):
            vmap(output_to_grad_is_vmapped)(input_tensor)

        output = (input_tensor ** 2).sum()

        def input_to_grad_is_vmapped(input_tensor):
            return torch.autograd.grad([output], [input_tensor])[0]

        with self.assertRaisesRegex(RuntimeError, err_msg):
            vmap(input_to_grad_is_vmapped)(input_tensor)

    def test_batched_gradient_basic(self):
        N = 3
        x = torch.randn(N, requires_grad=True)
        y = torch.randn(N)

        def vjp_mul(v):
            return torch.autograd.grad([x * y], [x], grad_outputs=[v])[0]

        batched_v = torch.eye(N)
        jacobian = vmap(vjp_mul)(batched_v)
        self.assertEqual(jacobian, torch.diagflat(y))

    def test_functools_partial(self):
        x = torch.randn(3)
        y = torch.randn(2, 3)
        result = vmap(functools.partial(torch.mul, x))(y)
        self.assertEqual(result, x * y)

    def test_nn_module(self):
        tensor = torch.randn(2, 3)
        model = torch.nn.Linear(3, 3, bias=False)
        result = vmap(model)(tensor)
        self.assertEqual(result, model(tensor))


def slice_inputs(inputs, bdims, i):
    result = []
    for inp, bdim in zip(inputs, bdims):
        if bdim is None:
            result.append(inp)
        else:
            result.append(inp.select(bdim, i))
    return tuple(result)


def reference_vmap(op, inputs, in_dims=0, out_dims=0):
    if isinstance(in_dims, int):
        in_dims = (in_dims,) * len(inputs)
    bdim_sizes = [inp.size(dim) for inp, dim in zip(inputs, in_dims) if dim is not None]
    assert all(bdim_size == bdim_sizes[0] for bdim_size in bdim_sizes)
    bdim_size = bdim_sizes[0]
    results = tuple(op(*slice_inputs(inputs, in_dims, i)) for i in range(bdim_size))

    assert len(results) > 0
    op_has_single_return = not isinstance(results[0], tuple)
    if op_has_single_return:
        assert all(isinstance(result, torch.Tensor) for result in results)
        if isinstance(out_dims, int):
            out_dims = (out_dims,) * 1
        return torch.stack(results, dim=out_dims[0])

    assert all(isinstance(result, tuple) for result in results)
    num_returns = len(results[0])
    assert all(len(result) == num_returns for result in results)
    if isinstance(out_dims, int):
        out_dims = (out_dims,) * num_returns
    return tuple(torch.stack(result_shards, out_dim)
                 for result_shards, out_dim in zip(zip(*results), out_dims))


class TensorFactory:
    @staticmethod
    def rand(size, device='cpu', dtype=torch.float):
        return torch.rand(size, device=device, dtype=dtype)

    @staticmethod
    def randn(size, device='cpu', dtype=torch.float):
        return torch.randn(size, device=device, dtype=dtype)

    @staticmethod
    def randp1(size, device='cpu', dtype=torch.float):
        return torch.rand(size, device=device, dtype=dtype) + 1

# Tests vmap(op, in_dims, out_dims)(*inputs) by comparing the output to a
# (slow) sequential map+stack fallback.
#
# check_view: Test if the first returned output is a view of the first input
# check_propagates_grad: Test if the operation propagates gradients.
def _vmap_test(self, op, inputs, in_dims=0, out_dims=0,
               check_view=False, check_propagates_grad=True):
    result = vmap(op, in_dims, out_dims)(*inputs)
    reference_result = reference_vmap(op, inputs, in_dims, out_dims)
    self.assertEqual(result, reference_result)
    op_has_single_return = not isinstance(result, tuple)

    if check_view:
        result_as_tuple = (result,) if op_has_single_return else result
        for output in result_as_tuple:
            input0_base = inputs[0] if inputs[0]._base is None else inputs[0]._base
            self.assertTrue(output._base is input0_base,
                            msg="result was not a view of the first input!")

    if not check_propagates_grad:
        return
    # Assuming input[0] is a floating-point tensor. Check if the vmap
    # operation propagates the requires_grad flag to the zeroth output.
    # Some vmap operators are implemented in a way that assumes that
    # they are composite with respect to autograd. If the operator ever is
    # changed to not be composite with respect to autograd, then the
    # following check should fail.
    inputs_clone = list(inputs)
    inputs_clone[0] = inputs[0].clone().requires_grad_()
    result = vmap(op, in_dims, out_dims)(*inputs_clone)
    result_as_tuple = (result,) if op_has_single_return else result
    self.assertTrue(result[0].requires_grad)

class TestVmapOperators(TestCase):
    def _vmap_test(self, *args, **kwargs):
        return _vmap_test(self, *args, **kwargs)

    def _vmap_view_test(self, *args, **kwargs):
        self._vmap_test(*args, **kwargs, check_view=True)

    def _assert_doesnt_use_vmap_fallback(self, vmap_args, inputs):
        regex = r'falling back to slow \(for loop and stack\) implementation'
        with warnings.catch_warnings(record=True) as wa:
            result = vmap(*vmap_args)(*inputs)
            for captured_warning in wa:
                self.assertNotRegex(str(captured_warning.message), regex)

    def test_assert_doesnt_use_vmap_fallback(self):
        with self.assertRaises(AssertionError):
            # One day we'll implement a batching rule for torch.var_mean.
            # When that happens, please change the example to use an
            # operator that doesn't have a batching rule implemented.
            self._assert_doesnt_use_vmap_fallback([torch.var_mean], [torch.rand(3)])

    def _test_unary(self, op, getter, device):
        test = self._vmap_test
        B0, B1 = 7, 11

        self._assert_doesnt_use_vmap_fallback([op], [getter([B0], device)])

        # Single vmap, various in_dims / out_dims
        test(op, [getter([B0, 3], device)])
        test(op, [getter([2, 5, B0, 3], device)], in_dims=2)
        test(op, [getter([2, 5, B0, 3], device)], in_dims=2, out_dims=2)

        # Doubly nested vmap
        test(vmap(op), [getter([B0, B1], device)])
        test(vmap(op), [getter([B1, 2, 5, B0, 3], device)], in_dims=2)
        test(vmap(op, in_dims=2), [getter([2, 5, B0, B1, 3], device)],
             in_dims=2, out_dims=2)

    def test_unary_pointwise_ops(self):
        cases = [
            (torch.abs, TensorFactory.randn),
            (torch.acos, TensorFactory.rand),
            (torch.asin, TensorFactory.rand),
            (torch.atan, TensorFactory.rand),
            (torch.ceil, TensorFactory.randn),
            (torch.cos, TensorFactory.rand),
            (torch.cosh, TensorFactory.rand),
            (torch.digamma, TensorFactory.rand),
            (torch.exp, TensorFactory.randn),
            (torch.expm1, TensorFactory.randn),
            (torch.floor, TensorFactory.randn),
            (torch.frac, TensorFactory.randn),
            (torch.lgamma, TensorFactory.rand),
            (torch.log, TensorFactory.randp1),
            (torch.log10, TensorFactory.randp1),
            (torch.log1p, TensorFactory.randp1),
            (torch.log2, TensorFactory.randp1),
            (torch.neg, TensorFactory.randn),
            (torch.reciprocal, TensorFactory.randp1),
            (torch.relu, TensorFactory.randn),
            (torch.round, TensorFactory.randn),
            (torch.rsqrt, TensorFactory.randp1),
            (torch.sigmoid, TensorFactory.randn),
            (torch.sign, TensorFactory.randn),
            (torch.sin, TensorFactory.rand),
            (torch.sinh, TensorFactory.rand),
            (torch.sqrt, TensorFactory.rand),
            (torch.tan, TensorFactory.rand),
            (torch.tanh, TensorFactory.rand),
            (torch.trunc, TensorFactory.randn),
        ]
        for op, getter in cases:
            self._test_unary(op, getter, 'cpu')

    def test_binary_pointwise_ops(self):
        def get_number(getter):
            return getter([]).item()

        def make_case(op, input_getter=TensorFactory.randn):
            return (op, input_getter)

        cases = [
            # Basic arithmetic
            make_case(torch.add),
            make_case(lambda x, y: x + y),
            make_case(torch.sub),
            make_case(lambda x, y: x - y),
            make_case(torch.mul),
            make_case(lambda x, y: x * y),
            make_case(torch.div, input_getter=TensorFactory.randp1),
            make_case(lambda x, y: x / y, input_getter=TensorFactory.randp1),
            make_case(torch.pow, input_getter=TensorFactory.randp1),
            make_case(lambda x, y: x ** y, input_getter=TensorFactory.randp1),
        ]
        test = self._vmap_test

        for op, getter in cases:
            device = 'cpu'
            B0, B1 = 7, 11

            self._assert_doesnt_use_vmap_fallback(
                [op], (getter([B0], device), getter([B0], device)))

            # Single vmap: op(Tensor, Tensor)
            test(op, (getter([B0, 3], device), getter([B0, 3], device)))
            test(op, (getter([B0], device), getter([B0, 2, 3], device)))
            test(op, (getter([B0], device), getter([2, B0, 3], device)), in_dims=(0, 1))
            test(op, (getter([B0], device), getter([2, B0, 3], device)),
                 in_dims=(0, 1), out_dims=1)
            test(op, (getter([B0], device), getter([2, 3], device)), in_dims=(0, None))
            test(op, (getter([2, 3], device), getter([B0, 3], device)), in_dims=(0, None))

            # Nested vmap: op(Tensor, Tensor)
            test(vmap(op), (getter([B0, B1, 2, 3], device), getter([B0, B1, 3], device)))
            test(vmap(op, in_dims=(None, 0)),
                 (getter([B0, 2, 3], device), getter([B1, 3], device)), in_dims=(0, None))

            # Python number overload: op(Tensor, Number) (and vice-versa)
            number = get_number(getter)
            self._test_unary(lambda t: op(t, number), getter, device)
            number = get_number(getter)
            self._test_unary(lambda t: op(number, t), getter, device)

            # Type promotion: op(Logical Scalar Tensor, Logical Scalar Tensor)
            test(op, (getter([B0], device), getter([B0], device, dtype=torch.double)))
            test(op, (getter([B0], device, dtype=torch.double), getter([B0], device)))
            test(op, (getter([B0], device), getter([B0], device)))

            # Type promotion: op(Tensor, Logical Scalar Tensor) (and vice-versa)
            test(op, (getter([B0, 2], device), getter([B0], device, torch.double)))
            test(op, (getter([B0], device, torch.double), getter([B0, 2], device)))

            if not torch.cuda.is_available():
                continue

            # Test cross-device scalars
            number = get_number(getter)
            self._test_unary(lambda t: op(t, number), getter, device='cuda')
            self._test_unary(lambda t: op(number, t), getter, device='cuda')
            self._test_unary(lambda t: op(t, torch.tensor(number)), getter, device='cuda')

    def test_chunk(self):
        test = self._vmap_view_test
        op = torch.chunk
        B0, B1, B2 = 7, 11, 13

        # tests for torch.split(self, split_size: int, dim)
        test(op, (torch.rand(B0, 2, 1024), 15, -1), in_dims=(0, None, None))
        test(op, (torch.rand(2, B0, 1024), 9, 1), in_dims=(1, None, None))
        test(vmap(op, in_dims=(0, None, None)), (torch.rand(B1, 1023, B0, 5), 4, 0),
             in_dims=(2, None, None))
        test(vmap(vmap(lambda t: op(t, 4, 1), in_dims=2)),
             (torch.rand(B1, 2, B0, 64, B2),), in_dims=2)

    def test_diagonal(self):
        tensor = torch.randn(3, 5, 7, 11, 13)
        test = self._vmap_view_test
        op = torch.diagonal
        test(op, (tensor, 1, 0, 1), in_dims=(0, None, None, None))
        test(op, (tensor, 0, 2, -1), in_dims=(0, None, None, None))
        test(op, (tensor, 2, 1, 2), in_dims=(1, None, None, None))
        test(op, (tensor, 0, -2, -1), in_dims=(1, None, None, None), out_dims=1)
        test(vmap(lambda t: op(t, 0, 0, -1)), (tensor,), in_dims=1, out_dims=1)
        test(vmap(vmap(lambda t: op(t, 0, 0, 1), in_dims=1), in_dims=3),
             (tensor,), in_dims=1, out_dims=1)

    def test_expand_as(self):
        op = torch.Tensor.expand_as
        test = self._vmap_view_test
        B0, B1, B2 = 7, 11, 13
        test(op, (torch.rand(B0, 1, 5), torch.rand(B0, 2, 3, 5)))
        test(op, (torch.rand(B0, 1, 5), torch.rand(2, 3, 5)), in_dims=(0, None))
        test(op, (torch.rand(1, 5), torch.rand(B0, 2, 3, 5)), in_dims=(None, 0))
        test(vmap(op), (torch.rand(B0, B1, 1, 5), torch.rand(B0, B1, 2, 3, 5)))
        test(vmap(op), (torch.rand(B0, B1, 1, 5), torch.rand(B1, B0, 2, 3, 5)), in_dims=(0, 1))
        test(vmap(op), (torch.rand(B0, B1), torch.rand(B1, 2, 3, 5)), in_dims=(0, None))
        test(vmap(vmap(op)), (torch.rand(B0, B1, B2), torch.rand(B0, B1, B2, 2, 3, 5)))

    def test_movedim(self):
        op = torch.movedim
        test = self._vmap_view_test
        B0, B1, B2 = 7, 11, 13

        # movedim(tensor, int, int) variant
        test(op, (torch.rand(B0, 2, 5), 0, 1), in_dims=(0, None, None))
        test(op, (torch.rand(2, B0, 5), 0, 1), in_dims=(1, None, None))
        test(vmap(op, in_dims=(0, None, None)), (torch.rand(B1, 2, B0, 5), 0, 1), in_dims=(2, None, None))
        test(vmap(vmap(op, in_dims=(2, None, None)), in_dims=(0, None, None)),
             (torch.rand(B1, 2, B0, 5, B2), 0, 1), in_dims=(2, None, None))

        # movedim(tensor, intlist, intlist) variant
        test(op, (torch.rand(B0, 2, 3, 5), [1, 0], [0, 2]), in_dims=(0, None, None))
        test(op, (torch.rand(2, 3, B0, 5), [1, 0], [0, 2]), in_dims=(1, None, None))
        test(vmap(op, in_dims=(0, None, None)),
             (torch.rand(B1, 2, B0, 5), [0, 1], [1, 0]), in_dims=(2, None, None))
        test(vmap(vmap(op, in_dims=(2, None, None)), in_dims=(0, None, None)),
             (torch.rand(B1, 2, B0, 5, B2), [0, 1], [1, 0]), in_dims=(2, None, None))

    def test_narrow(self):
        op = torch.narrow
        test = self._vmap_view_test
        B0, B1, B2 = 7, 11, 13

        test(op, (torch.rand(B0, 2, 5), -1, 1, 3), in_dims=(0, None, None, None))
        test(op, (torch.rand(2, B0, 5), 1, 1, 3), in_dims=(1, None, None, None))
        test(vmap(op, in_dims=(0, None, None, None)),
             (torch.rand(B1, 2, B0, 5), 1, 0, 0), in_dims=(2, None, None, None))
        test(vmap(vmap(op, in_dims=(2, None, None, None)), in_dims=(0, None, None, None)),
             (torch.rand(B1, 2, B0, 5, B2), -1, 2, 3), in_dims=(2, None, None, None))

    def test_select(self):
        op = torch.select
        test = self._vmap_view_test
        B0, B1, B2 = 7, 11, 13
        test(op, (torch.rand(B0, 2, 5), 0, 0), in_dims=(0, None, None))
        test(op, (torch.rand(2, B0, 5), 1, 1), in_dims=(1, None, None))
        test(vmap(lambda t: op(t, 1, 1)), (torch.rand(B1, 2, B0, 5),), in_dims=2)
        test(vmap(vmap(lambda t: op(t, 1, 1), in_dims=1)), (torch.rand(B1, 2, B0, B2, 5),), in_dims=2)

    def test_slice(self):
        test = self._vmap_view_test
        B0, B1, B2 = 7, 11, 13
        test(lambda t: t[0:1], (torch.rand(B0, 3, 5),))
        test(lambda t: t[:, 1:3], (torch.rand(3, 5, B0),), in_dims=2)
        test(vmap(lambda t: t[:, 0:1], in_dims=2), (torch.rand(3, 5, B0, B1),), in_dims=2)
        test(vmap(vmap(lambda t: t[0:1], in_dims=2), in_dims=2),
             (torch.rand(3, 5, B0, B1, B2),), in_dims=2)

    def test_reshape(self):
        test = self._vmap_test
        B0, B1, B2 = 7, 11, 13
        op = torch.reshape
        test(op, (torch.rand(B0, 2 * 5), [2, 5]), in_dims=(0, None), check_view=True)
        test(op, (torch.rand(2, B0, 5), [1, 1, 10]), in_dims=(1, None), check_view=False)
        test(vmap(lambda t: t.reshape([-1])), (torch.rand(B0, B1, 2, 5),), check_view=True)
        test(vmap(vmap(lambda t: t.reshape([-1]), in_dims=2), in_dims=1),
             (torch.rand(3, B1, 2, B2, 5, B0),), in_dims=5, check_view=False)

    def test_reshape_as(self):
        test = self._vmap_test
        B0, B1, B2 = 7, 11, 13
        op = torch.Tensor.reshape_as
        test(op, (torch.rand(B0, 2 * 5), torch.rand(B0, 2, 5)), check_view=True)
        test(op, (torch.rand(2 * 5), torch.rand(B0, 2, 5)), in_dims=(None, 0), check_view=True)
        test(op, (torch.rand(B0, 2 * 5), torch.rand(2, 5)), in_dims=(0, None), check_view=True)

        test(op, (torch.rand(2, B0, 5), torch.rand(1, 1, 10)), in_dims=(1, None), check_view=False)

        test(vmap(op), (torch.rand(B0, B1, 2, 5), torch.randn(B0, B1, 10)), check_view=True)
        test(vmap(vmap(op, in_dims=(2, None)), in_dims=(1, None)),
             (torch.rand(3, B1, 2, B2, 5, B0), torch.rand(B0, 3 * 2 * 5)),
             in_dims=(5, 0), check_view=False)

    def test_result_type(self):
        def scalar_tensor_with_dtype(op):
            def wrapped(*args, **kwargs):
                dtype = op(*args, **kwargs)
                return torch.ones([], dtype=dtype)
            return wrapped

        test = self._vmap_test
        op = scalar_tensor_with_dtype(torch.result_type)

        B0 = 2

        test(op, (torch.randn(B0), torch.randn(B0, dtype=torch.float64)),
             check_propagates_grad=False)
        test(op, (torch.randn(B0), torch.randint(10, [B0], dtype=torch.int64)),
             check_propagates_grad=False)

        test(lambda x: op(x, 1), (torch.randn(B0),), check_propagates_grad=False)
        test(lambda x: op(x, 1.6), (torch.randn(B0),), check_propagates_grad=False)

        test(lambda x: op(x, torch.tensor(1)), (torch.randn(B0),),
             check_propagates_grad=False)
        test(lambda x: op(x, torch.tensor(1.6, dtype=torch.double)),
             (torch.randn(B0),), check_propagates_grad=False)

        test(op, (torch.randn(B0, 2), torch.randn(B0, 2, dtype=torch.float64)),
             check_propagates_grad=False)
        test(op, (torch.randn(B0, 2), torch.randint(10, [B0, 2], dtype=torch.int64)),
             check_propagates_grad=False)

        test(lambda x: op(x, 1), (torch.randn(B0, 2),), check_propagates_grad=False)
        test(lambda x: op(x, 1.6), (torch.randn(B0, 2),), check_propagates_grad=False)

        test(lambda x: op(x, torch.tensor(1)), (torch.randn(B0, 2),),
             check_propagates_grad=False)
        test(lambda x: op(x, torch.tensor(1.6, dtype=torch.double)),
             (torch.randn(B0, 2),), check_propagates_grad=False)

        test(op, (torch.randn(B0, 2), torch.randn(B0, dtype=torch.float64)),
             check_propagates_grad=False)
        test(op, (torch.randn(B0, 2), torch.randint(10, [B0], dtype=torch.int64)),
             check_propagates_grad=False)

    def test_split(self):
        test = self._vmap_view_test
        op = torch.split
        B0, B1, B2 = 7, 11, 13

        # tests for torch.split(self, split_size: int, dim)
        test(op, (torch.rand(B0, 2, 1024), 101, -1), in_dims=(0, None, None))
        test(op, (torch.rand(2, B0, 1024), 130, 1), in_dims=(1, None, None))
        test(vmap(op, in_dims=(0, None, None)), (torch.rand(B1, 1023, B0, 5), 256, 0),
             in_dims=(2, None, None))
        test(vmap(vmap(lambda t: op(t, 4, 1), in_dims=2)),
             (torch.rand(B1, 2, B0, 64, B2),), in_dims=2)

        # tests for torch.split(self, split_size: List[int], dim)
        test(op, (torch.rand(B0, 2, 1024), [1, 1020, 3], -1), in_dims=(0, None, None))
        test(op, (torch.rand(2, B0, 1024), [100] * 10 + [24], 1), in_dims=(1, None, None))
        test(vmap(op, in_dims=(0, None, None)), (torch.rand(B1, 1023, B0, 5), [256] * 3 + [255], 0),
             in_dims=(2, None, None))
        test(vmap(vmap(lambda t: op(t, [4] * 8 + [8] * 4, 1), in_dims=2)),
             (torch.rand(B1, 2, B0, 64, B2),), in_dims=2)

    def test_t(self):
        op = torch.t
        test = self._vmap_view_test
        B0, B1, B2 = 7, 11, 13
        test(op, (torch.rand(B0, 2, 5),))
        test(op, (torch.rand(2, B0, 5),), in_dims=1)
        test(vmap(op), (torch.rand(B1, 2, B0, 5),), in_dims=2)
        test(vmap(vmap(op, in_dims=2)), (torch.rand(B1, 2, B0, 5, B2),), in_dims=2)

    def test_T_numpy(self):
        def op(t):
            return t.T

        test = self._vmap_view_test
        B0, B1, B2 = 7, 11, 13
        test(op, (torch.rand(B0, 2, 3, 5),))
        test(op, (torch.rand(B0),))
        test(op, (torch.rand(2, B0, 3, 5),), in_dims=1)
        test(vmap(op), (torch.rand(B1, 2, B0, 5),), in_dims=2)
        test(vmap(op), (torch.rand(B1, 2, B0, 3, 5),), in_dims=2)
        test(vmap(vmap(op, in_dims=2)), (torch.rand(B1, 2, B0, 3, B2, 5),), in_dims=2)

    def test_to(self):
        test = self._vmap_test
        B0, B1 = 7, 11

        test(lambda t: t.to('cpu'), (torch.rand(B0),))
        test(lambda t: t.to(torch.double), (torch.rand(B0),))
        test(lambda t, o: t.to(o), (torch.rand(B0), torch.randn(B0, dtype=torch.float64)))
        test(lambda t, o: t.to(o),
             (torch.rand(B0), torch.randn(B0, dtype=torch.float64)),
             in_dims=(0, None))
        test(vmap(lambda t: t.to(torch.double)), (torch.rand(B0, B1, 3),))

        # also test some casting methods
        test(lambda t: t.double(), (torch.rand(B0),))
        test(lambda t: t.float(), (torch.rand(B0),))
        test(lambda t: t.int(), (torch.rand(B0),), check_propagates_grad=False)
        test(lambda t: t.long(), (torch.rand(B0),), check_propagates_grad=False)

    def test_unfold(self):
        op = torch.Tensor.unfold
        test = self._vmap_view_test
        B0, B1, B2 = 3, 2, 5

        test(op, (torch.rand(B0, 7, 11), 0, 2, 1), in_dims=(0, None, None, None))
        test(op, (torch.rand(7, B0, 11), 1, 4, 2), in_dims=(1, None, None, None))
        test(vmap(op, in_dims=(0, None, None, None)),
             (torch.rand(B1, 7, B0, 11), 1, 5, 1), in_dims=(2, None, None, None))
        test(vmap(vmap(op, in_dims=(2, None, None, None)), in_dims=(0, None, None, None)),
             (torch.rand(B1, 7, B0, 11, B2), -1, 2, 4), in_dims=(2, None, None, None))

    def test_unbind(self):
        test = self._vmap_view_test
        op = torch.unbind
        B0, B1, B2 = 7, 11, 13

        test(op, (torch.rand(B0, 2, 1024), -1), in_dims=(0, None))
        test(op, (torch.rand(B0, 2, 0),))
        test(op, (torch.rand(2, B0, 7), 0), in_dims=(1, None))
        test(vmap(op, in_dims=(0, None)), (torch.rand(B1, 1023, B0, 5), 1),
             in_dims=(2, None))
        test(vmap(vmap(lambda t: op(t, dim=1), in_dims=2)),
             (torch.rand(B1, 2, B0, 32, B2),), in_dims=2)

    def test_view(self):
        test = self._vmap_view_test
        B0, B1, B2 = 7, 11, 13
        op = torch.Tensor.view

        # We should error out if the view would produce an incorrect result
        with self.assertRaises(RuntimeError):
            vmap(op, in_dims=(1, None))(torch.rand(2, B0, 5), [10])

        test(op, (torch.rand(B0, 2 * 5), [2, 5]), in_dims=(0, None))
        test(op, (torch.rand(B0, 4, 5), [1, 2, 1, 10]), in_dims=(0, None))
        test(vmap(lambda t: t.view([-1])), (torch.rand(B0, B1, 2, 5, 3),))
        test(vmap(vmap(lambda t: t.reshape([-1])), in_dims=1),
             (torch.rand(B2, B0, B1, 3, 2, 5),), in_dims=1)

    def test_view_as(self):
        test = self._vmap_view_test
        B0, B1, B2 = 7, 11, 13
        op = torch.Tensor.view_as

        # We should error out if the view would produce an incorrect result
        with self.assertRaises(RuntimeError):
            vmap(op, in_dims=(1, 0))(torch.rand(2, B0, 5), torch.rand(B0, 10))

        test(op, (torch.rand(B0, 2 * 5), torch.rand(B0, 2, 5)))
        test(op, (torch.rand(2 * 5), torch.rand(B0, 2, 5)), in_dims=(None, 0))
        test(op, (torch.rand(B0, 2 * 5), torch.rand(2, 5)), in_dims=(0, None))

        test(op, (torch.rand(B0, 4, 5), torch.rand(2, 1, 1, 10)), in_dims=(0, None))

        test(vmap(op), (torch.rand(B0, B1, 2, 5), torch.randn(B0, B1, 10)))
        test(vmap(vmap(op, in_dims=(0, None)), in_dims=(0, None)),
             (torch.rand(B1, B2, B0, 3, 2, 5), torch.rand(B0, 3 * 2 * 5)),
             in_dims=(2, 0))

    def test_no_random_op_support(self):
        B0 = 2

        captured = torch.rand(3)

        random_ops = [
            # out-of-place on BatchedTensor
            (torch.bernoulli, (torch.rand(B0, 1),)),
            (lambda t: torch.bernoulli(t, p=0.5), (torch.rand(B0, 1),)),
            (lambda t: torch.multinomial(t, 2), (torch.rand(B0, 3),)),
            (torch.normal, (torch.randn(B0, 1), torch.randn(B0, 1))),
            (lambda t: torch.normal(t, 1.), (torch.randn(B0, 1),)),
            (lambda t: torch.normal(0., t), (torch.randn(B0, 1),)),
            (torch.poisson, (torch.rand(B0, 1),)),
            (torch.rand_like, (torch.rand(B0, 1),)),
            (torch.randn_like, (torch.rand(B0, 1),)),
            (lambda t: torch.randint_like(t, 2), (torch.rand(B0, 1),)),
            (lambda t: torch.randint_like(t, 0, 2), (torch.rand(B0, 1),)),

            # out-of-place on captured tensor
            (lambda t: torch.bernoulli(captured), (torch.rand(B0),)),
            (lambda t: torch.bernoulli(captured, p=0.5), (torch.rand(B0),)),
            (lambda t: torch.multinomial(captured, 2), (torch.rand(B0),)),
            (lambda t: torch.normal(captured, captured), (torch.randn(B0),)),
            (lambda t: torch.normal(captured, 1.), (torch.randn(B0),)),
            (lambda t: torch.normal(0., captured), (torch.randn(B0),)),
            (lambda t: torch.poisson(captured), (torch.rand(B0),)),
            (lambda t: torch.rand_like(captured), (torch.rand(B0),)),
            (lambda t: torch.randn_like(captured) , (torch.rand(B0),)),
            (lambda t: torch.randint_like(captured, 2), (torch.rand(B0),)),
            (lambda t: torch.randint_like(captured, 0, 2), (torch.rand(B0),)),

            # in-place on BatchedTensor
            (lambda t: t.bernoulli_(), (torch.randn(B0, 1),)),
            (lambda t: t.cauchy_(), (torch.randn(B0, 1),)),
            (lambda t: t.exponential_(), (torch.randn(B0, 1),)),
            (lambda t: t.geometric_(0.5), (torch.randn(B0, 1),)),
            (lambda t: t.log_normal_(), (torch.randn(B0, 1),)),
            (lambda t: t.normal_(), (torch.randn(B0, 1),)),
            (lambda t: t.random_(), (torch.randn(B0, 1),)),
            (lambda t: t.random_(0, 2), (torch.randn(B0, 1),)),
            (lambda t: t.random_(2), (torch.randn(B0, 1),)),
            (lambda t: t.uniform_(), (torch.randn(B0, 1),)),

            # in-place on captured tensor
            (lambda t: captured.bernoulli_(), (torch.randn(B0),)),
            (lambda t: captured.cauchy_(), (torch.randn(B0),)),
            (lambda t: captured.exponential_(), (torch.randn(B0),)),
            (lambda t: captured.geometric_(0.5), (torch.randn(B0),)),
            (lambda t: captured.log_normal_(), (torch.randn(B0),)),
            (lambda t: captured.normal_(), (torch.randn(B0),)),
            (lambda t: captured.random_(), (torch.randn(B0),)),
            (lambda t: captured.random_(0, 2), (torch.randn(B0),)),
            (lambda t: captured.random_(2), (torch.randn(B0),)),
            (lambda t: captured.uniform_(), (torch.randn(B0),)),

            # factory functions
            (lambda t: torch.rand(1), (torch.randn(B0),)),
            (lambda t: torch.randn(1), (torch.randn(B0),)),
            (lambda t: torch.randint(5, [1]), (torch.randn(B0),)),
            (lambda t: torch.randperm(5), (torch.randn(B0),)),
        ]
        for op, args in random_ops:
            with self.assertRaisesRegex(RuntimeError,
                                        'vmap: We do not yet support calling random operations'):
                vmap(op)(*args)

def construct_v(output, batch_size):
    return torch.randn(batch_size, *output.shape,
                       dtype=output.dtype, device=output.device)

def as_tuple(x):
    if isinstance(x, tuple):
        return x
    elif isinstance(x, list):
        return tuple(x)
    else:
        return x,

def differentiable(args):
    return tuple(arg for arg in as_tuple(args)
                 if isinstance(arg, torch.Tensor) and arg.requires_grad)

class TestVmapBatchedGradient(TestCase):
    def _vmap_test(self, *args, **kwargs):
        return _vmap_test(self, *args, **kwargs)

    # Tests batched gradient computation of outputs = op(*args, **kwargs)
    # by comparing it to a sequential map+stack fallback.
    #
    # output_process_fn: a function that maps the outputs to the part
    #       that should be differentiated.
    # batch_size: the batch dim size for the batched grad
    def _batched_grad_test(self, op, args, kwargs, output_process_fn=lambda x: x, batch_size=3):
        outputs = op(*args, **kwargs)
        outputs = differentiable(output_process_fn(outputs))
        batched_vectors = tuple(construct_v(out, batch_size) for out in outputs)

        def vector_jacobian_product(*vectors):
            return torch.autograd.grad(outputs, differentiable(args), vectors,
                                       retain_graph=True)
        self._vmap_test(vector_jacobian_product, batched_vectors,
                        check_propagates_grad=False)

    # Tests batched second grad computation of outputs = op(*args, **kwargs).
    # by comparing it to a sequential map+stack fallback.
    #
    # output_process_fn: a function that maps the outputs to the part
    #       that should be differentiated.
    # batch_size: the batch dim size for the batched grad
    #
    # NB: we only test computing batched gradients in the second gradient
    # computation. One specific use case that does this is computing the hessian
    # matrix of a scalar-valued function; this is useful in Bayesian Logistic
    # Regression.
    # It might be useful to have a test that computes batched first gradients and
    # then uses those to compute batched second gradients in the future.
    def _batched_grad_grad_test(self, op, args, kwargs, output_process_fn=lambda x: x, batch_size=3):
        outputs = op(*args, **kwargs)
        outputs = differentiable(output_process_fn(outputs))
        ones = tuple(torch.ones_like(out) for out in outputs)
        # Same thing as summing together all of the outputs and calling .backward()
        first_grads = torch.autograd.grad(outputs, differentiable(args), ones,
                                          create_graph=True)
        first_grads = differentiable(first_grads)
        self.assertNotEqual(
            len(first_grads), 0, "None of the first grads depend on the input!")

        batched_vectors = tuple(construct_v(grad, batch_size) for grad in first_grads)

        def vector_hessian_product(*vectors):
            outputs = torch.autograd.grad(first_grads, differentiable(args), vectors,
                                          retain_graph=True, allow_unused=True)
            outputs = tuple(out for out in outputs if out is not None)
            assert len(outputs) > 0
            return outputs

        self._vmap_test(vector_hessian_product, batched_vectors,
                        check_propagates_grad=False)

    def test_sigmoid(self, device):
        # Maybe we can make the "check that the slow fallback was not invoked"
        # into a context manager, because it's used a lot. I'll leave that for
        # future work.
        regex = r'falling back to slow \(for loop and stack\) implementation'
        with warnings.catch_warnings(record=True) as wa:
            warnings.simplefilter('always')
            x = torch.randn(2, 3, requires_grad=True, device=device)
            self._batched_grad_test(Tensor.sigmoid, (x,), {})
            self._batched_grad_grad_test(Tensor.sigmoid, (x,), {})

            for captured_warning in wa:
                self.assertNotRegex(str(captured_warning.message), regex)

instantiate_device_type_tests(
    TestVmapBatchedGradient,
    globals(),
    # Excluding ROCM
    except_for='cuda' if TEST_WITH_ROCM else None,
    only_for=['cuda', 'cpu'],
)

if __name__ == '__main__':
    run_tests()
