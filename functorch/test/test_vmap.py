# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.testing._internal.common_utils import TestCase, run_tests
import torch
import torch.nn.functional as F
import torch.utils._pytree as pytree
from torch import Tensor
import functools
import itertools
import textwrap
import warnings
import unittest
import re
from torch.testing._internal.common_device_type import instantiate_device_type_tests, \
    skipCUDAIfNoMagma
from torch.testing._internal.common_device_type import ops, onlyCPU
from functorch_lagging_op_db import functorch_lagging_op_db
from functorch_additional_op_db import additional_op_db
from torch.utils._pytree import tree_map
from common_utils import (
    parameterized,
    parameterized_with_device,
    instantiate_parameterized_methods,
    get_fallback_and_vmap_exhaustive,
    opinfo_in_dict,
    xfail,
    skipOps,
    check_vmap_fallback,
)
import types

import functorch
from functorch import vmap, functional_init_with_buffers
from functorch._C import reshape_dim_into, reshape_dim_outof

FALLBACK_REGEX = 'There is a performance drop'


class EnableVmapFallbackWarnings:
    def __enter__(self):
        self.prev_state = torch._C._debug_only_are_vmap_fallback_warnings_enabled()
        torch._C._debug_only_display_vmap_fallback_warnings(True)

    def __exit__(self, *ignored):
        torch._C._debug_only_display_vmap_fallback_warnings(self.prev_state)


class TestVmapAPI(TestCase):
    def test_non_tensor_output_raises(self):
        with self.assertRaisesRegex(ValueError, "got type <class 'float'> as a return"):
            output = vmap(lambda x: 3.14)(torch.ones(3))

        def multiple_outputs(x):
            return x, 3

        with self.assertRaisesRegex(ValueError, "got type <class 'int'> as a return"):
            vmap(multiple_outputs)(torch.ones(3))

    def test_different_map_dim_size_raises(self):
        x = torch.randn(2)
        y = torch.randn(3)
        expected_msg = 'Expected all tensors to have the same size in the mapped dimension'
        with self.assertRaisesRegex(ValueError, expected_msg):
            vmap(torch.mul)(x, y)
        with self.assertRaisesRegex(ValueError, expected_msg):
            vmap(lambda z: z[0] + z[1], in_dims=((0, 0),))((x, y))
        with self.assertRaisesRegex(ValueError, expected_msg):
            vmap(lambda z: z['x'] + z['y'], in_dims=({'x': 0, 'y': 0},))({'x': x, 'y': y})

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

    def test_multiple_outputs2(self):
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
        vmap(returns_list_of_two_tensors)(x)
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
        msg = (
            r"Batching rule not implemented for aten::.+; the "
            r"fallback path doesn't work on out= or view ops"
        )
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(torch.ravel)(tensor)

        def out_op(x, y):
            return torch.abs(x, out=y)

        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(out_op)(tensor, tensor)

        tensor = torch.randn(2)
        # The fallback doesn't support TensorList
        with self.assertRaisesRegex(RuntimeError, 'Batching rule not implemented'):
            vmap(lambda t: torch.atleast_1d([t]))(tensor)

        # Don't support non-tensor returns. This is a limitation of vmap;
        # functions that don't return tensors must be special cased
        with self.assertRaisesRegex(RuntimeError, 'Batching rule not implemented'):
            vmap(torch.Tensor.item)(tensor)

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

    def test_pytree_returns(self):
        x = torch.randn(2, 3)

        def f(x):
            y = x.sin()
            return y, (y, y), [y, (y, y)]

        y0, (y1, y2), (y3, (y4, y5)) = vmap(f)(x)
        self.assertEqual(y0, x.sin())
        self.assertEqual(y0, y1)
        self.assertEqual(y2, y1)
        self.assertEqual(y2, y3)
        self.assertEqual(y4, y3)
        self.assertEqual(y5, y4)

    def test_pytree_returns_outdims(self):
        x = torch.randn(2, 3)

        def f(x):
            y = x.sin()
            return y, (y, y)

        y0, (y1, y2) = vmap(f, out_dims=(0, (0, 1)))(x)
        self.assertEqual(y0, x.sin())
        self.assertEqual(y1, x.sin())
        self.assertEqual(y2, x.sin().t())

    def test_pytree_returns_broadcast_simple(self):
        x = torch.randn(2, 3)

        def f(x):
            y = x.sin()
            return y, (y, y)

        y0, (y1, y2) = vmap(f, out_dims=1)(x)
        self.assertEqual(y0, x.sin().t())
        self.assertEqual(y1, y0)
        self.assertEqual(y2, y0)

    def test_pytree_returns_broadcast_nested(self):
        x = torch.randn(2, 3)

        def f(x):
            y = x.sin()
            return y, (y, y)

        y0, (y1, y2) = vmap(f, out_dims=(0, 1))(x)
        self.assertEqual(y0, x.sin())
        self.assertEqual(y1, y0.t())
        self.assertEqual(y2, y0.t())

    def test_out_dims_must_be_int_or_collection_of_int_err_msg(self):
        msg = 'must be an int or a python collection of ints'
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
        msg = 'not compatible'
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

    def test_accepts_nested_inputs(self):
        B0 = 2
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)

        # Single layer of nesting
        out = vmap(lambda z: z[0] + z[1])((x, y))
        self.assertEqual(out, x + y)
        out = vmap(lambda z: z[0] + z[1], in_dims=(0,))((x, y))
        self.assertEqual(out, x + y)
        out = vmap(lambda z: z[0] + z[1], in_dims=((0, 0),))((x, y))
        self.assertEqual(out, x + y)

        out = vmap(lambda z: z[0] + z[1])([x, y])
        self.assertEqual(out, x + y)
        out = vmap(lambda z: z[0] + z[1], in_dims=(0,))([x, y])
        self.assertEqual(out, x + y)
        out = vmap(lambda z: z[0] + z[1], in_dims=([0, 0],))([x, y])
        self.assertEqual(out, x + y)

        out = vmap(lambda z: z['x'] + z['y'])({'x': x, 'y': y})
        self.assertEqual(out, x + y)
        out = vmap(lambda z: z['x'] + z['y'], in_dims=(0,))({'x': x, 'y': y})
        self.assertEqual(out, x + y)
        out = vmap(lambda z: z['x'] + z['y'], in_dims=({'x': 0, 'y': 0},))({'x': x, 'y': y})
        self.assertEqual(out, x + y)

        # Multiple layers of nesting
        out_fn = vmap(lambda z: z['x'][0] + z['x'][1][0] + z['y'][0] + z['y'][1])
        out = out_fn({'x': [x, (x,)], 'y': [y, y]})
        self.assertEqual(out, x + x + y + y)

    def test_in_dims_wrong_type_err_msg(self):
        x = torch.randn(3)
        y = torch.randn(3)
        msg = r'expected `in_dims` to be int or a \(potentially nested\) tuple'
        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.mul, [0, 0])(x, y)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.mul, set({0, 0}))(x, y)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.mul, 'lol')(x, y)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda z: z[0] + z[1], in_dims=[0, 0])([x, y])
        # The following should not throw
        vmap(torch.mul, (0, 0))(x, y)

    def test_not_enough_in_dims_err_msg(self):
        x = torch.randn(3)
        y = torch.randn(3)
        msg = r'in_dims is not compatible with the structure of `inputs`'

        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.mul, (0,))(x, y)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.mul, (0, 0, 0))(x, y)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda z: z[0] + z[1], in_dims=([0],))([x, y])
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda z: z[0] + z[1], in_dims=((0, 0),))([x, y])
        # The following should not throw
        vmap(torch.mul, (0, 0))(x, y)

    def test_integer_in_dim_but_not_tensor_input_err_msg(self):
        def foo(xy):
            return xy[0] * xy[1]

        def bar(x, yz):
            return x * yz[0] * yz[1]

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)

        # the following are errors in jax (and will always be errors)
        msg = 'Got in_dim=0 for an input but the input is of type'
        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.sum)(x, 0)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.sum, (0, 0))(x, 0)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda z: z[0] + z[1], in_dims=([0, 0],))([x, 1])
        # The following should not throw
        vmap(torch.sum, (0, None))(x, 0)

    def test_in_dim_not_in_tensor_err_msg(self):
        def foo(x):
            return x * x

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)

        msg = r'Got in_dim=-?\w for some input, but that input is a Tensor of dimensionality \w'
        with self.assertRaisesRegex(ValueError, msg):
            vmap(foo)(torch.randn([]))
        with self.assertRaisesRegex(ValueError, msg):
            vmap(foo, in_dims=(0,))(torch.randn([]))
        with self.assertRaisesRegex(ValueError, msg):
            vmap(foo, in_dims=(-1,))(x)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(foo, in_dims=(2,))(y)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda z: z[0] + z[1], in_dims=([3, 0],))([x, y])
        # the following should not throw
        vmap(foo, in_dims=(0,))(torch.randn(2, 3))
        vmap(foo, in_dims=(1,))(torch.randn(2, 3))

    def test_fallback_does_not_warn_by_default(self):
        # NB: One day we will implement a batching rule for torch.atan2.
        # If/when we do, this test should be replaced to test the fallback
        # path on another operator to avoid bitrot.
        op = torch.copysign
        x = torch.randn(11)
        y = torch.randn(11)
        with warnings.catch_warnings(record=True) as wa:
            result = vmap(op)(x, y)
            # The single warning here is the "vmap is experimental"
            # warning, not a warning from the vmap fallback path.
            self.assertEqual(len(wa), 1)

    @unittest.expectedFailure
    def test_fallback_warns_when_warnings_are_enabled(self):
        # NB: One day we will implement a batching rule for torch.atan2.
        # If/when we do, this test should be replaced to test the fallback
        # path on another operator to avoid bitrot.
        op = torch.copysign
        x = torch.randn(11)
        y = torch.randn(11)
        with warnings.catch_warnings(record=True) as wa:
            with EnableVmapFallbackWarnings():
                result = vmap(op)(x, y)
            self.assertEqual(len(wa), 2)
            self.assertRegex(str(wa[-1].message), FALLBACK_REGEX)

    def _assert_uses_vmap_fallback(self, vmap_args, inputs):
        return
        with warnings.catch_warnings(record=True) as wa:
            with EnableVmapFallbackWarnings():
                result = vmap(*vmap_args)(*inputs)
            self.assertEqual(len(wa), 2)
            self.assertRegex(str(wa[-1].message), FALLBACK_REGEX)

    def test_fallback_zero_dim(self):
        # NB: One day we will implement a batching rule for torch.atan2.
        # If/when we do, this test should be replaced to test the fallback
        # path on another operator to avoid bitrot.
        op = torch.copysign
        x = torch.randn(11)
        y = torch.randn(11)
        self._assert_uses_vmap_fallback((op,), (x, y))

        B0, B1 = 0, 3
        x = torch.randn(B0, 11)
        y = torch.randn(11)

        msg = 'The fallback path does not support vmap over dims of size 0'

        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, (0, None))(x, y)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, (None, 0))(y, x)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op)(x, x)

        x = torch.randn(B0, B1, 11)
        y = torch.randn(B1, 11)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, (0, None))(x, y)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, (None, 0))(y, x)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op)(x, x)

    def test_fallback_atan2(self):
        # NB: One day we will implement a batching rule for torch.atan2.
        # If/when we do, this test should be replaced to test the fallback
        # path on another operator to avoid bitrot.
        op = torch.copysign

        x = torch.randn(5, 7, 11)
        y = torch.randn(5, 7, 11)

        self._assert_uses_vmap_fallback((op,), (x, y))

        # fallback on torch.atan2
        x = torch.randn(7, 11, 5)
        y = torch.randn(5, 7, 11)
        result = vmap(op, (2, 0))(x, y)
        self.assertEqual(result, op(x.permute(2, 0, 1), y))

        # fallback on torch.atan2, nested vmap
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

    def test_inplace_fallback_unary(self):
        # Test the in-place fallback on an in-place method that takes no
        # additional Tensor arguments. This is the simplest case of the fallback.
        # NB: One day we will implement a batching rule for acos_.
        # If/when we do, this test should be replaced to test the fallback
        # path on another operator to avoid bitrot.
        op = Tensor.acos_
        B0, B1, B2 = 2, 3, 10000

        x = torch.randn(B0, 5)
        self._assert_uses_vmap_fallback((op,), (x,))

        # Single vmap
        x_orig = torch.rand(B0, 5)
        x = x_orig.clone()
        result = vmap(op)(x)
        self.assertTrue(result is x)
        self.assertEqual(result, x_orig.acos())

        # Single vmap + different out_dim produces a view(!)
        x_orig = torch.rand(B0, 5)
        x = x_orig.clone()
        result = vmap(op, out_dims=(1,))(x)
        self.assertTrue(result._base is x)
        self.assertEqual(result, x_orig.t().acos())

        # Nested vmap
        x_orig = torch.randn(B0, B1, 5)
        x = x_orig.clone()
        result = vmap(vmap(op))(x)
        self.assertTrue(result is x)
        self.assertEqual(result, x_orig.acos())

        # Nested vmap, large batch size
        x_orig = torch.randn(B0, B1, B2, 5)
        x = x_orig.clone()
        result = vmap(vmap(vmap(op)))(x)
        self.assertTrue(result is x)
        self.assertEqual(result, x_orig.acos())

    def test_inplace_fallback_nary_same_levels(self):
        # NB: One day we will implement a batching rule for atan2_
        # If/when we do, this test should be replaced to test the fallback
        # path on another operator to avoid bitrot.
        op = Tensor.atan2_
        outplace_op = torch.atan2

        x = torch.randn(5, 7, 11)
        y = torch.randn(5, 7, 11)
        self._assert_uses_vmap_fallback((op,), (x, y))

        # Single vmap
        B0 = 5
        x_orig = torch.randn(7, 11, B0)
        x = x_orig.clone()
        y = torch.randn(B0, 7, 11)
        vmap(op, (2, 0))(x, y)
        self.assertEqual(x, outplace_op(x_orig, y.movedim(0, 2)))

        # Nested vmap
        B0, B1 = 5, 7
        x_orig = torch.randn(B1, 11, B0)
        x = x_orig.clone()
        y = torch.randn(B0, B1, 11)
        vmap(vmap(op), (2, 0))(x, y)
        self.assertEqual(x, outplace_op(x_orig, y.movedim([0, 1], [2, 0])))

        # big batch size (total 10000)
        B0, B1, B2 = 100, 10, 10
        x_orig = torch.randn(B0, B1, B2, 5)
        x = x_orig.clone()
        y = torch.randn(B0, B1, B2)
        result = vmap(vmap(vmap(op)))(x, y)
        self.assertEqual(x, outplace_op(x_orig, y.view(B0, B1, B2, 1)))

    # ("Fallback isInplaceVmapCompatible check is broken")
    @unittest.expectedFailure
    def test_inplace_fallback_nary_different_levels(self):
        # NB: One day we will implement a batching rule for atan2_
        # If/when we do, this test should be replaced to test the fallback
        # path on another operator to avoid bitrot.
        op = Tensor.atan2_
        outplace_op = torch.atan2
        B0, B1, B2 = 2, 3, 5

        x = torch.rand(B0, 7)
        y = torch.rand(7)
        self._assert_uses_vmap_fallback((op, (0, None)), (x, y))

        # op(left, right): All of the levels in right are found in left
        x_orig = torch.rand(B0, 7)
        x = x_orig.clone()
        y = torch.rand(7)
        vmap(op, in_dims=(0, None))(x, y)
        self.assertEqual(x, outplace_op(x_orig, y))

        x_orig = torch.rand(B0, B1, 7)
        x = x_orig.clone()
        y = torch.rand(B0, 7)
        vmap(vmap(op, in_dims=(0, None)))(x, y)
        self.assertEqual(x, outplace_op(x_orig, y.view(B0, 1, 7)))

        # op(left, right): Some of the levels in right are not found in left
        msg = r'vmap: aten::atan2_\(self, \*extra_args\) is not possible'
        x = torch.rand(7)
        y = torch.rand(B0, 7)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(None, 0))(x, y)

        x = torch.rand(B1, 7)
        y = torch.rand(B0, 7)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(vmap(op, in_dims=(0, None)), in_dims=(None, 0))(x, y)

        x = torch.rand(B1, 7)
        y = torch.rand(7, B0)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(vmap(op, in_dims=(0, None)), in_dims=(None, 1))(x, y)

        x = torch.rand(B0, 7)
        y = torch.rand(B0, B1, 7)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(vmap(op, in_dims=(None, 0)))(x, y)

    def test_backward_unsupported_interaction(self):
        x = torch.randn(3, requires_grad=True)
        y = torch.randn(5)
        grad = torch.randn_like(x)
        err_msg = r'backward\(\) called inside a functorch transform'

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
            return y

        with self.assertRaisesRegex(RuntimeError, err_msg):
            vmap(completely_unrelated_backward)(y)

    @unittest.expectedFailure
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

    def test_fallback_with_undefined_grad(self):
        B0 = 7
        x = torch.randn(2, 3, 4, 5, requires_grad=True)
        weight = torch.randn(3, 3, 1, 1)
        v = torch.randn(B0, 2, 3, 4, 5)

        def get_vjp(v):
            result = torch.nn.functional.conv2d(x, weight)
            grad_x, = torch.autograd.grad(result, x, v)
            return grad_x

        # Runs vmap(get_vjp)(v), which should not error out.
        # The backward formula for convolution returns an undefined
        # Tensor for grad_bias because the original bias does not exist.
        #
        # In the future we'll probably add a batching rule for convolution
        # backward. When this happens, we should modify this test to use a
        # different op (and/or create and use a dummy operator) to avoid bitrot.
        self._assert_uses_vmap_fallback([get_vjp], [v])

    def test_reshape_dim_into(self):
        x = torch.randn(2, 3, 5, 7)

        y = reshape_dim_into(0, 0, x)
        self.assertEqual(y, x.reshape(6, 5, 7))

        y = reshape_dim_into(0, 1, x)
        self.assertEqual(y, x.movedim(0, 1).reshape(3, 2 * 5, 7))

        y = reshape_dim_into(0, 2, x)
        self.assertEqual(y, x.movedim(0, 2).reshape(3, 5, 2 * 7))

        y = reshape_dim_into(1, 2, x)
        self.assertEqual(y, x.movedim(1, 2).reshape(2, 5, 3 * 7))

        y = reshape_dim_into(0, -2, x)
        self.assertEqual(y, x.movedim(0, 1).reshape(3, 2 * 5, 7))

        y = reshape_dim_into(0, -1, x)
        self.assertEqual(y, x.movedim(0, 2).reshape(3, 5, 2 * 7))

        y = reshape_dim_into(-4, -1, x)
        self.assertEqual(y, x.movedim(0, 2).reshape(3, 5, 2 * 7))

    def test_reshape_dim_outof(self):
        x = torch.randn(12, 12, 12).permute(2, 1, 0)

        y = reshape_dim_outof(0, 2, x)
        self.assertEqual(y, x.reshape(2, 6, 12, 12))

        y = reshape_dim_outof(1, 4, x)
        self.assertEqual(y, x.reshape(12, 4, 3, 12))

        y = reshape_dim_outof(2, 6, x)
        self.assertEqual(y, x.reshape(12, 12, 6, 2))

        y = reshape_dim_outof(-1, 6, x)
        self.assertEqual(y, x.reshape(12, 12, 6, 2))

    def test_batch_rule_does_not_need_to_handle_no_batched_input(self):
        def f(x, y):
            res = torch.dot(y, torch.ones(2))
            return x + res

        x = torch.randn(7, 5)
        y = torch.randn(3, 2)
        out = vmap(vmap(f, in_dims=(0, None)), in_dims=(None, 0))(x, y)
        expected = torch.mv(y, torch.ones(2)).view(3, 1, 1) + x
        self.assertEqual(out, expected)

    def test_tensor_print(self):
        x = torch.tensor([[3.14]])
        buf = None

        def foo(x):
            nonlocal buf
            buf = repr(x)
            return x

        vmap(vmap(foo))(x)
        expected = textwrap.dedent("""\
            BatchedTensor(lvl=3, bdim=0, value=\\
              BatchedTensor(lvl=2, bdim=0, value=\\
                tensor([[3.1400]])))""")
        self.assertEqual(buf, expected)


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

def should_allow_vmap_fallback_usage(fn):
    return getattr(fn, '_allow_vmap_fallback_usage', False)

def allowVmapFallbackUsage(fn):
    fn._allow_vmap_fallback_usage = True
    return fn

# All tests of TestVmapBase check that the slow vmap fallback is never invoked.
# This is so that we can incrementally add batching rules for operators to
# replace the slow vmap fallback path for said operators. To skip this check,
# please use the allowVmapFallbackUsage decorator.
#
# NB: Don't add tests to TestVmapBase directly, unless you want them to run
# on every subclass of TestVmapBase. Add them to e.g. TestVmapOperators.
#
# NB: TestVmapBase is a nested class. This prevents test runners from picking
# it up and running it.
class Namespace:
    class TestVmapBase(TestCase):
        def __init__(self, method_name='runTest'):
            super().__init__(method_name)

            test_method = getattr(self, method_name, None)
            if test_method is None:
                return

            if not should_allow_vmap_fallback_usage(test_method):
                setattr(self, method_name,
                        self._wrap_method_with_vmap_fallback_check(test_method))

        def _wrap_method_with_vmap_fallback_check(self, method):
            msg = (
                'Expected the test to not invoke the vmap fallback path, i.e., '
                'all of the operators being tested in this test should have batching '
                'rules implemented. If you are intentionally testing something to '
                'do with the fallback path, use allowVmapFallbackUsage. Otherwise, '
                'please make sure that batching rules are implemented for the '
                'operator(s) being tested.'
            )

            @functools.wraps(method)
            def wrapper(self, *args, **kwargs):
                with warnings.catch_warnings(record=True) as wa:
                    warnings.simplefilter('always')
                    with EnableVmapFallbackWarnings():
                        method(*args, **kwargs)
                    # for captured_warning in wa:
                    #     self.assertNotRegex(str(captured_warning.message), FALLBACK_REGEX, msg)
            return types.MethodType(wrapper, self)

        @allowVmapFallbackUsage
        def test_vmap_fallback_check_ok(self):
            # One day we'll implement a batching rule for torch.var_mean.
            # When that happens, please change the example to use an
            # operator that doesn't have a batching rule implemented.
            op_using_fallback = torch.var_mean
            vmap(op_using_fallback)(torch.rand(3))

        @unittest.expectedFailure
        def test_vmap_fallback_check(self):
            @self._wrap_method_with_vmap_fallback_check
            def no_fallback(self):
                pass

            # One day we'll implement a batching rule for torch.var_mean.
            # When that happens, please change the example to use an
            # operator that doesn't have a batching rule implemented.
            op_using_fallback = torch.var_mean

            @self._wrap_method_with_vmap_fallback_check
            def uses_fallback(self):
                vmap(op_using_fallback)(torch.rand(3))

            no_fallback(self)

            with self.assertRaises(AssertionError):
                uses_fallback(self)

def _make_case(op, input_getter=TensorFactory.randn):
    return (op, input_getter)


class TestVmapOperators(Namespace.TestVmapBase):
    def _vmap_test(self, *args, **kwargs):
        return _vmap_test(self, *args, **kwargs)

    def _vmap_view_test(self, *args, **kwargs):
        self._vmap_test(*args, **kwargs, check_view=True)

    def _test_unary(self, op, getter, device, *args, **kwargs):
        test = functools.partial(self._vmap_test, *args, **kwargs)
        B0, B1 = 7, 11

        # Single vmap, various in_dims / out_dims
        test(op, [getter([B0, 3], device)])
        test(op, [getter([2, 5, B0, 3], device)], in_dims=2)
        test(op, [getter([2, 5, B0, 3], device)], in_dims=2, out_dims=2)

        # Doubly nested vmap
        test(vmap(op), [getter([B0, B1], device)])
        test(vmap(op), [getter([B1, 2, 5, B0, 3], device)], in_dims=2)
        test(vmap(op, in_dims=2), [getter([2, 5, B0, B1, 3], device)],
             in_dims=2, out_dims=2)

    @parameterized("case", {
        'abs': (torch.abs, TensorFactory.randn),
        'acos': (torch.acos, TensorFactory.rand),
        'asin': (torch.asin, TensorFactory.rand),
        'atan': (torch.atan, TensorFactory.rand),
        'ceil': (torch.ceil, TensorFactory.randn),
        'cos': (torch.cos, TensorFactory.rand),
        'cosh': (torch.cosh, TensorFactory.rand),
        'digamma': (torch.digamma, TensorFactory.rand),
        'exp': (torch.exp, TensorFactory.randn),
        'expm1': (torch.expm1, TensorFactory.randn),
        'floor': (torch.floor, TensorFactory.randn),
        'frac': (torch.frac, TensorFactory.randn),
        'lgamma': (torch.lgamma, TensorFactory.rand),
        'log': (torch.log, TensorFactory.randp1),
        'log10': (torch.log10, TensorFactory.randp1),
        'log1p': (torch.log1p, TensorFactory.randp1),
        'log2': (torch.log2, TensorFactory.randp1),
        'neg': (torch.neg, TensorFactory.randn),
        'reciprocol': (torch.reciprocal, TensorFactory.randp1),
        'relu': (torch.relu, TensorFactory.randn),
        'round': (torch.round, TensorFactory.randn),
        'rsqrt': (torch.rsqrt, TensorFactory.randp1),
        'sigmoid': (torch.sigmoid, TensorFactory.randn),
        'sign': (torch.sign, TensorFactory.randn),
        'sin': (torch.sin, TensorFactory.rand),
        'sinh': (torch.sinh, TensorFactory.rand),
        'sqrt': (torch.sqrt, TensorFactory.rand),
        'tan': (torch.tan, TensorFactory.rand),
        'tanh': (torch.tanh, TensorFactory.rand),
        'trunc': (torch.trunc, TensorFactory.randn),
    })
    def test_unary_pointwise(self, case):
        op, getter = case
        self._test_unary(op, getter, 'cpu')

        # test in-place
        method = getattr(Tensor, f'{op.__name__ + "_"}')
        self._test_unary(method, getter, 'cpu', check_propagates_grad=False)

    def test_clone(self):
        # Some basic tests
        self._test_unary(lambda x: x.clone(), TensorFactory.randn, 'cpu')
        self._test_unary(lambda x: x.clone(memory_format=torch.preserve_format),
                         TensorFactory.randn, 'cpu')
        self._test_unary(lambda x: x.clone(memory_format=torch.contiguous_format),
                         TensorFactory.randn, 'cpu')

        # Test that the per-examples are contiguous when using torch.contiguous_format
        def clone_contiguous(x):
            return x.clone(memory_format=torch.contiguous_format)

        B0, B1 = 3, 5
        x = torch.randn(2, B0, 7)
        y = vmap(clone_contiguous, in_dims=1, out_dims=1)(x)
        self.assertTrue(y.movedim(1, 0).is_contiguous())
        self.assertTrue(y[:, 0, :].is_contiguous())

        x = torch.randn(2, B0, 7, B1)
        y = vmap(vmap(clone_contiguous, in_dims=2), in_dims=1)(x)
        self.assertTrue(y.is_contiguous())
        self.assertTrue(y[0][0].is_contiguous())


        msg = r'only supported with memory_format torch.preserve_format or torch.contiguous_format'
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(lambda x: x.clone(memory_format=torch.channels_last))(torch.randn(B0))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(lambda x: x.clone(memory_format=torch.channels_last_3d))(torch.randn(B0))

    @parameterized('case', {
        'clamp_min': _make_case(torch.clamp_min),
        'clamp_max': _make_case(torch.clamp_max),
    })
    def test_clamp_variant(self, case):
        test = self._vmap_test

        def get_number(getter):
            return getter([]).item()

        op, getter = case
        device = 'cpu'
        B0, B1 = 7, 11

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

        # Python number overload: op(Tensor, Number)
        number = get_number(getter)
        self._test_unary(lambda t: op(t, number), getter, device)

    @parameterized('case', {
        'add': _make_case(torch.add),
        'add_dunder': _make_case(lambda x, y: x + y),
        'sub': _make_case(torch.sub),
        'sub_dunder': _make_case(lambda x, y: x - y),
        'mul': _make_case(torch.mul),
        'mul_dunder': _make_case(lambda x, y: x * y),
        'div': _make_case(torch.div, input_getter=TensorFactory.randp1),
        'div_dunder': _make_case(lambda x, y: x / y, input_getter=TensorFactory.randp1),
        'pow': _make_case(torch.pow, input_getter=TensorFactory.randp1),
        'pow_dunder': _make_case(lambda x, y: x ** y, input_getter=TensorFactory.randp1),
    })
    def test_arithmetic(self, case):
        test = self._vmap_test

        def get_number(getter):
            return getter([]).item()

        op, getter = case
        device = 'cpu'
        B0, B1 = 7, 11

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
            return

        # TODO(rzou): fix the following
        # # Test cross-device scalars
        # number = get_number(getter)
        # self._test_unary(lambda t: op(t, number), getter, device='cuda')
        # self._test_unary(lambda t: op(number, t), getter, device='cuda')
        # self._test_unary(lambda t: op(t, torch.tensor(number)), getter, device='cuda')

    # TODO: as_strided BR
    @unittest.expectedFailure
    def test_as_strided(self):
        def _test(sizes, strides, offset, tensor, lambd):
            result = vmap(lambda t: t.as_strided(sizes, strides, offset))(tensor)
            expected = vmap(lambd)(tensor)
            self.assertTrue(result._base is expected._base)
            self.assertEqual(result, expected)

        # single vmap test
        B0 = 5
        tensors = [
            # contiguous
            torch.randn(B0, 2, 3),
            # non-contiguous
            torch.randn(B0, 3, 2).transpose(1, 2),
            # non-zero storage offset
            torch.randn(2, B0, 2, 3)[1],
            # non-contiguous strides, zero storage offset
            torch.randn(B0, 2, 4, 3, 7)[:, :, 0, :, 0],
            # non-contiguous strides, non-zero storage offset
            torch.randn(B0, 2, 4, 3, 7)[:, :, 2, :, 1],
        ]

        for x in tensors:
            S0, S1 = x.stride()[1:]
            offset = x.storage_offset()

            # Broadcast
            _test([5, 5, 2, 3], [0, 0, S0, S1], offset, x, lambda x: x.expand(5, 5, 2, 3))
            # transpose
            _test([3, 2], [S1, S0], offset, x, lambda x: x.transpose(0, 1))
            # select
            _test([2], [S0], offset + S1, x, lambda x: x[:, 1])

        # Nested vmap test
        B1 = 7
        x = torch.randn(B1, B0, 2, 3)
        S0, S1 = x.stride()[2:]
        result = vmap(vmap(lambda t: t.as_strided([5, 5, 2, 3], [0, 0, S0, S1])), in_dims=1)(x)
        expected = vmap(vmap(lambda t: t.expand(5, 5, 2, 3)), in_dims=1)(x)
        self.assertTrue(result._base is expected._base)
        self.assertEqual(result, expected)

        # Check that mal-formatted size/strides doesn't crash
        with self.assertRaisesRegex(RuntimeError, 'size and stride must have the same length'):
            x = torch.randn(B0, 2, 3).transpose(0, 1)
            vmap(lambda x: x.as_strided([1, 1, 1], [1, 1]))(x)

        # Sanity check #1: we require the batch dims to be at the front of the
        # tensor (in memory layout).
        msg = 'batch dims being vmapped over are at the front of the tensor'
        with self.assertRaisesRegex(RuntimeError, msg):
            x = torch.randn(2, B0, 3).transpose(0, 1)
            vmap(lambda x: x.as_strided([2, 3], [B0 * 3, 1]))(x)
        with self.assertRaisesRegex(RuntimeError, msg):
            x = torch.randn(B0, 2, 3, B1).movedim(3, 1)
            vmap(vmap(lambda x: x.as_strided([2, 3], [B1 * 3, B1])))(x)

        # All the Sanity check #2{a,b,c} cases check that
        # xs[i].as_strided(sizes, strides, offset + xs[i].offset() - xs.offset())
        # doesn't index memory that is out of bounds of xs[i]. This condition
        # is important to the correctness of the as_strided batching rule
        # (see NOTE: [When will the as_strided_batching_rule fail?])

        # Sanity check #2a: The maximum indexable location of
        # xs[i].as_strided(sizes, strides, offset + xs[i].offset() - xs.offset())
        # is less than or equal to the maximum indexable location of xs[i].
        msg = 'This is not supported inside of vmap'
        with self.assertRaisesRegex(RuntimeError, msg):
            x = torch.randn(B0, 3)
            vmap(lambda x: x.as_strided([3], [1], 1))(x)
        with self.assertRaisesRegex(RuntimeError, msg):
            x = torch.randn(B0, 3, 5)
            vmap(lambda x: x.as_strided([4, 4], [4, 1], 0))(x)
        with self.assertRaisesRegex(RuntimeError, msg):
            x = torch.randn(B0, B1, 3, 5)
            vmap(vmap(lambda x: x.as_strided([4, 4], [4, 1], 0)))(x)

        # Sanity check #2b: The min indexable location of
        # xs[i].as_strided(sizes, strides, offset + xs[i].offset() - xs.offset())
        # is greater than or equal to the min indexable location of xs[i].
        with self.assertRaisesRegex(RuntimeError, msg):
            x = torch.randn(2, B0, 3)[1]
            vmap(lambda x: x.as_strided([3], [1], B0 * 3 - 1))(x)

        # Sanity check #2c:
        # xs[i] is a zero-dim tensor, but
        # xs[i].as_strided(sizes, strides, offset + xs[i].offset() - xs.offset())
        # is not
        with self.assertRaisesRegex(RuntimeError, msg):
            x = torch.randn(B0, 0, 3)
            vmap(lambda x: x.as_strided([3], [1]))(x)

    def test_nll_loss(self):
        test = self._vmap_test
        op = F.nll_loss
        B = 3

        y = torch.randn(B, 2, 5)
        t = torch.randint(0, 5, (B, 2))
        test(op, (y, t))
        test(functools.partial(op, reduction='sum'), (y, t))
        test(functools.partial(op, reduction='none'), (y, t))

        y = torch.randn(B, 2, 5)
        t = torch.randint(0, 5, (2,))
        test(op, (y, t), in_dims=(0, None))
        test(functools.partial(op, reduction='sum'), (y, t), in_dims=(0, None))
        test(functools.partial(op, reduction='none'), (y, t), in_dims=(0, None))

    def test_adaptive_avg_pool2d(self):
        test = self._vmap_test
        op = functools.partial(F.adaptive_avg_pool2d, output_size=(3, 3))

        x = torch.randn(3, 5, 7, 9, 11)
        test(op, (x,))
        test(op, (x,), in_dims=(1,))
        test(op, (x,), in_dims=(4,))

    def test_bmm(self):
        op = torch.bmm
        test = self._vmap_test
        B0, B1 = 7, 11

        # shape mismatch
        msg = ""
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op)(torch.randn(B0, 2, 2, 2), torch.randn(B0, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(0, None))(torch.randn(B0, 3, 3, 2), torch.randn(2, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(None, 0))(torch.randn(2, 2), torch.randn(B0, 2, 2, 2))

        # left arg is vmapped
        test(op, (torch.rand(B0, 2, 3, 5), torch.rand(2, 5, 3)), in_dims=(0, None))
        test(vmap(op, in_dims=(0, None)), (torch.rand(B1, B0, 2, 3, 5), torch.rand(2, 5, 3)),
             in_dims=(1, None))

        # right arg is vmapped
        test(op, (torch.rand(2, 5, 3), torch.rand(B0, 2, 3, 5)), in_dims=(None, 0))
        test(vmap(op, in_dims=(None, 0)), (torch.rand(2, 5, 3), torch.rand(B1, B0, 2, 3, 5)),
             in_dims=(None, 1))

        # both args are vmapped
        test(op, (torch.rand(B0, 2, 3, 5), torch.rand(B0, 2, 5, 3)))
        test(vmap(op), (torch.rand(B1, B0, 2, 3, 5), torch.rand(B0, B1, 2, 5, 3)), in_dims=(1, 0))
        test(vmap(op, in_dims=(0, None)),
             (torch.rand(B1, 2, 3, 5), torch.rand(B0, 2, 5, 3)), in_dims=(None, 0))

    def test_cat(self):
        test = self._vmap_test
        B0, B1 = 5, 7

        # Quick hack b/c vmap can't accept a list of tensors as an argument
        def get_op(dim):
            def op(*tensors):
                return torch.cat(tensors, dim=dim)
            return op

        test(get_op(0), (torch.rand(B0, 2), torch.rand(B0, 3)))
        test(get_op(0), (torch.rand(2), torch.rand(B0, 3)), in_dims=(None, 0))
        test(get_op(0), (torch.rand(2, 17), torch.rand(3, 17, B0)), in_dims=(None, 2))
        test(get_op(-1), (torch.rand(17, 2), torch.rand(17, 3, B0)), in_dims=(None, 2))
        test(vmap(get_op(0), in_dims=(0, None)),
             (torch.rand(B1, 2), torch.rand(B0, 3)), in_dims=(None, 0))
        test(vmap(get_op(0), in_dims=(0, 0)),
             (torch.rand(B1, 2), torch.rand(B0, B1, 3)), in_dims=(None, 0))

    def test_unsafe_view(self):
        # Unsafe view isn't exposed, so we get at it via
        # vmap(grad(matmul))
        test = functools.partial(self._vmap_test, check_propagates_grad=False)
        B = 2
        x = torch.randn(B, 2, 3, 3)
        y = torch.randn(B, 3, 3)

        def baz(x, y):
            return (x @ y).sum()

        test(functorch.grad(baz), (x, y))

    def test_conj(self):
        op = torch.conj

        def run_test(dtype):
            def get(shape):
                return torch.randn(shape, dtype=dtype)
            B0, B1 = 7, 11
            test = self._vmap_test

            # Single vmap, various in_dims / out_dims
            test(op, [get([B0, 3])])
            test(op, [get([2, 5, B0, 3])], in_dims=2)
            test(op, [get([2, 5, B0, 3])], in_dims=2, out_dims=2)

            # Doubly nested vmap
            test(vmap(op), [get([B0, B1])])
            test(vmap(op), [get([B1, 2, 5, B0, 3])], in_dims=2)
            test(vmap(op, in_dims=2), [get([2, 5, B0, B1, 3])],
                 in_dims=2, out_dims=2)

        # correctness tests
        run_test(torch.float)
        run_test(torch.cfloat)

        # check that torch.conj on a non-complex tensor returns the same tensor
        real_tensor = torch.randn(3)
        result = vmap(op)(real_tensor)
        self.assertEqual(result.data_ptr(), real_tensor.data_ptr())

    def test_contiguous(self):
        op = Tensor.contiguous

        self._test_unary(op, TensorFactory.randn, 'cpu')

        # check that contiguous returns the original tensor if the per-examples
        # are already contiguous
        B0 = 3
        x = torch.randn(B0, 2, 5, 7)
        x = x.movedim(0, 2)
        result = vmap(Tensor.contiguous, in_dims=2, out_dims=2)(x)
        self.assertTrue(result is x)

        msg = 'NYI: querying is_contiguous inside of vmap for memory_format'
        tensor = torch.randn(B0, 3)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(functools.partial(op, memory_format=torch.channels_last))(tensor)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(functools.partial(op, memory_format=torch.channels_last_3d))(tensor)

    def test_stride(self):
        B0 = 3

        x = torch.randn(B0, 2, 5, 7)

        def foo(x):
            assert x.stride() == (7 * 5, 7, 1)
            return x

        vmap(foo)(x)

        x = torch.randn(2, B0, 5, 7).movedim(1, 0)

        def bar(x):
            assert x.stride() == (7 * 5 * B0, 7, 1)
            return x

        vmap(bar)(x)

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

    def test_clamp(self):
        clamp_cases = (
            (lambda t: t.clamp(min=-0.5), TensorFactory.randn),
            (lambda t: t.clamp(max=0.5), TensorFactory.randn),
            (lambda t: t.clamp(min=-0.5, max=0.5), TensorFactory.randn),
            (lambda t: t.clamp_min(min=-0.5), TensorFactory.randn),
            (lambda t: t.clamp_max(max=0.5), TensorFactory.randn),
        )
        for op, getter in clamp_cases:
            self._test_unary(op, getter, 'cpu')

    def test_comparison_ops(self):
        test = functools.partial(self._vmap_test, check_propagates_grad=False)

        getter = TensorFactory.randn
        B0, B1 = 7, 11

        ops = (
            torch.eq, lambda x, y: x == y,
            torch.gt, lambda x, y: x > y,
            torch.ge, lambda x, y: x >= y,
            torch.le, lambda x, y: x <= y,
            torch.lt, lambda x, y: x < y,
            torch.ne, lambda x, y: x != y,
        )

        for op in ops:
            # Single vmap: op(Tensor, Tensor)
            test(op, (getter([B0, 3]), getter([B0, 3])))
            test(op, (getter([B0]), getter([B0, 2, 3])))
            test(op, (getter([B0]), getter([2, B0, 3])), in_dims=(0, 1))
            test(op, (getter([B0]), getter([2, B0, 3])), in_dims=(0, 1), out_dims=1)
            test(op, (getter([B0]), getter([2, 3])), in_dims=(0, None))
            test(op, (getter([2, 3]), getter([B0, 3])), in_dims=(0, None))

            # Nested vmap: op(Tensor, Tensor)
            test(vmap(op), (getter([B0, B1, 2, 3]), getter([B0, B1, 3])))
            test(vmap(op, in_dims=(None, 0)),
                 (getter([B0, 2, 3]), getter([B1, 3])), in_dims=(0, None))

            # test number as inputs
            number = getter([]).item()
            self._test_unary(lambda t: op(t, number), getter, 'cpu', check_propagates_grad=False)

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

    def test_dot(self):
        op = torch.dot
        test = self._vmap_test
        B0, B1 = 7, 11

        # shape mismatch
        msg = ""
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op)(torch.randn(B0, 2, 2, 2), torch.randn(B0, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(0, None))(torch.randn(B0, 2), torch.randn(2, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(None, 0))(torch.randn(2, 2), torch.randn(B0, 2))

        # left arg is vmapped
        test(op, (torch.rand(B0, 5), torch.rand(5)), in_dims=(0, None))
        test(vmap(op, in_dims=(0, None)), (torch.rand(B1, B0, 5), torch.rand(5)),
             in_dims=(1, None))

        # right arg is vmapped
        test(op, (torch.rand(5), torch.rand(B0, 5)), in_dims=(None, 0))
        test(vmap(op, in_dims=(None, 0)), (torch.rand(5), torch.rand(B1, B0, 5)),
             in_dims=(None, 1))

        # both args are vmapped
        test(op, (torch.rand(B0, 5), torch.rand(B0, 5)))
        test(vmap(op), (torch.rand(B1, B0, 5), torch.rand(B0, B1, 5)), in_dims=(1, 0))
        test(vmap(op, in_dims=(0, None)),
             (torch.rand(B1, 5), torch.rand(B0, 5)), in_dims=(None, 0))

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

    def test_fill_and_zero_inplace(self):
        test = functools.partial(self._vmap_test, check_propagates_grad=False)
        B0, B1 = 7, 11
        ops = (
            lambda t: t.fill_(0.1),
            lambda t: t.fill_(torch.tensor(0.2)),
            lambda t: t.zero_(),
        )

        for op in ops:
            # Single vmap, various in_dims / out_dims
            test(op, [TensorFactory.randn([B0, 3])])
            test(op, [TensorFactory.randn([2, 5, B0, 3])], in_dims=2)
            test(op, [TensorFactory.randn([2, 5, B0, 3])], in_dims=2, out_dims=2)

            # Doubly nested vmap
            test(vmap(op), [TensorFactory.randn([B0, B1])])
            test(vmap(op), [TensorFactory.randn([B1, 2, 5, B0, 3])], in_dims=2)
            test(vmap(op, in_dims=2), [TensorFactory.randn([2, 5, B0, B1, 3])],
                 in_dims=2, out_dims=2)

        # test when value is a batched tensor for fill_ operator
        B0, B1 = 3, 5
        test(Tensor.fill_, [TensorFactory.randn([B0, B1]), TensorFactory.randn(B0)])

        with self.assertRaisesRegex(RuntimeError,
                                    ""):
            # Runtime Error is thrown when the tensor being written to isn't being vmapped over
            vmap(Tensor.fill_, (None, 0))(TensorFactory.randn([B0, B1]),
                                          TensorFactory.randn([B0]))

    def _test_complex_views(self, op, dtypes):
        test = self._vmap_view_test

        def run_test(op, dtype):
            def get(shape):
                return torch.randn(shape, dtype=dtype)

            B0, B1 = 7, 11

            # Single vmap, various in_dims / out_dims
            test(op, [get([B0, 3])])
            test(op, [get([3, B0])], in_dims=1)
            test(op, [get([2, 5, B0, 3])], in_dims=2)
            test(op, [get([2, 5, B0, 3])], in_dims=2, out_dims=2)

            # Doubly nested vmap
            test(vmap(op), [get([B0, B1])])
            test(vmap(op), [get([B1, 2, 5, 3, B0])], in_dims=4)
            test(vmap(op, in_dims=2), [get([2, 5, B0, B1, 3])],
                 in_dims=2, out_dims=2)

        for dtype in dtypes:
            run_test(op, dtype)

    def test_real(self):
        self._test_complex_views(torch.real, dtypes=[torch.cfloat, torch.cdouble])

    def test_imag(self):
        self._test_complex_views(torch.imag, dtypes=[torch.cfloat, torch.cdouble])

    def test_view_as_real(self):
        self._test_complex_views(torch.view_as_real, dtypes=[torch.cfloat, torch.cdouble])

    def test_view_as_complex(self):
        def run_test(dtype):
            def get(shape):
                return torch.randn(shape, dtype=dtype)

            op = torch.view_as_complex
            test = self._vmap_view_test
            B0, B1 = 7, 11

            # Single vmap, various in_dims / out_dims
            test(op, [get([B0, 3, 2])])
            test(op, [get([2, 5, B0, 3, 2])], in_dims=2)
            test(op, [get([2, 5, B0, 3, 2])], in_dims=2, out_dims=2)

            # Doubly nested vmap
            test(vmap(op), [get([B0, B1, 2])])
            test(vmap(op), [get([B1, 2, 5, B0, 3, 2])], in_dims=2)
            test(vmap(op, in_dims=2), [get([2, 5, B0, B1, 3, 2])],
                 in_dims=2, out_dims=2)

            # Interesting case #1: Batch dim directly before dim of size 2
            test(op, [get([3, B0, 2])], in_dims=1)
            test(vmap(op, in_dims=1), [get([3, B1, B0, 2])], in_dims=2)

            # Interesting case #2: Batch dim at end of tensor, success cases
            # view_as_complex requires that the dim with size 2 have stride 1
            # in order for the view to function propertly
            test(op, [get([B0, 2]).transpose(0, 1)], in_dims=1)
            test(vmap(op, in_dims=1), [get([B0, B1, 2]).movedim(1, 2)])
            test(vmap(op, in_dims=2), [get([B0, 3, B1, 2]).movedim(2, 3)])

            # Interesting case #3: Batch dim at end of tensor, failure cases
            msg = "Tensor must have a last dimension with stride 1"
            with self.assertRaisesRegex(RuntimeError, msg):
                vmap(op, in_dims=1)(get([2, B0]))
            with self.assertRaisesRegex(RuntimeError, msg):
                vmap(vmap(op, in_dims=1), in_dims=1)(get([2, B0, B1]))

            # Invalid input: no dimension of size 2
            msg = 'Input tensor must have one or more dimensions'
            with self.assertRaisesRegex(RuntimeError, msg):
                vmap(op)(get([B0]))
            with self.assertRaisesRegex(RuntimeError, msg):
                vmap(vmap(op))(get([B0, B1]))

            # Invalid input: Batch dim has size 2, but the logical last dim does
            # not have size 2
            msg = 'Tensor must have a last dimension of size 2'
            with self.assertRaisesRegex(RuntimeError, msg):
                vmap(op, in_dims=1)(get([3, 2]))

        for dtype in [torch.float, torch.double]:
            run_test(dtype)

    def test_is_complex(self):
        ctensor = torch.randn(3, dtype=torch.cfloat)
        tensor = torch.randn(3)

        def foo(x):
            if x.is_complex():
                return torch.tensor(1)
            else:
                return torch.tensor(0)

        self.assertEqual(vmap(foo)(ctensor), torch.tensor([1, 1, 1]))
        self.assertEqual(vmap(foo)(tensor), torch.tensor([0, 0, 0]))

    def test_is_floating_point(self):
        float_tensor = torch.tensor([1., 2., 3.])
        long_tensor = torch.tensor([1, 2, 3])

        def foo(x):
            if x.is_floating_point():
                return torch.tensor(1)
            else:
                return torch.tensor(0)

        self.assertEqual(vmap(foo)(float_tensor), torch.tensor([1, 1, 1]))
        self.assertEqual(vmap(foo)(long_tensor), torch.tensor([0, 0, 0]))

    def test_is_contiguous(self):
        def foo(x):
            if x.is_contiguous():
                return torch.tensor(1.)
            else:
                return torch.tensor(0.)

        B0, B1 = 3, 5

        # Single batch dim
        contig = torch.randn(B0, 2, 7)
        self.assertEqual(vmap(foo)(contig), torch.ones(B0))

        noncontig = torch.randn(2, B0, 7)
        self.assertEqual(vmap(foo, in_dims=1)(noncontig), torch.zeros(B0))

        noncontig = torch.randn(2, B0, 7).movedim(1, 0)
        self.assertEqual(vmap(foo)(noncontig), torch.zeros(B0))

        noncontig = torch.randn(2, 7, B0)
        self.assertEqual(vmap(foo, in_dims=2)(noncontig), torch.zeros(B0))

        # Multiple batch dims
        contig = torch.randn(B0, B1, 3)
        self.assertEqual(vmap(vmap(foo))(contig), torch.ones(B0, B1))

        contig = torch.randn(B1, B0, 3)
        self.assertEqual(vmap(vmap(foo), in_dims=1)(contig), torch.ones(B0, B1))

        contig = torch.randn(B1, B0, 3).movedim(0, 1)
        self.assertEqual(vmap(vmap(foo))(contig), torch.ones(B0, B1))

        noncontig = torch.randn(B0, 3, B1)
        self.assertEqual(vmap(vmap(foo, in_dims=1))(noncontig), torch.zeros(B0, B1))

        # is_contiguous on empty tensor is True
        def bar(x):
            assert x.is_contiguous()
            return x

        vmap(bar)(torch.randn(B0, 0, 3))
        vmap(bar, in_dims=1)(torch.randn(0, B0, 3))
        vmap(bar)(torch.randn(B0, 0, 3).transpose(-1, -2))

        # is_contiguous with other memory formats
        def baz(x, memory_format):
            x.is_contiguous(memory_format=memory_format)
            return x

        msg = 'NYI: querying is_contiguous inside of vmap for memory_format'
        tensor = torch.randn(B0, 2, 7, 3)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(functools.partial(baz, memory_format=torch.channels_last))(tensor)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(functools.partial(baz, memory_format=torch.channels_last_3d))(tensor)

    def test_unsqueeze(self):
        op = torch.unsqueeze
        test = self._vmap_view_test
        B0, B1, B2 = 7, 11, 13

        # unsqueeze dim 0
        test(op, (torch.rand(B0, 2, 5), 0), in_dims=(0, None))
        test(op, (torch.rand(2, B0, 5), 0), in_dims=(1, None))

        # unsqueeze last dim (positive)
        test(op, (torch.rand(B0, 2, 5), 2), in_dims=(0, None))
        test(op, (torch.rand(2, B0, 5), 2), in_dims=(1, None))

        # unsqueeze last dim (negative)
        test(op, (torch.rand(B0, 2, 5), -1), in_dims=(0, None))
        test(op, (torch.rand(2, B0, 5), -1), in_dims=(1, None))

        # nested vmaps
        def unsqueeze_0(x):
            return torch.unsqueeze(x, 0)

        def unsqueeze_last(x):
            return torch.unsqueeze(x, -1)

        # bdims in canonical order
        test(vmap(unsqueeze_0), (torch.rand(B0, B1, 2), ))
        test(vmap(unsqueeze_last), (torch.rand(B0, B1, 2),))

        # wild bdims
        test(vmap(unsqueeze_0), (torch.rand(B1, 2, B0),), in_dims=2)
        test(vmap(unsqueeze_0, in_dims=1), (torch.rand(2, B1, B0),), in_dims=2)
        test(vmap(unsqueeze_last), (torch.rand(B1, 2, B0),), in_dims=2)
        test(vmap(unsqueeze_last, in_dims=1), (torch.rand(2, B1, B0),), in_dims=2)

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

    def test_mm(self):
        op = torch.mm
        test = self._vmap_test
        B0, B1 = 7, 11

        # shape mismatch
        msg = "Shape mismatch"
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op)(torch.randn(B0, 2, 2, 2), torch.randn(B0, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(0, None))(torch.randn(B0, 2), torch.randn(2, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(None, 0))(torch.randn(2, 2), torch.randn(B0, 2, 2, 2))

        # left arg is vmapped
        test(op, (torch.rand(B0, 2, 5), torch.rand(5, 2)), in_dims=(0, None))
        test(vmap(op, in_dims=(0, None)), (torch.rand(B1, B0, 2, 5), torch.rand(5, 2)),
             in_dims=(1, None))

        # right arg is vmapped
        test(op, (torch.rand(2, 5), torch.rand(B0, 5, 2)), in_dims=(None, 0))
        test(vmap(op, in_dims=(None, 0)), (torch.rand(2, 5), torch.rand(B1, B0, 5, 2)),
             in_dims=(None, 1))

        # both args are vmapped
        test(op, (torch.rand(B0, 2, 5), torch.rand(B0, 5, 2)))
        test(vmap(op), (torch.rand(B1, B0, 2, 5), torch.rand(B0, B1, 5, 2)), in_dims=(1, 0))
        test(vmap(op, in_dims=(0, None)),
             (torch.rand(B1, 2, 5), torch.rand(B0, 5, 2)), in_dims=(None, 0))

    def test_mv(self):
        op = torch.mv
        test = self._vmap_test
        B0, B1 = 7, 11

        # shape mismatch
        msg = ""
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op)(torch.randn(B0, 2, 2, 2), torch.randn(B0, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(0, None))(torch.randn(B0, 2, 2), torch.randn(2, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(None, 0))(torch.randn(2, 2), torch.randn(B0, 2, 2))

        # left arg is vmapped
        test(op, (torch.rand(B0, 2, 5), torch.rand(5)), in_dims=(0, None))
        test(vmap(op, in_dims=(0, None)), (torch.rand(B1, B0, 2, 5), torch.rand(5)),
             in_dims=(1, None))

        # right arg is vmapped
        test(op, (torch.rand(2, 5), torch.rand(B0, 5)), in_dims=(None, 0))
        test(vmap(op, in_dims=(None, 0)), (torch.rand(2, 5), torch.rand(B1, B0, 5)),
             in_dims=(None, 1))

        # both args are vmapped
        test(op, (torch.rand(B0, 2, 5), torch.rand(B0, 5)))
        test(vmap(op), (torch.rand(B1, B0, 2, 5), torch.rand(B0, B1, 5)), in_dims=(1, 0))
        test(vmap(op, in_dims=(0, None)),
             (torch.rand(B1, 2, 5), torch.rand(B0, 5)), in_dims=(None, 0))

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

    def test_new_empty(self):
        # Empty is non-deterministic so we just check that the shape of the
        # output tensor is what we expect and that the vmap fallback isn't used.
        op = Tensor.new_empty

        B0, B1 = 7, 11

        result = vmap(lambda x: op(x, [2, 3]))(torch.randn(B0))
        self.assertEqual(result.shape, [B0, 2, 3])

        result = vmap(lambda x: op(x, []))(torch.randn(B0))
        self.assertEqual(result.shape, [B0])

        result = vmap(vmap(lambda x: op(x, [2, 3])))(torch.randn(B0, B1))
        self.assertEqual(result.shape, [B0, B1, 2, 3])

    # TODO: new_empty_strided BR
    @unittest.expectedFailure
    def test_new_empty_strided(self):
        # Empty is non-deterministic so we just check that the size and shape
        # of the output are what we expect and that the vmap fallback isn't used
        B0, B1 = 7, 11

        def _test_single_vmap(size, stride, B0):
            x = torch.randn(B0)
            result = vmap(lambda x: x.new_empty_strided(size, stride))(x)
            S = torch.empty_strided(size, stride).storage().size()
            self.assertEqual(result.shape, [B0] + size)
            self.assertEqual(result.stride(), [S] + stride)

        def _test_double_vmap(size, stride, B0, B1):
            x = torch.randn(B0, B1)
            result = vmap(vmap(lambda x: x.new_empty_strided(size, stride)))(x)
            S = torch.empty_strided(size, stride).storage().size()
            self.assertEqual(result.shape, [B0, B1] + size)
            self.assertEqual(result.stride(), [B1 * S, S] + stride)

            x = torch.randn(B1, B0)
            result = vmap(vmap(lambda x: x.new_empty_strided(size, stride)), in_dims=1)(x)
            S = x.new_empty_strided(size, stride).storage().size()
            self.assertEqual(result.shape, [B0, B1] + size)
            self.assertEqual(result.stride(), [B1 * S, S] + stride)

        # contiguous case
        _test_single_vmap([2, 3, 5], [3 * 5, 5, 1], B0)
        _test_double_vmap([2, 3, 5], [3 * 5, 5, 1], B0, B1)

        # expanded
        _test_single_vmap([2, 3, 5], [0, 5, 1], B0)
        _test_double_vmap([2, 3, 5], [0, 5, 1], B0, B1)

        # some of these cases are pretty strange, just verifying that if
        # empty_strided allows them then BatchedTensor.new_empty_strided
        # can as well
        for shape in [[2, 3, 4], [0, 2, 0]]:
            for strides in [[12, 4, 1], [2, 4, 6], [0, 0, 0]]:
                _test_single_vmap(shape, strides, B0)
                _test_double_vmap(shape, strides, B0, B1)

    def test_new_zeros(self):
        op = Tensor.new_zeros
        test = functools.partial(self._vmap_test, check_propagates_grad=False)
        B0, B1 = 7, 11

        test(lambda x: op(x, 2, 3), (torch.rand(B0),))
        test(lambda x: op(x, []), (torch.rand(B0),))
        test(vmap(lambda x: op(x, 3, 5)), (torch.rand(B0, B1),))

    def test_select(self):
        op = torch.select
        test = self._vmap_view_test
        B0, B1, B2 = 7, 11, 13
        test(op, (torch.rand(B0, 2, 5), 0, 0), in_dims=(0, None, None))
        test(op, (torch.rand(2, B0, 5), 1, 1), in_dims=(1, None, None))
        test(vmap(lambda t: op(t, 1, 1)), (torch.rand(B1, 2, B0, 5),), in_dims=2)
        test(vmap(vmap(lambda t: op(t, 1, 1), in_dims=1)), (torch.rand(B1, 2, B0, B2, 5),), in_dims=2)

    def test_stack(self):
        test = self._vmap_test
        B0, B1 = 5, 7

        # Quick hack b/c vmap can't accept a list of tensors as an argument
        def get_op(dim):
            def op(*tensors):
                return torch.stack(tensors, dim=dim)
            return op

        test(get_op(0), (torch.rand(B0, 3), torch.rand(B0, 3)))
        test(get_op(0), (torch.rand(3), torch.rand(B0, 3)), in_dims=(None, 0))
        test(get_op(0), (torch.rand(2, 17), torch.rand(2, 17, B0)), in_dims=(None, 2))
        test(get_op(-1), (torch.rand(2, 17), torch.rand(2, 17, B0)), in_dims=(None, 2))
        test(vmap(get_op(0), in_dims=(0, None)),
             (torch.rand(B1, 2), torch.rand(B0, 2)), in_dims=(None, 0))
        test(vmap(get_op(0), in_dims=(0, 0)),
             (torch.rand(B1, 2), torch.rand(B0, B1, 2)), in_dims=(None, 0))


    def test_slice(self):
        test = self._vmap_view_test
        B0, B1, B2 = 7, 11, 13
        test(lambda t: t[0:1], (torch.rand(B0, 3, 5),))
        test(lambda t: t[:, 1:3], (torch.rand(3, 5, B0),), in_dims=2)
        test(vmap(lambda t: t[:, 0:1], in_dims=2), (torch.rand(3, 5, B0, B1),), in_dims=2)
        test(vmap(vmap(lambda t: t[0:1], in_dims=2), in_dims=2),
             (torch.rand(3, 5, B0, B1, B2),), in_dims=2)

    def test_squeeze(self):
        def verify_behavior(op, min_ndim=1):
            test = self._vmap_view_test
            B0, B1 = 1, 11
            # These tests cannot be used with an operator that requires more
            # than 1 dimension after batching.
            if min_ndim <= 1:
                test(op, (torch.rand(B0),))
                test(op, (torch.rand(B1),))
                test(vmap(op), (torch.rand(B0, B1, 1),))
                test(vmap(op), (torch.rand(B1, 1, B0),), in_dims=2)
            test(op, (torch.rand(B0, 3, 5),))
            test(op, (torch.rand(1, B0, 5),), in_dims=1)
            test(op, (torch.rand(B0, 0, 1, 5, 1),))
            test(op, (torch.rand(B0, 1, 1, 1, 1),))
            test(vmap(op), (torch.rand(B0, B1, 1, 3, 4),))
            test(vmap(op), (torch.rand(B1, 1, B0, 4, 5),), in_dims=2)

        verify_behavior(torch.squeeze)
        verify_behavior(lambda x: torch.squeeze(x, dim=0), min_ndim=1)
        verify_behavior(lambda x: torch.squeeze(x, dim=1), min_ndim=2)
        verify_behavior(lambda x: torch.squeeze(x, dim=-1), min_ndim=2)
        verify_behavior(lambda x: torch.squeeze(x, dim=-2), min_ndim=3)

        msg = ""
        try:
            torch.squeeze(torch.rand(10), dim=1)
        except IndexError as err:
            msg = str(err)
        with self.assertRaises(RuntimeError, msg=msg):
            vmap(lambda x: torch.squeeze(x, dim=1))(torch.rand(10))

    def _test_mean_sum_dim(self, op):
        test = self._vmap_test
        B0, B1 = 5, 7

        # Single vmap, various in_dims / out_dims
        test(lambda x: op(x, 0), [torch.randn([B0])])
        test(lambda x: op(x, -1), [torch.randn([B0])])
        test(lambda x: op(x, 0), [torch.randn([B0, 3])])
        test(lambda x: op(x, -1), [torch.randn([2, 5, B0, 3])], in_dims=2)
        test(lambda x: op(x, 2), [torch.randn([2, 5, B0, 3])], in_dims=2, out_dims=2)

        # Doubly nested vmap
        test(vmap(lambda x: op(x, 0)), [torch.randn([B0, B1])])
        test(vmap(lambda x: op(x, -1)), [torch.randn([B0, B1])])
        test(vmap(lambda x: op(x, -2)), [torch.randn([B1, 2, 5, B0, 3])], in_dims=2)
        test(vmap(lambda x: op(x, 2), in_dims=2), [torch.randn([2, 5, B0, B1, 3])],
             in_dims=2, out_dims=2)

    def test_sum_dim(self):
        self._test_mean_sum_dim(torch.sum)

    def test_mean_dim(self):
        self._test_mean_sum_dim(torch.mean)

    def test_argmax_dim(self):
        def test(f, args):
            for loop_out, batched_out in get_fallback_and_vmap_exhaustive(f, args, {}):
                self.assertEqual(loop_out, batched_out)
        B0 = 5
        test(lambda x: torch.argmax(x), [torch.randn(B0)])
        test(lambda x: torch.argmax(x), [torch.randn(B0, 2, 3)])
        test(lambda x: torch.argmax(x, 0), [torch.randn(B0, 2, 3)])
        test(lambda x: torch.argmax(x, -1), [torch.randn(B0, 2, 3)])
        test(lambda x: torch.argmax(x, 2), [torch.randn(B0, 2, 3)])

    def _test_sum_mean(self, op):
        test = self._vmap_test
        B0, B1 = 5, 7

        # Single vmap, various in_dims / out_dims
        test(op, [torch.randn([B0])])
        test(op, [torch.randn([B0, 3])])
        test(op, [torch.randn([2, 5, B0, 3])], in_dims=2)
        test(op, [torch.randn([2, 5, B0, 3])], in_dims=2)

        # Doubly nested vmap
        test(vmap(op), [torch.randn([B0, B1])])
        test(vmap(op), [torch.randn([B1, 2, 5, B0, 3])])
        test(vmap(op), [torch.randn([2, 5, B0, B1, 3])], in_dims=2)

    def test_sum(self):
        self._test_sum_mean(torch.sum)

    def test_mean(self):
        self._test_sum_mean(torch.mean)

    def test_repeat(self):
        test = self._vmap_test
        B0 = 7
        op = Tensor.repeat
        test(lambda x: op(x, (2, 3)), (torch.rand(B0, 1, 1),))
        test(lambda x: op(x, (2, 3)), (torch.rand(1, B0, 1),), in_dims=1)

    def test_slogdet(self):
        test = functools.partial(self._vmap_test, check_propagates_grad=False)
        B0 = 7
        op = torch.linalg.slogdet
        test(op, (torch.rand(B0, 1, 1),))
        test(op, (torch.rand(B0, 2, 2),))
        test(op, (torch.rand(B0, 3, 2, 2),))
        test(op, (torch.rand(3, 2, 2, B0),), in_dims=3)

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

    def test_tensor_split(self):
        test = self._vmap_view_test
        op = torch.tensor_split
        B0, B1, B2 = 7, 11, 13

        # tests for torch.tensor_split(self, indices_or_sections: int, dim)
        test(op, (torch.rand(B0, 2, 1024), 5, -1), in_dims=(0, None, None))
        test(op, (torch.rand(2, B0, 1024), 150, 1), in_dims=(1, None, None))
        test(vmap(op, in_dims=(0, None, None)), (torch.rand(B1, 1023, B0, 5), 256, 0),
             in_dims=(2, None, None))
        test(vmap(vmap(lambda t: op(t, 4, 1), in_dims=2)),
             (torch.rand(B1, 2, B0, 64, B2),), in_dims=2)

        # tests for torch.tensor_split(self, indices_or_sections: List[int], dim)
        test(op, (torch.rand(B0, 2, 1024), [50, 100, 378, 890], -1), in_dims=(0, None, None))
        test(op, (torch.rand(2, B0, 1024), [50, 100, 212, 345, 0, 378, 890], 1), in_dims=(1, None, None))
        test(vmap(op, in_dims=(0, None, None)), (torch.rand(B1, 1023, B0, 5), [50, 100, 212, 345, 0, 378, 890], 0),
             in_dims=(2, None, None))
        test(vmap(vmap(lambda t: op(t, [4, 8, 9, 34, 29], 1), in_dims=2)),
             (torch.rand(B1, 2, B0, 64, B2),), in_dims=2)

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

    def test_trace(self):
        op = torch.trace
        test = self._vmap_test
        B0, B1, B2 = 7, 11, 13
        test(op, (torch.rand(B0, 2, 5),))
        test(op, (torch.rand(2, B0, 5),), in_dims=1)
        test(vmap(op), (torch.rand(B1, 2, B0, 5),), in_dims=2)
        test(vmap(vmap(op, in_dims=2)), (torch.rand(B1, 2, B0, 5, B2),), in_dims=2)

    def test_transpose(self):
        op = torch.transpose
        test = self._vmap_view_test

        B0, B1, B2 = 7, 11, 13
        test(lambda x: op(x, 0, 1), (torch.rand(B0, 2, 5),))
        test(lambda x: op(x, -1, -2), (torch.rand(B0, 2, 5),))
        test(lambda x: op(x, 3, 1), (torch.rand(B0, 2, 5, 4, 6),))
        test(lambda x: op(x, 1, 0), (torch.rand(2, B0, 5),), in_dims=1)
        test(vmap(lambda x: op(x, 0, 1)), (torch.rand(B1, 2, B0, 5),), in_dims=2)
        test(vmap(vmap(lambda x: op(x, 0, 1), in_dims=2)),
             (torch.rand(B1, 2, B0, 5, B2),), in_dims=2)

        # Special case: scalar tensor
        for dim1, dim2 in itertools.product([0, -1], [0, -1]):
            x = torch.rand(B0)
            result = vmap(lambda x: op(x, dim1, dim2))(x)
            self.assertTrue(result is x)

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

    # TODO: reenable the random op failures
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
            # (torch.rand_like, (torch.rand(B0, 1),)),
            # (torch.randn_like, (torch.rand(B0, 1),)),
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
            # (lambda t: torch.rand_like(captured), (torch.rand(B0),)),
            # (lambda t: torch.randn_like(captured) , (torch.rand(B0),)),
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
            # (lambda t: torch.randn(1), (torch.randn(B0),)),
            (lambda t: torch.randint(5, [1]), (torch.randn(B0),)),
            (lambda t: torch.randperm(5), (torch.randn(B0),)),
        ]
        for op, args in random_ops:
            with self.assertRaisesRegex(RuntimeError,
                                        'vmap: We do not yet support calling random operations'):
                vmap(op)(*args)

    def test_conv2d(self):
        conv_setups = [
            (torch.nn.Conv1d, torch.conv1d, [2, 4, 15]),
            (torch.nn.Conv2d, torch.conv2d, [2, 4, 15, 20]),
            (torch.nn.Conv3d, torch.conv3d, [2, 4, 15, 20, 25]),
            # (torch.nn.ConvTranspose2d, torch.conv_transpose2d, [2, 4, 15, 20])
        ]
        for conv_mod, conv_fn, inp_shape in conv_setups:
            mod = conv_mod(4, 8, kernel_size=3)
            arg_values = [torch.randn(inp_shape), mod.weight, mod.bias]
            kwarg_values = {}
            for loop_out, batched_out in get_fallback_and_vmap_exhaustive(conv_fn, arg_values, kwarg_values):
                self.assertEqual(loop_out, batched_out)

            arg_values = [torch.randn(inp_shape), mod.weight, None]
            for loop_out, batched_out in get_fallback_and_vmap_exhaustive(conv_fn, arg_values, kwarg_values):
                self.assertEqual(loop_out, batched_out)

            mod2 = torch.nn.Conv2d(4, 8, kernel_size=3, groups=2, stride=3, padding=1, dilation = 2)
            arg_values = [torch.randn(inp_shape), mod.weight, mod.bias]
            for loop_out, batched_out in get_fallback_and_vmap_exhaustive(conv_fn, arg_values, kwarg_values):
                self.assertEqual(loop_out, batched_out)

            arg_values = [torch.randn(inp_shape), mod.weight, None]
            for loop_out, batched_out in get_fallback_and_vmap_exhaustive(conv_fn, arg_values, kwarg_values):
                self.assertEqual(loop_out, batched_out)

    def test_one_hot(self):
        sample_inputs = [
            (torch.randint(0, 3, []), 3),
            (torch.randint(0, 3, [2, 3, 4]), 4),
        ]
        for args in sample_inputs:
            for loop_out, batched_out in get_fallback_and_vmap_exhaustive(F.one_hot, args, {}):
                self.assertEqual(loop_out, batched_out)

    def test_mode_key(self):
        def vmap_f(x):
            return x + torch.randn(())

        def naive_f(x, shape):
            return x + torch.randn(shape)

        torch.manual_seed(0)
        out1 = vmap(vmap(vmap_f))(torch.ones(2, 3))

        torch.manual_seed(0)
        out2 = naive_f(torch.ones(2, 3), (2, 3))
        self.assertEqual(out1, out2)

        torch.manual_seed(0)
        out1 = vmap(vmap(vmap_f))(torch.ones(2, 3, 4))

        torch.manual_seed(0)
        out2 = naive_f(torch.ones(2, 3, 4), (2, 3, 1))
        self.assertEqual(out1, out2)

        self.assertTrue(torch.randn(()).dim() == 0)

    @parameterized('op', {'abs': torch.abs, 'acos': torch.acos})
    def test_parameterize(self, op):
        pass

    @parameterized('op2', {'cos': torch.cos, 'cosh': torch.cosh})
    @parameterized('op1', {'sin': torch.sin, 'sinh': torch.sinh})
    def test_parameterize_multiple(self, op1, op2):
        pass

instantiate_parameterized_methods(TestVmapOperators)


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

def _get_rand_no_zeros(*args, **kwargs):
    requires_grad = kwargs.get('requires_grad', False)
    kwargs_without_requires_grad = kwargs.copy()
    kwargs_without_requires_grad['requires_grad'] = False
    result = torch.rand(*args, **kwargs_without_requires_grad)
    return result.clamp_min_(0.1).requires_grad_(requires_grad)

class TestVmapBatchedGradient(Namespace.TestVmapBase):
    def _vmap_test(self, *args, **kwargs):
        return _vmap_test(self, *args, **kwargs)

    # Tests batched gradient computation of outputs = op(*args, **kwargs)
    # by comparing it to a sequential map+stack fallback.
    #
    # output_process_fn: a function that maps the outputs to the part
    #       that should be differentiated.
    # batch_size: the batch dim size for the batched grad
    def _batched_grad_test(self, op, args, kwargs=None, output_process_fn=lambda x: x, batch_size=3):
        if kwargs is None:
            kwargs = {}
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
    def _batched_grad_grad_test(self, op, args, kwargs=None, output_process_fn=lambda x: x, batch_size=3):
        if kwargs is None:
            kwargs = {}
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

    def _test_arithmetic(self, op, device, test_grad_grad=True):
        x = torch.randn(2, 3, requires_grad=True, device=device)
        y = _get_rand_no_zeros(2, 3, device=device, requires_grad=True)
        scalar = 3.14
        self._batched_grad_test(op, (x, y))
        self._batched_grad_test(op, (scalar, y))
        self._batched_grad_test(op, (x, scalar))

        if test_grad_grad:
            self._batched_grad_grad_test(op, (x, y))

    def test_add(self, device):
        self._test_arithmetic(torch.add, device, test_grad_grad=False)
        self._test_arithmetic(lambda x, y: x + y, device, test_grad_grad=False)

    def test_sub(self, device):
        self._test_arithmetic(torch.sub, device, test_grad_grad=False)
        self._test_arithmetic(lambda x, y: x - y, device, test_grad_grad=False)

    def test_mul(self, device):
        self._test_arithmetic(torch.mul, device)
        self._test_arithmetic(lambda x, y: x * y, device)

    def test_div(self, device):
        self._test_arithmetic(torch.div, device)
        self._test_arithmetic(lambda x, y: x / y, device)

    @allowVmapFallbackUsage
    @unittest.expectedFailure
    def test_binary_cross_entropy(self, device):
        x = F.sigmoid(torch.randn(3, 2, device=device, requires_grad=True))
        target = torch.rand(3, 2, device=device)

        op = functools.partial(F.binary_cross_entropy, target=target)

        self._batched_grad_test(op, (x,), {})
        self._batched_grad_grad_test(op, (x,), {})

    def test_log_softmax(self, device):
        op = functools.partial(torch.log_softmax, dim=-1)
        x = torch.randn(3, 2, device=device, requires_grad=True)

        self._batched_grad_test(op, (x,), {})
        self._batched_grad_grad_test(op, (x,), {})

    def test_expand(self, device):
        x = torch.randn(2, 3, device=device, requires_grad=True)

        def op(x):
            return x.expand(5, 5, 2, 3)
        self._batched_grad_test(op, (x,))

    @allowVmapFallbackUsage
    def test_index(self, device):
        x = torch.randn(2, 3, requires_grad=True, device=device)
        index = torch.tensor([[0, 0], [1, 1]], device=device)

        def op(x):
            y = x * x
            return y[index]

        self._batched_grad_test(op, (x,))
        self._batched_grad_grad_test(op, (x,))

    def test_lgamma(self, device):
        x = torch.randn(2, 3, requires_grad=True, device=device)
        self._batched_grad_test(Tensor.lgamma, (x,))
        self._batched_grad_grad_test(Tensor.lgamma, (x,))

    def test_log(self, device):
        x = _get_rand_no_zeros(2, 3, device=device, requires_grad=True)
        self._batched_grad_test(torch.log, (x,))
        self._batched_grad_grad_test(torch.log, (x,))

    def test_logsumexp(self, device):
        x = _get_rand_no_zeros(2, 3, device=device, requires_grad=True)

        def op(x):
            return torch.logsumexp(x, -1)

        self._batched_grad_test(op, (x,))
        self._batched_grad_grad_test(op, (x,))

    def test_log1p(self, device):
        x = _get_rand_no_zeros(2, 3, device=device, requires_grad=True)
        self._batched_grad_test(torch.log1p, (x,))
        self._batched_grad_grad_test(torch.log1p, (x,))

    @parameterized_with_device('param', {'foo': None, 'bar': None})
    def test_param_device(self, device, param):
        pass

    @allowVmapFallbackUsage
    def test_max(self, device):
        x = torch.randn(2, 3, requires_grad=True, device=device)
        self._batched_grad_test(torch.max, (x,))

    @allowVmapFallbackUsage
    def test_median(self, device):
        x = torch.randn(2, 3, requires_grad=True, device=device)
        self._batched_grad_test(torch.median, (x,))

    @allowVmapFallbackUsage
    def test_min(self, device):
        x = torch.randn(2, 3, requires_grad=True, device=device)
        self._batched_grad_test(torch.min, (x,))

    def test_permute(self, device):
        x = torch.randn(2, 3, 5, requires_grad=True, device=device)

        def op(x):
            return x.permute(2, 0, 1)

        self._batched_grad_test(op, (x,))

    def test_reshape(self, device):
        x = torch.randn(2, 3, 5, requires_grad=True, device=device)

        def op(x):
            return x.reshape([2 * 3, 5])

        self._batched_grad_test(op, (x,))

    def test_sigmoid(self, device):
        x = torch.randn(2, 3, requires_grad=True, device=device)
        self._batched_grad_test(Tensor.sigmoid, (x,))
        self._batched_grad_grad_test(Tensor.sigmoid, (x,))

    def test_stack(self, device):
        x = torch.randn(2, 3, device=device, requires_grad=True)
        y = torch.randn(2, 3, device=device, requires_grad=True)

        def op(x, y):
            return torch.stack([x, y])
        self._batched_grad_test(op, (x, y))

    def test_select(self, device):
        x = torch.randn(2, 3, device=device, requires_grad=True)
        self._batched_grad_test(lambda x: x[1], (x,))
        self._batched_grad_test(lambda x: x.select(1, 2), (x,))
        self._batched_grad_test(lambda x: x.select(-1, 0), (x,))

    def test_slice(self, device):
        x = torch.randn(2, 3, 5, device=device, requires_grad=True)
        self._batched_grad_test(lambda x: x[0:1], (x,))
        self._batched_grad_test(lambda x: x[:, 1:3], (x,))
        self._batched_grad_test(lambda x: x[..., 1:3], (x,))

    def test_trace(self, device):
        x = torch.randn(2, 3, device=device, requires_grad=True)
        self._batched_grad_test(Tensor.trace, (x,))

    @skipCUDAIfNoMagma
    @allowVmapFallbackUsage
    def test_symeig(self, device):
        def op(x):
            return torch.symeig(x, eigenvectors=True)[0]

        x = torch.randn(3, 3, device=device, requires_grad=True)
        self._batched_grad_test(op, (x,), {})
        self._batched_grad_grad_test(op, (x,), {})

    def test_threshold(self, device):
        x = torch.randn(2, 3, device=device, requires_grad=True)
        self._batched_grad_test(lambda x: F.threshold(x, 0.5, 0.0), (x,))

    @allowVmapFallbackUsage
    def test_inplace_view(self, device):
        leaf = torch.randn(4, 5, requires_grad=True)

        def func(leaf):
            # Make sure the function is non-trivially twice differentiable
            base = leaf * leaf
            view = base[0]
            view.cos_()
            return view

        self._batched_grad_test(func, (leaf,), {})
        self._batched_grad_grad_test(func, (leaf,), {})

    @allowVmapFallbackUsage
    def test_inplace_manyview(self, device):
        leaf = torch.randn(4, 4, 5, requires_grad=True)

        def func(leaf):
            # Make sure the function is non-trivially twice differentiable
            base = leaf * leaf
            view = base.transpose(0, 2)
            view = view[1]
            view = view.diagonal()
            view = view[::2]
            view.cos_()
            return view

        self._batched_grad_test(func, (leaf,), {})
        self._batched_grad_grad_test(func, (leaf,), {})

    def test_diagonal(self, device):
        x = torch.randn(4, 5, device=device, requires_grad=True)
        self._batched_grad_test(lambda x: x.diagonal(1, 0, 1), (x,))

        x = torch.randn(3, 4, 5, device=device, requires_grad=True)
        self._batched_grad_test(lambda x: x.diagonal(0, -1, -2), (x,))

    @allowVmapFallbackUsage
    def test_unrelated_output(self, device):
        B0 = 3
        x = torch.randn([], requires_grad=True)
        y = torch.randn([], requires_grad=True)
        gy = torch.randn(B0, requires_grad=True)

        def vjp(v):
            res, = torch.autograd.grad(y, x, v, allow_unused=True)
            return torch.zeros_like(x) if res is None else res

        result = vmap(vjp)(gy)
        self.assertEqual(result, torch.zeros(B0, *x.shape, device=device))

    @allowVmapFallbackUsage
    def test_unrelated_output_multiple_grad(self, device):
        B0 = 3
        x = torch.randn([], requires_grad=True)
        y = torch.randn([], requires_grad=True)
        gy = torch.randn(B0, requires_grad=True)

        def vjp(v):
            res, = torch.autograd.grad(y, x, v, allow_unused=True)
            return torch.zeros_like(x) if res is None else res

        _ = vjp(gy[0])
        result = vmap(vjp)(gy)
        self.assertEqual(result, torch.zeros(B0, *x.shape, device=device))

class TestVmapOperatorsOpInfo(TestCase):
    @ops(functorch_lagging_op_db + additional_op_db, allowed_dtypes=(torch.float,))
    @skipOps('TestVmapOperatorsOpInfo', 'test_vmap_exhaustive', {
        # These are ops that we can't generate fallbacks for
        xfail('broadcast_to'),
        xfail('dsplit'),
        xfail('fill_'),
        xfail('gradient'),
        xfail('hsplit'),
        xfail('nn.functional.pad', 'circular'),
        xfail('ravel'),
        xfail('resize_as_'),
        xfail('resolve_conj'),
        xfail('resolve_neg'),
        xfail('tensor_split'),
        xfail('to_sparse'),
        xfail('vsplit'),
        xfail('hstack'),
        xfail('vstack'),
        xfail('dstack'),
        xfail('linalg.multi_dot'),
        xfail('nanmean'),
        xfail('block_diag'),
        xfail('nn.functional.dropout'),

        # entries in here need don't work and need to be fixed.
        # Each one of these is a bug
        xfail('unfold'),
        xfail('svd', device_type='cuda'),
        xfail('linalg.svd', device_type='cuda'),
        xfail('index_put'),
        xfail('nn.functional.max_pool2d'),
    })
    def test_vmap_exhaustive(self, device, dtype, op):
        sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=False)
        for sample_input in sample_inputs_itr:
            arg_values = [sample_input.input] + list(sample_input.args)
            kwarg_values = sample_input.kwargs
            try:
                for loop_out, batched_out in get_fallback_and_vmap_exhaustive(op.op, arg_values, kwarg_values):
                    self.assertEqual(loop_out, batched_out, atol=1e-4, rtol=1e-4)
            except Exception as e:
                # Checking if we're throwing an error because of dynamic shapes.
                if "dynamic" in e.args[0]:
                    continue
                raise e

    @ops(functorch_lagging_op_db + additional_op_db, allowed_dtypes=(torch.float,))
    @skipOps('TestVmapOperatorsOpInfo', 'test_op_has_batch_rule', {
        # xfail('__getitem__'),
        xfail('aminmax'),
        xfail('broadcast_to'),
        xfail('cdist'),
        xfail('complex'),
        xfail('copysign'),
        xfail('cross'),
        xfail('diag_embed'),
        xfail('dsplit'),
        xfail('eig'),
        xfail('fft.fftn'),
        xfail('fft.hfft'),
        xfail('fft.ifftn'),
        xfail('fill_'),
        xfail('gradient'),
        xfail('histogram'),
        xfail('hsplit'),
        xfail('index_add'),
        xfail('index_copy'),
        xfail('index_fill'),
        xfail('index_put'),
        xfail('index_select'),
        xfail('isin'),
        xfail('linalg.cholesky'),
        xfail('linalg.eigvals'),
        xfail('linalg.eigvalsh'),
        xfail('linalg.householder_product'),
        xfail('linalg.inv'),
        xfail('linalg.lstsq'),
        xfail('linalg.matrix_norm'),
        xfail('linalg.matrix_power'),
        xfail('linalg.matrix_rank'),
        xfail('linalg.matrix_rank', 'hermitian'),
        xfail('linalg.norm'),
        xfail('linalg.solve'),
        xfail('linalg.svdvals'),
        xfail('linalg.tensorinv'),
        xfail('lu'),
        xfail('lu_solve'),
        xfail('lu_unpack'),
        xfail('masked_fill'),
        xfail('masked_scatter'),
        xfail('masked_select'),
        xfail('nanquantile'),
        xfail('nn.functional.conv_transpose2d'),
        xfail('nn.functional.cross_entropy', 'mean'),
        xfail('nn.functional.cross_entropy', 'none'),
        xfail('nn.functional.cross_entropy', 'sum'),
        xfail('nn.functional.interpolate', 'area'),
        xfail('nn.functional.pad', 'circular'),
        xfail('nn.functional.unfold'),
        xfail('norm', 'fro'),
        xfail('norm', 'nuc'),
        xfail('ormqr'),
        xfail('put'),
        xfail('quantile'),
        xfail('ravel'),
        xfail('renorm'),
        xfail('resize_as_'),
        xfail('resolve_conj'),
        xfail('resolve_neg'),
        xfail('roll'),
        xfail('rot90'),
        xfail('scatter'),
        xfail('scatter_add'),
        xfail('take'),
        xfail('take_along_dim'),
        xfail('tensor_split'),
        xfail('to_sparse'),
        xfail('unfold'),
        xfail('vdot'),
        xfail('vsplit'),
        xfail('__getitem__'),
        xfail('all'),
        xfail('any'),
        xfail('count_nonzero'),
        xfail('dstack'),
        xfail('hstack'),
        xfail('linalg.multi_dot'),
        xfail('nanmean'),
        xfail('nn.functional.cosine_similarity'),
        xfail('nn.functional.layer_norm'),
        xfail('nn.functional.nll_loss'),
        xfail('vstack'),
        xfail('block_diag'),
        xfail('nn.functional.dropout'),
        xfail('nn.functional.max_pool2d'),
    })
    def test_op_has_batch_rule(self, device, dtype, op):
        def test():
            sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=False)
            for sample_input in sample_inputs_itr:
                arg_values = [sample_input.input] + list(sample_input.args)
                kwarg_values = sample_input.kwargs
                for _ in get_fallback_and_vmap_exhaustive(op.op, arg_values, kwarg_values, compute_loop_out=False):
                    pass
        check_vmap_fallback(self, test, op)

    def test_isnan(self, device):
        test = functools.partial(_vmap_test, check_propagates_grad=False)

        B, N, C, H, W = 2, 3, 24, 5, 7
        op = torch.isnan

        x = torch.randn(B, N, C, H, W)
        x[x > 0] = float('nan')
        test(self, op, (x,), in_dims=(0))

    def test_isinf(self, device):
        test = functools.partial(_vmap_test, check_propagates_grad=False)

        B, N, C, H, W = 2, 3, 24, 5, 7
        op = torch.isinf

        x = torch.randn(B, N, C, H, W)
        x[x > 0] = float('inf')
        test(self, op, (x,), in_dims=(0))

    def test_foo_like(self, device):
        test = functools.partial(_vmap_test, check_propagates_grad=False)

        B, N, C, H, W = 2, 3, 24, 5, 7
        for op in [torch.ones_like, torch.zeros_like, torch.randn_like, torch.rand_like]:
            x = torch.randn(B, N, C, H, W)
            # todo(chilli): test these better
            # Not testing correctness, just that they run
            vmap(op, in_dims=(0,))(x,)

    def test_flatten(self, device):
        test = functools.partial(_vmap_test, check_propagates_grad=False)

        op = torch.flatten

        x = torch.randn(2, 3, 4, 5)
        test(self, op, (x, 1, 2), in_dims=(0, None, None))

    def test_group_norm(self, device):
        test = functools.partial(_vmap_test, check_propagates_grad=False)

        B, N, C, H, W = 2, 3, 24, 5, 7
        op = F.group_norm

        x = torch.randn(B, N, C, H, W)
        weight = torch.randn(C)
        bias = torch.randn(C)
        test(self, op, (x, 3, weight, bias), in_dims=(0, None, None, None))

        x = torch.randn(B, N, C, H, W)
        weight = torch.randn(B, C)
        bias = torch.randn(B, C)
        test(self, op, (x, 4, weight, bias), in_dims=(0, None, 0, 0))

    def test_index_put(self, device):
        test = functools.partial(_vmap_test, check_propagates_grad=False)

        x = torch.arange(3*4*5).reshape(3,4,5)
        def f(x, y, z):
            x[y] = z
            return x
        x = torch.randn(3, 4, 5)
        y = torch.zeros((3, 2)).long()
        z = torch.randn(3, 2, 5)
        base = f(x[0], y[0], z[0])
        self.assertEqual(vmap(f, in_dims=(0,0,0))(x, y, z)[0], base)
        self.assertEqual(vmap(f, in_dims=(0,None,None))(x, y[0], z[0])[0], base)
        self.assertEqual(vmap(f, in_dims=(0,None,0))(x, y[0], z)[0], base)
        self.assertEqual(vmap(f, in_dims=(0,0,None))(x, y, z[0])[0], base)

    @parameterized_with_device('training', {'train': True, 'eval': False})
    @parameterized_with_device('track_running_stats', {'running_stats1': True, 'running_stats0': False})
    @parameterized_with_device('affine', {'affine1': True, 'affine0': False})
    def test_batch_norm(self, device, affine, track_running_stats, training):
        if not track_running_stats and not training:
            return

        test = functools.partial(_vmap_test, check_propagates_grad=False)
        BN = torch.nn.BatchNorm2d
        ensemble_size = 10
        hidden_dim = 3

        weights, buffers, _, _, _ = \
            functional_init_with_buffers(BN, [ensemble_size])(
                hidden_dim, affine=affine, track_running_stats=track_running_stats)

        inputs = [torch.randn(ensemble_size, 32, hidden_dim, 16, 16, device=device)]
        in_dims = [0]

        def append(inp, in_dim):
            inputs.append(inp)
            in_dims.append(in_dim)

        if track_running_stats:
            running_mean, running_var, _ = buffers
            append(running_mean.to(device), 0)
            append(running_var.to(device), 0)
        else:
            append(None, None)
            append(None, None)

        if affine:
            weight, bias = weights
            append(weight.to(device), 0)
            append(bias.to(device), 0)
        else:
            append(None, None)
            append(None, None)

        append(training, None)

        def op(inp, running_mean, running_var, weight, bias, training):
            res = F.batch_norm(inp, running_mean, running_var, weight, bias, training)
            if track_running_stats:
                return res, running_mean, running_var
            return res

        test(self, op, tuple(inputs), in_dims=tuple(in_dims))


only_for = ("cpu", "cuda")
instantiate_parameterized_methods(TestVmapOperatorsOpInfo)
instantiate_device_type_tests(TestVmapOperatorsOpInfo, globals(), only_for=only_for)

instantiate_parameterized_methods(TestVmapBatchedGradient)
instantiate_device_type_tests(
    TestVmapBatchedGradient,
    globals(),
    only_for=only_for,
)

if __name__ == '__main__':
    run_tests()
