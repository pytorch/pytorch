# Owner(s): ["module: functorch"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import functools
import itertools
import os
import random
import types
import unittest
import warnings
from collections import namedtuple, OrderedDict
from unittest.case import skipIf

from common_utils import (
    check_vmap_fallback,
    compute_quantities_for_vmap_test,
    decorate,
    DisableVmapFallback,
    generate_vmap_inputs,
    get_fallback_and_vmap_exhaustive,
    is_batch_norm_training,
    is_valid_inplace_sample_input,
    opsToleranceOverride,
    skip,
    skipOps,
    tol1,
    xfail,
    xfailIf,
)
from functorch_additional_op_db import additional_op_db

import functorch
import torch
import torch.nn.functional as F
from functorch import grad, grad_and_value, jacfwd, jvp, vjp, vmap
from functorch.experimental import chunk_vmap
from torch import Tensor
from torch._C._functorch import reshape_dim_into, reshape_dim_outof
from torch._functorch.make_functional import functional_init_with_buffers
from torch._functorch.vmap import restore_vmap
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.testing._internal.autograd_function_db import autograd_function_db
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_CUDNN_ATTENTION,
    PLATFORM_SUPPORTS_FLASH_ATTENTION,
    PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
    tf32_on_and_off,
    with_tf32_off,
)
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
    OpDTypes,
    ops,
    tol,
    toleranceOverride,
)
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_WINDOWS,
    markDynamoStrictTest,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    subtest,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
    unMarkDynamoStrictTest,
    xfailIfTorchDynamo,
)
from torch.testing._internal.custom_op_db import custom_op_db
from torch.utils import _pytree as pytree


def get_platform_specific_sdpa():
    ret = [SDPBackend.MATH]
    if PLATFORM_SUPPORTS_FLASH_ATTENTION:
        ret.append(SDPBackend.FLASH_ATTENTION)
    if PLATFORM_SUPPORTS_MEM_EFF_ATTENTION:
        ret.append(SDPBackend.EFFICIENT_ATTENTION)
    if PLATFORM_SUPPORTS_CUDNN_ATTENTION:
        ret.append(SDPBackend.CUDNN_ATTENTION)
    return ret


PLATFORM_SPECIFIC_SDPA = get_platform_specific_sdpa()

FALLBACK_REGEX = "There is a performance drop"


class EnableVmapFallbackWarnings:
    def __enter__(self):
        self.prev_state = torch._C._debug_only_are_vmap_fallback_warnings_enabled()
        torch._C._debug_only_display_vmap_fallback_warnings(True)

    def __exit__(self, *ignored):
        torch._C._debug_only_display_vmap_fallback_warnings(self.prev_state)


@markDynamoStrictTest
class TestVmapAPI(TestCase):
    def test_non_tensor_output_raises(self):
        with self.assertRaisesRegex(ValueError, "got type <class 'float'>"):
            vmap(lambda x: 3.14)(torch.ones(3))

        def multiple_outputs(x):
            return x, 3

        with self.assertRaisesRegex(ValueError, "got type <class 'int'>"):
            vmap(multiple_outputs)(torch.ones(3))

    def test_different_map_dim_size_raises(self):
        x = torch.randn(2)
        y = torch.randn(3)
        expected_msg = (
            "Expected all tensors to have the same size in the mapped dimension"
        )
        with self.assertRaisesRegex(ValueError, expected_msg):
            vmap(torch.mul)(x, y)
        with self.assertRaisesRegex(ValueError, expected_msg):
            vmap(lambda z: z[0] + z[1], in_dims=((0, 0),))((x, y))
        with self.assertRaisesRegex(ValueError, expected_msg):
            vmap(lambda z: z["x"] + z["y"], in_dims=({"x": 0, "y": 0},))(
                {"x": x, "y": y}
            )

    def test_func_with_no_inputs(self):
        expected_msg = "got no inputs"

        def foo():
            return torch.randn(3)

        def bar(x):
            return torch.randn(3)

        with self.assertRaisesRegex(ValueError, expected_msg):
            vmap(foo)()

        with self.assertRaisesRegex(ValueError, expected_msg):
            vmap(bar)()

    def test_func_with_no_tensors(self):
        def foo(x):
            return torch.randn(3)

        with self.assertRaisesRegex(ValueError, "at least one Tensor"):
            vmap(foo, (None,))(1)

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

    def test_nested_with_diag_embed(self):
        # diag_embed requires special testing because it is registered with conditional functionalization.
        x = torch.randn(3, 3, 5)
        output = vmap(vmap(torch.diag_embed))(x)
        self.assertEqual(output, torch.diag_embed(x))

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

    def test_checkpoint(self):
        A = torch.randn((3, 8, 8), dtype=torch.float64, requires_grad=True)

        def get_grad(checkpoint):
            A.grad = None

            def get_loss(A):
                ortho_A, _ = torch.func.vmap(torch.linalg.qr)(A)
                return torch.sum(ortho_A)

            if checkpoint:
                loss = torch.utils.checkpoint.checkpoint(
                    get_loss, A, use_reentrant=False
                )
            else:
                loss = get_loss(A)
            loss.backward()
            return A.grad

        expected = get_grad(checkpoint=False)
        result = get_grad(checkpoint=True)
        self.assertEqual(result, expected)

    def test_unsupported_op_err_msg(self):
        # Unsupported view op
        tensor = torch.randn(2, 3)
        msg = (
            r"Batching rule not implemented for aten::.+; the "
            r"fallback path doesn't work on out= or view ops"
        )
        # TODO: find a view op
        # with self.assertRaisesRegex(RuntimeError, msg):
        #     vmap(torch.ravel)(tensor)

        def out_op(x, y):
            return torch.abs(x, out=y)

        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(out_op)(tensor, tensor)

        # Don't support non-tensor returns. This is a limitation of vmap;
        # functions that don't return tensors must be special cased
        with self.assertRaisesRegex(RuntimeError, "Batching rule not implemented"):
            vmap(torch.equal)(tensor, tensor)

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
        self.assertEqual(
            result, (tensor.permute(1, 2, 0, 3), other.permute(1, 2, 0, 3))
        )

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
            (
                x.permute(1, 0, 2),
                (x * y).permute(1, 0, 2),
                (x * y * y).permute(1, 0, 2),
            ),
        )

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

    def test_out_dims_none_tuple(self):
        def foo(x):
            return x, "hello world"

        tensor = torch.randn(2, 3)
        result = vmap(foo, out_dims=(0, None))(tensor)
        self.assertEqual(result[1], "hello world")
        self.assertEqual(result[0], tensor)

        def foo(x):
            x.add_(1)
            return None, "hello world"

        result = vmap(foo, out_dims=(None, None))(tensor)
        self.assertEqual(result, (None, "hello world"))

    def test_out_dims_none(self):
        def foo(x):
            return x

        tensor = torch.randn(2, 3)
        with self.assertRaisesRegex(
            ValueError, "can not return a BatchedTensor when out_dim is None"
        ):
            vmap(foo, out_dims=None)(tensor)

        def foo(x):
            x.add_(1)
            return "hello world"

        result = vmap(foo, out_dims=None)(tensor)
        self.assertEqual(result, "hello world")

    def test_out_dims_normal_tensor(self):
        def foo(x):
            return torch.arange(3)

        tensor = torch.randn(2, 3)
        result = vmap(foo)(tensor)
        self.assertEqual(result.shape, [2, 3])

        result = vmap(foo, out_dims=None)(tensor)
        self.assertEqual(result, torch.arange(3))

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

    def test_pytree_odict_returns(self):
        x = torch.randn(2, 3)

        def f(t):
            y = t.sin()
            return OrderedDict([("sin", y), ("cos", t.cos())])

        out = vmap(f)(x)
        assert isinstance(out, OrderedDict)
        expected = f(x)
        self.assertEqual(out["sin"], expected["sin"])
        self.assertEqual(out["cos"], expected["cos"])

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
        msg = "must be an int, None or a python collection of ints"
        tensor = torch.randn(2, 3)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: x, out_dims="lol")(tensor)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: x, out_dims=("lol",))(tensor)

    def test_out_dims_and_num_outputs_mismatch_err_msg(self):
        msg = "not compatible"
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
        msg = "Dimension out of range"
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

    def test_nested_negative_in_dims(self):
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        output = vmap(torch.mul, (-1, -1))(x, y)
        self.assertEqual(output.shape, (3, 2))
        self.assertEqual(output, (x * y).permute(1, 0))

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

    def test_item_throws(self):
        def f(x):
            return x.item()

        with self.assertRaisesRegex(RuntimeError, r"item\(\) on a Tensor"):
            vmap(f)(torch.randn(3))

    def test_data_dependent_control_flow_throws(self):
        def f(x):
            if x:
                return x
            return 0

        with self.assertRaisesRegex(RuntimeError, r"data-dependent control flow"):
            vmap(f)(torch.randn(3))

    def test_accepts_nested_inputs(self):
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

        out = vmap(lambda z: z["x"] + z["y"])({"x": x, "y": y})
        self.assertEqual(out, x + y)
        out = vmap(lambda z: z["x"] + z["y"], in_dims=(0,))({"x": x, "y": y})
        self.assertEqual(out, x + y)
        out = vmap(lambda z: z["x"] + z["y"], in_dims=({"x": 0, "y": 0},))(
            {"x": x, "y": y}
        )
        self.assertEqual(out, x + y)

        # Multiple layers of nesting
        out_fn = vmap(lambda z: z["x"][0] + z["x"][1][0] + z["y"][0] + z["y"][1])
        out = out_fn({"x": [x, (x,)], "y": [y, y]})
        self.assertEqual(out, x + x + y + y)

    def test_in_dims_wrong_type_err_msg(self):
        x = torch.randn(3)
        y = torch.randn(3)
        msg = r"expected `in_dims` to be int or a \(potentially nested\) tuple"
        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.mul, [0, 0])(x, y)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.mul, set({0}))(x, y)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.mul, "lol")(x, y)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda z: z[0] + z[1], in_dims=[0, 0])([x, y])
        # The following should not throw
        vmap(torch.mul, (0, 0))(x, y)

    def test_not_enough_in_dims_err_msg(self):
        x = torch.randn(3)
        y = torch.randn(3)
        msg = r"in_dims is not compatible with the structure of `inputs`"

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
        # noqa: F841

        def foo(xy):
            return xy[0] * xy[1]

        def bar(x, yz):
            return x * yz[0] * yz[1]

        x = torch.randn(2, 3)

        # the following are errors in jax (and will always be errors)
        msg = "Got in_dim=0 for an input but the input is of type"
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

        msg = r"Got in_dim=-?\w for some input, but that input is a Tensor of dimensionality \w"
        with self.assertRaisesRegex(ValueError, msg):
            vmap(foo)(torch.randn([]))
        with self.assertRaisesRegex(ValueError, msg):
            vmap(foo, in_dims=(0,))(torch.randn([]))
        with self.assertRaisesRegex(ValueError, msg):
            vmap(foo, in_dims=(-3,))(x)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(foo, in_dims=(2,))(y)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda z: z[0] + z[1], in_dims=([3, 0],))([x, y])
        # the following should not throw
        vmap(foo, in_dims=(0,))(torch.randn(2, 3))
        vmap(foo, in_dims=(1,))(torch.randn(2, 3))

    def test_fallback_does_not_warn_by_default(self):
        op = torch._test_functorch_fallback
        x = torch.randn(11)
        y = torch.randn(11)
        with warnings.catch_warnings(record=True) as wa:
            torch.vmap(op)(x, y)
            # The single warning here is the "vmap is experimental"
            # warning, not a warning from the vmap fallback path.
            self.assertEqual(len(wa), 1)

    @unittest.expectedFailure
    def test_fallback_warns_when_warnings_are_enabled(self):
        # NB: One day we will implement a batching rule for torch.atan2.
        # If/when we do, this test should be replaced to test the fallback
        # path on another operator to avoid bitrot.
        op = torch._test_functorch_fallback
        x = torch.randn(11)
        y = torch.randn(11)
        with warnings.catch_warnings(record=True) as wa:
            with EnableVmapFallbackWarnings():
                torch.vmap(op)(x, y)
            self.assertEqual(len(wa), 2)
            self.assertRegex(str(wa[-1].message), FALLBACK_REGEX)

    def _assert_uses_vmap_fallback(self, vmap_args, inputs):
        return
        # with warnings.catch_warnings(record=True) as wa:
        #     with EnableVmapFallbackWarnings():
        #         result = vmap(*vmap_args)(*inputs)
        #     self.assertEqual(len(wa), 2)
        #     self.assertRegex(str(wa[-1].message), FALLBACK_REGEX)

    def test_fallback_zero_dim(self):
        op = torch._test_functorch_fallback
        x = torch.randn(11)
        y = torch.randn(11)
        self._assert_uses_vmap_fallback((op,), (x, y))

        B0, B1 = 0, 3
        x = torch.randn(B0, 11)
        y = torch.randn(11)

        msg = "The fallback path does not support vmap over dims of size 0"

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

    def test_fallback_warning(self):
        # We use a dummy function _test_functorch_fallback
        # defined in prim_native_functions.cpp for this
        op = torch._test_functorch_fallback

        x = torch.randn(5, 7, 11)
        y = torch.randn(5, 7, 11)

        self._assert_uses_vmap_fallback((op,), (x, y))

        x = torch.randn(7, 11, 5)
        y = torch.randn(5, 7, 11)
        result = vmap(op, (2, 0))(x, y)
        self.assertEqual(result, op(x.permute(2, 0, 1), y))

        # nested vmap
        x = torch.randn(7, 11, 5)
        y = torch.randn(5, 7, 11)
        result = vmap(vmap(op), (2, 0))(x, y)
        self.assertEqual(result, op(x.permute(2, 0, 1), y))

        # big batch size (total 10000)
        x = torch.randn(100, 10, 10, 5)
        y = torch.randn(100, 10, 10)
        result = vmap(vmap(vmap(op)))(x, y)
        self.assertEqual(result, op(x, y.view(100, 10, 10, 1)))

    # TODO: No clue what is wrong here.
    @unittest.skip
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

            self._assert_uses_vmap_fallback(
                (torch.index_add, (0, None, None, 0)), (x, dim, index, values)
            )

            result = vmap(torch.index_add, (0, None, None, 0))(x, dim, index, values)
            expected = torch.index_add(x, dim + 1, index, values.view(B0, 3, 1, 13))
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
        vmap(vmap(vmap(op)))(x, y)
        self.assertEqual(x, outplace_op(x_orig, y.view(B0, B1, B2, 1)))

    # ("Fallback isInplaceVmapCompatible check is broken")
    @unittest.expectedFailure
    def test_inplace_fallback_nary_different_levels(self):
        # NB: One day we will implement a batching rule for atan2_
        # If/when we do, this test should be replaced to test the fallback
        # path on another operator to avoid bitrot.
        op = Tensor.atan2_
        outplace_op = torch.atan2
        B0, B1 = 2, 3

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
        msg = r"vmap: aten::atan2_\(self, \*extra_args\) is not possible"
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
        err_msg = r"backward\(\) called inside a functorch transform"

        def backward_on_vmapped_tensor(x):
            x.sum().backward()

        # FIXME
        return self.skipTest(
            "error: element 0 of tensors does not require grad and does not have a grad_fn"
        )
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
        err_msg = "autograd.grad.* called inside torch.vmap"

        captured = torch.randn(3, requires_grad=True)

        def output_to_grad_is_vmapped(input_tensor):
            output = (captured * input_tensor).sum()
            return torch.autograd.grad([output], [captured])[0]

        with self.assertRaisesRegex(RuntimeError, err_msg):
            vmap(output_to_grad_is_vmapped)(input_tensor)

        output = (input_tensor**2).sum()

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
            (grad_x,) = torch.autograd.grad(result, x, v)
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

        # Case: `0` sized dim.
        x = torch.randn(12, 12, 0)
        y = reshape_dim_outof(-1, 6, x)
        self.assertEqual(y.shape, torch.Size((12, 12, 6, 0)))

    def test_batch_rule_does_not_need_to_handle_no_batched_input(self):
        def f(x, y):
            res = torch.dot(y, torch.ones(2))
            return x + res

        x = torch.randn(7, 5)
        y = torch.randn(3, 2)
        out = vmap(vmap(f, in_dims=(0, None)), in_dims=(None, 0))(x, y)
        expected = torch.mv(y, torch.ones(2)).view(3, 1, 1) + x
        self.assertEqual(out, expected)

    def test_decomposition_under_python_dispatcher(self):
        # This test will raise an error if the vmap fallback gets invoked.
        # Here we test that decomps registered to FuncTorchBatchedDecomposition
        # are respected by the Python Dispatcher.
        t = torch.ones(3, 3) * 5
        with DisableVmapFallback():
            with torch._dispatch.python.enable_python_dispatcher():
                o = torch.vmap(torch.square)(t)
        self.assertEqual(o, torch.square(t))

    def _test_vmap_autocast(self, device):
        if torch.device(device).type == "cpu":
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16

        a_float32 = torch.rand(4, 2, 3, device=device)
        b_float32 = torch.rand(4, 3, 2, device=device)
        c_float32 = torch.rand(4, 2, 2, device=device)
        d_float32 = torch.rand(4, 3, 2, device=device)

        # Case 1, autocast inside vmapped function
        def func1(x, y, z, w):
            with torch.autocast(dtype=amp_dtype, device_type=device):
                e_float16 = torch.matmul(x, y)
                assert e_float16.dtype == amp_dtype, e_float16.dtype
                f_float16 = torch.matmul(z, e_float16)
                assert f_float16.dtype == amp_dtype, f_float16.dtype
            return torch.matmul(w, f_float16.float())

        expected = func1(a_float32, b_float32, c_float32, d_float32)
        out = vmap(func1)(a_float32, b_float32, c_float32, d_float32)
        assert expected.allclose(out)

        # Case 2, autocast decorator inside vmapped function
        @torch.autocast(dtype=amp_dtype, device_type=device)
        def func2(x, y, z, w):
            e_float16 = torch.matmul(x, y)
            assert e_float16.dtype == amp_dtype, e_float16.dtype
            f_float16 = torch.matmul(z, e_float16)
            assert f_float16.dtype == amp_dtype, f_float16.dtype
            return torch.matmul(w, f_float16)

        expected = func2(a_float32, b_float32, c_float32, d_float32)
        out = vmap(func2)(a_float32, b_float32, c_float32, d_float32)
        assert expected.allclose(out)

        # Case 3, autocast is outside vmapped function
        def func3(x, y, z, w):
            e_float16 = torch.matmul(x, y)
            assert e_float16.dtype == amp_dtype, e_float16.dtype
            f_float16 = torch.matmul(z, e_float16)
            assert f_float16.dtype == amp_dtype, f_float16.dtype
            return torch.matmul(w, f_float16)

        with torch.autocast(dtype=amp_dtype, device_type=device):
            expected = func3(a_float32, b_float32, c_float32, d_float32)
            out = vmap(func3)(a_float32, b_float32, c_float32, d_float32)

        assert expected.allclose(out)

    @unittest.skip("Somehow, vmap and autocast do not work on CPU")
    def test_vmap_autocast_cpu(self):
        self._test_vmap_autocast("cpu")

    @skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    def test_vmap_autocast_cuda(self):
        self._test_vmap_autocast("cuda")

    def test_restore_vmap_pytree_input_output(self):
        def f(x, y):
            output0 = x[0] + x[1]
            output1 = y
            return {"a": output0, "b": output1}

        B = 2
        x0 = torch.randn(B, 3)
        x1 = torch.randn(B)
        y = torch.randn(4, B)

        out, out_dims = restore_vmap(f, ((0, 0), 1), B, "error")((x0, x1), y)
        expected = vmap(f, in_dims=((0, 0), 1), out_dims={"a": 0, "b": 1})((x0, x1), y)
        self.assertEqual(out, expected)
        self.assertEqual(out_dims, {"a": 0, "b": 1})

    def test_restore_vmap_no_vmapped_inputs(self):
        def f(x, y, z):
            return x, y * z, z

        B = 2
        # Mix of tensor and non-tensor inputs
        x = torch.randn(3)
        y = torch.randn(4)
        z = 5
        out, out_dims = restore_vmap(f, (None, None, None), B, "error")(x, y, z)
        self.assertEqual(out, f(x, y, z))
        self.assertEqual(out_dims, (None, None, None))

    def test_restore_vmap_unexpanded_outputs(self):
        def f(x, y):
            # Mix of tensor and non-tensor outputs
            return 3 * y, y.sum(), None

        B = 2
        x = torch.randn(B, 3)
        y = torch.randn(4)
        out, out_dims = restore_vmap(f, (0, None), B, "error")(x, y)
        self.assertEqual(out, f(None, y))
        self.assertEqual(out_dims, (None, None, None))

    def test_data_attribute(self):
        def foo(x):
            y = x.data  # noqa: F841
            return x

        with self.assertRaisesRegex(
            RuntimeError, "accessing `data` under vmap transform"
        ):
            torch.func.vmap(foo)(torch.randn(3, 3))

        def foo(x):
            x.data = torch.ones(3, 3)
            return x

        with self.assertRaisesRegex(
            RuntimeError, "mutating directly with `.data` under vmap"
        ):
            torch.func.vmap(foo)(torch.randn(3, 3))


def slice_inputs(inputs, bdims, i):
    result = []
    for inp, bdim in zip(inputs, bdims):
        if bdim is None:
            result.append(inp)
        else:
            result.append(inp.select(bdim, i))
    return tuple(result)


def reference_vmap(op, inputs, in_dims=0, out_dims=0, return_nt=False):
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
        if return_nt:
            return torch.nested.nested_tensor(list(results))
        else:
            return torch.stack(results, dim=out_dims[0])

    assert all(isinstance(result, tuple) for result in results)
    num_returns = len(results[0])
    assert all(len(result) == num_returns for result in results)
    if isinstance(out_dims, int):
        out_dims = (out_dims,) * num_returns
    if return_nt:
        return tuple(
            torch.nested.nested_tensor(list(result_shards))
            for result_shards in zip(*results)
        )
    else:
        return tuple(
            torch.stack(result_shards, out_dim)
            for result_shards, out_dim in zip(zip(*results), out_dims)
        )


class TensorFactory:
    @staticmethod
    def rand(size, device="cpu", dtype=torch.float):
        return torch.rand(size, device=device, dtype=dtype)

    @staticmethod
    def randn(size, device="cpu", dtype=torch.float):
        return torch.randn(size, device=device, dtype=dtype)

    @staticmethod
    def randp1(size, device="cpu", dtype=torch.float):
        return torch.rand(size, device=device, dtype=dtype) + 1


# Tests vmap(op, in_dims, out_dims)(*inputs) by comparing the output to a
# (slow) sequential map+stack fallback.
#
# check_view: Test if the first returned output is a view of the first input
# check_propagates_grad: Test if the operation propagates gradients.


def _vmap_test(
    self,
    op,
    inputs,
    in_dims=0,
    out_dims=0,
    check_view=False,
    check_propagates_grad=True,
):
    result = vmap(op, in_dims, out_dims)(*inputs)
    are_nested = [t.is_nested for t in pytree.tree_leaves(result)]
    reference_result = reference_vmap(
        op, inputs, in_dims, out_dims, return_nt=any(are_nested)
    )
    self.assertEqual(result, reference_result)
    op_has_single_return = not isinstance(result, tuple)

    if check_view:
        result_as_tuple = (result,) if op_has_single_return else result
        for output in result_as_tuple:
            input0_base = inputs[0] if inputs[0]._base is None else inputs[0]._base
            self.assertTrue(
                output._base is input0_base,
                msg="result was not a view of the first input!",
            )

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
    return getattr(fn, "_allow_vmap_fallback_usage", False)


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
        def __init__(self, method_name="runTest"):
            super().__init__(method_name)

            test_method = getattr(self, method_name, None)
            if test_method is None:
                return

            if not should_allow_vmap_fallback_usage(test_method):
                setattr(
                    self,
                    method_name,
                    self._wrap_method_with_vmap_fallback_check(test_method),
                )

        def _wrap_method_with_vmap_fallback_check(self, method):
            # msg = (
            #     'Expected the test to not invoke the vmap fallback path, i.e., '
            #     'all of the operators being tested in this test should have batching '
            #     'rules implemented. If you are intentionally testing something to '
            #     'do with the fallback path, use allowVmapFallbackUsage. Otherwise, '
            #     'please make sure that batching rules are implemented for the '
            #     'operator(s) being tested.'
            # )

            @functools.wraps(method)
            def wrapper(self, *args, **kwargs):
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always")
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


@markDynamoStrictTest
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
        test(
            vmap(op, in_dims=2),
            [getter([2, 5, B0, B1, 3], device)],
            in_dims=2,
            out_dims=2,
        )

    @parametrize(
        "case",
        [
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
        ],
        name_fn=lambda x: x[0].__name__,
    )
    def test_unary_pointwise(self, case):
        op, getter = case
        self._test_unary(op, getter, "cpu")

        # test in-place
        method = getattr(Tensor, f'{op.__name__ + "_"}')
        self._test_unary(method, getter, "cpu", check_propagates_grad=False)

    def test_clone(self):
        # Some basic tests
        self._test_unary(lambda x: x.clone(), TensorFactory.randn, "cpu")
        self._test_unary(
            lambda x: x.clone(memory_format=torch.preserve_format),
            TensorFactory.randn,
            "cpu",
        )
        self._test_unary(
            lambda x: x.clone(memory_format=torch.contiguous_format),
            TensorFactory.randn,
            "cpu",
        )

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

        msg = r"only supported with memory_format torch.preserve_format or torch.contiguous_format"
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(lambda x: x.clone(memory_format=torch.channels_last))(torch.randn(B0))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(lambda x: x.clone(memory_format=torch.channels_last_3d))(
                torch.randn(B0)
            )

    def test_weird_matmul_case(self):
        # Check that this doesn't crash.
        # https://github.com/pytorch/functorch/issues/417
        x = torch.randn(5, 2, 2, 2)
        y = torch.randn(5, 7, 2)

        vmap(vmap(torch.matmul, in_dims=(None, 0)))(x, y)

    @parametrize(
        "case",
        (
            (torch.clamp_min_, TensorFactory.randn),
            (torch.clamp_max_, TensorFactory.randn),
        ),
        name_fn=lambda x: x[0].__name__,
    )
    def test_clamp_inplace_variant(self, case):
        test = self._vmap_test

        def get_number(getter):
            return getter([]).item()

        op, getter = case
        device = "cpu"
        B0, B1 = 7, 11

        # Single vmap: op(Tensor, Tensor)
        test(
            op,
            (getter([B0, 3], device), getter([B0, 3], device)),
            check_propagates_grad=False,
        )
        test(
            op,
            (getter([B0], device), getter([B0], device)),
            check_propagates_grad=False,
        )
        test(
            op,
            (getter([2, B0, 3], device), getter([2, B0, 3], device)),
            in_dims=(1, 1),
            check_propagates_grad=False,
        )
        test(
            op,
            (getter([B0, 2, 3], device), getter([2, B0, 3], device)),
            in_dims=(0, 1),
            out_dims=1,
            check_propagates_grad=False,
        )
        test(
            op,
            (getter([B0, 2, 3], device), getter([1, 1], device)),
            in_dims=(0, None),
            check_propagates_grad=False,
        )
        test(
            op,
            (getter([B0, 3], device), getter([B0, 3], device)),
            in_dims=(0, 0),
            check_propagates_grad=False,
        )

        # Nested vmap: op(Tensor, Tensor)
        test(
            vmap(op),
            (getter([B0, B1, 2, 3], device), getter([B0, B1, 1, 3], device)),
            check_propagates_grad=False,
        )

        # Python number overload: op(Tensor, Number)
        number = get_number(getter)
        self._test_unary(
            lambda t: op(t, number), getter, device, check_propagates_grad=False
        )

    @parametrize(
        "case",
        [
            subtest(_make_case(torch.clamp_min), name="clamp_min"),
            subtest(_make_case(torch.clamp_max), name="clamp_max"),
        ],
    )
    def test_clamp_variant(self, case):
        test = self._vmap_test

        def get_number(getter):
            return getter([]).item()

        op, getter = case
        device = "cpu"
        B0, B1 = 7, 11

        # Single vmap: op(Tensor, Tensor)
        test(op, (getter([B0, 3], device), getter([B0, 3], device)))
        test(op, (getter([B0], device), getter([B0, 2, 3], device)))
        test(op, (getter([B0], device), getter([2, B0, 3], device)), in_dims=(0, 1))
        test(
            op,
            (getter([B0], device), getter([2, B0, 3], device)),
            in_dims=(0, 1),
            out_dims=1,
        )
        test(op, (getter([B0], device), getter([2, 3], device)), in_dims=(0, None))
        test(op, (getter([2, 3], device), getter([B0, 3], device)), in_dims=(None, 0))

        # Nested vmap: op(Tensor, Tensor)
        test(vmap(op), (getter([B0, B1, 2, 3], device), getter([B0, B1, 3], device)))
        test(
            vmap(op, in_dims=(None, 0)),
            (getter([B0, 2, 3], device), getter([B1, 3], device)),
            in_dims=(0, None),
        )

        # Python number overload: op(Tensor, Number)
        number = get_number(getter)
        self._test_unary(lambda t: op(t, number), getter, device)

    def test_copy_(self):
        x = torch.randn(3)
        y = torch.randn(3)
        vmap(Tensor.copy_)(x, y)
        self.assertEqual(x, y)

        x = torch.randn(3)
        y = torch.randn(3, 2)
        vmap(Tensor.copy_, in_dims=(1, None))(y, x)
        self.assertEqual(y, x.expand(2, 3).t())

        x = torch.randn(3)
        y = torch.randn(2, 3)
        with self.assertRaisesRegex(RuntimeError, "inplace"):
            vmap(Tensor.copy_, in_dims=(None, 0))(x, y)

    def test_silu_backward(self):
        test = self._vmap_test
        device = "cpu"
        getter = TensorFactory.randp1
        B0 = 7
        op = torch.ops.aten.silu_backward

        # Single vmap: op(Tensor, Tensor)
        test(op, (getter([B0, 3], device), getter([B0, 3], device)))
        test(op, (getter([], device), getter([B0], device)), in_dims=(None, 0))
        test(op, (getter([2, B0], device), getter([2], device)), in_dims=(1, None))

    @skipIf(
        TEST_WITH_TORCHDYNAMO
        and os.getenv("BUILD_ENVIRONMENT", "") == "linux-focal-py3.8-clang10",
        "Segfaults with dynamo on focal, see https://github.com/pytorch/pytorch/issues/107173",
    )
    @parametrize(
        "case",
        [
            subtest(_make_case(torch.add), name="add"),
            subtest(_make_case(lambda x, y: x + y), name="add_dunder"),
            subtest(_make_case(torch.sub), name="sub"),
            subtest(_make_case(lambda x, y: x - y), name="sub_dunder"),
            subtest(_make_case(torch.mul), name="mul"),
            subtest(_make_case(lambda x, y: x * y), name="mul_dunder"),
            subtest(
                _make_case(torch.div, input_getter=TensorFactory.randp1), name="div"
            ),
            subtest(
                _make_case(lambda x, y: x / y, input_getter=TensorFactory.randp1),
                name="div_dunder",
            ),
            subtest(
                _make_case(torch.pow, input_getter=TensorFactory.randp1), name="pow"
            ),
            subtest(
                _make_case(lambda x, y: x**y, input_getter=TensorFactory.randp1),
                name="pow_dunder",
            ),
        ],
    )
    def test_arithmetic(self, case):
        test = self._vmap_test

        def get_number(getter):
            return getter([]).item()

        op, getter = case
        device = "cpu"
        B0, B1 = 7, 11

        # Single vmap: op(Tensor, Tensor)
        test(op, (getter([B0, 3], device), getter([B0, 3], device)))
        test(op, (getter([B0], device), getter([B0, 2, 3], device)))
        test(op, (getter([B0], device), getter([2, B0, 3], device)), in_dims=(0, 1))
        test(
            op,
            (getter([B0], device), getter([2, B0, 3], device)),
            in_dims=(0, 1),
            out_dims=1,
        )
        test(op, (getter([B0], device), getter([2, 3], device)), in_dims=(0, None))
        test(op, (getter([2, 3], device), getter([B0, 3], device)), in_dims=(0, None))

        # Nested vmap: op(Tensor, Tensor)
        test(vmap(op), (getter([B0, B1, 2, 3], device), getter([B0, B1, 3], device)))
        test(
            vmap(op, in_dims=(None, 0)),
            (getter([B0, 2, 3], device), getter([B1, 3], device)),
            in_dims=(0, None),
        )

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

    def test_as_strided(self):
        def _test(sizes, strides, offset, tensor, lambd):
            # bdim at dim 0 test
            result = vmap(lambda t: t.as_strided(sizes, strides, offset))(tensor)
            expected = vmap(lambd)(tensor)
            self.assertTrue(result._base is expected._base)
            self.assertEqual(result, expected)

            # bdim at dim -1 test
            tensor = tensor.movedim(0, -1)
            result = vmap(lambda t: t.as_strided(sizes, strides, offset), -1)(tensor)
            expected = vmap(lambd, -1)(tensor)
            self.assertTrue(result._base is expected._base)
            self.assertEqual(result, expected)

        # single vmap test
        B0 = 5
        # Each Tensor has shape [B0, 2, 3]; the expressions below
        # are just to get tensors of different strides that have shape [B0, 2, 3]
        tensors = [
            # contiguous
            torch.randn(B0, 2, 3),
            # non-contiguous
            torch.randn(B0, 3, 2).transpose(1, 2),
            torch.randn(3, 2, B0).movedim(-1, 0).transpose(1, 2),
            # non-zero storage offset
            torch.randn(2, B0, 2, 3)[1],
            torch.randn(2, 2, B0, 3)[1].movedim(1, 0),
            # non-contiguous strides, zero storage offset
            torch.randn(B0, 2, 4, 3, 7)[:, :, 0, :, 0],
            torch.randn(2, 4, B0, 3, 7).movedim(2, 0)[:, :, 0, :, 0],
            # non-contiguous strides, non-zero storage offset
            torch.randn(B0, 2, 4, 3, 7)[:, :, 2, :, 1],
            torch.randn(2, 4, 3, 7, B0).movedim(-1, 0)[:, :, 2, :, 1],
        ]

        for x in tensors:
            S0, S1 = x.stride()[1:]
            offset = x.storage_offset()

            # Broadcast
            _test(
                [5, 5, 2, 3], [0, 0, S0, S1], offset, x, lambda x: x.expand(5, 5, 2, 3)
            )
            # transpose
            _test([3, 2], [S1, S0], offset, x, lambda x: x.transpose(0, 1))
            # select
            _test([2], [S0], offset + S1, x, lambda x: x[:, 1])
            # diagonal
            _test([2], [S0 + S1], offset, x, lambda x: x.diagonal())
            # strided slice
            _test([2], [S1 * 2], offset, x, lambda x: x[0, ::2])

        # Nested vmap test
        B1 = 7
        x = torch.randn(B1, B0, 2, 3)
        S0, S1 = x.stride()[2:]
        result = vmap(
            vmap(lambda t: t.as_strided([5, 5, 2, 3], [0, 0, S0, S1])), in_dims=1
        )(x)
        expected = vmap(vmap(lambda t: t.expand(5, 5, 2, 3)), in_dims=1)(x)
        self.assertTrue(result._base is expected._base)
        self.assertEqual(result, expected)

        # Check that mal-formatted size/strides doesn't crash
        with self.assertRaisesRegex(
            RuntimeError, "size and stride must have the same length"
        ):
            x = torch.randn(B0, 2, 3).transpose(0, 1)
            vmap(lambda x: x.as_strided([1, 1, 1], [1, 1]))(x)

        # All the Sanity check #1{a,b,c} cases check that
        # xs[i].as_strided(sizes, strides, offset + xs[i].offset() - xs.offset())
        # doesn't index memory that is out of bounds of xs[i]. This condition
        # is important to the correctness of the as_strided batching rule
        # (see NOTE: [When will the as_strided_batching_rule fail?])

        # Sanity check #1a: The maximum indexable location of
        # xs[i].as_strided(sizes, strides, offset + xs[i].offset() - xs.offset())
        # is less than or equal to the maximum indexable location of xs[i].
        msg = "This is not supported inside of vmap"
        with self.assertRaisesRegex(RuntimeError, msg):
            x = torch.randn(B0, 3)
            vmap(lambda x: x.as_strided([3], [1], 1))(x)
        with self.assertRaisesRegex(RuntimeError, msg):
            x = torch.randn(B0, 3, 5)
            vmap(lambda x: x.as_strided([4, 4], [4, 1], 0))(x)
        with self.assertRaisesRegex(RuntimeError, msg):
            x = torch.randn(B0, B1, 3, 5)
            vmap(vmap(lambda x: x.as_strided([4, 4], [4, 1], 0)))(x)

        # Sanity check #1b: The min indexable location of
        # xs[i].as_strided(sizes, strides, offset + xs[i].offset() - xs.offset())
        # is greater than or equal to the min indexable location of xs[i].
        with self.assertRaisesRegex(RuntimeError, msg):
            x = torch.randn(2, B0, 3)[1]
            vmap(lambda x: x.as_strided([3], [1], B0 * 3 - 1))(x)

        # Sanity check #1c:
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
        test(functools.partial(op, reduction="sum"), (y, t))
        test(functools.partial(op, reduction="none"), (y, t))

        y = torch.randn(B, 2, 5)
        t = torch.randint(0, 5, (2,))
        test(op, (y, t), in_dims=(0, None))
        test(functools.partial(op, reduction="sum"), (y, t), in_dims=(0, None))
        test(functools.partial(op, reduction="none"), (y, t), in_dims=(0, None))

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
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, B0, 2, 3, 5), torch.rand(2, 5, 3)),
            in_dims=(1, None),
        )

        # right arg is vmapped
        test(op, (torch.rand(2, 5, 3), torch.rand(B0, 2, 3, 5)), in_dims=(None, 0))
        test(
            vmap(op, in_dims=(None, 0)),
            (torch.rand(2, 5, 3), torch.rand(B1, B0, 2, 3, 5)),
            in_dims=(None, 1),
        )

        # both args are vmapped
        test(op, (torch.rand(B0, 2, 3, 5), torch.rand(B0, 2, 5, 3)))
        test(
            vmap(op),
            (torch.rand(B1, B0, 2, 3, 5), torch.rand(B0, B1, 2, 5, 3)),
            in_dims=(1, 0),
        )
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, 2, 3, 5), torch.rand(B0, 2, 5, 3)),
            in_dims=(None, 0),
        )

    def test_cat(self):
        test = self._vmap_test
        B0, B1 = 5, 7

        # Quick hack b/c vmap can't accept a list of tensors as an argument
        def get_op(dim):
            def op(*tensors):
                return torch.cat(tensors, dim=dim)

            return op

        test(get_op(0), (torch.rand(B0, 2), torch.rand(B0, 3)))
        test(get_op(0), (torch.rand(B0, 0), torch.rand(B0, 0)))
        test(get_op(0), (torch.rand(2), torch.rand(B0, 0)), in_dims=(None, 0))
        test(
            get_op(1),
            (torch.rand(2, 5), torch.rand(B0, 0), torch.rand(2, 3)),
            in_dims=(None, 0, None),
        )
        test(get_op(1), (torch.rand(B0, 2, 3), torch.rand(B0, 0)))
        test(get_op(1), (torch.rand(B0, 2, 3, 4), torch.rand(0)), in_dims=(0, None))
        test(
            get_op(0),
            (torch.rand(0), torch.rand(B0, 2), torch.rand(B0, 0)),
            in_dims=(None, 0, 0),
        )
        test(get_op(0), (torch.rand(2), torch.rand(B0, 3)), in_dims=(None, 0))
        test(get_op(0), (torch.rand(2, 17), torch.rand(3, 17, B0)), in_dims=(None, 2))
        test(get_op(-1), (torch.rand(17, 2), torch.rand(17, 3, B0)), in_dims=(None, 2))
        test(
            vmap(get_op(0), in_dims=(0, None)),
            (torch.rand(B1, 2), torch.rand(B0, 3)),
            in_dims=(None, 0),
        )
        test(
            vmap(get_op(0), in_dims=(0, 0)),
            (torch.rand(B1, 2), torch.rand(B0, B1, 3)),
            in_dims=(None, 0),
        )

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
            test(vmap(op, in_dims=2), [get([2, 5, B0, B1, 3])], in_dims=2, out_dims=2)

        # correctness tests
        run_test(torch.float)
        run_test(torch.cfloat)

        # check that torch.conj on a non-complex tensor returns the same tensor
        real_tensor = torch.randn(3)
        result = vmap(op)(real_tensor)
        self.assertEqual(result.data_ptr(), real_tensor.data_ptr())

    def test_contiguous(self):
        op = Tensor.contiguous

        self._test_unary(op, TensorFactory.randn, "cpu")

        # check that contiguous returns the original tensor if the per-examples
        # are already contiguous
        B0 = 3
        x = torch.randn(B0, 2, 5, 7)
        x = x.movedim(0, 2)
        result = vmap(Tensor.contiguous, in_dims=2, out_dims=2)(x)
        self.assertTrue(result is x)

        msg = "NYI: querying is_contiguous inside of vmap for memory_format"
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
        test(
            vmap(op, in_dims=(0, None, None)),
            (torch.rand(B1, 1023, B0, 5), 4, 0),
            in_dims=(2, None, None),
        )
        test(
            vmap(vmap(lambda t: op(t, 4, 1), in_dims=2)),
            (torch.rand(B1, 2, B0, 64, B2),),
            in_dims=2,
        )

    def test_clamp(self):
        clamp_cases = (
            (lambda t: t.clamp(min=-0.5), TensorFactory.randn),
            (lambda t: t.clamp(max=0.5), TensorFactory.randn),
            (lambda t: t.clamp(min=-0.5, max=0.5), TensorFactory.randn),
            (lambda t: t.clamp_min(min=-0.5), TensorFactory.randn),
            (lambda t: t.clamp_max(max=0.5), TensorFactory.randn),
        )
        for op, getter in clamp_cases:
            self._test_unary(op, getter, "cpu")

    def test_comparison_ops(self):
        test = functools.partial(self._vmap_test, check_propagates_grad=False)

        getter = TensorFactory.randn
        B0, B1 = 7, 11

        ops = (
            torch.eq,
            lambda x, y: x == y,
            torch.gt,
            lambda x, y: x > y,
            torch.ge,
            lambda x, y: x >= y,
            torch.le,
            lambda x, y: x <= y,
            torch.lt,
            lambda x, y: x < y,
            torch.ne,
            lambda x, y: x != y,
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
            test(
                vmap(op, in_dims=(None, 0)),
                (getter([B0, 2, 3]), getter([B1, 3])),
                in_dims=(0, None),
            )

            # test number as inputs
            number = getter([]).item()
            self._test_unary(
                lambda t: op(t, number), getter, "cpu", check_propagates_grad=False
            )

    def test_cross_batch_size_three(self):
        # Let's test corner case when batch_size is 3 and cross' dim argument is not specified
        # According to the cross API, dim will be assigned to the first dim with value 3
        # In this test we ensure that found dim is not batch dim.
        op = torch.cross
        test = self._vmap_test
        B0 = B1 = 3
        test(op, (torch.rand(B0, 2, 3), torch.rand(B0, 2, 3)))
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B0, B1, 2, 3), torch.rand(B0, B1, 2, 3)),
            in_dims=(None, 1),
        )

    def test_diagonal(self):
        tensor = torch.randn(3, 5, 7, 11, 13)
        test = self._vmap_view_test
        op = torch.diagonal
        test(op, (tensor, 1, 0, 1), in_dims=(0, None, None, None))
        test(op, (tensor, 0, 2, -1), in_dims=(0, None, None, None))
        test(op, (tensor, 2, 1, 2), in_dims=(1, None, None, None))
        test(op, (tensor, 0, -2, -1), in_dims=(1, None, None, None), out_dims=1)
        test(vmap(lambda t: op(t, 0, 0, -1)), (tensor,), in_dims=1, out_dims=1)
        test(
            vmap(vmap(lambda t: op(t, 0, 0, 1), in_dims=1), in_dims=3),
            (tensor,),
            in_dims=1,
            out_dims=1,
        )

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
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, B0, 5), torch.rand(5)),
            in_dims=(1, None),
        )

        # right arg is vmapped
        test(op, (torch.rand(5), torch.rand(B0, 5)), in_dims=(None, 0))
        test(
            vmap(op, in_dims=(None, 0)),
            (torch.rand(5), torch.rand(B1, B0, 5)),
            in_dims=(None, 1),
        )

        # both args are vmapped
        test(op, (torch.rand(B0, 5), torch.rand(B0, 5)))
        test(vmap(op), (torch.rand(B1, B0, 5), torch.rand(B0, B1, 5)), in_dims=(1, 0))
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, 5), torch.rand(B0, 5)),
            in_dims=(None, 0),
        )

    def test_expand_as(self):
        op = torch.Tensor.expand_as
        test = self._vmap_view_test
        B0, B1, B2 = 7, 11, 13
        test(op, (torch.rand(B0, 1, 5), torch.rand(B0, 2, 3, 5)))
        test(op, (torch.rand(B0, 1, 5), torch.rand(2, 3, 5)), in_dims=(0, None))
        test(op, (torch.rand(1, 5), torch.rand(B0, 2, 3, 5)), in_dims=(None, 0))
        test(vmap(op), (torch.rand(B0, B1, 1, 5), torch.rand(B0, B1, 2, 3, 5)))
        test(
            vmap(op),
            (torch.rand(B0, B1, 1, 5), torch.rand(B1, B0, 2, 3, 5)),
            in_dims=(0, 1),
        )
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
            test(
                vmap(op, in_dims=2),
                [TensorFactory.randn([2, 5, B0, B1, 3])],
                in_dims=2,
                out_dims=2,
            )

        # test when value is a batched tensor for fill_ operator
        B0, B1 = 3, 5
        test(Tensor.fill_, [TensorFactory.randn([B0, B1]), TensorFactory.randn(B0)])

        with self.assertRaisesRegex(RuntimeError, ""):
            # Runtime Error is thrown when the tensor being written to isn't being vmapped over
            vmap(Tensor.fill_, (None, 0))(
                TensorFactory.randn([B0, B1]), TensorFactory.randn([B0])
            )

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
            test(vmap(op, in_dims=2), [get([2, 5, B0, B1, 3])], in_dims=2, out_dims=2)

        for dtype in dtypes:
            run_test(op, dtype)

    def test_real(self):
        self._test_complex_views(torch.real, dtypes=[torch.cfloat, torch.cdouble])

    def test_imag(self):
        self._test_complex_views(torch.imag, dtypes=[torch.cfloat, torch.cdouble])

    def test_view_as_real(self):
        self._test_complex_views(
            torch.view_as_real, dtypes=[torch.cfloat, torch.cdouble]
        )

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
            test(
                vmap(op, in_dims=2), [get([2, 5, B0, B1, 3, 2])], in_dims=2, out_dims=2
            )

            # Interesting case #1: Batch dim directly before dim of size 2
            test(op, [get([3, B0, 2])], in_dims=1)
            test(vmap(op, in_dims=1), [get([3, B1, B0, 2])], in_dims=2)

            # Interesting case #2: Batch dim at end of tensor, success cases
            # view_as_complex requires that the dim with size 2 have stride 1
            # in order for the view to function property
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
            msg = "Input tensor must have one or more dimensions"
            with self.assertRaisesRegex(RuntimeError, msg):
                vmap(op)(get([B0]))
            with self.assertRaisesRegex(RuntimeError, msg):
                vmap(vmap(op))(get([B0, B1]))

            # Invalid input: Batch dim has size 2, but the logical last dim does
            # not have size 2
            msg = "Tensor must have a last dimension of size 2"
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
        float_tensor = torch.tensor([1.0, 2.0, 3.0])
        long_tensor = torch.tensor([1, 2, 3])

        def foo(x):
            if x.is_floating_point():
                return torch.tensor(1)
            else:
                return torch.tensor(0)

        self.assertEqual(vmap(foo)(float_tensor), torch.tensor([1, 1, 1]))
        self.assertEqual(vmap(foo)(long_tensor), torch.tensor([0, 0, 0]))

    @unittest.skipIf(IS_WINDOWS, reason="Windows not yet supported for torch.compile")
    def test_is_contiguous(self):
        def foo(x):
            if x.is_contiguous():
                return torch.tensor(1.0)
            else:
                return torch.tensor(0.0)

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

        msg = "NYI: querying is_contiguous inside of vmap for memory_format"
        tensor = torch.randn(B0, 2, 7, 3)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(functools.partial(baz, memory_format=torch.channels_last))(tensor)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(functools.partial(baz, memory_format=torch.channels_last_3d))(tensor)

        for mf in (torch.channels_last, torch.channels_last_3d):

            @torch.compile(backend="eager", fullgraph=True)
            def f(x):
                if x.is_contiguous(memory_format=mf):
                    return x.sin()
                return x.cos()

            with self.assertRaisesRegex(RuntimeError, msg):
                vmap(f)(torch.randn(3, 3))

    def test_unsqueeze(self):
        op = torch.unsqueeze
        test = self._vmap_view_test
        B0, B1 = 7, 11

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
        test(vmap(unsqueeze_0), (torch.rand(B0, B1, 2),))
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
        test(
            vmap(op, in_dims=(0, None, None)),
            (torch.rand(B1, 2, B0, 5), 0, 1),
            in_dims=(2, None, None),
        )
        test(
            vmap(vmap(op, in_dims=(2, None, None)), in_dims=(0, None, None)),
            (torch.rand(B1, 2, B0, 5, B2), 0, 1),
            in_dims=(2, None, None),
        )

        # movedim(tensor, intlist, intlist) variant
        test(op, (torch.rand(B0, 2, 3, 5), [1, 0], [0, 2]), in_dims=(0, None, None))
        test(op, (torch.rand(2, 3, B0, 5), [1, 0], [0, 2]), in_dims=(1, None, None))
        test(
            vmap(op, in_dims=(0, None, None)),
            (torch.rand(B1, 2, B0, 5), [0, 1], [1, 0]),
            in_dims=(2, None, None),
        )
        test(
            vmap(vmap(op, in_dims=(2, None, None)), in_dims=(0, None, None)),
            (torch.rand(B1, 2, B0, 5, B2), [0, 1], [1, 0]),
            in_dims=(2, None, None),
        )

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
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, B0, 2, 5), torch.rand(5, 2)),
            in_dims=(1, None),
        )

        # right arg is vmapped
        test(op, (torch.rand(2, 5), torch.rand(B0, 5, 2)), in_dims=(None, 0))
        test(
            vmap(op, in_dims=(None, 0)),
            (torch.rand(2, 5), torch.rand(B1, B0, 5, 2)),
            in_dims=(None, 1),
        )

        # both args are vmapped
        test(op, (torch.rand(B0, 2, 5), torch.rand(B0, 5, 2)))
        test(
            vmap(op),
            (torch.rand(B1, B0, 2, 5), torch.rand(B0, B1, 5, 2)),
            in_dims=(1, 0),
        )
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, 2, 5), torch.rand(B0, 5, 2)),
            in_dims=(None, 0),
        )

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
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, B0, 2, 5), torch.rand(5)),
            in_dims=(1, None),
        )

        # right arg is vmapped
        test(op, (torch.rand(2, 5), torch.rand(B0, 5)), in_dims=(None, 0))
        test(
            vmap(op, in_dims=(None, 0)),
            (torch.rand(2, 5), torch.rand(B1, B0, 5)),
            in_dims=(None, 1),
        )

        # both args are vmapped
        test(op, (torch.rand(B0, 2, 5), torch.rand(B0, 5)))
        test(
            vmap(op), (torch.rand(B1, B0, 2, 5), torch.rand(B0, B1, 5)), in_dims=(1, 0)
        )
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, 2, 5), torch.rand(B0, 5)),
            in_dims=(None, 0),
        )

    def test_narrow(self):
        op = torch.narrow
        test = self._vmap_view_test
        B0, B1, B2 = 7, 11, 13

        test(op, (torch.rand(B0, 2, 5), -1, 1, 3), in_dims=(0, None, None, None))
        test(op, (torch.rand(2, B0, 5), 1, 1, 3), in_dims=(1, None, None, None))
        test(
            vmap(op, in_dims=(0, None, None, None)),
            (torch.rand(B1, 2, B0, 5), 1, 0, 0),
            in_dims=(2, None, None, None),
        )
        test(
            vmap(
                vmap(op, in_dims=(2, None, None, None)), in_dims=(0, None, None, None)
            ),
            (torch.rand(B1, 2, B0, 5, B2), -1, 2, 3),
            in_dims=(2, None, None, None),
        )

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
            result = vmap(vmap(lambda x: x.new_empty_strided(size, stride)), in_dims=1)(
                x
            )
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
        test(
            vmap(vmap(lambda t: op(t, 1, 1), in_dims=1)),
            (torch.rand(B1, 2, B0, B2, 5),),
            in_dims=2,
        )

    def test_roll_no_dims(self):
        op = torch.roll
        test = self._vmap_test
        B0, B1, B2 = 7, 11, 13
        test(op, (torch.rand(B0, 2, 5), 2), in_dims=(0, None))
        test(op, (torch.rand(2, B0, 5), 3), in_dims=(1, None))
        test(vmap(lambda t: op(t, 3)), (torch.rand(B1, 2, B0, 5),), in_dims=2)
        test(
            vmap(vmap(lambda t: op(t, 3), in_dims=1)),
            (torch.rand(B1, 2, B0, B2, 5),),
            in_dims=2,
        )

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
        test(
            vmap(get_op(0), in_dims=(0, None)),
            (torch.rand(B1, 2), torch.rand(B0, 2)),
            in_dims=(None, 0),
        )
        test(
            vmap(get_op(0), in_dims=(0, 0)),
            (torch.rand(B1, 2), torch.rand(B0, B1, 2)),
            in_dims=(None, 0),
        )

    def test_slice(self):
        test = self._vmap_view_test
        B0, B1, B2 = 7, 11, 13
        test(lambda t: t[0:1], (torch.rand(B0, 3, 5),))
        test(lambda t: t[:, 1:3], (torch.rand(3, 5, B0),), in_dims=2)
        test(
            vmap(lambda t: t[:, 0:1], in_dims=2), (torch.rand(3, 5, B0, B1),), in_dims=2
        )
        test(
            vmap(vmap(lambda t: t[0:1], in_dims=2), in_dims=2),
            (torch.rand(3, 5, B0, B1, B2),),
            in_dims=2,
        )

    @xfailIfTorchDynamo
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
        test(
            vmap(lambda x: op(x, 2), in_dims=2),
            [torch.randn([2, 5, B0, B1, 3])],
            in_dims=2,
            out_dims=2,
        )

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

    @skipIfTorchDynamo()
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
        test(
            op, (torch.rand(2, B0, 5), [1, 1, 10]), in_dims=(1, None), check_view=False
        )
        test(
            vmap(lambda t: t.reshape([-1])),
            (torch.rand(B0, B1, 2, 5),),
            check_view=True,
        )
        test(
            vmap(vmap(lambda t: t.reshape([-1]), in_dims=2), in_dims=1),
            (torch.rand(3, B1, 2, B2, 5, B0),),
            in_dims=5,
            check_view=False,
        )

    def test_reshape_as(self):
        test = self._vmap_test
        B0, B1, B2 = 7, 11, 13
        op = torch.Tensor.reshape_as
        test(op, (torch.rand(B0, 2 * 5), torch.rand(B0, 2, 5)), check_view=True)
        test(
            op,
            (torch.rand(2 * 5), torch.rand(B0, 2, 5)),
            in_dims=(None, 0),
            check_view=True,
        )
        test(
            op,
            (torch.rand(B0, 2 * 5), torch.rand(2, 5)),
            in_dims=(0, None),
            check_view=True,
        )

        test(
            op,
            (torch.rand(2, B0, 5), torch.rand(1, 1, 10)),
            in_dims=(1, None),
            check_view=False,
        )

        test(
            vmap(op),
            (torch.rand(B0, B1, 2, 5), torch.randn(B0, B1, 10)),
            check_view=True,
        )
        test(
            vmap(vmap(op, in_dims=(2, None)), in_dims=(1, None)),
            (torch.rand(3, B1, 2, B2, 5, B0), torch.rand(B0, 3 * 2 * 5)),
            in_dims=(5, 0),
            check_view=False,
        )

    def test_result_type(self):
        def scalar_tensor_with_dtype(op):
            def wrapped(*args, **kwargs):
                dtype = op(*args, **kwargs)
                return torch.ones([], dtype=dtype)

            return wrapped

        test = self._vmap_test
        op = scalar_tensor_with_dtype(torch.result_type)

        B0 = 2

        test(
            op,
            (torch.randn(B0), torch.randn(B0, dtype=torch.float64)),
            check_propagates_grad=False,
        )
        test(
            op,
            (torch.randn(B0), torch.randint(10, [B0], dtype=torch.int64)),
            check_propagates_grad=False,
        )

        test(lambda x: op(x, 1), (torch.randn(B0),), check_propagates_grad=False)
        test(lambda x: op(x, 1.6), (torch.randn(B0),), check_propagates_grad=False)

        test(
            lambda x: op(x, torch.tensor(1)),
            (torch.randn(B0),),
            check_propagates_grad=False,
        )
        test(
            lambda x: op(x, torch.tensor(1.6, dtype=torch.double)),
            (torch.randn(B0),),
            check_propagates_grad=False,
        )

        test(
            op,
            (torch.randn(B0, 2), torch.randn(B0, 2, dtype=torch.float64)),
            check_propagates_grad=False,
        )
        test(
            op,
            (torch.randn(B0, 2), torch.randint(10, [B0, 2], dtype=torch.int64)),
            check_propagates_grad=False,
        )

        test(lambda x: op(x, 1), (torch.randn(B0, 2),), check_propagates_grad=False)
        test(lambda x: op(x, 1.6), (torch.randn(B0, 2),), check_propagates_grad=False)

        test(
            lambda x: op(x, torch.tensor(1)),
            (torch.randn(B0, 2),),
            check_propagates_grad=False,
        )
        test(
            lambda x: op(x, torch.tensor(1.6, dtype=torch.double)),
            (torch.randn(B0, 2),),
            check_propagates_grad=False,
        )

        test(
            op,
            (torch.randn(B0, 2), torch.randn(B0, dtype=torch.float64)),
            check_propagates_grad=False,
        )
        test(
            op,
            (torch.randn(B0, 2), torch.randint(10, [B0], dtype=torch.int64)),
            check_propagates_grad=False,
        )

    def test_tensor_split(self):
        test = self._vmap_view_test
        op = torch.tensor_split
        B0, B1, B2 = 7, 11, 13

        # tests for torch.tensor_split(self, indices_or_sections: int, dim)
        test(op, (torch.rand(B0, 2, 1024), 5, -1), in_dims=(0, None, None))
        test(op, (torch.rand(2, B0, 1024), 150, 1), in_dims=(1, None, None))
        test(
            vmap(op, in_dims=(0, None, None)),
            (torch.rand(B1, 1023, B0, 5), 256, 0),
            in_dims=(2, None, None),
        )
        test(
            vmap(vmap(lambda t: op(t, 4, 1), in_dims=2)),
            (torch.rand(B1, 2, B0, 64, B2),),
            in_dims=2,
        )

        # tests for torch.tensor_split(self, indices_or_sections: List[int], dim)
        test(
            op,
            (torch.rand(B0, 2, 1024), [50, 100, 378, 890], -1),
            in_dims=(0, None, None),
        )
        test(
            op,
            (torch.rand(2, B0, 1024), [50, 100, 212, 345, 0, 378, 890], 1),
            in_dims=(1, None, None),
        )
        test(
            vmap(op, in_dims=(0, None, None)),
            (torch.rand(B1, 1023, B0, 5), [50, 100, 212, 345, 0, 378, 890], 0),
            in_dims=(2, None, None),
        )
        test(
            vmap(vmap(lambda t: op(t, [4, 8, 9, 34, 29], 1), in_dims=2)),
            (torch.rand(B1, 2, B0, 64, B2),),
            in_dims=2,
        )

    @skipIfTorchDynamo("really slow")
    def test_split(self):
        test = self._vmap_view_test
        op = torch.split
        B0, B1, B2 = 7, 11, 13

        # tests for torch.split(self, split_size: int, dim)
        test(op, (torch.rand(B0, 2, 1024), 101, -1), in_dims=(0, None, None))
        test(op, (torch.rand(2, B0, 1024), 130, 1), in_dims=(1, None, None))
        test(
            vmap(op, in_dims=(0, None, None)),
            (torch.rand(B1, 1023, B0, 5), 256, 0),
            in_dims=(2, None, None),
        )
        test(
            vmap(vmap(lambda t: op(t, 4, 1), in_dims=2)),
            (torch.rand(B1, 2, B0, 64, B2),),
            in_dims=2,
        )

        # tests for torch.split(self, split_size: List[int], dim)
        test(op, (torch.rand(B0, 2, 1024), [1, 1020, 3], -1), in_dims=(0, None, None))
        test(
            op, (torch.rand(2, B0, 1024), [100] * 10 + [24], 1), in_dims=(1, None, None)
        )
        test(
            vmap(op, in_dims=(0, None, None)),
            (torch.rand(B1, 1023, B0, 5), [256] * 3 + [255], 0),
            in_dims=(2, None, None),
        )
        test(
            vmap(vmap(lambda t: op(t, [4] * 8 + [8] * 4, 1), in_dims=2)),
            (torch.rand(B1, 2, B0, 64, B2),),
            in_dims=2,
        )

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
        test(
            vmap(vmap(lambda x: op(x, 0, 1), in_dims=2)),
            (torch.rand(B1, 2, B0, 5, B2),),
            in_dims=2,
        )

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
        test(op, (torch.rand(2, B0, 3, 5),), in_dims=1)
        test(vmap(op), (torch.rand(B1, 2, B0, 5),), in_dims=2)
        test(vmap(op), (torch.rand(B1, 2, B0, 3, 5),), in_dims=2)
        test(vmap(vmap(op, in_dims=2)), (torch.rand(B1, 2, B0, 3, B2, 5),), in_dims=2)

    def test_to(self):
        test = self._vmap_test
        B0, B1 = 7, 11

        test(lambda t: t.to("cpu"), (torch.rand(B0),))
        test(lambda t: t.to(torch.double), (torch.rand(B0),))
        test(
            lambda t, o: t.to(o), (torch.rand(B0), torch.randn(B0, dtype=torch.float64))
        )
        test(
            lambda t, o: t.to(o),
            (torch.rand(B0), torch.randn(B0, dtype=torch.float64)),
            in_dims=(0, None),
        )
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
        test(
            vmap(op, in_dims=(0, None, None, None)),
            (torch.rand(B1, 7, B0, 11), 1, 5, 1),
            in_dims=(2, None, None, None),
        )
        test(
            vmap(
                vmap(op, in_dims=(2, None, None, None)), in_dims=(0, None, None, None)
            ),
            (torch.rand(B1, 7, B0, 11, B2), -1, 2, 4),
            in_dims=(2, None, None, None),
        )

    def test_unbind(self):
        test = self._vmap_view_test
        op = torch.unbind
        B0, B1, B2 = 7, 11, 13

        test(op, (torch.rand(B0, 2, 1024), -1), in_dims=(0, None))
        test(op, (torch.rand(B0, 2, 0),))
        test(op, (torch.rand(2, B0, 7), 0), in_dims=(1, None))
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, 1023, B0, 5), 1),
            in_dims=(2, None),
        )
        test(
            vmap(vmap(lambda t: op(t, dim=1), in_dims=2)),
            (torch.rand(B1, 2, B0, 32, B2),),
            in_dims=2,
        )

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
        test(
            vmap(vmap(lambda t: t.reshape([-1])), in_dims=1),
            (torch.rand(B2, B0, B1, 3, 2, 5),),
            in_dims=1,
        )

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
        test(
            vmap(vmap(op, in_dims=(0, None)), in_dims=(0, None)),
            (torch.rand(B1, B2, B0, 3, 2, 5), torch.rand(B0, 3 * 2 * 5)),
            in_dims=(2, 0),
        )

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
            for loop_out, batched_out in get_fallback_and_vmap_exhaustive(
                conv_fn, arg_values, kwarg_values
            ):
                self.assertEqual(loop_out, batched_out)

            arg_values = [torch.randn(inp_shape), mod.weight, None]
            for loop_out, batched_out in get_fallback_and_vmap_exhaustive(
                conv_fn, arg_values, kwarg_values
            ):
                self.assertEqual(loop_out, batched_out)

            mod2 = conv_mod(
                4, 8, kernel_size=3, groups=2, stride=3, padding=1, dilation=2
            )
            arg_values = [torch.randn(inp_shape), mod2.weight, mod2.bias]
            kwarg_values = dict(groups=2, stride=3, padding=1, dilation=2)
            for loop_out, batched_out in get_fallback_and_vmap_exhaustive(
                conv_fn, arg_values, kwarg_values
            ):
                self.assertEqual(loop_out, batched_out)

            arg_values = [torch.randn(inp_shape), mod2.weight, None]
            for loop_out, batched_out in get_fallback_and_vmap_exhaustive(
                conv_fn, arg_values, kwarg_values
            ):
                self.assertEqual(loop_out, batched_out)

    def test_one_hot(self):
        sample_inputs = [
            (torch.randint(0, 3, []), 3),
            (torch.randint(0, 3, [2, 3, 4]), 4),
        ]
        for args in sample_inputs:
            for loop_out, batched_out in get_fallback_and_vmap_exhaustive(
                F.one_hot, args, {}
            ):
                self.assertEqual(loop_out, batched_out)

    def test_conj_bit(self):
        x = torch.tensor([1 + 1j, 2 + 1j])

        def foo(x):
            assert not x.is_conj()
            y = x.conj()
            assert y.is_conj()
            return y

        res = vmap(foo)(x)
        self.assertEqual(res, x.conj())

    def test_mode_key(self):
        def vmap_f(x):
            return x + torch.randn(())

        def naive_f(x, shape):
            return x + torch.randn(shape)

        torch.manual_seed(0)
        out1 = vmap(vmap(vmap_f, randomness="different"), randomness="different")(
            torch.ones(2, 3)
        )

        torch.manual_seed(0)
        out2 = naive_f(torch.ones(2, 3), (2, 3))
        self.assertEqual(out1, out2)

        torch.manual_seed(0)
        out1 = vmap(vmap(vmap_f, randomness="different"), randomness="different")(
            torch.ones(2, 3, 4)
        )

        torch.manual_seed(0)
        out2 = naive_f(torch.ones(2, 3, 4), (2, 3, 1))
        self.assertEqual(out1, out2)

        self.assertTrue(torch.randn(()).dim() == 0)

    @parametrize("in_dim", [0, 1, 2])
    @parametrize("out_dim", [0, 1, 2])
    @parametrize("randomness", ["error", "same"])
    def test_chunk_vmap(self, in_dim, out_dim, randomness):
        x = torch.randn(4, 5, 6)

        def f(x):
            y = x.sin()
            if randomness != "error":
                y = y + torch.rand_like(x)
            return y

        rs = torch.get_rng_state()
        expected = vmap(f, in_dims=in_dim, out_dims=out_dim, randomness=randomness)(x)

        for chunks in [1, 2, 3, 4, 7, 10, 16]:
            torch.set_rng_state(rs)
            output = chunk_vmap(
                f,
                in_dims=in_dim,
                out_dims=out_dim,
                randomness=randomness,
                chunks=chunks,
            )(x)
            self.assertEqual(output, expected)

    @parametrize("in_dim", [0, 1, 2])
    @parametrize("out_dim", [0, 1, 2])
    @parametrize("randomness", ["error", "same"])
    def test_vmap_chunksize(self, in_dim, out_dim, randomness):
        x = torch.randn(4, 5, 6)
        y = torch.randn_like(x)

        # fn: Single Input/Single Output
        def f(x):
            y = x.sin()
            if randomness != "error":
                y = y + torch.rand_like(x)
            return y

        f_args = (x,)
        f_kwargs = {"in_dims": in_dim, "out_dims": out_dim, "randomness": randomness}

        # fn: Nested Input/Single Output
        def f1(pair):
            x, y = pair
            z = x.sin() + y.cos()
            if randomness != "error":
                z = z + torch.rand_like(z)
            return z

        f1_args = ((x, y),)
        f1_kwargs = {
            "in_dims": ((in_dim,) * 2,),
            "out_dims": out_dim,
            "randomness": randomness,
        }

        # fn: Single Input/Nested Output
        def f2(x):
            y = x.sin()
            if randomness != "error":
                y = y + torch.rand_like(x)
            return {"out": y, "out1": y + 2}

        f2_args = (x,)
        f2_kwargs = {"in_dims": in_dim, "out_dims": out_dim, "randomness": randomness}

        # fn: Nested Input/Nested Output (first tensor is not vmapped).
        def f3(inp_dict):
            x = inp_dict["inp"]
            y = inp_dict["inp1"]
            z = x.sin() + y.cos()
            if randomness != "error":
                z = z + torch.rand_like(z)
            return {"z": z, "tuple": (z, z + 1)}

        f3_args = (
            {
                "inp": x.index_select(in_dim, torch.tensor([0])).squeeze(in_dim),
                "inp1": y,
            },
        )
        f3_kwargs = {
            "in_dims": ({"inp": None, "inp1": in_dim},),
            "out_dims": out_dim,
            "randomness": randomness,
        }

        # fn: Nested Input/Nested Output (first argument is not a Tensor).
        def f4(inp_dict):
            x = inp_dict["inp"]
            y = inp_dict["inp1"]
            z = x + y.cos()
            if randomness != "error":
                z = z + torch.rand_like(z)
            return {"z": z, "tuple": (z, z + 1)}

        f4_args = ({"inp": 2.0, "inp1": y},)
        f4_kwargs = {
            "in_dims": ({"inp": None, "inp1": in_dim},),
            "out_dims": out_dim,
            "randomness": randomness,
        }

        fns_and_args = (
            (f, f_args, f_kwargs),
            (f1, f1_args, f1_kwargs),
            (f2, f2_args, f2_kwargs),
            (f3, f3_args, f3_kwargs),
            (f4, f4_args, f4_kwargs),
        )
        for fn, args, kwargs in fns_and_args:
            rs = torch.get_rng_state()
            expected_vmap = vmap(fn, **kwargs)(*args)
            for chunk_size in (1, 2, 3, 4, 7, 10, 16, 100):
                torch.set_rng_state(rs)
                output = vmap(fn, chunk_size=chunk_size, **kwargs)(*args)
                self.assertEqual(output, expected_vmap)

    @parametrize("in_dim", [0, 1])
    @parametrize("out_dim", [0, 1])
    @parametrize("randomness", ["error", "same"])
    def test_vmap_chunksize_error(self, in_dim, out_dim, randomness):
        x = torch.randn(4, 5, 6)

        def f(x):
            y = x.sin()
            if randomness != "error":
                y = y + torch.rand_like(x)
            return y

        # Incorrect `chunk_size`
        for chunk_size in (-1, 0):
            with self.assertRaisesRegex(
                ValueError, "vmap: chunk_size should be None or greater than 0."
            ):
                vmap(
                    f,
                    in_dims=in_dim,
                    out_dims=out_dim,
                    randomness=randomness,
                    chunk_size=chunk_size,
                )(x)

        # Incorrect `out_dims`
        msg = "out_dims is not compatible with the structure of `outputs`"
        with self.assertRaisesRegex(ValueError, msg):
            vmap(
                f,
                in_dims=in_dim,
                out_dims=(out_dim, out_dim),
                randomness=randomness,
                chunk_size=2,
            )(x)

    @parametrize("in_dim", [0, 1])
    @parametrize("out_dim", [0, 1])
    @parametrize("randomness", ["error", "same"])
    def test_vmap_chunksize_composition(self, in_dim, out_dim, randomness):
        x = torch.randn(4, 5, 6)
        y = torch.randn_like(x)

        # fn: Single Input/Single Output
        def f(x):
            y = x.sin()
            if randomness != "error":
                y = y + torch.rand_like(x)
            return y

        f_args = (x,)

        # fn: Nested Input/Single Output
        def f1(pair):
            x, y = pair
            z = x.sin() + y.cos()
            if randomness != "error":
                z = z + torch.rand_like(z)
            return z

        f1_args = ((x, y),)

        # fn: Single Input/Nested Output
        def f2(x):
            y = x.sin()
            if randomness != "error":
                y = y + torch.rand_like(x)
            return {"out": y, "out1": y + 2}

        f2_args = (x,)

        # fn: Nested Input/Nested Output
        def f3(inp_dict):
            x = inp_dict["inp"]
            y = inp_dict["inp1"]
            z = x.sin() + y.cos()
            if randomness != "error":
                z = z + torch.rand_like(z)
            return {"z": z, "tuple": (z, z + 1)}

        f3_args = ({"inp": x, "inp1": y},)

        for fn, args in ((f, f_args), (f1, f1_args), (f2, f2_args), (f3, f3_args)):
            rs = torch.get_rng_state()
            expected = vmap(
                vmap(fn, in_dims=in_dim, out_dims=out_dim, randomness=randomness),
                in_dims=in_dim,
                out_dims=out_dim,
                randomness=randomness,
            )(*args)
            for chunk_size in (1, 2, 3, 4, 7, 10, 16, 100):
                torch.set_rng_state(rs)
                actual = vmap(
                    vmap(
                        fn,
                        in_dims=in_dim,
                        out_dims=out_dim,
                        randomness=randomness,
                        chunk_size=chunk_size,
                    ),
                    in_dims=in_dim,
                    out_dims=out_dim,
                    randomness=randomness,
                    chunk_size=chunk_size,
                )(*args)
                self.assertEqual(actual, expected)


instantiate_parametrized_tests(TestVmapOperators)


def construct_v(output, batch_size, contig=False):
    if contig:
        return torch.randn(
            batch_size, *output.shape, dtype=output.dtype, device=output.device
        )
    result = torch.randn(
        *output.shape, batch_size, dtype=output.dtype, device=output.device
    )
    return result.movedim(-1, 0)


def as_tuple(x):
    if isinstance(x, tuple):
        return x
    elif isinstance(x, list):
        return tuple(x)
    else:
        return (x,)


def differentiable(args):
    return tuple(
        arg
        for arg in as_tuple(args)
        if isinstance(arg, torch.Tensor) and arg.requires_grad
    )


def _get_rand_no_zeros(*args, **kwargs):
    requires_grad = kwargs.get("requires_grad", False)
    kwargs_without_requires_grad = kwargs.copy()
    kwargs_without_requires_grad["requires_grad"] = False
    result = torch.rand(*args, **kwargs_without_requires_grad)
    return result.clamp_min_(0.1).requires_grad_(requires_grad)


@markDynamoStrictTest
class TestVmapBatchedGradient(Namespace.TestVmapBase):
    def _vmap_test(self, *args, **kwargs):
        return _vmap_test(self, *args, **kwargs)

    # Tests batched gradient computation of outputs = op(*args, **kwargs)
    # by comparing it to a sequential map+stack fallback.
    #
    # output_process_fn: a function that maps the outputs to the part
    #       that should be differentiated.
    # batch_size: the batch dim size for the batched grad
    def _batched_grad_test(
        self, op, args, kwargs=None, output_process_fn=lambda x: x, batch_size=3
    ):
        if kwargs is None:
            kwargs = {}
        outputs = op(*args, **kwargs)
        outputs = differentiable(output_process_fn(outputs))
        for contig in [True, False]:
            batched_vectors = tuple(
                construct_v(out, batch_size, contig) for out in outputs
            )

            def vector_jacobian_product(*vectors):
                return torch.autograd.grad(
                    outputs, differentiable(args), vectors, retain_graph=True
                )

            self._vmap_test(
                vector_jacobian_product, batched_vectors, check_propagates_grad=False
            )

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
    def _batched_grad_grad_test(
        self, op, args, kwargs=None, output_process_fn=lambda x: x, batch_size=3
    ):
        if kwargs is None:
            kwargs = {}
        outputs = op(*args, **kwargs)
        outputs = differentiable(output_process_fn(outputs))
        ones = tuple(torch.ones_like(out) for out in outputs)
        # Same thing as summing together all of the outputs and calling .backward()
        first_grads = torch.autograd.grad(
            outputs, differentiable(args), ones, create_graph=True
        )
        first_grads = differentiable(first_grads)
        self.assertNotEqual(
            len(first_grads), 0, "None of the first grads depend on the input!"
        )

        for contig in [True, False]:
            batched_vectors = tuple(
                construct_v(grad, batch_size, contig) for grad in first_grads
            )

            def vector_hessian_product(*vectors):
                outputs = torch.autograd.grad(
                    first_grads,
                    differentiable(args),
                    vectors,
                    retain_graph=True,
                    allow_unused=True,
                )
                outputs = tuple(out for out in outputs if out is not None)
                assert len(outputs) > 0
                return outputs

            self._vmap_test(
                vector_hessian_product, batched_vectors, check_propagates_grad=False
            )

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

        x = torch.randn(3, 2, 2, device=device)

        def sum_grad_trace(x):
            return grad(torch.trace)(x).sum()

        output = vmap(grad(sum_grad_trace))(x)
        self.assertEqual(output, torch.zeros_like(output))

    def test_where(self, device):
        x = torch.randn(3, 2, device=device)
        y = torch.ones(3, 2, device=device)

        def f(x, y):
            return torch.where(x > 0, x, y)

        # Check that there is no runtime error, exactness tests are done with opinfo
        vmap(f)(x, y)

        x = torch.randint(0, 2, size=(4, 3), dtype=torch.float)

        def f(t):
            return torch.where(t)

        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to vmap over aten::where"
        ):
            vmap(f)(x)

    def test_threshold(self, device):
        x = torch.randn(2, 3, device=device, requires_grad=True)
        self._batched_grad_test(lambda x: F.threshold(x, 0.5, 0.0), (x,))

    @parametrize("backend", PLATFORM_SPECIFIC_SDPA)
    def test_sdpa(self, device, backend):
        if device == "cpu":
            raise unittest.SkipTest("This test is only for CUDA for now")

        def T(*args):
            return torch.randn(*args, dtype=torch.float16, device=device)

        backend_ctx = sdpa_kernel([backend])
        with backend_ctx:
            for batching in [
                (True, True, True),
                (True, False, False),
                (False, True, True),
            ]:
                size = [8, 4, 128, 64]
                if batching[0]:
                    query = T(3, *size)
                else:
                    query = T(*size)
                if batching[1]:
                    key = T(3, *size)
                else:
                    key = T(*size)
                if batching[2]:
                    value = T(3, *size)
                else:
                    value = T(*size)
                in_dims = tuple(0 if b else None for b in batching)
                attention = F.scaled_dot_product_attention

                self._vmap_test(
                    attention,
                    (query, key, value),
                    in_dims=in_dims,
                )
                # Backwards test doesn't work yet
                # self._batched_grad_test(
                #     lambda query, key, value: F.scaled_dot_product_attention(
                #         query, key, value
                #     ),
                #     (query, key, value),
                # )

            B = 4
            query = torch.rand(4, 32, B, 8, 128, dtype=torch.float16, device=device)
            key = torch.rand(4, B, 32, 8, 128, dtype=torch.float16, device=device)
            value = torch.rand(4, 32, 8, 128, dtype=torch.float16, device=device)
            self._vmap_test(
                F.scaled_dot_product_attention,
                (query, key, value),
                in_dims=(2, 1, None),
            )

    @parametrize("backend", PLATFORM_SPECIFIC_SDPA)
    @parametrize("randomness", ["error", "same", "different"])
    def test_randomness(self, device, randomness, backend):
        if device == "cpu":
            raise unittest.SkipTest("This test is only for CUDA for now")
        backend_ctx = sdpa_kernel([backend])
        with backend_ctx:
            B = 4
            query = torch.rand(B, 4, 32, 8, 128, dtype=torch.float16, device=device)
            key = torch.rand(B, 4, 32, 8, 128, dtype=torch.float16, device=device)
            value = torch.rand(B, 4, 32, 8, 128, dtype=torch.float16, device=device)

            def f(q, k, v, dropout):
                return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout)

            # No matter the randomness mode, dropout=0.0 should pass
            vmap(
                functools.partial(f, dropout=0.0),
                in_dims=(0, 0, 0),
                randomness=randomness,
            )(query, key, value)

            fail_with_randomness = randomness == "error"
            if backend != SDPBackend.MATH:
                fail_with_randomness |= randomness == "same"
            context = (
                self.assertRaises(RuntimeError)
                # We currently don't support randomness == "same", and "error" should always error with randomness
                if fail_with_randomness
                else contextlib.nullcontext()
            )
            with context:
                vmap(
                    functools.partial(f, dropout=0.5),
                    in_dims=(0, 0, 0),
                    randomness=randomness,
                )(query, key, value)

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
            (res,) = torch.autograd.grad(y, x, v, allow_unused=True)
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
            (res,) = torch.autograd.grad(y, x, v, allow_unused=True)
            return torch.zeros_like(x) if res is None else res

        _ = vjp(gy[0])
        result = vmap(vjp)(gy)
        self.assertEqual(result, torch.zeros(B0, *x.shape, device=device))


def discover_variants(opinfo):
    aliases = []
    inplace_variants = []

    if opinfo.inplace_variant:
        inplace_variants.append(opinfo.inplace_variant)

    aliases.append(opinfo.op)
    for alias in opinfo.aliases:
        aliases.append(alias.op)
        if alias.inplace_variant:
            inplace_variants.append(alias.inplace_variant)
    return aliases, inplace_variants


# TODO: enable this when we get a bit closer to getting torch.vmap x torch.compile working.
# @markDynamoStrictTest
@unMarkDynamoStrictTest
class TestVmapOperatorsOpInfo(TestCase):
    def vmap_outplace_test(
        self,
        func,
        args,
        kwargs,
        in_dims,
        check_shape_only=False,
        postprocess_fn=None,
        out_dim=0,
    ):
        for vmap_out, loop_out in compute_quantities_for_vmap_test(
            func, args, kwargs, in_dims, out_dim=out_dim
        ):
            if postprocess_fn is not None:
                loop_out = postprocess_fn(loop_out)
                vmap_out = postprocess_fn(vmap_out)
            if check_shape_only:
                self.assertEqual(vmap_out.shape, loop_out.shape)
                continue
            self.assertEqual(vmap_out, loop_out)

    def vmap_inplace_test(
        self, func, args, kwargs, in_dims, postprocess_fn=None, out_dim=0
    ):
        # NB: This test assumes that the first argument is being modified.
        # This is OK because it's what every other OpInfo-based test assumes,
        # but it is going to need a more robust solution eventually.
        if in_dims[0] is None:
            # Check that we correctly raise an error when vmap is impossible
            # on the in-place operation
            with self.assertRaises(RuntimeError):
                for _ in compute_quantities_for_vmap_test(
                    func,
                    args,
                    kwargs,
                    in_dims,
                    out_dim=out_dim,
                    compute_loop_out=False,
                    clone_inputs=True,
                ):
                    pass
            return
        for vmap_out, loop_out in compute_quantities_for_vmap_test(
            func,
            args,
            kwargs,
            in_dims,
            clone_inputs=True,
            out_dim=out_dim,
        ):
            if postprocess_fn is not None:
                loop_out = postprocess_fn(loop_out)
                vmap_out = postprocess_fn(vmap_out)
            self.assertEqual(vmap_out, loop_out)

    def opinfo_vmap_test(
        self,
        device,
        dtype,
        op,
        check_has_batch_rule,
        skip_inplace=(),
        postprocess_fn=None,
    ):
        def test():
            # Error inputs check
            if op.error_inputs_func is not None:
                error_inputs = op.error_inputs(device)
                for error_input in error_inputs:
                    sample_input = error_input.sample_input
                    args = (sample_input.input,) + tuple(sample_input.args)
                    kwargs = sample_input.kwargs
                    for batched_args, in_dims, _ in generate_vmap_inputs(args, {}):
                        with self.assertRaises(Exception):
                            vmap(op, in_dims)(*batched_args, **kwargs)

            # Sample inputs check
            sample_inputs_op = {
                # Take too long with reference inputs
                "special.chebyshev_polynomial_t",
                "special.chebyshev_polynomial_u",
                "special.chebyshev_polynomial_v",
                "special.chebyshev_polynomial_w",
                "special.hermite_polynomial_he",
                "special.laguerre_polynomial_l",
                "special.legendre_polynomial_p",
                "special.shifted_chebyshev_polynomial_t",
                "special.shifted_chebyshev_polynomial_u",
                "special.shifted_chebyshev_polynomial_v",
                "special.shifted_chebyshev_polynomial_w",
            }
            if op.name in sample_inputs_op:
                sample_inputs_itr = op.sample_inputs(
                    device, dtype, requires_grad=False, use_subtests=True
                )
            else:
                sample_inputs_itr = op.reference_inputs(
                    device, dtype, requires_grad=False, use_subtests=True
                )
            aliases, inplace_aliases = discover_variants(op)
            check_shape_only = op.name in ("empty_like", "new_empty")
            for sample_input, subtest_ctx, skip_xfail_ctx in sample_inputs_itr:
                with subtest_ctx(self), skip_xfail_ctx(self):
                    args = (sample_input.input,) + sample_input.args
                    if not any(isinstance(arg, torch.Tensor) for arg in args):
                        # Atleast one tensor required for vmap.
                        continue
                    kwargs = sample_input.kwargs
                    is_batch_norm_and_training = is_batch_norm_training(op.name, kwargs)
                    out_dim = 0
                    if op.name == "NumpySplitCopyWithIntCustomOp":
                        # special case for this custom op
                        def sample_vmap_out_dim_numpy_split_copy_with_int(
                            x, splits, dim
                        ):
                            return [0 for _ in range(len(splits) + 1)], None

                        out_dim = sample_vmap_out_dim_numpy_split_copy_with_int(*args)
                    for batched_args, in_dims, _ in generate_vmap_inputs(
                        args, {}, is_batch_norm_and_training=is_batch_norm_and_training
                    ):
                        for func in aliases:
                            self.vmap_outplace_test(
                                func,
                                batched_args,
                                kwargs,
                                in_dims,
                                check_shape_only,
                                postprocess_fn,
                                out_dim=out_dim,
                            )
                        if op.name in skip_inplace:
                            continue
                        if not is_valid_inplace_sample_input(
                            sample_input, op, op.inplace_variant
                        ):
                            continue
                        for func in inplace_aliases:
                            self.vmap_inplace_test(
                                func, batched_args, kwargs, in_dims, postprocess_fn
                            )

        if check_has_batch_rule:
            check_vmap_fallback(self, test, op)
        else:
            test()

    vmap_fail = {
        # -------------------- ALLOWED FAILURES --------------------------------
        # These are things that we either cannot fix or are not actually problems
        xfail("resize_"),
        xfail("resize_as_"),
        xfail("to_sparse"),
        xfail("__getitem__"),  # dynamic mask
        xfail("index_put"),  # dynamic mask
        xfail(
            "nn.functional.dropout"
        ),  # works, can't check against for loop because of randomness inconsistency
        xfail("nn.functional.scaled_dot_product_attention"),  # randomness
        xfail("nn.functional.multi_head_attention_forward"),  # randomness
        xfail("masked_select"),  # dynamic op
        xfail("nonzero"),  # dynamic op
        xfail("unique", ""),  # dynamic op
        xfail("unique_consecutive", ""),  # dynamic op
        xfail("allclose"),  # returns a boolean
        xfail("uniform"),  # randomness is tested separately
        xfail("rand_like"),  # randomness is tested separately
        xfail("randint_like"),  # randomness is tested separately
        xfail("randn_like"),  # randomness is tested separately
        xfail("bernoulli", ""),  # randomness is tested separately
        xfail("normal", ""),  # randomness is tested separately
        xfail("normal", "number_mean"),  # randomness is tested separately
        xfail("multinomial", ""),  # randomness
        xfail("nn.functional.embedding", ""),  # we only support some cases
        xfail("nn.functional.rrelu"),  # randomness
        xfail("nn.functional.dropout2d", ""),  # randomness
        xfail("nn.functional.dropout3d", ""),  # randomness
        xfail("nn.functional.alpha_dropout", ""),  # randomness
        xfail("nn.functional.feature_alpha_dropout", "with_train"),  # randomness
        xfail("as_strided"),  # Our test runner can't handle this; manual test exists
        xfail("as_strided_copy"),
        xfail(
            "as_strided_scatter"
        ),  # no batching rule implemented, default doesnt work
        skip(
            "new_empty_strided"
        ),  # empty tensor data is garbage so it's hard to make comparisons with it
        xfail("nn.functional.fractional_max_pool3d"),  # randomness
        xfail("nn.functional.fractional_max_pool2d"),  # randomness
        xfail("pca_lowrank", ""),  # random operation
        xfail("svd_lowrank", ""),  # random operation
        xfail("sparse.sampled_addmm"),  # sparse
        xfail("sparse.mm", "reduce"),  # sparse
        xfail(
            "NumpyCubeNotComposableAutogradFunction"
        ),  # Not composable autograd.Function
        skip("_softmax_backward_data"),
        skip(
            "linalg.eigh", ""
        ),  # not always return the same result for the same input, see test_linalg_eigh for manual test
        # UnimplementedError: data-dependent operators cannot be vmapped
        xfail("NumpyNonzeroCustomOp"),
        xfail("NumpyNMSCustomOp"),
        # ----------------------------------------------------------------------
        # ---------------------------- BUGS ------------------------------------
        # entries in here don't work and need to be fixed.
        # Each one of these is a bug
        decorate("frexp", decorator=skipIfTorchDynamo()),
        xfail("clamp_min", ""),  # Exception not raised on error input
        xfail("clamp_max", ""),  # Exception not raised on error input
        xfail(
            "view_as_complex"
        ),  # RuntimeError: Tensor must have a last dimension with stride 1
        xfail("tensor_split"),  # data_ptr
        xfail(
            "histogramdd"
        ),  # expected Tensor as element 0 in argument 0, but got tuple
        xfail("nn.functional.gaussian_nll_loss"),  # data-dependent control flow error
        xfail(
            "nn.functional.embedding_bag"
        ),  # embedding renorm vmap inplace incompatible
        xfail("narrow"),  # Batching rule not implemented for aten::narrow.Tensor
        # required rank 4 tensor to use channels_last format
        xfail("bfloat16"),
        xfail("bool"),
        xfail("byte"),
        xfail("char"),
        xfail("double"),
        xfail("float"),
        xfail("half"),
        xfail("int"),
        xfail("long"),
        xfail("short"),
        xfail("cdouble"),
        xfail("cfloat"),
        xfail(
            "jiterator_binary", device_type="cuda"
        ),  # NYI: querying is_contiguous inside of vmap
        xfail(
            "jiterator_binary_return_by_ref", device_type="cuda"
        ),  # NYI: querying is_contiguous inside of vmap
        xfail(
            "jiterator_4inputs_with_extra_args", device_type="cuda"
        ),  # NYI: querying is_contiguous inside of vmap
        xfail(
            "equal", ""
        ),  # TypeError: object of type 'bool' has no len(); likely testrunner problem
        xfail(
            "jiterator_unary", device_type="cuda"
        ),  # NYI: querying is_contiguous inside of vmap
        xfail(
            "jiterator_2inputs_2outputs", device_type="cuda"
        ),  # NYI: querying is_contiguous inside of vmap
        # ---------------------------------------------------------------------
        # TypeError: expected Tensor as element 0 in argument 0, but got NotImplementedType
        xfail("__rsub__"),
        # RuntimeError: Batching rule not implemented for aten::moveaxis.int;
        # the fallback path doesn't work on out= or view ops.
        xfail("movedim"),
        # RuntimeError: NYI: querying is_contiguous inside of vmap for
        # memory_format other than torch.contiguous_format
        xfail("contiguous"),
        # RuntimeError: NYI: Tensor.clone(memory_format) inside vmap is only supported
        # with memory_format torch.preserve_format or torch.contiguous_format (got ChannelsLast)
        xfail("clone"),
        # RuntimeError: When vmap-ing torch.nn.functional.one_hot,
        # please provide an explicit positive num_classes argument.
        xfail("nn.functional.one_hot"),
        # RuntimeError: Expected all tensors to be on the same device,
        # but found at least two devices, cuda:0 and cpu!
        xfail("eq", device_type="cuda"),
        xfail("ge", device_type="cuda"),
        xfail("gt", device_type="cuda"),
        xfail("le", device_type="cuda"),
        xfail("lt", device_type="cuda"),
        xfail("ne", device_type="cuda"),
        # RuntimeError: aten::_flash_attention_forward hit the vmap fallback which is currently disabled
        xfail("torch.ops.aten._flash_attention_forward"),
    }

    @with_tf32_off  # https://github.com/pytorch/pytorch/issues/86798
    @ops(
        op_db + additional_op_db + autograd_function_db + custom_op_db,
        dtypes=OpDTypes.any_one,
    )
    @opsToleranceOverride(
        "TestVmapOperatorsOpInfo",
        "test_vmap_exhaustive",
        (
            tol1(
                "linalg.det",
                {torch.float32: tol(atol=1e-04, rtol=1e-04)},
                device_type="cuda",
            ),
            # The following is often flaky, but just on windows.
            # We should investigate if it's actually a problem or not.
            tol1(
                "nn.functional.conv_transpose3d",
                {torch.float32: tol(atol=1e-04, rtol=1e-02)},
                device_type="cuda",
            ),
        ),
    )
    @toleranceOverride(
        {
            torch.float32: tol(atol=1e-04, rtol=1e-04),
            torch.complex64: tol(atol=1e-04, rtol=1e-04),
        }
    )
    @skipOps(
        "TestVmapOperatorsOpInfo",
        "test_vmap_exhaustive",
        vmap_fail.union(
            {
                # RuntimeError: Batch norm got a batched tensor as input while the running_mean or running_var,
                # which will be updated in place, were not batched.
                xfail("native_batch_norm"),
                xfail("_native_batch_norm_legit"),
                # TODO: implement batching rule
                xfail("_batch_norm_with_update"),
                xfail("tril"),  # Exception not raised on error input
                xfail("triu"),  # Exception not raised on error input
                xfail("as_strided", "partial_views"),
                # RuntimeError: output with shape [4, 4] doesn't match the broadcast shape [1, 4, 4]
                xfail("addcdiv"),
                xfail("addcmul"),
                xfail("clamp"),
                xfail("torch.ops.aten._efficient_attention_forward"),  # outputs ints
                # TypeError: expected Tensor as element 0 in argument 0, but got float
                xfail("item"),
                # RuntimeError: required rank 4 tensor to use channels_last format
                xfailIf(
                    "to",
                    lambda sample: (
                        sample.kwargs["memory_format"] == torch.channels_last
                    ),
                ),
            }
        ),
    )
    def test_vmap_exhaustive(self, device, dtype, op):
        # needs to be fixed
        inplace_failure_list = ()
        self.opinfo_vmap_test(
            device,
            dtype,
            op,
            check_has_batch_rule=False,
            skip_inplace=inplace_failure_list,
        )

    @with_tf32_off
    @ops(
        op_db + additional_op_db + autograd_function_db + custom_op_db,
        dtypes=OpDTypes.any_one,
    )
    @opsToleranceOverride(
        "TestVmapOperatorsOpInfo",
        "test_op_has_batch_rule",
        (
            tol1(
                "linalg.det",
                {torch.float32: tol(atol=1e-04, rtol=1e-04)},
                device_type="cuda",
            ),
        ),
    )
    @toleranceOverride(
        {
            torch.float32: tol(atol=1e-04, rtol=1e-04),
            torch.complex64: tol(atol=1e-04, rtol=1e-04),
        }
    )
    @skipOps(
        "TestVmapOperatorsOpInfo",
        "test_op_has_batch_rule",
        vmap_fail.union(
            {
                xfail("as_strided", "partial_views"),
                skip(
                    "to"
                ),  # RuntimeError: required rank 4 tensor to use channels_last format
                xfail("fill"),
                # Batch norm got a batched tensor as input while the running_mean or running_var,
                # which will be updated in place, were not batched.
                xfail("native_batch_norm"),
                xfail("_native_batch_norm_legit"),
                # TODO: implement batching rule
                xfail("_batch_norm_with_update"),
                xfail("histogram"),
                # `index_put` OpInfo in pytorch/pytorch has
                # masked index as input which is not supported
                xfail("index_put", ""),
                xfail("isin"),
                xfail("masked_fill"),
                xfail("masked_scatter"),
                xfail("masked_select"),
                xfail("nanquantile"),
                xfail("ormqr"),
                xfail("put"),
                xfail("quantile"),
                xfail("renorm"),
                xfail("squeeze_copy"),
                xfail("resize_as_"),
                xfail("take"),
                xfail("tensor_split"),
                xfail("transpose_copy"),
                xfail("to_sparse"),
                # TypeError: expected Tensor as element 0 in argument 0, but got float
                xfail("item"),
                xfail("tril"),  # Exception not raised on error input
                xfail("triu"),  # Exception not raised on error input
                xfail("__getitem__", ""),
                xfail("count_nonzero"),
                xfail(
                    "nn.functional.dropout"
                ),  # works, can't check against for loop because of randomness inconsistency
                xfail("nn.functional.scaled_dot_product_attention"),  # randomness
                xfail("nn.functional.multi_head_attention_forward"),  # randomness
                xfail("torch.ops.aten._efficient_attention_forward"),  # outputs ints
                xfail("resize_"),
                xfail("view_as_complex"),
                xfail("matrix_exp"),
                xfail("fft.ihfft2"),
                xfail("fft.ihfftn"),
                xfail("allclose"),
                xfail("argwhere"),
                xfail("unique_consecutive"),
                xfail("unique"),
                xfail("nn.functional.ctc_loss"),
                xfail("nn.functional.gaussian_nll_loss"),
                xfail("histc"),
                xfail("as_strided"),
                xfail("as_strided_copy"),
                xfail("permute_copy"),
                xfail("t_copy"),
                xfail("unsqueeze_copy"),
                xfail("istft"),
                xfail("nonzero"),
                xfail("nn.functional.fractional_max_pool2d"),
                xfail("stft"),
                xfail("isclose"),
                xfail("nn.functional.fractional_max_pool3d"),
                xfail("nn.functional.bilinear"),
                xfail("nn.functional.embedding_bag"),
                xfail("linalg.tensorsolve"),
                xfail("bernoulli", ""),
                xfail("nn.functional.feature_alpha_dropout", "with_train"),
                xfail("nn.functional.kl_div", ""),
                xfail("multinomial", ""),
                xfail("pca_lowrank", ""),
                xfail("normal", ""),
                xfail("nn.functional.dropout2d", ""),
                xfail("normal", "number_mean"),
                xfail("svd_lowrank", ""),
                xfail("diagflat", ""),
                xfail("special.log_ndtr"),
                xfail(
                    "narrow"
                ),  # Batching rule not implemented for aten::narrow.Tensor
                xfail("nn.functional.triplet_margin_loss", ""),
                xfail("nn.functional.pdist", ""),
                xfail("nn.functional.max_unpool1d", "grad"),
                xfail("nn.functional.multi_margin_loss", ""),
                xfail("nn.functional.multilabel_margin_loss", ""),
                xfail("nn.functional.max_unpool3d", "grad"),
                xfail("nn.functional.max_unpool2d", ""),
                xfail("nn.functional.max_unpool2d", "grad"),
                xfail("nn.functional.margin_ranking_loss", ""),
                xfail("nn.functional.max_unpool1d", ""),
                xfail("nn.functional.soft_margin_loss", ""),
                xfail("nn.functional.max_unpool3d", ""),
                xfail("linalg.ldl_solve", "", device_type="cpu"),
                xfail("chalf", ""),
                xfail("clamp_max", ""),
                xfail("jiterator_binary_return_by_ref", device_type="cuda"),
                xfail("jiterator_unary", device_type="cuda"),
                xfail("jiterator_2inputs_2outputs", device_type="cuda"),
                xfail("special.airy_ai"),
                xfail("clamp_min", ""),
                xfail("sparse.sampled_addmm"),
                xfail("sparse.mm", "reduce"),
                xfail("special.chebyshev_polynomial_u"),
                xfail("_segment_reduce", "offsets"),
                xfail("index_reduce", "prod"),
                xfail("index_reduce", "mean"),
                xfail("index_reduce", "amin"),
                xfail("index_reduce", "amax"),
                xfail("special.laguerre_polynomial_l"),
                xfail("special.hermite_polynomial_h"),
                xfail("jiterator_binary", device_type="cuda"),
                xfail("jiterator_4inputs_with_extra_args", device_type="cuda"),
                xfail("_segment_reduce", "lengths"),
                xfail("lu_solve", ""),
                xfail("special.hermite_polynomial_he"),
                xfail("nn.functional.dropout3d", ""),
                xfail("special.chebyshev_polynomial_t"),
                xfail("as_strided_scatter", ""),
                xfail("equal", ""),
                xfail("linalg.lu", ""),
                skip("linalg.ldl_solve", ""),
                skip("_softmax_backward_data"),
                # One or more of the overload doesn't have a Batch rule.
                xfail("bincount"),
                # RuntimeError: Expected all tensors to be on the same device,
                # but found at least two devices, cuda:0 and cpu!
                xfail("ge", device_type="cuda"),
                xfail(
                    "searchsorted"
                ),  # aten::searchsorted.Scalar hit the vmap fallback which is currently disabled
            }
        ),
    )
    def test_op_has_batch_rule(self, device, dtype, op):
        # needs to be fixed
        inplace_failures = (
            "addbmm",
            "addcdiv",
            "addcmul",
            "addmm",
            "addmv",
            "addr",
            "baddbmm",
            "clamp",
            "conj_physical",
            "cumprod",
            "cumsum",
            "floor_divide",
            "fmod",
            "heaviside",
            "hypot",
            "igamma",
            "igammac",
            "index_copy",
            "ldexp",
            "lerp",
            "neg",
            "nextafter",
            "polygamma",
            "pow",
            "remainder",
            "scatter_add",
            "scatter",
            "square",
            "sub",
            "trunc",
            "xlogy",
        )
        self.opinfo_vmap_test(
            device, dtype, op, check_has_batch_rule=True, skip_inplace=inplace_failures
        )

    def test_linalg_svd(self, device):
        # linalg_svd returns a tuple of three tensors, (U, S, Vh).
        # Given the same input, it may return different tensors,
        # because svd isn't unique. To test that the svd is correct, we multiply
        # U @ diag(S) @ Vh and check that the output from vmap matches the
        # output from a for-loop.
        def compute_A(out):
            U, S, Vh = out
            m = U.shape[-1]
            n = Vh.shape[-2]
            diag_S = S.new_zeros(*S.shape[:-1], m, n)
            diag_S.diagonal(offset=0, dim1=-2, dim2=-1).copy_(S)
            return U @ diag_S @ Vh

        opinfos = [op for op in op_db if op.name == "linalg.svd"]
        assert len(opinfos) > 0

        for op in opinfos:
            self.opinfo_vmap_test(
                device,
                torch.float,
                op,
                check_has_batch_rule=True,
                postprocess_fn=compute_A,
            )

    def test_linalg_eigh(self, device):
        # linalg_svd returns two tensors, (Q, L).
        # Given the same input, it may return different tensors,
        # because the eig decomposition isn't unique.
        # To test that eigh is correct, we multiply
        # Q @ diag(L) @ Qh and check that the output from vmap matches the
        # output from a for-loop.
        def compute_A(out):
            L, Q = out
            n = Q.shape[-1]
            diag_L = L.new_zeros(*L.shape[:-1], n, n)
            diag_L.diagonal(offset=0, dim1=-2, dim2=-1).copy_(L)
            Qh = Q.transpose(-2, -1).conj()
            return Q @ diag_L @ Qh

        opinfos = [op for op in op_db if op.name == "linalg.eigh"]
        assert len(opinfos) > 0

        for op in opinfos:
            self.opinfo_vmap_test(
                device,
                torch.float,
                op,
                check_has_batch_rule=True,
                postprocess_fn=compute_A,
            )

    @skipIfTorchDynamo()
    def test_slogdet(self, device):
        # There's no OpInfo for this
        def test():
            B = 2
            x = torch.randn(B, 5, 5, device=device)
            self.vmap_outplace_test(torch.slogdet, (x,), {}, (0,))

        check_vmap_fallback(self, test, torch.slogdet)

    def test_index_fill(self, device):
        # There's no OpInfo for these tests

        B = 2

        def test1():
            # negative dim
            x = torch.randn(B, 5, 5, device=device)
            dim = -2
            index = torch.tensor([[2, 3], [0, 4]], device=device)
            value = 5.0
            self.vmap_outplace_test(
                torch.index_fill, (x, dim, index, value), {}, (None, None, 0, None)
            )

        def test2():
            # self batched, self logical rank 1, index logical rank 1
            x = torch.zeros(B, 3, device=device)
            dim = 0
            index = torch.tensor([[0], [1]], device=device)
            for value in (1.0, torch.rand((), device=device)):
                self.vmap_outplace_test(
                    torch.index_fill, (x, dim, index, value), {}, (0, None, 0, None)
                )

        def test3():
            # self batched, self logical rank 1, index logical rank 0
            x = torch.zeros(B, 3, device=device)
            dim = 0
            index = torch.tensor([0, 1], device=device)
            for value in (1.0, torch.rand((), device=device)):
                self.vmap_outplace_test(
                    torch.index_fill, (x, dim, index, value), {}, (0, None, 0, None)
                )

        def test4():
            # self not batched, self logical rank 0, index logical rank 1
            x = torch.zeros([], device=device)
            dim = 0
            index = torch.tensor([[0], [0]], device=device)
            for value in (1.0, torch.rand((), device=device)):
                self.vmap_outplace_test(
                    torch.index_fill, (x, dim, index, value), {}, (None, None, 0, None)
                )

        def test5():
            # self not batched, self logical rank 0, index logical rank 0
            x = torch.zeros([], device=device)
            dim = 0
            index = torch.tensor([0, 0], device=device)
            for value in (1.0, torch.rand((), device=device)):
                self.vmap_outplace_test(
                    torch.index_fill, (x, dim, index, value), {}, (None, None, 0, None)
                )

        def test6():
            # self not batched, self logical rank 0, index logical rank 1
            x = torch.zeros(3, device=device)
            dim = 0
            index = torch.tensor([[0], [1]], device=device)
            for value in (1.0, torch.rand((), device=device)):
                self.vmap_outplace_test(
                    torch.index_fill, (x, dim, index, value), {}, (None, None, 0, None)
                )

        def test7():
            # self not batched, self logical rank 0, index logical rank 0
            x = torch.zeros(3, device=device)
            dim = 0
            index = torch.tensor([0, 1], device=device)
            for value in (1.0, torch.rand((), device=device)):
                self.vmap_outplace_test(
                    torch.index_fill, (x, dim, index, value), {}, (None, None, 0, None)
                )

        def test8():
            # self batched, self logical rank > 1, index logical rank 0
            x = torch.zeros(B, 3, 3, device=device)
            dim = 0
            index = torch.tensor([0, 1], device=device)
            for value in (1.0, torch.rand((), device=device)):
                self.vmap_outplace_test(
                    torch.index_fill, (x, dim, index, value), {}, (0, None, 0, None)
                )

        for test in (test1, test2, test3, test4, test5, test6, test7, test8):
            check_vmap_fallback(self, test, torch.index_fill)

    def test_fill__Tensor(self, device):
        # There's no OpInfo for fill_.Tensor, so here's an extra test for it.
        def test():
            B = 2
            args = (torch.randn(B, 3, device=device), torch.randn(B))
            self.vmap_inplace_test(Tensor.fill_, args, {}, (0, 0))

            args = (torch.randn(3, B, device=device), torch.randn(B))
            self.vmap_inplace_test(Tensor.fill_, args, {}, (-1, 0))

            args = (torch.randn(3, device=device), torch.randn(B))
            self.vmap_inplace_test(Tensor.fill_, args, {}, (None, 0))

            args = (torch.randn(3, B, device=device), torch.randn([]))
            self.vmap_inplace_test(Tensor.fill_, args, {}, (1, None))

        check_vmap_fallback(self, test, Tensor.fill_)

    @tf32_on_and_off(0.005)
    def test_conv_double_backward(self, device):
        images = torch.randn(2, 1, 5, 5, device=device)
        weight = torch.randn(2, 1, 2, 2, device=device)
        bias = torch.randn(2, device=device)
        ggI = torch.randn_like(images)
        ggW = torch.randn_like(weight)
        ggb = torch.randn_like(bias)
        stride = (1, 1)
        padding = (0, 0)
        dilation = (1, 1)
        transposed = False
        output_padding = (0, 0)
        groups = 1
        output_mask = (True, True, True)
        gO = torch.randn_like(
            F.conv2d(images, weight, bias, stride, padding, dilation, groups)
        )

        args = (
            ggI,
            ggW,
            ggb,
            gO,
            weight,
            images,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            output_mask,
        )
        op = torch.ops.aten._convolution_double_backward

        generator = get_fallback_and_vmap_exhaustive(op, args, {})
        is_cuda_sm86 = device.startswith("cuda") and torch.cuda.get_device_capability(
            0
        ) == (8, 6)
        atol, rtol = (1e-3, 1e-3) if is_cuda_sm86 else (1e-4, 1e-4)

        def test():
            for loop_out, batched_out in generator:
                self.assertEqual(loop_out, batched_out, atol=atol, rtol=rtol)

        check_vmap_fallback(self, test, op)

    def test_isnan(self, device):
        test = functools.partial(_vmap_test, check_propagates_grad=False)

        B, N, C, H, W = 2, 3, 24, 5, 7
        op = torch.isnan

        x = torch.randn(B, N, C, H, W)
        x[x > 0] = float("nan")
        test(self, op, (x,), in_dims=(0))

    def test_sum_scalar(self, device):
        x = torch.tensor([10.0], device=device)
        y = vmap(torch.sum)(x)
        self.assertEqual(y, x)

        y = vmap(lambda x: x.sum(0))(x)
        self.assertEqual(y, x)

        y = vmap(lambda x: x.sum(-1))(x)
        self.assertEqual(y, x)

    def test_isinf(self, device):
        test = functools.partial(_vmap_test, check_propagates_grad=False)

        B, N, C, H, W = 2, 3, 24, 5, 7
        op = torch.isinf

        x = torch.randn(B, N, C, H, W)
        x[x > 0] = float("inf")
        test(self, op, (x,), in_dims=(0))

    def test_foo_like(self, device):
        # vfdev-5: Probably, we can remove this line. Flake8 reported as unused
        # test = functools.partial(_vmap_test, check_propagates_grad=False)

        B, N, C, H, W = 2, 3, 24, 5, 7
        for op in [torch.ones_like, torch.zeros_like]:
            x = torch.randn(B, N, C, H, W)
            # todo(chilli): test these better
            # Not testing correctness, just that they run
            vmap(op, in_dims=(0,))(
                x,
            )

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
        def test(f, t, idx, values):
            base = f(t[0], idx[0], values[0])
            self.assertEqual(vmap(f, in_dims=(0, 0, 0))(t, idx, values)[0], base)
            self.assertEqual(
                vmap(f, in_dims=(0, None, None))(t, idx[0], values[0])[0], base
            )
            self.assertEqual(vmap(f, in_dims=(0, None, 0))(t, idx[0], values)[0], base)
            self.assertEqual(vmap(f, in_dims=(0, 0, None))(t, idx, values[0])[0], base)

        def f(x, y, z):
            x[y] = z
            return x

        x = torch.randn(3, 4, 5, device=device)
        y = torch.zeros((3, 2), device=device).long()
        z = torch.randn(3, 2, 5, device=device)
        test(f, x, y, z)

        # indexing innermost dim
        def f(t, idx, values):
            t[:, idx] = values
            return t

        t = torch.zeros((3, 2, 3))
        values = torch.ones((3, 1, 2))
        idx = torch.tensor([[1, 2]]).expand((3, 2))
        test(f, t, idx, values)

        # indexing middle dim
        def f(t, idx, values):
            t[:, idx, :] = values
            return t

        t = torch.zeros((3, 2, 3, 3))
        values = torch.ones((3, 1, 2, 3))
        idx = torch.tensor([[0, 2]]).expand((3, 2))
        test(f, t, idx, values)

        # indexing with slices
        def f(t, values):
            t[:, :2, :] = values
            return t

        base = f(t[0], values[0])
        self.assertEqual(vmap(f, in_dims=(0, 0))(t, values)[0], base)
        self.assertEqual(vmap(f, in_dims=(0, None))(t, values[0])[0], base)

        # index_put_
        tensor = torch.zeros(3, 3, 4)
        value = torch.ones(3, 2)
        idxs = (
            torch.tensor([[0], [1], [2]]),
            torch.tensor([[0]]),
            torch.tensor([1, 2]),
        )
        expected = torch.index_put_(tensor.clone(), idxs, value)

        def f(t, idx, v):
            torch.index_put_(t, idx, v)
            return t

        self.assertEqual(
            vmap(f, in_dims=(0, (None, None), 0))(tensor, idxs[1:], value), expected
        )
        self.assertEqual(
            vmap(f, in_dims=(0, (None, None), None))(tensor, idxs[1:], value[0]),
            expected,
        )

        # boolean mask
        B = 2
        x = torch.randn(1, 3, 3)
        gy = torch.randn(B, 1, 3, 3)

        def f(x, gy):
            mask = x < 1e-09
            zeros = torch.zeros([])
            index_put = torch.ops.aten.index_put.default(gy, [mask], zeros)
            return index_put

        self.vmap_outplace_test(f, (x, gy), {}, in_dims=(None, 0))

    @onlyCUDA
    @parametrize("inplace", [True, False])
    def test_0d_tensor_index_put(self, device, inplace):
        def f(t, idx, v):
            fn = torch.index_put_ if inplace else torch.index_put
            return fn(t, idx, v)

        N = 2
        t = torch.zeros((N, 5), device="cuda")
        idx = torch.tensor([1, 3])
        v = torch.tensor(1, dtype=t.dtype, device="cpu")

        expected = torch.tensor([[0, 1, 0, 1, 0], [0, 1, 0, 1, 0]], dtype=t.dtype)
        self.assertEqual(expected, vmap(f, in_dims=(0, None, None))(t, (idx,), v))

    @parametrize("training", [True, False])
    @parametrize("track_running_stats", [True, False])
    @parametrize("affine", [True, False])
    def test_batch_norm(self, device, affine, track_running_stats, training):
        if not track_running_stats and not training:
            return

        test = functools.partial(_vmap_test, check_propagates_grad=False)
        BN = torch.nn.BatchNorm2d
        ensemble_size = 10
        hidden_dim = 3

        weights, buffers, _, _, _ = functional_init_with_buffers(BN, [ensemble_size])(
            hidden_dim, affine=affine, track_running_stats=track_running_stats
        )

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

    def test_torch_return_types_returns(self, device):
        t = torch.randn(3, 2, 2, device=device)
        self.assertTrue(
            isinstance(vmap(torch.min, (0, None))(t, 0), torch.return_types.min)
        )
        self.assertTrue(
            isinstance(vmap(torch.max, (0, None))(t, 0), torch.return_types.max)
        )
        self.assertTrue(
            isinstance(
                vmap(torch.topk, (0, None, None))(t, 1, 0), torch.return_types.topk
            )
        )
        self.assertTrue(
            isinstance(vmap(torch.linalg.eig, (0))(t), torch.return_types.linalg_eig)
        )

    def test_namedtuple_returns(self, device):
        Point = namedtuple("Point", ["x", "y"])

        def f(x, y):
            return Point(x=x, y=y)

        x = torch.randn(2, 5, device=device)
        y = torch.randn(2, 3, device=device)
        self.assertTrue(isinstance(vmap(f)(x, y), Point))

    def test_inplace_on_view(self, device):
        def func(leaf):
            base = leaf * leaf
            view = base.transpose(0, 1)
            view[2:4, 2:4] *= 2
            view[0:2, 0:2].diagonal().sin_()
            view = view[1:3, 1:3]
            view.cos_()
            return view

        def push_vjp(leaf, gout):
            _, vjp_fn = vjp(func, leaf)
            (result,) = vjp_fn(gout)
            return result

        leaf = torch.randn(4, 4, device=device)
        gout = torch.randn(2, 2, device=device)
        args = (leaf, gout)

        for (
            batched_args,
            in_dims,
            _,
        ) in generate_vmap_inputs(args, {}):
            if in_dims[1] is None:
                # triggers some composite compliance problem
                continue
            self.vmap_outplace_test(push_vjp, batched_args, {}, in_dims)

    def test_advanced_indexing(self, device):
        def test(f, args):
            for loop_out, batched_out in get_fallback_and_vmap_exhaustive(f, args, {}):
                self.assertEqual(loop_out, batched_out)

        def f(x, idx):
            return x[:, idx]

        def f2(x, idx):
            return x[idx, :]

        def f3(x, idx):
            return x[:, :, idx]

        inps = (
            torch.randn(5, 5, 5, device=device),
            torch.randn(5, 5, 5, 5, device=device),
            torch.randn(5, 5, 5, 5, 5, device=device),
        )
        idxes = (
            torch.tensor([0, 1, 2], device=device),
            torch.tensor([0, 1, 2], device=device).reshape(3, 1),
            torch.tensor([0, 1, 2], device=device).reshape(3, 1, 1),
        )
        for inp, idx in itertools.product(inps, idxes):
            test(f, (inp, idx))
            test(f2, (inp, idx))
            test(f3, (inp, idx))

    def test_nested_advanced_indexing(self, device):
        e = torch.rand(7, 4, device=device)
        idx = torch.tensor([0, 1], device=device).view(2, 1)

        # simple reference implementation for comparison
        def _fake_vmap(f, in_dims=0, out_dims=0):
            def w(input):
                r = [f(input.select(in_dims, i)) for i in range(input.size(in_dims))]
                return torch.stack(r, out_dims)

            return w

        def with_vmap(_vmap):
            def g(idx_):
                def f(e_):
                    return e_[idx_]

                return _vmap(f, in_dims=1)(e)

            r = _vmap(g)(idx)
            return r

        a = with_vmap(vmap)
        b = with_vmap(_fake_vmap)
        self.assertEqual(a, b)

    @ops(
        filter(lambda op: "linalg" in op.name, op_db + additional_op_db),
        allowed_dtypes=(torch.float,),
    )
    @skipOps(
        "TestVmapOperatorsOpInfo",
        "test_vmap_linalg_failure_1D_input",
        {
            xfail("linalg.vector_norm"),  # can accept vector inputs
            xfail("linalg.norm"),  # can accept vector inputs
            xfail("linalg.norm", "subgradients_at_zero"),  # can accept vector inputs
            xfail("linalg.vander"),  # can accept vector inputs
            skip(
                "linalg.multi_dot"
            ),  # accepts list of tensor inputs, has its own special test
            xfail("linalg.vecdot"),
            # throws in vmap on CUDA
            # IndexError: Dimension out of range (expected to be in range of [-1, 0], but got -2)
            # https://github.com/pytorch/pytorch/runs/8110653462?check_suite_focus=true
            # but it passes locally
            xfail("linalg.diagonal"),
            skip("linalg.matrix_norm", ""),
            skip("linalg.ldl_solve", ""),
        },
    )
    def test_vmap_linalg_failure_1D_input(self, device, dtype, op):
        for sample in op.sample_inputs(device, dtype, requires_grad=False):
            if sample.input.dim() != 2 or sample.input.shape[0] == 0:
                continue
            test_input = sample.input[
                0
            ]  # using the sample input avoids numerical inconsistency issues
            with self.assertRaisesRegex(RuntimeError, "dimension"):
                op(test_input, *sample.args, **sample.kwargs)

            def op_wrapper(inp):
                return op(inp, *sample.args, **sample.kwargs)

            # square inputs are more likely to pass linalg checks
            test_input = test_input.expand(test_input.shape[0], test_input.shape[0])
            with self.assertRaisesRegex(RuntimeError, "dimension"):
                return vmap(op_wrapper)(test_input)

    def test_vmap_multi_dot_failure_1D_input(self):
        # special exception for first and last tensors so making giving 3 items avoids special cases
        inputs = (torch.randn(3, 3), torch.randn(3), torch.randn(3, 3))
        with self.assertRaisesRegex(RuntimeError, "tensor 1 must be 2D but got 1D"):
            torch.linalg.multi_dot(inputs)

        # square inputs are more likely to pass linalg checks
        inputs = tuple(i.expand(i.shape[0], i.shape[0]) for i in inputs)
        with self.assertRaisesRegex(RuntimeError, "tensor 1 must be 2D but got 1D"):
            return vmap(torch.linalg.multi_dot)(inputs)

    def test_vmap_escaped_error(self):
        escaped = None

        def f(x):
            nonlocal escaped
            escaped = x
            return x**2

        x = torch.randn([3, 3, 3, 3, 3])
        vmap(f)(x)

        common_message = (
            r"your tensor may have escaped from inside a function being vmapped.*{0}.*"
        )

        # Note: These are not a complete set of tests for all possible functions calling 'vmap_check_escaped'

        with self.assertRaisesRegex(
            RuntimeError, common_message.format("gen_vmap_plumbing")
        ):
            escaped.sin()

        with self.assertRaisesRegex(
            RuntimeError, common_message.format("boxed_tensor_inputs_batch_rule")
        ):
            escaped.sin_()

        with self.assertRaisesRegex(
            RuntimeError, common_message.format("gen_vmap_inplace_plumbing")
        ):
            escaped.mul_(1)

        with self.assertRaisesRegex(
            RuntimeError, common_message.format("binary_cross_entropy_plumbing")
        ):
            torch.nn.functional.binary_cross_entropy(escaped, torch.zeros([3, 3, 3, 3]))

        with self.assertRaisesRegex(
            RuntimeError, common_message.format("boxed_existing_bdim_all_batch_rule")
        ):
            torch.nn.functional.adaptive_max_pool2d(escaped, output_size=(1, 1))

        with self.assertRaisesRegex(
            RuntimeError, common_message.format("boxed_reduction_batch_rule")
        ):
            escaped.argmin()

        a = torch.zeros([4, 4, 4, 4])
        b = torch.zeros([4, 4, 4, 4], dtype=torch.long)
        with self.assertRaisesRegex(
            RuntimeError, common_message.format("boxed_all_tensors_have_optional_bdim")
        ):
            torch.ops.aten.adaptive_max_pool2d_backward(escaped, a, b)

        vmap(f)(torch.tensor([[0, 0], [0, 0]], dtype=torch.int))
        with self.assertRaisesRegex(
            RuntimeError, common_message.format("gen_vmap_plumbing_no_returns")
        ):
            torch.ops.aten._linalg_check_errors(escaped, "linalg.inv", is_matrix=False)

    def test_vmap_with_anomaly_detection(self):
        with torch.autograd.set_detect_anomaly(True):
            x = torch.zeros(3) - 1

            def fn(x):
                return x.sum()

            per_sample_grad = vmap(grad(fn))(x)
            self.assertEqual(per_sample_grad, torch.ones_like(x))

            def bad_fn(x):
                return x.sqrt().sum()

            err_msg = "Function 'SqrtBackward0' returned nan values in its 0th output."
            with self.assertRaisesRegex(RuntimeError, err_msg):
                vmap(grad(bad_fn))(x)

    def test_searchsorted_bucketize(self, device):
        # OpInfo generates test with repeated samples in batch dim.
        # Thus we test explicitly with different samples across a batch.

        def test():
            boundaries = torch.tensor(
                [[1, 4, 5, 7, 9], [1, 2, 6, 8, 10]], device=device
            )
            v = torch.tensor(3, device=device)
            self.vmap_outplace_test(torch.searchsorted, (boundaries, v), {}, (0, None))
            self.vmap_outplace_test(torch.bucketize, (v, boundaries), {}, (None, 0))
            boundaries = torch.tensor([[1, 4, 5, 7, 9], [1, 2, 4, 8, 9]], device=device)
            v = torch.tensor([3, 4], device=device)
            self.vmap_outplace_test(torch.searchsorted, (boundaries, v), {}, (0, 0))
            self.vmap_outplace_test(torch.bucketize, (v, boundaries), {}, (0, 0))

        test()


@markDynamoStrictTest
class TestRandomness(TestCase):
    def _reset_random(self, generator, orig_state, use_generator, seed):
        return (
            generator.set_state(orig_state)
            if use_generator
            else torch.manual_seed(seed)
        )

    def _get_image(self, batched_input, batch_size, device):
        if batched_input == "first":
            return torch.ones([batch_size, 3, 3, 14, 14], device=device)
        if batched_input == "last":
            return torch.ones([3, 3, 14, 14, batch_size], device=device)
        assert batched_input == "none"
        return torch.ones([3, 3, 14, 14], device=device)

    def _assert_all_slices_equal(self, tensor):
        expected = tensor[0]
        self.assertTrue((tensor == expected).all())

    def _assert_all_slices_unique(self, tensor):
        B0 = tensor.shape[0]
        slices_equal = vmap(vmap(lambda x, y: (x == y).all(), (0, None)), (None, 0))(
            tensor, tensor
        )
        assert slices_equal.shape == (B0, B0)
        slices_equal.diagonal().zero_()
        self.assertEqual(slices_equal, torch.zeros_like(slices_equal))

    def _assert_throws_in_error_mode(self, fn, args, in_dims):
        with self.assertRaisesRegex(
            RuntimeError, r"called random operation while in randomness error mode"
        ):
            vmap(fn, in_dims=in_dims, randomness="error")(*args)

    def _assert_throws_in_different_mode_inplace(self, fn, args, in_dims):
        with self.assertRaisesRegex(
            RuntimeError, r"different inplace randomness on an unbatched tensor"
        ):
            vmap(fn, in_dims=in_dims, randomness="different")(*args)

    def _assert_throws_in_same_mode_batched(self, fn, args, in_dims):
        with self.assertRaisesRegex(
            RuntimeError,
            r"Vmap does not currently support same randomness with a batched tensor input",
        ):
            vmap(fn, in_dims=in_dims, randomness="same")(*args)

    def _in_dims(self, *batched_strings):
        def get_in_dim(batched_string):
            if batched_string == "first":
                return 0
            if batched_string == "last":
                return -1
            assert batched_string == "none"
            return None

        batched_strings = batched_strings + (
            "first",
        )  # for the always batched as first dim dummy argument
        return tuple(get_in_dim(batched_string) for batched_string in batched_strings)

    @parametrize("randomness", ["same", "different", "error"])
    @parametrize("use_generator", [True, False])
    def test_factory_ops(self, device, randomness, use_generator):
        generator = torch.Generator(device=device)
        orig_state = generator.get_state()
        kwargs = (
            {"device": device, "generator": generator}
            if use_generator
            else {"device": device}
        )
        ops = [
            lambda _, shape: torch.randn(shape, **kwargs),
            lambda _, shape: torch.rand(shape, **kwargs),
            lambda _, shape: torch.randint(100, shape, **kwargs),
            lambda _, shape: torch.randint(5, 100, shape, **kwargs),
            lambda _, shape: torch.normal(0.0, 1.0, shape, **kwargs),
        ]
        B0 = 4
        shape = (3, 3)
        seed = 1234567

        for op in ops:
            passed = torch.randn(B0, device=device)
            if randomness == "error":
                self._assert_throws_in_error_mode(
                    op, (passed, shape), in_dims=(0, None)
                )
                return

            generator = self._reset_random(generator, orig_state, use_generator, seed)
            vmap_result = vmap(op, in_dims=(0, None), randomness=randomness)(
                passed, shape
            )

            generator = self._reset_random(generator, orig_state, use_generator, seed)
            if randomness == "different":
                expected = op(passed, [B0, *shape])
                self._assert_all_slices_unique(vmap_result)
                self.assertEqual(vmap_result, expected)
            else:
                expected = op(passed, shape)
                self._assert_all_slices_equal(vmap_result)
                for i in range(B0):
                    self.assertEqual(vmap_result[i], expected)

    @parametrize("randomness", ["same", "different", "error"])
    @parametrize("use_generator", [True, False])
    def test_randperm(self, device, randomness, use_generator):
        # needs a special case because randperm doesn't take a batch size
        B0 = 4
        seed = 1234567
        passed = torch.randn(B0, device=device)

        torch.manual_seed(seed)
        generator = torch.Generator(device=device)
        orig_state = generator.get_state()

        kwargs = (
            {"device": device, "generator": generator}
            if use_generator
            else {"device": device}
        )

        if randomness == "error":
            with self.assertRaisesRegex(
                RuntimeError, r"called random operation while in randomness error mode"
            ):
                vmap(lambda _: torch.randperm(10, **kwargs), randomness=randomness)(
                    passed
                )
            return

        vmap_result = vmap(
            lambda _: torch.randperm(10, **kwargs), randomness=randomness
        )(passed)
        generator = generator.set_state(orig_state)
        torch.manual_seed(seed)
        if randomness == "different":
            for i in range(B0):
                expected = torch.randperm(10, **kwargs)
                # RNG differs between eager and via dynamo trace on CUDA
                if TEST_WITH_TORCHDYNAMO and torch.device(device).type == "cuda":
                    self._assert_all_slices_unique(vmap_result)
                else:
                    self.assertEqual(vmap_result[i], expected)
        else:
            expected = torch.randperm(10, **kwargs)
            # RNG differs between eager and via dynamo trace on CUDA
            if TEST_WITH_TORCHDYNAMO and torch.device(device).type == "cuda":
                self._assert_all_slices_equal(vmap_result)
            else:
                for i in range(B0):
                    self.assertEqual(vmap_result[i], expected)

    @parametrize("randomness", ["error", "same", "different"])
    @parametrize("batched_input", ["first", "last", "none"])
    def test_dropout(self, device, randomness, batched_input):
        def op(t, ignored):
            return torch.nn.functional.dropout(torch.ones_like(t), training=True)

        B0 = 4
        always_batched = torch.randn((B0,))
        passed = self._get_image(batched_input, B0, device)
        in_dims = self._in_dims(batched_input)

        if randomness == "error":
            with self.assertRaisesRegex(
                RuntimeError, r"called random operation while in randomness error mode"
            ):
                vmap(op, randomness=randomness, in_dims=in_dims)(passed, always_batched)
            return

        vmap_result = vmap(op, randomness=randomness, in_dims=in_dims)(
            passed, always_batched
        )

        # Check that the randomness is within bounds...
        # ideally this is close to 0.5
        p_estimate = vmap_result.mean() / 2
        self.assertTrue(p_estimate < 0.75)
        self.assertTrue(p_estimate > 0.25)

        if randomness == "different":
            self._assert_all_slices_unique(vmap_result)
            return

        assert randomness == "same"
        self._assert_all_slices_equal(vmap_result)

    @parametrize("randomness", ["error", "same", "different"])
    @parametrize("batched_input", ["first", "last", "none"])
    def test_alpha_dropout(self, device, randomness, batched_input):
        def op(t, ignored):
            return torch.nn.functional.alpha_dropout(torch.ones_like(t), training=True)

        B0 = 4
        always_batched = torch.randn((B0,))
        passed = self._get_image(batched_input, B0, device)
        in_dims = self._in_dims(batched_input)

        if randomness == "error":
            with self.assertRaisesRegex(
                RuntimeError, r"called random operation while in randomness error mode"
            ):
                vmap(op, randomness=randomness, in_dims=in_dims)(passed, always_batched)
            return

        # I have no clue how to actually test correctness of alpha dropout because the docs
        # seem wrong: https://github.com/pytorch/pytorch/issues/74004
        vmap_result = vmap(op, randomness=randomness, in_dims=in_dims)(
            passed, always_batched
        )
        if randomness == "different":
            self._assert_all_slices_unique(vmap_result)
            return

        assert randomness == "same"
        self._assert_all_slices_equal(vmap_result)

    @parametrize("randomness", ["error", "same", "different"])
    @parametrize("batched_input", ["first", "last", "none"])
    @parametrize("dim", [2, 3])
    def test_feature_dropout(self, device, randomness, batched_input, dim):
        def op(t, ignored):
            f = (
                torch.nn.functional.dropout2d
                if dim == 2
                else torch.nn.functional.dropout3d
            )
            return f(torch.ones_like(t), training=True)

        B0 = 4
        always_batched = torch.randn((B0,))
        passed = self._get_image(batched_input, B0, device)
        if dim == 3:
            unsqueeze_dim = -2 if batched_input == "last" else -1
            passed = passed.unsqueeze(unsqueeze_dim)
        in_dims = self._in_dims(batched_input)

        if randomness == "error":
            with self.assertRaisesRegex(
                RuntimeError, r"called random operation while in randomness error mode"
            ):
                vmap(op, randomness=randomness, in_dims=in_dims)(passed, always_batched)
            return

        vmap_result = vmap(op, randomness=randomness, in_dims=in_dims)(
            passed, always_batched
        )

        # Check the "feature" pattern
        dims = [-1, -2] if dim == 2 else [-1, -2, -3]
        planes_numel = (
            2
            * vmap_result.numel()
            / (vmap_result.shape[0] * vmap_result.shape[1] * vmap_result.shape[2])
        )
        planes = vmap_result.sum(dims)
        result = (planes == 0) ^ (planes == planes_numel)
        self.assertEqual(result, torch.ones_like(result, dtype=torch.bool))

        if randomness == "different":
            self._assert_all_slices_unique(vmap_result)
            return

        assert randomness == "same"
        self._assert_all_slices_equal(vmap_result)

    @parametrize("randomness", ["error", "same", "different"])
    @parametrize("batched_input", ["first", "last", "none"])
    def test_feature_alpha_dropout(self, device, randomness, batched_input):
        def op(t, ignored):
            return torch.nn.functional.feature_alpha_dropout(
                torch.ones_like(t), training=True
            )

        B0 = 4
        always_batched = torch.randn((B0,))
        passed = self._get_image(batched_input, B0, device)
        unsqueeze_dim = -2 if batched_input == "last" else -1
        passed = passed.unsqueeze(unsqueeze_dim)
        in_dims = self._in_dims(batched_input)

        if randomness == "error":
            with self.assertRaisesRegex(
                RuntimeError, r"called random operation while in randomness error mode"
            ):
                vmap(op, randomness=randomness, in_dims=in_dims)(passed, always_batched)
            return

        vmap_result = vmap(op, randomness=randomness, in_dims=in_dims)(
            passed, always_batched
        )

        # I have no clue how to actually test correctness of alpha dropout because the docs
        # seem wrong: https://github.com/pytorch/pytorch/issues/74004

        # Check the "feature" pattern
        dims = [-1, -2, -3]
        planes = vmap_result.sum(dims)
        max_elt = planes.max()
        min_elt = planes.min()
        result = (planes == min_elt) ^ (planes == max_elt)
        self.assertEqual(result, torch.ones_like(result, dtype=torch.bool))

        if randomness == "different":
            self._assert_all_slices_unique(vmap_result)
            return

        assert randomness == "same"
        self._assert_all_slices_equal(vmap_result)

    @parametrize("randomness", ["error", "same", "different"])
    @parametrize("batched_input", ["first", "last", "none"])
    def test_like_functions(self, device, randomness, batched_input):
        seed = 1234567
        supported_ops = [
            lambda t, _: torch.randint_like(t, 20),
            lambda t, _: torch.randint_like(t, 0, 20),
            lambda t, _: torch.rand_like(t),
            lambda t, _: torch.randn_like(t),
        ]
        B0 = 4

        for op in supported_ops:
            always_batched = torch.randn(B0)
            passed = self._get_image(batched_input, B0, device)
            in_dims = self._in_dims(batched_input)

            if randomness == "error":
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"called random operation while in randomness error mode",
                ):
                    vmap(op, in_dims=in_dims, randomness=randomness)(
                        passed, always_batched
                    )
                return

            torch.manual_seed(seed)
            vmap_result = vmap(op, randomness=randomness, in_dims=in_dims)(
                passed, always_batched
            )

            torch.manual_seed(seed)

            if batched_input == "last":
                passed = passed.movedim(-1, 0)
            if randomness == "different":
                if batched_input == "none":
                    passed = passed.expand(B0, *passed.shape)
                expected = op(passed, 0)

                self._assert_all_slices_unique(vmap_result)
                # RNG differs between eager and via dynamo trace on CUDA
                if not (TEST_WITH_TORCHDYNAMO and torch.device(device).type == "cuda"):
                    self.assertEqual(expected, vmap_result)
                return

            assert randomness == "same"
            if batched_input != "none":
                passed = passed[0]
            expected = op(passed, 0)
            self._assert_all_slices_equal(vmap_result)
            # RNG differs between eager and via dynamo trace on CUDA
            if not (TEST_WITH_TORCHDYNAMO and torch.device(device).type == "cuda"):
                for i in range(B0):
                    self.assertEqual(expected, vmap_result[i])

    @parametrize("use_generator", [True, False])
    @parametrize("randomness", ["error", "same", "different"])
    @parametrize("batched_input", ["first", "last", "none"])
    def test_random_unary_inplace(
        self, device, use_generator, randomness, batched_input
    ):
        generator = torch.Generator(device=device)
        orig_state = generator.get_state()
        kwargs = {"generator": generator} if use_generator else {}
        ops = [
            lambda t, _: t.random_(**kwargs),
            lambda t, _: t.random_(100, **kwargs),
            lambda t, _: t.random_(-5, 100, **kwargs),
            lambda t, _: t.normal_(**kwargs),
            lambda t, _: t.bernoulli_(**kwargs),
            lambda t, _: t.cauchy_(**kwargs),
            lambda t, _: t.exponential_(**kwargs),
            lambda t, _: t.geometric_(0.5, **kwargs),
            lambda t, _: t.log_normal_(**kwargs),
            lambda t, _: t.uniform_(**kwargs),
        ]
        B0 = 4
        seed = 1234567
        in_dims = self._in_dims(batched_input)

        for op in ops:
            # because of in place updates, clone inputs
            always_batched = torch.randn(B0, device=device)
            passed = self._get_image(batched_input, B0, device)
            passed_expected = passed.clone()

            if randomness == "error":
                self._assert_throws_in_error_mode(
                    op, (passed, always_batched), in_dims=in_dims
                )
                return
            if randomness == "different" and batched_input == "none":
                self._assert_throws_in_different_mode_inplace(
                    op, (passed, always_batched), in_dims=in_dims
                )
                return

            generator = self._reset_random(generator, orig_state, use_generator, seed)
            vmap_result = vmap(op, in_dims=in_dims, randomness=randomness)(
                passed, always_batched
            )

            if batched_input == "last":
                passed_expected = passed_expected.movedim(-1, 0)
            generator = self._reset_random(generator, orig_state, use_generator, seed)
            if randomness == "different":
                expected = op(passed_expected, always_batched)
                self._assert_all_slices_unique(vmap_result)
                self.assertEqual(vmap_result, expected)
            else:
                if batched_input != "none":
                    passed_expected = passed_expected[
                        0
                    ].clone()  # bug in pytorch, normal_ on views doesn't work
                expected = op(passed_expected, always_batched)
                self._assert_all_slices_equal(vmap_result)
                for i in range(B0):
                    self.assertEqual(vmap_result[i], expected)

    @parametrize("use_generator", [True, False])
    @parametrize("randomness", ["error", "same", "different"])
    @parametrize("batched_input", ["first", "last", "none"])
    @parametrize("batched_probability", ["first", "last", "none"])
    def test_bernoulli_in_place(
        self, device, use_generator, randomness, batched_input, batched_probability
    ):
        B0 = 4
        seed = 1234567
        generator = torch.Generator(device=device)
        orig_state = generator.get_state()
        kwargs = {"generator": generator} if use_generator else {}
        in_dims = self._in_dims(batched_input, batched_probability)

        def op(t, p, ignored):
            return t.bernoulli_(p, **kwargs)

        # because of in place updates, clone inputs
        always_batched = torch.randn(B0, device=device)
        input = self._get_image(batched_input, B0, device)
        input_expected = input.clone()
        probability = self._get_image(batched_probability, B0, device) - 0.5

        if randomness == "error":
            self._assert_throws_in_error_mode(
                op, (input, probability, always_batched), in_dims=in_dims
            )
            return
        if randomness == "same" and batched_probability != "none":
            self._assert_throws_in_same_mode_batched(
                op, (input, probability, always_batched), in_dims=in_dims
            )
            return
        if batched_input == "none" and batched_probability != "none":
            regex = r"there exists a Tensor `other` in extra_args that has more elements than `self`"
            with self.assertRaisesRegex(RuntimeError, regex):
                vmap(op, in_dims=in_dims, randomness=randomness)(
                    input, probability, always_batched
                )
            return
        if randomness == "different" and batched_input == "none":
            self._assert_throws_in_different_mode_inplace(
                op, (input, probability, always_batched), in_dims=in_dims
            )
            return

        self._reset_random(generator, orig_state, use_generator, seed)
        vmap_result = vmap(op, in_dims=in_dims, randomness=randomness)(
            input, probability, always_batched
        )

        self._reset_random(generator, orig_state, use_generator, seed)
        if batched_input == "last":
            input_expected = input_expected.movedim(-1, 0)
        if batched_probability == "last":
            probability = probability.movedim(-1, 0)
        if randomness == "different":
            expected = op(input_expected, probability, always_batched)
            self._assert_all_slices_unique(vmap_result)
            self.assertEqual(vmap_result, expected)
        else:
            if batched_input != "none":
                input_expected = input_expected[0]
            expected = op(input_expected, probability, always_batched)
            self._assert_all_slices_equal(vmap_result)
            for i in range(B0):
                self.assertEqual(vmap_result[i], expected)

    @parametrize("use_generator", [True, False])
    @parametrize("randomness", ["error", "same", "different"])
    @parametrize("batched_input", ["first", "last", "none"])
    @parametrize("batched_other", ["first", "last", "none"])
    def test_random_binary_out_of_place(
        self, device, use_generator, randomness, batched_input, batched_other
    ):
        generator = torch.Generator(device=device)
        orig_state = generator.get_state()
        kwargs = {"generator": generator} if use_generator else {}
        ops = [
            lambda t, o, _: torch.normal(t, o, **kwargs),
            lambda t, o, _: torch.binomial(t, (o - 0.5), **kwargs),
        ]

        B0 = 4
        seed = 1234567
        in_dims = self._in_dims(batched_input, batched_other)

        for op in ops:
            always_batched = torch.randn(B0, device=device)
            input = self._get_image(batched_input, B0, device)
            other = self._get_image(batched_other, B0, device)

            if randomness == "error":
                self._assert_throws_in_error_mode(
                    op, (input, other, always_batched), in_dims=in_dims
                )
                return
            if randomness == "same" and (
                batched_input != "none" or batched_other != "none"
            ):
                self._assert_throws_in_same_mode_batched(
                    op, (input, other, always_batched), in_dims=in_dims
                )
                return

            generator = self._reset_random(generator, orig_state, use_generator, seed)
            vmap_result = vmap(op, in_dims=in_dims, randomness=randomness)(
                input, other, always_batched
            )

            if batched_input == "last":
                input = input.movedim(-1, 0)
            if batched_other == "last":
                other = other.movedim(-1, 0)

            generator = self._reset_random(generator, orig_state, use_generator, seed)
            if randomness == "different":
                if batched_input == "none":
                    input = input.expand(B0, *input.shape)
                expected = op(input, other, always_batched)
                self._assert_all_slices_unique(vmap_result)
                self.assertEqual(vmap_result, expected)
            else:
                assert batched_input == "none" and batched_other == "none"
                expected = op(input, other, always_batched)
                self._assert_all_slices_equal(vmap_result)
                for i in range(B0):
                    self.assertEqual(vmap_result[i], expected)

    @parametrize("use_generator", [True, False])
    @parametrize("randomness", ["error", "same", "different"])
    @parametrize("batched_input", ["first", "last", "none"])
    def test_random_unary_out_of_place(
        self, device, use_generator, randomness, batched_input
    ):
        generator = torch.Generator(device=device)
        orig_state = generator.get_state()
        kwargs = {"generator": generator} if use_generator else {}
        ops = [
            lambda t, _: torch.normal(0.0, torch.abs(t), **kwargs),
            lambda t, _: torch.normal(t, 1.0, **kwargs),
            lambda t, _: torch.bernoulli(t - 0.5, **kwargs),
            lambda t, _: torch.bernoulli(t, 0.5, **kwargs),
            lambda t, _: torch._standard_gamma(t, **kwargs),
            lambda t, _: torch._sample_dirichlet(t, **kwargs),
            lambda t, _: torch.poisson(t, **kwargs),
        ]

        B0 = 4
        seed = 1234567
        in_dims = self._in_dims(batched_input)

        for op in ops:
            always_batched = torch.randn(B0, device=device)
            passed = self._get_image(batched_input, B0, device)
            if randomness == "error":
                self._assert_throws_in_error_mode(
                    op, (passed, always_batched), in_dims=in_dims
                )
                return
            if randomness == "same" and batched_input != "none":
                self._assert_throws_in_same_mode_batched(
                    op, (passed, always_batched), in_dims=in_dims
                )
                return

            generator = self._reset_random(generator, orig_state, use_generator, seed)
            vmap_result = vmap(op, in_dims=in_dims, randomness=randomness)(
                passed, always_batched
            )

            generator = self._reset_random(generator, orig_state, use_generator, seed)
            if randomness == "different":
                if batched_input == "none":
                    passed = passed.expand(B0, *passed.shape)
                if batched_input == "last":
                    passed = passed.movedim(-1, 0)
                expected = op(passed, always_batched)
                self._assert_all_slices_unique(vmap_result)
                self.assertEqual(vmap_result, expected)
            else:
                expected = op(passed, always_batched)
                self._assert_all_slices_equal(vmap_result)
                for i in range(B0):
                    self.assertEqual(vmap_result[i], expected)

    @parametrize("use_generator", [True, False])
    @parametrize("randomness", ["error", "same", "different"])
    @parametrize("batched_call", [True, False])
    @parametrize("batched_input", ["first", "last", "none"])
    def test_multinomial(
        self, device, use_generator, randomness, batched_call, batched_input
    ):
        def flatten_input(input, batch_call, batch_location):
            if batch_call and batch_location != "none":
                final_size = 3  # [B0, B, N]
            elif not batch_call and batch_location == "none":
                final_size = 1  # [N]
            else:
                final_size = 2  # [B0, N] or [B, N]

            start_idx = final_size - 1
            end_idx = -1
            if batch_location == "last":
                start_idx -= 1
                end_idx -= (
                    1  # gets to correct final size because using negative indices
                )

            ret = input.flatten(start_idx, end_idx)
            assert ret.dim() == final_size
            return ret

        def op(input, _):
            return torch.multinomial(input, 10, **kwargs)

        generator = torch.Generator(device=device)
        orig_state = generator.get_state()
        kwargs = {"generator": generator} if use_generator else {}

        B0 = 4
        seed = 1234567
        in_dims = self._in_dims(batched_input)

        always_batched = torch.randn(B0, device=device)
        passed = self._get_image(batched_input, B0, device)
        passed = flatten_input(passed, batched_call, batched_input)
        if randomness == "error":
            self._assert_throws_in_error_mode(
                op, (passed, always_batched), in_dims=in_dims
            )
            return
        if randomness == "same" and batched_input != "none":
            self._assert_throws_in_same_mode_batched(
                op, (passed, always_batched), in_dims=in_dims
            )
            return

        generator = self._reset_random(generator, orig_state, use_generator, seed)
        vmap_result = vmap(op, in_dims=in_dims, randomness=randomness)(
            passed, always_batched
        )

        generator = self._reset_random(generator, orig_state, use_generator, seed)

        if randomness == "different":
            if batched_input == "none":
                passed = passed.expand(B0, *passed.shape)
            if batched_input == "last":
                passed = passed.movedim(-1, 0)
            orig_passed_size = passed.shape[:2] if batched_call else passed.shape[:1]
            passed = passed.flatten(0, 1) if batched_call else passed
            expected = op(passed, always_batched)
            expected = expected.reshape(*orig_passed_size, 10)
            self._assert_all_slices_unique(vmap_result)
            self.assertEqual(vmap_result, expected)
        else:
            expected = op(passed, always_batched)
            self._assert_all_slices_equal(vmap_result)
            for i in range(B0):
                self.assertEqual(vmap_result[i], expected)

    def test_unsupported_random(self, device):
        x = torch.randn(3, device=device)
        y = x.abs()
        z = x.abs()
        with self.assertRaisesRegex(RuntimeError, "calling out variants"):

            def f(x):
                return torch.randn(3, device=device, out=y)

            vmap(f, randomness="same")(x)
        with self.assertRaisesRegex(RuntimeError, "calling out variants"):

            def f(x0, x1):
                return torch.normal(x, y, out=x)

            vmap(f, randomness="same")(z, z)
        with self.assertRaisesRegex(RuntimeError, "do not yet support"):

            def f(z):
                return torch.rrelu(x)

            vmap(f, randomness="same")(z)

    @parametrize("in_dim", [0, 1, 2])
    @parametrize("out_dim", [0, 1, 2])
    def test_chunk_vmap(self, in_dim, out_dim):
        randomness = "different"

        x = torch.randn(4, 5, 6)

        def f(x):
            y = x.sin() + torch.rand_like(x)
            return y

        for chunks in [1, 2, 3, 4, 7, 10, 16]:
            output = chunk_vmap(
                f,
                in_dims=in_dim,
                out_dims=out_dim,
                randomness=randomness,
                chunks=chunks,
            )(x)
            self._assert_all_slices_unique(output)

    @parametrize("in_dim", [0, 1, 2])
    @parametrize("out_dim", [0, 1, 2])
    def test_vmap_chunksize(self, in_dim, out_dim):
        randomness = "different"

        x = torch.randn(4, 5, 6)

        def f(x):
            y = x.sin() + torch.rand_like(x)
            return y

        for chunk_size in [1, 2, 3, 4, 7, 10, 16, 100]:
            output = vmap(
                f,
                in_dims=in_dim,
                out_dims=out_dim,
                randomness=randomness,
                chunk_size=chunk_size,
            )(x)
            self._assert_all_slices_unique(output)

    def test_jacfwd_with_random(self):
        # checks on behavior are above, this just checks that jacfwd respects
        # the randomness param

        x = torch.rand(3, 4)
        with self.assertRaisesRegex(
            RuntimeError, r"called random operation while in randomness error mode"
        ):
            jacfwd(torch.bernoulli)(x)

        # x isn't batched so use bernoulli since it doesn't do inplace randomness
        jacfwd(torch.bernoulli, randomness="same")(x)
        jacfwd(torch.bernoulli, randomness="different")(x)

    @parametrize("randomness", ["error", "same", "different"])
    def test_dropout_unbatched(self, device, randomness):
        x = torch.randn(3, device=device)
        y = torch.randn(1, 3, device=device)

        def fn(x, y):
            # output from dropout should be a Tensor[B, 1, 3] (B=3)
            return x + torch.nn.functional.dropout(y, p=0.5).mean(1)

        # We just verify that this doesn't raise an error for
        # `same` and `different` randomness.
        # Ref: https://github.com/pytorch/pytorch/issues/92283
        context = (
            self.assertRaises(RuntimeError)
            if randomness == "error"
            else contextlib.nullcontext()
        )
        with context:
            vmap(fn, in_dims=(0, None), randomness=randomness)(x, y)


@markDynamoStrictTest
class TestTransformFailure(TestCase):
    @skipIfTorchDynamo()
    @parametrize(
        "transform",
        ["vmap", "grad", "grad_and_value", "vjp", "jvp", "jacrev", "jacfwd"],
    )
    def test_fails_with_autograd_function(self, device, transform):
        failed_build_envs = ("linux-focal-py3.8-clang10", "linux-focal-py3.11-clang10")
        if (
            device == "cpu"
            and transform in ["grad", "vmap"]
            and TEST_WITH_TORCHDYNAMO
            and os.getenv("BUILD_ENVIRONMENT", "") in failed_build_envs
        ):
            raise unittest.SkipTest(
                "Unexpected successes on focal with dynamo,"
                + " see https://github.com/pytorch/pytorch/issues/107173"
            )

        class Test(torch.autograd.Function):
            @staticmethod
            def forward(_, input):
                return input

            @staticmethod
            def backward(_, grad_input):
                return grad_input

        transform = getattr(functorch, transform)

        def f(x):
            return Test.apply(x)

        if transform in (grad, grad_and_value):
            input = torch.tensor(4.0)
        else:
            input = torch.randn(5)

        if transform == vjp:
            transform = functools.partial(transform, f)
        elif transform == jvp:
            input = (input,)
            transform = functools.partial(transform, f, input)
        else:
            transform = transform(f)

        with self.assertRaisesRegex(RuntimeError, "autograd.Function"):
            transform(input)


@markDynamoStrictTest
class TestVmapDeviceType(Namespace.TestVmapBase):
    def _vmap_test(self, *args, **kwargs):
        return _vmap_test(self, *args, **kwargs)

    def test__is_all_true(self, device):
        def test():
            def f(x, *, expected_result):
                result = torch.ops.aten._is_all_true(x)
                self.assertFalse(torch._C._functorch.is_batchedtensor(result))
                self.assertEqual(result.shape, torch.Size([]))
                self.assertEqual(result.item(), expected_result)
                return result

            x = torch.rand(10, device=device)
            vmap(f)(x >= 0, expected_result=True)
            vmap(f)(x < 0, expected_result=False)

            x[random.choice(range(10))] *= -1
            vmap(f)(x >= 0, expected_result=False)
            vmap(f)(x < 0, expected_result=False)

            x = -torch.rand(10, device=device)
            vmap(f)(x > 0, expected_result=False)
            vmap(f)(x <= 0, expected_result=True)

        check_vmap_fallback(self, test, torch._is_all_true)

    def test__is_any_true(self, device):
        def test():
            def f(x, *, expected_result):
                result = torch.ops.aten._is_any_true(x)
                self.assertFalse(torch._C._functorch.is_batchedtensor(result))
                self.assertEqual(result.shape, torch.Size([]))
                self.assertEqual(result.item(), expected_result)
                return result

            x = torch.zeros(10, device=device, dtype=torch.bool)
            vmap(f)(x > 0, expected_result=False)

            x[5] = True
            vmap(f)(x > 0, expected_result=True)
            vmap(f)(x[1::2], expected_result=True)
            vmap(f)(x[0::2], expected_result=False)

        check_vmap_fallback(self, test, torch._is_any_true)

    def test_check_tensor(self, device):
        def test():
            test_sizes = [
                (1,),
                (10,),
                (1, 1),
                (1, 10),
                (10, 1),
                (10, 10),
                (1, 1, 1),
                (10, 1, 1),
                (1, 10, 1),
                (10, 10, 10),
            ]

            def check_gte_0(t):
                return torch._test_check_tensor(t >= 0)

            error_message = "Test message for TORCH_CHECK_TENSOR_ALL"

            for size in test_sizes:
                t_all_gte_0 = torch.rand(size, device=device)
                t_all_lt_0 = t_all_gte_0 - 1

                vmap(check_gte_0)(t_all_gte_0)

                if len(size) >= 2:
                    vmap(vmap(check_gte_0))(t_all_gte_0)

                with self.assertRaisesRegex(RuntimeError, error_message):
                    vmap(check_gte_0)(t_all_lt_0)

                if len(size) >= 2:
                    with self.assertRaisesRegex(RuntimeError, error_message):
                        vmap(vmap(check_gte_0))(t_all_lt_0)

                if t_all_gte_0.numel() > 1:
                    t_all_gte_0_but_one = t_all_gte_0.clone()
                    idx = (random.choice(range(dim_size)) for dim_size in size)
                    t_all_gte_0_but_one[(..., *idx)] = -1

                    with self.assertRaisesRegex(RuntimeError, error_message):
                        vmap(check_gte_0)(t_all_gte_0_but_one)

                    if len(size) >= 2:
                        with self.assertRaisesRegex(RuntimeError, error_message):
                            vmap(vmap(check_gte_0))(t_all_gte_0_but_one)

        check_vmap_fallback(self, test, torch._test_check_tensor)


@markDynamoStrictTest
class TestVmapNestedTensor(Namespace.TestVmapBase):
    def _vmap_test(self, *args, **kwargs):
        return _vmap_test(self, *args, **kwargs)

    # dims should be something like [5, None, 10], with None indicating that a
    # random ragged structure should be used
    def _create_nt(self, dims, device):
        sizes = [
            [
                d if d is not None else torch.randint(2, 10, size=(1,)).item()
                for d in dims[1:]
            ]
            for d in range(dims[0])
        ]
        return torch.nested.nested_tensor(
            [torch.randn(*size) for size in sizes], device=device
        )

    # Creates an NT matching another NT's number of components and
    # shape / ragged structure for all dims specified to be -1.
    def _nt_from_similar(self, other, dims):
        assert len(dims) == other.dim()
        assert dims[0] == -1 or dims[0] == other.size(0)

        ret_sizes = []
        for t in other.unbind():
            other_size = t.shape
            ret_size = []
            for i, d in enumerate(dims[1:]):
                if d == -1:
                    ret_size.append(other_size[i])
                else:
                    ret_size.append(d)
            ret_sizes.append(ret_size)

        return torch.nested.nested_tensor(
            [torch.randn(*size) for size in ret_sizes], device=other.device
        )

    @allowVmapFallbackUsage
    def test_fallback_unary(self, device):
        def f(x):
            return x.sin() * 5.0 + 4.0

        nt = self._create_nt([4, None, 3], device=device)
        self._vmap_test(f, (nt,))

    @allowVmapFallbackUsage
    def test_fallback_binary(self, device):
        def f(x, y):
            return x @ y

        x = self._create_nt([5, None, 3], device=device)
        y = self._create_nt([5, 3, None], device=device)
        self._vmap_test(f, (x, y))

    @allowVmapFallbackUsage
    def test_fallback_binary_nt_and_unbatched_dense(self, device):
        def f(x, y):
            return x @ y

        x = self._create_nt([5, None, 3], device=device)
        y = torch.randn(3, 4, device=device)
        self._vmap_test(f, (x, y), in_dims=(0, None))

    @allowVmapFallbackUsage
    def test_fallback_binary_nt_and_batched_dense(self, device):
        def f(x, y):
            return x @ y

        x = self._create_nt([5, None, 3], device=device)
        y = torch.randn(5, 3, 4, device=device)
        self._vmap_test(f, (x, y))

    def test_nt_acts_as_dense_in_vmap(self, device):
        def f(x):
            assert not x.is_nested
            return x

        x = self._create_nt([5, None, 3], device=device)
        self._vmap_test(f, (x,))

    def test_cat_batching_rule(self, device):
        def f(x, y, dim):
            return torch.cat([x, y], dim=dim)

        # Different nested structure, same other dims
        x = self._create_nt([3, None, 2], device=device)
        y = self._create_nt([3, None, 2], device=device)
        self._vmap_test(functools.partial(f, dim=0), (x, y))

        x = self._create_nt([3, 2, None], device=device)
        y = self._create_nt([3, 2, None], device=device)
        self._vmap_test(functools.partial(f, dim=1), (x, y))

        # Same nested structure, different other dims
        x = self._create_nt([3, 2, None], device=device)
        y = self._nt_from_similar(x, [-1, 4, -1])
        self._vmap_test(functools.partial(f, dim=0), (x, y))

        x = self._create_nt([3, None, 2], device=device)
        y = self._nt_from_similar(x, [-1, -1, 4])
        self._vmap_test(functools.partial(f, dim=1), (x, y))

    # .shape calls don't work on NTs
    # TODO: Fix this somehow?
    @unittest.expectedFailure
    def test_shape_call(self, device):
        def f(x):
            x.shape[0]
            return x

        x = self._create_nt([3, None, 2])
        self._vmap_test(f, (x,))

    def test_nt_with_nonzero_in_dim_raises(self, device):
        def f(x):
            return x

        x = self._create_nt([3, None, 2], device=device)
        with self.assertRaisesRegex(
            RuntimeError, "Nested tensors can only be vmapped over dim=0"
        ):
            vmap(f, in_dims=2)(x)

    def test_nt_with_nonzero_out_dim_raises(self, device):
        def f(x):
            return x

        x = self._create_nt([3, None, 2], device=device)
        with self.assertRaisesRegex(
            RuntimeError, "Nested tensors can only be vmapped over dim=0"
        ):
            vmap(f, out_dims=2)(x)

    def test_fallback_with_nt_and_batched_dense_with_nonzero_bdim_raises(self, device):
        def f(x, y):
            return x @ y

        x = self._create_nt([5, None, 3], device=device)
        y = torch.randn(3, 5, 4, device=device)

        with self.assertRaisesRegex(
            RuntimeError,
            "Fallback not supported for mixed nested / non-nested arguments without bdim=0",
        ):
            vmap(f, in_dims=(0, 1))(x, y)

    def test_multilevel_vmap_raises(self, device):
        def f(x):
            return x.sin() * 4.0 + 3.0

        x = self._create_nt([2, 2, 2, None], device=device)

        with self.assertRaisesRegex(
            RuntimeError, "Only one level of vmap is supported"
        ):
            vmap(vmap(f))(x)

        with self.assertRaisesRegex(
            RuntimeError, "Only one level of vmap is supported"
        ):
            vmap(vmap(vmap(f)))(x)


only_for = ("cpu", "cuda")
instantiate_device_type_tests(TestVmapOperatorsOpInfo, globals(), only_for=only_for)

instantiate_device_type_tests(
    TestVmapBatchedGradient,
    globals(),
    only_for=only_for,
)
instantiate_device_type_tests(TestTransformFailure, globals(), only_for=only_for)
instantiate_device_type_tests(TestRandomness, globals(), only_for=only_for)
instantiate_device_type_tests(TestVmapDeviceType, globals(), only_for=only_for)
instantiate_device_type_tests(TestVmapNestedTensor, globals(), only_for=only_for)

if __name__ == "__main__":
    run_tests()
