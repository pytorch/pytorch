# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.testing._internal.common_utils import TestCase, run_tests, is_iterable_of_tensors
import torch
from torch import Tensor
import torch.nn.functional as F
import functools
import unittest
import itertools
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_device_type import ops
from torch.testing._internal.common_dtype import integral_types
from functorch_lagging_op_db import functorch_lagging_op_db
from functorch_additional_op_db import additional_op_db
from common_utils import (
    get_fallback_and_vmap_exhaustive,
    get_exhaustive_batched_inputs,
    xfail,
    skip,
    skipOps,
    check_vmap_fallback,
    loop,
    IS_FBCODE,
)
from torch.utils._pytree import tree_flatten, tree_unflatten, tree_map
from functorch import grad, vjp, vmap
import torch.autograd.forward_ad as fwAD
from functorch._src.eager_transforms import _as_tuple, jvp
from functorch.compile import decomposition_table
aten = torch.ops.aten

# Version of autograd.grad that handles outputs that don't depend on inputs


def _autograd_grad(outputs, inputs, grad_outputs=None, retain_graph=False, create_graph=True):
    inputs, inputs_spec = tree_flatten(inputs)
    result = [torch.zeros_like(inp) for inp in inputs]
    diff_argnums = tuple(i for i, inp in enumerate(inputs) if inp.requires_grad)
    inputs = tuple(inputs[i] for i in diff_argnums)
    if grad_outputs is None:
        diff_outputs = tuple(out for out in outputs if out.requires_grad)
    else:
        something = [(out, go) for out, go in zip(outputs, grad_outputs)
                     if out.requires_grad]
        if len(something) == 0:
            diff_outputs, grad_outputs = (), ()
        else:
            diff_outputs, grad_outputs = zip(*something)
    if len(diff_outputs) == 0:
        return tuple(torch.zeros_like(inp) for inp in inputs)
    grad_inputs = torch.autograd.grad(diff_outputs, inputs, grad_outputs,
                                      retain_graph=retain_graph,
                                      create_graph=create_graph,
                                      allow_unused=True)
    grad_inputs = tuple(torch.zeros_like(inp) if gi is None else gi
                        for gi, inp in zip(grad_inputs, inputs))
    for idx, grad_inp in zip(diff_argnums, grad_inputs):
        result[idx] = grad_inp
    return tree_unflatten(result, inputs_spec)


def diff_arg(arg, requires_grad=True):
    def is_differentiable_arg(arg):
        if requires_grad:
            return arg.requires_grad
        else:
            return arg.is_floating_point() or arg.is_complex()
    if is_iterable_of_tensors(arg):
        if all([is_differentiable_arg(a) for a in arg]):
            return True
        if all([not is_differentiable_arg(a) for a in arg]):
            return False
        raise RuntimeError("NYI: The test runner can't handle this")
    return isinstance(arg, Tensor) and is_differentiable_arg(arg)


# Given f, returns an f' such that:
# - f' takes only positional arguments
# - All arguments to f' are floating-point Tensors
# - All outputs of f' are floating-point Tensors
def normalize_op_input_output2(f, args, kwargs, output_process_fn_grad=None, requires_grad=True):
    flat_args, args_spec = tree_flatten(args)
    diff_argnums = tuple(i for i, arg in enumerate(flat_args) if diff_arg(arg, requires_grad=requires_grad))
    assert len(diff_argnums) > 0
    primals = tuple(flat_args[i] for i in diff_argnums)

    @functools.wraps(f)
    def wrapped(*primals):
        _args = list(flat_args)
        for num, arg in zip(diff_argnums, primals):
            _args[num] = arg
        _args = tree_unflatten(_args, args_spec)
        result = f(*_args, **kwargs)
        if output_process_fn_grad is not None:
            result = output_process_fn_grad(result)
        if isinstance(result, tuple):
            # TODO: Remove the following hack for namedtuples
            result = tuple(result)
            result = tuple(r for r in result if torch.is_floating_point(r))
            assert len(result) > 0
        return result
    return wrapped, primals


# TODO: consolidate with normalize_op_input_output2
def normalize_op_input_output3(f, args, kwargs, sample_args, output_process_fn_grad=None):
    flat_args, args_spec = tree_flatten(args)
    flat_sample_args, _ = tree_flatten(sample_args)
    diff_argnums = tuple(i for i, (arg, sample) in enumerate(zip(flat_args, flat_sample_args))
                         if diff_arg(sample, requires_grad=True))
    assert len(diff_argnums) > 0
    primals = tuple(flat_args[i] for i in diff_argnums)

    @functools.wraps(f)
    def wrapped(*primals):
        _args = list(flat_args)
        for num, arg in zip(diff_argnums, primals):
            _args[num] = arg
        _args = tree_unflatten(_args, args_spec)
        result = f(*_args, **kwargs)
        if output_process_fn_grad is not None:
            result = output_process_fn_grad(result)
        if isinstance(result, tuple):
            # TODO: Remove the following hack for namedtuples
            result = tuple(result)
            result = tuple(r for r in result if torch.is_floating_point(r))
            assert len(result) > 0
        return result
    return wrapped, primals


def normalize_op_input_output(f, sample, requires_grad=True):
    args = tuple([sample.input] + list(sample.args))
    return normalize_op_input_output2(
        f, args, sample.kwargs, sample.output_process_fn_grad, requires_grad=requires_grad
    )


def ref_vjp(f, *primals):
    result = f(*primals)

    def wrapped(cotangents):
        return _autograd_grad(_as_tuple(result), primals, _as_tuple(cotangents))

    return result, wrapped


def ref_jvp(f, primals, tangents):
    with fwAD.dual_level():
        duals = tuple(fwAD.make_dual(p, t) for p, t in zip(primals, tangents))
        result_duals = f(*duals)
        result_duals, spec = tree_flatten(result_duals)
        primals_out, tangents_out = zip(*(fwAD.unpack_dual(d) for d in result_duals))
        return tree_unflatten(primals_out, spec), tree_unflatten(tangents_out, spec)


def get_sample_cotangents(f, sample):
    fn, primals = normalize_op_input_output(f, sample)
    output = fn(*primals)
    if isinstance(output, tuple):
        # TODO: Remove the following hack for torch.return_types
        output = tuple(output)
    return tree_map(torch.randn_like, output)


# returns a new function g(*args, *cotangents)
# that computes vjps and (*args, cotangents)
def get_vjp_fn_and_args_with_cotangents(f, sample, cotangents):
    args = tuple([sample.input] + list(sample.args))
    kwargs = sample.kwargs
    flat_args, args_spec = tree_flatten(args)
    flat_cotangents, cotangents_spec = tree_flatten(cotangents)

    @functools.wraps(f)
    def wrapped(*args):
        assert len(args) == len(flat_args) + len(flat_cotangents)
        actual_args = args[:len(flat_args)]
        cotangents = args[len(flat_args):]
        actual_args = tree_unflatten(actual_args, args_spec)
        cotangents = tree_unflatten(cotangents, cotangents_spec)

        fn, primals = normalize_op_input_output3(f, actual_args, kwargs,
                                                 flat_args,
                                                 sample.output_process_fn_grad)
        _, vjp_fn = vjp(fn, *primals)
        return vjp_fn(cotangents)

    return wrapped, tuple(flat_args + flat_cotangents)


# Returns a new function g(*args, *cotangents) that computes vjps and
# sample (*args, *cotangents)
def get_vjpfull_variant(f, sample):
    fn, primals = normalize_op_input_output(f, sample)
    result = fn(*primals)
    cotangents = _as_tuple(
        tree_map(lambda x: torch.randn_like(x, requires_grad=True), result))
    num_primals = len(primals)
    args = (*primals, *cotangents)

    @functools.wraps(f)
    def wrapped(*args):
        primals = args[:num_primals]
        cotangents = args[num_primals:]
        result, vjp_fn = vjp(fn, *primals)
        if isinstance(result, torch.Tensor):
            assert len(cotangents) == 1
            cotangents = cotangents[0]
        return vjp_fn(cotangents)

    return wrapped, args


def get_jvp_variant(f, sample):
    # We want this higher-order variant of jvp, so that it can
    # be used to wrap vmap
    fn, primals = normalize_op_input_output(f, sample, requires_grad=False)
    tangents = _as_tuple(
        tree_map(lambda x: torch.randn_like(x), primals))

    @functools.wraps(f)
    def wrapped(*args):
        tangents = args
        primals_out, tangents_out = jvp(fn, primals, tangents)

        if isinstance(primals_out, torch.Tensor):
            return (primals_out, tangents_out)
        else:
            flat_primals_out, _ = tree_flatten(primals_out)
            flat_tangents_out, _ = tree_flatten(tangents_out)
            return tuple(flat_primals_out + flat_tangents_out)

    return wrapped, tangents


def get_jvp_variant_primals_tangents(f, sample):
    # We want this higher-order variant of jvp, so that it can
    # be used to wrap vmap
    fn, primals = normalize_op_input_output(f, sample, requires_grad=False)
    tangents = _as_tuple(
        tree_map(lambda x: torch.randn_like(x), primals))

    @functools.wraps(f)
    def wrapped(*args):
        primals_in = args[:len(primals)]
        tangents_in = args[len(primals):]
        primals_out, tangents_out = jvp(fn, primals_in, tangents_in)

        if isinstance(primals_out, torch.Tensor):
            return (primals_out, tangents_out)
        else:
            flat_primals_out, _ = tree_flatten(primals_out)
            flat_tangents_out, _ = tree_flatten(tangents_out)
            return tuple(flat_primals_out + flat_tangents_out)

    return wrapped, primals + tangents


def is_inplace(op, variant):
    if hasattr(variant, "__wrapped__"):
        return variant.__wrapped__ is op.get_inplace()
    return variant is op.get_inplace()


vjp_fail = {
    skip('nn.functional.dropout'),  # randomness testing artifact
    skip('nn.functional.rrelu'),  # randomness testing artifact
    xfail('tensor_split'),
    xfail('to_sparse'),
    xfail('nn.functional.ctc_loss'),
}


class TestOperators(TestCase):
    @ops(functorch_lagging_op_db + additional_op_db, allowed_dtypes=(torch.float,))
    @skipOps('TestOperators', 'test_grad', vjp_fail.union({
        skip('nn.functional.fractional_max_pool2d'),  # fails on cuda, runs okay on cpu
        skip('nn.functional.fractional_max_pool3d'),  # fails on cuda, runs okay on cpu
    }))
    def test_grad(self, device, dtype, op):
        if op.name in vjp_fail:
            self.skipTest("Skipped; Expected failures")
            return

        if not op.supports_autograd:
            self.skipTest("Skipped! Autograd not supported.")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=True)

        # TODO: test in-place
        if is_inplace(op, op.get_op()):
            self.skipTest("Skipped! NYI: inplace-testing not supported.")
            return

        for sample in samples:
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs

            diff_argnums = tuple(i for i, arg in enumerate(args) if diff_arg(arg))
            assert len(diff_argnums) > 0
            diff_args = tuple(args[i] for i in diff_argnums)

            def wrapped_fn(*args, **kwargs):
                result = op(*args, **kwargs)
                if sample.output_process_fn_grad is not None:
                    result = sample.output_process_fn_grad(result)

                # Reduce into single value for grad
                if isinstance(result, torch.Tensor):
                    return result.sum()
                result = sum([res.sum() for res in result])
                return result

            result = grad(wrapped_fn, diff_argnums)(*args, **kwargs)
            expected = _autograd_grad(_as_tuple(wrapped_fn(*args, **kwargs)), diff_args)

            self.assertEqual(result, expected)

    @ops(functorch_lagging_op_db + additional_op_db, allowed_dtypes=(torch.float,))
    @skipOps('TestOperators', 'test_jvp', set({
        skip('nn.functional.dropout'),  # randomness testing artifact; not actually a problem
        skip('nn.functional.rrelu'),  # randomness testing artifact; not actually a problem
        skip('nn.functional.fractional_max_pool2d'),  # fails on cuda, runs okay on cpu
        skip('nn.functional.fractional_max_pool3d'),  # fails on cuda, runs okay on cpu
        skip('nn.functional.max_pool1d'),  # fails on cpu, runs okay on cuda
        xfail('nn.functional.batch_norm', device_type='cuda'),
        xfail('nn.functional.batch_norm', 'without_cudnn', device_type='cuda'),
        skip('nn.functional.conv_transpose3d', device_type='cuda'),

        # See https://github.com/pytorch/pytorch/issues/69034
        # RuntimeError: expected scalar type double but found float
        xfail('minimum'),
        xfail('min', 'binary'),
        xfail('maximum'),
        xfail('max', 'binary'),

        # The following don't have a forward-mode AD formula in PyTorch core
        # (check derivatives.yaml).
        xfail('var_mean'),
        xfail('std_mean'),
        # https://gist.github.com/zou3519/f62a167fb46cda01d7f238f61dd9ccf9
        xfail('linalg.eigvalsh'),
        # https://gist.github.com/zou3519/b86616d01ca375a4bd17403277f49225
        xfail('nn.functional.dropout', device_type='cuda'),

        # =============================================
        # NB: The above failures also fail using PyTorch core's
        #     forward-mode AD and vmap.
        #     The failures below are functorch-specific issues
        # =============================================

        # Composite ops that do bad things. Need to be fixed in PyTorch core.
        # RuntimeError: Cannot access data pointer of Tensor that doesn't have storage
        xfail('linalg.eigvals'),
        xfail('tensor_split'),

        # Causing a CUDA assert, needs investigation
        skip('div', 'floor_rounding', device_type='cuda'),
        skip('div', 'no_rounding_mode', device_type='cuda'),
        skip('div', 'trunc_rounding', device_type='cuda'),
        skip('true_divide', device_type='cuda'),

        # Causing an error with calling a forward mode of a forward mode
        xfail('nn.functional.batch_norm', device_type='cpu'),

        # Some kind of issue with unsymmetric tangent type
        # Runtime Error: The tangent part of the matrix A should also be symmetric.
        xfail('linalg.eigh'),


    }))
    def test_jvp(self, device, dtype, op):
        # TODO: when we change supports_autograd to supports_backward_ad, also change in this file
        if not op.supports_forward_ad:
            self.skipTest("Skipped! Forward AD not supported.")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=True)
        # TODO: test in-place
        if is_inplace(op, op.get_op()):
            self.skipTest("Skipped! NYI: inplace-testing not supported.")
            return

        for sample in samples:
            fn, primals = normalize_op_input_output(op, sample, requires_grad=False)
            tangents = tree_map(lambda x: torch.randn_like(x), primals)
            primal_outs, tangent_outs = jvp(fn, primals, tangents)
            expected_primal_outs, expected_tangent_outs = ref_jvp(fn, primals, tangents)
            self.assertEqual(primal_outs, expected_primal_outs)
            self.assertEqual(tangent_outs, expected_tangent_outs)

    @ops(functorch_lagging_op_db + additional_op_db, allowed_dtypes=(torch.float,))
    @skipOps('TestOperators', 'test_vjp', vjp_fail.union({
        skip('nn.functional.fractional_max_pool2d'),  # fails on cpu, runs okay on cuda
        skip('nn.functional.fractional_max_pool3d'),  # fails on cpu, runs okay on cuda
        skip('nn.functional.conv_transpose3d', device_type='cuda'),  # numerical precision
    }))
    def test_vjp(self, device, dtype, op):
        if not op.supports_autograd:
            self.skipTest("Skipped! Autograd not supported.")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=True)

        # TODO: test in-place
        if is_inplace(op, op.get_op()):
            self.skipTest("Skipped! NYI: inplace-testing not supported.")
            return

        def _test(_op):
            for sample in samples:
                fn, primals = normalize_op_input_output(_op, sample)
                result = fn(*primals)
                cotangents = tree_map(lambda x: torch.randn_like(x), result)

                out, vjp_fn = vjp(fn, *primals)
                self.assertEqual(out, result)
                result_vjps = vjp_fn(cotangents)

                _, vjp_fn = ref_vjp(fn, *primals)
                expected_vjps = vjp_fn(cotangents)

                self.assertEqual(result_vjps, expected_vjps)

        _test(op)
        for a_op in op.aliases:
            _test(a_op)

    @ops(functorch_lagging_op_db + additional_op_db, allowed_dtypes=(torch.float,))
    @skipOps('TestOperators', 'test_vjpvjp', vjp_fail.union({
        skip('nn.functional.fractional_max_pool2d'),  # fails on cuda, runs okay on cpu
        xfail('nn.functional.fractional_max_pool3d'),
        skip('nn.functional.conv_transpose3d'),  # numerical precision problem
        xfail('nn.functional.binary_cross_entropy'),  # testing problem
    }))
    def test_vjpvjp(self, device, dtype, op):
        if not op.supports_autograd:
            self.skipTest("Skipped! Autograd not supported.")
            return
        if not op.supports_gradgrad:
            self.skipTest("Skipped! Operation does not support gradgrad")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=True)

        # TODO: test in-place
        if is_inplace(op, op.get_op()):
            self.skipTest("Skipped! NYI: inplace-testing not supported.")
            return

        for sample in samples:
            fn, args = get_vjpfull_variant(op, sample)
            result = fn(*args)
            cotangents = tree_map(lambda x: torch.randn_like(x), result)

            # Compute vjp of vjp
            _, vjp_fn = vjp(fn, *args)
            result_vjps = vjp_fn(cotangents)

            # Compute ref_vjp of vjp. We could have done ref_vjp of ref_vjp,
            # but since we're confident that vjp works by itself, this is
            # an equivalent way to test that.
            _, vjp_fn = ref_vjp(fn, *args)
            expected_vjps = vjp_fn(cotangents)

            self.assertEqual(result_vjps, expected_vjps)

    @ops(functorch_lagging_op_db + additional_op_db, allowed_dtypes=(torch.float,))
    def test_vmapvjpvjp(self, device, dtype, op):
        self.skipTest("Skipped; these tests take too long")
        op_skip = set({
        })
        op_skip = op_skip.union(vjp_fail)
        if op.name in op_skip:
            self.skipTest("Skipped; Expected failures")
            return

        if not op.supports_autograd:
            self.skipTest("Skipped! Autograd not supported.")
            return
        if not op.supports_gradgrad:
            self.skipTest("Skipped! Operation does not support gradgrad")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=True)

        # TODO: test in-place
        if is_inplace(op, op.get_op()):
            self.skipTest("Skipped! NYI: inplace-testing not supported.")
            return

        for sample in samples:
            fn, args = get_vjpfull_variant(op, sample)
            result = fn(*args)
            cotangents = tree_map(lambda x: torch.randn_like(x), result)
            cotangents, _ = tree_flatten(cotangents)
            num_args = len(args)

            args_and_cotangents = tuple(args) + tuple(cotangents)

            def vjp_of_vjp(*args_and_cotangents):
                args = args_and_cotangents[:num_args]
                cotangents = args_and_cotangents[num_args:]
                result, vjp_fn = vjp(fn, *args)
                result_vjps = vjp_fn(cotangents)
                result, _ = tree_flatten(result)
                result_vjps, _ = tree_flatten(result_vjps)
                return (*result, *result_vjps)

            generator = get_fallback_and_vmap_exhaustive(vjp_of_vjp, args_and_cotangents, {}, opinfo=op)
            for loop_out, batched_out in generator:
                self.assertEqual(loop_out, batched_out, atol=1e-4, rtol=1e-4)
    vmapvjp_fail = vjp_fail.union({
        # The following are not bugs and are expected behavior
        xfail('fill_'),  # Not possible, wontfix
        xfail('masked_select'),  # Not possible due to dynamic shapes

        # All of the following are bugs and need to be fixed
        xfail('eig'),
        xfail('view_as_complex'),
        xfail('fft.ihfft'),
        xfail('fft.ihfft'),
        xfail('fft.rfft'),
        xfail('fft.rfft'),
        xfail('fft.rfftn'),
        xfail('linalg.det', ''),
        xfail('linalg.cholesky'),
        xfail('linalg.eig'),  # Uses aten::allclose
        xfail('linalg.eigh'),
        xfail('linalg.householder_product'),
        xfail('linalg.inv'),
        xfail('linalg.matrix_norm'),
        xfail('linalg.matrix_power'),
        xfail('linalg.norm'),
        xfail('linalg.slogdet'),
        # really annoying thing where it passes correctness check but not has_batch_rule
        skip('linalg.svdvals'),
        xfail('logdet'),
        xfail('lu_unpack'),
        xfail('masked_scatter'),
        xfail('matrix_exp'),
        xfail('nanquantile'),
        xfail('norm', 'nuc'),
        xfail('prod'),
        xfail('put'),
        xfail('quantile'),
        xfail('symeig'),
        xfail('take'),
        xfail('linalg.tensorinv'),
        xfail('block_diag'),
        xfail('nn.functional.dropout'),
        xfail('fft.ihfft2'),
        xfail('fft.ihfftn'),
        xfail('double', 'channels_last'),
        xfail('nn.functional.gaussian_nll_loss'),
        xfail('fft.rfft2'),
        skip('qr'),  # Nondetermistic
        xfail('_masked.prod'),  # calls aten::item
        xfail('stft'),
        xfail('nn.functional.glu'),
        xfail('nn.functional.fractional_max_pool3d'),
        xfail('as_strided'),
        xfail('nn.functional.fractional_max_pool2d'),
        xfail('__getitem__', ''),
        xfail('index_put', ''),
        xfail('lu_solve'),
        xfail('nn.functional.instance_norm'),
    })

    @ops(functorch_lagging_op_db + additional_op_db, allowed_dtypes=(torch.float,))
    @skipOps('TestOperators', 'test_vmapvjp', vmapvjp_fail)
    def test_vmapvjp(self, device, dtype, op):
        if not op.supports_autograd:
            self.skipTest("Skipped! Autograd not supported.")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=True)

        # TODO: test in-place
        if is_inplace(op, op.get_op()):
            self.skipTest("Skipped! NYI: inplace-testing not supported.")
            return

        for sample in samples:
            cotangents = get_sample_cotangents(op, sample)
            fn, args = get_vjp_fn_and_args_with_cotangents(op, sample, cotangents)
            for loop_out, batched_out in get_fallback_and_vmap_exhaustive(fn, args, {}, opinfo=op):
                self.assertEqual(loop_out, batched_out, atol=1e-4, rtol=1e-4)

    # There are several variations we care about
    # 1) primal batched (TODO)
    # 2) tangent batched (batched grads) <--
    # 3) both batched (TODO)
    # The below tests (2) only.
    @ops(functorch_lagging_op_db, allowed_dtypes=(torch.float,))
    @skipOps('TestOperators', 'test_vmapjvp', {
        skip('nn.functional.dropout'),  # randomness
        skip('nn.functional.rrelu'),  # randomness
        skip('nn.functional.fractional_max_pool2d'),  # randomness
        skip('nn.functional.fractional_max_pool3d'),  # randomness
        skip('nn.functional.max_pool1d'),  # fails on cpu, runs on cuda

        # TODO: fails in core due to in-place batched nto non-batched
        # but fails here for a different reason
        xfail('linalg.householder_product'),

        # Try to in-place batched tensor into non-batched tensor
        xfail('matrix_exp'),
        xfail('fill_'),
        xfail('block_diag'),  # TODO: We expect this to fail in core, but it doesn't

        # https://gist.github.com/zou3519/c42d032c0111c6b65235583d391bf7a3
        xfail('nn.functional.linear'),

        # These are issues that should be fixed in core. See repro in core:
        # https://github.com/pytorch/functorch/pull/232#discussion_r751405155
        # RuntimeError: expected scalar type double but found float
        xfail('minimum'),
        xfail('min', 'binary'),
        xfail('maximum'),
        xfail('max', 'binary'),

        # Apprently these support forward AD, but we get "Trying to use forward AD..."
        # These are cases where OpInfo has supports_forward_ad=True, but disables
        # the test
        xfail('var_mean'),
        xfail('std_mean'),
        xfail('linalg.eigvalsh'),

        # functorch doesn't support channels_last
        # PyTorch core's vmap doesn't have a batching rule for `double`, if it
        # did it would also not support channels last, so I'm including this
        # xfail "above the line".
        xfail('double', 'channels_last'),

        # RuntimeError: expand: the number of sizes provided (1) must be greater or
        # equal to the number of dimensions in the tensor (2)
        xfail('nanquantile'),
        xfail('quantile'),

        # RuntimeError: vmap: inplace arithmetic(self, *extra_args)
        xfail('nn.functional.gelu'),

        # Not implemented
        xfail('scatter'),

        # =============================================
        # NB: The above failures also fail in PyTorch core.
        #     The failures below only fail in functorch
        # =============================================

        # Composite ops that do bad things. Need to be fixed in PyTorch core.
        # RuntimeError: Cannot access data pointer of Tensor that doesn't have storage
        xfail('tensor_split'),
        xfail('linalg.eigvals'),

        # Causing a CUDA assert, needs investigation
        skip('div', 'floor_rounding', device_type='cuda'),
        skip('div', 'no_rounding_mode', device_type='cuda'),
        skip('div', 'trunc_rounding', device_type='cuda'),
        skip('true_divide', device_type='cuda'),

        # Causing multiple forward mode AD issues, needs investigation
        xfail('nn.functional.batch_norm'),
        xfail('nn.functional.batch_norm', 'without_cudnn', device_type='cuda'),

        # Some kind of issue with unsymmetric tangent type
        # Runtime Error: The tangent part of the matrix A should also be symmetric.
        xfail('linalg.eigh'),
    })
    def test_vmapjvp(self, device, dtype, op):
        if is_inplace(op, op.get_op()):
            # TODO: test in-place
            self.skipTest("Skipped! NYI: inplace-testing not supported.")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=False)

        if not op.supports_forward_ad:
            self.skipTest("Skipped! Forward AD not supported.")
            return

        for sample in samples:
            arg_values = [sample.input] + list(sample.args)
            kwarg_values = sample.kwargs
            args = tuple([*arg_values, *kwarg_values])
            fn, args = get_jvp_variant(op, sample)
            for loop_out, batched_out in get_fallback_and_vmap_exhaustive(fn, args, {}, opinfo=op, bdims=(0,)):
                self.assertEqual(loop_out, batched_out, atol=1e-4, rtol=1e-4)

    @ops(functorch_lagging_op_db, allowed_dtypes=(torch.float,))
    @skipOps('TestOperators', 'test_vmapjvpall', {
        skip('nn.functional.dropout'),  # randomness
        skip('nn.functional.rrelu'),  # randomness
        xfail('nn.functional.fractional_max_pool2d'),  # Cannot access data pointer of Tensor that doesn't have storage
        xfail('nn.functional.fractional_max_pool3d'),  # Cannot access data pointer of Tensor that doesn't have storage
        skip('nn.functional.max_pool1d'),  # fails on cpu, runs on cuda
        xfail('_masked.mean', device_type='cuda'),
        xfail('_masked.prod', device_type='cuda'),
        xfail('nn.functional.batch_norm', device_type='cuda'),
        xfail('nn.functional.batch_norm', 'without_cudnn', device_type='cuda'),
        xfail('nn.functional.hinge_embedding_loss', device_type='cuda'),
        xfail('nn.functional.instance_norm', device_type='cuda'),

        # Causing a CUDA assert, needs investigation
        skip('div', 'floor_rounding', device_type='cuda'),
        skip('div', 'no_rounding_mode', device_type='cuda'),
        skip('div', 'trunc_rounding', device_type='cuda'),
        skip('true_divide', device_type='cuda'),

        # Causing issues with multiple cpu levels of forward mode AD
        xfail('_masked.mean', device_type='cpu'),
        xfail('_masked.prod', device_type='cpu'),
        xfail('nn.functional.batch_norm', device_type='cpu'),
        xfail('nn.functional.hinge_embedding_loss', device_type='cpu'),
        xfail('nn.functional.instance_norm', device_type='cpu'),

        # xfail list
        xfail('norm', 'nuc'),
        xfail('linalg.inv'),
        xfail('linalg.tensorinv'),
        xfail('linalg.matrix_power'),
        xfail('maximum'),
        xfail('linalg.householder_product'),
        xfail('tensor_split'),
        xfail('nn.functional.gelu'),
        xfail('quantile'),
        xfail('var_mean'),
        xfail('as_strided'),
        xfail('linalg.eigvals'),
        xfail('linalg.eigvalsh'),
        xfail('fill_'),
        xfail('linalg.cholesky'),
        xfail('max', 'binary'),
        xfail('nn.functional.gaussian_nll_loss'),
        xfail('min', 'binary'),
        xfail('std_mean'),
        xfail('double', 'channels_last'),
        xfail('block_diag'),
        xfail('minimum'),
        xfail('scatter'),
        xfail('matrix_exp'),
        xfail('nanquantile'),
        xfail('nn.functional.linear'),
        xfail('view_as_complex'),
        xfail('prod'),

        # Some kind of issue with unsymmetric tangent type
        # Runtime Error: The tangent part of the matrix A should also be symmetric.
        xfail('linalg.eigh'),
    })
    # This is technically a superset of test_vmapjvp. We should either delete test_vmapjvp
    # or figure out if we can split vmapjvpall. It's useful to keep test_vmapjvp intact
    # because that coresponds to "batched forward-mode AD" testing in PyTorch core
    def test_vmapjvpall(self, device, dtype, op):
        if is_inplace(op, op.get_op()):
            # TODO: test in-place
            self.skipTest("Skipped! NYI: inplace-testing not supported.")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=False)

        if not op.supports_forward_ad:
            self.skipTest("Skipped! Forward AD not supported.")
            return

        for sample in samples:
            arg_values = [sample.input] + list(sample.args)
            kwarg_values = sample.kwargs
            args = tuple([*arg_values, *kwarg_values])
            fn, args = get_jvp_variant_primals_tangents(op, sample)
            for loop_out, batched_out in get_fallback_and_vmap_exhaustive(fn, args, {}, opinfo=op):
                self.assertEqual(loop_out, batched_out, atol=1e-4, rtol=1e-4)

    @ops(functorch_lagging_op_db + additional_op_db, allowed_dtypes=(torch.float,))
    @skipOps('TestOperators', 'test_vmapvjp_has_batch_rule', vmapvjp_fail.union({
        xfail('view_as_complex'),
        xfail('__getitem__', ''),
        xfail('cholesky'),
        xfail('complex'),
        xfail('copysign'),
        xfail('cummax'),
        xfail('cummin'),
        xfail('cumprod'),
        xfail('eig'),
        xfail('fmin'),
        xfail('fmax'),
        xfail('fft.ihfft'),
        xfail('fft.rfft'),
        xfail('fft.rfftn'),
        xfail('fill_'),
        xfail('index_copy'),
        xfail('index_fill'),
        xfail('linalg.cholesky'),
        xfail('linalg.cholesky_ex'),
        xfail('linalg.det'),
        xfail('linalg.eig'),
        xfail('linalg.eigh'),
        xfail('linalg.eigvals'),
        xfail('linalg.householder_product'),
        xfail('linalg.lstsq'),
        xfail('linalg.inv'),
        xfail('linalg.matrix_norm'),
        xfail('linalg.matrix_power'),
        xfail('linalg.norm'),
        xfail('linalg.pinv'),
        xfail('linalg.qr'),
        xfail('linalg.pinv', 'hermitian'),
        xfail('linalg.slogdet'),
        xfail('linalg.solve'),
        xfail('linalg.tensorinv'),
        xfail('logdet'),
        xfail('lu'),
        xfail('lu_solve'),
        xfail('lu_unpack'),
        xfail('masked_fill'),
        xfail('masked_scatter'),
        xfail('masked_select'),
        xfail('matrix_exp'),
        xfail('nanquantile'),
        xfail('nn.functional.gelu'),
        xfail('norm', 'nuc'),
        xfail('pinverse'),
        xfail('prod'),
        xfail('put'),
        xfail('quantile'),
        xfail('renorm'),
        xfail('solve'),
        xfail('symeig'),
        xfail('take'),
        xfail('tensor_split'),
        xfail('to_sparse'),
        xfail('trace'),
        xfail('unfold'),
        xfail('vdot'),
        xfail('block_diag'),
        xfail('nn.functional.dropout'),
        xfail('_masked.prod'),
        xfail('fft.ihfft2'),
        xfail('fft.ihfftn'),
        xfail('fft.rfft2'),
        xfail('cross'),
        xfail('double', 'channels_last'),
        xfail('linalg.cross'),
        xfail('nn.functional.gaussian_nll_loss'),
        xfail('nn.functional.huber_loss'),
        xfail('nn.functional.instance_norm'),
        xfail('nn.functional.poisson_nll_loss'),
        xfail('nn.functional.bilinear'),
        xfail('nn.functional.prelu'),
        xfail('nn.functional.glu'),
        xfail('nn.functional.fractional_max_pool3d'),
        xfail('as_strided'),
        xfail('linalg.solve_triangular'),
        xfail('stft'),
        xfail('nn.functional.rrelu'),
        xfail('nn.functional.embedding_bag'),
        xfail('nn.functional.max_pool3d'),
        xfail('istft'),
        xfail('nn.functional.fractional_max_pool2d'),
        xfail('linalg.tensorsolve'),
    }))
    def test_vmapvjp_has_batch_rule(self, device, dtype, op):
        if not op.supports_autograd:
            self.skipTest("Skipped! Autograd not supported.")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=True)

        # TODO: test in-place
        if is_inplace(op, op.get_op()):
            self.skipTest("Skipped! NYI: inplace-testing not supported.")
            return

        def test():
            for sample in samples:
                cotangents = get_sample_cotangents(op, sample)
                fn, args = get_vjp_fn_and_args_with_cotangents(op, sample, cotangents)
                for loop_out, batched_out in get_fallback_and_vmap_exhaustive(
                        fn, args, {}, opinfo=op, compute_loop_out=False):
                    pass
                for a_op in op.aliases:
                    fn, args = get_vjp_fn_and_args_with_cotangents(a_op, sample, cotangents)
                    for loop_out, batched_out in get_fallback_and_vmap_exhaustive(
                            fn, args, {}, opinfo=op, compute_loop_out=False):
                        pass

        check_vmap_fallback(self, test, op, dry_run=False)

    @ops(functorch_lagging_op_db + additional_op_db, allowed_dtypes=(torch.float,))
    @skipOps('TestOperators', 'test_vjpvmap', vjp_fail.union({
        # fallback path doesn't work
        # All of the following are bugs and need to be fixed
        xfail('__getitem__', ''),
        xfail('clamp', ''),
        xfail('fill_'),
        xfail('index_put', ''),
        xfail('lu_solve'),
        xfail('lu_unpack'),
        xfail('matrix_exp'),
        xfail('view_as_complex'),
        xfail('nn.functional.gaussian_nll_loss'),
        xfail('double', 'channels_last'),
        xfail('masked_select'),
        xfail('nn.functional.fractional_max_pool3d'),
        xfail('nn.functional.glu'),
        xfail('as_strided'),
        xfail('nn.functional.fractional_max_pool2d'),
        skip('solve'),
    }))
    def test_vjpvmap(self, device, dtype, op):
        # NB: there is no vjpvmap_has_batch_rule test because that is almost
        # certainly redundant with the vmap_has_batch_rule test in test_vmap.py

        # one-off skip
        if op.name == 'nn.functional.dropout':
            self.skipTest("Skipped!")

        if not op.supports_autograd:
            # If the op doesn't support autograd, vmap(op) won't either
            self.skipTest("Skipped! Autograd not supported.")
            return

        # TODO: test in-place
        if is_inplace(op, op.get_op()):
            self.skipTest("Skipped! NYI: inplace-testing not supported.")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=True)
        is_batch_norm = op.name == "nn.functional.batch_norm"

        for sample in samples:
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs
            generator = get_exhaustive_batched_inputs(args, kwargs, for_batch_norm=is_batch_norm)

            for batched_args, in_dims, kwargs in generator:
                vmapped_op = vmap(op, in_dims)
                fn, primals = normalize_op_input_output2(vmapped_op, batched_args, kwargs,
                                                         sample.output_process_fn_grad)
                result = fn(*primals)
                cotangents = tree_map(lambda x: torch.randn_like(x), result)

                _, vjp_fn = vjp(fn, *primals)
                result_vjps = vjp_fn(cotangents)

                _, vjp_fn = ref_vjp(fn, *primals)
                expected_vjps = vjp_fn(cotangents)

                self.assertEqual(result_vjps, expected_vjps)


class InplaceError(Exception):
    def __repr__(self):
        return "Decomposition Tensor with no elem was created (probably due to an in-place op)"


def ref_vjp_no_create(f, *primals):
    result = f(*primals)

    def wrapped(cotangents):
        return _autograd_grad(_as_tuple(result), primals, _as_tuple(cotangents), create_graph=False)

    return result, wrapped


run_decompositions = set()
run_ops = set()


class TestDecompositionOpInfo(TestCase):

    @unittest.skipIf(IS_FBCODE, "__torch_dispatch__ is buggy")
    @ops(
        functorch_lagging_op_db + additional_op_db,
        allowed_dtypes=[torch.float32, torch.float64, torch.float16, torch.bfloat16] + [*integral_types()]
    )
    # entries in here need don't work and need to be fixed.
    # Each one of these is a bug (or needs to be investigated)
    @skipOps('TestDecompositionOpInfo', 'test_decomposition', {
        skip('view_as_complex'),
        # skip('log_softmax', device_type='cuda'),
        # skip('nn.functional.softmin', device_type='cuda'),
        xfail('linalg.cholesky'),
        xfail('linalg.inv'),
        skip('linalg.det', 'singular', device_type='cuda'),  # this is nasty and seems to stop the test suite
        xfail('linalg.matrix_power'),
        xfail('linalg.tensorinv'),
        xfail('to_sparse'),
        skip('tensor_split'),
        skip('mvlgamma'),
        skip('eig'),
        skip('nn.functional.dropout'),
        skip('_masked.softmin'),
        skip('_masked.log_softmax'),
        skip('stft'),
        skip('_masked.softmax'),
        skip('_masked.normalize'),
        # Some weird matmul stuff with int64 matmuls
        # inplace op
        skip('resize_'),
    })
    def test_decomposition(self, device, dtype, op):
        # copied from common_utils.py
        dtype_precisions = {
            torch.float16: (0.001, 1e-5),
            torch.bfloat16: (0.016, 1e-4),  # zzzz bfloat16 precision
            torch.float32: (1.3e-6, 1e-5),
            torch.float64: (1e-7, 1e-7),
            torch.complex32: (0.001, 1e-5),
            torch.complex64: (1.3e-6, 1e-5),
            torch.complex128: (1e-7, 1e-7),
        }
        # Returns the "default" rtol and atol for comparing scalars or
        # tensors of the given dtypes.

        def _getDefaultRtolAndAtol(dtype0, dtype1):
            rtol = max(dtype_precisions.get(dtype0, (0, 0))[0],
                       dtype_precisions.get(dtype1, (0, 0))[0])
            atol = max(dtype_precisions.get(dtype0, (0, 0))[1],
                       dtype_precisions.get(dtype1, (0, 0))[1])
            return rtol, atol

        def op_assert_equal(op, a, b):
            assert a.dtype == b.dtype
            # Some ops, like those involving reductions, are fundamentally non-decomposable with precision guarantees
            tol_table = {
                # aggghhhhhhhhhh I hate reductions and floating point
                (torch.float16, aten.huber_loss_backward): (0.002, 1e-5),
                (torch.bfloat16, aten.tanh_backward): (0.03, 1e-4),
                (torch.bfloat16, aten._log_softmax): (0.05, 1e-2),
                (torch.bfloat16, aten._softmax_backward_data): (0.016, 5e-3),
                (torch.bfloat16, aten._log_softmax_backward_data): (0.016, 5e-3),
            }
            if (b.dtype, op) in tol_table:
                rtol, atol = tol_table[(b.dtype, op)]
            else:
                rtol, atol = _getDefaultRtolAndAtol(a.dtype, b.dtype)
            if not torch.allclose(a, b, rtol=rtol, atol=atol):
                atol_diff = (a - b).abs().max()
                rtol_diff = ((a - b).abs()/b.abs()).nan_to_num(0).max()
                msg = f"{op.__name__} decomposition failed, max rel: {rtol_diff}, max abs: {atol_diff}"
                raise RuntimeError(msg)

        # We check the correctness of each decomposition right after running it.
        # So, when we encounter a decomposition, we run the function normally, and
        # then run the decomposition, and ensure they're identical.
        # The way this is implemented, there could .... technically be an exponential blow up,
        # but it's probably fine for now.
        class DecompositionTensor(torch.Tensor):
            elem: torch.Tensor

            __slots__ = ['elem']

            @staticmethod
            def __new__(cls, elem):
                r = torch.Tensor._make_wrapper_subclass(
                    cls, elem.size(),
                    strides=elem.stride(), storage_offset=elem.storage_offset(),
                    dtype=elem.dtype, layout=elem.layout,
                    device=elem.device, requires_grad=elem.requires_grad
                )

                r.elem = elem
                return r

            def __repr__(self):
                return f"DecompositionTensor(elem={self.elem})"

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                global run_ops
                run_ops.add(func)

                def unwrap_tensor(e):
                    if isinstance(e, DecompositionTensor):
                        if not hasattr(e, 'elem'):
                            raise InplaceError()
                        return e.elem
                    return e

                real_out = func(*tree_map(unwrap_tensor, args), **tree_map(unwrap_tensor, kwargs))

                if func in decomposition_table and func != torch.ops.aten.detach:
                    upcast_table = set([
                        aten.tanh_backward,
                        aten._log_softmax,
                        aten._log_softmax_backward_data,
                        aten._softmax_backward_data,
                    ])
                    decomposition = decomposition_table[func]
                    global run_decompositions
                    run_decompositions.add(func)
                    input_dtypes = set()

                    def upcast_tensor(x):
                        if isinstance(x, Tensor) and (x.dtype == torch.bfloat16 or x.dtype == torch.float16):
                            input_dtypes.add(x.dtype)
                            x = x.to(dtype=torch.float32)
                        return x
                    # Theoretically, most PyTorch ops compute intermediates as fp32. But this breaks some ops...
                    if func in upcast_table:
                        args = tree_map(upcast_tensor, args)
                        kwargs = tree_map(upcast_tensor, kwargs)
                    decomp_out = decomposition(*args, **kwargs)
                    real_out_flat = tree_flatten(real_out)[0]
                    decomp_out_flat = tree_flatten(decomp_out)[0]
                    assert(len(real_out_flat) == len(decomp_out_flat))
                    assert(len(input_dtypes) <= 1)
                    input_dtypes = list(input_dtypes)
                    for a, b in zip(real_out_flat, decomp_out_flat):
                        if len(input_dtypes) > 0:
                            b = b.to(dtype=input_dtypes[0])
                        op_assert_equal(func, a, b)

                def wrap_tensor(e):
                    if e is None:
                        return DecompositionTensor(torch.empty(()))
                    return DecompositionTensor(e) if type(e) == torch.Tensor else e
                wrapped_out = tree_map(wrap_tensor, real_out)
                return wrapped_out

        if dtype not in op.supported_dtypes(dtype):
            self.skipTest("Dtype not in op's supported dtypes")
            return
        if is_inplace(op, op.get_op()):
            self.skipTest("op is inplace")
            return
        _requires_grad = op.supports_autograd and dtype.is_floating_point

        samples = op.sample_inputs(device, dtype, requires_grad=_requires_grad)

        # Acquires variants to test
        def wrap_tensor(x):
            if type(x) == torch.Tensor:
                return DecompositionTensor(x)
            return x

        try:
            func = op.get_op()
            for sample_input in samples:
                if _requires_grad:
                    fn, primals = normalize_op_input_output(func, sample_input)
                    primals = tree_map(lambda x: x.abs() if isinstance(x, torch.Tensor) else x, primals)

                    decomp_out, decomp_vjp_fn = ref_vjp_no_create(fn, *tree_map(wrap_tensor, primals))
                    cotangents = tree_map(lambda x: torch.randn_like(x), decomp_out)

                    _ = decomp_vjp_fn(cotangents)

                else:
                    args = [sample_input.input] + list(sample_input.args)
                    kwargs = sample_input.kwargs
                    _ = func(*args, **kwargs)

                    args = tree_map(wrap_tensor, args)
                    kwargs = tree_map(wrap_tensor, kwargs)
                    decomp_out = func(*args, **kwargs)

        except InplaceError:
            self.skipTest("op is inplace")
            return
        except RuntimeError as e:
            if "not implemented for" in str(e):
                self.skipTest(str(e))
                return
            if "Mismatch in shape: grad_output" in str(e):
                self.skipTest("Some weird issue with autograd engine and tensor subclasses")
                return
            raise e

    @unittest.skipIf(IS_FBCODE, "__torch_dispatch__ is buggy")
    def test_placeholder(self):
        global run_ops, run_decompositions
        with open('op_analysis/run_ops.txt', 'w') as f:
            def get_names(inpt):
                return sorted([x.__name__ for x in inpt])
            for op in get_names(run_ops):
                f.write(f'{op}\n')
        with open('op_analysis/run_decompositions.txt', 'w') as f:
            for op in get_names(run_decompositions):
                f.write(f'{op}\n')

    def test_group_norm_backward(self, device):
        # group norm will hit the decomposable ``infinitely_differentiable_group_norm_backward`` when
        # GradMode is on, which happens by default in the grad transform. This avoids that
        def f(x, weight, bias, grad_out):
            output = F.group_norm(x, 6, weight, bias)
            inputs = []
            for input in (x, weight, bias):
                if input.requires_grad:
                    inputs.append(input)
            return torch.autograd.grad(outputs=output, inputs=inputs, grad_outputs=grad_out)

        B, N, C, H, W = 2, 3, 24, 5, 7
        for (input_grad, weight_grad, bias_grad) in itertools.product((True, False), (True, False), (True, False)):
            if not input_grad and not weight_grad and not bias_grad:
                continue
            x = torch.randn(N, C, H, W, device=device, requires_grad=input_grad)
            weight = torch.randn(C, device=device, requires_grad=weight_grad)
            bias = torch.randn(C, device=device, requires_grad=bias_grad)
            grad_out = torch.randn(B, N, C, H, W, device=device)
            loop_out = loop(f, (None, None, None, 0), 0, 2, x, weight, bias, grad_out)
            batched_out = vmap(f, (None, None, None, 0), 0)(x, weight, bias, grad_out)
            self.assertEqual(loop_out, batched_out)


only_for = ("cpu", "cuda")
instantiate_device_type_tests(TestOperators, globals(), only_for=only_for)
instantiate_device_type_tests(TestDecompositionOpInfo, globals(), only_for=only_for)

if __name__ == '__main__':
    run_tests()
