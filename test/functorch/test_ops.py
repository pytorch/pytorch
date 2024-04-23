# Owner(s): ["module: functorch"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import itertools
import unittest

import torch
import torch.autograd.forward_ad as fwAD
from common_utils import (
    check_vmap_fallback,
    decorate,
    expectedFailureIf,
    generate_vmap_inputs,
    get_fallback_and_vmap_exhaustive,
    is_batch_norm_training,
    is_valid_inplace_sample_input,
    loop,
    loop2,
    opsToleranceOverride,
    skip,
    skipOps,
    tol1,
    tol2,
    xfail,
)
from functorch import grad, jacfwd, jacrev, vjp, vmap
from functorch_additional_op_db import additional_op_db
from torch import Tensor
from torch._functorch.eager_transforms import _as_tuple, jvp
from torch.testing._internal.autograd_function_db import autograd_function_db
from torch.testing._internal.common_cuda import with_tf32_off
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
    tol,
    toleranceOverride,
)
from torch.testing._internal.common_methods_invocations import op_db

from torch.testing._internal.common_utils import (
    is_iterable_of_tensors,
    IS_MACOS,
    IS_X86,
    noncontiguous_like,
    parametrize,
    run_tests,
    runOnRocm,
    skipIfRocm,
    TEST_WITH_ASAN,
    TEST_WITH_ROCM,
    TestCase,
    unMarkDynamoStrictTest,
)

from torch.testing._internal.opinfo.core import SampleInput
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

aten = torch.ops.aten


# Version of autograd.grad with some differences:
#   - pytree inputs is allowed (but leaves of the pytree have to all
#     be tensors)
#   - if an input is not used as part of derivatives, we will return a
#     zero-filled tensor for the result
def _autograd_grad(
    outputs, inputs, grad_outputs=None, retain_graph=False, create_graph=True
):
    inputs, inputs_spec = tree_flatten(inputs)
    diff_inputs = tuple(inp for inp in inputs if inp.requires_grad)
    if grad_outputs is None:
        diff_outputs = tuple(out for out in outputs if out.requires_grad)
    else:
        diff_grad_outputs = [
            (out, go) for out, go in zip(outputs, grad_outputs) if out.requires_grad
        ]
        if len(diff_grad_outputs) == 0:
            diff_outputs, grad_outputs = (), ()
        else:
            diff_outputs, grad_outputs = zip(*diff_grad_outputs)
    grad_inputs = torch.autograd.grad(
        diff_outputs,
        diff_inputs,
        grad_outputs,
        retain_graph=retain_graph,
        create_graph=create_graph,
        allow_unused=True,
    )
    result = []
    grad_inputs_iter = iter(grad_inputs)
    for inp in inputs:
        if inp.requires_grad:
            grad_input = next(grad_inputs_iter)
            if grad_input is None:
                result.append(torch.zeros_like(inp))
            else:
                result.append(grad_input)
        else:
            result.append(torch.zeros_like(inp))
    return tree_unflatten(result, inputs_spec)


def diff_arg(arg, requires_grad=True):
    def is_differentiable_arg(arg):
        if requires_grad:
            return arg.requires_grad
        else:
            return arg.is_floating_point() or arg.is_complex()

    if is_iterable_of_tensors(arg):
        if all(is_differentiable_arg(a) for a in arg):
            return True
        if all(not is_differentiable_arg(a) for a in arg):
            return False
        raise RuntimeError("NYI: The test runner can't handle this")
    return isinstance(arg, Tensor) and is_differentiable_arg(arg)


# Given f, returns an f' such that:
# - f' takes only positional arguments
# - All arguments to f' are floating-point Tensors
# - All outputs of f' are floating-point Tensors
def normalize_op_input_output2(
    f, args, kwargs, output_process_fn_grad=None, requires_grad=True
):
    flat_args, args_spec = tree_flatten(args)
    diff_argnums = tuple(
        i
        for i, arg in enumerate(flat_args)
        if diff_arg(arg, requires_grad=requires_grad)
    )
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
            result = tuple(r for r in result if torch.is_floating_point(r))
            assert len(result) > 0
        return result

    return wrapped, primals


# TODO: consolidate with normalize_op_input_output2
def normalize_op_input_output3(
    f, args, kwargs, sample_args, output_process_fn_grad=None
):
    flat_args, args_spec = tree_flatten(args)
    flat_sample_args = pytree.tree_leaves(sample_args)
    diff_argnums = tuple(
        i
        for i, (arg, sample) in enumerate(zip(flat_args, flat_sample_args))
        if diff_arg(sample, requires_grad=True)
    )
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
            result = tuple(r for r in result if torch.is_floating_point(r))
            assert len(result) > 0
        return result

    return wrapped, primals


def normalize_op_input_output(f, sample, requires_grad=True):
    args = tuple([sample.input] + list(sample.args))
    return normalize_op_input_output2(
        f,
        args,
        sample.kwargs,
        sample.output_process_fn_grad,
        requires_grad=requires_grad,
    )


def ref_vjp(f, *primals):
    result = f(*primals)

    def wrapped(cotangents):
        return _autograd_grad(_as_tuple(result), primals, _as_tuple(cotangents))

    return result, wrapped


def simulate_jvp(f, primals, tangents):
    primals_out, tangents_out = torch.autograd.functional.jvp(f, primals, tangents)
    return primals_out, tangents_out


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
        actual_args = args[: len(flat_args)]
        cotangents = args[len(flat_args) :]
        actual_args = tree_unflatten(actual_args, args_spec)
        cotangents = tree_unflatten(cotangents, cotangents_spec)

        fn, primals = normalize_op_input_output3(
            f, actual_args, kwargs, flat_args, sample.output_process_fn_grad
        )
        _, vjp_fn = vjp(fn, *primals)
        return vjp_fn(cotangents)

    return wrapped, tuple(flat_args + flat_cotangents)


# Returns a new function g(*args, *cotangents) that computes vjps and
# sample (*args, *cotangents)
def get_vjpfull_variant(f, sample):
    fn, primals = normalize_op_input_output(f, sample)
    return _get_vjpfull_variant(fn, primals)


def get_vjpfull_variant2(f, args, kwargs):
    fn, primals = normalize_op_input_output2(f, args, kwargs)
    return _get_vjpfull_variant(fn, primals)


def _get_vjpfull_variant(fn, primals):
    result = fn(*primals)
    cotangents = _as_tuple(
        tree_map(lambda x: torch.randn_like(x, requires_grad=True), result)
    )
    num_primals = len(primals)
    args = (*primals, *cotangents)

    @functools.wraps(fn)
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
    tangents = _as_tuple(tree_map(lambda x: torch.randn_like(x), primals))

    @functools.wraps(f)
    def wrapped(*args):
        tangents = args
        primals_out, tangents_out = jvp(fn, primals, tangents)

        if isinstance(primals_out, torch.Tensor):
            return (primals_out, tangents_out)
        else:
            flat_primals_out = pytree.tree_leaves(primals_out)
            flat_tangents_out = pytree.tree_leaves(tangents_out)
            return tuple(flat_primals_out + flat_tangents_out)

    return wrapped, tangents


def get_jvp_variant_primals_tangents2(
    f, args, kwargs, output_process_fn_grad=None, requires_grad=False
):
    fn, primals = normalize_op_input_output2(
        f, args, kwargs, output_process_fn_grad, requires_grad
    )
    tangents = _as_tuple(tree_map(lambda x: torch.randn_like(x), primals))
    return _get_jvp_variant(fn, primals, tangents)


def get_jvp_variant_primals_tangents(f, sample):
    # We want this higher-order variant of jvp, so that it can
    # be used to wrap vmap
    fn, primals = normalize_op_input_output(f, sample, requires_grad=False)
    tangents = _as_tuple(tree_map(lambda x: torch.randn_like(x), primals))
    return _get_jvp_variant(fn, primals, tangents)


def _get_jvp_variant(fn, primals, tangents):
    @functools.wraps(fn)
    def wrapped(*args):
        primals_in = args[: len(primals)]
        tangents_in = args[len(primals) :]
        primals_out, tangents_out = jvp(fn, primals_in, tangents_in)

        if isinstance(primals_out, torch.Tensor):
            return (primals_out, tangents_out)
        else:
            flat_primals_out = pytree.tree_leaves(primals_out)
            flat_tangents_out = pytree.tree_leaves(tangents_out)
            return tuple(flat_primals_out + flat_tangents_out)

    return wrapped, primals + tangents


def is_inplace(op, variant):
    if hasattr(variant, "__wrapped__"):
        return variant.__wrapped__ is op.get_inplace()
    return variant is op.get_inplace()


vjp_fail = {
    xfail("tensor_split"),  # data_ptr composite compliance
    decorate("nn.functional.batch_norm", decorator=skipIfRocm),
    decorate("nn.functional.instance_norm", decorator=skipIfRocm),
    # https://github.com/pytorch/pytorch/issues/96560
    decorate("nn.functional.scaled_dot_product_attention", decorator=skipIfRocm),
}

aliasing_ops = {
    "T",
    "broadcast_to",
    "conj",
    "contiguous",
    "diagonal",  # linalg.diagonal is an alias
    "expand",
    "flatten",
    "imag",
    "mH",  # adjoint is an alias
    "mT",
    "movedim",  # moveaxis is an alias
    "narrow",
    "permute",
    "positive",
    # 'ravel', is composite implicit autograd and may call clone
    "real",
    "reshape",
    "resolve_conj",
    "resolve_neg",
    "select",
    "squeeze",
    "transpose",  # swapdims and swapaxes are aliases
    "unflatten",
    "unfold",
    "unsqueeze",
    "view",
    "view_as",
    "view_as_complex",
    "view_as_real",
}

aliasing_ops_list_return = {
    "chunks",
    "dsplit",
    "hsplit",
    "split",
    "unbind",
    "vsplit",
    # 'tensor_split' not composite compliant, see vjp_fail
}

skip_noncontig = {
    "_batch_norm_with_update",
}


@unittest.skipIf(TEST_WITH_ASAN, "tests time out with asan, are probably redundant")
@unMarkDynamoStrictTest
class TestOperators(TestCase):
    @with_tf32_off  # https://github.com/pytorch/pytorch/issues/86798
    @ops(op_db + additional_op_db + autograd_function_db, allowed_dtypes=(torch.float,))
    @skipOps(
        "TestOperators",
        "test_grad",
        vjp_fail.union(
            {
                xfail(
                    "chalf", "", device_type="cpu"
                ),  # RuntimeError: "sum_cpu" not implemented for 'ComplexHalf'
                xfail(
                    "sparse.sampled_addmm", ""
                ),  # RuntimeError: Sparse CSR tensors do not have strides
                xfail(
                    "sparse.mm", "reduce"
                ),  # RuntimeError: Sparse CSR tensors do not have strides
                # Non-contiguous Bugs
                #
                # AssertionError: Tensor-likes are not close!
                xfail("_softmax_backward_data", device_type="cpu"),
                xfail("as_strided"),
                xfail("as_strided", "partial_views"),
                # RuntimeError: !self.requires_grad() || self.is_contiguous()
                xfail("as_strided_scatter"),
                # RuntimeError: Tensor must have a last dimension with stride 1
                xfail("view_as_complex"),
                # query: last dimension must be contiguous
                # Fused attention kernels require last dim to be contiguous
                xfail("nn.functional.scaled_dot_product_attention"),
                xfail("torch.ops.aten._flash_attention_forward"),
                xfail("torch.ops.aten._efficient_attention_forward"),
                # RuntimeError: Expected contiguous tensor, but got
                # non-contiguous tensor for argument #2 'grad_output'
                decorate(
                    "_batch_norm_with_update",
                    decorator=expectedFailureIf(TEST_WITH_ROCM),
                    device_type="cuda",
                ),
            }
        ),
    )
    @opsToleranceOverride(
        "TestOperators",
        "test_grad",
        (
            tol1(
                "nn.functional.binary_cross_entropy_with_logits",
                {torch.float32: tol(atol=1e-04, rtol=1e-04)},
            ),
            tol1("masked.cumprod", {torch.float32: tol(atol=1e-05, rtol=1e-05)}),
            tol1(
                "svd_lowrank",
                {torch.float32: tol(atol=3e-04, rtol=3e-04)},
                device_type="cuda",
            ),
            tol1(
                "linalg.tensorsolve",
                {torch.float32: tol(atol=3e-04, rtol=3e-04)},
                device_type="cuda",
            ),
            tol1(
                "nn.functional.multi_head_attention_forward",
                {torch.float32: tol(atol=8e-04, rtol=1e-03)},
            ),
            tol1(
                "__rmatmul__",
                {torch.float32: tol(atol=3e-04, rtol=3e-04)},
                device_type="cuda",
            ),
            tol1(
                "matmul",
                {torch.float32: tol(atol=3e-04, rtol=3e-04)},
                device_type="cuda",
            ),
        ),
    )
    def test_grad(self, device, dtype, op):
        if op.name in vjp_fail:
            self.skipTest("Skipped; Expected failures")
            return

        if not op.supports_autograd:
            self.skipTest("Skipped! Autograd not supported.")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=True)

        if is_inplace(op, op.get_op()):
            self.skipTest("Skipped for redundancy. test_vjp handles in-place testing.")
            return

        for sample in samples:
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs

            if op.name not in skip_noncontig:
                noncontig_sample = sample.noncontiguous()
                noncontig_args = [noncontig_sample.input] + list(noncontig_sample.args)
                noncontig_kwargs = noncontig_sample.kwargs

            diff_argnums = tuple(i for i, arg in enumerate(args) if diff_arg(arg))
            assert len(diff_argnums) > 0
            diff_args = tuple(args[i] for i in diff_argnums)

            def wrapped_fn(*args, **kwargs):
                result = op(*args, **kwargs)
                if sample.output_process_fn_grad is not None:
                    result = sample.output_process_fn_grad(result)

                def abs_if_complex(t):
                    if t.dtype.is_complex:
                        return t.abs()
                    return t

                # Reduce into single value for grad
                if isinstance(result, torch.Tensor):
                    return abs_if_complex(result.sum())
                result = sum(abs_if_complex(res.sum()) for res in result)
                return result

            result = grad(wrapped_fn, diff_argnums)(*args, **kwargs)
            expected = _autograd_grad(_as_tuple(wrapped_fn(*args, **kwargs)), diff_args)
            self.assertEqual(result, expected)

            if op.name not in skip_noncontig:
                result_noncontig = grad(wrapped_fn, diff_argnums)(
                    *noncontig_args, **noncontig_kwargs
                )
                self.assertEqual(result_noncontig, expected)

    @with_tf32_off  # https://github.com/pytorch/pytorch/issues/86798
    @ops(op_db + additional_op_db + autograd_function_db, allowed_dtypes=(torch.float,))
    @skipOps(
        "TestOperators",
        "test_jvp",
        set(
            {
                # Composite ops that do bad things. Need to be fixed in PyTorch core.
                # RuntimeError: Cannot access data pointer of Tensor that doesn't have storage
                xfail("tensor_split"),
                # BUG: silent incorrectness: runs and produces numerical differences
                skip("nn.functional.max_unpool1d"),  # fails everywhere except on mac
                skip(
                    "nn.functional.max_unpool2d"
                ),  # fails everywhere except on windows
                skip("nn.functional.max_unpool3d"),  # fails everywhere except on mac
                xfail(
                    "native_batch_norm"
                ),  # TODO: fails comparing None to tensor of 0s for saved_mean/var tangents
                xfail(
                    "_native_batch_norm_legit"
                ),  # TODO: fails comparing None to tensor of 0s for saved_mean/var tangents
                xfail(
                    "_batch_norm_with_update"
                ),  # TODO: fails comparing None to tensor of 0s for saved_mean/var tangents
                xfail("nn.functional.scaled_dot_product_attention"),
                xfail("torch.ops.aten._flash_attention_forward"),
                xfail("torch.ops.aten._efficient_attention_forward"),
                xfail(
                    "nn.functional.rrelu"
                ),  # in-place test errors out with no formula implemented
                xfail(
                    "NumpyExpMarkDirtyAutogradFunction"
                ),  # TODO: https://github.com/pytorch/pytorch/issues/91280
                # https://github.com/pytorch/pytorch/issues/96560
                # ROCm: NotImplementedError
                decorate("nn.functional.batch_norm", decorator=skipIfRocm),
                # ROCm: NotImplementedError
                decorate("nn.functional.instance_norm", decorator=skipIfRocm),
                # --- Non-Contiguous Failures! ---
                # This is expected to fail as the operator
                # expects last dim to have stride=1
                xfail("view_as_complex"),
                # BUG
                # AssertionError: Tensor-likes are not close!
                xfail("as_strided"),
                xfail("as_strided", "partial_views"),
                xfail("as_strided_scatter"),
                decorate(
                    "linalg.det",
                    "singular",
                    decorator=expectedFailureIf(IS_MACOS and IS_X86),
                ),
            }
        ),
    )
    @opsToleranceOverride(
        "TestOperators",
        "test_jvp",
        (
            tol1(
                "nn.functional.conv_transpose3d",
                {torch.float32: tol(atol=1e-04, rtol=1.3e-06)},
                device_type="cuda",
            ),
            tol1(
                "linalg.tensorsolve",
                {torch.float32: tol(atol=1e-04, rtol=1.3e-05)},
                device_type="cuda",
            ),
            tol1(
                "nn.functional.binary_cross_entropy_with_logits",
                {torch.float32: tol(atol=4e-04, rtol=4e-04)},
            ),
            tol1(
                "nn.functional.batch_norm", {torch.float32: tol(atol=4e-05, rtol=5e-05)}
            ),
            tol1("nn.functional.conv2d", {torch.float32: tol(atol=4e-05, rtol=5e-05)}),
            tol1("pca_lowrank", {torch.float32: tol(atol=5e-05, rtol=5e-05)}),
            tol1(
                "nn.functional.multi_head_attention_forward",
                {torch.float32: tol(atol=6e-05, rtol=2e-05)},
            ),
        ),
    )
    def test_jvp(self, device, dtype, op):
        # TODO: get rid of vjp_decomp when we add decomposition support to
        # PyTorch's forward-mode ad. Currently the decomposition support only
        # works for functorch.jvp
        VJP_DECOMP = {
            "nn.functional.logsigmoid",
        }
        if op.name in VJP_DECOMP:
            fixme_ref_jvp_local = simulate_jvp
        else:
            fixme_ref_jvp_local = ref_jvp

        if not op.supports_forward_ad and op.name not in VJP_DECOMP:
            self.skipTest("Skipped! Forward AD not supported.")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=True)

        outplace_variant = op if not is_inplace(op, op.get_op()) else None
        inplace_variant = op.inplace_variant if op.supports_inplace_autograd else None

        for sample in samples:
            if outplace_variant:
                self.jvp_opinfo_test(
                    outplace_variant,
                    sample,
                    sample.output_process_fn_grad,
                    clone_inputs=False,
                    fixme_ref_jvp_local=fixme_ref_jvp_local,
                    test_noncontig=op.name not in skip_noncontig,
                )
            if is_valid_inplace_sample_input(sample, op, inplace_variant):
                self.jvp_opinfo_test(
                    inplace_variant,
                    sample,
                    sample.output_process_fn_grad,
                    clone_inputs=True,
                    fixme_ref_jvp_local=fixme_ref_jvp_local,
                    test_noncontig=op.name not in skip_noncontig,
                )

    def jvp_opinfo_test(
        self,
        fn,
        sample,
        output_process_fn,
        clone_inputs,
        fixme_ref_jvp_local,
        test_noncontig,
    ):
        # NB: we used requires_grad=True to determine where the primals are,
        # but don't need that information otherwise
        args = (sample.input,) + sample.args
        kwargs = sample.kwargs
        contig_fn, primals = normalize_op_input_output2(
            fn, args, kwargs, output_process_fn, requires_grad=True
        )
        orig_primals = tree_map(lambda x: x.detach(), primals)
        orig_tangents = tree_map(lambda x: torch.randn_like(x), primals)

        def maybe_clone_inputs():
            if clone_inputs:
                primals = tree_map(torch.clone, orig_primals)
                tangents = tree_map(torch.clone, orig_tangents)
                return primals, tangents
            return orig_primals, orig_tangents

        primals, tangents = maybe_clone_inputs()
        expected_primal_outs, expected_tangent_outs = fixme_ref_jvp_local(
            contig_fn, primals, tangents
        )

        primals, tangents = maybe_clone_inputs()
        primal_outs, tangent_outs = jvp(contig_fn, primals, tangents)

        self.assertEqual(primal_outs, expected_primal_outs)
        self.assertEqual(tangent_outs, expected_tangent_outs)

        if test_noncontig:
            noncontig_sample = sample.noncontiguous()
            noncontig_args = (noncontig_sample.input,) + noncontig_sample.args
            noncontig_kwargs = sample.kwargs
            noncontig_fn, primals = normalize_op_input_output2(
                fn,
                noncontig_args,
                noncontig_kwargs,
                output_process_fn,
                requires_grad=True,
            )
            noncontig_primals = tree_map(lambda x: x.detach(), primals)
            noncontig_tangents = tree_map(
                lambda x: noncontiguous_like(x), orig_tangents
            )
            noncontig_primal_outs, noncontig_tangent_outs = jvp(
                noncontig_fn, noncontig_primals, noncontig_tangents
            )

            self.assertEqual(noncontig_primal_outs, expected_primal_outs)
            self.assertEqual(noncontig_tangent_outs, expected_tangent_outs)

    @with_tf32_off  # https://github.com/pytorch/pytorch/issues/86798
    @ops(op_db + additional_op_db + autograd_function_db, allowed_dtypes=(torch.float,))
    @skipOps(
        "TestOperators",
        "test_vjp",
        vjp_fail.union(
            {
                xfail("sparse.sampled_addmm", ""),
                xfail("sparse.mm", "reduce"),
                # ---- Non-Contiguous Failures ----
                # This is expected to fail as the operator
                # expects last dim to have stride=1
                xfail("view_as_complex"),
                # RuntimeError: query: last dimension must be contiguous
                # The fused attention kernels require the last dim to be contiguous
                xfail("nn.functional.scaled_dot_product_attention"),
                xfail("torch.ops.aten._flash_attention_forward"),
                xfail("torch.ops.aten._efficient_attention_forward"),
                # BUG
                # AssertionError: Tensor-likes are not close!
                xfail("as_strided"),
                xfail("as_strided_scatter"),
                xfail("_softmax_backward_data", device_type="cpu"),
                xfail("as_strided", "partial_views"),
            }
        ),
    )
    @opsToleranceOverride(
        "TestOperators",
        "test_vjp",
        (
            tol1(
                "nn.functional.conv_transpose3d",
                {torch.float32: tol(atol=5e-05, rtol=9e-05)},
                device_type="cuda",
            ),
            tol1(
                "nn.functional.binary_cross_entropy_with_logits",
                {torch.float32: tol(atol=1e-04, rtol=1e-04)},
            ),
            tol1(
                "nn.functional.multi_head_attention_forward",
                {torch.float32: tol(atol=2e-03, rtol=2e-04)},
            ),
            tol1("__rmatmul__", {torch.float32: tol(atol=1e-05, rtol=1e-05)}),
            tol1("matmul", {torch.float32: tol(atol=1e-05, rtol=1e-05)}),
            tol2(
                "linalg.pinv", "hermitian", {torch.float32: tol(atol=1e-05, rtol=1e-05)}
            ),
            tol1("linalg.tensorsolve", {torch.float32: tol(atol=1e-05, rtol=1e-05)}),
            tol1("linalg.multi_dot", {torch.float32: tol(atol=1e-04, rtol=1e-04)}),
            tol1("svd_lowrank", {torch.float32: tol(atol=1e-04, rtol=1e-04)}),
            tol1("pca_lowrank", {torch.float32: tol(atol=1e-04, rtol=1e-04)}),
        ),
    )
    def test_vjp(self, device, dtype, op):
        if not op.supports_autograd:
            self.skipTest("Skipped! Autograd not supported.")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=True)

        def _test(_op, inplace=False):
            for sample in samples:
                if inplace and not is_valid_inplace_sample_input(
                    sample, op, op.inplace_variant
                ):
                    continue
                fn, primals = normalize_op_input_output(_op, sample)
                result = fn(*primals)
                cotangents = tree_map(lambda x: torch.randn_like(x), result)

                out, vjp_fn = vjp(fn, *primals)
                self.assertEqual(out, result)
                result_vjps = vjp_fn(cotangents)

                _, vjp_fn = ref_vjp(fn, *primals)
                expected_vjps = vjp_fn(cotangents)

                self.assertEqual(result_vjps, expected_vjps)

                if op.name not in skip_noncontig:
                    noncontig_fn, noncontig_primals = normalize_op_input_output(
                        _op, sample.noncontiguous()
                    )
                    noncontig_cotangents = tree_map(
                        lambda x: noncontiguous_like(x), cotangents
                    )
                    out_noncontig, vjp_fn = vjp(noncontig_fn, *noncontig_primals)
                    self.assertEqual(out_noncontig, result)
                    noncontig_result_vjps = vjp_fn(noncontig_cotangents)
                    self.assertEqual(noncontig_result_vjps, expected_vjps)

        _test(op)
        for a_op in op.aliases:
            _test(a_op)
        if op.inplace_variant:

            def f(inp, *args, **kwargs):
                return op.inplace_variant(inp.clone(), *args, **kwargs)

            _test(f, inplace=True)

    @ops(op_db + additional_op_db + autograd_function_db, allowed_dtypes=(torch.float,))
    @skipOps(
        "TestOperators",
        "test_vjpvjp",
        vjp_fail.union(
            {
                skip("nn.functional.max_unpool1d"),  # silent incorrectness; Flaky
                skip("nn.functional.max_unpool2d"),  # silent incorrectness; Flaky
                xfail("nn.functional.ctc_loss"),  # Not Implemented
                xfail(
                    "native_layer_norm", ""
                ),  # Expected a proper Tensor but got None for argument #1 'other'
                xfail("sparse.sampled_addmm", ""),  # sparse tensors have no strides
                xfail("sparse.mm", "reduce"),  # sparse tensors have no strides
                skip("nn.functional.scaled_dot_product_attention"),
                xfail("torch.ops.aten._flash_attention_forward"),
                xfail("torch.ops.aten._efficient_attention_forward"),
                # AssertionError: Tensor-likes are not close!
                # Mismatched elements: 1 / 15 (6.7%)
                # Greatest absolute difference: 24.0 at index (2, 4) (up to 1e-05 allowed)
                # Greatest relative difference: 1.7933241714393998e-06 at index (2, 4) (up to 1.3e-06 allowed)
                # The failure occurred for item [0]
                xfail("masked.prod"),
            }
        ),
    )
    @opsToleranceOverride(
        "TestOperators",
        "test_vjpvjp",
        (
            tol1(
                "nn.functional.conv_transpose3d",
                {torch.float32: tol(atol=5e-05, rtol=9e-05)},
                device_type="cuda",
            ),
            tol1("prod", {torch.float32: tol(atol=2e-05, rtol=1e-04)}),
            tol1("masked.cumprod", {torch.float32: tol(atol=5e-04, rtol=5e-04)}),
            tol1("cumprod", {torch.float32: tol(atol=5e-04, rtol=5e-04)}),
            tol1("linalg.vander", {torch.float32: tol(atol=5e-04, rtol=5e-04)}),
            tol2(
                "linalg.det", "singular", {torch.float32: tol(atol=2e-05, rtol=2e-05)}
            ),
        ),
    )
    def test_vjpvjp(self, device, dtype, op):
        if not op.supports_autograd:
            self.skipTest("Skipped! Autograd not supported.")
            return
        if not op.supports_gradgrad:
            self.skipTest("Skipped! Operation does not support gradgrad")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=True)

        def test(_op, inplace=False):
            for sample in samples:
                if inplace and not is_valid_inplace_sample_input(
                    sample, op, op.inplace_variant
                ):
                    continue
                fn, args = get_vjpfull_variant(_op, sample)
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

        test(op)
        if op.inplace_variant:

            def fn(inp, *args, **kwargs):
                return op.inplace_variant(inp.clone(), *args, **kwargs)

            test(fn, inplace=True)

    @with_tf32_off  # https://github.com/pytorch/pytorch/issues/86798
    @skipOps(
        "TestOperators",
        "test_vmapvjpvjp",
        vjp_fail.union(
            {
                skip("atleast_1d"),  # Takes too long
                skip("atleast_2d"),  # Takes too long
                skip("atleast_3d"),  # Takes too long
                skip("ormqr"),  # Takes too long
                xfail("as_strided"),  # incorrect output
                xfail("as_strided", "partial_views"),  # incorrect output
                xfail("as_strided_scatter"),  # incorrect output
                skip("bernoulli"),  # calls random op
                xfail("bfloat16"),  # rank 4 tensor for channels_last
                xfail("cdouble"),  # rank 4 tensor for channels_last
                xfail("cfloat"),  # rank 4 tensor for channels_last
                xfail("chalf"),  # rank 4 tensor for channels_last
                xfail("double"),  # rank 4 tensor for channels_last
                xfail("float"),  # rank 4 tensor for channels_last
                xfail("half"),  # rank 4 tensor for channels_last
                xfail(
                    "NumpyCubeNotComposableAutogradFunction"
                ),  # Not composable autograd.Function
                # It looks like you're either (1) calling .item() on a Tensor or
                # (2) attempting to use a Tensor in some data-dependent control flow or
                # (3) encountering this error in PyTorch internals.
                xfail("index_reduce", "prod"),
                decorate(
                    "linalg.householder_product", decorator=runOnRocm
                ),  # works on ROCm
                xfail(
                    "nanquantile", device_type="cpu"
                ),  # vmap not implemented for at::equal.
                xfail("native_layer_norm"),  # vmap: inplace into a regular tensor
                # got a batched tensor as input while the running_mean or running_var,
                # which will be updated in place, were not batched.
                xfail("nn.functional.batch_norm"),
                xfail(
                    "nn.functional.binary_cross_entropy"
                ),  # vmap: inplace into a regular tensor
                xfail(
                    "nn.functional.ctc_loss"
                ),  # derivate not implemented for _ctc_loss_backward
                # flaky on ROCM needs investigation
                decorate("nn.functional.conv_transpose2d", decorator=skipIfRocm),
                skip("nn.functional.dropout"),  # calls random op
                skip("nn.functional.dropout2d"),  # calls random op
                skip("nn.functional.dropout3d"),  # calls random op
                skip("nn.functional.alpha_dropout"),  # calls random op
                skip(
                    "nn.functional.feature_alpha_dropout", "with_train"
                ),  # calls random op
                skip("nn.functional.fractional_max_pool2d"),  # calls random op
                skip("nn.functional.fractional_max_pool3d"),  # calls random op
                xfail("nn.functional.scaled_dot_product_attention"),  # randomness
                xfail("torch.ops.aten._efficient_attention_forward"),  # outputs ints
                xfail("nn.functional.multi_head_attention_forward"),  # randomness
                # It looks like you're either (1) calling .item() on a Tensor or
                # (2) attempting to use a Tensor in some data-dependent control flow or
                # (3) encountering this error in PyTorch internals.
                xfail("nn.functional.gaussian_nll_loss"),
                # got a batched tensor as input while the running_mean or running_var,
                # which will be updated in place, were not batched.
                xfail("nn.functional.instance_norm"),
                xfail(
                    "nn.functional.layer_norm"
                ),  # vmap: inplace into a regular tensor
                # RuntimeError: NYI: querying is_contiguous inside of vmap
                # for memory_format other than torch.contiguous_formats
                xfail("nn.functional.max_pool2d"),
                # RuntimeError: NYI: Tensor.clone(memory_format) inside vmap is only
                # supported with memory_format torch.preserve_format or
                # torch.contiguous_format (got ChannelsLast)
                xfail("nn.functional.max_unpool2d"),
                # RuntimeError: NYI: Tensor.clone(memory_format) inside vmap is only
                # supported with memory_format torch.preserve_format
                # or torch.contiguous_format (got ChannelsLast)s
                xfail("nn.functional.max_unpool2d", "grad"),
                xfail(
                    "nn.functional.rrelu"
                ),  # RuntimeError: vmap: we do not yet support aten::rrelu_with_noise.
                xfail("normal"),  # calls random op
                xfail("normal", "number_mean"),  # calls random op
                xfail("pca_lowrank"),  # calls random op
                # https://github.com/pytorch/pytorch/issues/96560
                decorate("linalg.pinv", "hermitian", decorator=skipIfRocm),
                xfail(
                    "quantile", device_type="cpu"
                ),  # Batching rule not implemented for `at::equal`
                xfail(
                    "scatter_reduce", "prod"
                ),  # vmap (looks like you are calling item/data-dependent)
                xfail(
                    "sparse.sampled_addmm"
                ),  # RuntimeError: Sparse CSR tensors do not have strides
                xfail(
                    "sparse.mm", "reduce"
                ),  # RuntimeError: Sparse CSR tensors do not have strides
                xfail("svd_lowrank"),  # calls random op
                xfail("to"),  # rank 4 tensor for channels_last
                xfail(
                    "view_as_complex"
                ),  # RuntimeError: Tensor must have a last dimension with stride 1
                # got a batched tensor as input while the running_mean or running_var,
                # which will be updated in place, were not batched.
                xfail("nn.functional.batch_norm", "without_cudnn"),
                # view doesn't work on sparse
                xfail("to_sparse"),
                xfail("native_batch_norm"),
                xfail("_native_batch_norm_legit"),
                # TODO: implement batching rule
                xfail("_batch_norm_with_update"),
            }
        ),
    )
    @ops(op_db + additional_op_db + autograd_function_db, allowed_dtypes=(torch.float,))
    @toleranceOverride({torch.float32: tol(atol=1e-04, rtol=1e-04)})
    @opsToleranceOverride(
        "TestOperators",
        "test_vmapvjpvjp",
        (
            tol1("linalg.svd", {torch.float32: tol(atol=1e-03, rtol=5e-04)}),
            tol1("linalg.lu_factor", {torch.float32: tol(atol=2e-03, rtol=2e-02)}),
            tol1("svd", {torch.float32: tol(atol=1e-03, rtol=5e-04)}),
            tol1("matrix_exp", {torch.float32: tol(atol=1e-03, rtol=5e-04)}),
        ),
    )
    @skipOps(
        "TestOperators",
        "test_vmapvjpvjp",
        {
            xfail("as_strided", "partial_views"),
        },
    )
    def test_vmapvjpvjp(self, device, dtype, op):
        # Since, we test `vjpvjp` independently,
        # for this test, we just verify that vmap
        # of `vjpvjp` is correct.
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
            cotangents = pytree.tree_leaves(cotangents)
            num_args = len(args)

            args_and_cotangents = tuple(args) + tuple(cotangents)

            def vjp_of_vjp(*args_and_cotangents):
                args = args_and_cotangents[:num_args]
                cotangents = args_and_cotangents[num_args:]
                result, vjp_fn = vjp(fn, *args)
                result_vjps = vjp_fn(cotangents)
                result = pytree.tree_leaves(result)
                result_vjps = pytree.tree_leaves(result_vjps)
                return (*result, *result_vjps)

            is_batch_norm_and_training = is_batch_norm_training(op.name, sample.kwargs)
            generator = get_fallback_and_vmap_exhaustive(
                vjp_of_vjp,
                args_and_cotangents,
                {},
                is_batch_norm_and_training=is_batch_norm_and_training,
            )
            for loop_out, batched_out in generator:
                self.assertEqual(loop_out, batched_out)

    vmapvjp_fail = vjp_fail.union(
        {
            # -------------------- ALLOWED FAILURES --------------------------------
            # The following are not bugs and are expected behavior
            xfail("masked_select"),  # Not possible due to dynamic shapes
            skip("bernoulli"),  # randomness
            skip("normal", ""),  # randomness
            skip("normal", "number_mean"),  # randomness
            skip("nn.functional.rrelu"),  # randomness
            skip("nn.functional.feature_alpha_dropout", "with_train"),  # randomness
            skip("nn.functional.feature_alpha_dropout", "without_train"),  # randomness
            skip("nn.functional.dropout"),  # randomness
            skip("nn.functional.dropout2d"),  # randomness
            skip("nn.functional.dropout3d", ""),  # randomness
            skip("nn.functional.alpha_dropout"),  # randomness
            skip("nn.functional.scaled_dot_product_attention"),  # randomness
            xfail("torch.ops.aten._efficient_attention_forward"),  # outputs ints
            skip("nn.functional.multi_head_attention_forward"),  # randomness
            xfail(
                "index_put", ""
            ),  # not possible due to dynamic shapes; we support a subset
            xfail("nn.functional.fractional_max_pool2d"),  # random
            xfail("nn.functional.fractional_max_pool3d"),  # random
            xfail("pca_lowrank", ""),  # randomness
            xfail("svd_lowrank", ""),  # randomness
            xfail("to_sparse", ""),  # non-dense output
            skip(
                "to"
            ),  # RuntimeError: required rank 4 tensor to use channels_last format
            xfail("as_strided", "partial_views"),
            xfail(
                "NumpyCubeNotComposableAutogradFunction"
            ),  # Not composable autograd.Function
            # ----------------------------------------------------------------------
            # ---------------------------- BUGS ------------------------------------
            # All of the following are bugs and need to be fixed
            skip(
                "linalg.svdvals"
            ),  # # really annoying thing where it passes correctness check but not has_batch_rule
            skip("native_batch_norm"),
            skip("_native_batch_norm_legit"),
            # TODO: implement batching rule
            skip("_batch_norm_with_update"),
            xfail("__getitem__", ""),  # dynamic error
            xfail("nanquantile", device_type="cpu"),  # checks q via a .item() call
            xfail("nn.functional.gaussian_nll_loss"),  # checks var for if any value < 0
            xfail("narrow"),  # .item() call
            xfail("quantile", device_type="cpu"),  # checks q via a .item() call
            xfail("view_as_complex"),  # Tensor must have a last dimension with stride 1
            # required rank 4 tensor to use channels_last format
            xfail("bfloat16"),
            xfail("double"),
            xfail("float"),
            xfail("half"),
            xfail("cdouble", ""),
            xfail("cfloat", ""),
            xfail("chalf", ""),
            xfail("scatter_reduce", "prod"),  # item call
            # Batching rule not implemented for aten::_use_cudnn_ctc_loss.Tensor
            xfail("nn.functional.ctc_loss", device_type="cuda"),
            # NYI: querying is_contiguous inside of vmap for memory_format other than torch.contiguous_format
            xfail("nn.functional.max_unpool2d"),
            xfail("nn.functional.max_unpool2d", "grad"),
            xfail("sparse.sampled_addmm", ""),
            xfail("sparse.mm", "reduce"),
            xfail("as_strided_scatter", ""),  # calls as_strided
            xfail("index_reduce", "prod"),  # .item() call
            # ---------------------------------------------------------------------
        }
    )

    @with_tf32_off  # https://github.com/pytorch/pytorch/issues/86798
    @ops(op_db + additional_op_db + autograd_function_db, allowed_dtypes=(torch.float,))
    @toleranceOverride({torch.float32: tol(atol=1e-04, rtol=1e-04)})
    @opsToleranceOverride(
        "TestOperators",
        "test_vmapvjp",
        (
            tol1(
                "linalg.svd",
                {torch.float32: tol(atol=5e-04, rtol=1e-04)},
                device_type="cuda",
            ),
            tol1(
                "svd", {torch.float32: tol(atol=5e-04, rtol=1e-04)}, device_type="cuda"
            ),
            tol1(
                "linalg.householder_product",
                {torch.float32: tol(atol=1e-04, rtol=1e-04)},
            ),
            tol1(
                "matrix_exp",
                {torch.float32: tol(atol=5e-04, rtol=1e-04)},
                device_type="cuda",
            ),
        ),
    )
    @skipOps(
        "TestOperators",
        "test_vmapvjp",
        vmapvjp_fail.union(
            {
                xfail("as_strided"),
                xfail("as_strided", "partial_views"),
            }
        ),
    )
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
            is_batch_norm_and_training = is_batch_norm_training(op.name, sample.kwargs)
            generator = get_fallback_and_vmap_exhaustive(
                fn, args, {}, is_batch_norm_and_training=is_batch_norm_and_training
            )
            for loop_out, batched_out in generator:
                self.assertEqual(loop_out, batched_out)

    vmapjvpall_fail = {
        # -------------------- ALLOWED FAILURES --------------------------------
        # The following are expected (not a bug)
        skip("bernoulli", ""),  # randomness
        skip("nn.functional.dropout"),  # randomness
        skip("nn.functional.rrelu"),  # randomness
        skip("nn.functional.dropout2d", ""),
        skip("nn.functional.dropout3d", ""),
        skip("nn.functional.scaled_dot_product_attention"),  # randomness
        xfail("torch.ops.aten._efficient_attention_forward"),  # outputs ints
        skip("nn.functional.multi_head_attention_forward"),  # randomness
        skip("nn.functional.alpha_dropout"),  # randomness
        skip("nn.functional.feature_alpha_dropout", "without_train"),
        skip("nn.functional.feature_alpha_dropout", "with_train"),
        xfail(
            "nn.functional.fractional_max_pool2d"
        ),  # Cannot access data pointer of Tensor that doesn't have storage
        xfail(
            "nn.functional.fractional_max_pool3d"
        ),  # Cannot access data pointer of Tensor that doesn't have storage
        # Not actually a problem: embedding with max_norm mutates the weight
        # and causes different runs to produce different results.
        # skip because this is flaky depending on what the max_norm is!
        skip("nn.functional.embedding", ""),
        skip("to"),  # RuntimeError: required rank 4 tensor to use channels_last format
        xfail(
            "NumpyExpMarkDirtyAutogradFunction"
        ),  # vmap: inplace into a regular tensor
        # ----------------------------------------------------------------------
        # ---------------------------- BUGS ------------------------------------
        # The following are bugs that we should fix
        xfail("masked.mean"),  # silent incorrectness (nan difference)
        xfail("as_strided", "partial_views"),  # Tensor-likes are not close!
        xfail(
            "nn.functional.soft_margin_loss", ""
        ),  # soft_margin_loss_backward does not support forward-ad
        xfail("tensor_split"),  # data_ptr composite compliance
        xfail("quantile"),  # at::equal batching rule (cpu), also, in-place vmap (cuda)
        skip("as_strided"),  # Test runner cannot handle this
        # requires special handling, and does not yet have a batching rule. Feel free to file a github issue!
        xfail("as_strided_scatter"),
        xfail(
            "nn.functional.gaussian_nll_loss"
        ),  # .item or data-dependent control flow
        xfail("scatter"),  # forward-mode AD does not support at::scatter
        xfail(
            "nanquantile"
        ),  # at::equal batching rule (cpu), also, in-place vmap (cuda)
        xfail("view_as_complex"),  # Tensor must have a last dimension with stride 1
        skip("pca_lowrank", ""),  # randomness
        skip("svd_lowrank", ""),  # randomness
        xfail("double"),  # required rank 4 tensor to use channels_last format
        xfail("cdouble"),  # required rank 4 tensor to use channels_last format
        # potential silent incorrectness
        skip(
            "nn.functional.max_unpool1d"
        ),  # Flaky, seems to sometimes his max_unpool2d
        skip("nn.functional.max_unpool2d"),  # fails everywhere except on mac
        skip("nn.functional.max_unpool3d"),  # fails everywhere except on mac
        # erroring because running_mean and running_var aren't differentiable
        xfail("nn.functional.batch_norm"),
        xfail("nn.functional.batch_norm", "without_cudnn"),
        xfail("native_batch_norm"),
        xfail("_native_batch_norm_legit"),
        # TODO: implement batching rule
        xfail("_batch_norm_with_update"),
        # https://github.com/pytorch/pytorch/issues/96560
        # ROCm: NotImplementedError
        decorate("nn.functional.instance_norm", decorator=skipIfRocm),
        # ----------------------------------------------------------------------
    }

    @with_tf32_off  # https://github.com/pytorch/pytorch/issues/86798
    @ops(op_db + additional_op_db + autograd_function_db, allowed_dtypes=(torch.float,))
    @toleranceOverride({torch.float32: tol(atol=1e-04, rtol=1e-04)})
    @opsToleranceOverride(
        "TestOperators",
        "test_vmapjvpall",
        (
            tol1(
                "nn.functional.conv_transpose3d",
                {torch.float32: tol(atol=2e-04, rtol=9e-3)},
                device_type="cuda",
            ),
            tol1(
                "linalg.householder_product",
                {torch.float32: tol(atol=2e-04, rtol=9e-3)},
            ),
        ),
    )
    @skipOps(
        "TestOperators",
        "test_vmapjvpall",
        vmapjvpall_fail.union(
            {
                decorate(
                    "linalg.det",
                    "singular",
                    decorator=expectedFailureIf(IS_MACOS and IS_X86),
                ),
            }
        ),
    )
    # This is technically a superset of test_vmapjvp. We should either delete test_vmapjvp
    # or figure out if we can split vmapjvpall. It's useful to keep test_vmapjvp intact
    # because that corresponds to "batched forward-mode AD" testing in PyTorch core
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
            args = tuple(arg_values) + tuple(kwarg_values)
            fn, args = get_jvp_variant_primals_tangents(op, sample)
            is_batch_norm_and_training = is_batch_norm_training(op.name, kwarg_values)
            generator = get_fallback_and_vmap_exhaustive(
                fn, args, {}, is_batch_norm_and_training=is_batch_norm_and_training
            )
            for loop_out, batched_out in generator:
                self.assertEqual(loop_out, batched_out)

    @ops(op_db + additional_op_db + autograd_function_db, allowed_dtypes=(torch.float,))
    @skipOps(
        "TestOperators",
        "test_vmapjvpall_has_batch_rule",
        vmapjvpall_fail.union(
            {
                skip(
                    "to"
                ),  # RuntimeError: required rank 4 tensor to use channels_last format
                xfail(
                    "cdouble"
                ),  # RuntimeError: required rank 4 tensor to use channels_last format
                xfail("cumprod"),
                xfail("masked_fill"),
                xfail("fill"),
                skip("masked.mean"),  # ???
                xfail("masked_scatter"),
                xfail("put"),
                xfail("take"),
                xfail("nn.functional.feature_alpha_dropout", "without_train"),
                xfail("nn.functional.dropout2d", ""),
                xfail("pca_lowrank", ""),
                xfail("svd_lowrank", ""),
                xfail("nn.functional.feature_alpha_dropout", "with_train"),
                xfail("special.log_ndtr", ""),
                xfail("fft.ihfft2"),  # conj_physical fallback
                xfail("fft.ihfftn"),  # conj_physical fallback
                xfail("nn.functional.max_unpool3d", "grad"),
                xfail("nn.functional.max_unpool2d", "grad"),
                xfail("nn.functional.soft_margin_loss", ""),
                xfail("nn.functional.max_unpool1d", "grad"),
                xfail("nn.functional.embedding", ""),
                xfail(
                    "scatter_reduce", "sum"
                ),  # aten::scatter_reduce.two hit the vmap fallback
                xfail(
                    "scatter_reduce", "mean"
                ),  # aten::scatter_reduce.two hit the vmap fallback
                xfail(
                    "scatter_reduce", "amin"
                ),  # aten::scatter_reduce.two hit the vmap fallback
                xfail(
                    "scatter_reduce", "amax"
                ),  # aten::scatter_reduce.two hit the vmap fallback
                xfail("nn.functional.glu"),
                xfail("nn.functional.bilinear"),  # trilinear doesn't have batching rule
                xfail("linalg.lu", ""),
                xfail("nn.functional.dropout3d", ""),
                xfail("as_strided_scatter", ""),
                xfail("masked.cumprod", ""),
                xfail("renorm"),  # hit vmap fallback, which is disabled
            }
        ),
    )
    @toleranceOverride({torch.float32: tol(atol=1e-04, rtol=1e-04)})
    def test_vmapjvpall_has_batch_rule(self, device, dtype, op):
        if is_inplace(op, op.get_op()):
            # TODO: test in-place
            self.skipTest("Skipped! NYI: inplace-testing not supported.")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=False)

        if not op.supports_forward_ad:
            self.skipTest("Skipped! Forward AD not supported.")
            return

        def test():
            for sample in samples:
                arg_values = [sample.input] + list(sample.args)
                kwarg_values = sample.kwargs
                args = tuple(arg_values) + tuple(kwarg_values)
                fn, args = get_jvp_variant_primals_tangents(op, sample)
                is_batch_norm_and_training = is_batch_norm_training(
                    op.name, kwarg_values
                )
                for loop_out, batched_out in get_fallback_and_vmap_exhaustive(
                    fn,
                    args,
                    {},
                    is_batch_norm_and_training=is_batch_norm_and_training,
                    compute_loop_out=False,
                ):
                    pass

        check_vmap_fallback(self, test, op, dry_run=False)

    @ops(op_db + additional_op_db + autograd_function_db, allowed_dtypes=(torch.float,))
    @toleranceOverride({torch.float32: tol(atol=1e-04, rtol=1e-04)})
    @skipOps(
        "TestOperators",
        "test_vmapvjp_has_batch_rule",
        vmapvjp_fail.union(
            {
                skip(
                    "to"
                ),  # RuntimeError: required rank 4 tensor to use channels_last format
                xfail("view_as_complex"),
                xfail("cummax"),
                xfail("cummin"),
                xfail("fill"),
                xfail(
                    "narrow"
                ),  # Batching rule not implemented for `narrow.Tensor` (and view op)
                xfail("special.log_ndtr"),
                xfail("linalg.householder_product"),
                xfail("masked_fill"),
                xfail("masked_scatter"),
                xfail("masked_select"),
                xfail("nanquantile"),
                xfail("ormqr"),
                xfail("put"),
                xfail(
                    "scatter_reduce", "sum"
                ),  # aten::scatter_reduce.two hit the vmap fallback
                xfail(
                    "scatter_reduce", "mean"
                ),  # aten::scatter_reduce.two hit the vmap fallback
                xfail(
                    "scatter_reduce", "amin"
                ),  # aten::scatter_reduce.two hit the vmap fallback
                xfail(
                    "scatter_reduce", "amax"
                ),  # aten::scatter_reduce.two hit the vmap fallback
                xfail("quantile"),
                xfail("renorm"),
                xfail("take"),
                xfail("tensor_split"),
                xfail("to_sparse"),
                xfail("unfold"),
                xfail("unfold_copy"),
                xfail("nn.functional.dropout"),
                xfail("fft.ihfft2"),
                xfail("fft.ihfftn"),
                xfail("nn.functional.gaussian_nll_loss"),
                xfail("nn.functional.bilinear"),
                xfail("nn.functional.fractional_max_pool3d"),
                xfail("nn.functional.ctc_loss"),
                xfail("nn.functional.rrelu"),
                xfail("nn.functional.embedding_bag"),
                xfail("nn.functional.fractional_max_pool2d"),
                xfail("nn.functional.feature_alpha_dropout", "with_train"),
                xfail("pca_lowrank", ""),
                xfail("nn.functional.dropout2d", ""),
                xfail("nn.functional.feature_alpha_dropout", "without_train"),
                xfail("svd_lowrank", ""),
                xfail("nn.functional.max_unpool2d", ""),
                xfail("nn.functional.multi_margin_loss", ""),
                xfail("nn.functional.multilabel_margin_loss", ""),
                xfail("nn.functional.pdist", ""),
                xfail("scatter_reduce", "prod"),
                xfail("nn.functional.max_unpool1d", ""),
                xfail("nn.functional.max_unpool3d", ""),
                xfail("nn.functional.max_unpool3d", "grad"),
                xfail("nn.functional.soft_margin_loss", ""),
                xfail("nn.functional.max_unpool1d", "grad"),
                xfail("nn.functional.max_unpool2d", "grad"),
                xfail("linalg.lu", ""),
                xfail("cdouble", ""),
                xfail("cfloat", ""),
                xfail("chalf", ""),
                xfail(
                    "index_reduce", "prod"
                ),  # aten::index_reduce hit the vmap fallback which is currently disabled
                xfail(
                    "index_reduce", "mean"
                ),  # aten::index_reduce hit the vmap fallback which is currently disabled
                xfail(
                    "index_reduce", "amax"
                ),  # aten::index_reduce hit the vmap fallback which is currently disabled
                xfail(
                    "index_reduce", "amin"
                ),  # aten::index_reduce hit the vmap fallback which is currently disabled
                xfail("nn.functional.dropout3d", ""),
                xfail("as_strided_scatter", ""),
                xfail("_segment_reduce", "offsets"),
                xfail("_segment_reduce", "lengths"),
                xfail("sparse.sampled_addmm", ""),
                xfail("sparse.mm", "reduce"),
                xfail("native_batch_norm"),
                xfail("_native_batch_norm_legit"),
                # TODO: implement batching rule
                xfail("_batch_norm_with_update"),
                xfail("native_dropout_backward"),
                xfail(
                    "index_fill"
                ),  # aten::_unique hit the vmap fallback which is currently disabled
            }
        ),
    )
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
                is_batch_norm_and_training = is_batch_norm_training(
                    op.name, sample.kwargs
                )
                for loop_out, batched_out in get_fallback_and_vmap_exhaustive(
                    fn,
                    args,
                    {},
                    is_batch_norm_and_training=is_batch_norm_and_training,
                    compute_loop_out=False,
                ):
                    pass
                for a_op in op.aliases:
                    fn, args = get_vjp_fn_and_args_with_cotangents(
                        a_op, sample, cotangents
                    )
                    for loop_out, batched_out in get_fallback_and_vmap_exhaustive(
                        fn,
                        args,
                        {},
                        is_batch_norm_and_training=is_batch_norm_and_training,
                        compute_loop_out=False,
                    ):
                        pass

        check_vmap_fallback(self, test, op, dry_run=False)

    @ops(op_db + additional_op_db + autograd_function_db, allowed_dtypes=(torch.float,))
    @skipOps(
        "TestOperators",
        "test_vjpvmap",
        vjp_fail.union(
            {
                skip("bernoulli", ""),  # vjpvmap testing can't handle randomness
                skip("normal", ""),  # vjpvmap testing can't handle randomness
                skip(
                    "normal", "number_mean"
                ),  # vjpvmap testing can't handle randomness
                skip("nn.functional.rrelu"),  # randomness
                skip("nn.functional.feature_alpha_dropout", "with_train"),  # randomness
                skip(
                    "nn.functional.feature_alpha_dropout", "without_train"
                ),  # randomness
                skip("nn.functional.scaled_dot_product_attention"),
                xfail("torch.ops.aten._efficient_attention_forward"),  # outputs ints
                skip("nn.functional.multi_head_attention_forward"),  # randomness
                skip("nn.functional.alpha_dropout"),  # randomness
                skip(
                    "to"
                ),  # RuntimeError: required rank 4 tensor to use channels_last format
                skip("to_sparse", ""),  # non-dense output
                skip("ormqr", ""),  # takes too long
                xfail(
                    "NumpyCubeNotComposableAutogradFunction"
                ),  # Not composable autograd.Function
                # fallback path doesn't work
                # All of the following are bugs and need to be fixed
                xfail("__getitem__", ""),
                xfail("index_put", ""),
                xfail("view_as_complex"),
                xfail("nn.functional.gaussian_nll_loss"),
                xfail("masked_select"),
                xfail(
                    "narrow"
                ),  # Batching rule not implemented for `narrow.Tensor` (and view op)
                skip(
                    "nn.functional.fractional_max_pool3d"
                ),  # generator works on cpu, fails on cuda
                skip(
                    "nn.functional.fractional_max_pool2d"
                ),  # generator works on cpu, fails on cuda
                xfail("column_stack", ""),
                xfail("nn.functional.dropout2d", ""),
                xfail("svd_lowrank", ""),
                xfail("pca_lowrank", ""),
                xfail("clamp"),
                # something weird happening with channels_last
                xfail("bfloat16"),
                xfail("double"),
                xfail("float"),
                xfail("half"),
                xfail("cdouble"),
                xfail("cfloat"),
                xfail("nn.functional.dropout3d", ""),
                xfail("as_strided_scatter", ""),
                xfail("sparse.sampled_addmm", ""),
                xfail("sparse.mm", "reduce"),
                xfail("native_batch_norm"),
                xfail("_native_batch_norm_legit"),
                # TODO: implement batching rule
                xfail("_batch_norm_with_update"),
                xfail("as_strided", "partial_views"),
            }
        ),
    )
    def test_vjpvmap(self, device, dtype, op):
        # NB: there is no vjpvmap_has_batch_rule test because that is almost
        # certainly redundant with the vmap_has_batch_rule test in test_vmap.py

        # one-off skip
        if op.name == "nn.functional.dropout":
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
        batch_norm_fns = (
            "nn.functional.batch_norm",
            "nn.functional.instance_norm",
        )  # instance norm calls batch norm
        is_batch_norm = op.name in batch_norm_fns

        for sample in samples:
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs

            is_batch_norm_and_training = is_batch_norm and is_batch_norm_training(
                op.name, kwargs
            )
            generator = generate_vmap_inputs(
                args, kwargs, is_batch_norm_and_training=is_batch_norm_and_training
            )

            for batched_args, in_dims, kwargs in generator:
                vmapped_op = vmap(op, in_dims)
                fn, primals = normalize_op_input_output2(
                    vmapped_op, batched_args, kwargs, sample.output_process_fn_grad
                )
                result = fn(*primals)
                cotangents = tree_map(lambda x: torch.randn_like(x), result)

                _, vjp_fn = vjp(fn, *primals)
                result_vjps = vjp_fn(cotangents)

                _, vjp_fn = ref_vjp(fn, *primals)
                expected_vjps = vjp_fn(cotangents)

                self.assertEqual(result_vjps, expected_vjps)

    def _compare_jacobians_of_vjp(
        self, fn, cotangents_and_primals, argnums=None, atol_rtol=None
    ):
        if argnums is None:
            argnums = tuple(range(len(cotangents_and_primals)))

        def get_vjp(cotangents, *primals):
            _, vjp_fn = vjp(fn, *primals)
            return vjp_fn(cotangents)

        jacobian_jvp = jacfwd(get_vjp, argnums)(*cotangents_and_primals)
        jacobian_vjp = jacrev(get_vjp, argnums)(*cotangents_and_primals)

        # For dtype changing operations, the jacobians have different dtype.
        jacobian_jvp = tree_map(lambda x: x.to(torch.float), jacobian_jvp)
        jacobian_vjp = tree_map(lambda x: x.to(torch.float), jacobian_vjp)

        if atol_rtol is not None:
            (atol, rtol) = atol_rtol
            self.assertEqual(jacobian_jvp, jacobian_vjp, atol=atol, rtol=rtol)
        else:
            self.assertEqual(jacobian_jvp, jacobian_vjp)

    @ops(op_db + additional_op_db + autograd_function_db, allowed_dtypes=(torch.float,))
    @skipOps(
        "TestOperators",
        "test_jvpvjp",
        vjp_fail.union(
            {
                xfail("to_sparse", ""),  # NYI
                # RuntimeError: Trying to set a forward gradient that has a different size than that of the original Tensor,
                # this is not supported. Tensor is of size [5, 2, 3] while the given forward gradient is of size [1, 2, 3].
                xfail("normal", ""),
                xfail("cdist", ""),  # NYI: forward-AD for _cdist_forward
                xfail("cholesky", ""),  # NYI: forward-AD for cholesky
                xfail(
                    "nn.functional.embedding_bag", ""
                ),  # NYI: forward-AD for _embedding_bag
                xfail(
                    "nn.functional.grid_sample", ""
                ),  # NYI: forward AD for grid_sampler_2d
                xfail("grid_sampler_2d", ""),  # NYI: forward AD for grid_sampler_2d
                xfail(
                    "nn.functional.hardsigmoid", ""
                ),  # NYI: forward AD for hardsigmoid_backward
                xfail(
                    "nn.functional.huber_loss", ""
                ),  # NYI: forward AD for huber_loss_backward
                xfail("NumpyCubeNotComposableAutogradFunction"),  # not composable
                xfail("ormqr", ""),  # NYI: forward AD for ormqr
                xfail(
                    "nn.functional.multilabel_margin_loss", ""
                ),  # NYI: multilabel_margin_loss_forward
                xfail(
                    "nn.functional.soft_margin_loss", ""
                ),  # NYI: forward-AD for soft_margin_loss_backward
                xfail("nn.functional.ctc_loss", ""),  # NYI: forward-AD for _ctc_loss
                xfail("nn.functional.pdist", ""),  # NYI: forward-AD with _pdist_forward
                skip("nn.functional.scaled_dot_product_attention"),
                xfail("torch.ops.aten._efficient_attention_forward"),  # outputs ints
                xfail(
                    "nn.functional.multi_margin_loss", ""
                ),  # NYI: forward AD with multi_margin_loss
                skip(
                    "linalg.householder_product", "", device_type="cuda"
                ),  # flaky, I'm not sure why
                xfail("sparse.sampled_addmm", ""),  # Sparse tensors have no strides
                xfail(
                    "_segment_reduce", "offsets"
                ),  # NYI: forward-AD for _segment_reduce
                xfail("sparse.mm", "reduce"),  # Sparse tensors have no strides
                xfail("index_reduce", "prod"),  # NYI: forward-AD for index_reduce
                xfail("index_reduce", "mean"),  # NYI: forward-AD for index_reduce
                xfail("index_reduce", "amax"),  # NYI: forward-AD for index_reduce
                xfail("index_reduce", "amin"),  # NYI: forward-AD for index_reduce
                xfail(
                    "_segment_reduce", "lengths"
                ),  # NYI: forward-AD for _segment_reduce
                xfail("native_dropout_backward"),  # NYI
            }
        ),
    )
    @opsToleranceOverride(
        "TestOperators",
        "test_jvpvjp",
        (
            tol1("masked.prod", {torch.float32: tol(atol=1e-04, rtol=1.3e-05)}),
            tol1("masked.cumprod", {torch.float32: tol(atol=1e-04, rtol=5e-04)}),
            tol1(
                "cumprod",
                {torch.float32: tol(atol=1e-04, rtol=1.3e-05)},
                device_type="cuda",
            ),
            tol1(
                "linalg.vander",
                {torch.float32: tol(atol=1e-04, rtol=1.3e-05)},
                device_type="cuda",
            ),
            tol1(
                "nn.functional.group_norm", {torch.float32: tol(atol=1e-03, rtol=1e-03)}
            ),
            tol2(
                "linalg.pinv", "hermitian", {torch.float32: tol(atol=5e-03, rtol=5e-03)}
            ),
        ),
    )
    def test_jvpvjp(self, device, dtype, op):
        if not op.supports_autograd:
            self.skipTest("Skipped! Autograd not supported.")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=True)

        # TODO: test in-place
        if is_inplace(op, op.get_op()):
            self.skipTest("Skipped! NYI: inplace-testing not supported.")
            return

        for sample in samples:
            fn, primals = normalize_op_input_output(op, sample)
            result = fn(*primals)
            cotangents = tree_map(lambda x: torch.randn_like(x), result)

            primals_tangents = tree_map(lambda x: torch.randn_like(x), primals)
            cotangents_tangents = tree_map(lambda x: torch.randn_like(x), cotangents)

            def push_vjp(primals, cotangents):
                _, vjp_fn = vjp(fn, *primals)
                return vjp_fn(cotangents)

            result = jvp(
                push_vjp, (primals, cotangents), (primals_tangents, cotangents_tangents)
            )
            self.assertEqual(len(result), 2)

            def tree_map2(fn, first, second):
                flat_first, spec_first = tree_flatten(first)
                flat_second, spec_second = tree_flatten(second)
                assert spec_first == spec_second
                flat_result = [fn(f, s) for f, s in zip(flat_first, flat_second)]
                return tree_unflatten(flat_result, spec_first)

            def reference(primals, cotangents, primals_tangents, cotangents_tangents):
                with fwAD.dual_level():
                    primal_duals = tree_map2(fwAD.make_dual, primals, primals_tangents)
                    _, vjp_fn = ref_vjp(fn, *primal_duals)

                    cotangent_duals = tree_map2(
                        fwAD.make_dual, cotangents, cotangents_tangents
                    )
                    result = vjp_fn(cotangent_duals)

                    flat_result, spec = tree_flatten(result)
                    primals_out, tangents_out = zip(
                        *[fwAD.unpack_dual(r) for r in flat_result]
                    )
                    tangents_out = [
                        t if t is not None else torch.zeros_like(p)
                        for p, t in zip(primals_out, tangents_out)
                    ]
                    expected = (
                        tree_unflatten(primals_out, spec),
                        tree_unflatten(tangents_out, spec),
                    )
                return expected

            expected = reference(
                primals, cotangents, primals_tangents, cotangents_tangents
            )
            self.assertEqual(result, expected)

    @with_tf32_off  # https://github.com/pytorch/pytorch/issues/86798
    @skipOps(
        "TestOperators",
        "test_vmapjvpvjp",
        vjp_fail.union(
            {
                # Following operators take too long, hence skipped
                skip("atleast_1d"),
                skip("atleast_2d"),
                skip("atleast_3d"),
                skip("meshgrid", "list_of_tensors"),
                skip("meshgrid", "variadic_tensors"),
                skip("broadcast_tensors"),
                skip("linalg.lstsq"),
                skip("nn.functional.bilinear"),
                skip("native_layer_norm"),
                skip("ormqr"),
                # Not actually a problem
                xfail("NumpyCubeNotComposableAutogradFunction"),  # not composable
                xfail(
                    "NumpyExpMarkDirtyAutogradFunction"
                ),  # vmap: inplace into a regular tensor
                # Potential bugs/errors
                xfail("as_strided"),  # AssertionError: Tensor-likes are not close!
                xfail(
                    "as_strided", "partial_views"
                ),  # AssertionError: Tensor-likes are not close!
                xfail(
                    "as_strided_scatter"
                ),  # AssertionError: Tensor-likes are not close!
                xfail("bernoulli"),  # calls random op
                xfail("bfloat16"),  # required rank 4 tensor to use channels_last format
                xfail("cdist"),  # Forward AD not implemented and no decomposition
                xfail("cdouble"),  # required rank 4 tensor to use channels_last format
                xfail("cfloat"),  # required rank 4 tensor to use channels_last format
                xfail("chalf"),  # required rank 4 tensor to use channels_last format
                xfail("cholesky"),  # Forward AD not implemented and no decomposition
                xfail("ormqr"),  # Forward AD not implemented and no decomposition
                xfail("double"),  # required rank 4 tensor to use channels_last format
                xfail("float"),  # required rank 4 tensor to use channels_last format
                xfail("half"),  # required rank 4 tensor to use channels_last format
                xfail("index_reduce", "prod"),  # NYI: forward AD for index_reduce
                xfail("index_reduce", "mean"),  # NYI: forward AD for index_reduce
                xfail("index_reduce", "amax"),  # NYI: forward AD for index_reduce
                xfail("index_reduce", "amin"),  # NYI: forward AD for index_reduce
                xfail(
                    "mvlgamma", "mvlgamma_p_1"
                ),  # vmap: inplace into a regular tensor
                xfail(
                    "mvlgamma", "mvlgamma_p_3"
                ),  # vmap: inplace into a regular tensor
                xfail(
                    "mvlgamma", "mvlgamma_p_5"
                ),  # vmap: inplace into a regular tensor
                xfail("nanquantile"),  # Batching rule not implemented for aten::equal
                # RuntimeError: Batch norm got a batched tensor as input while the
                # running_mean or running_var, which will be updated in place,
                # were not batched.
                xfail("nn.functional.batch_norm"),
                xfail("nn.functional.batch_norm", "without_cudnn"),
                xfail(
                    "nn.functional.ctc_loss"
                ),  # ForwardAD not implemented and no decomposition
                xfail("nn.functional.dropout2d"),  # calls random op
                xfail("nn.functional.dropout3d"),  # calls random op
                xfail("nn.functional.dropout"),  # calls random op
                xfail("nn.functional.scaled_dot_product_attention"),  # randomness
                xfail("torch.ops.aten._efficient_attention_forward"),  # outputs ints
                xfail("nn.functional.multi_head_attention_forward"),  # randomness
                xfail(
                    "nn.functional.embedding_bag"
                ),  # Forward AD not implemented and no decomposition
                xfail("nn.functional.alpha_dropout"),  # calls randomn op
                xfail(
                    "nn.functional.feature_alpha_dropout", "with_train"
                ),  # calls random op
                xfail("nn.functional.fractional_max_pool2d"),  # calls random op
                xfail("nn.functional.fractional_max_pool3d"),  # calls random op
                xfail("nn.functional.gaussian_nll_loss"),  # data depenedant flow
                xfail(
                    "nn.functional.grid_sample"
                ),  # Forward AD not implemented and no decomposition
                xfail(
                    "grid_sampler_2d"
                ),  # Forward AD not implemented and no decomposition
                xfail(
                    "nn.functional.hardsigmoid"
                ),  # Forward AD not implemented and no decomposition
                xfail(
                    "nn.functional.hinge_embedding_loss"
                ),  # vmap: inplace into a regular tensor
                xfail(
                    "nn.functional.huber_loss"
                ),  # Forward AD not implemented and no decomposition
                # RuntimeError: Batch norm got a batched tensor as input while the
                # running_mean or running_var, which will be updated in place,
                # were not batched.
                xfail("nn.functional.instance_norm"),
                # NYI: Tensor.clone(memory_format) inside vmap is only supported with
                # memory_format torch.preserve_format or torch.contiguous_format (got ChannelsLast)
                xfail("nn.functional.max_unpool2d"),
                xfail("nn.functional.max_unpool2d", "grad"),
                xfail(
                    "nn.functional.multi_margin_loss"
                ),  # Forward AD not implemented and no decomposition
                xfail(
                    "nn.functional.multilabel_margin_loss"
                ),  # Forward AD not implemented and no decomposition
                xfail(
                    "nn.functional.pdist"
                ),  # Forward AD not implemented and no decomposition
                xfail(
                    "nn.functional.rrelu"
                ),  # vmap: we do not yet support aten::rrelu_with_noise.
                xfail(
                    "nn.functional.soft_margin_loss"
                ),  # Forward AD not implemented and no decomposition
                xfail("normal"),  # calls random op
                xfail("normal", "number_mean"),  # calls random op
                xfail("pca_lowrank"),  # calls random op
                xfail("quantile"),  # Batching rule not implemented for aten::equal
                xfail(
                    "scatter_reduce", "prod"
                ),  # Forward AD not implemented and no decomposition
                xfail(
                    "_segment_reduce", "lengths"
                ),  # Forward AD not implemented and no decomposition
                xfail(
                    "_segment_reduce", "offsets"
                ),  # Forward AD not implemented and no decomposition
                xfail(
                    "sparse.sampled_addmm"
                ),  # RuntimeError: Sparse CSR tensors do not have strides
                xfail(
                    "sparse.mm", "reduce"
                ),  # RuntimeError: Sparse CSR tensors do not have strides
                xfail("svd_lowrank"),  # calls random op
                xfail(
                    "to"
                ),  # RuntimeError: required rank 4 tensor to use channels_last format
                xfail("to_sparse"),  # Forward AD not implemented and no decomposition
                xfail(
                    "view_as_complex"
                ),  # RuntimeError: Tensor must have a last dimension with stride 1
                # RuntimeError: Batch norm got a batched tensor as
                # input while the running_mean or running_var, which will be updated in
                # place, were not batched.
                xfail("native_batch_norm"),
                xfail("_native_batch_norm_legit"),
                # TODO: implement batching rule
                xfail("_batch_norm_with_update"),
                xfail("native_dropout_backward"),
            }
        ),
    )
    @ops(op_db + additional_op_db + autograd_function_db, allowed_dtypes=(torch.float,))
    @toleranceOverride({torch.float32: tol(atol=1e-04, rtol=1e-04)})
    @opsToleranceOverride(
        "TestOperators",
        "test_vmapjvpvjp",
        (
            tol1("linalg.svd", {torch.float32: tol(atol=5e-04, rtol=5e-04)}),
            tol1(
                "linalg.householder_product",
                {torch.float32: tol(atol=5e-03, rtol=5e-03)},
            ),
            tol1("linalg.multi_dot", {torch.float32: tol(atol=5e-04, rtol=5e-04)}),
            tol2(
                "linalg.pinv", "hermitian", {torch.float32: tol(atol=5e-04, rtol=5e-04)}
            ),
            tol1("svd", {torch.float32: tol(atol=5e-04, rtol=5e-04)}),
            tol1("matrix_exp", {torch.float32: tol(atol=5e-04, rtol=5e-04)}),
        ),
    )
    def test_vmapjvpvjp(self, device, dtype, op):
        # Since we test `jvpvjp` separately,
        # in this we just check that vmap of `jvpvjp`
        # is correct.
        if not op.supports_autograd:
            self.skipTest("Skipped! Autograd not supported.")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=True)

        # TODO: test in-place
        if is_inplace(op, op.get_op()):
            self.skipTest("Skipped! NYI: inplace-testing not supported.")
            return

        for sample in samples:
            fn, primals = normalize_op_input_output(op, sample)
            result = fn(*primals)
            cotangents = tree_map(lambda x: torch.randn_like(x), result)

            primals_tangents = tree_map(lambda x: torch.randn_like(x), primals)
            cotangents_tangents = tree_map(lambda x: torch.randn_like(x), cotangents)

            def push_vjp(primals, cotangents):
                _, vjp_fn = vjp(fn, *primals)
                return vjp_fn(cotangents)

            args, spec = tree_flatten(
                ((primals, cotangents), (primals_tangents, cotangents_tangents))
            )

            def jvp_of_vjp(*args):
                (primals, tangents) = tree_unflatten(args, spec)
                primals_out, tangents_out = jvp(push_vjp, primals, tangents)

                flat_primals_out = pytree.tree_leaves(primals_out)
                flat_tangents_out = pytree.tree_leaves(tangents_out)
                return tuple(flat_primals_out + flat_tangents_out)

            is_batch_norm_and_training = is_batch_norm_training(op, sample.kwargs)
            generator = get_fallback_and_vmap_exhaustive(
                jvp_of_vjp,
                args,
                {},
                is_batch_norm_and_training=is_batch_norm_and_training,
            )
            for loop_out, batched_out in generator:
                self.assertEqual(loop_out, batched_out)

    def _make_extremal_inputs(self, shape, device):
        if shape is None:
            return (None,)
        return (
            torch.full(shape, -1000.0, device=device),
            torch.zeros(shape, device=device),
            torch.full(shape, 1000.0, device=device),
        )

    def _arg_and_kwarg_options(self, args_options, kwargs_options):
        return itertools.product(*args_options, kwargs_options)

    def test_extremal_numerics_nll_loss(self, device):
        N, C = 3, 4
        d1, d2, d3 = 5, 6, 7
        shapes = (
            ((N, C), (N,), (C,)),
            ((N, C), (N,), None),
            ((N, C, d1, d2, d3), (N, d1, d2, d3), (C,)),
            ((N, C, d1, d2, d3), (N, d1, d2, d3), None),
        )
        kwargs_options = (
            {"ignore_index": 0, "reduction": "mean"},
            {"reduction": "sum"},
            {"reduction": "none"},
            {},
        )
        for input_shape, target_shape, weight_shape in shapes:
            input_options = self._make_extremal_inputs(input_shape, device)
            for input, kwargs in self._arg_and_kwarg_options(
                (input_options,), kwargs_options
            ):
                if weight_shape is None:
                    weight = None
                else:
                    weight = torch.randn(weight_shape, device=device)
                target = torch.randint(0, C, target_shape, device=device)
                target[
                    0
                ] = 1  # since we're ignoring index 0, at least one element must be non-zero

                fn = functools.partial(
                    torch.nn.functional.nll_loss, target=target, weight=weight, **kwargs
                )
                result = fn(input)
                cotangents = torch.randn_like(result, device=device)
                self._compare_jacobians_of_vjp(fn, (cotangents, input))

    def test_extremal_numerics_l1_loss(self, device):
        N, C, H, W = 3, 4, 5, 6
        shapes = ((N, C), (N, C, H), (N, C, H, W))
        kwargs_options = ({"reduction": "sum"}, {"reduction": "none"}, {})
        for shape in shapes:
            input_options = self._make_extremal_inputs(shape, device)
            target_options = self._make_extremal_inputs(shape, device)
            for input, target, kwargs in self._arg_and_kwarg_options(
                (input_options, target_options), kwargs_options
            ):
                result = torch.nn.functional.l1_loss(input, target)
                cotangents = torch.randn_like(result, device=device)
                self._compare_jacobians_of_vjp(
                    torch.nn.functional.l1_loss, (cotangents, input, target)
                )

    def test_extremal_numerics_mse_loss(self, device):
        N, C, H, W = 3, 4, 5, 6
        shapes = ((N, C), (N, C, H), (N, C, H, W))
        kwargs_options = ({"reduction": "sum"}, {"reduction": "none"}, {})
        for shape in shapes:
            input_options = self._make_extremal_inputs(shape, device)
            target_options = self._make_extremal_inputs(shape, device)
            for input, target, kwargs in self._arg_and_kwarg_options(
                (input_options, target_options), kwargs_options
            ):
                result = torch.nn.functional.mse_loss(input, target)
                cotangents = torch.randn_like(result, device=device)
                self._compare_jacobians_of_vjp(
                    torch.nn.functional.mse_loss, (cotangents, input, target)
                )

    def test_extremal_numerics_softmax(self, device):
        N, C, H, W = 3, 4, 5, 6
        shapes = ((N, C), (N, C, H), (N, C, H, W))
        kwargs_options = ({"dim": 1}, {})
        for shape in shapes:
            input_options = self._make_extremal_inputs(shape, device)
            for input, kwargs in self._arg_and_kwarg_options(
                (input_options,), kwargs_options
            ):
                result = torch.nn.functional.softmax(input)
                cotangents = torch.randn_like(result, device=device)
                self._compare_jacobians_of_vjp(
                    torch.nn.functional.softmax, (cotangents, input)
                )

    def test_extremal_numerics_log_softmax(self, device):
        N, C, H, W = 3, 4, 5, 6
        shapes = ((N, C), (N, C, H), (N, C, H, W))
        kwargs_options = ({"dim": 1}, {})
        for shape in shapes:
            input_options = self._make_extremal_inputs(shape, device)
            for input, kwargs in self._arg_and_kwarg_options(
                (input_options,), kwargs_options
            ):
                result = torch.nn.functional.log_softmax(input)
                cotangents = torch.randn_like(result, device=device)
                self._compare_jacobians_of_vjp(
                    torch.nn.functional.log_softmax, (cotangents, input)
                )

    def test_extremal_numerics_cross_entropy(self, device):
        N, C = 3, 4
        d1, d2, d3 = 5, 6, 7
        shapes = (
            ((N, C), (N,), (C,)),
            ((N, C), (N,), None),
            ((N, C), (N, C), (C,)),
            ((N, C), (N, C), None),
            ((C,), (), (C,)),
            ((C,), (), None),
            ((C,), (C,), (C,)),
            ((C,), (C,), None),
            ((N, C, d1, d2, d3), (N, d1, d2, d3), (C,)),
            ((N, C, d1, d2, d3), (N, d1, d2, d3), None),
            ((N, C, d1, d2, d3), (N, C, d1, d2, d3), (C,)),
            ((N, C, d1, d2, d3), (N, C, d1, d2, d3), None),
        )
        for input_shape, target_shape, weight_shape in shapes:
            input_options = self._make_extremal_inputs(input_shape, device)
            kwargs_options = [{"reduction": "sum"}, {"reduction": "none"}, {}]
            if input_shape != target_shape:
                kwargs_options.append({"ignore_index": 0, "reduction": "mean"})

            for input, kwargs in self._arg_and_kwarg_options(
                (input_options,), kwargs_options
            ):
                if weight_shape is None:
                    weight = None
                else:
                    weight = torch.randn(weight_shape, device=device)

                if input_shape == target_shape:
                    target = torch.rand(target_shape, device=device)
                elif len(target_shape) == 0:
                    target = torch.tensor(
                        1, device=device
                    )  # must be non-zero since ignore_index may be 0
                else:
                    target = torch.randint(0, C, target_shape, device=device)

                fn = functools.partial(
                    torch.nn.functional.cross_entropy,
                    target=target,
                    weight=weight,
                    **kwargs,
                )
                result = fn(input)
                cotangents = torch.randn_like(result, device=device)
                self._compare_jacobians_of_vjp(
                    fn, (cotangents, input), atol_rtol=(1e-4, 1e-5)
                )

    def test_extremal_numerics_binary_cross_entropy(self, device):
        N, C, H, W = 3, 4, 5, 6
        shapes = ((N, C), (N, C, H), (N, C, H, W))
        for shape in shapes:
            weight_options = self._make_extremal_inputs(shape, device)
            kwargs_options = [{"reduction": "sum"}, {"reduction": "none"}, {}]

            for weight, kwargs in self._arg_and_kwarg_options(
                (weight_options,), kwargs_options
            ):
                input = torch.rand(shape, device=device)
                target = torch.rand(shape, device=device)
                fn = functools.partial(
                    torch.nn.functional.binary_cross_entropy,
                    target=target,
                    weight=weight,
                    **kwargs,
                )
                result = fn(input)
                cotangents = torch.randn_like(result, device=device)
                self._compare_jacobians_of_vjp(
                    fn, (cotangents, input), atol_rtol=(1e-4, 2e-5)
                )

    def test_extremal_numerics_layer_norm(self, device):
        N, C, H, W = 3, 4, 5, 6
        shapes = ((N, C), (N, C, H), (N, C, H, W))
        for shape in shapes:
            input_options = self._make_extremal_inputs(shape, device)
            normalized_shape = shape[1:]
            weight_options = self._make_extremal_inputs(normalized_shape, device)
            bias_options = self._make_extremal_inputs(normalized_shape, device)

            for input, bias, weight in self._arg_and_kwarg_options(
                (input_options, bias_options, weight_options), ()
            ):

                def fn(input, weight, bias):
                    return torch.nn.functional.layer_norm(
                        input, normalized_shape, weight=weight, bias=bias
                    )

                result = fn(input, weight, bias)
                cotangents = torch.randn_like(result, device=device)
                self._compare_jacobians_of_vjp(fn, (cotangents, input, weight, bias))

    @with_tf32_off  # https://github.com/pytorch/pytorch/issues/86798
    @ops(
        op_db + additional_op_db + autograd_function_db,
        allowed_dtypes=(torch.float32, torch.double),
    )
    @skipOps(
        "TestOperators",
        "test_vmap_autograd_grad",
        {
            # The size of tensor a (4) must match the size of tensor b (10) at non-singleton dimension 0
            xfail("masked_select"),
            xfail("nn.functional.max_unpool2d", "grad"),  # contiguous call
            xfail("nn.functional.max_unpool2d"),  # contiguous call
            xfail("to_sparse"),  # dispatch key issue
            xfail("torch.ops.aten._efficient_attention_forward"),  # outputs ints
            # https://github.com/pytorch/pytorch/issues/96560
            decorate("xlogy", decorator=skipIfRocm),
            # numerical inconsistencies, look like bugs
            skip(
                "matrix_exp", dtypes=(torch.float32,), device_type="cuda"
            ),  # fails on linux, passes on windows
            skip(
                "ldexp", dtypes=(torch.float32,), device_type="cpu"
            ),  # fails on all but mac
            skip("__rmatmul__"),  # flaky needs investigation
            skip("matmul"),  # flaky needs investigation
            skip("nn.functional.conv_transpose3d"),  # flaky needs investigation
            skip("nn.functional.conv_transpose2d"),  # flaky needs investigation
            skip("nn.functional.conv_transpose1d"),  # flaky needs investigation
            skip(
                "nn.functional.layer_norm", dtypes=(torch.float32,), device_type="cpu"
            ),  # fails on windows
            skip(
                "linalg.lu_factor", dtypes=(torch.float32,), device_type="cuda"
            ),  # fails on all but windows
            skip(
                "linalg.lu_factor_ex", dtypes=(torch.float32,), device_type="cuda"
            ),  # fails on all but windows
            skip("linalg.multi_dot", "", device_type="cpu"),
            skip("sparse.sampled_addmm", ""),
            skip("sparse.mm", "reduce"),
            skip("native_layer_norm", "", device_type="cpu"),
            # RuntimeError: Expected contiguous tensor, but got
            # non-contiguous tensor for argument #2 'grad_output'
            decorate(
                "_batch_norm_with_update",
                decorator=expectedFailureIf(TEST_WITH_ROCM),
                device_type="cuda",
            ),
        },
    )
    @opsToleranceOverride(
        "TestOperators",
        "test_vmap_autograd_grad",
        (
            tol1(
                "linalg.householder_product",
                {torch.float32: tol(atol=5e-04, rtol=9e-03)},
                device_type="cuda",
            ),
            tol1(
                "linalg.householder_product",
                {torch.float32: tol(atol=1e-04, rtol=1e-04)},
                device_type="cpu",
            ),
            tol1(
                "linalg.multi_dot",
                {torch.float32: tol(atol=2e-04, rtol=1e-04)},
                device_type="cuda",
            ),
            tol2(
                "linalg.pinv", "hermitian", {torch.float32: tol(atol=5e-06, rtol=5e-06)}
            ),
            tol1("nn.functional.conv3d", {torch.float32: tol(atol=5e-04, rtol=9e-03)}),
        ),
    )
    def test_vmap_autograd_grad(self, device, dtype, op):
        def is_differentiable(inp):
            return isinstance(inp, Tensor) and (
                inp.grad_fn is not None or inp.requires_grad
            )

        def get_flat_differentiable(tree):
            flattened = pytree.tree_leaves(tree)
            return tuple(i for i in flattened if is_differentiable(i))

        def get_differentiable_linked(list1, list2):
            paired_list = zip(list1, list2)
            paired_list = tuple(
                (first, second)
                for (first, second) in paired_list
                if is_differentiable(first)
            )
            return zip(*paired_list)

        def filter_none(out):
            flattened = pytree.tree_leaves(out)
            return tuple(o for o in flattened if o is not None)

        if not op.supports_autograd:
            self.skipTest("Skipped! Autograd not supported.")
            return

        sample_inputs = op.sample_inputs(device, dtype, requires_grad=True)

        for sample_input in sample_inputs:
            fn, primals = normalize_op_input_output(op, sample_input)
            out = fn(*primals)
            cotangents = tree_map(torch.randn_like, out)

            def compute_grad(cotangents):
                out_flattened = out
                cotangents_flattened = cotangents
                if not isinstance(out_flattened, torch.Tensor):
                    out_flattened = pytree.tree_leaves(out)
                    cotangents_flattened = pytree.tree_leaves(cotangents)
                    out_flattened, cotangents_flattened = get_differentiable_linked(
                        out_flattened, cotangents_flattened
                    )

                return filter_none(
                    torch.autograd.grad(
                        out_flattened,
                        get_flat_differentiable(primals),
                        cotangents_flattened,
                        retain_graph=True,
                        allow_unused=True,
                    )
                )

            is_batch_norm_and_training = is_batch_norm_training(op, sample_input.kwargs)
            generator = get_fallback_and_vmap_exhaustive(
                compute_grad,
                (cotangents,),
                {},
                is_batch_norm_and_training=is_batch_norm_and_training,
            )
            for loop_out, batched_out in generator:
                self.assertEqual(loop_out, batched_out)

    def test_vmapvmapjvp_linalg_solve(self):
        ops = [op for op in op_db if op.name == "linalg.solve"]
        assert len(ops) > 0

        # this specializes a lot of code from the get_fallback_and_vmap_exhaustive test. If we need this more
        # generally, this could go for a refactor

        B0 = 2
        B1 = 3

        # we want to check the case where A will be seen as contiguous by jvp but during the vmap calls will become
        # non-contiguous because vmap will expand. This will happen during both levels of vmap
        A = torch.randn(4, 4)
        k = torch.randn(4, 5, B1, B0)
        fn, args = get_jvp_variant_primals_tangents(
            torch.linalg.solve, SampleInput(A, args=(k,))
        )

        in_dims_all = (None, -1, None, -1)
        batched_out = vmap(vmap(fn, in_dims=in_dims_all), in_dims=in_dims_all)(*args)
        loop_out = loop2(fn, in_dims_all, in_dims_all, 0, 0, B0, B1, *args)
        self.assertEqual(loop_out, batched_out)

    @ops(
        filter(lambda op: op.name in aliasing_ops, op_db + additional_op_db),
        allowed_dtypes=(torch.float,),
    )
    @parametrize("grad_op", ["jvp", "vjp"])
    def test_view_then_inplace(self, device, dtype, op, grad_op):
        for sample_input in op.sample_inputs(device, dtype):

            def f(x):
                op(sample_input.input, *sample_input.args, **sample_input.kwargs).copy_(
                    x
                )
                return x

            without_grad = op(
                sample_input.input, *sample_input.args, **sample_input.kwargs
            )
            if grad_op == "jvp":
                with self.assertRaisesRegex(
                    RuntimeError,
                    "During a grad .* attempted to call in-place operation",
                ):
                    jvp(
                        f,
                        (torch.randn_like(without_grad),),
                        (torch.randn_like(without_grad),),
                    )
            else:
                assert grad_op == "vjp"
                with self.assertRaisesRegex(
                    RuntimeError,
                    "During a grad .* attempted to call in-place operation",
                ):
                    vjp(f, torch.randn_like(without_grad))

    @ops(
        filter(
            lambda op: op.name in aliasing_ops_list_return, op_db + additional_op_db
        ),
        allowed_dtypes=(torch.float,),
    )
    @parametrize("grad_op", ["jvp", "vjp"])
    def test_view_then_inplace_list_return(self, device, dtype, op, grad_op):
        for sample_input in op.sample_inputs(device, dtype):

            def f(x):
                op(sample_input.input, *sample_input.args, **sample_input.kwargs)[
                    0
                ].copy_(x)
                return x

            without_grad = op(
                sample_input.input, *sample_input.args, **sample_input.kwargs
            )[0]
            with self.assertRaisesRegex(
                RuntimeError, "During a grad .* attempted to call in-place operation"
            ):
                if grad_op == "jvp":
                    jvp(
                        f,
                        (torch.randn_like(without_grad),),
                        (torch.randn_like(without_grad),),
                    )
                else:
                    assert grad_op == "vjp"
                    vjp(f, torch.randn_like(without_grad))

    @parametrize("grad_op", ["jvp", "vjp"])
    def test_view_then_inplace_special(self, grad_op):
        # some things in __getitem__ use at::index, which doesn't alias, so this tests a subset of them that do alias
        ops = [
            lambda x: x[0],
            lambda x: x[0, 0, 0],
            lambda x: x[:1],
            lambda x: x[:, :1],
            lambda x: x[:, :1, :],
        ]

        for op in ops:

            def f(x):
                op(captured).copy_(x)
                return x

            captured = torch.randn(4, 3, 3)
            without_grad = op(captured)
            if grad_op == "jvp":
                with self.assertRaisesRegex(
                    RuntimeError,
                    "During a grad .* attempted to call in-place operation",
                ):
                    jvp(
                        f,
                        (torch.randn_like(without_grad),),
                        (torch.randn_like(without_grad),),
                    )
            else:
                assert grad_op == "vjp"
                with self.assertRaisesRegex(
                    RuntimeError,
                    "During a grad .* attempted to call in-place operation",
                ):
                    vjp(f, torch.randn_like(without_grad))

    @with_tf32_off  # https://github.com/pytorch/pytorch/issues/86798
    # NOTE: [three-transform testing]
    # We only test the autograd_function_db tests here.
    #
    # Usually testing the composition of two transforms is sufficient to convince
    # ourselves that an operator is correctly implemented. For the following cases,
    # we want to be extra sure, so we send those through some three-transform tests:
    # - autograd.Function. The mechanism is via PyDispatcher/HigherOrderOperator, not the
    #   regular PyTorch dispatcher, so it's good to exercise more caution.
    @ops(autograd_function_db, allowed_dtypes=(torch.float32,))
    @skipOps(
        "TestOperators",
        "test_vmapvjpvmap",
        {
            xfail("NumpyCubeNotComposableAutogradFunction"),  # Not composable
        },
    )
    def test_vmapvjpvmap(self, device, dtype, op):
        samples = op.sample_inputs(device, dtype, requires_grad=True)
        B = 2
        for sample in samples:
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs
            generator = generate_vmap_inputs(args, kwargs, batch_size=B)
            for batched_args, in_dims, kwargs in generator:
                inner_vmapped_op = vmap(op, in_dims)
                inner_mapped_op = functools.partial(loop, op, in_dims, 0, B)

                inner_vmapped_fn, primals = normalize_op_input_output2(
                    inner_vmapped_op,
                    batched_args,
                    kwargs,
                    sample.output_process_fn_grad,
                )
                inner_mapped_fn, _ = normalize_op_input_output2(
                    inner_mapped_op, batched_args, kwargs, sample.output_process_fn_grad
                )
                result = inner_mapped_fn(*primals)
                cotangents = tree_map(lambda x: torch.rand_like(x), result)

                def apply_vjp(fn):
                    def inner(primals, cotangents):
                        _, vjp_fn = vjp(fn, *primals)
                        return vjp_fn(cotangents)

                    return inner

                vjpvmap_fn = apply_vjp(inner_vmapped_fn)
                vjpmap_fn = apply_vjp(inner_mapped_fn)
                batched_args = (primals, cotangents)
                generator = generate_vmap_inputs(batched_args, {})

                for batched_args, in_dims, _ in generator:
                    # strategy: compare vmap(vjp(vmap(op)) vs map(vjp(map(op))
                    vmapvjpvmap_fn = vmap(vjpvmap_fn, in_dims)
                    mapvjpmap_fn = functools.partial(loop, vjpmap_fn, in_dims, 0, B)

                    result = vmapvjpvmap_fn(*batched_args)
                    expected = mapvjpmap_fn(*batched_args)
                    self.assertEqual(result, expected)

    # See NOTE: [three-transform testing]
    @ops(autograd_function_db, allowed_dtypes=(torch.float32,))
    @skipOps(
        "TestOperators",
        "test_vjpvmapvmap",
        {
            xfail("NumpyCubeNotComposableAutogradFunction"),  # Not composable
        },
    )
    def test_vjpvmapvmap(self, device, dtype, op):
        samples = op.sample_inputs(device, dtype, requires_grad=True)
        B = 2
        for sample in samples:
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs
            generator = generate_vmap_inputs(args, kwargs, batch_size=B)
            for batched_args, inner_in_dims, kwargs in generator:
                inner_vmapped_op = vmap(op, inner_in_dims)
                inner_mapped_op = functools.partial(loop, op, inner_in_dims, 0, B)
                generator = generate_vmap_inputs(batched_args, kwargs)
                for batched_args, in_dims, kwargs in generator:
                    # strategy: compare vjp(vmap(vmap(op)) vs vjp(map(map(op))
                    vmapped_op = vmap(inner_vmapped_op, in_dims)
                    mapped_op = functools.partial(loop, inner_mapped_op, in_dims, 0, B)

                    vmapped_fn, primals = normalize_op_input_output2(
                        vmapped_op, batched_args, kwargs, sample.output_process_fn_grad
                    )
                    mapped_fn, _ = normalize_op_input_output2(
                        mapped_op, batched_args, kwargs, sample.output_process_fn_grad
                    )

                    result = mapped_fn(*primals)
                    cotangents = tree_map(lambda x: torch.rand_like(x), result)

                    _, vjp_fn = vjp(mapped_fn, *primals)
                    expected_vjps = vjp_fn(cotangents)

                    _, vjp_fn = vjp(vmapped_fn, *primals)
                    result_vjps = vjp_fn(cotangents)

                    self.assertEqual(result_vjps, expected_vjps)

    # See NOTE: [three-transform testing]
    @ops(autograd_function_db, allowed_dtypes=(torch.float32,))
    @skipOps(
        "TestOperators",
        "test_vjpvjpvmap",
        {
            xfail("NumpyCubeNotComposableAutogradFunction"),  # Not composable
        },
    )
    def test_vjpvjpvmap(self, device, dtype, op):
        samples = op.sample_inputs(device, dtype, requires_grad=True)
        B = 2
        for sample in samples:
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs
            generator = generate_vmap_inputs(args, kwargs, batch_size=B)
            for batched_args, in_dims, kwargs in generator:
                inner_vmapped_op = vmap(op, in_dims)
                inner_mapped_op = functools.partial(loop, op, in_dims, 0, B)

                vjpmap_fn, args = get_vjpfull_variant2(
                    inner_mapped_op, batched_args, kwargs
                )
                vjpvmap_fn, _ = get_vjpfull_variant2(
                    inner_vmapped_op, batched_args, kwargs
                )

                vjpvjpvmap_fn, new_args = get_vjpfull_variant2(vjpvmap_fn, args, {})
                vjpvjpmap_fn, _ = get_vjpfull_variant2(vjpmap_fn, args, {})

                expected = vjpvjpmap_fn(*new_args)
                result = vjpvjpvmap_fn(*new_args)
                self.assertEqual(result, expected)

    # We're generally convinced that jvp x vmap works (vmap turns an operator
    # into another operator and we test jvp support for operators). So
    # we only test it on the things we're not sure about:
    # - the autograd.Function <> functorch interaction
    @ops(autograd_function_db, allowed_dtypes=(torch.float32,))
    @skipOps(
        "TestOperators",
        "test_jvpvmap",
        {
            xfail("NumpyCubeNotComposableAutogradFunction"),  # Not composable
        },
    )
    def test_jvpvmap(self, device, dtype, op):
        samples = op.sample_inputs(device, dtype, requires_grad=True)
        B = 2
        for sample in samples:
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs
            generator = generate_vmap_inputs(args, kwargs, batch_size=B)
            for batched_args, in_dims, kwargs in generator:
                inner_vmapped_op = vmap(op, in_dims)
                inner_mapped_op = functools.partial(loop, op, in_dims, 0, B)

                jvpvmap_op, primals = get_jvp_variant_primals_tangents2(
                    inner_vmapped_op,
                    batched_args,
                    kwargs,
                    sample.output_process_fn_grad,
                )
                jvpmap_op, _ = get_jvp_variant_primals_tangents2(
                    inner_mapped_op, batched_args, kwargs, sample.output_process_fn_grad
                )

                expected = jvpmap_op(*primals)
                result = jvpvmap_op(*primals)
                self.assertEqual(result, expected)

    # See NOTE: [three-transform testing]
    @ops(autograd_function_db, allowed_dtypes=(torch.float32,))
    @skipOps(
        "TestOperators",
        "test_jvpvmapvmap",
        {
            xfail("NumpyCubeNotComposableAutogradFunction"),  # Not composable
        },
    )
    def test_jvpvmapvmap(self, device, dtype, op):
        samples = op.sample_inputs(device, dtype, requires_grad=True)
        B = 2
        for sample in samples:
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs
            generator = generate_vmap_inputs(args, kwargs, batch_size=B)
            for batched_args, inner_in_dims, kwargs in generator:
                inner_vmapped_op = vmap(op, inner_in_dims)
                inner_mapped_op = functools.partial(loop, op, inner_in_dims, 0, B)
                generator = generate_vmap_inputs(batched_args, kwargs)
                for batched_args, in_dims, kwargs in generator:
                    # strategy: compare jvp(vmap(vmap(op)) vs jvp(map(map(op))
                    vmapped_op = vmap(inner_vmapped_op, in_dims)
                    mapped_op = functools.partial(loop, inner_mapped_op, in_dims, 0, B)

                    jvpvmapvmap_fn, primals = get_jvp_variant_primals_tangents2(
                        vmapped_op, batched_args, kwargs, sample.output_process_fn_grad
                    )
                    jvpmapmap_fn, _ = get_jvp_variant_primals_tangents2(
                        mapped_op, batched_args, kwargs, sample.output_process_fn_grad
                    )

                    expected = jvpmapmap_fn(*primals)
                    result = jvpvmapvmap_fn(*primals)
                    self.assertEqual(result, expected)

    # See NOTE: [three-transform testing]
    @with_tf32_off  # https://github.com/pytorch/pytorch/issues/86798
    @ops(autograd_function_db, allowed_dtypes=(torch.float32,))
    @skipOps(
        "TestOperators",
        "test_vmapjvpvmap",
        {
            xfail("NumpyCubeNotComposableAutogradFunction"),  # Not composable
        },
    )
    def test_vmapjvpvmap(self, device, dtype, op):
        samples = op.sample_inputs(device, dtype, requires_grad=True)
        B = 2
        for sample in samples:
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs
            generator = generate_vmap_inputs(args, kwargs, batch_size=B)
            for batched_args, in_dims, kwargs in generator:
                inner_vmapped_op = vmap(op, in_dims)
                inner_mapped_op = functools.partial(loop, op, in_dims, 0, B)

                jvpvmap_fn, primals = get_jvp_variant_primals_tangents2(
                    inner_vmapped_op,
                    batched_args,
                    kwargs,
                    sample.output_process_fn_grad,
                )
                jvpmap_fn, _ = get_jvp_variant_primals_tangents2(
                    inner_mapped_op, batched_args, kwargs, sample.output_process_fn_grad
                )

                generator = generate_vmap_inputs(primals, {})

                for batched_args, in_dims, _ in generator:
                    # strategy: compare vmap(jvp(vmap(op)) vs map(jvp(map(op))
                    vmapjvpvmap_fn = vmap(jvpvmap_fn, in_dims)
                    mapjvpmap_fn = functools.partial(loop, jvpmap_fn, in_dims, 0, B)

                    result = vmapjvpvmap_fn(*batched_args)
                    expected = mapjvpmap_fn(*batched_args)
                    self.assertEqual(result, expected)

    # See NOTE: [three-transform testing]
    @ops(autograd_function_db, allowed_dtypes=(torch.float32,))
    @skipOps(
        "TestOperators",
        "test_jvpjvpvmap",
        {
            xfail("NumpyCubeNotComposableAutogradFunction"),  # Not composable
        },
    )
    def test_jvpjvpvmap(self, device, dtype, op):
        samples = op.sample_inputs(device, dtype, requires_grad=True)
        B = 2
        for sample in samples:
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs
            generator = generate_vmap_inputs(args, kwargs, batch_size=B)
            for batched_args, in_dims, kwargs in generator:
                inner_vmapped_op = vmap(op, in_dims)
                inner_mapped_op = functools.partial(loop, op, in_dims, 0, B)

                jvpmap_fn, args = get_jvp_variant_primals_tangents2(
                    inner_mapped_op, batched_args, kwargs, sample.output_process_fn_grad
                )
                jvpvmap_fn, _ = get_jvp_variant_primals_tangents2(
                    inner_vmapped_op,
                    batched_args,
                    kwargs,
                    sample.output_process_fn_grad,
                )

                jvpjvpvmap_fn, new_args = get_jvp_variant_primals_tangents2(
                    jvpvmap_fn, args, {}
                )
                jvpjvpmap_fn, _ = get_jvp_variant_primals_tangents2(jvpmap_fn, args, {})

                expected = jvpjvpmap_fn(*new_args)
                result = jvpjvpvmap_fn(*new_args)
                self.assertEqual(result, expected)

    # See NOTE: [three-transform testing]
    @ops(autograd_function_db, allowed_dtypes=(torch.float32,))
    @skipOps(
        "TestOperators",
        "test_jvpvjpvmap",
        {
            xfail("NumpyCubeNotComposableAutogradFunction"),  # Not composable
        },
    )
    def test_jvpvjpvmap(self, device, dtype, op):
        samples = op.sample_inputs(device, dtype, requires_grad=True)
        B = 2
        for sample in samples:
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs
            generator = generate_vmap_inputs(args, kwargs, batch_size=B)
            for batched_args, in_dims, kwargs in generator:
                inner_vmapped_op = vmap(op, in_dims)
                inner_mapped_op = functools.partial(loop, op, in_dims, 0, B)

                vjpmap_fn, args = get_vjpfull_variant2(
                    inner_mapped_op, batched_args, kwargs
                )
                vjpvmap_fn, _ = get_vjpfull_variant2(
                    inner_vmapped_op, batched_args, kwargs
                )

                jvpvjpvmap_fn, new_args = get_jvp_variant_primals_tangents2(
                    vjpvmap_fn, args, {}
                )
                jvpvjpmap_fn, _ = get_jvp_variant_primals_tangents2(vjpmap_fn, args, {})

                expected = jvpvjpmap_fn(*new_args)
                result = jvpvjpvmap_fn(*new_args)
                self.assertEqual(result, expected)

    def test_data_write_errors_under_transform(self, device):
        t = torch.randn(3, 3, device=device)

        def fn(t):
            t.data = torch.randn(3, 3)
            return t.sum()

        msg = "mutating directly with `.data` inside functorch transform"
        with self.assertRaisesRegex(RuntimeError, msg):
            grad(fn)(t)

        with self.assertRaisesRegex(RuntimeError, msg):
            vjp(fn, t)

        with self.assertRaisesRegex(RuntimeError, msg):
            jvp(fn, (t,), (torch.randn_like(t),))

    def test_tensor_with_scalar_list(self, device):
        x = torch.randn((), device=device)

        def func_list_of_scalar(x):
            return torch.tensor([x], device=device)

        def func(x):
            return torch.tensor(x, device=device).view(1)

        actual_o, actual_fn = vjp(func_list_of_scalar, x)
        expected_o, expected_fn = vjp(func, x)

        self.assertEqual(actual_o, expected_o)
        self.assertEqual(
            expected_fn(torch.ones_like(expected_o)),
            actual_fn(torch.ones_like(actual_o)),
        )


only_for = ("cpu", "cuda")
instantiate_device_type_tests(TestOperators, globals(), only_for=only_for)

if __name__ == "__main__":
    run_tests()
