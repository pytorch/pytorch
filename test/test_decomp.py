# Owner(s): ["module: decompositions"]

import functools
import itertools
import re
import unittest
from collections import defaultdict
from functools import partial

import torch._inductor.decomposition
import torch.autograd
from torch import Tensor
from torch._decomp import core_aten_decompositions, decomposition_table
from torch._dispatch.python import enable_python_dispatcher
from torch._ops import DispatchKey
from torch.testing import make_tensor
from torch.testing._internal.common_cuda import tf32_off
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCPU,
    onlyCUDA,
    onlyNativeDeviceTypes,
    ops,
)
from torch.testing._internal.common_methods_invocations import (
    op_db,
    skip,
    skipOps,
    xfail,
)
from torch.testing._internal.common_modules import module_db, modules
from torch.testing._internal.common_utils import (
    is_iterable_of_tensors,
    run_tests,
    skipIfCrossRef,
    skipIfTorchDynamo,
    suppress_warnings,
    TEST_WITH_ASAN,
    TEST_WITH_SLOW,
    TestCase,
    unMarkDynamoStrictTest,
)
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten


aten = torch.ops.aten


# TODO: this isn't going to work with non-aten namespaces
def overload_to_aten_name(op):
    return op._schema.name.split("::")[1]


# All operators that can have decomp tests
decomposition_names = {
    overload_to_aten_name(k)
    for k in decomposition_table
    if isinstance(k, torch._ops.OpOverload)
}
core_decomposition_names = {
    overload_to_aten_name(k)
    for k in core_aten_decompositions()
    if isinstance(k, torch._ops.OpOverload)
}
_decomp_test_ops = [
    op
    for op in op_db
    if op.aten_name in decomposition_names
    or op.aten_backward_name in decomposition_names
]
_decomp_test_ops_core_autograd = [
    op
    for op in op_db
    if op.aten_name in core_decomposition_names and op.supports_autograd
]
_sdpa_op_info = [op for op in op_db if "scaled_dot_product_attention" in op.aten_name]


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


def _as_tuple(val):
    if isinstance(val, tuple):
        return val
    return (val,)


def ref_vjp_no_create(f, *primals):
    result = f(*primals)

    def wrapped(cotangents):
        return _autograd_grad(
            _as_tuple(result),
            primals,
            _as_tuple(cotangents),
            create_graph=False,
            retain_graph=True,
        )

    return result, wrapped


dtype_precisions = {
    torch.float16: (0.001, 1e-5),
    torch.bfloat16: (0.016, 1e-4),
    torch.float32: (1.3e-6, 1e-5),
    torch.float64: (1e-7, 1e-7),
    torch.complex32: (0.001, 1e-5),
    torch.complex64: (1.3e-6, 1e-5),
    torch.complex128: (1e-7, 1e-7),
}
# Returns the "default" rtol and atol for comparing scalars or
# tensors of the given dtypes.


def _getDefaultRtolAndAtol(dtype0, dtype1):
    rtol = max(
        dtype_precisions.get(dtype0, (0, 0))[0], dtype_precisions.get(dtype1, (0, 0))[0]
    )
    atol = max(
        dtype_precisions.get(dtype0, (0, 0))[1], dtype_precisions.get(dtype1, (0, 0))[1]
    )
    return rtol, atol


def op_assert_ref(test_case, op, test_dtype, i, orig, decomp, ref, args, kwargs):
    assert orig.dtype == decomp.dtype, f"{i} Operation:  {op}"
    if orig.numel() == 0 or decomp.numel() == 0:
        assert orig.numel() == decomp.numel()
        return
    assert orig.shape == decomp.shape, f"{i} Operation:  {op}"
    tol_table = {
        (torch.bfloat16, torch.ops.aten.native_layer_norm.default): 1e-5,
        (torch.float16, torch.ops.aten.native_layer_norm.default): 1e-5,
        (torch.float16, torch.ops.aten.native_layer_norm_backward.default): 1e-3,
        (torch.bfloat16, torch.ops.aten.native_layer_norm_backward.default): 2e-2,
        (torch.bfloat16, torch.ops.aten.native_batch_norm.default): 1e-5,
        (torch.float16, torch.ops.aten.native_batch_norm.default): 1e-5,
        (torch.bfloat16, torch.ops.aten._native_batch_norm_legit.default): 1e-5,
        (torch.bfloat16, torch.ops.aten._native_batch_norm_legit.no_stats): 1e-5,
        (torch.float16, torch.ops.aten._native_batch_norm_legit.default): 1e-5,
        (torch.float16, torch.ops.aten._native_batch_norm_legit.no_stats): 1e-5,
        (torch.bfloat16, torch.ops.aten.linalg_vector_norm.default): 1e-4,
        (torch.float16, torch.ops.aten.linalg_vector_norm.default): 1e-4,
        (torch.bfloat16, torch.ops.aten.var_mean.correction): 5e-7,
        (torch.float16, torch.ops.aten.var_mean.correction): 5e-7,
        (torch.bfloat16, torch.ops.aten.var_mean.dim): 5e-7,
        (torch.float16, torch.ops.aten.var_mean.dim): 5e-7,
        (torch.float16, torch.ops.aten.nll_loss_forward.default): 1e-2,
        (torch.bfloat16, torch.ops.aten.nll_loss_forward.default): 1e-1,
        (torch.float16, torch.ops.aten.nll_loss2d_forward.default): 1e-2,
        (torch.bfloat16, torch.ops.aten.nll_loss2d_forward.default): 2e-1,
        (torch.float16, torch.ops.aten.hardswish.default): 2e-7,
        (torch.bfloat16, torch.ops.aten.hardswish.default): 2e-7,
        (torch.float16, torch.ops.aten.multi_margin_loss.default): 3e-2,
        (torch.bfloat16, torch.ops.aten.multi_margin_loss.default): 5e-2,
        (torch.float16, torch.ops.aten.multilabel_margin_loss_forward.default): 3e-2,
        (torch.bfloat16, torch.ops.aten.multilabel_margin_loss_forward.default): 3e-2,
        (torch.float16, torch.ops.aten.reflection_pad1d_backward.default): 5e-3,
        (torch.bfloat16, torch.ops.aten.reflection_pad1d_backward.default): 5e-3,
        (torch.float16, torch.ops.aten.reflection_pad2d_backward.default): 5e-3,
        (torch.bfloat16, torch.ops.aten.reflection_pad2d_backward.default): 5e-3,
        (torch.float16, torch.ops.aten.reflection_pad3d_backward.default): 5e-3,
        (torch.bfloat16, torch.ops.aten.reflection_pad3d_backward.default): 5e-2,
        # see https://github.com/pytorch/pytorch/pull/96264
        (torch.float16, torch.ops.aten.mv.default): 1e-5,
        (torch.bfloat16, torch.ops.aten.mv.default): 1e-5,
        (torch.float16, torch.ops.aten.log_sigmoid_backward.default): 2e-5,
        (torch.float16, torch.ops.aten._softmax_backward_data.default): 3e-7,
    }
    if ref.is_floating_point():
        orig_diff = (orig - ref).abs().max()
        decomp_diff = (decomp - ref).abs().max()
        atol = tol_table.get((test_dtype, op), 1e-7)
        if decomp_diff > orig_diff + atol:
            raise RuntimeError(
                f"Difference from float64 is larger with decomposition {op.__name__}"
                f" than original on output {i}. Original max diff: {orig_diff}, Decomp max diff: {decomp_diff}\n"
                f"atol = {atol}\n"
                f"args = {args}\n"
                f"kwargs = {kwargs}"
            )
    else:
        test_case.assertEqual(
            orig, decomp, msg=f"{op.__name__}\nargs = {args}\nkwargs = {kwargs}"
        )


def op_assert_equal(test_case, op, test_dtype, orig, decomp, args, kwargs):
    test_case.assertEqual(
        orig.dtype,
        decomp.dtype,
        f"Operation: {op}, orig.dtype: {orig.dtype}, decomp.dtype: {decomp.dtype}, {args}, {kwargs}",
    )
    # Before adding an entry to this table, make sure your decomposition is right :)
    tol_table = {
        # Due to strange epsilon behaviors, see https://github.com/pytorch/pytorch/issues/73161
        (torch.float32, torch.ops.aten.native_layer_norm.default): (1e-3, 1e-3),
        (torch.float32, torch.ops.aten.native_layer_norm_backward.default): (
            1e-3,
            1e-3,
        ),
        (torch.float64, torch.ops.aten.native_layer_norm.default): (1e-6, 1e-6),
        # This exceeds default tolerances only on CPU, on CUDA it's fine
        (torch.float32, torch.ops.aten.grid_sampler_2d.default): (7e-6, 3e-5),
        # Exceeds tolerances on CUDA, likely due to fma
        (torch.float32, torch.ops.aten.mv.default): (1e-5, 3e-5),
        (torch.complex64, torch.ops.aten.mv.default): (5e-5, 5e-5),
        (torch.float64, torch.ops.aten.upsample_bicubic2d.vec): (1e-5, 5e-4),
        (torch.float64, torch.ops.aten.upsample_bicubic2d.default): (1e-5, 5e-4),
        # The decomposition is TOO correct. It computes everything in int64, so sometimes
        # there's an off-by-one error. See
        # https://github.com/pytorch/pytorch/issues/81996
        # https://github.com/pytorch/pytorch/issues/82230
        (torch.int8, torch.ops.aten.linspace.default): (0, 1),
        (torch.uint8, torch.ops.aten.linspace.default): (0, 1),
        (torch.int16, torch.ops.aten.linspace.default): (0, 1),
        (torch.int32, torch.ops.aten.linspace.default): (0, 1),
        (torch.int64, torch.ops.aten.linspace.default): (0, 1),
        (torch.int8, torch.ops.aten.linspace.Tensor_Tensor): (0, 1),
        (torch.uint8, torch.ops.aten.linspace.Tensor_Tensor): (0, 1),
        (torch.int16, torch.ops.aten.linspace.Tensor_Tensor): (0, 1),
        (torch.int32, torch.ops.aten.linspace.Tensor_Tensor): (0, 1),
        (torch.int64, torch.ops.aten.linspace.Tensor_Tensor): (0, 1),
        (torch.int8, torch.ops.aten.linspace.Tensor_Scalar): (0, 1),
        (torch.uint8, torch.ops.aten.linspace.Tensor_Scalar): (0, 1),
        (torch.int16, torch.ops.aten.linspace.Tensor_Scalar): (0, 1),
        (torch.int32, torch.ops.aten.linspace.Tensor_Scalar): (0, 1),
        (torch.int64, torch.ops.aten.linspace.Tensor_Scalar): (0, 1),
        (torch.int8, torch.ops.aten.linspace.Scalar_Tensor): (0, 1),
        (torch.uint8, torch.ops.aten.linspace.Scalar_Tensor): (0, 1),
        (torch.int16, torch.ops.aten.linspace.Scalar_Tensor): (0, 1),
        (torch.int32, torch.ops.aten.linspace.Scalar_Tensor): (0, 1),
        (torch.int64, torch.ops.aten.linspace.Scalar_Tensor): (0, 1),
    }
    if (decomp.dtype, op) in tol_table:
        rtol, atol = tol_table[(decomp.dtype, op)]
    else:
        rtol, atol = _getDefaultRtolAndAtol(orig.dtype, decomp.dtype)
    test_case.assertEqual(
        orig,
        decomp,
        rtol=rtol,
        atol=atol,
        msg=f"{op.__name__}\nargs = {args}\nkwargs = {kwargs}",
    )


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
            # TODO We should check that the integer outputs also agree
            result = tuple(
                r
                for r in result
                if isinstance(r, Tensor) and (r.is_floating_point() or r.is_complex())
            )
            assert len(result) > 0
        return result

    return wrapped, primals


# NB: This also upcasts dtype arguments
# TODO: handle complex correctly
def upcast_tensor(x, dtype=torch.float32):
    if isinstance(x, Tensor) and x.dtype.is_floating_point:
        return x.to(dtype=dtype)
    elif isinstance(x, torch.dtype) and x in [
        torch.float16,
        torch.bfloat16,
        torch.float,
    ]:
        return dtype
    else:
        return x


def normalize_op_input_output(f, sample, requires_grad=True):
    args = tuple([sample.input] + list(sample.args))
    return normalize_op_input_output2(
        f,
        args,
        sample.kwargs,
        sample.output_process_fn_grad,
        requires_grad=requires_grad,
    )


CROSS_REF_EXCLUDE_SET = {
    # CUBLAS_STATUS_NOT_SUPPORTED when calling
    # `cublasGemmStridedBatchedExFix(handle, opa, opb, (int)m, (int)n, (int)k,
    # (void*)&falpha, a, CUDA_R_16BF, (int)lda, stridea, b, CUDA_R_16BF,
    # (int)ldb, strideb, (void*)&fbeta, c, CUDA_R_16BF, (int)ldc, stridec,
    # (int)num_batches, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)`
    ("cuda", torch.bfloat16, "nn.functional.bilinear"),
    # randomness
    (None, None, "special.ndtr"),  # aten.special_ndtr was not decomposed
    (None, None, "new_empty"),
    (None, None, "empty_like"),
    (None, None, "empty"),
    # AssertionError: False is not true : aten.item was not decomposed, saw calls for: aten._local_scalar_dense.default.
    (None, None, "item"),
    # It's the only in-place op without an out-of-place equivalent in the Python API
    # Its OpInfo wrongly registers it as `torch.zero_(x.clone())`.
    (None, None, "zero_"),
    # No idea what's going on here
    # In the recursive test logsumexp.default fails with args = (torch.tensor(-math.inf), [])
    # in the test, but it seems to pass when tested locally and in the logsumexp test
    (None, torch.float32, "masked.logsumexp"),
    (None, torch.float64, "masked.logsumexp"),
    # exp_vml_cpu not implemented for Half
    (torch.cpu, torch.float16, "signal.windows.exponential"),
    (torch.cpu, torch.float16, "signal.windows.gaussian"),
    # sin_vml_cpu not implemented for Half
    (torch.cpu, torch.float16, "signal.windows.cosine"),
    # CompositeAutogradImplicit
    # See https://github.com/pytorch/pytorch/issues/81669
    (None, None, "nn.functional.relu6"),
    # This decomp runs before autograd.
    (None, None, "nn.functional.rrelu"),
    (None, None, "meshgrid"),
    # Decomposition registered as Autograd
    (None, None, "nn.functional.hardshrink"),
    (None, None, "nn.functional.softshrink"),
    # diag was not decomposed (it just registers a decomp for diag_out, torch.diag is CompImplicit)
    (None, None, "diag"),
    # _softmax_backward_data's CPU kernel for bfloat16 always return the grad_input as float32
    ("cpu", torch.bfloat16, "_softmax_backward_data"),
    (None, None, "norm"),
    # native_batch_norm is only implicit when python dispatcher is on (and noncomposite otherwise)
    (None, None, "native_batch_norm"),
    (None, None, "_upsample_bilinear2d_aa"),
    (None, None, "empty_strided"),  # aten.empty_strided was not decomposed
}

CROSS_REF_BACKWARD_EXCLUDE_SET = {
    # Decomposed backward formula is not as precise
    ("cpu", torch.bfloat16, "nn.functional.hardswish"),
    ("cuda", torch.float16, "nn.functional.cross_entropy"),
}

all_decomposed = set()
all_called = defaultdict(int)

# Helpful snippet for testing coverage
"""
import atexit
def check_coverage():
    print("missing coverage:")
    print("\n".join(map(str, decomposition_table.keys() - all_decomposed)))
atexit.register(check_coverage)
"""

# Helpful snippet for Horace to create his google sheet :)
"""
import atexit
def dump_ops():
    with open('run_ops.txt', 'w') as f, open('count_ops.txt', 'w') as g:
        for op, count in sorted(all_called.items(), key=lambda x: x[0].__name__):
            f.write(f'{op.__name__}\n')
            g.write(f'{count}\n')
    with open('run_decompositions.txt', 'w') as f:
        for op in sorted([i.__name__ for i in all_decomposed]):
            f.write(f'{op}\n')

atexit.register(dump_ops)
"""


def any_unsupported(args, kwargs):
    def test_unsupported(t):
        if type(t) is torch.Tensor or type(t) is torch.nn.Parameter:
            # These are all things that we haven't coded decompositions
            # to handle correctly.  Maybe they should.
            return any(
                [
                    t.is_sparse_csr,
                    t.is_sparse,
                    t.is_onednn,
                    t.is_quantized,
                    t.is_nested,
                    torch._is_functional_tensor(t),
                ]
            )
        elif torch.overrides.is_tensor_like(t):
            # Decompositions will generally change the behavior of Tensor-like
            # subclasses, so bypass tests in this case too
            return True
        else:
            return False

    flat_args = pytree.arg_tree_leaves(*args, **kwargs)
    return any(test_unsupported(x) for x in flat_args)


core_backward_failures = {
    skip("_softmax_backward_data"),  # slow: fails with --timeout=360 secs
    xfail("addcdiv"),
    skip("addcmul"),  # slow: fails with --timeout=360 secs
    skip("deg2rad"),  # slow: fails with --timeout=360 secs
    skip("diag_embed"),  # slow: fails with --timeout=360 secs
    skip("frac"),  # slow: fails with --timeout=360 secs
    skip("grid_sampler_2d"),  # slow: fails with --timeout=360 secs
    xfail("lerp"),
    skip("logaddexp"),  # slow: fails with --timeout=360 secs
    skip("native_dropout_backward"),  # slow: fails with --timeout=360 secs
    xfail("nn.functional.binary_cross_entropy_with_logits"),
    skip("nn.functional.glu"),  # slow: fails with --timeout=360 secs
    xfail("nn.functional.hardshrink"),
    xfail("nn.functional.softshrink"),
    skip("nn.functional.unfold"),  # slow: fails with --timeout=360 secs
    xfail("norm"),
    xfail("norm", "fro"),
    xfail("norm", "inf"),
    xfail("norm", "nuc"),
    skip("rad2deg"),  # slow: fails with --timeout=360 secs
    skip("renorm"),  # slow: fails with --timeout=360 secs
    skip("rot90"),  # slow: fails with --timeout=360 secs
    skip("rsub"),  # slow: fails with --timeout=360 secs
    skip("sgn"),  # slow: fails with --timeout=360 secs
    skip("special.xlog1py"),  # slow: fails with --timeout=360 secs
    xfail("stack"),
    skip("tril"),  # slow: fails with --timeout=360 secs
    skip("triu"),  # slow: fails with --timeout=360 secs
    skip("unfold_copy"),  # slow: fails with --timeout=360 secs
    skip("xlogy"),  # slow: fails with --timeout=360 secs
    xfail("zero_"),
}
if not TEST_WITH_SLOW:
    core_backward_failures.update(
        {
            skip("addr"),  # slow: takes 46 sec on A100
            skip("baddbmm"),  # slow: takes 800+ sec on A100
            skip("clamp_min"),  # slow: takes 800 sec on A100
            skip("clamp_max"),  # slow: takes 800 sec on A100
            skip("logit"),  # slow: takes 44 sec on A100
            skip("nn.functional.hardswish"),  # slow: takes 60 sec on A100
            skip("std_mean"),  # slow: takes 170 sec on A100
            skip("split", variant_name="list_args"),  # slow: takes 118 sec on A100
            skip("transpose"),  # slow: takes 50 sec on A100
            skip("unbind"),  # slow: takes 70 sec on A100
            skip("unsafe_split"),  # slow: takes 49 sec on A100
        }
    )

comprehensive_failures = {
    xfail(
        "nn.functional.interpolate", "bilinear", dtypes=(torch.uint8,)
    ),  # off by one error
    xfail(
        "nn.functional.interpolate", "bicubic", dtypes=(torch.uint8,)
    ),  # off by one error
    xfail(
        "nn.functional.upsample_bilinear", "", dtypes=(torch.uint8,)
    ),  # off by one error
}


@unMarkDynamoStrictTest
class TestDecomp(TestCase):
    longMessage = True

    # NB: This actually overlaps with test_comprehensive, but it only
    # runs on things that are definitely decomposed so it's a lot faster
    # to run
    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    @onlyNativeDeviceTypes
    @skipIfCrossRef
    @suppress_warnings
    @ops(_decomp_test_ops)
    def test_quick(self, device, dtype, op):
        self.do_cross_ref(device, dtype, op, run_all=False)

    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    @skipOps("TestDecomp", "test_quick_core_backward", core_backward_failures)
    @onlyNativeDeviceTypes
    @skipIfCrossRef
    @suppress_warnings
    @ops(_decomp_test_ops_core_autograd, allowed_dtypes=(torch.float64,))
    def test_quick_core_backward(self, device, dtype, op):
        for sample_input in op.sample_inputs(device, dtype, requires_grad=True):
            aten_name = op.decomp_aten_name or op.aten_name
            args = [sample_input.input] + list(sample_input.args)
            kwargs = sample_input.kwargs
            func = partial(op.get_op(), **kwargs)
            with self.DecompCrossRefMode(
                self, self.precision, self.rel_tol, dtype, run_all=False
            ) as mode, enable_python_dispatcher():
                torch.autograd.gradcheck(func, args)
            self.check_decomposed(aten_name, mode)

    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    @onlyNativeDeviceTypes
    @skipIfCrossRef
    @skipOps("TestDecomp", "test_comprehensive", comprehensive_failures)
    @suppress_warnings
    @ops(op_db)
    def test_comprehensive(self, device, dtype, op):
        self.do_cross_ref(device, dtype, op, run_all=True)

    def test_uniform(self, device):
        size = (2, 3, 4, 5)
        dtype = torch.float32
        x = make_tensor(size, dtype=dtype, device=device)
        low = 0.3
        high = 0.9

        torch.manual_seed(123)
        ref = torch.ops.aten.uniform(x, low, high)
        torch.manual_seed(123)
        res = torch._decomp.decompositions.uniform(x, low=low, high=high)
        self.assertEqual(ref, res)

    def test_broadcasting_index_copy(self, device):
        x = torch.zeros([1, 10], device=device)
        xs = torch.ones([2, 10], device=device)

        def index_copy(xs, x):
            torch._decomp.decompositions.index_copy_(
                xs, 0, torch.tensor(0).to(device), x
            )

        index_copy(xs, x)

        xs_two = torch.ones([2, 10], device=device)
        xs_two[0] = x

        self.assertEqual(xs, xs_two)

    def test_cat_single_input(self, device):
        decomp_table = torch._inductor.decomposition.select_decomp_table()
        cat_inductor = decomp_table[torch.ops.aten.cat.default]

        inp = torch.rand([2048, 2048], device=device)
        inps = [inp for _ in range(10)]

        for dim in (-1, 0, 1):
            self.assertEqual(torch.cat(inps, dim), cat_inductor(inps, dim))

    def test_rrelu_with_noise(self, device):
        # rrelu_with_noise behavior depends on a) whether elements in the input
        # are <= 0, and b) whether we're in training mode. Cover all cases:
        dtype = torch.float64
        x = torch.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0], dtype=dtype, device=device)
        lower = 1.0
        upper = 4.0
        training = False

        torch.manual_seed(123)
        noise_ref = torch.zeros(x.shape, dtype=dtype, device=device)
        ref = torch.ops.aten.rrelu_with_noise(x, noise_ref, lower, upper, training)

        torch.manual_seed(123)
        noise_res = torch.zeros(x.shape, dtype=dtype, device=device)
        res = torch._decomp.decompositions.rrelu_with_noise(
            x,
            noise_res,
            lower,
            upper,
            training,
        )
        self.assertEqual(ref, res)
        self.assertEqual(noise_ref, noise_res)

        # Now with training=True:
        training = True

        torch.manual_seed(123)
        noise_ref = torch.zeros(x.shape, dtype=dtype, device=device)
        ref = torch.ops.aten.rrelu_with_noise(x, noise_ref, lower, upper, training)

        torch.manual_seed(123)
        noise_res = torch.zeros(x.shape, dtype=dtype, device=device)
        res = torch._decomp.decompositions.rrelu_with_noise(
            x,
            noise_res,
            lower,
            upper,
            training,
        )
        self.assertEqual(ref, res)
        self.assertEqual(noise_ref, noise_res)

    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    @suppress_warnings
    @tf32_off()
    # only tests RNNs since we have py dispsatcher decomps for them
    @modules(
        filter(
            lambda m: m.module_cls in (torch.nn.RNN, torch.nn.LSTM, torch.nn.GRU),
            module_db,
        )
    )
    def test_rnn_decomp_module(self, device, dtype, module_info, training):
        module_cls = module_info.module_cls
        module_inputs = module_info.module_inputs_func(
            module_info,
            device=device,
            dtype=dtype,
            requires_grad=True,
            training=training,
        )
        for module_input in module_inputs:
            if module_input.forward_input is None:
                continue
            args, kwargs = (
                module_input.constructor_input.args,
                module_input.constructor_input.kwargs,
            )
            m = module_cls(*args, **kwargs)
            m.to(device).to(dtype)

            args, kwargs = (
                module_input.forward_input.args,
                module_input.forward_input.kwargs,
            )
            with self.DecompCrossRefMode(
                self, self.precision, self.rel_tol, dtype, run_all=True
            ), enable_python_dispatcher():
                decomp_out = m(*args, **kwargs)

            non_decomp_out = m(*args, **kwargs)
            # without this check, incorrect decomps at the python dispatcher level can still pass because
            # they're checking aten decomps at the torch_dispatch level
            self.assertEqual(decomp_out, non_decomp_out)

    def test_batch_norm_unflatten_weight_bias(self, device):
        # https://github.com/pytorch/pytorch/issues/100970
        shape = (1, 3, 2, 2)
        input = torch.randn(shape, device=device)
        weight = torch.randn((3, 1, 1, 1), device=device)
        bias = torch.randn(3, device=device)
        mean = torch.randn(3, device=device)
        var = torch.randn(3, device=device)
        res = torch._decomp.decompositions.native_batch_norm(
            input, weight, bias, mean, var, False, 1, 1e-05
        )
        self.assertEqual(shape, res[0].shape)

    def test_arange_graph(self, device):
        from torch.fx.experimental.proxy_tensor import make_fx

        def func(x, start):
            le = x.shape[-1]
            if start is None:
                a = torch.arange(le, dtype=torch.float32, device=x.device)
            else:
                a = torch.arange(start, le, dtype=torch.float32, device=x.device)
            return a

        pattern = r", device = device\(.+\), requires_grad = False"

        cfunc = make_fx(func, decomposition_table=decomposition_table)
        fx_g = cfunc(torch.rand(10, device=device), None)
        fx_g_code = fx_g.code.strip()
        # Remove device and requires_grad
        fx_g_code = re.sub(pattern, "", fx_g_code)
        self.assertExpectedInline(
            fx_g_code,
            """\
def forward(self, x_1, start_1):
    iota = torch.ops.prims.iota.default(10, start = 0, step = 1, dtype = torch.int64)
    mul = torch.ops.prims.mul.default(iota, 1);  iota = None
    add = torch.ops.prims.add.default(mul, 0);  mul = None
    convert_element_type = torch.ops.prims.convert_element_type.default(add, torch.float32);  add = None
    return convert_element_type""",
        )

        fx_g = cfunc(torch.rand(10, device=device), 1)
        fx_g_code = fx_g.code.strip()
        # Remove device and requires_grad
        fx_g_code = re.sub(pattern, "", fx_g_code)
        self.assertExpectedInline(
            fx_g_code,
            """\
def forward(self, x_1, start_1):
    iota = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64)
    mul = torch.ops.prims.mul.default(iota, 1);  iota = None
    add = torch.ops.prims.add.default(mul, 1);  mul = None
    convert_element_type = torch.ops.prims.convert_element_type.default(add, torch.float32);  add = None
    return convert_element_type""",
        )

    def test_masked_fill(self, device):
        from torch.fx.experimental.proxy_tensor import make_fx

        if torch.device(device).type not in [
            "xpu",
            "cuda",
            torch._C._get_privateuse1_backend_name(),
        ]:
            self.skipTest("only runs on XPU and CUDA and PrivateUse1.")

        def func(scores, mask, value):
            return scores.masked_fill(mask, value)

        scores_t = torch.tensor([1, 2, 3, 4], device=device)
        mask_t = torch.tensor([True, True, True, True], device=device)
        value_t = torch.tensor(0, dtype=scores_t.dtype)
        cfunc = make_fx(func, decomposition_table=decomposition_table)
        fx_g = cfunc(scores_t, mask_t, value_t)
        self.assertExpectedInline(
            fx_g.code.strip(),
            """\
def forward(self, scores_1, mask_1, value_1):
    where = torch.ops.prims.where.default(mask_1, value_1, scores_1);  mask_1 = value_1 = scores_1 = None
    return where""",
        )

    class DecompCrossRefMode(TorchDispatchMode):
        def __init__(self, test_case, saved_precision, saved_rel_tol, dtype, run_all):
            self.test_case = test_case
            self.saved_precision = saved_precision
            self.saved_rel_tol = saved_rel_tol
            self.test_dtype = dtype
            self.run_all = run_all

            # We check the correctness of each decomposition right after running it.
            # So, when we encounter a decomposition, we run the function normally, and
            # then run the decomposition, and ensure they're identical.
            self.called = set()
            self.decomposed = set()

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.test_case.precision = self.saved_precision
            self.test_case.rel_tol = self.saved_rel_tol

            self.called.add(func)
            all_called[func] += 1

            # Stuff we shouldn't bother testing
            # (TODO: remove detach from the decomp table?)
            # N.b. Testing in-place ops would need dedicated logic
            in_place = func.name()[-1] == "_"
            ignored_ops = [
                torch.ops.aten.detach.default,
                # non-deterministic ops
                torch.ops.aten.empty.memory_format,
                torch.ops.aten.empty_like.default,
                torch.ops.aten.new_empty.default,
                torch.ops.aten.empty_strided.default,
                torch.ops.aten.new_empty_strided.default,
                torch.ops.aten.randn.default,
                torch.ops.aten.native_dropout.default,
            ]
            if (
                func not in decomposition_table
                or func in ignored_ops
                or torch.Tag.nondeterministic_seeded in func.tags
                or any_unsupported(args, kwargs)
                or in_place
            ):
                return func(*args, **kwargs)

            self.decomposed.add(func)
            all_decomposed.add(func)

            # We take 2 main strategies for verifying correctness/numerical stability of decompositions
            # The first one is simply tolerance checking between decomp_out and pytorch_out
            # However, for fp16/bf16 and reductions, this becomes very
            # finicky, as there are not many guarantees we can make.
            # So, for fp16/bf16, we instead compare the difference of
            # {decomp_out, pytorch_out_64} and {pytorch_out,
            # pytorch_out_64}. In other words, we compare how far the
            # decomposition and pytorch are from the "ground truth" (i.e.
            # fp64). If the decomposition results in more error, we error

            # We also decompose the decomposition recursively for
            # further coverage, as some paths not be exercised directly by
            # OpInfos (sadly) but just by other ops

            decomposition = decomposition_table[func]

            do_relative_check = self.test_dtype in [torch.float16, torch.bfloat16]
            if self.run_all:
                # Execute recursively via DFS, to find the root of a possible error first
                with self:
                    decomp_out = pytree.tree_leaves(decomposition(*args, **kwargs))
            else:
                decomp_out = pytree.tree_leaves(decomposition(*args, **kwargs))

            # At this stage we should not be decomposing an in-place op
            # We'd like to have decompositions that decompose out-of-place ops into out-of-place ops
            #  because decompositions are run after functionalisation and we would not like them to
            #  de-functionalise the graph, as that would break AoTAutograd
            # We run the real function *after* the decomposition to make sure that the
            # decomposition does not modify any of the inputs in-place. If it does
            # real_out should be differen than decom_out so we should catch this
            real_out_unflat = func(*args, **kwargs)
            real_out = pytree.tree_leaves(real_out_unflat)

            assert len(real_out) == len(decomp_out)

            if do_relative_check:
                upcast = partial(upcast_tensor, dtype=torch.float64)
                real_out_double, _ = tree_flatten(
                    func(*tree_map(upcast, args), **tree_map(upcast, kwargs))
                )
                for i, (orig, decomp, ref) in enumerate(
                    zip(real_out, decomp_out, real_out_double)
                ):
                    if not isinstance(orig, torch.Tensor):
                        assert type(orig) == type(decomp)
                        assert orig == decomp
                        continue
                    op_assert_ref(
                        self.test_case,
                        func,
                        self.test_dtype,
                        i,
                        orig,
                        decomp,
                        ref,
                        args,
                        kwargs,
                    )
            else:
                for orig, decomp in zip(real_out, decomp_out):
                    if not isinstance(orig, torch.Tensor):
                        assert type(orig) == type(decomp)
                        assert orig == decomp
                        continue
                    op_assert_equal(
                        self.test_case,
                        func,
                        self.test_dtype,
                        orig,
                        decomp,
                        args,
                        kwargs,
                    )

            return real_out_unflat

    def check_decomposed(self, aten_name, mode):
        self.assertTrue(
            any(overload_to_aten_name(c) == aten_name for c in mode.decomposed),
            msg=(
                f"aten.{aten_name} was not decomposed, saw calls for: "
                f"{', '.join(map(str, list(mode.called)))}. If your op is  "
                f"CompositeImplicitAutograd you should skip this test "
                f"by updating CROSS_REF_EXCLUDE_SET."
            ),
        )

    @skipIfTorchDynamo("Test does not work with TorchDynamo")
    def do_cross_ref(self, device, dtype, op, *, run_all):
        test_keys = [
            (torch.device(device).type, dtype, op.name),
            (None, dtype, op.name),
            (None, None, op.name),
        ]
        if any(key in CROSS_REF_EXCLUDE_SET for key in test_keys):
            self.skipTest(f"{op.name} in {dtype} not supported")

        skip_decomp_vjp = any(
            key in CROSS_REF_BACKWARD_EXCLUDE_SET for key in test_keys
        )

        requires_grad = (
            op.supports_autograd
            and dtype in op.supported_backward_dtypes(torch.device(device).type)
            # TODO: OpInfo really ought to error out for this case, but it's
            # not exercised in test_ops_gradients atm.  The problem is not
            # complex32 per-se (which is supported by data movement only ops)
            # but that when we do backwards we expect other ops like add to work
            and not dtype == torch.complex32
        )
        samples = op.sample_inputs(device, dtype, requires_grad=requires_grad)

        aten_name = op.decomp_aten_name or op.aten_name

        func = op.get_op()

        def run_without_python_dispatcher(mode):
            return any(
                isinstance(op, torch._ops.OpOverload)
                and op.has_kernel_for_dispatch_key(
                    DispatchKey.CompositeImplicitAutograd
                )
                for op in mode.decomposed.union([func])
            )

        for sample_input in samples:
            if requires_grad:
                fn, primals = normalize_op_input_output(func, sample_input)
                primals = tree_map(
                    lambda x: x if isinstance(x, torch.Tensor) else x, primals
                )

                # Once https://github.com/pytorch/pytorch/pull/75965/ I can
                # store the called list on the mode object instance and no
                # explicit clearing is necessary as I will create a fresh mode
                # for each region
                with self.DecompCrossRefMode(
                    self, self.precision, self.rel_tol, dtype, run_all
                ) as mode, enable_python_dispatcher():
                    decomp_out, decomp_vjp_fn = ref_vjp_no_create(fn, *primals)
                if run_without_python_dispatcher(mode):
                    # without this check, incorrect decomps at the python dispatcher level can still pass because
                    # they're checking aten decomps at the torch_dispatch level.
                    with self.DecompCrossRefMode(
                        self, self.precision, self.rel_tol, dtype, run_all
                    ) as mode:
                        decomp_out, decomp_vjp_fn = ref_vjp_no_create(fn, *primals)
                if aten_name in decomposition_names:
                    self.check_decomposed(aten_name, mode)

                if not skip_decomp_vjp and (
                    op.aten_backward_name in decomposition_names or run_all
                ):
                    cotangents = tree_map(lambda x: torch.randn_like(x), decomp_out)

                    with self.DecompCrossRefMode(
                        self, self.precision, self.rel_tol, dtype, run_all
                    ) as mode, enable_python_dispatcher():
                        decomp_vjp_fn(cotangents)
                    if run_without_python_dispatcher(mode):
                        # without this check, incorrect decomps at the python dispatcher level can still pass because
                        # they're checking aten decomps at the torch_dispatch level.
                        with self.DecompCrossRefMode(
                            self, self.precision, self.rel_tol, dtype, run_all
                        ) as mode:
                            decomp_vjp_fn(cotangents)
                    if not run_all:
                        self.check_decomposed(op.aten_backward_name, mode)

            elif aten_name in decomposition_names or run_all:
                args = [sample_input.input] + list(sample_input.args)
                kwargs = sample_input.kwargs
                # A failure here might be because the decomposition for the op is wrong or because a
                # decomposition used by the particular op is wrong.
                with self.DecompCrossRefMode(
                    self, self.precision, self.rel_tol, dtype, run_all
                ) as mode, enable_python_dispatcher():
                    func(*args, **kwargs)

                if run_without_python_dispatcher(mode):
                    # without this check, incorrect decomps at the python dispatcher level can still pass because
                    # they're checking aten decomps at the torch_dispatch level.
                    with self.DecompCrossRefMode(
                        self, self.precision, self.rel_tol, dtype, run_all
                    ) as mode:
                        func(*args, **kwargs)

                if not run_all:
                    self.check_decomposed(aten_name, mode)
            else:
                assert op.supports_autograd
                self.skipTest(
                    "only backwards is decomposed, but dtype doesn't support AD"
                )


instantiate_device_type_tests(TestDecomp, globals())


class DecompOneOffTests(TestCase):
    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    @onlyNativeDeviceTypes
    @skipIfCrossRef
    def test_contiguous_softmax(self, device):
        size = (2, 4, 3, 3)
        stride = (9, 18, 3, 1)
        dtype = torch.float32

        x = torch.randn(size, dtype=dtype, device=device)
        x = torch.as_strided(x, size, stride)

        ref = torch.ops.aten._softmax(x, -1, False)
        res = torch._decomp.decompositions._softmax(x, -1, False)
        self.assertEqual(ref.stride(), res.stride())

    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    @onlyNativeDeviceTypes
    @skipIfCrossRef
    def test_contiguous_log_softmax(self, device):
        size = (2, 4, 3, 3)
        stride = (9, 18, 3, 1)

        dtype = torch.float32
        x = torch.randn(size, dtype=dtype, device=device)
        x = torch.as_strided(x, size, stride)

        ref = torch.ops.aten._log_softmax(x, -1, False)
        res = torch._decomp.decompositions._log_softmax(x, -1, False)
        self.assertEqual(ref.stride(), res.stride())

    @onlyCUDA
    def test_exponential_non_inf(self, device):
        inp = torch.empty((4, 400, 256), device=device)

        with torch._dynamo.utils.preserve_rng_state():
            exp_ref = inp.exponential_()
        exp = torch._refs.exponential(inp)

        self.assertEqual(exp, exp_ref)
        self.assertFalse(exp.isinf().any())

    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    @skipIfCrossRef
    @onlyCUDA
    def test_amp_batch_norm_backward(self):
        device = "cuda"
        grad_out = torch.randn((1, 2, 16, 16), dtype=torch.float16, device=device)
        x = torch.randn((1, 2, 16, 16), dtype=torch.float16, device=device)
        weight = torch.randn((2,), dtype=torch.float32, device=device)
        rmean = torch.randn((2,), dtype=torch.float32, device=device)
        rvar = torch.randn((2,), dtype=torch.float32, device=device)
        mean = torch.randn((0,), dtype=torch.float32, device=device)

        ref = torch.ops.aten.native_batch_norm_backward(
            grad_out,
            x,
            weight,
            rmean,
            rvar,
            mean,
            mean,
            False,
            1e-05,
            [True, True, True],
        )
        res = torch._decomp.decompositions.native_batch_norm_backward(
            grad_out,
            x,
            weight,
            rmean,
            rvar,
            mean,
            mean,
            False,
            1e-05,
            [True, True, True],
        )
        for a, b in zip(ref, res):
            self.assertEqual(a.stride(), b.stride())
            self.assertEqual(a.dtype, b.dtype)

    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    @onlyNativeDeviceTypes
    @skipIfCrossRef
    def test_elu_backward(self, device):
        size = (2, 4, 3, 3)
        dtype = torch.float32
        grad_out = torch.randn(size, dtype=dtype, device=device)
        out = torch.randn(size, dtype=dtype, device=device)

        ref = torch.ops.aten.elu_backward(grad_out, 1.0, 1, 1, True, out)
        res = torch._decomp.decompositions.elu_backward(grad_out, 1.0, 1, 1, True, out)
        self.assertEqual(ref, res)

    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    @onlyNativeDeviceTypes
    @skipIfCrossRef
    def test_threshold_backward_dtype(self, device):
        grad = torch.randint(10, (4,), device=device)
        input_tensor = torch.randint(10, (4,), device=device)

        ref = torch.ops.aten.threshold_backward(grad, input_tensor, 1)
        res = torch._decomp.decompositions.threshold_backward(grad, input_tensor, 1)
        self.assertEqual(ref.dtype, res.dtype)

    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    @onlyNativeDeviceTypes
    @skipIfCrossRef
    def test_weight_norm_interface(self, device):
        g = torch.randn((3, 10, 10), device=device)
        v = torch.randn((1, 1, 10), device=device)

        ref = torch.ops.aten._weight_norm_interface(g, v, 2)
        res = torch._decomp.decompositions._weight_norm_interface(g, v, 2)
        self.assertTrue(torch.allclose(ref[0], res[0]))
        self.assertTrue(torch.allclose(ref[1], res[1]))

        inp = torch.rand([30, 10], device=device)
        inp2 = torch.rand([30, 1], device=device)

        self.assertEqual(
            torch.ops.aten._weight_norm_interface(inp, inp2),
            torch._decomp.decompositions._weight_norm_interface(inp, inp2),
        )

    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    @onlyCPU
    @skipIfCrossRef
    @skipOps(
        "DecompOneOffTests",
        "test_sdpa",
        [
            xfail(
                "nn.functional.scaled_dot_product_attention",
                dtypes=[torch.half],
            ),
        ],
    )
    @ops(_sdpa_op_info)
    def test_sdpa(self, device, dtype, op):
        # SDPA doesn't support float16, this is aligned with aten/src/ATen/native/transformers/attention.cpp. If we
        # add support for float16 over there we should update this test as well.

        class ScaledDotProductAttention(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(
                self, query_layer, key_layer, value_layer, mask=None, is_causal=True
            ):
                attn_output = op(
                    query_layer,
                    key_layer,
                    value_layer,
                    attn_mask=mask,
                    dropout_p=0.0,
                    is_causal=is_causal,
                )
                return attn_output

        query_layer = torch.randn(1, 128, 100, 64, device=device, dtype=dtype)
        key_layer = torch.randn(1, 128, 100, 64, device=device, dtype=dtype)
        value_layer = torch.randn(1, 128, 100, 64, device=device, dtype=dtype)
        masks = [None, torch.ones((1, 1, 100, 100), device=device, dtype=torch.bool)]

        atol, rtol = dtype_precisions[dtype]

        for mask in masks:
            is_causal = mask is None
            attention = ScaledDotProductAttention()
            decomposed_res = (
                torch._decomp.decompositions.scaled_dot_product_flash_attention_for_cpu(
                    query_layer, key_layer, value_layer, 0.0, is_causal, attn_mask=mask
                )
            )
            eager_res = op(
                query_layer,
                key_layer,
                value_layer,
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=is_causal,
            )

            self.assertTrue(
                torch.allclose(decomposed_res[0], eager_res, atol=atol, rtol=rtol)
            )


instantiate_device_type_tests(DecompOneOffTests, globals())


class HasDecompTest(TestCase):
    def setUp(self):
        super().setUp()
        self.maxDiff = None

    @staticmethod
    def _can_appear_in_trace(op: torch._ops.OpOverload) -> bool:
        has_tensor_arg = any(
            "Tensor" in str(a.type)
            for a in itertools.chain(op._schema.arguments, op._schema.returns)
        )
        if not has_tensor_arg:
            return False

        try:
            # CompositeImplicitAutograd ops are transparent to the tracer, so don't need decompositions
            return not op.has_kernel_for_dispatch_key(
                DispatchKey.CompositeImplicitAutograd
            )
        except RuntimeError as e:
            # has_key fails for some jit-registered ops, which shouldn't be
            # relevant here anyway
            if "does not exist" in str(e):
                return False
            raise

    def test_has_decomposition(self):
        def all_aten_overloads():
            for name in torch._C._dispatch_get_all_op_names():
                if not name.startswith("aten::"):
                    continue

                name = name[6:]
                if "." in name:
                    packet_name, overload_name = name.split(".")
                else:
                    packet_name, overload_name = name, "default"

                packet = getattr(aten, packet_name)
                assert isinstance(packet, torch._ops.OpOverloadPacket)
                op = getattr(packet, overload_name)
                yield op

        # This is for operators that are only registered in some CI
        # configurations, so would cause the test to fail
        allow_list = {aten.get_gradients.default}

        overloads_wanting_decomp = {
            op for op in all_aten_overloads() if self._can_appear_in_trace(op)
        }
        ops_missing_decomp = overloads_wanting_decomp - decomposition_table.keys()
        ops_missing_decomp -= allow_list
        self.assertExpected(
            "".join(sorted(op.name() + "\n" for op in ops_missing_decomp))
        )

    def test_aten_core_operators(self):
        # If a decomposition isn't included in the core decompositions,
        # then it must decompose a core ATen operator.
        #
        # See NOTE [Core ATen Ops]
        #
        # If this test fails then either:
        # - Add the decomposition to torch._decomp.core_aten_decompositions,
        #   if decomposition should be used by inductor (not a core operator).
        # - Run this test again with EXPECTTEST_ACCEPT=1 to update the list of
        #   core ATen operators (and inductor will not use the decomposition).

        # Some decompositions are registered for CompositeImplicitAutograd
        # operators, which never appear in AOTAutograd's graph so are never used.
        useful_decomps = {
            op
            for op in decomposition_table.keys()
            if isinstance(op, torch._ops.OpOverload) and self._can_appear_in_trace(op)
        }
        core_decomps = torch._decomp.core_aten_decompositions().keys()
        core_aten_ops = useful_decomps - core_decomps
        self.assertExpected("".join(sorted(op.name() + "\n" for op in core_aten_ops)))


if __name__ == "__main__":
    run_tests()
