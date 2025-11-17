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
from torch._export.utils import _is_cia_op
from torch._ops import DispatchKey
from torch.testing import make_tensor
from torch.testing._internal.common_cuda import (
    SM70OrLater,
    tf32_off,
    _get_torch_cuda_version,
)
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
    if isinstance(k, torch._ops.OpOverload) and not _is_cia_op(k)
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
        (torch.float16, torch.ops.aten._batch_norm_with_update.default): 2e-7,
        (torch.bfloat16, torch.ops.aten._batch_norm_with_update.default): 2e-7,
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
    (
        None,
        None,
        "bernoulli",
    ),  # bernoulli is a function of randomness, so couldn't do cross-reference.
}

CROSS_REF_BACKWARD_EXCLUDE_SET = {
    # Decomposed backward formula is not as precise
    ("cpu", torch.bfloat16, "nn.functional.hardswish"),
    ("cuda", torch.float16, "nn.functional.cross_entropy"),
    (
        None,
        None,
        "bernoulli",
    ),  # bernoulli is a function of randomness, so couldn't do cross-reference.
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
                    t.is_mkldnn,
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
    skip(
        "torch._scaled_mm", "", dtypes=(torch.float8_e4m3fn,)
    ),  # Skip _scaled_mm with FP8 on CUDA 13.0+
}


@unMarkDynamoStrictTest
class TestDecomp(TestCase):
    longMessage = True

    # NB: This actually overlaps with test_comprehensive, but it only
    # runs on things that are definitely decomposed so it's a lot faster
    # to run
    @onlyNativeDeviceTypes
    @skipIfCrossRef
    @suppress_warnings
    @ops(_decomp_test_ops)
    def test_quick(self, device, dtype, op):
        self.do_cross_ref(device, dtype, op, run_all=False)

    @skipOps("TestDecomp", "test_quick_core_backward", core_backward_failures)
    @onlyNativeDeviceTypes
    @skipIfCrossRef
    @suppress_warnings
    @ops(_decomp_test_ops_core_autograd, allowed_dtypes=(torch.float64,))
    def test_quick_core_backward(self, device, dtype, op):
        test_keys = [
            (torch.device(device).type, dtype, op.name),
            (None, dtype, op.name),
            (None, None, op.name),
        ]
        if any(key in CROSS_REF_BACKWARD_EXCLUDE_SET for key in test_keys):
            self.skipTest(f"{op.name} in {dtype} not supported")
        for sample_input in op.sample_inputs(device, dtype, requires_grad=True):
            aten_name = op.decomp_aten_name or op.aten_name
            args = [sample_input.input] + list(sample_input.args)
            kwargs = sample_input.kwargs
            func = partial(op.get_op(), **kwargs)
            with (
                self.DecompCrossRefMode(
                    self, self.precision, self.rel_tol, dtype, run_all=False
                ) as mode,
                enable_python_dispatcher(),
            ):
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

    def test_bernoulli_default(self, device):
        p = 0.3
        p_t = p * torch.ones(5, 5)
        torch.manual_seed(123)
        ref = torch.ops.aten.bernoulli.default(p_t)
        torch.manual_seed(123)
        res = torch._decomp.decompositions.bernoulli(p_t)
        ref_p = ref.sum() / torch.prod(torch.tensor(ref.size()))
        res_p = res.sum() / torch.prod(torch.tensor(res.size()))
        self.assertEqual(ref_p, res_p, atol=0.06 * p, rtol=0.06)

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
            with (
                self.DecompCrossRefMode(
                    self, self.precision, self.rel_tol, dtype, run_all=True
                ),
                enable_python_dispatcher(),
            ):
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
            # real_out should be different than decom_out so we should catch this
            real_out_unflat = func(*args, **kwargs)
            real_out = pytree.tree_leaves(real_out_unflat)

            assert len(real_out) == len(decomp_out)

            if do_relative_check:
                device_arg = kwargs.get("device", None)

                def upcast(x):
                    if (isinstance(x, Tensor) and x.device.type == "mps") or (
                        device_arg and torch.device(device_arg).type == "mps"
                    ):
                        return upcast_tensor(x, dtype=torch.float32)
                    else:
                        return upcast_tensor(x, dtype=torch.float64)

                real_out_double, _ = tree_flatten(
                    func(*tree_map(upcast, args), **tree_map(upcast, kwargs))
                )
                for i, (orig, decomp, ref) in enumerate(
                    zip(real_out, decomp_out, real_out_double)
                ):
                    if not isinstance(orig, torch.Tensor):
                        assert type(orig) is type(decomp)
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
                        assert type(orig) is type(decomp)
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
            and dtype != torch.complex32
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
                with (
                    self.DecompCrossRefMode(
                        self, self.precision, self.rel_tol, dtype, run_all
                    ) as mode,
                    enable_python_dispatcher(),
                ):
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

                    with (
                        self.DecompCrossRefMode(
                            self, self.precision, self.rel_tol, dtype, run_all
                        ) as mode,
                        enable_python_dispatcher(),
                    ):
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
                with (
                    self.DecompCrossRefMode(
                        self, self.precision, self.rel_tol, dtype, run_all
                    ) as mode,
                    enable_python_dispatcher(),
                ):
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

    @onlyNativeDeviceTypes
    @skipIfCrossRef
    def test_threshold_backward_dtype(self, device):
        grad = torch.randint(10, (4,), device=device)
        input_tensor = torch.randint(10, (4,), device=device)

        ref = torch.ops.aten.threshold_backward(grad, input_tensor, 1)
        res = torch._decomp.decompositions.threshold_backward(grad, input_tensor, 1)
        self.assertEqual(ref.dtype, res.dtype)

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
        query_layer = torch.randn(1, 128, 100, 64, device=device, dtype=dtype)
        key_layer = torch.randn(1, 128, 100, 64, device=device, dtype=dtype)
        value_layer = torch.randn(1, 128, 100, 64, device=device, dtype=dtype)
        masks = [None, torch.ones((1, 1, 100, 100), device=device, dtype=torch.bool)]

        atol, rtol = dtype_precisions[dtype]

        for mask in masks:
            is_causal = mask is None
            decomposed_res = (
                torch._decomp.decompositions.scaled_dot_product_flash_attention_for_cpu(
                    query_layer, key_layer, value_layer, 0.0, is_causal, attn_mask=mask
                )
            )
            actual_res = decomposed_res[0]
            # Output has form (N, H, L, E), but should be continuous on (L, N, H, E)
            # in order for subsequent view(L * N, H * E) to be valid.
            # So permute(2, 0, 1, 3) before checking that tensor is contiguous
            self.assertTrue(actual_res.permute(2, 0, 1, 3).is_contiguous())

            eager_res = op(
                query_layer,
                key_layer,
                value_layer,
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=is_causal,
            )

            self.assertTrue(torch.allclose(actual_res, eager_res, atol=atol, rtol=rtol))

    @onlyCPU
    def test_native_layer_norm_cpu_decomp(self, device):
        def f(x, w, b):
            return torch.ops.aten.native_layer_norm.default(x, [1, 2, 3], w, b, eps=0.5)

        x = torch.randn(1, 2, 3, dtype=torch.bfloat16, device="cpu")
        w = torch.randn(1, 2, 3, dtype=torch.bfloat16, requires_grad=True, device="cpu")
        b = torch.randn(1, 2, 3, dtype=torch.bfloat16, requires_grad=True, device="cpu")
        out_ref = f(x, w, b)

        from torch._subclasses.fake_tensor import FakeTensorMode

        with enable_python_dispatcher(), FakeTensorMode():
            x = torch.randn(1, 2, 3, dtype=torch.bfloat16, device="cpu")
            w = torch.randn(
                1, 2, 3, dtype=torch.bfloat16, requires_grad=True, device="cpu"
            )
            b = torch.randn(
                1, 2, 3, dtype=torch.bfloat16, requires_grad=True, device="cpu"
            )
            out = f(x, w, b)

        for o_ref, o in zip(out_ref, out):
            self.assertEqual(o_ref.dtype, o.dtype)

    @onlyCUDA
    @unittest.skipIf(not SM70OrLater, "triton")
    def test_rms_norm_decomp_cuda(self, device):
        @torch.compile
        def rms_norm_sinh(a, b, c):
            output = torch.nn.functional.rms_norm(a, b, c)
            return torch.sinh(output)

        normalized_shape_arg = (3, 3, 3)
        input_tensor = torch.randn(3, 3, 3, device=device, requires_grad=True)
        weight_tensor = torch.randn(3, 3, 3, device=device, requires_grad=True)

        def forward_pass_fn():
            return rms_norm_sinh(input_tensor, normalized_shape_arg, weight_tensor)

        model_output, generated_codes = torch._inductor.utils.run_fw_bw_and_get_code(
            forward_pass_fn
        )

        # check RMSNorm was fused with sinh
        self.assertTrue("triton_per_fused__fused_rms_norm_sinh" in generated_codes[0])
        self.assertTrue(
            "triton_per_fused__fused_rms_norm__fused_rms_norm_backward_cosh_mul"
            in generated_codes[1]
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
            return not _is_cia_op(op)
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
            for op in decomposition_table
            if isinstance(op, torch._ops.OpOverload) and self._can_appear_in_trace(op)
        }
        core_decomps = torch._decomp.core_aten_decompositions().keys()
        core_aten_ops = useful_decomps - core_decomps
        self.assertExpected("".join(sorted(op.name() + "\n" for op in core_aten_ops)))

    def test_conv1d_decomposition(self):
        from torch._inductor.decomposition import conv1d_to_conv2d

        def check_case(
            N=2,
            C_in=3,
            C_out=5,
            L=37,
            K=5,
            stride=2,
            padding=3,
            dilation=1,
            groups=1,
            dtype=torch.float32,
            device="cpu",
        ):
            torch.manual_seed(0)
            x = torch.randn(N, C_in, L, dtype=dtype, device=device)
            w = torch.randn(C_out, C_in // groups, K, dtype=dtype, device=device)
            b = torch.randn(C_out, dtype=dtype, device=device)

            ref = torch.ops.aten.conv1d.default(
                x,
                w,
                b,
                stride=[stride],
                padding=[padding],
                dilation=[dilation],
                groups=groups,
            )
            got = conv1d_to_conv2d(
                x,
                w,
                b,
                stride=[stride],
                padding=[padding],
                dilation=[dilation],
                groups=groups,
            )
            self.assertTrue(torch.allclose(ref, got, atol=1e-5, rtol=1e-5))

        # A few cases
        check_case()  # default
        check_case(stride=1, padding=0, K=3)
        check_case(stride=3, padding=4, K=7)
        check_case(dilation=2, padding=6, K=5)  # dilation
        check_case(groups=1, C_in=8, C_out=12)  # groups=1 bigger
        check_case(groups=2, C_in=8, C_out=12)  # grouped conv

    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    @torch._dynamo.config.patch("capture_dynamic_output_shape_ops", True)
    def test_mm_decompose_mm_dde(self):
        def fuzzed_program(
            arg_0,
            arg_1,
            arg_2,
            arg_3,
            arg_4,
            arg_5,
            arg_6,
            arg_7,
            arg_8,
            arg_9,
            arg_10,
            arg_11,
            arg_12,
            arg_13,
            arg_14,
            arg_15,
            arg_16,
            arg_17,
            arg_18,
            sentinel,
        ):
            var_node_6 = (
                arg_0  # size=(9, 9, 9), stride=(81, 9, 1), dtype=float64, device=cuda
            )
            var_node_7 = (
                arg_1  # size=(9, 9, 11), stride=(99, 11, 1), dtype=float64, device=cuda
            )
            var_node_5 = torch.matmul(
                var_node_6.to(torch.float64), var_node_7.to(torch.float64)
            )  # size=(9, 9, 11), stride=(99, 11, 1), dtype=float64, device=cuda
            var_node_9 = torch.full(
                (9, 11, 12), 1.5758497316910556, dtype=torch.float64
            )  # size=(9, 11, 12), stride=(132, 12, 1), dtype=float64, device=cuda
            var_node_10 = (
                arg_2  # size=(9, 12, 8), stride=(96, 8, 1), dtype=float64, device=cuda
            )
            var_node_8 = torch.matmul(
                var_node_9.to(torch.float64), var_node_10.to(torch.float64)
            )  # size=(9, 11, 8), stride=(88, 8, 1), dtype=float64, device=cuda
            var_node_4 = torch.matmul(
                var_node_5.to(torch.float64), var_node_8.to(torch.float64)
            )  # size=(9, 9, 8), stride=(72, 8, 1), dtype=float64, device=cuda
            var_node_13 = arg_3  # size=(9, 8, 13), stride=(104, 13, 1), dtype=float64, device=cuda
            var_node_14 = (
                arg_4  # size=(9, 13, 7), stride=(91, 7, 1), dtype=float64, device=cuda
            )
            var_node_12 = torch.matmul(
                var_node_13.to(torch.float64), var_node_14.to(torch.float64)
            )  # size=(9, 8, 7), stride=(56, 7, 1), dtype=float64, device=cuda
            var_node_15 = arg_5  # size=(9, 7, 16), stride=(112, 16, 1), dtype=float64, device=cuda
            var_node_11 = torch.matmul(
                var_node_12.to(torch.float64), var_node_15.to(torch.float64)
            )  # size=(9, 8, 16), stride=(128, 16, 1), dtype=float64, device=cuda
            var_node_3 = torch.matmul(
                var_node_4.to(torch.float64), var_node_11.to(torch.float64)
            )  # size=(9, 9, 16), stride=(144, 16, 1), dtype=float64, device=cuda
            var_node_17 = arg_6  # size=(9, 16, 12), stride=(192, 12, 1), dtype=float64, device=cuda
            var_node_18 = arg_7  # size=(9, 12, 11), stride=(132, 11, 1), dtype=float64, device=cuda
            var_node_16 = torch.matmul(
                var_node_17.to(torch.float64), var_node_18.to(torch.float64)
            )  # size=(9, 16, 11), stride=(176, 11, 1), dtype=float64, device=cuda
            var_node_2 = torch.matmul(
                var_node_3.to(torch.float64), var_node_16.to(torch.float64)
            )  # size=(9, 9, 11), stride=(99, 11, 1), dtype=float64, device=cuda
            var_node_23 = torch.full(
                (156, 8), -0.5249394453404403, dtype=torch.float64
            )  # size=(156, 8), stride=(8, 1), dtype=float64, device=cuda
            var_node_24 = torch.full(
                (8, 9), 0.9331226188585692, dtype=torch.float64
            )  # size=(8, 9), stride=(9, 1), dtype=float64, device=cuda
            var_node_22 = torch.matmul(
                var_node_23.to(torch.float64), var_node_24.to(torch.float64)
            )  # size=(156, 9), stride=(9, 1), dtype=float64, device=cuda
            var_node_26 = torch.full(
                (9, 13), -0.9276381954691514, dtype=torch.float64
            )  # size=(9, 13), stride=(13, 1), dtype=float64, device=cuda
            var_node_27 = torch.full(
                (13, 16), 0.024752238943232543, dtype=torch.float64
            )  # size=(13, 16), stride=(16, 1), dtype=float64, device=cuda
            var_node_25 = torch.matmul(
                var_node_26.to(torch.float64), var_node_27.to(torch.float64)
            )  # size=(9, 16), stride=(16, 1), dtype=float64, device=cuda
            var_node_21 = torch.matmul(
                var_node_22.to(torch.float64), var_node_25.to(torch.float64)
            )  # size=(156, 16), stride=(16, 1), dtype=float64, device=cuda
            var_node_29 = arg_8
            _x_nz = torch.zeros(
                (9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                dtype=torch.bool,
                device=var_node_29.device,
            )
            _x_nz_flat = _x_nz.reshape(-1)
            _x_nz_flat[:9] = True
            var_node_28 = torch.nonzero(
                _x_nz
            )  # size=(9, 11), stride=(11, 1), dtype=int64, device=cuda
            var_node_20 = torch.nn.functional.embedding(
                torch.clamp(var_node_28.to(torch.int64), 0, var_node_21.size(0) - 1),
                var_node_21,
            )  # size=(9, 11, 16), stride=(176, 16, 1), dtype=float64, device=cuda
            var_node_33 = torch.full(
                (9, 16, 5), 1.0707914920634904, dtype=torch.float64
            )  # size=(9, 16, 5), stride=(80, 5, 1), dtype=float64, device=cuda
            var_node_34 = torch.full(
                (9, 5, 10), -0.44934093079047227, dtype=torch.float64
            )  # size=(9, 5, 10), stride=(50, 10, 1), dtype=float64, device=cuda
            var_node_32 = torch.matmul(
                var_node_33.to(torch.float64), var_node_34.to(torch.float64)
            )  # size=(9, 16, 10), stride=(160, 10, 1), dtype=float64, device=cuda
            var_node_36 = (
                arg_9  # size=(9, 10, 1), stride=(10, 1, 1), dtype=float64, device=cuda
            )
            var_node_37 = torch.full(
                (9, 1, 11), -1.874293687140311, dtype=torch.float64
            )  # size=(9, 1, 11), stride=(11, 11, 1), dtype=float64, device=cuda
            var_node_35 = torch.matmul(
                var_node_36.to(torch.float64), var_node_37.to(torch.float64)
            )  # size=(9, 10, 11), stride=(110, 11, 1), dtype=float64, device=cuda
            var_node_31 = torch.matmul(
                var_node_32.to(torch.float64), var_node_35.to(torch.float64)
            )  # size=(9, 16, 11), stride=(176, 11, 1), dtype=float64, device=cuda
            var_node_40 = torch.full(
                (990, 2), 0.4084376380351558, dtype=torch.float64
            )  # size=(990, 2), stride=(2, 1), dtype=float64, device=cuda
            var_node_41 = torch.full(
                (2,), 0.982671965550022, dtype=torch.float64
            )  # size=(2,), stride=(1,), dtype=float64, device=cuda
            var_node_39 = torch.matmul(
                var_node_40.to(torch.float64), var_node_41.to(torch.float64)
            )  # size=(990,), stride=(1,), dtype=float64, device=cuda
            var_node_38 = torch.reshape(
                var_node_39, [9, 11, 10]
            )  # size=(9, 11, 10), stride=(110, 10, 1), dtype=float64, device=cuda
            var_node_30 = torch.matmul(
                var_node_31.to(torch.float64), var_node_38.to(torch.float64)
            )  # size=(9, 16, 10), stride=(160, 10, 1), dtype=float64, device=cuda
            var_node_19 = torch.matmul(
                var_node_20.to(torch.float64), var_node_30.to(torch.float64)
            )  # size=(9, 11, 10), stride=(110, 10, 1), dtype=float64, device=cuda
            var_node_1 = torch.matmul(
                var_node_2.to(torch.float64), var_node_19.to(torch.float64)
            )  # size=(9, 9, 10), stride=(90, 10, 1), dtype=float64, device=cuda
            var_node_47 = arg_10  # size=(9, 10, 15), stride=(150, 15, 1), dtype=float64, device=cuda
            var_node_48 = torch.full(
                (9, 15, 2), -0.3349339402390618, dtype=torch.float64
            )  # size=(9, 15, 2), stride=(30, 2, 1), dtype=float64, device=cuda
            var_node_46 = torch.matmul(
                var_node_47.to(torch.float64), var_node_48.to(torch.float64)
            )  # size=(9, 10, 2), stride=(20, 2, 1), dtype=float64, device=cuda
            var_node_50 = (
                arg_11  # size=(9, 2, 7), stride=(14, 7, 1), dtype=float64, device=cuda
            )
            var_node_51 = (
                arg_12  # size=(9, 7, 2), stride=(14, 2, 1), dtype=float64, device=cuda
            )
            var_node_49 = torch.matmul(
                var_node_50.to(torch.float64), var_node_51.to(torch.float64)
            )  # size=(9, 2, 2), stride=(4, 2, 1), dtype=float64, device=cuda
            var_node_45 = torch.matmul(
                var_node_46.to(torch.float64), var_node_49.to(torch.float64)
            )  # size=(9, 10, 2), stride=(20, 2, 1), dtype=float64, device=cuda
            var_node_52 = torch.full(
                (9, 2, 1), -0.4046675639434615, dtype=torch.float64
            )  # size=(9, 2, 1), stride=(2, 1, 1), dtype=float64, device=cuda
            var_node_44 = torch.matmul(
                var_node_45.to(torch.float64), var_node_52.to(torch.float64)
            )  # size=(9, 10, 1), stride=(10, 1, 1), dtype=float64, device=cuda
            var_node_56 = (
                arg_13  # size=(9, 1, 1), stride=(1, 1, 1), dtype=float64, device=cuda
            )
            var_node_55 = torch.nn.functional.rms_norm(
                var_node_56.to(torch.float64), (1,)
            )  # size=(9, 1, 1), stride=(1, 1, 1), dtype=float64, device=cuda
            var_node_57 = torch.full(
                (9, 1, 8), 0.17877664640931384, dtype=torch.float64
            )  # size=(9, 1, 8), stride=(8, 8, 1), dtype=float64, device=cuda
            var_node_54 = torch.matmul(
                var_node_55.to(torch.float64), var_node_57.to(torch.float64)
            )  # size=(9, 1, 8), stride=(8, 8, 1), dtype=float64, device=cuda
            var_node_60 = arg_14  # size=(9, 8, 10), stride=(80, 10, 1), dtype=float64, device=cuda
            var_node_61 = torch.full(
                (9, 10, 6), 0.43614806380221494, dtype=torch.float64
            )  # size=(9, 10, 6), stride=(60, 6, 1), dtype=float64, device=cuda
            var_node_59 = torch.matmul(
                var_node_60.to(torch.float64), var_node_61.to(torch.float64)
            )  # size=(9, 8, 6), stride=(48, 6, 1), dtype=float64, device=cuda
            var_node_63 = (
                arg_15  # size=(9, 6, 3), stride=(18, 3, 1), dtype=float64, device=cuda
            )
            var_node_64 = torch.full(
                (9, 3, 8), -0.042774422041922854, dtype=torch.float64
            )  # size=(9, 3, 8), stride=(24, 8, 1), dtype=float64, device=cuda
            var_node_62 = torch.matmul(
                var_node_63.to(torch.float64), var_node_64.to(torch.float64)
            )  # size=(9, 6, 8), stride=(48, 8, 1), dtype=float64, device=cuda
            var_node_58 = torch.matmul(
                var_node_59.to(torch.float64), var_node_62.to(torch.float64)
            )  # size=(9, 8, 8), stride=(64, 8, 1), dtype=float64, device=cuda
            var_node_53 = torch.matmul(
                var_node_54.to(torch.float64), var_node_58.to(torch.float64)
            )  # size=(9, 1, 8), stride=(8, 8, 1), dtype=float64, device=cuda
            var_node_43 = torch.matmul(
                var_node_44.to(torch.float64), var_node_53.to(torch.float64)
            )  # size=(9, 10, 8), stride=(80, 8, 1), dtype=float64, device=cuda
            var_node_68 = arg_16  # size=(9, 8, 16), stride=(128, 16, 1), dtype=float64, device=cuda
            var_node_70 = torch.full(
                (9, 16, 15), 0.24947808634496438, dtype=torch.float64
            )  # size=(9, 16, 15), stride=(240, 15, 1), dtype=float64, device=cuda
            var_node_71 = torch.full(
                (9, 15, 7), -0.09035245509773453, dtype=torch.float64
            )  # size=(9, 15, 7), stride=(105, 7, 1), dtype=float64, device=cuda
            var_node_69 = torch.matmul(
                var_node_70.to(torch.float64), var_node_71.to(torch.float64)
            )  # size=(9, 16, 7), stride=(112, 7, 1), dtype=float64, device=cuda
            var_node_67 = torch.matmul(
                var_node_68.to(torch.float64), var_node_69.to(torch.float64)
            )  # size=(9, 8, 7), stride=(56, 7, 1), dtype=float64, device=cuda
            var_node_74 = torch.full(
                (9, 7, 1), 0.05671950481832341, dtype=torch.float64
            )  # size=(9, 7, 1), stride=(7, 1, 1), dtype=float64, device=cuda
            var_node_73 = torch.nn.functional.gelu(
                var_node_74
            )  # size=(9, 7, 1), stride=(7, 1, 1), dtype=float64, device=cuda
            var_node_76 = torch.full(
                (9, 1, 2), -0.019912810353597852, dtype=torch.float64
            )  # size=(9, 1, 2), stride=(2, 2, 1), dtype=float64, device=cuda
            var_node_77 = (
                arg_17  # size=(9, 2, 7), stride=(14, 7, 1), dtype=float64, device=cuda
            )
            var_node_75 = torch.matmul(
                var_node_76.to(torch.float64), var_node_77.to(torch.float64)
            )  # size=(9, 1, 7), stride=(7, 7, 1), dtype=float64, device=cuda
            var_node_72 = torch.matmul(
                var_node_73.to(torch.float64), var_node_75.to(torch.float64)
            )  # size=(9, 7, 7), stride=(49, 7, 1), dtype=float64, device=cuda
            var_node_66 = torch.matmul(
                var_node_67.to(torch.float64), var_node_72.to(torch.float64)
            )  # size=(9, 8, 7), stride=(56, 7, 1), dtype=float64, device=cuda
            var_node_78 = arg_18  # size=(9, 7, 13), stride=(91, 13, 1), dtype=float64, device=cuda
            var_node_65 = torch.matmul(
                var_node_66.to(torch.float64), var_node_78.to(torch.float64)
            )  # size=(9, 8, 13), stride=(104, 13, 1), dtype=float64, device=cuda
            var_node_42 = torch.matmul(
                var_node_43.to(torch.float64), var_node_65.to(torch.float64)
            )  # size=(9, 10, 13), stride=(130, 13, 1), dtype=float64, device=cuda
            var_node_0 = torch.matmul(
                var_node_1.to(torch.float64), var_node_42.to(torch.float64)
            )  # size=(9, 9, 13), stride=(117, 13, 1), dtype=float64, device=cuda
            # Ensure gradient computation by multiplying with sentinel and taking real part
            result = var_node_0 * sentinel
            if result.is_complex():
                result = result.real
            return result

        # Sentinel tensor to ensure gradient computation
        sentinel = torch.tensor(1.0, requires_grad=True)

        arg_0 = torch.as_strided(
            torch.randn(729).to(torch.float64), (9, 9, 9), (81, 9, 1)
        )
        arg_1 = torch.as_strided(
            torch.randn(891).to(torch.float64), (9, 9, 11), (99, 11, 1)
        )
        arg_2 = torch.as_strided(
            torch.randn(864).to(torch.float64), (9, 12, 8), (96, 8, 1)
        )
        arg_3 = torch.as_strided(
            torch.randn(936).to(torch.float64), (9, 8, 13), (104, 13, 1)
        )
        arg_4 = torch.as_strided(
            torch.randn(819).to(torch.float64), (9, 13, 7), (91, 7, 1)
        )
        arg_5 = torch.as_strided(
            torch.randn(1008).to(torch.float64), (9, 7, 16), (112, 16, 1)
        )
        arg_6 = torch.as_strided(
            torch.randn(1728).to(torch.float64), (9, 16, 12), (192, 12, 1)
        )
        arg_7 = torch.as_strided(
            torch.randn(1188).to(torch.float64), (9, 12, 11), (132, 11, 1)
        )
        arg_8 = torch.as_strided(
            torch.randint(0, 2, (1,), dtype=torch.int8).bool(),
            (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
            (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
        )
        arg_9 = torch.as_strided(
            torch.randn(90).to(torch.float64), (9, 10, 1), (10, 1, 1)
        )
        arg_10 = torch.as_strided(
            torch.randn(1350).to(torch.float64), (9, 10, 15), (150, 15, 1)
        )
        arg_11 = torch.as_strided(
            torch.randn(126).to(torch.float64), (9, 2, 7), (14, 7, 1)
        )
        arg_12 = torch.as_strided(
            torch.randn(126).to(torch.float64), (9, 7, 2), (14, 2, 1)
        )
        arg_13 = torch.as_strided(
            torch.randn(9).to(torch.float64), (9, 1, 1), (1, 1, 1)
        )
        arg_14 = torch.as_strided(
            torch.randn(720).to(torch.float64), (9, 8, 10), (80, 10, 1)
        )
        arg_15 = torch.as_strided(
            torch.randn(162).to(torch.float64), (9, 6, 3), (18, 3, 1)
        )
        arg_16 = torch.as_strided(
            torch.randn(1152).to(torch.float64), (9, 8, 16), (128, 16, 1)
        )
        arg_17 = torch.as_strided(
            torch.randn(126).to(torch.float64), (9, 2, 7), (14, 7, 1)
        )
        arg_18 = torch.as_strided(
            torch.randn(819).to(torch.float64), (9, 7, 13), (91, 13, 1)
        )

        args = (
            arg_0,
            arg_1,
            arg_2,
            arg_3,
            arg_4,
            arg_5,
            arg_6,
            arg_7,
            arg_8,
            arg_9,
            arg_10,
            arg_11,
            arg_12,
            arg_13,
            arg_14,
            arg_15,
            arg_16,
            arg_17,
            arg_18,
        ) + (sentinel,)
        result_original = fuzzed_program(*args)
        compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
        result_compiled = compiled_program(*args)

        # Both should succeed without NameError
        self.assertTrue(
            torch.allclose(result_original, result_compiled, rtol=1e-5, atol=1e-5)
        )


if __name__ == "__main__":
    run_tests()
