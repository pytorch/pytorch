# Owner(s): ["module: primTorch", "module: decompositions"]

from collections import defaultdict
from torch import Tensor
import torch.autograd
from torch._decomp import decomposition_table
from torch.utils._python_dispatch import TorchDispatchMode

from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten
from torch.utils._mode_utils import no_dispatch
from torch.testing._internal.common_utils import (
    is_iterable_of_tensors,
    TestCase,
    skipIfCrossRef,
    suppress_warnings,
    TEST_WITH_ASAN,
    run_tests,
    skipIfSlowGradcheckEnv,
    skipIfTorchDynamo,
)
from torch.testing._internal.common_device_type import (
    onlyNativeDeviceTypes,
    ops,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_methods_invocations import op_db
from torch._dispatch.python import enable_python_dispatcher

import itertools
import functools
from functools import partial
import unittest

aten = torch.ops.aten


# TODO: this isn't going to work with non-aten namespaces
def overload_to_aten_name(overload):
    return overload._schema.name.split("::")[1]


# All operators that can have decomp tests
decomposition_names = {overload_to_aten_name(k) for k in decomposition_table}
_decomp_test_ops = [
    op
    for op in op_db
    if op.aten_name in decomposition_names
    or op.aten_backward_name in decomposition_names
]


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
            _as_tuple(result), primals, _as_tuple(cotangents), create_graph=False
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
        (torch.bfloat16, torch.ops.aten.linalg_vector_norm.default): 1e-6,
        (torch.float16, torch.ops.aten.linalg_vector_norm.default): 1e-6,
        (torch.float16, torch.ops.aten.nll_loss_forward.default): 1e-2,
        (torch.bfloat16, torch.ops.aten.nll_loss_forward.default): 1e-1,
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
        orig.dtype, decomp.dtype, f"Operation: {op}, orig.dtype: {orig.dtype}, decomp.dtype: {decomp.dtype}, {args}, {kwargs}")
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
        (torch.float32, torch.ops.aten.grid_sampler_2d.default) : (7e-6, 3e-5),
        # Exceeds tolerances on CUDA, likely due to fma
        (torch.float32, torch.ops.aten.mv.default) : (1e-5, 3e-5),
        (torch.float64, torch.ops.aten.upsample_bicubic2d.vec) : (1e-5, 5e-4),
        (torch.float64, torch.ops.aten.upsample_bicubic2d.default) : (1e-5, 5e-4),
    }
    if (test_dtype, op) in tol_table:
        rtol, atol = tol_table[(decomp.dtype, op)]
    else:
        rtol, atol = _getDefaultRtolAndAtol(orig.dtype, decomp.dtype)
    test_case.assertEqual(orig, decomp, rtol=rtol, atol=atol, msg=f"{op.__name__}\nargs = {args}\nkwargs = {kwargs}")


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
            # TODO: Remove the following hack for namedtuples
            result = tuple(result)
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
    elif (isinstance(x, torch.dtype)
          and x in [torch.float16, torch.bfloat16, torch.float]):
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
    ("cuda", torch.float16, "nn.functional.dropout"),
    ("cuda", torch.bfloat16, "nn.functional.dropout"),
    ("cuda", torch.float64, "nn.functional.dropout"),
    ("cuda", torch.float32, "nn.functional.dropout"),
    (None, None, "special.ndtr"),  # aten.special_ndtr was not decomposed
    (None, None, "new_empty"),
    (None, None, "empty_like"),
    (None, None, "empty"),

    # CompositeAutogradImplicit
    # See https://github.com/pytorch/pytorch/issues/81669
    (None, None, "nn.functional.relu6"),
    (None, None, "meshgrid"),
    # diag was not decomposed (it just registers a decomp for diag_out, torch.diag is CompImplicit)
    (None, None, "diag"),
}

CROSS_REF_BACKWARD_EXCLUDE_SET = {
    # Decomposed backward formula is not as precise
    ("cuda", torch.float16, "nn.functional.embedding"),
    ("cuda", torch.bfloat16, "nn.functional.embedding"),
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
            return any([
                t.is_sparse_csr, t.is_sparse, t.is_mkldnn, t.is_quantized,
                t.is_nested, torch._is_functional_tensor(t),
            ])
        elif torch.overrides.is_tensor_like(t):
            # Decompositions will generally change the behavior of Tensor-like
            # subclasses, so bypass tests in this case too
            return True
        else:
            return False

    flat_args, _ = tree_flatten(args)
    flat_kwargs, _ = tree_flatten(kwargs)
    return any(test_unsupported(x) for x in itertools.chain(flat_args, flat_kwargs))


@skipIfSlowGradcheckEnv
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
    @onlyNativeDeviceTypes
    @skipIfCrossRef
    @suppress_warnings
    @ops(op_db)
    def test_comprehensive(self, device, dtype, op):
        self.do_cross_ref(device, dtype, op, run_all=True)

    @skipIfTorchDynamo("Test does not work with TorchDynamo")
    def do_cross_ref(self, device, dtype, op, *, run_all):
        test_keys = [
            (torch.device(device).type, dtype, op.name),
            (None, dtype, op.name),
            (None, None, op.name),
        ]
        if any(key in CROSS_REF_EXCLUDE_SET for key in test_keys):
            self.skipTest(f"{op.name} in {dtype} not supported")

        skip_decomp_vjp = any(key in CROSS_REF_BACKWARD_EXCLUDE_SET for key in test_keys)
        test_dtype = dtype

        # We check the correctness of each decomposition right after running it.
        # So, when we encounter a decomposition, we run the function normally, and
        # then run the decomposition, and ensure they're identical.
        called = set()
        decomposed = set()

        saved_precision = self.precision
        saved_rel_tol = self.rel_tol
        test_case = self

        class DecompCrossRefMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                with no_dispatch():
                    return self._torch_dispatch(func, types, args, kwargs)

            def _torch_dispatch(self, func, types, args=(), kwargs=None):
                test_case.precision = saved_precision
                test_case.rel_tol = saved_rel_tol

                called.add(func)
                all_called[func] += 1

                # Stuff we shouldn't bother testing
                # (TODO: remove detach from the decomp table?)
                if func not in decomposition_table or func in [
                    torch.ops.aten.detach.default,
                    # non-deterministic ops
                    torch.ops.aten.empty.memory_format,
                    torch.ops.aten.empty_like.default,
                    torch.ops.aten.new_empty.default
                ] or any_unsupported(args, kwargs):
                    return func(*args, **kwargs)

                decomposed.add(func)
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

                decomposition = decomposition_table[func]

                do_relative_check = test_dtype in [torch.float16, torch.bfloat16]
                real_out_unflat = func(*args, **kwargs)
                real_out, _ = tree_flatten(real_out_unflat)
                decomp_out, _ = tree_flatten(decomposition(*args, **kwargs))
                assert len(real_out) == len(decomp_out)

                if do_relative_check:
                    upcast = partial(upcast_tensor, dtype=torch.float64)
                    real_out_double, _ = tree_flatten(
                        func(*tree_map(upcast, args), **tree_map(upcast, kwargs))
                    )
                    for i, orig, decomp, ref in zip(range(len(real_out)), real_out, decomp_out, real_out_double):
                        if not isinstance(orig, torch.Tensor):
                            assert type(orig) == type(decomp)
                            assert orig == decomp
                            continue
                        op_assert_ref(test_case, func, test_dtype, i, orig, decomp, ref, args, kwargs)
                else:
                    for orig, decomp in zip(real_out, decomp_out):
                        if not isinstance(orig, torch.Tensor):
                            assert type(orig) == type(decomp)
                            assert orig == decomp
                            continue
                        op_assert_equal(test_case, func, test_dtype, orig, decomp, args, kwargs)

                return real_out_unflat

        requires_grad = (
            op.supports_autograd
            and dtype in op.supported_backward_dtypes(torch.device(device).type)
            # TODO: OpInfo really ought to error out for this case, but it's
            # not exercised in test_ops_gradients atm.  The problem is not
            # complex32 per-se (which is supported by data movement only ops)
            # but that when we do backwards we expect other ops like add to work
            and not dtype == torch.complex32
        )
        samples = op.sample_inputs(device, test_dtype, requires_grad=requires_grad)

        def check_decomposed(aten_name):
            self.assertTrue(
                any(overload_to_aten_name(c) == aten_name for c in decomposed),
                msg=(f"aten.{aten_name} was not decomposed, saw calls for: "
                     f"{', '.join(map(str, list(called)))}. If your op is  "
                     f"CompositeImplicitAutograd you should skip this test "
                     "by updating CROSS_REF_EXCLUDE_SET.")
            )

        aten_name = op.decomp_aten_name or op.aten_name

        func = op.get_op()
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
                decomposed.clear()
                with DecompCrossRefMode(), enable_python_dispatcher():
                    decomp_out, decomp_vjp_fn = ref_vjp_no_create(fn, *primals)
                if aten_name in decomposition_names:
                    check_decomposed(aten_name)

                if not skip_decomp_vjp and (op.aten_backward_name in decomposition_names or run_all):
                    cotangents = tree_map(lambda x: torch.randn_like(x), decomp_out)

                    decomposed.clear()
                    with DecompCrossRefMode(), enable_python_dispatcher():
                        decomp_vjp_fn(cotangents)
                    if not run_all:
                        check_decomposed(op.aten_backward_name)

            elif aten_name in decomposition_names or run_all:
                args = [sample_input.input] + list(sample_input.args)
                kwargs = sample_input.kwargs
                decomposed.clear()
                with DecompCrossRefMode(), enable_python_dispatcher():
                    func(*args, **kwargs)
                if not run_all:
                    check_decomposed(aten_name)
            else:
                assert op.supports_autograd
                self.skipTest(
                    "only backwards is decomposed, but dtype doesn't support AD"
                )

instantiate_device_type_tests(TestDecomp, globals())

class DecompContiguousTests(TestCase):
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

instantiate_device_type_tests(DecompContiguousTests, globals())


if __name__ == "__main__":
    run_tests()
