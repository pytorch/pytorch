# Owner(s): ["module: unknown"]
import contextlib
import copy
import inspect
import itertools
import os
import re
import unittest
import warnings
from collections import defaultdict
from collections.abc import Sequence
from functools import partial
from importlib import import_module

import torch
import torch._prims as prims
import torch.utils._pytree as pytree
from torch._prims.context import TorchRefsMode
from torch._prims_common.wrappers import _maybe_remove_out_wrapper
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch._subclasses.fake_utils import outputs_alias_inputs
from torch.testing import make_tensor
from torch.testing._internal import composite_compliance, opinfo
from torch.testing._internal.common_cuda import with_tf32_off
from torch.testing._internal.common_device_type import (
    deviceCountAtLeast,
    instantiate_device_type_tests,
    onlyCPU,
    onlyCUDA,
    onlyNativeDeviceTypesAnd,
    onlyOn,
    OpDTypes,
    ops,
    skipCUDAIfNotRocm,
    skipMeta,
    skipXPU,
)
from torch.testing._internal.common_dtype import (
    all_types_and_complex_and,
    floating_and_complex_types_and,
    integral_types_and,
)
from torch.testing._internal.common_methods_invocations import (
    BinaryUfuncInfo,
    op_db,
    ops_and_refs,
    python_ref_db,
    ReductionOpInfo,
    ReductionPythonRefInfo,
    skip,
    skipOps,
    SpectralFuncInfo,
    UnaryUfuncInfo,
    xfail,
)
from torch.testing._internal.common_utils import (
    clone_input_helper,
    first_sample,
    IS_CI,
    IS_FBCODE,
    is_iterable_of_tensors,
    IS_SANDCASTLE,
    noncontiguous_like,
    parametrize,
    run_tests,
    set_default_dtype,
    skipIfTorchDynamo,
    skipIfTorchInductor,
    suppress_warnings,
    TEST_WITH_ROCM,
    TEST_WITH_TORCHDYNAMO,
    TEST_WITH_TORCHINDUCTOR,
    TestCase,
    unMarkDynamoStrictTest,
)
from torch.testing._internal.inductor_utils import maybe_skip_size_asserts
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map


assert torch.get_default_dtype() == torch.float32

# variant testing is only done with torch.float and torch.cfloat to avoid
#   excessive test times and maximize signal to noise ratio
_variant_ops = partial(
    ops, dtypes=OpDTypes.supported, allowed_dtypes=(torch.float, torch.cfloat)
)

# Get names of all the operators which have ref in their entry in OpInfo (testing infra)
#   except for elementwise unary operators (separately implemented in test/test_unary_ufuncs.py),
#   elementwise binary operators (separately implemented in test_binary_ufuncs.py),
#   reduction operations (separately implemented in test_reductions.py),
#   and Spectral Functions (separately implemented for only 1D as of now, in test/test_spectral_ops.py)
_ref_test_ops = tuple(
    filter(
        lambda op: not isinstance(
            op, (UnaryUfuncInfo, ReductionOpInfo, SpectralFuncInfo, BinaryUfuncInfo)
        )
        and op.ref is not None,
        op_db,
    )
)


def reduction_dtype_filter(op):
    if (
        not isinstance(op, ReductionPythonRefInfo)
        or not op.supports_out
        or torch.int16 not in op.dtypes
    ):
        return False
    return "dtype" in inspect.getfullargspec(op.op).kwonlyargs


def has_reduction_tag(op):
    """Check if an op has the reduction tag."""
    if not hasattr(torch.ops.aten, op.name):
        return False
    aten_op = getattr(torch.ops.aten, op.name)
    if not hasattr(aten_op, "default"):
        return False
    return torch.Tag.reduction in aten_op.default.tags


# Create a list of operators that are a subset of _ref_test_ops but don't have a
# numpy ref to compare them too, If both CPU and CUDA are compared to numpy
# then they do not need to be compared to each other
_ops_and_refs_with_no_numpy_ref = [op for op in ops_and_refs if op.ref is None]

aten = torch.ops.aten

meta_consistency_out_dtype_mismatch_xfails = {
    xfail("all"),
    xfail("amax"),
    xfail("amin"),
    xfail("aminmax"),
    xfail("any"),
    xfail("bucketize"),
    xfail("conj_physical"),
    xfail("cross"),
    xfail("cummax"),
    xfail("cummin"),
    xfail("diag"),
    xfail("fft.ihfft2"),
    xfail("fft.ihfftn"),
    xfail("frexp"),
    xfail("geqrf"),
    xfail("heaviside"),
    xfail("histc"),
    xfail("index_add"),
    xfail("index_copy"),
    xfail("index_select"),
    xfail("isin"),
    xfail("kthvalue"),
    xfail("lerp"),
    xfail("linalg.cross"),
    xfail("linalg.eigh"),
    xfail("linalg.eigvalsh"),
    xfail("linalg.ldl_factor"),
    xfail("linalg.ldl_factor_ex"),
    xfail("linalg.ldl_solve"),
    xfail("linalg.lu"),
    xfail("linalg.lu_factor"),
    xfail("linalg.lu_factor_ex"),
    xfail("linalg.lu_solve"),
    xfail("linalg.qr"),
    xfail("linalg.slogdet"),
    xfail("linalg.solve"),
    xfail("linalg.solve_ex"),
    xfail("linalg.solve_triangular"),
    xfail("logcumsumexp"),
    xfail("lu_solve"),
    xfail("lu_unpack"),
    xfail("mode"),
    xfail("msort"),
    xfail("multinomial"),
    xfail("nan_to_num"),
    xfail("native_batch_norm"),
    xfail("neg"),
    xfail("nn.functional.avg_pool3d"),
    xfail("nn.functional.gelu"),
    xfail("nn.functional.hardshrink"),
    xfail("nn.functional.logsigmoid"),
    xfail("nn.functional.softplus"),
    xfail("nn.functional.softshrink"),
    xfail("ormqr"),
    xfail("qr"),
    xfail("renorm"),
    xfail("round"),
    xfail("round", "decimals_0"),
    xfail("scatter_reduce", "amax"),
    xfail("scatter_reduce", "amin"),
    xfail("scatter_reduce", "mean"),
    xfail("scatter_reduce", "prod"),
    xfail("scatter_reduce", "sum"),
    xfail("searchsorted"),
    xfail("slice_scatter"),
    xfail("softmax"),
    xfail("sort"),
    xfail("sparse.sampled_addmm"),
    xfail("take"),
    xfail("tril"),
    xfail("triu"),
    xfail("unfold_copy"),
    # Output has dynamic shape.
    # Does not have a meta kernel implementation.
    skip("linalg.lstsq"),
}


# Tests that apply to all operators and aren't related to any particular
#   system
@unMarkDynamoStrictTest
class TestCommon(TestCase):
    exact_dtype = True

    # Verifies, on teardown, that no OpInfo is still using dynamic dtypes in CI
    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

        if IS_CI:
            err_msg = (
                "The operator(s) below is(are) using dynamic_dtypes in the OpInfo entries."
                "This is OK for testing, but be sure to set the dtypes manually before landing your PR!"
            )
            # Assure no opinfo entry has dynamic_dtypes
            filtered_ops = list(filter(opinfo.utils.is_dynamic_dtype_set, op_db))
            for op in filtered_ops:
                fmt_str = opinfo.utils.str_format_dynamic_dtype(op)
                err_msg += "\n" + fmt_str

            assert len(filtered_ops) == 0, err_msg

    # Validates that each OpInfo works correctly on different CUDA devices
    @onlyOn(["cuda", "xpu"])
    @deviceCountAtLeast(2)
    @ops(op_db, allowed_dtypes=(torch.float32, torch.long))
    def test_multiple_devices(self, devices, dtype, op):
        for cuda_device_str in devices:
            cuda_device = torch.device(cuda_device_str)
            # NOTE: only tests on first sample
            samples = op.sample_inputs(cuda_device, dtype)
            sample = first_sample(self, samples)
            result = op(sample.input, *sample.args, **sample.kwargs)

            if isinstance(result, torch.Tensor):
                self.assertTrue(result.device == cuda_device)
            elif is_iterable_of_tensors(result):
                self.assertTrue(all(t.device == cuda_device for t in result))
            else:
                self.skipTest(
                    "Skipped! Only supports single tensor or iterable of tensor outputs."
                )

    def test_pointwise_tag_coverage(self):
        pytorch_dir = os.path.abspath(__file__ + "/../../")
        files = [
            "aten/src/ATen/native/UnaryOps.cpp",
            "aten/src/ATen/native/BinaryOps.cpp",
            "aten/src/ATen/native/PointwiseOps.cpp",
            "aten/src/ATen/native/TensorCompare.cpp",
        ]

        allowed_functions = (
            # reduction version of these operators
            "aten.max.default",
            "aten.max.dim",
            "aten.max.dim_max",
            "aten.max.names_dim",
            "aten.max.names_dim_max",
            "aten.max.unary_out",
            "aten.min.default",
            "aten.min.dim",
            "aten.min.dim_min",
            "aten.min.names_dim",
            "aten.min.names_dim_min",
            "aten.min.unary_out",
            # not pointwise
            "aten.isin.Tensor_Tensor",
            "aten.isin.Tensor_Tensor_out",
            "aten.isin.Tensor_Scalar",
            "aten.isin.Tensor_Scalar_out",
            "aten.isin.Scalar_Tensor",
            "aten.isin.Scalar_Tensor_out",
            "aten.mode.default",
            "aten.mode.dimname",
            "aten.mode.dimname_out",
            "aten.mode.values",
        )

        regex = re.compile(r"DEFINE_DISPATCH\(.*_stub")

        def get_opoverloadpacket_from_dispatch(kernel):
            if hasattr(torch.ops.aten, kernel):
                return kernel
            if hasattr(torch.ops.aten, f"__{kernel}__"):
                return f"__{kernel}__"
            if hasattr(torch.ops.aten, f"special_{kernel}"):
                return f"special_{kernel}"
            if "_" in kernel:
                kernel_split = kernel.split("_")
                new_kernel = "_".join(kernel_split[:-1])
                if hasattr(torch.ops.aten, new_kernel):
                    return new_kernel

            # could not find op from kernel dispatch string
            self.assertTrue(False)

        for file_name in files:
            with open(os.path.join(pytorch_dir, file_name)) as f:
                lines = f.read()
                matches = regex.findall(lines)
                for match in matches:
                    kernel = match[len("DEFINE_DISPATCH(") : -len("_stub")]

                    # no op definition for it, but defined with DEFINE_DISPATCH ?
                    if kernel == "trigamma":
                        continue

                    kernel = get_opoverloadpacket_from_dispatch(kernel)
                    overloadpacket = getattr(torch.ops.aten, kernel)

                    for overload_name in overloadpacket.overloads():
                        overload = getattr(overloadpacket, overload_name)

                        if not torch._C._dispatch_has_kernel(overload.name()):
                            continue

                        # TODO: tags are not propagated to generated overload,
                        # and there's no way of specifying them
                        if torch.Tag.generated in overload.tags:
                            continue

                        if str(overload) in allowed_functions:
                            continue

                        self.assertTrue(torch.Tag.pointwise in overload.tags)

    def test_reduction_tag_coverage(self):
        """Test that operators with reduction tag are from reduction operator files."""
        pytorch_dir = os.path.abspath(__file__ + "/../../")
        files = [
            "aten/src/ATen/native/ReduceOps.cpp",
            "aten/src/ATen/native/ReduceAllOps.h",
        ]

        # Operators that are not pure reduction but have reduction overloads
        allowed_functions = (
            # min/max have both elementwise (binary) and reduction versions
            "aten.min.other",
            "aten.min.out",
            "aten.max.other",
            "aten.max.out",
        )

        regex = re.compile(r"DEFINE_DISPATCH\(.*_stub")

        def get_opoverloadpacket_from_dispatch(kernel):
            # Skip cumulative operations - they're in ReduceOps.cpp but aren't reductions
            if kernel in ("cumsum", "cumprod", "logcumsumexp", "xor_sum"):
                return None

            # Special mappings for ambiguous kernel names
            if kernel == "and":
                return "all"
            if kernel == "or":
                return "any"

            if hasattr(torch.ops.aten, kernel):
                return kernel
            if hasattr(torch.ops.aten, f"__{kernel}__"):
                return f"__{kernel}__"
            if hasattr(torch.ops.aten, f"special_{kernel}"):
                return f"special_{kernel}"
            if "_" in kernel:
                kernel_split = kernel.split("_")
                new_kernel = "_".join(kernel_split[:-1])
                if hasattr(torch.ops.aten, new_kernel):
                    return new_kernel

            # could not find op from kernel dispatch string
            return None

        for file_name in files:
            file_path = os.path.join(pytorch_dir, file_name)
            if not os.path.exists(file_path):
                continue

            with open(file_path) as f:
                lines = f.read()
                matches = regex.findall(lines)
                for match in matches:
                    kernel = match[len("DEFINE_DISPATCH(") : -len("_stub")]

                    kernel = get_opoverloadpacket_from_dispatch(kernel)
                    if kernel is None:
                        continue

                    overloadpacket = getattr(torch.ops.aten, kernel)

                    for overload_name in overloadpacket.overloads():
                        overload = getattr(overloadpacket, overload_name)

                        if not torch._C._dispatch_has_kernel(overload.name()):
                            continue

                        # TODO: tags are not propagated to generated overload,
                        # and there's no way of specifying them
                        if torch.Tag.generated in overload.tags:
                            continue

                        if str(overload) in allowed_functions:
                            continue

                        self.assertTrue(
                            torch.Tag.reduction in overload.tags,
                            f"{overload} should have reduction tag",
                        )

    @ops([op for op in op_db if has_reduction_tag(op)], dtypes=OpDTypes.none)
    def test_reduction_ops_reduce(self, device, op):
        """Test that operators with reduction tag actually reduce numel when dim is specified."""
        samples = op.sample_inputs(device, torch.float32)

        for sample in samples:
            if "dim" not in sample.kwargs:
                continue

            dim_val = sample.kwargs["dim"]

            # Call the operation
            result = op(sample.input, *sample.args, **sample.kwargs)

            if isinstance(result, torch.Tensor):
                if dim_val is None:
                    dim_val = list(range(sample.input.ndim))
                reduction_dims = [dim_val] if isinstance(dim_val, int) else dim_val

                # Skip 0 dim for now
                if any(abs(dim) >= sample.input.ndim for dim in reduction_dims):
                    continue

                reduction_factor = 1
                for dim in reduction_dims:
                    reduction_factor *= sample.input.shape[dim]

                expected_numel = sample.input.numel() // reduction_factor

                self.assertEqual(
                    result.numel(),
                    expected_numel,
                    f"{op.name} with dim={dim_val} should reduce numel by factor of {reduction_factor} "
                    f"(input: {sample.input.numel()}, expected: {expected_numel}, got: {result.numel()})",
                )

    # Tests that the function and its (ndarray-accepting) reference produce the same
    #   values on the tensors from sample_inputs func for the corresponding op.
    # This test runs in double and complex double precision because
    # NumPy does computation internally using double precision for many functions
    # resulting in possible equality check failures.
    # skip windows case on CPU due to https://github.com/pytorch/pytorch/issues/129947
    # XPU test will be enabled step by step. Skip the tests temporarily.
    @skipXPU
    @onlyNativeDeviceTypesAnd(["hpu"])
    @suppress_warnings
    @ops(_ref_test_ops, allowed_dtypes=(torch.float64, torch.long, torch.complex128))
    def test_numpy_ref(self, device, dtype, op):
        if (
            TEST_WITH_TORCHINDUCTOR
            and op.formatted_name
            in ("signal_windows_exponential", "signal_windows_bartlett")
            and dtype == torch.float64
            and ("cuda" in device or "xpu" in device)
            or "cpu" in device
        ):  # noqa: E121
            raise unittest.SkipTest("XXX: raises tensor-likes are not close.")

        # Sets the default dtype to NumPy's default dtype of double
        with set_default_dtype(torch.double):
            for sample_input in op.reference_inputs(device, dtype):
                self.compare_with_reference(
                    op, op.ref, sample_input, exact_dtype=(dtype is not torch.long)
                )

    # Tests that the cpu and gpu results are consistent
    @onlyOn(["cuda", "xpu"])
    @suppress_warnings
    @skipCUDAIfNotRocm
    @ops(_ops_and_refs_with_no_numpy_ref, dtypes=OpDTypes.any_common_cpu_cuda_one)
    def test_compare_cpu(self, device, dtype, op):
        def to_cpu(arg):
            if isinstance(arg, torch.Tensor):
                return arg.to(device="cpu")
            return arg

        samples = op.reference_inputs(device, dtype)

        for sample in samples:
            cpu_sample = sample.transform(to_cpu)
            cuda_results = op(sample.input, *sample.args, **sample.kwargs)
            cpu_results = op(cpu_sample.input, *cpu_sample.args, **cpu_sample.kwargs)

            # output_process_fn_grad has a very unfortunate name
            # We use this function in linalg extensively to postprocess the inputs of functions
            # that are not completely well-defined. Think svd and multiplying the singular vectors by -1.
            # CPU and CUDA implementations of the SVD can return valid SVDs that are different.
            # We use this function to compare them.
            cuda_results = sample.output_process_fn_grad(cuda_results)
            cpu_results = cpu_sample.output_process_fn_grad(cpu_results)

            atol, rtol = 0, 0
            if dtype.is_floating_point or dtype.is_complex:
                atol, rtol = 1e-3, 1e-3
            self.assertEqual(cuda_results, cpu_results, atol=atol, rtol=rtol)

    # Tests that experimental Python References can propagate shape, dtype,
    # and device metadata properly.
    # See https://github.com/pytorch/pytorch/issues/78050 for a discussion of stride propagation.
    @skipXPU
    @onlyNativeDeviceTypesAnd(["hpu"])
    @ops(python_ref_db)
    @skipIfTorchInductor("Takes too long for inductor")
    def test_python_ref_meta(self, device, dtype, op):
        CHECK_CONJ_SKIPS = {
            torch._refs.linalg.svd,
        }

        with FakeTensorMode() as mode:
            pass

        def _to_tensormeta(x):
            if isinstance(x, torch.Tensor):
                out = FakeTensor.from_tensor(x, mode)
                return out
            return x

        # TODO: iterate over requires_grad true/false
        for sample in op.reference_inputs(device, dtype, requires_grad=False):
            result = op(sample.input, *sample.args, **sample.kwargs)

            meta_sample = sample.transform(_to_tensormeta)
            try:
                with mode:
                    meta_result = op(
                        meta_sample.input, *meta_sample.args, **meta_sample.kwargs
                    )
            except torch._subclasses.fake_tensor.UnsupportedFakeTensorException:
                continue
            except torch._subclasses.fake_tensor.DataDependentOutputException:
                continue
            except torch._subclasses.fake_tensor.UnsupportedOperatorException:
                continue

            if isinstance(result, torch.Tensor):
                self.assertTrue(isinstance(meta_result, FakeTensor))
                prims.utils.compare_tensor_meta(
                    result, meta_result, check_conj=op.op not in CHECK_CONJ_SKIPS
                )
            elif isinstance(result, Sequence):
                for a, b in zip(result, meta_result):
                    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
                        self.assertTrue(isinstance(b, FakeTensor))
                        prims.utils.compare_tensor_meta(
                            a, b, check_conj=op.op not in CHECK_CONJ_SKIPS
                        )

    def _ref_test_helper(
        self,
        ctx,
        device,
        dtype,
        op,
        skip_zero_numel=False,
        skip_zero_dim=False,
        skip_bfloat=False,
        skip_view_consistency=False,
    ):
        # NOTE: this test works by comparing the reference
        for sample in op.reference_inputs(device, dtype, requires_grad=False):
            ex = None
            if (
                isinstance(sample.input, torch.Tensor)
                and sample.input.numel() == 0
                and skip_zero_numel
            ):
                continue
            if (
                isinstance(sample.input, torch.Tensor)
                and sample.input.ndim == 0
                and skip_zero_dim
            ):
                continue

            if skip_bfloat and (
                (
                    isinstance(sample.input, torch.Tensor)
                    and sample.input.dtype == torch.bfloat16
                )
                or any(
                    isinstance(arg, torch.Tensor) and arg.dtype == torch.bfloat16
                    for arg in sample.args
                )
            ):
                continue
            with ctx():
                ref_result = op(sample.input, *sample.args, **sample.kwargs)
            torch_result = op.torch_opinfo(sample.input, *sample.args, **sample.kwargs)

            for a, b in zip(
                pytree.tree_leaves(ref_result), pytree.tree_leaves(torch_result)
            ):
                if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
                    prims.utils.compare_tensor_meta(a, b)
                    if (
                        getattr(op, "validate_view_consistency", True)
                        and not skip_view_consistency
                    ):
                        msg = (
                            f"The torch implementation {'returns' if b._is_view() else 'does not return'} "
                            f"a view, while the reference {'does' if a._is_view() else 'does not'}"
                        )
                        self.assertEqual(a._is_view(), b._is_view(), msg)

            # Computes the dtype the more precise computatino would occur in
            precise_dtype = torch.bool
            if prims.utils.is_integer_dtype(dtype):
                # Note: bool and integer dtypes do not have more
                # precise dtypes -- they simply must be close
                precise_dtype = dtype
            if prims.utils.is_float_dtype(dtype):
                precise_dtype = torch.double
            if prims.utils.is_complex_dtype(dtype):
                precise_dtype = torch.cdouble

            # Checks if the results are close
            try:
                self.assertEqual(
                    ref_result,
                    torch_result,
                    exact_stride=False,
                    exact_device=True,
                    exact_layout=True,
                    exact_is_coalesced=True,
                )
            except AssertionError as e:
                # Raises the error if the precise dtype comparison wouldn't be
                # different
                if dtype is precise_dtype:
                    raise e

                ex = e

            # Goes to next sample if these results are close
            if not ex:
                continue

            # If the results are not close, checks that the
            # reference is more accurate than the torch op
            def _make_precise(x):
                if isinstance(x, torch.dtype):
                    return precise_dtype
                if isinstance(x, torch.Tensor) and x.dtype is dtype:
                    return x.to(precise_dtype)
                return x

            precise_sample = sample.transform(_make_precise)
            precise_result = op.torch_opinfo(
                precise_sample.input, *precise_sample.args, **precise_sample.kwargs
            )

            def _distance(a, b):
                # Special-cases boolean comparisons
                if prims.utils.is_boolean_dtype(a.dtype):
                    assert b.dtype is torch.bool
                    return (a ^ b).sum()

                same = a == b
                if prims.utils.is_float_dtype(a.dtype) or prims.utils.is_complex_dtype(
                    a.dtype
                ):
                    same = torch.logical_or(
                        same, torch.logical_and(torch.isnan(a), torch.isnan(b))
                    )

                actual_error = torch.where(same, 0, torch.abs(a - b)).sum()
                return actual_error

            ref_distance = 0
            for a, b in zip(
                pytree.tree_leaves(ref_result), pytree.tree_leaves(precise_result)
            ):
                ref_distance = ref_distance + _distance(a, b)

            torch_distance = 0
            for a, b in zip(
                pytree.tree_leaves(torch_result), pytree.tree_leaves(precise_result)
            ):
                torch_distance = torch_distance + _distance(a, b)

            # TODO: consider adding some tolerance to this comparison
            msg = (
                f"Reference result was farther ({ref_distance}) from the precise "
                f"computation than the torch result was ({torch_distance})!"
            )
            self.assertTrue(ref_distance <= torch_distance, msg=msg)

        # Reports numerical accuracy discrepancies
        if ex is not None:
            msg = "Test passed because the reference was more accurate than the torch operator."
            warnings.warn(msg)

    # Tests that experimental Python References perform the same computation
    # as the operators they reference, when operator calls in the torch
    # namespace are remapped to the refs namespace (torch.foo becomes refs.foo).
    @skipXPU
    @onlyNativeDeviceTypesAnd(["hpu"])
    @ops(python_ref_db)
    @skipIfTorchInductor("Takes too long for inductor")
    def test_python_ref(self, device, dtype, op):
        # In this test, primTorch refs call into the refs namespace
        # For example, a ref with torch.foo in it will calls refs.foo instead
        # Direct calls to refs and prims are not affected
        if (
            TEST_WITH_ROCM
            and (op.name == "_refs.fft.ihfftn" or op.name == "_refs.fft.ihfft2")
            and dtype == torch.float16
        ):
            self.skipTest("Skipped on ROCm")
        self._ref_test_helper(lambda: TorchRefsMode(strict=True), device, dtype, op)

    # Tests that experimental Python References perform the same computation
    # as the operators they reference, when operator calls in the torch
    # namespace are preserved (torch.foo remains torch.foo).
    @skipXPU
    @onlyNativeDeviceTypesAnd(["hpu"])
    @ops(python_ref_db)
    @skipIfTorchInductor("Takes too long for inductor")
    def test_python_ref_torch_fallback(self, device, dtype, op):
        # In this test, refs call into the torch namespace (after the initial invocation)
        # For example, a ref with torch.foo in it will call torch.foo instead of refs.foo
        # Direct calls to refs and prims are not translated
        if op.full_name == "_refs.div.floor_rounding" and dtype == torch.bfloat16:
            self.skipTest(
                "Skipped _refs.div.floor_rounding with bfloat16"
                "Divide by 0: _refs produces NaN, torch produces +/-inf"
            )
        self._ref_test_helper(contextlib.nullcontext, device, dtype, op)

    @onlyCUDA
    @ops(python_ref_db)
    @parametrize("executor", ["aten"])
    @skipIfTorchInductor("Takes too long for inductor")
    def test_python_ref_executor(self, device, dtype, op, executor):
        from copy import copy

        from torch._prims.executor import make_traced

        op = copy(op)
        op.op = partial(make_traced(op.op), executor=executor)
        self._ref_test_helper(contextlib.nullcontext, device, dtype, op)

    @skipXPU
    @skipMeta
    @onlyNativeDeviceTypesAnd(["hpu"])
    @ops([op for op in op_db if op.error_inputs_func is not None], dtypes=OpDTypes.none)
    def test_errors(self, device, op):
        error_inputs = op.error_inputs(device)
        for ei in error_inputs:
            si = ei.sample_input
            with self.assertRaisesRegex(ei.error_type, ei.error_regex):
                out = op(si.input, *si.args, **si.kwargs)
                self.assertFalse(isinstance(out, type(NotImplemented)))

    @skipXPU
    @skipMeta
    @onlyNativeDeviceTypesAnd(["hpu"])
    @ops(
        [op for op in op_db if op.error_inputs_sparse_func is not None],
        dtypes=OpDTypes.none,
    )
    @parametrize(
        "layout",
        (
            torch.sparse_csr,
            torch.sparse_csc,
            torch.sparse_bsr,
            torch.sparse_bsc,
            torch.sparse_coo,
        ),
    )
    def test_errors_sparse(self, device, op, layout):
        for ei in op.error_inputs_sparse(device, layout):
            si = ei.sample_input
            with self.assertRaisesRegex(ei.error_type, ei.error_regex):
                out = op(si.input, *si.args, **si.kwargs)
                self.assertFalse(isinstance(out, type(NotImplemented)))

    @skipXPU
    @skipMeta
    @onlyNativeDeviceTypesAnd(["hpu"])
    @ops(
        [op for op in python_ref_db if op.error_inputs_func is not None],
        dtypes=OpDTypes.none,
    )
    @skipIfTorchInductor("Takes too long for inductor")
    def test_python_ref_errors(self, device, op):
        mode = FakeTensorMode()
        with mode:
            pass

        def _to_tensormeta(x):
            if isinstance(x, torch.Tensor):
                return FakeTensor.from_tensor(x, mode)
            return x

        error_inputs = op.error_inputs(device)
        for ei in error_inputs:
            si = ei.sample_input
            meta_sample = si.transform(_to_tensormeta)
            with self.assertRaisesRegex(ei.error_type, ei.error_regex):
                op(meta_sample.input, *meta_sample.args, **meta_sample.kwargs)

    # Tests that the function produces the same result when called with
    #   noncontiguous tensors.
    @skipXPU
    @with_tf32_off
    @onlyNativeDeviceTypesAnd(["hpu"])
    @suppress_warnings
    @ops(op_db, allowed_dtypes=(torch.float32, torch.long, torch.complex64))
    def test_noncontiguous_samples(self, device, dtype, op):
        test_grad = dtype in op.supported_backward_dtypes(torch.device(device).type)
        sample_inputs = op.sample_inputs(device, dtype, requires_grad=test_grad)
        for sample_input in sample_inputs:
            t_inp, t_args, t_kwargs = (
                sample_input.input,
                sample_input.args,
                sample_input.kwargs,
            )
            noncontig_sample = sample_input.noncontiguous()
            n_inp, n_args, n_kwargs = (
                noncontig_sample.input,
                noncontig_sample.args,
                noncontig_sample.kwargs,
            )

            # validates forward
            expected = op(t_inp, *t_args, **t_kwargs)
            actual = op(n_inp, *n_args, **n_kwargs)

            self.assertEqual(actual, expected)

            # Validate backward
            # Short-circuits if the op doesn't support grad in this device x dtype
            if not test_grad:
                continue

            expected = sample_input.output_process_fn_grad(expected)
            actual = sample_input.output_process_fn_grad(actual)

            if isinstance(expected, torch.Tensor):
                grad_for_expected = torch.randn_like(expected)
                grad_for_actual = noncontiguous_like(grad_for_expected)
            elif isinstance(expected, Sequence):
                # Filter output elements that do not require grad
                expected = [
                    t
                    for t in expected
                    if isinstance(t, torch.Tensor) and t.requires_grad
                ]
                actual = [
                    n for n in actual if isinstance(n, torch.Tensor) and n.requires_grad
                ]
                grad_for_expected = [torch.randn_like(t) for t in expected]
                grad_for_actual = [noncontiguous_like(n) for n in grad_for_expected]
            else:
                # Nothing to do if it returns a scalar or things like that
                continue

            # Concatenate inputs into a tuple
            t_inputs = (
                (t_inp,) + t_args
                if isinstance(t_inp, torch.Tensor)
                else tuple(t_inp) + t_args
            )
            n_inputs = (
                (n_inp,) + n_args
                if isinstance(n_inp, torch.Tensor)
                else tuple(n_inp) + n_args
            )

            # Filter the elements that are tensors that require grad
            t_input_tensors = [
                t for t in t_inputs if isinstance(t, torch.Tensor) and t.requires_grad
            ]
            n_input_tensors = [
                n for n in n_inputs if isinstance(n, torch.Tensor) and n.requires_grad
            ]

            self.assertEqual(len(t_input_tensors), len(n_input_tensors))

            # Some functions may not use all the inputs to generate gradients. One of the
            # few examples of this "odd" behaviour is F.hinge_embedding_loss
            t_grads = torch.autograd.grad(
                expected, t_input_tensors, grad_for_expected, allow_unused=True
            )
            n_grads = torch.autograd.grad(
                actual, n_input_tensors, grad_for_actual, allow_unused=True
            )

            msg = "Got different gradients for contiguous / non-contiguous inputs wrt input {}."
            for i, (t, n) in enumerate(zip(t_grads, n_grads)):
                self.assertEqual(t, n, msg=msg.format(i))

    # Separates one case from the following test_out because many ops don't properly implement the
    #   incorrectly sized out parameter warning properly yet
    # Cases test here:
    #   - out= with the correct dtype and device, but the wrong shape
    @skipXPU
    @ops(ops_and_refs, dtypes=OpDTypes.none)
    def test_out_warning(self, device, op):
        if TEST_WITH_TORCHDYNAMO and op.name == "_refs.clamp":
            self.skipTest("flaky")
        # Prefers running in float32 but has a fallback for the first listed supported dtype
        supported_dtypes = op.supported_dtypes(self.device_type)
        if len(supported_dtypes) == 0:
            self.skipTest("Skipped! Op has not supported dtypes on this device.")
        dtype = (
            torch.float32
            if torch.float32 in supported_dtypes
            else next(iter(supported_dtypes))
        )

        # Ops from python_ref_db point to python decomps that are potentially
        # wrapped with `torch._prims_common.wrappers.out_wrapper`. Unwrap these
        # ops before testing to avoid clashing with OpInfo.supports_out
        if not op.supports_out:
            op = copy.copy(op)
            op.op = _maybe_remove_out_wrapper(op.op)

        samples = op.sample_inputs(device, dtype)
        for sample in samples:
            # calls it normally to get the expected result
            expected = op(sample.input, *sample.args, **sample.kwargs)
            op_out = partial(op, sample.input, *sample.args, **sample.kwargs)

            # Short-circuits if output is not a single tensor or an
            #   iterable of tensors
            if not isinstance(expected, torch.Tensor) and not is_iterable_of_tensors(
                expected, include_empty=True
            ):
                self.skipTest(
                    "Skipped! Only supports single tensor or iterable of tensor outputs."
                )

            # Validates the op doesn't support out if it claims not to
            if not op.supports_out:
                with self.assertRaises(Exception):
                    assert op_out(out=expected) != NotImplemented
                return

            # A wrapper around map that works with single tensors and always
            #   instantiates the map. Used below to apply transforms to
            #   single tensor and iterable tensor outputs.
            def _apply_out_transform(fn, out):
                if isinstance(out, torch.Tensor):
                    return fn(out)

                # assumes (see above) that out is an iterable of tensors
                return tuple(map(fn, out))

            # Extracts strides from a tensor or iterable of tensors into a tuple
            def _extract_strides(out):
                if isinstance(out, torch.Tensor):
                    return (out.stride(),)

                # assumes (see above) that out is an iterable of tensors
                return tuple(t.stride() for t in out)

            # Extracts data pointers from a tensor or iterable of tensors into a tuple
            # NOTE: only extracts on the CPU and CUDA device types since some
            #   device types don't have storage
            def _extract_data_ptrs(out):
                if self.device_type != "cpu" and self.device_type != "cuda":
                    return ()

                if isinstance(out, torch.Tensor):
                    return (out.data_ptr(),)

                # assumes (see above) that out is an iterable of tensors
                return tuple(t.data_ptr() for t in out)

            @suppress_warnings
            def _compare_out(transform, *, compare_strides_and_data_ptrs=True):
                out = _apply_out_transform(transform, expected)
                original_strides = _extract_strides(out)
                original_ptrs = _extract_data_ptrs(out)

                op_out(out=out)
                final_strides = _extract_strides(out)
                final_ptrs = _extract_data_ptrs(out)

                self.assertEqual(expected, out)

                if compare_strides_and_data_ptrs:
                    stride_msg = (
                        f"Strides are not the same! Original strides were {original_strides} "
                        f"and strides are now {final_strides}"
                    )
                    self.assertEqual(original_strides, final_strides, msg=stride_msg)
                    self.assertEqual(original_ptrs, final_ptrs)

            # Case Zero: out= with the correct dtype and device, but the wrong shape
            #   Expected behavior: if nonempty, resize with a warning.
            def _case_zero_transform(t):
                wrong_shape = list(t.shape)

                if len(wrong_shape) == 0:
                    # Handles scalar tensor case (empty list)
                    wrong_shape = [2]
                else:
                    wrong_shape[-1] = wrong_shape[-1] + 1
                return make_tensor(wrong_shape, dtype=t.dtype, device=t.device)

            # Verifies the out values are correct
            _compare_out(_case_zero_transform, compare_strides_and_data_ptrs=False)

            # Additionally validates that the appropriate warning is thrown if a nonempty
            #   tensor is resized.
            def _any_nonempty(out):
                if isinstance(out, torch.Tensor):
                    return out.numel() > 0

                return any(x.numel() > 0 for x in out)

            out = _apply_out_transform(_case_zero_transform, expected)
            msg_fail = "Resized a non-empty tensor but did not warn about it."
            if _any_nonempty(out):
                with self.assertWarnsRegex(
                    UserWarning, "An output with one or more elements", msg=msg_fail
                ):
                    op_out(out=out)

    # Validates ops implement the correct out= behavior
    # See https://github.com/pytorch/pytorch/wiki/Developer-FAQ#how-does-out-work-in-pytorch
    #   for a description of the correct behavior
    # Validates the following cases:
    #   - Case 0: out has the correct shape, dtype, and device but is full of extremal values
    #   - Case 1: out has the correct shape, dtype, and device but is noncontiguous
    #   - Case 2: out has the correct dtype and device, but is zero elements
    #   - Case 3: out has the correct shape and dtype, but is on a different device type
    #   - Case 4: out has the correct shape and device, but a dtype that cannot
    #       "safely" cast to
    #
    # Case 3 and 4 are slightly different when the op is a factory function:
    #   - if device, dtype are NOT passed, any combination of dtype/device should be OK for out
    #   - if device, dtype are passed, device and dtype should match
    @skipXPU
    @ops(ops_and_refs, dtypes=OpDTypes.any_one)
    def test_out(self, device, dtype, op):
        # Prefers running in float32 but has a fallback for the first listed supported dtype
        samples = op.sample_inputs(device, dtype)

        # Ops from python_ref_db point to python decomps that are potentially
        # wrapped with `torch._prims_common.wrappers.out_wrapper`. Unwrap these
        # ops before testing to avoid clashing with OpInfo.supports_out
        if not op.supports_out:
            op = copy.copy(op)
            op.op = _maybe_remove_out_wrapper(op.op)

        for sample in samples:
            # calls it normally to get the expected result
            expected = op(sample.input, *sample.args, **sample.kwargs)
            op_out = partial(op, sample.input, *sample.args, **sample.kwargs)

            # Short-circuits if output is not a single tensor or an
            #   iterable of tensors
            if not isinstance(expected, torch.Tensor) and not is_iterable_of_tensors(
                expected, include_empty=True
            ):
                self.skipTest(
                    "Skipped! Only supports single tensor or iterable of tensor outputs."
                )

            # Validates the op doesn't support out if it claims not to
            if not op.supports_out:
                with self.assertRaises(Exception):
                    assert op_out(out=expected) != NotImplemented
                return

            # A wrapper around map that works with single tensors and always
            #   instantiates the map. Used below to apply transforms to
            #   single tensor and iterable tensor outputs.
            def _apply_out_transform(fn, out):
                if isinstance(out, torch.Tensor):
                    return fn(out)

                # assumes (see above) that out is an iterable of tensors
                return tuple(map(fn, out))

            # Extracts strides from a tensor or iterable of tensors into a tuple
            def _extract_strides(out):
                if isinstance(out, torch.Tensor):
                    return (out.stride(),)

                # assumes (see above) that out is an iterable of tensors
                return tuple(t.stride() for t in out)

            # Extracts data pointers from a tensor or iterable of tensors into a tuple
            # NOTE: only extracts on the CPU and CUDA device types since some
            #   device types don't have storage
            def _extract_data_ptrs(out):
                if self.device_type != "cpu" and self.device_type != "cuda":
                    return ()

                if isinstance(out, torch.Tensor):
                    return (out.data_ptr(),)

                # assumes (see above) that out is an iterable of tensors
                return tuple(t.data_ptr() for t in out)

            def _compare_out(transform, *, compare_strides_and_data_ptrs=True):
                out = _apply_out_transform(transform, expected)
                original_strides = _extract_strides(out)
                original_ptrs = _extract_data_ptrs(out)

                op_out(out=out)
                final_strides = _extract_strides(out)
                final_ptrs = _extract_data_ptrs(out)
                self.assertEqual(expected, out)

                if compare_strides_and_data_ptrs:
                    stride_msg = (
                        "Strides are not the same! "
                        f"Original strides were {original_strides} and strides are now {final_strides}"
                    )
                    self.assertEqual(original_strides, final_strides, msg=stride_msg)
                    self.assertEqual(original_ptrs, final_ptrs)

            # Case 0: out= with the correct shape, dtype, and device
            #   but NaN values for floating point and complex tensors, and
            #   maximum values for integer tensors.
            #   Expected behavior: out= values have no effect on the computation.
            def _case_zero_transform(t):
                try:
                    info = torch.iinfo(t.dtype)
                    return torch.full_like(t, info.max)
                except TypeError:
                    # for non-integer types fills with NaN
                    return torch.full_like(t, float("nan"))

            _compare_out(_case_zero_transform)

            # Case 1: out= with the correct shape, dtype, and device,
            #   but noncontiguous.
            #   Expected behavior: strides are respected and `out` storage is not changed.
            def _case_one_transform(t):
                return make_tensor(
                    t.shape, dtype=t.dtype, device=t.device, noncontiguous=True
                )

            _compare_out(_case_one_transform)

            # Case 2: out= with the correct dtype and device, but has no elements.
            #   Expected behavior: resize without warning.
            def _case_two_transform(t):
                return make_tensor((0,), dtype=t.dtype, device=t.device)

            _compare_out(_case_two_transform, compare_strides_and_data_ptrs=False)

            # Also validates that no warning is thrown when this out is resized
            out = _apply_out_transform(_case_two_transform, expected)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                op_out(out=out)

            # Verifies no warning is a resize warning
            for w in caught:
                if "An output with one or more elements" in str(w.message):
                    self.fail(
                        "Resizing an out= argument with no elements threw a resize warning!"
                    )

            # Case 3: out= with correct shape and dtype, but wrong device.
            #   Expected behavior: throws an error.
            #   This case is ignored on CPU to allow some scalar operations to succeed.
            factory_fn_msg = (
                "\n\nNOTE: If your op is a factory function (i.e., it accepts TensorOptions) you should mark its "
                "OpInfo with `is_factory_function=True`."
            )

            if torch.device(device).type != "cpu":
                wrong_device = "cpu"

                def _case_three_transform(t):
                    return make_tensor(t.shape, dtype=t.dtype, device=wrong_device)

                out = _apply_out_transform(_case_three_transform, expected)

                if op.is_factory_function and sample.kwargs.get("device", None) is None:
                    op_out(out=out)
                else:
                    msg_fail = (
                        f"Expected RuntimeError when calling with input.device={device} and out.device={wrong_device}."
                    ) + factory_fn_msg
                    with self.assertRaises(RuntimeError, msg=msg_fail):
                        op_out(out=out)

            # Case 4: out= with correct shape and device, but a dtype
            #   that output cannot be "safely" cast to (long).
            #   Expected behavior: error.
            # NOTE: this case is filtered by dtype since some ops produce
            #   bool tensors, for example, which can be safely cast to any
            #   dtype. It is applied when single tensors are floating point or complex
            #   dtypes, or if an op returns multiple tensors when at least one such
            #   tensor is a floating point or complex dtype.
            _dtypes = floating_and_complex_types_and(torch.float16, torch.bfloat16)
            if (
                isinstance(expected, torch.Tensor)
                and expected.dtype in _dtypes
                or (
                    not isinstance(expected, torch.Tensor)
                    and any(t.dtype in _dtypes for t in expected)
                )
            ):

                def _case_four_transform(t):
                    return make_tensor(t.shape, dtype=torch.long, device=t.device)

                out = _apply_out_transform(_case_four_transform, expected)
                msg_fail = "Expected RuntimeError when doing an unsafe cast!"
                msg_fail = (
                    msg_fail
                    if not isinstance(expected, torch.Tensor)
                    else (
                        "Expected RuntimeError when doing an unsafe cast from a result of dtype "
                        f"{expected.dtype} into an out= with dtype torch.long"
                    )
                ) + factory_fn_msg

                if op.is_factory_function and sample.kwargs.get("dtype", None) is None:
                    op_out(out=out)
                else:
                    # TODO: Remove me when all ops will raise type error on mismatched types
                    exc_type = (
                        TypeError
                        if op.name
                        in [
                            "_chunk_cat",
                            "cat",
                            "column_stack",
                            "dstack",
                            "hstack",
                            "vstack",
                            "stack",
                        ]
                        else RuntimeError
                    )
                    with self.assertRaises(exc_type, msg=msg_fail):
                        op_out(out=out)

    @skipXPU
    @ops(
        [
            op
            for op in op_db
            if op.supports_out and (op.supports_autograd or op.is_factory_function)
        ],
        dtypes=OpDTypes.supported,
        allowed_dtypes=[torch.float, torch.cfloat],
    )
    def test_out_requires_grad_error(self, device, dtype, op):
        sample = first_sample(self, op.sample_inputs(device, dtype))

        # Call op to get prototype for out arguments
        with maybe_skip_size_asserts(op):
            expect = op(sample.input, *sample.args, **sample.kwargs)
        any_requires_grad = False

        def set_requires_grad(x):
            nonlocal any_requires_grad
            if isinstance(x, torch.Tensor) and (
                x.is_floating_point() or x.is_complex()
            ):
                any_requires_grad = True
                x.requires_grad_(True)
            return x

        out = pytree.tree_map_(set_requires_grad, expect)
        if not any_requires_grad:
            # Skip ops without any floating point outputs, e.g. isnan
            return

        msg = (
            "functions with out=... arguments don't support automatic "
            "differentiation, but one of the arguments requires grad."
        )
        with self.assertRaises(RuntimeError, msg=msg), maybe_skip_size_asserts(op):
            op(sample.input, *sample.args, **sample.kwargs, out=out)

    @skipXPU
    @ops(filter(reduction_dtype_filter, ops_and_refs), dtypes=(torch.int16,))
    def test_out_integral_dtype(self, device, dtype, op):
        def helper(with_out, expectFail, op_to_test, inputs, *args, **kwargs):
            out = None
            try:
                if with_out:
                    out = torch.empty(0, dtype=torch.int32, device=device)
                    op_to_test(inputs, *args, out=out, **kwargs)
                else:
                    out = op_to_test(inputs, *args, **kwargs)
                self.assertFalse(expectFail)
            except RuntimeError as err:
                self.assertEqual(
                    str(err), "dtype argument and out dtype must match in reduction"
                )
                self.assertTrue(expectFail)
            return out

        samples = op.sample_inputs(device, dtype)
        for sample in samples:
            if "dtype" not in sample.kwargs:
                helper(False, False, op, sample.input, *sample.args, **sample.kwargs)
                helper(True, False, op, sample.input, *sample.args, **sample.kwargs)
                sample.kwargs["dtype"] = torch.int16
                helper(False, False, op, sample.input, *sample.args, **sample.kwargs)
                helper(True, True, op, sample.input, *sample.args, **sample.kwargs)
                sample.kwargs["dtype"] = torch.int32
                helper(False, False, op, sample.input, *sample.args, **sample.kwargs)
                helper(True, False, op, sample.input, *sample.args, **sample.kwargs)
            else:
                helper(False, False, op, sample.input, *sample.args, **sample.kwargs)
                helper(
                    True,
                    sample.kwargs["dtype"] != torch.int32,
                    op,
                    sample.input,
                    *sample.args,
                    **sample.kwargs,
                )

    # Tests that the forward and backward passes of operations produce the
    #   same values for the cross-product of op variants (method, inplace)
    #   against eager's gold standard op function variant
    @skipXPU
    @_variant_ops(op_db)
    def test_variant_consistency_eager(self, device, dtype, op):
        # Acquires variants (method variant, inplace variant, operator variant, inplace_operator variant, aliases)

        method = op.method_variant
        inplace = op.inplace_variant
        operator = op.operator_variant
        inplace_operator = op.inplace_operator_variant

        # list of all inplace ops: inplace variant + alias inplace variants if exist
        inplace_ops = [inplace, inplace_operator]
        variants = [method, inplace, operator, inplace_operator]
        operators = [operator, inplace_operator]

        for a_op in op.aliases:
            variants.append(a_op.op)
            variants.append(a_op.method_variant)
            variants.append(a_op.inplace_variant)
            inplace_ops.append(a_op.inplace_variant)

        inplace_variants = tuple(filter(None, inplace_ops))
        variants = tuple(filter(None, variants))
        operators = tuple(filter(None, operators))

        _requires_grad = dtype in op.supported_backward_dtypes(
            torch.device(device).type
        )

        include_conjugated_inputs = op.test_conjugated_samples and dtype.is_complex
        samples = op.sample_inputs(
            device,
            dtype,
            requires_grad=_requires_grad,
            include_conjugated_inputs=include_conjugated_inputs,
        )
        samples = list(samples)

        def _test_consistency_helper(samples, variants):
            for sample in samples:
                # TODO: Check grad for all Tensors requiring grad if sample.input is TensorList
                tensor = (
                    sample.input
                    if isinstance(sample.input, torch.Tensor)
                    else sample.input[0]
                )

                # Computes function forward and backward values
                tensor.grad = None
                expected_forward = op(sample.input, *sample.args, **sample.kwargs)
                expected_grad = None

                output_process_fn_grad = (
                    sample.output_process_fn_grad
                    if sample.output_process_fn_grad
                    else lambda x: x
                )

                # Skips inplace variants if the output dtype is not the same as
                #   the input dtype
                skip_inplace = False
                if (
                    isinstance(expected_forward, torch.Tensor)
                    and expected_forward.dtype is not tensor.dtype
                ):
                    skip_inplace = True

                # TODO: backward consistency only supported for single tensor outputs
                # TODO: backward consistency only checked on sample.input, not all
                #   tensor inputs
                # TODO: update to handle checking grads of all tensor inputs as
                #   derived from each tensor output
                if isinstance(
                    expected_forward, torch.Tensor
                ) and dtype in op.supported_backward_dtypes(torch.device(device).type):
                    out = output_process_fn_grad(expected_forward).sum()
                    if out.dtype.is_complex:
                        out = out.abs()
                    out.backward()
                    expected_grad = tensor.grad

                # Test eager consistency
                for variant in variants:
                    # Skips inplace ops
                    if variant in inplace_ops and skip_inplace:
                        continue

                    # Compares variant's forward
                    # Note: copies the to-be-modified input when testing the inplace variant
                    tensor.grad = None
                    cloned = (
                        clone_input_helper(sample.input)
                        if variant in inplace_ops
                        else sample.input
                    )

                    if variant in inplace_ops and sample.broadcasts_input:
                        with self.assertRaises(
                            RuntimeError,
                            msg=(
                                "inplace variant either incorrectly allowed "
                                f"resizing or you have marked the sample {sample.summary()}"
                                " incorrectly with `broadcasts_self=True"
                            ),
                        ):
                            variant_forward = variant(
                                cloned, *sample.args, **sample.kwargs
                            )
                        continue

                    if variant in operators and sample.kwargs:
                        # skip samples with kwargs for operator variants
                        continue

                    variant_forward = variant(cloned, *sample.args, **sample.kwargs)
                    self.assertEqual(expected_forward, variant_forward)

                    # Compares variant's backward
                    if expected_grad is not None and (
                        variant not in inplace_ops or op.supports_inplace_autograd
                    ):
                        out = output_process_fn_grad(variant_forward).sum()
                        if out.dtype.is_complex:
                            out = out.abs()
                        out.backward()
                        self.assertEqual(expected_grad, tensor.grad)

        _test_consistency_helper(samples, variants)

        def _test_inplace_preserve_storage(samples, variants):
            for sample in samples:
                # Skips inplace variants if the output dtype is not the same as
                #   the input dtype
                expected_forward = op(sample.input, *sample.args, **sample.kwargs)
                tensor = (
                    sample.input
                    if isinstance(sample.input, torch.Tensor)
                    else sample.input[0]
                )
                skip_inplace = False
                if (
                    isinstance(expected_forward, torch.Tensor)
                    and expected_forward.dtype is not tensor.dtype
                ):
                    skip_inplace = True
                if skip_inplace:
                    return
                for variant in variants:
                    cloned = (
                        clone_input_helper(sample.input)
                        if variant in inplace_ops
                        else sample.input
                    )
                    inp_tensor = (
                        cloned if isinstance(cloned, torch.Tensor) else cloned[0]
                    )
                    data_ptr = inp_tensor.data_ptr()
                    if variant in operators and sample.kwargs:
                        # skip samples with kwargs for operator variants
                        continue

                    variant_forward = variant(cloned, *sample.args, **sample.kwargs)
                    # TODO Support non-tensor outputs if they exist for inplace ops
                    if isinstance(variant_forward, torch.Tensor):
                        self.assertEqual(
                            data_ptr, variant_forward.data_ptr(), atol=0, rtol=0
                        )
                    else:
                        self.assertTrue(
                            False,
                            "Non-tensor outputs for inplace ops are not supported",
                        )

        if len(inplace_ops) > 0:
            inplace_samples = list(
                filter(lambda sample: not sample.broadcasts_input, samples)
            )
            _test_inplace_preserve_storage(inplace_samples, inplace_variants)

    # Reference testing for operations in complex32 against complex64.
    # NOTE: We test against complex64 as NumPy doesn't have a complex32 equivalent dtype.
    @skipXPU
    @ops(op_db, allowed_dtypes=(torch.complex32,))
    def test_complex_half_reference_testing(self, device, dtype, op):
        if not op.supports_dtype(torch.complex32, device):
            unittest.skip("Does not support complex32")

        for sample in op.sample_inputs(device, dtype):
            actual = op(sample.input, *sample.args, **sample.kwargs)
            # sample.transform applies the lambda to torch.Tensor and torch.dtype.
            # However, we only want to apply it to Tensors with dtype `torch.complex32`..
            transformed_sample = sample.transform(
                lambda x: x.to(torch.complex64)
                if isinstance(x, torch.Tensor) and x.dtype is torch.complex32
                else x
            )
            expected = op(
                transformed_sample.input,
                *transformed_sample.args,
                **transformed_sample.kwargs,
            )
            # Since range of chalf is much less compared to cfloat,
            # we get `inf`s easily (eg. with `pow`, `exp`),
            # so we cast `cfloat` back to `chalf`.
            expected = tree_map(
                lambda x: x.to(torch.complex32)
                if isinstance(x, torch.Tensor) and x.dtype is torch.complex64
                else x,
                expected,
            )

            # `exact_dtype` is False because for ops like real, imag
            # we get different dtypes for `actual` and `expected`
            # `chalf` input -> `half` output
            # `cfloat` input -> `float` output
            self.assertEqual(actual, expected, exact_dtype=False)

    @skipXPU
    @ops(op_db, allowed_dtypes=(torch.bool,))
    def test_non_standard_bool_values(self, device, dtype, op):
        # Test boolean values other than 0x00 and 0x01 (gh-54789)
        def convert_boolean_tensors(x):
            if not isinstance(x, torch.Tensor) or x.dtype != torch.bool:
                return x

            # Map False -> 0 and True -> Random value in [2, 255]
            true_vals = torch.randint(
                2, 255, x.shape, dtype=torch.uint8, device=x.device
            )
            false_vals = torch.zeros((), dtype=torch.uint8, device=x.device)
            x_int = torch.where(x, true_vals, false_vals)

            ret = x_int.view(torch.bool)
            self.assertEqual(ret, x)
            return ret

        for sample in op.sample_inputs(device, dtype):
            expect = op(sample.input, *sample.args, **sample.kwargs)

            transformed = sample.transform(convert_boolean_tensors)
            actual = op(transformed.input, *transformed.args, **transformed.kwargs)

            self.assertEqual(expect, actual)

    # Validates that each OpInfo specifies its forward and backward dtypes
    #   correctly for CPU and CUDA devices
    @skipXPU
    @skipMeta
    @onlyNativeDeviceTypesAnd(["hpu"])
    @ops(ops_and_refs, dtypes=OpDTypes.none)
    def test_dtypes(self, device, op):
        # Check complex32 support only if the op claims.
        # TODO: Once the complex32 support is better, we should add check for complex32 unconditionally.
        device_type = torch.device(device).type
        include_complex32 = (
            (torch.complex32,)
            if op.supports_dtype(torch.complex32, device_type)
            else ()
        )

        # dtypes to try to backward in
        allowed_backward_dtypes = floating_and_complex_types_and(
            *((torch.half, torch.bfloat16) + include_complex32)
        )

        # lists for (un)supported dtypes
        supported_dtypes = set()
        unsupported_dtypes = set()
        supported_backward_dtypes = set()
        unsupported_backward_dtypes = set()
        dtype_error: dict[torch.dtype, Exception] = {}

        def unsupported(dtype, e):
            dtype_error[dtype] = e
            unsupported_dtypes.add(dtype)
            if dtype in allowed_backward_dtypes:
                unsupported_backward_dtypes.add(dtype)

        for dtype in all_types_and_complex_and(
            *((torch.half, torch.bfloat16, torch.bool) + include_complex32)
        ):
            # tries to acquire samples - failure indicates lack of support
            requires_grad = dtype in allowed_backward_dtypes
            try:
                samples = tuple(
                    op.sample_inputs(device, dtype, requires_grad=requires_grad)
                )
            except Exception as e:
                unsupported(dtype, e)
                continue

            for sample in samples:
                # tries to call operator with the sample - failure indicates
                #   lack of support
                try:
                    result = op(sample.input, *sample.args, **sample.kwargs)
                    supported_dtypes.add(dtype)
                except Exception as e:
                    # NOTE: some ops will fail in forward if their inputs
                    #   require grad but they don't support computing the gradient
                    #   in that type! This is a bug in the op!
                    unsupported(dtype, e)
                    continue

                # Checks for backward support in the same dtype, if the input has
                # one or more tensors requiring grad
                def _tensor_requires_grad(x):
                    if isinstance(x, dict):
                        for v in x.values():
                            if _tensor_requires_grad(v):
                                return True
                    if isinstance(x, (list, tuple)):
                        for a in x:
                            if _tensor_requires_grad(a):
                                return True
                    if isinstance(x, torch.Tensor) and x.requires_grad:
                        return True

                    return False

                requires_grad = (
                    _tensor_requires_grad(sample.input)
                    or _tensor_requires_grad(sample.args)
                    or _tensor_requires_grad(sample.kwargs)
                )
                if not requires_grad:
                    continue

                try:
                    result = sample.output_process_fn_grad(result)
                    if isinstance(result, torch.Tensor):
                        backward_tensor = result
                    elif isinstance(result, Sequence) and isinstance(
                        result[0], torch.Tensor
                    ):
                        backward_tensor = result[0]
                    else:
                        continue

                    # Note: this grad may not have the same dtype as dtype
                    # For functions like complex (float -> complex) or abs
                    #   (complex -> float) the grad tensor will have a
                    #   different dtype than the input.
                    #   For simplicity, this is still modeled as these ops
                    #   supporting grad in the input dtype.
                    grad = torch.randn_like(backward_tensor)
                    backward_tensor.backward(grad)
                    supported_backward_dtypes.add(dtype)
                except Exception as e:
                    dtype_error[dtype] = e
                    unsupported_backward_dtypes.add(dtype)

        # Checks that dtypes are listed correctly and generates an informative
        #   error message

        supported_forward = supported_dtypes - unsupported_dtypes
        partially_supported_forward = supported_dtypes & unsupported_dtypes
        unsupported_forward = unsupported_dtypes - supported_dtypes
        supported_backward = supported_backward_dtypes - unsupported_backward_dtypes
        partially_supported_backward = (
            supported_backward_dtypes & unsupported_backward_dtypes
        )
        unsupported_backward = unsupported_backward_dtypes - supported_backward_dtypes

        device_type = torch.device(device).type

        claimed_forward = set(op.supported_dtypes(device_type))
        supported_but_unclaimed_forward = supported_forward - claimed_forward
        claimed_but_unsupported_forward = claimed_forward & unsupported_forward

        claimed_backward = set(op.supported_backward_dtypes(device_type))
        supported_but_unclaimed_backward = supported_backward - claimed_backward
        claimed_but_unsupported_backward = claimed_backward & unsupported_backward

        # Partially supporting a dtype is not an error, but we print a warning
        if (len(partially_supported_forward) + len(partially_supported_backward)) > 0:
            msg = f"Some dtypes for {op.name} on device type {device_type} are only partially supported!\n"
            if len(partially_supported_forward) > 0:
                msg = (
                    msg
                    + f"The following dtypes only worked on some samples during forward: {partially_supported_forward}.\n"
                )
            if len(partially_supported_backward) > 0:
                msg = (
                    msg
                    + f"The following dtypes only worked on some samples during backward: {partially_supported_backward}.\n"
                )
            print(msg)

        if (
            len(supported_but_unclaimed_forward)
            + len(claimed_but_unsupported_forward)
            + len(supported_but_unclaimed_backward)
            + len(claimed_but_unsupported_backward)
        ) == 0:
            return

        if TEST_WITH_TORCHDYNAMO:
            # NOTE: Also for TEST_WITH_TORCHINDUCTOR tests
            # Under compile, some ops may be decomposed into supported ops
            # So it is okay to have supported_but_unclaimed_*
            if (
                len(claimed_but_unsupported_forward)
                + len(claimed_but_unsupported_backward)
            ) == 0:
                return

        # Reference operators often support additional dtypes, and that's OK
        if op in python_ref_db:
            if (
                len(claimed_but_unsupported_forward)
                + len(claimed_but_unsupported_backward)
            ) == 0:
                return

        # Generates error msg
        msg = f"The supported dtypes for {op.name} on device type {device_type} are incorrect!\n"
        if len(supported_but_unclaimed_forward) > 0:
            msg = (
                msg
                + "The following dtypes worked in forward but are not listed by the OpInfo: "
                + f"{supported_but_unclaimed_forward}.\n"
            )
        if len(supported_but_unclaimed_backward) > 0:
            msg = (
                msg
                + "The following dtypes worked in backward but are not listed by the OpInfo: "
                + f"{supported_but_unclaimed_backward}.\n"
            )
        if len(claimed_but_unsupported_forward) > 0:
            msg = (
                msg
                + "The following dtypes did not work in forward but are listed by the OpInfo: "
                + f"{claimed_but_unsupported_forward}.\n"
            )
        if len(claimed_but_unsupported_backward) > 0:
            msg = (
                msg
                + "The following dtypes did not work in backward "
                + f"but are listed by the OpInfo: {claimed_but_unsupported_backward}.\n"
            )

        all_claimed_but_unsupported = set.union(
            claimed_but_unsupported_backward, claimed_but_unsupported_forward
        )
        if all_claimed_but_unsupported:
            msg += "Unexpected failures raised the following errors:\n"
            for dtype in all_claimed_but_unsupported:
                msg += f"{dtype} - {dtype_error[dtype]}\n"

        self.fail(msg)

    # Validates that each OpInfo that sets promotes_int_to_float=True does as it says
    @skipXPU
    @skipMeta
    @onlyNativeDeviceTypesAnd(["hpu"])
    @ops(
        (op for op in op_db if op.promotes_int_to_float),
        allowed_dtypes=integral_types_and(torch.bool),
    )
    def test_promotes_int_to_float(self, device, dtype, op):
        for sample in op.sample_inputs(device, dtype):
            output = op(sample.input, *sample.args, **sample.kwargs)
            if not output.dtype.is_floating_point:
                self.fail(
                    f"The OpInfo sets `promotes_int_to_float=True`, but {dtype} was promoted to {output.dtype}."
                )

    # Checks whether running the operations on both CPU and meta devices raise errors
    # when the output tensors have mismatching data-types (i.e. data-types that are
    # different from the expected one).
    #
    # The idea is that the meta implementations should correctly reflect on the behavior
    # of other concrete devices (e.g. CPU and CUDA).
    @onlyCPU
    @ops([op for op in op_db if op.supports_out], allowed_dtypes=(torch.float32,))
    @skipOps(
        "TestCommon",
        "test_meta_consistency_out_dtype_mismatch",
        meta_consistency_out_dtype_mismatch_xfails,
    )
    @skipIfTorchDynamo("meta device runs only on eager")
    def test_meta_consistency_out_dtype_mismatch(self, device, dtype, op):
        samples = op.sample_inputs(device, dtype)

        for sample in samples:
            input, args, kwargs = (sample.input, sample.args, sample.kwargs)

            try:
                # Call the functional version of the operation, using a real device, so that
                # we get the actual expected result.
                expected = op(input, *args, **kwargs)

                if isinstance(expected, tuple):
                    # Some operations return named tuples. However, pytree does not work well
                    # with that, so we turn it into a plain tuple.
                    expected = tuple(expected)
            except Exception:
                # If that doesn't work out, go to the next sample.
                continue

            def run_on(dev):
                # Create new outputs in the desired device, with a mismatching data type of
                # the same kind.
                out = pytree.tree_map_only(
                    torch.Tensor,
                    lambda t: torch.empty_like(t, device=dev, dtype=torch.float64),
                    expected,
                )

                # Move inputs to the desired device.
                arguments = (input, args, kwargs)
                arguments = pytree.tree_map_only(
                    torch.Tensor, lambda t: t.to(dev), arguments
                )
                # Also, replace every instance of 'cpu' arguments by whatever the desired
                # device really should be.
                arguments = pytree.tree_map_only(
                    torch.device, lambda d: torch.device(dev), arguments
                )
                arguments = pytree.tree_map_only(
                    str, lambda v: dev if v == device else v, arguments
                )
                input_, args_, kwargs_ = arguments

                # Try running the operation, and return the raised error, if any.
                try:
                    op(input_, *args_, **kwargs_, out=out)
                except Exception as e:
                    return e

            # Run the operation with the sample arguments on both CPU and meta devices, capturing
            # the raised error, if any.
            device_err = run_on(device)
            meta_err = run_on("meta")

            # Check whether they disagree on the result.
            #
            # In case there is an inconsistency of whether an error was raised using the real device,
            # but not when using the meta device, we raise a RuntimeError, chaining with the captured
            # one.
            #
            # We could just assertEquals here, but chaining the errors is more informative.
            if device_err is None and meta_err is not None:
                raise RuntimeError(f"{device} didn't fail, but meta did.") from meta_err
            elif device_err is not None and meta_err is None:
                raise RuntimeError(f"{device} failed, but meta didn't.") from device_err


@unMarkDynamoStrictTest
class TestCompositeCompliance(TestCase):
    # Checks if the operator (if it is composite) is written to support most
    # backends and Tensor subclasses. See "CompositeImplicitAutograd Compliance"
    # in aten/src/ATen/native/README.md for more details
    @unittest.skipIf(
        IS_FBCODE or IS_SANDCASTLE, "__torch_dispatch__ does not work in fbcode"
    )
    @ops(op_db, allowed_dtypes=(torch.float,))
    def test_operator(self, device, dtype, op):
        samples = op.sample_inputs(device, dtype, requires_grad=False)

        for sample in samples:
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs
            composite_compliance.check_with_mode(op, args, kwargs, self.assertEqual)
            composite_compliance.check_all_permutations(
                op, args, kwargs, self.assertEqual
            )

    @unittest.skipIf(
        IS_FBCODE or IS_SANDCASTLE, "__torch_dispatch__ does not work in fbcode"
    )
    @ops([op for op in op_db if op.supports_autograd], allowed_dtypes=(torch.float,))
    def test_backward(self, device, dtype, op):
        samples = op.sample_inputs(device, dtype, requires_grad=True)

        for sample in samples:
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs
            # We pass assertEqual so that decorators like `toleranceOverride`
            # actually work (otherwise they silently do nothing!)
            composite_compliance.check_backward_formula(
                op.get_op(),
                args,
                kwargs,
                sample.output_process_fn_grad,
                op.gradcheck_wrapper,
                self.assertEqual,
            )

    @unittest.skipIf(
        IS_FBCODE or IS_SANDCASTLE, "__torch_dispatch__ does not work in fbcode"
    )
    @ops(op_db, allowed_dtypes=(torch.float,))
    def test_forward_ad(self, device, dtype, op):
        if torch.float not in op.supported_backward_dtypes(device):
            raise unittest.SkipTest("Does not support autograd")

        if not op.supports_forward_ad:
            raise unittest.SkipTest("Does not support forward_ad")

        samples = op.sample_inputs(device, dtype, requires_grad=True)

        for sample in samples:
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs
            # We pass assertEqual so that decorators like `toleranceOverride`
            # actually work (otherwise they silently do nothing!)
            composite_compliance.check_forward_ad_formula(
                op.get_op(), args, kwargs, op.gradcheck_wrapper, self.assertEqual
            )

    @ops(op_db, allowed_dtypes=(torch.float,))
    def test_cow_input(self, device, dtype, op):
        samples = op.sample_inputs(device, dtype, requires_grad=op.supports_autograd)

        def is_strided_tensor(arg):
            return torch.is_tensor(arg) and arg.layout == torch.strided

        def check_ignore_materialize(idx_or_kw, allow_list):
            return (allow_list is not None) and (idx_or_kw in allow_list)

        def check_cow_input(
            arg,
            arg_copy,
            arg_raw,
            idx_or_kw,
            backward_or_forward="forward",
            supports_cow_input_no_materialize=op.supports_cow_input_no_materialize_forward,
            allow_list=op.allow_cow_input_materialize_forward,
        ):
            arg_name = (
                f"Argument {idx_or_kw}"
                if isinstance(idx_or_kw, int)
                else f"Keyword argument '{idx_or_kw}'"
            ) + f" during {backward_or_forward} call"

            if is_strided_tensor(arg):
                self.assertTrue(
                    torch._C._is_cow_tensor(arg_raw),
                    msg=(
                        f"{arg_name} raw input should remain COW, but it "
                        "unexpectedly materialized."
                    ),
                )
                is_cow = torch._C._is_cow_tensor(arg)

                if supports_cow_input_no_materialize and not check_ignore_materialize(
                    idx_or_kw, allow_list
                ):
                    self.assertTrue(
                        is_cow,
                        msg=(
                            f"{arg_name} unexpectedly materializes. "
                            f"Either set `supports_cow_input_no_materialize_{backward_or_forward}=False` "
                            "in this operation's OpInfo, add the arg to the OpInfo's "
                            f"`allow_cow_input_materialize_{backward_or_forward}` list, or change the "
                            "implementation to avoid materialization."
                        ),
                    )

                if is_cow:
                    self.assertTrue(
                        torch.allclose(arg, arg_copy, rtol=0, atol=0, equal_nan=True),
                        msg=(
                            f"{arg_name} avoided materialization, "
                            "but the operation mutated its data."
                        ),
                    )
                else:
                    self.assertTrue(
                        torch.allclose(
                            arg_raw, arg_copy, rtol=0, atol=0, equal_nan=True
                        ),
                        msg=(
                            f"{arg_name} materialized, which is allowed in this "
                            "case, but the COW input data was mutated, which is "
                            "not allowed."
                        ),
                    )

        for sample in samples:
            args_raw = [sample.input] + list(sample.args)
            kwargs_raw = sample.kwargs
            args_copy = []
            args = []
            kwargs_copy = {}
            kwargs = {}

            # Convert strided tensor inputs to COW tensors and make copies of
            # all inputs
            for arg in args_raw:
                if is_strided_tensor(arg):
                    args_copy.append(arg.detach().clone())
                    args.append(torch._lazy_clone(arg))
                else:
                    if torch.is_tensor(arg):
                        args_copy.append(arg.detach().clone())
                    else:
                        args_copy.append(copy.deepcopy(arg))
                    args.append(arg)

            for kw, arg in kwargs_raw.items():
                if is_strided_tensor(arg):
                    kwargs_copy[kw] = arg.detach().clone()
                    kwargs[kw] = torch._lazy_clone(arg)
                else:
                    if torch.is_tensor(arg):
                        kwargs_copy[kw] = arg.detach().clone()
                    else:
                        kwargs_copy[kw] = copy.deepcopy(arg)
                    kwargs[kw] = arg

            leaf_tensors = composite_compliance.gather_leaf_tensors(args, kwargs)

            # Call forward op
            results_raw = op.get_op()(*args, **kwargs)

            # Check that COW inputs remain COW after the forward op is executed
            for idx, arg in enumerate(args):
                check_cow_input(arg, args_copy[idx], args_raw[idx], idx)

            for kw, arg in kwargs.items():
                check_cow_input(arg, kwargs_copy[kw], kwargs_raw[kw], kw)

            # Call backward op if it is supported. This part of the test is
            # based on `composite_compliance.check_backward_formula`
            if (
                op.supports_autograd
                and len(leaf_tensors) > 0
                and not op.skip_cow_input_backward
            ):
                if sample.output_process_fn_grad is not None:
                    results_raw = sample.output_process_fn_grad(results_raw)

                leaf_results = pytree.tree_leaves(results_raw)
                results = [
                    r
                    for r in leaf_results
                    if isinstance(r, torch.Tensor) and r.requires_grad
                ]

                all_results_strided = all(
                    is_strided_tensor(result) for result in results
                )

                # Only test backward if the results are strided tensors
                if all_results_strided:
                    output_grads_raw = [
                        torch.ones(r.shape, device=r.device, dtype=r.dtype)
                        for r in results
                    ]
                    output_grads_copy = []
                    output_grads = []

                    # Convert output grads to COW tensors and make copies
                    for output_grad in output_grads_raw:
                        output_grads_copy.append(output_grad.detach().clone())
                        output_grads.append(torch._lazy_clone(output_grad))

                    torch.autograd.grad(
                        results,
                        leaf_tensors,
                        output_grads,
                        allow_unused=True,
                        retain_graph=True,
                    )

                    # Check that COW inputs remain COW after the backward op is executed
                    for idx, arg in enumerate(args):
                        check_cow_input(
                            arg,
                            args_copy[idx],
                            args_raw[idx],
                            idx,
                            backward_or_forward="backward",
                            supports_cow_input_no_materialize=op.supports_cow_input_no_materialize_backward,
                            allow_list=op.allow_cow_input_materialize_backward,
                        )

                    # Check that COW inputs remain COW after the backward op is executed
                    for idx, output_grad in enumerate(output_grads):
                        check_cow_input(
                            output_grad,
                            output_grads_copy[idx],
                            output_grads_raw[idx],
                            f"output grad {idx}",
                            backward_or_forward="backward",
                            supports_cow_input_no_materialize=op.supports_cow_input_no_materialize_backward,
                            allow_list=op.allow_cow_input_materialize_backward,
                        )

    @ops(op_db, allowed_dtypes=(torch.float,))
    def test_view_replay(self, device, dtype, op):
        def _assert_match_metadata(a, b):
            self.assertEqual(a.size(), b.size())
            self.assertEqual(a.stride(), b.stride())
            self.assertEqual(a.storage_offset(), b.storage_offset())
            self.assertEqual(a.device, b.device)
            self.assertEqual(a.dtype, b.dtype)

        # ensure view replay is enabled
        with torch.autograd._force_original_view_tracking(True):
            for sample in op.sample_inputs(device, dtype, requires_grad=False):
                inp = sample.input
                outs = op(inp, *sample.args, **sample.kwargs)
                if not isinstance(outs, (tuple, list)):
                    outs = [outs]

                # for all outputs that are views of the input, we should be able to replay the
                # forward and reverse views via a functioning view_func() / rev_view_func().
                for out in outs:
                    if not (
                        isinstance(out, torch.Tensor)
                        and out._is_view()
                        and out._base is inp
                    ):
                        continue

                    # forward view_func
                    new_inp = inp.clone()
                    _assert_match_metadata(new_inp, inp)
                    new_out = out._view_func_unsafe(new_inp)
                    _assert_match_metadata(new_out, out)
                    self.assertEqual(new_out, out)

                    # reverse view_func
                    new_out = out.detach()
                    new_inp = out._rev_view_func_unsafe(new_out)
                    _assert_match_metadata(new_inp, inp)
                    self.assertTrue(new_inp._is_view())
                    self.assertTrue(new_inp._base is new_out)


@unMarkDynamoStrictTest
class TestMathBits(TestCase):
    # Tests that
    # 1. The operator's output for physically conjugated/negated tensors and conjugate/negative view tensors
    # produces the same value
    # 2. The gradients are same in both cases mentioned in (1)
    # 3. If the operator's inplace variant is supported, tests that the inplace operation
    #    produces the correct value when called on a conjugate/negative view tensor and that the output
    #    has its conj/neg bit set to true
    # This test only runs for C -> R and C -> C functions
    # TODO: add tests for `R->C` functions
    # Note: This test runs for functions that take both tensors and tensorlists as input.
    def _test_math_view(
        self,
        device,
        dtype,
        op,
        samples,
        math_op_physical,
        math_op_view,
        is_bit_set,
        out_type,
    ):
        inplace_variant = op.inplace_variant

        # helper function to clone and conjugate/negate the input if its a tensor
        # else clone the sequence and conjugate/negate the first element in the sequence
        # If a requires_grad argument is provided the tensor being conjugated/negated will
        # have its requires_grad set to that value.
        def clone_and_perform_view(input, **kwargs):
            if isinstance(input, torch.Tensor):
                requires_grad = kwargs.get("requires_grad", input.requires_grad)
                with torch.no_grad():
                    # Ensure view represents the original sample input
                    input = math_op_physical(input)
                # Note: .conj() is not called under no_grad mode since it's not allowed to modify a
                # view created in no_grad mode. Here it's ok to do so, so as a workaround we call conj
                # before resetting the requires_grad field for input
                input = math_op_view(input)
                assert input.is_leaf
                return input.requires_grad_(requires_grad)

            if isinstance(input, Sequence):
                out = list(map(clone_input_helper, input))
                out[0] = clone_and_perform_view(out[0])
                return tuple(out)

        for sample in samples:
            tensor = (
                sample.input
                if isinstance(sample.input, torch.Tensor)
                else sample.input[0]
            )
            cloned1 = clone_and_perform_view(sample.input)

            # Computes function forward value with a physically conjugated/negated tensor and
            # a conj/neg view tensor and verifies that the output in both case are equal.
            expected_forward = op(sample.input, *sample.args, **sample.kwargs)
            forward_with_mathview = op(cloned1, *sample.args, **sample.kwargs)
            self.assertEqual(expected_forward, forward_with_mathview)

            # If the op has an inplace variant, and the input doesn't require broadcasting
            # and has the same dtype as output, verify that the inplace operation on a conjugated/negated
            # input produces correct output, and the output tensor has the conj/neg bit set to True
            if inplace_variant is not None and not sample.broadcasts_input:
                cloned2 = clone_and_perform_view(tensor, requires_grad=False)
                if (
                    isinstance(expected_forward, torch.Tensor)
                    and expected_forward.dtype is tensor.dtype
                ):
                    inplace_forward = inplace_variant(
                        cloned2, *sample.args, **sample.kwargs
                    )
                    self.assertTrue(is_bit_set(inplace_forward))
                    self.assertEqual(inplace_forward, expected_forward)

            # TODO: backward consistency only supported for single tensor outputs
            # TODO: backward consistency only checked on sample.input, not all
            #   tensor inputs
            # TODO: update to handle checking grads of all tensor inputs as
            #   derived from each tensor output
            if (
                isinstance(expected_forward, torch.Tensor)
                and expected_forward.requires_grad
            ):
                output_process_fn_grad = sample.output_process_fn_grad or (lambda x: x)
                expected_forward = output_process_fn_grad(expected_forward)
                forward_with_mathview = output_process_fn_grad(forward_with_mathview)

                tensor = (
                    sample.input
                    if isinstance(sample.input, torch.Tensor)
                    else sample.input[0]
                )
                expected_forward.sum().abs().backward(retain_graph=True)
                forward_with_mathview.sum().abs().backward(retain_graph=True)
                if tensor.grad is not None:
                    cloned1_tensor = (
                        cloned1 if isinstance(cloned1, torch.Tensor) else cloned1[0]
                    )
                    self.assertEqual(tensor.grad, cloned1_tensor.grad)

                    tensor.grad, cloned1_tensor.grad = None, None

                    # a repeat of the above test if output is not complex valued
                    if out_type(expected_forward):
                        grad = torch.randn_like(expected_forward)
                        expected_forward.backward(grad)
                        forward_with_mathview.backward(
                            math_op_view(math_op_physical(grad))
                        )

                        self.assertEqual(tensor.grad, cloned1_tensor.grad)

    @ops(ops_and_refs, allowed_dtypes=(torch.cfloat,))
    def test_conj_view(self, device, dtype, op):
        if not op.test_conjugated_samples:
            self.skipTest("Operation doesn't support conjugated inputs.")
        math_op_physical = torch.conj_physical
        math_op_view = torch.conj
        _requires_grad = torch.cfloat in op.supported_backward_dtypes(
            torch.device(device).type
        )
        is_bit_set = torch.is_conj
        samples = op.sample_inputs(device, dtype, requires_grad=_requires_grad)
        self._test_math_view(
            device,
            dtype,
            op,
            samples,
            math_op_physical,
            math_op_view,
            is_bit_set,
            torch.is_complex,
        )

    @ops(ops_and_refs, allowed_dtypes=(torch.double,))
    def test_neg_view(self, device, dtype, op):
        if not op.test_neg_view:
            self.skipTest("Operation not tested with tensors with negative bit.")
        math_op_physical = torch.neg
        math_op_view = torch._neg_view
        is_bit_set = torch.is_neg
        samples = op.sample_inputs(device, dtype, requires_grad=op.supports_autograd)
        self._test_math_view(
            device,
            dtype,
            op,
            samples,
            math_op_physical,
            math_op_view,
            is_bit_set,
            lambda x: True,
        )

    @ops(ops_and_refs, allowed_dtypes=(torch.cdouble,))
    def test_neg_conj_view(self, device, dtype, op):
        if not op.test_neg_view:
            self.skipTest("Operation not tested with tensors with negative bit.")
        if not op.test_conjugated_samples:
            self.skipTest("Operation doesn't support conjugated inputs.")

        def math_op_physical(x):
            return -x.conj_physical()

        def math_op_view(x):
            return torch._neg_view(x).conj()

        def is_bit_set(x):
            return torch.is_neg(x) and torch.is_conj(x)

        _requires_grad = dtype in op.supported_backward_dtypes(
            torch.device(device).type
        )
        samples = op.sample_inputs(device, dtype, requires_grad=_requires_grad)
        # Only test one sample
        samples = itertools.islice(samples, 1)
        self._test_math_view(
            device,
            dtype,
            op,
            samples,
            math_op_physical,
            math_op_view,
            is_bit_set,
            torch.is_complex,
        )


# input strides and size may have been altered due to the result of an inplace op
def check_inplace_view(func, input, rs, input_size, input_strides):
    if func is None:
        return
    # TODO: extend this test to test ops with multiple outputs and ops like native_batch_norm(_legit).out
    # which mutate not necessarily the first input.
    if isinstance(rs, torch.Tensor) and rs is input:
        unequal_size = rs.size() != input_size
        unequal_strides = rs.stride() != input_strides
        # resize_ should probably have inplace_view tag. Not adding the tag since it
        # breaks some codegen logic
        if unequal_size or unequal_strides:
            if isinstance(func, torch._ops.OpOverloadPacket):
                func = func.default
            # Reference: https://github.com/pytorch/pytorch/issues/78759
            if func is not torch.ops.aten.resize_.default:
                # TODO: use self.assertIn when we have separate tests for each tag
                assert torch.Tag.inplace_view in func.tags


# A mode that when enabled runs correctness checks to ensure
# that operators have expected tags based on their input and
# output tensor properties
class _TestTagsMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if isinstance(args[0], torch.Tensor):
            old_size = args[0].size()
            old_stride = args[0].stride()
            rs = func(*args, **kwargs)
            check_inplace_view(func, args[0], rs, old_size, old_stride)
        else:
            rs = func(*args, **kwargs)
        return rs


# Test to verify the correctness for tags in `tags.yaml`, also available for access through `torch.Tags`
@unMarkDynamoStrictTest
class TestTags(TestCase):
    @onlyCPU
    @ops(ops_and_refs, dtypes=OpDTypes.any_one)
    def test_tags(self, device, dtype, op):
        samples = op.sample_inputs(device, dtype, requires_grad=False)
        for sample in samples:
            # TODO: Test tags for ops that return a list of tensors
            input = sample.input
            if isinstance(input, torch.Tensor):
                old_size = input.size()
                old_stride = input.stride()
                with _TestTagsMode():
                    rs = op(input, *sample.args, **sample.kwargs)
                # TODO: add test for aliases: https://github.com/pytorch/pytorch/issues/78761
                aten_name = op.aten_name if op.aten_name is not None else op.name
                opoverloadpacket = getattr(torch.ops.aten, aten_name, None)
                check_inplace_view(opoverloadpacket, input, rs, old_size, old_stride)


class TestSelfKwarg(TestCase):
    def test_self_kwargs(self):
        """Verify that we can call the aten ops with all kwargs even if the
        argument's name is "self"
        """
        torch.ops.aten.reshape.default(self=torch.rand(1, 2), shape=[2])
        torch.ops.aten.min.default(self=torch.rand(100))


@unMarkDynamoStrictTest
class TestRefsOpsInfo(TestCase):
    import_paths = [
        "_refs",
        "_refs.special",
        "_refs.nn.functional",
        "_refs.fft",
        "_refs._conversions",
    ]
    module_alls = [
        (path, import_module(f"torch.{path}").__all__) for path in import_paths
    ]
    ref_ops_names = tuple(
        itertools.chain.from_iterable(
            [f"{path}.{op}" for op in module_all] for path, module_all in module_alls
        )
    )
    ref_db_names = {ref_op.name for ref_op in python_ref_db}

    # TODO: References that do not have an entry in python_ref_db
    skip_ref_ops = {
        "_refs.alias",
        "_refs.bitwise_right_shift",
        "_refs.copy_to",
        "_refs.empty_permuted",
        "_refs.empty_strided",
        "_refs.equal",
        "_refs.full",
        "_refs.full_like",
        "_refs.is_complex",
        "_refs.to",
        "_refs.mvlgamma",
        "_refs.ones",
        "_refs.ones_like",
        "_refs.special.expit",
        "_refs.std_var",
        "_refs.swap_axes",
        "_refs.uniform",
        "_refs.scalar_tensor",
        "_refs.trunc_divide",
        "_refs.zero",
        "_refs.zeros",
        "_refs.zeros_like",
        "_refs.rfloordiv",
        "_refs.rtruediv",
        "_refs.rpow",
        # These should be tested with their out-of-place counterparts
        "_refs.index_add_",
        "_refs.index_copy_",
        "_refs.index_fill_",
        "_refs.native_group_norm",
    }

    not_in_decomp_table = {
        # duplicated in _decomp and _refs
        "_refs.nn.functional.group_norm",
        "_refs.nn.functional.mse_loss",
        "_refs.floor_divide",
        # duplicated as refs do not have decent support for advanced indexing
        "_refs.index_copy",
        "_refs.index_copy_",
        "_refs.index_add",
        "_refs.index_add_",
        # these are not aten ops?
        "_refs._conversions.bfloat16",
        "_refs._conversions.bool",
        "_refs._conversions.byte",
        "_refs._conversions.char",
        "_refs._conversions.double",
        "_refs._conversions.float",
        "_refs._conversions.half",
        "_refs._conversions.int",
        "_refs._conversions.long",
        "_refs._conversions.short",
        "_refs._conversions.chalf",
        "_refs._conversions.cfloat",
        "_refs._conversions.cdouble",
        "_refs.broadcast_shapes",
        "_refs.broadcast_tensors",
        "_refs.mvlgamma",
        "_refs.nn.functional.layer_norm",
        "_refs.nn.functional.tanhshrink",
        "_refs.nn.functional.triplet_margin_loss",
        "_refs.rfloordiv",
        "_refs.rtruediv",
        "_refs.rpow",
        # CompositeImplicitAutograd
        "_refs.allclose",
        "_refs.atleast_1d",
        "_refs.atleast_2d",
        "_refs.atleast_3d",
        "_refs.broadcast_to",
        "_refs.chunk",
        "_refs.column_stack",
        "_refs.contiguous",
        "_refs.dsplit",
        "_refs.dstack",
        "_refs.fill",
        "_refs.fill_",
        "_refs.flatten",
        "_refs.fliplr",
        "_refs.flipud",
        "_refs.float_power",
        "_refs.hsplit",
        "_refs.hstack",
        "_refs.isclose",
        "_refs.isfinite",
        "_refs.isreal",
        "_refs.istft",
        "_refs.log_softmax",
        "_refs.movedim",
        "_refs.narrow",
        "_refs.nn.functional.dropout",
        "_refs.nn.functional.l1_loss",
        "_refs.nn.functional.smooth_l1_loss",
        "_refs.nn.functional.log_softmax",
        "_refs.nn.functional.poisson_nll_loss",
        "_refs.nn.functional.softmax",
        "_refs.nn.functional.softmin",
        "_refs.positive",
        "_refs.ravel",
        "_refs.reshape",
        "_refs.softmax",
        "_refs.special.expit",
        "_refs.special.log_softmax",
        "_refs.special.softmax",
        "_refs.square",
        "_refs.stft",
        "_refs.T",
        "_refs.take_along_dim",
        "_refs.tensor_split",
        "_refs.to",
        "_refs.true_divide",
        "_refs.trunc_divide",
        "_refs.vsplit",
        "_refs.vstack",
        "_refs.linalg.matrix_norm",
        "_refs.linalg.norm",
        "_refs.linalg.svd",
        "_refs.linalg.svdvals",
        "_refs.unflatten",
        "_refs.sum_to_size",
        # ref implementation missing kwargs
        "_refs.full_like",  # missing "layout"
        "_refs.scalar_tensor",  # missing "layout"
        # other
        "_refs.block_diag",  # only refs._block_diag_iterable is in decomposition table
        "_refs.empty",  # intentional; direct empty is faster and has less guards
        "_refs.empty_permuted",  # intentional; direct empty is faster and has less guards
        "_refs.expand_as",
        "_refs.as_strided",  # _prims._as_strided_meta: "reduce() of empty sequence with no initial value"
        "_refs.copy_to",  # torch._C._jit_get_operation: No such operator aten::copy_to
        "_refs.equal",  # 'bool' object has no attribute 'dtype'
        "_refs.conj",  # Calls _prims.conj
        "_refs.real",
        "_refs.imag",
        "_refs.reshape_as",
        "_refs.view_as",
        "_refs.view_as_complex",  # TorchInductor does not support complex at the moment.
        # the decompositions for these ops are slightly different
        # because of out handling
        "_refs.var_mean",
        "_refs.std_mean",
        "_refs.native_layer_norm",
    }

    @parametrize("op", ref_ops_names)
    def test_refs_are_in_python_ref_db(self, op):
        inplace = op[-1] == "_"
        if op in self.skip_ref_ops:
            raise unittest.SkipTest(f"{op} does not have an entry in python_ref_db")
        elif inplace:
            self.assertNotIn(
                op,
                self.ref_db_names,
                msg=f"{op} is an in-place operation and should not have an OpInfo",
            )
        else:
            # Intentionally don't use assertIn to avoid printing the
            # (very large) container
            self.assertTrue(op in self.ref_db_names, msg=f"{op} not in ref_db_names")

    @parametrize("op", ref_ops_names)
    def test_refs_are_in_decomp_table(self, op):
        path = op.split(".")
        module_path = ".".join(path[:-1])
        op_name = path[-1]
        op_impl = getattr(import_module(f"torch.{module_path}"), op_name)

        if op in self.not_in_decomp_table:
            self.assertNotIn(
                op_impl,
                torch._decomp.decomposition_table.values(),
                f"Unexpectedly found {op} in torch._decomp.decomposition_table.values()",
            )
        else:
            self.assertIn(
                op_impl,
                torch._decomp.decomposition_table.values(),
                f"Did not find {op} in torch._decomp.decomposition_table.values()",
            )


fake_skips = (
    "aminmax",  # failing input
    "cov",  # aweights cannot be negtaive
    "istft",  # window overlap add min: 0
    "linalg.eigvals",  # The tensor has a non-zero number of elements, but its data is not allocated yet
    "linalg.eigvalsh",  # aten::linalg_eigvalsh.out' with arguments from the 'Meta' backend
    "linalg.matrix_power",  # Could not run 'aten::eye.m_out' with arguments from the 'Meta' backend
    # "linalg.pinv",  # Could not run 'aten::pinv.out' with arguments from the 'Meta' backend
    "linalg.matrix_rank.hermitian",  # Could not run 'aten::linalg_eigvalsh.out' with arguments from the 'Meta' backend
    "linalg.pinv.hermitian",  # tensor.mH is only supported on matrices or batches of matrices. Got 1-D tensor
    "linalg.solve",  # Could not run 'aten::linalg_solve' with arguments from the 'Meta' backend
    "linalg.tensorsolve",  # Could not run 'aten::linalg_solve' with arguments from the 'Meta'
    "lu_solve",  # MALLOC ERROR: debug
    "multinomial",  # Could not run 'aten::multinomial' with arguments from the 'Meta' backend
    "mvlgamma.mvlgamma_p_1",  # Could not run 'aten::_local_scalar_dense' with arguments from the 'Meta' backend
    "mvlgamma.mvlgamma_p_3",  # Could not run 'aten::_local_scalar_dense' with arguments from the 'Meta' backend
    "mvlgamma.mvlgamma_p_5",  # Could not run 'aten::_local_scalar_dense' with arguments from the 'Meta' backend
    "quantile",  # quantile() q values must be in the range [0, 1]
    "nanquantile",  # quantile() q values must be in the range [0, 1]
    "nn.functional.ctc_loss",  # The tensor has a non-zero number of elements, but its data is not allocated yet
    "nn.functional.embedding_bag",  # sometimes errors
    "nn.functional.nll_loss",  # sometimes errors
    "nn.functional.max_pool1d",  # The tensor has a non-zero number of elements
    "to_sparse",  # Could not run 'aten::_to_sparse' with arguments from the 'Meta' backend
    "tensor_split",  # The tensor has a non-zero number of elements, but its data is not allocated yet
    "repeat_interleave",  # cannot repeat_interleave a meta tensor without output_size
    "sparse.sampled.addmm",  # sparsity not supported
    # Can not infer total number of classes from meta. no way at present to throw DynamicOutputShapeException
    "nn.functional.one_hot",
    "narrow",  # Fails only for one overload with DataDependentOutputException (hence skip).
)

fake_autocast_device_skips = defaultdict(dict)

# TODO: investigate/fix
fake_autocast_device_skips["cpu"] = {"linalg.pinv"}
fake_autocast_device_skips["cuda"] = {"linalg.pinv", "pinverse"}


dynamic_output_op_tests = (
    "argwhere",
    "bincount",
    "combinations",
    "linalg.lstsq",
    "masked_select",
    "nonzero",
    "unique_consecutive",
    "unique",
    "linalg.lstsq.grad_oriented",
)

# Ops that have dynamic output shapes that we can handle when
# allow_dynamic_shape_ops is True in fake tensor shape environment.
supported_dynamic_output_op_tests = (
    "nonzero",
    "unique",
    "repeat_interleave",
    "masked_select",
)

# some inputs invoke dynamic output shape operators, some do not
sometimes_dynamic_output_op_test = ("__getitem__", "index_select")

data_dependent_op_tests = (
    "equal",
    "corrcoef",
    "nn.functional.gaussian_nll_loss",
    "allclose",
)

aliasing_failures = ("histogramdd",)

fake_backward_skips = {
    "linalg.cond",
    "linalg.matrix_norm",
    "linalg.norm",
    "linalg.svd",
    "linalg.svdvals",
    "pca_lowrank",
    "roll",
    "svd_lowrank",
    "sgn",
}

fake_backward_xfails = {skip(s) for s in fake_backward_skips} | {
    skip("nn.functional.ctc_loss"),
}

fake_autocast_backward_xfails = {
    skip("nn.functional.binary_cross_entropy"),
    skip("sparse.sampled_addmm"),
    skip("linalg.pinv"),
    skip("linalg.pinv", "hermitian"),
    skip("linalg.pinv", "singular"),
    skip("pinverse"),
}


@unMarkDynamoStrictTest
class TestFakeTensor(TestCase):
    def setUp(self):
        super().setUp()
        # Turn on FakeTensor caching and cross-checking for these tests:
        cache_enabled = unittest.mock.patch(
            "torch._dynamo.config.fake_tensor_cache_enabled", True
        )
        cache_enabled.start()
        self.addCleanup(cache_enabled.stop)

        cache_crosscheck = unittest.mock.patch(
            "torch._dynamo.config.fake_tensor_cache_crosscheck_enabled", True
        )
        cache_crosscheck.start()
        self.addCleanup(cache_crosscheck.stop)

    def _test_fake_helper(self, device, dtype, op, context):
        name = op.name
        if op.variant_test_name:
            name += "." + op.variant_test_name
        if name in fake_skips or "sparse" in name or "jiterator" in name:
            self.skipTest("Skip failing test")

        samples = op.sample_inputs(device, dtype, requires_grad=False)
        for sample in samples:
            mode = FakeTensorMode()

            from torch.fx.experimental.symbolic_shapes import ShapeEnv

            allow_dynamic_output_shape_shape_env = ShapeEnv(
                allow_dynamic_output_shape_ops=True
            )

            allow_dynamic_output_shape_mode = FakeTensorMode(
                shape_env=allow_dynamic_output_shape_shape_env
            )

            try:
                with context():
                    res = op(sample.input, *sample.args, **sample.kwargs)
            except Exception:
                continue

            def run_with_fake_mode_and_verify(fake_mode, match_results=True):
                def map_to_fake(e):
                    if isinstance(e, torch.Tensor):
                        return fake_mode.from_tensor(e)
                    else:
                        return e

                input = tree_map(map_to_fake, sample.input)
                args = tree_map(map_to_fake, sample.args)
                kwargs = tree_map(map_to_fake, sample.kwargs)

                try:
                    with context():
                        with fake_mode:
                            res_fake = op(input, *args, **kwargs)

                    if not match_results:
                        return

                    for fake_out, real_out in zip(
                        pytree.tree_leaves(res_fake), pytree.tree_leaves(res)
                    ):
                        if not isinstance(fake_out, torch.Tensor):
                            self.assertTrue(not isinstance(real_out, torch.Tensor))
                            self.assertEqual(fake_out, real_out)
                            continue

                        self.assertTrue(isinstance(fake_out, FakeTensor))
                        # if you see a shape exception here, you may need to add
                        # a `dynamic_output_shape` tag to an operator

                        if op.op not in [
                            torch.ops.aten._efficient_attention_forward,
                            torch.ops.aten._flash_attention_forward,
                        ]:
                            # prims/decomps must correctly model strides,
                            # see https://github.com/pytorch/pytorch/issues/78050#issuecomment-1253950325

                            # note: the excluded ops have intentionally incorrect device;
                            # see "Note [Seed and Offset]" (_meta_registrations.py)
                            prims.utils.compare_tensor_meta(fake_out, real_out, True)

                        if name not in aliasing_failures:
                            fake_aliasing = outputs_alias_inputs(
                                (input, args, kwargs), res_fake
                            )
                            real_aliasing = outputs_alias_inputs(
                                (sample.input, sample, args, sample.kwargs), res
                            )
                            self.assertEqual(fake_aliasing, real_aliasing)

                    self.assertTrue(
                        name not in dynamic_output_op_tests
                        and name not in data_dependent_op_tests
                    )

                except torch._subclasses.fake_tensor.UnsupportedFakeTensorException:
                    pass
                except torch._subclasses.fake_tensor.UnsupportedOperatorException:
                    pass
                except torch._subclasses.fake_tensor.DynamicOutputShapeException:
                    self.assertTrue(
                        name in dynamic_output_op_tests
                        or name in sometimes_dynamic_output_op_test
                    )
                    self.assertTrue(
                        fake_mode.shape_env is None
                        or not fake_mode.shape_env.allow_dynamic_output_shape_ops
                        or name not in supported_dynamic_output_op_tests
                    )
                except torch._subclasses.fake_tensor.DataDependentOutputException:
                    self.assertTrue(name in data_dependent_op_tests)

            run_with_fake_mode_and_verify(mode)
            if name in supported_dynamic_output_op_tests:
                run_with_fake_mode_and_verify(
                    allow_dynamic_output_shape_mode, match_results=False
                )

    @ops(op_db, dtypes=OpDTypes.any_one)
    def test_pointwise_ops(self, device, dtype, op):
        name = op.name
        if op.variant_test_name:
            name += "." + op.variant_test_name
        if name in fake_skips or "sparse" in name or "jiterator" in name:
            self.skipTest("Skip failing test")

        test_self = self

        class TestPointwiseMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}

                out = func(*args, **kwargs)

                if torch.Tag.pointwise in func.tags:
                    shapes = []
                    for inp in pytree.arg_tree_leaves(*args, **kwargs):
                        if isinstance(inp, torch.Tensor):
                            shapes.append(inp.shape)

                    out_shape = torch._refs._broadcast_shapes(*shapes)

                    for out_elem in pytree.tree_leaves(out):
                        if isinstance(out_elem, torch.Tensor):
                            test_self.assertEqual(out_elem.shape, out_shape)

                return out

        samples = op.sample_inputs(device, dtype, requires_grad=False)
        for sample in samples:
            mode = FakeTensorMode()

            def map_to_fake(e):
                if isinstance(e, torch.Tensor):
                    return mode.from_tensor(e)
                else:
                    return e

            input = tree_map(map_to_fake, sample.input)
            args = tree_map(map_to_fake, sample.args)
            kwargs = tree_map(map_to_fake, sample.kwargs)

            try:
                op(input, *args, **kwargs)
            except Exception:
                continue

            with TestPointwiseMode():
                with mode:
                    op(input, *args, **kwargs)

    @ops(op_db, dtypes=OpDTypes.any_one)
    def test_fake(self, device, dtype, op):
        self._test_fake_helper(device, dtype, op, contextlib.nullcontext)

    @ops(op_db, dtypes=OpDTypes.any_one)
    def test_fake_autocast(self, device, dtype, op):
        device_type = torch.device(device).type
        if op.name in fake_autocast_device_skips[device_type]:
            self.skipTest("Skip failing test")

        def context_fn():
            return torch.amp.autocast(device_type)

        self._test_fake_helper(device, dtype, op, context_fn)

    def _test_fake_crossref_helper(self, device, dtype, op, context):
        samples = op.sample_inputs(device, dtype, requires_grad=True)

        for sample in samples:
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs

            # skip these to speed up tests
            common_skip_ops = (
                aten.detach.default,
                aten.empty_strided.default,
                aten.copy_.default,
                aten.is_same_size.default,
            )

            # TODO: enable check_aliasing, batch norm fails
            try:
                with torch._subclasses.CrossRefFakeMode(
                    ignore_op_fn=lambda fn: fn in common_skip_ops, check_aliasing=True
                ):
                    with (
                        warnings.catch_warnings(),
                        context(),
                        torch.autograd.set_multithreading_enabled(False),
                    ):
                        composite_compliance.compute_expected_grads(
                            op.get_op(),
                            args,
                            kwargs,
                            sample.output_process_fn_grad,
                            op.gradcheck_wrapper,
                        )
            except torch._subclasses.fake_tensor.UnsupportedOperatorException:
                pass

    @onlyCUDA
    @ops([op for op in op_db if op.supports_autograd], allowed_dtypes=(torch.float,))
    @skipOps(
        "TestFakeTensor", "test_fake_crossref_backward_no_amp", fake_backward_xfails
    )
    def test_fake_crossref_backward_no_amp(self, device, dtype, op):
        self._test_fake_crossref_helper(device, dtype, op, contextlib.nullcontext)

    @onlyCUDA
    @ops([op for op in op_db if op.supports_autograd], allowed_dtypes=(torch.float,))
    @skipOps(
        "TestFakeTensor",
        "test_fake_crossref_backward_amp",
        fake_backward_xfails | fake_autocast_backward_xfails,
    )
    def test_fake_crossref_backward_amp(self, device, dtype, op):
        self._test_fake_crossref_helper(device, dtype, op, torch.cuda.amp.autocast)

    @ops([op for op in ops_and_refs if op.is_factory_function])
    def test_strided_layout(self, device, dtype, op):
        samples = op.sample_inputs(device, dtype)
        for sample in samples:
            kwargs = sample.kwargs.copy()
            kwargs["layout"] = torch.strided
            strided_result = op(sample.input, *sample.args, **kwargs)
            self.assertEqual(strided_result.layout, torch.strided)


class TestForwardADWithScalars(TestCase):
    @ops(
        [op for op in op_db if op.name in ["mul", "add", "div"]],
        allowed_dtypes=(torch.float32,),
    )
    def test_0d_tensor_with_python_scalar(self, device, dtype, op):
        """Test that forward AD preserves dtype when combining 0D tensors with Python scalars."""
        if torch.float not in op.supported_backward_dtypes(device):
            raise unittest.SkipTest("Does not support autograd")

        # skip if operator doesn't support forward AD
        if not op.supports_forward_ad:
            raise unittest.SkipTest("Does not support forward_ad")

        # create 0D tensors
        primal0d = torch.ones((), device=device, dtype=dtype)
        tangent0d = torch.ones((), device=device, dtype=dtype)

        with torch.autograd.forward_ad.dual_level():
            dual0d = torch.autograd.forward_ad.make_dual(primal0d, tangent0d)

            # Test with scalar on RHS
            if op.supports_rhs_python_scalar:
                result = op(dual0d, 2.0)
                p, t = torch.autograd.forward_ad.unpack_dual(result)
                self.assertEqual(
                    p.dtype, t.dtype, f"{op.name} and scalar on RHS - dtype mismatch"
                )
            # Test with scalar on LHS
            if op.supports_one_python_scalar:
                result = op(2.0, dual0d)
                p, t = torch.autograd.forward_ad.unpack_dual(result)
                self.assertEqual(
                    p.dtype, t.dtype, f"{op.name} and scalar on LHS - dtype mismatch"
                )


instantiate_device_type_tests(TestCommon, globals(), allow_xpu=True)
instantiate_device_type_tests(TestCompositeCompliance, globals())
instantiate_device_type_tests(TestMathBits, globals())
instantiate_device_type_tests(TestRefsOpsInfo, globals(), only_for="cpu")
instantiate_device_type_tests(TestFakeTensor, globals())
instantiate_device_type_tests(TestTags, globals())
instantiate_device_type_tests(TestForwardADWithScalars, globals())

if __name__ == "__main__":
    TestCase._default_dtype_check_enabled = True
    run_tests()
