# Owner(s): ["module: meta tensors"]
# ruff: noqa: F841


import contextlib
import copy
import dataclasses
import gc
import inspect
import io
import itertools
import pickle
import re
import subprocess
import sys
import unittest
import weakref
from functools import partial
from unittest.mock import patch

import numpy as np

import torch
import torch._dynamo
import torch._functorch.config
import torch._prims as prims
import torch.testing._internal.optests as optests
import torch.utils._pytree as pytree
from torch import distributed as dist
from torch._C._functorch import _add_batch_dim, get_unwrapped, is_batchedtensor
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.source import LocalSource
from torch._dynamo.testing import make_test_cls_with_patches, rand_strided
from torch._guards import tracing, TracingContext
from torch._higher_order_ops.scan import scan
from torch._subclasses.fake_tensor import (
    _CacheKeyState,
    _check_for_subclass_arg,
    CppFakeTensorMode,
    DynamicOutputShapeException,
    extract_tensor_metadata,
    FakeTensor,
    FakeTensorConverter,
    FakeTensorDeviceMismatchError,
    FakeTensorMode,
    in_kernel_invocation_manager,
    is_fake_tensor,
    maybe_get_fake_constant,
    maybe_get_fake_device,
    maybe_get_fake_dispatch_keys,
    maybe_get_fake_mode,
    maybe_get_item_memo,
    maybe_get_real_tensor,
    MetadataMismatchError,
    unset_fake_temporarily,
    UnsupportedOperatorException,
)
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import (
    DimDynamic,
    free_symbols,
    free_unbacked_symbols,
    guarding_hint_or_throw,
    GuardOnDataDependentSymNode,
    ShapeEnv,
    ShapeEnvSettings,
    StatefulSymbolicContext,
    StatelessSymbolicContext,
    statically_known_true,
)
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FLASH_ATTENTION
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    onlyAccelerator,
    OpDTypes,
    ops,
    skipXPUIf,
)
from torch.testing._internal.common_dtype import all_types_complex_float8_and
from torch.testing._internal.common_utils import (
    expectedIfCppFakeTensor,
    fake_tensor_modes,
    HardwareClassification,
    instantiate_parametrized_tests,
    IS_LINUX,
    parametrize,
    run_tests,
    skipIfCppFakeTensor,
    skipIfCrossRef,
    skipIfTorchDynamo,
    skipIfWindows,
    subtest,
    TemporaryFileName,
    TEST_ACCELERATOR,
    TEST_WITH_ROCM,
    TEST_WITH_SLOW,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
    xfailIf,
    xfailIfTorchDynamo,
)
from torch.testing._internal.custom_op_db import custom_op_db
from torch.testing._internal.inductor_utils import GPU_TYPE
from torch.testing._internal.jit_utils import RUN_CUDA
from torch.testing._internal.two_tensor import TwoTensor
from torch.utils._mode_utils import no_dispatch
from torch.utils._python_dispatch import (
    _disable_current_modes,
    _get_current_dispatch_mode,
    BaseTorchDispatchMode,
    TorchDispatchMode,
)


aten = torch.ops.aten

torch._dynamo.config.fake_tensor_cache_enabled = True
torch._dynamo.config.fake_tensor_cache_crosscheck_enabled = True

# Follow the library switch so the test harness (N/A skips, cpp mode
# construction, cpp-specific assertions) always matches how FakeTensorMode is
# actually built, rather than reading the env var with a different default.
CPP_FAKETENSOR = torch._dynamo.config.use_cpp_fake_tensor

if CPP_FAKETENSOR:

    def FakeTensorMode(
        *,
        shape_env=None,
        allow_fallback_kernels=True,
        allow_non_fake_inputs=False,
        static_shapes=None,
        **kwargs,
    ):
        if not allow_fallback_kernels or kwargs:
            raise NotImplementedError("CppFakeTensorMode has no such options yet")
        return CppFakeTensorMode(
            shape_env=shape_env,
            allow_non_fake_inputs=allow_non_fake_inputs,
            static_shapes=static_shapes,
        )

    FakeTensorModeCls = CppFakeTensorMode
else:
    FakeTensorModeCls = FakeTensorMode

_CPP_FAKE_DYNAMO_SKIP = "Dynamo does not support C++ fakes as inputs yet"

# A conv with cpu inputs and xpu weights dispatches to aten::_slow_conv2d_forward, which
# is unimplemented on XPU, so the expected device-mismatch error never surfaces.
XPU_SLOW_CONV2D_SKIP = "XPU lacks aten::_slow_conv2d_forward for mixed cpu/xpu conv, see https://github.com/intel/torch-xpu-ops/issues/5125"


def expectedFailurePropagateRealTensors(fn):
    fn._expected_failure_propagate_real_tensors = True
    return fn


def skipIfPropagateRealTensors(reason):
    def decorator(fn):
        fn._skip_propagate_real_tensors = reason
        return fn

    return decorator


class _TestFakeTensorHelpers:
    """Assertions shared by TestFakeTensor and TestFakeTensorDevice."""

    def checkType(self, t, device_str, size):
        self.assertTrue(is_fake_tensor(t))
        self.assertEqual(t.device.type, device_str)
        self.assertEqual(list(t.size()), size)

    def checkMetaProps(self, t1, t2):
        prims.utils.compare_tensor_meta(t1, t2, check_strides=True)

    def fake_from_meta(self, mode, meta, device):
        if isinstance(mode, CppFakeTensorMode):
            return torch._C._from_meta_and_device(meta, torch.device(device), mode)
        return FakeTensor(mode, meta, torch.device(device))


class TestFakeTensor(_TestFakeTensorHelpers, TestCase):
    hw_classification = HardwareClassification.GENERIC

    @fake_tensor_modes
    def test_fake_mode_dispatch_api(self, mode_cls):
        mode = mode_cls()
        mode_key = torch._C._TorchDispatchModeKey.FAKE
        with mode:
            self.assertIs(torch._C._get_dispatch_mode(mode_key), mode)
            unset_mode = torch._C._unset_dispatch_mode(mode_key)
            self.assertIs(unset_mode, mode)
            self.assertIsNone(torch._C._get_dispatch_mode(mode_key))
            torch._C._set_dispatch_mode(unset_mode)
            self.assertIs(torch._C._get_dispatch_mode(mode_key), mode)
        self.assertIsNone(torch._C._get_dispatch_mode(mode_key))

    def test_like_out_overload_meta_out(self):
        # An out overload returns its out argument, a fake already
        with torch._subclasses.fake_tensor.FakeTensorMode():
            x = torch.empty(2, device="meta")
            out = torch.empty(2, device="meta")
            self.assertIs(torch.ops.aten.ones_like.out(x, out=out), out)

    def fake_with_unbacked_batch(self, *tensors):
        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True, shape_env=shape_env)
        fakes = []
        # These are fake inputs created before tracing, not fresh op outputs for make_fx.
        with fake_mode, shape_env.ignore_fresh_unbacked_symbols():
            for i, tensor in enumerate(tensors):
                fakes.append(
                    fake_mode.from_tensor(
                        tensor,
                        source=LocalSource(f"arg{i}", is_input=True),
                        symbolic_context=StatelessSymbolicContext(
                            dynamic_sizes=[
                                DimDynamic.UNBACKED,
                                *[DimDynamic.STATIC] * (tensor.dim() - 1),
                            ],
                        ),
                    )
                )
        return fake_mode, fakes

    @fake_tensor_modes
    @parametrize(
        "op",
        [torch.add, torch.matmul, aten.mse_loss.default],
        name_fn=lambda op: op.__name__.split(".")[0],
    )
    @parametrize("inference", [False, True])
    def test_basic(self, mode_cls, op, inference):
        if mode_cls is CppFakeTensorMode:
            if TEST_WITH_TORCHDYNAMO:
                self.skipTest(_CPP_FAKE_DYNAMO_SKIP)
            if torch._functorch.config.fake_tensor_propagate_real_tensors:
                self.skipTest(
                    "C++ fake fallback does not support propagate_real_tensors"
                )
        x = torch.empty(2, 2, device="cpu")
        y = torch.empty((2, 2) if op is torch.matmul else (4, 2, 2), device="cpu")
        expected = op(x, y)
        with mode_cls() as mode, torch.inference_mode(inference):
            x = mode.from_tensor(x)
            y = mode.from_tensor(y)
            z = op(x, y)
            self.checkMetaProps(z, expected)
            self.assertTrue(is_fake_tensor(z))
            self.assertIs(maybe_get_fake_mode(z), mode)

    def test_sparse_compressed_tensor_creation(self):
        def csr():
            crow = torch.tensor([0, 2, 4])
            col = torch.tensor([0, 1, 0, 1])
            return torch.sparse_csr_tensor(crow, col, torch.randn(4), size=(2, 2))

        def csc():
            ccol = torch.tensor([0, 2, 4])
            row = torch.tensor([0, 1, 0, 1])
            return torch.sparse_csc_tensor(ccol, row, torch.randn(4), size=(2, 2))

        def bsr():
            crow = torch.tensor([0, 1, 2])
            col = torch.tensor([0, 1])
            return torch.sparse_bsr_tensor(crow, col, torch.randn(2, 2, 2), size=(4, 4))

        def bsc():
            ccol = torch.tensor([0, 1, 2])
            row = torch.tensor([0, 1])
            return torch.sparse_bsc_tensor(ccol, row, torch.randn(2, 2, 2), size=(4, 4))

        for fn in [csr, csc, bsr, bsc]:
            ref = fn()
            ref_values_shape = ref.values().shape
            with FakeTensorMode():
                t = fn()
                self.assertTrue(is_fake_tensor(t))
                self.assertEqual(t.layout, ref.layout)
                self.assertEqual(t.shape, ref.shape)
                self.assertEqual(t.device, ref.device)
                self.assertEqual(t.values().shape, ref_values_shape)
                d = t.to_dense()
                self.assertTrue(is_fake_tensor(d))
                self.assertEqual(d.shape, ref.shape)
                self.assertEqual(d.layout, torch.strided)

    @skipIfPropagateRealTensors("real pin_memory needs an accelerator")
    def test_sparse_compressed_tensor_creation_pin_memory(self):
        if torch._functorch.config.fake_tensor_propagate_real_tensors:
            self.skipTest("real pin_memory needs an accelerator")
        with FakeTensorMode():
            crow = torch.tensor([0, 2, 4])
            col = torch.tensor([0, 1, 0, 1])
            t = torch.sparse_csr_tensor(
                crow, col, torch.randn(4), size=(2, 2), pin_memory=True
            )
            self.assertTrue(is_fake_tensor(t))
            self.assertEqual(t.layout, torch.sparse_csr)
            self.assertEqual(t.shape, (2, 2))
            self.assertEqual(t.device, torch.device("cpu"))

    def test_nansum_nanmean_empty_dim(self):
        # nansum/nanmean reduce over all dimensions when dim=() or dim=[] is
        # passed, matching eager. The meta kernel used to preserve the input
        # shape instead. See https://github.com/pytorch/pytorch/issues/191188
        x = torch.randn(2, 3)
        for op in (torch.nansum, torch.nanmean):
            for dim in ((), []):
                for keepdim in (False, True):
                    eager = op(x, dim=dim, keepdim=keepdim)
                    with FakeTensorMode() as mode:
                        fake = op(mode.from_tensor(x), dim=dim, keepdim=keepdim)
                    self.assertEqual(fake.shape, eager.shape)

    def test_inplace_non_broadcastable_raises(self):
        # Ops decomposed via _make_inplace used to silently resize the fake
        # self tensor to the broadcast shape instead of raising like eager.
        # See https://github.com/pytorch/pytorch/issues/191283
        with FakeTensorMode():
            for name in ["pow_", "atan2_", "eq_", "fmod_", "remainder_", "add_"]:
                a = torch.empty(1)
                b = torch.empty(2, 1, 2, 1)
                msg = "doesn't match the broadcast shape"
                with self.assertRaisesRegex(RuntimeError, msg):
                    getattr(a, name)(b)
                self.assertEqual(a.shape, (1,))

    @skipIfPropagateRealTensors("Propagate real tensor not supported")
    def test_mm_out_dtype(self):
        # The out_dtype dtype restriction in mm/bmm/baddbmm is a property of the
        # in-tree CUDA/XPU backends. Out-of-tree backends may support arbitrary
        # out_dtype combinations

        if torch._functorch.config.fake_tensor_propagate_real_tensors:
            self.skipTest("Propagate real tensor not supported")

        with FakeTensorMode():
            a = torch.empty((8, 2048), dtype=torch.float8_e4m3fn, device="cpu")
            b = torch.empty((2048, 768), dtype=torch.float8_e4m3fn, device="cpu")
            out = torch.mm(a, b, out_dtype=torch.float32)
            self.assertEqual(out.dtype, torch.float32)
            self.assertEqual(out.shape, (8, 768))

            a3 = torch.empty((2, 8, 2048), dtype=torch.float8_e4m3fn, device="cpu")
            b3 = torch.empty((2, 2048, 768), dtype=torch.float8_e4m3fn, device="cpu")
            out3 = torch.bmm(a3, b3, out_dtype=torch.float32)
            self.assertEqual(out3.dtype, torch.float32)
            self.assertEqual(out3.shape, (2, 8, 768))

        # On CUDA the restriction must still be enforced by the meta kernel.
        with FakeTensorMode(allow_non_fake_inputs=True):
            a = torch.empty((8, 2048), dtype=torch.float8_e4m3fn, device="cuda")
            b = torch.empty((2048, 768), dtype=torch.float8_e4m3fn, device="cuda")
            with self.assertRaisesRegex(RuntimeError, "out_dtype must be the same"):
                torch.mm(a, b, out_dtype=torch.float32)

            # fp16/bf16 -> fp32 is still allowed on CUDA.
            c = torch.empty((8, 2048), dtype=torch.bfloat16, device="cuda")
            d = torch.empty((2048, 768), dtype=torch.bfloat16, device="cuda")
            out = torch.mm(c, d, out_dtype=torch.float32)
            self.assertEqual(out.dtype, torch.float32)

    def test_sparse_spdiags(self):
        with FakeTensorMode(allow_non_fake_inputs=True, shape_env=ShapeEnv()):
            out = torch.sparse.spdiags(torch.randn(2, 3), torch.tensor([0, -1]), (2, 3))

        self.assertTrue(is_fake_tensor(out))
        self.assertEqual(out.layout, torch.sparse_coo)
        self.assertEqual(out.shape, (2, 3))

    @skipIfPropagateRealTensors(
        "propagate_real_tensors does not support sparse strides"
    )
    def test_sparse_spdiags_compressed_layouts(self):
        if torch._functorch.config.fake_tensor_propagate_real_tensors:
            self.skipTest("propagate_real_tensors does not support sparse strides")

        real_offsets = torch.tensor([0, -1])
        with FakeTensorMode(allow_non_fake_inputs=True, shape_env=ShapeEnv()) as mode:
            offsets = mode.fake_tensor_converter.from_real_tensor(
                mode, real_offsets, make_constant=True
            )
            csr_out = torch.sparse.spdiags(
                torch.randn(2, 3),
                offsets,
                (2, 3),
                layout=torch.sparse_csr,
            )
            csc_out = torch.sparse.spdiags(
                torch.randn(2, 3),
                offsets,
                (2, 3),
                layout=torch.sparse_csc,
            )
            dynamic_offsets = torch.tensor([-1, -2])
            dynamic_csr_out = torch.sparse.spdiags(
                torch.randn(2, 100),
                dynamic_offsets,
                (100, 2),
                layout=torch.sparse_csr,
            )
            dynamic_csc_out = torch.sparse.spdiags(
                torch.randn(2, 100),
                dynamic_offsets,
                (100, 2),
                layout=torch.sparse_csc,
            )

        self.assertEqual(csr_out.layout, torch.sparse_csr)
        self.assertEqual(csc_out.layout, torch.sparse_csc)
        self.assertEqual(dynamic_csr_out.layout, torch.sparse_csr)
        self.assertEqual(dynamic_csc_out.layout, torch.sparse_csc)
        self.assertEqual(dynamic_csr_out.crow_indices().shape, (101,))
        self.assertEqual(
            dynamic_csr_out.col_indices().shape, dynamic_csr_out.values().shape
        )
        self.assertEqual(dynamic_csc_out.ccol_indices().shape, (3,))
        self.assertEqual(
            dynamic_csc_out.row_indices().shape, dynamic_csc_out.values().shape
        )

    @skipIfPropagateRealTensors("propagate_real_tensors resolves the real nnz")
    def test_sparse_spdiags_dynamic_nnz_bound(self):
        if torch._functorch.config.fake_tensor_propagate_real_tensors:
            self.skipTest("propagate_real_tensors resolves the real nnz")

        with FakeTensorMode(allow_non_fake_inputs=True, shape_env=ShapeEnv()) as mode:
            out = torch.sparse.spdiags(
                torch.randn(2, 100), torch.tensor([-1, -2]), (100, 2)
            )

        nnz = out._values().shape[0]
        nnz_range = mode.shape_env.var_to_range[nnz.node.expr]
        self.assertEqual(nnz_range.lower, 0)
        self.assertEqual(nnz_range.upper, 200)

    def test_sparse_spdiags_coalesced_metadata(self):
        zero_nnz_offsets = torch.tensor([3])
        one_nnz_offsets = torch.tensor([0])
        with FakeTensorMode(allow_non_fake_inputs=True, shape_env=ShapeEnv()) as mode:
            zero_nnz_offsets = mode.fake_tensor_converter.from_real_tensor(
                mode, zero_nnz_offsets, make_constant=True
            )
            one_nnz_offsets = mode.fake_tensor_converter.from_real_tensor(
                mode, one_nnz_offsets, make_constant=True
            )
            zero_nnz = torch.sparse.spdiags(torch.randn(1, 3), zero_nnz_offsets, (3, 3))
            one_nnz = torch.sparse.spdiags(torch.randn(1, 1), one_nnz_offsets, (1, 1))
            two_nnz = torch.sparse.spdiags(torch.randn(1, 2), one_nnz_offsets, (2, 2))

        self.assertTrue(zero_nnz.is_coalesced())
        self.assertTrue(one_nnz.is_coalesced())
        self.assertFalse(two_nnz.is_coalesced())

    def test_sparse_spdiags_zero_dim_static_offsets(self):
        duplicate_offsets = torch.tensor([0, 0])
        out_of_range_offsets = torch.tensor([4])
        with FakeTensorMode(allow_non_fake_inputs=True, shape_env=ShapeEnv()) as mode:
            duplicate_offsets = mode.fake_tensor_converter.from_real_tensor(
                mode, duplicate_offsets, make_constant=True
            )
            out_of_range_offsets = mode.fake_tensor_converter.from_real_tensor(
                mode, out_of_range_offsets, make_constant=True
            )

            with self.assertRaisesRegex(
                RuntimeError, "Offset tensor contains duplicate values"
            ):
                torch.sparse.spdiags(torch.randn(2, 3), duplicate_offsets, (0, 3))

            out = torch.sparse.spdiags(torch.randn(1, 3), out_of_range_offsets, (0, 3))

        self.assertEqual(out.layout, torch.sparse_coo)
        self.assertEqual(out.shape, (0, 3))
        self.assertTrue(out.is_coalesced())

    def test_sparse_spdiags_static_offset_errors(self):
        duplicate_offsets = torch.tensor([0, 0])
        out_of_range_offsets = torch.tensor([4])
        with FakeTensorMode(allow_non_fake_inputs=True, shape_env=ShapeEnv()) as mode:
            duplicate_offsets = mode.fake_tensor_converter.from_real_tensor(
                mode, duplicate_offsets, make_constant=True
            )
            out_of_range_offsets = mode.fake_tensor_converter.from_real_tensor(
                mode, out_of_range_offsets, make_constant=True
            )
            with self.assertRaisesRegex(
                RuntimeError, "Offset tensor contains duplicate values"
            ):
                torch.sparse.spdiags(torch.randn(2, 3), duplicate_offsets, (2, 3))

            with self.assertRaisesRegex(RuntimeError, "negative dimension"):
                torch.sparse.spdiags(torch.randn(1, 3), out_of_range_offsets, (3, 3))

    # _spdiags allocates unbacked sizes, which the C++ fallback rejects.
    @xfailIf(CPP_FAKETENSOR)
    def test_sparse_spdiags_cuda_not_supported(self):
        diagonals_meta = torch.empty(2, 3, device="meta")
        offsets_meta = torch.empty(2, dtype=torch.long, device="meta")
        with FakeTensorMode() as fake_mode:
            diagonals = fake_mode.fake_tensor_converter.from_meta_and_device(
                fake_mode, diagonals_meta, torch.device("cuda")
            )
            offsets = fake_mode.fake_tensor_converter.from_meta_and_device(
                fake_mode, offsets_meta, torch.device("cuda")
            )
            with self.assertRaisesRegex(
                RuntimeError, "_spdiags is only implemented for CPU tensors"
            ):
                torch.sparse.spdiags(diagonals, offsets, (2, 3))

    def test_sparse_spdiags_compile_backend_eager(self):
        def fn():
            diags = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            return torch.sparse.spdiags(
                diags,
                offsets=torch.tensor([0, -1]),
                shape=(diags.size(0), diags.size(1)),
            )

        self.assertEqual(torch.compile(fn, backend="eager")(), fn())

    def test_custom_op_fallback(self):
        from torch.library import _scoped_library, impl

        with _scoped_library("my_test_op", "DEF") as test_lib:
            test_lib.define("foo(Tensor self) -> Tensor")

            @impl(test_lib, "foo", "CPU")
            def foo_impl(self):
                return self.cos()

            x = torch.empty(2, 2, device="cpu")
            exc, msg = expectedIfCppFakeTensor(
                (RuntimeError, "my_test_op::foo"),
                (UnsupportedOperatorException, "my_test_op.foo.default"),
            )
            with self.assertRaisesRegex(exc, msg):
                with FakeTensorMode(allow_fallback_kernels=True) as mode:
                    x = mode.from_tensor(x)
                    torch.ops.my_test_op.foo(x)

    def test_custom_op_fake_impl_with_parameter_subclass(self):
        class ParameterSubclass(torch.nn.Parameter):
            pass

        @torch.library.custom_op(
            "_torch_testing::fake_tensor_parameter_subclass", mutates_args=()
        )
        def custom_op(weight: torch.Tensor) -> torch.Tensor:
            return weight.clone()

        @custom_op.register_fake
        def custom_op_fake(weight: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(weight)

        weight = ParameterSubclass(torch.randn(10, 10))
        with self.assertRaisesRegex(
            (AssertionError, RuntimeError),
            "Please convert all Tensors to FakeTensors first",
        ):
            with FakeTensorMode():
                torch.ops._torch_testing.fake_tensor_parameter_subclass(weight)

        with FakeTensorMode(allow_non_fake_inputs=True):
            result = torch.ops._torch_testing.fake_tensor_parameter_subclass(weight)

        self.assertTrue(is_fake_tensor(result))
        self.assertEqual(result.shape, weight.shape)
        self.assertEqual(result.device, torch.device("cpu"))

    @fake_tensor_modes
    def test_parameter_subclass_with_torch_dispatch_gets_subclass_dispatch(
        self, mode_cls
    ):
        if (
            mode_cls is CppFakeTensorMode
            and torch._functorch.config.fake_tensor_propagate_real_tensors
        ):
            self.skipTest("C++ fake fallback does not support propagate_real_tensors")
        dispatch_calls = []
        fake_impl_calls = []

        @torch.library.custom_op(
            "_torch_testing::fake_tensor_dispatch_parameter_subclass",
            mutates_args=(),
        )
        def custom_op(weight: torch.Tensor) -> torch.Tensor:
            return weight.clone()

        @custom_op.register_fake
        def custom_op_fake(weight: torch.Tensor) -> torch.Tensor:
            fake_impl_calls.append(True)
            return torch.empty_like(weight)

        op = custom_op._opoverload

        class DispatchParameter(torch.nn.Parameter):
            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                dispatch_calls.append(func)
                if func is op:
                    weight = args[0]
                    return torch.empty(
                        weight.shape, dtype=weight.dtype, device=weight.device
                    )
                return NotImplemented

        weight = DispatchParameter(torch.randn(10, 10))
        with mode_cls() as mode:
            result = op(weight)

        self.assertTrue(is_fake_tensor(result))
        self.assertIs(maybe_get_fake_mode(result), mode)
        self.assertEqual(result.shape, weight.shape)
        self.assertEqual(result.device, torch.device("cpu"))
        self.assertEqual(dispatch_calls, [op])
        self.assertEqual(fake_impl_calls, [])

    def test_check_for_subclass_arg_parameter_subclass(self):
        class ParameterSubclass(torch.nn.Parameter):
            pass

        class DispatchParameter(torch.nn.Parameter):
            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        self.assertFalse(_check_for_subclass_arg(torch.randn(10)))
        self.assertFalse(_check_for_subclass_arg(torch.nn.Parameter(torch.randn(10))))
        self.assertFalse(_check_for_subclass_arg(ParameterSubclass(torch.randn(10))))
        self.assertTrue(_check_for_subclass_arg(DispatchParameter(torch.randn(10))))

        with FakeTensorMode() as mode:
            fake = mode.from_tensor(torch.randn(10))
        self.assertFalse(_check_for_subclass_arg(fake))

        tt_param = torch.nn.Parameter(TwoTensor(torch.randn(4, 4), torch.randn(4, 4)))
        self.assertTrue(_check_for_subclass_arg(tt_param))

    def test_parameter_instantiation(self):
        with FakeTensorMode():
            x = torch.rand([4])
            y = torch.nn.parameter.Parameter(x)
            self.assertTrue(isinstance(y, torch.nn.Parameter))
            # A fake Parameter is a detached fake of the same type with the
            # Parameter marker; for a C++ fake that type is torch.Tensor
            self.assertIs(type(y), type(x))
            self.assertTrue(y._is_param)
            self.assertIsNot(y, x)
            self.assertFalse(torch.nn.Parameter(x, requires_grad=False).requires_grad)

    @unittest.skipIf(
        not CPP_FAKETENSOR, "lazy modules cannot be built under this fake mode"
    )
    def test_parameter_from_lazy_module_parameter(self):
        # A lazy module's UninitializedParameter needs _make_subclass on a fake,
        # which fails for a Python FakeTensor.
        if not CPP_FAKETENSOR:
            self.skipTest("lazy modules cannot be built under this fake mode")
        with FakeTensorMode():
            lin = torch.nn.LazyLinear(4)
            lin(torch.randn(2, 3))
            # The materialized weight's exact type is Parameter
            self.assertIs(type(lin.weight), torch.nn.Parameter)
            tied = torch.nn.Parameter(lin.weight)
        self.assertTrue(is_fake_tensor(tied))
        self.assertIsInstance(tied, torch.nn.Parameter)
        # tied gets the marker form; lin.weight is left as it was
        self.assertIs(type(tied), torch.Tensor)
        self.assertTrue(tied._is_param)
        self.assertIsNot(tied, lin.weight)
        self.assertIs(type(lin.weight), torch.nn.Parameter)

    @skipIfTorchDynamo("uses torch.compile")
    def test_compile_module_with_fake_parameters(self):
        # The parameter guards must accept the same fake parameters on later calls
        mode = FakeTensorMode()
        with mode:
            m = torch.nn.Linear(4, 4)
            x = torch.randn(2, 4)
        cnt = torch._dynamo.testing.CompileCounter()
        f = torch.compile(m, backend=cnt)
        try:
            with mode:
                for _ in range(3):
                    f(x)
            self.assertEqual(cnt.frame_count, 1)
        finally:
            torch._dynamo.reset()

    @skipIfTorchDynamo("uses torch.compile")
    @unittest.skipIf(
        not CPP_FAKETENSOR, "TENSOR_MATCH already tells a Python FakeTensor apart"
    )
    def test_compile_parameter_subclass_with_fake_input(self):
        # Parameter.__new__ treats a C++ fake differently, so a graph traced for
        # a real input must not be reused for one
        if not CPP_FAKETENSOR:
            self.skipTest("TENSOR_MATCH already tells a Python FakeTensor apart")

        class MyParam(torch.nn.Parameter):
            pass

        def fn(x):
            return MyParam(x, requires_grad=False)

        cnt = torch._dynamo.testing.CompileCounter()
        f = torch.compile(fn, backend=cnt, fullgraph=True)
        try:
            self.assertIs(type(f(torch.randn(2))), MyParam)
            with FakeTensorMode():
                out = f(torch.randn(2))
            self.assertIs(type(out), torch.Tensor)
            self.assertTrue(out._is_param)
            self.assertEqual(cnt.frame_count, 2)
            self.assertIs(type(f(torch.randn(2))), MyParam)
            self.assertEqual(cnt.frame_count, 2)
        finally:
            torch._dynamo.reset()

    @skipIfCppFakeTensor("exercises Python FakeTensor parameter reconstruction")
    def test_parameter_state_reconstruction_from_dict(self):
        with FakeTensorMode():
            param = torch.nn.Conv2d(3, 4, 1).weight

        fake_device = param.fake_device
        param_cls = type(param)
        kwargs = dict(param.__dict__)
        kwargs["requires_grad"] = param.requires_grad

        reconstructed = param_cls(param.to("meta"), **kwargs)

        self.assertTrue(is_fake_tensor(reconstructed))
        self.assertIsInstance(reconstructed, torch.nn.Parameter)
        self.assertEqual(reconstructed.requires_grad, param.requires_grad)
        self.assertEqual(reconstructed.fake_device, fake_device)
        self.assertEqual(reconstructed.device, fake_device)
        self.assertIs(reconstructed.fake_mode, param.fake_mode)
        self.assertIn("_fake_device", reconstructed.__dict__)
        self.assertNotIn("fake_device", reconstructed.__dict__)
        self.assertEqual(reconstructed.to("meta").fake_device, torch.device("meta"))

    @skipIfCppFakeTensor("exercises Python FakeTensor parameter reconstruction")
    def test_parameter_state_reconstruction_rejects_device_conflict(self):
        with FakeTensorMode() as mode:
            elem = torch.empty(2, 2, device="meta")
            with self.assertRaisesRegex(ValueError, "conflicting device values"):
                FakeTensor(
                    mode,
                    elem,
                    device=torch.device("cpu"),
                    fake_device=torch.device("cuda:0"),
                )

    @skipIfCppFakeTensor("exercises Python FakeTensor parameter reconstruction")
    def test_parameter_state_reconstruction_accepts_normalized_device_aliases(self):
        with FakeTensorMode() as mode:
            elem = torch.empty(2, 2, device="meta")
            fake = FakeTensor(
                mode,
                elem,
                device=torch.device("cuda"),
                fake_device=torch.device("cuda:0"),
            )

        self.assertEqual(fake.fake_device, torch.device("cuda:0"))

    @skipIfCppFakeTensor("exercises Python FakeTensor parameter reconstruction")
    @parametrize("device_kwarg_name", ["fake_device", "device"])
    def test_parameter_state_reconstruction_device_names(self, device_kwarg_name):
        with FakeTensorMode():
            param = torch.nn.Conv2d(3, 4, 1).weight

        kwargs = dict(param.__dict__)
        fake_device = kwargs.pop("_fake_device")
        kwargs[device_kwarg_name] = fake_device
        kwargs["requires_grad"] = param.requires_grad

        reconstructed = type(param)(param.to("meta"), **kwargs)

        self.assertTrue(is_fake_tensor(reconstructed))
        self.assertIsInstance(reconstructed, torch.nn.Parameter)
        self.assertEqual(reconstructed.requires_grad, param.requires_grad)
        self.assertEqual(reconstructed.fake_device, fake_device)
        self.assertEqual(reconstructed.device, fake_device)
        self.assertIs(reconstructed.fake_mode, param.fake_mode)
        self.assertIn("_fake_device", reconstructed.__dict__)
        self.assertNotIn("fake_device", reconstructed.__dict__)
        self.assertEqual(reconstructed.to("meta").fake_device, torch.device("meta"))

    @unittest.skipIf(not dist.is_available(), "requires distributed")
    def test_fsdp_flat_param(self):
        from torch.distributed.fsdp._flat_param import FlatParameter

        with FakeTensorMode() as m:
            data = torch.randn(2, 2)
            param = FlatParameter(data, requires_grad=True)
        self.assertIsInstance(param, FlatParameter)
        self.assertIsInstance(param, torch.nn.Parameter)
        self.assertTrue(is_fake_tensor(param))

    @fake_tensor_modes
    @parametrize("leaf", [False, True])
    def test_non_parameter_grad(self, mode_cls, leaf):
        mode = mode_cls()
        t = torch.rand([4], requires_grad=True)
        if not leaf:
            t = t * 2
        fake_t = mode.from_tensor(t)
        self.assertEqual(fake_t.requires_grad, t.requires_grad)
        self.assertEqual(fake_t.is_leaf, t.is_leaf)
        self.assertEqual(fake_t.device, t.device)
        self.assertIs(maybe_get_fake_mode(fake_t), mode)

    def test_non_parameter_grad_tensor_subclass_stateful_context(self):
        mode = FakeTensorMode(shape_env=ShapeEnv())
        t = torch.ones(2, requires_grad=True)
        t.grad = TwoTensor(torch.ones(2), torch.ones(2))
        source = LocalSource("t", is_input=True)
        symbolic_context = StatefulSymbolicContext(
            dynamic_sizes=[DimDynamic.STATIC] * t.dim(),
            constraint_sizes=[None] * t.dim(),
            tensor_source=source,
        )

        fake_t = mode.from_tensor(t, source=source, symbolic_context=symbolic_context)

        self.assertIsInstance(fake_t.grad, TwoTensor)
        self.assertTrue(is_fake_tensor(fake_t.grad.a))
        self.assertTrue(is_fake_tensor(fake_t.grad.b))

    @skipIfCppFakeTensor("exercises Python FakeTensor scalar indexing")
    @skipIfTorchDynamo("test directly checks FakeTensorMode outputs")
    def test_scalar_tensor_index(self):
        for dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            with self.subTest(dtype=dtype):
                with FakeTensorMode() as mode:
                    x, idx = map(
                        mode.from_tensor,
                        (torch.empty(3, 4), torch.tensor(0, dtype=dtype)),
                    )
                    out = x[idx]
                self.checkType(out, "cpu", [4])

    def test_index_tensor_types_match_eager(self):
        # index and index_put reject the same index dtypes as eager, with the
        # same error
        ops = (
            lambda x, index, value: torch.ops.aten.index(x, [index]),
            lambda x, index, value: torch.ops.aten.index_put(x, [index], value),
            lambda x, index, value: torch.ops.aten.index_put_(x, [index], value),
            lambda x, index, value: torch.ops.aten._unsafe_index_put(x, [index], value),
        )
        for i, op in enumerate(ops):
            for dtype in (torch.int8, torch.int16, torch.float):
                with self.subTest(op=i, dtype=dtype):
                    args = (
                        torch.zeros(4),
                        torch.tensor([1, 0, 1, 0], dtype=dtype),
                        torch.tensor(1.0),
                    )
                    with self.assertRaises(IndexError) as eager:
                        op(*args)
                    # Under dynamo, the message continues with where it was raised
                    message = str(eager.exception).splitlines()[0]
                    with FakeTensorMode() as mode:
                        fake_args = [mode.from_tensor(a) for a in args]
                        with self.assertRaisesRegex(IndexError, re.escape(message)):
                            op(*fake_args)

    @fake_tensor_modes
    def test_static_scalar_tensor_with_shape_env_has_no_concrete_item_memo(
        self, mode_cls
    ):
        shape_env = ShapeEnv()
        for dtype in (torch.int64, torch.float64):
            with self.subTest(dtype=dtype):
                scalar = torch.tensor(0, dtype=dtype)
                with mode_cls(shape_env=shape_env) as mode:
                    scalar = mode.from_tensor(
                        scalar,
                        static_shapes=True,
                    )
                    self.assertIsNone(maybe_get_item_memo(scalar))

    @fake_tensor_modes
    def test_scalar_tensor_item_memo_invalidates_on_epoch(self, mode_cls):
        shape_env = ShapeEnv()
        scalar = torch.tensor(2, dtype=torch.int64)
        with mode_cls(shape_env=shape_env) as mode:
            scalar = mode.from_tensor(scalar, source=LocalSource("scalar"))
            item_memo = maybe_get_item_memo(scalar)
            self.assertIsInstance(item_memo, torch.SymInt)
            self.assertEqual(int(item_memo), 2)

            mode.epoch += 1

            self.assertIsNone(maybe_get_item_memo(scalar))

    def test_repr(self):
        name = expectedIfCppFakeTensor("tensor", "FakeTensor")
        with FakeTensorMode():
            x = torch.empty(2, 2, device="cpu")
            self.assertEqual(repr(x), f"{name}(..., size=(2, 2))")
            x = torch.empty(2, 2, device="meta")
            self.assertEqual(repr(x), f"{name}(..., device='meta', size=(2, 2))")

    @skipIfCppFakeTensor("exercises the Python fake_device property")
    def test_fake_device_property_normalization(self):
        """Test that fake_device property normalizes device on assignment."""
        with FakeTensorMode() as mode:
            # Test CPU device - should remain unchanged
            cpu_tensor = torch.empty(2, 2, device="meta")
            fake_cpu = FakeTensor(mode, cpu_tensor, torch.device("cpu"))
            self.assertEqual(fake_cpu.fake_device, torch.device("cpu"))
            fake_cpu_from_alias = FakeTensor(
                mode, cpu_tensor, fake_device=torch.device("cpu")
            )
            self.assertEqual(fake_cpu_from_alias.fake_device, torch.device("cpu"))

            # Test device with explicit index - should remain unchanged
            cuda_device_with_index = torch.device("cuda:0")
            fake_cuda = FakeTensor(mode, cpu_tensor, cuda_device_with_index)
            self.assertEqual(fake_cuda.fake_device, torch.device("cuda:0"))

            # Test MPS device without index - should normalize to mps:0
            mps_device = torch.device("mps")
            fake_mps = FakeTensor(mode, cpu_tensor, mps_device)
            self.assertEqual(fake_mps.fake_device, torch.device("mps:0"))

            # Test property setter normalization with MPS
            fake_tensor = FakeTensor(mode, cpu_tensor, torch.device("cpu"))
            fake_tensor.fake_device = torch.device("mps")
            self.assertEqual(fake_tensor.fake_device, torch.device("mps:0"))

            # Test property setter normalization with CUDA
            fake_tensor.fake_device = torch.device("cuda")
            self.assertEqual(fake_tensor.fake_device, torch.device("cuda:0"))

    def test_convert_fake_to_real(self):
        x = torch.ones([20])
        with FakeTensorMode(allow_non_fake_inputs=True) as m:
            _ = x + 1

        out = torch._subclasses.fake_utils.try_convert_fake_to_real([x[0:10]])

        self.assertEqual(torch.ones([10]), out[0])

    def test_conv_nhwc(self):
        x = torch.randn([1, 1024, 16, 16]).to(memory_format=torch.channels_last)
        w = torch.randn([256, 1024, 4, 4]).to(memory_format=torch.channels_last)
        b = torch.randn([256])

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, w, b):
                return torch.ops.aten.convolution(
                    x, w, b, [1, 1], [0, 0], [1, 1], False, [0, 0], 1
                )

        model = Model()
        with FakeTensorMode(allow_non_fake_inputs=True) as mode:
            fake_out = model.forward(x, w, b)
        eager_out = model.forward(x, w, b)
        self.assertEqual(fake_out.stride(), eager_out.stride())

    def test_private_convolution_symint(self):
        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True, shape_env=shape_env)
        x = fake_mode.from_tensor(
            torch.randn(1, 3, 16, 16),
            symbolic_context=StatelessSymbolicContext(
                dynamic_sizes=[
                    DimDynamic.DYNAMIC,
                    DimDynamic.STATIC,
                    DimDynamic.DYNAMIC,
                    DimDynamic.DYNAMIC,
                ],
                constraint_sizes=[None, None, None, None],
            ),
        )
        w = fake_mode.from_tensor(torch.randn(32, 3, 3, 3), static_shapes=True)

        with fake_mode:
            out_default = aten._convolution.default(
                x,
                w,
                None,
                [2, 2],
                [0, 0],
                [1, 1],
                False,
                [0, 0],
                1,
                False,
                False,
                True,
                True,
            )
            out_deprecated = aten._convolution.deprecated(
                x,
                w,
                None,
                [2, 2],
                [0, 0],
                [1, 1],
                False,
                [0, 0],
                1,
                False,
                False,
                True,
            )

        for out in (out_default, out_deprecated):
            self.assertEqual(out.shape[:2], (1, 32))
            self.assertTrue(
                statically_known_true(out.shape[2] == (x.shape[2] - 3) // 2 + 1)
            )
            self.assertTrue(
                statically_known_true(out.shape[3] == (x.shape[3] - 3) // 2 + 1)
            )

    @skipIfCppFakeTensor("no unspecified fake-device index support")
    def test_conv_allows_unspecified_fake_device_index(self):
        with FakeTensorMode() as mode:
            x = FakeTensor(
                mode,
                torch.empty(1, 3, 8, 8, device="meta"),
                torch.device("cpu"),
            )
            w = FakeTensor(
                mode,
                torch.empty(3, 3, 3, 3, device="meta"),
                torch.device("cpu:0"),
            )
            b = FakeTensor(
                mode,
                torch.empty(3, device="meta"),
                torch.device("cpu"),
            )

            torch.ops.aten.convolution.default(
                x, w, b, [1, 1], [1, 1], [1, 1], False, [0, 0], 1
            )

    def test_nan_to_num(self):
        with FakeTensorMode():
            for dtype in [torch.float16, torch.float32]:
                x = torch.rand([4], dtype=dtype)
                y = torch.nan_to_num(x, nan=None)
                z = torch.nan_to_num(x, 0.0)
                self.assertEqual(dtype, y.dtype)
                self.assertEqual(dtype, z.dtype)

    def test_fake_dispatch_keys(self):
        with FakeTensorMode():
            x = torch.rand([4])
            f = (
                FileCheck()
                .check("CPU")
                .check("ADInplaceOrView")
                .check("AutogradCPU")
                .check("AutocastCPU")
            )
            f.run(torch._C._dispatch_key_set(x))

            with torch.inference_mode():
                x = torch.rand([4])
                y = x + x
                FileCheck().check("CPU").check("AutocastCPU").run(
                    torch._C._dispatch_key_set(y)
                )
                FileCheck().check_not("ADInplaceOrView").check_not("Autograd").run(
                    torch._C._dispatch_key_set(y)
                )

    # Conversion of real views to C++ fakes is not implemented.
    @xfailIf(CPP_FAKETENSOR)
    def test_alias_fake_tensor(self):
        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)
        x = fake_mode.from_tensor(
            torch.randn(4, 6).t(),
            symbolic_context=StatelessSymbolicContext(
                dynamic_sizes=[DimDynamic.DYNAMIC, DimDynamic.DYNAMIC],
                constraint_sizes=[None, None],
            ),
        )
        self.assertIsInstance(x.shape[0], torch.SymInt)
        with fake_mode:
            y = torch.ops.aten.alias.default(x)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.stride(), x.stride())
        self.assertEqual(y.storage_offset(), x.storage_offset())
        self.assertEqual(y.untyped_storage()._cdata, x.untyped_storage()._cdata)

    @unittest.skipIf(not torch.backends.mkldnn.is_available(), "MKLDNN not available")
    @skipIfPropagateRealTensors("Propagate real tensor not supported")
    # mkldnn nyi in C++ fake tensors
    @xfailIf(CPP_FAKETENSOR)
    def test_alias_fake_mkldnn(self):
        if torch._functorch.config.fake_tensor_propagate_real_tensors:
            self.skipTest("Propagate real tensor not supported")
        with FakeTensorMode() as fake_mode:
            x = fake_mode.from_tensor(torch.randn(2, 3).to_mkldnn())
            y = torch.ops.aten.alias.default(x)
        self.assertTrue(y.is_mkldnn)
        self.assertEqual(torch._C._dispatch_key_set(y), torch._C._dispatch_key_set(x))

    @fake_tensor_modes
    def test_in_kernel_invocation_reports_meta_device(self, mode_cls):
        t = torch.empty(2, device="meta")
        torch._C._set_fake_device(t, torch.device("cpu"))
        self.assertEqual(t.device, torch.device("cpu"))
        fake_key = torch._C.DispatchKeySet(torch._C.DispatchKey.Fake)
        with torch._C._ExcludeDispatchKeyGuard(fake_key):
            self.assertEqual(t.device, torch.device("cpu"))
        with torch._C._InKernelInvocation():
            self.assertTrue(torch._C._in_kernel_invocation())
            self.assertEqual(t.device, torch.device("meta"))
        self.assertFalse(torch._C._in_kernel_invocation())
        self.assertEqual(t.device, torch.device("cpu"))
        with in_kernel_invocation_manager(mode_cls()):
            self.assertEqual(t.device, torch.device("meta"))
        self.assertEqual(t.device, torch.device("cpu"))

    @unittest.skipIf(not torch.backends.mkldnn.is_available(), "MKLDNN not available")
    # mkldnn nyi in C++ fake tensors
    @xfailIf(CPP_FAKETENSOR)
    def test_mkldnn_to_dense(self):
        from torch._subclasses.functional_tensor import (
            FunctionalTensor,
            FunctionalTensorMode,
        )

        if torch._functorch.config.fake_tensor_propagate_real_tensors:
            self.skipTest("Propagate real tensor not supported")
        real = torch.randn(2, 3).to_mkldnn()
        with FakeTensorMode() as fake_mode:
            x = fake_mode.from_tensor(real)
            self.assertTrue(x.is_mkldnn)
            self.assertEqual(x.layout, torch._mkldnn)  # type: ignore[attr-defined]

            y = x.to_dense()
            self.assertFalse(y.is_mkldnn)
            self.assertEqual(y.layout, torch.strided)
            self.assertEqual(y.stride(), (3, 1))

            y = torch.ops.aten.to_dense.default(x)
            self.assertFalse(y.is_mkldnn)
            self.assertEqual(y.layout, torch.strided)
            self.assertEqual(y.stride(), (3, 1))

            y = x.to_dense(torch.float8_e4m3fn)
            self.assertFalse(y.is_mkldnn)
            self.assertEqual(y.layout, torch.strided)
            self.assertEqual(y.dtype, torch.float8_e4m3fn)
            self.assertEqual(y.stride(), (3, 1))

            detached = x.detach()
            self.assertTrue(detached.is_mkldnn)
            y = detached.to_dense()
            self.assertFalse(y.is_mkldnn)
            self.assertEqual(y.layout, torch.strided)
            self.assertEqual(y.stride(), (3, 1))

            aliased = torch.ops.aten.alias.default(x)
            self.assertTrue(aliased.is_mkldnn)
            y = aliased.to_dense()
            self.assertFalse(y.is_mkldnn)
            self.assertEqual(y.layout, torch.strided)
            self.assertEqual(y.stride(), (3, 1))

            dense = torch.randn(2, 3)
            mkldnn = torch.ops.aten.to_mkldnn.default(dense)
            self.assertTrue(mkldnn.is_mkldnn)
            y = torch.ops.aten.to_dense.default(mkldnn)
            self.assertFalse(y.is_mkldnn)
            self.assertEqual(y.layout, torch.strided)
            self.assertEqual(y.stride(), (3, 1))

            with FunctionalTensorMode():
                functional = FunctionalTensor.to_functional(x)
                y = functional.detach().to_dense()
                y_unwrapped = torch._from_functional_tensor(y.elem)
                self.assertFalse(y_unwrapped.is_mkldnn)
                self.assertEqual(y_unwrapped.layout, torch.strided)
                self.assertEqual(y_unwrapped.stride(), (3, 1))

                functional = FunctionalTensor.to_functional(x)
                y = torch.ops.aten.alias.default(functional).to_dense()
                y_unwrapped = torch._from_functional_tensor(y.elem)
                self.assertFalse(y_unwrapped.is_mkldnn)
                self.assertEqual(y_unwrapped.layout, torch.strided)
                self.assertEqual(y_unwrapped.stride(), (3, 1))

    @unittest.skipIf(not torch.backends.mkldnn.is_available(), "MKLDNN not available")
    @skipIfPropagateRealTensors("Propagate real tensor not supported")
    # mkldnn nyi in C++ fake tensors
    @xfailIf(CPP_FAKETENSOR)
    def test_mkldnn_unsafe_fallback(self):
        if torch._functorch.config.fake_tensor_propagate_real_tensors:
            self.skipTest("Propagate real tensor not supported")
        with FakeTensorMode() as fake_mode:
            x = fake_mode.from_tensor(torch.randn(1, 2, 8, 8).to_mkldnn())
            y = torch.nn.functional.max_pool2d(x, 3, 2, 1)
            self.assertTrue(is_fake_tensor(y))
            self.assertEqual(y.shape, (1, 2, 4, 4))

    def test_compare_tensor_meta_unbacked_numel(self):
        from torch.fx.experimental.symbolic_shapes import _constrain_range_for_size

        shape_env = ShapeEnv()
        with FakeTensorMode(shape_env=shape_env):
            u0 = shape_env.create_unbacked_symint()
            _constrain_range_for_size(u0, min=0)
            a = torch.empty(u0, 16, device="cpu")
            b = torch.empty(u0, 16, device="cpu")
        prims.utils.compare_tensor_meta(a, b, check_strides=True)

    def test_stack_symbolic_shapes(self):
        # torch.stack, for a non-trailing dim, is computed as
        # cat(...).view(result_sizes). The inner view must not require concrete
        # sizes when the stacked tensors have symbolic shapes, otherwise it
        # raises "SymIntArrayRef expected to contain only concrete integers".
        shape_env = ShapeEnv()
        with FakeTensorMode(shape_env=shape_env):
            u0 = shape_env.create_unbacked_symint()
            u1 = shape_env.create_unbacked_symint()
            t = torch.empty(u0, u1, device="cpu")
            out = torch.stack([t, t], dim=1)
            self.assertEqual(out.dim(), 3)
            self.assertEqual(out.size(1), 2)

    def test_batch_tensor(self):
        x = torch.rand((3, 4, 5))
        b = _add_batch_dim(x, 0, 0)
        mode = FakeTensorMode()
        fake_b = mode.from_tensor(b)
        prims.utils.compare_tensor_meta(b, fake_b, check_strides=True)

        b1 = _add_batch_dim(x, 1, 1)
        b2 = _add_batch_dim(b1, 0, 2)
        fake_b2 = mode.from_tensor(b2)
        prims.utils.compare_tensor_meta(b2, fake_b2, check_strides=True)
        self.assertTrue(is_batchedtensor(fake_b2))
        fake_b1 = get_unwrapped(fake_b2)
        self.assertTrue(is_batchedtensor(fake_b1))
        fake_tensor = get_unwrapped(fake_b1)
        self.assertTrue(is_fake_tensor(fake_tensor))

    @fake_tensor_modes
    @parametrize(
        "op,args",
        [
            (torch.rand, ()),
            (torch.empty, ()),
            (torch.zeros, ()),
            (torch.full, (1,)),
            subtest(
                (
                    partial(
                        prims.empty_strided,
                        strides=(4, 1),
                        dtype=torch.float32,
                        requires_grad=False,
                    ),
                    (),
                ),
                name="prims_empty_strided",
            ),
        ],
        name_fn=lambda op, args: op.__name__,
    )
    def test_constructor(self, mode_cls, op, args):
        if (
            mode_cls is CppFakeTensorMode
            and torch._functorch.config.fake_tensor_propagate_real_tensors
        ):
            self.skipTest("C++ fake fallback does not support propagate_real_tensors")
        with mode_cls() as mode:
            x = op([4, 4], *args, device="cpu")

        self.assertTrue(is_fake_tensor(x))
        self.assertTrue(x.device.type == "cpu")
        self.assertEqual(x.shape, (4, 4))
        self.checkMetaProps(x, op([4, 4], *args, device="cpu"))
        self.assertIs(maybe_get_fake_mode(x), mode)

    def test_constructor_like_custom_op_without_device_arg(self):
        with torch.library._scoped_library(
            "fake_tensor_issue_163196", "FRAGMENT"
        ) as lib:
            lib.define("moo(int x) -> Tensor")

            @torch.library.impl(
                "fake_tensor_issue_163196::moo",
                "CompositeExplicitAutograd",
                lib=lib,
            )
            def moo(x):
                return torch.empty(3).fill_(x)

            def f(x):
                return torch.ops.fake_tensor_issue_163196.moo(x.item())

            gm = make_fx(f, tracing_mode="symbolic")(torch.tensor(4))
            target = torch.ops.fake_tensor_issue_163196.moo.default
            (node,) = [n for n in gm.graph.nodes if n.target is target]
            self.assertTrue(is_fake_tensor(node.meta["val"]))
            self.assertEqual(node.meta["val"].shape, (3,))
            self.assertEqual(node.meta["val"].device.type, "cpu")

    @fake_tensor_modes
    def test_constructor_like_custom_op_with_device_arg(self, mode_cls):
        if (
            mode_cls is CppFakeTensorMode
            and torch._functorch.config.fake_tensor_propagate_real_tensors
        ):
            self.skipTest("C++ fake fallback does not support propagate_real_tensors")
        with torch.library._scoped_library(
            "fake_tensor_issue_163196_device_arg", "FRAGMENT"
        ) as lib:
            lib.define("moo(int x, *, Device? device=None) -> Tensor")
            seen_devices = []

            @torch.library.impl(
                "fake_tensor_issue_163196_device_arg::moo",
                "CompositeExplicitAutograd",
                lib=lib,
            )
            def moo(x, device=None):
                self.assertIsNone(torch._guards.active_fake_mode())
                seen_devices.append(device)
                return torch.empty(3, device=device).fill_(x)

            with mode_cls():
                out = torch.ops.fake_tensor_issue_163196_device_arg.moo(4, device="cpu")

            self.assertTrue(is_fake_tensor(out))
            self.assertEqual(out.device.type, "cpu")
            self.assertTrue(torch.device("meta") in seen_devices)

    def test_constructor_like_custom_op_with_non_device_arg_named_device(self):
        with torch.library._scoped_library(
            "fake_tensor_issue_163196_non_device_arg", "FRAGMENT"
        ) as lib:
            lib.define("moo(str device) -> Tensor")

            @torch.library.impl(
                "fake_tensor_issue_163196_non_device_arg::moo",
                "CompositeExplicitAutograd",
                lib=lib,
            )
            def moo(device):
                return torch.empty(len(device))

            def f(x):
                return torch.ops.fake_tensor_issue_163196_non_device_arg.moo("cpu")

            gm = make_fx(f, tracing_mode="symbolic")(torch.tensor(4))
            target = torch.ops.fake_tensor_issue_163196_non_device_arg.moo.default
            (node,) = [n for n in gm.graph.nodes if n.target is target]
            self.assertTrue(is_fake_tensor(node.meta["val"]))
            self.assertEqual(node.meta["val"].shape, (3,))
            self.assertEqual(node.meta["val"].device.type, "cpu")

    @fake_tensor_modes
    @parametrize(
        "op",
        [
            torch.nn.functional.hardswish,
            torch.square,
            subtest(lambda x: x.view(2, 2), name="view"),
            subtest(lambda x: x.select(0, 1), name="select"),
            subtest(lambda x: x.detach(), name="detach"),
            subtest(lambda x: x.data, name="data"),
            subtest(lambda x: aten.copy(x, x), name="copy"),
            subtest(lambda x: x[x.new_zeros(2, dtype=torch.int64)], name="index"),
            subtest(
                lambda x: aten.index_put(
                    x, [x.new_zeros(2, dtype=torch.int64)], x.new_ones(2)
                ),
                name="index_put",
            ),
            subtest(lambda x: torch._foreach_add([x, x], [x, x]), name="foreach_add"),
            subtest(aten.log_sigmoid_forward.default, name="log_sigmoid_forward"),
            subtest(lambda x: prims.add(x, x), name="prims_add"),
            subtest(
                lambda x: prims.convert_element_type(x, torch.float64),
                name="prims_convert_element_type",
            ),
        ],
        name_fn=lambda op: op.__name__,
    )
    @parametrize("active", [False, True])
    @parametrize("inference", [False, True])
    @skipIfTorchDynamo(_CPP_FAKE_DYNAMO_SKIP)
    def test_mode(self, mode_cls, op, active, inference):
        if (
            mode_cls is CppFakeTensorMode
            and torch._functorch.config.fake_tensor_propagate_real_tensors
        ):
            self.skipTest("C++ fake fallback does not support propagate_real_tensors")
        real = torch.rand([4], device="cpu")
        mode = mode_cls()
        with torch.inference_mode(inference):
            fake = mode.from_tensor(real)
        with (
            mode if active else contextlib.nullcontext(),
            torch.inference_mode(inference),
        ):
            out = op(fake)
        for result, expected in zip(
            pytree.tree_leaves(out), pytree.tree_leaves(op(real)), strict=True
        ):
            self.checkMetaProps(result, expected)
            self.assertTrue(is_fake_tensor(result))
            self.assertIs(maybe_get_fake_mode(result), mode)
            self.assertEqual(result.storage_offset(), expected.storage_offset())
            self.assertEqual(
                torch._C._is_alias_of(result, fake),
                torch._C._is_alias_of(expected, real),
            )
            with mode:
                self.assertEqual(aten.size.default(result), list(expected.shape))
                self.assertEqual(aten.stride.default(result), list(expected.stride()))
                self.assertEqual(
                    aten.storage_offset.default(result), expected.storage_offset()
                )
                self.assertEqual(torch.ops.prim.device.default(result), expected.device)

    def test_full(self):
        # Test torch.full returns tensor with correct dtype
        with torch._subclasses.CrossRefFakeMode():
            y = torch.full((4, 4), 1)

    def test_tensor_constructor_meta_device(self):
        eager_indexed = torch.tensor([[1.0, 2.0], [3.0]], device="meta:0")
        eager_values = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="meta")

        self.assertEqual(eager_indexed.device, torch.device("meta"))
        self.assertEqual(eager_indexed.shape, (2, 2))
        self.assertEqual(eager_values.shape, (2, 2))

        with FakeTensorMode(), torch.device("meta"):
            scalar = torch.tensor(3.0)
            values = torch.tensor([1.0, 2.0, 3.0])

        self.assertTrue(is_fake_tensor(scalar))
        self.assertEqual(scalar.device, torch.device("meta"))
        self.assertEqual(scalar.shape, ())
        self.assertTrue(is_fake_tensor(values))
        self.assertEqual(values.device, torch.device("meta"))
        self.assertEqual(values.shape, (3,))

        with FakeTensorMode():
            explicit = torch.tensor(3.0, device="meta")
            explicit_indexed = torch.tensor(3.0, device="meta:0")

        self.assertTrue(is_fake_tensor(explicit))
        self.assertEqual(explicit.device, torch.device("meta"))
        self.assertEqual(explicit.shape, ())
        self.assertTrue(is_fake_tensor(explicit_indexed))
        self.assertEqual(explicit_indexed.device, torch.device("meta"))
        self.assertEqual(explicit_indexed.shape, ())

    @parametrize("device", ("meta", "meta:0"))
    def test_tensor_constructor_meta_device_from_storage(self, device):
        storage = torch.tensor([1.0, 2.0]).storage()

        with patch.object(FakeTensorModeCls, "avoid_device_init", True):
            with FakeTensorMode():
                tensor = torch.tensor(storage, device=device)

        self.assertTrue(is_fake_tensor(tensor))
        self.assertEqual(tensor.device, torch.device("meta"))
        self.assertEqual(tensor.shape, (2,))

    @expectedFailurePropagateRealTensors
    @parametrize(
        "device,expected_device",
        (("cpu", "cpu"), ("cpu:0", "cpu"), ("cuda:1", "cuda:1")),
    )
    def test_tensor_constructor_meta_storage_device(self, device, expected_device):
        storage = torch.empty(2, device="meta").storage()

        with patch.object(FakeTensorModeCls, "avoid_device_init", True):
            with FakeTensorMode():
                tensor = torch.tensor(storage, dtype=torch.float32, device=device)

        self.assertTrue(is_fake_tensor(tensor))
        self.assertEqual(tensor.device, torch.device(expected_device))
        self.assertEqual(tensor.shape, (2,))

    def test_tensor_constructor_meta_device_disallowed(self):
        with patch.object(torch._functorch.config, "fake_tensor_allow_meta", False):
            exc = expectedIfCppFakeTensor(RuntimeError, AssertionError)
            msg = "device.type must not be 'meta' when allow_meta is False"
            with self.assertRaisesRegex(exc, msg):
                with FakeTensorMode():
                    torch.tensor(3.0, device="meta")

    def check_function_with_fake(self, fn):
        out = fn()
        with torch._subclasses.FakeTensorMode():
            out_fake = fn()

        for a, b in zip(pytree.tree_leaves(out), pytree.tree_leaves(out_fake)):
            if not isinstance(a, torch.Tensor):
                self.assertTrue(not isinstance(b, torch.Tensor))
                continue

            prims.utils.compare_tensor_meta(a, b, check_strides=True)

    def test_non_overlapping_stride_zero(self):
        def foo():
            x = torch.empty_strided([1, 3, 427, 640], (0, 1, 1920, 3))
            return x.half()

        self.check_function_with_fake(foo)

    def test_fake_mode_error(self):
        x = torch.rand([4, 4])

        with self.assertRaisesRegex(Exception, "Please convert all Tensors"):
            with FakeTensorMode():
                y = x[0]

    def test_no_tag_func(self):
        import functools

        from torch.nn.attention.flex_attention import _identity, flex_attention

        def create_attention(score_mod, block_mask, enable_gqa=False):
            return functools.partial(
                flex_attention,
                score_mod=score_mod,
                block_mask=block_mask,
                enable_gqa=enable_gqa,
            )

        input_shape = (4, 16, 128, 64)
        q = torch.randn(
            input_shape,
            dtype=torch.bfloat16,
            device="cpu",
            requires_grad=False,
        )
        k = torch.randn(
            input_shape,
            dtype=torch.bfloat16,
            device="cpu",
            requires_grad=False,
        )
        v = torch.randn(
            input_shape,
            dtype=torch.bfloat16,
            device="cpu",
            requires_grad=False,
        )
        sdpa_partial = create_attention(_identity, None)
        with FakeTensorMode(allow_non_fake_inputs=True):
            sdpa_partial(q, k, v, return_lse=False)

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO, "isinstance check for FakeTensor won't work with compile"
    )
    def test_fake_grad_copy(self):
        x = torch.rand([4, 4], requires_grad=True)
        x.grad = torch.rand([4, 4])
        mode = FakeTensorMode()
        fake_x = mode.from_tensor(x)
        prims.utils.compare_tensor_meta(fake_x, x)
        prims.utils.compare_tensor_meta(fake_x.grad, x.grad)

        self.assertTrue(is_fake_tensor(fake_x.grad))

    def test_binary_op_type_promotion(self):
        with FakeTensorMode():
            x = torch.empty([2, 2], dtype=torch.float)
            y = torch.empty([2, 2], dtype=torch.int64)
            out = x / y
            self.assertEqual(out.dtype, torch.float)
            self.assertEqual(out.device.type, "cpu")

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO, "isinstance check for FakeTensor won't work with compile"
    )
    def test_from_numpy(self):
        with FakeTensorMode():
            x = torch.tensor(np.zeros([4, 4]))
            self.checkType(x, "cpu", [4, 4])

    def test_randperm(self):
        x = torch.randperm(10)
        y = torch.randperm(5, device="cpu")
        with FakeTensorMode():
            x1 = torch.randperm(10)
            prims.utils.compare_tensor_meta(x, x1)
            y1 = torch.randperm(5, device="cpu")
            prims.utils.compare_tensor_meta(y, y1)

    def test_print_in_fake_mode(self):
        x = torch.zeros(2)
        # does not fail
        with FakeTensorMode():
            out = str(x)
        if "FakeTensor" in out:
            raise AssertionError("FakeTensor should not be in output")

    def test_split_return_self(self):
        def fn(x):
            return torch.functional.split(x, 0)[0]

        # meta should not return self
        with FakeTensorMode(), enable_python_dispatcher():
            out_fake = fn(torch.empty((0,)))

        out_eager = fn(torch.empty((0,)))
        self.checkMetaProps(out_fake, out_eager)

    def test_split_empty_dim(self):
        def fn(x):
            return torch.split(x, 2)[0]

        with FakeTensorMode(), enable_python_dispatcher():
            out_fake = fn(torch.empty((0,)))

        out_eager = fn(torch.empty((0,)))
        self.checkMetaProps(out_fake, out_eager)

    def test_as_strided_negative_stride_error(self):
        error = (
            r"as_strided: Negative strides are not supported at the "
            r"moment, got strides: \[-?[0-9]+(, -?[0-9]+)*\]"
        )
        with FakeTensorMode():
            x = torch.empty(0)
            with self.assertRaisesRegex(RuntimeError, error):
                torch.as_strided(x, (17, 18), (-80, 1), 1)

    def test_recursive_invocation(self):
        mode = FakeTensorMode()
        with mode:
            x = torch.tensor(2)
            mode.in_kernel_invocation = True
            y = x + x
            self.assertTrue(mode.in_kernel_invocation)

    @skipIfCppFakeTensor("no data-dependent output support")
    def test_data_dependent_operator(self):
        with FakeTensorMode(allow_fallback_kernels=False):
            x = torch.rand([10, 10])

            self.assertRaises(DynamicOutputShapeException, lambda: torch.nonzero(x))

    def test_symbolic_number_tensor_argument_not_constant(self):
        shape_env = ShapeEnv()
        symint = shape_env.create_unbacked_symint()
        with FakeTensorMode(shape_env=shape_env):
            result = torch.div(symint, 4, rounding_mode="trunc")
        self.assertIsNone(maybe_get_fake_constant(result))

    # Conversion of real views to C++ fakes is not implemented.
    @xfailIf(CPP_FAKETENSOR)
    def test_parameter_view(self):
        x = torch.nn.Parameter(torch.randn(4))
        x_view = x.view(4)
        mode = FakeTensorMode()
        fake_x_view = mode.from_tensor(x_view)
        fake_x = mode.from_tensor(x)
        self.assertFalse(isinstance(fake_x_view, torch.nn.Parameter))
        self.assertTrue(isinstance(fake_x, torch.nn.Parameter))

    def test_tolist(self):
        shape_env = ShapeEnv()
        with FakeTensorMode(allow_fallback_kernels=False, shape_env=shape_env):
            # 1-D: flat list of (symbolic) scalars.
            flat = torch.rand([10]).tolist()
            self.assertEqual(len(flat), 10)
            self.assertNotIsInstance(flat[0], list)
            # Multi-dim exercises the recursive fake tolist path.
            nested = torch.rand([2, 3]).tolist()
            self.assertEqual(len(nested), 2)
            self.assertTrue(all(len(row) == 3 for row in nested))
            # 0-D returns a scalar, not a list.
            self.assertNotIsInstance(torch.rand(()).tolist(), list)

    def test_tolist_traces_unbind(self):
        def f(x):
            return x + sum(v for row in x.tolist() for v in row)

        gm = make_fx(f, tracing_mode="symbolic")(torch.randint(0, 9, (2, 2)))
        targets = {n.target for n in gm.graph.nodes if n.op == "call_function"}
        self.assertIn(torch.ops.aten.unbind.int, targets)
        self.assertNotIn(torch.ops.aten.select.int, targets)

    # Propagate real tensors doesn't work with fake-on-fake
    @expectedFailurePropagateRealTensors
    def test_same_shape_env_preserved(self):
        shape_env = ShapeEnv()
        mode1 = FakeTensorMode(shape_env=shape_env)
        t1 = mode1.from_tensor(
            torch.randn(10),
            symbolic_context=StatelessSymbolicContext(
                dynamic_sizes=[DimDynamic.DYNAMIC], constraint_sizes=[None]
            ),
        )
        mode2 = FakeTensorMode(shape_env=shape_env)
        t2 = mode2.from_tensor(t1)
        # t2.size(0) is still dynamic, even though we didn't pass DYNAMIC here
        self.assertIsNot(t2, t1)
        self.assertIs(maybe_get_fake_mode(t1), mode1)
        self.assertIs(maybe_get_fake_mode(t2), mode2)
        self.assertIs(t2.size(0).node.shape_env, t1.size(0).node.shape_env)
        self.assertEqual(str(t2.size(0)), str(t1.size(0)))

    def test_linear_rank3_unbacked_batch_dim(self):
        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True, shape_env=shape_env)
        fake_x = fake_mode.from_tensor(
            torch.randn(4, 16, 256),
            symbolic_context=StatelessSymbolicContext(
                dynamic_sizes=[
                    DimDynamic.UNBACKED,
                    DimDynamic.STATIC,
                    DimDynamic.STATIC,
                ],
                constraint_sizes=[None, None, None],
            ),
        )
        fake_weight = fake_mode.from_tensor(torch.randn(3072, 256), static_shapes=True)

        def f(x):
            return torch.nn.functional.linear(x, fake_weight)

        self.assertTrue(free_unbacked_symbols(fake_x.shape[0]))
        with fake_mode:
            out = f(fake_x)

        self.assertEqual(out.shape[1:], (16, 3072))
        self.assertTrue(free_unbacked_symbols(out.shape[0]))

        x = torch.randn(4, 16, 256)
        weight = torch.randn(3072, 256)
        torch._dynamo.decorators.mark_unbacked(
            x, 0, min=2, max=8, hint_override=x.shape[0]
        )

        def trace_f(x, weight):
            return torch.nn.functional.linear(x, weight)

        gm = make_fx(trace_f, tracing_mode="symbolic")(x, weight)
        replay_out = gm(torch.randn(5, 16, 256), weight)
        self.assertEqual(replay_out.shape, (5, 16, 3072))

    def test_view_unbacked_contiguous_relation_from_hints(self):
        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True, shape_env=shape_env)
        with fake_mode:
            seq = shape_env.create_unbacked_symint()
            torch._dynamo.override_optimization_hint(seq, 16)
            base = torch.empty(4, seq, 256)
            chunked = torch.as_strided(
                base,
                (4, 2 * (seq // 2), 256),
                (256 * seq, 256, 1),
            )
            out = chunked.view(4 * 2 * (seq // 2), 256)

        self.assertEqual(out.shape, (4 * 2 * (seq // 2), 256))
        self.assertTrue(free_unbacked_symbols(out.shape[0]))

    def test_meta_storage_trace_uses_hint_for_symbolic_size(self):
        from torch._subclasses.meta_utils import MetaStorageDesc, MetaStorageId

        shape_env = ShapeEnv()
        size = shape_env.create_unbacked_symint()
        torch._dynamo.override_optimization_hint(size, 16)

        metadata = MetaStorageDesc(MetaStorageId(0), 3 * size, None).as_json(1)
        self.assertEqual(metadata["size"], "3*u0")
        self.assertEqual(metadata["size_hint"], 48)

    def test_matmul_rank4_unbacked_batch_dim(self):
        fake_mode, (q, k) = self.fake_with_unbacked_batch(
            torch.randn(4, 2, 8, 16),
            torch.randn(4, 2, 8, 16),
        )

        def f(q, k):
            return torch.matmul(q, k.transpose(-2, -1))

        with fake_mode:
            gm = make_fx(f)(q, k)
            out = gm(q, k)

        self.assertEqual(out.shape[1:], (2, 8, 8))
        self.assertTrue(free_unbacked_symbols(out.shape[0]))

    def test_sdpa_math_unbacked_batch_dim(self):
        fake_mode, (q, k, v) = self.fake_with_unbacked_batch(
            torch.randn(4, 2, 8, 16),
            torch.randn(4, 2, 8, 16),
            torch.randn(4, 2, 8, 16),
        )

        def f(q, k, v):
            with sdpa_kernel([SDPBackend.MATH], set_priority=True):
                return torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, is_causal=True
                )

        with fake_mode:
            gm = make_fx(f)(q, k, v)
            out = gm(q, k, v)

        self.assertEqual(out.shape[1:], (2, 8, 16))
        self.assertTrue(free_unbacked_symbols(out.shape[0]))

    def test_size_default_unbacked_batch_dim(self):
        fake_mode, (x,) = self.fake_with_unbacked_batch(torch.randn(4, 2))

        def f(x):
            return x.reshape(torch.ops.aten.size.default(x))

        with fake_mode:
            gm = make_fx(f)(x)
            out = gm(x)

        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, x_1):
    sym_size_int = torch.ops.aten.sym_size.int(x_1, 0)
    view = torch.ops.aten.view.default(x_1, [sym_size_int, 2]);  x_1 = sym_size_int = None
    return view""",
        )
        self.assertEqual(out.shape[1:], (2,))
        self.assertTrue(free_unbacked_symbols(out.shape[0]))

    def test_cudnn_sdpa_unbacked_batch_dim(self):
        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True, shape_env=shape_env)
        batch = shape_env.create_unbacked_symint()

        def fake_cuda_attn_tensor(*shape):
            with fake_mode:
                return torch.empty(shape, device="cuda", dtype=torch.float16)

        q = fake_cuda_attn_tensor(batch, 2, 8, 24)
        k = fake_cuda_attn_tensor(batch, 2, 8, 24)
        v = fake_cuda_attn_tensor(batch, 2, 8, 16)
        self.assertTrue(free_unbacked_symbols(q.shape[0]))

        with fake_mode:
            out = torch.ops.aten._scaled_dot_product_cudnn_attention.default(
                q, k, v, None, False, 0.0, True
            )[0]
            out_size = torch.ops.aten.sym_size.default(out)

        self.assertEqual(out.shape[1:], (2, 8, 16))
        self.assertEqual(out_size[1:], (2, 8, 16))
        self.assertTrue(free_unbacked_symbols(out.shape[0]))
        self.assertTrue(free_unbacked_symbols(out_size[0]))

    # TODO: Support NJT.  There's also some funny business with dynamic shapes
    # which would need to be dealt with as well
    @expectedFailurePropagateRealTensors
    @skipIfCppFakeTensor("no nested tensor support")
    def test_jagged_fake_to_fake_preserved(self):
        from torch.nested._internal.nested_tensor import jagged_from_list

        S0, S1, S2 = 3, 4, 5
        D = 4
        a = torch.randn(S0, D, requires_grad=True, dtype=torch.float64)
        b = torch.randn(S1, D, requires_grad=True, dtype=torch.float64)
        c = torch.randn(S2, D, requires_grad=True, dtype=torch.float64)
        offsets = None
        jt, _ = jagged_from_list([a, b, c], offsets)
        shape_env = ShapeEnv()
        mode1 = FakeTensorMode(shape_env=shape_env)
        t1 = mode1.from_tensor(jt)
        mode2 = FakeTensorMode(shape_env=shape_env)
        t2 = mode2.from_tensor(t1)
        # It's not obvious that the invocation above makes it dynamic but it
        # does!
        self.assertTrue(free_symbols(t1.size()))
        self.assertIsNot(t2, t1)
        self.assertIs(maybe_get_fake_mode(t1.offsets()), mode1)
        self.assertIs(maybe_get_fake_mode(t2.offsets()), mode2)
        self.assertIs(t2.size(1).node.shape_env, t1.size(1).node.shape_env)
        self.assertEqual(str(t2.size(1)), str(t1.size(1)))

    @skipIfCrossRef
    def test_deepcopy(self):
        with FakeTensorMode() as mode:
            pass
        mod = torch.nn.BatchNorm2d(10)
        with torch._subclasses.fake_tensor.FakeCopyMode(mode):
            mod_copied = copy.deepcopy(mod)

        def check_copy(mod, mod_copied):
            for name, param in itertools.chain(
                mod.named_parameters(), mod.named_buffers()
            ):
                param_copied = getattr(mod_copied, name)
                self.checkMetaProps(param, param_copied)
                self.assertTrue(is_fake_tensor(param_copied))
                self.assertEqual(
                    isinstance(param, torch.nn.Parameter),
                    isinstance(param_copied, torch.nn.Parameter),
                )
                self.assertEqual(param.requires_grad, param_copied.requires_grad)

        check_copy(mod, mod_copied)

        class ModuleNew(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = torch.rand([10, 2])
                self.b = self.a
                self.c = self.a[0]

        mod = ModuleNew()
        with torch._subclasses.fake_tensor.FakeCopyMode(mode):
            mod_copied = copy.deepcopy(mod)

        self.assertIs(mod_copied.a, mod_copied.b)
        self.assertEqual(mod_copied.b.storage()._cdata, mod_copied.a.storage()._cdata)

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO, "isinstance check for FakeTensor won't work with compile"
    )
    def test_scalar_inputs(self):
        with FakeTensorMode():
            self.checkType(torch.div(3, 2), "cpu", [])
            ten = torch.zeros(2, dtype=torch.int32) * 2.0
            self.assertEqual(ten.dtype, torch.float)
            self.checkType(ten, "cpu", [2])

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO, "isinstance check for FakeTensor won't work with compile"
    )
    def test_allow_meta(self):
        def run_meta():
            with FakeTensorMode():
                x = torch.rand([4], device="meta")
                return x + x

        self.checkType(run_meta(), "meta", [4])

        with patch.object(torch._functorch.config, "fake_tensor_allow_meta", False):
            self.assertRaises(Exception, run_meta)

    def test_embedding_bag_meta(self):
        def f():
            # This behavior was originally unintentional but we see people
            # relying on it
            embedding = torch.nn.EmbeddingBag(10, 3, mode="sum", device="meta")
            input = torch.tensor([1, 2, 4, 5, 4, 3, 2, 9], dtype=torch.long)
            offsets = torch.tensor([0, 4], dtype=torch.long)
            return embedding(input, offsets)

        real_out = f()
        with FakeTensorMode():
            fake_out = f()

        for r, f in zip(real_out, fake_out):
            self.assertEqual(r.size(), f.size())
            self.assertEqual(r.device, f.device)

    def test_weight_int4pack_mm(self):
        # XPU packs w as 2D [N, K // 8], other devices as 4D [N // 8, ...].
        for device, w_shape, n in [("cuda", (8, 2, 32, 4), 64), ("xpu", (8, 8), 8)]:
            with self.subTest(device=device), FakeTensorMode() as mode:

                def fake(shape, dtype):
                    with unset_fake_temporarily():
                        meta = torch.empty(shape, dtype=dtype, device="meta")
                    return mode.fake_tensor_converter.from_meta_and_device(
                        mode, meta, torch.device(device)
                    )

                x = fake((4, 64), torch.bfloat16)
                w = fake(w_shape, torch.int32)
                q_scale_and_zeros = fake((2, n, 2), torch.bfloat16)
                out = torch.ops.aten._weight_int4pack_mm(x, w, 32, q_scale_and_zeros)
                self.assertEqual(out.shape, (4, n))
                self.assertEqual(out.device.type, device)

    def test_embedding_meta_indices(self):
        with FakeTensorMode():
            weight = torch.empty(20, 8)
            # Indices must be in range: under propagate_real_tensors the real
            # embedding gather runs, and uninitialized (torch.empty) indices read
            # out of bounds -- a hard GPU fault on some backends (e.g. ROCm).
            indices = torch.zeros(2, 3, dtype=torch.long, device="meta")
            out = torch.nn.functional.embedding(indices, weight)

            meta_weight = torch.empty(20, 8, dtype=torch.float64, device="meta")
            cpu_indices = torch.zeros(4, 5, dtype=torch.long)
            meta_weight_out = torch.nn.functional.embedding(cpu_indices, meta_weight)

        self.assertTrue(is_fake_tensor(out))
        self.assertEqual(out.shape, (2, 3, 8))
        self.assertEqual(out.dtype, weight.dtype)
        self.assertEqual(out.device, weight.device)
        self.assertEqual(maybe_get_fake_device(out), maybe_get_fake_device(weight))

        self.assertTrue(is_fake_tensor(meta_weight_out))
        self.assertEqual(meta_weight_out.shape, (4, 5, 8))
        self.assertEqual(meta_weight_out.dtype, meta_weight.dtype)
        self.assertEqual(meta_weight_out.device, meta_weight.device)
        self.assertEqual(
            maybe_get_fake_device(meta_weight_out), maybe_get_fake_device(meta_weight)
        )

    @skipIfTorchDynamo("uses torch.compile")
    @skipIfPropagateRealTensors("uses default FakeTensorMode propagation")
    def test_embedding_meta_indices_torch_compile(self):
        if torch._functorch.config.fake_tensor_propagate_real_tensors:
            self.skipTest("uses default FakeTensorMode propagation")

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.position_embedding = torch.nn.Embedding(20, 8)

            def forward(self, x):
                positions = (
                    torch.arange(0, x.shape[1])
                    .expand(x.shape[0], x.shape[1])
                    .to("meta")
                )
                return self.position_embedding(positions)

        m = M().eval()
        x = torch.randint(0, 20, (2, 3))
        eager = m(x)
        try:
            compiled = torch.compile(m, backend="eager")(x)
        finally:
            torch._dynamo.reset()

        self.assertEqual(compiled.shape, eager.shape)
        self.assertEqual(compiled.device, eager.device)

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO, "isinstance check for FakeTensor won't work with compile"
    )
    @parametrize("allow", [False, True])
    @parametrize("override", [None, False, True])
    @parametrize("channels", [1, 2])
    def test_mixed_real_and_fake_inputs(self, allow, override, channels):
        from torch._subclasses.fake_tensor import fake_tensor_tls

        class _TestPattern(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(channels, channels, 1)
                self.bn = torch.nn.BatchNorm2d(channels)

            def forward(self, input):
                running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
                scale_factor = self.bn.weight / running_std
                weight_shape = [1] * len(self.conv.weight.shape)
                weight_shape[0] = -1
                bias_shape = [1] * len(self.conv.weight.shape)
                bias_shape[1] = -1
                scaled_weight = self.conv.weight * scale_factor.reshape(weight_shape)
                zero_bias = torch.zeros_like(self.conv.bias, dtype=input.dtype)
                conv = self.conv._conv_forward(input, scaled_weight, zero_bias)
                conv_orig = conv / scale_factor.reshape(bias_shape)
                conv_orig = conv_orig + self.conv.bias.reshape(bias_shape)
                conv = self.bn(conv_orig)
                return conv

        mod = _TestPattern()
        mode = FakeTensorMode(allow_non_fake_inputs=allow)
        allowed = allow if override is None else override
        with (
            patch.object(fake_tensor_tls, "allow_non_fake_inputs_override", override),
            mode,
        ):
            x = torch.randn(1, channels, 3, 3)
            if allowed:
                out = mod(x)
                self.checkType(out, "cpu", (1, channels, 3, 3))
                self.assertIs(maybe_get_fake_mode(out), mode)
            else:
                with self.assertRaisesRegex(
                    AssertionError, "Please convert all Tensors"
                ):
                    mod(x)
        self.assertEqual(mode.allow_non_fake_inputs, allow)

    def test__adaptive_avg_pool2d_backward(self):
        with FakeTensorMode():
            grad_out = torch.rand(2, 3, 4, 4)
            inp = torch.rand(2, 3, 4, 4).to(memory_format=torch.channels_last)
            grad_in = torch.ops.aten._adaptive_avg_pool2d_backward(grad_out, inp)
            self.assertTrue(
                torch._prims_common.suggest_memory_format(grad_in)
                == torch.channels_last
            )

    def test_suggest_memory_format_with_degenerate_dimensions(self):
        """
        Test that suggest_memory_format correctly returns contiguous_format for
        tensors with degenerate dimensions (H=1, W=1) that have ambiguous strides.
        This is a regression test for a bug where the Python implementation checked
        strides[d] > 1 instead of shape[d] > 1, causing incorrect channels_last
        classification. See c10/core/MemoryFormat.h is_channels_last_strides_2d_s4.
        """
        # Create a tensor that mimics the problematic case:
        # shape (1, 512, 1, 1) with strides (512, 1, 512, 1)
        # This comes from permute(0, 2, 1).contiguous().unsqueeze(-1)
        x = torch.randn(1, 1, 512)
        x = x.permute(0, 2, 1).contiguous()  # shape (1, 512, 1), strides (512, 1, 512)
        x = x.unsqueeze(-1)  # shape (1, 512, 1, 1), strides (512, 1, 512, 1)

        # In eager mode, this should be contiguous_format
        self.assertEqual(
            torch._prims_common.suggest_memory_format(x),
            torch.contiguous_format,
            "Eager tensor with degenerate dims should be contiguous_format",
        )

        # In FakeTensor mode, it should also be contiguous_format (not channels_last)
        with FakeTensorMode() as mode:
            fake_x = mode.from_tensor(torch.randn(1, 1, 512))
            fake_x = fake_x.permute(0, 2, 1).contiguous()
            fake_x = fake_x.unsqueeze(-1)

            self.assertEqual(
                torch._prims_common.suggest_memory_format(fake_x),
                torch.contiguous_format,
                "FakeTensor with degenerate dims should be contiguous_format",
            )

        # Test that upsample output has matching strides in eager vs FakeTensor
        upsample = torch.nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )

        eager_out = upsample(x)
        self.assertTrue(
            eager_out.is_contiguous(), "Eager upsample output should be contiguous"
        )

        with FakeTensorMode() as mode:
            fake_x = mode.from_tensor(torch.randn(1, 1, 512))
            fake_x = fake_x.permute(0, 2, 1).contiguous().unsqueeze(-1)
            fake_out = upsample(fake_x)

            self.assertTrue(
                fake_out.is_contiguous(),
                lambda msg: f"{msg}\nFakeTensor upsample output should be contiguous, got strides {fake_out.stride()}",
            )

    def test_export_numpy(self):
        class MyNumpyModel(torch.nn.Module):
            def forward(self, input):
                input = input.numpy()
                return input + np.random.randn(*input.shape)

        with FakeTensorMode():
            ep = torch.export.export(
                MyNumpyModel(), args=(torch.randn(1000),), strict=True
            )
            self.assertTrue(isinstance(ep, torch.export.ExportedProgram))

    def test_unsqueeze_copy(self):
        shape_env = ShapeEnv()
        t1 = torch.ones(2, 2, 768)
        with FakeTensorMode(shape_env=shape_env) as fake_mode:
            t = fake_mode.from_tensor(
                t1,
                symbolic_context=StatelessSymbolicContext(
                    dynamic_sizes=[
                        DimDynamic.DYNAMIC,
                        DimDynamic.STATIC,
                        DimDynamic.STATIC,
                    ],
                ),
            )

        self.assertEqual(t.shape[0], torch.ops.aten.unsqueeze_copy(t, 1).shape[0])

    def test_alias_call(self):
        fwAD = torch.autograd.forward_ad

        def f(x):
            return 4312491 * x

        with torch._subclasses.fake_tensor.FakeTensorMode():
            with fwAD.dual_level():
                x = torch.randn(3, device="cpu")
                y = torch.ones_like(x)
                dual = fwAD.make_dual(x, y)
                r = f(dual)

        self.assertTrue(is_fake_tensor(r))
        self.assertEqual(r.size(), [3])

    @parametrize("reverse", [False, True])
    def test_scan(self, reverse):
        def add(x, y):
            return x + y, x + y

        with torch._subclasses.fake_tensor.FakeTensorMode():
            x = torch.randn((3, 5, 7), device="cpu")
            init = torch.randn((3, 7), device="cpu")
            r = scan(add, init, x, dim=1, reverse=reverse)

        self.assertTrue(is_fake_tensor(r[0]))
        self.assertTrue(is_fake_tensor(r[1]))

    def test_fast_div_int_to_float(self):
        mode = FakeTensorMode()
        with mode:
            x = torch.empty(2, 2, device="cpu", dtype=torch.int32)
            y = torch.empty(2, 2, device="cpu", dtype=torch.int32)
        from torch._subclasses.fake_impls import get_fast_op_impls

        fast_div = get_fast_op_impls()[torch.ops.aten.div.Tensor]
        z = fast_div(mode, x, y)
        self.assertEqual(z.dtype, torch.float32)

    def test_fast_div(self):
        mode = FakeTensorMode()
        with mode:
            x = torch.empty(2, 2, device="cpu", dtype=torch.int32)
        from torch._subclasses.fake_impls import get_fast_op_impls

        fast_div = get_fast_op_impls()[torch.ops.aten.div.Tensor]
        y = fast_div(mode, x, 2)
        self.assertEqual(y.dtype, torch.float32)

    def test_nanmean_out(self):
        # Regression test to ensure we don't error out.
        with torch._subclasses.fake_tensor.FakeTensorMode() as mode:
            x = torch.randn(10)
            out = torch.empty(())
            torch.nanmean(x, out=out)

        self.assertEqual(out.dtype, x.dtype)

    def test_unbind_copy_out(self):
        # Regression test to ensure we don't error out.
        with torch._subclasses.fake_tensor.FakeTensorMode() as mode:
            eye = torch.eye(3)
            out = (torch.zeros(3), torch.zeros(3), torch.zeros(3))
            torch.unbind_copy(eye, out=out)

        self.assertEqual(out[0].dtype, eye.dtype)
        self.assertEqual(out[1].dtype, eye.dtype)
        self.assertEqual(out[2].dtype, eye.dtype)

    @unittest.skipIf(
        TEST_ACCELERATOR, "Only execute when an accelerator is not present"
    )
    def test_avoid_device_init_without_backends(self):
        fake_mode = FakeTensorMode()

        self.assertTrue(
            fake_mode.avoid_device_init,
            "Expected avoid_device_init to return True when no backends are registered",
        )

    @parametrize("is_available", [False, True])
    @skipIfTorchDynamo(
        "TorchDynamo exposes https://github.com/pytorch/pytorch/issues/166696"
    )
    @unittest.skipIf(
        TEST_ACCELERATOR, "Only execute when an accelerator is not present"
    )
    def test_avoid_device_init_with_privateuse1_backend(self, is_available):
        class _DummyPrivateUse1Module:
            @staticmethod
            def is_available() -> bool:
                return is_available

        backend_name = "privateuseone"

        try:
            torch._register_device_module(backend_name, _DummyPrivateUse1Module)

            fake_mode = FakeTensorMode()

            self.assertEqual(fake_mode.avoid_device_init, not is_available)
        finally:
            delattr(torch, backend_name)
            del sys.modules[f"torch.{backend_name}"]

    def test_unique_output_dtype(self):
        shape_env = ShapeEnv()
        for input_dtype in [torch.float32, torch.float16, torch.bfloat16]:
            x_real = torch.randn(10, dtype=input_dtype)
            real_unique, real_inverse, real_counts = torch.unique(
                x_real, return_inverse=True, return_counts=True
            )
            with FakeTensorMode(shape_env=shape_env):
                x = torch.randn(10, dtype=input_dtype)
                fake_unique, fake_inverse, fake_counts = torch.unique(
                    x, return_inverse=True, return_counts=True
                )
                self.assertEqual(fake_unique.dtype, real_unique.dtype)
                self.assertEqual(fake_inverse.dtype, real_inverse.dtype)
                self.assertEqual(fake_counts.dtype, real_counts.dtype)

        # Also test with dim argument
        x_real = torch.randn(3, 4, dtype=torch.float32)
        real_unique, real_inverse, real_counts = torch.unique(
            x_real, dim=0, return_inverse=True, return_counts=True
        )
        with FakeTensorMode(shape_env=shape_env):
            x = torch.randn(3, 4, dtype=torch.float32)
            fake_unique, fake_inverse, fake_counts = torch.unique(
                x, dim=0, return_inverse=True, return_counts=True
            )
            self.assertEqual(fake_unique.dtype, real_unique.dtype)
            self.assertEqual(fake_inverse.dtype, real_inverse.dtype)
            self.assertEqual(fake_counts.dtype, real_counts.dtype)

    def test_assert_tensor_metadata_with_list(self):
        with FakeTensorMode():
            x = torch.empty(10)
            torch.ops.aten._assert_tensor_metadata.default(x, size=[10], stride=[1])

    def test_unique_default_dynamic_output_shape(self):
        x = torch.tensor([2, 1, 4, 2, 2])
        with FakeTensorMode() as mode:
            fake_x = mode.from_tensor(x)
            with self.assertRaises(DynamicOutputShapeException):
                torch.ops.aten._unique.default(fake_x, False, False)

        shape_env = ShapeEnv(allow_dynamic_output_shape_ops=True)
        with FakeTensorMode(shape_env=shape_env) as mode:
            fake_x = mode.from_tensor(x)
            fake_unique, fake_inverse = torch.ops.aten._unique.default(
                fake_x, False, True
            )

        self.assertEqual(fake_unique.dtype, x.dtype)
        self.assertEqual(fake_inverse.dtype, torch.int64)
        self.assertEqual(fake_inverse.shape, x.shape)
        self.assertTrue(free_unbacked_symbols(fake_unique.shape[0]))

    def test_select_out_of_bounds(self):
        with FakeTensorMode():
            x = torch.randn(3, 4)
            with self.assertRaisesRegex(IndexError, "index .* out of range"):
                torch.select(x, dim=1, index=10)
            with self.assertRaisesRegex(IndexError, "index .* out of range"):
                torch.select(x, dim=1, index=-10)

    @fake_tensor_modes
    def test_fake_mode_lookup(self, mode_cls):
        from torch._guards import active_fake_mode, detect_fake_mode

        # Distinct creation sites must appear in the mismatch diagnostic.
        # Keep these frames in the creation stacks under Dynamo.
        @torch.compiler.disable
        def make_mode():
            return mode_cls()

        @torch.compiler.disable
        def make_other():
            return mode_cls()

        mode, other = make_mode(), make_other()
        cpu = torch.device("cpu")

        def make_fake(m):
            meta = torch.empty(2, device="meta")
            return self.fake_from_meta(m, meta, cpu)

        fake, other_fake = make_fake(mode), make_fake(other)
        self.assertIs(maybe_get_fake_mode(fake), mode)
        self.assertIs(maybe_get_fake_mode(TwoTensor(fake, make_fake(mode))), mode)
        self.assertIsNone(maybe_get_fake_mode(torch.empty(2)))
        self.assertIs(detect_fake_mode([fake]), mode)
        self.assertIsNone(active_fake_mode())
        with mode:
            self.assertIs(active_fake_mode(), mode)
            self.assertIs(detect_fake_mode(), mode)
            # The inputs' mode must match the active one.
            not_other = r"(?:(?!in make_other)[\s\S])*"
            not_mode = r"(?:(?!in make_mode)[\s\S])*"
            msg = (
                rf"from active fake mode 0 allocated at:\n{not_other}in make_mode"
                rf"{not_other}from fake tensor input 0 allocated at:\n"
                rf"{not_mode}in make_other{not_mode}\Z"
            )
            with self.assertRaisesRegex(AssertionError, msg):
                detect_fake_mode([other_fake])
        self.assertIsNone(active_fake_mode())

        with mode_cls() as nested_mode:
            with BaseTorchDispatchMode():
                self.assertIs(active_fake_mode(), nested_mode)
            with unset_fake_temporarily():
                self.assertIsNone(active_fake_mode())

    @fake_tensor_modes
    @parametrize(
        "shape,input_device,other_device",
        [
            ((2, 3), "cpu", "cpu"),
            ((), "cpu", "cuda:0"),
            ((), "cuda:0", "cpu"),
        ],
    )
    @skipIfTorchDynamo(_CPP_FAKE_DYNAMO_SKIP)
    def test_inplace_and_out_identity(
        self, mode_cls, shape, input_device, other_device
    ):
        if (
            mode_cls is CppFakeTensorMode
            and torch._functorch.config.fake_tensor_propagate_real_tensors
        ):
            self.skipTest("C++ fake fallback does not support propagate_real_tensors")
        mode = mode_cls()
        fake, out, other = (
            self.fake_from_meta(mode, torch.empty(shape, device="meta"), device)
            for device in (input_device, input_device, other_device)
        )
        # Inspect the fallback's return, without ADInplaceOrView returning its input.
        with torch._C._AutoDispatchBelowADInplaceOrView():
            self.assertIs(torch.ops.aten.add_.Tensor(fake, 1), fake)
            for op, destination in (
                (lambda: aten.add_.Tensor(fake, other), fake),
                (lambda: aten.add.out(fake, other, out=out), out),
            ):
                if input_device == "cpu" and other_device != "cpu":
                    with self.assertRaisesRegex(
                        RuntimeError, "FakeTensor is wrapped to wrong device"
                    ):
                        op()
                else:
                    self.assertIs(op(), destination)
                self.assertEqual(destination.device, torch.device(input_device))
                self.assertIs(maybe_get_fake_mode(destination), mode)

    @fake_tensor_modes
    @skipIfTorchDynamo(_CPP_FAKE_DYNAMO_SKIP)
    @parametrize("layout", [torch.sparse_coo, torch.sparse_csr])
    def test_sparse_metadata_without_backend_kernel(self, mode_cls, layout):
        if (
            mode_cls is CppFakeTensorMode
            and torch._functorch.config.fake_tensor_propagate_real_tensors
        ):
            self.skipTest("C++ fake fallback does not support propagate_real_tensors")
        mode = mode_cls()
        meta = torch.eye(2).to_sparse(layout=layout).to("meta")
        sparse = self.fake_from_meta(mode, meta, "privateuseone:0")
        for context in (contextlib.nullcontext(), mode):
            with context:
                self.assertEqual(sparse.sparse_dim(), 2)
                self.assertEqual(sparse.dense_dim(), 0)
                if layout == torch.sparse_coo:
                    self.assertTrue(sparse.is_coalesced())
        self.assertEqual(sparse.device, torch.device("privateuseone:0"))
        self.assertIs(maybe_get_fake_mode(sparse), mode)

    @fake_tensor_modes
    @skipIfTorchDynamo(_CPP_FAKE_DYNAMO_SKIP)
    def test_sparse_meta_kernel_dispatch(self, mode_cls):
        if (
            mode_cls is CppFakeTensorMode
            and torch._functorch.config.fake_tensor_propagate_real_tensors
        ):
            self.skipTest("C++ fake fallback does not support propagate_real_tensors")
        mode = mode_cls()

        def check(out, shape):
            self.checkType(out, "cpu", list(shape))
            self.assertIs(maybe_get_fake_mode(out), mode)

        real = torch.eye(2).to_sparse()
        meta = real.to("meta")
        sparse = self.fake_from_meta(mode, meta, "cpu")
        with mode:
            for tensor in (real, sparse):
                self.assertEqual(tensor.sparse_dim(), 2)
                self.assertEqual(tensor.dense_dim(), 0)
                self.assertTrue(tensor.is_coalesced())
        self.assertFalse(is_fake_tensor(real))
        # Bypass ADInplaceOrView to check the sparse fallback's return.
        with torch._C._AutoDispatchBelowADInplaceOrView():
            self.assertIs(aten._coalesced_(sparse, False), sparse)
        self.assertFalse(sparse.is_coalesced())
        for out in (sparse.clone(), sparse + sparse, sparse.detach()):
            check(out, (2, 2))
            self.assertTrue(out.is_sparse)
        check(sparse._values(), meta._values().shape)
        check(sparse._indices(), meta._indices().shape)
        csr_meta = torch.eye(2).to_sparse_csr().to("meta")
        csr = self.fake_from_meta(mode, csr_meta, "cpu")
        check(csr.crow_indices(), csr_meta.crow_indices().shape)
        check(csr.detach(), csr.shape)
        for tensor in (sparse, csr):
            dense = tensor.to_dense()
            check(dense, (2, 2))
            self.assertEqual(dense.layout, torch.strided)


instantiate_parametrized_tests(TestFakeTensor)


def make_propagate_real_tensors_cls(cls):
    cls = make_test_cls_with_patches(
        cls,
        "PropagateRealTensors",
        "_propagate_real_tensors",
        (torch._functorch.config, "fake_tensor_propagate_real_tensors", True),
        xfail_prop="_expected_failure_propagate_real_tensors",
        decorator=skipIfTorchDynamo("propagate_real_tensors affects Dynamo"),
    )
    cls = skipIfCppFakeTensor("no propagate_real_tensors support")(cls)
    if not getattr(cls, "__unittest_skip__", False) and not TEST_WITH_TORCHDYNAMO:
        for name, fn in vars(cls).items():
            reason = getattr(fn, "_skip_propagate_real_tensors", None)
            if reason is not None and not getattr(fn, "__unittest_skip__", False):
                setattr(cls, name, unittest.skip(reason)(fn))
    cls.__file__ = __file__
    cls.__module__ = __name__
    globals()[cls.__name__] = cls


make_propagate_real_tensors_cls(TestFakeTensor)


class TestFakeTensorDevice(_TestFakeTensorHelpers, TestCase):
    """TestFakeTensor cases whose behavior depends on the tensor's device.

    Tests that mix ``device`` with a cpu tensor, or that target the accelerator
    implementation, are marked ``@onlyAccelerator``; the rest also run on cpu.
    """

    hw_classification = HardwareClassification.ACCELERATOR

    @onlyAccelerator
    def test_accelerator_initialized(self, device):
        # doesn't error
        with FakeTensorMode():
            p = torch.randn(4, 2, requires_grad=True, device=device)
            x = torch.randn(8, 4, device=device)
            y = torch.mm(x, p).square().sum()
            y.backward()

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO, "isinstance check for FakeTensor won't work with compile"
    )
    @onlyAccelerator
    @dtypes(*all_types_complex_float8_and())
    def test_index_with_cpu_indices(self, device, dtype):
        with FakeTensorMode():
            x = torch.ones([2048], device=device, dtype=dtype)
            out = x[torch.zeros([36], dtype=torch.int64)]
            self.checkType(out, self.device_type, [36])
            self.assertEqual(out.dtype, dtype)

    @onlyAccelerator
    def test_shape_take_not_device(self, device):
        with FakeTensorMode():
            x = torch.empty(1, device="cpu")
            y = torch.empty(8, 8, device=device)
            out = x.resize_as_(y)
            self.assertEqual(out.shape, (8, 8))
            self.assertEqual(out.device.type, "cpu")
            self.assertTrue(is_fake_tensor(out))

    @onlyAccelerator
    def test_conv_rejects_mismatched_fake_devices(self, device):
        if (
            torch._functorch.config.fake_tensor_propagate_real_tensors
            and self.device_type == "xpu"
        ):
            # Runs a real conv, which needs aten::_slow_conv2d_forward.
            self.skipTest(XPU_SLOW_CONV2D_SKIP)
        with FakeTensorMode():
            x = torch.empty(1, 3, 8, 8)
            w = torch.empty(3, 3, 3, 3, device=device)
            b = torch.empty(3, device=device)

            with self.assertRaisesRegex(
                RuntimeError,
                f"Expected all tensors to be on the same device.*weight is on {device}",
            ):
                torch.ops.aten.convolution.default(
                    x, w, b, [1, 1], [1, 1], [1, 1], False, [0, 0], 1
                )

    # The fake half of this comparison is covered on XPU by
    # test_conv_rejects_mismatched_fake_devices.
    @skipXPUIf(True, XPU_SLOW_CONV2D_SKIP)
    @onlyAccelerator
    def test_conv_mismatched_device_error_matches_eager(self, device):
        error = f"Expected all tensors to be on the same device.*weight is on {device}"

        for name, ctx in (
            ("eager", contextlib.nullcontext()),
            ("fake", FakeTensorMode()),
        ):
            with self.subTest(name), ctx:
                x = torch.empty(1, 3, 8, 8)
                w = torch.empty(3, 3, 3, 3, device=device)
                b = torch.empty(3, device=device)

                with self.assertRaisesRegex(RuntimeError, error):
                    torch.ops.aten.convolution.default(
                        x, w, b, [1, 1], [1, 1], [1, 1], False, [0, 0], 1
                    )

    @onlyAccelerator
    @fake_tensor_modes
    @parametrize("reverse", [False, True])
    @parametrize("constant", [False, True])
    def test_zero_dim(self, device, mode_cls, reverse, constant):
        if mode_cls is CppFakeTensorMode and constant:
            self.skipTest("C++ fake fallback does not support constant inputs")
        if (
            mode_cls is CppFakeTensorMode
            and torch._functorch.config.fake_tensor_propagate_real_tensors
        ):
            self.skipTest("C++ fake fallback does not support propagate_real_tensors")
        with mode_cls() as mode:
            x = torch.tensor(0.0) if constant else torch.empty(())
            y = torch.rand([4, 4], device=device)
            out = y + x if reverse else x + y
            self.assertEqual(out.shape, (4, 4))
            self.assertEqual(out.device, y.device)
            self.assertTrue(is_fake_tensor(out))
            self.assertIs(maybe_get_fake_mode(out), mode)

    @onlyAccelerator
    @skipIfPropagateRealTensors("Propagate real tensor not supported")
    @fake_tensor_modes
    def test_op_with_zero_dim_bypassed(self, device, mode_cls):
        if torch._functorch.config.fake_tensor_propagate_real_tensors:
            self.skipTest("Propagate real tensor not supported")
        shape_env = ShapeEnv()
        mode = mode_cls(shape_env=shape_env)
        x = torch.tensor(1.0, device=device)
        y = torch.tensor(2.0)
        fake_x = mode.from_tensor(x)
        fake_y = mode.from_tensor(y)

        with self.assertRaisesRegex(
            FakeTensorDeviceMismatchError,
            re.escape(f"found at least two devices, {fake_x.device} and cpu!"),
        ):
            torch.nextafter(fake_x, fake_y)

    @onlyAccelerator
    def test_diagonal_scatter_one_dim_single_elem_cpu_with_device_tensor(self, device):
        with FakeTensorMode():
            base = torch.zeros((1, 2), device=device)
            src = torch.tensor([1.0])
            out = torch.diagonal_scatter(base, src, dim1=0, dim2=1)
            self.assertEqual(out.shape, (1, 2))
            self.assertEqual(out.device, base.device)
            self.assertTrue(is_fake_tensor(out))

    @onlyAccelerator
    def test_diagonal_scatter_two_dim_cpu_with_device_tensor(self, device):
        with FakeTensorMode():
            base = torch.zeros((3, 3, 3))
            src = torch.ones((3, 3), device=device)
            out = torch.diagonal_scatter(base, src)
            self.assertEqual(out.shape, (3, 3, 3))
            self.assertEqual(out.device, base.device)
            self.assertTrue(is_fake_tensor(out))

    @onlyAccelerator
    @skipIfPropagateRealTensors("Propagate real tensor not supported")
    @fake_tensor_modes
    @parametrize("reverse", [False, True])
    def test_add_one_dim_single_elem_cpu_with_device_tensor(
        self, device, mode_cls, reverse
    ):
        if torch._functorch.config.fake_tensor_propagate_real_tensors:
            self.skipTest("Propagate real tensor not supported")
        with mode_cls():
            x = torch.randn([1])
            y = torch.randn(10, device=device)
            lhs, rhs = (y, x) if reverse else (x, y)

            with self.assertRaisesRegex(
                FakeTensorDeviceMismatchError,
                re.escape(
                    f"found at least two devices, {lhs.device} and {rhs.device}!"
                ),
            ):
                lhs + rhs

    @onlyAccelerator
    def test_throw(self, device):
        x = torch.tensor(0.0)  # TODO: tensor() errors
        with FakeTensorMode() as mode:
            x_conv = mode.from_tensor(x)
            y = torch.rand([4, 4], device=device)
            z = torch.rand([4, 4], device="cpu")
            self.assertRaises(Exception, lambda: torch.lerp(x_conv, y, z))

    @onlyAccelerator
    def test_type_as(self, device):
        with FakeTensorMode():
            x = torch.rand([16, 1], device="cpu")
            y = torch.rand([4, 4], device=device)
            out = x.type_as(y)
            self.assertEqual(out.device.type, self.device_type)
            self.assertTrue(is_fake_tensor(out))

    def test_setitem(self, device):
        for dev in ("cpu", device):
            with FakeTensorMode():
                x = torch.rand([16, 1], device=dev)
                x[..., 0] = 0

    @onlyAccelerator
    def test_device_inplace_copy(self, device):
        with FakeTensorMode():
            x = torch.rand([8, 8], device="cpu")
            y = torch.rand([8, 8], device=device)
            if x.copy_(y).device.type != "cpu":
                raise AssertionError("expected cpu device")
            if y.copy_(x).device.type != self.device_type:
                raise AssertionError(f"expected {self.device_type} device")

    @onlyAccelerator
    @skipIfCppFakeTensor("sets the Python fake_device property")
    def test_fake_device(self, device):
        t = torch.ones(3)
        t = t.view(1, 3)

        fake_mode1 = FakeTensorMode(allow_non_fake_inputs=True)
        fake_t = fake_mode1.from_tensor(t)
        fake_t.fake_device = torch.device(device)

        fake_mode2 = FakeTensorMode(allow_non_fake_inputs=True)
        new_fake_t = fake_mode2.from_tensor(fake_t)

        self.assertEqual(new_fake_t.device, fake_t.device)

    def test_non_kwarg_device(self, device):
        with FakeTensorMode():
            x = torch.rand([16, 1], device="cpu")
            y = x.to(torch.device("cpu"))
            self.assertIs(x, y)
            z = x.to(torch.device(device))
            self.assertEqual(z.device.type, self.device_type)

    @onlyAccelerator
    def test_index_put_error(self, device):
        mode = FakeTensorMode()
        for context in [contextlib.nullcontext, lambda: mode]:
            with context():
                y = torch.randn(2, 2, 3)
                x = torch.randn(2, 2, 3).to(device)
                with self.assertRaises(RuntimeError):
                    x[[1, 1]] = y

                with self.assertRaises(RuntimeError):
                    torch.ops.aten.index_put(x, torch.tensor([1, 1], device=device), y)

                # no error
                torch.ops.aten.index_put(
                    x, torch.tensor([1, 1], device=device), torch.tensor(5.0)
                )
                torch.ops.aten.index_put_(
                    x, torch.tensor([1, 1], device=device), torch.tensor(5.0)
                )

    @fake_tensor_modes
    @parametrize(
        "op",
        [
            torch.ones_like,
            subtest(lambda x, **kw: x.new_empty(x.shape, **kw), name="new_empty"),
        ],
        name_fn=lambda op: op.__name__,
    )
    def test_like_constructor(self, device, mode_cls, op):
        if (
            mode_cls is CppFakeTensorMode
            and torch._functorch.config.fake_tensor_propagate_real_tensors
        ):
            self.skipTest("C++ fake fallback does not support propagate_real_tensors")
        with mode_cls() as mode:
            x = torch.rand([4, 4])
            y = op(x)
            self.assertTrue(is_fake_tensor(y))
            self.assertEqual(y.device.type, "cpu")
            z = op(x, device=device)
            self.assertTrue(is_fake_tensor(z))
            self.assertEqual(z.device.type, self.device_type)
            self.assertIs(maybe_get_fake_mode(z), mode)
            self.assertEqual(z.shape, x.shape)

    def test_upsample_bilinear_small_channels(self, device):
        out = []
        mode = FakeTensorMode()
        for context in [contextlib.nullcontext, lambda: mode]:
            with context():
                arg0_1 = torch.empty_strided(
                    (3, 427, 640), (1, 1920, 3), dtype=torch.float32, device=device
                )
                unsqueeze = torch.ops.aten.unsqueeze.default(arg0_1, 0)
                out.append(
                    torch.ops.aten.upsample_bilinear2d.default(
                        unsqueeze, [800, 1199], False
                    )
                )

        self.assertTrue(out[1].is_contiguous())
        self.checkMetaProps(out[0], out[1])

    @onlyAccelerator
    def test_cpu_fallback(self, device):
        with FakeTensorMode(allow_fallback_kernels=False):
            filters = torch.randn(8, 4, 3, 3).to(device)
            inputs = torch.randn(1, 4, 5, 5).to(device)
            out = torch.nn.functional.conv2d(inputs, filters, padding=1)
            self.assertEqual(out.device.type, self.device_type)
            self.assertEqual(list(out.size()), [1, 8, 5, 5])

        with FakeTensorMode(allow_fallback_kernels=True):
            # intentionally bad inputs
            filters = torch.randn(8, 20, 3, 3).to(device)
            inputs = torch.randn(1, 7, 10, 5).to(device)
            with self.assertRaises(RuntimeError):
                torch.nn.functional.conv2d(inputs, filters, padding=1)

        with FakeTensorMode(allow_fallback_kernels=True):
            filters = torch.randn(8, 4, 3, 3).to(device)
            inputs = torch.randn(1, 4, 5, 5).to(device)

            out = torch.nn.functional.conv2d(inputs, filters, padding=1)
            self.assertEqual(out.device.type, self.device_type)
            self.assertEqual(list(out.size()), [1, 8, 5, 5])

    @onlyAccelerator
    def test_out_multi_device(self, device):
        with FakeTensorMode():
            x = torch.rand([4])
            y = torch.rand([4], device=device)

            with self.assertRaisesRegex(Exception, "found.+two.+devices"):
                torch.sin(x, out=y)

            with self.assertRaisesRegex(Exception, "found.+two.+devices"):
                x.add_(y)

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO, "isinstance check for FakeTensor won't work with compile"
    )
    def test_normalize_device(self, device):
        with FakeTensorMode():
            # device carries an index, self.device_type does not
            x = torch.empty(1, device=self.device_type)
            y = torch.empty(1, device=device)
            out = x + y
        self.checkType(out, self.device_type, [1])

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO, "isinstance check for FakeTensor won't work with compile"
    )
    @onlyAccelerator
    def test_new(self, device):
        with FakeTensorMode():
            a = torch.rand([16, 1])
            self.checkType(a.new(10, 10), "cpu", [10, 10])
            self.checkType(a.new([1, 2, 3, 4]), "cpu", [4])
            b = torch.rand([4, 4], device=device)
            self.checkType(b.new(device=device), self.device_type, [0])
            self.checkType(a.new(torch.rand([1])), "cpu", [1])

    @onlyAccelerator
    def test_embedding_device_indices(self, device):
        with FakeTensorMode():
            weight = torch.empty(20, 8, device=device)
            # Indices must be in range: under propagate_real_tensors the real
            # embedding gather runs, and uninitialized (torch.empty) indices read
            # out of bounds -- a hard GPU fault on some backends (e.g. ROCm).
            indices = torch.zeros(2, 3, dtype=torch.long, device=device)
            out = torch.nn.functional.embedding(indices, weight)

        self.assertTrue(is_fake_tensor(out))
        self.assertEqual(out.shape, (2, 3, 8))
        self.assertEqual(out.device, weight.device)
        self.assertEqual(maybe_get_fake_device(out), maybe_get_fake_device(weight))

        with FakeTensorMode():
            cpu_weight = torch.empty(20, 8)
            indices = torch.zeros(2, 3, dtype=torch.long, device=device)
            with self.assertRaisesRegex(
                (FakeTensorDeviceMismatchError, RuntimeError),
                "same device|different from other tensors",
            ):
                torch.nn.functional.embedding(indices, cpu_weight)

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO, "isinstance check for FakeTensor won't work with compile"
    )
    @onlyAccelerator
    def test_aten_copy_multi_device(self, device):
        with FakeTensorMode():
            x1 = torch.rand(4, device="cpu")
            x2 = torch.rand(4, device=device)
            copy1 = torch.ops.aten.copy.default(x1, x2)
            copy2 = torch.ops.aten.copy.default(x2, x1)
            out = torch.empty(4, device="cpu")
            torch.ops.aten.copy.out(x1, x2, out=out)
        self.checkType(copy1, "cpu", (4,))
        self.checkType(copy2, self.device_type, (4,))
        self.checkType(out, "cpu", (4,))

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO, "isinstance check for FakeTensor won't work with compile"
    )
    @onlyAccelerator
    def test_aten_index_multi_device(self, device):
        with FakeTensorMode():
            x1 = torch.rand(4, 4, device="cpu")
            x2 = torch.rand(4, 4, device=device)
            i1 = torch.tensor([0, 1], device=device)
            i2 = torch.tensor([0, 1], device="cpu")
            # NB: This one does not work: accelerator indices not allowed on cpu
            # tensor
            # r1 = torch.ops.aten.index(x1, i1)
            r2 = torch.ops.aten.index(x2, i2)

            y1 = torch.rand(4, device="cpu")
            y2 = torch.rand(4, device=device)
            j1 = torch.tensor([2], device=device)
            j2 = torch.tensor([2], device="cpu")
            r3 = torch.ops.aten.index_put.default(x1, j1, y1)
            r4 = torch.ops.aten.index_put.default(x2, j2, y2)
        # self.checkType(r1, "cpu", ())
        self.checkType(r2, self.device_type, ())
        self.checkType(r3, "cpu", (4, 4))
        self.checkType(r4, self.device_type, (4, 4))

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO, "isinstance check for FakeTensor won't work with compile"
    )
    @onlyAccelerator
    def test_aten_slice_scatter_multi_device(self, device):
        with FakeTensorMode():
            x1 = torch.rand(4, 4, device="cpu")
            y1 = torch.rand(2, 4, device=device)
            x2 = torch.rand(4, 4, device=device)
            y2 = torch.rand(2, 4, device="cpu")
            out = torch.empty(4, 4, device="cpu")
            r1 = torch.ops.aten.slice_scatter.default(x1, y1, start=2)
            r2 = torch.ops.aten.slice_scatter.default(x2, y2, start=2)
            r3 = torch.ops.aten.slice_scatter.out(x1, y1, out=out, start=2)
        self.checkType(r1, "cpu", (4, 4))
        self.checkType(r2, self.device_type, (4, 4))
        self.checkType(r3, "cpu", (4, 4))
        self.checkType(out, "cpu", (4, 4))

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO, "isinstance check for FakeTensor won't work with compile"
    )
    @onlyAccelerator
    def test_aten_set_data_multi_device(self, device):
        with FakeTensorMode():
            x1 = torch.rand(4, device="cpu")
            x2 = torch.rand(4, device=device)
            # cpu -> accelerator
            torch.ops.aten.shallow_copy_data_(x1, x2)
            self.checkType(x1, self.device_type, (4,))
            # accelerator -> cpu
            x3 = torch.rand(4, device=device)
            x4 = torch.rand(4, device="cpu")
            torch.ops.aten.shallow_copy_data_(x3, x4)
            self.checkType(x3, "cpu", (4,))

    @onlyAccelerator
    def test_lstm(self, device):
        # Ensure the non-cuDNN impl succeeds with fake tensors.
        with torch.backends.cudnn.flags(enabled=False):
            fake_tensor_mode = FakeTensorMode(allow_fallback_kernels=False)
            with fake_tensor_mode:
                N = 5
                L = 4
                H_in = 2
                hidden_size = 3
                proj_size = 2
                num_layers = 2
                bidir = False
                D = 2 if bidir else 1
                H_out = proj_size if proj_size > 0 else hidden_size

                lstm = torch.nn.LSTM(
                    input_size=H_in,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    proj_size=proj_size,
                    batch_first=False,
                    bias=True,
                    bidirectional=bidir,
                    device=device,
                )

                h_0 = torch.randn((num_layers * D, N, H_out), device=device)
                c_0 = torch.randn((num_layers * D, N, hidden_size), device=device)
                inp = torch.randn((L, N, H_in), device=device)
                (output, (h_n, c_n)) = lstm(inp, (h_0, c_0))
                output.sum().backward()

                self.assertEqual(output.shape, (L, N, D * H_out))
                self.assertEqual(h_n.shape, (D * num_layers, N, H_out))
                self.assertEqual(c_n.shape, (D * num_layers, N, hidden_size))

    @onlyAccelerator
    def test_gru(self, device):
        with torch.backends.cudnn.flags(enabled=False):
            fake_tensor_mode = FakeTensorMode(allow_fallback_kernels=False)
            with fake_tensor_mode:
                N = 5
                L = 4
                H_in = 2
                hidden_size = 3
                num_layers = 2
                bidir = False
                D = 2 if bidir else 1

                gru = torch.nn.GRU(
                    input_size=H_in,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=False,
                    bias=True,
                    bidirectional=bidir,
                    device=device,
                )

                h_0 = torch.randn((num_layers * D, N, hidden_size), device=device)
                inp = torch.randn((L, N, H_in), device=device)
                output, h_n = gru(inp, h_0)
                output.sum().backward()

                self.assertEqual(output.shape, (L, N, D * hidden_size))
                self.assertEqual(h_n.shape, (D * num_layers, N, hidden_size))


make_propagate_real_tensors_cls(TestFakeTensorDevice)
instantiate_device_type_tests(TestFakeTensorDevice, globals(), allow_xpu=True)
instantiate_device_type_tests(
    PropagateRealTensorsTestFakeTensorDevice,  # noqa: F821
    globals(),
    allow_xpu=True,
)


class TestFakeTensorCUDA(TestCase):
    """TestFakeTensor cases that need CUDA hardware or a CUDA build."""

    hw_classification = HardwareClassification.CUDA

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO, "isinstance check for FakeTensor won't work with compile"
    )
    @parametrize(
        "allow_fallback_kernels",
        [False, True],
        lambda a: "with_fallback" if a else "without_fallback",
    )
    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_cudnn_rnn(self, allow_fallback_kernels):
        def fn(
            a0,
            b0,
            b1,
            b2,
            b3,
            b4,
            b5,
            b6,
            b7,
            b8,
            b9,
            b10,
            b11,
            b12,
            b13,
            b14,
            b15,
            a3,
            a4,
            a5,
        ):
            a1 = [
                b0,
                b1,
                b2,
                b3,
                b4,
                b5,
                b6,
                b7,
                b8,
                b9,
                b10,
                b11,
                b12,
                b13,
                b14,
                b15,
            ]

            if torch.version.cuda:
                return torch.ops.aten._cudnn_rnn(
                    a0,
                    a1,
                    4,
                    a3,
                    a4,
                    a5,
                    2,
                    2048,
                    0,
                    2,
                    False,
                    0.0,
                    False,
                    True,
                    [],
                    None,
                )
            else:
                return torch.ops.aten.miopen_rnn(
                    a0,
                    a1,
                    4,
                    # weight_buf is cudnn only
                    a4,
                    a5,
                    2,
                    2048,
                    # proj_size is cudnn only
                    2,
                    False,
                    0.0,
                    False,
                    True,
                    [],
                    None,
                )

        mode = FakeTensorMode(allow_fallback_kernels=allow_fallback_kernels)
        for i, context in enumerate([contextlib.nullcontext, lambda: mode]):
            with context():
                inps1 = [
                    torch.randn([92, 8, 2048]).cuda(),
                    torch.randn([8192, 2048]).cuda(),
                    torch.randn([8192, 2048]).cuda(),
                    torch.randn([8192]).cuda(),
                    torch.randn([8192]).cuda(),
                    torch.randn([8192, 2048]).cuda(),
                    torch.randn([8192, 2048]).cuda(),
                    torch.randn([8192]).cuda(),
                    torch.randn([8192]).cuda(),
                    torch.randn([8192, 4096]).cuda(),
                    torch.randn([8192, 2048]).cuda(),
                    torch.randn([8192]).cuda(),
                    torch.randn([8192]).cuda(),
                    torch.randn([8192, 4096]).cuda(),
                    torch.randn([8192, 2048]).cuda(),
                    torch.randn([8192]).cuda(),
                    torch.randn([8192]).cuda(),
                    torch.randn([167837696]).cuda(),
                    torch.randn([4, 8, 2048]).cuda(),
                    torch.randn([4, 8, 2048]).cuda(),
                ]
                inps2 = inps1
                inps2[len(inps2) - 1] = None  # argument `cx` can be None

                for inps in [inps1, inps2]:
                    out = fn(*inps)
                    # weight_buf is cudnn only <- this is a3 and inps[-3]
                    # in cudnn_rnn, it passthroughs the inps[-3] to out[4],
                    #   so the test expects self.assertIs(out[4], inps[-3])
                    # in miopen_rnn, it does not need the input argument weight_buf,
                    #   the miopen_rnn generate a new weight_buf instead,
                    #   and it returns via out[4]
                    if torch.version.hip:
                        self.assertIsNotNone(out[4])
                    else:
                        self.assertIs(out[4], inps[-3])
                    for ten in out:
                        if i == 1:
                            self.assertTrue(is_fake_tensor(ten))
                        self.assertEqual(ten.device.type, "cuda")

    @unittest.skipIf(not torch.backends.cuda.is_built(), "requires CUDA build")
    def test_sparse_compressed_tensor_creation_device(self):
        if torch._functorch.config.fake_tensor_propagate_real_tensors and not RUN_CUDA:
            self.skipTest("propagate_real_tensors requires real CUDA tensors")
        cuda0 = torch.device("cuda:0")
        with FakeTensorMode():
            crow = torch.tensor([0, 2, 4])
            col = torch.tensor([0, 1, 0, 1])
            v = torch.randn(4)
            t = torch.sparse_csr_tensor(crow, col, v, size=(2, 2), device="cuda")
            self.assertEqual(t.device, cuda0)
            self.assertEqual(t.values().device, cuda0)
            self.assertEqual(t.crow_indices().device, cuda0)
            self.assertEqual(t.to_dense().device, cuda0)

    @unittest.skipIf(not torch.backends.cuda.is_built(), "requires CUDA build")
    @skipIfPropagateRealTensors("runs the real op, which reports its own device errors")
    def test_sparse_compressed_tensor_creation_device_mismatch(self):
        # the python ctor moves inputs to the requested device, so only the op trips this
        if torch._functorch.config.fake_tensor_propagate_real_tensors:
            self.skipTest("runs the real op, which reports its own device errors")
        error = "need to be on the same device"
        with FakeTensorMode():
            crow = torch.tensor([0, 2, 4])
            col = torch.tensor([0, 1, 0, 1])
            v = torch.randn(4)
            with self.assertRaisesRegex(RuntimeError, error):
                torch.ops.aten.sparse_compressed_tensor.comp_plain_value_size(
                    crow, col, v, [2, 2], layout=torch.sparse_csr, device="cuda"
                )
            cuda_args = [crow.to("cuda"), col.to("cuda"), v.to("cuda")]
            with self.assertRaisesRegex(RuntimeError, error):
                torch.ops.aten.sparse_compressed_tensor.comp_plain_value_size(
                    *cuda_args, [2, 2], layout=torch.sparse_csr, device="cuda:1"
                )
            with self.assertRaisesRegex(RuntimeError, error):
                torch.ops.aten.sparse_compressed_tensor.comp_plain_value_size(
                    *cuda_args, [2, 2], layout=torch.sparse_csr
                )

    @unittest.skipIf(not torch.cuda._is_compiled(), "requires CUDA-compiled PyTorch")
    def test_fake_device_guard_no_use_after_free(self):
        # Regression test: when CUDA is compiled but no devices are visible,
        # FakeTensorMode installs a FakeGuardImpl into the global
        # device_guard_impl_registry.  The impl must outlive all threads that
        # use it; previously it was stored as a thread-local unique_ptr, so
        # the pointer became dangling after the thread exited, causing a
        # segfault when a later thread called deviceCount() on it.
        #
        # CUDA_VISIBLE_DEVICES must be set before torch is imported, so we
        # run the repro in a subprocess.

        script = """\
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import threading
import torch

def f(x):
    return x * x

def run_compile():
    g = torch.compile(f, backend="eager")
    for i in range(10):
        g(torch.randn(i))

threads = [threading.Thread(target=run_compile) for _ in range(2)]
for t in threads:
    t.start()
for t in threads:
    t.join()

torch._dynamo.reset()

threads = [threading.Thread(target=run_compile) for _ in range(2)]
for t in threads:
    t.start()
for t in threads:
    t.join()
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            timeout=60,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=lambda msg: f"{msg}\nsubprocess failed:\n{result.stderr.decode()}",
        )

    @unittest.skipIf(not torch.cuda._is_compiled(), "requires CUDA-compiled PyTorch")
    def test_cuda_fake_tensor_new_methods_no_device_init(self):
        # Device visibility must be set before torch is imported, so run the
        # repro in a subprocess.

        script = """\
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""

# Import numpy before torch to avoid an MKL threading-layer conflict in this
# subprocess environment.
import numpy
import torch
from torch._subclasses.fake_tensor import (
    FakeTensorMode,
    is_fake_tensor,
    maybe_get_fake_constant,
)

fake_mode = FakeTensorMode()
with fake_mode:
    x = torch.randn(3, device="cuda")
    constant = torch.tensor([1.0])
    moved = constant.to(device="cuda")
    placed = torch.ops.prims.device_put.default(constant, torch.device("cuda"))

assert not torch.cuda.is_initialized()

new_tensors = [
    x.new_empty([2, 1], device="cuda:0", dtype=torch.float32),
    x.new_empty_strided([2, 1], [1, 2], device="cuda:0", dtype=torch.float32),
    x.new_full([2, 1], 1.0, device="cuda:0", dtype=torch.float32),
    x.new_ones([2, 1], device="cuda:0", dtype=torch.float32),
    x.new_zeros([2, 1], device="cuda:0", dtype=torch.float32),
    moved,
    placed,
]

for y in new_tensors:
    assert is_fake_tensor(y)
    assert y.device == torch.device("cuda:0")
    assert y.dtype is torch.float32

assert maybe_get_fake_constant(constant) is not None
assert maybe_get_fake_constant(moved) is None
assert maybe_get_fake_constant(placed) is None

assert not torch.cuda.is_initialized()
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            timeout=60,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=lambda msg: f"{msg}\nsubprocess failed:\n{result.stderr.decode()}",
        )


instantiate_parametrized_tests(TestFakeTensorCUDA)
make_propagate_real_tensors_cls(TestFakeTensorCUDA)


class TestFakeTensorMoveModule(TestCase):
    # No hw_classification: it uses GPU_TYPE, so it runs on whichever accelerator
    # is present (CUDA, XPU or MPS), which neither GENERIC nor one device label fits.
    # It isn't ACCELERATOR (a device parameter and instantiate_device_type_tests)
    # because the fake-only path has to stay reachable on a CUDA build with no GPU
    # present, where no cuda test variant would be generated.

    @unittest.skipIf(
        not TEST_ACCELERATOR and not torch.backends.cuda.is_built(),
        "requires an accelerator, or a CUDA build for the fake-only path",
    )
    def test_move_module_under_fake(self):
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)
                self.buffer = torch.nn.Buffer(torch.rand(2, 2))
                self.param = torch.nn.Parameter(torch.rand(2, 2))

            def forward(self, x):
                return self.linear(x) + self.buffer + self.param

        m = Module()
        input = torch.rand(2, 2)
        gpu_device = torch.device(GPU_TYPE, 0)

        with FakeTensorMode(allow_non_fake_inputs=True):
            m.to(device=gpu_device)
            arg = input.to(device=gpu_device)
            out = m(arg)

        for p in m.parameters():
            self.assertTrue(is_fake_tensor(p))
            self.assertEqual(p.device, gpu_device)
        for b in m.buffers():
            self.assertTrue(is_fake_tensor(b))
            self.assertEqual(b.device, gpu_device)

        self.assertTrue(is_fake_tensor(out))
        self.assertEqual(out.device, gpu_device)


class TestFakeTensorConstHandling(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def assertConst(self, *args):
        for arg in args:
            self.assertTrue(maybe_get_fake_constant(arg) is not None)

    def assertNotConst(self, *args):
        for arg in args:
            self.assertTrue(maybe_get_fake_constant(arg) is None)

    def test_simple(self):
        with FakeTensorMode():
            x = torch.tensor(4.0)
            self.assertEqual(x.item(), 4.0)

    def test_inplace_add(self):
        with FakeTensorMode():
            x = torch.tensor(4.0)
            y = x.add_(1)
            self.assertEqual(x.item(), 5.0)
            self.assertEqual(y.item(), 5.0)
            self.assertConst(x, y)

    def test_shared_storages(self):
        with FakeTensorMode():
            x = torch.tensor([4.0])
            y = x[:]

            self.assertEqual(x.storage()._cdata, y.storage()._cdata)
            xc, yc = maybe_get_fake_constant(x), maybe_get_fake_constant(y)
            self.assertEqual(xc.storage()._cdata, yc.storage()._cdata)

    def test_constant_invalidation(self):
        with FakeTensorMode():
            x = torch.tensor([1.0])
            self.assertConst(x)
            y = torch.rand([1])
            x.add_(y)
            self.assertNotConst(x)

    def test_inplace_view_invalidation(self):
        with FakeTensorMode():
            x = torch.tensor([1])
            self.assertConst(x)
            x.resize_([2])
            self.assertEqual(x.size(0), 2)
            self.assertNotConst(x)

    def test_fake_tensor_in_intlist_repro(self):
        def fn(tensors):
            max_size = torch.tensor([800, 1216], dtype=torch.int64)
            batch_shape = [len(tensors)] + list(tensors[0].shape[:-2]) + list(max_size)
            return tensors[0].new_full(batch_shape, 0.0)

        with self.assertRaises(
            torch._subclasses.fake_tensor.DataDependentOutputException
        ):
            with torch._subclasses.fake_tensor.FakeTensorMode():
                a = torch.randn(3, 800, 1199)
                b = torch.randn(3, 800, 800)
                inputs = [a, b]
                ref = fn(inputs)

    def test_fake_tensor_batch_norm_cpu(self):
        with torch._subclasses.CrossRefFakeMode():
            m = torch.nn.Sequential(
                torch.nn.BatchNorm2d(10),
                torch.nn.ReLU(),
            )
            m.eval()
            out = m(torch.randn([2, 10, 8, 8]))

    def test_shared_storage_invalidation(self):
        with FakeTensorMode():
            x = torch.tensor([1.0])
            y = x[:]
            self.assertConst(x, y)
            y.add_(torch.rand([1]))
            self.assertNotConst(x, y)

    def test_aliased_const_write(self):
        with FakeTensorMode():
            x = torch.tensor([1])
            y = x.expand([4])
            self.assertNotConst(y)
            y[0] = 1
            self.assertNotConst(x)

    def test_constant_propagate_through_functions(self):
        with FakeTensorMode():
            y = torch.div(4, 4, rounding_mode="trunc")
            self.assertConst(y)


make_propagate_real_tensors_cls(TestFakeTensorConstHandling)


def contains_type(type: torch.Type, maybe_contained_type: torch.Type):
    return maybe_contained_type.isSubtypeOf(type) or any(
        contains_type(e, maybe_contained_type) for e in type.containedTypes()
    )


class TestFakeTensorOpInfoDevice(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    @ops(custom_op_db, dtypes=OpDTypes.any_one)
    def test_fake(self, device, dtype, op):
        sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=False)
        for sample_input in sample_inputs_itr:
            args = (sample_input.input,) + sample_input.args
            kwargs = sample_input.kwargs
            optests.fake_check(op, args, kwargs)


make_propagate_real_tensors_cls(TestFakeTensorOpInfoDevice)
instantiate_device_type_tests(
    TestFakeTensorOpInfoDevice,
    globals(),
    allow_xpu=True,
)
instantiate_device_type_tests(
    PropagateRealTensorsTestFakeTensorOpInfoDevice,  # noqa: F821
    globals(),
    only_for=("cpu",),
)


@instantiate_parametrized_tests
class TestFakeTensorConverter(TestCase):
    hw_classification = HardwareClassification.GENERIC

    @fake_tensor_modes
    @parametrize("active", [False, True])
    def test_memoized_conversion_to_meta(self, mode_cls, active):
        x = torch.rand(2, 2, 2)
        mode = mode_cls()
        with mode if active else contextlib.nullcontext():
            fake = mode.from_tensor(x)
        self.assertIs(mode.from_tensor(x), fake)
        self.assertIs(mode.from_tensor(fake), fake)
        self.assertIs(maybe_get_fake_mode(fake), mode)
        self.assertEqual(fake.shape, x.shape)
        self.assertEqual(fake.stride(), x.stride())
        self.assertEqual(fake.dtype, x.dtype)
        self.assertEqual(fake.device, x.device)

    @fake_tensor_modes
    @parametrize("pytype", [None, torch.Tensor, torch.nn.Parameter])
    def test_memoized_conversion_from_meta(self, mode_cls, pytype):
        if mode_cls is CppFakeTensorMode and pytype is torch.nn.Parameter:
            self.skipTest("C++ fake tensors do not support Parameter pytype")
        if TEST_WITH_TORCHDYNAMO and mode_cls is FakeTensorMode and pytype is not None:
            self.skipTest(
                "Dynamo guards on pytype, which differs from the actual FakeTensor type"
            )
        x = torch.rand(2, 2).to(device="meta")
        mode = mode_cls()
        converter = mode.fake_tensor_converter
        keys = torch._C._dispatch_keys(torch.empty(2))
        fake = converter.from_meta_and_device(
            mode, x, "cpu", pytype=pytype, dispatch_keys=keys
        )
        self.assertIs(converter.from_meta_and_device(mode, x, "cpu"), fake)
        self.assertIs(maybe_get_fake_mode(fake), mode)
        self.assertEqual(fake.shape, x.shape)
        self.assertEqual(fake.device, torch.device("cpu"))
        if mode_cls is FakeTensorMode:
            self.assertIs(fake.pytype, pytype)
        self.assertEqual(maybe_get_fake_dispatch_keys(fake), keys)

    @fake_tensor_modes
    def test_from_meta_and_device_rejects_fake(self, mode_cls):
        mode = mode_cls()
        converter = mode.fake_tensor_converter
        meta_fake = converter.from_meta_and_device(
            mode, torch.empty(2, device="meta"), "meta"
        )
        cpu_fake = converter.from_meta_and_device(
            mode, torch.empty(2, device="meta"), "cpu"
        )
        with self.assertRaisesRegex(AssertionError, "not fake"):
            converter.from_meta_and_device(mode, meta_fake, "cpu")
        with in_kernel_invocation_manager(mode):
            with self.assertRaisesRegex(AssertionError, "not fake"):
                converter.from_meta_and_device(mode, cpu_fake, "cpu")

        cpp_mode = CppFakeTensorMode()
        cpp_fake = torch._C._from_meta_and_device(
            torch.empty(2, device="meta"), torch.device("meta"), cpp_mode
        )
        with self.assertRaisesRegex(AssertionError, "not fake"):
            converter.from_meta_and_device(mode, cpp_fake, "cpu")

    # Conversion of real views to C++ fakes is not implemented.
    @xfailIf(CPP_FAKETENSOR)
    def test_separate_tensor_storages_view(self):
        x = torch.rand(2, 2, 2)
        y = x[0]
        mode = FakeTensorMode()
        converter = mode.fake_tensor_converter
        x_conv = converter.from_real_tensor(mode, x)
        y_conv = converter.from_real_tensor(mode, y)
        self.assertEqual(torch._C._storage_id(x_conv), torch._C._storage_id(y_conv))

    @xfailIfTorchDynamo
    @fake_tensor_modes
    def test_separate_tensor_storages_non_view(self, mode_cls):
        x = torch.rand(2, 2, 2)
        y = torch.rand(4, 2)
        y.set_(x.storage())
        mode = mode_cls()
        converter = mode.fake_tensor_converter
        x_conv = converter.from_real_tensor(mode, x)
        y_conv = converter.from_real_tensor(mode, y)
        stor_id = torch._C._storage_id(x_conv)
        self.assertEqual(stor_id, torch._C._storage_id(y_conv))
        del x
        del x_conv
        self.assertEqual(len(converter.tensor_memo), 1)
        self.assertEqual(len(converter.meta_converter.storage_memo), 1)
        del y
        del y_conv
        self.assertEqual(len(converter.tensor_memo), 0)
        self.assertEqual(len(converter.meta_converter.storage_memo), 0)

    def test_parameter_views_keep_full_storage(self):
        base = torch.arange(12, dtype=torch.float16)
        x = torch.nn.Parameter(base[:6].view(2, 3))
        y = torch.nn.Parameter(base[6:].view(2, 3))
        self.assertFalse(x._is_view())
        self.assertFalse(y._is_view())

        mode = FakeTensorMode()
        x_conv = mode.from_tensor(x, static_shapes=True)
        y_conv = mode.from_tensor(y, static_shapes=True)

        self.assertEqual(torch._C._storage_id(x_conv), torch._C._storage_id(y_conv))
        self.assertEqual(
            x_conv.untyped_storage().nbytes(), base.untyped_storage().nbytes()
        )
        self.assertEqual(y_conv.storage_offset(), y.storage_offset())
        if torch._functorch.config.fake_tensor_propagate_real_tensors:
            x_real = maybe_get_real_tensor(x_conv)
            y_real = maybe_get_real_tensor(y_conv)
            self.assertEqual(
                x_real.untyped_storage().nbytes(),
                base.untyped_storage().nbytes(),
            )
            self.assertEqual(y_real.storage_offset(), y.storage_offset())
            self.assertEqual(x_real, x)
            self.assertEqual(y_real, y)

    def test_parameter_views_keep_full_storage_symbolic_propagate_real(self):
        base = torch.arange(12, dtype=torch.float16)
        x = torch.nn.Parameter(base[:6].view(2, 3))
        y = torch.nn.Parameter(base[6:].view(2, 3))

        with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
            mode = FakeTensorMode(shape_env=ShapeEnv())
            x_conv = mode.from_tensor(x)
            y_conv = mode.from_tensor(y)

        x_real = maybe_get_real_tensor(x_conv)
        y_real = maybe_get_real_tensor(y_conv)
        self.assertEqual(torch._C._storage_id(x_conv), torch._C._storage_id(y_conv))
        self.assertEqual(
            x_real.untyped_storage().nbytes(), base.untyped_storage().nbytes()
        )
        self.assertEqual(
            y_real.untyped_storage().nbytes(), base.untyped_storage().nbytes()
        )
        self.assertEqual(y_real.storage_offset(), y.storage_offset())
        self.assertEqual(x_real, x)
        self.assertEqual(y_real, y)

    def test_strided_parameter_views_keep_real_storage_data(self):
        base = torch.arange(11, dtype=torch.float32)
        x = torch.nn.Parameter(base[::2])
        y = torch.nn.Parameter(base[1::2])
        self.assertFalse(x._is_view())
        self.assertFalse(y._is_view())

        with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
            mode = FakeTensorMode()
            x_conv = mode.from_tensor(x, static_shapes=True)
            y_conv = mode.from_tensor(y, static_shapes=True)

        self.assertEqual(torch._C._storage_id(x_conv), torch._C._storage_id(y_conv))
        self.assertEqual(maybe_get_real_tensor(x_conv), x)
        self.assertEqual(maybe_get_real_tensor(y_conv), y)

    def test_functional_tensor_offset_storage_copy(self):
        from torch._subclasses.functional_tensor import (
            FunctionalTensor,
            FunctionalTensorMode,
        )

        base = torch.arange(12, dtype=torch.float32)
        y = torch.nn.Parameter(base[6:].view(2, 3))

        with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
            mode = FakeTensorMode()
            y_fake = mode.from_tensor(y, static_shapes=True)
            with FunctionalTensorMode():
                y_functional = FunctionalTensor.to_functional(y_fake)
                refake_mode = FakeTensorMode()
                y_refake = refake_mode.from_tensor(y_functional, static_shapes=True)

        self.assertEqual(y_refake.storage_offset(), y.storage_offset())
        self.assertEqual(maybe_get_real_tensor(y_refake), y)

    def test_refake_unbacked_storage_static_shapes(self):
        with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=False):
            shape_env = ShapeEnv()
            fake_mode = FakeTensorMode(shape_env=shape_env)
            with shape_env.ignore_fresh_unbacked_symbols():
                fake = fake_mode.from_tensor(
                    torch.randn(2, 3),
                    source=LocalSource("x", is_input=True),
                    symbolic_context=StatelessSymbolicContext(
                        dynamic_sizes=[DimDynamic.UNBACKED, DimDynamic.STATIC],
                        constraint_sizes=[None, None],
                    ),
                )

            refake_mode = FakeTensorMode()
            refake = refake_mode.from_tensor(fake, static_shapes=True)

        self.assertTrue(free_unbacked_symbols(refake.shape[0]))
        self.assertEqual(refake.shape[1], 3)

    # Conversion of real views to C++ fakes is not implemented.
    @xfailIf(CPP_FAKETENSOR)
    def test_dead_weak_ref(self):
        x = torch.rand(2, 2, 2)
        y = x[0]
        mode = FakeTensorMode()
        converter = FakeTensorConverter(cpp_fakes=CPP_FAKETENSOR)
        x_conv = converter.from_real_tensor(mode, x)
        x_conv_storage = x_conv.untyped_storage()
        del x_conv
        self.assertFalse(x in converter.tensor_memo)
        y_conv = converter.from_real_tensor(mode, y)
        self.assertIs(x_conv_storage, y_conv.untyped_storage())

    @xfailIfTorchDynamo
    @fake_tensor_modes
    def test_dead_key(self, mode_cls):
        x = torch.rand(2, 2, 2)
        mode = mode_cls()
        converter = mode.fake_tensor_converter
        x_conv = converter.from_real_tensor(mode, x)
        self.assertEqual(len(converter.tensor_memo), 1)
        x_conv2 = converter.from_real_tensor(mode, x)
        if x_conv2 is not x_conv:
            raise AssertionError("expected x_conv2 is x_conv")
        del x
        del x_conv
        del x_conv2
        self.assertEqual(len(converter.tensor_memo), 0)

    @fake_tensor_modes
    def test_no_active_mode(self, mode_cls):
        if mode_cls is CppFakeTensorMode:
            if TEST_WITH_TORCHDYNAMO:
                self.skipTest(_CPP_FAKE_DYNAMO_SKIP)
            if torch._functorch.config.fake_tensor_propagate_real_tensors:
                self.skipTest(
                    "C++ fake fallback does not support propagate_real_tensors"
                )
        x = torch.empty(2, 2, device="cpu")
        y = torch.empty(2, 2, device="cpu")
        with mode_cls() as mode:
            x = mode.from_tensor(x)
            y = mode.from_tensor(y)

        out = x + y
        self.assertEqual(mode, maybe_get_fake_mode(out))
        self.assertTrue(is_fake_tensor(out))
        self.assertEqual(out.device.type, "cpu")

    @fake_tensor_modes
    def test_multiple_modes(self, mode_cls):
        if mode_cls is CppFakeTensorMode:
            if TEST_WITH_TORCHDYNAMO:
                self.skipTest(_CPP_FAKE_DYNAMO_SKIP)
            if torch._functorch.config.fake_tensor_propagate_real_tensors:
                self.skipTest(
                    "C++ fake fallback does not support propagate_real_tensors"
                )
        t = torch.rand([4])
        t2 = torch.rand([4])
        with mode_cls() as m:
            with mode_cls() as m2:
                t_fake = m.from_tensor(t)
                t2_fake = m2.from_tensor(t2)

                with self.assertRaisesRegex(AssertionError, "Mixing fake modes"):
                    t_fake + t2_fake

    def test_separate_mode_error(self):
        with FakeTensorMode():
            x = torch.empty(2, 2, device="cpu")
        with FakeTensorMode():
            y = torch.empty(2, 2, device="cpu")
        self.assertRaises(Exception, lambda: x, y)

    @skipIfCppFakeTensor("no Python FakeTensor reference-cycle support")
    @xfailIfTorchDynamo
    def test_no_ref_cycle(self):
        x = torch.rand([4])
        mode = FakeTensorMode()
        y = mode.from_tensor(x)
        self.assertEqual(len(mode.fake_tensor_converter.tensor_memo), 1)
        mode_weak = weakref.ref(mode)
        y_weak = weakref.ref(mode)
        del mode
        del y
        if mode_weak() is not None:
            raise AssertionError("expected mode_weak() is None")
        if y_weak() is not None:
            raise AssertionError("expected y_weak() is None")

    def test_grad_dtype_preserved(self):
        t = torch.randn(4, dtype=torch.bfloat16, requires_grad=True)
        t.grad_dtype = torch.float32

        mode = FakeTensorMode()
        fake_t = mode.from_tensor(t)
        self.assertEqual(fake_t.grad_dtype, torch.float32)

    def test_grad_dtype_none_preserved(self):
        t = torch.randn(4, dtype=torch.bfloat16, requires_grad=True)
        t.grad_dtype = None

        mode = FakeTensorMode()
        fake_t = mode.from_tensor(t)
        self.assertIsNone(fake_t.grad_dtype)

    def test_grad_dtype_functional_tensor_no_crash(self):
        from torch._subclasses.functional_tensor import (
            FunctionalTensor,
            FunctionalTensorMode,
        )

        t = torch.randn(4, dtype=torch.bfloat16, requires_grad=True)
        t.grad_dtype = torch.float32

        mode = FakeTensorMode()
        fake_t = mode.from_tensor(t)
        self.assertEqual(fake_t.grad_dtype, torch.float32)

        with FunctionalTensorMode():
            func_t = FunctionalTensor.to_functional(fake_t)
            # Re-fakifying a FunctionalTensor should not crash even though
            # the inner tensor has a custom grad_dtype.
            re_faked = mode.from_tensor(func_t)
        self.assertTrue(re_faked.requires_grad)

    @skipIfTorchDynamo("make_fx tracing is incompatible with dynamo")
    def test_grad_dtype_make_fx(self):
        def train_step(w):
            y = (w.float() * 2).sum()
            (g,) = torch.autograd.grad(y, w)
            return g

        w = torch.randn(4, dtype=torch.bfloat16, requires_grad=True)
        w.grad_dtype = torch.float32

        g_eager = train_step(w)

        fake_mode = FakeTensorMode()
        w_fake = fake_mode.from_tensor(w)
        with fake_mode:
            gm = make_fx(train_step)(w_fake)
        g_traced = gm(w)

        self.assertEqual(g_eager.dtype, g_traced.dtype)

    @fake_tensor_modes
    def test_static_shapes(self, mode_cls):
        self.assertTrue(mode_cls().static_shapes)
        mode = mode_cls(shape_env=ShapeEnv())
        self.assertFalse(mode.static_shapes)
        # It decides what from_tensor builds when not told
        self.assertIsInstance(mode.from_tensor(torch.randn(4)).shape[0], torch.SymInt)
        static = mode_cls(shape_env=ShapeEnv(), static_shapes=True)
        self.assertIsInstance(static.from_tensor(torch.randn(4)).shape[0], int)
        # A symbolic_context picks the dynamic sizes, and conflicts with
        # static_shapes
        context = StatelessSymbolicContext(
            dynamic_sizes=[DimDynamic.DYNAMIC, DimDynamic.STATIC],
            constraint_sizes=[None, None],
        )
        dynamic = mode.from_tensor(torch.randn(4, 5), symbolic_context=context)
        self.assertIsInstance(dynamic.shape[0], torch.SymInt)
        self.assertIsInstance(dynamic.shape[1], int)
        real = torch.randn(4, 5)
        with self.assertRaisesRegex(AssertionError, "cannot set both"):
            mode.from_tensor(real, static_shapes=True, symbolic_context=context)

    @fake_tensor_modes
    def test_constant_conversion(self, mode_cls):
        mode = mode_cls()
        real = torch.ones(2)
        converter = mode.fake_tensor_converter
        const = converter.from_real_tensor(mode, real, make_constant=True)
        self.assertIs(maybe_get_fake_constant(const), real)
        # A 0-d CPU tensor converted without a ShapeEnv memoizes its value
        scalar = mode.from_tensor(torch.tensor(3))
        self.assertEqual(maybe_get_item_memo(scalar), 3)


make_propagate_real_tensors_cls(TestFakeTensorConverter)


class TestFakeTensorOperatorInvariants(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def get_aten_op(self, schema):
        namespace, name = schema.name.split("::")
        overload = schema.overload_name if schema.overload_name else "default"
        if namespace != "aten":
            raise AssertionError(f"expected namespace 'aten', got {namespace!r}")
        return getattr(getattr(torch.ops.aten, name), overload)

    def get_all_aten_schemas(self):
        for schema in torch._C._jit_get_all_schemas():
            namespace = schema.name.split("::")[0]
            if namespace != "aten":
                continue
            yield schema

    def test_non_kwarg_only_device(self):
        for schema in self.get_all_aten_schemas():
            ten_type = torch._C.TensorType.get()
            if not any(
                contains_type(arg.type, ten_type)
                for arg in itertools.chain(schema.arguments, schema.returns)
            ):
                continue

            opt_device = torch._C.OptionalType(torch._C.DeviceObjType.get())
            has_non_kwarg_device = any(
                not arg.kwarg_only and arg.type.isSubtypeOf(opt_device)
                for arg in schema.arguments
            )
            if has_non_kwarg_device:
                self.assertTrue(
                    self.get_aten_op(schema)
                    in torch._subclasses.fake_tensor._device_not_kwarg_ops
                )

    def test_tensor_constructors_all_have_kwarg_device(self):
        for schema in self.get_all_aten_schemas():
            op = self.get_aten_op(schema)
            if not torch._subclasses.fake_tensor._is_tensor_constructor(op):
                continue

            opt_device = torch._C.OptionalType(torch._C.DeviceObjType.get())
            has_kwarg_device = any(
                arg.kwarg_only and arg.type.isSubtypeOf(opt_device)
                for arg in schema.arguments
            )

            self.assertTrue(
                has_kwarg_device or op == torch.ops.aten._list_to_tensor.default
            )

    @unittest.expectedFailure
    def test_sparse_new(self):
        with FakeTensorMode():
            indices = torch.randn(1, 1, dtype=torch.int64)
            values = torch.randn(1)
            extra = (2,)
            sparse = torch.randn(1).to_sparse()
            # This used to segfault, now it does not, but it still raises an
            # error
            sparse2 = sparse.new(indices, values, extra)

    def test_tensor_new(self):
        with FakeTensorMode():
            x = torch.Tensor([1, 2, 3])
        self.assertTrue(is_fake_tensor(x))

    def test_like_ops(self):
        for schema in self.get_all_aten_schemas():
            if "_like" == schema.name[-5:]:
                op = self.get_aten_op(schema)
                self.assertIn(
                    op, torch._subclasses.fake_tensor._like_tensor_constructors
                )

    def test_str_storage(self):
        x = torch.zeros(3)
        with FakeTensorMode() as m:
            y = m.from_tensor(x)
            self.assertExpectedInline(
                str(x.storage()),
                """\
 0.0
 0.0
 0.0
[torch.storage.TypedStorage(dtype=torch.float32, device=cpu) of size 3]""",
            )
            self.assertExpectedInline(
                str(y.storage()),
                """\
...
[torch.storage.TypedStorage(dtype=torch.float32, device=meta) of size 3]""",
            )

        self.assertExpectedInline(
            str(y.storage()),
            """\
...
[torch.storage.TypedStorage(dtype=torch.float32, device=meta) of size 3]""",
        )

    # at::_embedding_bag has no op info,
    # and returns extra tensors that at::embedding bag throws away
    def test_embedding_bag_private(self):
        args = [
            torch.ones(6, 1),
            torch.ones(6, dtype=torch.int64),
            torch.arange(2, dtype=torch.int64),
            False,
            2,  # mode = max
        ]

        ref_out = torch.ops.aten._embedding_bag(*args)
        with FakeTensorMode() as m:
            meta_args = [
                m.from_tensor(a) if isinstance(a, torch.Tensor) else a for a in args
            ]
            meta_out = torch.ops.aten._embedding_bag(*meta_args)

        self.assertEqual(len(ref_out), len(meta_out))
        for ref_o, meta_o in zip(ref_out, meta_out):
            self.assertEqual(ref_o.size(), meta_o.size())

    def test_cross_entropy_loss(self):
        inp = torch.randn(3, 5)
        target = torch.randint(5, (3,), dtype=torch.long)
        weight = torch.rand(5)
        fn = torch.nn.functional.cross_entropy
        for w in (weight, None):
            args = (inp, target, w)
            ref = fn(*args)
            with FakeTensorMode() as m:
                meta_args = [
                    m.from_tensor(a) if isinstance(a, torch.Tensor) else a for a in args
                ]
                meta_out = torch.nn.functional.cross_entropy(
                    *meta_args, label_smoothing=0.5
                )

            self.assertEqual(ref.size(), meta_out.size())

    # IMPORTANT!!! Always run even if CUDA is not available: this guards that
    # naming a GPU device under fake mode skips device lazy init (#122636).
    # Hence GPU_TYPE rather than a device-generic class -- the latter only emits
    # variants for devices the machine has, degrading this to cpu == cpu.
    @skipIfPropagateRealTensors("Propagate real tensor not supported")
    def test_fake_gpu_no_init(self):
        # Skip this test, we will try to run CUDA operations to real prop so
        # it clearly will not work on CPU runner
        if torch._functorch.config.fake_tensor_propagate_real_tensors:
            self.skipTest("Propagate real tensor not supported")

        with FakeTensorMode(allow_non_fake_inputs=True):
            self.assertEqual(torch.empty(10, device=GPU_TYPE).device.type, GPU_TYPE)
            self.assertEqual(torch.ones(10, device=GPU_TYPE).device.type, GPU_TYPE)
            self.assertEqual(torch.zeros(10, device=GPU_TYPE).device.type, GPU_TYPE)
            self.assertEqual(torch.rand(10, device=GPU_TYPE).device.type, GPU_TYPE)
            self.assertEqual(torch.tensor(3.14, device=GPU_TYPE).device.type, GPU_TYPE)
            self.assertEqual(
                torch.tensor([[3.14, 2], [1, 2]], device=GPU_TYPE).device.type, GPU_TYPE
            )

    def test_convolution_backward_channels_last_memory_format(self):
        """Regression test: meta convolution_backward must predict channels_last
        output strides when inputs are channels_last, matching CUDA/MPS backends.
        See https://github.com/pytorch/pytorch/issues/171622
        """
        with FakeTensorMode():
            # channels_last inputs: outputs should be channels_last
            grad_out = torch.randn(2, 3, 4, 4).to(memory_format=torch.channels_last)
            inp = torch.randn(2, 3, 4, 4).to(memory_format=torch.channels_last)
            weight = torch.randn(3, 3, 3, 3).to(memory_format=torch.channels_last)
            grad_input, grad_weight, _ = torch.ops.aten.convolution_backward(
                grad_out,
                inp,
                weight,
                [3],
                [1, 1],
                [1, 1],
                [1, 1],
                False,
                [0, 0],
                1,
                [True, True, True],
            )
            self.assertTrue(grad_input.is_contiguous(memory_format=torch.channels_last))
            self.assertTrue(
                grad_weight.is_contiguous(memory_format=torch.channels_last)
            )

            # contiguous inputs: outputs should be contiguous
            grad_out_c = torch.randn(2, 3, 4, 4)
            inp_c = torch.randn(2, 3, 4, 4)
            weight_c = torch.randn(3, 3, 3, 3)
            grad_input_c, grad_weight_c, _ = torch.ops.aten.convolution_backward(
                grad_out_c,
                inp_c,
                weight_c,
                [3],
                [1, 1],
                [1, 1],
                [1, 1],
                False,
                [0, 0],
                1,
                [True, True, True],
            )
            self.assertTrue(grad_input_c.is_contiguous())
            self.assertTrue(grad_weight_c.is_contiguous())

    def test_no_dispatch_with_like_function(self):
        class CountingMode(TorchDispatchMode):
            def __init__(self) -> None:
                self.count = 0

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                self.count += 1
                return func(*args, **kwargs)

        with FakeTensorMode():
            x = torch.randn(2)
            with CountingMode() as mode:
                with no_dispatch():
                    torch.zeros_like(x)

        self.assertEqual(mode.count, 0)


make_propagate_real_tensors_cls(TestFakeTensorOperatorInvariants)


class TestFakeTensorOperatorInvariantsDevice(TestCase):
    r"""Accelerator-only TestFakeTensorOperatorInvariants cases (see except_for below)."""

    hw_classification = HardwareClassification.ACCELERATOR

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION,
        "Does not support SDPA or pre-SM80 hardware",
    )
    def test_flash_attention(self, device):
        class Repro(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, arg1, arg2, arg3):
                torch.ops.aten._scaled_dot_product_flash_attention(
                    arg1, arg2, arg3, scale=0.17677669529663687
                )

        args_new = [
            [
                ((1, 48, 64, 64), (0, 4096, 64, 1), torch.float16, device),
                ((1, 48, 64, 64), (0, 4096, 64, 1), torch.float16, device),
                ((1, 48, 64, 64), (0, 4096, 64, 1), torch.float16, device),
            ],
            [
                ((4, 2, 16, 32), (1024, 512, 32, 1), torch.float16, device),
                ((4, 2, 16, 32), (1024, 512, 32, 1), torch.float16, device),
                ((4, 2, 16, 32), (1024, 512, 32, 1), torch.float16, device),
            ],
        ]
        for args_list in args_new:
            args = [
                rand_strided(bsz, num_heads, seq_len, head_dim)
                for (bsz, num_heads, seq_len, head_dim) in args_list
            ]
            try:
                with torch._subclasses.CrossRefFakeMode():
                    Repro()(*args)
            except MetadataMismatchError as e:
                # We expect the cross ref to succeed for the first output to fail
                # for the rng state, see Note [Seed and Offset]
                self.assertTrue("output[0]" not in str(e))
                if self.__class__.__name__.startswith("PropagateRealTensors"):
                    self.assertTrue(
                        "Real tensor propagation found a metadata mismatch" in str(e)
                    )
                else:
                    self.assertTrue(
                        "found mismatched tensor metadata for output" in str(e)
                    )

    @skipIfPropagateRealTensors("Propagate real tensor not supported")
    def test_move_meta_tensor(self, device):
        if torch._functorch.config.fake_tensor_propagate_real_tensors:
            self.skipTest("Propagate real tensor not supported")

        meta_tensor = torch.ones(2, device="meta")
        with FakeTensorMode(allow_non_fake_inputs=True):
            self.assertEqual(meta_tensor.to(device="cpu").device.type, "cpu")
            self.assertEqual(
                meta_tensor.to(device=device).device.type, self.device_type
            )

    def test_conv_c1_backward(self, device):
        class Repro(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, arg1, arg2, arg3):
                torch.ops.aten.convolution_backward.default(
                    arg1,
                    arg2,
                    arg3,
                    [1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    False,
                    [0, 0],
                    1,
                    [True, True, False],
                )

        args_new = [
            ((16, 1, 128, 128), (16384, 16384, 128, 1), torch.float16, device),
            ((16, 64, 128, 128), (1048576, 1, 8192, 64), torch.float16, device),
            ((1, 64, 3, 3), (576, 9, 3, 1), torch.float16, device),
        ]
        args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args_new]

        with torch._subclasses.CrossRefFakeMode():
            Repro()(*args)

    # PropagateRealTensors installs weakrefs
    @unittest.skipIf(
        IS_LINUX or TEST_WITH_ROCM or TEST_WITH_SLOW,
        "https://github.com/pytorch/pytorch/issues/165387",
    )
    @expectedFailurePropagateRealTensors
    def test_module_to(self, device):
        def _check_device(sd, expected):
            for v in sd.values():
                self.assertEqual(v.device.type, expected)

        with FakeTensorMode():
            m = torch.nn.Linear(2, 2)
            _check_device(m.state_dict(), "cpu")
            m.to(device)
            _check_device(m.state_dict(), self.device_type)


make_propagate_real_tensors_cls(TestFakeTensorOperatorInvariantsDevice)
instantiate_device_type_tests(
    TestFakeTensorOperatorInvariantsDevice,
    globals(),
    except_for=("cpu",),
    allow_xpu=True,
)
instantiate_device_type_tests(
    PropagateRealTensorsTestFakeTensorOperatorInvariantsDevice,  # noqa: F821
    globals(),
    except_for=("cpu",),
    allow_xpu=True,
)


class TestFakeTensorProp(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_fake_tensor_prop_on_nn_module(self):
        class ToyNnModuleWithParameters(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer1 = torch.nn.Linear(4, 3)
                self.layer2 = torch.nn.Linear(3, 2)

            def forward(self, value):
                value = self.layer1(value)
                value = torch.relu(value)
                value = self.layer2(value)
                return value

        model = ToyNnModuleWithParameters()
        value = torch.randn(5, 4)
        # Convert nn.Module to GraphModule so that FakeTensorProp runs.
        graph_model = torch.fx.symbolic_trace(model, (value,))
        # The following block runs FakeTensorProp on graph_module w/to the same FakeTensorMode
        #
        # TODO(wschin): there should be an API to run FakeTensorProp for GraphModule
        # with parameters and buffers.
        with FakeTensorMode() as fake_tensor_mode:

            def to_fake_tensor(x):
                if isinstance(x, torch.Tensor) and not is_fake_tensor(x):
                    return fake_tensor_mode.from_tensor(x)
                return x

            fake_parameters_and_buffers = {
                k: to_fake_tensor(v)
                for k, v in itertools.chain(
                    graph_model.named_parameters(), graph_model.named_buffers()
                )
            }
            with torch.nn.utils.stateless._reparametrize_module(
                graph_model, fake_parameters_and_buffers
            ):
                # This case uses the **same** fake tensor mode to
                #  1. create fake parameters and fake buffers, and
                #  2. run FakeTensorProp
                # The result should be correct.
                result = FakeTensorProp(graph_model, fake_tensor_mode).propagate(value)
                self.assertTrue(is_fake_tensor(result))
                self.assertEqual(result.shape, (5, 2))
                # This case uses the **different** fake tensor modes to
                #  1. create fake parameters and fake buffers, and
                #  2. run FakeTensorProp
                # The following code should fail.
                failed = False
                # Python FakeTensor fails the refake with AssertionError (device
                # must be meta). A C++ fake is a plain torch.Tensor already bound
                # to a Tensor PyObject, so the mismatched mode instead trips
                # _make_subclass with a RuntimeError; both signal the same reject.
                refake_errors = expectedIfCppFakeTensor(
                    (AssertionError, RuntimeError), AssertionError
                )
                try:
                    FakeTensorProp(graph_model).propagate(value)
                except refake_errors:
                    failed = True
                self.assertTrue(failed)

    def test_fake_tensor_prop_on_nn_module_with_optional_args(self):
        class OptionalArgumentInBetween(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer1 = torch.nn.Linear(4, 3)
                self.layer2 = torch.nn.Linear(3, 2)

            def forward(self, value, another_value=None, another_optional_value=None):
                # Mimic huggingface's `forward` methods which have several optional arguments.
                # For example, GPT accepts forward(self, input_ids, None, attention_mask, ...).
                # To apply FakeTensorProp, its from_real_tensor(...) needs to accept None.
                if another_value is None:
                    another_value = torch.rand_like(value)
                if another_optional_value is None:
                    another_optional_value = torch.rand_like(value)
                value = value + another_value + another_optional_value
                return value * value

        fake_mode = FakeTensorMode(
            allow_non_fake_inputs=True, allow_fallback_kernels=False
        )
        with fake_mode:
            model = OptionalArgumentInBetween()
            value = torch.randn(5, 4)
            another_optional_value = torch.randn(5, 4)
            graph_model = torch.fx.symbolic_trace(
                model, (value, None, another_optional_value)
            )
            FakeTensorProp(graph_model, fake_mode).propagate(
                value, None, another_optional_value
            )

    def test_unbacked_shape_realloc(self):
        def f(x):
            return x.nonzero()

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)
        with fake_mode:
            value = torch.randn(5)
            gm = make_fx(f)(value)
        nonzero_nodes = [
            n for n in gm.graph.nodes if n.target is torch.ops.aten.nonzero.default
        ]
        self.assertEqual(len(nonzero_nodes), 1)
        self.assertIsInstance(nonzero_nodes[0].meta["val"].shape[0], torch.SymInt)
        u0 = nonzero_nodes[0].meta["val"].shape[0]
        FakeTensorProp(gm, fake_mode).propagate(value)
        u1 = nonzero_nodes[0].meta["val"].shape[0]
        # Test that this test is actually doing something in that the
        # FakeTensorProp actually triggered a reallocation.  If this assert is
        # failing, it could be because we started memoizing the nnz count for
        # nonzero, which is nice in some sense (no reallocation) but not
        # helpful for this test, which is checking what we do when we have
        # to reallocate.  If so, you need to make this example more
        # complicated (e.g., maybe have a nontrivial computation on the input
        # before feeding it into nonzero, or have some sort of randomness)
        self.assertIsNot(u0, u1)
        self.assertTrue(statically_known_true(u0 == u1))

    def test_nonzero_stride(self):
        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)
        with fake_mode:
            value = torch.ones(5)
            fake_r = value.nonzero()

        r = torch.ones(5).nonzero()

        self.assertEqual(fake_r.T.is_contiguous(), r.T.is_contiguous())

    def test_nonzero_numpy_arity_matches_eager(self):
        # Normal dispatch decomposes this CIA op before fake mode sees it; the
        # fake rule is only used when export preserves the op, so call it directly.
        from torch._subclasses.fake_impls import op_implementations_dict

        op = torch.ops.aten.nonzero_numpy.default
        rule = op_implementations_dict[op]
        for shape in [(), (3,), (2, 3), (2, 0, 3)]:
            x = torch.randn(shape)
            expected = torch.nonzero(x, as_tuple=True)
            fake_mode = FakeTensorMode(shape_env=ShapeEnv())
            actual = rule(fake_mode, op, fake_mode.from_tensor(x))
            self.assertEqual(len(actual), len(expected))
            self.assertEqual([t.dim() for t in actual], [t.dim() for t in expected])

    def test_nan_to_num(self):
        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)
        with fake_mode:
            x = torch.randn(5, 10).t()
            y = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        self.assertEqual(x.size(), y.size())
        self.assertEqual(x.stride(), y.stride())

    def test_inference_mode_nonzero_memo(self):
        """D106102842: nonzero() memo must be stored for inference FakeTensors.

        Without the fix, SymNumberMemoDescriptor.__set__ skipped memo storage
        for inference tensors, so each boolean indexing (x[mask]) allocated a
        fresh unbacked symint. Two tensors filtered by the same mask would get
        independent symbols, causing GuardOnDataDependentSymNode failures.
        """
        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)
        with fake_mode:
            with torch.inference_mode():
                x = torch.randn(10, 3)
                y = torch.randn(10, 5)
                mask = torch.randn(10) > 0

                self.assertTrue(is_fake_tensor(x))
                self.assertTrue(x.is_inference())

                filtered_x = x[mask]
                filtered_y = y[mask]

                sx = filtered_x.shape[0]
                sy = filtered_y.shape[0]

                # Both shapes should be the same unified symbol because
                # mask.nonzero() is memoized.
                self.assertTrue(statically_known_true(sx == sy))

    @parametrize(
        "op",
        [torch.nonzero, torch.unique, torch.unique_consecutive],
        name_fn=lambda op: op.__name__,
    )
    def test_nonzero_unique_memo_invalidation(self, op):
        # The memo is reused until the input is mutated or the epoch changes
        mode = FakeTensorMode(shape_env=ShapeEnv())
        with mode:
            x = torch.randint(0, 3, (10,))
            n = op(x).shape[0]
            self.assertTrue(statically_known_true(op(x).shape[0] == n))
            x.add_(1)
            m = op(x).shape[0]
            self.assertFalse(statically_known_true(m == n))
            mode.epoch += 1
            self.assertFalse(statically_known_true(op(x).shape[0] == m))

    @parametrize(
        "op",
        [torch.nonzero, torch.unique, torch.unique_consecutive],
        name_fn=lambda op: op.__name__,
    )
    @unittest.skipIf(
        not CPP_FAKETENSOR, "Python FakeTensor keeps the memo across set_data"
    )
    def test_nonzero_unique_memo_dropped_by_set_data(self, op):
        # A C++ fake keeps these memos on the C++ tensor, which set_data
        # replaces. A Python FakeTensor keeps them, since set_data does not bump
        # the version counter.
        if not CPP_FAKETENSOR:
            self.skipTest("Python FakeTensor keeps the memo across set_data")
        with FakeTensorMode(shape_env=ShapeEnv()):
            x = torch.randint(0, 3, (10,))
            n = op(x).shape[0]
            x.data = torch.randint(0, 3, (20,))
            self.assertFalse(statically_known_true(op(x).shape[0] == n))

    @unittest.skipIf(not CPP_FAKETENSOR, "Python FakeTensor raises on the get as well")
    def test_failed_sym_memo_set_leaves_no_memo(self):
        # Recording a memo reads the version counter, which a non-inference
        # tensor has disabled after inference_tensor.data = tensor. The failed
        # set must not leave a memo behind for a later get.
        if not CPP_FAKETENSOR:
            self.skipTest("Python FakeTensor raises on the get as well")
        from torch._subclasses.fake_tensor import maybe_get_sym_memo, maybe_set_sym_memo

        with FakeTensorMode(shape_env=ShapeEnv()):
            with torch.inference_mode():
                x = torch.randint(0, 3, (10,))
            x.data = torch.randint(0, 3, (10,))
            self.assertFalse(x.is_inference())
            with self.assertRaisesRegex(RuntimeError, "version counter"):
                maybe_set_sym_memo(x, "nonzero_memo", 3)
            self.assertIsNone(maybe_get_sym_memo(x, "nonzero_memo"))

    @parametrize(
        "op",
        [torch.nonzero, torch.unique, torch.unique_consecutive],
        name_fn=lambda op: op.__name__,
    )
    def test_save_after_nonzero_unique_memo(self, op):
        # The memos must not become tensor state that torch.save pickles, which
        # would also make the default weights_only torch.load reject the file
        with FakeTensorMode(shape_env=ShapeEnv()):
            x = torch.randint(0, 3, (10,))
            op(x)
        buf = io.BytesIO()
        with torch.serialization.skip_data(materialize_fake_tensors=True):
            torch.save(x, buf)
        buf.seek(0)
        self.assertEqual(torch.load(buf).shape, x.shape)


instantiate_parametrized_tests(TestFakeTensorProp)
make_propagate_real_tensors_cls(TestFakeTensorProp)


class TestFakeTensorPropDevice(TestCase):
    r"""Accelerator-only TestFakeTensorProp cases (see except_for below)."""

    hw_classification = HardwareClassification.ACCELERATOR

    def test_torch_load_with_fake_mode(self, device):
        model = torch.nn.Linear(5, 10)
        sd = model.state_dict()
        sd["tt"] = TwoTensor(torch.randn(2), torch.randn(2))

        def _read_tensor_and_check(key, sd_loaded, sd_orig, all_bytes, device_type):
            dtype = torch.float32
            t = sd_loaded[key]
            self.assertEqual(t.device.type, device_type)
            if isinstance(t, TwoTensor):
                untyped_storage_a, untyped_storage_b = (
                    t.a.untyped_storage(),
                    t.b.untyped_storage(),
                )
                self.assertEqual(
                    untyped_storage_a.nbytes(),
                    sd_orig[key].a.untyped_storage().nbytes(),
                )
                self.assertEqual(
                    untyped_storage_b.nbytes(),
                    sd_orig[key].b.untyped_storage().nbytes(),
                )
                offset_a, offset_b = (
                    untyped_storage_a._checkpoint_offset,
                    untyped_storage_b._checkpoint_offset,
                )
                nbytes_a, nbytes_b = (
                    untyped_storage_a.nbytes() // 4,
                    untyped_storage_b.nbytes() // 4,
                )
                result_a = torch.frombuffer(
                    all_bytes, dtype=dtype, count=nbytes_a, offset=offset_a
                ).resize_(t.a.size())
                result_b = torch.frombuffer(
                    all_bytes, dtype=dtype, count=nbytes_b, offset=offset_b
                ).resize_(t.b.size())
                self.assertEqual(TwoTensor(result_a, result_b), sd_orig[key])
            else:
                untyped_storage = t.untyped_storage()
                self.assertEqual(
                    untyped_storage.nbytes(), sd_orig[key].untyped_storage().nbytes()
                )
                offset = untyped_storage._checkpoint_offset
                nbytes = untyped_storage.nbytes() // 4
                result = torch.frombuffer(
                    all_bytes, dtype=dtype, count=nbytes, offset=offset
                ).resize_(t.size())
                self.assertEqual(result, sd_orig[key])

        with TemporaryFileName() as f, torch.serialization.safe_globals([TwoTensor]):
            # Create state_dict to be loaded later
            torch.save(sd, f)
            with open(f, "rb") as g:
                all_bytes = g.read()

            fake_mode = FakeTensorMode()
            with fake_mode:
                sd_loaded = torch.load(f)
            for k in sd:
                _read_tensor_and_check(k, sd_loaded, sd, all_bytes, "cpu")
            with fake_mode:
                sd_loaded = torch.load(f, map_location=device)
            for k in sd:
                _read_tensor_and_check(k, sd_loaded, sd, all_bytes, self.device_type)

        for k in sd:
            sd[k] = sd[k].to(device)

        with TemporaryFileName() as f, torch.serialization.safe_globals([TwoTensor]):
            torch.save(sd, f)
            with open(f, "rb") as g:
                all_bytes = g.read()

            fake_mode = FakeTensorMode()
            with fake_mode:
                sd_loaded = torch.load(f)
            for k in sd:
                _read_tensor_and_check(k, sd_loaded, sd, all_bytes, self.device_type)
            with fake_mode:
                sd_loaded = torch.load(f, map_location="cpu")
            for k in sd:
                _read_tensor_and_check(k, sd_loaded, sd, all_bytes, "cpu")


make_propagate_real_tensors_cls(TestFakeTensorPropDevice)
instantiate_device_type_tests(
    TestFakeTensorPropDevice, globals(), except_for=("cpu",), allow_xpu=True
)
instantiate_device_type_tests(
    PropagateRealTensorsTestFakeTensorPropDevice,  # noqa: F821
    globals(),
    except_for=("cpu",),
    allow_xpu=True,
)


class TestFakeTensorSerialization(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_serialization(self):
        x = torch.tensor([0], device="cpu")
        with FakeTensorMode():
            y = pickle.loads(pickle.dumps(x))
            self.assertTrue(is_fake_tensor(y))
            self.assertEqual(y.device.type, "meta")

            with unset_fake_temporarily():
                y = pickle.loads(pickle.dumps(x))
                self.assertEqual(x.device, y.device)

    def test_serialization_with_tracing(self):
        x = torch.tensor([0], device="cpu")
        with tracing(TracingContext(FakeTensorMode())):
            y = pickle.loads(pickle.dumps(x))
            self.assertEqual(x.device, y.device)


class _TestFakeTensorDispatchCacheHelpers:
    """Cache assertions shared with TestFakeTensorDispatchCacheDevice."""

    def _test_cache_key(self, fm, x, y, z):
        """
        Helper for all test_cache_key_* tests below. Assert that the
        cache keys for inputs x and y are the same, but z is different.
        """
        func = aten.add.Tensor
        state = _CacheKeyState()
        key_x = fm._cache_key(state, func, [x], {})
        key_y = fm._cache_key(state, func, [y], {})
        key_z = fm._cache_key(state, func, [z], {})

        self.assertEqual(key_x, key_y)
        self.assertNotEqual(key_x, key_z)

    def assertHitsMisses(self, hits, misses):
        """
        Helper to assert on the number of recorded hits and misses.
        """
        info = FakeTensorMode.cache_info()
        self.assertEqual(info.hits, hits)
        self.assertEqual(info.misses, misses)

    def assertBypasses(self, reason, count):
        """
        Helper to assert on the number of recorded bypasses.
        """
        info = FakeTensorMode.cache_info()
        if count > 0:
            self.assertIn(reason, info.bypasses)
            self.assertEqual(info.bypasses[reason], count)
        else:
            self.assertNotIn(reason, info.bypasses)


# The dispatch cache is a Python FakeTensorMode feature with no C++ equivalent.
@skipIfCppFakeTensor("no Python dispatch cache support")
class TestFakeTensorDispatchCache(_TestFakeTensorDispatchCacheHelpers, TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_shape_env_settings(self):
        """
        Validation that any boolean settings in ShapeEnv are present in the
        ShapeEnvSettings. We hope to ensure that any new settings that might
        affect FakeTensor dispatch are included in the cache key calculation.
        If this test fails, consider updating ShapeEnvSettings or change this
        test to omit checking for the new field.
        """
        init_sig = inspect.signature(ShapeEnv._init)
        args = [
            name
            for name, param in init_sig.parameters.items()
            if type(param.default) is bool
        ]

        settings = [f.name for f in dataclasses.fields(ShapeEnvSettings)]
        for arg in args:
            self.assertTrue(arg in settings)

    def test_cache_key_dtype(self):
        with FakeTensorMode() as fm:
            x = torch.randn(4, 3, dtype=torch.float16)
            y = torch.randn(4, 3, dtype=torch.float16)
            z = x.to(dtype=torch.float32)
            self._test_cache_key(fm, x, y, z)

    def test_cache_key_shape(self):
        with FakeTensorMode() as fm:
            x = torch.randn(4, 3)
            y = torch.randn(4, 3)
            z = torch.randn(4, 2)
            self._test_cache_key(fm, x, y, z)

    def test_cache_key_stride(self):
        with FakeTensorMode() as fm:
            x = torch.randn(4, 2)
            y = torch.randn(4, 2)
            z = x.as_strided((4, 2), (1, 2))
            self._test_cache_key(fm, x, y, z)

    def test_cache_key_memory_format(self):
        with FakeTensorMode() as fm:
            x = torch.randn(1, 2, 3, 4)
            y = torch.randn(1, 2, 3, 4)
            z = x.to(memory_format=torch.channels_last)
            self._test_cache_key(fm, x, y, z)

    def test_cache_key_storage_offset(self):
        with FakeTensorMode() as fm:
            x = torch.randn(3)[1:]
            y = torch.randn(3)[1:]
            z = torch.randn(2)
            self._test_cache_key(fm, x, y, z)

    def test_cache_key_requires_grad(self):
        with FakeTensorMode() as fm:
            x = torch.randn(4, 3)
            y = torch.randn(4, 3)
            z = torch.randn(4, 3, requires_grad=True)
            self._test_cache_key(fm, x, y, z)

    def test_cache_key_is_conj(self):
        with FakeTensorMode() as fm:
            x = torch.randn(4, 3, dtype=torch.complex64)
            y = torch.randn(4, 3, dtype=torch.complex64)
            z = torch.randn(4, 3, dtype=torch.complex64)
            torch._C._set_conj(z, not z.is_conj())
            self._test_cache_key(fm, x, y, z)

    def test_cache_key_is_neg(self):
        with FakeTensorMode() as fm:
            x = torch.randn(4, 3, dtype=torch.complex64)
            y = torch.randn(4, 3, dtype=torch.complex64)
            z = torch.randn(4, 3, dtype=torch.complex64)
            torch._C._set_neg(z, not z.is_neg())
            self._test_cache_key(fm, x, y, z)

    def test_cache_key_is_inference(self):
        with torch.inference_mode(True):
            t = torch.randn(4, 3)
        with FakeTensorMode() as fm:
            x = torch.randn(4, 3)
            y = torch.randn(4, 3)
            z = fm.from_tensor(t)
            self._test_cache_key(fm, x, y, z)

    def test_cache_key_constants(self):
        with FakeTensorMode() as fm:
            # Python hashes 1.0 to the same value as 1. Make sure the
            # cache key calculation differentiates them.
            self._test_cache_key(fm, 1.0, 1.0, 1)
            self._test_cache_key(fm, 0.0, 0.0, 0)

    def test_empty_list(self):
        with FakeTensorMode() as fm:
            func = aten.any.dims
            state = _CacheKeyState()
            x = torch.ones((2, 3))
            key_x = fm._cache_key(state, func, [x, []], {})
            key_y = fm._cache_key(state, func, [x], {})

        self.assertNotEqual(key_x, key_y)

    def test_cache_hit(self):
        """
        Test that cache hit/miss counters are updated correctly.
        """
        with FakeTensorMode():
            x = torch.randn(4, 3)
            y = torch.randn(4, 3)

            FakeTensorMode.cache_clear()
            self.assertHitsMisses(0, 0)
            res1 = x + y
            self.assertHitsMisses(0, 1)
            res2 = x + y
            self.assertHitsMisses(1, 1)

            self.assertEqual(
                extract_tensor_metadata(res1),
                extract_tensor_metadata(res2),
            )

    def _symbolic_cache_input(self, *, dtype=torch.float32, requires_grad=False):
        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)
        x = fake_mode.from_tensor(
            torch.randn(4, 8, dtype=dtype, requires_grad=requires_grad),
            source=LocalSource("x", is_input=True),
            symbolic_context=StatelessSymbolicContext(
                dynamic_sizes=[DimDynamic.DYNAMIC, DimDynamic.STATIC],
                constraint_sizes=[None, None],
            ),
        )
        return fake_mode, x

    def _assert_symbolic_cache_hit(self, before_hit, actual, expected):
        after_hit = FakeTensorMode.cache_info()
        self.assertEqual(after_hit.hits, before_hit.hits + 1)
        self.assertEqual(after_hit.misses, before_hit.misses)
        self.assertEqual(
            extract_tensor_metadata(actual),
            extract_tensor_metadata(expected),
        )

    def test_cache_symbolic_contiguous_output_avoids_empty_strided(self):
        fake_mode, x = self._symbolic_cache_input()
        fake_mode.cache_crosscheck_enabled = False

        with fake_mode:
            FakeTensorMode.cache_clear()
            expected = aten.clone.default(x)
            before_hit = FakeTensorMode.cache_info()

            with patch.object(
                torch, "empty_strided", wraps=torch.empty_strided
            ) as empty_strided:
                actual = aten.clone.default(x)

            empty_strided.assert_not_called()
            self._assert_symbolic_cache_hit(before_hit, actual, expected)

    def test_cache_contiguous_output_preserves_stride_expressions(self):
        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)
        fake_mode.cache_crosscheck_enabled = False

        with fake_mode:
            u0 = shape_env.create_unbacked_symint()
            u1 = shape_env.create_unbacked_symint()
            torch._check(u0 >= 1)
            torch._check(u1 >= 1)
            x = torch.empty_strided((u0, u1), (u1, 1))
            FakeTensorMode.cache_clear()
            expected = aten.stack.default([x, x], 0)
            before_hit = FakeTensorMode.cache_info()

            with patch.object(
                torch, "empty_strided", wraps=torch.empty_strided
            ) as empty_strided:
                actual = aten.stack.default([x, x], 0)

            empty_strided.assert_called_once()
            self.assertEqual(
                tuple(
                    s.node.expr if isinstance(s, torch.SymInt) else s
                    for s in actual.stride()
                ),
                tuple(
                    s.node.expr if isinstance(s, torch.SymInt) else s
                    for s in expected.stride()
                ),
            )
            self._assert_symbolic_cache_hit(before_hit, actual, expected)

    def test_cache_symbolic_noncontiguous_output_preserves_strides(self):
        fake_mode, x = self._symbolic_cache_input()
        fake_mode.cache_crosscheck_enabled = False

        with fake_mode:
            x = aten.transpose.int(x, 0, 1)
            FakeTensorMode.cache_clear()
            expected = aten.clone.default(x, memory_format=torch.preserve_format)
            before_hit = FakeTensorMode.cache_info()

            with patch.object(
                torch, "empty_strided", wraps=torch.empty_strided
            ) as empty_strided:
                actual = aten.clone.default(x, memory_format=torch.preserve_format)

            empty_strided.assert_called_once()
            self._assert_symbolic_cache_hit(before_hit, actual, expected)

    def test_cache_symbolic_view_output_preserves_strides_and_storage_offset(self):
        fake_mode, x = self._symbolic_cache_input()
        fake_mode.cache_crosscheck_enabled = False

        with fake_mode:
            x = aten.transpose.int(x, 0, 1)
            FakeTensorMode.cache_clear()
            expected = aten.slice.Tensor(x, 0, 1, 8)
            before_hit = FakeTensorMode.cache_info()

            with patch.object(
                torch, "as_strided", wraps=torch.as_strided
            ) as as_strided:
                actual = aten.slice.Tensor(x, 0, 1, 8)

            as_strided.assert_called_once()
            self.assertNotEqual(expected.storage_offset(), 0)
            self.assertFalse(expected.is_contiguous())
            self.assertEqual(
                actual.untyped_storage()._cdata,
                x.untyped_storage()._cdata,
            )
            self._assert_symbolic_cache_hit(before_hit, actual, expected)

    def test_cache_view_output_preserves_resized_storage(self):
        fake_mode = FakeTensorMode()

        with fake_mode:
            x = torch.randn(4, 8)
            torch.ops.inductor.resize_storage_bytes_(x, 0)
            FakeTensorMode.cache_clear()
            expected = aten.slice.Tensor(x, 0, 1, 3)
            before_hit = FakeTensorMode.cache_info()
            actual = aten.slice.Tensor(x, 0, 1, 3)

            self.assertEqual(x.untyped_storage().nbytes(), 0)
            self.assertEqual(expected.untyped_storage().nbytes(), 0)
            self.assertEqual(actual.untyped_storage().nbytes(), 0)
            self._assert_symbolic_cache_hit(before_hit, actual, expected)

    @parametrize(
        "func,set_flag,flag_name,dtype",
        (
            (aten._conj.default, torch._C._set_conj, "is_conj", torch.complex64),
            (aten._neg_view.default, torch._C._set_neg, "is_neg", torch.float32),
        ),
    )
    def test_cache_symbolic_view_output_restores_view_bits(
        self, func, set_flag, flag_name, dtype
    ):
        fake_mode, x = self._symbolic_cache_input(dtype=dtype)

        with fake_mode:
            set_flag(x, True)
            FakeTensorMode.cache_clear()
            expected = func(x)
            before_hit = FakeTensorMode.cache_info()
            actual = func(x)

            self.assertEqual(
                getattr(actual, flag_name)(),
                getattr(expected, flag_name)(),
            )
            self._assert_symbolic_cache_hit(before_hit, actual, expected)

    def test_cache_symbolic_contiguous_output_preserves_stride_backrefs(self):
        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)
        fake_mode.cache_crosscheck_enabled = False
        x = fake_mode.from_tensor(
            torch.randn(4, 8),
            source=LocalSource("x", is_input=True),
            symbolic_context=StatelessSymbolicContext(
                dynamic_sizes=[DimDynamic.DYNAMIC, DimDynamic.DYNAMIC],
                dynamic_strides=[DimDynamic.DYNAMIC, DimDynamic.DYNAMIC],
            ),
        )

        with fake_mode:
            FakeTensorMode.cache_clear()
            expected = aten.clone.default(x, memory_format=torch.preserve_format)
            before_hit = FakeTensorMode.cache_info()
            with patch.object(
                torch, "empty_strided", wraps=torch.empty_strided
            ) as empty_strided:
                actual = aten.clone.default(x, memory_format=torch.preserve_format)

            empty_strided.assert_called_once()
            self._assert_symbolic_cache_hit(before_hit, actual, expected)

            FakeTensorMode.cache_clear()
            aten.sin.default(expected)
            before_downstream_hit = FakeTensorMode.cache_info()
            aten.sin.default(actual)
            after_downstream_hit = FakeTensorMode.cache_info()
            self.assertEqual(
                after_downstream_hit.hits,
                before_downstream_hit.hits + 1,
            )
            self.assertEqual(
                after_downstream_hit.misses,
                before_downstream_hit.misses,
            )

    def test_cache_symbolic_contiguous_output_preserves_storage_bytes(self):
        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)
        x = fake_mode.from_tensor(
            torch.randn(4, 8),
            source=LocalSource("x", is_input=True),
            symbolic_context=StatelessSymbolicContext(
                dynamic_sizes=[DimDynamic.DYNAMIC, DimDynamic.DYNAMIC],
                constraint_sizes=[None, None],
            ),
        )

        with fake_mode:
            FakeTensorMode.cache_clear()
            expected = aten.mul.Tensor(x, 2)
            before_hit = FakeTensorMode.cache_info()
            actual = aten.mul.Tensor(x, 2)
            self._assert_symbolic_cache_hit(before_hit, actual, expected)

            expected_storage_bytes = expected.untyped_storage().nbytes()
            actual_storage_bytes = actual.untyped_storage().nbytes()
            self.assertIsInstance(expected_storage_bytes, torch.SymInt)
            self.assertIsInstance(actual_storage_bytes, torch.SymInt)
            self.assertEqual(
                actual_storage_bytes.node.expr,
                expected_storage_bytes.node.expr,
            )

            FakeTensorMode.cache_clear()
            aten.sin.default(expected)
            before_downstream_hit = FakeTensorMode.cache_info()
            aten.sin.default(actual)
            after_downstream_hit = FakeTensorMode.cache_info()
            self.assertEqual(
                after_downstream_hit.hits,
                before_downstream_hit.hits + 1,
            )
            self.assertEqual(
                after_downstream_hit.misses,
                before_downstream_hit.misses,
            )

    def test_cache_symbolic_dtype_changing_view_preserves_dtype(self):
        fake_mode, x = self._symbolic_cache_input(dtype=torch.complex64)
        fake_mode.cache_crosscheck_enabled = False

        with fake_mode:
            x = aten.slice.Tensor(x, 0, 1, 4)
            FakeTensorMode.cache_clear()
            expected = aten.view_as_real.default(x)
            before_hit = FakeTensorMode.cache_info()

            with patch.object(
                torch, "as_strided", wraps=torch.as_strided
            ) as as_strided:
                actual = aten.view_as_real.default(x)

            as_strided.assert_not_called()
            self.assertEqual(actual.dtype, torch.float32)
            self.assertNotEqual(expected.storage_offset(), 0)
            self.assertEqual(
                actual.untyped_storage()._cdata,
                x.untyped_storage()._cdata,
            )
            self._assert_symbolic_cache_hit(before_hit, actual, expected)

    def test_cache_symbolic_detach_output_clears_requires_grad(self):
        fake_mode, x = self._symbolic_cache_input(requires_grad=True)

        with fake_mode:
            FakeTensorMode.cache_clear()
            expected = aten.detach.default(x)
            before_hit = FakeTensorMode.cache_info()
            actual = aten.detach.default(x)

            self.assertFalse(actual.requires_grad)
            self._assert_symbolic_cache_hit(before_hit, actual, expected)

    def test_cache_bypass_custom_op_without_cache_key(self):
        """
        Test that a custom op, which can't be cached, is bypassed without
        building a cache key for it.
        """

        @torch.library.custom_op("_torch_testing::fake_cache_bypass", mutates_args=())
        def f(x: torch.Tensor) -> torch.Tensor:
            return x.clone()

        outputs = []
        f.register_fake(lambda x: outputs[0])

        with FakeTensorMode():
            x = torch.randn(3)
            outputs.append(torch.empty(3))
            FakeTensorMode.cache_clear()
            built_key = AssertionError("built a cache key")
            with patch.object(FakeTensorMode, "_cache_key", side_effect=built_key):
                f(x)
                f(x)
            self.assertBypasses("non-builtin", 2)
            self.assertEqual(FakeTensorMode.cache_info().size, 0)

    def test_cache_bypass_prims_as_strided(self):
        x = torch.empty(0, 8)
        y = torch.empty(0, 8)
        mode = FakeTensorMode(shape_env=ShapeEnv())
        x = mode.from_tensor(x, static_shapes=True)
        y = mode.from_tensor(y, static_shapes=True)

        with mode:
            FakeTensorMode.cache_clear()
            prims.as_strided(x, (0, 8), (8, 1), 0)
            prims.as_strided(y, (0, 8), (8, 1), 0)

            self.assertBypasses("prims.as_strided", 2)

    def test_cache_bypass(self):
        """
        Test that cache bypass counters are updated correctly.
        """
        with FakeTensorMode():
            x = torch.randn(1, 2)

            FakeTensorMode.cache_clear()
            self.assertBypasses("inplace view", 0)

            x.unsqueeze_(0)
            self.assertBypasses("inplace view", 1)

    def test_cache_default_dtype(self):
        """
        Test that the default dtype is respected when serving cached results.
        """
        with FakeTensorMode():
            x = torch.tensor([1, 2], dtype=torch.int32)
            torch.set_default_dtype(torch.float32)

            FakeTensorMode.cache_clear()
            self.assertHitsMisses(0, 0)

            y = x + 1.0
            self.assertEqual(y.dtype, torch.float32)
            self.assertHitsMisses(0, 1)

            torch.set_default_dtype(torch.float16)
            y = x + 1.0
            self.assertEqual(y.dtype, torch.float16)
            self.assertHitsMisses(0, 2)

            torch.set_default_dtype(torch.float32)
            y = x + 1.0
            self.assertEqual(y.dtype, torch.float32)
            self.assertHitsMisses(1, 2)

    def test_cache_inplace_op(self):
        """
        Test that inplace ops served from the cache correctly reference the
        input parameter.
        """
        with FakeTensorMode():
            x = torch.randn(1, 2)
            y = torch.randn(1, 2)

            FakeTensorMode.cache_clear()
            self.assertHitsMisses(0, 0)

            z = x.add_(y)
            self.assertHitsMisses(0, 1)
            self.assertEqual(id(x), id(z))

            w = x.add_(y)
            self.assertHitsMisses(1, 1)
            self.assertEqual(id(x), id(w))

    def test_cache_view_op(self):
        """
        Test that view ops are handled correctly when served from the cache.
        """
        with FakeTensorMode():
            x1 = torch.ones(2, requires_grad=True).clone()
            x2 = torch.ones(2, requires_grad=True).clone()
            y2 = x2.view(-1)

            # Test operating on a non-view tensor, then the same operation
            # on a view tensor. Assert that the view property is set correctly.
            z1 = x1.mul_(2)
            self.assertFalse(z1._is_view())

            z2 = y2.mul_(2)
            self.assertTrue(z2._is_view())

            # Now the other way around: first operate on a view tensor, then
            # the same operation on a non-view tensor.
            z2 = y2.mul_(2)
            self.assertTrue(z2._is_view())

            z1 = x1.mul_(2)
            self.assertFalse(z1._is_view())

    def test_cache_unsafe_view_aliasing(self):
        """
        _unsafe_view reports is_view=False, but its output still shares storage
        with its input. A cache hit must reproduce that aliasing against the new
        input rather than hand back a freshly allocated storage.

        Storage identity is what this checks. Neither extract_tensor_metadata
        nor the crosscheck's assert_metadata_eq compares storage identity, and
        _is_view() is False for a correct and an incorrect output alike, so
        nothing else here would catch a regression.
        """
        with FakeTensorMode():
            x = torch.randn(4, 4)
            y = torch.randn(4, 4)

            FakeTensorMode.cache_clear()
            ref = aten._unsafe_view.default(x, [16])
            self.assertEqual(ref.untyped_storage()._cdata, x.untyped_storage()._cdata)

            # Same shapes and dtypes, so this call is served from the cache. The
            # hit count is compared relatively: the fake implementation
            # re-dispatches internally, so the absolute counts are not 1.
            hits = FakeTensorMode.cache_info().hits
            res = aten._unsafe_view.default(y, [16])
            self.assertEqual(FakeTensorMode.cache_info().hits, hits + 1)

            self.assertEqual(res.untyped_storage()._cdata, y.untyped_storage()._cdata)
            self.assertNotEqual(
                res.untyped_storage()._cdata, x.untyped_storage()._cdata
            )
            self.assertEqual(
                extract_tensor_metadata(ref),
                extract_tensor_metadata(res),
            )

    def test_cache_dispatch_key_set(self):
        """
        Test that operations that change the dispatch key set bypass caching.
        """
        with FakeTensorMode():
            FakeTensorMode.cache_clear()
            self.assertBypasses("dispatch_key_set mismatch", 0)

            x = torch._efficientzerotensor(3)
            self.assertTrue(x._is_zerotensor())
            self.assertBypasses("dispatch_key_set mismatch", 1)

            y = torch._efficientzerotensor(3)
            self.assertTrue(y._is_zerotensor())
            self.assertBypasses("dispatch_key_set mismatch", 2)

    def test_fft_hfft2_issue145522(self):
        with FakeTensorMode():
            s0 = 5
            s1 = 6
            s2 = 7
            s3 = 3
            s4 = 10
            s5 = 2
            x = torch.randn(s0, s1, s2)
            out = torch.randn(s0, s3, s4)
            kwargs = {
                "s": (s3, s4),
                "dim": (1, s5),
                "norm": "ortho",
            }
            r = torch._C._fft.fft_hfft2(x, **kwargs, out=out)
            self.assertEqual(r.shape, out.shape)

    def test_inference_mode(self):
        """
        Test that caching handles inference mode correctly.
        """
        with FakeTensorMode():
            x = torch.randn(4, 3)
            y = torch.randn(4, 3)

            FakeTensorMode.cache_clear()
            self.assertHitsMisses(0, 0)

            # Expect a miss when the inference mode is different
            res1 = x + y
            with torch.inference_mode():
                res2 = x + y

            self.assertHitsMisses(0, 2)
            self.assertFalse(res1.is_inference())
            self.assertTrue(res2.is_inference())

            # Second tries should see hits
            res3 = x + y

            self.assertHitsMisses(1, 2)
            self.assertFalse(res3.is_inference())
            self.assertEqual(
                extract_tensor_metadata(res1),
                extract_tensor_metadata(res3),
            )

            with torch.inference_mode():
                res4 = x + y

            self.assertHitsMisses(2, 2)
            self.assertTrue(res4.is_inference())
            self.assertEqual(
                extract_tensor_metadata(res2),
                extract_tensor_metadata(res4),
            )

    def test__upsample_bilinear2d_aa_backward_dynamic_shapes(self):
        def f(x):
            return torch.nn.functional.interpolate(
                x,
                size=[256, 256],
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )

        shape_env = ShapeEnv()
        fake_m = FakeTensorMode(shape_env=shape_env)
        x = fake_m.from_tensor(
            torch.randn(1, 3, 2005, 1920, requires_grad=True),
            symbolic_context=StatelessSymbolicContext(
                dynamic_sizes=[
                    DimDynamic.STATIC,
                    DimDynamic.STATIC,
                    DimDynamic.DYNAMIC,
                    DimDynamic.DYNAMIC,
                ],
                constraint_sizes=[None, None, None, None],
            ),
        )
        with fake_m, enable_python_dispatcher():
            out = f(x)
            out.sum().backward()
            self.assertEqual(x.shape, x.grad.shape)

    @fake_tensor_modes
    def test_from_buffer(self, mode_cls):
        with mode_cls() as mode:
            obj = [1, 2]
            f = io.BytesIO()
            pickle.Pickler(f).dump(obj)
            storage = torch.UntypedStorage.from_buffer(f.getvalue(), dtype=torch.uint8)
            self.assertEqual(storage.device, torch.device("meta"))

            t = torch.ByteTensor(storage)
            self.assertTrue(is_fake_tensor(t))
            self.assertEqual(t.device, torch.device("cpu"))
            self.assertIs(maybe_get_fake_mode(t), mode)

    def test_meta_tensor_to_fake_cpu(self):
        x = torch.randn(4, 4, device="meta")
        with FakeTensorMode(allow_non_fake_inputs=True):
            x_cpu = x.to(device="cpu")
        self.assertTrue(is_fake_tensor(x_cpu))
        self.assertEqual(x_cpu.device, torch.device("cpu"))

    def test_cache_tuple_outputs(self):
        """
        Test to check that ops with tuple outputs work.
        """
        with FakeTensorMode():
            x = torch.randn(6, 4)
            y = torch.randn(6, 4)

            FakeTensorMode.cache_clear()
            self.assertHitsMisses(0, 0)

            ref = torch.split(x, 2)
            self.assertHitsMisses(0, 1)

            res = torch.split(y, 2)
            self.assertHitsMisses(1, 1)
            self.assertEqual(len(ref), len(res))
            for a, b in zip(ref, res):
                self.assertEqual(
                    extract_tensor_metadata(a),
                    extract_tensor_metadata(b),
                )

    def test_cache_aten_index(self):
        with FakeTensorMode():
            x = torch.randn(4, 4, 4)
            idx_tensor1 = torch.tensor([0, 2, 3])
            idx_tensor2 = torch.tensor([0, 1, 2])

            FakeTensorMode.cache_clear()
            self.assertHitsMisses(0, 0)

            ref = torch.ops.aten.index(x, [None, idx_tensor1, idx_tensor2])
            self.assertHitsMisses(0, 3)

            res = torch.ops.aten.index(x, [None, idx_tensor1, idx_tensor2])
            self.assertHitsMisses(1, 3)
            self.assertEqual(extract_tensor_metadata(ref), extract_tensor_metadata(res))

        with FakeTensorMode():
            x = torch.randn(4, 4, 4)
            idx_tensor1 = torch.tensor([True, True, False, True])
            self.assertRaises(
                DynamicOutputShapeException,
                lambda: torch.ops.aten.index(x, [None, idx_tensor1]),
            )

            # uint8 is a (deprecated) mask, like bool
            idx_tensor1 = torch.tensor([1, 1, 0, 1], dtype=torch.uint8)
            self.assertRaises(
                DynamicOutputShapeException,
                lambda: torch.ops.aten.index(x, [None, idx_tensor1]),
            )

            # int8 is not a valid index dtype; eager raises too
            idx_tensor1 = torch.tensor([1, -2, 3, -4], dtype=torch.int8)
            self.assertRaises(
                IndexError,
                lambda: torch.ops.aten.index(x, [None, idx_tensor1]),
            )

        # With a ShapeEnv, indexing with a mask succeeds with an unbacked size,
        # and is not cached because its output shape depends on the mask's data.
        with FakeTensorMode(shape_env=ShapeEnv()):
            x = torch.randn(4, 4, 4)
            for dtype in (torch.bool, torch.uint8):
                mask = torch.tensor([1, 1, 0, 1], dtype=dtype)
                FakeTensorMode.cache_clear()
                torch.ops.aten.index(x, [None, mask])
                # Both index and the nonzero it calls on the mask
                self.assertBypasses("dynamic output shape", 2)
                self.assertBypasses("unrepresented symbol in output", 0)

    def test_cache_output_synthesis_ignores_proxy_tracing(self):
        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True, shape_env=shape_env)

        # Create a cache entry whose output metadata contains an untracked
        # symbolic size.  Rebuilding that FakeTensor inside make_fx should not
        # try to record cache-internal SymInt checks in the proxy graph.
        with fake_mode, shape_env.ignore_fresh_unbacked_symbols():
            u0 = shape_env.create_unbacked_symint()
            x = torch.empty(u0)
            output = x.clone()

        state = _CacheKeyState(shape_env)
        args = (x,)
        key = fake_mode._cache_key(state, aten.clone.default, args, {})
        entry = fake_mode._make_cache_entry(
            state, key, aten.clone.default, args, {}, output
        )

        def f(dummy):
            fake_mode._output_from_cache_entry(state, entry, key, args)
            return dummy + 1

        gm = make_fx(f)(torch.randn(1))
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, dummy_1):
    add = torch.ops.aten.add.Tensor(dummy_1, 1);  dummy_1 = None
    return add""",
        )

    def test_cache_crosscheck_ignores_proxy_tracing(self):
        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True, shape_env=shape_env)
        with fake_mode, shape_env.ignore_fresh_unbacked_symbols():
            u0 = shape_env.create_unbacked_symint()
            x = torch.empty(u0)

        FakeTensorMode.cache_clear()
        dispatch_types = (FakeTensor,)
        dispatch_args = (x,)
        cache_hits = FakeTensorMode.cache_hits

        def f(dummy):
            fake_mode.dispatch(aten.sum.default, dispatch_types, dispatch_args, {})
            fake_mode.dispatch(aten.sum.default, dispatch_types, dispatch_args, {})
            return dummy

        gm = make_fx(f)(torch.randn(1))
        self.assertEqual(FakeTensorMode.cache_hits, cache_hits + 1)
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, dummy_1):
    _tensor_constant0 = self._tensor_constant0
    sum_1 = torch.ops.aten.sum.dim_IntList(_tensor_constant0, []);  _tensor_constant0 = sum_1 = None
    return dummy_1""",
        )

    @skipIfWindows(
        msg="weird bug - cache may not be cleared after https://github.com/pytorch/pytorch/pull/154283"
    )
    @skipIfTorchDynamo("cache hit/miss changes with invoke_subgraph caching")
    def test_invoke_subgraph(self):
        """
        Tests invoke subgraph
        """
        invoke_subgraph = torch._higher_order_ops.invoke_subgraph

        def run():
            def fn(x, y):
                return (x + y * 2,)

            # Ensure there is no caching for non-Fx graph module inputs
            with FakeTensorMode():
                x = torch.randn(6, 4)
                y = torch.randn(6, 4)

                FakeTensorMode.cache_clear()
                self.assertHitsMisses(0, 0)

                ref = invoke_subgraph(fn, "subgraph", x, y)
                self.assertHitsMisses(0, 2)
                self.assertBypasses("function argument", 1)

                res = invoke_subgraph(fn, "subgraph", x, y)
                # The hits are from the ops inside fn
                self.assertHitsMisses(2, 2)
                self.assertBypasses("function argument", 2)

                res = invoke_subgraph(fn, "subgraph", x, y)
                # The hits are from the ops inside fn
                self.assertHitsMisses(4, 2)
                self.assertBypasses("function argument", 3)

            # Get the mod as if its going through torch.compile
            backend = torch._dynamo.testing.AotEagerAndRecordGraphs()
            x = torch.randn(6, 4)
            y = torch.randn(6, 4)
            torch.compile(fn, backend=backend, fullgraph=True)(x, y)
            self.assertEqual(len(backend.fw_graphs), 1)
            mod = backend.fw_graphs[0]

            # Ensure that we see hits every time
            with FakeTensorMode():
                x = torch.randn(6, 4)
                y = torch.randn(6, 4)

                FakeTensorMode.cache_clear()
                self.assertHitsMisses(0, 0)

                ref = invoke_subgraph(mod, "subgraph", x, y)
                self.assertHitsMisses(0, 3)

                res = invoke_subgraph(mod, "subgraph", x, y)
                # The hits are from re-running the subgraph
                self.assertHitsMisses(1, 3)

                res = invoke_subgraph(mod, "subgraph", x, y)
                # The hits are from re-running the subgraph
                self.assertHitsMisses(2, 3)

                self.assertEqual(len(ref), len(res))
                for a, b in zip(ref, res):
                    self.assertEqual(
                        extract_tensor_metadata(a),
                        extract_tensor_metadata(b),
                    )
            self.assertTrue(count_invoke_subgraph_keys() > 0)

        def count_invoke_subgraph_keys():
            invoke_subgraph_keys = 0
            for cache_key in FakeTensorMode.cache:
                if isinstance(cache_key.key[0], torch._ops.HigherOrderOperator):
                    invoke_subgraph_keys += 1
            return invoke_subgraph_keys

        # Check that the graph gc clears the cache
        run()
        torch.compiler.reset()
        gc.collect()
        gc.collect()  # second collection needed for 3.14t
        self.assertTrue(count_invoke_subgraph_keys() == 0)

    @skipIfTorchDynamo("cache hit/miss changes with invoke_subgraph caching")
    def test_invoke_subgraph_cacheable_inplace(self):
        invoke_subgraph = torch._higher_order_ops.invoke_subgraph

        def fn(x, y):
            # aten ops are used so that eager backend graph is suitable for fake
            # tensor testing
            cos = torch.ops.aten.cos.default(x)
            # inplace-view - this should cause the whole invoke_subgraph to not
            # being able to cache
            t = torch.ops.aten.t_.default(cos)
            mul = torch.ops.aten.mul.Tensor(t, y)
            return (mul,)

        # Get the mod as if its going through torch.compile
        backend = torch._dynamo.testing.AotEagerAndRecordGraphs()
        x = torch.randn(4, 4)
        y = torch.randn(4, 4)
        torch.compile(fn, backend=backend, fullgraph=True)(x, y)
        self.assertEqual(len(backend.graphs), 1)
        mod = backend.graphs[0]

        # Ensure that invoke_subgraph result is still cached
        with FakeTensorMode():
            x = torch.randn(4, 4)
            y = torch.randn(4, 4)

            FakeTensorMode.cache_clear()
            self.assertHitsMisses(0, 0)

            ref = invoke_subgraph(mod, "subgraph", x, y)
            self.assertHitsMisses(0, 3)

            res = invoke_subgraph(mod, "subgraph", x, y)
            # The hits are from the ops inside fn and not the subgraph
            self.assertHitsMisses(1, 3)

            res = invoke_subgraph(mod, "subgraph", x, y)
            # The hits are from the ops inside fn and not the subgraph
            self.assertHitsMisses(2, 3)

            self.assertEqual(len(ref), len(res))
            for a, b in zip(ref, res):
                self.assertEqual(
                    extract_tensor_metadata(a),
                    extract_tensor_metadata(b),
                )

    @skipIfTorchDynamo("cache hit/miss changes with invoke_subgraph caching")
    def test_unbacked_output(self):
        # The point of this test is to have an op which has no symbols as input
        # but a symbol as an output and make sure that we skip caching it.
        class LengthsGather(torch.nn.Module):
            def forward(
                self,
                input: torch.Tensor,
                lengths: torch.Tensor,
                indices: torch.Tensor,
                offsets: torch.Tensor,
            ) -> torch.Tensor:
                bias = torch.gather(offsets, 0, indices)
                lengths_selected = torch.gather(lengths, 0, indices)
                index = torch.repeat_interleave(bias, lengths_selected, dim=0)
                return index

        input = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        lengths = torch.tensor([0, 2, 3, 1, 4])
        indices = torch.tensor([2, 3, 4, 6, 7, 8, 9])
        offsets = torch.cumsum(lengths, 0)
        ep = torch.export.export(
            LengthsGather(), (input, lengths, indices, offsets), strict=False
        )

        FakeTensorMode.cache_clear()
        ep.run_decompositions({})
        self.assertBypasses("unrepresented symbol in output", 2)


instantiate_parametrized_tests(TestFakeTensorDispatchCache)


@skipIfCppFakeTensor("no Python dispatch cache support")
class TestFakeTensorDispatchCacheDevice(_TestFakeTensorDispatchCacheHelpers, TestCase):
    r"""Accelerator-only TestFakeTensorDispatchCache cases (see except_for below)."""

    hw_classification = HardwareClassification.ACCELERATOR

    def test_cache_key_device(self, device):
        with FakeTensorMode() as fm:
            x = torch.randn(4, 3)
            y = torch.randn(4, 3)
            z = x.to(device=device)
            self._test_cache_key(fm, x, y, z)

    def test_cache_key_indexless_device_pins_current_index(self, device):
        # An index-less device argument means "the current one", so it denotes a
        # different device depending on ambient state. Keying it unresolved lets a
        # call made while device 0 is current serve one made while device 1 is, and
        # the cached output then carries the wrong device. Patching current_device
        # is enough to see it: the resolution happens in the key, so this needs one
        # device.
        func = aten.zeros.default
        state = _CacheKeyState()
        device_module = getattr(torch, self.device_type)
        with FakeTensorMode() as fm:
            args = [[4, 3]]
            kwargs = {"device": torch.device(self.device_type)}
            with patch.object(device_module, "current_device", return_value=0):
                key_dev0 = fm._cache_key(state, func, args, kwargs)
            with patch.object(device_module, "current_device", return_value=1):
                key_dev1 = fm._cache_key(state, func, args, kwargs)
            # An explicit index is already unambiguous and must be unaffected.
            explicit = {"device": torch.device(self.device_type, 0)}
            with patch.object(device_module, "current_device", return_value=1):
                key_explicit = fm._cache_key(state, func, args, explicit)

        self.assertNotEqual(key_dev0, key_dev1)
        self.assertEqual(key_dev0, key_explicit)

    def test_cache_default_device(self, device):
        """
        Test that the default device is respected when serving cached results.
        """
        with FakeTensorMode():
            FakeTensorMode.cache_clear()
            self.assertHitsMisses(0, 0)

            try:
                torch.set_default_device("cpu")
                x = torch.tensor([1, 2])
                y = x + 1.0
                self.assertEqual(y.device.type, "cpu")
                self.assertHitsMisses(0, 1)

                torch.set_default_device(device)
                x = torch.tensor([1, 2])
                y = x + 1.0
                self.assertEqual(y.device.type, self.device_type)
                self.assertHitsMisses(0, 2)

                torch.set_default_device("cpu")
                x = torch.tensor([1, 2])
                y = x + 1.0
                self.assertEqual(y.device.type, "cpu")
                self.assertHitsMisses(1, 2)
            finally:
                torch.set_default_device(None)

    def test_wrapper_tensor_subclass_different_device(self, device):
        class DifferentDeviceTensor(torch.Tensor):
            @staticmethod
            def __new__(cls, a):
                kwargs = {}
                kwargs["strides"] = a.stride()
                kwargs["storage_offset"] = a.storage_offset()
                kwargs["device"] = torch.device("cpu")
                kwargs["layout"] = a.layout
                kwargs["requires_grad"] = a.requires_grad
                kwargs["dtype"] = a.dtype
                out = torch.Tensor._make_wrapper_subclass(cls, a.size(), **kwargs)
                return out

            def __init__(self, a):
                self.inner_tensor = a

            def __repr__(self):
                return f"DifferentDeviceTensor({repr(self.inner_tensor)})"

            def __tensor_flatten__(self):
                return ["inner_tensor"], None

            @staticmethod
            def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
                if meta is not None:
                    raise AssertionError(f"expected meta is None, got {meta}")
                return DifferentDeviceTensor(inner_tensors["inner_tensor"])

            @classmethod
            def __torch_dispatch__(cls, func, types, args, kwargs):
                if kwargs is None:
                    kwargs = {}
                args = pytree.tree_map_only(
                    DifferentDeviceTensor, lambda x: x.inner_tensor, args
                )
                kwargs = pytree.tree_map_only(
                    DifferentDeviceTensor, lambda x: x.inner_tensor, kwargs
                )
                # Returns unwrapped tensor
                return func(*args, **kwargs)

        a = torch.ones(2, 2, 768, device=device)
        wrapped_a = DifferentDeviceTensor(a)

        # Outer Tensor is on cpu, inner is on the accelerator
        self.assertTrue(wrapped_a.is_cpu)
        self.assertFalse(wrapped_a.inner_tensor.is_cpu)

        with FakeTensorMode() as fake_mode:
            fake_wrapped_a = fake_mode.from_tensor(wrapped_a)

        self.assertTrue(fake_wrapped_a.is_cpu)
        if not isinstance(fake_wrapped_a, DifferentDeviceTensor):
            raise AssertionError(
                f"expected DifferentDeviceTensor, got {type(fake_wrapped_a)}"
            )
        self.assertFalse(fake_wrapped_a.inner_tensor.is_cpu)


instantiate_device_type_tests(
    TestFakeTensorDispatchCacheDevice,
    globals(),
    except_for=("cpu",),
    allow_xpu=True,
)


@instantiate_parametrized_tests
class TestFakeTensorPreferDeviceType(TestCase):
    hw_classification = HardwareClassification.GENERIC

    @fake_tensor_modes
    def test_fake_tensor_prefer_device_type_cpu_only(self, mode_cls):
        """
        Test that fake_tensor_prefer_device_type works correctly when only CPU tensors are involved.
        """
        # The preferred type only has to differ from the tensors' device; nothing
        # is ever dispatched to it, so this needs no accelerator.
        with torch._functorch.config.patch(fake_tensor_prefer_device_type="cuda"):
            with mode_cls():
                # When all tensors are CPU, the result should still be CPU
                x = torch.randn(2, 3, device="cpu")
                y = torch.randn(2, 3, device="cpu")
                result = x + y
                self.assertEqual(result.device.type, "cpu")
                self.assertTrue(is_fake_tensor(result))

    @torch._functorch.config.patch(fake_tensor_prefer_device_type=None)
    @fake_tensor_modes
    @parametrize("reverse", [False, True])
    @parametrize(
        "devices,preference,expected_device",
        [
            (("cuda:0", "cuda:1"), "cuda", None),
            (("cuda:0", "cuda:1"), "uda", None),
            (("cpu", "cuda:1"), "cuda", "cuda:1"),
            (("cpu", "cuda:1"), "cpu", "cpu"),
            (("cpu", "cuda:1"), "uda", "cuda:1"),
            (("cpu", "cuda:1"), "cp", "cpu"),
            (("cpu", "cuda:1"), "xpu", None),
            (("cpu", "cuda:1"), "u", None),
            (("cpu", "cuda:1"), "", None),
            (("cpu", "cuda:1"), "cuda:1", None),
        ],
    )
    def test_fake_tensor_prefer_device_type_fallback(
        self, mode_cls, reverse, devices, preference, expected_device
    ):
        # Synthetic accelerator indices must not initialize GPU contexts.
        with (
            patch("torch._subclasses.fake_tensor.init_gpu_context", lambda _: None),
            patch.object(mode_cls, "avoid_device_init", True),
            torch.library._scoped_library("_test_fake_device_conflict", "DEF") as lib,
            torch._functorch.config.patch(fake_tensor_prefer_device_type=preference),
        ):
            mode = mode_cls()
            a, b = (
                mode.fake_tensor_converter.from_meta_and_device(
                    mode, torch.empty(2, 3, device="meta"), torch.device(device)
                )
                for device in devices
            )
            if reverse:
                a, b = b, a

            lib.define("resolve(Tensor a, Tensor b) -> Tensor")
            # A fresh Meta output forces common-device resolution in the fallback.
            lib.impl("resolve", lambda a, b: torch.empty(2, 3, device="meta"), "Meta")
            op = torch.ops._test_fake_device_conflict.resolve.default

            def check(expected):
                if expected is None:
                    msg = re.escape(
                        f"found at least two devices, {a.device} and {b.device}!"
                    )
                    with self.assertRaisesRegex(
                        FakeTensorDeviceMismatchError, msg
                    ) as exc:
                        op(a, b)
                    self.assertIs(exc.exception.func, op)
                    self.assertEqual(exc.exception.common_device, a.device)
                    self.assertEqual(exc.exception.device, b.device)
                else:
                    result = op(a, b)
                    self.assertEqual(result.device, torch.device(expected))
                    self.assertEqual(result.shape, (2, 3))
                    self.assertIs(maybe_get_fake_mode(result), mode)

            with mode:
                check(expected_device)
                # This Meta-output path uses C++'s captured preference; Python
                # mode reads live config. Python op_impls also read live config
                # for C++ modes; see Note [C++ fake fallback].
                with torch._functorch.config.patch(fake_tensor_prefer_device_type=None):
                    check(expected_device if mode_cls is CppFakeTensorMode else None)
                check(expected_device)


class TestFakeTensorPreferDeviceTypeDevice(TestCase):
    r"""Accelerator-only TestFakeTensorPreferDeviceType cases (see except_for below)."""

    hw_classification = HardwareClassification.ACCELERATOR

    @torch._dynamo.config.patch(fake_tensor_cache_enabled=False)
    @fake_tensor_modes
    @parametrize("reverse", [False, True])
    def test_fake_tensor_prefer_device_type(self, device, mode_cls, reverse):
        for preference in (None, self.device_type):
            with (
                torch._functorch.config.patch(
                    fake_tensor_prefer_device_type=preference
                ),
                mode_cls() as mode,
            ):
                gpu_tensor = torch.randn(3, 4, device=device)
                cpu_tensor = torch.arange(3).unsqueeze(-1)
                lhs, rhs = (
                    (cpu_tensor, gpu_tensor) if reverse else (gpu_tensor, cpu_tensor)
                )
                if preference is None:
                    msg = re.escape(
                        f"found at least two devices, {lhs.device} and {rhs.device}!"
                    )
                    with self.assertRaisesRegex(FakeTensorDeviceMismatchError, msg):
                        lhs + rhs
                    continue
                result = lhs + rhs
                self.assertEqual(result.device, gpu_tensor.device)
                self.assertEqual(result.shape, (3, 4))
                self.assertIs(maybe_get_fake_mode(result), mode)
                for tensor in (gpu_tensor, cpu_tensor):
                    self.assertEqual((tensor + tensor).device, tensor.device)


instantiate_device_type_tests(
    TestFakeTensorPreferDeviceTypeDevice,
    globals(),
    except_for=("cpu",),
    allow_xpu=True,
)


class TestFakeTensorMetaDevicePropagationDevice(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    @fake_tensor_modes
    @parametrize("meta_rhs", [False, True])
    def test_inplace_add_with_meta_rhs_keeps_destination_device(
        self, device, mode_cls, meta_rhs
    ):
        with mode_cls() as mode:
            lhs_device, rhs_device = (device, "meta") if meta_rhs else ("meta", device)
            log_det = torch.zeros(2, device=lhs_device)
            expected_device = log_det.device
            self.assertIs(log_det.add_(torch.zeros(2, device=rhs_device)), log_det)

            self.assertEqual(log_det.device, expected_device)
            self.assertTrue(is_fake_tensor(log_det))
            self.assertIs(maybe_get_fake_mode(log_det), mode)


instantiate_device_type_tests(
    TestFakeTensorMetaDevicePropagationDevice,
    globals(),
    allow_xpu=True,
)


class TestFakeTensorViewCopy(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_expand_then_view_copy_matches_eager_mode(self):
        x = torch.arange(7)
        y = x.expand(12, 7)

        # Eager baseline
        eager = torch.ops.aten.view_copy.default(y, [84])
        self.assertEqual(eager.shape, (84,))
        self.assertIsNone(eager._base)  # non-aliasing

        # FakeTensor behavior should match eager
        with FakeTensorMode():
            xf = torch.arange(7)
            yf = xf.expand(12, 7)
            fake = torch.ops.aten.view_copy.default(yf, [84])
            self.assertEqual(fake.shape, (84,))
            self.assertIsNone(fake._base)  # non-aliasing

    def test_expand_then_view_still_not_allowed(self):
        with FakeTensorMode():
            xf = torch.arange(7)
            yf = xf.expand(12, 7)
            with self.assertRaisesRegex(
                ValueError, "Cannot view a tensor with shape *"
            ):
                _ = yf.view(-1)

    def test_expand_then_view_copy_unbacked_matches_eager_mode(self):
        with torch.fx.experimental._config.patch(backed_size_oblivious=True):
            with FakeTensorMode():
                xf = torch.arange(7)
                yf = xf.expand(12, 7)
                fake = torch.ops.aten.view_copy.default(yf, [84])
                self.assertEqual(fake.shape, (84,))
                self.assertIsNone(fake._base)  # non-aliasing

    def test_expand_then_view_unbacked_still_not_allowed(self):
        with torch.fx.experimental._config.patch(backed_size_oblivious=True):
            with FakeTensorMode():
                xf = torch.arange(7)
                yf = xf.expand(12, 7)
                with self.assertRaisesRegex(
                    ValueError, "Cannot view a tensor with shape *"
                ):
                    _ = yf.view(-1)


@instantiate_parametrized_tests
class TestCppFakeTensorModeBindings(TestCase):
    def _installed(self):
        return (
            torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.FAKE),
            torch._C._dispatch_tls_is_dispatch_key_included(torch._C.DispatchKey.Fake),
        )

    def test_create(self):
        mode = CppFakeTensorMode()
        self.assertIsInstance(mode, CppFakeTensorMode)
        self.assertTrue(mode.allow_meta)
        mode.allow_meta = False
        self.assertFalse(mode.allow_meta)
        with torch._functorch.config.patch(fake_tensor_allow_meta=False):
            mode = CppFakeTensorMode(shape_env=ShapeEnv())
        self.assertFalse(mode.allow_meta)
        with self.assertRaisesRegex(RuntimeError, "Expected a FakeTensorConverter"):
            torch._C._CppFakeTensorMode(None, None)
        # Only a CppFakeTensorMode stands for its C++ mode
        self.addCleanup(
            torch._C._unset_dispatch_mode, torch._C._TorchDispatchModeKey.FAKE
        )
        bare = torch._C._CppFakeTensorMode.__new__(torch._C._CppFakeTensorMode)
        uninitialized = CppFakeTensorMode.__new__(CppFakeTensorMode)
        with self.assertRaisesRegex(RuntimeError, "not initialized"):
            torch._C._set_dispatch_mode(uninitialized)
        converter = mode.fake_tensor_converter
        with self.assertRaisesRegex(RuntimeError, "already initialized"):
            mode.__init__()
        self.assertIs(mode.fake_tensor_converter, converter)
        with self.assertRaisesRegex(TypeError, "Expected a CppFakeTensorMode"):
            torch._C._from_meta_and_device(
                torch.empty(2, device="meta"), torch.device("cpu"), bare
            )
        with self.assertRaisesRegex(TypeError, "Expected a CppFakeTensorMode"):
            torch._C._set_dispatch_mode(bare)

        # Nor does a subclass, whose __dict__ could hold a fake
        class Subclass(CppFakeTensorMode):  # type: ignore[misc]
            pass

        with self.assertRaisesRegex(TypeError, "Expected a CppFakeTensorMode"):
            torch._C._from_meta_and_device(
                torch.empty(2, device="meta"), torch.device("cpu"), Subclass()
            )
        with self.assertRaisesRegex(TypeError, "Expected a CppFakeTensorMode"):
            torch._C._set_dispatch_mode(Subclass())

    def test_from_meta_and_device(self):
        mode = CppFakeTensorMode()
        cpu = torch.device("cpu")
        # A plain meta tensor gets a new detached fake on device sharing its
        # storage, as in Python, and keeps its own device and autograd state
        meta = torch.empty(2, 3, device="meta", requires_grad=True) * 2
        layouts = (torch.sparse_coo, torch.sparse_csr)
        sparse_metas = [
            torch.empty(2, 2, layout=layout, device="meta") for layout in layouts
        ]
        ops = []

        class RecordingMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                ops.append(func)
                return func(*args, **(kwargs or {}))

        with RecordingMode():
            fake = torch._C._from_meta_and_device(meta, cpu, mode)
            sparse_fakes = [
                torch._C._from_meta_and_device(t, cpu, mode) for t in sparse_metas
            ]
        self.assertEqual(ops, [])
        self.assertIsNot(fake, meta)
        self.assertTrue(torch._C._is_fake_tensor(fake))
        self.assertEqual(fake.device, cpu)
        self.assertEqual(fake.shape, (2, 3))
        self.assertTrue(fake.requires_grad)
        self.assertTrue(fake.is_leaf)
        self.assertEqual(fake.untyped_storage()._cdata, meta.untyped_storage()._cdata)
        self.assertFalse(torch._C._is_fake_tensor(meta))
        self.assertTrue(meta.is_meta)
        self.assertIsNotNone(meta.grad_fn)
        no_grad_meta = torch.empty(2, device="meta")
        no_grad = torch._C._from_meta_and_device(no_grad_meta, cpu, mode)
        self.assertFalse(no_grad.requires_grad)

        def sparse_parts(t):
            if t.layout == torch.sparse_coo:
                return t._indices(), t._values()
            return t.crow_indices(), t.col_indices(), t.values()

        fake_key = torch._C.DispatchKeySet(torch._C.DispatchKey.Fake)
        for sparse_meta, sparse, layout in zip(sparse_metas, sparse_fakes, layouts):
            self.assertTrue(torch._C._is_fake_tensor(sparse))
            self.assertEqual(sparse.layout, layout)
            self.assertEqual(sparse.device, cpu)
            self.assertEqual(sparse.shape, (2, 2))
            self.assertEqual(sparse.dtype, torch.float32)
            # It shares the input's indices and values (read with Fake excluded)
            with torch._C._ExcludeDispatchKeyGuard(fake_key):
                shared = sparse_parts(sparse)
            for x, y in zip(shared, sparse_parts(sparse_meta)):
                self.assertEqual(x.untyped_storage()._cdata, y.untyped_storage()._cdata)

    @skipIfTorchDynamo(_CPP_FAKE_DYNAMO_SKIP)
    def test_from_meta_and_device_rejects(self):
        mode = CppFakeTensorMode()
        meta_device, cpu = torch.device("meta"), torch.device("cpu")
        meta = torch.empty(2, device="meta")
        fake = torch._C._from_meta_and_device(meta, cpu, mode)
        not_fake = (ValueError, "not fake")
        not_meta = (ValueError, "device must be 'meta', got cpu")
        wrapper = (TypeError, "does not accept tensor subclasses")

        def rejects(x, error, msg, mode=mode, device=cpu):
            with self.assertRaisesRegex(error, msg):
                torch._C._from_meta_and_device(x, device, mode)
            return x

        # A fake also reads as meta, on the meta device or inside a kernel
        rejects(torch._C._from_meta_and_device(meta, meta_device, mode), *not_fake)
        with torch._C._InKernelInvocation():
            self.assertTrue(fake.is_meta)
            rejects(fake, *not_fake)
        rejects(fake, *not_meta)
        rejects(torch.empty(2), *not_meta)
        # Wrappers are checked before the device, so a cpu subclass gets the
        # same error
        rejects(TwoTensor(meta, meta.clone()), *wrapper)
        rejects(TwoTensor(torch.empty(2), torch.empty(2)), *wrapper)
        rejects(torch._to_functional_tensor(meta), *wrapper)
        torch.func.grad(lambda x: rejects(x, *wrapper).sum())(meta)
        torch.vmap(lambda x: rejects(x, *wrapper).sum())(meta.expand(3, 2))
        with torch._functorch.config.patch(fake_tensor_allow_meta=False):
            strict_mode = CppFakeTensorMode()
        msg = "allow_meta is False"
        rejects(meta, RuntimeError, msg, mode=strict_mode, device=meta_device)

    def test_from_meta_and_device_failure_leaves_input_unmarked(self):
        # requires_grad is set before the shared storage is marked, so failing
        # to set it leaves the input's storage unmarked
        mode = CppFakeTensorMode()
        mode._allow_unsafe_data_ptr_access = False
        with torch.inference_mode():
            inference = torch.empty(2, device="meta", requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, "inference tensor"):
            torch._C._from_meta_and_device(inference, torch.device("cpu"), mode)
        inference.data_ptr()

    @skipIfTorchDynamo(_CPP_FAKE_DYNAMO_SKIP)
    def test_maybe_get_fake_mode(self):
        mode = CppFakeTensorMode()
        other_mode = CppFakeTensorMode()
        meta = torch.empty(2, device="meta")
        fake = torch._C._from_meta_and_device(meta, torch.device("cpu"), mode)
        other = torch._C._from_meta_and_device(meta, torch.device("cpu"), other_mode)
        self.assertIs(torch._C._maybe_get_fake_mode(fake), mode)
        self.assertIs(torch._C._maybe_get_fake_mode(other), other_mode)
        self.assertIsNone(torch._C._maybe_get_fake_mode(meta))
        self.assertIsNone(torch._C._maybe_get_fake_mode(torch.empty(2)))
        no_mode = torch.empty(2, device="meta")
        torch._C._set_fake_device(no_mode, torch.device("cpu"))
        self.assertTrue(torch._C._is_fake_tensor(no_mode))
        self.assertIsNone(torch._C._maybe_get_fake_mode(no_mode))
        # The functional wrapper copies the fake's ExtraMeta, mode included, but
        # drops the Fake key
        functional = torch._to_functional_tensor(fake)
        self.assertFalse(torch._C._is_fake_tensor(functional))
        self.assertEqual(torch._C._fake_device(functional), torch.device("cpu"))
        self.assertIsNone(torch._C._maybe_get_fake_mode(functional))
        self.assertIsNone(torch._C._maybe_get_fake_mode(TwoTensor(fake, fake)))
        self.assertIsNone(torch._C._maybe_get_fake_mode(_add_batch_dim(fake, 0, 1)))
        python_mode = torch._subclasses.fake_tensor.FakeTensorMode()
        python_fake = python_mode.from_tensor(torch.empty(2))
        self.assertIsNone(torch._C._maybe_get_fake_mode(python_fake))

    def test_unsafe_data_ptr_access(self):
        def make_fake(mode):
            meta = torch.empty(2, device="meta")
            return torch._C._from_meta_and_device(meta, torch.device("cpu"), mode)

        warning = "Accessing the data pointer of FakeTensor"
        data_ptr_error = r"Cannot access data pointer of Tensor \(e\.g\. FakeTensor"
        config = torch._functorch.config
        with config.patch(fake_tensor_allow_unsafe_data_ptr_access=False):
            strict_mode = CppFakeTensorMode()
        self.assertFalse(strict_mode._allow_unsafe_data_ptr_access)
        with self.assertRaisesRegex(RuntimeError, data_ptr_error):
            make_fake(strict_mode).data_ptr()

        mode = CppFakeTensorMode()
        self.assertTrue(mode._allow_unsafe_data_ptr_access)
        fake = make_fake(mode)
        with self.assertWarnsOnceRegex(UserWarning, warning):
            fake.data_ptr()

        # Reassigning applies to fakes created afterwards
        mode._allow_unsafe_data_ptr_access = False
        self.assertFalse(mode._allow_unsafe_data_ptr_access)
        with self.assertRaisesRegex(RuntimeError, data_ptr_error):
            make_fake(mode).data_ptr()
        with self.assertWarnsOnceRegex(UserWarning, warning):
            fake.data_ptr()

    def test_fake_dispatch_mode_slot(self):
        mode = CppFakeTensorMode()
        fake = torch._C._TorchDispatchModeKey.FAKE

        self.assertEqual(self._installed(), (None, False))
        msg = "not a CppFakeTensorMode and has no _mode_key"
        with self.assertRaisesRegex(RuntimeError, msg):
            torch._C._set_dispatch_mode(object())
        # A C++ mode and a Python FakeTensorMode cannot both be installed
        python_mode = torch._subclasses.fake_tensor.FakeTensorMode()
        try:
            with python_mode:
                msg = "Python FakeTensorMode is set"
                with self.assertRaisesRegex(RuntimeError, msg):
                    torch._C._set_dispatch_mode(mode)
            self.assertEqual(self._installed(), (None, False))
            torch._C._set_dispatch_mode(mode)
            self.assertEqual(self._installed(), (mode, True))
            other_mode = CppFakeTensorMode()
            with self.assertRaisesRegex(RuntimeError, "one already exists"):
                torch._C._set_dispatch_mode(other_mode)
            msg = r"C\+\+ FakeTensorMode is set"
            for install in (
                torch._C._set_dispatch_mode,
                torch._C._push_on_torch_dispatch_stack,
            ):
                with self.assertRaisesRegex(RuntimeError, msg):
                    install(python_mode)
            self.assertEqual(self._installed(), (mode, True))
            # Only the FAKE key sees it, and the mode stack does not
            proxy = torch._C._TorchDispatchModeKey.PROXY
            self.assertIsNone(torch._C._get_dispatch_mode(proxy))
            self.assertIsNone(torch._C._unset_dispatch_mode(proxy))
            self.assertEqual(torch._C._len_torch_dispatch_stack(), 0)
            self.assertEqual(self._installed(), (mode, True))
            self.assertIs(torch._C._unset_dispatch_mode(fake), mode)
            self.assertEqual(self._installed(), (None, False))
            self.assertIsNone(torch._C._unset_dispatch_mode(fake))
        finally:
            # A rejection that regressed can leave both a Python and a C++ mode
            # installed, which _unset_dispatch_mode rejects, so the Python one
            # is popped first.
            if torch._C._get_dispatch_mode(fake) is python_mode:
                torch._C._pop_torch_dispatch_stack(fake)
            torch._C._unset_dispatch_mode(fake)
        self.assertEqual(self._installed(), (None, False))

    @skipIfTorchDynamo("Dynamo suspends the fake mode, rebuilding its Python object")
    def test_unset_rebuilds_mode(self):
        mode = CppFakeTensorMode()
        converter = mode.fake_tensor_converter
        fake = torch._C._TorchDispatchModeKey.FAKE
        self.addCleanup(torch._C._unset_dispatch_mode, fake)
        torch._C._set_dispatch_mode(mode)
        ref = weakref.ref(mode)
        del mode
        gc.collect()
        self.assertIsNone(ref())

        with patch.object(CppFakeTensorMode, "__init__", side_effect=AssertionError):
            with unset_fake_temporarily():
                self.assertEqual(self._installed(), (None, False))
            got = torch._C._get_dispatch_mode(fake)
            self.assertIs(got.fake_tensor_converter, converter)
            self.assertEqual(self._installed(), (got, True))
            self.assertIs(torch._C._unset_dispatch_mode(fake), got)
            self.assertEqual(self._installed(), (None, False))

    @skipIfTorchDynamo("Dynamo keeps references that stop the mode being collected")
    @parametrize("replace_with_fake", [False, True])
    def test_set_data_releases_mode_after_metadata_copy(self, replace_with_fake):
        cpu = torch.device("cpu")
        mode = CppFakeTensorMode()
        fake = torch._C._from_meta_and_device(torch.empty(2, device="meta"), cpu, mode)
        real = torch.arange(16, dtype=torch.float64).reshape(4, 4)[1:].t()
        replacement = real
        if replace_with_fake:
            replacement_mode = CppFakeTensorMode()
            replacement = torch._C._from_meta_and_device(
                torch.empty_like(real, device="meta"), cpu, replacement_mode
            )

        def metadata(tensor):
            return (
                tensor.shape,
                tensor.stride(),
                tensor.storage_offset(),
                tensor.dtype,
                tensor.device,
                str(torch._C._dispatch_keys(tensor)),
                tensor.untyped_storage().device,
                torch._C._maybe_get_fake_mode(tensor),
            )

        observed = []
        ref = weakref.ref(
            mode.fake_tensor_converter, lambda _: observed.append(metadata(fake))
        )
        del mode
        self.assertIsNotNone(ref())
        expected = metadata(replacement)
        fake.data = replacement
        self.assertIsNone(ref())
        self.assertEqual(observed, [expected])
        self.assertEqual(metadata(fake), expected)
        if not replace_with_fake:
            self.assertEqual(fake, real)

    def test_from_meta_and_device_sparse_mode_installed(self):
        # The placeholder meta indices and values that the sparse impls' copies
        # allocate do not dispatch to the installed mode. Such dispatch would
        # fail or make meta fakes, which this mode's allow_meta rejects.
        with torch._functorch.config.patch(fake_tensor_allow_meta=False):
            mode = CppFakeTensorMode()
        layouts = (torch.sparse_coo, torch.sparse_csr)
        metas = [torch.empty(2, 2, layout=layout, device="meta") for layout in layouts]
        cpu = torch.device("cpu")
        torch._C._set_dispatch_mode(mode)
        try:
            fakes = [torch._C._from_meta_and_device(t, cpu, mode) for t in metas]
        finally:
            torch._C._unset_dispatch_mode(torch._C._TorchDispatchModeKey.FAKE)
        for fake, layout in zip(fakes, layouts):
            self.assertTrue(torch._C._is_fake_tensor(fake))
            self.assertEqual(fake.layout, layout)
            self.assertEqual(fake.device, cpu)

    def test_suspend_for_real_tensors(self):
        mode = CppFakeTensorMode()
        torch._C._set_dispatch_mode(mode)
        try:
            with unset_fake_temporarily() as old:
                self.assertIs(old, mode)
                self.assertEqual(self._installed(), (None, False))
                self.assertFalse(torch._C._is_fake_tensor(torch.empty(2)))
            self.assertEqual(self._installed(), (mode, True))
            # Python modes on the stack are suspended and restored with it
            with BaseTorchDispatchMode() as py_mode:
                with _disable_current_modes():
                    self.assertEqual(self._installed(), (None, False))
                    self.assertIsNone(_get_current_dispatch_mode())
                    self.assertFalse(torch._C._is_fake_tensor(torch.empty(2)))
                self.assertEqual(self._installed(), (mode, True))
                self.assertIs(_get_current_dispatch_mode(), py_mode)
                # A failed C++ mode restore still restores the Python modes
                other_mode = CppFakeTensorMode()
                with self.assertRaisesRegex(RuntimeError, "one already exists"):
                    with _disable_current_modes():
                        self.assertEqual(self._installed(), (None, False))
                        torch._C._set_dispatch_mode(other_mode)
                # The restore raised, leaving other_mode installed
                self.assertEqual(self._installed(), (other_mode, True))
                self.assertIs(_get_current_dispatch_mode(), py_mode)
        finally:
            torch._C._unset_dispatch_mode(torch._C._TorchDispatchModeKey.FAKE)
        self.assertEqual(self._installed(), (None, False))

    def test_item_memo(self):
        mode = CppFakeTensorMode()
        get_memo = torch._C._get_fake_item_memo

        def set_memo(fake, memo, keep):
            torch._C._set_fake_item_memo(fake, memo, keep_across_epochs=keep)

        def make_fake():
            meta = torch.empty((), device="meta")
            return torch._C._from_meta_and_device(meta, torch.device("cpu"), mode)

        class IntSubclass(int):
            pass

        # Only Python and symbolic numbers are accepted, and returned as is
        fake = make_fake()
        self.assertIsNone(get_memo(fake))
        for bad in (fake, None, 1j, IntSubclass(1)):
            with self.assertRaisesRegex(TypeError, "Expected an int, float, bool"):
                set_memo(fake, bad, False)
        env = ShapeEnv()
        syms = (env.create_unbacked_symint(), env.create_unbacked_symfloat())
        for good in (2, 2.5, False, *syms, env.create_unbacked_symbool()):
            set_memo(fake, good, False)
            self.assertIs(get_memo(fake), good)

        # Reading keeps the memo; it is dropped once read in another epoch, and
        # stays dropped back in the first, unless kept across epochs
        set_memo(fake, 2, False)
        self.assertEqual(get_memo(fake), 2)
        self.assertEqual(get_memo(fake), 2)
        mode.epoch += 1
        self.assertIsNone(get_memo(fake))
        mode.epoch -= 1
        self.assertIsNone(get_memo(fake))
        set_memo(fake, 3, True)
        for _ in range(2):
            mode.epoch += 1
            self.assertEqual(get_memo(fake), 3)
        # Dropped once the tensor is mutated, whether or not kept across epochs
        for keep in (False, True):
            set_memo(fake, 4, keep)
            torch.autograd.graph.increment_version(fake)
            self.assertIsNone(get_memo(fake))

        # An inference fake has no version counter, but epochs still apply
        with torch.inference_mode():
            inference = make_fake()
        set_memo(inference, 2, False)
        self.assertEqual(get_memo(inference), 2)
        mode.epoch += 1
        self.assertIsNone(get_memo(inference))
        set_memo(inference, 3, True)
        for _ in range(2):
            mode.epoch += 1
            self.assertEqual(get_memo(inference), 3)

        # set_data drops the destination's memo and does not copy the source's
        dst, src = make_fake(), make_fake()
        set_memo(dst, 2, True)
        set_memo(src, 3, True)
        self.assertEqual((get_memo(dst), get_memo(src)), (2, 3))
        dst.data = src
        self.assertIsNone(get_memo(dst))
        # An inference fake's version counter is disabled; once set_data gives
        # it a non-inference key set, reading the version throws, so setting
        # fails and leaves no memo
        with torch.inference_mode():
            disabled = make_fake()
        disabled.data = make_fake()
        with self.assertRaisesRegex(RuntimeError, "version counter"):
            set_memo(disabled, 2, False)
        self.assertIsNone(get_memo(disabled))

        # A fake without a mode stores no memo (the getter would return None
        # anyway, so check the memo is not kept alive); a real tensor has none
        # and is rejected by the setters
        no_mode = torch.empty((), device="meta")
        torch._C._set_fake_device(no_mode, torch.device("cpu"))
        unstored = env.create_unbacked_symfloat()
        ref = weakref.ref(unstored)
        set_memo(no_mode, unstored, True)
        del unstored
        gc.collect()
        self.assertIsNone(ref())
        self.assertIsNone(get_memo(no_mode))
        real = torch.empty(())
        with self.assertRaisesRegex(RuntimeError, "Expected a fake tensor"):
            set_memo(real, 2, False)
        with self.assertRaisesRegex(RuntimeError, "Expected a fake tensor"):
            torch._C._set_fake_sym_memo(real, "nonzero_memo", 2)
        self.assertIsNone(get_memo(real))
        self.assertIsNone(torch._C._get_fake_sym_memo(real, "nonzero_memo"))

    def test_memo_helpers(self):
        # The helpers reach a C++ fake's memos, and, like SymNumberMemoDescriptor,
        # keep only a backed SymFloat item memo across epochs
        from torch._subclasses.fake_tensor import (
            maybe_get_item_memo,
            maybe_get_sym_memo,
            maybe_set_item_memo,
            maybe_set_sym_memo,
        )
        from torch.fx.experimental.sym_node import _NO_HINT, SymNode

        shape_env = ShapeEnv()
        mode = CppFakeTensorMode(shape_env=shape_env)
        meta = torch.empty((), device="meta")
        fake = torch._C._from_meta_and_device(meta, torch.device("cpu"), mode)
        source = LocalSource("i")
        sym = shape_env.create_symbol(2, source, DimDynamic.DYNAMIC)
        backed_int = shape_env.create_symintnode(sym, hint=2, source=source)
        source = LocalSource("x")
        sym = shape_env.create_unspecified_symbol(2.5, source, DimDynamic.DYNAMIC)
        backed_float = shape_env.create_symfloatnode(sym, hint=2.5, source=source)
        unbacked_float = shape_env.create_unbacked_symfloat()
        hintless_float = torch.SymFloat(SymNode(sym, shape_env, float, hint=_NO_HINT))
        self.assertIsNone(hintless_float.node.hint)

        # The memo is stored with the mode's current epoch
        mode.epoch += 1
        maybe_set_sym_memo(fake, "nonzero_memo", backed_int)
        self.assertIs(maybe_get_sym_memo(fake, "nonzero_memo").node, backed_int.node)
        mode.epoch += 1
        self.assertIsNone(maybe_get_sym_memo(fake, "nonzero_memo"))
        for memo in (backed_int, unbacked_float, hintless_float, backed_float):
            maybe_set_item_memo(fake, memo)
            self.assertIs(maybe_get_item_memo(fake), memo)
            if memo is hintless_float:
                self.assertEqual(guarding_hint_or_throw(memo), 2.5)
                self.assertIsNone(memo.node.hint)
            mode.epoch += 1
            kept = memo is backed_float
            self.assertIs(maybe_get_item_memo(fake), memo if kept else None)
        mode.epoch += 1
        self.assertIs(maybe_get_item_memo(fake), backed_float)
        with self.assertRaises(GuardOnDataDependentSymNode):
            guarding_hint_or_throw(unbacked_float.node)
        self.assertIsNone(unbacked_float.node.hint)

    @skipIfTorchDynamo(_CPP_FAKE_DYNAMO_SKIP)
    def test_set_get_fake_dispatch_keys(self):
        mode = CppFakeTensorMode()
        meta = torch.empty(2, device="meta")
        meta_fake = torch._C._from_meta_and_device(meta, torch.device("meta"), mode)
        self.assertIsNone(maybe_get_fake_dispatch_keys(meta_fake))
        keys = torch._C._dispatch_keys(torch.empty(2))
        torch._C._set_fake_dispatch_keys(meta_fake, keys)
        self.assertEqual(maybe_get_fake_dispatch_keys(meta_fake), keys)
        # Excluding Fake runs the kernels themselves. detach keeps them, like a
        # Python FakeTensor's. alias's kernel does not: ops such as _conj call
        # it internally, and Python gives their outputs no keys.
        fake_key = torch._C.DispatchKeySet(torch._C.DispatchKey.Fake)
        with torch._C._ExcludeDispatchKeyGuard(fake_key):
            detached = meta_fake.detach()
            aliased = torch.ops.aten.alias(meta_fake)
        self.assertEqual(maybe_get_fake_dispatch_keys(detached), keys)
        self.assertTrue(torch._C._is_fake_tensor(aliased))
        self.assertIsNone(torch._C._get_fake_dispatch_keys(aliased))
        self.assertEqual(maybe_get_fake_dispatch_keys(meta_fake), keys)
        # A functional wrapper copies the fake's ExtraMeta, keys included, but
        # is not itself a fake
        functional = torch._to_functional_tensor(meta_fake)
        self.assertIsNone(maybe_get_fake_dispatch_keys(functional))
        for real in (torch.empty(2), functional):
            with self.assertRaisesRegex(RuntimeError, "Expected a fake tensor"):
                torch._C._set_fake_dispatch_keys(real, keys)
            with self.assertRaisesRegex(RuntimeError, "Expected a fake tensor"):
                torch._C._get_fake_dispatch_keys(real)
        torch._C._set_fake_dispatch_keys(meta_fake, None)
        self.assertIsNone(maybe_get_fake_dispatch_keys(meta_fake))
        # set_data replaces the whole ExtraMeta, so the destination takes the
        # source's keys, or none
        cpu = torch.device("cpu")
        dst = torch._C._from_meta_and_device(torch.empty(2, device="meta"), cpu, mode)
        src = torch._C._from_meta_and_device(torch.empty(2, device="meta"), cpu, mode)
        torch._C._set_fake_dispatch_keys(dst, keys)
        dst.data = src
        self.assertIsNone(maybe_get_fake_dispatch_keys(dst))
        torch._C._set_fake_dispatch_keys(src, keys)
        dst.data = src
        self.assertEqual(maybe_get_fake_dispatch_keys(dst), keys)
        # Guards serialize the recorded keys, or the fake's own without them,
        # and a loaded guard carries them, as tested for Python fakes in
        # test_guard_serialization.py
        from torch._dynamo.guards import GuardsStatePickler
        from torch._dynamo.package import load_guards_state

        torch._C._set_fake_dispatch_keys(src, None)
        for fake, expected in ((dst, keys), (src, torch._C._dispatch_keys(src))):
            buf = io.BytesIO()
            pickler = GuardsStatePickler({id(fake): fake}, {}, {}, {}, buf)
            template, _, _, keys_raw, _ = pickler.reducer_override(fake)[1]
            self.assertIs(type(template), torch.Tensor)
            self.assertFalse(torch._C._is_fake_tensor(template))
            self.assertEqual(keys_raw, expected.raw_repr())
            pickler.dump({"t": fake})
            # Neither the C++ mode nor its Python wrapper is pickled
            self.assertNotIn(b"FakeTensorMode", buf.getvalue())
            out = load_guards_state(buf.getvalue())["t"]
            self.assertEqual(maybe_get_fake_dispatch_keys(out), expected)

    def test_cpp_fake_tensor_mode_init(self):
        shape_env = ShapeEnv()
        with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
            mode = CppFakeTensorMode(shape_env=shape_env)
        self.assertTrue(mode.fake_tensor_converter.meta_converter.copy_data)
        self.assertIs(mode.shape_env, shape_env)
        self.assertIn("test_cpp_fake_tensor_mode_init", mode.stack)
        defaults = CppFakeTensorMode()
        self.assertFalse(defaults.fake_tensor_converter.meta_converter.copy_data)
        self.assertIsNone(defaults.shape_env)
        with self.assertRaisesRegex(AttributeError, "unknown_attribute"):
            mode.unknown_attribute = True
        self.assertEqual(mode.epoch, 0)
        mode.epoch += 1
        self.assertEqual(mode.epoch, 1)
        with self.assertRaises(TypeError):
            mode.epoch = -1
        with self.assertRaises(AttributeError):
            del mode.epoch
        self.assertEqual(mode.epoch, 1)
        meta = torch.empty((), device="meta")
        fake = torch._C._from_meta_and_device(meta, torch.device("cpu"), mode)
        self.assertIs(torch._C._maybe_get_fake_mode(fake), mode)
        for op in (torch.ones, torch.hann_window):
            with (
                mode,
                self.assertRaisesRegex(
                    NotImplementedError, "does not support propagate_real_tensors"
                ),
            ):
                op(4)

    @fake_tensor_modes
    def test_cpp_fake_tensor_mode_converter_kind(self, mode_cls):
        mode = mode_cls()
        converter = FakeTensorConverter(cpp_fakes=mode_cls is not CppFakeTensorMode)
        with self.assertRaisesRegex(AssertionError, "cpp_fakes does not match"):
            converter.from_real_tensor(mode, torch.randn(2))
        meta = torch.empty(2, device="meta")
        with self.assertRaisesRegex(AssertionError, "cpp_fakes does not match"):
            converter.from_meta_and_device(mode, meta, torch.device("cpu"))

    def test_in_kernel_invocation_with_native_guard(self):
        with torch._C._InKernelInvocation():
            with in_kernel_invocation_manager(CppFakeTensorMode()):
                self.assertTrue(torch._C._meta_in_tls_dispatch_include())
            self.assertTrue(torch._C._in_kernel_invocation())
            self.assertFalse(torch._C._meta_in_tls_dispatch_include())

    def test_fake_fallback_static_shape_decomps(self):
        from torch._decomp import decomposition_table, is_torch_decomp_decomposition
        from torch._library.utils import lookup_op
        from torch._meta_registrations import cpp_meta_supports_symint

        contains = torch._C._fake_dispatch_contains_static_shape_decomp
        self.assertTrue(contains("aten::log_sigmoid_forward", ""))
        self.assertFalse(contains("aten::arange", "start"))

        mismatched = set()
        for name in torch._C._dispatch_get_all_op_names():
            if not name.startswith("aten::"):
                continue
            op = lookup_op(name)
            schema = op._schema
            decomposes = (
                op in decomposition_table
                and is_torch_decomp_decomposition(op)
                and not cpp_meta_supports_symint(op)
            )
            if contains(schema.name, schema.overload_name) != decomposes:
                mismatched.add(name)
        self.assertEqual(mismatched, set())

    @skipIfTorchDynamo(_CPP_FAKE_DYNAMO_SKIP)
    def test_fake_fallback_decomposition_on_launch_thread(self):
        import functools
        import threading

        from torch._decomp import decomposition_table
        from torch.utils._python_dispatch import _cpp_fake_mode_tls

        caller = threading.get_ident()
        mode = CppFakeTensorMode()
        fake = mode.from_tensor(torch.randn(2, 3))
        seen = []
        hardswish = aten.hardswish.default
        decomp = decomposition_table[hardswish]

        def record(name, t):
            seen.append(
                (
                    name,
                    threading.get_ident() != caller,
                    t.is_meta,
                    torch._guards.active_fake_mode(),
                )
            )

        @functools.wraps(decomp)
        def spy(t):
            record("decomp", t)
            return decomp(t)

        def python_cia(t):
            record("python_cia", t)
            return t.sin()

        def native_cia(t):
            record("native_cia", t)
            return t.sin()

        with torch.library._scoped_library("_test_fake_cia_launch", "DEF") as lib:
            lib.define("python_cia(Tensor x) -> Tensor")
            lib.define("native_cia(Tensor x) -> Tensor")
            torch.ops._test_fake_cia_launch.python_cia.default.py_impl(
                torch._C.DispatchKey.CompositeImplicitAutograd
            )(python_cia)
            lib.impl("native_cia", native_cia, "CompositeImplicitAutograd")

            def work(t):
                ops = torch.ops._test_fake_cia_launch
                return ops.native_cia(ops.python_cia(torch.nn.functional.hardswish(t)))

            @torch.jit._script.script
            def forked(t):
                return torch.jit._wait(torch.jit._fork(work, t))

            with (
                patch.dict(decomposition_table, {hardswish: spy}),
                torch.inference_mode(),
            ):
                for active in (False, True):
                    with mode if active else contextlib.nullcontext():
                        before = self._installed()
                        stack = list(_cpp_fake_mode_tls.stack)
                        result = forked(fake)
                        self.assertIs(maybe_get_fake_mode(result), mode)
                        self.assertEqual(self._installed(), before)
                        self.assertEqual(_cpp_fake_mode_tls.stack, stack)
        names = ["decomp", "python_cia", "native_cia"]
        self.assertEqual(seen, [(n, True, False, mode) for n in names] * 2)

    def test_fake_fallback_static_shape_decomp_registration(self):
        from torch._decomp import (
            decomposition_table,
            register_decomposition,
            remove_decompositions,
        )
        from torch._decomp.decompositions import hardswish

        contains = torch._C._fake_dispatch_contains_static_shape_decomp
        with torch.library._scoped_library("_test_static_shape_decomp", "DEF") as lib:
            lib.define("op(Tensor x) -> Tensor")
            op = torch.ops._test_static_shape_decomp.op.default
            self.addCleanup(remove_decompositions, decomposition_table, [op])

            self.assertFalse(contains(op.name(), ""))
            register_decomposition(op)(hardswish)
            self.assertTrue(contains(op.name(), ""))
            remove_decompositions(decomposition_table, [op])
            self.assertFalse(contains(op.name(), ""))

    @skipIfTorchDynamo(_CPP_FAKE_DYNAMO_SKIP)
    @parametrize(
        "op,preserve_keys",
        [
            (aten.alias.default, True),
            (torch.detach, True),
            subtest((lambda x: x.data, True), name="data"),
            (torch.t, False),
        ],
    )
    def test_fake_fallback_dispatch_keys(self, op, preserve_keys):
        mode = CppFakeTensorMode()
        real = torch.randn(2, 3)
        fake = mode.from_tensor(real)
        keys = torch._C._dispatch_keys(real)
        torch._C._set_fake_dispatch_keys(fake, keys)
        self.assertEqual(
            maybe_get_fake_dispatch_keys(op(fake)), keys if preserve_keys else None
        )

    @skipIfTorchDynamo(_CPP_FAKE_DYNAMO_SKIP)
    @parametrize("active", [False, True])
    @parametrize("transform", ["grad", "vjp", "jvp"])
    def test_fake_data_functorch(self, active, transform):
        mode = CppFakeTensorMode()
        captured = mode.from_tensor(torch.ones(2, requires_grad=True))
        arg = mode.from_tensor(torch.ones(2))
        copies = []

        def fn(x):
            copies.append(captured.data)
            return x.sum()

        with mode if active else contextlib.nullcontext():
            if transform == "grad":
                torch.func.grad(fn)(arg)
            elif transform == "vjp":
                torch.func.vjp(fn, arg)
            else:
                torch.func.jvp(fn, (arg,), (arg,))
        self.assertEqual(len(copies), 1)
        result = copies[0]
        self.assertFalse(torch._C._functorch.is_gradtrackingtensor(result))
        self.assertTrue(is_fake_tensor(result))
        self.assertIs(maybe_get_fake_mode(result), mode)
        self.assertFalse(result.requires_grad)
        version = captured._version
        result.add_(1)
        self.assertEqual(captured._version, version)
        self.assertEqual(result._version, 1)

    @skipIfTorchDynamo(_CPP_FAKE_DYNAMO_SKIP)
    def test_fake_data_trace(self):
        mode = CppFakeTensorMode()
        captured = mode.from_tensor(torch.ones(2, requires_grad=True))
        copies = []

        def fn(x):
            copies.append(captured.data)
            return x + 1

        traced = torch.jit._trace.trace(fn, torch.ones(2), check_trace=False)
        self.assertNotIn("aten::detach", str(traced.graph))
        self.assertEqual(len(copies), 1)
        self.assertFalse(copies[0].requires_grad)
        self.assertIs(maybe_get_fake_mode(copies[0]), mode)

    @skipIfTorchDynamo(_CPP_FAKE_DYNAMO_SKIP)
    @parametrize("op", ["data", "saved_output"])
    def test_fake_data_profiler(self, op):
        mode = CppFakeTensorMode()
        fake = mode.from_tensor(torch.ones(2, requires_grad=True))
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as prof:
            result = fake.data if op == "data" else fake.sqrt()
        self.assertNotIn("aten::detach", [event.key for event in prof.key_averages()])
        if op == "saved_output":
            copies = []
            result.grad_fn._raw_saved_result.register_hooks(
                lambda t: copies.append(t) or t, lambda t: t
            )
            saved = copies[0]
            self.assertFalse(saved.requires_grad)
            version = result._version
            saved.add_(1)
            self.assertEqual(result._version, version + 1)
            self.assertEqual(saved._version, result._version)

    @skipIfTorchDynamo(_CPP_FAKE_DYNAMO_SKIP)
    @fake_tensor_modes
    @parametrize("exclude_python", [False, True])
    def test_fake_data_python_hook(self, mode_cls, exclude_python):
        mode = mode_cls()
        fake = mode.from_tensor(torch.ones(2, requires_grad=True))
        replacement = mode.from_tensor(torch.ones(3))
        calls = []

        class DataMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                calls.append(func)
                return replacement

        with (
            DataMode(),
            torch._C._ExcludeDispatchKeyGuard(
                torch._C.DispatchKeySet(torch._C.DispatchKey.Python)
            )
            if exclude_python
            else contextlib.nullcontext(),
        ):
            result = fake.data
        self.assertEqual(calls, [] if exclude_python else [aten.detach.default])
        if exclude_python:
            self.assertEqual(result.shape, fake.shape)
            self.assertTrue(torch._C._is_alias_of(result, fake))
        else:
            self.assertIs(result, replacement)

    @skipIfTorchDynamo(_CPP_FAKE_DYNAMO_SKIP)
    @fake_tensor_modes
    @parametrize("active", [False, True])
    @parametrize("device,source_device", [("cpu", "cuda:0"), ("cuda:0", "cpu")])
    def test_fake_shallow_copy_data(self, mode_cls, active, device, source_device):
        mode = mode_cls()
        meta_dst = torch.empty(2, device="meta", requires_grad=True)
        meta_src = torch.empty(3, 4, device="meta")
        if mode_cls is CppFakeTensorMode:
            dst = torch._C._from_meta_and_device(meta_dst, torch.device(device), mode)
            src = torch._C._from_meta_and_device(
                meta_src, torch.device(source_device), mode
            )
        else:
            dst = FakeTensor(mode, meta_dst, torch.device(device))
            src = FakeTensor(mode, meta_src, torch.device(source_device))
        version = dst._version
        # Autograd's CUDA gradient accumulator needs a CUDA-enabled build.
        with torch.no_grad(), mode if active else contextlib.nullcontext():
            result = aten.shallow_copy_data_.default(dst, src)
        self.assertIs(result, dst)
        self.assertIs(maybe_get_fake_mode(dst), mode)
        self.assertEqual(dst.device, src.device)
        self.assertEqual(dst.shape, src.shape)
        self.assertEqual(dst.stride(), src.stride())
        self.assertTrue(torch._C._is_alias_of(dst, src))
        self.assertEqual(dst._version, version)
        self.assertTrue(dst.requires_grad)
        self.assertFalse(src.requires_grad)
        self.assertFalse(torch._C._in_kernel_invocation())

    @skipIfTorchDynamo(_CPP_FAKE_DYNAMO_SKIP)
    @fake_tensor_modes
    @parametrize(
        "kind", ["tensor", "parameter", "parameter_subclass", "tensor_subclass"]
    )
    def test_fake_fallback_subclasses_without_python_dispatch(self, mode_cls, kind):
        class ParameterSubclass(torch.nn.Parameter):
            pass

        class TensorSubclass(torch.Tensor):
            pass

        tensor_cls = {
            "tensor": torch.Tensor,
            "parameter": torch.nn.Parameter,
            "parameter_subclass": ParameterSubclass,
            "tensor_subclass": TensorSubclass,
        }[kind]
        real = torch.empty(2, 3, device="cpu")
        if tensor_cls is not torch.Tensor:
            real = torch.Tensor._make_subclass(tensor_cls, real)
        self.assertIs(type(real).__torch_dispatch__, torch.Tensor.__torch_dispatch__)
        self.assertFalse(torch._C._dispatch_keys(real).has(torch._C.DispatchKey.Python))
        mode = mode_cls(allow_non_fake_inputs=True)
        fake = mode.from_tensor(torch.empty(2, 3, device="cpu"))
        with mode, torch._C.DisableTorchFunctionSubclass():
            if kind == "tensor_subclass":
                if mode_cls is CppFakeTensorMode:
                    error, msg = (
                        RuntimeError,
                        "does not support converting tensor subclass TensorSubclass",
                    )
                else:
                    error, msg = TypeError, "Multiple dispatch failed"
                with self.assertRaisesRegex(error, msg):
                    torch.add(real, fake)
            else:
                result = torch.add(real, fake)
                self.assertTrue(is_fake_tensor(result))
                self.assertIs(maybe_get_fake_mode(result), mode)
                self.assertEqual(result.shape, (2, 3))
                self.assertEqual(result.device, torch.device("cpu"))

    @skipIfTorchDynamo(_CPP_FAKE_DYNAMO_SKIP)
    @torch._dynamo.config.patch(use_cpp_fake_tensor=False)
    @fake_tensor_modes
    @parametrize("active", [False, True])
    @parametrize("device", [None, "cuda:0"])
    def test_fake_fallback_retained_outputs(self, mode_cls, active, device):
        mode = mode_cls()
        fake = mode.from_tensor(torch.randn(2, 3))
        retained = torch.empty(2, 3, device="meta")
        self.assertFalse(is_fake_tensor(retained))

        def meta_impl(x, *, device=None):
            return retained, [retained, retained], [None, retained]

        with torch.library._scoped_library("_test_fake_fallback", "DEF") as lib:
            lib.define(
                "retained(Tensor x, *, Device? device=None)"
                " -> (Tensor, Tensor[], Tensor?[])"
            )
            lib.impl("retained", meta_impl, "Meta")
            previous = None
            with mode if active else contextlib.nullcontext():
                # In-place finalization does not preserve raw Meta tensors for
                # converter memo reuse across calls with a device override.
                for _ in range(2 if device is None else 1):
                    result, repeated, optional = torch.ops._test_fake_fallback.retained(
                        fake, device=device
                    )
                    for output in (result, *repeated, optional[1]):
                        self.assertIs(output, result)
                        self.assertIs(maybe_get_fake_mode(output), mode)
                        self.assertEqual(output.device, torch.device(device or "cpu"))
                        self.assertEqual(output.shape, retained.shape)
                    self.assertIsNone(optional[0])
                    self.assertTrue(torch._C._is_alias_of(result, retained))
                    if previous is not None:
                        self.assertIs(result, previous)
                    previous = result
                    if mode_cls is CppFakeTensorMode:
                        self.assertIs(result, retained)
                        self.assertTrue(is_fake_tensor(retained))
                    else:
                        self.assertIsNot(result, retained)
                        self.assertFalse(is_fake_tensor(retained))
                        self.assertEqual(retained.device, torch.device("meta"))

    @skipIfTorchDynamo(_CPP_FAKE_DYNAMO_SKIP)
    @torch._dynamo.config.patch(use_cpp_fake_tensor=False)
    @parametrize("active", [False, True])
    @parametrize("output_device", ["cpu", "meta"])
    @parametrize("device", [None, "cuda:0"])
    @parametrize("implementation", ["meta", "custom"])
    @fake_tensor_modes
    def test_fake_fallback_existing_fake_output(
        self, active, output_device, device, implementation, mode_cls
    ):
        mode = mode_cls()
        fake = mode.from_tensor(torch.empty(2))
        retained = mode.from_tensor(torch.empty(2, device=output_device))
        with torch.library._scoped_library("_test_fake_fallback", "DEF") as lib:
            lib.define("retained(Tensor x, *, Device? device=None) -> Tensor")
            op = torch.ops._test_fake_fallback.retained.default
            if implementation == "custom":
                torch.library.register_fake(
                    op, lambda x, device=None: retained, lib=lib
                )
            else:
                lib.impl("retained", lambda x, device=None: retained, "Meta")
            with mode if active else contextlib.nullcontext():
                if implementation == "custom" or output_device == "cpu":
                    self.assertIs(op(fake, device=device), retained)
                else:
                    msg = "FakeTensor is wrapped to wrong device, found meta, expected cpu"
                    with self.assertRaisesRegex(RuntimeError, msg):
                        op(fake, device=device)
        self.assertEqual(retained.device, torch.device(output_device))
        self.assertIs(maybe_get_fake_mode(retained), mode)

    @skipIfTorchDynamo(_CPP_FAKE_DYNAMO_SKIP)
    @parametrize("alias", ["detach", "data"])
    def test_fake_fallback_clears_inherited_dispatch_keys(self, alias):
        mode = CppFakeTensorMode()
        real = torch.randn(2, 3)
        fake = mode.from_tensor(real)
        keys = torch._C._dispatch_keys(real)
        torch._C._set_fake_dispatch_keys(fake, keys)

        def meta_impl(x):
            result = x.detach() if alias == "detach" else x.data
            self.assertTrue(result.is_meta)
            self.assertFalse(is_fake_tensor(result))
            self.assertIsNone(maybe_get_fake_dispatch_keys(result))
            return result

        with torch.library._scoped_library("_test_fake_fallback", "DEF") as lib:
            lib.define("detached(Tensor x) -> Tensor")
            lib.impl("detached", meta_impl, "Meta")
            result = torch.ops._test_fake_fallback.detached(fake)
        self.assertIsNone(maybe_get_fake_dispatch_keys(result))

    @skipIfTorchDynamo(_CPP_FAKE_DYNAMO_SKIP)
    @torch._dynamo.config.patch(use_cpp_fake_tensor=False)
    @parametrize("active", [False, True])
    @parametrize("implementation", ["meta", "python_cia", "declining_python_cia"])
    @fake_tensor_modes
    def test_fake_fallback_unsets_mode_in_meta_kernel(
        self, active, implementation, mode_cls
    ):
        mode = mode_cls()
        fake = mode.from_tensor(torch.randn(2, 3))
        seen = []

        def meta_impl(x):
            seen.append(("meta", x.is_meta, torch._guards.active_fake_mode()))
            self.assertIsNone(torch._guards.active_fake_mode())
            return x.new_empty(x.shape)

        def python_impl(x):
            seen.append(("python", x.is_meta, torch._guards.active_fake_mode()))
            return (
                NotImplemented if implementation == "declining_python_cia" else x.sum()
            )

        with torch.library._scoped_library("_test_fake_fallback", "DEF") as lib:
            lib.define("probe(Tensor x) -> Tensor")
            lib.impl(
                "probe",
                meta_impl,
                "Meta" if implementation == "meta" else "CompositeImplicitAutograd",
            )
            op = torch.ops._test_fake_fallback.probe.default
            # Export writes py_kernels directly, bypassing the PythonCIA table.
            overrides = (
                {}
                if implementation == "meta"
                else {torch._C.DispatchKey.CompositeImplicitAutograd: python_impl}
            )
            with (
                patch.dict(op.py_kernels, overrides),
                torch.inference_mode(),
                mode if active else contextlib.nullcontext(),
            ):
                before = self._installed()
                result = op(fake)
                self.assertEqual(self._installed(), before)
        expected = [] if implementation == "meta" else [("python", False, mode)]
        if implementation != "python_cia":
            expected.append(("meta", True, None))
        self.assertEqual(seen, expected)
        self.assertEqual(result.shape, () if implementation == "python_cia" else (2, 3))
        self.assertIs(maybe_get_fake_mode(result), mode)
        self.assertEqual(self._installed(), (None, False))

    @skipIfTorchDynamo(_CPP_FAKE_DYNAMO_SKIP)
    def test_fake_fallback_reentrant_mode_release(self):
        mode = CppFakeTensorMode()
        fake = mode.from_tensor(torch.randn(2, 3))
        restored = []

        def meta_impl(x):
            result = x.new_empty(x.shape)
            replacement = CppFakeTensorMode()
            weakref.finalize(
                replacement.fake_tensor_converter,
                lambda: restored.append(self._installed()),
            )
            torch._C._set_dispatch_mode(replacement)
            return result

        with torch.library._scoped_library("_test_fake_fallback", "DEF") as lib:
            lib.define("probe(Tensor x) -> Tensor")
            lib.impl("probe", meta_impl, "Meta")
            with mode:
                torch.ops._test_fake_fallback.probe(fake)
                self.assertEqual(restored, [(mode, True)])
                self.assertEqual(self._installed(), (mode, True))
        self.assertEqual(self._installed(), (None, False))

    def test_require_python_fake_mode(self):
        from torch._guards import require_python_fake_mode

        msg = "CppFakeTensorMode is not supported here yet"
        with self.assertRaisesRegex(RuntimeError, msg):
            require_python_fake_mode(CppFakeTensorMode())
        py_mode = torch._subclasses.fake_tensor.FakeTensorMode()
        self.assertIs(require_python_fake_mode(py_mode), py_mode)
        self.assertIsNone(require_python_fake_mode(None))

    def test_require_python_fake_mode_caller(self):
        from torch.fx.passes.shape_prop import ShapeProp

        # ShapeProp needs a Python FakeTensorMode, so it rejects a C++ one,
        # given to it or found by detect_fake_mode
        gm = torch.fx.symbolic_trace(lambda x: x + 1)
        msg = "CppFakeTensorMode is not supported here yet"
        with self.assertRaisesRegex(RuntimeError, msg):
            ShapeProp(gm, fake_mode=CppFakeTensorMode())
        with tracing(TracingContext(CppFakeTensorMode())):
            with self.assertRaisesRegex(RuntimeError, msg):
                ShapeProp(gm)

    @parametrize(
        "callback,propagate_real_tensors",
        [
            (None, False),
            ("cia", False),
            ("op_impl", False),
            ("custom", False),
            (None, True),
        ],
    )
    def test_cpp_fake_tensor_mode_rebuild(self, callback, propagate_real_tensors):
        from torch._subclasses.fake_tensor import CppFakeTensorMode

        shape_env = ShapeEnv()
        with torch._functorch.config.patch(
            fake_tensor_propagate_real_tensors=propagate_real_tensors
        ):
            mode = CppFakeTensorMode(shape_env=shape_env)
        converter = mode.fake_tensor_converter
        creation_traceback = mode._creation_traceback
        mode.static_shapes = True
        mode.allow_non_fake_inputs = True
        mode.epoch += 2
        epoch = mode.epoch
        mode.allow_meta = False
        mode._allow_unsafe_data_ptr_access = False
        meta = torch.empty((), device="meta")
        fake = converter.from_meta_and_device(mode, meta, "cpu")

        # A collected wrapper is rebuilt around the same C++ mode and state
        ref = weakref.ref(mode)
        del mode
        gc.collect()
        self.assertIsNone(ref())

        def check_mode(rebuilt):
            self.assertIsInstance(rebuilt, CppFakeTensorMode)
            self.assertIs(rebuilt.fake_tensor_converter, converter)
            self.assertIs(rebuilt.shape_env, shape_env)
            self.assertIs(rebuilt._creation_traceback, creation_traceback)
            self.assertEqual(rebuilt.epoch, epoch)
            self.assertTrue(rebuilt.static_shapes)
            self.assertTrue(rebuilt.allow_non_fake_inputs)
            self.assertFalse(rebuilt.allow_meta)
            self.assertFalse(rebuilt._allow_unsafe_data_ptr_access)
            self.assertIs(converter.from_meta_and_device(rebuilt, meta, "cpu"), fake)
            self.assertIs(torch._C._maybe_get_fake_mode(fake), rebuilt)
            with rebuilt:
                self.assertEqual(self._installed(), (rebuilt, True))
                if propagate_real_tensors:
                    with self.assertRaisesRegex(
                        NotImplementedError, "does not support propagate_real_tensors"
                    ):
                        torch.ones(2)

        if callback:
            seen = []

            def kernel(t, rebuilt=None):
                if rebuilt is None:
                    rebuilt = torch._guards.active_fake_mode()
                check_mode(rebuilt)
                self.assertIs(maybe_get_fake_mode(t), rebuilt)
                seen.append(weakref.ref(rebuilt))
                return t.sin()

            with (
                torch.library._scoped_library("_test_fake_rebuilt_mode", "DEF") as lib,
                torch.inference_mode(),
            ):
                lib.define("probe(Tensor x) -> Tensor")
                op = torch.ops._test_fake_rebuilt_mode.probe.default
                if callback == "cia":
                    lib.impl("probe", kernel, "CompositeImplicitAutograd")
                elif callback == "custom":
                    torch.library.register_fake(op, kernel, lib=lib)
                else:
                    from torch._subclasses.fake_impls import (
                        _deregister_op_impl,
                        register_op_impl,
                    )

                    @register_op_impl(op)
                    def op_impl(rebuilt, func, t):
                        self.assertIsNone(torch._guards.active_fake_mode())
                        return kernel(t, rebuilt)

                    self.addCleanup(_deregister_op_impl, op)
                for _ in range(2):
                    result = op(fake)
                    self.assertEqual(result.shape, fake.shape)
                    self.assertIsNone(seen[-1]())
                    self.assertEqual(self._installed(), (None, False))
            self.assertEqual(len(seen), 2)
        else:
            with (
                patch.object(CppFakeTensorMode, "__new__", side_effect=AssertionError),
                patch.object(CppFakeTensorMode, "__init__", side_effect=AssertionError),
            ):
                check_mode(torch._C._maybe_get_fake_mode(fake))
        self.assertEqual(self._installed(), (None, False))

    @xfailIfTorchDynamo
    def test_cpp_fake_tensor_mode_rebuild_threads(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier

        mode = CppFakeTensorMode()
        converter = mode.fake_tensor_converter
        fake = torch._C._from_meta_and_device(
            torch.empty((), device="meta"), torch.device("cpu"), mode
        )
        ref = weakref.ref(mode)
        del mode
        barrier = Barrier(4)

        def lookup(_):
            barrier.wait()
            return torch._C._maybe_get_fake_mode(fake)

        for _ in range(2):
            gc.collect()
            self.assertIsNone(ref())
            with ThreadPoolExecutor(max_workers=4) as pool:
                modes = list(pool.map(lookup, range(4)))
            self.assertTrue(all(mode is modes[0] for mode in modes))
            self.assertIs(modes[0].fake_tensor_converter, converter)
            self.assertIs(torch._C._maybe_get_fake_mode(fake), modes[0])
            ref = weakref.ref(modes[0])
            del modes


if __name__ == "__main__":
    run_tests()
