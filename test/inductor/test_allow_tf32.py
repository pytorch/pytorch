# Owner(s): ["module: inductor"]

import unittest
from unittest import mock

import sympy

import torch
from torch._dynamo.device_interface import (
    CudaInterface,
    device_interfaces,
    DeviceInterface,
    get_interface_for_device,
    register_interface_for_device,
    XpuInterface,
)
from torch._dynamo.source import ConstantSource
from torch._inductor import config as inductor_config
from torch._inductor.codecache import compiled_fx_graph_hash, FxGraphHashDetails
from torch._inductor.graph import GraphLowering
from torch._inductor.heuristics.template.triton import (
    mm_allow_tf32,
    MMTemplateConfigMixin,
)
from torch._inductor.ir import Buffer, FixedLayout
from torch._inductor.kernel_inputs import MMKernelInputs
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.virtualized import V
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


class _TensorNode:
    def __init__(self, device: torch.device, size: tuple[int, ...] = (2, 2)) -> None:
        self._device = device
        self._size = size

    def get_device(self) -> torch.device:
        return self._device

    def get_dtype(self) -> torch.dtype:
        return torch.float32

    def get_size(self) -> tuple[int, ...]:
        return self._size

    def get_stride(self) -> tuple[int, ...]:
        return (self._size[-1], 1)


class _MutableAllowTf32Interface(DeviceInterface):
    enabled = False

    @staticmethod
    def allow_tf32() -> bool:
        return _MutableAllowTf32Interface.enabled

    @staticmethod
    def is_available() -> bool:
        return True


class _AllowTf32Interface(DeviceInterface):
    @staticmethod
    def allow_tf32() -> bool:
        return True

    @staticmethod
    def is_available() -> bool:
        return True


class TestAllowTf32Interface(TestCase):
    def test_device_interface_default_false(self):
        self.assertFalse(DeviceInterface.allow_tf32())

    def test_cuda_allow_tf32_follows_fp32_precision(self):
        orig = torch.backends.cuda.matmul.fp32_precision
        try:
            torch.backends.cuda.matmul.fp32_precision = "tf32"
            self.assertTrue(CudaInterface.allow_tf32())
            torch.backends.cuda.matmul.fp32_precision = "ieee"
            self.assertFalse(CudaInterface.allow_tf32())
        finally:
            torch.backends.cuda.matmul.fp32_precision = orig

    def test_xpu_allow_tf32_matches_mkldnn(self):
        # Without USE_XPU the raw flag is None. The interface still returns a bool.
        self.assertEqual(
            XpuInterface.allow_tf32(), bool(torch.backends.mkldnn.allow_tf32)
        )

    @unittest.skipUnless(
        torch.xpu._is_compiled(), "oneDNN TF32 flag is a no-op without USE_XPU"
    )
    def test_xpu_allow_tf32_follows_mkldnn_flag(self):
        with torch.backends.mkldnn.flags(allow_tf32=True):
            self.assertTrue(XpuInterface.allow_tf32())
        with torch.backends.mkldnn.flags(allow_tf32=False):
            self.assertFalse(XpuInterface.allow_tf32())


@instantiate_parametrized_tests
class TestAllowTf32DynamicShapes(TestCase):
    def _gate(
        self, m, *, aot_mode=False, precision="tf32", dtype=torch.float32, tf32=True
    ):
        shape_env = ShapeEnv()
        if m == "unbacked":
            sym_m = shape_env.create_unbacked_symint().node.expr
        else:
            sym_m = shape_env.create_symbol(m, ConstantSource("m"))
        gm = make_fx(lambda: torch.zeros(2, 2))()
        graph = GraphLowering(gm, shape_env=shape_env, aot_mode=aot_mode)
        orig = torch.backends.cuda.matmul.fp32_precision
        try:
            torch.backends.cuda.matmul.fp32_precision = precision
            with (
                V.set_graph_handler(graph),
                mock.patch("torch.cuda.is_available", return_value=True),
                mock.patch("torch.cuda.is_tf32_supported", return_value=tf32),
                mock.patch(
                    f"{mm_allow_tf32.__module__}.using_b200", return_value=False
                ),
            ):
                dtype_arg = () if dtype is None else (dtype,)
                result = mm_allow_tf32(sym_m, 512, 512, "cuda", *dtype_arg)
        finally:
            torch.backends.cuda.matmul.fp32_precision = orig
        return result, [
            str(g.expr.subs(sym_m, sympy.Symbol("m"))) for g in shape_env.guards
        ]

    @parametrize(
        "m,expected,guards",
        [(1024, True, ["m >= 16"]), (8, False, ["m < 16"]), ("unbacked", False, [])],
    )
    def test_cuda_gate_guards_backed_dims(self, m, expected, guards):
        self.assertEqual(self._gate(m), (expected, guards))

    def test_cuda_gate_no_guard_without_tf32(self):
        self.assertEqual(self._gate(1024, precision="ieee"), (False, []))

    @parametrize("dtype", (torch.bfloat16, torch.float16))
    def test_cuda_gate_no_guard_for_non_fp32(self, dtype):
        # ALLOW_TF32 only affects fp32 dots, so other dtypes keep the proof.
        self.assertEqual(self._gate(1024, dtype=dtype), (False, []))

    def test_cuda_gate_no_guard_without_dtype(self):
        # Callers that don't pass a dtype (4-arg form) keep the guard-free check.
        self.assertEqual(self._gate(1024, dtype=None), (False, []))

    def test_cuda_gate_no_guard_without_tf32_hardware(self):
        self.assertEqual(self._gate(1024, tf32=False), (False, []))

    def test_cuda_gate_aot_mode_needs_proof(self):
        # AOT mode never checks guards at runtime, so a hint alone is not enough.
        self.assertEqual(self._gate(1024, aot_mode=True), (False, []))


@instantiate_parametrized_tests
class TestAllowTf32ExtraKwargs(TestCase):
    mixin = MMTemplateConfigMixin()

    def setUp(self):
        super().setUp()
        self._prev_pua = device_interfaces.get("privateuseone")
        gm = make_fx(lambda: torch.zeros(2, 2))()
        self.graph = GraphLowering(gm)
        self._graph_ctx = V.set_graph_handler(self.graph)
        self._graph_ctx.__enter__()
        self.addCleanup(self._graph_ctx.__exit__, None, None, None)
        # GEMM templates (and so ALLOW_TF32) only run under max-autotune.
        config_ctx = inductor_config.patch(max_autotune=True)
        config_ctx.__enter__()
        self.addCleanup(config_ctx.__exit__, None, None, None)

    def tearDown(self):
        if self._prev_pua is None:
            device_interfaces.pop("privateuseone", None)
        else:
            device_interfaces["privateuseone"] = self._prev_pua
        super().tearDown()

    def _mm_kernel_inputs(
        self,
        device_type: str,
        m: int = 2,
        n: int = 2,
        k: int = 2,
        dtype: torch.dtype = torch.float32,
    ) -> MMKernelInputs:
        device = torch.device(device_type)
        mat1 = Buffer(
            name="mat1",
            layout=FixedLayout(device, dtype, (m, k)),
        )
        mat2 = Buffer(
            name="mat2",
            layout=FixedLayout(device, dtype, (k, n)),
        )
        return MMKernelInputs([mat1, mat2])

    def test_registered_privateuse1_allow_tf32_true(self):
        register_interface_for_device("privateuseone", _AllowTf32Interface)
        kernel_inputs = self._mm_kernel_inputs("privateuseone")
        extra = self.mixin.get_extra_kwargs(kernel_inputs, "mm")
        self.assertTrue(extra["ALLOW_TF32"])

    def test_unregistered_device_allow_tf32_false(self):
        # meta is a real device type with no DeviceInterface. A name torch.device
        # rejects never reaches get_interface_for_device.
        kernel_inputs = self._mm_kernel_inputs("meta")
        extra = self.mixin.get_extra_kwargs(kernel_inputs, "mm")
        self.assertFalse(extra["ALLOW_TF32"])

    def test_default_device_interface_allow_tf32_false(self):
        register_interface_for_device("privateuseone", DeviceInterface)
        kernel_inputs = self._mm_kernel_inputs("privateuseone")
        extra = self.mixin.get_extra_kwargs(kernel_inputs, "mm")
        self.assertFalse(extra["ALLOW_TF32"])

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    def test_cuda_small_shapes_disable_allow_tf32(self):
        orig = torch.backends.cuda.matmul.fp32_precision
        try:
            torch.backends.cuda.matmul.fp32_precision = "tf32"
            kernel_inputs = self._mm_kernel_inputs("cuda", m=2, n=2, k=2)
            extra = self.mixin.get_extra_kwargs(kernel_inputs, "mm")
            self.assertFalse(extra["ALLOW_TF32"])
        finally:
            torch.backends.cuda.matmul.fp32_precision = orig

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    def test_cuda_large_shapes_enable_allow_tf32(self):
        orig = torch.backends.cuda.matmul.fp32_precision
        try:
            torch.backends.cuda.matmul.fp32_precision = "tf32"
            kernel_inputs = self._mm_kernel_inputs("cuda", m=64, n=1024, k=1024)
            extra = self.mixin.get_extra_kwargs(kernel_inputs, "mm")
            self.assertTrue(extra["ALLOW_TF32"])
        finally:
            torch.backends.cuda.matmul.fp32_precision = orig

    @parametrize(
        "b200,hip,mnk,dtype,expected",
        [
            (False, None, (16, 512, 512), torch.float32, True),
            (False, None, (1365, 512, 320), torch.float32, False),
            (False, None, (15, 4096, 4096), torch.float32, False),
            (True, None, (1365, 512, 320), torch.float32, True),
            (True, None, (16, 128, 128), torch.float32, True),
            (True, None, (15, 4096, 4096), torch.float32, False),
            (True, "7.0", (1365, 512, 320), torch.float32, False),
            (True, None, (1365, 512, 320), torch.bfloat16, False),
        ],
    )
    def test_cuda_shape_threshold_b200(self, b200, hip, mnk, dtype, expected):
        # B200 drops the min(N, K) >= 512 term for fp32. Other GPUs, ROCm and
        # non-fp32 dtypes keep it.
        kernel_inputs = self._mm_kernel_inputs("cuda", *mnk, dtype=dtype)
        orig = torch.backends.cuda.matmul.fp32_precision
        try:
            torch.backends.cuda.matmul.fp32_precision = "tf32"
            with (
                mock.patch(f"{mm_allow_tf32.__module__}.using_b200", return_value=b200),
                mock.patch("torch.version.hip", hip),
                mock.patch("torch.cuda.is_available", return_value=True),
                mock.patch("torch.cuda.is_tf32_supported", return_value=True),
            ):
                extra = self.mixin.get_extra_kwargs(kernel_inputs, "mm")
        finally:
            torch.backends.cuda.matmul.fp32_precision = orig
        self.assertEqual(extra["ALLOW_TF32"], expected)

    def test_xpu_unaffected_by_cuda_shape_threshold(self):
        # Platform check. On a non-XPU build both sides are False, so this does
        # not prove the shape gate was skipped. CPU CI coverage for that is
        # test_registered_privateuse1_allow_tf32_true, which uses a 2x2 shape.
        # CUDA would force ALLOW_TF32 off for these shapes. XPU follows the flag.
        kernel_inputs = self._mm_kernel_inputs("xpu", m=2, n=2, k=2)
        if torch.xpu._is_compiled():
            with torch.backends.mkldnn.flags(allow_tf32=True):
                extra = self.mixin.get_extra_kwargs(kernel_inputs, "mm")
            self.assertTrue(extra["ALLOW_TF32"])
            return
        extra = self.mixin.get_extra_kwargs(kernel_inputs, "mm")
        self.assertEqual(extra["ALLOW_TF32"], XpuInterface.allow_tf32())

    def test_cache_key_records_registered_allow_tf32(self):
        details = FxGraphHashDetails(None, [torch.zeros(2, 2)], {}, [])
        self.assertIn(("cpu", False), details.mm_template_allow_tf32)

    def test_cache_key_skips_unregistered_device(self):
        details = FxGraphHashDetails(None, [torch.zeros(2, 2, device="meta")], {}, [])
        recorded = [device for device, _flag in details.mm_template_allow_tf32]
        self.assertNotIn("meta", recorded)

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    def test_cache_key_tracks_cuda_allow_tf32(self):
        tensor = torch.zeros(2, 2, device="cuda")
        orig = torch.backends.cuda.matmul.fp32_precision
        try:
            torch.backends.cuda.matmul.fp32_precision = "tf32"
            enabled = FxGraphHashDetails(None, [tensor], {}, [])
            torch.backends.cuda.matmul.fp32_precision = "ieee"
            disabled = FxGraphHashDetails(None, [tensor], {}, [])
        finally:
            torch.backends.cuda.matmul.fp32_precision = orig
        self.assertIn(("cuda", True), enabled.mm_template_allow_tf32)
        self.assertIn(("cuda", False), disabled.mm_template_allow_tf32)

    def test_non_cuda_flag_changes_fx_graph_cache_key(self):
        # cuda_matmul_settings already hashes CUDA fp32_precision. This swaps
        # the CPU interface so a flag outside that tuple changes the cache key.
        get_interface_for_device("cpu")
        previous = device_interfaces["cpu"]
        gm = make_fx(lambda x: x * 2)(torch.zeros(2, 2))
        example = [torch.zeros(2, 2)]
        try:
            register_interface_for_device("cpu", _MutableAllowTf32Interface)
            _MutableAllowTf32Interface.enabled = True
            enabled = FxGraphHashDetails(gm, example, {}, [])
            enabled_key, _ = compiled_fx_graph_hash(gm, example, {}, [])
            _MutableAllowTf32Interface.enabled = False
            disabled = FxGraphHashDetails(gm, example, {}, [])
            disabled_key, _ = compiled_fx_graph_hash(gm, example, {}, [])
            self.assertEqual(enabled.mm_template_allow_tf32, (("cpu", True),))
            self.assertEqual(disabled.mm_template_allow_tf32, (("cpu", False),))
            self.assertNotEqual(enabled_key, disabled_key)
        finally:
            _MutableAllowTf32Interface.enabled = False
            register_interface_for_device("cpu", previous)


if __name__ == "__main__":
    run_tests()
