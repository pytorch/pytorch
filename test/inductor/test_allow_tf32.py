# Owner(s): ["module: inductor"]

import unittest
from unittest import mock

import torch
from torch._dynamo.device_interface import (
    CudaInterface,
    device_interfaces,
    DeviceInterface,
    register_interface_for_device,
    XpuInterface,
)
from torch._inductor.graph import GraphLowering
from torch._inductor.heuristics.template.triton import MMTemplateConfigMixin
from torch._inductor.ir import Buffer, FixedLayout
from torch._inductor.kernel_inputs import MMKernelInputs
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.virtualized import V
from torch.fx.experimental.proxy_tensor import make_fx


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
        self.assertEqual(XpuInterface.allow_tf32(), torch.backends.mkldnn.allow_tf32)


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

    def tearDown(self):
        if self._prev_pua is None:
            device_interfaces.pop("privateuseone", None)
        else:
            device_interfaces["privateuseone"] = self._prev_pua
        super().tearDown()

    def _mm_kernel_inputs(
        self, device_type: str, m: int = 2, n: int = 2, k: int = 2
    ) -> MMKernelInputs:
        device = torch.device(device_type)
        mat1 = Buffer(
            name="mat1",
            layout=FixedLayout(device, torch.float32, (m, k)),
        )
        mat2 = Buffer(
            name="mat2",
            layout=FixedLayout(device, torch.float32, (k, n)),
        )
        return MMKernelInputs([mat1, mat2])

    def test_registered_privateuse1_allow_tf32_true(self):
        register_interface_for_device("privateuseone", _AllowTf32Interface)
        kernel_inputs = self._mm_kernel_inputs("privateuseone")
        extra = self.mixin.get_extra_kwargs(kernel_inputs, "mm")
        self.assertTrue(extra["ALLOW_TF32"])

    def test_unregistered_device_allow_tf32_false(self):
        kernel_inputs = self._mm_kernel_inputs("cpu")
        with mock.patch(
            "torch._inductor.heuristics.template.triton.get_interface_for_device",
            side_effect=NotImplementedError,
        ):
            extra = self.mixin.get_extra_kwargs(kernel_inputs, "mm")
        self.assertFalse(extra["ALLOW_TF32"])

    @mock.patch(
        "torch._inductor.heuristics.template.triton.get_interface_for_device",
        return_value=DeviceInterface,
    )
    def test_default_device_interface_allow_tf32_false(self, _mock_get_iface):
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

    def test_xpu_unaffected_by_cuda_shape_threshold(self):
        kernel_inputs = self._mm_kernel_inputs("xpu", m=2, n=2, k=2)
        with (
            mock.patch.object(XpuInterface, "allow_tf32", return_value=True),
            mock.patch(
                "torch._inductor.heuristics.template.triton.get_interface_for_device",
                return_value=XpuInterface,
            ),
        ):
            extra = self.mixin.get_extra_kwargs(kernel_inputs, "mm")
        self.assertTrue(extra["ALLOW_TF32"])


if __name__ == "__main__":
    run_tests()
