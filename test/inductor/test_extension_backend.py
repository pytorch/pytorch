# Owner(s): ["module: inductor"]
import os
import sys
import tempfile
import unittest

import torch
import torch._dynamo
import torch.utils.cpp_extension
from torch._C import FileCheck
from torch.testing._internal.common_utils import skipIfWindows
from torch.testing._internal.inductor_utils import has_cpp_wrapper_for_device


try:
    from extension_backends.cpp.extension_codegen_backend import (  # @manual=fbcode//caffe2/test/inductor/extension_backends:extension_codegen_backend
        ExtensionCppWrapperCodegen,
        ExtensionScheduling,
        ExtensionWrapperCodegen,
    )
except ImportError:
    from .extension_backends.cpp.extension_codegen_backend import (
        ExtensionCppWrapperCodegen,
        ExtensionScheduling,
        ExtensionWrapperCodegen,
    )

import torch._inductor.config as config
from torch._inductor import cpu_vec_isa, metrics
from torch._inductor.codegen.common import (
    device_op_overrides_dict,
    DeviceOpOverrides,
    get_scheduling_for_device,
    get_wrapper_codegen_for_device,
    register_backend_for_device,
    register_device_op_overrides,
)
from torch._inductor.codegen.cpp_utils import device_to_aten
from torch._inductor.codegen.cpu_device_op_overrides import CpuDeviceOpOverrides
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_FBCODE,
    IS_MACOS,
    parametrize,
    xfailIfS390X,
)


class ExtensionDeviceOpOverrides(CpuDeviceOpOverrides):
    def aten_device_type(self) -> str:
        return "at::kPrivateUse1"


class MissingAtenDeviceTypeOverrides(DeviceOpOverrides):
    pass


class InvalidAtenDeviceTypeOverrides(CpuDeviceOpOverrides):
    def aten_device_type(self) -> str:
        return "c10::DeviceType::PrivateUse1"


try:
    try:
        from . import test_torchinductor
    except ImportError:
        import test_torchinductor  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
except unittest.SkipTest:
    if __name__ == "__main__":
        sys.exit(0)
    raise


run_and_get_cpp_code = test_torchinductor.run_and_get_cpp_code
TestCase = test_torchinductor.TestCase


class DeviceToAtenTests(TestCase):
    @parametrize(
        "device, expected",
        [
            ("cpu", "at::kCPU"),
            ("cuda", "at::kCUDA"),
            ("xpu", "at::kXPU"),
            ("mps", "at::kMPS"),
            ("meta", "at::kMeta"),
        ],
    )
    def test_builtin_device_types(self, device, expected):
        self.assertEqual(device_to_aten(device), expected)

    def test_unregistered_device_type(self):
        with self.assertRaisesRegex(RuntimeError, "No ATen device type mapping"):
            device_to_aten("unregistered_aten_device_type")

    def test_missing_aten_device_type(self):
        device = "missing_aten_device_type"
        register_device_op_overrides(device, MissingAtenDeviceTypeOverrides())
        self.addCleanup(device_op_overrides_dict.pop, device, None)
        with self.assertRaisesRegex(RuntimeError, "No ATen device type mapping"):
            device_to_aten(device)

    def test_invalid_aten_device_type(self):
        device = "invalid_aten_device_type"
        register_device_op_overrides(device, InvalidAtenDeviceTypeOverrides())
        self.addCleanup(device_op_overrides_dict.pop, device, None)
        with self.assertRaisesRegex(RuntimeError, "must return.*at::k"):
            device_to_aten(device)

    @parametrize("device", ["tpu", "mtia"])
    def test_unsupported_device_types(self, device):
        # MTIA has no aoti_torch_device_type_mtia shim yet; update this when it does.
        with self.assertRaisesRegex(RuntimeError, "No ATen device type mapping"):
            device_to_aten(device)


class BaseExtensionBackendTests(TestCase):
    module = None

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls._build_dir = tempfile.TemporaryDirectory()
        source_file_path = os.path.dirname(os.path.abspath(__file__))
        source_file = os.path.join(
            source_file_path, "extension_backends/cpp/extension_device.cpp"
        )
        cls.module = torch.utils.cpp_extension.load(
            name="extension_device",
            sources=[
                str(source_file),
            ],
            extra_cflags=["-g"],
            verbose=True,
            build_directory=cls._build_dir.name,
        )

    @classmethod
    def tearDownClass(cls):
        cls._stack.close()
        super().tearDownClass()

        cls._build_dir.cleanup()

    def setUp(self):
        torch._dynamo.reset()
        super().setUp()

        # cpp extensions use relative paths. Those paths are relative to
        # this file, so we'll change the working directory temporarily
        self.old_working_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        if self.module is None:
            raise AssertionError

    def tearDown(self):
        super().tearDown()
        torch._dynamo.reset()

        backend_name = torch._C._get_privateuse1_backend_name()
        if hasattr(torch, backend_name):
            delattr(torch, backend_name)
        if f"torch.{backend_name}" in sys.modules:
            del sys.modules[f"torch.{backend_name}"]

        os.chdir(self.old_working_dir)


@unittest.skipIf(IS_FBCODE, "cpp_extension doesn't work in fbcode right now")
class ExtensionBackendTests(BaseExtensionBackendTests):
    @xfailIfS390X
    @skipIfWindows
    def test_open_device_registration(self):
        torch.utils.rename_privateuse1_backend("extension_device")
        torch._register_device_module("extension_device", self.module)

        register_backend_for_device(
            "extension_device",
            ExtensionScheduling,
            ExtensionWrapperCodegen,
            ExtensionCppWrapperCodegen,
        )
        register_device_op_overrides("extension_device", ExtensionDeviceOpOverrides())
        self.assertEqual(
            device_to_aten("extension_device"),
            "at::kPrivateUse1",
        )
        self.assertTrue(
            get_scheduling_for_device("extension_device") == ExtensionScheduling
        )
        self.assertTrue(
            get_wrapper_codegen_for_device("extension_device")
            == ExtensionWrapperCodegen
        )
        self.assertTrue(
            get_wrapper_codegen_for_device("extension_device", True)
            == ExtensionCppWrapperCodegen
        )

        self.assertFalse(self.module.custom_op_called())
        device = self.module.custom_device()
        x = torch.empty(2, 16).to(device=device).fill_(1)
        self.assertTrue(self.module.custom_op_called())
        y = torch.empty(2, 16).to(device=device).fill_(2)
        z = torch.empty(2, 16).to(device=device).fill_(3)
        ref = torch.empty(2, 16).fill_(5)

        self.assertTrue(x.device == device)
        self.assertTrue(y.device == device)
        self.assertTrue(z.device == device)

        def fn(a, b, c):
            return a * b + c

        for cpp_wrapper_flag in [True, False]:
            with config.patch({"cpp_wrapper": cpp_wrapper_flag}):
                metrics.reset()
                opt_fn = torch.compile()(fn)
                _, code = run_and_get_cpp_code(opt_fn, x, y, z)
                if (
                    cpu_vec_isa.valid_vec_isa_list()
                    and os.getenv("ATEN_CPU_CAPABILITY") != "default"
                ):
                    load_expr = "loadu"
                else:
                    load_expr = " = in_ptr0[static_cast<int64_t>(x0)];"
                FileCheck().check("void").check(load_expr).check(
                    "extension_device"
                ).run(code)
                if cpp_wrapper_flag:
                    self.assertIn("CACHE_TORCH_DEVICE(privateuse1);", code)
                opt_fn(x, y, z)
                res = opt_fn(x, y, z)
                self.assertEqual(ref, res.to(device="cpu"))

    @parametrize("device", ["cpu", "has_cpp_wrapper_extension_device"])
    def test_has_cpp_wrapper_for_device(self, device: str):
        # Check that calling the function without having registered a backend
        # ourselves doesn't error.
        _ = has_cpp_wrapper_for_device(device)

        # Check when we don't have a C++ wrapper
        register_backend_for_device(
            device,
            ExtensionScheduling,
            ExtensionWrapperCodegen,
        )
        self.assertFalse(has_cpp_wrapper_for_device(device))

        # Check when we have a C++ wrapper
        register_backend_for_device(
            device,
            ExtensionScheduling,
            ExtensionWrapperCodegen,
            ExtensionCppWrapperCodegen,
        )
        self.assertTrue(has_cpp_wrapper_for_device(device))


instantiate_parametrized_tests(DeviceToAtenTests)
instantiate_parametrized_tests(ExtensionBackendTests)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests
    from torch.testing._internal.inductor_utils import HAS_CPU

    # cpp_extension doesn't work in fbcode right now
    if HAS_CPU and not IS_MACOS and not IS_FBCODE:
        run_tests()
