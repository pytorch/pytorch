# Owner(s): ["module: inductor"]
import contextlib
import pickle
import tempfile
from pathlib import Path
from unittest import mock

import torch
from torch._dynamo import device_interface as di
from torch._dynamo.device_interface import CpuInterface, CudaInterface, DeviceInterface
from torch._inductor import config, cpp_builder
from torch._inductor.output_code import CompiledAOTI
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


@instantiate_parametrized_tests
class TestAOTIDeviceInterface(TestCase):
    hw_classification = HardwareClassification.GENERIC

    @parametrize(
        "device,interface,runner_name,args",
        [
            ("cpu", CpuInterface, "AOTIModelContainerRunnerCpu", ("model.so", 1)),
            (
                "cuda:1",
                CudaInterface,
                "AOTIModelContainerRunnerCuda",
                ("model.so", 1, "cuda:1", "", True),
            ),
            (
                "xpu:2",
                di.XpuInterface,
                "AOTIModelContainerRunnerXpu",
                ("model.so", 1, "xpu:2", ""),
            ),
        ],
    )
    def test_builtin_constructor_contract(self, device, interface, runner_name, args):
        with mock.patch.object(torch._C._aoti, runner_name, create=True) as constructor:
            loaded = interface.load_aoti_callable("model.so", device)
            constructor.assert_called_once_with(*args)
            self.assertIs(loaded, constructor.return_value.run)

    def test_default_rejects_unsupported_device(self):
        with self.assertRaises(NotImplementedError):
            DeviceInterface.load_aoti_callable("model.so", "mps")


@instantiate_parametrized_tests
class TestCompiledAOTIDeviceDispatch(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def setUp(self):
        super().setUp()
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(
            config.patch(
                {
                    "aot_inductor.link_libtorch": True,
                    "aot_inductor.cross_target_platform": None,
                    "aot_inductor.package_cpp_only": False,
                    "enable_autograd_for_aot": True,
                }
            )
        )
        stack.enter_context(mock.patch.object(cpp_builder, "_IS_MACOS", False))
        stack.enter_context(mock.patch.object(cpp_builder, "_IS_WINDOWS", False))

    @parametrize("device", ["cpu", "cuda", "cuda:1", "xpu", "xpu:2"])
    def test_dispatch_preserves_full_device(self, device):
        interface = di.get_interface_for_device(torch.device(device))
        with mock.patch.object(interface, "load_aoti_callable") as load:
            compiled = CompiledAOTI(filename="model.so", device_type=device)
            load.assert_called_once_with("model.so", device)
            self.assertIs(compiled.current_callable, load.return_value)
            self.assertTrue(compiled._boxed_call)
            inputs = [object()]
            self.assertIs(compiled(inputs), load.return_value.return_value)
            load.return_value.assert_called_once_with(inputs)

    def test_registered_external_device(self):
        load = mock.Mock()

        class ExternalInterface(DeviceInterface):
            load_aoti_callable = staticmethod(load)

        name = torch._C._get_privateuse1_backend_name()
        di.get_registered_device_interfaces()
        with mock.patch.dict(di.device_interfaces):
            di.register_interface_for_device(name, ExternalInterface)
            compiled = CompiledAOTI(filename="model.so", device_type=f"{name}:3")
            load.assert_called_once_with("model.so", f"{name}:3")
            self.assertIs(compiled.current_callable, load.return_value)
            inputs = [object()]
            self.assertIs(compiled(inputs), load.return_value.return_value)
            load.return_value.assert_called_once_with(inputs)

    def test_registered_device_without_aoti_support(self):
        with self.assertRaisesRegex(
            RuntimeError, "unsupported device type mps"
        ) as caught:
            CompiledAOTI(filename="model.so", device_type="mps")
        self.assertIsInstance(caught.exception.__cause__, NotImplementedError)

    @parametrize("device", ["cpu", "cpu:1"])
    def test_loader_not_implemented_reports_unsupported_device(self, device):
        error = NotImplementedError("AOTI loading is unavailable")
        with mock.patch.object(CpuInterface, "load_aoti_callable", side_effect=error):
            with self.assertRaisesRegex(
                RuntimeError, f"unsupported device type {device}"
            ) as caught:
                CompiledAOTI(filename="model.so", device_type=device)
        self.assertIs(type(caught.exception), RuntimeError)
        self.assertIs(caught.exception.__cause__, error)

    def test_unregistered_device(self):
        di.get_registered_device_interfaces()
        with mock.patch.dict(di.device_interfaces):
            di.device_interfaces.pop("meta", None)
            with self.assertRaisesRegex(
                NotImplementedError, "No interface for device meta"
            ):
                CompiledAOTI(filename="model.so", device_type="meta")

    def test_invalid_device(self):
        with self.assertRaises(RuntimeError):
            CompiledAOTI(filename="model.so", device_type="cuda_invalid")

    def test_loader_error_propagates(self):
        error = OSError("cannot load model.so")
        with mock.patch.object(CpuInterface, "load_aoti_callable", side_effect=error):
            with self.assertRaises(OSError) as caught:
                CompiledAOTI(filename="model.so", device_type="cpu")
        self.assertIs(caught.exception, error)

    def test_selects_shared_library_from_file_list(self):
        with mock.patch.object(CpuInterface, "load_aoti_callable") as load:
            CompiledAOTI(filename=["model.cpp", "model.so"], device_type="cpu")
            load.assert_called_once_with("model.so", "cpu")

    def test_graph_module_bypasses_loader(self):
        gm = torch.fx.symbolic_trace(torch.nn.Identity())
        with mock.patch.object(CpuInterface, "load_aoti_callable") as load:
            compiled = CompiledAOTI(filename=gm, device_type="cpu")
            self.assertIs(compiled.current_callable, gm)
            load.assert_not_called()

    @parametrize(
        "settings",
        [
            {"aot_inductor.link_libtorch": False},
            {"aot_inductor.cross_target_platform": "windows"},
            {"aot_inductor.package_cpp_only": True},
            {"enable_autograd_for_aot": False},
        ],
    )
    def test_disabled_loading(self, settings):
        with config.patch(settings):
            with mock.patch.object(CpuInterface, "load_aoti_callable") as load:
                compiled = CompiledAOTI(filename="model.so", device_type="cpu")
                self.assertIsNone(compiled.current_callable)
                load.assert_not_called()

    @parametrize("platform_flag", ["_IS_MACOS", "_IS_WINDOWS"])
    def test_unsupported_host_bypasses_loader(self, platform_flag):
        with mock.patch.object(cpp_builder, platform_flag, True):
            with mock.patch.object(CpuInterface, "load_aoti_callable") as load:
                compiled = CompiledAOTI(filename="model.so", device_type="cpu")
                self.assertIsNone(compiled.current_callable)
                load.assert_not_called()

    def test_serialization_and_post_compile_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.so"
            path.write_bytes(b"test AOTI artifact")
            with mock.patch.object(CpuInterface, "load_aoti_callable") as load:
                compiled = CompiledAOTI(filename=str(path), device_type="cpu")
                compiled.prepare_for_serialization()
                restored = pickle.loads(pickle.dumps(compiled))
                self.assertIsNone(restored.current_callable)
                self.assertEqual(restored._cached_files, {str(path): path.read_bytes()})
                load.reset_mock()
                restored.post_compile([], mock.Mock(), {})
                load.assert_called_once_with(str(path), "cpu")
                self.assertIs(restored.current_callable, load.return_value)
                self.assertTrue(restored._boxed_call)
                restored.post_compile([], mock.Mock(), {})
                load.assert_called_once()


if __name__ == "__main__":
    run_tests()
