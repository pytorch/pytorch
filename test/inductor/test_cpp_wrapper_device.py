# Owner(s): ["module: inductor"]

from unittest.mock import patch

import torch
from torch._inductor import config
from torch._inductor.codegen.common import get_device_op_overrides
from torch._inductor.codegen.cpp_wrapper_gpu import (
    CppWrapperGpu,
    DeferredTritonCallWrapper,
)
from torch._inductor.graph import GraphLowering
from torch._inductor.test_case import TestCase
from torch._inductor.utils import IndentedBuffer
from torch._inductor.virtualized import V
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)


@instantiate_parametrized_tests
class TestCppWrapperDevice(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def make_graph(self, aot_mode):
        x = torch.ones(4)
        gm = make_fx(lambda value: value + 1)(x)
        return GraphLowering(gm, [x], cpp_wrapper=True, aot_mode=aot_mode)

    @parametrize("aot_mode", (False, True))
    @parametrize("device_type", ("cuda", "xpu"))
    def test_explicit_device(self, aot_mode, device_type):
        graph = self.make_graph(aot_mode)
        with (
            V.set_graph_handler(graph),
            patch(
                "torch._inductor.codegen.cpp_wrapper_gpu.get_gpu_type",
                side_effect=AssertionError("global device detection was called"),
            ) as detect,
        ):
            wrapper = CppWrapperGpu(device_type=device_type)
        detect.assert_not_called()
        self.assertEqual(wrapper.device, device_type)
        self.assertIs(wrapper.device_codegen, get_device_op_overrides(device_type))

    @parametrize("aot_mode", (False, True))
    def test_default_device_detection(self, aot_mode):
        graph = self.make_graph(aot_mode)
        with (
            V.set_graph_handler(graph),
            patch(
                "torch._inductor.codegen.cpp_wrapper_gpu.get_gpu_type",
                return_value="cuda",
            ) as detect,
        ):
            wrapper = CppWrapperGpu()
        detect.assert_called_once_with()
        self.assertEqual(wrapper.device, "cuda")
        self.assertIs(wrapper.device_codegen, get_device_op_overrides("cuda"))

    @parametrize("aot_mode", (False, True))
    @parametrize("device_type", ("cuda", "xpu"))
    def test_lazy_scratch_uses_selected_device(self, aot_mode, device_type):
        graph = self.make_graph(aot_mode)
        with (
            V.set_graph_handler(graph),
            patch(
                "torch._inductor.codegen.cpp_wrapper_gpu.get_gpu_type",
                side_effect=AssertionError("global device detection was called"),
            ) as detect,
        ):
            wrapper = CppWrapperGpu(device_type=device_type)
            call = DeferredTritonCallWrapper("call_test", "test_kernel", {}, [])
            prefix = IndentedBuffer()
            args = call._generate_lazy_scratch(prefix, wrapper, "arg0")

        detect.assert_not_called()
        self.assertEqual(args, "arg0, &global_scratch_ptr, &profile_scratch_ptr")
        source = prefix.getvalue()
        self.assertEqual(source.count(f"cached_torch_device_type_{device_type}"), 2)
        other_device = "xpu" if device_type == "cuda" else "cuda"
        self.assertNotIn(f"cached_torch_device_type_{other_device}", source)
        self.assertIn(device_type, wrapper.used_cached_devices)

    @parametrize("aot_mode", (False, True))
    def test_subclass_inherits_kernel_registration(self, aot_mode):
        class FixedDeviceWrapper(CppWrapperGpu):
            def __init__(self):
                super().__init__(device_type="cuda")

        graph = self.make_graph(aot_mode)
        with (
            V.set_graph_handler(graph),
            config.patch("triton.autotune_at_compile_time", False),
        ):
            wrapper = FixedDeviceWrapper()
            graph.wrapper_code = wrapper
            wrapper._define_kernel_helper("test_kernel", "kernel source")
            wrapper._generate_kernel_call_helper(
                "test_kernel",
                [],
                device=torch.device("cuda", 0),
                arg_types=[],
                triton_meta={"signature": {}},
                inductor_meta={},
            )

        self.assertEqual(wrapper.get_autotuning_input_name(3), "_REAL_AUTOTUNE_INPUT_3")
        call = wrapper._triton_call_wrappers["call_test_kernel"]
        self.assertEqual(call.kernel_name, "test_kernel")
        self.assertEqual(call.kernel_name_to_body["test_kernel"], "kernel source")


if __name__ == "__main__":
    run_tests()
