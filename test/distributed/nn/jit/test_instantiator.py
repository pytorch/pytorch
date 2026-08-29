#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]

import sys

import torch
import torch.distributed as dist
from torch import nn, Tensor


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.distributed.nn.jit import instantiator
from torch.testing._internal.common_utils import run_tests, TestCase


@torch.jit.interface
class MyModuleInterface:
    def forward(
        self, tensor: Tensor, number: int, word: str = "default"
    ) -> tuple[Tensor, int, str]:
        pass


class MyModule(nn.Module):
    pass


def create_module():
    return MyModule()


class TestInstantiator(TestCase):
    def test_get_arg_return_types_from_interface(self):
        (
            args_str,
            arg_types_str,
            return_type_str,
        ) = instantiator.get_arg_return_types_from_interface(MyModuleInterface)
        self.assertEqual(args_str, "tensor, number, word")
        self.assertEqual(arg_types_str, "tensor: Tensor, number: int, word: str")
        self.assertEqual(return_type_str, "Tuple[Tensor, int, str]")

    def test_instantiate_scripted_remote_module_template(self):
        generated_module = instantiator.instantiate_scriptable_remote_module_template(
            MyModuleInterface
        )
        self.assertTrue(hasattr(generated_module, "_remote_forward"))
        self.assertTrue(hasattr(generated_module, "_generated_methods"))

    def test_instantiate_non_scripted_remote_module_template(self):
        generated_module = (
            instantiator.instantiate_non_scriptable_remote_module_template()
        )
        self.assertTrue(hasattr(generated_module, "_remote_forward"))
        self.assertTrue(hasattr(generated_module, "_generated_methods"))

    def test_template_variant_cache_key(self):
        # The same interface class can be used by both a RemoteModule on an
        # accelerator device and one on cpu. The two must instantiate different
        # template variants: the cache key in ``sys.modules`` must include
        # ``enable_moving_cpu_tensors_to_cuda``. Otherwise the second
        # instantiation silently reuses the first one's variant.
        device_module = instantiator.instantiate_scriptable_remote_module_template(
            MyModuleInterface, enable_moving_cpu_tensors_to_cuda=True
        )
        cpu_module = instantiator.instantiate_scriptable_remote_module_template(
            MyModuleInterface, enable_moving_cpu_tensors_to_cuda=False
        )
        self.assertIsNot(device_module, cpu_module)
        # Repeated instantiation of the same variant still hits the cache.
        self.assertIs(
            cpu_module,
            instantiator.instantiate_scriptable_remote_module_template(
                MyModuleInterface, enable_moving_cpu_tensors_to_cuda=False
            ),
        )
        # The device variant moves CPU tensors to the device; the cpu variant
        # must not contain the moving logic at all.
        device_source = device_module.__loader__.get_source(device_module.__name__)
        cpu_source = cpu_module.__loader__.get_source(cpu_module.__name__)
        self.assertIn("out_args", device_source)
        self.assertNotIn("out_args", cpu_source)

    def test_template_device_branch_is_device_generic(self):
        # The moving template must gate on ``device.type == "cpu"`` (early
        # return for cpu) instead of ``device.type != "cuda"``, so that
        # tensors are moved for any non-cpu device (cuda, xpu, npu, ...)
        # registered by a backend, not just cuda.
        generated_module = instantiator.instantiate_scriptable_remote_module_template(
            MyModuleInterface, enable_moving_cpu_tensors_to_cuda=True
        )
        source = generated_module.__loader__.get_source(generated_module.__name__)
        self.assertIn('if device.type == "cpu":', source)
        self.assertNotIn('!= "cuda"', source)


if __name__ == "__main__":
    run_tests()
