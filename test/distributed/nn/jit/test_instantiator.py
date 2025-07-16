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


if __name__ == "__main__":
    run_tests()
