#!/usr/bin/env python3
import unittest
import uuid
from typing import Tuple

import torch
from torch import Tensor, nn
from torch.distributed.nn.jit import instantiator
from torch.testing._internal.common_utils import run_tests


@torch.jit.interface
class MyModuleInterface:
    def forward(
        self, tensor: Tensor, number: int, word: str = "default"
    ) -> Tuple[Tensor, int, str]:
        pass


class MyModule(nn.Module):
    pass


def create_module() -> MyModuleInterface:
    return MyModule()


class TestInstantiator(unittest.TestCase):
    def test_get_return_type_from_callable_expect_module_interface(self):
        return_type = instantiator.get_return_type_from_callable(create_module)
        self.assertEqual(return_type, MyModuleInterface)

    def test_get_return_type_from_callable_expect_module_type(self):
        return_type = instantiator.get_return_type_from_callable(MyModule)
        self.assertEqual(return_type, MyModule)

    def test_get_arg_return_types_from_interface(self):
        args_str, arg_types_str, return_type_str = instantiator.get_arg_return_types_from_interface(
            MyModuleInterface
        )
        self.assertEqual(args_str, "tensor, number, word")
        self.assertEqual(arg_types_str, "tensor: Tensor, number: int, word: str")
        self.assertEqual(return_type_str, "Tuple[Tensor, int, str]")

    def test_instantiate_remote_module_template(self):
        generated_module_name = f"_remote_module_{uuid.uuid4().hex}"
        generated_module = instantiator.instantiate_remote_module_template(
            generated_module_name, MyModuleInterface, True  # is_scriptable
        )
        self.assertTrue(hasattr(generated_module, "_RemoteModule"))
        self.assertTrue(hasattr(generated_module, "_remote_forward"))


if __name__ == "__main__":
    run_tests()
