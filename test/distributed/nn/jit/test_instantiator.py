#!/usr/bin/env python3
import unittest
import pathlib
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


def create_module():
    return MyModule()


class TestInstantiator(unittest.TestCase):
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

        dir_path = pathlib.Path(instantiator.INSTANTIATED_TEMPLATE_DIR_PATH)
        num_files_before_cleanup = len(list(dir_path.iterdir()))
        instantiator.cleanup_generated_modules()
        num_files_after_cleanup = len(list(dir_path.iterdir()))
        # This test assumes single RPC worker group in the same time.
        self.assertGreater(num_files_before_cleanup, num_files_after_cleanup)

    def test_instantiate_non_scripted_remote_module_template(self):
        generated_module = instantiator.instantiate_non_scriptable_remote_module_template()
        self.assertTrue(hasattr(generated_module, "_remote_forward"))
        self.assertTrue(hasattr(generated_module, "_generated_methods"))

        dir_path = pathlib.Path(instantiator.INSTANTIATED_TEMPLATE_DIR_PATH)
        num_files_before_cleanup = len(list(dir_path.iterdir()))
        instantiator.cleanup_generated_modules()
        num_files_after_cleanup = len(list(dir_path.iterdir()))
        # This test assumes single RPC worker group in the same time.
        self.assertGreater(num_files_before_cleanup, num_files_after_cleanup)

if __name__ == "__main__":
    run_tests()
