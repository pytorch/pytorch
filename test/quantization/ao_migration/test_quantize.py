from torch.testing._internal.common_utils import TestCase

import importlib
from typing import List


class AOMigrationTestCase(TestCase):
    def _test_package_import(self, package_name: str):
        r"""Tests the module import by making sure that all the internals match
        (except the dunder methods)."""
        old_module = importlib.import_module(f'torch.quantization.{package_name}')
        new_module = importlib.import_module(f'torch.ao.quantization.{package_name}')
        old_module_dir = set(dir(old_module))
        new_module_dir = set(dir(new_module))
        # Remove magic modules from checking in subsets
        for el in list(old_module_dir):
            if el[:2] == '__' and el[-2:] == '__':
                old_module_dir.remove(el)
        assert (old_module_dir <= new_module_dir), \
            f"Importing {old_module} vs. {new_module} does not match: " \
            f"{old_module_dir - new_module_dir}"

    def _test_function_import(self, package_name: str, function_list: List[str]):
        r"""Tests individual function list import by comparing the functions
        and their hashes."""
        old_location = importlib.import_module(f'torch.quantization.{package_name}')
        new_location = importlib.import_module(f'torch.ao.quantization.{package_name}')
        for fn_name in function_list:
            old_function = getattr(old_location, fn_name)
            new_function = getattr(new_location, fn_name)
            assert old_function == new_function, f"Functions don't match: {fn_name}"
            assert hash(old_function) == hash(new_function), \
                f"Hashes don't match: {old_function}({hash(old_function)}) vs. " \
                f"{new_function}({hash(new_function)})"


class TestAOMigrationQuantizePy(AOMigrationTestCase):
    def test_package_import(self):
        self._test_package_import('quantize')

    def test_function_import(self):
        function_list = [
            '_convert',
            '_observer_forward_hook',
            '_propagate_qconfig_helper',
            '_remove_activation_post_process',
            '_remove_qconfig',
            'add_observer_',
            'add_quant_dequant',
            'convert',
            'get_observer_dict',
            'get_unique_devices_',
            'is_activation_post_process',
            'prepare',
            'prepare_qat',
            'propagate_qconfig_',
            'quantize',
            'quantize_dynamic',
            'quantize_qat',
            'register_activation_post_process_hook',
            'swap_module',
        ]
        self._test_function_import('quantize', function_list)
