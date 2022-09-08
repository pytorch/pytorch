from torch.testing._internal.common_utils import TestCase

import importlib
from typing import List, Optional

class AOMigrationTestCase(TestCase):
    def _test_package_import(self, package_name: str,
                             base: Optional[str] = None,
                             skip: List[str] = None):
        r"""Tests the module import by making sure that all the internals match
        (except the dunder methods).

        Args:
            package_name: The name of the package to be tested
            base: The base namespace where the `package_name` resides
            skip: The list of the subpackages/modules/functions to skip
        """
        skip = skip or []
        base = base or 'quantization'
        old_base = 'torch.' + base
        new_base = 'torch.ao.' + base
        old_module = importlib.import_module(f'{old_base}.{package_name}')
        new_module = importlib.import_module(f'{new_base}.{package_name}')
        old_module_dir = set(dir(old_module))
        new_module_dir = set(dir(new_module))
        # Remove magic modules from checking in subsets
        for el in list(old_module_dir):
            if el.startswith('__') and el.endswith('__'):
                # Remove dunder
                old_module_dir.remove(el)
            if el in skip:
                # Remove skips
                old_module_dir.remove(el)
        assert (old_module_dir <= new_module_dir), \
            f"Importing {old_module} vs. {new_module} does not match: " \
            f"{old_module_dir - new_module_dir}"

    def _test_function_import(self, package_name: str, function_list: List[str],
                              base: Optional[str] = None):
        r"""Tests individual function list import by comparing the functions
        and their hashes."""
        if base is None:
            base = 'quantization'
        old_base = 'torch.' + base
        new_base = 'torch.ao.' + base
        old_location = importlib.import_module(f'{old_base}.{package_name}')
        new_location = importlib.import_module(f'{new_base}.{package_name}')
        for fn_name in function_list:
            old_function = getattr(old_location, fn_name)
            new_function = getattr(new_location, fn_name)
            assert old_function == new_function, f"Functions don't match: {fn_name}"
            assert hash(old_function) == hash(new_function), \
                f"Hashes don't match: {old_function}({hash(old_function)}) vs. " \
                f"{new_function}({hash(new_function)})"

    def _test_dict_import(self, package_name: str, dict_list: List[str],
                          base: Optional[str] = None):
        r"""Tests individual function list import by comparing the functions
        and their hashes."""
        if base is None:
            base = 'quantization'
        old_base = 'torch.' + base
        new_base = 'torch.ao.' + base
        old_location = importlib.import_module(f'{old_base}.{package_name}')
        new_location = importlib.import_module(f'{new_base}.{package_name}')
        for dict_name in dict_list:
            old_dict = getattr(old_location, dict_name)
            new_dict = getattr(new_location, dict_name)
            assert old_dict == new_dict, f"Dicts don't match: {dict_name}"
            for key in new_dict.keys():
                assert old_dict[key] == new_dict[key], f"Dicts don't match: {dict_name} for key {key}"
