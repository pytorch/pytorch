import importlib
from typing import List, Optional

from torch.testing._internal.common_utils import TestCase


class AOMigrationTestCase(TestCase):
    def _test_function_import(
        self,
        package_name: str,
        function_list: List[str],
        base: Optional[str] = None,
        new_package_name: Optional[str] = None,
    ):
        r"""Tests individual function list import by comparing the functions
        and their hashes."""
        if base is None:
            base = "quantization"
        old_base = "torch." + base
        new_base = "torch.ao." + base
        if new_package_name is None:
            new_package_name = package_name
        old_location = importlib.import_module(f"{old_base}.{package_name}")
        new_location = importlib.import_module(f"{new_base}.{new_package_name}")
        for fn_name in function_list:
            old_function = getattr(old_location, fn_name)
            new_function = getattr(new_location, fn_name)
            assert old_function == new_function, f"Functions don't match: {fn_name}"
            assert hash(old_function) == hash(new_function), (
                f"Hashes don't match: {old_function}({hash(old_function)}) vs. "
                f"{new_function}({hash(new_function)})"
            )

    def _test_dict_import(
        self, package_name: str, dict_list: List[str], base: Optional[str] = None
    ):
        r"""Tests individual function list import by comparing the functions
        and their hashes."""
        if base is None:
            base = "quantization"
        old_base = "torch." + base
        new_base = "torch.ao." + base
        old_location = importlib.import_module(f"{old_base}.{package_name}")
        new_location = importlib.import_module(f"{new_base}.{package_name}")
        for dict_name in dict_list:
            old_dict = getattr(old_location, dict_name)
            new_dict = getattr(new_location, dict_name)
            assert old_dict == new_dict, f"Dicts don't match: {dict_name}"
            for key in new_dict.keys():
                assert (
                    old_dict[key] == new_dict[key]
                ), f"Dicts don't match: {dict_name} for key {key}"
