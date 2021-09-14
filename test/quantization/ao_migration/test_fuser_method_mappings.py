from torch.testing._internal.common_utils import TestCase


class TestFuserMethodMappingsPyImport(TestCase):
    def test_package_import(self):
        import torch.ao.quantization.fuser_method_mappings as new_module
        import torch.quantization.fuser_method_mappings as old_module

        old_module_dir = set(dir(old_module))
        new_module_dir = set(dir(new_module))
        # Remove magic modules from checking in subsets
        for el in list(old_module_dir):
            if el[:2] == "__" and el[-2:] == "__":
                old_module_dir.remove(el)
        assert old_module_dir <= new_module_dir, (
            f"Importing {old_module} vs. {new_module} does not match: "
            f"{old_module_dir - new_module_dir}"
        )

    def test_function_import(self):
        function_list = [
            "fuse_conv_bn",
            "fuse_conv_bn_relu",
            "fuse_linear_bn",
            "DEFAULT_OP_LIST_TO_FUSER_METHOD",
            "get_fuser_method",
        ]
        import torch.ao.quantization.fuser_method_mappings as new_location
        import torch.quantization.fuser_method_mappings as old_location

        for fn_name in function_list:
            old_function = getattr(old_location, fn_name)
            new_function = getattr(new_location, fn_name)
            assert old_function == new_function, f"Imports don't match: {fn_name}"
