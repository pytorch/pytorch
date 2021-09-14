from torch.testing._internal.common_utils import TestCase


class TestQconfigPyImport(TestCase):
    def test_package_import(self):
        import torch.ao.quantization.qconfig as new_module
        import torch.quantization.qconfig as old_module

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
            "QConfig",
            "default_qconfig",
            "default_debug_qconfig",
            "default_per_channel_qconfig",
            "QConfigDynamic",
            "default_dynamic_qconfig",
            "float16_dynamic_qconfig",
            "float16_static_qconfig",
            "per_channel_dynamic_qconfig",
            "float_qparams_weight_only_qconfig",
            "default_qat_qconfig",
            "default_weight_only_qconfig",
            "default_activation_only_qconfig",
            "default_qat_qconfig_v2",
            "get_default_qconfig",
            "get_default_qat_qconfig",
            "assert_valid_qconfig",
            "QConfigAny",
            "add_module_to_qconfig_obs_ctr",
            "qconfig_equals"
        ]
        import torch.ao.quantization.qconfig as new_location
        import torch.quantization.qconfig as old_location

        for fn_name in function_list:
            old_function = getattr(old_location, fn_name)
            new_function = getattr(new_location, fn_name)
            assert old_function == new_function, f"Imports don't match: {fn_name}"
