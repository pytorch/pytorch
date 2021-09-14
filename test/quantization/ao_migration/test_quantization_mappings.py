from torch.testing._internal.common_utils import TestCase


class TestQuantizationMappingsPyImport(TestCase):
    def test_package_import(self):
        import torch.ao.quantization.quantization_mappings as new_module
        import torch.quantization.quantization_mappings as old_module

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
            "DEFAULT_REFERENCE_STATIC_QUANT_MODULE_MAPPINGS",
            "DEFAULT_STATIC_QUANT_MODULE_MAPPINGS",
            "DEFAULT_QAT_MODULE_MAPPINGS",
            "DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS",
            "_INCLUDE_QCONFIG_PROPAGATE_LIST",
            "DEFAULT_FLOAT_TO_QUANTIZED_OPERATOR_MAPPINGS",
            "DEFAULT_MODULE_TO_ACT_POST_PROCESS",
            "no_observer_set",
            "get_default_static_quant_module_mappings",
            "get_static_quant_module_class",
            "get_dynamic_quant_module_class",
            "get_default_qat_module_mappings",
            "get_default_dynamic_quant_module_mappings",
            "get_default_qconfig_propagation_list",
            "get_default_compare_output_module_list",
            "get_default_float_to_quantized_operator_mappings",
            "get_quantized_operator",
            "_get_special_act_post_process",
            "_has_special_act_post_process",
        ]
        import torch.ao.quantization.quantization_mappings as new_location
        import torch.quantization.quantization_mappings as old_location

        for fn_name in function_list:
            old_function = getattr(old_location, fn_name)
            new_function = getattr(new_location, fn_name)
            assert old_function == new_function, f"Imports don't match: {fn_name}"
