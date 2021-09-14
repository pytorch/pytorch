from torch.testing._internal.common_utils import TestCase


class TestObserverPyImport(TestCase):
    def test_package_import(self):
        import torch.ao.quantization.observer as new_module
        import torch.quantization.observer as old_module

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
            "_PartialWrapper",
            "_with_args",
            "_with_callable_args",
            "ABC",
            "ObserverBase",
            "_ObserverBase",
            "MinMaxObserver",
            "MovingAverageMinMaxObserver",
            "PerChannelMinMaxObserver",
            "MovingAveragePerChannelMinMaxObserver",
            "HistogramObserver",
            "PlaceholderObserver",
            "RecordingObserver",
            "NoopObserver",
            "_is_activation_post_process",
            "_is_per_channel_script_obs_instance",
            "get_observer_state_dict",
            "load_observer_state_dict",
            "default_observer",
            "default_placeholder_observer",
            "default_debug_observer",
            "default_weight_observer",
            "default_histogram_observer",
            "default_per_channel_weight_observer",
            "default_dynamic_quant_observer",
            "default_float_qparams_observer",
        ]
        import torch.ao.quantization.observer as new_location
        import torch.quantization.observer as old_location

        for fn_name in function_list:
            old_function = getattr(old_location, fn_name)
            new_function = getattr(new_location, fn_name)
            assert old_function == new_function, f"Imports don't match: {fn_name}"
