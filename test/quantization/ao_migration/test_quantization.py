from .common import AOMigrationTestCase


class TestAOMigrationQuantization(AOMigrationTestCase):
    r"""Modules and functions related to the
    `torch/quantization` migration to `torch/ao/quantization`.
    """
    def test_package_import_quantize(self):
        self._test_package_import('quantize')

    def test_function_import_quantize(self):
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

    def test_package_import_fuse_modules(self):
        self._test_package_import('fuse_modules')

    def test_function_import_fuse_modules(self):
        function_list = [
            '_fuse_modules',
            '_get_module',
            '_set_module',
            'fuse_conv_bn',
            'fuse_conv_bn_relu',
            'fuse_known_modules',
            'fuse_modules',
            'get_fuser_method',
        ]
        self._test_function_import('fuse_modules', function_list)
