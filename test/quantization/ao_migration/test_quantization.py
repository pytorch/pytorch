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

    def test_package_import_stubs(self):
        self._test_package_import('stubs')

    def test_function_import_stubs(self):
        function_list = [
            'QuantStub',
            'DeQuantStub',
            'QuantWrapper',
        ]
        self._test_function_import('stubs', function_list)

    def test_package_import_quantize_jit(self):
        self._test_package_import('quantize_jit')

    def test_function_import_quantize_jit(self):
        function_list = [
            '_check_is_script_module',
            '_check_forward_method',
            'script_qconfig',
            'script_qconfig_dict',
            'fuse_conv_bn_jit',
            '_prepare_jit',
            'prepare_jit',
            'prepare_dynamic_jit',
            '_convert_jit',
            'convert_jit',
            'convert_dynamic_jit',
            '_quantize_jit',
            'quantize_jit',
            'quantize_dynamic_jit',
        ]
        self._test_function_import('quantize_jit', function_list)

    def test_package_import_fake_quantize(self):
        self._test_package_import('fake_quantize')

    def test_function_import_fake_quantize(self):
        function_list = [
            '_is_per_channel',
            '_is_per_tensor',
            '_is_symmetric_quant',
            'FakeQuantizeBase',
            'FakeQuantize',
            'FixedQParamsFakeQuantize',
            'FusedMovingAvgObsFakeQuantize',
            'default_fake_quant',
            'default_weight_fake_quant',
            'default_symmetric_fixed_qparams_fake_quant',
            'default_affine_fixed_qparams_fake_quant',
            'default_per_channel_weight_fake_quant',
            'default_histogram_fake_quant',
            'default_fused_act_fake_quant',
            'default_fused_wt_fake_quant',
            'default_fused_per_channel_wt_fake_quant',
            '_is_fake_quant_script_module',
            'disable_fake_quant',
            'enable_fake_quant',
            'disable_observer',
            'enable_observer',
        ]
        self._test_function_import('fake_quantize', function_list)

    def test_package_import_observer(self):
        self._test_package_import('observer')

    def test_function_import_observer(self):
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
        self._test_function_import('observer', function_list)

    def test_package_import_qconfig(self):
        self._test_package_import('qconfig')

    def test_function_import_qconfig(self):
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
        self._test_function_import('qconfig', function_list)
