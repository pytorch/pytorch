# Owner(s): ["oncall: quantization"]

from .common import AOMigrationTestCase

class TestAOMigrationQuantizationFx(AOMigrationTestCase):
    def test_function_import_quantize_fx(self):
        function_list = [
            '_check_is_graph_module',
            '_swap_ff_with_fxff',
            '_fuse_fx',
            'QuantizationTracer',
            '_prepare_fx',
            '_prepare_standalone_module_fx',
            'fuse_fx',
            'Scope',
            'ScopeContextManager',
            'prepare_fx',
            'prepare_qat_fx',
            '_convert_fx',
            'convert_fx',
            '_convert_standalone_module_fx',
        ]
        self._test_function_import('quantize_fx', function_list)

    def test_function_import_fx(self):
        function_list = [
            'prepare',
            'convert',
            'fuse',
        ]
        self._test_function_import('fx', function_list)

    def test_function_import_fx_graph_module(self):
        function_list = [
            'FusedGraphModule',
            'ObservedGraphModule',
            '_is_observed_module',
            'ObservedStandaloneGraphModule',
            '_is_observed_standalone_module',
            'QuantizedGraphModule'
        ]
        self._test_function_import('fx.graph_module', function_list)

    def test_function_import_fx_pattern_utils(self):
        function_list = [
            'QuantizeHandler',
            '_register_fusion_pattern',
            'get_default_fusion_patterns',
            '_register_quant_pattern',
            'get_default_quant_patterns',
            'get_default_output_activation_post_process_map'
        ]
        self._test_function_import('fx.pattern_utils', function_list)

    def test_function_import_fx_equalize(self):
        function_list = [
            'reshape_scale',
            '_InputEqualizationObserver',
            '_WeightEqualizationObserver',
            'calculate_equalization_scale',
            'EqualizationQConfig',
            'input_equalization_observer',
            'weight_equalization_observer',
            'default_equalization_qconfig',
            'fused_module_supports_equalization',
            'nn_module_supports_equalization',
            'node_supports_equalization',
            'is_equalization_observer',
            'get_op_node_and_weight_eq_obs',
            'maybe_get_weight_eq_obs_node',
            'maybe_get_next_input_eq_obs',
            'maybe_get_next_equalization_scale',
            'scale_input_observer',
            'scale_weight_node',
            'scale_weight_functional',
            'clear_weight_quant_obs_node',
            'remove_node',
            'update_obs_for_equalization',
            'convert_eq_obs',
            '_convert_equalization_ref',
            'get_layer_sqnr_dict',
            'get_equalization_qconfig_dict'
        ]
        self._test_function_import('fx._equalize', function_list)

    def test_function_import_fx_quantization_patterns(self):
        function_list = [
            'QuantizeHandler',
            'BinaryOpQuantizeHandler',
            'CatQuantizeHandler',
            'ConvReluQuantizeHandler',
            'LinearReLUQuantizeHandler',
            'BatchNormQuantizeHandler',
            'EmbeddingQuantizeHandler',
            'RNNDynamicQuantizeHandler',
            'DefaultNodeQuantizeHandler',
            'FixedQParamsOpQuantizeHandler',
            'CopyNodeQuantizeHandler',
            'CustomModuleQuantizeHandler',
            'GeneralTensorShapeOpQuantizeHandler',
            'StandaloneModuleQuantizeHandler'
        ]
        self._test_function_import(
            'fx.quantization_patterns',
            function_list,
            new_package_name='fx.quantize_handler',
        )

    def test_function_import_fx_match_utils(self):
        function_list = [
            '_MatchResult',
            'MatchAllNode',
            '_is_match',
            '_find_matches'
        ]
        self._test_function_import('fx.match_utils', function_list)

    def test_function_import_fx_prepare(self):
        function_list = [
            'prepare'
        ]
        self._test_function_import('fx.prepare', function_list)

    def test_function_import_fx_convert(self):
        function_list = [
            'convert'
        ]
        self._test_function_import('fx.convert', function_list)

    def test_function_import_fx_fuse(self):
        function_list = ['fuse']
        self._test_function_import('fx.fuse', function_list)

    def test_function_import_fx_fusion_patterns(self):
        function_list = [
            'FuseHandler',
            'DefaultFuseHandler'
        ]
        self._test_function_import(
            'fx.fusion_patterns',
            function_list,
            new_package_name='fx.fuse_handler',
        )

    # we removed matching test for torch.quantization.fx.quantization_types
    # old: torch.quantization.fx.quantization_types
    # new: torch.ao.quantization.utils
    # both are valid, but we'll deprecate the old path in the future

    def test_function_import_fx_utils(self):
        function_list = [
            'get_custom_module_class_keys',
            'get_linear_prepack_op_for_dtype',
            'get_qconv_prepack_op',
            'get_new_attr_name_with_prefix',
            'graph_module_from_producer_nodes',
            'assert_and_get_unique_device',
            'create_getattr_from_value',
            'all_node_args_have_no_tensors',
            'get_non_observable_arg_indexes_and_types',
            'maybe_get_next_module'
        ]
        self._test_function_import('fx.utils', function_list)
