# Owner(s): ["oncall: quantization"]

from .common import AOMigrationTestCase


class TestAOMigrationNNQuantized(AOMigrationTestCase):
    def test_functional_import(self):
        r"""Tests the migration of the torch.nn.quantized.functional"""
        function_list = [
            'avg_pool2d',
            'avg_pool3d',
            'adaptive_avg_pool2d',
            'adaptive_avg_pool3d',
            'conv1d',
            'conv2d',
            'conv3d',
            'interpolate',
            'linear',
            'max_pool1d',
            'max_pool2d',
            'celu',
            'leaky_relu',
            'hardtanh',
            'hardswish',
            'threshold',
            'elu',
            'hardsigmoid',
            'clamp',
            'upsample',
            'upsample_bilinear',
            'upsample_nearest',
        ]
        self._test_function_import('functional', function_list, base='nn.quantized')

    def test_modules_import(self):
        module_list = [
            # Modules
            'BatchNorm2d',
            'BatchNorm3d',
            'Conv1d',
            'Conv2d',
            'Conv3d',
            'ConvTranspose1d',
            'ConvTranspose2d',
            'ConvTranspose3d',
            'DeQuantize',
            'ELU',
            'Embedding',
            'EmbeddingBag',
            'GroupNorm',
            'Hardswish',
            'InstanceNorm1d',
            'InstanceNorm2d',
            'InstanceNorm3d',
            'LayerNorm',
            'LeakyReLU',
            'Linear',
            'MaxPool2d',
            'Quantize',
            'ReLU6',
            'Sigmoid',
            'Softmax',
            'Dropout',
            # Wrapper modules
            'FloatFunctional',
            'FXFloatFunctional',
            'QFunctional',
        ]
        self._test_function_import('modules', module_list, base='nn.quantized')

    def test_modules_activation(self):
        function_list = [
            'ReLU6',
            'Hardswish',
            'ELU',
            'LeakyReLU',
            'Sigmoid',
            'Softmax',
        ]
        self._test_function_import('activation', function_list,
                                   base='nn.quantized.modules')

    def test_modules_batchnorm(self):
        function_list = [
            'BatchNorm2d',
            'BatchNorm3d',
        ]
        self._test_function_import('batchnorm', function_list,
                                   base='nn.quantized.modules')

    def test_modules_conv(self):
        function_list = [
            '_reverse_repeat_padding',
            'Conv1d',
            'Conv2d',
            'Conv3d',
            'ConvTranspose1d',
            'ConvTranspose2d',
            'ConvTranspose3d',
        ]

        self._test_function_import('conv', function_list,
                                   base='nn.quantized.modules')

    def test_modules_dropout(self):
        function_list = [
            'Dropout',
        ]
        self._test_function_import('dropout', function_list,
                                   base='nn.quantized.modules')

    def test_modules_embedding_ops(self):
        function_list = [
            'EmbeddingPackedParams',
            'Embedding',
            'EmbeddingBag',
        ]
        self._test_function_import('embedding_ops', function_list,
                                   base='nn.quantized.modules')

    def test_modules_functional_modules(self):
        function_list = [
            'FloatFunctional',
            'FXFloatFunctional',
            'QFunctional',
        ]
        self._test_function_import('functional_modules', function_list,
                                   base='nn.quantized.modules')

    def test_modules_linear(self):
        function_list = [
            'Linear',
            'LinearPackedParams',
        ]
        self._test_function_import('linear', function_list,
                                   base='nn.quantized.modules')

    def test_modules_normalization(self):
        function_list = [
            'LayerNorm',
            'GroupNorm',
            'InstanceNorm1d',
            'InstanceNorm2d',
            'InstanceNorm3d',
        ]
        self._test_function_import('normalization', function_list,
                                   base='nn.quantized.modules')

    def test_modules_utils(self):
        function_list = [
            '_ntuple_from_first',
            '_pair_from_first',
            '_quantize_weight',
            '_hide_packed_params_repr',
            'WeightedQuantizedModule',
        ]
        self._test_function_import('utils', function_list,
                                   base='nn.quantized.modules')

    def test_import_nn_quantized_dynamic_import(self):
        module_list = [
            # Modules
            'Linear',
            'LSTM',
            'GRU',
            'LSTMCell',
            'RNNCell',
            'GRUCell',
            'Conv1d',
            'Conv2d',
            'Conv3d',
            'ConvTranspose1d',
            'ConvTranspose2d',
            'ConvTranspose3d',
        ]
        self._test_function_import('dynamic', module_list, base='nn.quantized')

    def test_import_nn_quantizable_activation(self):
        module_list = [
            # Modules
            'MultiheadAttention',
        ]
        self._test_function_import('activation', module_list, base='nn.quantizable.modules')

    def test_import_nn_quantizable_rnn(self):
        module_list = [
            # Modules
            'LSTM',
            'LSTMCell',
        ]
        self._test_function_import('rnn', module_list, base='nn.quantizable.modules')

    def test_import_nn_qat_conv(self):
        module_list = [
            'Conv1d',
            'Conv2d',
            'Conv3d',
        ]
        self._test_function_import('conv', module_list, base='nn.qat.modules')

    def test_import_nn_qat_embedding_ops(self):
        module_list = [
            'Embedding',
            'EmbeddingBag',
        ]
        self._test_function_import('embedding_ops', module_list, base='nn.qat.modules')

    def test_import_nn_qat_linear(self):
        module_list = [
            'Linear',
        ]
        self._test_function_import('linear', module_list, base='nn.qat.modules')

    def test_import_nn_qat_dynamic_linear(self):
        module_list = [
            'Linear',
        ]
        self._test_function_import('linear', module_list, base='nn.qat.dynamic.modules')


class TestAOMigrationNNIntrinsic(AOMigrationTestCase):
    def test_modules_import_nn_intrinsic(self):
        module_list = [
            # Modules
            '_FusedModule',
            'ConvBn1d',
            'ConvBn2d',
            'ConvBn3d',
            'ConvBnReLU1d',
            'ConvBnReLU2d',
            'ConvBnReLU3d',
            'ConvReLU1d',
            'ConvReLU2d',
            'ConvReLU3d',
            'LinearReLU',
            'BNReLU2d',
            'BNReLU3d',
            'LinearBn1d',
        ]
        self._test_function_import('intrinsic', module_list, base='nn')

    def test_modules_nn_intrinsic_fused(self):
        function_list = [
            '_FusedModule',
            'ConvBn1d',
            'ConvBn2d',
            'ConvBn3d',
            'ConvBnReLU1d',
            'ConvBnReLU2d',
            'ConvBnReLU3d',
            'ConvReLU1d',
            'ConvReLU2d',
            'ConvReLU3d',
            'LinearReLU',
            'BNReLU2d',
            'BNReLU3d',
            'LinearBn1d',
        ]
        self._test_function_import('fused', function_list,
                                   base='nn.intrinsic.modules')

    def test_modules_import_nn_intrinsic_qat(self):
        module_list = [
            "LinearReLU",
            "LinearBn1d",
            "ConvReLU1d",
            "ConvReLU2d",
            "ConvReLU3d",
            "ConvBn1d",
            "ConvBn2d",
            "ConvBn3d",
            "ConvBnReLU1d",
            "ConvBnReLU2d",
            "ConvBnReLU3d",
            "update_bn_stats",
            "freeze_bn_stats",
        ]
        self._test_function_import('qat', module_list, base='nn.intrinsic')

    def test_modules_intrinsic_qat_conv_fused(self):
        function_list = [
            'ConvBn1d',
            'ConvBnReLU1d',
            'ConvReLU1d',
            'ConvBn2d',
            'ConvBnReLU2d',
            'ConvReLU2d',
            'ConvBn3d',
            'ConvBnReLU3d',
            'ConvReLU3d',
            'update_bn_stats',
            'freeze_bn_stats'
        ]
        self._test_function_import('conv_fused', function_list,
                                   base='nn.intrinsic.qat.modules')

    def test_modules_intrinsic_qat_linear_fused(self):
        function_list = [
            'LinearBn1d',
        ]
        self._test_function_import('linear_fused', function_list,
                                   base='nn.intrinsic.qat.modules')

    def test_modules_intrinsic_qat_linear_relu(self):
        function_list = [
            'LinearReLU',
        ]
        self._test_function_import('linear_relu', function_list,
                                   base='nn.intrinsic.qat.modules')

    def test_modules_import_nn_intrinsic_quantized(self):
        module_list = [
            'BNReLU2d',
            'BNReLU3d',
            'ConvReLU1d',
            'ConvReLU2d',
            'ConvReLU3d',
            'LinearReLU',
        ]
        self._test_function_import('quantized', module_list, base='nn.intrinsic')

    def test_modules_intrinsic_quantized_bn_relu(self):
        function_list = [
            'BNReLU2d',
            'BNReLU3d',
        ]
        self._test_function_import('bn_relu', function_list,
                                   base='nn.intrinsic.quantized.modules')

    def test_modules_intrinsic_quantized_conv_relu(self):
        function_list = [
            'ConvReLU1d',
            'ConvReLU2d',
            'ConvReLU3d',
        ]
        self._test_function_import('conv_relu', function_list,
                                   base='nn.intrinsic.quantized.modules')

    def test_modules_intrinsic_quantized_linear_relu(self):
        function_list = [
            'LinearReLU',
        ]
        self._test_function_import('linear_relu', function_list,
                                   base='nn.intrinsic.quantized.modules')

    def test_modules_no_import_nn_intrinsic_quantized_dynamic(self):
        # TODO(future PR): generalize this
        import torch
        _ = torch.ao.nn.intrinsic.quantized.dynamic
        _ = torch.nn.intrinsic.quantized.dynamic
