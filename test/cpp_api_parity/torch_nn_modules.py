import torch

from cpp_api_parity import TorchNNModuleMetadata

# NOTE: In order to let Python/C++ API parity test pass for any of the modules here,
# after fixing the C++ module implementation you should change change its "Implementation Parity"
# from "No" to "Yes" in parity-tracker.md, and change the module's `TorchNNModuleMetadata` here
# to look like:
#
# 'Linear': TorchNNModuleMetadata(
#     cpp_default_constructor_args="(3, 4)",
#     num_attrs_recursive=4,
# )
#
# The meaning of each field of `TorchNNModuleMetadata` is as follows:
#
# `cpp_default_constructor_args`: string that represents the required non-keyword arguments
#     for the C++ module constructor. For example, since `LinearOptions` expects two non-keyword
#     arguments `(in_features, out_features)`, the `cpp_default_constructor_args` for `Linear`
#     will be the string representation of any integer 2-tuple, such as "(3, 4)".
#     Note that the C++ module constructor must take the exact same number of non-keyword arguments
#     as the Python module constructor.
#
# `num_attrs_recursive`: the number of attributes (including parameters, buffers and non-tensor
#     attributes, but excluding the attributes in `python_ignored_attrs`) of the Python module.
#     If the module contains any submodule, the submodule's attributes also need to be counted.
#
# `python_ignored_constructor_args`: (optional) list of Python constructor args that are
#     ignored in Python/C++ API parity test.
#
# `python_ignored_attrs`: (optional) list of Python module attributes (including parameters,
#     buffers and non-tensor attributes) that are ignored in Python/C++ API parity test.
#
# `python_optional_attribute_to_jit_type`: (optional) map between Python None-able module
#     attribute to its corresponding JIT type. For example, in `AvgPool2d`:
#     { "divisor_override": torch._C.OptionalType(torch._C.IntType.get()) }
module_metadata_map = {
    'Conv1d': TorchNNModuleMetadata(),
    'Conv2d': TorchNNModuleMetadata(),
    'Conv3d': TorchNNModuleMetadata(),
    'ConvTranspose1d': TorchNNModuleMetadata(),
    'ConvTranspose2d': TorchNNModuleMetadata(),
    'ConvTranspose3d': TorchNNModuleMetadata(),
    'Unfold': TorchNNModuleMetadata(),
    'Fold': TorchNNModuleMetadata(
        cpp_default_constructor_args="(3, 2)",
        num_attrs_recursive=5,
    ),
    'MaxPool1d': TorchNNModuleMetadata(
        cpp_default_constructor_args="(2)",
        num_attrs_recursive=5,
        python_ignored_constructor_args=['return_indices'],
        python_ignored_attrs=['return_indices'],
    ),
    'MaxPool2d': TorchNNModuleMetadata(
        cpp_default_constructor_args="(2)",
        num_attrs_recursive=5,
        python_ignored_constructor_args=['return_indices'],
        python_ignored_attrs=['return_indices'],
    ),
    'MaxPool3d': TorchNNModuleMetadata(
        cpp_default_constructor_args="(2)",
        num_attrs_recursive=5,
        python_ignored_constructor_args=['return_indices'],
        python_ignored_attrs=['return_indices'],
    ),
    'MaxUnpool1d': TorchNNModuleMetadata(),
    'MaxUnpool2d': TorchNNModuleMetadata(),
    'MaxUnpool3d': TorchNNModuleMetadata(),
    'AvgPool1d': TorchNNModuleMetadata(
        cpp_default_constructor_args="(2)",
        num_attrs_recursive=5,
    ),
    'AvgPool2d': TorchNNModuleMetadata(
        cpp_default_constructor_args="(2)",
        num_attrs_recursive=6,
        python_optional_attribute_to_jit_type={
            "divisor_override": torch._C.OptionalType(torch._C.IntType.get()),
        }
    ),
    'AvgPool3d': TorchNNModuleMetadata(
        cpp_default_constructor_args="(2)",
        num_attrs_recursive=6,
        python_optional_attribute_to_jit_type={
            "divisor_override": torch._C.OptionalType(torch._C.IntType.get()),
        }
    ),
    'FractionalMaxPool2d': TorchNNModuleMetadata(),
    'LPPool1d': TorchNNModuleMetadata(),
    'LPPool2d': TorchNNModuleMetadata(),
    'AdaptiveMaxPool1d': TorchNNModuleMetadata(),
    'AdaptiveMaxPool2d': TorchNNModuleMetadata(),
    'AdaptiveMaxPool3d': TorchNNModuleMetadata(),
    'AdaptiveAvgPool1d': TorchNNModuleMetadata(),
    'AdaptiveAvgPool2d': TorchNNModuleMetadata(),
    'AdaptiveAvgPool3d': TorchNNModuleMetadata(),
    'ReflectionPad1d': TorchNNModuleMetadata(),
    'ReflectionPad2d': TorchNNModuleMetadata(),
    'ReplicationPad1d': TorchNNModuleMetadata(),
    'ReplicationPad2d': TorchNNModuleMetadata(),
    'ReplicationPad3d': TorchNNModuleMetadata(),
    'ZeroPad2d': TorchNNModuleMetadata(),
    'ConstantPad1d': TorchNNModuleMetadata(),
    'ConstantPad2d': TorchNNModuleMetadata(),
    'ConstantPad3d': TorchNNModuleMetadata(),
    'ELU': TorchNNModuleMetadata(),
    'Hardshrink': TorchNNModuleMetadata(),
    'Hardtanh': TorchNNModuleMetadata(),
    'LeakyReLU': TorchNNModuleMetadata(),
    'LogSigmoid': TorchNNModuleMetadata(),
    'MultiheadAttention': TorchNNModuleMetadata(),
    'PReLU': TorchNNModuleMetadata(),
    'ReLU': TorchNNModuleMetadata(),
    'ReLU6': TorchNNModuleMetadata(),
    'RReLU': TorchNNModuleMetadata(),
    'SELU': TorchNNModuleMetadata(),
    'CELU': TorchNNModuleMetadata(),
    'GELU': TorchNNModuleMetadata(),
    'Sigmoid': TorchNNModuleMetadata(),
    'Softplus': TorchNNModuleMetadata(),
    'Softshrink': TorchNNModuleMetadata(),
    'Softsign': TorchNNModuleMetadata(),
    'Tanh': TorchNNModuleMetadata(),
    'Tanhshrink': TorchNNModuleMetadata(),
    'Threshold': TorchNNModuleMetadata(),
    'Softmin': TorchNNModuleMetadata(),
    'Softmax': TorchNNModuleMetadata(),
    'Softmax2d': TorchNNModuleMetadata(),
    'LogSoftmax': TorchNNModuleMetadata(),
    'AdaptiveLogSoftmaxWithLoss': TorchNNModuleMetadata(),
    'BatchNorm1d': TorchNNModuleMetadata(),
    'BatchNorm2d': TorchNNModuleMetadata(),
    'BatchNorm3d': TorchNNModuleMetadata(),
    'GroupNorm': TorchNNModuleMetadata(),
    'SyncBatchNorm': TorchNNModuleMetadata(),
    'InstanceNorm1d': TorchNNModuleMetadata(),
    'InstanceNorm2d': TorchNNModuleMetadata(),
    'InstanceNorm3d': TorchNNModuleMetadata(),
    'LayerNorm': TorchNNModuleMetadata(),
    'LocalResponseNorm': TorchNNModuleMetadata(),
    'RNN': TorchNNModuleMetadata(),
    'LSTM': TorchNNModuleMetadata(),
    'GRU': TorchNNModuleMetadata(),
    'RNNCell': TorchNNModuleMetadata(),
    'LSTMCell': TorchNNModuleMetadata(),
    'GRUCell': TorchNNModuleMetadata(),
    'Transformer': TorchNNModuleMetadata(),
    'TransformerEncoder': TorchNNModuleMetadata(),
    'TransformerDecoder': TorchNNModuleMetadata(),
    'TransformerEncoderLayer': TorchNNModuleMetadata(),
    'TransformerDecoderLayer': TorchNNModuleMetadata(),
    'Identity': TorchNNModuleMetadata(),
    'Linear': TorchNNModuleMetadata(),
    'Bilinear': TorchNNModuleMetadata(),
    'Dropout': TorchNNModuleMetadata(),
    'Dropout2d': TorchNNModuleMetadata(),
    'Dropout3d': TorchNNModuleMetadata(),
    'AlphaDropout': TorchNNModuleMetadata(),
    'Embedding': TorchNNModuleMetadata(),
    'EmbeddingBag': TorchNNModuleMetadata(),
    'CosineSimilarity': TorchNNModuleMetadata(),
    'PairwiseDistance': TorchNNModuleMetadata(),
    'L1Loss': TorchNNModuleMetadata(
        cpp_default_constructor_args="()",
        num_attrs_recursive=1,
        python_ignored_constructor_args=['size_average', 'reduce'],
    ),
    'MSELoss': TorchNNModuleMetadata(),
    'CrossEntropyLoss': TorchNNModuleMetadata(),
    'CTCLoss': TorchNNModuleMetadata(),
    'NLLLoss': TorchNNModuleMetadata(),
    'PoissonNLLLoss': TorchNNModuleMetadata(),
    'KLDivLoss': TorchNNModuleMetadata(),
    'BCELoss': TorchNNModuleMetadata(),
    'BCEWithLogitsLoss': TorchNNModuleMetadata(),
    'MarginRankingLoss': TorchNNModuleMetadata(),
    'HingeEmbeddingLoss': TorchNNModuleMetadata(),
    'MultiLabelMarginLoss': TorchNNModuleMetadata(),
    'SmoothL1Loss': TorchNNModuleMetadata(),
    'SoftMarginLoss': TorchNNModuleMetadata(),
    'MultiLabelSoftMarginLoss': TorchNNModuleMetadata(),
    'CosineEmbeddingLoss': TorchNNModuleMetadata(),
    'MultiMarginLoss': TorchNNModuleMetadata(),
    'TripletMarginLoss': TorchNNModuleMetadata(),
    'PixelShuffle': TorchNNModuleMetadata(),
    'Upsample': TorchNNModuleMetadata(),
    'UpsamplingNearest2d': TorchNNModuleMetadata(),
    'UpsamplingBilinear2d': TorchNNModuleMetadata(),
    'Flatten': TorchNNModuleMetadata(),
    'CrossMapLRN2d': TorchNNModuleMetadata(),
    'FractionalMaxPool3d': TorchNNModuleMetadata(),
    'GLU': TorchNNModuleMetadata(),
}
