# NOTE: In order to let Python/C++ API parity test pass for any of the modules here,
# you should update their `cpp_default_constructor_args` and `num_attrs_recursive` values here,
# and also change their "Implementation Parity" bit from "No" to "Yes" in parity-tracker.md.
#
# `cpp_default_constructor_args`: string that represents the required non-keyword arguments
# for the C++ module constructor. For example, since `LinearOptions` expects two non-keyword
# arguments `(in_features, out_features)`, the `cpp_default_constructor_args` for `Linear`
# will be the string representation of any integer 2-tuple, such as "(3, 4)".
# Note that the C++ module constructor must take the exact same number of non-keyword arguments
# as the Python module constructor.
#
# `num_attrs_recursive`: the number of attributes (including parameters, buffers and non-tensor
# attributes) of a module. If the module contains any submodule, the submodule's attributes
# also need to be counted.
module_metadata_map = {
    'Conv1d': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'Conv2d': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'Conv3d': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'ConvTranspose1d': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'ConvTranspose2d': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'ConvTranspose3d': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'Unfold': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'Fold': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'MaxPool1d': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'MaxPool2d': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'MaxPool3d': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'MaxUnpool1d': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'MaxUnpool2d': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'MaxUnpool3d': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'AvgPool1d': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'AvgPool2d': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'AvgPool3d': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'FractionalMaxPool2d': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'LPPool1d': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'LPPool2d': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'AdaptiveMaxPool1d': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'AdaptiveMaxPool2d': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'AdaptiveMaxPool3d': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'AdaptiveAvgPool1d': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'AdaptiveAvgPool2d': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'AdaptiveAvgPool3d': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'ReflectionPad1d': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'ReflectionPad2d': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'ReplicationPad1d': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'ReplicationPad2d': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'ReplicationPad3d': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'ZeroPad2d': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'ConstantPad1d': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'ConstantPad2d': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'ConstantPad3d': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'ELU': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'Hardshrink': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'Hardtanh': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'LeakyReLU': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'LogSigmoid': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'MultiheadAttention': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'PReLU': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'ReLU': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'ReLU6': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'RReLU': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'SELU': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'CELU': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'Sigmoid': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'Softplus': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'Softshrink': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'Softsign': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'Tanh': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'Tanhshrink': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'Threshold': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'Softmin': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'Softmax': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'Softmax2d': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'LogSoftmax': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'AdaptiveLogSoftmaxWithLoss': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'BatchNorm1d': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'BatchNorm2d': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'BatchNorm3d': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'GroupNorm': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'SyncBatchNorm': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'InstanceNorm1d': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'InstanceNorm2d': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'InstanceNorm3d': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'LayerNorm': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'LocalResponseNorm': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'RNN': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'LSTM': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'GRU': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'RNNCell': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'LSTMCell': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'GRUCell': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'Transformer': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'TransformerEncoder': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'TransformerDecoder': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'TransformerEncoderLayer': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'TransformerDecoderLayer': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'Identity': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'Linear': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'Bilinear': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'Dropout': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'Dropout2d': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'Dropout3d': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'AlphaDropout': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'Embedding': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'EmbeddingBag': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'CosineSimilarity': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'PairwiseDistance': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'L1Loss': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'MSELoss': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'CrossEntropyLoss': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'CTCLoss': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'NLLLoss': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'PoissonNLLLoss': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'KLDivLoss': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'BCELoss': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'BCEWithLogitsLoss': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'MarginRankingLoss': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'HingeEmbeddingLoss': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'MultiLabelMarginLoss': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'SmoothL1Loss': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'SoftMarginLoss': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'MultiLabelSoftMarginLoss': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'CosineEmbeddingLoss': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'MultiMarginLoss': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'TripletMarginLoss': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'PixelShuffle': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'Upsample': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'UpsamplingNearest2d': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'UpsamplingBilinear2d': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'Flatten': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'CrossMapLRN2d': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'FractionalMaxPool3d': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
    'GLU': dict(
        cpp_default_constructor_args=None,
        num_attrs_recursive=None,
    ),
}
