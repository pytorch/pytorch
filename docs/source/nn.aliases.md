```{eval-rst}
.. role:: hidden
    :class: hidden-section
```


# Aliases in torch.nn
```{eval-rst}
.. automodule:: torch.nn.modules
```


The following are aliases to their counterparts in ``torch.nn`` in nested namespaces.

## torch.nn.modules

The following are aliases to their counterparts in ``torch.nn`` in the ``torch.nn.modules`` namespace.

### Containers (Aliases)
```{eval-rst}
.. currentmodule:: torch.nn.modules
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    container.Sequential
    container.ModuleList
    container.ModuleDict
    container.ParameterList
    container.ParameterDict

```

### Convolution Layers (Aliases)
```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    conv.Conv1d
    conv.Conv2d
    conv.Conv3d
    conv.ConvTranspose1d
    conv.ConvTranspose2d
    conv.ConvTranspose3d
    conv.LazyConv1d
    conv.LazyConv2d
    conv.LazyConv3d
    conv.LazyConvTranspose1d
    conv.LazyConvTranspose2d
    conv.LazyConvTranspose3d
    fold.Unfold
    fold.Fold

```

### Pooling layers (Aliases)
```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    pooling.MaxPool1d
    pooling.MaxPool2d
    pooling.MaxPool3d
    pooling.MaxUnpool1d
    pooling.MaxUnpool2d
    pooling.MaxUnpool3d
    pooling.AvgPool1d
    pooling.AvgPool2d
    pooling.AvgPool3d
    pooling.FractionalMaxPool2d
    pooling.FractionalMaxPool3d
    pooling.LPPool1d
    pooling.LPPool2d
    pooling.LPPool3d
    pooling.AdaptiveMaxPool1d
    pooling.AdaptiveMaxPool2d
    pooling.AdaptiveMaxPool3d
    pooling.AdaptiveAvgPool1d
    pooling.AdaptiveAvgPool2d
    pooling.AdaptiveAvgPool3d

```

### Padding Layers (Aliases)
```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    padding.ReflectionPad1d
    padding.ReflectionPad2d
    padding.ReflectionPad3d
    padding.ReplicationPad1d
    padding.ReplicationPad2d
    padding.ReplicationPad3d
    padding.ZeroPad1d
    padding.ZeroPad2d
    padding.ZeroPad3d
    padding.ConstantPad1d
    padding.ConstantPad2d
    padding.ConstantPad3d
    padding.CircularPad1d
    padding.CircularPad2d
    padding.CircularPad3d

```

### Non-linear Activations (weighted sum, nonlinearity) (Aliases)
```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    activation.ELU
    activation.Hardshrink
    activation.Hardsigmoid
    activation.Hardtanh
    activation.Hardswish
    activation.LeakyReLU
    activation.LogSigmoid
    activation.MultiheadAttention
    activation.PReLU
    activation.ReLU
    activation.ReLU6
    activation.RReLU
    activation.SELU
    activation.CELU
    activation.GELU
    activation.Sigmoid
    activation.SiLU
    activation.Mish
    activation.Softplus
    activation.Softshrink
    activation.Softsign
    activation.Tanh
    activation.Tanhshrink
    activation.Threshold
    activation.GLU

```

### Non-linear Activations (other) (Aliases)
```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    activation.Softmin
    activation.Softmax
    activation.Softmax2d
    activation.LogSoftmax
    adaptive.AdaptiveLogSoftmaxWithLoss

```

### Normalization Layers (Aliases)
```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    batchnorm.BatchNorm1d
    batchnorm.BatchNorm2d
    batchnorm.BatchNorm3d
    batchnorm.LazyBatchNorm1d
    batchnorm.LazyBatchNorm2d
    batchnorm.LazyBatchNorm3d
    normalization.GroupNorm
    batchnorm.SyncBatchNorm
    instancenorm.InstanceNorm1d
    instancenorm.InstanceNorm2d
    instancenorm.InstanceNorm3d
    instancenorm.LazyInstanceNorm1d
    instancenorm.LazyInstanceNorm2d
    instancenorm.LazyInstanceNorm3d
    normalization.LayerNorm
    normalization.LocalResponseNorm
    normalization.RMSNorm

```

### Recurrent Layers (Aliases)
```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    rnn.RNNBase
    rnn.RNN
    rnn.LSTM
    rnn.GRU
    rnn.RNNCell
    rnn.LSTMCell
    rnn.GRUCell

```

### Transformer Layers (Aliases)
```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    transformer.Transformer
    transformer.TransformerEncoder
    transformer.TransformerDecoder
    transformer.TransformerEncoderLayer
    transformer.TransformerDecoderLayer

```

### Linear Layers (Aliases)
```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    linear.Identity
    linear.Linear
    linear.Bilinear
    linear.LazyLinear

```

### Dropout Layers (Aliases)
```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    dropout.Dropout
    dropout.Dropout1d
    dropout.Dropout2d
    dropout.Dropout3d
    dropout.AlphaDropout
    dropout.FeatureAlphaDropout

```

### Sparse Layers (Aliases)
```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    sparse.Embedding
    sparse.EmbeddingBag

```

### Distance Functions (Aliases)
```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    distance.CosineSimilarity
    distance.PairwiseDistance

```

### Loss Functions (Aliases)
```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    loss.L1Loss
    loss.MSELoss
    loss.CrossEntropyLoss
    loss.CTCLoss
    loss.NLLLoss
    loss.PoissonNLLLoss
    loss.GaussianNLLLoss
    loss.KLDivLoss
    loss.BCELoss
    loss.BCEWithLogitsLoss
    loss.MarginRankingLoss
    loss.HingeEmbeddingLoss
    loss.MultiLabelMarginLoss
    loss.HuberLoss
    loss.SmoothL1Loss
    loss.SoftMarginLoss
    loss.MultiLabelSoftMarginLoss
    loss.CosineEmbeddingLoss
    loss.MultiMarginLoss
    loss.TripletMarginLoss
    loss.TripletMarginWithDistanceLoss

```

### Vision Layers (Aliases)
```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    pixelshuffle.PixelShuffle
    pixelshuffle.PixelUnshuffle
    upsampling.Upsample
    upsampling.UpsamplingNearest2d
    upsampling.UpsamplingBilinear2d

```

### Shuffle Layers (Aliases)
```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    channelshuffle.ChannelShuffle

```

## torch.nn.utils

The following are aliases to their counterparts in ``torch.nn.utils`` in nested namespaces.

Utility functions to clip parameter gradients.

```{eval-rst}
.. currentmodule:: torch.nn.utils
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    clip_grad.clip_grad_norm_
    clip_grad.clip_grad_norm
    clip_grad.clip_grad_value_


```

Utility functions to flatten and unflatten Module parameters to and from a single vector.

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    convert_parameters.parameters_to_vector
    convert_parameters.vector_to_parameters

```

Utility functions to fuse Modules with BatchNorm modules.

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    fusion.fuse_conv_bn_eval
    fusion.fuse_conv_bn_weights
    fusion.fuse_linear_bn_eval
    fusion.fuse_linear_bn_weights

```

Utility functions to convert Module parameter memory formats.

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    memory_format.convert_conv2d_weight_memory_format
    memory_format.convert_conv3d_weight_memory_format

```

Utility functions to apply and remove weight normalization from Module parameters.

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    weight_norm.weight_norm
    weight_norm.remove_weight_norm
    spectral_norm.spectral_norm
    spectral_norm.remove_spectral_norm

```

Utility functions for initializing Module parameters.

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    init.skip_init
```