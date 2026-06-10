```{eval-rst}
.. role:: hidden
    :class: hidden-section
```

# torch.nn.functional

```{eval-rst}
.. currentmodule:: torch.nn.functional
```

## Convolution functions

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    conv1d
    conv2d
    conv3d
    conv_transpose1d
    conv_transpose2d
    conv_transpose3d
    unfold
    fold
```

## Pooling functions

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    avg_pool1d
    avg_pool2d
    avg_pool3d
    max_pool1d
    max_pool2d
    max_pool3d
    max_unpool1d
    max_unpool2d
    max_unpool3d
    lp_pool1d
    lp_pool2d
    lp_pool3d
    adaptive_max_pool1d
    adaptive_max_pool2d
    adaptive_max_pool3d
    adaptive_avg_pool1d
    adaptive_avg_pool2d
    adaptive_avg_pool3d
    fractional_max_pool2d
    fractional_max_pool3d
```

## Attention Mechanisms

The {mod}`torch.nn.attention.bias` module contains attention_biases that are designed to be used with
scaled_dot_product_attention.

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    scaled_dot_product_attention
```

## Non-linear activation functions

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    threshold
    threshold_
    relu
    relu_
    hardtanh
    hardtanh_
    hardswish
    relu6
    elu
    elu_
    selu
    celu
    leaky_relu
    leaky_relu_
    prelu
    rrelu
    rrelu_
    glu
    gelu
    logsigmoid
    hardshrink
    tanhshrink
    softsign
    softplus
    softmin
    softmax
    softshrink
    gumbel_softmax
    log_softmax
    tanh
    sigmoid
    hardsigmoid
    silu
    mish
    batch_norm
    group_norm
    instance_norm
    layer_norm
    local_response_norm
    rms_norm
    normalize

.. _Link 1: https://arxiv.org/abs/1611.00712
.. _Link 2: https://arxiv.org/abs/1611.01144
```

## Linear functions

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    linear
    bilinear
```

## Dropout functions

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    dropout
    alpha_dropout
    feature_alpha_dropout
    dropout1d
    dropout2d
    dropout3d
```

## Sparse functions

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    embedding
    embedding_bag
    one_hot
```

## Distance functions

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    pairwise_distance
    cosine_similarity
    pdist
```

## Loss functions

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    binary_cross_entropy
    binary_cross_entropy_with_logits
    poisson_nll_loss
    cosine_embedding_loss
    cross_entropy
    ctc_loss
    gaussian_nll_loss
    hinge_embedding_loss
    kl_div
    l1_loss
    linear_cross_entropy
    mse_loss
    margin_ranking_loss
    multilabel_margin_loss
    multilabel_soft_margin_loss
    multi_margin_loss
    nll_loss
    huber_loss
    smooth_l1_loss
    soft_margin_loss
    triplet_margin_loss
    triplet_margin_with_distance_loss
```

## Vision functions

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    pixel_shuffle
    pixel_unshuffle
    pad
    interpolate
    upsample
    upsample_nearest
    upsample_bilinear
    grid_sample
    affine_grid
```

## DataParallel functions (multi-GPU, distributed)

### {hidden}`data_parallel`

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    torch.nn.parallel.data_parallel
```

## Low-Precision functions

```{eval-rst}
.. NOTE: ScalingType and SwizzleType are pybind11 enums imported via type alias
   (e.g. ``_ScalingType as ScalingType``). Sphinx autodoc/autosummary treats
   these as aliases and renders "alias of _ScalingType" instead of the class
   docstring and members. Manual ``.. class::`` with ``.. attribute::`` entries
   is required. When adding new enum values, update both the C++ definition in
   aten/src/ATen/BlasBackend.h and the entries below.

.. class:: ScalingType

    Enum describing how a tensor's scaling factors are organized for use with
    :func:`~torch.nn.functional.scaled_mm` and :func:`~torch.nn.functional.scaled_grouped_mm`.

    .. attribute:: TensorWise

        Single ``float32`` scale for the entire tensor.

    .. attribute:: RowWise

        One ``float32`` scale per row.

    .. attribute:: BlockWise1x16

        One ``float8_e4m3fn`` scale per 16 contiguous values.

    .. attribute:: BlockWise1x32

        One ``float8_e8m0fnu`` scale per 32 contiguous values (OCP MX format).

    .. attribute:: BlockWise1x128

        One ``float32`` scale per 128 contiguous values (OCP MX format).

    .. attribute:: BlockWise128x128

        One ``float32`` scale per 128x128 tile.

.. class:: SwizzleType

    Enum describing the swizzling pattern of scale tensors for use with
    :func:`~torch.nn.functional.scaled_mm` and :func:`~torch.nn.functional.scaled_grouped_mm`.

    .. attribute:: NO_SWIZZLE

        No swizzling.

    .. attribute:: SWIZZLE_32_4_4

        Blackwell-style 32x4x4 swizzle.
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    grouped_mm
    scaled_mm
    scaled_grouped_mm
```
