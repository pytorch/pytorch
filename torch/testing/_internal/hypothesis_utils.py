from collections import defaultdict
import numpy as np
import torch

import hypothesis
from hypothesis import assume
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as stnp
from hypothesis.strategies import SearchStrategy

from torch.testing._internal.common_quantized import _calculate_dynamic_qparams, _calculate_dynamic_per_channel_qparams

# Setup for the hypothesis tests.
# The tuples are (torch_quantized_dtype, zero_point_enforce), where the last
# element is enforced zero_point. If None, any zero_point point within the
# range of the data type is OK.

# Tuple with all quantized data types.
_ALL_QINT_TYPES = (
    torch.quint8,
    torch.qint8,
    torch.qint32,
)

# Enforced zero point for every quantized data type.
# If None, any zero_point point within the range of the data type is OK.
_ENFORCED_ZERO_POINT = defaultdict(lambda: None, {
    torch.quint8: None,
    torch.qint8: None,
    torch.qint32: 0
})

def _get_valid_min_max(qparams):
    scale, zero_point, quantized_type = qparams
    adjustment = 1 + torch.finfo(torch.float).eps
    _long_type_info = torch.iinfo(torch.long)
    long_min, long_max = _long_type_info.min / adjustment, _long_type_info.max / adjustment
    # make sure intermediate results are within the range of long
    min_value = max((long_min - zero_point) * scale, (long_min / scale + zero_point))
    max_value = min((long_max - zero_point) * scale, (long_max / scale + zero_point))
    return np.float32(min_value), np.float32(max_value)

# This wrapper wraps around `st.floats` and checks the version of `hypothesis`, if
# it is too old, removes the `width` parameter (which was introduced)
# in 3.67.0
def _floats_wrapper(*args, **kwargs):
    if 'width' in kwargs and hypothesis.version.__version_info__ < (3, 67, 0):
        kwargs.pop('width')
    return st.floats(*args, **kwargs)

def floats(*args, **kwargs):
    if 'width' not in kwargs:
        kwargs['width'] = 32
    return _floats_wrapper(*args, **kwargs)

"""Hypothesis filter to avoid overflows with quantized tensors.

Args:
    tensor: Tensor of floats to filter
    qparams: Quantization parameters as returned by the `qparams`.

Returns:
    True

Raises:
    hypothesis.UnsatisfiedAssumption

Note: This filter is slow. Use it only when filtering of the test cases is
      absolutely necessary!
"""
def assume_not_overflowing(tensor, qparams):
    min_value, max_value = _get_valid_min_max(qparams)
    assume(tensor.min() >= min_value)
    assume(tensor.max() <= max_value)
    return True

"""Strategy for generating the quantization parameters.

Args:
    dtypes: quantized data types to sample from.
    scale_min / scale_max: Min and max scales. If None, set to 1e-3 / 1e3.
    zero_point_min / zero_point_max: Min and max for the zero point. If None,
        set to the minimum and maximum of the quantized data type.
        Note: The min and max are only valid if the zero_point is not enforced
              by the data type itself.

Generates:
    scale: Sampled scale.
    zero_point: Sampled zero point.
    quantized_type: Sampled quantized type.
"""
@st.composite
def qparams(draw, dtypes=None, scale_min=None, scale_max=None,
            zero_point_min=None, zero_point_max=None):
    if dtypes is None:
        dtypes = _ALL_QINT_TYPES
    if not isinstance(dtypes, (list, tuple)):
        dtypes = (dtypes,)
    quantized_type = draw(st.sampled_from(dtypes))

    _type_info = torch.iinfo(quantized_type)
    qmin, qmax = _type_info.min, _type_info.max

    # TODO: Maybe embed the enforced zero_point in the `torch.iinfo`.
    _zp_enforced = _ENFORCED_ZERO_POINT[quantized_type]
    if _zp_enforced is not None:
        zero_point = _zp_enforced
    else:
        _zp_min = qmin if zero_point_min is None else zero_point_min
        _zp_max = qmax if zero_point_max is None else zero_point_max
        zero_point = draw(st.integers(min_value=_zp_min, max_value=_zp_max))

    if scale_min is None:
        scale_min = torch.finfo(torch.float).eps
    if scale_max is None:
        scale_max = torch.finfo(torch.float).max
    scale = draw(floats(min_value=scale_min, max_value=scale_max, width=32))

    return scale, zero_point, quantized_type

"""Strategy to create different shapes.
Args:
    min_dims / max_dims: minimum and maximum rank.
    min_side / max_side: minimum and maximum dimensions per rank.

Generates:
    Possible shapes for a tensor, constrained to the rank and dimensionality.

Example:
    # Generates 3D and 4D tensors.
    @given(Q = qtensor(shapes=array_shapes(min_dims=3, max_dims=4))
    some_test(self, Q):...
"""
@st.composite
def array_shapes(draw, min_dims=1, max_dims=None, min_side=1, max_side=None):
    """Return a strategy for array shapes (tuples of int >= 1)."""
    assert(min_dims < 32)
    if max_dims is None:
        max_dims = min(min_dims + 2, 32)
    assert(max_dims < 32)
    if max_side is None:
        max_side = min_side + 5
    return draw(st.lists(
        st.integers(min_side, max_side), min_size=min_dims, max_size=max_dims
    ).map(tuple))


"""Strategy for generating test cases for tensors.
The resulting tensor is in float32 format.

Args:
    shapes: Shapes under test for the tensor. Could be either a hypothesis
            strategy, or an iterable of different shapes to sample from.
    elements: Elements to generate from for the returned data type.
              If None, the strategy resolves to float within range [-1e6, 1e6].
    qparams: Instance of the qparams strategy. This is used to filter the tensor
             such that the overflow would not happen.

Generates:
    X: Tensor of type float32. Note that NaN and +/-inf is not included.
    qparams: (If `qparams` arg is set) Quantization parameters for X.
        The returned parameters are `(scale, zero_point, quantization_type)`.
        (If `qparams` arg is None), returns None.
"""
@st.composite
def tensor(draw, shapes=None, elements=None, qparams=None):
    if isinstance(shapes, SearchStrategy):
        _shape = draw(shapes)
    else:
        _shape = draw(st.sampled_from(shapes))
    if qparams is None:
        if elements is None:
            elements = floats(-1e6, 1e6, allow_nan=False, width=32)
        X = draw(stnp.arrays(dtype=np.float32, elements=elements, shape=_shape))
        assume(not (np.isnan(X).any() or np.isinf(X).any()))
        return X, None
    qparams = draw(qparams)
    if elements is None:
        min_value, max_value = _get_valid_min_max(qparams)
        elements = floats(min_value, max_value, allow_infinity=False,
                          allow_nan=False, width=32)
    X = draw(stnp.arrays(dtype=np.float32, elements=elements, shape=_shape))
    # Recompute the scale and zero_points according to the X statistics.
    scale, zp = _calculate_dynamic_qparams(X, qparams[2])
    enforced_zp = _ENFORCED_ZERO_POINT.get(qparams[2], None)
    if enforced_zp is not None:
        zp = enforced_zp
    return X, (scale, zp, qparams[2])

@st.composite
def per_channel_tensor(draw, shapes=None, elements=None, qparams=None):
    if isinstance(shapes, SearchStrategy):
        _shape = draw(shapes)
    else:
        _shape = draw(st.sampled_from(shapes))
    if qparams is None:
        if elements is None:
            elements = floats(-1e6, 1e6, allow_nan=False, width=32)
        X = draw(stnp.arrays(dtype=np.float32, elements=elements, shape=_shape))
        assume(not (np.isnan(X).any() or np.isinf(X).any()))
        return X, None
    qparams = draw(qparams)
    if elements is None:
        min_value, max_value = _get_valid_min_max(qparams)
        elements = floats(min_value, max_value, allow_infinity=False,
                          allow_nan=False, width=32)
    X = draw(stnp.arrays(dtype=np.float32, elements=elements, shape=_shape))
    # Recompute the scale and zero_points according to the X statistics.
    scale, zp = _calculate_dynamic_per_channel_qparams(X, qparams[2])
    enforced_zp = _ENFORCED_ZERO_POINT.get(qparams[2], None)
    if enforced_zp is not None:
        zp = enforced_zp
    # Permute to model quantization along an axis
    axis = int(np.random.randint(0, X.ndim, 1))
    permute_axes = np.arange(X.ndim)
    permute_axes[0] = axis
    permute_axes[axis] = 0
    X = np.transpose(X, permute_axes)

    return X, (scale, zp, axis, qparams[2])

"""Strategy for generating test cases for tensors used in Conv.
The resulting tensors is in float32 format.

Args:
    spatial_dim: Spatial Dim for feature maps.
    batch_size_range: Range to generate `batch_size`.
                      Must be tuple of `(min, max)`.
    input_channels_per_group_range:
        Range to generate `input_channels_per_group`.
        Must be tuple of `(min, max)`.
    output_channels_per_group_range:
        Range to generate `output_channels_per_group`.
        Must be tuple of `(min, max)`.
    feature_map_range: Range to generate feature map size for each spatial_dim.
                       Must be tuple of `(min, max)`.
    kernel_range: Range to generate kernel size for each spatial_dim. Must be
                  tuple of `(min, max)`.
    max_groups: Maximum number of groups to generate.
    elements: Elements to generate from for the returned data type.
              If None, the strategy resolves to float within range [-1e6, 1e6].
    qparams: Strategy for quantization parameters. for X, w, and b.
             Could be either a single strategy (used for all) or a list of
             three strategies for X, w, b.
Generates:
    (X, W, b, g): Tensors of type `float32` of the following drawen shapes:
        X: (`batch_size, input_channels, H, W`)
        W: (`output_channels, input_channels_per_group) + kernel_shape
        b: `(output_channels,)`
        groups: Number of groups the input is divided into
Note: X, W, b are tuples of (Tensor, qparams), where qparams could be either
      None or (scale, zero_point, quantized_type)


Example:
    @given(tensor_conv(
        spatial_dim=2,
        batch_size_range=(1, 3),
        input_channels_per_group_range=(1, 7),
        output_channels_per_group_range=(1, 7),
        feature_map_range=(6, 12),
        kernel_range=(3, 5),
        max_groups=4,
        elements=st.floats(-1.0, 1.0),
        qparams=qparams()
    ))
"""
@st.composite
def tensor_conv(
    draw, spatial_dim=2, batch_size_range=(1, 4),
    input_channels_per_group_range=(3, 7),
    output_channels_per_group_range=(3, 7), feature_map_range=(6, 12),
    kernel_range=(3, 7), max_groups=1, elements=None, qparams=None
):

    # Resolve the minibatch, in_channels, out_channels, iH/iW, iK/iW
    batch_size = draw(st.integers(*batch_size_range))
    input_channels_per_group = draw(
        st.integers(*input_channels_per_group_range))
    output_channels_per_group = draw(
        st.integers(*output_channels_per_group_range))
    groups = draw(st.integers(1, max_groups))
    input_channels = input_channels_per_group * groups
    output_channels = output_channels_per_group * groups

    feature_map_shape = []
    for i in range(spatial_dim):
        feature_map_shape.append(draw(st.integers(*feature_map_range)))

    kernels = []
    for i in range(spatial_dim):
        kernels.append(draw(st.integers(*kernel_range)))

    # Resolve the tensors
    if qparams is not None:
        if isinstance(qparams, (list, tuple)):
            assert(len(qparams) == 3), "Need 3 qparams for X, w, b"
        else:
            qparams = [qparams] * 3

    X = draw(tensor(shapes=(
        (batch_size, input_channels) + tuple(feature_map_shape),),
        elements=elements, qparams=qparams[0]))
    W = draw(tensor(shapes=(
        (output_channels, input_channels_per_group) + tuple(kernels),),
        elements=elements, qparams=qparams[1]))
    b = draw(tensor(shapes=(output_channels,), elements=elements,
                    qparams=qparams[2]))

    return X, W, b, groups

# We set the deadline in the currently loaded profile.
# Creating (and loading) a separate profile overrides any settings the user
# already specified.
hypothesis_version = hypothesis.version.__version_info__
current_settings = settings._profiles[settings._current_profile].__dict__
current_settings['deadline'] = None
if hypothesis_version >= (3, 16, 0) and hypothesis_version < (5, 0, 0):
    current_settings['timeout'] = hypothesis.unlimited
def assert_deadline_disabled():
    if hypothesis_version < (3, 27, 0):
        import warnings
        warning_message = (
            "Your version of hypothesis is outdated. "
            "To avoid `DeadlineExceeded` errors, please update. "
            "Current hypothesis version: {}".format(hypothesis.__version__)
        )
        warnings.warn(warning_message)
    else:
        assert settings().deadline is None
