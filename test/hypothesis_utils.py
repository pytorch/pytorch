from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import numpy as np
import torch

from hypothesis import assume
from hypothesis import strategies as st
from hypothesis.extra import numpy as stnp
from hypothesis.searchstrategy import SearchStrategy

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

# Map from torch data type to its underlying non-quantized data type.
_UNDERLYING_TYPE = {
    torch.quint8: torch.uint8,
    torch.qint8: torch.int8,
    torch.qint32: torch.int32
}

# Enforced zero point for every quantized data type.
# If None, any zero_point point within the range of the data type is OK.
_ENFORCED_ZERO_POINT = defaultdict(lambda: None, {
    torch.quint8: None,
    torch.qint8: None,
    torch.qint32: 0
})

@st.composite
def qparams(draw, dtypes=None, scale_min=None, scale_max=None,
            zero_point_min=None, zero_point_max=None):
    if dtypes is None:
        dtypes = _ALL_QINT_TYPES
    if not isinstance(dtypes, (list, tuple)):
        dtypes = (dtypes,)
    quantized_type = draw(st.sampled_from(dtypes))
    _underlying_type = _UNDERLYING_TYPE[quantized_type]

    _type_info = torch.iinfo(_underlying_type)
    qmin, qmax = _type_info.min, _type_info.max

    _zp_enforced = _ENFORCED_ZERO_POINT[quantized_type]
    if _zp_enforced is not None:
        zero_point = _zp_enforced
    else:
        _zp_min = qmin if zero_point_min is None else zero_point_min
        _zp_max = qmax if zero_point_max is None else zero_point_max
        zero_point = draw(st.integers(min_value=_zp_min, max_value=_zp_max))

    _long_type_info = torch.iinfo(torch.long)
    if scale_min is None:
        scale_min = 1e-3
    if scale_max is None:
        scale_max = 1e3
    scale = draw(st.floats(min_value=scale_min, max_value=scale_max))

    return (scale, zero_point), (qmin, qmax), quantized_type

"""Strategy to create different shapes.

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
    shapes
Generates:
    (X, w, b, g): Tensors of type `float32` of the following drawen shapes:
        X: (`nbatch, iChannels, H, W`)
        w: (`oChannels, iChannels // groups, kH, kW)
        b: `(oChannels,)`
        g: Number of groups the input is divided into
"""
@st.composite
def tensor(draw, shapes=None, float_min=-1e6, float_max=1e6):
    if isinstance(shapes, SearchStrategy):
        _shape = draw(shapes)
    else:
        _shape = draw(st.sampled_from(shapes))

    _float_type_info = torch.finfo(torch.float)
    if float_min is None:
        float_min = _float_type_info.min + _float_type_info.eps
    if float_max is None:
        float_max = _float_type_info.max - _float_type_info.eps

    X = draw(stnp.arrays(dtype=np.float32,
                         elements=st.floats(min_value=float_min,
                                            max_value=float_max),
                         shape=_shape))
    return X

"""Strategy for generating test cases for tensors used in Conv2D.
The resulting tensors is in float32 format.

Args:
    min_batch, max_batch: Range to generate `nbatch`
    min_in_channels, max_in_channels: Range to generate `iChannels`
    min_out_channels, max_out_channels: Range to generate `oChannels`
    H_range, W_range: Ranges to generate height and width of matrix. Must be
                      tuples of `(min, max)`
    kH_range, kW_range: Ranges to generate kernel height and width. Must be
                        tuples of `(min, max)`
    max_groups: Maximum number of groups to generate
Generates:
    (X, w, b, g): Tensors of type `float32` of the following drawen shapes:
        X: (`nbatch, iChannels, H, W`)
        w: (`oChannels, iChannels // groups, kH, kW)
        b: `(oChannels,)`
        g: Number of groups the input is divided into
"""
@st.composite
def tensor_conv2d(draw,
                  min_batch=1, max_batch=3,
                  min_in_channels=3, max_in_channels=7,
                  min_out_channels=3, max_out_channels=7,
                  H_range=(6, 12), W_range=(6, 12),
                  kH_range=(3, 7), kW_range=(3, 7),
                  max_groups=1, dtypes=None):
    _float_min = -1e6
    _float_max = 1e6
    # Resolve the minibatch, in_channels, out_channels, iH/iW, iK/iW
    _minibatch = draw(st.integers(min_batch, max_batch))
    _in_channels = draw(st.integers(min_in_channels, max_in_channels))
    _out_channels = draw(st.integers(min_out_channels, max_out_channels))
    g = draw(st.integers(1, max_groups))
    assume(_in_channels % g == 0)
    assume(_out_channels % g == 0)

    _iH = draw(st.integers(H_range[0], H_range[1]))
    _iW = draw(st.integers(W_range[0], W_range[1]))
    _kH = draw(st.integers(kH_range[0], kH_range[1]))
    _kW = draw(st.integers(kW_range[0], kW_range[1]))

    # Resolve the tensors
    X = draw(stnp.arrays(dtype=np.float32,
                         elements=st.floats(_float_min, _float_max),
                         shape=(_minibatch, _in_channels, _iH, _iW)))
    w = draw(stnp.arrays(dtype=np.float32,
                         elements=st.floats(_float_min, _float_max),
                         shape=(_out_channels, _in_channels // g, _kH, _kW)))
    b = draw(stnp.arrays(dtype=np.float32,
                         elements=st.floats(_float_min, _float_max),
                         shape=(_out_channels,)))
    return X, w, b, g
