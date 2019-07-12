from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

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
ALL_QINT_TYPES = (
    (torch.quint8, None),
    (torch.qint8, None),
    (torch.qint32, 0),  # We enforce zero_point = 0 for this case.
)

"""Strategy for generating test cases for quantized tensors.
The resulting tensor is in float32 format.

Args:
    shapes: A list of shapes to generate.
    dtypes: A list of data types to generate. See note below.
    float_min, float_max: Min and max FP value for the output.
Generates:
    Xhy: Tensor of type `float32` and shape drawn from the `shapes`.
    (scale, zero_point): Drawn from valid ranges derived from the dtypes
    (qmin, qmax): Valid quantization ranges derived from the dtypes.
    quantized_type: Data types for conversion in test.
Note:
    - The `dtypes` argument is used to infer the ranges. The elements should be
    of length 1 or 2:
        If the length is 1 -- The zero_point is not enforced.
        If the length is 2 -- Argument 0 is torch dtype,
            the zero_point is also enforced by argument 1
"""
@st.composite
def qtensor(draw, shapes, dtypes=None, float_min=None, float_max=None):
    # In case shape is a strategy
    if isinstance(shapes, SearchStrategy):
        shape = draw(shapes)
    else:
        shape = draw(st.sampled_from(shapes))
    # Resolve types
    if dtypes is None:
        dtypes = ALL_QINT_TYPES
    _dtypes = draw(st.sampled_from(dtypes))
    assert len(_dtypes) in [1, 2]
    if len(_dtypes) == 1:
        quantized_type = _dtypes[0]
        _zp_enforce = None
    elif len(_dtypes) == 2:
        quantized_type, _zp_enforce = _dtypes[:2]
    _qtype_info = torch.iinfo(quantized_type)
    qmin, qmax = _qtype_info.min, _qtype_info.max
    # Resolve zero_point
    if _zp_enforce is not None:
        zero_point = _zp_enforce
    else:
        zero_point = draw(st.integers(min_value=qmin, max_value=qmax))
    if float_min is None or float_max is None:
        _float_type_info = torch.finfo(torch.float)
        float_min = _float_type_info.min
        float_max = _float_type_info.max
    else:
        assert float_min <= float_max, 'float_min must be <= float_max'
    float_eps = _float_type_info.eps
    # Resolve scale
    scale = draw(st.floats(min_value=float_eps,
                           max_value=float_max))

    adjustment = 1 + float_eps
    _long_type_info = torch.iinfo(torch.long)
    long_min, long_max = _long_type_info.min / adjustment, _long_type_info.max / adjustment
    # make sure intermediate results are within the range of long
    min_value = max((long_min - zero_point) * scale, (long_min / scale + zero_point), float_min)
    max_value = min((long_max - zero_point) * scale, (long_max / scale + zero_point), float_max)
    # Resolve the tensor
    Xhy = draw(stnp.arrays(dtype=np.float32,
                           elements=st.floats(min_value=min_value, max_value=max_value),
                           shape=shape))
    return Xhy, (scale, zero_point), (qmin, qmax), quantized_type

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

"""Strategy for generating test cases for quantized tensors.
The resulting tensor is in float32 format.

Args:
    min_batch, max_batch: Range to generate `nbatch`
    min_in_channels, max_in_channels: Range to generate `iChannels`
    min_out_channels, max_out_channels: Range to generate `oChannels`
    H_range, W_range: Ranges to generate height and width of matrix. Must be
                      tuples of `(min, max)`
    kH_range, kW_range: Ranges to generate kernel height and width. Must be
                        tuples of `(min, max)`
    max_groups: Maximum number of groups to generate
    dtypes: A list of data types to generate. See note below.
Generates:
    (X, w, b, g): Tensors of type `float32` of the following drawen shapes:
        X: (`nbatch, iChannels, H, W`)
        w: (`oChannels, iChannels // groups, kH, kW)
        b: `(oChannels,)`
        g: Number of groups the input is divided into
    (scale, zero_point): Drawn from valid ranges derived from the dtypes
    (qmin, qmax): Valid quantization ranges derived from the dtypes.
    quantized_type: Data types for conversion in test.
Note:
    - The `dtypes` argument is used to infer the ranges. The elements should be
    of length 1 or 2:
        If the length is 1 -- The zero_point is not enforced.
        If the length is 2 -- Argument 0 is torch dtype,
            the zero_point is also enforced by argument 1
"""
@st.composite
def qtensors_conv(draw, min_batch=1, max_batch=3,
                  min_in_channels=3, max_in_channels=7,
                  min_out_channels=3, max_out_channels=7,
                  H_range=(6, 12), W_range=(6, 12),
                  kH_range=(3, 7), kW_range=(3, 7),
                  max_groups=1, dtypes=None):
    _float_type_info = torch.finfo(torch.float)
    float_min = _float_type_info.min
    float_max = _float_type_info.max
    float_eps = _float_type_info.eps
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
                         elements=st.floats(float_min, float_max),
                         shape=(_minibatch, _in_channels, _iH, _iW)))
    w = draw(stnp.arrays(dtype=np.float32,
                         elements=st.floats(float_min, float_max),
                         shape=(_out_channels, _in_channels // g,
                                _kH, _kW)))
    b = draw(stnp.arrays(dtype=np.float32,
                         elements=st.floats(float_min, float_max),
                         shape=(_out_channels,)))

    # Resolve types
    if dtypes is None:
        dtypes = ALL_QINT_TYPES
    _dtypes = draw(st.sampled_from(dtypes))
    assert len(_dtypes) in [1, 2]
    if len(_dtypes) == 1:
        quantized_type = _dtypes[0]
        _zp_enforce = None
    elif len(_dtypes) == 2:
        quantized_type, _zp_enforce = _dtypes[:2]
    _type_info = torch.iinfo(UNDERLYING_TYPE[quantized_type])
    qmin, qmax = _type_info.min, _type_info.max
    # Resolve zero_point
    if _zp_enforce is not None:
        zero_point = _zp_enforce
    else:
        zero_point = draw(st.integers(min_value=qmin, max_value=qmax))
    # Resolve scale
    scale = draw(st.floats(min_value=float_eps,
                           max_value=float_max))
    return ((X, w, b, g), (scale, zero_point), (qmin, qmax),
            quantized_type)
