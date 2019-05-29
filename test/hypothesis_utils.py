from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch

from hypothesis import strategies as st
from hypothesis.extra import numpy as stnp

# Setup for the hypothesis tests.
# The tuples are (torch_type, np_type, zero_point_enforce), where the last
# element is enforced zero_point. If None, any zero_point point within the
# range of the data type is OK.
ALL_QINT_AND_NP_TYPES = (
    (torch.quint8, np.uint8, None),
    (torch.qint8, np.int8, None),
    (torch.qint32, np.int32, 0),  # We enforce zero_point = 0 for this case.
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
    (torch_type, np_type): Data types (torch and numpy) for conversion in test.
Note:
    The `dtypes` argument is used to infer the ranges. The elements should be
    of length 1, 2, or 3:
        If the length is 1 -- the torch_type is assumed to be the same as
            np_type. The zero_point is not enforced.
        If the length is 2 -- the torch_type and np_type are set, while the
            zero_point is not enforced.
        If the length is 3 -- zero_point is forced to be the fixed at element 3
            of the tuple.
"""
@st.composite
def qtensor(draw, shapes, dtypes=None, float_min=-1e6, float_max=1e6):
    # Resolve types
    if dtypes is None:
        dtypes = ALL_QINT_AND_NP_TYPES
    _dtypes = draw(st.sampled_from(dtypes))
    if len(_dtypes) == 1:
        torch_type = np_type = _dtypes[0]
        _zp_enforce = None
    elif len(_dtypes) == 2:
        torch_type, np_type = _dtypes
        _zp_enforce = None
    else:
        torch_type, np_type, _zp_enforce = _dtypes[:3]
    _type_info = np.iinfo(np_type)
    qmin, qmax = _type_info.min, _type_info.max
    # Resolve zero_point
    if _zp_enforce is not None:
        zero_point = _zp_enforce
    else:
        zero_point = draw(st.integers(min_value=qmin, max_value=qmax))
    # Resolve scale
    scale = draw(st.floats(min_value=np.finfo(np.float32).resolution,
                           max_value=(np.finfo(np.float32).max)))
    # Resolve the tensor
    Xhy = draw(stnp.arrays(dtype=np.float32,
                           elements=st.floats(float_min, float_max),
                           shape=draw(st.sampled_from(shapes))))
    return Xhy, (scale, zero_point), (qmin, qmax), (torch_type, np_type)
