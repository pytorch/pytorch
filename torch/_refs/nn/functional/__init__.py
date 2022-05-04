import torch

import torch._prims.utils as utils
from torch._prims.utils import TensorLikeType, TensorLike, NumberType, Number
import torch._refs as refs

from typing import Optional

all = [
    "elu",
]

# elu is implemented specially because it has an alpha argument
def elu(a: TensorLikeType, alpha: Optional[NumberType] = None, inplace: bool = False):
    """
    Reference implementation of torch.nn.functional.elu
    """

    if inplace:
        raise NotImplementedError

    # Type checks
    assert isinstance(a, TensorLike)
    assert alpha is None or isinstance(alpha, Number)

    computation_dtype, result_dtype = refs._elementwise_dtypes(
        a, type_promotion=refs.ELEMENTWISE_TYPE_PROMOTION_KIND.OP_MATH
    )

    (a,) = refs._convert_dtype(a, dtype=computation_dtype)

    if alpha is not None:
        alpha_promotion_type = utils.dtype_to_type(computation_dtype)
        assert utils.is_lesser_type(type(alpha), alpha_promotion_type)
        alpha = alpha_promotion_type(alpha)

    # Performs computation
    x = refs.maximum(0, a)

    y = refs.sub(refs.exp(a), 1)
    if alpha is not None:
        y = y * alpha
    y = refs.minimum(0, y)

    result = refs.add(x, y)

    (result,) = refs._convert_dtype(result, dtype=result_dtype)
    return result
