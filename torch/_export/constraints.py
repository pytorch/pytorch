from typing import Optional, Callable, Union
import sympy
import weakref

import torch
from torch import SymInt, SymFloat
from torch._dynamo import allow_in_graph
from torch.fx.experimental.symbolic_shapes import constrain_range_int
from torch.utils._sympy.value_ranges import ValueRangeError, ValueRanges

from torch.fx.experimental.symbolic_shapes import StrictMinMaxConstraint
from torch._dynamo.exc import UserError, UserErrorType
from torch._constraint import Constraint

# Note - [On Export Dynamic Dimension UX]
#
# After a lot of discussion, we have settled on a dynamic marking API
# for export that meets the following constraints:
# 1) Stateless
# 2) Safe for numerous .export calls within a single process
# 3) Simple to use
# 4) Can be extended to constraints easily
#
# While the underlying API is still torch._dynamo.mark_dynamic, we offer a higher
# level API that meets the constraints above.
#
# This API produces an object that is meant to be passed into torch._dynamo.export
# constraints field. See docs on torch._dynamo.export for more details.
#
# Note - The output type and structure here is NOT BC and NOT A CONTRACT, we reserve
# the right to change the output here at any time, and will do so as we extend the API.
#
# result = torch.export(
#     my_model,
#     sixtyfour_tensors,
#     constraints=[
#         # if you do only dynamic_dim, this is sugar for
#         # -Inf <= dynamic_dim(blah, 0) <= Inf; we don’t otherwise
#         # permit direct int->bool conversion
#         dynamic_dim(blah, 0),
#         # operator overloading because it makes it clear whether
#         # or not you’re inclusive-exclusive range or not
#         0 <= dynamic_dim(blah, 1) <= 100,
#         # NB: But we actually truncate ranges to be >= 2, because of
#         # 0/1 specialization
#     ]
# )
def dynamic_dim(t: torch.Tensor, index: int):
    if not isinstance(t, torch.Tensor):
        raise UserError(
            UserErrorType.DYNAMIC_DIM,
            f"Expected tensor as input to dynamic_dim but got {type(t)}"
        )

    if t.dim() < 1:
        raise UserError(
            UserErrorType.DYNAMIC_DIM,
            "Cannot mark 0-dimension tensors to be dynamic"
        )

    if index >= t.dim():
        raise UserError(
            UserErrorType.DYNAMIC_DIM,
            f"Expected the dimension passed to dynamic_dim to be in the range [0:{t.dim()-1}]"
            f" but got {index}, which is out of bounds for the given tensor."
        )

    return Constraint(
        weakref.ref(t),
        id(t),
        index,
        StrictMinMaxConstraint(
            vr=ValueRanges(lower=2, upper=sympy.oo), warn_only=False
        ),
    )



# `Scalar` type used in native_functions.ymal will be translated to `Union[Number, _complex]`
# could cause type error during since `SymInt` or `SymFloat` will be used.
# Here manually specify the type explicitly.
sym_constrain_range: Callable[
    [Union[int, float, SymInt, SymFloat], Optional[int], Optional[int]],
    None,
] = torch.sym_constrain_range  # type: ignore[assignment]


# TODO: we want to hide this min/max stuff under some abstraction similar to
# DynamicDim
@allow_in_graph
def constrain_as_value(symbol, min: Optional[int] = None, max: Optional[int] = None):
    """
    Add min/max constraint on the intermediate symbol at tracing time
    """

    if not isinstance(symbol, SymInt):
        constrain_range_int(symbol, min=min, max=max)
    else:
        sym_constrain_range(symbol, min, max)

    return symbol


# TODO: we want to hide this min/max stuff under some abstraction similar to
# DynamicDim
@allow_in_graph
def constrain_as_size(symbol, min: int = 2, max: Optional[int] = None):
    """
    Add min/max constraint on the intermediate symbol which will be used as a size
    """

    # TODO: we should investigate turning off 0/1 specialization for unbacked
    # SymInts
    if min < 2:
        raise ValueRangeError(
            "Unable to set min size to be <= 2 because we specialize on 0/1 sizes."
        )
    return constrain_as_value(symbol, min, max)
