import dataclasses
import weakref
from typing import Any, Callable, List, Tuple, Optional

import sympy
import torch
import torch._dynamo
import torch.fx
from torch._decomp import core_aten_decompositions
from torch._dynamo.eval_frame import Constraint

from torch.fx.experimental.symbolic_shapes import StrictMinMaxConstraint
from torch.utils._sympy.value_ranges import ValueRanges

Value = Any


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
# result = torch._dynamo.export(
#     my_model,
#     *sixtyfour_tensors,
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
    return Constraint(
        weakref.ref(t), id(t), index, StrictMinMaxConstraint(vr=ValueRanges(lower=2, upper=sympy.oo), warn_only=False)
    )


@dataclasses.dataclass
class ExportDynamoConfig:
    """
    Manage Export-specific configurations of Dynamo.
    TODO add tests to make sure the flags are not outdated
    """
    capture_scalar_outputs: bool = True
    capture_dynamic_output_shape_ops: bool = True
    guard_nn_modules: bool = True
    dynamic_shapes: bool = True
    specialize_int: bool = True
    allow_rnn: bool = True


DECOMP_TABLE = core_aten_decompositions()


def _export(
    f: Callable,
    args: Tuple[Value],
    constraints: Optional[List[Constraint]] = None,
) -> torch.fx.GraphModule:
    """
    Private API to export a single entry point or a free function. It is meant to be used
    inside top level torch.export.
    """
    with torch._dynamo.config.patch(dataclasses.asdict(ExportDynamoConfig())):  # type: ignore[attr-defined]
        gm, _ = torch._dynamo.export(
            f,
            *args,
            aten_graph=True,
            tracing_mode="symbolic",
            decomposition_table=DECOMP_TABLE,
            constraints=constraints,
        )
    return gm
