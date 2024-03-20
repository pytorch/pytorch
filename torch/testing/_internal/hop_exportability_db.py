# mypy: ignore-errors

import torch
import functools
from torch.testing import make_tensor
from functorch.experimental.control_flow import map
from torch.testing._internal.opinfo.core import (
    OpInfo,
    SampleInput,
)
from torch.testing._internal.common_dtype import all_types_and

def sample_inputs_cond(opinfo, device, dtype, requires_grad, **kwargs):
    make_arg = functools.partial(
        make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    yield SampleInput(make_arg(2, 2, 2, low=0.1, high=2))


def simple_cond(x):
    return torch.cond(x.shape[0] > 2, lambda x: x.cos(), lambda x: x.sin(), [x])

hop_export_opinfo_db = {
    "cond": [
        OpInfo(
            "CondSingleLevel",
            op=simple_cond,
            sample_inputs_func=sample_inputs_cond,
            dtypes=all_types_and(torch.bool, torch.half),
            supports_out=False,
            check_batched_grad=False,
            check_batched_gradgrad=False,
            check_batched_forward_grad=False,
            check_inplace_batched_forward_grad=False,
        )
    ]
}
