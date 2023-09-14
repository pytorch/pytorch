import torch
import functools
from torch.testing import make_tensor
from functorch.experimental.control_flow import map
from torch.testing._internal.opinfo.core import (
    OpInfo,
    SampleInput,
)
from torch.testing._internal.common_dtype import all_types_and

def sample_inputs_map(opinfo, device, dtype, requires_grad, **kwargs):
    make_arg = functools.partial(
        make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    yield SampleInput([make_arg(2, 2, 2, low=0.1, high=2), make_arg(2, 2, 2, low=0.1, high=2)],
                      args=(make_arg(1, low=0.1, high=2), make_arg(1, low=0.1, high=2)))

def inner_f(x, y0, y1):
    return [x[0].cos().add_(1.) * y0, (x[1] + y1.sin()).cos_().view(x[1].size())]

def simple_map(xs, y0, y1):
    def f(x, y0, y1):
        return inner_f(x, y0, y1)
    return map(f, xs, y0, y1)

def nested_map(xs, y0, y1):
    def f1(xx, y0, y1):
        def f2(x, y0, y1):
            return inner_f(x, y0, y1)
        return map(f2, xx, y0, y1)
    return map(f1, xs, y0, y1)

def triple_nested_map(xs, y0, y1):
    def f0(xs, y0, y1):
        def f1(xx, y0, y1):
            def f2(x, y0, y1):
                return inner_f(x, y0, y1)
            return map(f2, xx, y0, y1)
        return map(f1, xs, y0, y1)
    return map(f0, xs, y0, y1)

control_flow_opinfo_db = [
    OpInfo(
        "MapControlflowOp",
        op=simple_map,
        sample_inputs_func=sample_inputs_map,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
        check_batched_grad=False,
        check_batched_gradgrad=False,
        check_batched_forward_grad=False,
        check_inplace_batched_forward_grad=False,
    ),
    OpInfo(
        "NestedMapControlflowOp",
        op=nested_map,
        sample_inputs_func=sample_inputs_map,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
        check_batched_grad=False,
        check_batched_gradgrad=False,
        check_batched_forward_grad=False,
        check_inplace_batched_forward_grad=False,
    ),
    OpInfo(
        "TripleNestedMapControlflowOp",
        op=triple_nested_map,
        sample_inputs_func=sample_inputs_map,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
        check_batched_grad=False,
        check_batched_gradgrad=False,
        check_batched_forward_grad=False,
        check_inplace_batched_forward_grad=False,
    )
]
