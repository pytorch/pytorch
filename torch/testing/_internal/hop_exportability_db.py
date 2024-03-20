# mypy: ignore-errors

import functools

import torch
from torch.testing import make_tensor
from torch.testing._internal.common_dtype import all_types_and
from torch.testing._internal.opinfo.core import OpInfo, SampleInput

torch.library.define(
    "testlib::mutating_custom_op",
    "(Tensor(a!) x, Tensor(b!) z) -> (Tensor, Tensor, Tensor)",
    tags=torch.Tag.pt2_compliant_tag,
)


@torch.library.impl("testlib::mutating_custom_op", "cpu")
def foo_impl_cpu(x, z):
    x.add_(5)
    z.add_(5)
    return x, z, x + z


@torch.library.impl_abstract("testlib::mutating_custom_op")
def foo_impl_abstract(x, z):
    return x, z, x + z


def sample_inputs_cond(opinfo, device, dtype, requires_grad, **kwargs):
    make_arg = functools.partial(
        make_tensor, device=device, dtype=dtype, requires_grad=requires_grad
    )
    yield SampleInput((make_arg(2, 2, 2, low=0.1, high=2),))


def simple_cond(x):
    return torch.cond(x.shape[0] > 2, lambda x: x.cos(), lambda x: x.sin(), [x])


def sample_inputs_auto_functionalize(opinfo, device, dtype, requires_grad, **kwargs):
    make_arg = functools.partial(
        make_tensor, device=device, dtype=dtype, requires_grad=False
    )
    yield SampleInput(
        (make_arg(2, 2, 2, low=0.1, high=2), make_arg(2, 2, 2, low=0.1, high=2))
    )


def simple_auto_functionalize(x, z):
    return torch.ops.testlib.mutating_custom_op(x, z)


hop_that_doesnt_need_export_support = [
    "custom_function_call",
    "autograd_function_apply",
    "run_and_save_rng_state",
    "run_with_rng_state",
    "out_dtype",
    "trace_wrapped",
    "map",
    "map_impl",
    "with_effects",
    "strict_mode",
    "_export_tracepoint",
]

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
    ],
    "auto_functionalize": [
        OpInfo(
            "AutoFunctionalize",
            op=simple_auto_functionalize,
            sample_inputs_func=sample_inputs_auto_functionalize,
            dtypes=all_types_and(torch.bool, torch.half),
            supports_out=False,
            check_batched_grad=False,
            check_batched_gradgrad=False,
            check_batched_forward_grad=False,
            check_inplace_batched_forward_grad=False,
        )
    ],
}
