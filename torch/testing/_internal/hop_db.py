# mypy: ignore-errors

import torch
import functools
from torch.testing import make_tensor
import unittest
from functorch.experimental.control_flow import map
from torch.testing._internal.opinfo.core import (
    OpInfo,
    SampleInput,
)
from torch.testing._internal.common_dtype import all_types_and, custom_types
from torch.testing._internal.opinfo.core import DecorateInfo
from torch.nn.attention.flex_attention import flex_attention, _create_empty_block_mask

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


# Please consult with torch.export team before
# adding new entry to this list.
hop_that_doesnt_have_opinfo_test_allowlist = [
    "custom_function_call",
    "autograd_function_apply",
    "run_and_save_rng_state",
    "run_with_rng_state",
    "out_dtype",
    "trace_wrapped",
    "map",  # T183144629
    "map_impl",
    "with_effects",
    "strict_mode",
    "_export_tracepoint",
    "call_torchbind",
    "triton_kernel_wrapper_mutation",
    "triton_kernel_wrapper_functional",
    "hints_wrapper",
]

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


@torch.library.impl("testlib::mutating_custom_op", "cuda")
def foo_impl_cuda(x, z):
    x.add_(5)
    z.add_(5)
    return x, z, x + z


@torch.library.register_fake("testlib::mutating_custom_op")
def foo_impl_abstract(x, z):
    return x, z, x + z


def sample_inputs_cond(opinfo, device, dtype, requires_grad, **kwargs):
    make_arg = functools.partial(
        make_tensor, device=device, dtype=dtype, requires_grad=requires_grad
    )
    yield SampleInput(make_arg(2, 2, 2, low=0.1, high=2))


def simple_cond(x):
    return torch.cond(x.sum() > 2, lambda x: (x.cos(),), lambda x: (x.sin(),), [x])


def sample_inputs_auto_functionalize(opinfo, device, dtype, requires_grad, **kwargs):
    make_arg = functools.partial(
        make_tensor, device=device, dtype=dtype, requires_grad=False
    )
    yield SampleInput(make_arg(2, 2, 2, low=0.1, high=2), make_arg(2, 2, 2, low=0.1, high=2))


def simple_auto_functionalize(x, z):
    return torch.ops.testlib.mutating_custom_op(x, z)


def sample_inputs_flex_attention(opinfo, device, dtype, requires_grad, **kwargs):
    make_arg = functools.partial(
        make_tensor, device=device, dtype=dtype, requires_grad=requires_grad
    )

    def score_mod(score, b, h, m, n):
        return score + h

    q, k, v = (make_arg(2, 2, 128, 8, low=0.1, high=2) for _ in range(3))
    block_mask = _create_empty_block_mask(q, k)
    yield SampleInput(
        q,
        k,
        v,
        score_mod,
        block_mask
    )

def sample_inputs_while_loop(opinfo, device, dtype, requires_grad, **kwargs):
    make_arg = functools.partial(
        make_tensor, device=device, dtype=dtype, requires_grad=False
    )
    yield SampleInput(
        torch.tensor(3),
        make_arg(2, 3, 4, low=0.1, high=2),
    )

def simple_while_loop(iter_t, x):
    def cond_fn(iter_t, x):
        return iter_t > 0

    def body_fn(iter_t, x):
        return iter_t - 1, x.cos()

    return torch._higher_order_ops.while_loop(cond_fn, body_fn, (iter_t, x))


hop_db = [
    OpInfo(
        name="map",
        variant_test_name="simple",
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
        name="map",
        variant_test_name="nested",
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
        name="map",
        variant_test_name="triple_nested",
        op=triple_nested_map,
        sample_inputs_func=sample_inputs_map,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
        check_batched_grad=False,
        check_batched_gradgrad=False,
        check_batched_forward_grad=False,
        check_inplace_batched_forward_grad=False,
    ),
    OpInfo(
        name="cond",
        variant_test_name="simple",
        op=simple_cond,
        sample_inputs_func=sample_inputs_cond,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
        check_batched_grad=False,
        check_batched_gradgrad=False,
        check_batched_forward_grad=False,
        check_inplace_batched_forward_grad=False,
        supports_autograd=True,
        # "torch.compile with aot_autograd does not currently support double backward."
        supports_gradgrad=False,
    ),
    OpInfo(
        name="while_loop",
        variant_test_name="simple",
        op=simple_while_loop,
        sample_inputs_func=sample_inputs_while_loop,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
        check_batched_grad=False,
        check_batched_gradgrad=False,
        check_batched_forward_grad=False,
        check_inplace_batched_forward_grad=False,
        supports_autograd=False,
    ),
    OpInfo(
        name="auto_functionalize",
        variant_test_name="simple",
        op=simple_auto_functionalize,
        sample_inputs_func=sample_inputs_auto_functionalize,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
        check_batched_grad=False,
        check_batched_gradgrad=False,
        check_batched_forward_grad=False,
        check_inplace_batched_forward_grad=False,
        supports_autograd=False,
    ),
    OpInfo(
        name="flex_attention",
        variant_test_name="simple",
        op=flex_attention,
        sample_inputs_func=sample_inputs_flex_attention,
        dtypes=custom_types(torch.float16, torch.float32),
        supports_out=False,
        check_batched_grad=False,
        check_batched_gradgrad=False,
        check_batched_forward_grad=False,
        check_inplace_batched_forward_grad=False,
        skips=(
            DecorateInfo(unittest.expectedFailure, "TestHOP", "test_aot_export"),
            DecorateInfo(unittest.expectedFailure, "TestHOP", "test_pre_dispatch_export"),
            DecorateInfo(unittest.expectedFailure, "TestHOP", "test_serialize_export"),
            DecorateInfo(unittest.expectedFailure, "TestHOP", "test_retrace_export"),
        ),
    ),
    OpInfo(
        name="flex_attention_backward",
        variant_test_name="simple",
        op=flex_attention,
        sample_inputs_func=sample_inputs_flex_attention,
        dtypes=custom_types(torch.float16, torch.float32),
        supports_out=False,
        check_batched_grad=False,
        check_batched_gradgrad=False,
        check_batched_forward_grad=False,
        check_inplace_batched_forward_grad=False,
        skips=(
            DecorateInfo(unittest.expectedFailure, "TestHOP", "test_aot_export"),
            DecorateInfo(unittest.expectedFailure, "TestHOP", "test_pre_dispatch_export"),
            DecorateInfo(unittest.expectedFailure, "TestHOP", "test_serialize_export"),
            DecorateInfo(unittest.expectedFailure, "TestHOP", "test_retrace_export"),
        ),
    )
]
