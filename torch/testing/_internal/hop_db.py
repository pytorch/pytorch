# mypy: ignore-errors

import functools
import unittest

import torch
from functorch.experimental.control_flow import map
from torch.nn.attention.flex_attention import _create_empty_block_mask, flex_attention
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import onlyCUDA
from torch.testing._internal.common_dtype import all_types_and, custom_types
from torch.testing._internal.opinfo.core import DecorateInfo, OpInfo, SampleInput
from torch._higher_order_ops.invoke_subgraph import mark_compile_region
from torch._higher_order_ops import InvokeQuant, invoke_quant_packed


def sample_inputs_map(opinfo, device, dtype, requires_grad, **kwargs):
    make_arg = functools.partial(
        make_tensor, device=device, dtype=dtype, requires_grad=requires_grad
    )
    yield SampleInput(
        [make_arg(2, 2, 2, low=0.1, high=2), make_arg(2, 2, 2, low=0.1, high=2)],
        args=(make_arg(1, low=0.1, high=2), make_arg(1, low=0.1, high=2)),
    )


def inner_f(x, y0, y1):
    return [x[0].cos().add_(1.0) * y0, (x[1] + y1.sin()).cos_().view(x[1].size())]


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


# PLEASE DON'T ADD ANYTHING NEW TO THIS LIST,
# and do add an OpInfo for your HOP.
# The OpInfo lets us do automated testing for the HOP to check that
# your HOP will work correctly with PyTorch!
#
# Your new HOP may fail some automated testing. That's OK. If you don't
# care about certain features (like torch.export), it's fine to xfail those
# failing tests. It is less fine to xfail a more critical check (like checking
# if torch.compile works with your HOP, or if your HOP has a docstring).
# If you don't know if a test is fine to xfail, please ask.
#
# There are legitimate reasons why something cannot be added to this list
# (e.g. it uses executorch which is not in PyTorch). If that's the case then
# please leave a comment.
FIXME_hop_that_doesnt_have_opinfo_test_allowlist = [
    "custom_function_call",
    "autograd_function_apply",
    "run_and_save_rng_state",
    "run_with_rng_state",
    "graphsafe_run_with_rng_state",
    "out_dtype",
    "trace_wrapped",
    'tag_activation_checkpoint',
    'executorch_call_delegate',
    'wrap',
    'wrap_with_set_grad_enabled',
    'auto_functionalized_v2',
    'associative_scan',
    'flat_apply',  # is WIP, doesn't pass any of the tests yet
    'wrap_with_autocast',
    'wrap_activation_checkpoint',
    'run_const_graph',
    'auto_functionalized',
    "map",  # T183144629
    "map_impl",
    "with_effects",
    "strict_mode",
    "_export_tracepoint",
    "call_torchbind",
    "triton_kernel_wrapper_mutation",
    "triton_kernel_wrapper_functional",
    "hints_wrapper",
    "dynamo_bypassing_wrapper",  # TODO(soulitzer)
    "foreach_map",
    "aoti_call_delegate",
    "print",
    "inductor_compiled_code",  # Tested separately in test_inductor_wrap_inductor_compile_regions
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
    return x.clone(), z.clone(), x + z


@torch.library.impl("testlib::mutating_custom_op", "cuda")
def foo_impl_cuda(x, z):
    x.add_(5)
    z.add_(5)
    return x.clone(), z.clone(), x + z


@torch.library.impl("testlib::mutating_custom_op", "xpu")
def foo_impl_xpu(x, z):
    x.add_(5)
    z.add_(5)
    return x.clone(), z.clone(), x + z


@torch.library.register_fake("testlib::mutating_custom_op")
def foo_impl_abstract(x, z):
    return x.clone(), z.clone(), x + z


def sample_inputs_cond(opinfo, device, dtype, requires_grad, **kwargs):
    make_arg = functools.partial(
        make_tensor, device=device, dtype=dtype, requires_grad=requires_grad
    )
    yield SampleInput(make_arg(2, 2, 2, low=0.1, high=2))


def simple_cond(x):
    return torch.cond(x.sum() > 2, lambda x: (x.cos(),), lambda x: (x.sin(),), [x])


def sample_inputs_invoke_subgraph(opinfo, device, dtype, requires_grad, **kwargs):
    make_arg = functools.partial(
        make_tensor, device=device, dtype=dtype, requires_grad=requires_grad
    )
    yield SampleInput(make_arg(2, 2, 2, low=0.1, high=2))


@mark_compile_region
def fn_for_invoke_subgraph(x):
    return torch.sin(x)


def simple_invoke_subgraph(x):
    return fn_for_invoke_subgraph(x)


def sample_inputs_auto_functionalize(opinfo, device, dtype, requires_grad, **kwargs):
    make_arg = functools.partial(
        make_tensor, device=device, dtype=dtype, requires_grad=False
    )
    yield SampleInput(
        make_arg(2, 2, 2, low=0.1, high=2), make_arg(2, 2, 2, low=0.1, high=2)
    )


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
    yield SampleInput(q, k, v, score_mod, block_mask)


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


def simple_while_loop_stack_output(iter_t, x):
    def cond_fn(iter_t, x):
        return iter_t > 0

    def body_fn(iter_t, x):
        return iter_t - 1, x.cos()

    return torch._higher_order_ops.while_loop_stack_output(
        cond_fn, body_fn, (iter_t, x), tuple()
    )


def sample_inputs_local_map_hop(opinfo, device, dtype, requires_grad, **kwargs):
    # TODO: once HOPs support DTensor inputs, we should also test DTensors
    make_arg = functools.partial(
        make_tensor, device=device, dtype=dtype, requires_grad=False
    )
    yield SampleInput(
        make_arg(2, 3, 4, low=0.1, high=2),
        make_arg(2, 3, 4, low=0.1, high=2),
    )


def simple_local_map_hop(inp1, inp2):
    def body_gm(inp1, inp2):
        return inp1.cos() + inp2.sin()

    gm = torch.fx.symbolic_trace(body_gm)

    assert torch.distributed.is_available()
    from torch.distributed.tensor.placement_types import Replicate

    gm.meta["local_map_kwargs"] = {
        "in_placements": (Replicate(), Replicate(), Replicate()),
        "out_placements": ((Replicate(), Replicate(), Replicate()),),
    }

    # TODO: Dynamo would rewrite this op differently
    return torch._higher_order_ops.local_map_hop(gm, inp1, inp2)


def sample_inputs_scan(opinfo, device, dtype, requires_grad, **kwargs):
    make_arg = functools.partial(
        make_tensor, device=device, dtype=dtype, requires_grad=requires_grad
    )
    yield SampleInput(
        make_arg(2, 2, low=0.1, high=2),
        make_arg(2, 2, 2, low=0.1, high=2),
    )


def simple_scan(init, xs):
    def combine_fn(carry, x):
        result = carry @ x + x
        return result, carry.clone()

    return torch._higher_order_ops.scan(combine_fn, init, xs)


quant_tracer = InvokeQuant()


def simple_invoke_quant(x):
    def fn(x, y):
        return (torch.sin(x) * y,)

    return quant_tracer(fn, x, x)[0] * 2.0


def simple_invoke_quant_packed(x):
    def fn(x):
        return (torch.sin(x),)

    return invoke_quant_packed(fn, x)[0] * 2.0


hop_db = [
    OpInfo(
        name="scan",
        variant_test_name="simple",
        op=simple_scan,
        sample_inputs_func=sample_inputs_scan,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
        check_batched_grad=False,
        check_batched_gradgrad=False,
        check_batched_forward_grad=False,
        check_inplace_batched_forward_grad=False,
        supports_autograd=False,
        # "torch.compile with aot_autograd does not currently support double backward."
        supports_gradgrad=False,
    ),
    OpInfo(
        name="invoke_subgraph",
        variant_test_name="simple",
        op=simple_invoke_subgraph,
        sample_inputs_func=sample_inputs_invoke_subgraph,
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
        name="invoke_quant",
        variant_test_name="simple",
        op=simple_invoke_quant,
        sample_inputs_func=sample_inputs_invoke_subgraph,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
        check_batched_grad=False,
        check_batched_gradgrad=False,
        check_batched_forward_grad=False,
        check_inplace_batched_forward_grad=False,
        supports_autograd=True,
        # "torch.compile with aot_autograd does not currently support double backward."
        skips=(
            DecorateInfo(unittest.expectedFailure, "TestHOP", "test_aot_export"),
            DecorateInfo(
                unittest.expectedFailure, "TestHOP", "test_pre_dispatch_export"
            ),
            DecorateInfo(unittest.expectedFailure, "TestHOP", "test_serialize_export"),
            DecorateInfo(unittest.expectedFailure, "TestHOP", "test_retrace_export"),
        ),
        # "torch.compile with aot_autograd does not currently support double backward."
        supports_gradgrad=False,
    ),
    OpInfo(
        name="invoke_quant_packed",
        variant_test_name="simple",
        op=simple_invoke_quant_packed,
        sample_inputs_func=sample_inputs_invoke_subgraph,
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
        name="while_loop_stack_output",
        variant_test_name="simple",
        op=simple_while_loop_stack_output,
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
            DecorateInfo(
                unittest.expectedFailure, "TestHOP", "test_pre_dispatch_export"
            ),
            DecorateInfo(unittest.expectedFailure, "TestHOP", "test_serialize_export"),
            DecorateInfo(unittest.expectedFailure, "TestHOP", "test_retrace_export"),
        ),
        decorators=[onlyCUDA],
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
            DecorateInfo(
                unittest.expectedFailure, "TestHOP", "test_pre_dispatch_export"
            ),
            DecorateInfo(unittest.expectedFailure, "TestHOP", "test_serialize_export"),
            DecorateInfo(unittest.expectedFailure, "TestHOP", "test_retrace_export"),
        ),
        decorators=[onlyCUDA],
    ),
    OpInfo(
        name="local_map_hop",
        variant_test_name="simple",
        op=simple_local_map_hop,
        sample_inputs_func=sample_inputs_local_map_hop,
        dtypes=custom_types(torch.float16, torch.float32),
        supports_out=False,
        check_batched_grad=False,
        check_batched_gradgrad=False,
        check_batched_forward_grad=False,
        check_inplace_batched_forward_grad=False,
        skips=(
            DecorateInfo(unittest.expectedFailure, "TestHOP", "test_aot_export"),
            DecorateInfo(
                unittest.expectedFailure, "TestHOP", "test_pre_dispatch_export"
            ),
            DecorateInfo(unittest.expectedFailure, "TestHOP", "test_serialize_export"),
            DecorateInfo(unittest.expectedFailure, "TestHOP", "test_retrace_export"),
        ),
        decorators=[
            onlyCUDA,
            unittest.skipIf(
                not torch.distributed.is_available(), "requires distributed build"
            ),
        ],
    ),
]
