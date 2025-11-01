# Owner(s): ["module: higher order operators"]
# flake8: noqa: B950


import contextlib
import functools
import unittest

import torch
import torch._dynamo
import torch._functorch
import torch._inductor
import torch._inductor.decomposition
import torch.nn.functional as F
from torch import nn
from torch._dynamo.variables.higher_order_ops import LocalMapWrappedHigherOrderVariable
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.utils.checkpoint import create_selective_checkpoint_contexts


if torch.distributed.is_available():
    from torch.distributed._tensor.experimental import local_map
    from torch.distributed.tensor.placement_types import Replicate, Shard

from torch.testing._internal.common_utils import (
    run_tests,
    TEST_WITH_CROSSREF,
    TEST_WITH_TORCHDYNAMO,
    TEST_WITH_TORCHINDUCTOR,
    TestCase,
)


nested_compile_region = torch.compiler.nested_compile_region


def get_skip_reasons():
    msg = ""
    if not torch.distributed.is_available():
        msg += "Torch distributed not available. "
    if TEST_WITH_TORCHINDUCTOR or TEST_WITH_TORCHDYNAMO:
        msg += "Already manually torch.compile'd. "

    return msg != "", msg


class MyTransform(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x + 100

    @staticmethod
    def backward(ctx, grad):
        return grad + 100


def context_parallel_attention(query, key, value):
    out = F.scaled_dot_product_attention(
        query=query, key=key, value=value, is_causal=False
    )
    return out


# NOTE: we use this function directly in the node checks
def save_scalar_muls(ctx, op, *args, **kwargs):
    if op == torch.ops.aten.mul.Scalar:
        return torch.utils.checkpoint.CheckpointPolicy.MUST_SAVE
    return torch.utils.checkpoint.CheckpointPolicy.MUST_RECOMPUTE


def create_model(attention_fn, nheads, dim1, dim2, sac_policy=None):
    class LocalMapTransformerBlock(nn.Module):
        def __init__(self, nheads, dim1, dim2):
            super().__init__()
            self.nheads = nheads
            bias = False
            self.wq = nn.Linear(dim1, dim1, bias=bias)
            self.wk = nn.Linear(dim1, dim1, bias=bias)
            self.wv = nn.Linear(dim1, dim1, bias=bias)
            self.wo = nn.Linear(dim1, dim1, bias=bias)
            self.w1 = nn.Linear(dim1, dim2, bias=bias)
            self.w2 = nn.Linear(dim2, dim1, bias=bias)
            if sac_policy:
                self.sac_context_fn = functools.partial(
                    create_selective_checkpoint_contexts, sac_policy
                )
            else:
                self.sac_context_fn = None

        def _forward(self, x):
            q = self.wq(x)
            k = self.wk(x)
            v = self.wv(x)

            q = q.unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)
            k = k.unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)
            v = v.unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)

            o = attention_fn(q, k, v)
            o = o.permute(0, 2, 1, 3).flatten(-2)

            o = self.wo(o)

            o0 = o + x

            o = self.w1(o0)
            o = torch.nn.functional.relu(o)
            o = self.w2(o)

            o = o0 + o
            return o

        def forward(self, x):
            if self.sac_context_fn is not None:
                return torch.utils.checkpoint.checkpoint(
                    self._forward,
                    x,
                    use_reentrant=False,
                    context_fn=self.sac_context_fn,
                )
            return self._forward(x)

    return LocalMapTransformerBlock(nheads, dim1, dim2)


def get_local_mapped_functions():
    assert torch.distributed.is_available()

    @local_map(
        out_placements=((Shard(0), Shard(1), Shard(2)),),
        in_placements=(
            (Shard(0), Shard(1), Shard(2)),  # query
            (Shard(0), Shard(1), Replicate()),  # key
            (Shard(0), Shard(1), Replicate()),  # value
        ),
        redistribute_inputs=True,
        in_grad_placements=None,
        device_mesh=None,
    )
    def cp_decorated(query, key, value):
        return context_parallel_attention(query, key, value)

    cp_function = local_map(
        context_parallel_attention,
        out_placements=(Shard(0), Shard(1), Shard(2)),
        in_placements=(
            (Shard(0), Shard(1), Shard(2)),  # query
            (Shard(0), Shard(1), Replicate()),  # key
            (Shard(0), Shard(1), Replicate()),  # value
        ),
        redistribute_inputs=True,
        in_grad_placements=None,
        device_mesh=None,
    )

    return cp_decorated, cp_function


class TestLocalMap(TestCase):
    def setUp(self):
        self.exit_stack = contextlib.ExitStack()
        self.exit_stack.enter_context(sdpa_kernel(backends=[SDPBackend.MATH]))

    def tearDown(self):
        self.exit_stack.close()

    @unittest.skipIf(*get_skip_reasons())
    def test_simple(self):
        cp_decorated, cp_function = get_local_mapped_functions()
        bs = 8 * 1
        dim1 = 96
        dim2 = dim1 * 4
        nheads = 16
        seq_len = 16

        from torch._dynamo.testing import EagerAndRecordGraphs, normalize_gm

        backend = EagerAndRecordGraphs()

        model = create_model(cp_decorated, nheads, dim1, dim2)
        inputs = (torch.randn(bs, seq_len, dim1, requires_grad=True),)
        with LocalMapWrappedHigherOrderVariable.enable():
            out = torch.compile(model, backend=backend)(*inputs)
        out.sum().backward()

        model = create_model(cp_function, nheads, dim1, dim2)
        inputs = (torch.randn(bs, seq_len, dim1, requires_grad=True),)
        with LocalMapWrappedHigherOrderVariable.enable():
            out = torch.compile(model, backend=backend)(*inputs)
        out.sum().backward()

        if not TEST_WITH_CROSSREF:
            self.assertEqual(len(backend.graphs), 2)
            self.assertEqual(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                normalize_gm(backend.graphs[1].print_readable(print_output=False)),
            )
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_self_modules_wq_parameters_weight_: "f32[96, 96]", L_x_: "f32[8, 16, 96]", L_self_modules_wk_parameters_weight_: "f32[96, 96]", L_self_modules_wv_parameters_weight_: "f32[96, 96]", L_self_modules_wo_parameters_weight_: "f32[96, 96]", L_self_modules_w1_parameters_weight_: "f32[384, 96]", L_self_modules_w2_parameters_weight_: "f32[96, 384]"):
        l_self_modules_wq_parameters_weight_ = L_self_modules_wq_parameters_weight_
        l_x_ = L_x_
        l_self_modules_wk_parameters_weight_ = L_self_modules_wk_parameters_weight_
        l_self_modules_wv_parameters_weight_ = L_self_modules_wv_parameters_weight_
        l_self_modules_wo_parameters_weight_ = L_self_modules_wo_parameters_weight_
        l_self_modules_w1_parameters_weight_ = L_self_modules_w1_parameters_weight_
        l_self_modules_w2_parameters_weight_ = L_self_modules_w2_parameters_weight_

        q: "f32[8, 16, 96]" = torch._C._nn.linear(l_x_, l_self_modules_wq_parameters_weight_, None);  l_self_modules_wq_parameters_weight_ = None

        k: "f32[8, 16, 96]" = torch._C._nn.linear(l_x_, l_self_modules_wk_parameters_weight_, None);  l_self_modules_wk_parameters_weight_ = None

        v: "f32[8, 16, 96]" = torch._C._nn.linear(l_x_, l_self_modules_wv_parameters_weight_, None);  l_self_modules_wv_parameters_weight_ = None

        unflatten: "f32[8, 16, 16, 6]" = q.unflatten(-1, (16, -1));  q = None
        q_1: "f32[8, 16, 16, 6]" = unflatten.permute(0, 2, 1, 3);  unflatten = None

        unflatten_1: "f32[8, 16, 16, 6]" = k.unflatten(-1, (16, -1));  k = None
        k_1: "f32[8, 16, 16, 6]" = unflatten_1.permute(0, 2, 1, 3);  unflatten_1 = None

        unflatten_2: "f32[8, 16, 16, 6]" = v.unflatten(-1, (16, -1));  v = None
        v_1: "f32[8, 16, 16, 6]" = unflatten_2.permute(0, 2, 1, 3);  unflatten_2 = None

        subgraph_0 = self.subgraph_0
        local_map_hop = torch.ops.higher_order.local_map_hop(subgraph_0, q_1, k_1, v_1);  subgraph_0 = q_1 = k_1 = v_1 = None
        o: "f32[8, 16, 16, 6]" = local_map_hop[0];  local_map_hop = None

        permute_3: "f32[8, 16, 16, 6]" = o.permute(0, 2, 1, 3);  o = None
        o_1: "f32[8, 16, 96]" = permute_3.flatten(-2);  permute_3 = None

        o_2: "f32[8, 16, 96]" = torch._C._nn.linear(o_1, l_self_modules_wo_parameters_weight_, None);  o_1 = l_self_modules_wo_parameters_weight_ = None

        o0: "f32[8, 16, 96]" = o_2 + l_x_;  o_2 = l_x_ = None

        o_3: "f32[8, 16, 384]" = torch._C._nn.linear(o0, l_self_modules_w1_parameters_weight_, None);  l_self_modules_w1_parameters_weight_ = None

        o_4: "f32[8, 16, 384]" = torch.nn.functional.relu(o_3);  o_3 = None

        o_5: "f32[8, 16, 96]" = torch._C._nn.linear(o_4, l_self_modules_w2_parameters_weight_, None);  o_4 = l_self_modules_w2_parameters_weight_ = None

        o_6: "f32[8, 16, 96]" = o0 + o_5;  o0 = o_5 = None
        return (o_6,)

    class subgraph_0(torch.nn.Module):
        def forward(self, q_1: "f32[8, 16, 16, 6]", k_1: "f32[8, 16, 16, 6]", v_1: "f32[8, 16, 16, 6]"):
            out: "f32[8, 16, 16, 6]" = torch._C._nn.scaled_dot_product_attention(query = q_1, key = k_1, value = v_1, is_causal = False);  q_1 = k_1 = v_1 = None
            return (out,)
""",
            )

    @unittest.skipIf(*get_skip_reasons())
    def test_sac(self):
        cp_decorated, cp_function = get_local_mapped_functions()
        bs = 8 * 1
        dim1 = 96
        dim2 = dim1 * 4
        nheads = 16
        seq_len = 16

        from torch._dynamo.testing import AotEagerAndRecordGraphs, normalize_gm

        backend = AotEagerAndRecordGraphs()

        model = create_model(
            cp_decorated, nheads, dim1, dim2, sac_policy=save_scalar_muls
        )
        inputs = (torch.randn(bs, seq_len, dim1, requires_grad=True),)
        with LocalMapWrappedHigherOrderVariable.enable():
            out = torch.compile(model, backend=backend)(*inputs)
        out.sum().backward()

        model = create_model(
            cp_function, nheads, dim1, dim2, sac_policy=save_scalar_muls
        )
        inputs = (torch.randn(bs, seq_len, dim1, requires_grad=True),)
        with LocalMapWrappedHigherOrderVariable.enable():
            out = torch.compile(model, backend=backend)(*inputs)
        out.sum().backward()

        if not TEST_WITH_CROSSREF:
            self.assertEqual(len(backend.graphs), 2)
            self.assertEqual(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                normalize_gm(backend.graphs[1].print_readable(print_output=False)),
            )
            self.assertEqual(
                normalize_gm(backend.fw_graphs[0].print_readable(print_output=False)),
                normalize_gm(backend.fw_graphs[1].print_readable(print_output=False)),
            )
            self.assertEqual(
                normalize_gm(backend.bw_graphs[0].print_readable(print_output=False)),
                normalize_gm(backend.bw_graphs[1].print_readable(print_output=False)),
            )
            self.assertEqual(
                len(
                    backend.graphs[0].graph.find_nodes(
                        op="call_function",
                        target=torch._higher_order_ops.wrap.tag_activation_checkpoint,
                    )
                ),
                1,
            )
            # TODO: add joint to the testing compile backend
            fw_outs = {
                n.name
                for n in backend.fw_graphs[0].graph.find_nodes(op="output")[0].args[0]
            }
            bw_ins = {
                n.name for n in backend.bw_graphs[0].graph.find_nodes(op="placeholder")
            }
            for node in backend.fw_graphs[0].graph.nodes:
                if "recompute" in node.meta:
                    expected = save_scalar_muls(None, node.target, None, None)
                    actual = node.meta["recompute"]
                    self.assertEqual(expected, actual)
                    if actual == torch.utils.checkpoint.CheckpointPolicy.MUST_SAVE:
                        self.assertTrue(node.name in fw_outs and node.name in bw_ins)
                    elif (
                        actual == torch.utils.checkpoint.CheckpointPolicy.MUST_RECOMPUTE
                    ):
                        # can still be in fw_outs for post-graph bytecode
                        self.assertFalse(node.name in bw_ins)

    @unittest.skipIf(*get_skip_reasons())
    def test_sac_deferred(self):
        # This test is in a bit of a weird state, it needs compositional compile API
        # so that we can defer inlining for up until AOTAutograd stage 1.
        # Then we should be inlined by stage 2. But we can't do that today.

        cp_decorated, cp_function = get_local_mapped_functions()
        bs = 8 * 1
        dim1 = 96
        dim2 = dim1 * 4
        nheads = 16
        seq_len = 16

        from torch._dynamo.testing import AotEagerAndRecordGraphs, normalize_gm

        backend = AotEagerAndRecordGraphs()

        model = create_model(
            cp_decorated, nheads, dim1, dim2, sac_policy=save_scalar_muls
        )
        inputs = (torch.randn(bs, seq_len, dim1, requires_grad=True),)
        try:
            with (
                LocalMapWrappedHigherOrderVariable.enable(),
                torch._higher_order_ops.local_map.defer_inlining(),
            ):
                out = torch.compile(model, backend=backend)(*inputs)
            out.sum().backward()
        except AttributeError as e:
            # TODO: get rid of this when we can install as a subgraph
            self.assertTrue(
                "module 'torch._higher_order_ops.local_map' has no attribute 'call_local_map'"
                in str(e)
            )

        model = create_model(
            cp_function, nheads, dim1, dim2, sac_policy=save_scalar_muls
        )
        inputs = (torch.randn(bs, seq_len, dim1, requires_grad=True),)
        try:
            with (
                LocalMapWrappedHigherOrderVariable.enable(),
                torch._higher_order_ops.local_map.defer_inlining(),
            ):
                out = torch.compile(model, backend=backend)(*inputs)
            out.sum().backward()
        except AttributeError as e:
            # TODO: get rid of this when we can install as a subgraph
            self.assertTrue(
                "module 'torch._higher_order_ops.local_map' has no attribute 'call_local_map'"
                in str(e)
            )

        # TODO: re-enable tests on backward when we can install as a subgraph
        if not TEST_WITH_CROSSREF:
            self.assertEqual(len(backend.graphs), 2)
            self.assertEqual(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                normalize_gm(backend.graphs[1].print_readable(print_output=False)),
            )
            self.assertEqual(
                normalize_gm(backend.fw_graphs[0].print_readable(print_output=False)),
                normalize_gm(backend.fw_graphs[1].print_readable(print_output=False)),
            )
            # self.assertEqual(
            #     normalize_gm(backend.bw_graphs[0].print_readable(print_output=False)),
            #     normalize_gm(backend.bw_graphs[1].print_readable(print_output=False)),
            # )
            self.assertEqual(
                len(
                    backend.graphs[0].graph.find_nodes(
                        op="call_function",
                        target=torch._higher_order_ops.wrap.tag_activation_checkpoint,
                    )
                ),
                1,
            )
            # TODO: add joint to the testing compile backend
            fw_outs = {
                n.name
                for n in backend.fw_graphs[0].graph.find_nodes(op="output")[0].args[0]
            }
            # bw_ins = {
            #     n.name for n in backend.bw_graphs[0].graph.find_nodes(op="placeholder")
            # }
            for node in backend.fw_graphs[0].graph.nodes:
                if "recompute" in node.meta:
                    expected = save_scalar_muls(None, node.target, None, None)
                    actual = node.meta["recompute"]
                    self.assertEqual(expected, actual)
                    if actual == torch.utils.checkpoint.CheckpointPolicy.MUST_SAVE:
                        self.assertTrue(node.name in fw_outs)
                    #     self.assertTrue(node.name in fw_outs and node.name in bw_ins)
                    # elif (
                    #     actual == torch.utils.checkpoint.CheckpointPolicy.MUST_RECOMPUTE
                    # ):
                    #     # can still be in fw_outs for post-graph bytecode
                    #     self.assertFalse(node.name in bw_ins)


if __name__ == "__main__":
    run_tests()
