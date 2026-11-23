# Owner(s): ["module: inductor"]
"""InductorChoices contributors that lower whole flex_attention calls
(lower_flex_attention / lower_flex_attention_backward)."""

from __future__ import annotations

import contextlib
import math
import unittest
from typing import Any

import torch
import torch.fx
from torch._inductor.choices import (
    InductorChoices,
    register_inductor_choices,
    registered_inductor_choices,
    unregister_inductor_choices,
)
from torch._inductor.kernel.flex.backend import (
    FlexAttentionBackwardCall,
    FlexAttentionBackwardOutputs,
    FlexAttentionCall,
    FlexAttentionDeclined,
    FlexAttentionOutputs,
)
from torch._inductor.lowering import fallback_handler
from torch._inductor.utils import fresh_cache, run_and_get_code
from torch.nn.attention.flex_attention import AuxRequest, flex_attention
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TestCase,
)
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON


def _math(q, k, v, scale):
    scores = (q.float() @ k.float().transpose(-2, -1)) * scale
    lse = torch.logsumexp(scores, dim=-1)
    out = torch.softmax(scores, dim=-1) @ v.float()
    return out.to(q.dtype), lse, scores.amax(dim=-1)


@torch.library.custom_op("toyflex::attention", mutates_args=())
def _toy_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out, lse, row_max = _math(q, k, v, scale)
    # The contract: statistics in base 2.
    return out, lse / math.log(2.0), row_max / math.log(2.0)


@_toy_attention.register_fake
def _(q, k, v, scale):
    stats = q.new_empty(q.shape[:-1], dtype=torch.float32)
    return q.new_empty(*q.shape[:-1], v.shape[-1]), stats, stats.clone()


@torch.library.custom_op("toyflex::attention_backward", mutates_args=())
def _toy_attention_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_out: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Autograd is off inside a custom op, so the softmax backward by hand.
    qf, kf, vf, do = (t.float() for t in (q, k, v, grad_out))
    p = torch.softmax((qf @ kf.transpose(-2, -1)) * scale, dim=-1)
    dp = do @ vf.transpose(-2, -1)
    ds = p * (dp - (dp * p).sum(dim=-1, keepdim=True))
    dq = (ds @ kf) * scale
    dk = (ds.transpose(-2, -1) @ qf) * scale
    dv = p.transpose(-2, -1) @ do
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


@_toy_attention_backward.register_fake
def _(q, k, v, grad_out, scale):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _is_identity_score_mod(gm: torch.fx.GraphModule) -> bool:
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    (output,) = (n for n in gm.graph.nodes if n.op == "output")
    return output.args[0] is placeholders[0]


class AppendHooksBackend(InductorChoices):
    """An old-style backend: it only overrides the append hooks, which
    Inductor calls with its own Triton candidates."""

    def __init__(self) -> None:
        super().__init__()
        self.forward_calls = 0
        self.backward_calls = 0

    def uuid(self) -> str:
        return "append-hooks-backend"

    def append_flex_attention_choices(self, choices, *args, **kwargs):
        self.forward_calls += 1
        return choices

    def append_flex_attention_backward_choices(self, choices, *args, **kwargs):
        self.backward_calls += 1
        return choices


class ToyFlexBackend(InductorChoices):
    """Plain softmax attention: an identity score_mod and no BlockMask; it
    declines everything else."""

    def __init__(self) -> None:
        super().__init__()
        self.forward_calls: list[FlexAttentionCall] = []
        self.backward_calls: list[FlexAttentionBackwardCall] = []

    def uuid(self) -> str:
        return "toy-flex-backend"

    def _check(self, score_mod, call) -> None:
        if not isinstance(score_mod, torch.fx.GraphModule):
            raise AssertionError(f"score_mod is a {type(score_mod)}")
        if not _is_identity_score_mod(score_mod):
            raise FlexAttentionDeclined("score_mod is not the identity")
        if not call.block_mask.is_full_default:
            raise FlexAttentionDeclined("a BlockMask")

    def lower_flex_attention(
        self, call: FlexAttentionCall
    ) -> FlexAttentionOutputs | None:
        self.forward_calls.append(call)
        self._check(call.score_mod, call)
        out, lse, row_max = fallback_handler(
            torch.ops.toyflex.attention.default, add_to_fallback_set=False
        )(call.query, call.key, call.value, call.scale)
        return FlexAttentionOutputs(out, lse, row_max if call.output_max else None)

    def lower_flex_attention_backward(
        self, call: FlexAttentionBackwardCall
    ) -> FlexAttentionBackwardOutputs | None:
        self.backward_calls.append(call)
        self._check(call.fw_graph, call)
        dq, dk, dv = fallback_handler(
            torch.ops.toyflex.attention_backward.default, add_to_fallback_set=False
        )(call.query, call.key, call.value, call.grad_out, call.scale)
        return FlexAttentionBackwardOutputs(dq, dk, dv, ())


class NoneBackend(InductorChoices):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def uuid(self) -> str:
        return "none-backend"

    def lower_flex_attention(self, call: Any) -> None:
        self.calls += 1
        return None


@unittest.skipUnless(HAS_CUDA_AND_TRITON, "requires CUDA and Triton")
class FlexBackendHooksTest(TestCase):
    hw_classification = HardwareClassification.CUDA

    def setUp(self) -> None:
        super().setUp()
        self._prior = registered_inductor_choices()
        for key, _ in self._prior:
            unregister_inductor_choices(key)
        torch._dynamo.reset()
        # Lowering, and so the hooks, runs only on a cache miss.
        self._cache = contextlib.ExitStack()
        self._cache.enter_context(fresh_cache())

    def tearDown(self) -> None:
        self._cache.close()
        for key, _ in registered_inductor_choices():
            unregister_inductor_choices(key)
        for key, factory in self._prior:
            register_inductor_choices(key, factory)
        torch._dynamo.reset()
        super().tearDown()

    def register(self, backend: InductorChoices) -> InductorChoices:
        register_inductor_choices("test-backend", lambda: backend)
        return backend

    def tensors(self, requires_grad: bool = False) -> list[torch.Tensor]:
        return [
            torch.randn(
                2,
                4,
                256,
                64,
                device="cuda",
                dtype=torch.float16,
                requires_grad=requires_grad,
            )
            for _ in range(3)
        ]

    def test_append_hooks_still_get_the_calls_nobody_lowers(self) -> None:
        # The append hooks keep working: alone, and after a contributor
        # declines the call.
        q, k, v = self.tensors(requires_grad=True)

        def biased(score, b, h, m, n):
            return score + (m - n).to(score.dtype) * 0.01

        def run(score_mod=None):
            torch._dynamo.reset()
            out = torch.compile(
                lambda q, k, v: flex_attention(q, k, v, score_mod=score_mod)
            )(q, k, v)
            out.sum().backward()

        old_style = AppendHooksBackend()
        register_inductor_choices("old-style", lambda: old_style)
        run()
        self.assertEqual((old_style.forward_calls, old_style.backward_calls), (1, 1))

        toy = self.register(ToyFlexBackend())
        run(biased)
        self.assertEqual(len(toy.forward_calls), 1)
        self.assertEqual((old_style.forward_calls, old_style.backward_calls), (2, 2))

    def test_lowered_call_skips_the_append_hooks(self) -> None:
        # A call a contributor lowers whole never builds Inductor's candidates,
        # so the append hooks are not asked about it.
        old_style = AppendHooksBackend()
        register_inductor_choices("old-style", lambda: old_style)
        toy = self.register(ToyFlexBackend())
        q, k, v = self.tensors(requires_grad=True)
        torch.compile(lambda q, k, v: flex_attention(q, k, v))(q, k, v).sum().backward()
        self.assertEqual((len(toy.forward_calls), len(toy.backward_calls)), (1, 1))
        self.assertEqual((old_style.forward_calls, old_style.backward_calls), (0, 0))

    def test_contributor_lowers_the_forward(self) -> None:
        backend = self.register(ToyFlexBackend())
        q, k, v = self.tensors()

        def f(q, k, v):
            return flex_attention(
                q, k, v, return_aux=AuxRequest(lse=True, max_scores=True)
            )

        (out, aux), (code,) = run_and_get_code(torch.compile(f), q, k, v)
        expected, lse, row_max = _math(q, k, v, 1 / math.sqrt(64))
        torch.testing.assert_close(out, expected, atol=2e-3, rtol=2e-3)
        # The base-2 statistics come back to the user in natural units.
        torch.testing.assert_close(aux.lse, lse, atol=2e-3, rtol=2e-3)
        torch.testing.assert_close(aux.max_scores, row_max, atol=2e-3, rtol=2e-3)
        (call,) = backend.forward_calls
        self.assertTrue(call.block_mask.is_full_default)
        self.assertEqual(call.backend, "AUTO")
        self.assertTrue(call.output_logsumexp and call.output_max)
        self.assertNotIn("BACKEND", call.kernel_options)
        # The contributor's kernel runs; Inductor built nothing of its own.
        self.assertIn("toyflex.attention", code)
        self.assertNotIn("triton_tem_", code)

    def test_contributor_lowers_the_backward(self) -> None:
        backend = self.register(ToyFlexBackend())
        q, k, v = self.tensors(requires_grad=True)
        grad = torch.randn_like(q)

        def f(q, k, v):
            return flex_attention(q, k, v)

        out = torch.compile(f)(q, k, v)
        grads = torch.autograd.grad(out, (q, k, v), grad)
        q2, k2, v2 = (t.detach().requires_grad_() for t in (q, k, v))
        expected = torch.autograd.grad(
            _math(q2, k2, v2, 1 / math.sqrt(64))[0], (q2, k2, v2), grad
        )
        for actual, wanted in zip(grads, expected):
            torch.testing.assert_close(actual, wanted, atol=2e-2, rtol=2e-2)
        self.assertEqual(len(backend.forward_calls), 1)
        (call,) = backend.backward_calls
        self.assertIsInstance(call.joint_graph, torch.fx.GraphModule)

    def test_declined_call_is_lowered_by_inductor(self) -> None:
        backend = self.register(ToyFlexBackend())
        q, k, v = self.tensors()

        def biased(score, b, h, m, n):
            return score + (m - n).to(score.dtype) * 0.01

        out, (code,) = run_and_get_code(
            torch.compile(lambda q, k, v: flex_attention(q, k, v, score_mod=biased)),
            q,
            k,
            v,
        )
        torch.testing.assert_close(
            out, flex_attention(q, k, v, score_mod=biased), atol=2e-3, rtol=2e-3
        )
        self.assertEqual(len(backend.forward_calls), 1)
        self.assertNotIn("toyflex", code)
        self.assertIn("triton_tem_", code)

    def test_contributor_returning_none_leaves_the_call(self) -> None:
        backend = self.register(NoneBackend())
        q, k, v = self.tensors()
        out = torch.compile(lambda q, k, v: flex_attention(q, k, v))(q, k, v)
        torch.testing.assert_close(out, flex_attention(q, k, v), atol=2e-3, rtol=2e-3)
        self.assertEqual(backend.calls, 1)

    def test_named_builtin_backend_skips_contributors(self) -> None:
        backend = self.register(ToyFlexBackend())
        q, k, v = self.tensors()
        out = torch.compile(
            lambda q, k, v: flex_attention(
                q, k, v, kernel_options={"BACKEND": "TRITON"}
            )
        )(q, k, v)
        torch.testing.assert_close(out, flex_attention(q, k, v), atol=2e-3, rtol=2e-3)
        self.assertEqual(backend.forward_calls, [])


if __name__ == "__main__":
    run_tests()
