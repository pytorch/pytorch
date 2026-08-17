# Owner(s): ["module: inductor"]
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch._inductor.codegen.flydsl.flydsl_template import FlyDSLTemplate
from torch._inductor.codegen.flydsl.flydsl_utils import runtime_available
from torch._inductor.kernel.flex import flex_flydsl_attention
from torch._inductor.kernel.flex.flex_flydsl_attention import (
    _can_use_flydsl_flex_attention_backward,
    flex_flydsl_backward_template,
)
from torch.testing._internal.common_utils import run_tests, TestCase


def _score_graph(fn) -> SimpleNamespace:
    gm = torch.fx.symbolic_trace(fn)
    return SimpleNamespace(graph_module=gm)


def _identity_score_graph() -> SimpleNamespace:
    return _score_graph(lambda score, b, h, m, n: score)


def _nontrivial_score_graph() -> SimpleNamespace:
    return _score_graph(lambda score, b, h, m, n: score * 2.0)


def _fake_query(dtype: torch.dtype) -> SimpleNamespace:
    return SimpleNamespace(get_dtype=lambda: dtype)


class TestFlexFlyDSLGates(TestCase):
    """Gate-logic tests: run in CI even without the flydsl runtime installed."""

    def test_template_registered(self):
        self.assertIn("flex_flydsl_backward", FlyDSLTemplate.all_templates)
        self.assertIs(
            FlyDSLTemplate.all_templates["flex_flydsl_backward"],
            flex_flydsl_backward_template,
        )

    def test_gate_declines_when_runtime_unavailable(self):
        with mock.patch.object(
            flex_flydsl_attention, "runtime_available", return_value=False
        ):
            can_use, reason = _can_use_flydsl_flex_attention_backward(
                _identity_score_graph(),
                _identity_score_graph(),
                _fake_query(torch.bfloat16),
            )
        self.assertFalse(can_use)
        self.assertIn("unavailable", reason)

    def test_gate_declines_when_not_rocm(self):
        with (
            mock.patch.object(
                flex_flydsl_attention, "runtime_available", return_value=True
            ),
            mock.patch.object(torch.version, "hip", None),
        ):
            can_use, reason = _can_use_flydsl_flex_attention_backward(
                _identity_score_graph(),
                _identity_score_graph(),
                _fake_query(torch.bfloat16),
            )
        self.assertFalse(can_use)
        self.assertIn("ROCm", reason)

    def test_gate_declines_when_not_bf16(self):
        with (
            mock.patch.object(
                flex_flydsl_attention, "runtime_available", return_value=True
            ),
            mock.patch.object(torch.version, "hip", "6.0.0"),
        ):
            can_use, reason = _can_use_flydsl_flex_attention_backward(
                _identity_score_graph(),
                _identity_score_graph(),
                _fake_query(torch.float16),
            )
        self.assertFalse(can_use)
        self.assertIn("bf16", reason)

    def test_gate_declines_when_score_mod_nontrivial(self):
        with (
            mock.patch.object(
                flex_flydsl_attention, "runtime_available", return_value=True
            ),
            mock.patch.object(torch.version, "hip", "6.0.0"),
        ):
            can_use, reason = _can_use_flydsl_flex_attention_backward(
                _nontrivial_score_graph(),
                _identity_score_graph(),
                _fake_query(torch.bfloat16),
            )
        self.assertFalse(can_use)
        self.assertIn("identity score_mod", reason)


@unittest.skipUnless(runtime_available(), "flydsl runtime not available")
class TestFlexFlyDSLRuntime(TestCase):
    """Runtime tests: only exercised when the flydsl runtime is installed."""

    def test_gate_allows_trivial_bf16_on_rocm(self):
        can_use, reason = _can_use_flydsl_flex_attention_backward(
            _identity_score_graph(),
            _identity_score_graph(),
            _fake_query(torch.bfloat16),
        )
        self.assertTrue(can_use, reason)


if __name__ == "__main__":
    run_tests()
