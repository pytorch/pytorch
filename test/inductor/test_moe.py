# Owner(s): ["module: inductor"]
import unittest

import torch
import torch.nn as nn

try:
    from torch._dynamo.utils import counters
except ImportError:
    counters = None  # type: ignore[assignment]

try:
    from torch._inductor.test_case import TestCase as InductorTestCase
except ImportError:
    InductorTestCase = unittest.TestCase  # type: ignore[misc, assignment]

try:
    from torch.testing._internal.common_utils import skipIfRocm
except ImportError:
    def skipIfRocm(fn):  # type: ignore[misc]
        return fn


def _grouped_mm_available():
    try:
        return callable(torch.nn.functional.grouped_mm)
    except AttributeError:
        return False


class MoEExpertModule(nn.Module):
    def __init__(self, num_experts, hidden_size, expert_dim):
        super().__init__()
        self.num_experts = num_experts
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, hidden_size, 2 * expert_dim, dtype=torch.bfloat16)
        )
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, expert_dim, hidden_size, dtype=torch.bfloat16)
        )
        nn.init.normal_(self.gate_up_proj)
        nn.init.normal_(self.down_proj)

    def forward(self, hidden_3d):
        gate_up = torch.bmm(hidden_3d, self.gate_up_proj)
        gate, up = gate_up.chunk(2, dim=-1)
        return torch.bmm(up * torch.nn.functional.silu(gate), self.down_proj)


class MoEExpertGeluModule(nn.Module):
    def __init__(self, num_experts, hidden_size, expert_dim):
        super().__init__()
        self.num_experts = num_experts
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, hidden_size, 2 * expert_dim, dtype=torch.bfloat16)
        )
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, expert_dim, hidden_size, dtype=torch.bfloat16)
        )
        nn.init.normal_(self.gate_up_proj)
        nn.init.normal_(self.down_proj)

    def forward(self, hidden_3d):
        gate_up = torch.bmm(hidden_3d, self.gate_up_proj)
        gate, up = gate_up.chunk(2, dim=-1)
        return torch.bmm(up * torch.nn.functional.gelu(gate), self.down_proj)


def _pick_device():
    if torch.cuda.is_available():
        # Verify BF16 support
        props = torch.cuda.get_device_properties(0)
        if props.major >= 8:
            return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return None


class TestMoEPatternMatch(InductorTestCase):
    def setUp(self):
        super().setUp()
        if counters is not None:
            counters.clear()
        self.device = _pick_device()
        if self.device is None:
            self.skipTest("No device with BF16 support available")
        if not _grouped_mm_available():
            self.skipTest("torch.nn.functional.grouped_mm not available")

    def _make_inputs(self, num_experts=4, tokens_per_expert=16, hidden_size=64, expert_dim=32):
        device = self.device
        hidden_3d = torch.randn(
            num_experts, tokens_per_expert, hidden_size,
            dtype=torch.bfloat16, device=device,
        )
        gate_up_proj = torch.randn(
            num_experts, hidden_size, 2 * expert_dim,
            dtype=torch.bfloat16, device=device,
        )
        down_proj = torch.randn(
            num_experts, expert_dim, hidden_size,
            dtype=torch.bfloat16, device=device,
        )
        return hidden_3d, gate_up_proj, down_proj

    def test_moe_expert_pattern_inference(self):
        hidden_3d, gate_up_proj, down_proj = self._make_inputs()

        def original_fn(hidden_3d, gate_up_proj, down_proj):
            gate_up = torch.bmm(hidden_3d, gate_up_proj)
            gate, up = gate_up.chunk(2, dim=-1)
            return torch.bmm(up * torch.nn.functional.silu(gate), down_proj)

        def replacement_fn(hidden_3d, gate_up_proj, down_proj):
            gate_up = torch.nn.functional.grouped_mm(hidden_3d, gate_up_proj)
            gate, up = gate_up.chunk(2, dim=-1)
            return torch.nn.functional.grouped_mm(
                up * torch.nn.functional.silu(gate), down_proj
            )

        expected = original_fn(hidden_3d, gate_up_proj, down_proj)
        actual = replacement_fn(hidden_3d, gate_up_proj, down_proj)

        self.assertTrue(
            torch.allclose(expected, actual, rtol=1e-2, atol=1e-2),
            f"Max diff: {(expected - actual).abs().max().item()}",
        )

    def test_moe_expert_gelu_inference(self):
        hidden_3d, gate_up_proj, down_proj = self._make_inputs()

        def original_fn(hidden_3d, gate_up_proj, down_proj):
            gate_up = torch.bmm(hidden_3d, gate_up_proj)
            gate, up = gate_up.chunk(2, dim=-1)
            return torch.bmm(up * torch.nn.functional.gelu(gate), down_proj)

        def replacement_fn(hidden_3d, gate_up_proj, down_proj):
            gate_up = torch.nn.functional.grouped_mm(hidden_3d, gate_up_proj)
            gate, up = gate_up.chunk(2, dim=-1)
            return torch.nn.functional.grouped_mm(
                up * torch.nn.functional.gelu(gate), down_proj
            )

        expected = original_fn(hidden_3d, gate_up_proj, down_proj)
        actual = replacement_fn(hidden_3d, gate_up_proj, down_proj)

        self.assertTrue(
            torch.allclose(expected, actual, rtol=1e-2, atol=1e-2),
            f"Max diff: {(expected - actual).abs().max().item()}",
        )

    def test_moe_pattern_match_count(self):
        if counters is None:
            self.skipTest("counters not available")

        try:
            from torch._inductor.fx_passes.moe import _moe_init
        except ImportError:
            self.skipTest("torch._inductor.fx_passes.moe not importable")

        num_experts, hidden_size, expert_dim = 4, 64, 32
        module = MoEExpertModule(num_experts, hidden_size, expert_dim).to(self.device)

        hidden_3d = torch.randn(
            num_experts, 16, hidden_size,
            dtype=torch.bfloat16, device=self.device,
        )

        # Trigger pattern registration
        _moe_init()

        counters["inductor"]["fuse_moe"] = 0

        with torch.no_grad():
            compiled_fn = torch.compile(module, backend="inductor")
            _ = compiled_fn(hidden_3d)

        self.assertGreater(
            counters["inductor"]["fuse_moe"],
            0,
            "Expected fuse_moe counter to be incremented after pattern match",
        )

    def test_moe_expert_module_numerical(self):
        """Verify MoEExpertModule output is numerically consistent across runs."""
        num_experts, hidden_size, expert_dim = 4, 64, 32
        module = MoEExpertModule(num_experts, hidden_size, expert_dim).to(self.device)

        hidden_3d = torch.randn(
            num_experts, 16, hidden_size,
            dtype=torch.bfloat16, device=self.device,
        )

        out1 = module(hidden_3d)
        out2 = module(hidden_3d)

        self.assertTrue(
            torch.allclose(out1, out2, rtol=0, atol=0),
            "Deterministic forward pass produced different results",
        )


if __name__ == "__main__":
    unittest.main()
