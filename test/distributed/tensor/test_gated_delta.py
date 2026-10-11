# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
"""Gated DeltaNet context parallel: all-to-all heads, full timeline, autograd."""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.experimental import context_parallel
from torch.distributed.tensor.experimental._context_parallel._attention import (
    _cp_options,
)
from torch.distributed.tensor.experimental._context_parallel._gated_delta import (
    a2a_feat_to_seq,
    a2a_seq_to_feat,
)
from torch.nn.modules.gated_delta import GatedDeltaNet, _TinyGatedDeltaModel
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests, skipIfRocm
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


class GatedDeltaCPTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    @skipIfRocm
    @with_comms
    def test_a2a_roundtrip_autograd(self) -> None:
        group = dist.group.WORLD
        torch.manual_seed(0)
        x = torch.randn(2, 8, 16, device=self.device_type, requires_grad=True)
        y = a2a_seq_to_feat(x, group)
        z = a2a_feat_to_seq(y, group)
        self.assertEqual(x.shape, z.shape)
        torch.testing.assert_close(z, x, atol=1e-5, rtol=1e-5)
        z.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertGreater(x.grad.abs().sum().item(), 0.0)

    @skip_if_lt_x_gpu(2)
    @skipIfRocm
    @with_comms
    def test_module_matches_unsharded_and_trains_in_proj(self) -> None:
        device = self.device_type
        torch.manual_seed(0)
        hidden, heads, head_dim, seq = 32, 4, 8, 16
        ref = GatedDeltaNet(hidden, heads, head_dim, device=device, dtype=torch.float32)
        cp_model = GatedDeltaNet(
            hidden, heads, head_dim, device=device, dtype=torch.float32
        )
        cp_model.load_state_dict(ref.state_dict())

        tokens = torch.randn(1, seq, hidden, device=device, dtype=torch.float32)
        dist.broadcast(tokens, src=0)
        cp_in = tokens.detach().clone()
        mesh = init_device_mesh(device, (self.world_size,), mesh_dim_names=("cp",))
        _cp_options.enable_load_balance = False
        with context_parallel(mesh["cp"], buffers=[cp_in], buffer_seq_dims=[1]):
            cp_out = cp_model(cp_in)
            cp_out.pow(2).mean().backward()

        parts = [torch.empty_like(cp_out) for _ in range(self.world_size)]
        dist.all_gather(parts, cp_out.detach(), group=dist.group.WORLD)
        full = torch.cat(parts, dim=1)
        with torch.no_grad():
            want = ref(tokens)
        torch.testing.assert_close(full, want, atol=2e-3, rtol=2e-3)
        self.assertIsNotNone(cp_model.in_proj.weight.grad)
        self.assertGreater(cp_model.in_proj.weight.grad.abs().sum().item(), 0.0)


class GatedDeltaFSDPSmokeTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    @skipIfRocm
    @with_comms
    def test_tiny_model_backward_reaches_embed(self) -> None:
        mesh = init_device_mesh(
            self.device_type, (1, self.world_size), mesh_dim_names=("fsdp", "cp")
        )
        model = _TinyGatedDeltaModel(
            vocab_size=32,
            hidden_size=32,
            num_heads=4,
            head_dim=8,
            num_layers=1,
            device=self.device_type,
            dtype=torch.float32,
        )
        _cp_options.enable_load_balance = False
        tokens = torch.randint(0, 32, (1, 16), device=self.device_type)
        dist.broadcast(tokens, src=0)
        with context_parallel(mesh["cp"], buffers=[tokens], buffer_seq_dims=[1]):
            logits = model(tokens)
            logits.float().pow(2).mean().backward()
        self.assertIsNotNone(model.embed.weight.grad)
        self.assertIsNotNone(model.layers[0].in_proj.weight.grad)


if __name__ == "__main__":
    run_tests()
