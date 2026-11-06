# Owner(s): ["oncall: distributed"]

import os
from unittest import mock

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy
from torch.distributed.fsdp._flex_shard.placements.shard import Shard as EngineShard
from torch.distributed.fsdp._fully_shard._fsdp_engine import ENGINE_ENV_VAR
from torch.distributed.tensor import distribute_tensor, Shard
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, get_devtype
from torch.testing._internal.common_utils import run_tests


device_type = torch.device(get_devtype())


class _Net(nn.Module):
    """Uneven Shard(0) params, a frozen fp32 param next to bf16 params (a
    mixed-dtype group), and a Shard(1)-able weight."""

    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.inner = nn.Linear(16, 33, dtype=dtype)
        self.frozen = nn.Linear(16, 33)
        self.frozen.requires_grad_(False)
        self.out = nn.Linear(33, 8, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frozen = self.frozen(x.to(self.frozen.weight.dtype)).to(x.dtype)
        return self.out(torch.relu(self.inner(x) + frozen)).float()


class TestFullyShardEngine(FSDPTest):
    """``fully_shard`` on the FlexShard engine matches the legacy implementation
    bitwise and keeps each group's sharded parameters in one engine bucket."""

    @property
    def world_size(self) -> int:
        return min(4, torch.get_device_module(device_type).device_count())

    def _train(
        self,
        engine: str,
        *,
        mesh=None,
        offload=False,
        mp=False,
        raf=True,
        shard_dim=0,
        dtype=torch.float32,
        steps=3,
    ):
        with mock.patch.dict(os.environ, {ENGINE_ENV_VAR: engine}):
            torch.manual_seed(0)
            model = _Net(dtype).to(device_type)
            kwargs = {"reshard_after_forward": raf, "mesh": mesh}
            if offload:
                kwargs["offload_policy"] = CPUOffloadPolicy()
            if mp:
                kwargs["mp_policy"] = MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16, reduce_dtype=torch.float32
                )

            def placement_fn(param: nn.Parameter) -> Shard:
                # FSDP shards Shard(i > 0) evenly only
                if param.ndim > 1 and param.size(shard_dim) % self.world_size == 0:
                    return Shard(shard_dim)
                return Shard(0)

            fully_shard(model.out, shard_placement_fn=placement_fn, **kwargs)
            fully_shard(model, shard_placement_fn=placement_fn, **kwargs)
        group = model._get_fsdp_state()._fsdp_param_group
        self.assertEqual(group._engine is not None, engine == "flex")
        optim = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-2
        )
        results = []
        for step in range(steps):
            torch.manual_seed(1 + self.rank + step)
            inp = torch.randn(4, 16, device=device_type, dtype=dtype)
            loss = model(inp).sum()
            loss.backward()
            results.append(loss.detach())
            # Each rank compares its shards, which also works for CPU offload
            results += [
                p.grad.to_local() for p in model.parameters() if p.requires_grad
            ]
            optim.step()
            optim.zero_grad()
        results += [p.to_local() for p in model.parameters()]
        return model, results

    def _assert_bitwise(self, **kwargs):
        _, expected = self._train("legacy", **kwargs)
        model, actual = self._train("flex", **kwargs)
        self.assertEqual(len(expected), len(actual))
        for exp, act in zip(expected, actual):
            self.assertEqual(exp, act, atol=0, rtol=0)
        return model

    @skip_if_lt_x_gpu(2)
    def test_bitwise_parity(self):
        for mp, raf, shard_dim, dtype in (
            (False, True, 0, torch.float32),
            (True, True, 0, torch.float32),
            (False, False, 1, torch.float32),
            (True, 2, 0, torch.float32),
            (False, True, 0, torch.bfloat16),
            (True, 2, 0, torch.bfloat16),
        ):
            if isinstance(raf, int) and not isinstance(raf, bool):
                if self.world_size % raf:
                    continue
            with self.subTest(mp=mp, raf=raf, shard_dim=shard_dim, dtype=dtype):
                self._assert_bitwise(mp=mp, raf=raf, shard_dim=shard_dim, dtype=dtype)

    @skip_if_lt_x_gpu(2)
    def test_bitwise_parity_cpu_offload(self):
        self._assert_bitwise(offload=True)

    @skip_if_lt_x_gpu(4)
    def test_bitwise_parity_hsdp(self):
        mesh = init_device_mesh(
            device_type.type, (2, self.world_size // 2), mesh_dim_names=("rep", "shard")
        )
        self._assert_bitwise(mesh=mesh, mp=True)

    @skip_if_lt_x_gpu(2)
    def test_params_live_in_one_engine_bucket(self):
        model, _ = self._train("flex", steps=1, dtype=torch.bfloat16)
        group = model._get_fsdp_state()._fsdp_param_group
        storage = group._engine.storage
        self.assertEqual(storage.dtype, torch.uint8)
        for fsdp_param, info in zip(group.fsdp_params, group._engine.infos):
            self.assertIsInstance(info.placement, EngineShard)
            local = fsdp_param.sharded_param._local_tensor
            self.assertEqual(
                local.untyped_storage().data_ptr(), storage.untyped_storage().data_ptr()
            )
            self.assertEqual(
                local.storage_offset() * local.element_size(), info.byte_offset
            )

    @skip_if_lt_x_gpu(2)
    def test_meta_init_and_load_state_dict_assign(self):
        _, expected = self._train("legacy", steps=1)
        with mock.patch.dict(os.environ, {ENGINE_ENV_VAR: "flex"}):
            torch.manual_seed(0)
            reference = _Net(torch.float32).to(device_type)
            with torch.device("meta"):
                model = _Net(torch.float32)
            fully_shard(model.out)
            fully_shard(model)
        # to_empty replaces the sharded parameters, which lazy init re-homes
        # into the engine bucket
        model.to_empty(device=device_type)
        full_sd = reference.state_dict()
        sharded_sd = {}
        for name, param in model.state_dict().items():
            sharded_sd[name] = distribute_tensor(
                full_sd[name], param.device_mesh, param.placements
            )
        model.load_state_dict(sharded_sd, assign=True)
        optim = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-2
        )
        torch.manual_seed(1 + self.rank)
        inp = torch.randn(4, 16, device=device_type)
        loss = model(inp).sum()
        loss.backward()
        actual = [loss.detach()]
        actual += [p.grad.to_local() for p in model.parameters() if p.requires_grad]
        optim.step()
        actual += [p.to_local() for p in model.parameters()]
        group = model._get_fsdp_state()._fsdp_param_group
        self.assertTrue(group._engine.homed(group.fsdp_params))
        for exp, act in zip(expected, actual):
            self.assertEqual(exp, act, atol=0, rtol=0)
        dist.barrier()


if __name__ == "__main__":
    run_tests()
