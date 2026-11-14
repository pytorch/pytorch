# Owner(s): ["oncall: distributed"]

import copy
import io

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import fully_shard
from torch.distributed.fsdp._flex_shard.placements.shard import Shard as EngineShard
from torch.distributed.tensor import distribute_tensor, Shard
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, get_devtype
from torch.testing._internal.common_utils import HardwareClassification, run_tests


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


class _UnevenNet(nn.Module):
    """Weights with 33 columns, which Shard(1) splits unevenly over 2 or 4
    ranks; the frozen bf16 weight beside fp32 ones makes its group's dtypes
    differ, so that it all-gathers bytes."""

    def __init__(self) -> None:
        super().__init__()
        self.inner = nn.Linear(33, 33)
        self.frozen = nn.Linear(33, 33, dtype=torch.bfloat16)
        self.frozen.requires_grad_(False)
        self.out = nn.Linear(33, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frozen = self.frozen(x.to(torch.bfloat16)).to(x.dtype)
        return self.out(torch.relu(self.inner(x) + frozen))


class TestFullyShardEngine(FSDPTest):
    """``fully_shard`` runs on the FlexShard engine: each group's sharded
    parameters live in one engine bucket, also after meta-device init."""

    hw_classification = HardwareClassification.CUDA

    @property
    def world_size(self) -> int:
        return min(4, torch.get_device_module(device_type).device_count())

    def _shard(self, model: _Net) -> None:
        def placement_fn(param: nn.Parameter) -> Shard:
            # Shard(1) where it splits evenly, which the engine's Shard gathers
            if param.ndim > 1 and param.size(1) % self.world_size == 0:
                return Shard(1)
            return Shard(0)

        fully_shard(model.out, shard_placement_fn=placement_fn)
        fully_shard(model, shard_placement_fn=placement_fn)

    def _train_step(self, model: _Net, dtype: torch.dtype) -> list[torch.Tensor]:
        optim = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-2
        )
        torch.manual_seed(1 + self.rank)
        inp = torch.randn(4, 16, device=device_type, dtype=dtype)
        loss = model(inp).sum()
        loss.backward()
        results = [loss.detach()]
        results += [p.grad.to_local() for p in model.parameters() if p.requires_grad]
        optim.step()
        results += [p.to_local() for p in model.parameters()]
        return results

    @skip_if_lt_x_gpu(2)
    def test_uneven_shard_dim1_matches_unsharded_reference(self):
        # The engine's Shard all-gathers the uneven Shard(1) weights itself:
        # in the out group as its dtype, in the root group as bytes
        torch.manual_seed(0)
        model = _UnevenNet().to(device_type)
        reference = copy.deepcopy(model)
        for module in (model.out, model):
            fully_shard(
                module, shard_placement_fn=lambda p: Shard(1 if p.ndim > 1 else 0)
            )
            infos = module._get_fsdp_state()._fsdp_param_groups[0].storage.param_infos
            for info in infos.values():
                self.assertIs(type(info.placement), EngineShard)
        torch.manual_seed(1 + self.rank)
        for _ in range(2):
            inp = torch.randn(4, 33, device=device_type)
            output, ref_output = model(inp), reference(inp)
            self.assertEqual(output, ref_output)
            output.sum().backward()
            ref_output.sum().backward()
            for param, ref_param in zip(
                model.parameters(), reference.parameters(), strict=True
            ):
                if ref_param.grad is None:
                    continue
                dist.all_reduce(ref_param.grad)
                ref_param.grad.div_(self.world_size)
                self.assertEqual(param.grad.full_tensor(), ref_param.grad)
                param.grad = ref_param.grad = None

    @skip_if_lt_x_gpu(2)
    def test_backward_prefetch_skips_buckets_without_unshard_in_backward(self):
        # A bucket that does not unshard in backward never takes a backward
        # prefetch, so one would stay in flight past a backward that keeps
        # prefetches, into the next forward after the optimizer step
        model = nn.Sequential(nn.Embedding(16, 8), nn.Linear(8, 8)).to(device_type)
        fully_shard(model[0])
        fully_shard(model[1])
        fully_shard(model)
        # Embedding weights are not needed for the embedding backward
        model[0].set_unshard_in_backward(False)
        model.set_is_last_backward(False)
        model(torch.randint(0, 16, (4,), device=device_type)).sum().backward()
        state = model._get_fsdp_state()
        context = state._engine_context
        self.assertEqual(context.pending_unshards, [])
        model.set_is_last_backward(True)
        self.assertTrue(context.wait_reductions_at_backward_end)

    @skip_if_lt_x_gpu(2)
    def test_params_live_in_one_engine_bucket(self):
        torch.manual_seed(0)
        model = _Net(torch.bfloat16).to(device_type)
        self._shard(model)
        self._train_step(model, torch.bfloat16)
        bucket = model._get_fsdp_state()._fsdp_param_groups[0]
        storage = bucket.storage.byte_storage
        self.assertEqual(storage.dtype, torch.uint8)
        infos = list(bucket.storage.param_infos.values())
        for fsdp_param, info in zip(bucket.fsdp_params, infos, strict=True):
            self.assertIsInstance(info.placement, EngineShard)
            local = fsdp_param.sharded_param._local_tensor
            self.assertEqual(
                local.untyped_storage().data_ptr(), storage.untyped_storage().data_ptr()
            )
            self.assertEqual(
                local.storage_offset() * local.element_size(), info.byte_offset
            )

    @skip_if_lt_x_gpu(2)
    def test_state_dict_deepcopy_and_save(self):
        # The sharded parameters' local tensors view the bucket at nonzero
        # offsets, which the state dict's detached DTensors keep
        torch.manual_seed(0)
        model = _Net(torch.float32).to(device_type)
        self._shard(model)
        self._train_step(model, torch.float32)
        state_dict = model.state_dict()
        for name, value in copy.deepcopy(state_dict).items():
            self.assertEqual(value, state_dict[name])
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        buffer.seek(0)
        for name, value in torch.load(buffer, weights_only=True).items():
            self.assertEqual(value, state_dict[name])

    @skip_if_lt_x_gpu(2)
    def test_meta_init_of_grouped_modules(self):
        # to_empty() visits a group's modules one by one, so the bucket sees
        # some of its parameters materialized while others are still on meta
        def build(device: torch.device | str) -> nn.Module:
            with torch.device(device):
                model = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 16))
            fully_shard([model[0], model[1]])
            fully_shard(model)
            return model

        torch.manual_seed(0)
        reference = build(device_type)
        model = build("meta")
        model.to_empty(device=device_type)
        with torch.no_grad():
            for param, ref_param in zip(
                model.parameters(), reference.parameters(), strict=True
            ):
                param.copy_(ref_param)
        inp = torch.randn(4, 16, device=device_type)
        self.assertEqual(model(inp), reference(inp))
        self.assertTrue(model[0]._get_fsdp_state()._fsdp_param_groups[0].homed())

    @skip_if_lt_x_gpu(2)
    def test_meta_init_and_load_state_dict_assign(self):
        torch.manual_seed(0)
        reference = _Net(torch.float32).to(device_type)
        full_sd = reference.state_dict()
        self._shard(reference)
        expected = self._train_step(reference, torch.float32)

        with torch.device("meta"):
            model = _Net(torch.float32)
        self._shard(model)
        # to_empty replaces the sharded parameters, which lazy init re-homes
        # into the engine bucket
        model.to_empty(device=device_type)
        sharded_sd = {
            name: distribute_tensor(full_sd[name], param.device_mesh, param.placements)
            for name, param in model.state_dict().items()
        }
        model.load_state_dict(sharded_sd, assign=True)
        actual = self._train_step(model, torch.float32)
        self.assertTrue(model._get_fsdp_state()._fsdp_param_groups[0].homed())
        self.assertEqual(len(expected), len(actual))
        for exp, act in zip(expected, actual):
            self.assertEqual(exp, act, atol=0, rtol=0)
        dist.barrier()


if __name__ == "__main__":
    run_tests()
