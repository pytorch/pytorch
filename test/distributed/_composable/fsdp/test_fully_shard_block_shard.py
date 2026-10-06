# Owner(s): ["oncall: distributed"]

import copy

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
    StateDictOptions,
)
from torch.distributed.fsdp import DataParallelMeshDims, fully_shard
from torch.distributed.fsdp._fully_shard._all_gather_layout import (
    _default_all_gather_output_fn,
)
from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
    _default_reduce_scatter_input_fn,
)
from torch.distributed.fsdp.experimental import DefaultAllGatherLayout
from torch.distributed.tensor import (
    distribute_tensor,
    DTensor,
    init_device_mesh,
    Replicate,
    Shard,
)
from torch.distributed.tensor.placement_types import BlockShard
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTestContinuous, get_devtype
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir


device_type = torch.device(get_devtype())

# 3 experts over 4 ranks: w1 has 15 merged rows, so rank rows cross expert
# boundaries and the last rank is short.
NUM_EXPERTS, DIM, HIDDEN = 3, 8, 5


class Experts(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(NUM_EXPERTS, HIDDEN, DIM) * 0.1)
        self.w2 = nn.Parameter(torch.randn(NUM_EXPERTS, DIM, HIDDEN) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.bmm(x, self.w1.transpose(1, 2)).relu()
        return torch.bmm(h, self.w2.transpose(1, 2))


class MoEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(DIM, DIM)
        self.experts = Experts()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        return self.experts(h.unsqueeze(0).expand(NUM_EXPERTS, -1, -1)).sum(0)


def _block_shard_placement_fn(param: nn.Parameter) -> BlockShard | None:
    if param.ndim == 3:
        return BlockShard.split_leading(param.shape, 2)
    return None


class TestFullyShardBlockShard(FSDPTestContinuous):
    world_size = 4

    def _init_models(self, mesh=None, reshard_after_forward=True):
        torch.manual_seed(42)
        model = MoEModel().to(device_type)
        ref_model = copy.deepcopy(model)
        fully_shard(
            model.experts,
            mesh=mesh,
            shard_placement_fn=_block_shard_placement_fn,
            reshard_after_forward=reshard_after_forward,
        )
        fully_shard(model, mesh=mesh, reshard_after_forward=reshard_after_forward)
        return model, ref_model

    @skip_if_lt_x_gpu(4)
    def test_init(self):
        model, ref_model = self._init_models()
        for name in ("w1", "w2"):
            param = getattr(model.experts, name)
            ref_param = getattr(ref_model.experts, name)
            placement = BlockShard((ref_param.shape[2],))
            self.assertIsInstance(param, DTensor)
            self.assertEqual(param.placements, (placement,))
            # Rank r owns merged rows [r * q, (r + 1) * q).
            rows = ref_param.reshape(-1, ref_param.shape[2])
            q = -(-rows.shape[0] // self.world_size)
            self.assertEqual(
                param.to_local(), rows[self.rank * q : (self.rank + 1) * q]
            )
            self.assertEqual(param.full_tensor(), ref_param)
        self.assertEqual(model.in_proj.weight.placements, (Shard(0),))

    @skip_if_lt_x_gpu(4)
    @parametrize("reshard_after_forward", [True, False, 2])
    def test_train_parity(self, reshard_after_forward):
        model, ref_model = self._init_models(
            reshard_after_forward=reshard_after_forward
        )
        ref_optim = torch.optim.AdamW(ref_model.parameters(), lr=1e-2, foreach=True)
        optim = torch.optim.AdamW(model.parameters(), lr=1e-2, foreach=True)
        torch.manual_seed(42 + self.rank)
        inp = torch.randn(4, DIM, device=device_type)
        for _ in range(4):
            ref_loss = ref_model(inp).sum()
            loss = model(inp).sum()
            self.assertEqual(loss, ref_loss)
            ref_loss.backward()
            loss.backward()
            for param in ref_model.parameters():
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
            for name in ("w1", "w2"):
                grad = getattr(model.experts, name).grad
                self.assertEqual(
                    grad.placements, getattr(model.experts, name).placements
                )
                self.assertEqual(
                    grad.full_tensor(), getattr(ref_model.experts, name).grad
                )
            ref_norm = torch.nn.utils.clip_grad_norm_(ref_model.parameters(), 1.0)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            self.assertEqual(norm.full_tensor(), ref_norm)
            ref_optim.step()
            optim.step()
            ref_optim.zero_grad()
            optim.zero_grad()
        for param, ref_param in zip(model.parameters(), ref_model.parameters()):
            self.assertEqual(param.full_tensor(), ref_param)

    @skip_if_lt_x_gpu(4)
    def test_train_parity_hsdp(self):
        mesh = init_device_mesh(
            device_type.type, (2, 2), mesh_dim_names=("replicate", "shard")
        )
        model, ref_model = self._init_models(mesh=mesh)
        self.assertEqual(model.experts.w1.placements, (Replicate(), BlockShard((DIM,))))
        ref_optim = torch.optim.AdamW(ref_model.parameters(), lr=1e-2)
        optim = torch.optim.AdamW(model.parameters(), lr=1e-2)
        torch.manual_seed(42 + self.rank)
        inp = torch.randn(4, DIM, device=device_type)
        for _ in range(3):
            ref_loss = ref_model(inp).sum()
            loss = model(inp).sum()
            self.assertEqual(loss, ref_loss)
            ref_loss.backward()
            loss.backward()
            for param in ref_model.parameters():
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
            ref_optim.step()
            optim.step()
            ref_optim.zero_grad()
            optim.zero_grad()
        for param, ref_param in zip(model.parameters(), ref_model.parameters()):
            self.assertEqual(param.full_tensor(), ref_param)

    @skip_if_lt_x_gpu(4)
    def test_meta_init(self):
        # Init after sharding draws the same values per element as Shard(0) on
        # the merged view, so ranks don't repeat values.
        with torch.device("meta"):
            model = MoEModel()
        fully_shard(model.experts, shard_placement_fn=_block_shard_placement_fn)
        fully_shard(model)
        model.to_empty(device=device_type)
        mesh = model.experts.w1.device_mesh
        for name in ("w1", "w2"):
            param = getattr(model.experts, name)
            merged_shape = param.placements[0]._merged_shape(param.shape)
            merged = distribute_tensor(
                torch.zeros(merged_shape, device=device_type), mesh, [Shard(0)]
            )
            torch.manual_seed(3)
            torch.nn.init.trunc_normal_(param)
            torch.manual_seed(3)
            torch.nn.init.trunc_normal_(merged)
            self.assertEqual(param.to_local(), merged.to_local())

    @skip_if_lt_x_gpu(4)
    def test_no_reorder_copies(self):
        # BlockShard all-gather and reduce-scatter use the dim-0 path, so no
        # chunk-cat reassembly or gradient reordering is needed.
        model, _ = self._init_models()
        outer_sizes, grad_shapes, shard_dims = [], [], []

        def output_fn(out, outputs, split_sizes, sizes, world_size):
            outer_sizes.extend(sizes)
            _default_all_gather_output_fn(out, outputs, split_sizes, sizes, world_size)

        def reduce_scatter_input_fn(grads, dims, world_size):
            grad_shapes.extend(grad.shape for grad in grads)
            shard_dims.extend(dims)
            return _default_reduce_scatter_input_fn(grads, dims, world_size)

        model.experts.set_all_gather_layout(DefaultAllGatherLayout(output_fn))
        model.experts.set_reduce_scatter_input_fn(reduce_scatter_input_fn)
        model(torch.randn(4, DIM, device=device_type)).sum().backward()
        self.assertEqual(set(outer_sizes), {1})
        # Reduce-scatter input functions get each gradient as merged rows
        self.assertEqual(shard_dims, [0, 0])
        merged = [(NUM_EXPERTS * HIDDEN, DIM), (NUM_EXPERTS * DIM, HIDDEN)]
        self.assertEqual(grad_shapes, merged)

    @skip_if_lt_x_gpu(4)
    def test_state_dict_round_trip(self):
        model, ref_model = self._init_models()
        state_dict = model.state_dict()
        w1 = state_dict["experts.w1"]
        self.assertEqual(w1.placements, (BlockShard((DIM,)),))
        self.assertEqual(w1.full_tensor(), ref_model.experts.w1)

        new_model, _ = self._init_models()
        with torch.no_grad():
            for param in new_model.parameters():
                param.zero_()
        new_model.load_state_dict(state_dict)
        for param, ref_param in zip(new_model.parameters(), ref_model.parameters()):
            self.assertEqual(param.full_tensor(), ref_param)
        inp = torch.randn(4, DIM, device=device_type)
        self.assertEqual(new_model(inp), ref_model(inp))

    @skip_if_lt_x_gpu(4)
    def test_full_state_dict_broadcast(self):
        model, ref_model = self._init_models()
        with torch.no_grad():
            for param in model.parameters():
                param.zero_()
        full_state_dict = ref_model.state_dict() if self.rank == 0 else {}
        set_model_state_dict(
            model,
            full_state_dict,
            options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True),
        )
        for param, ref_param in zip(model.parameters(), ref_model.parameters()):
            self.assertEqual(param.full_tensor(), ref_param)
        gathered = get_model_state_dict(
            model, options=StateDictOptions(full_state_dict=True)
        )
        if self.rank == 0:
            for name, value in ref_model.state_dict().items():
                self.assertEqual(gathered[name], value)

    @skip_if_lt_x_gpu(4)
    @with_temp_dir
    def test_dcp_round_trip(self):
        model, ref_model = self._init_models()
        dcp.save(get_model_state_dict(model), checkpoint_id=self.temp_dir)
        new_model, _ = self._init_models()
        with torch.no_grad():
            for param in new_model.parameters():
                param.zero_()
        state_dict = get_model_state_dict(new_model)
        dcp.load(state_dict, checkpoint_id=self.temp_dir)
        set_model_state_dict(new_model, state_dict)
        for param, ref_param in zip(new_model.parameters(), ref_model.parameters()):
            self.assertEqual(param.full_tensor(), ref_param)


# Expert parallelism on the experts dim, then FSDP BlockShard on the merged rows
# of the local experts, laid out like torchtitan's sparse mesh (efsdp, ep). Each
# EP rank holds 3 experts, so efsdp rank rows cross expert boundaries.
EP_NUM_EXPERTS = 6


class EPExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(EP_NUM_EXPERTS, HIDDEN, DIM) * 0.1)
        self.w2 = nn.Parameter(torch.randn(EP_NUM_EXPERTS, DIM, HIDDEN) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.bmm(x, self.w1.transpose(1, 2)).relu()
        return torch.bmm(h, self.w2.transpose(1, 2))


class TestFullyShardBlockShardExpertParallel(FSDPTestContinuous):
    world_size = 4

    _ep_mesh = None

    def _mesh(self):
        # FSDP2's spmd-mesh path requires the same mesh object across
        # fully_shard calls in a process, and worker processes are reused
        # across tests, so build the mesh once.
        if TestFullyShardBlockShardExpertParallel._ep_mesh is None:
            TestFullyShardBlockShardExpertParallel._ep_mesh = init_device_mesh(
                device_type.type, (2, 2), mesh_dim_names=("efsdp", "ep")
            )
        return TestFullyShardBlockShardExpertParallel._ep_mesh

    def _init_models(self, mesh, reshard_after_forward=True):
        torch.manual_seed(42)
        ref_model = EPExperts().to(device_type)
        model = copy.deepcopy(ref_model)
        for name, param in list(model.named_parameters()):
            dparam = distribute_tensor(param.detach(), mesh, [Replicate(), Shard(0)])
            setattr(model, name, nn.Parameter(dparam))
        fully_shard(
            model,
            mesh=mesh,
            dp_mesh_dims=DataParallelMeshDims(shard="efsdp"),
            shard_placement_fn=_block_shard_placement_fn,
            reshard_after_forward=reshard_after_forward,
        )
        return model, ref_model

    def _input(self, mesh):
        # Same tokens on every efsdp rank, so the averaged gradients match the
        # reference gradients.
        torch.manual_seed(7)
        x = torch.randn(EP_NUM_EXPERTS, 4, DIM, device=device_type)
        return x, distribute_tensor(x, mesh, [Replicate(), Shard(0)])

    @skip_if_lt_x_gpu(4)
    def test_init(self):
        mesh = self._mesh()
        model, ref_model = self._init_models(mesh)
        for name in ("w1", "w2"):
            param, ref_param = getattr(model, name), getattr(ref_model, name)
            self.assertEqual(
                param.placements, (BlockShard((ref_param.shape[2],)), Shard(0))
            )
            self.assertEqual(param.full_tensor(), ref_param)
            # Rank rows come from the local experts of this rank's EP group.
            efsdp_rank, ep_rank = mesh.get_coordinate()
            local_experts = EP_NUM_EXPERTS // 2
            rows = ref_param[ep_rank * local_experts : (ep_rank + 1) * local_experts]
            rows = rows.reshape(-1, ref_param.shape[2])
            q = -(-rows.shape[0] // 2)
            self.assertEqual(
                param.to_local(), rows[efsdp_rank * q : (efsdp_rank + 1) * q]
            )

    @skip_if_lt_x_gpu(4)
    @parametrize("reshard_after_forward", [True, False])
    def test_train_parity(self, reshard_after_forward):
        mesh = self._mesh()
        model, ref_model = self._init_models(mesh, reshard_after_forward)
        ref_optim = torch.optim.AdamW(ref_model.parameters(), lr=1e-2, foreach=True)
        optim = torch.optim.AdamW(model.parameters(), lr=1e-2, foreach=True)
        x, dx = self._input(mesh)
        for _ in range(4):
            ref_loss = ref_model(x).sum()
            loss = model(dx).sum()
            self.assertEqual(loss.full_tensor(), ref_loss)
            ref_loss.backward()
            loss.backward()
            for name in ("w1", "w2"):
                grad = getattr(model, name).grad
                self.assertEqual(grad.placements, getattr(model, name).placements)
                self.assertEqual(grad.full_tensor(), getattr(ref_model, name).grad)
            ref_norm = torch.nn.utils.clip_grad_norm_(ref_model.parameters(), 1.0)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            self.assertEqual(norm.full_tensor(), ref_norm)
            ref_optim.step()
            optim.step()
            ref_optim.zero_grad()
            optim.zero_grad()
        for param, ref_param in zip(model.parameters(), ref_model.parameters()):
            self.assertEqual(param.full_tensor(), ref_param)

    @skip_if_lt_x_gpu(4)
    @with_temp_dir
    def test_checkpoint(self):
        mesh = self._mesh()
        model, ref_model = self._init_models(mesh)
        dcp.save(get_model_state_dict(model), checkpoint_id=self.temp_dir)
        new_model, _ = self._init_models(mesh)
        with torch.no_grad():
            for param in new_model.parameters():
                param.zero_()
        state_dict = get_model_state_dict(new_model)
        dcp.load(state_dict, checkpoint_id=self.temp_dir)
        set_model_state_dict(new_model, state_dict)
        for param, ref_param in zip(new_model.parameters(), ref_model.parameters()):
            self.assertEqual(param.full_tensor(), ref_param)

        with torch.no_grad():
            for param in new_model.parameters():
                param.zero_()
        full_state_dict = ref_model.state_dict() if self.rank == 0 else {}
        set_model_state_dict(
            new_model,
            full_state_dict,
            options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True),
        )
        for param, ref_param in zip(new_model.parameters(), ref_model.parameters()):
            self.assertEqual(param.full_tensor(), ref_param)


instantiate_parametrized_tests(TestFullyShardBlockShard)
instantiate_parametrized_tests(TestFullyShardBlockShardExpertParallel)


if __name__ == "__main__":
    run_tests()
