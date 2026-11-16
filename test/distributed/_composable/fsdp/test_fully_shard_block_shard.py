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
from torch.distributed.fsdp._flex_shard.all_gather_layout import (
    _default_all_gather_output_fn,
)
from torch.distributed.fsdp._flex_shard.placements.shard import (
    reduce_scatter_input_fn as default_reduce_scatter_input_fn,
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
    HardwareClassification,
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
    def __init__(self, num_experts: int = NUM_EXPERTS):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(num_experts, HIDDEN, DIM) * 0.1)
        self.w2 = nn.Parameter(torch.randn(num_experts, DIM, HIDDEN) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _experts_forward(x, self.w1, self.w2)


class MergedExperts(nn.Module):
    """``Experts`` that store each weight as its merged view ``(E * O, I)``."""

    def __init__(self, experts: Experts):
        super().__init__()
        self.w1 = nn.Parameter(experts.w1.detach().flatten(0, 1).clone())
        self.w2 = nn.Parameter(experts.w2.detach().flatten(0, 1).clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1 = self.w1.view(NUM_EXPERTS, HIDDEN, DIM)
        w2 = self.w2.view(NUM_EXPERTS, DIM, HIDDEN)
        return _experts_forward(x, w1, w2)


def _experts_forward(
    x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
) -> torch.Tensor:
    h = torch.bmm(x, w1.transpose(1, 2)).relu()
    return torch.bmm(h, w2.transpose(1, 2))


class MoEModel(nn.Module):
    def __init__(self, experts: nn.Module):
        super().__init__()
        self.in_proj = nn.Linear(DIM, DIM)
        self.experts = experts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        return self.experts(h.unsqueeze(0).expand(NUM_EXPERTS, -1, -1)).sum(0)


def _block_shard_placement_fn(param: nn.Parameter) -> BlockShard | None:
    if param.ndim == 3:
        return BlockShard.split_leading(param.shape, 2)
    return None


class TestFullyShardBlockShard(FSDPTestContinuous):
    hw_classification = HardwareClassification.CUDA
    world_size = 4

    def _init_models(self, mesh=None, reshard_after_forward=True):
        """The BlockShard model, a model that keeps the merged view of each
        expert weight with FSDP's default Shard(0), and an unsharded reference."""
        torch.manual_seed(42)
        ref_model = MoEModel(Experts()).to(device_type)
        model = copy.deepcopy(ref_model)
        merged_model = copy.deepcopy(ref_model)
        merged_model.experts = MergedExperts(ref_model.experts)
        for module, placement_fn in (
            (model, _block_shard_placement_fn),
            (merged_model, None),
        ):
            fully_shard(
                module.experts,
                mesh=mesh,
                shard_placement_fn=placement_fn,
                reshard_after_forward=reshard_after_forward,
            )
            fully_shard(module, mesh=mesh, reshard_after_forward=reshard_after_forward)
        return model, merged_model, ref_model

    def _check_train_parity(self, model, merged_model, ref_model):
        # BlockShard is Shard(0) on the merged view, so it matches the merged
        # model bitwise: the same shards, collectives and optimizer steps
        models = (model, merged_model, ref_model)
        optims = [
            torch.optim.AdamW(module.parameters(), lr=1e-2, foreach=True)
            for module in models
        ]
        torch.manual_seed(42 + self.rank)
        inp = torch.randn(4, DIM, device=device_type)
        for _ in range(4):
            loss, merged_loss, ref_loss = (module(inp).sum() for module in models)
            self.assertEqual(loss, ref_loss)
            self.assertEqual(loss, merged_loss, atol=0, rtol=0)
            for module_loss in (loss, merged_loss, ref_loss):
                module_loss.backward()
            for param in ref_model.parameters():
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
            for name in ("w1", "w2"):
                param = getattr(model.experts, name)
                self.assertEqual(param.grad.placements, param.placements)
                ref_grad = getattr(ref_model.experts, name).grad
                self.assertEqual(param.grad.full_tensor(), ref_grad)
            for param, merged_param in zip(
                model.parameters(), merged_model.parameters(), strict=True
            ):
                self.assertEqual(
                    param.grad.to_local(), merged_param.grad.to_local(), atol=0, rtol=0
                )
            norm, merged_norm, ref_norm = (
                torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
                for module in models
            )
            self.assertEqual(norm.full_tensor(), ref_norm)
            self.assertEqual(
                norm.full_tensor(), merged_norm.full_tensor(), atol=0, rtol=0
            )
            for optim in optims:
                optim.step()
                optim.zero_grad()
        for param, merged_param, ref_param in zip(
            model.parameters(),
            merged_model.parameters(),
            ref_model.parameters(),
            strict=True,
        ):
            self.assertEqual(param.full_tensor(), ref_param)
            self.assertEqual(param.to_local(), merged_param.to_local(), atol=0, rtol=0)

    @skip_if_lt_x_gpu(4)
    def test_init(self):
        model, _, ref_model = self._init_models()
        for name in ("w1", "w2"):
            param = getattr(model.experts, name)
            ref_param = getattr(ref_model.experts, name)
            self.assertIsInstance(param, DTensor)
            self.assertEqual(param.placements, (BlockShard((ref_param.shape[2],)),))
            # Rank r owns merged rows [r * q, (r + 1) * q)
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
        self._check_train_parity(
            *self._init_models(reshard_after_forward=reshard_after_forward)
        )

    @skip_if_lt_x_gpu(4)
    def test_train_parity_hsdp(self):
        mesh = init_device_mesh(
            device_type.type, (2, 2), mesh_dim_names=("replicate", "shard")
        )
        model, merged_model, ref_model = self._init_models(mesh=mesh)
        self.assertEqual(model.experts.w1.placements, (Replicate(), BlockShard((DIM,))))
        self._check_train_parity(model, merged_model, ref_model)

    @skip_if_lt_x_gpu(4)
    def test_hsdp_accumulation_with_unused_weight(self):
        class ExpertsWithOptionalW2(Experts):
            def __init__(self):
                super().__init__()
                self.use_w2 = True

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                h = torch.bmm(x, self.w1.transpose(1, 2)).relu()
                return torch.bmm(h, self.w2.transpose(1, 2)) if self.use_w2 else h

        torch.manual_seed(42)
        ref_model = ExpertsWithOptionalW2().to(device_type)
        model = copy.deepcopy(ref_model)
        mesh = init_device_mesh(
            device_type.type, (2, 2), mesh_dim_names=("replicate", "shard")
        )
        fully_shard(model, mesh=mesh, shard_placement_fn=_block_shard_placement_fn)
        torch.manual_seed(42 + self.rank)
        for microbatch_idx in range(2):
            # HSDP accumulates reduce-scatter outputs until the all-reduce, so
            # w2's zero gradients in the first microbatch take the layout of its
            # merged view, as its gradients in the second do
            model.set_requires_all_reduce(microbatch_idx == 1)
            model.use_w2 = ref_model.use_w2 = microbatch_idx == 1
            inp = torch.randn(NUM_EXPERTS, 4, DIM, device=device_type)
            model(inp).sum().backward()
            ref_model(inp).sum().backward()
        for param in ref_model.parameters():
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
        for param, ref_param in zip(model.parameters(), ref_model.parameters()):
            self.assertEqual(param.grad.full_tensor(), ref_param.grad)

    @skip_if_lt_x_gpu(4)
    def test_meta_init(self):
        # Init after sharding draws the same values per element as Shard(0) on
        # the merged view, so ranks don't repeat values, and forward uses them
        with torch.device("meta"):
            model = MoEModel(Experts())
        fully_shard(model.experts, shard_placement_fn=_block_shard_placement_fn)
        fully_shard(model)
        model.to_empty(device=device_type)
        model.in_proj.reset_parameters()
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
        ref_model = MoEModel(Experts()).to(device_type)
        with torch.no_grad():
            for param, ref_param in zip(model.parameters(), ref_model.parameters()):
                ref_param.copy_(param.full_tensor())
        inp = torch.randn(4, DIM, device=device_type)
        self.assertEqual(model(inp), ref_model(inp))

    @skip_if_lt_x_gpu(4)
    def test_no_reorder_copies(self):
        # BlockShard all-gather and reduce-scatter use the dim-0 path, so no
        # chunk-cat reassembly or gradient reordering is needed
        model, _, _ = self._init_models()
        outer_sizes, grad_shapes, grad_shard_dims = [], [], []

        def output_fn(all_gather_output, outputs, **sizes):
            outer_sizes.extend(sizes["outer_sizes"])
            _default_all_gather_output_fn(all_gather_output, outputs, **sizes)

        def reduce_scatter_input_fn(unsharded_grads, shard_dims, world_size):
            grad_shapes.extend(grad.shape for grad in unsharded_grads)
            grad_shard_dims.extend(shard_dims)
            return default_reduce_scatter_input_fn(
                unsharded_grads, shard_dims=shard_dims, world_size=world_size
            )

        model.experts.set_all_gather_layout(DefaultAllGatherLayout(output_fn))
        model.experts.set_reduce_scatter_input_fn(reduce_scatter_input_fn)
        model(torch.randn(4, DIM, device=device_type)).sum().backward()
        self.assertEqual(set(outer_sizes), {1})
        # The reduce-scatter pads merged rows: it gets the merged views
        self.assertEqual(grad_shard_dims, [0, 0])
        merged = [(NUM_EXPERTS * HIDDEN, DIM), (NUM_EXPERTS * DIM, HIDDEN)]
        self.assertEqual(grad_shapes, merged)

    @skip_if_lt_x_gpu(4)
    def test_state_dict_round_trip(self):
        model, _, ref_model = self._init_models()
        state_dict = model.state_dict()
        w1 = state_dict["experts.w1"]
        self.assertEqual(w1.placements, (BlockShard((DIM,)),))
        self.assertEqual(w1.full_tensor(), ref_model.experts.w1)

        new_model, _, _ = self._init_models()
        with torch.no_grad():
            for param in new_model.parameters():
                param.to_local().zero_()
        new_model.load_state_dict(state_dict)
        for param, ref_param in zip(new_model.parameters(), ref_model.parameters()):
            self.assertEqual(param.full_tensor(), ref_param)
        inp = torch.randn(4, DIM, device=device_type)
        self.assertEqual(new_model(inp), ref_model(inp))

    @skip_if_lt_x_gpu(4)
    def test_full_state_dict_broadcast(self):
        model, _, ref_model = self._init_models()
        with torch.no_grad():
            for param in model.parameters():
                param.to_local().zero_()
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
        model, _, ref_model = self._init_models()
        dcp.save(get_model_state_dict(model), checkpoint_id=self.temp_dir)
        new_model, _, _ = self._init_models()
        with torch.no_grad():
            for param in new_model.parameters():
                param.to_local().zero_()
        state_dict = get_model_state_dict(new_model)
        dcp.load(state_dict, checkpoint_id=self.temp_dir)
        set_model_state_dict(new_model, state_dict)
        for param, ref_param in zip(new_model.parameters(), ref_model.parameters()):
            self.assertEqual(param.full_tensor(), ref_param)

    @skip_if_lt_x_gpu(4)
    def test_unsupported_compositions_raise(self):
        mesh = init_device_mesh(device_type.type, (2, 2), mesh_dim_names=("dp", "tp"))

        def distributed_experts(param_mesh, placements):
            experts = Experts().to(device_type)
            for name, param in list(experts.named_parameters()):
                dparam = distribute_tensor(param.detach(), param_mesh, placements)
                setattr(experts, name, nn.Parameter(dparam))
            return experts

        # Blocks that are not the trailing dims of the parameter
        with self.assertRaisesRegex(NotImplementedError, "not supported for shape"):
            fully_shard(
                Experts().to(device_type),
                shard_placement_fn=lambda param: BlockShard((HIDDEN,)),
            )
        # BlockShard composes with Shard(0), not with TP's Shard(1)
        with self.assertRaisesRegex(NotImplementedError, "at most one Shard"):
            fully_shard(
                distributed_experts(mesh["tp"], [Shard(1)]),
                mesh=mesh["dp"],
                shard_placement_fn=_block_shard_placement_fn,
            )
        # Several DP shard mesh dims
        with self.assertRaisesRegex(NotImplementedError, "exactly one DP shard"):
            fully_shard(
                distributed_experts(mesh, [Replicate(), Replicate()]),
                mesh=mesh,
                dp_mesh_dims=DataParallelMeshDims(shard=("dp", "tp")),
                shard_placement_fn=_block_shard_placement_fn,
            )


# Expert parallelism on the experts dim, then FSDP BlockShard on the merged rows
# of the local experts. Each EP rank holds 3 experts, so efsdp rank rows cross
# expert boundaries.
EP_NUM_EXPERTS = 6


class TestFullyShardBlockShardExpertParallel(FSDPTestContinuous):
    hw_classification = HardwareClassification.CUDA
    world_size = 4

    _cached_mesh = None

    def _mesh(self):
        # FSDP2's spmd-mesh path requires the parameters' mesh object to be the
        # one passed to fully_shard, and worker processes are reused across
        # tests, so build the mesh once
        if TestFullyShardBlockShardExpertParallel._cached_mesh is None:
            TestFullyShardBlockShardExpertParallel._cached_mesh = init_device_mesh(
                device_type.type, (2, 2), mesh_dim_names=("efsdp", "ep")
            )
        return TestFullyShardBlockShardExpertParallel._cached_mesh

    def _init_models(self, layout, reshard_after_forward=True):
        """The EP experts sharded with BlockShard and an unsharded reference,
        laid out as torchtitan does: EP DTensors on the full (efsdp, ep) mesh
        that FSDP shards over its efsdp dim (``"spmd_mesh"``), or EP DTensors
        on the ep mesh and FSDP on the efsdp mesh (``"separate_ep_mesh"``)."""
        mesh = self._mesh()
        if layout == "spmd_mesh":
            param_mesh, placements = mesh, [Replicate(), Shard(0)]
            fsdp_kwargs = {
                "mesh": mesh,
                "dp_mesh_dims": DataParallelMeshDims(shard="efsdp"),
            }
        else:
            param_mesh, placements = mesh["ep"], [Shard(0)]
            fsdp_kwargs = {"mesh": mesh["efsdp"]}
        torch.manual_seed(42)
        ref_model = Experts(EP_NUM_EXPERTS).to(device_type)
        model = copy.deepcopy(ref_model)
        for name, param in list(model.named_parameters()):
            dparam = distribute_tensor(param.detach(), param_mesh, placements)
            setattr(model, name, nn.Parameter(dparam))
        fully_shard(
            model,
            shard_placement_fn=_block_shard_placement_fn,
            reshard_after_forward=reshard_after_forward,
            **fsdp_kwargs,
        )
        # Same tokens on every efsdp rank, so the averaged gradients match the
        # reference gradients
        torch.manual_seed(7)
        x = torch.randn(EP_NUM_EXPERTS, 4, DIM, device=device_type)
        dx = distribute_tensor(x, param_mesh, placements)
        return model, ref_model, x, dx

    @skip_if_lt_x_gpu(4)
    @parametrize("layout", ["spmd_mesh", "separate_ep_mesh"])
    def test_init(self, layout):
        model, ref_model, _, _ = self._init_models(layout)
        efsdp_rank, ep_rank = self._mesh().get_coordinate()
        for name in ("w1", "w2"):
            param, ref_param = getattr(model, name), getattr(ref_model, name)
            self.assertEqual(
                param.placements, (BlockShard((ref_param.shape[2],)), Shard(0))
            )
            self.assertEqual(param.full_tensor(), ref_param)
            # Rank rows come from the local experts of this rank's EP group
            local_experts = EP_NUM_EXPERTS // 2
            rows = ref_param[ep_rank * local_experts : (ep_rank + 1) * local_experts]
            rows = rows.reshape(-1, ref_param.shape[2])
            q = -(-rows.shape[0] // 2)
            self.assertEqual(
                param.to_local(), rows[efsdp_rank * q : (efsdp_rank + 1) * q]
            )

    @skip_if_lt_x_gpu(4)
    @parametrize("layout", ["spmd_mesh", "separate_ep_mesh"])
    @parametrize("reshard_after_forward", [True, False])
    def test_train_parity(self, layout, reshard_after_forward):
        model, ref_model, x, dx = self._init_models(layout, reshard_after_forward)
        ref_optim = torch.optim.AdamW(ref_model.parameters(), lr=1e-2, foreach=True)
        optim = torch.optim.AdamW(model.parameters(), lr=1e-2, foreach=True)
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
    @parametrize("layout", ["spmd_mesh", "separate_ep_mesh"])
    @with_temp_dir
    def test_checkpoint(self, layout):
        model, ref_model, _, _ = self._init_models(layout)
        dcp.save(get_model_state_dict(model), checkpoint_id=self.temp_dir)
        new_model, _, _, _ = self._init_models(layout)
        with torch.no_grad():
            for param in new_model.parameters():
                param.to_local().zero_()
        state_dict = get_model_state_dict(new_model)
        dcp.load(state_dict, checkpoint_id=self.temp_dir)
        set_model_state_dict(new_model, state_dict)
        for param, ref_param in zip(new_model.parameters(), ref_model.parameters()):
            self.assertEqual(param.full_tensor(), ref_param)

        with torch.no_grad():
            for param in new_model.parameters():
                param.to_local().zero_()
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
