# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import copy
import os
from tempfile import TemporaryDirectory
from typing import Any, cast

import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.fsdp._flex_shard import (
    BucketSpec,
    flex_shard,
    get_global_layout,
    GlobalLayout,
    register_optimizer_checkpoint_hook,
    set_global_layout,
    set_state_dict_global_layouts,
)
from torch.distributed.fsdp._flex_shard.bucket_storage import ShardedBucketStorage
from torch.distributed.fsdp._flex_shard.flex_shard import (
    _attach_flex_shard_module_state,
)
from torch.distributed.fsdp._flex_shard.placement_contract import (
    compose_global_layouts,
    Placement,
)
from torch.distributed.fsdp._flex_shard.placements.block_shard import (
    BlockShard,
    BucketedBlockShard,
)
from torch.distributed.fsdp._flex_shard.placements.owned import (
    BucketedOwned,
    BucketedOwnedSegmentSpec,
)
from torch.distributed.fsdp._flex_shard.placements.shard import (
    per_param_placements,
    Shard,
)
from torch.distributed.tensor import distribute_tensor, DTensor
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, get_devtype
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir
from torch.testing._internal.distributed.common_flex_shard import (
    expected_shard,
    single_rank_cpu_mesh,
)


device_type = torch.device(get_devtype())


class _Mesh:
    def __init__(self, rank: int, world_size: int) -> None:
        self._rank = rank
        self._world_size = world_size

    def get_local_rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._world_size


def _global_layouts(
    named_params: list[tuple[str, nn.Parameter]],
    placements: dict[str, tuple[Placement, ...]],
    rank: int,
    world_size: int,
) -> dict[str, GlobalLayout]:
    infos, _ = ShardedBucketStorage.create_param_infos(
        named_params, cast(Any, _Mesh(rank, world_size)), placements
    )
    return {
        fqn: info.placement.global_layout(info, rank, world_size)
        for fqn, info in infos.items()
    }


def _single_param_layout(
    placement: Placement, shape: tuple[int, ...], rank: int, world_size: int
) -> GlobalLayout:
    return _global_layouts(
        [("weight", nn.Parameter(torch.empty(shape)))],
        {"weight": (placement,)},
        rank,
        world_size,
    )["weight"]


def _shard_model(
    model: nn.Module,
    placement: Placement,
    mesh: Any,
    shared_names: dict[str, list[str]] | None = None,
) -> None:
    named_params = list(model.named_parameters())
    storage = ShardedBucketStorage.from_bucket(
        model,
        named_params,
        {fqn: (placement,) for fqn, _ in named_params},
        mesh,
        torch.device("cpu"),
        BucketSpec(
            ["*"],
            placement_fn=lambda named_params, mesh: {
                fqn: (placement,) for fqn, _ in named_params
            },
            mesh=mesh,
            reshard_after_forward=False,
        ),
        shared_names,
    )
    _attach_flex_shard_module_state(model, [storage])


def _checkpoint_state_dict(model: nn.Module) -> dict[str, Any]:
    state_dict = model.state_dict()
    set_state_dict_global_layouts(model, state_dict)
    return state_dict


class TestFlexShardGlobalLayout(TestCase):
    def test_shard_uneven_and_empty_ranks(self) -> None:
        self.assertEqual(
            _single_param_layout(Shard(0), (5, 4), 1, 2),
            GlobalLayout((5, 4), ((3, 0),), ((0, 0),), ((2, 4),)),
        )
        self.assertEqual(
            _single_param_layout(Shard(0), (1, 4), 2, 3),
            GlobalLayout((1, 4), (), (), ()),
        )

    def test_block_shard(self) -> None:
        placement = BlockShard(blocks_per_rank=(2, 0, 1), dim=1)
        self.assertEqual(
            _single_param_layout(placement, (2, 6), 2, 3),
            GlobalLayout((2, 6), ((0, 4),), ((0, 0),), ((2, 2),)),
        )

    def test_bucketed_owned(self) -> None:
        placement = BucketedOwned(
            {
                "slabs": [
                    BucketedOwnedSegmentSpec(f"slabs#{row}", "slabs", 4 * row, 4, owner)
                    for row, owner in enumerate((0, 0, 1))
                ],
            }
        )
        named_params = [("slabs", nn.Parameter(torch.empty(3, 4)))]
        placements = {"slabs": (placement,)}
        self.assertEqual(
            _global_layouts(named_params, placements, 1, 2)["slabs"],
            GlobalLayout((3, 4), ((2, 0),), ((0, 0),), ((1, 4),)),
        )

    def test_bucketed_block_shard(self) -> None:
        placement = BucketedBlockShard(dims=(0,), blocks_per_rank=(1, 1))
        named_params = [
            ("prefix", nn.Parameter(torch.empty(2, 3))),
            ("weight", nn.Parameter(torch.empty(4, 3))),
        ]
        placements = {fqn: (placement,) for fqn, _ in named_params}
        self.assertEqual(
            _global_layouts(named_params, placements, 1, 2)["weight"],
            GlobalLayout((4, 3), ((1, 0),), ((0, 0),), ((3, 3),)),
        )

    def test_compose_multi_chunk(self) -> None:
        # The intermediate tensor stacks global rows [1, 3) and [5, 7).
        outer = GlobalLayout(
            global_shape=(8, 6),
            global_offsets=((1, 0), (5, 0)),
            local_offsets=((0, 0), (2, 0)),
            local_sizes=((2, 6), (2, 6)),
        )
        # The local tensor holds intermediate row 1 and a column slab of rows 2-3.
        inner = GlobalLayout(
            global_shape=(4, 6),
            global_offsets=((1, 0), (2, 2)),
            local_offsets=((0, 0), (1, 2)),
            local_sizes=((1, 6), (2, 4)),
        )
        self.assertEqual(
            compose_global_layouts(outer, inner),
            GlobalLayout(
                global_shape=(8, 6),
                global_offsets=((2, 0), (5, 2)),
                local_offsets=((0, 0), (1, 2)),
                local_sizes=((1, 6), (2, 4)),
            ),
        )


class TestFlexShardCheckpointStateDict(TestCase):
    def test_dcp_round_trip_of_checkpoint_wrapped_submodule(self) -> None:
        expected = torch.arange(20, dtype=torch.float32).view(5, 4)
        linear = nn.Linear(4, 5, bias=False)
        with torch.no_grad():
            linear.weight.copy_(expected)
        model = nn.Module()
        model.layer = checkpoint_wrapper(linear)
        with single_rank_cpu_mesh() as mesh:
            _shard_model(model.layer, Shard(0), mesh)
            state_dict = _checkpoint_state_dict(model)
            self.assertEqual(
                get_global_layout(state_dict["layer.weight"]),
                GlobalLayout((5, 4), ((0, 0),), ((0, 0),), ((5, 4),)),
            )

            with TemporaryDirectory() as checkpoint_dir:
                dcp.save(state_dict, checkpoint_id=checkpoint_dir, no_dist=True)
                next(model.parameters()).detach().fill_(-1)
                load_state_dict = _checkpoint_state_dict(model)
                dcp.load(load_state_dict, checkpoint_id=checkpoint_dir, no_dist=True)
                model.load_state_dict(load_state_dict)

            self.assertEqual(next(model.parameters()), expected)

    def test_dcp_round_trip_of_tied_weights(self) -> None:
        # state_dict() lists both names of a tied weight, each as its own
        # detached tensor. Both carry the layout, so DCP saves and loads each
        # as the shard it is, and the weight stays shared.
        expected = torch.arange(20, dtype=torch.float32).view(5, 4)
        model = nn.Module()
        model.embed = nn.Embedding(5, 4)
        model.head = nn.Linear(4, 5, bias=False)
        model.head.weight = model.embed.weight
        with torch.no_grad():
            model.embed.weight.copy_(expected)
        with single_rank_cpu_mesh() as mesh:
            _shard_model(
                model, Shard(0), mesh, shared_names={"embed.weight": ["head.weight"]}
            )
            self.assertIs(model.head.weight, model.embed.weight)
            state_dict = _checkpoint_state_dict(model)
            layout = GlobalLayout((5, 4), ((0, 0),), ((0, 0),), ((5, 4),))
            for key in ("embed.weight", "head.weight"):
                self.assertEqual(get_global_layout(state_dict[key]), layout)

            with TemporaryDirectory() as checkpoint_dir:
                dcp.save(state_dict, checkpoint_id=checkpoint_dir, no_dist=True)
                model.embed.weight.detach().fill_(-1)
                load_state_dict = _checkpoint_state_dict(model)
                dcp.load(load_state_dict, checkpoint_id=checkpoint_dir, no_dist=True)
                model.load_state_dict(load_state_dict)

            self.assertIs(model.head.weight, model.embed.weight)
            self.assertEqual(model.embed.weight, expected)

    def test_state_dict_composes_declared_outer_layout(self) -> None:
        model = nn.Module()
        model.weight = nn.Parameter(torch.empty(4, 3))
        layout = GlobalLayout((8, 3), ((4, 0),), ((0, 0),), ((4, 3),))
        set_global_layout(model.weight, layout)
        with single_rank_cpu_mesh() as mesh:
            _shard_model(model, Shard(0), mesh)
            state_dict = _checkpoint_state_dict(model)
        self.assertEqual(get_global_layout(state_dict["weight"]), layout)

    def test_unsupported_placement_raises(self) -> None:
        model = nn.Module()
        model.weight = nn.Parameter(torch.arange(24.0).view(2, 3, 4))
        with single_rank_cpu_mesh() as mesh:
            _shard_model(model, BucketedBlockShard(dims=(0, 1)), mesh)
            with self.assertRaisesRegex(
                NotImplementedError, "distributed checkpointing supports only dims"
            ):
                _checkpoint_state_dict(model)


def _adam_step(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    for param in model.parameters():
        param.grad = torch.ones_like(param)
    optimizer.step()


class TestFlexShardOptimizerCheckpointHook(TestCase):
    def test_states_take_their_parameters_layouts(self) -> None:
        model = nn.Module()
        model.weight = nn.Parameter(torch.arange(12.0).view(4, 3))
        model.bias = nn.Parameter(torch.arange(4.0))
        # The weight is rows [4, 8) of an outer (e.g. expert-parallel) sharding.
        weight_layout = GlobalLayout((8, 3), ((4, 0),), ((0, 0),), ((4, 3),))
        set_global_layout(model.weight, weight_layout)
        with single_rank_cpu_mesh() as mesh:
            _shard_model(model, Shard(0), mesh)
            optimizer = torch.optim.Adam(model.parameters(), amsgrad=True)
            register_optimizer_checkpoint_hook(optimizer, model)
            _adam_step(model, optimizer)
            state = optimizer.state_dict()["state"]
        bias_layout = GlobalLayout((4,), ((0,),), ((0,),), ((4,),))
        for param_id, layout in ((0, weight_layout), (1, bias_layout)):
            for state_name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                self.assertEqual(get_global_layout(state[param_id][state_name]), layout)
            self.assertIsNone(get_global_layout(state[param_id]["step"]))

    def test_dcp_saves_states_as_chunks_of_the_full_state(self) -> None:
        model = nn.Module()
        model.weight = nn.Parameter(torch.arange(12.0).view(4, 3))
        set_global_layout(
            model.weight, GlobalLayout((8, 3), ((4, 0),), ((0, 0),), ((4, 3),))
        )
        with single_rank_cpu_mesh() as mesh:
            _shard_model(model, Shard(0), mesh)
            optimizer = torch.optim.AdamW(model.parameters())
            register_optimizer_checkpoint_hook(optimizer, model)
            _adam_step(model, optimizer)
            expected = copy.deepcopy(optimizer.state_dict()["state"])
            with TemporaryDirectory() as checkpoint_dir:
                dcp.save(
                    {"optim": optimizer.state_dict()},
                    checkpoint_id=checkpoint_dir,
                    no_dist=True,
                )
                metadata = dcp.FileSystemReader(checkpoint_dir).read_metadata()
                _adam_step(model, optimizer)
                state_dict = {"optim": optimizer.state_dict()}
                dcp.load(state_dict, checkpoint_id=checkpoint_dir, no_dist=True)
                optimizer.load_state_dict(state_dict["optim"])
        for state_name in ("exp_avg", "exp_avg_sq"):
            tensor_metadata = metadata.state_dict_metadata[
                f"optim.state.0.{state_name}"
            ]
            self.assertEqual(tensor_metadata.size, torch.Size((8, 3)))
            self.assertEqual(
                [chunk.offsets for chunk in tensor_metadata.chunks],
                [torch.Size((4, 0))],
            )
        self.assertEqual(
            metadata.state_dict_metadata["optim.state.0.step"].size, torch.Size(())
        )
        self.assertEqual(optimizer.state_dict()["state"], expected, atol=0, rtol=0)

    def test_unsupported_optimizer_raises(self) -> None:
        model = nn.Linear(4, 5)
        with single_rank_cpu_mesh() as mesh:
            _shard_model(model, Shard(0), mesh)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
            with self.assertRaisesRegex(
                NotImplementedError, "supports Adam and AdamW, got SGD"
            ):
                register_optimizer_checkpoint_hook(optimizer, model)

    def test_parameter_outside_module_raises(self) -> None:
        model = nn.Module()
        model.first = nn.Linear(4, 5)
        model.second = nn.Linear(4, 5)
        with single_rank_cpu_mesh() as mesh:
            _shard_model(model.first, Shard(0), mesh)
            _shard_model(model.second, Shard(0), mesh)
            optimizer = torch.optim.Adam(model.parameters())
            with self.assertRaisesRegex(
                ValueError, "not a sharded parameter of the module"
            ):
                register_optimizer_checkpoint_hook(optimizer, model.first)

    def test_state_shape_mismatch_raises(self) -> None:
        model = nn.Linear(4, 5)
        with single_rank_cpu_mesh() as mesh:
            _shard_model(model, Shard(0), mesh)
            optimizer = torch.optim.Adam(model.parameters())
            register_optimizer_checkpoint_hook(optimizer, model)
            _adam_step(model, optimizer)
            optimizer.state[model.weight]["exp_avg"] = torch.zeros(1)
            with self.assertRaisesRegex(
                NotImplementedError,
                r"'exp_avg' of FlexShard parameter 'weight' has shape \(1,\)",
            ):
                optimizer.state_dict()

    def test_registering_again_replaces_the_hook(self) -> None:
        model = nn.Linear(4, 5)
        with single_rank_cpu_mesh() as mesh:
            _shard_model(model, Shard(0), mesh)
            optimizer = torch.optim.Adam(model.parameters())
            register_optimizer_checkpoint_hook(optimizer, model)
            register_optimizer_checkpoint_hook(optimizer, model)
        self.assertEqual(len(optimizer._optimizer_state_dict_post_hooks), 1)


def _uneven_model() -> nn.Module:
    # Shard(0) splits the 5 rows as 2, 2, 1, 0 over four ranks and as 3, 2
    # over two; the 3 rows as 1, 1, 1, 0 and as 2, 1.
    return nn.Sequential(nn.Linear(8, 5), nn.ReLU(), nn.Linear(5, 3)).to(device_type)


def _adamw(model: nn.Module) -> torch.optim.Optimizer:
    # Single-tensor AdamW, so local shards match the reference bit for bit.
    return torch.optim.AdamW(
        model.parameters(), lr=0.1, weight_decay=0.1, foreach=False
    )


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


# A model and its optimizer.
_Trained = tuple[nn.Module, torch.optim.Optimizer]


class _OptimizerCheckpointRoundTrip(FSDPTest):
    """Saves a model with AdamW state under one backend and split, and loads it
    under another, checking both against an unsharded reference.

    A split over ``num_shards`` ranks shards over them and replicates over the
    other ranks.
    """

    def _mesh(self, num_shards: int):
        return init_device_mesh(
            device_type.type,
            (self.world_size // num_shards, num_shards),
            mesh_dim_names=("replicate", "shard"),
        )["shard"]

    def _shard(self, backend: str, model: nn.Module, mesh) -> torch.optim.Optimizer:
        if backend == "fsdp2":
            fully_shard(model, mesh=mesh)
        else:
            flex_shard(
                model,
                buckets=[
                    BucketSpec(["*"], placement_fn=per_param_placements, mesh=mesh)
                ],
            )
        optimizer = _adamw(model)
        if backend == "flex_shard":
            register_optimizer_checkpoint_hook(optimizer, model)
        return optimizer

    def _set_grads(self, model: nn.Module, grads: list[torch.Tensor], mesh) -> None:
        for param, grad in zip(model.parameters(), grads, strict=True):
            if isinstance(param, DTensor):
                param.grad = distribute_tensor(grad, mesh, param.placements)
            else:
                param.grad = expected_shard(
                    grad, rank=mesh.get_local_rank(), world_size=mesh.size()
                )

    def _step(self, reference: _Trained, sharded: _Trained, mesh, seed: int) -> None:
        """Step both with the same random gradients."""
        generator = torch.Generator().manual_seed(seed)
        grads = [
            torch.randn(param.shape, generator=generator).to(device_type)
            for param in reference[0].parameters()
        ]
        for param, grad in zip(reference[0].parameters(), grads, strict=True):
            param.grad = grad
        reference[1].step()
        self._set_grads(sharded[0], grads, mesh)
        sharded[1].step()

    def _check(self, reference: _Trained, sharded: _Trained, mesh) -> None:
        """Check the local parameters and states against the reference's, bit for bit."""
        rank, world_size = mesh.get_local_rank(), mesh.size()
        for reference_param, param in zip(
            reference[0].parameters(), sharded[0].parameters(), strict=True
        ):
            reference_state = reference[1].state[reference_param]
            state = sharded[1].state[param]
            self.assertEqual(state.keys(), reference_state.keys())
            self.assertEqual(state["step"], reference_state["step"])
            for full, local in (
                (reference_param.detach(), param.detach()),
                *(
                    (reference_state[name], state[name])
                    for name in state
                    if name != "step"
                ),
            ):
                self.assertEqual(
                    _local(local),
                    expected_shard(full, rank=rank, world_size=world_size),
                    atol=0,
                    rtol=0,
                )

    def _checkpoint(self, backend: str, trained: _Trained) -> dict[str, Any]:
        model, optimizer = trained
        model_state_dict = model.state_dict()
        if backend == "flex_shard":
            set_state_dict_global_layouts(model, model_state_dict)
        return {"model": model_state_dict, "optim": optimizer.state_dict()}

    def _round_trip(
        self,
        save: tuple[str, int],
        load: tuple[str, int],
        meshes: dict[int, Any],
        checkpoint_dir: str,
    ) -> None:
        (save_backend, save_shards), (load_backend, load_shards) = save, load
        torch.manual_seed(0)
        reference_model = _uneven_model()
        model = copy.deepcopy(reference_model)
        reference = (reference_model, _adamw(reference_model))
        saved = (model, self._shard(save_backend, model, meshes[save_shards]))
        for seed in (1, 2):
            self._step(reference, saved, meshes[save_shards], seed)
        self._check(reference, saved, meshes[save_shards])
        dcp.save(self._checkpoint(save_backend, saved), checkpoint_id=checkpoint_dir)

        torch.manual_seed(1)
        model = _uneven_model()
        loaded = (model, self._shard(load_backend, model, meshes[load_shards]))
        # Create the states that DCP loads into.
        self._set_grads(
            model,
            [torch.zeros_like(param) for param in reference_model.parameters()],
            meshes[load_shards],
        )
        loaded[1].step()
        state_dict = self._checkpoint(load_backend, loaded)
        dcp.load(state_dict, checkpoint_id=checkpoint_dir)
        model.load_state_dict(state_dict["model"])
        loaded[1].load_state_dict(state_dict["optim"])
        self._check(reference, loaded, meshes[load_shards])
        # Training continues as it would have without the round trip.
        self._step(reference, loaded, meshes[load_shards], seed=3)
        self._check(reference, loaded, meshes[load_shards])

    def _run_round_trips(self) -> None:
        all_shards, half_shards = self.world_size, self.world_size // 2
        meshes = {
            num_shards: self._mesh(num_shards)
            for num_shards in (all_shards, half_shards)
        }
        for save, load in (
            (("flex_shard", all_shards), ("flex_shard", all_shards)),
            (("flex_shard", all_shards), ("flex_shard", half_shards)),
            (("flex_shard", half_shards), ("flex_shard", all_shards)),
            (("fsdp2", all_shards), ("flex_shard", all_shards)),
            (("flex_shard", all_shards), ("fsdp2", all_shards)),
            (("fsdp2", all_shards), ("flex_shard", half_shards)),
        ):
            with self.subTest(save=save, load=load):
                checkpoint_dir = os.path.join(
                    self.temp_dir, f"{save[0]}_{save[1]}_to_{load[0]}_{load[1]}"
                )
                self._round_trip(save, load, meshes, checkpoint_dir)


class TestFlexShardOptimizerCheckpointTwoRanks(_OptimizerCheckpointRoundTrip):
    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    @with_temp_dir
    def test_dcp_round_trip(self) -> None:
        self._run_round_trips()


class TestFlexShardOptimizerCheckpointFourRanks(_OptimizerCheckpointRoundTrip):
    @property
    def world_size(self) -> int:
        return 4

    @skip_if_lt_x_gpu(4)
    @with_temp_dir
    def test_dcp_round_trip(self) -> None:
        self._run_round_trips()


if __name__ == "__main__":
    run_tests()
