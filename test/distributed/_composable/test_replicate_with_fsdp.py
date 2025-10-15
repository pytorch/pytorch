# Owner(s): ["oncall: distributed"]

import copy
import functools
import os
from copy import deepcopy

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed._composable.contract import _get_registry
from torch.distributed._composable.replicate_with_fsdp import (
    _get_managed_modules,
    replicate,
)
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import Replicate, Shard
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    run_subtests,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_fsdp import check_sharded_parity, MLPStack
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
)


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 2)
        self.fc3 = nn.Linear(2, 2)

    def forward(self, x):
        return self.fc3(self.fc2(self.fc1(x)))


class ReplicateTest(MultiProcessTestCase):
    @property
    def world_size(self) -> int:
        return 4

    def init_replicate_tp_mesh(self) -> DeviceMesh:
        # Prefer to test with >=4 GPUs, but for 2 GPUs, use 2-way TP
        replicate_size = 2
        return init_device_mesh(
            "cuda",
            (replicate_size, 1, self.world_size // replicate_size),
            mesh_dim_names=("replicate", "shard", "tp"),
        )

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def _init_pg(self):
        # Set the device explicitly before initializing the process group

        torch.cuda.set_device(self.rank % self.world_size)
        dist.init_process_group(
            backend="nccl",
            rank=self.rank,
            world_size=self.world_size,
            store=dist.FileStore(self.file_name, self.world_size),
        )

    @skip_if_lt_x_gpu(2)
    def test_replicate_transformer(self):
        """
        This tests that replicate works on a transformer model with fully_shard and replicate layers
        """
        self._init_pg()
        run_subtests(
            self,
            {
                "sharding_strategy": ["replicate", "fully_shard"],
            },
            self._test_replicate_transformer,
        )

    def _composable_api_module_check(self, module, sharding_strategy):
        if sharding_strategy == "replicate":
            self.assertTrue("replicate" in _get_registry(module))
        else:
            self.assertTrue("fully_shard" in _get_registry(module))

    def _test_replicate_transformer(self, sharding_strategy):
        model_args = ModelArgs()

        model = Transformer(model_args)
        replicate_model = deepcopy(model)

        for i, layer in enumerate(replicate_model.layers):
            if i % 2 == 0:
                replicate(layer)
            elif i % 2 == 1:
                fully_shard(layer)

        if sharding_strategy == "replicate":
            replicate_model = replicate(replicate_model)

        else:
            replicate_model = fully_shard(replicate_model)

        self._composable_api_module_check(replicate_model, sharding_strategy)

        for i, layer in enumerate(replicate_model.layers):
            if i % 2 == 0:
                self.assertTrue("replicate" in _get_registry(layer))
                for parameter in layer.parameters():
                    self.assertEqual(parameter.placements, (Replicate(),))
            elif i % 2 == 1:
                self.assertTrue("fully_shard" in _get_registry(layer))
                for parameter in layer.parameters():
                    self.assertEqual(parameter.placements, (Shard(dim=0),))

    @skip_if_lt_x_gpu(2)
    def test_replicate_transformer_managed_modules(self):
        """
        This tests that replicate managed modules works properly. In this test we use a Transformer Module with 3 layers,
        which means there are 49 submodules. We apply replicate on the first layer and fully shard on the second layer,
        each consisting of 14 submodules, leaving 21 remaining submodules. I have shown below how there are this many submodules

        1. Transformer Module
            2. tok_embeddings
            3. pos_embeddings
            4. dropout
            5. layers
            6. norm
            7. output

        In the layers we have Transformer Blocks

        1. Transformer Block
            2. attention_norm
            3. Attention
                4. resid_dropout
                5. wq
                6. wk
                7. wv
                8. wo
            9. ffn_norm
            10. Feed_forward
                11. w1
                12. gelu
                13. w2
                14. resid_dropout

        """
        self._init_pg()

        model_args = ModelArgs()
        model_args.n_layers = 3

        model = Transformer(model_args)
        replicate_model = deepcopy(model)

        self.assertEqual(len(_get_managed_modules((replicate_model,))), 49)

        for i, layer in enumerate(replicate_model.layers):
            if i % 3 == 0:
                replicate(layer)
            elif i % 3 == 1:
                fully_shard(layer)

        replicate_model = replicate(replicate_model)
        self.assertEqual(len(_get_managed_modules((replicate_model,))), 21)

    @skip_if_lt_x_gpu(2)
    def test_replicate_tp_device_mesh(self):
        """
        This tests that a user can pass in a device mesh to replicate a module
        """

        self._init_pg()

        device = torch.device(f"cuda:{self.rank % torch.cuda.device_count()}")
        model = Net().to(device)
        replicate_model = deepcopy(model)

        layers = [
            replicate_model.fc1,
            replicate_model.fc2,
            replicate_model.fc3,
        ]

        global_mesh = self.init_replicate_tp_mesh()
        replicate_mesh = global_mesh["replicate"]

        for layer in layers:
            replicate(layer, device_mesh=replicate_mesh)

            for parameter in layer.parameters():
                self.assertEqual(parameter.device_mesh.shape, (2,))
                self.assertEqual(parameter.placements, (Replicate(),))

    @skip_if_lt_x_gpu(2)
    def test_train_replicate_fsdp(self):
        """
        Tests that replicate_model has the same behavior as original model when training
        """
        self._init_pg()

        device = torch.device(f"cuda:{self.rank % torch.cuda.device_count()}")
        model = Net().to(device)
        replicate_model = deepcopy(model)

        layers = [
            replicate_model.fc1,
            replicate_model.fc2,
            replicate_model.fc3,
        ]

        for layer in layers:
            replicate(layer)

        replicate_model = replicate(replicate_model)

        optim = torch.optim.Adam(model.parameters(), lr=0.01)
        replicate_optim = torch.optim.Adam(replicate_model.parameters(), lr=0.01)

        torch.manual_seed(42 + self.rank + 1)
        inp = torch.randn(2, 2, device=device)

        for _ in range(10):
            loss = model(inp).sum()
            loss.backward()

            for param in model.parameters():
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)

            replicate_loss = replicate_model(inp).sum()
            replicate_loss.backward()

            optim.step()
            replicate_optim.step()

            optim.zero_grad()
            replicate_optim.zero_grad()

            self.assertEqual(replicate_loss, loss)
            check_sharded_parity(self, model, replicate_model)

    @skip_if_lt_x_gpu(2)
    def test_train_parity_2d_mlp(self):
        """
        Verifies when a device mesh is passed in, the model has the same behavior as the original model when training
        """
        self._init_pg()
        global_mesh = self.init_replicate_tp_mesh()
        run_subtests(
            self,
            {
                "use_activation_checkpointing": [False, True],
                "mlp_dim": [3, 16, 17],
            },
            functools.partial(self._test_train_parity_2d_mlp, global_mesh),
        )

    def _test_train_parity_2d_mlp(
        self,
        global_mesh: DeviceMesh,
        use_activation_checkpointing: bool,
        mlp_dim: int,
    ):
        replicate_shard_mesh, tp_mesh = (
            global_mesh["replicate", "shard"],
            global_mesh["tp"],
        )
        replicate_mesh = global_mesh["replicate"]
        replicate_pg = replicate_mesh.get_group()  # used for `replicate()`

        torch.manual_seed(42)
        model = MLPStack(mlp_dim)
        ref_model = copy.deepcopy(model).cuda()
        replicate(ref_model, device_mesh=replicate_mesh)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2, foreach=False)
        model.parallelize(
            tp_mesh,
            replicate_shard_mesh,
            use_activation_checkpointing,
        )
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=False)

        torch.manual_seed(42 + replicate_pg.rank() + 1)
        device = torch.device("cuda")
        for iter_idx in range(10):
            inp = torch.randn((8, mlp_dim), device=device)
            losses: list[torch.Tensor] = []
            for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
                losses.append(_model(inp).sum())
                losses[-1].backward()
                _optim.step()
            self.assertEqual(losses[0], losses[1])


if __name__ == "__main__":
    run_tests()
