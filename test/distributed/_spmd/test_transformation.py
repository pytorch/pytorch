# Owner(s): ["oncall: distributed"]

import unittest
from copy import deepcopy
from functools import wraps
from unittest.mock import MagicMock

import torch
import torch.nn as nn
from torch._inductor.utils import has_triton
from torch.distributed._spmd.api import compile
from torch.distributed._spmd.gm_transformation import GraphModuleTransformation
from torch.distributed._spmd.graph_optimization import (
    _optimized_func,
    comm_fusion_with_concat,
    get_all_fused_optimizer_blocks,
    graph_optimization_pass,
    iter_move_grads_and_optimizers,
    remove_copy_from_optimizer,
    schedule_comm_wait,
    split_fused_optimizer,
)
from torch.distributed._spmd.iter_graph_module import IterGraphModule
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests, skipIfRocm
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms as base_with_comms,
)


def with_comms(func):
    @base_with_comms
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # make sure we set different random seeds for each rank
        # otherwise we dont need DDP / SPMD
        # (we would have the same parameters and inputs everywhere)
        torch.manual_seed(self.rank)
        return func(self, *args, **kwargs)

    return wrapper


class DummyModel(nn.Module):
    def __init__(self, layers: int, dim: int):
        super().__init__()
        modules = []
        for _ in range(layers):
            modules.extend([nn.Linear(dim, dim), nn.ReLU()])
        self.mod = nn.Sequential(*modules)

    def forward(self, x):
        return self.mod(x)


class GraphPassWrapperTest(DTensorTestBase):
    @property
    def world_size(self):
        return 1

    def test_order(self):
        @graph_optimization_pass(
            prerequisites=[],
            apply_after=[],
        )
        def my_pass1(gm) -> None:
            return

        @graph_optimization_pass(
            prerequisites=[my_pass1],
            apply_after=[],
        )
        def my_pass2(gm) -> None:
            return

        @graph_optimization_pass(
            prerequisites=[],
            apply_after=[my_pass1],
        )
        def my_pass3(gm) -> None:
            return

        gm = MagicMock(spec=IterGraphModule)
        # No errors happen.
        my_pass1(gm)
        my_pass3(gm)
        my_pass2(gm)
        _optimized_func.clear()

        # Only my_pass3 is okay as it has no prerequisites.
        my_pass3(gm)
        _optimized_func.clear()

        # Prerequisite condition does not match.
        with self.assertRaisesRegex(AssertionError, "are the prerequisites of"):
            my_pass2(gm)
        _optimized_func.clear()

        # my_pass3 must be applied after my_pass1
        with self.assertRaisesRegex(AssertionError, "must be applied after"):
            my_pass3(gm)
            my_pass1(gm)
        _optimized_func.clear()


class TransformationTest(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    def _init(self, batch_size, layers, dim, foreach: bool = True, fused: bool = False):
        torch.manual_seed(0)
        model = DummyModel(layers, dim).cuda()
        ddp_model = DDP(deepcopy(model), device_ids=[self.rank])
        optim = torch.optim.Adam(
            model.parameters(), lr=0.01, foreach=foreach, fused=fused, capturable=True
        )
        ddp_optim = torch.optim.Adam(
            ddp_model.parameters(),
            lr=0.01,
            foreach=foreach,
            fused=fused,
            capturable=True,
        )
        batch = torch.randn(batch_size, dim).cuda()

        # materialize optimizer states
        out = model(batch)
        out.sum().backward()
        optim.step()
        optim.zero_grad()

        ddp_out = ddp_model(batch)
        ddp_out.sum().backward()
        ddp_optim.step()
        ddp_optim.zero_grad()

        self.assertEqual(ddp_out, out)
        self.assertEqual(list(ddp_model.parameters()), list(model.parameters()))
        return model, optim, ddp_model, ddp_optim

    def _test_train_step(
        self, train_step, num_iters, batch_size, layers, dim, use_fused_optimizer=False
    ):
        def _ddp_train_step(model, optim, batch):
            model(batch).sum().backward()
            with torch.no_grad():
                for p in model.parameters():
                    p.grad *= self.world_size
            optim.step()
            optim.zero_grad()

        model, optim, ddp_model, ddp_optim = self._init(
            batch_size,
            layers,
            dim,
            foreach=(not use_fused_optimizer),
            fused=use_fused_optimizer,
        )
        for i in range(num_iters):
            batch = torch.randn(batch_size, dim).cuda()
            kwargs = {} if i < num_iters - 1 else {"last_train_step": True}
            out = train_step(model, optim, batch, **kwargs)
            ddp_out = _ddp_train_step(ddp_model, ddp_optim, batch)
        self.assertEqual(list(ddp_model.parameters()), list(model.parameters()))

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_basic_transformation(self):
        batch_size = 100
        layers = 10
        dim = 100
        num_iters = 5

        @compile(gm_transformation=GraphModuleTransformation())
        def train_step(model, optim, batch):
            model(batch).sum().backward()
            optim.step()
            optim.zero_grad()

        self._test_train_step(train_step, num_iters, batch_size, layers, dim)

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    @skipIfRocm
    @with_comms
    def test_inductor(self):
        batch_size = 100
        # Too many layers will cause test timeout due to the compilation.
        layers = 2
        dim = 100
        num_iters = 5

        @compile(
            gm_transformation=GraphModuleTransformation(
                enable_inductor=True, dump_graphs=True
            )
        )
        def train_step(model, optim, batch):
            model(batch).sum().backward()
            optim.step()
            optim.zero_grad()

        self._test_train_step(
            train_step, num_iters, batch_size, layers, dim, use_fused_optimizer=True
        )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_graph_optimization_with_foreach(self):
        batch_size = 100
        layers = 2
        dim = 4096
        num_iters = 5

        @compile(
            gm_transformation=GraphModuleTransformation(
                enable_graph_optimization=True,
                dump_graphs=False,
            )
        )
        def train_step(model, optim, batch):
            model(batch).sum().backward()
            optim.step()
            optim.zero_grad()

        self._test_train_step(train_step, num_iters, batch_size, layers, dim)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_graph_optimization_with_fused(self):
        batch_size = 100
        layers = 2
        dim = 4096
        num_iters = 5

        @compile(
            gm_transformation=GraphModuleTransformation(
                enable_graph_optimization=True,
                dump_graphs=False,
            )
        )
        def train_step(model, optim, batch):
            model(batch).sum().backward()
            optim.step()
            optim.zero_grad()

        self._test_train_step(
            train_step, num_iters, batch_size, layers, dim, use_fused_optimizer=True
        )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_split_fused_optimizer(self):
        batch_size = 100
        layers = 2
        dim = 4096
        num_iters = 5

        def my_transformation(gm):
            gm = IterGraphModule(gm)
            remove_copy_from_optimizer(gm)
            opt_block = get_all_fused_optimizer_blocks(gm, "_fused_adam")[0]
            gradients = {
                opt_block.optim.optim_node.args[1][1],
                opt_block.optim.optim_node.args[1][2],
            }
            split_fused_optimizer(gm, opt_block, gradients)
            gm.graph.eliminate_dead_code()
            gm.recompile()
            self.assertEquals(len(get_all_fused_optimizer_blocks(gm, "_fused_adam")), 2)
            gm.finalize_setup()
            return gm

        @compile(gm_transformation=my_transformation)
        def train_step(model, optim, batch):
            model(batch).sum().backward()
            optim.step()
            optim.zero_grad()

        self._test_train_step(
            train_step, num_iters, batch_size, layers, dim, use_fused_optimizer=True
        )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_iter_move_blocks_and_optimizers(self):
        batch_size = 100
        layers = 5
        dim = 4096
        num_iters = 5

        def my_transformation(gm):
            gm = IterGraphModule(gm)
            comm_fusion_with_concat(gm, 100)
            schedule_comm_wait(gm)
            remove_copy_from_optimizer(gm)
            iter_move_grads_and_optimizers(gm, "all_reduce_default_1", "relu")
            gm.finalize_setup()
            return gm

        @compile(gm_transformation=my_transformation)
        def train_step(model, optim, batch):
            model(batch).sum().backward()
            optim.step()
            optim.zero_grad()

        self._test_train_step(
            train_step, num_iters, batch_size, layers, dim, use_fused_optimizer=True
        )


if __name__ == "__main__":
    run_tests()
