# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# type: ignore

import copy
import os
import sys
from contextlib import suppress
from typing import List, Any, Type, cast

import numpy as np
import torch
import torch.distributed as dist
if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.distributed.optim.zero_redundancy_optimizer import _broadcast_object
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD
from torch.testing._internal import common_utils, common_distributed

BACKEND = dist.Backend.NCCL if torch.cuda.is_available() else dist.Backend.GLOO  # type: ignore
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def check_same_model_params(model_a: torch.nn.Module, model_b: torch.nn.Module, message: str = "") -> None:
    for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.allclose(p_a, p_b, atol=1e-3), f"Model parameters differ\n{p_a} {p_b}\n" + message

    for b_a, b_b in zip(model_a.buffers(), model_b.buffers()):
        assert torch.allclose(b_a, b_b), f"Model buffers differ {b_a} - {b_b}\n" + message


class TestZeroRedundancyOptimizer(common_distributed.MultiProcessTestCase):
    def setUp(self):
        super(TestZeroRedundancyOptimizer, self).setUp()
        os.environ["WORLD_SIZE"] = str(self.world_size)

        self._spawn_processes()

    @property
    def device(self):
        return torch.device(self.rank) if BACKEND == dist.Backend.NCCL else torch.device("cpu")

    @property
    def world_size(self):
        return 1

    def tearDown(self):
        try:
            torch.distributed.destroy_process_group()
        except AssertionError:
            pass

        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def dist_init(self, rank, world_size=-1):
        store = dist.FileStore(self.file_name, self.world_size if world_size < 1 else world_size)
        return dist.init_process_group(backend=BACKEND, store=store, rank=rank, world_size=self.world_size)


class TestZeroRedundancyOptimizerSingleRank(TestZeroRedundancyOptimizer):
    def test_state_dict(self):
        """Check that the ZeroRedundancyOptimizer exposes the expected state dict interface,
        irrespective of the sharding.
        """
        self.dist_init(self.rank)
        x = torch.tensor([1.0], device=DEVICE, requires_grad=True)
        o = ZeroRedundancyOptimizer([x], optimizer_class=SGD, lr=0.1, momentum=0.9)
        x.backward()
        o.step()
        self.assertEqual(x, torch.tensor([0.9], device=DEVICE))
        self.assertEqual(o.optim.state[x]["momentum_buffer"], torch.tensor([1.0], device=DEVICE))

        o.zero_grad()
        o.consolidate_state_dict()  # Sync state dict in between replicas - even if there are none
        state_dict = o.state_dict()

        # Check that the state dict is pytorch-compliant key wise
        self.assertIn("param_groups", state_dict.keys())
        self.assertIn("state", state_dict.keys())

        # Check that the pulled state is what we expect, and that we have all the expected keys
        self.assertEqual(state_dict["param_groups"][0]["lr"], 0.1)
        self.assertEqual(state_dict["param_groups"][0]["momentum"], 0.9)
        self.assertFalse(state_dict["param_groups"][0]["nesterov"])
        self.assertEqual(state_dict["param_groups"][0]["weight_decay"], 0.0)
        self.assertEqual(state_dict["param_groups"][0]["dampening"], 0.0)

        # Check that the pulled state and the .param_groups attribute are in sync
        for k in state_dict["param_groups"][0].keys():
            if k != "params":
                self.assertEqual(state_dict["param_groups"][0][k], o.param_groups[0][k])

        # Check that it's correctly loaded
        o = ZeroRedundancyOptimizer([x], optimizer_class=SGD, lr=0.01)
        o.load_state_dict(state_dict)

        # Check that state is correct and on proper device
        self.assertEqual(o.optim.state[x]["momentum_buffer"], torch.tensor([1.0], device=DEVICE))

        # We should now be using a lr of 0.1, both within the optimizer
        # and as exposed by the .param_groups attribute
        assert o.param_groups[0]["lr"] == 0.1
        x.backward()
        o.step()
        self.assertEqual(x, torch.tensor([0.71], device=DEVICE))
        self.assertEqual(o.optim.state[x]["momentum_buffer"], torch.tensor([1.9], device=DEVICE))

        # Check that the exposed param_groups are on the proper device
        self.assertEqual(o.param_groups[0]["params"][0].device, x.device)

    def test_lr_scheduler(self):
        """ Check that a normal torch lr_scheduler is usable with ZeroRedundancyOptimizer"""

        self.dist_init(self.rank)
        x = torch.tensor([1.0], device=DEVICE, requires_grad=True)
        x2 = torch.tensor([1.0], device=DEVICE, requires_grad=True)
        o = ZeroRedundancyOptimizer([x], optimizer_class=SGD, lr=0.01)
        o2 = torch.optim.SGD([x2], lr=0.01)
        s = torch.optim.lr_scheduler.StepLR(o, 1)
        s2 = torch.optim.lr_scheduler.StepLR(o2, 1)
        for _ in range(5):
            x.backward()
            o.zero_grad()
            o.step()
            s.step()
            x2.backward()
            o2.zero_grad()
            o2.step()
            s2.step()
            self.assertEqual(x, x2)

    def test_step_with_kwargs(self):
        """ Check that the `step(**kwargs)` interface is properly exposed"""
        self.dist_init(self.rank)

        class SGDWithStepKWArg(torch.optim.SGD):
            def step(self, closure=None, kwarg=None):
                super().step()
                kwarg.append(5)

        kwarg: List[Any] = []
        x = torch.tensor([1.0], device=DEVICE, requires_grad=True)
        o = ZeroRedundancyOptimizer([x], optimizer_class=SGDWithStepKWArg, lr=0.1)
        x.backward()
        o.step(0, kwarg=kwarg)
        self.assertEqual(kwarg, [5])
        self.assertEqual(x, torch.tensor([0.9], device=DEVICE))

    def test_step_with_extra_inner_key(self):
        """Check that an optimizer adding extra keys to the param_groups
        is properly handled, in that the new key is exposed to the user
        """
        self.dist_init(self.rank)

        class SGDWithNewKey(torch.optim.SGD):
            # Dummy optimizer which adds a new key to the param groups
            def step(self, closure=None):
                super().step()
                self.param_groups[0]["new_key"] = 0.1

        x = torch.tensor([1.0], device=DEVICE, requires_grad=True)
        o = ZeroRedundancyOptimizer([x], optimizer_class=SGDWithNewKey, lr=0.1)
        x.backward()
        o.step()
        self.assertEqual(o.param_groups[0]["new_key"], 0.1)
        self.assertEqual(x, torch.tensor([0.9], device=DEVICE))

    def test_step_without_closure(self):
        """Check that the step() method (without closure) is handlded as expected"""
        self.dist_init(self.rank)

        class SGDWithoutClosure(torch.optim.SGD):
            def step(self):
                return super().step()

        x = torch.tensor([1.0], device=DEVICE, requires_grad=True)
        o = ZeroRedundancyOptimizer([x], optimizer_class=SGDWithoutClosure, lr=0.1)
        x.backward()
        o.step()
        self.assertEqual(x, torch.tensor([0.9], device=DEVICE))

    def test_zero_grad(self):
        """Check that the zero_grad attribute is properly handled"""
        self.dist_init(self.rank)
        x = torch.rand(1)
        m = torch.nn.Linear(1, 1)
        o = ZeroRedundancyOptimizer(m.parameters(), optimizer_class=SGD, lr=0.1)
        y = m(x)
        y.backward(x)
        self.assertNotEqual(m.weight.grad, torch.zeros_like(m.weight))
        self.assertNotEqual(m.weight.grad, torch.zeros_like(m.weight))
        o.zero_grad()
        self.assertFalse(m.weight.grad)
        self.assertFalse(m.bias.grad)


class TestZeroRedundancyOptimizerDistributed(TestZeroRedundancyOptimizer):
    @property
    def world_size(self):
        return max(2, torch.cuda.device_count())

    @common_distributed.skip_if_rocm
    def test_step(self):
        """ Check that the ZeroRedundancyOptimizer wrapper properly exposes the `.step()` interface"""
        if self.rank > 1 or (BACKEND == dist.Backend.NCCL and torch.cuda.device_count() < 2):
            return

        self.dist_init(self.rank, world_size=2)

        context = suppress() if not torch.cuda.is_available() else torch.cuda.device(self.rank)

        with context:
            x = torch.tensor([float(self.rank + 1)], device=self.device)
            m = torch.nn.Linear(1, 1)
            m.weight.data = torch.tensor([[1.0]])
            m.bias.data = torch.tensor([2.0])
            m.to(self.device)

            o = ZeroRedundancyOptimizer(m.parameters(), optimizer_class=SGD, lr=0.1)
            y = m(x)
            y.backward(x)
            for p in m.parameters():
                dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
                p.grad.data /= self.world_size
            o.step()
            self.assertEqual(m.weight, torch.tensor([[0.75]], device=self.device))
            self.assertEqual(m.bias, torch.tensor([1.85], device=self.device))

    @common_distributed.skip_if_rocm
    def test_step_with_closure(self):
        """ Check that the ZeroRedundancyOptimizer wrapper properly exposes the `.step(closure)` interface"""

        if self.rank > 1 or (BACKEND == dist.Backend.NCCL and torch.cuda.device_count() < 2):
            return

        self.dist_init(self.rank, world_size=2)

        context = suppress() if not torch.cuda.is_available() else torch.cuda.device(self.rank)

        with context:
            for bucket_view in [False, True]:
                x_val = self.rank + 1
                weight = 1.0
                bias = 2.0
                error = 1.0
                target = torch.tensor([x_val * weight + bias + error], device=self.device)
                loss_fn = torch.nn.L1Loss()

                x = torch.tensor([float(x_val)], device=self.device)
                m = torch.nn.Linear(1, 1)
                m.weight.data = torch.tensor([[weight]])
                m.bias.data = torch.tensor([bias])
                m.to(self.device)

                o = ZeroRedundancyOptimizer(
                    m.parameters(),
                    optimizer_class=SGD,
                    parameters_as_bucket_view=bucket_view,
                    lr=0.1,
                )

                y = m(x)
                y.backward(x)
                for p in m.parameters():
                    dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
                    p.grad.data /= self.world_size

                def closure():
                    o.zero_grad()
                    output = m(x)
                    loss = loss_fn(output, target)
                    loss.backward()
                    return loss

                loss = o.step(closure=closure)

                self.assertEqual(loss, torch.tensor(error))
                self.assertEqual(m.weight, torch.tensor([[1.1]]))
                self.assertEqual(m.bias, torch.tensor([2.1]))

    def test_sharding(self):
        """ Check the sharding at construction time"""
        self.dist_init(self.rank)
        sizes = [9, 7, 5, 3]
        params = []
        for size in sizes * self.world_size:
            params.append(torch.rand(size, 1))
        o = ZeroRedundancyOptimizer(params, optimizer_class=SGD, lr=0.1)
        self.assertEqual(sum([x.numel() for x in o.optim.param_groups[0]["params"]]), sum(sizes))

    def test_add_param_group(self):
        """Check that ZeroRedundancyOptimizer properly handles adding a new param_group a posteriori,
        and that all ranks get a shard
        """
        self.dist_init(self.rank)

        # Test with all parameters trainable to begin with
        def all_trainable():
            params = []
            sizes = [9, 7, 5, 3]
            sizes_world = sizes * self.world_size
            for size in sizes_world[:-1]:
                params.append(torch.rand(size, 1))

            # Make sure that the params are trainable, enforces size-based partitioning
            for p in params:
                p.requires_grad = True

            o = ZeroRedundancyOptimizer(params, optimizer_class=SGD, lr=0.1)

            assert len(o.param_groups) == 1
            o.add_param_group({"params": [torch.rand(3, 1)]})

            assert len(o.param_groups) == 2
            # Verify that added group is added to the correct partition making all have the same elements.
            assert sum([x.numel() for g in o.optim.param_groups for x in g["params"]]) == sum(sizes)
            assert len(o.optim.param_groups) == 2

        # Test a pathological config with a first big non-trainable param
        def some_trainable():
            params = []
            for size in [100, 3, 5, 2, 6, 4]:
                params.append(torch.rand(size, 1))

            # Make sure that the params are trainable, enforces size-based partitioning
            for p in params[1:]:
                p.requires_grad = True

            o = ZeroRedundancyOptimizer(params, optimizer_class=SGD, lr=0.1)

            assert len(o.param_groups) == 1
            o.add_param_group({"params": [torch.rand(3, 1)]})

            assert len(o.param_groups) == 2
            assert len(o.optim.param_groups) == 2

        all_trainable()
        some_trainable()

    @common_distributed.skip_if_not_multigpu
    def test_collect_shards(self):
        """ Check the state consolidation mechanism, and the state dict exposed by ZeroRedundancyOptimizer"""
        self.dist_init(self.rank)
        RECIPIENT_RANK = 0

        # Run a dummy step so that the optimizer state dict exists
        batch, input_width, hidden, target_width = 3, 20, 10, 5
        target = torch.rand((batch, target_width), device=self.device)
        inputs = torch.rand((batch, input_width), device=self.device)

        model = torch.nn.Sequential(torch.nn.Linear(input_width, hidden), torch.nn.Linear(hidden, target_width))
        model.to(self.device)

        loss_fn = torch.nn.L1Loss()
        loss_fn.to(self.device)

        # With SGD, Momentum is required to get a state to shard
        optimizer = ZeroRedundancyOptimizer(model.parameters(), optimizer_class=SGD, lr=0.1, momentum=0.99)

        def closure():
            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_fn(output, target)
            loss.backward()
            return loss

        _ = optimizer.step(closure=closure)

        # Update the optimizer state on the reference rank
        optimizer.consolidate_state_dict(to=RECIPIENT_RANK)

        # Fetch the state on the reference rank
        # - check that it has the correct size
        # - load it again
        if self.rank == RECIPIENT_RANK:
            optimizer_state_dict = optimizer.state_dict()
            self.assertEqual(len(optimizer_state_dict["state"]), len(list(model.parameters())))
        else:
            optimizer_state_dict = {}

        optimizer_state_dict = _broadcast_object(
            optimizer_state_dict,
            src_rank=RECIPIENT_RANK,
            group=dist.group.WORLD,
            dist_device=self.device,
        )

        # Load the optimizer state dict, check that no exception is raised
        optimizer.load_state_dict(optimizer_state_dict)

    def test_multiple_groups(self):
        """ Check that the ZeroRedundancyOptimizer handles working with multiple process groups"""
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(backend="gloo", store=store, rank=self.rank, world_size=self.world_size)

        # Only work with the even ranks, to check that the global_rank indexing is properly used
        sub_group_ranks = list(filter(lambda x: x % 2 == 0, range(self.world_size)))
        process_group = torch.distributed.new_group(ranks=sub_group_ranks, backend="gloo")

        # Make sure that all the ranks get different training data
        # So that the sync check in between their models is meaningful
        torch.manual_seed(self.rank)
        np.random.seed(self.rank)

        # Standard deep learning setup
        epochs, batch, input_width, hidden, target_width = 5, 3, 20, 10, 5
        loss_fn = torch.nn.L1Loss().to(self.device)

        def check(optimizer):
            # Just run a couple of epochs, check that the model is properly updated
            for _ in range(epochs):
                target = torch.rand((batch, target_width), device=self.device)
                inputs = torch.rand((batch, input_width), device=self.device)

                def closure():
                    optimizer.zero_grad()
                    output = model(inputs)
                    loss = loss_fn(output, target)
                    loss /= self.world_size
                    loss.backward()
                    dist.all_reduce(loss, group=process_group)  # Not strictly needed for the test below

                    return loss

                _ = optimizer.step(closure=closure)

                # Check that all the params are the same on all ranks
                for pg in optimizer.param_groups:
                    for p in pg["params"]:
                        receptacle = [p.clone() for _ in sub_group_ranks] if self.rank == 0 else []
                        dist.gather(p, receptacle, dst=0, group=process_group)
                        if self.rank == 0:
                            for sync_p in receptacle[1:]:
                                assert torch.all(torch.eq(receptacle[0], sync_p)), "Models differ in between ranks"

        if self.rank in sub_group_ranks:
            # Model fitting in the broadcast bucket
            model = torch.nn.Sequential(
                torch.nn.Linear(input_width, hidden),
                torch.nn.Linear(hidden, target_width),
            ).to(self.device)

            # With SGD, Momentum is required to get a state to shard
            optimizer = ZeroRedundancyOptimizer(
                model.parameters(), optimizer_class=SGD, lr=0.1, momentum=0.99, group=process_group
            )
            check(optimizer)

            # Model not-fitting in the broadcast bucket
            model = torch.nn.Sequential(
                torch.nn.Linear(input_width, hidden),
                torch.nn.Linear(hidden, target_width),
            ).to(self.device)

            # With SGD, Momentum is required to get a state to shard
            optimizer = ZeroRedundancyOptimizer(
                model.parameters(),
                optimizer_class=SGD,
                lr=0.1,
                momentum=0.99,
                group=process_group,
            )
            check(optimizer)

    @common_distributed.skip_if_no_gpu
    def test_pytorch_parity(self):
        """When combined with DDP, check that ZeroRedundancyOptimizer(optimizer) and the same monolithic optimizer
        give the exact same results
        """

        self.dist_init(self.rank)
        BATCHS = 20

        with torch.cuda.device(self.rank):
            torch.manual_seed(self.rank)
            np.random.seed(self.rank)

            def check_optimizer_equivalence(optimizer: Type[torch.optim.Optimizer]):
                # Any model works. Add one different buffer per rank
                model = torch.nn.Sequential(
                    torch.nn.Linear(2, 3),
                    torch.nn.Linear(3, 3),
                    torch.nn.Linear(3, 3),
                )
                model.register_buffer("test_buffer", torch.ones((1)) * self.rank)
                model.to(self.device)

                sharded_optimizer = ZeroRedundancyOptimizer(params=model.parameters(), optimizer_class=optimizer, lr=1e-3)
                sharded_ddp_model = DDP(
                    module=model, device_ids=[self.rank], broadcast_buffers=True, find_unused_parameters=True
                )

                ddp_model_single = copy.deepcopy(model)
                ddp_model_single.to(self.device)

                ddp_optimizer = optimizer(ddp_model_single.parameters(), lr=1e-3)
                ddp_model = DDP(
                    ddp_model_single, device_ids=[self.rank], broadcast_buffers=True, find_unused_parameters=True
                )

                # The model should be synchronized in between the ranks at construction time, check that
                check_same_model_params(sharded_ddp_model, ddp_model, "Models differ from the start")

                def check_step():
                    input_tensor = torch.rand((64, 2))

                    def closure_ddp(input_tensor=input_tensor):
                        ddp_optimizer.zero_grad()
                        ddp_loss = ddp_model(input_tensor).abs().sum()
                        ddp_loss.backward()
                        return ddp_loss

                    def closure_sharded(input_tensor=input_tensor):
                        sharded_optimizer.zero_grad()
                        sharded_loss = sharded_ddp_model(input_tensor).abs().sum()
                        sharded_loss.backward()
                        return sharded_loss

                    loss_ddp = cast(torch.Tensor, ddp_optimizer.step(closure=closure_ddp))
                    loss_sharded_optim = cast(torch.Tensor, sharded_optimizer.step(closure=closure_sharded))

                    assert torch.allclose(
                        loss_ddp, loss_sharded_optim
                    ), "Losses differ in between Pytorch optim and ZeroRedundancyOptimizer"

                    check_same_model_params(sharded_ddp_model, ddp_model, "Models differ after a step")

                # The models should stay the same in between the ranks
                for i in range(BATCHS):
                    check_step()

                    # Change the models trainability, check that parity is maintained
                    # only check after a couple of constant batchs to go through both regimes
                    if i > BATCHS // 2:
                        next(ddp_model.parameters()).requires_grad = bool(i % 2)
                        next(sharded_ddp_model.parameters()).requires_grad = bool(i % 2)

                # Check that the checkpoints are compatible
                reference_rank = 0
                # - get states
                ddp_state_dict = ddp_optimizer.state_dict()
                sharded_optimizer.consolidate_state_dict(to=reference_rank)
                sharded_optim_state_dict = [sharded_optimizer.state_dict() if self.rank == reference_rank else {}]
                dist.broadcast_object_list(sharded_optim_state_dict, src=reference_rank, group=dist.group.WORLD)
                sharded_optim_state_dict = sharded_optim_state_dict[0]

                # - cross load the states
                # run one step and check that the models are still the same
                ddp_state_dict_ref = copy.deepcopy(ddp_state_dict)  # OSS will remove some states
                ddp_optimizer.load_state_dict(sharded_optim_state_dict)  # mixup on purpose !
                sharded_optimizer.load_state_dict(ddp_state_dict)
                check_step()

                #  - self load, rewind, check no problem
                # run one step and check that the models are still the same
                ddp_optimizer.load_state_dict(ddp_state_dict_ref)
                sharded_optimizer.load_state_dict(sharded_optim_state_dict)
                check_step()

            for opt in [torch.optim.SGD, torch.optim.Adam]:
                check_optimizer_equivalence(opt)


if __name__ == "__main__":
    # ! unittest should not be used here, else the tests are not properly registered
    common_utils.run_tests()
