# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
import sys
from contextlib import suppress
from typing import Any, List, Type, cast

import numpy as np

import torch
import torch.distributed as dist

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)
from torch.distributed.algorithms.join import _Join, _Joinable, _JoinHook
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.distributed.optim.zero_redundancy_optimizer import _broadcast_object
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD
from torch.testing._internal import common_distributed, common_utils

BACKEND = dist.Backend.NCCL if torch.cuda.is_available() else dist.Backend.GLOO
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

    def dist_init(self, rank, world_size=-1, backend=BACKEND):
        if (world_size < 1):
            world_size = self.world_size
        store = dist.FileStore(self.file_name, world_size)
        return dist.init_process_group(backend=backend, store=store, rank=rank, world_size=world_size)


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

    def test_constructor(self):
        """Check the robustness of the ZeroRedundancyOptimizer constructor by
        passing different values for `params`"""
        self.dist_init(self.rank)

        m = torch.nn.Linear(1, 1)
        # (input, expected error)
        inputs = [
            ([], ValueError),                           # empty parameter list
            (torch.randn(1), TypeError),                # non-iterable: `torch.Tensor`
            (1.2, TypeError),                           # non-iterable: `float`
            ([{"params": m.parameters()}], TypeError),  # iterable of dict
            (list(m.parameters()) + [42], TypeError),   # iterable containing non-`torch.Tensor`
            (m.parameters(), None),                     # `params` as a generator
            (list(m.parameters()), None)                # `params` as a list
        ]

        for input, error in inputs:
            if (error):
                with self.assertRaises(error):
                    ZeroRedundancyOptimizer(input, optimizer_class=SGD, lr=0.1)
            else:
                ZeroRedundancyOptimizer(input, optimizer_class=SGD, lr=0.1)

    def test_same_dense_param_type(self):
        """Check that ZeroRedundancyOptimizer raises an exception if the input
        parameters include sparse tensors or different dense types.

        NOTE: This test should be removed once support for sparse parameters
        and varying parameter types is added.
        """
        self.dist_init(self.rank)

        inputs = [
            [torch.sparse_coo_tensor(size=(2, 3))],
            [torch.FloatTensor(1), torch.DoubleTensor(1)],
            [torch.FloatTensor(1), torch.FloatTensor(1),
                torch.sparse_coo_tensor(size=(2, 3))]
        ]
        for input in inputs:
            with self.assertRaises(ValueError):
                ZeroRedundancyOptimizer(input, optimizer_class=SGD, lr=0.1)


class TestZeroRedundancyOptimizerDistributed(TestZeroRedundancyOptimizer):
    @property
    def world_size(self):
        return min(4, max(2, torch.cuda.device_count()))

    @common_distributed.skip_if_rocm
    def test_step(self):
        """ Check that the ZeroRedundancyOptimizer wrapper properly exposes the `.step()` interface"""

        if self.rank >= self.world_size or (BACKEND == dist.Backend.NCCL and torch.cuda.device_count() < 2):
            return

        self.dist_init(self.rank, world_size=self.world_size)

        context = suppress() if not torch.cuda.is_available() else torch.cuda.device(self.rank)

        with context:
            x = torch.tensor([float(self.rank + 1)], device=self.device)
            m = torch.nn.Linear(1, 1)
            m.weight.data = torch.tensor([[1.0]])
            m.bias.data = torch.tensor([2.0])
            m_zero = copy.deepcopy(m)
            m.to(self.device)
            m_zero.to(self.device)

            lr = 0.1
            o = SGD(m.parameters(), lr=lr)
            o_zero = ZeroRedundancyOptimizer(m_zero.parameters(), optimizer_class=SGD, lr=lr)

            y = m(x)
            y.backward(x)
            y_zero = m_zero(x)
            y_zero.backward(x)

            for p in m.parameters():
                dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
                p.grad.data /= self.world_size
            o.step()
            for p in m_zero.parameters():
                dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
                p.grad.data /= self.world_size
            o_zero.step()

            self.assertEqual(m.weight, m_zero.weight)
            self.assertEqual(m.bias, m_zero.bias)

    @common_distributed.skip_if_rocm
    def test_step_with_closure(self):
        """ Check that the ZeroRedundancyOptimizer wrapper properly exposes the `.step(closure)` interface"""

        if self.rank >= self.world_size or (BACKEND == dist.Backend.NCCL and torch.cuda.device_count() < 2):
            return

        self.dist_init(self.rank, world_size=self.world_size)

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
        """ Check the sharding at construction time

        NOTE: The correctness of this test depends on the ZeRO implementation
        using the sorted-greedy partitioning algorithm. For details, see
        `ZeroRedundancyOptimizer._partition_parameters()` in
        `zero_redundancy_optimizer.py`.
        """
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

        NOTE: The correctness of this test depends on the ZeRO implementation
        using the sorted-greedy partitioning algorithm. For details, see
        `ZeroRedundancyOptimizer._partition_parameters()` in
        `zero_redundancy_optimizer.py`.
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

    @common_distributed.skip_if_lt_x_gpu(2)
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
            device=self.device,
        )

        # Load the optimizer state dict, check that no exception is raised
        optimizer.load_state_dict(optimizer_state_dict)

    def test_multiple_groups(self):
        """ Check that the ZeroRedundancyOptimizer handles working with multiple process groups"""
        self.dist_init(self.rank, self.world_size, dist.Backend.GLOO)

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
                model.parameters(), optimizer_class=SGD, lr=0.1, momentum=0.99, process_group=process_group
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
                process_group=process_group,
            )
            check(optimizer)

    @common_distributed.skip_if_no_gpu
    def test_local_optimizer_parity(self):
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

                sharded_optimizer = ZeroRedundancyOptimizer(
                    params=model.parameters(), optimizer_class=optimizer, lr=1e-3
                )
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

    def _test_zero_join(self, device):
        r"""
        Check that the ZeRO join hook allows training with uneven inputs when using the given device.

        Arguments:
            device (torch.device): device used to store parameters and perform
                collective communications.
        """
        NUM_INPUTS = 3
        NUM_EPOCHS = 2
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        rank = self.rank
        world_size = self.world_size
        is_gpu = device.type == "cuda"
        backend = dist.Backend.NCCL if is_gpu else dist.Backend.GLOO
        self.dist_init(rank, world_size, backend)
        if BACKEND == dist.Backend.NCCL and is_gpu:
            torch.cuda.set_device(self.device)

        model = torch.nn.Sequential(
            torch.nn.Linear(2, 3),
            torch.nn.Linear(3, 3),
            torch.nn.Linear(3, 3),
        )
        model.to(device)

        # DDP ensures correct gradients in data parallel training, so DDP with
        # local optimizers on uneven inputs should be equivalent to ZeRO on
        # uneven inputs with gradients being manually set
        ddp_model = DDP(model, device_ids=[rank]) if is_gpu else DDP(model)
        local_optim = torch.optim.Adam(ddp_model.parameters(), lr=0.01)
        zero_model = copy.deepcopy(model)
        zero_model.to(device)
        zero_optim = ZeroRedundancyOptimizer(zero_model.parameters(), torch.optim.Adam, lr=0.01)
        loss_fn = torch.nn.MSELoss()

        # Use uneven inputs: rank i has i extra inputs
        inputs = [torch.randn(20, 2).to(device) for _ in range(NUM_INPUTS + rank)]
        labels = torch.randn(20, 3).to(device)

        # Save the gradients and parameters from DDP as the ground truth; do
        # so on the last-joining rank (in this case, the largest rank)
        grads_at_each_iter = []
        params_at_each_iter = []
        with ddp_model.join():
            for _ in range(NUM_EPOCHS):
                for input in inputs:
                    output = ddp_model(input)
                    loss_fn(output, labels).backward()
                    if rank == world_size - 1:
                        grads = []
                        for p in ddp_model.parameters():
                            grads.append(p.grad.detach().clone().to(device))
                    local_optim.step()
                    if rank == world_size - 1:
                        params = []
                        for p in ddp_model.parameters():
                            params.append(p.detach().clone().to(device))
                        grads_at_each_iter.append(grads)
                        params_at_each_iter.append(params)

        # Broadcast the saved gradients and parameters to all of the other
        # ranks (which joined early)
        grads_and_params = [grads_at_each_iter, params_at_each_iter]
        grads_and_params = _broadcast_object(grads_and_params, src_rank=world_size - 1, group=dist.group.WORLD, device=device)
        grads_at_each_iter = grads_and_params[0]
        params_at_each_iter = grads_and_params[1]
        # TODO: Replace this `_broadcast_object` with `broadcast_object_list`
        # once the latter supports loading to the destination device instead
        # of the source device

        # A process must still set the remaining gradients after joining, so we
        # define a join hook to do this before the ZeRO join hook
        class _JoinGradInfo():
            def __init__(self, grads):
                self.grads = grads  # remaining gradients to set (in order)
                self.index = 0

        class _SetGradsJoinHook(_JoinHook):
            def __init__(self, zero_optim, grads):
                zero_optim._join_grad_info = _JoinGradInfo(grads)
                self.zero = zero_optim
                super().__init__()

            def main_hook(self):
                grads = self.zero._join_grad_info.grads[self.zero._join_grad_info.index]
                self.zero._join_grad_info.index += 1
                for p, grad in zip(self.zero._all_params, grads):
                    p.grad = grad.detach().clone().to(device)

        class _GradientSetter(_Joinable):
            def __init__(self):
                super().__init__()

            def _join_hook(self, **kwargs):
                assert "zero_optim" in kwargs
                assert "grads" in kwargs
                zero_optim = kwargs["zero_optim"]
                grads = kwargs["grads"]
                return _SetGradsJoinHook(zero_optim, grads)

            @property
            def _join_device(self):
                return device

            @property
            def _join_process_group(self):
                return dist.group.WORLD

        num_grads_after_joining = NUM_EPOCHS * (world_size - rank - 1)
        grads = grads_at_each_iter[-num_grads_after_joining:]
        gradient_setter = _GradientSetter()
        iter = 0
        with _Join([gradient_setter, zero_optim], zero_optim=zero_optim, grads=grads):
            for _ in range(NUM_EPOCHS):
                for input in inputs:
                    # Notify join context that this process has not joined
                    _Join.notify_join_context(gradient_setter)

                    # Set gradients manually
                    for p, grad in zip(zero_model.parameters(), grads_at_each_iter[iter]):
                        p.grad = grad.detach().clone().to(device)

                    # Perform optimizer step and check parity
                    zero_optim.step()
                    for p, ddp_p in zip(zero_model.parameters(), params_at_each_iter[iter]):
                        assert torch.allclose(p, ddp_p), \
                            "Parameters differ between using ZeRO and local optimizer"
                    iter += 1

    @common_distributed.requires_nccl()
    @common_distributed.skip_if_lt_x_gpu(2)
    def test_zero_join_gpu(self):
        """Check that the ZeRO join hook allows training with uneven inputs on GPU."""
        self._test_zero_join(self.device)

    @common_distributed.requires_gloo()
    def test_zero_join_cpu(self):
        """Check that the ZeRO join hook allows training with uneven inputs on CPU."""
        self._test_zero_join(torch.device("cpu"))

    def _test_zero_model_parallel(self, parameters_as_bucket_view: bool):
        # Use two processes each with two GPUs
        assert self.rank < 2
        NUM_EPOCHS = 3
        NUM_INPUTS = 5
        LR = 0.01
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        class ModelParallelModel(torch.nn.Module):
            def __init__(self, dev0, dev1):
                super().__init__()
                self.dev0 = dev0
                self.dev1 = dev1
                self.net0 = torch.nn.Linear(10, 10).to(dev0)
                self.relu = torch.nn.ReLU()
                self.net1 = torch.nn.Linear(10, 5).to(dev1)

            def forward(self, x):
                x = x.to(self.dev0)
                x = self.relu(self.net0(x))
                x = x.to(self.dev1)
                return self.net1(x)

        class LocalModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net0 = torch.nn.Linear(10, 10)
                self.relu = torch.nn.ReLU()
                self.net1 = torch.nn.Linear(10, 5)

            def forward(self, x):
                return self.net1(self.relu(self.net0(x)))

        dev0 = 2 * self.rank
        dev1 = 2 * self.rank + 1
        mp_model = ModelParallelModel(dev0, dev1)
        ddp_model = DDP(mp_model)
        local_model = LocalModel()
        cpu_device = torch.device("cpu")
        # Ensure the parameters are the same across the two models
        local_model.net0.weight = torch.nn.Parameter(mp_model.net0.weight.detach().clone().to(cpu_device))
        local_model.net0.bias = torch.nn.Parameter(mp_model.net0.bias.detach().clone().to(cpu_device))
        local_model.net1.weight = torch.nn.Parameter(mp_model.net1.weight.detach().clone().to(cpu_device))
        local_model.net1.bias = torch.nn.Parameter(mp_model.net1.bias.detach().clone().to(cpu_device))

        # Compare parity between DDP with model parallelism using ZeRO and
        # a local model using a local optimizer
        zero_optim = ZeroRedundancyOptimizer(
            ddp_model.parameters(),
            optimizer_class=torch.optim.Adam,
            parameters_as_bucket_view=parameters_as_bucket_view,
            lr=LR
        )
        local_optim = torch.optim.Adam(local_model.parameters(), lr=LR)
        inputs = [torch.randn(20, 10) for _ in range(NUM_INPUTS)]

        for _ in range(NUM_EPOCHS):
            for input in inputs:
                def closure_local():
                    local_optim.zero_grad()
                    local_loss = local_model(input).abs().sum()
                    local_loss.backward()
                    return local_loss

                def closure_ddp():
                    zero_optim.zero_grad()
                    ddp_loss = ddp_model(input).abs().sum()
                    ddp_loss.backward()
                    return ddp_loss

                local_loss = cast(torch.Tensor, local_optim.step(closure=closure_local))
                ddp_loss = cast(torch.Tensor, zero_optim.step(closure=closure_ddp)).to(cpu_device)

                assert torch.allclose(
                    local_loss, ddp_loss
                ), "Losses differ between local optim and ZeroRedundancyOptimizer"

                for local_p, ddp_p in zip(local_model.parameters(), ddp_model.parameters()):
                    ddp_p = ddp_p.to(cpu_device)
                    assert torch.allclose(local_p, ddp_p), "Models differ after a step"

    @common_distributed.skip_if_lt_x_gpu(4)
    def test_zero_model_parallel_with_bucket_view(self):
        """
        Check that ZeRO works with model parallelism where layers are sharded
        across devices when ``parameters_as_bucket_view=True``.
        """
        if self.rank >= 2:
            return
        self.dist_init(self.rank, world_size=2)
        self._test_zero_model_parallel(parameters_as_bucket_view=True)

    @common_distributed.skip_if_lt_x_gpu(4)
    def test_zero_model_parallel_without_bucket_view(self):
        """
        Check that ZeRO works with model parallelism where layers are sharded
        across devices when ``parameters_as_bucket_view=False``.
        """
        if self.rank >= 2:
            return
        self.dist_init(self.rank, world_size=2)
        self._test_zero_model_parallel(parameters_as_bucket_view=False)


if __name__ == "__main__":
    # ! unittest should not be used here, else the tests are not properly registered
    common_utils.run_tests()
