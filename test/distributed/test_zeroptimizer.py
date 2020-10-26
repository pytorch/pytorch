# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import os

import numpy as np
import unittest
import torch
import torch.distributed as dist
from typing import List, Any
import io
from torch.distributed.optim import ZeROptimizer
from torch.optim import SGD
from torch.testing._internal.common_distributed import skip_if_no_gpu, MultiProcessTestCase

BACKEND = dist.Backend.NCCL if torch.cuda.is_available() else dist.Backend.GLOO  # type: ignore
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _broadcast_object(
    obj: Any, src_rank: int, group: object = dist.group.WORLD, dist_device: torch.device = torch.device("cpu")
) -> Any:
    """
    Either broadcast from master to the fleet (default),
    or use the src setting as the original rank.
    """

    if dist.get_rank() == src_rank:
        # Emit data
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        data = bytearray(buffer.getbuffer())
        length_tensor = torch.LongTensor([len(data)]).to(dist_device)
        data_send_tensor = torch.ByteTensor(data).to(dist_device)
        dist.broadcast(length_tensor, src=src_rank, group=group, async_op=False)
        dist.broadcast(data_send_tensor, src=src_rank, group=group, async_op=False)
    else:
        # Fetch from the source
        length_tensor = torch.LongTensor([0]).to(dist_device)
        dist.broadcast(length_tensor, src=src_rank, group=group, async_op=False)
        data_recv_tensor = torch.empty([int(length_tensor.item())], dtype=torch.uint8, device=dist_device)
        dist.broadcast(data_recv_tensor, src=src_rank, group=group, async_op=False)
        buffer = io.BytesIO(data_recv_tensor.cpu().numpy())
        obj = torch.load(buffer, map_location=dist_device)
    return obj


class TestZeROptimizer(MultiProcessTestCase):
    def setUp(self):
        super(TestZeROptimizer, self).setUp()
        self._fork_processes()

    def tearDown(self):
        try:
            torch.distributed.destroy_process_group()
        except RuntimeError:
            pass

    def dist_init(self, rank):
        store = dist.FileStore(self.file_name, self.world_size)
        return dist.init_process_group(backend=BACKEND, store=store, rank=rank, world_size=self.world_size)

    @skip_if_no_gpu
    def test_step(self):
        self.dist_init(self.rank)
        x = torch.tensor([float(self.rank + 1)], device=torch.device(self.rank))
        m = torch.nn.Linear(1, 1)
        m.weight.data = torch.tensor([[1.0]])
        m.bias.data = torch.tensor([2.0])
        m.to(self.rank)
        o = ZeROptimizer(m.parameters(), optim=SGD, lr=0.1)
        y = m(x)
        y.backward(x)
        for p in m.parameters():
            dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
            p.grad.data /= self.world_size
        o.step()
        self.assertEqual(m.weight, torch.tensor([[0.75]], device=torch.device(self.rank)))
        self.assertEqual(m.bias, torch.tensor([1.85], device=torch.device(self.rank)))

    @skip_if_no_gpu
    def test_step_with_closure(self):
        self.dist_init(self.rank)

        x_val = self.rank + 1
        weight = 1.0
        bias = 2.0
        error = 1.0
        target = torch.tensor([x_val * weight + bias + error], device=torch.device(self.rank))
        loss_fn = torch.nn.L1Loss()

        x = torch.tensor([float(x_val)], device=torch.device(self.rank))
        m = torch.nn.Linear(1, 1)
        m.weight.data = torch.tensor([[weight]])
        m.bias.data = torch.tensor([bias])
        m.to(self.rank)

        o = ZeROptimizer(m.parameters(), optim=SGD, lr=0.1)

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

        self.assertEqual(loss, torch.tensor(error, device=torch.device(self.rank)))
        self.assertEqual(m.weight, torch.tensor([[1.1]], device=torch.device(self.rank)))
        self.assertEqual(m.bias, torch.tensor([2.1], device=torch.device(self.rank)))


class TestZeROptimizerSingleRank(TestZeROptimizer):
    @property
    def world_size(self):
        return 1

    def test_create(self):
        self.dist_init(self.rank)
        params = [torch.rand(1)]
        o = ZeROptimizer(params, optim=SGD, lr=0.01)

    def test_state_dict(self):
        self.dist_init(self.rank)
        x = torch.tensor([1.0], device=DEVICE, requires_grad=True)
        o = ZeROptimizer([x], optim=SGD, lr=0.1, momentum=0.9)
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
        o = ZeROptimizer([x], optim=SGD, lr=0.01)
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
        self.dist_init(self.rank)
        x = torch.tensor([1.0], device=DEVICE, requires_grad=True)
        x2 = torch.tensor([1.0], device=DEVICE, requires_grad=True)
        o = ZeROptimizer([x], optim=SGD, lr=0.01)
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
        self.dist_init(self.rank)

        class SGDWithStepKWArg(torch.optim.SGD):
            def step(self, closure=None, kwarg=[]):
                super().step()
                kwarg.append(5)

        kwarg: List[Any] = []
        x = torch.tensor([1.0], device=DEVICE, requires_grad=True)
        o = ZeROptimizer([x], optim=SGDWithStepKWArg, lr=0.1)
        x.backward()
        o.step(0, kwarg=kwarg)
        self.assertEqual(kwarg, [5])
        self.assertEqual(x, torch.tensor([0.9], device=DEVICE))

    def test_step_with_extra_inner_key(self):
        self.dist_init(self.rank)

        class SGDWithNewKey(torch.optim.SGD):
            # Dummy optimizer which adds a new key to the param groups
            def step(self, closure=None):
                super().step()
                self.param_groups[0]["new_key"] = 0.1

        x = torch.tensor([1.0], device=DEVICE, requires_grad=True)
        o = ZeROptimizer([x], optim=SGDWithNewKey, lr=0.1)
        x.backward()
        o.step()
        self.assertEqual(o.param_groups[0]["new_key"], 0.1)
        self.assertEqual(x, torch.tensor([0.9], device=DEVICE))

    def test_step_without_closure(self):
        self.dist_init(self.rank)

        class SGDWithoutClosure(torch.optim.SGD):
            def step(self):
                return super().step()

        x = torch.tensor([1.0], device=DEVICE, requires_grad=True)
        o = ZeROptimizer([x], optim=SGDWithoutClosure, lr=0.1)
        x.backward()
        o.step()
        self.assertEqual(x, torch.tensor([0.9], device=DEVICE))

    def test_local_state_dict(self):
        self.dist_init(self.rank)

        x = torch.tensor([1.0], device=DEVICE, requires_grad=True)
        o = ZeROptimizer([x], optim=SGD, lr=0.1)
        local_state_dict = o.local_state_dict()
        o = ZeROptimizer([x], optim=SGD, lr=0.01)
        o.load_local_state_dict(local_state_dict)
        # We should now be using a lr of 0.1.
        self.assertEqual(o.optim.param_groups[0]["lr"], 0.1)
        self.assertEqual(o.param_groups[0]["lr"], 0.1)
        x.backward()
        o.step()
        self.assertEqual(x, torch.tensor([0.9], device=DEVICE))

    def test_implicit_local_state_dict(self):
        self.dist_init(self.rank)

        x = torch.tensor([1.0], device=DEVICE, requires_grad=True)
        o = ZeROptimizer([x], optim=SGD, lr=0.1)
        local_state_dict = o.state_dict()
        o = ZeROptimizer([x], optim=SGD, lr=0.01)
        o.load_state_dict(local_state_dict)
        # We should now be using a lr of 0.1.
        self.assertEqual(o.optim.param_groups[0]["lr"], 0.1)
        self.assertEqual(o.param_groups[0]["lr"], 0.1)
        x.backward()
        o.step()
        self.assertEqual(x, torch.tensor([0.9], device=DEVICE))


class TestZeROptimizerTwoRanks(TestZeROptimizer):
    @property
    def world_size(self):
        return 2

    def test_zero_grad(self):
        self.dist_init(self.rank)
        x = torch.rand(1)
        m = torch.nn.Linear(1, 1)
        o = ZeROptimizer(m.parameters(), optim=SGD, lr=0.1)
        y = m(x)
        y.backward(x)
        self.assertNotEqual(m.weight.grad, torch.zeros_like(m.weight))
        self.assertNotEqual(m.weight.grad, torch.zeros_like(m.weight))
        o.zero_grad()
        self.assertFalse(m.weight.grad)
        self.assertFalse(m.bias.grad)


class TestZeROptimizerThreeRanks(TestZeROptimizer):
    @property
    def world_size(self):
        return 3

    def test_add_param_group(self):
        self.dist_init(self.rank)

        params = []
        for size in [4, 5, 2, 6, 4]:
            params.append(torch.rand(size, 1))
        o = ZeROptimizer(params, optim=SGD, lr=0.1)
        self.assertEqual(len(o.param_groups), 1)
        o.add_param_group({"params": [torch.rand(3, 1)]})
        self.assertEqual(len(o.param_groups), 2)

        # Verify that added group is added to the correct partition making all have 8 elements.
        self.assertEqual(sum([x.numel() for g in o.optim.param_groups for x in g["params"]]), 8)
        self.assertEqual(len(o.optim.param_groups), 2)

    def test_sharding(self):
        self.dist_init(self.rank)
        params = []
        for size in [5, 4, 2, 6, 4, 3]:
            params.append(torch.rand(size, 1))
        o = ZeROptimizer(params, optim=SGD, lr=0.1)
        self.assertEqual(sum([x.numel() for x in o.optim.param_groups[0]["params"]]), 8)

    def test_collect_shards(self):
        self.dist_init(self.rank)
        device = torch.device(self.rank) if torch.cuda.device_count() > 1 else torch.device("cpu")
        reference_rank = 0

        # Run a dummy step so that the optimizer state dict exists
        batch, input_width, hidden, target_width = 3, 20, 10, 5
        target = torch.rand((batch, target_width), device=device)
        inputs = torch.rand((batch, input_width), device=device)

        model = torch.nn.Sequential(torch.nn.Linear(input_width, hidden), torch.nn.Linear(hidden, target_width))
        model.to(device)

        loss_fn = torch.nn.L1Loss()
        loss_fn.to(device)

        # With SGD, Momentum is required to get a state to shard
        optimizer = ZeROptimizer(model.parameters(), optim=SGD, lr=0.1, momentum=0.99)

        def closure():
            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_fn(output, target)
            loss.backward()
            return loss

        _ = optimizer.step(closure=closure)

        # Update the optimizer state on the reference rank
        optimizer.consolidate_state_dict(recipient_rank=reference_rank)

        # Fetch the state on the reference rank
        # - check that it has the correct size
        # - load it again
        if self.rank == reference_rank:
            optimizer_state_dict = optimizer.state_dict()
            self.assertEqual(len(optimizer_state_dict["state"]), self.world_size)
        else:
            optimizer_state_dict = {}

        optimizer_state_dict = _broadcast_object(
            optimizer_state_dict, src_rank=reference_rank, group=dist.group.WORLD, dist_device=device
        )

        # Load the optimizer state dict
        optimizer.load_state_dict(optimizer_state_dict)


class TestZeROptimizerSixRanks(TestZeROptimizer):
    @property
    def world_size(self):
        return 6

    def test_multiple_groups(self):
        # Only work with the even ranks, to check that the global_rank indexing is properly used
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29501"
        dist.init_process_group(backend="gloo", rank=self.rank, world_size=self.world_size)
        sub_group_ranks = [0, 2, 4]
        process_group = torch.distributed.new_group(ranks=sub_group_ranks, backend="gloo")

        # Make sure that all the ranks get different training data
        # So that the sync check in between their models is meaningful
        torch.manual_seed(self.rank)
        np.random.seed(self.rank)

        # Standard deep learning setup
        device = "cpu"
        epochs, batch, input_width, hidden, target_width = 5, 3, 20, 10, 5
        loss_fn = torch.nn.L1Loss().to(device)

        def check(optimizer):
            # Just run a couple of epochs, check that the model is properly updated
            for _ in range(epochs):
                target = torch.rand((batch, target_width), device=device)
                inputs = torch.rand((batch, input_width), device=device)

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
            model = torch.nn.Sequential(torch.nn.Linear(input_width, hidden), torch.nn.Linear(hidden, target_width)).to(
                device
            )

            # With SGD, Momentum is required to get a state to shard
            optimizer = ZeROptimizer(
                model.parameters(), optim=SGD, lr=0.1, momentum=0.99, group=process_group, bucket_cap_kb=2 ** 10
            )
            check(optimizer)

            # Model not-fitting in the broadcast bucket
            model = torch.nn.Sequential(torch.nn.Linear(input_width, hidden), torch.nn.Linear(hidden, target_width)).to(
                device
            )

            # With SGD, Momentum is required to get a state to shard
            optimizer = ZeROptimizer(
                model.parameters(), optim=SGD, lr=0.1, momentum=0.99, group=process_group, bucket_cap_kb=0
            )
            check(optimizer)


if __name__ == "__main__":
    unittest.main()
