from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
)

from torch.distributed._spmd import (
    AllReduceComm,
    DefaultBucketer,
    DefaultTrigger,
    Engine,
)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as c10d

import copy
import os

class EngineTest(MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        self._fork_processes()

    def tearDown(self):
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @property
    def world_size(self):
        return 2

    def test_engine_without_handler(self):
        net = nn.Linear(10, 10)
        engine = Engine([])
        engine.prepare_module(list(net.parameters()))

    def test_engine_with_default_handlers(self):
        torch.manual_seed(0)

        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size)

        net = nn.Linear(10, 10)
        ddp = copy.deepcopy(net)

        opt_net = optim.SGD(net.parameters(), lr=0.1)
        opt_ddp = optim.SGD(ddp.parameters(), lr=0.1)

        engine = Engine(
            [DefaultTrigger(), DefaultBucketer(), AllReduceComm(pg)]
        )
        engine.prepare_module(list(ddp.parameters()))

        for _ in range(3):
            inputs = torch.randn(10, 10)

            # run ddp
            ddp_inputs = inputs.chunk(self.world_size)
            engine.pre_forward()
            ddp(ddp_inputs[self.rank]).sum().backward()

            # run local model
            net(inputs).sum().backward()

            # verify grads
            for p_net, p_ddp in zip(net.parameters(), ddp.parameters()):
                self.assertEqual(p_net.grad, p_ddp.grad)

            opt_net.step()
            opt_ddp.step
            opt_net.zero_grad()
            opt_ddp.zero_grad()

    def test_invalid_event_graph(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size)

        with self.assertRaisesRegex(
            RuntimeError, "Invalid Event Handling Graph"
        ):
            engine = Engine([DefaultTrigger(), AllReduceComm(pg)])
