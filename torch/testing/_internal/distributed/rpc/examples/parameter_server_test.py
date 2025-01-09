# mypy: allow-untyped-defs

# If you need to modify this file to make this test pass, please also apply same edits accordingly to
# https://github.com/pytorch/examples/blob/master/distributed/rpc/batch/parameter_server.py
# and https://pytorch.org/tutorials/intermediate/rpc_async_execution.html#batch-updating-parameter-server

import threading
from datetime import datetime
from time import perf_counter

import torch
import torch.distributed.rpc as rpc
import torch.nn as nn
from torch import optim

from torch.testing._internal.dist_utils import (
    dist_init,
    worker_name,
)
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import RpcAgentTestFixture

batch_size = 20
in_features = 100
out_features = 30
num_batches = 4


def timed_log(text):
    print(f"{datetime.now().strftime('%H:%M:%S')} {text}")


class BatchUpdateParameterServer:

    def __init__(self, batch_update_size):
        self.model = nn.Linear(in_features, out_features)
        self.lock = threading.Lock()
        self.future_model = torch.futures.Future()
        self.batch_update_size = batch_update_size
        self.curr_update_size = 0
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        for p in self.model.parameters():
            p.grad = torch.zeros_like(p)

    def get_model(self):
        return self.model

    @staticmethod
    @rpc.functions.async_execution
    def update_and_fetch_model(ps_rref, grads):
        self = ps_rref.local_value()
        for p, g in zip(self.model.parameters(), grads):
            if p.grad is None:
                p.grad = g
            else:
                p.grad += g
        with self.lock:
            timed_log(f"PS got {self.curr_update_size}/{self.batch_update_size} updates")
            self.curr_update_size += 1
            fut = self.future_model

            if self.curr_update_size >= self.batch_update_size:
                for p in self.model.parameters():
                    p.grad /= self.batch_update_size
                self.curr_update_size = 0
                self.optimizer.step()
                self.optimizer.zero_grad()
                fut.set_result(self.model)
                timed_log("PS updated model")
                self.future_model = torch.futures.Future()

        return fut


class Trainer:

    def __init__(self, ps_rref):
        self.ps_rref = ps_rref
        self.loss_fn = nn.L1Loss()

    def get_next_batch(self):
        for _ in range(num_batches):
            inputs = torch.randn(batch_size, in_features)
            labels = torch.zeros(batch_size, out_features)
            yield inputs, labels

    def train(self):
        name = rpc.get_worker_info().name
        m = self.ps_rref.rpc_sync().get_model()
        for inputs, labels in self.get_next_batch():
            timed_log(f"{name} processing one batch")
            self.loss_fn(m(inputs), labels).backward()
            timed_log(f"{name} reporting grads")
            m = rpc.rpc_sync(
                self.ps_rref.owner(),
                BatchUpdateParameterServer.update_and_fetch_model,
                args=(self.ps_rref, [p.grad for p in m.cpu().parameters()]),
            )
            timed_log(f"{name} got updated model")


def run_trainer(ps_rref):
    trainer = Trainer(ps_rref)
    trainer.train()


def run_ps(trainers):
    timed_log("Start training")
    start = perf_counter()
    ps_rref = rpc.RRef(BatchUpdateParameterServer(len(trainers)))
    futs = [rpc.rpc_async(trainer, run_trainer, args=(ps_rref,)) for trainer in trainers]

    torch.futures.wait_all(futs)
    stop = perf_counter()
    timed_log("Finish training")
    timed_log(f"Time spent training: {stop - start}s")

class ParameterServerTest(RpcAgentTestFixture):

    @dist_init(setup_rpc=False)
    def test_batch_updating_parameter_server(self):

        if self.rank != 0:
            rpc.init_rpc(
                name=worker_name(self.rank),
                backend=self.rpc_backend,
                rank=self.rank,
                world_size=self.world_size,
                rpc_backend_options=self.rpc_backend_options,
            )
        else:
            rpc.init_rpc(
                name=worker_name(self.rank),
                backend=self.rpc_backend,
                rank=self.rank,
                world_size=self.world_size,
                rpc_backend_options=self.rpc_backend_options,
            )
            run_ps([f"{worker_name(r)}" for r in range(1, self.world_size)])

        rpc.shutdown()
