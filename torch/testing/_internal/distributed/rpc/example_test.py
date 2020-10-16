import os
import threading
from datetime import datetime
import math

import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn as nn
from torch import optim

from torch.testing._internal.dist_utils import (
    dist_init,
    get_function_event,
    initialize_pg,
    wait_until_node_failure,
    wait_until_pending_futures_and_users_flushed,
    wait_until_owners_and_forks_on_rank,
    worker_name,
    single_threaded_process_group_agent,
)

from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
    RpcAgentTestFixture,
)

# =========================== Example ===========================
# https://github.com/pytorch/examples/blob/master/distributed/rpc/batch/parameter_server.py

batch_size = 20
image_w = 64
image_h = 64
num_classes = 30
batch_update_size = 5
num_batches = 6

def timed_log(text):
    print(f"{datetime.now().strftime('%H:%M:%S')} {text}")


class BatchUpdateParameterServer(object):

    def __init__(self, batch_update_size=batch_update_size):
        self.model = nn.Linear(100, num_classes)
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
        timed_log(f"PS got {self.curr_update_size}/{batch_update_size} updates")
        for p, g in zip(self.model.parameters(), grads):
            p.grad += g
        with self.lock:
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


class Trainer(object):

    def __init__(self, ps_rref):
        self.ps_rref = ps_rref
        self.loss_fn = nn.MSELoss()
        self.one_hot_indices = torch.LongTensor(batch_size) \
                                    .random_(0, num_classes) \
                                    .view(batch_size, 1)

    def get_next_batch(self):
        for _ in range(num_batches):
            inputs = torch.randn(batch_size, 100)
            labels = torch.zeros(batch_size, num_classes) \
                        .scatter_(1, self.one_hot_indices, 1)
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
    ps_rref = rpc.RRef(BatchUpdateParameterServer())
    futs = []
    for trainer in trainers:
        futs.append(
            rpc.rpc_async(trainer, run_trainer, args=(ps_rref,))
        )

    torch.futures.wait_all(futs)
    timed_log("Finish training")


def run(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    options=rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=16,
        rpc_timeout=0  # infinite timeout
     )
    if rank != 0:
        rpc.init_rpc(
            f"trainer{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        # trainer passively waiting for ps to kick off training iterations
    else:
        rpc.init_rpc(
            "ps",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        run_ps([f"trainer{r}" for r in range(1, world_size)])

    # block until all rpcs finish
    rpc.shutdown()

# =========================== End exmaple ===========================

# if __name__=="__main__":
#     world_size = batch_update_size + 1
#     mp.spawn(run, args=(world_size, ), nprocs=world_size, join=True)

class ExampleTest(RpcAgentTestFixture):
    @dist_init
    def test_batch_updating_parameter_server(self):
        mp.spawn(run, args=(self.world_size, ), nprocs=self.world_size, join=True)
        self.assertEqual(1, 1)

def test_another_one():
    world_size = batch_update_size + 1
    mp.spawn(run, args=(world_size, ), nprocs=world_size, join=True)
