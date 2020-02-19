import logging
import multiprocessing
import os
import sys

import torch.distributed.autograd as dist_autograd
from torch.distributed import rpc
from torch.distributed.optim import DistributedOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_utils import TestCase, run_tests
import torch
import torch.distributed as dist
import torch.nn as nn

from torch._utils_internal import TEST_MASTER_ADDR as MASTER_ADDR
from torch._utils_internal import TEST_MASTER_PORT as MASTER_PORT


def init_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s %(filename)s:%(lineno)s %(levelname)s: %(message)s"
    )
    console.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(console)
    logger.info("Set up a logger.")
    return logger


gLogger = init_logger()


def _call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)


def _remote_method(method, rref, *args, **kwargs):
    args_tup = tuple([method, rref] + list(args))
    return rpc.rpc_sync(
        rref.owner(), _call_method, args=args_tup, kwargs=kwargs
    )


class SimpleNet(nn.Module):
    def __init__(self, d_in, d_out):
        gLogger.info(f"Initing SimpleNet with {d_in} {d_out}")
        super(SimpleNet, self).__init__()
        self.net = nn.Linear(d_in, d_out)
        self.relu = nn.ReLU()

    def forward(self, input):
        gLogger.info(f"Running SimpleNet.forward on: {input}")
        return self.relu(self.net(input))


class DdpModelWithRpc(nn.Module):
    def __init__(self, remote_server, process_group_name=None):
        ddp_transform = (
            lambda net, process_group_name: net if process_group_name is None else DDP
        )
        super(DdpModelWithRpc, self).__init__()
        self.net1 = ddp_transform(SimpleNet(4, 3), process_group_name)
        self.rref = rpc.remote(remote_server, SimpleNet, args=(3, 2))
        self.net2 = ddp_transform(SimpleNet(2, 1), process_group_name)

    def forward(self, x):
        gLogger.info(f"Running DdpModelWithRpc.forward on {x}")
        x = self.net1(x)
        x = _remote_method(SimpleNet.forward, self.rref, x)
        return self.net2(x)


class TestDdpWithRpc(TestCase):
    TRAINER_NAME_TEMPLATE = "trainer{:02d}"
    REMOTE_WORKER_NAME = "remote_worker"
    TRAINER_GROUP = "trainer_group"
    NUM_TRAINERS = 3

    def setUp(self):
        super(TestDdpWithRpc, self).setUp()
        self.processes = []
        os.environ["MASTER_ADDR"] = str(MASTER_ADDR)
        os.environ["MASTER_PORT"] = str(MASTER_PORT)
        os.environ["WORLD_SIZE"] = str(self.NUM_TRAINERS + 1)

    def tearDown(self):
        super(TestDdpWithRpc, self).tearDown()
        for p in self.processes:
            p.terminate()

    def join_processes(self):
        for p in self.processes:
            p.join()

    # TODO: make it work
    def run_one_mini_batch(self):
        with dist_autograd.context() as context_id:
            # Forward pass (create references on remote nodes).
            rref1 = rpc.remote(dst_name, random_tensor)
            rref2 = rpc.remote(dst_name, random_tensor)
            loss = rref1.to_here() + rref2.to_here()

            # Backward pass (run distributed autograd).
            dist_autograd.backward([loss.sum()])

            # Build DistributedOptimizer.
            dist_optim = DistributedOptimizer(
                optim.SGD, [rref1, rref2], lr=0.05
            )

            # Run the distributed optimizer step.
            dist_optim.step()

    def run_trainer(self, rank, func, use_ddp=True):
        trainer_name = self.TRAINER_NAME_TEMPLATE.format(rank)
        gLogger.info(f"Starting trainer: {trainer_name}...")
        if use_ddp:
            dist.init_process_group(
                # TODO: test other rpc backend.
                "gloo",
                group_name=self.TRAINER_GROUP,
                rank=rank,
                world_size=self.NUM_TRAINERS,
            )
        # This group includes the remote worker
        rpc.init_rpc(
            name=trainer_name, rank=rank, world_size=self.NUM_TRAINERS + 1
        )
        self.model = DdpModelWithRpc(
            self.REMOTE_WORKER_NAME, self.TRAINER_GROUP if use_ddp else None
        )
        func()
        rpc.shutdown()
        dist.destroy_process_group()

    def run_remote_worker(self):
        gLogger.info(f"Starting the remote worker...")
        # This group includes the remote worker
        rpc.init_rpc(
            name=self.REMOTE_WORKER_NAME,
            rank=self.NUM_TRAINERS,
            world_size=self.NUM_TRAINERS + 1,
        )
        rpc.shutdown()

    def spawn_processes(self, func, use_ddp=True):
        for rank in range(self.NUM_TRAINERS):
            process = multiprocessing.Process(
                target=self.run_trainer,
                name=self.TRAINER_NAME_TEMPLATE.format(rank),
                args=(rank, func, use_ddp),
            )
            process.start()
            self.processes.append(process)

        remote_worker_process = multiprocessing.Process(
            target=self.run_remote_worker, name=self.REMOTE_WORKER_NAME
        )
        remote_worker_process.start()
        self.processes.append(remote_worker_process)

    def test_rpc(self):
        def run_one_forward():
            gLogger.info(f"Forward pass on {self.model(torch.randn((2, 4)))}")

        self.spawn_processes(run_one_forward, False)
        self.join_processes()


if __name__ == "__main__":
    run_tests()
