import logging
import os
import sys
import enum

from typing import Callable, List, NamedTuple

import torch.distributed.autograd as dist_autograd
from torch.distributed import rpc
from torch.distributed.optim import DistributedOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_utils import TestCase, run_tests
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as multiprocessing

from torch._utils_internal import TEST_MASTER_ADDR as MASTER_ADDR
from torch._utils_internal import TEST_MASTER_PORT as MASTER_PORT


D_SPARSE = 3
D_DENSE = 2
D_OUT = 2
NUM_EM_ROW = 3


def init_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s %(filename)s:%(lineno)s %(levelname)s p:%(process)s t:%(thread)s: %(message)s"
    )
    console.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(console)
    logger.propagate = False
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


class FeatureSet(NamedTuple):
    dense_features: torch.Tensor
    sparse_features: torch.Tensor


class RemoteEM(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        gLogger.info(f"Initing RemoteEM with {num_embeddings} {embedding_dim}")
        super(RemoteEM, self).__init__()
        init_em = [1] * embedding_dim
        self.em = nn.EmbeddingBag(
            num_embeddings,
            embedding_dim,
            _weight=torch.Tensor([init_em] * num_embeddings),
        )

    def forward(self, input: torch.Tensor):
        gLogger.info(f"Running RemoteEM.forward() on: {input}")
        return self.em(input)


class HybridModel(nn.Module):
    def __init__(
        self, remote_em_rref: rpc.RRef, process_group_name_for_ddp: str = None
    ):
        super(HybridModel, self).__init__()
        self.rref = remote_em_rref
        self.local_net = nn.Linear(D_DENSE + D_SPARSE, D_OUT)
        if process_group_name_for_ddp is not None:
            gLogger.info(f"Use DDP for the local net.")
            self.local_net = DDP(
                self.local_net,
                process_group=process_group_name,
                check_reduction=True,
            )
        gLogger.info(
            f"HybridModel has {len(list(self.parameters()))} groups of parameters."
        )

    def forward(self, input: FeatureSet):
        gLogger.info(f"Running HybridModel.forward on {input}")
        sparse = _remote_method(
            RemoteEM.forward, self.rref, input.sparse_features
        )
        assert sparse.sharp[0] == input.dense_features.share[0]
        x = torch.cat((input.dense_features, sparse), 1)
        gLogger.info(f"Running the local net on joined feature: {x}")
        return self.local_net(x)


class RpcContext:
    def __init__(
        self, name: str, rank: int, world_size: int, group_name: str = ""
    ):
        self.group_name = group_name
        self.name = name
        gLogger.info(
            f"Initing RPC group [{group_name}]: {name} {rank} out of {world_size} peers."
        )
        rpc.init_rpc(name=name, rank=rank, world_size=world_size)

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        gLogger.info(f"Shutting down RPC group [{self.group_name}] from process self.name.")
        if exc_type is not None:
            raise exc_value
        rpc.shutdown()


class DdpMode(enum.Enum):
    # Don't apply DDP
    NONE = enum.auto()
    # Apply DDP to the top level nn.Module
    OUTSIDE = enum.auto()
    # Embed DDP inside the top level nn.Module
    INSIDE = enum.auto()


class TestDdpWithRpc(TestCase):
    TRAINER_NAME_TEMPLATE = "trainer:{:02d}"
    REMOTE_WORKER_NAME = "remote_worker"
    TRAINER_GROUP = "trainer_group"
    NUM_TRAINERS = 1
    MASTER_NAME = "master"

    @classmethod
    def _remote_worker(cls):
        gLogger.info(f"Starting the remote worker...")
        # This RPC group is only for the master and remote worker.
        with RpcContext(
            name=cls.REMOTE_WORKER_NAME,
            rank=1,
            world_size=2,
            group_name="master/remote RPC",
        ):
            # They master will make a RPC and create a rref.
            pass

        with RpcContext(
            name=cls.REMOTE_WORKER_NAME,
            rank=cls.NUM_TRAINERS,
            world_size=cls.NUM_TRAINERS + 1,
            group_name="trainers/remote RPC",
        ):
            pass
        gLogger.info(f"Exiting remote worker.")

    def spawn_remote_worker(self):
        remote_worker_process = multiprocessing.Process(
            target=self._remote_worker, name=self.REMOTE_WORKER_NAME
        )
        remote_worker_process.start()
        self.processes.append(remote_worker_process)

    def setUp(self):
        super(TestDdpWithRpc, self).setUp()

        os.environ["MASTER_ADDR"] = str(MASTER_ADDR)
        os.environ["MASTER_PORT"] = str(MASTER_PORT)
        # os.environ["WORLD_SIZE"] = str(self.NUM_TRAINERS + 1)

        self.processes: List[multiprocessing.Process] = []
        self.remote_em_rref: rpc.RRef = None

        self.spawn_remote_worker()

        with RpcContext(
            name=self.MASTER_NAME,
            rank=0,
            world_size=2,
            group_name="master/remote RPC",
        ):
            gLogger.info(f"Creating a remote em.")
            # Start the remote server first before creating a net rref on it.
            self.remote_em_rref = rpc.remote(
                self.REMOTE_WORKER_NAME, RemoteEM, args=(NUM_EM_ROW, D_SPARSE)
            )

    def tearDown(self):
        super(TestDdpWithRpc, self).tearDown()
        self.join_processes()
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

    @classmethod
    def _trainer(
        cls,
        remote_em_rref: rpc.RRef,
        rank: int,
        func: Callable[[nn.Module], None],
        ddp_mode: DdpMode,
    ):
        trainer_name = cls.TRAINER_NAME_TEMPLATE.format(rank)
        gLogger.info(f"Starting trainer {trainer_name}")
        if ddp_mode in (DdpMode.OUTSIDE, DdpMode.INSIDE):
            gLogger.info(f"Initing trainer process group for DDP.")
            dist.init_process_group(
                # TODO: test other rpc backend.
                "gloo",
                group_name=cls.TRAINER_GROUP,
                rank=rank,
                world_size=cls.NUM_TRAINERS,
            )
        gLogger.info(f"Creating a model on trainer #{rank}")
        model = HybridModel(
            remote_em_rref,
            cls.TRAINER_GROUP if ddp_mode == DdpMode.INSIDE else None,
        )
        if ddp_mode == DdpMode.OUTSIDE:
            gLogger.info(f"Apply DDP to the top level module. ")
            model = DDP(model)

        # This group includes the remote worker
        with RpcContext(
            name=trainer_name,
            rank=rank,
            world_size=cls.NUM_TRAINERS + 1,
            group_name="trainers/remote RPC",
        ):
            func(model)

        if ddp_mode in (DdpMode.OUTSIDE, DdpMode.INSIDE):
            gLogger.info(f"Destroying down trainer process group.")
            dist.destroy_process_group()

    def spawn_trainers(
        self, func: Callable[[nn.Module], None], ddp_mode: DdpMode
    ):
        for rank in range(self.NUM_TRAINERS):
            process = multiprocessing.Process(
                target=self._trainer,
                name=self.TRAINER_NAME_TEMPLATE.format(rank),
                args=(self.remote_em_rref, rank, func, ddp_mode),
            )
            process.start()
            self.processes.append(process)

    def test_no_ddp(self):
        def run_one_forward(model: nn.Module):
            n = 2
            features = FeatureSet(
                sparse_features=torch.LongTensor(
                    [[0, NUM_EM_ROW - 1], [NUM_EM_ROW - 1]]
                ),
                dense_features=torch.rannd((n, D_DENSE)),
            )
            gLogger.info(f"Forward pass on {model(torch.randn((2, 4)))}")

        self.spawn_trainers(run_one_forward, DdpMode.NONE)


if __name__ == "__main__":
    run_tests()
