import logging
import os
import sys
import enum

from typing import Callable, List, NamedTuple

import torch.distributed.autograd as dist_autograd
from torch.distributed import rpc
from torch.distributed.optim import DistributedOptimizer
from torch.nn.parallel import DistributedDataParallel
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
)
import torch
import torch.distributed.distributed_c10d as dist_c10d
import torch.distributed as c10d
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as multiprocessing

from torch._utils_internal import TEST_MASTER_ADDR as MASTER_ADDR
from torch._utils_internal import TEST_MASTER_PORT as MASTER_PORT


NUM_EM_ROW = 2
D_SPARSE = 3
D_DENSE = 2
D_HID = 3
D_OUT = 2

NUM_TRAINERS = 1
# Trainers + the master + the remote worker
WORLD_SIZE = NUM_TRAINERS + 2

gTrainerProcessGroup: [dist.ProcessGroup] = [None] * WORLD_SIZE


class DdpMode(enum.Enum):
    # Don't apply DDP
    NONE = enum.auto()
    # Apply DDP to the top level nn.Module
    OUTSIDE = enum.auto()
    # Embed DDP inside the top level nn.Module
    INSIDE = enum.auto()


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


def _remote_method_async(method, rref, *args, **kwargs):
    args_tup = tuple([method, rref] + list(args))
    return rpc.rpc_async(
        rref.owner(), _call_method, args=args_tup, kwargs=kwargs
    )


class FeatureSet(NamedTuple):
    """ A feature set has 2 types of features"""

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

    def print_parameters(self):
        gLogger.info(f"RemoteEM params: {self.parameters}")


class RemoteNet(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        gLogger.info(f"Initing RemoteNet with {d_in} {d_out}")
        super(RemoteNet, self).__init__()
        self.fc = nn.Linear(d_in, d_out)
        self.relu = nn.ReLU()

    def forward(self, input: torch.Tensor):
        gLogger.info(f"Running RemoteNet.forward() on: {input}")
        return self.relu(self.fc(input))

    def print_parameters(self):
        gLogger.info(f"RemoteNet params: {self.parameters}")


class HybridModel(nn.Module):
    def __init__(
        self,
        remote_em_rref: rpc.RRef,
        remote_net_rref: rpc.RRef,
        process_group_for_ddp: dist.ProcessGroup = None,
    ):
        super(HybridModel, self).__init__()
        self.remote_em_rref = remote_em_rref
        self.remote_net_rref = remote_net_rref
        self.fc1 = nn.Linear(D_DENSE, D_DENSE)
        self.fc2 = nn.Linear(D_HID, D_OUT)

        if process_group_for_ddp is not None:
            gLogger.info(f"Use DDP for the local nets.")
            self.fc1 = DistributedDataParallel(
                self.fc1,
                process_group=process_group_for_ddp,
                check_reduction=True,
            )
            self.fc2 = DistributedDataParallel(
                self.fc2,
                process_group=process_group_for_ddp,
                check_reduction=True,
            )

        gLogger.info(
            f"HybridModel has {len(list(self.parameters()))} groups of parameters."
        )

    def forward(self, input: FeatureSet):
        gLogger.info(f"Running HybridModel.forward on {input}")
        sparse = _remote_method(
            RemoteEM.forward, self.remote_em_rref, input.sparse_features
        )
        # The same size of mini batch.
        assert sparse.shape[0] == input.dense_features.shape[0]
        dense = self.fc1(input.dense_features)
        x = torch.cat((dense, sparse), 1)
        gLogger.info(f"Concatenated feature: {x}")
        x = _remote_method(RemoteNet.forward, self.remote_net_rref, x)
        return self.fc2(x)


class RpcContext:
    def __init__(
        self, name: str, rank: int, world_size: int, group_name: str = ""
    ):
        self.name = name
        self.rank = rank
        self.world_size = world_size
        self.group_name = group_name

    def __enter__(self):
        gLogger.info(
            f"Initing RPC [{self.group_name}] by {self.name} with rank "
            f"#{self.rank} out of {self.world_size} peers."
        )
        rpc.init_rpc(name=self.name, rank=self.rank, world_size=self.world_size)

    def __exit__(self, exc_type, exc_value, traceback):
        gLogger.info(
            f"Shutting down RPC group [{self.group_name}] from process {self.name}."
        )
        if exc_type is not None:
            raise exc_value
        rpc.shutdown()


class ProcessGroupContext:
    def __init__(self, name: str, ranks: [int], group_name: str = ""):
        self.name = name
        self.ranks = ranks
        self.group_name = group_name
        self.group = None

    def process_group(self) -> dist.ProcessGroup:
        return self.group

    def __enter__(self) -> dist.ProcessGroup:
        gLogger.info(
            f"Initing process group [{self.group_name}] by {self.name} with ranks {self.ranks}",
        )
        self.group = dist_c10d.new_group(ranks=self.ranks,)
        gLogger.info(f"{type(self.group)}")
        return self.group

    def __exit__(self, exc_type, exc_value, traceback):
        gLogger.info(
            f"Destroy process group [{self.group_name}] from process {self.name}."
        )
        if exc_type is not None:
            raise exc_value
        # Don't destroy it because it's a synchronous point among all processes.
        # dist.destroy_process_group(self.group)


class Trainer:
    def __init__(
        self,
        remote_em_rref: rpc.RRef,
        remote_net_rref: rpc.RRef,
        ddp_mode: DdpMode,
        rank: int,
    ):
        process_group_for_ddp = gTrainerProcessGroup[rank]
        gLogger.info(
            f"Initing a trainer with process group {process_group_for_ddp} ..."
        )

        self.hybrid_module = HybridModel(
            remote_em_rref,
            remote_net_rref,
            process_group_for_ddp if ddp_mode in (DdpMode.INSIDE,) else None,
        )
        if ddp_mode == DdpMode.OUTSIDE:
            gLogger.info(f"Wrapping the whole hybride module into DDP.")
            self.hybrid_module = DistributedDataParallel(
                self.hybrid_module,
                process_group=process_group_for_ddp,
                check_reduction=True,
            )
        gLogger.info(f"Succeeded in creating a HybridModel instance.")

    def do_forward_without_grad(self, input: FeatureSet):
        gLogger.info(f"Doing a forward pass on {input}")
        with torch.no_grad():
            return self.hybrid_module(input)

    def do_mini_batch(self, mini_batch: FeatureSet):
        pass


class TestDdpWithRpc(TestCase):
    TRAINER_NAME_TEMPLATE = "trainer:{}"
    REMOTE_WORKER_NAME = "remote_worker"
    TRAINER_GROUP = "trainer_group"
    DDP_TRAINER_RANKS = list(range(1, NUM_TRAINERS + 1))
    REMOTE_WORKER_RANK = NUM_TRAINERS + 1

    @classmethod
    def _remote_worker_process(cls):
        gLogger.info(f"Starting the remote worker...")
        with RpcContext(
            name=cls.REMOTE_WORKER_NAME,
            rank=cls.REMOTE_WORKER_RANK,
            world_size=WORLD_SIZE,
            group_name="trainers/remote RPC",
        ):
            with ProcessGroupContext(
                cls.REMOTE_WORKER_NAME,
                ranks=cls.DDP_TRAINER_RANKS,
                group_name="trainers_ddp",
            ) as process_group:
                if isinstance(process_group, dist.ProcessGroup):
                    process_group.barrier().wait()
                else:
                    gLogger.error(
                        f"The process group in the remote worker is type of {type(process_group)}"
                    )
                gLogger.info(f"The remote worker is running.")

        gLogger.info(f"Exiting remote worker.")
        # exit to avoid run teardown() for fork processes
        sys.exit(0)

    def spawn_remote_worker(self):
        remote_worker_process = multiprocessing.Process(
            target=self._remote_worker_process, name=self.REMOTE_WORKER_NAME,
        )
        remote_worker_process.start()
        self.processes.append(remote_worker_process)

    @classmethod
    def _trainer_process(cls, rank: int):
        gLogger.info(f"Starting the trainer #{rank}...")
        with RpcContext(
            name=cls.TRAINER_NAME_TEMPLATE.format(rank),
            rank=rank,
            world_size=WORLD_SIZE,
            group_name="trainers/remote RPC",
        ):
            with ProcessGroupContext(
                name=cls.TRAINER_NAME_TEMPLATE.format(rank),
                ranks=cls.DDP_TRAINER_RANKS,
                group_name="trainers_ddp",
            ) as process_group:
                gTrainerProcessGroup[rank] = process_group
                if isinstance(process_group, dist.ProcessGroup):
                    process_group.barrier().wait()
                else:
                    gLogger.error(
                        f"The process group in the trainer #{rank} is of type {type(process_group)}"
                    )
                gLogger.info(f"Trainer #{rank} is running.")

        gLogger.info(f"Exiting trainer #{rank}...")
        # exit to avoid run teardown() for fork processes
        sys.exit(0)

    def spawn_trainers(self):
        for rank in range(1, NUM_TRAINERS + 1):
            trainer = multiprocessing.Process(
                target=self._trainer_process,
                name=self.TRAINER_NAME_TEMPLATE.format(rank),
                args=(rank,),
            )
            trainer.start()
            self.processes.append(trainer)
            self.trainer_names.append(self.TRAINER_NAME_TEMPLATE.format(rank))

    def setUp(self):
        super(TestDdpWithRpc, self).setUp()

        os.environ["MASTER_ADDR"] = str(MASTER_ADDR)
        os.environ["MASTER_PORT"] = str(MASTER_PORT)
        # os.environ["WORLD_SIZE"] = str(NUM_TRAINERS + 1)

        self.processes: List[multiprocessing.Process] = []
        self.remote_em_rref: rpc.RRef = None
        self.remote_net_rref: rpc.RRef = None
        self.trainer_names = []
        self.trainer_rrefs = []

        self.spawn_remote_worker()
        self.spawn_trainers()

        self.rpc_context = RpcContext(
            "master_process", 0, WORLD_SIZE, "trainer/remote RPC"
        )
        self.rpc_context.__enter__()

        self.remote_em_rref = rpc.remote(
            self.REMOTE_WORKER_NAME, RemoteEM, args=(NUM_EM_ROW, D_SPARSE)
        )
        self.remote_net_rref = rpc.remote(
            self.REMOTE_WORKER_NAME, RemoteNet, args=(D_DENSE + D_SPARSE, D_HID)
        )

        self.process_group_context = ProcessGroupContext(
            "master", self.DDP_TRAINER_RANKS, "trainers_ddp"
        )
        self.process_group = self.process_group_context.__enter__()

    def tearDown(self):
        super(TestDdpWithRpc, self).tearDown()

        self.process_group_context.__exit__(None, None, None)
        self.rpc_context.__exit__(None, None, None)

        self.join_processes()
        for p in self.processes:
            p.terminate()

    def join_processes(self):
        for p in self.processes:
            p.join()

    def create_trainers(self, ddp_mode: DdpMode):
        # Wait for all trainers to get their own process group populated in
        # 'gTrainerProcessGroup'
        if isinstance(self.process_group, dist.ProcessGroup):
            self.process_group.barrier().wait()
        else:
            gLogger.error(
                f"The process group in the master process is type of {type(self.process_group)}"
            )
        for rank, trainer in enumerate(self.trainer_names):
            self.trainer_rrefs.append(
                rpc.remote(
                    trainer,
                    Trainer,
                    args=(
                        self.remote_em_rref,
                        self.remote_net_rref,
                        ddp_mode,
                        rank + 1,
                    ),
                )
            )

    def do_test_setup(self, ddp_mode: DdpMode):
        self.create_trainers(ddp_mode)
        futures = []
        for trainer_rref in self.trainer_rrefs:
            futures.append(
                _remote_method_async(
                    Trainer.do_forward_without_grad,
                    trainer_rref,
                    FeatureSet(
                        sparse_features=torch.LongTensor(
                            [[0, NUM_EM_ROW - 1], [NUM_EM_ROW - 1, 0]]
                        ),
                        dense_features=torch.randn((2, D_DENSE)),
                    ),
                )
            )
        for future in futures:
            future.wait()

    def test_setup_no_ddp(self):
        self.do_test_setup(DdpMode.NONE)

    def test_setup_ddp_outside(self):
        self.do_test_setup(DdpMode.OUTSIDE)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    run_tests()
