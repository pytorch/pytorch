import logging
import os
import sys
import enum

from typing import Callable, List, NamedTuple

import torch.distributed.autograd as dist_autograd
from torch.distributed import rpc
from torch.nn.parallel import DistributedDataParallel
from torch.testing._internal.common_utils import TestCase, run_tests
import torch
from torch import optim
import torch.distributed.distributed_c10d as dist_c10d
import torch.distributed as c10d
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as multiprocessing
from torch.distributed.optim import DistributedOptimizer

from torch._utils_internal import TEST_MASTER_ADDR as MASTER_ADDR
from torch._utils_internal import TEST_MASTER_PORT as MASTER_PORT


NUM_EM_ROW = 2
NUM_CATEGORIES = 2
D_SPARSE = 3
D_DENSE = 2
D_HID = 3
D_OUT = 2

NUM_TRAINERS = 2
# Trainers + the master + the remote worker
WORLD_SIZE = NUM_TRAINERS + 2

TRAINER_NAME_TEMPLATE = "trainer:{}"
REMOTE_WORKER_NAME = "remote_worker"
TRAINER_GROUP = "trainer_group"
MASTER_NAME = "master"
DDP_TRAINER_RANKS = list(range(1, NUM_TRAINERS + 1))
TRAINER_NAMES = [
    TRAINER_NAME_TEMPLATE.format(rank) for rank in DDP_TRAINER_RANKS
]
REMOTE_WORKER_RANK = NUM_TRAINERS + 1

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
    return logger


gLogger = init_logger()


class FeatureSet(NamedTuple):
    """ A feature set has 2 types of features"""

    dense_features: torch.Tensor
    sparse_features: torch.LongTensor
    labels: torch.LongTensor


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


def check_process_group_type(
    process_name: str, process_group: dist.ProcessGroup
):
    if not isinstance(process_group, dist.ProcessGroup):
        gLogger.warning(
            f"The process group in {process_name} process is of type {type(process_group)}"
        )


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
        return self.em(input, offsets=torch.LongTensor(range(input.shape[0])))

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
            f"Initing process group [{self.group_name}] by {self.name} with ranks {self.ranks}"
        )
        self.group = dist_c10d.new_group(ranks=self.ranks)
        return self.group

    def __exit__(self, exc_type, exc_value, traceback):
        gLogger.info(
            f"Destroy process group [{self.group_name}] from process {self.name}."
        )
        if exc_type is not None:
            raise exc_value
        dist.destroy_process_group(self.group)


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

        self.criterion = torch.nn.CrossEntropyLoss()
        self.remote_em_rref = remote_em_rref
        self.remote_net_rref = remote_net_rref
        self.hybrid_module = HybridModel(
            self.remote_em_rref,
            self.remote_net_rref,
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
            output = self.hybrid_module(input)
            return self.criterion(output, input.labels)

    def do_mini_batch(self, mini_batch: FeatureSet):
        gLogger.info(f"Doing a mini batch on {mini_batch}")
        loss = 0

        def optimize_remote_parameters():
            gLogger.info(f"Optimizing the remote parameters.")
            dist_optim = DistributedOptimizer(
                optim.SGD, [self.remote_em_rref, self.remote_net_rref], lr=0.05
            )
            dist_optim.step()

        with dist_autograd.context() as context_id:
            local_optim = optim.SGD(self.hybrid_module.parameters(), lr=0.05)
            local_optim.zero_grad()

            output = self.hybrid_module.forward(mini_batch)
            loss = self.criterion(output, mini_batch.labels)
            dist_autograd.backward([loss])
            grads_for_local_params = dist_autograd.get_gradients(context_id)
            gLogger.info(f"Distributed grads: {grads_for_local_params}")
            # TODO: use the local optimize to update local parameters
            with torch.no_grad():
                for param in self.hybrid_module.parameters():
                    if param in grads_for_local_params:
                        param += 0.05 * grads_for_local_params[param]
                    else:
                        gLogger.error(
                            f"Param not in distributed autograd: {param}"
                        )

            gLogger.info(
                f"Optimizing the {len(list(self.hybrid_module.parameters()))} local parameters"
            )
            local_optim.step()

        gLogger.info(
            f"Local parameters: {list(self.hybrid_module.parameters())}"
        )
        return loss


def get_training_examples():
    n = 8
    training_examples = FeatureSet(
        dense_features=torch.zeros((n, D_DENSE)),
        sparse_features=torch.zeros(n, dtype=torch.long),
        labels=torch.zeros(n, dtype=torch.long),
    )
    idx = 0
    for x in range(2):
        for y in range(2):
            for z in range(2):
                training_examples.dense_features[idx, :] = torch.Tensor(
                    (x, y)
                )
                training_examples.sparse_features[idx] = z
                training_examples.labels[idx] = x ^ y ^ z
                idx += 1
    return training_examples


class TestDdpWithRpc(TestCase):
    @classmethod
    def _remote_worker_process(cls):
        gLogger.info(f"Starting the remote worker...")
        with RpcContext(
            name=REMOTE_WORKER_NAME,
            rank=REMOTE_WORKER_RANK,
            world_size=WORLD_SIZE,
            group_name="trainers/remote RPC",
        ), ProcessGroupContext(
            REMOTE_WORKER_NAME,
            ranks=DDP_TRAINER_RANKS,
            group_name="trainers_ddp",
        ) as process_group:
            check_process_group_type("remote worker", process_group)
            dist.barrier()
            gLogger.info(f"The remote worker is running.")

        gLogger.info(f"Exiting remote worker.")
        # exit to avoid run teardown() for fork processes
        sys.exit(0)

    def spawn_remote_worker(self):
        remote_worker_process = multiprocessing.Process(
            target=self._remote_worker_process, name=REMOTE_WORKER_NAME
        )
        remote_worker_process.start()
        self.processes.append(remote_worker_process)

    @classmethod
    def _trainer_process(cls, rank: int):
        gLogger.info(f"Starting the trainer #{rank}...")
        with RpcContext(
            name=TRAINER_NAME_TEMPLATE.format(rank),
            rank=rank,
            world_size=WORLD_SIZE,
            group_name="trainers/remote RPC",
        ), ProcessGroupContext(
            name=TRAINER_NAME_TEMPLATE.format(rank),
            ranks=DDP_TRAINER_RANKS,
            group_name="trainers_ddp",
        ) as process_group:
            gTrainerProcessGroup[rank] = process_group
            check_process_group_type(f"trainer #{rank}", process_group)
            dist.barrier()
            gLogger.info(f"Trainer #{rank} is running.")

        gLogger.info(f"Exiting trainer #{rank}...")
        # exit to avoid run teardown() for fork processes
        sys.exit(0)

    def spawn_trainers(self):
        for rank in range(1, NUM_TRAINERS + 1):
            trainer = multiprocessing.Process(
                target=self._trainer_process,
                name=TRAINER_NAME_TEMPLATE.format(rank),
                args=(rank,),
            )
            trainer.start()
            self.processes.append(trainer)

    @classmethod
    def do_test_on_master(
        cls,
        ddp_mode: DdpMode,
        trainer_method: Callable[[FeatureSet], None],
        remote_em_rref: rpc.RRef,
        remote_net_rref: rpc.RRef,
    ):
        # Wait for all trainers to get their own process group populated in
        # 'gTrainerProcessGroup'
        dist.barrier()
        trainer_rrefs = []
        for rank, trainer in zip(DDP_TRAINER_RANKS, TRAINER_NAMES):
            trainer_rrefs.append(
                rpc.remote(
                    trainer,
                    Trainer,
                    args=(remote_em_rref, remote_net_rref, ddp_mode, rank + 1),
                )
            )

        for epoch in range(3):
            futures = []
            for trainer_rref in trainer_rrefs:
                futures.append(
                    _remote_method_async(
                        trainer_method, trainer_rref, get_training_examples()
                    )
                )
            log_loss = 0
            for future in futures:
                log_loss += future.wait()
            gLogger.info(f"Log loss at epoch #{epoch}: {log_loss}")

    @classmethod
    def _master_process(
        cls, ddp_mode: DdpMode, trainer_method: Callable[[FeatureSet], None]
    ):
        with RpcContext(
            "master_process", 0, WORLD_SIZE, "trainer/remote RPC"
        ), ProcessGroupContext(
            "master", DDP_TRAINER_RANKS, "trainers_ddp"
        ) as process_group:
            gLogger.info(f"Running the master process...")
            check_process_group_type("master", process_group)
            remote_em_rref = rpc.remote(
                REMOTE_WORKER_NAME, RemoteEM, args=(NUM_EM_ROW, D_SPARSE)
            )
            remote_net_rref = rpc.remote(
                REMOTE_WORKER_NAME, RemoteNet, args=(D_DENSE + D_SPARSE, D_HID)
            )
            gLogger.info(f"Created remote rrefs on master")
            cls.do_test_on_master(
                ddp_mode, trainer_method, remote_em_rref, remote_net_rref
            )

    def spawn_master_process(
        self, ddp_mode: DdpMode, trainer_method: Callable[[FeatureSet], None]
    ):
        master_process = multiprocessing.Process(
            target=self._master_process,
            name=MASTER_NAME,
            args=(ddp_mode, trainer_method),
        )
        master_process.start()
        self.processes.append(master_process)

    def setUp(self):
        super(TestDdpWithRpc, self).setUp()

        os.environ["MASTER_ADDR"] = str(MASTER_ADDR)
        os.environ["MASTER_PORT"] = str(MASTER_PORT)
        # os.environ["WORLD_SIZE"] = str(NUM_TRAINERS + 1)

        self.processes: List[multiprocessing.Process] = []

        self.spawn_remote_worker()
        self.spawn_trainers()

    def tearDown(self):
        super(TestDdpWithRpc, self).tearDown()
        self.join_processes()
        for p in self.processes:
            p.terminate()

    def join_processes(self):
        for p in self.processes:
            p.join()

    def test_forward_no_ddp(self):
        self.spawn_master_process(DdpMode.NONE, Trainer.do_forward_without_grad)

    def test_forward_ddp_outside(self):
        self.spawn_master_process(
            DdpMode.OUTSIDE, Trainer.do_forward_without_grad
        )

    def test_training_no_ddp(self):
        self.spawn_master_process(DdpMode.NONE, Trainer.do_mini_batch)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    run_tests()
