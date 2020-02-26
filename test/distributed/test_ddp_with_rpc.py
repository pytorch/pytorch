#!/usr/bin/env python3

from typing import Callable, NamedTuple
import enum
import logging
import os
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
import sys
>>>>>>> e13cee7da0... Add a Python unit test for combining DDP and Distributed Autograd/Optimizer
=======
>>>>>>> 56d6cd92d3... 1. Removed unused import
=======
import sys
>>>>>>> ebcd8eef67... Add a Python unit test for combining DDP and Distributed Autograd/Optimizer
=======
>>>>>>> 72f09f8f3c... 1. Removed unused import
=======
import sys
>>>>>>> 0614fdde87... Use a file store for init_process_group()
import enum

from typing import Callable, List, NamedTuple
>>>>>>> 1. Removed unused import

from torch.distributed import rpc
<<<<<<< HEAD
from torch.distributed.optim import DistributedOptimizer
<<<<<<< HEAD
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.dist_utils import dist_init
from torch._utils_internal import TEST_MASTER_ADDR as MASTER_ADDR
from torch._utils_internal import TEST_MASTER_PORT as MASTER_PORT
=======
=======
>>>>>>> Add backward()
from torch.nn.parallel import DistributedDataParallel
<<<<<<< HEAD
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
)
>>>>>>> Use a file store for init_process_group()
=======
from torch.testing._internal.common_utils import TestCase, run_tests
>>>>>>> fix things
import torch
from torch import optim
import torch.distributed.distributed_c10d as dist_c10d
import torch.distributed as c10d
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
import torch.distributed.distributed_c10d as dist_c10d
import torch.multiprocessing as multiprocessing
<<<<<<< HEAD
import torch.nn as nn


NUM_EM_ROW = 2
<<<<<<< HEAD
NUM_CATEGORIES = 2
=======
>>>>>>> Use a file store for init_process_group()
=======
from torch.distributed.optim import DistributedOptimizer

from torch._utils_internal import TEST_MASTER_ADDR as MASTER_ADDR
from torch._utils_internal import TEST_MASTER_PORT as MASTER_PORT


NUM_EM_ROW = 2
NUM_CATEGORIES = 2
>>>>>>> Add backward()
D_SPARSE = 3
D_DENSE = 2
D_HID = 3
D_OUT = 2

<<<<<<< HEAD
<<<<<<< HEAD

def get_env_int(key: str, default: int):
    return int(os.environ[key]) if key in os.environ else default


NUM_TRAINERS = get_env_int("trainers", 2)
NUM_EPOCH = get_env_int("epoch", 3)

# Trainers + the master + the remote worker
WORLD_SIZE = NUM_TRAINERS + 2

TRAINER_GROUP = "trainer_group"
TRAINER_RANKS = list(range(1, NUM_TRAINERS + 1))
REMOTE_WORKER_RANK = NUM_TRAINERS + 1
MASTER_RANK = 0

LR = 0.1
=======
NUM_TRAINERS = 1
=======
NUM_TRAINERS = 2
>>>>>>> Spawn a master process to do all driving work
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
>>>>>>> Use a file store for init_process_group()


class DdpMode(enum.Enum):
    # Don't apply DDP
    NONE = enum.auto()
    # Apply DDP to the top level nn.Module
    OUTSIDE = enum.auto()
    # Embed DDP inside the top level nn.Module
    INSIDE = enum.auto()


def init_logger():
    logger = logging.getLogger(__name__)
    level = logging.DEBUG if "debug" in os.environ else logging.INFO
    logger.setLevel(level)
    console = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s %(filename)s:%(lineno)s %(levelname)s p:%(processName)s t:%(threadName)s: %(message)s"
    )
    console.setFormatter(formatter)
    console.setLevel(level)
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


<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> Spawn a master process to do all driving work
def check_process_group_type(
    process_name: str, process_group: dist.ProcessGroup
):
    if not isinstance(process_group, dist.ProcessGroup):
        gLogger.warning(
            f"The process group in {process_name} process is of type {type(process_group)}"
        )
<<<<<<< HEAD
=======
class FeatureSet(NamedTuple):
    """ A feature set has 2 types of features"""

    dense_features: torch.Tensor
    sparse_features: torch.Tensor
>>>>>>> Use a file store for init_process_group()


=======
>>>>>>> Add backward()
=======


>>>>>>> Spawn a master process to do all driving work
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
<<<<<<< HEAD
        gLogger.debug(f"Running RemoteEM.forward() on: {input}")
        return self.em(input, offsets=torch.LongTensor(range(input.shape[0])))

    def get_parameter_rrefs(self):
        gLogger.debug(f"RemoteEM params: {list(self.parameters())}")
        return [rpc.RRef(param) for param in self.parameters()]


class RemoteNet(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        gLogger.info(f"Initing RemoteNet with {d_in} {d_out}")
        super(RemoteNet, self).__init__()
        self.fc = nn.Linear(d_in, d_out)
        self.relu = nn.ReLU()

    def forward(self, input: torch.Tensor):
        gLogger.debug(f"Running RemoteNet.forward() on: {input}")
        return self.relu(self.fc(input))

    def get_parameter_rrefs(self):
        gLogger.debug(f"RemoteNet params: {list(self.parameters())}")
        return [rpc.RRef(param) for param in self.parameters()]
=======
        gLogger.info(f"Running RemoteEM.forward() on: {input}")
        return self.em(input, offsets=torch.LongTensor(range(input.shape[0])))
>>>>>>> Add backward()

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
<<<<<<< HEAD
        self.remote_parameters_rrefs = _remote_method(
            RemoteEM.get_parameter_rrefs, self.remote_em_rref
        ) + _remote_method(RemoteNet.get_parameter_rrefs, self.remote_net_rref)
=======
>>>>>>> Use a file store for init_process_group()
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
        gLogger.debug(f"Running HybridModel.forward on {input}")
        sparse = _remote_method(
            RemoteEM.forward, self.remote_em_rref, input.sparse_features
        )
        # The same size of mini batch.
        assert sparse.shape[0] == input.dense_features.shape[0]
        dense = self.fc1(input.dense_features)
        x = torch.cat((dense, sparse), 1)
<<<<<<< HEAD
        gLogger.debug(f"Concatenated feature: {x}")
        x = _remote_method(RemoteNet.forward, self.remote_net_rref, x)
        return self.fc2(x)

    def print_parameters(self):
        gLogger.debug(f"Local parameters: {list(self.parameters())}")
        gLogger.debug(
            f"Remote parameters: {[param_rref.to_here() for param_rref in self.remote_parameters_rrefs]}"
        )
=======
        gLogger.info(f"Concatenated feature: {x}")
        x = _remote_method(RemoteNet.forward, self.remote_net_rref, x)
        return self.fc2(x)
>>>>>>> Use a file store for init_process_group()


class Trainer:
    def __init__(
        self,
        remote_em_rref: rpc.RRef,
        remote_net_rref: rpc.RRef,
        ddp_mode: DdpMode,
        rank: int,
    ):
<<<<<<< HEAD
        gLogger.info(
            f"Initing trainer process group by traner #{rank} with ranks {TRAINER_RANKS}"
        )
        self.process_group_for_ddp = dist_c10d.new_group(ranks=TRAINER_RANKS)
        check_process_group_type(f"trainer #{rank}", self.process_group_for_ddp)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.remote_em_rref = remote_em_rref
        self.remote_net_rref = remote_net_rref
        self.hybrid_module = HybridModel(
            self.remote_em_rref,
            self.remote_net_rref,
            self.process_group_for_ddp
            if ddp_mode in (DdpMode.INSIDE,)
            else None,
        )
        if ddp_mode == DdpMode.OUTSIDE:
            gLogger.info(f"Wrapping the whole hybride module into DDP.")
            self.hybrid_module = DistributedDataParallel(
                self.hybrid_module,
                process_group=self.process_group_for_ddp,
                check_reduction=True,
            )
        gLogger.info(f"Succeeded in creating a HybridModel instance.")

    def __del__(self):
        dist.destroy_process_group(self.process_group_for_ddp)

    def do_forward_without_grad(self, input: FeatureSet):
        gLogger.debug(f"Doing a forward pass on {input}")
        with torch.no_grad():
            output = self.hybrid_module(input)
            return self.criterion(output, input.labels)
=======
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
>>>>>>> Use a file store for init_process_group()

    def get_real_module(self):
        if isinstance(self.hybrid_module, DistributedDataParallel):
            return self.hybrid_module.module
        else:
            return self.hybrid_module

<<<<<<< HEAD
    def print_parameters(self):
        self.get_real_module().print_parameters()
=======
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
<<<<<<< HEAD
        pass
>>>>>>> Use a file store for init_process_group()

    def do_mini_batch(self, mini_batch: FeatureSet):
        gLogger.info(f"Doing a mini batch on {mini_batch}")
        loss = 0

<<<<<<< HEAD
        with dist_autograd.context() as context_id:
            output = self.hybrid_module.forward(mini_batch)
            with torch.no_grad():
                gLogger.debug(
                    f"Output: {output} softmax: {torch.softmax(output, 1)} labels: {mini_batch.labels}"
                )
            loss = self.criterion(output, mini_batch.labels)
            # grads will be stored in dist_autograd context
            dist_autograd.backward([loss])
            gLogger.debug(
                f"Distributed grads: {dist_autograd.get_gradients(context_id)}"
            )
            gLogger.info(f"Running distributed optimizer...")
            dist_optimizer = DistributedOptimizer(
                optim.SGD,
                self.get_real_module().remote_parameters_rrefs
                + [
                    rpc.RRef(local_param)
                    for local_param in self.hybrid_module.parameters()
                ],
                lr=LR,
            )
            dist_optimizer.step()

            self.print_parameters()

        gLogger.info(f"Loss: {loss}.")
=======
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
>>>>>>> Add backward()
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
<<<<<<< HEAD
                training_examples.dense_features[idx, :] = torch.Tensor((x, y))
                training_examples.sparse_features[idx] = z
                training_examples.labels[idx] = x ^ y ^ z
                idx += 1
    # Split the examples among NUM_TRAINERS trainers
    examples_per_trainer = int(n / NUM_TRAINERS)
    return [
        FeatureSet(
            dense_features=training_examples.dense_features[
                start : start + examples_per_trainer, :
            ],
            sparse_features=training_examples.sparse_features[
                start : start + examples_per_trainer
            ],
            labels=training_examples.labels[
                start : start + examples_per_trainer
            ],
=======
class TestDdpWithRpc(TestCase):
    TRAINER_NAME_TEMPLATE = "trainer:{}"
    REMOTE_WORKER_NAME = "remote_worker"
    TRAINER_GROUP = "trainer_group"
    DDP_TRAINER_RANKS = list(range(1, NUM_TRAINERS + 1))
    REMOTE_WORKER_RANK = NUM_TRAINERS + 1
=======
                training_examples.dense_features[idx, :] = torch.Tensor(
                    (x, y)
                )
                training_examples.sparse_features[idx] = z
                training_examples.labels[idx] = x ^ y ^ z
                idx += 1
    return training_examples
>>>>>>> Spawn a master process to do all driving work


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
<<<<<<< HEAD
            if not isinstance(process_group, dist.ProcessGroup):
                gLogger.error(
                    f"The process group in the remote worker is of type {type(process_group)}"
                )
=======
            check_process_group_type("remote worker", process_group)
>>>>>>> Spawn a master process to do all driving work
            dist.barrier()
            gLogger.info(f"The remote worker is running.")

        gLogger.info(f"Exiting remote worker.")
        # exit to avoid run teardown() for fork processes
        sys.exit(0)

    def spawn_remote_worker(self):
        remote_worker_process = multiprocessing.Process(
<<<<<<< HEAD
<<<<<<< HEAD
            target=self._remote_worker_process, name=self.REMOTE_WORKER_NAME,
>>>>>>> Use a file store for init_process_group()
=======
            target=self._remote_worker_process, name=self.REMOTE_WORKER_NAME
>>>>>>> fix things
=======
            target=self._remote_worker_process, name=REMOTE_WORKER_NAME
>>>>>>> Spawn a master process to do all driving work
        )
        for start in range(0, n, examples_per_trainer)
    ]


class TestDdpWithRpc(MultiProcessTestCase):
    rpc_backend = rpc.backend_registry.BackendType.PROCESS_GROUP
    rpc_backend_options = None

    @property
    def world_size(self) -> int:
        return WORLD_SIZE

    def remote_worker_name(self) -> str:
        # The name has to be consistent with that in 'dist_init' decorator.
        return f"worker{REMOTE_WORKER_RANK}"

    def trainer_name(self, rank):
        # The name has to be consistent with that in 'dist_init' decorator.
        return f"worker{rank}"

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
<<<<<<< HEAD
            if isinstance(process_group, dist.ProcessGroup):
                gLogger.error(
                    f"The process group in the trainer #{rank} is of type {type(process_group)}"
                )
=======
            check_process_group_type(f"trainer #{rank}", process_group)
>>>>>>> Spawn a master process to do all driving work
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
<<<<<<< HEAD
        self._spawn_processes()

    def tearDown(self):
        super(TestDdpWithRpc, self).tearDown()

    @dist_init
    def _remote_worker_process(self):
        process_group_for_ddp = dist_c10d.new_group(ranks=TRAINER_RANKS)
        gLogger.info(f"The remote worker is running.")
        dist.destroy_process_group(process_group_for_ddp)
        gLogger.info(f"Exiting remote worker.")

    @dist_init
    def _trainer_process(self, rank: int):
        gLogger.info(f"Running the trainer #{rank}...")

    def do_test_on_master(
        self,
        ddp_mode: DdpMode,
        trainer_method: Callable[[FeatureSet], None],
        remote_em_rref: rpc.RRef,
        remote_net_rref: rpc.RRef,
    ):
        trainer_rrefs = []
        for rank in TRAINER_RANKS:
            trainer = self.trainer_name(rank)
            trainer_rrefs.append(
                rpc.remote(
                    trainer,
                    Trainer,
                    args=(remote_em_rref, remote_net_rref, ddp_mode, rank),
                )
            )

        training_examples = get_training_examples()
        for epoch in range(NUM_EPOCH):
            futures = []
            for idx, trainer_rref in enumerate(trainer_rrefs):
                futures.append(
                    _remote_method_async(
                        trainer_method, trainer_rref, training_examples[idx]
                    )
                )
            log_loss = 0
            for future in futures:
                log_loss += future.wait()
            gLogger.info(
                f"Log loss at epoch #{epoch}: {log_loss / NUM_TRAINERS}"
            )

    @dist_init
    def _master_process(
        self, ddp_mode: DdpMode, trainer_method: Callable[[FeatureSet], None]
    ):
        gLogger.info(f"Running the master process...")
        process_group_for_ddp = dist_c10d.new_group(ranks=TRAINER_RANKS)
        remote_em_rref = rpc.remote(
            self.remote_worker_name(), RemoteEM, args=(NUM_EM_ROW, D_SPARSE)
        )
        remote_net_rref = rpc.remote(
            self.remote_worker_name(),
            RemoteNet,
            args=(D_DENSE + D_SPARSE, D_HID),
        )
        gLogger.info(f"Created remote rrefs on master")
        self.do_test_on_master(
            ddp_mode, trainer_method, remote_em_rref, remote_net_rref
        )
        dist.destroy_process_group(process_group_for_ddp)

    def _do_test(
        self, ddp_mode: DdpMode, trainer_method: Callable[[FeatureSet], None]
    ):
        if self.rank == MASTER_RANK:
            self._master_process(ddp_mode, trainer_method)
        elif self.rank == REMOTE_WORKER_RANK:
            self._remote_worker_process()
        elif self.rank in TRAINER_RANKS:
            self._trainer_process(self.rank)
        else:
            raise RuntimeError(f"Unknow process rank: {self.rank}")

    def test_forward_no_ddp(self):
        self._do_test(DdpMode.NONE, Trainer.do_forward_without_grad)

    def test_forward_ddp_outside(self):
        self._do_test(DdpMode.OUTSIDE, Trainer.do_forward_without_grad)

<<<<<<< HEAD
    def test_forward_ddp_inside(self):
        self._do_test(DdpMode.INSIDE, Trainer.do_forward_without_grad)

    def test_training_no_ddp(self):
        self._do_test(DdpMode.NONE, Trainer.do_mini_batch)

    def test_training_ddp_outside(self):
        self._do_test(DdpMode.OUTSIDE, Trainer.do_mini_batch)

    def test_training_ddp_inside(self):
        self._do_test(DdpMode.INSIDE, Trainer.do_mini_batch)
=======
    def spawn_trainers(
        self, func: Callable[[nn.Module], None], ddp_mode: DdpMode
    ):
        for rank in range(self.NUM_TRAINERS):
            process = multiprocessing.Process(
                target=self._trainer,
                name=self.TRAINER_NAME_TEMPLATE.format(rank),
                args=(self.remote_em_rref, rank, func, ddp_mode),
=======
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

<<<<<<< HEAD
    def create_trainers(self, ddp_mode: DdpMode):
        if isinstance(self.process_group, dist.ProcessGroup):
            gLogger.error(
                f"The process group in the master process is of type {type(self.process_group)}"
            )
        # Wait for all trainers to get their own process group populated in
        # 'gTrainerProcessGroup'
        dist.barrier()
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

    def do_test(
        self, ddp_mode: DdpMode, trainer_method: Callable[[FeatureSet], None]
    ):
        self.create_trainers(ddp_mode)
        for epoch in range(3):
            futures = []
            for trainer_rref in self.trainer_rrefs:
                futures.append(
                    _remote_method_async(
                        trainer_method, trainer_rref, self.training_examples
                    )
                )
<<<<<<< HEAD
>>>>>>> Use a file store for init_process_group()
=======
<<<<<<< HEAD
>>>>>>> Add backward()
            )
<<<<<<< HEAD
            process.start()
            self.processes.append(process)

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    @staticmethod
    def run_one_forward(model: nn.Module):
        n = 2
        features = FeatureSet(
            sparse_features=torch.LongTensor(
                [[0, NUM_EM_ROW - 1], [NUM_EM_ROW - 1, 0]]
            ),
            dense_features=torch.randn((n, D_DENSE)),
        )
        gLogger.info(f"Forward pass on {model(features)}")
=======
        for future in futures:
            future.wait()
>>>>>>> 0614fdde87... Use a file store for init_process_group()
=======
            log_loss = 0
            for future in futures:
                log_loss += future.wait()
            gLogger.info(f"Log loss at epoch #{epoch}: {log_loss}")

    def test_forward_no_ddp(self):
        self.do_test(DdpMode.NONE, Trainer.do_forward_without_grad)
>>>>>>> 0fcd3c0148... Add backward()
=======
    def test_forward_no_ddp(self):
        self.spawn_master_process(DdpMode.NONE, Trainer.do_forward_without_grad)
>>>>>>> e5b4e7d245... Spawn a master process to do all driving work

    def test_forward_ddp_outside(self):
        self.spawn_master_process(
            DdpMode.OUTSIDE, Trainer.do_forward_without_grad
        )

<<<<<<< HEAD
<<<<<<< HEAD
    def test_forward_with_ddp(self):
        self.spawn_trainers(self.run_one_forward, DdpMode.OUTSIDE)
=======
=======
>>>>>>> ebcd8eef67... Add a Python unit test for combining DDP and Distributed Autograd/Optimizer
    def test_no_ddp(self):
        def run_one_forward(model: nn.Module):
            n = 2
            features = FeatureSet(
                sparse_features=torch.LongTensor(
                    [[0, NUM_EM_ROW - 1], [NUM_EM_ROW - 1, 0]]
                ),
                dense_features=torch.randn((n, D_DENSE)),
            )
            gLogger.info(f"Forward pass on {model(features)}")

        self.spawn_trainers(run_one_forward, DdpMode.NONE)
<<<<<<< HEAD
>>>>>>> e13cee7da0... Add a Python unit test for combining DDP and Distributed Autograd/Optimizer
=======

=======

>>>>>>> 72f09f8f3c... 1. Removed unused import
    @staticmethod
    def run_one_forward(model: nn.Module):
        n = 2
        features = FeatureSet(
            sparse_features=torch.LongTensor(
                [[0, NUM_EM_ROW - 1], [NUM_EM_ROW - 1, 0]]
            ),
            dense_features=torch.randn((n, D_DENSE)),
        )
        gLogger.info(f"Forward pass on {model(features)}")

    def test_forward_without_ddp(self):
        self.spawn_trainers(self.run_one_forward, DdpMode.NONE)

    def test_forward_with_ddp(self):
        self.spawn_trainers(self.run_one_forward, DdpMode.OUTSIDE)
<<<<<<< HEAD
>>>>>>> 56d6cd92d3... 1. Removed unused import
<<<<<<< HEAD
>>>>>>> 1. Removed unused import
=======
=======
>>>>>>> ebcd8eef67... Add a Python unit test for combining DDP and Distributed Autograd/Optimizer
<<<<<<< HEAD
>>>>>>> Add a Python unit test for combining DDP and Distributed Autograd/Optimizer
=======
=======
>>>>>>> 72f09f8f3c... 1. Removed unused import
<<<<<<< HEAD
>>>>>>> 1. Removed unused import
=======
=======
    def test_setup_ddp_outside(self):
        self.do_test_setup(DdpMode.OUTSIDE)
>>>>>>> 0614fdde87... Use a file store for init_process_group()
<<<<<<< HEAD
>>>>>>> Use a file store for init_process_group()
=======
=======
    def test_training_no_ddp(self):
<<<<<<< HEAD
        self.do_test(DdpMode.NONE, Trainer.do_mini_batch)
>>>>>>> 0fcd3c0148... Add backward()
<<<<<<< HEAD
>>>>>>> Add backward()
=======
=======
        self.spawn_master_process(DdpMode.NONE, Trainer.do_mini_batch)
>>>>>>> e5b4e7d245... Spawn a master process to do all driving work
>>>>>>> Spawn a master process to do all driving work


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    run_tests()
