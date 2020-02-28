#!/usr/bin/env python3

from typing import Callable, NamedTuple
import enum
import logging
import os

from torch.distributed import rpc
from torch.distributed.optim import DistributedOptimizer
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.dist_utils import dist_init
from torch._utils_internal import TEST_MASTER_ADDR as MASTER_ADDR
from torch._utils_internal import TEST_MASTER_PORT as MASTER_PORT
import torch
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
import torch.distributed.distributed_c10d as dist_c10d
import torch.multiprocessing as multiprocessing
import torch.nn as nn


NUM_EM_ROW = 2
NUM_CATEGORIES = 2
D_SPARSE = 3
D_DENSE = 2
D_HID = 3
D_OUT = 2


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
        self.remote_parameters_rrefs = _remote_method(
            RemoteEM.get_parameter_rrefs, self.remote_em_rref
        ) + _remote_method(RemoteNet.get_parameter_rrefs, self.remote_net_rref)
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
        gLogger.debug(f"Concatenated feature: {x}")
        x = _remote_method(RemoteNet.forward, self.remote_net_rref, x)
        return self.fc2(x)

    def print_parameters(self):
        gLogger.debug(f"Local parameters: {list(self.parameters())}")
        gLogger.debug(
            f"Remote parameters: {[param_rref.to_here() for param_rref in self.remote_parameters_rrefs]}"
        )


class Trainer:
    def __init__(
        self,
        remote_em_rref: rpc.RRef,
        remote_net_rref: rpc.RRef,
        ddp_mode: DdpMode,
        rank: int,
    ):
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

    def get_real_module(self):
        if isinstance(self.hybrid_module, DistributedDataParallel):
            return self.hybrid_module.module
        else:
            return self.hybrid_module

    def print_parameters(self):
        self.get_real_module().print_parameters()

    def do_mini_batch(self, mini_batch: FeatureSet):
        gLogger.info(f"Doing a mini batch on {mini_batch}")
        loss = 0

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

    def setUp(self):
        super(TestDdpWithRpc, self).setUp()

        os.environ["MASTER_ADDR"] = str(MASTER_ADDR)
        os.environ["MASTER_PORT"] = str(MASTER_PORT)
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

    def test_forward_ddp_inside(self):
        self._do_test(DdpMode.INSIDE, Trainer.do_forward_without_grad)

    def test_training_no_ddp(self):
        self._do_test(DdpMode.NONE, Trainer.do_mini_batch)

    def test_training_ddp_outside(self):
        self._do_test(DdpMode.OUTSIDE, Trainer.do_mini_batch)

    def test_training_ddp_inside(self):
        self._do_test(DdpMode.INSIDE, Trainer.do_mini_batch)


if __name__ == "__main__":
    run_tests()
