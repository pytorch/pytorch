#!/usr/bin/env python3

import contextlib
import enum
import logging
import os
import threading
from typing import NamedTuple

import torch
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
import torch.nn as nn
from torch.distributed import rpc
from torch.distributed.nn import RemoteModule
from torch.nn.parallel import DistributedDataParallel
from torch.testing._internal.common_distributed import (
    requires_gloo,
    requires_nccl,
    skip_if_lt_x_gpu,
    skip_if_rocm,
)
from torch.testing._internal.dist_utils import INIT_METHOD_TEMPLATE, dist_init
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
    RpcAgentTestFixture,
)


NUM_EM_ROW = 2
D_SPARSE = 3
D_DENSE = 2
D_HID = 3
D_OUT = 1
NUM_TRAINERS = 4
# Trainers + the master + the remote worker
WORLD_SIZE = NUM_TRAINERS + 2
TRAINER_RANKS = list(range(NUM_TRAINERS))
REMOTE_WORKER_RANK = TRAINER_RANKS[-1] + 1
MASTER_RANK = REMOTE_WORKER_RANK + 1


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
    values: torch.Tensor


def _call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)


def _remote_method(method, rref, *args, **kwargs):
    args_tup = tuple([method, rref] + list(args))
    return rpc.rpc_sync(rref.owner(), _call_method, args=args_tup, kwargs=kwargs)


def _remote_method_async(method, rref, *args, **kwargs):
    args_tup = tuple([method, rref] + list(args))
    return rpc.rpc_async(rref.owner(), _call_method, args=args_tup, kwargs=kwargs)


class RemoteEM(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        gLogger.info(f"Initing RemoteEM with {num_embeddings} {embedding_dim}")
        super(RemoteEM, self).__init__()
        init_em = [0.5] * embedding_dim
        self.em = nn.EmbeddingBag(
            num_embeddings,
            embedding_dim,
            _weight=torch.tensor([init_em] * num_embeddings),
        )

    def forward(self, input: torch.Tensor):
        gLogger.debug(f"Running RemoteEM.forward() on: {input}")
        return self.em(input, offsets=torch.LongTensor(range(input.shape[0])))


# Return a linear module with predefined parameters.
def getLinear(d_in, d_out):
    l = nn.Linear(d_in, d_out, bias=False)
    w = torch.ones((d_out, d_in))
    w[0][0] = -1
    w.requires_grad_()
    l.weight.data = w
    return l


class RemoteNet(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        gLogger.info(f"Initing RemoteNet with {d_in} {d_out}")
        super(RemoteNet, self).__init__()
        self.fc = getLinear(d_in, d_out)
        self.relu = nn.ReLU()

    def forward(self, input: torch.Tensor):
        gLogger.debug(f"Running RemoteNet.forward() on: {input}")
        return self.relu(self.fc(input))


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
        self.fc1 = getLinear(D_DENSE, D_DENSE)
        self.fc2 = getLinear(D_HID, D_OUT)

        self.non_ddp_params = tuple(self.fc1.parameters()) + tuple(
            self.fc2.parameters()
        )
        self.ddp_params = ()

        if process_group_for_ddp is not None:
            self.non_ddp_params, self.ddp_params = (
                tuple(self.fc1.parameters()),
                tuple(self.fc2.parameters()),  # type: ignore[assignment]
            )
            gLogger.info("Use DDP for the second local net.")
            self.fc2 = DistributedDataParallel(
                self.fc2, check_reduction=True, process_group=process_group_for_ddp
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


class Trainer:
    def __init__(
        self,
        remote_em_rref: rpc.RRef,
        remote_net_rref: rpc.RRef,
        ddp_mode: DdpMode,
        rank: int,
    ):
        self.rank = rank
        self.trainer_group = (
            dist.new_group(TRAINER_RANKS)
            if ddp_mode in (DdpMode.INSIDE, DdpMode.OUTSIDE)
            else None
        )
        self.remote_em_rref = remote_em_rref
        self.remote_net_rref = remote_net_rref
        self.hybrid_module = HybridModel(
            self.remote_em_rref,
            self.remote_net_rref,
            self.trainer_group if ddp_mode in (DdpMode.INSIDE,) else None,
        )
        self.ddp_params, self.non_ddp_params = (
            self.hybrid_module.ddp_params,
            self.hybrid_module.non_ddp_params,
        )
        if ddp_mode == DdpMode.OUTSIDE:
            gLogger.info("Wrapping the whole hybrid module into DDP.")
            self.ddp_params += self.non_ddp_params  # type: ignore[assignment]
            self.non_ddp_params = ()
            self.hybrid_module = DistributedDataParallel(  # type: ignore[assignment]
                self.hybrid_module,
                check_reduction=True,
                process_group=self.trainer_group,
            )
        gLogger.info(
            f"Succeeded in creating a HybridModel instance with "
            f"{len(self.ddp_params)} ddp params and {len(self.non_ddp_params)} "
            f"other local params."
        )

    def destroy_pg(self):
        if self.trainer_group:
            dist.destroy_process_group(self.trainer_group)

    def train_batch(
        self,
        mini_batch: FeatureSet,
        trainer_has_less_inputs: bool,
        simulate_uneven_inputs: bool,
    ):
        grads_dict = None

        if not simulate_uneven_inputs:
            input_batches = [mini_batch]
        else:
            # Split into microbatches, and trim to simulate uneven inputs.
            dense_features = mini_batch.dense_features
            sparse_features = mini_batch.sparse_features
            values = mini_batch.values

            dense_microbatch = torch.split(dense_features, 2)
            sparse_microbatch = torch.split(sparse_features, 2)
            values_microbatch = torch.split(values, 2)
            batches = []
            for d, s, v in zip(dense_microbatch, sparse_microbatch, values_microbatch):
                feature_set = FeatureSet(dense_features=d, sparse_features=s, values=v)
                batches.append(feature_set)

            if trainer_has_less_inputs:
                input_batches = batches[: len(batches) // 2]
                gLogger.info(
                    f"""Trainer reduced input patches from {len(batches)}
                    to {len(input_batches)} to simulate uneven inputs."""
                )
            else:
                input_batches = batches

        with self.hybrid_module.join() if simulate_uneven_inputs else contextlib.suppress():  # type: ignore[operator]
            for b in input_batches:
                with dist_autograd.context() as context_id:
                    output = self.hybrid_module.forward(b)
                    loss = (output * mini_batch.values).sum()
                    dist_autograd.backward(context_id, [loss])
                    grads_dict = dist_autograd.get_gradients(context_id)
                    gLogger.info(
                        f"Loss is {loss} for mini batch: {mini_batch}. "
                        f"Grads dict has {len(grads_dict)} entries: {grads_dict}"
                    )
        return (
            tuple(grads_dict[param] for param in self.ddp_params),  # type: ignore[var-annotated]
            tuple(grads_dict[param] for param in self.non_ddp_params),  # type: ignore[index]
        )


def get_training_examples():
    n = 16
    training_examples = FeatureSet(
        dense_features=torch.zeros((n, D_DENSE)),
        sparse_features=torch.zeros(n, dtype=torch.long),  # type: ignore[arg-type]
        values=torch.zeros(n),
    )
    idx = 0
    # Every example has another one that has exactly the same features but an
    # opposite value. Therefore, their grads cancel each other in all-reduce.
    for value in (-1, 1):
        for x in (-1.0 * value, 1.0 * value):
            for y in (1.0 * value, -1.0 * value):
                for z in (0, 1):
                    training_examples.dense_features[idx, :] = torch.tensor((x, y))
                    training_examples.sparse_features[idx] = z
                    training_examples.values[idx] = value
                    idx += 1

    # Split the examples among NUM_TRAINERS trainers
    assert 0 == (n % NUM_TRAINERS)
    examples_per_trainer = int(n / NUM_TRAINERS)
    return [
        FeatureSet(
            dense_features=training_examples.dense_features[
                start : start + examples_per_trainer, :
            ],
            sparse_features=training_examples.sparse_features[  # type: ignore[arg-type]
                start : start + examples_per_trainer
            ],
            values=training_examples.values[start : start + examples_per_trainer],
        )
        for start in range(0, n, examples_per_trainer)
    ]


shutdown_signal = threading.Condition()


def set_shutdown_signal():
    global shutdown_signal
    with shutdown_signal:
        shutdown_signal.notify()


class DdpUnderDistAutogradTest(RpcAgentTestFixture):
    @property
    def world_size(self) -> int:
        return WORLD_SIZE

    def remote_worker_name(self) -> str:
        # The name has to be consistent with that in 'dist_init' decorator.
        return f"worker{REMOTE_WORKER_RANK}"

    def trainer_name(self, rank):
        # The name has to be consistent with that in 'dist_init' decorator.
        return f"worker{rank}"

    def _remote_worker_process(self, ddp_mode):
        gLogger.info("The remote worker is running.")
        dist.init_process_group(
            backend="gloo",
            init_method=INIT_METHOD_TEMPLATE.format(file_name=self.file_name),  # type: ignore[attr-defined]
            world_size=self.world_size,
            rank=self.rank,  # type: ignore[attr-defined]
        )

        if ddp_mode in (DdpMode.INSIDE, DdpMode.OUTSIDE):
            # new_group needs to be called on ranks.
            dist.new_group(TRAINER_RANKS)

        global shutdown_signal
        with shutdown_signal:
            shutdown_signal.wait()
        gLogger.info("Exiting remote worker.")
        dist.destroy_process_group()

    def _trainer_process(self, rank: int):
        gLogger.info(f"Running the trainer #{rank}...")
        gLogger.info(
            f"Initing trainer process group by trainer #{rank} with ranks {TRAINER_RANKS}"
        )
        dist.init_process_group(
            backend="gloo",
            init_method=INIT_METHOD_TEMPLATE.format(file_name=self.file_name),  # type: ignore[attr-defined]
            world_size=self.world_size,
            rank=self.rank,  # type: ignore[attr-defined]
        )

        gLogger.info(f"Waiting for shutdown signal on trainer #{rank}...")

        global shutdown_signal
        with shutdown_signal:
            shutdown_signal.wait()
        gLogger.info(f"Exiting the trainer #{rank}...")
        dist.destroy_process_group()

    def _master_process(self, ddp_mode: DdpMode, simulate_uneven_inputs: bool):
        gLogger.info("Running the master process...")
        dist.init_process_group(
            backend="gloo",
            init_method=INIT_METHOD_TEMPLATE.format(file_name=self.file_name),  # type: ignore[attr-defined]
            world_size=self.world_size,
            rank=self.rank,  # type: ignore[attr-defined]
        )

        remote_em_rref = rpc.remote(
            self.remote_worker_name(), RemoteEM, args=(NUM_EM_ROW, D_SPARSE)
        )
        remote_net_rref = rpc.remote(
            self.remote_worker_name(), RemoteNet, args=(D_DENSE + D_SPARSE, D_HID)
        )
        gLogger.info("Created remote rrefs on master")
        self.do_test_on_master(
            ddp_mode, simulate_uneven_inputs, remote_em_rref, remote_net_rref
        )

    def do_test_on_master(
        self,
        ddp_mode: DdpMode,
        simulate_uneven_inputs: bool,
        remote_em_rref: rpc.RRef,
        remote_net_rref: rpc.RRef,
    ):
        if simulate_uneven_inputs:
            gLogger.info(
                "Running DDP + RPC test with simulating uneven inputs across trainers."
            )

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

        if ddp_mode in (DdpMode.INSIDE, DdpMode.OUTSIDE):
            # new_group needs to be called on ranks.
            dist.new_group(TRAINER_RANKS)

        training_examples = get_training_examples()
        for _ in range(3):
            futures = []
            num_trainers = len(trainer_rrefs)
            for idx, trainer_rref in enumerate(trainer_rrefs):
                # Half the trainers will deplete inputs earlier than the rest.
                trainer_has_less_inputs = (
                    simulate_uneven_inputs and idx < num_trainers // 2
                )
                futures.append(
                    _remote_method_async(
                        Trainer.train_batch,
                        trainer_rref,
                        training_examples[idx],
                        trainer_has_less_inputs,
                        simulate_uneven_inputs,
                    )
                )

            for future in futures:
                ddp_grads, non_ddp_grads = future.wait()
                # When there are uneven inputs, it is not necessary that grads
                # cancel each other out, since some trainers contribute 0 grad.
                if not simulate_uneven_inputs:
                    for grad in ddp_grads:
                        self.assertEqual(  # type: ignore[attr-defined]
                            grad,
                            torch.zeros_like(grad),
                            msg=f"The grad for any ddp parameter should be zeros, because "
                            "the training examples' grads cancel each other. Received "
                            f"gradient {grad}",
                        )
                for grad in non_ddp_grads:
                    self.assertNotEqual(  # type: ignore[attr-defined]
                        grad,
                        torch.zeros_like(grad),
                        msg="The grad for any non-ddp parameter shouldn't be zeros",
                    )

        # Destroy process groups
        for idx, trainer_rref in enumerate(trainer_rrefs):
            _remote_method_async(Trainer.destroy_pg, trainer_rref).wait()

        # Send shutdown signals.
        for rank in TRAINER_RANKS:
            trainer = self.trainer_name(rank)
            rpc.rpc_sync(trainer, set_shutdown_signal, args=())

        rpc.rpc_sync(self.remote_worker_name(), set_shutdown_signal, args=())

    def _do_test(self, ddp_mode, simulate_uneven_inputs=False):
        if self.rank == MASTER_RANK:  # type: ignore[attr-defined]
            self._master_process(ddp_mode, simulate_uneven_inputs)
        elif self.rank == REMOTE_WORKER_RANK:  # type: ignore[attr-defined]
            self._remote_worker_process(ddp_mode)
        elif self.rank in TRAINER_RANKS:  # type: ignore[attr-defined]
            self._trainer_process(self.rank)  # type: ignore[attr-defined]
        else:
            raise RuntimeError(f"Unknow process rank: {self.rank}")  # type: ignore[attr-defined]

    @requires_gloo()
    @dist_init
    def test_backward_no_ddp(self):
        self._do_test(DdpMode.NONE)

    @requires_gloo()
    @dist_init
    def test_backward_ddp_outside(self):
        self._do_test(DdpMode.OUTSIDE)

    @requires_gloo()
    @dist_init
    def test_backward_ddp_outside_uneven_inputs(self):
        self._do_test(DdpMode.OUTSIDE, simulate_uneven_inputs=True)

    @requires_gloo()
    @dist_init
    def test_backward_ddp_inside(self):
        self._do_test(DdpMode.INSIDE)


# Common utils for both CPU and CUDA test suites
class CommonDdpComparisonTest(RpcAgentTestFixture):
    @property
    def world_size(self) -> int:
        return NUM_TRAINERS

    def trainer_name(self, rank):
        # The name has to be consistent with that in 'dist_init' decorator.
        return f"worker{rank}"

    @staticmethod
    def get_remote_grads(rref, context_id):
        return dist_autograd.get_gradients(context_id)[rref.local_value().weight]


class DdpComparisonTest(CommonDdpComparisonTest):
    def _run_test_ddp_comparision(self, simulate_uneven_inputs=False):
        gLogger.info(f"Running trainer rank: {self.rank}")  # type: ignore[attr-defined]
        # Each trainer uses a different random seed. Otherwise, they are going
        # to have exactly the same initial model parameters, input, and
        # therefore grads. That means the grads will be the same before and
        # after DDP's all-reduce.
        torch.manual_seed(self.rank)  # type: ignore[attr-defined]
        dist.init_process_group(
            backend="gloo",
            init_method=INIT_METHOD_TEMPLATE.format(file_name=self.file_name),  # type: ignore[attr-defined]
            world_size=self.world_size,
            rank=self.rank,  # type: ignore[attr-defined]
        )
        net = nn.Linear(2, 3)
        ddp_net = DistributedDataParallel(net)

        # Odd ranks join early if simulate_uneven_inputs.
        num_inputs = 1
        if simulate_uneven_inputs:
            if self.rank % 2 == 0:  # type: ignore[attr-defined]
                num_inputs += 2
        inputs_list = [torch.rand((3, 2)) for _ in range(num_inputs)]

        if simulate_uneven_inputs:
            gLogger.info(f"Rank {self.rank} training with {len(inputs_list)} inputs.")  # type: ignore[attr-defined]

        # Use distributed autograd. The gradients will be in RPC context map.
        grads_dict = {}
        with ddp_net.join(simulate_uneven_inputs):  # type: ignore[operator]
            for i, inputs in enumerate(inputs_list):
                with dist_autograd.context() as context_id:
                    loss = ddp_net(inputs).norm()
                    dist_autograd.backward(context_id, [loss])
                    grads_dict = dist_autograd.get_gradients(context_id)
                gLogger.info(f"Trainer #{self.rank} got grad dict: {grads_dict}")  # type: ignore[attr-defined]

                # Use local autograd. The gradients will be in each variable's '.grad'.
                ddp_net.zero_grad()
                loss = ddp_net(inputs).norm()
                loss.backward()

                # The gradients should be the same
                for param in net.parameters():
                    self.assertTrue(  # type: ignore[attr-defined]
                        param in grads_dict,
                        msg=f"Param {param} is not in dist_auto grad dict {grads_dict} for iteration {i}",
                    )
                    self.assertEqual(  # type: ignore[attr-defined]
                        grads_dict[param],
                        param.grad,
                        msg=f"The grads for param {param} are different under local "
                        f"and dist autograd: {param.grad} \n---\n {grads_dict[param]} for iteration {i}",
                    )
        dist.destroy_process_group()

    @requires_gloo()
    @dist_init
    def test_ddp_comparison(self):
        self._run_test_ddp_comparision()

    @requires_gloo()
    @dist_init
    def test_ddp_comparison_uneven_inputs(self):
        # test with simulating uneven inputs in DDP
        self._run_test_ddp_comparision(simulate_uneven_inputs=True)

    @requires_gloo()
    @dist_init
    def test_ddp_dist_autograd_sparse_grads(self):
        # Each trainer uses a different random seed. Otherwise, they are going
        # to have exactly the same initial model parameters, input, and
        # therefore grads. That means the grads will be the same before and
        # after DDP's all-reduce.
        torch.manual_seed(self.rank)  # type: ignore[attr-defined]
        dist.init_process_group(
            backend="gloo",
            init_method=INIT_METHOD_TEMPLATE.format(file_name=self.file_name),  # type: ignore[attr-defined]
            world_size=self.world_size,
            rank=self.rank,  # type: ignore[attr-defined]
        )

        model = nn.EmbeddingBag(10, 3, sparse=True)
        ddp_model = DistributedDataParallel(model)

        # Different inputs for each
        input = torch.LongTensor(10).random_(0, 10)
        offsets = torch.LongTensor([0, 4])

        # Run local.
        loss = ddp_model(input, offsets).sum()
        loss.backward()

        with dist_autograd.context() as context_id:
            loss = ddp_model(input, offsets).sum()
            dist_autograd.backward(context_id, [loss])
            grads_dict = dist_autograd.get_gradients(context_id)
            self.assertEqual(1, len(grads_dict))  # type: ignore[attr-defined]
            self.assertEqual(model.weight.grad, grads_dict[model.weight])  # type: ignore[attr-defined]

    @requires_gloo()
    @dist_init
    def test_ddp_dist_autograd_local_vs_remote(self):
        # Each trainer uses a different random seed. Otherwise, they are going
        # to have exactly the same initial model parameters, input, and
        # therefore grads. That means the grads will be the same before and
        # after DDP's all-reduce.
        torch.manual_seed(self.rank)  # type: ignore[attr-defined]
        dist.init_process_group(
            backend="gloo",
            init_method=INIT_METHOD_TEMPLATE.format(file_name=self.file_name),  # type: ignore[attr-defined]
            world_size=self.world_size,
            rank=self.rank,  # type: ignore[attr-defined]
        )

        # Use two different remote device input string, w/ and w/o the default
        # device string "cpu", respectively.
        for remote_device in ["worker0/cpu", "worker0"]:
            remote_layer1 = RemoteModule(
                remote_device=remote_device, module_cls=nn.Linear, args=(10, 5, False)  # type: ignore[arg-type]
            )
            layer1 = nn.Linear(10, 5, False)
            # Start with the same parameters for remote and local
            layer1.weight = remote_layer1.module_rref.to_here().weight

            # Run local case.
            layer2 = nn.Linear(5, 1)
            inputs = torch.rand((10, 10))
            ddp_model = DistributedDataParallel(layer2)
            loss = ddp_model(layer1(inputs)).sum()
            loss.backward()

            # Run remote case.
            with dist_autograd.context() as context_id:
                loss = ddp_model(remote_layer1(inputs)).sum()
                dist_autograd.backward(context_id, [loss])
                grads_dict = dist_autograd.get_gradients(context_id)
                dist.barrier()
                self.assertEqual(layer2.weight.grad, grads_dict[layer2.weight])  # type: ignore[attr-defined]
                self.assertEqual(  # type: ignore[attr-defined]
                    layer1.weight.grad,
                    rpc.rpc_sync(
                        "worker0",
                        CommonDdpComparisonTest.get_remote_grads,
                        args=(remote_layer1.module_rref, context_id),
                    ),
                )


class CudaDdpComparisonTest(CommonDdpComparisonTest):
    @skip_if_lt_x_gpu(NUM_TRAINERS)
    @requires_nccl()
    @dist_init
    @skip_if_rocm
    def test_ddp_dist_autograd_local_vs_remote_gpu(self):
        # Each trainer uses a different random seed. Otherwise, they are going
        # to have exactly the same initial model parameters, input, and
        # therefore grads. That means the grads will be the same before and
        # after DDP's all-reduce.
        torch.manual_seed(self.rank)  # type: ignore[attr-defined]
        dist.init_process_group(
            backend="gloo",
            init_method=INIT_METHOD_TEMPLATE.format(file_name=self.file_name),  # type: ignore[attr-defined]
            world_size=self.world_size,
            rank=self.rank,  # type: ignore[attr-defined]
        )

        remote_layer1 = RemoteModule(
            remote_device="worker0/cpu", module_cls=nn.Linear, args=(10, 7, False)  # type: ignore[arg-type]
        )
        layer1 = nn.Linear(10, 7, False)
        # Start with the same parameters for remote and local
        layer1.weight = remote_layer1.module_rref.to_here().weight

        layer2 = nn.Linear(7, 5).cuda(self.rank)  # type: ignore[attr-defined]
        ddp_layer2 = DistributedDataParallel(layer2, device_ids=[self.rank])  # type: ignore[attr-defined]

        remote_layer3 = RemoteModule(
            remote_device="worker0/cpu", module_cls=nn.Linear, args=(5, 3, False)  # type: ignore[arg-type]
        )
        layer3 = nn.Linear(5, 3, False)
        # Start with the same parameters for remote and local
        layer3.weight = remote_layer3.module_rref.to_here().weight

        layer4 = nn.Linear(3, 1).cuda(self.rank)  # type: ignore[attr-defined]
        ddp_layer4 = DistributedDataParallel(layer4, device_ids=[self.rank])  # type: ignore[attr-defined]

        # Run local case.
        inputs = torch.rand((10, 10))
        loss = ddp_layer4(
            layer3(ddp_layer2(layer1(inputs).cuda(self.rank)).cpu()).cuda(self.rank)  # type: ignore[attr-defined]
        ).sum()
        loss.backward()

        # Run remote case.
        with dist_autograd.context() as context_id:
            loss = ddp_layer4(
                remote_layer3(
                    ddp_layer2(remote_layer1(inputs).cuda(self.rank)).cpu()  # type: ignore[attr-defined]
                ).cuda(self.rank)  # type: ignore[attr-defined]
            ).sum()
            dist_autograd.backward(context_id, [loss])
            grads_dict = dist_autograd.get_gradients(context_id)
            dist.barrier()
            self.assertEqual(  # type: ignore[attr-defined]
                layer1.weight.grad,
                rpc.rpc_sync(
                    "worker0",
                    CommonDdpComparisonTest.get_remote_grads,
                    args=(remote_layer1.module_rref, context_id),
                ),
            )
            self.assertEqual(layer2.weight.grad, grads_dict[layer2.weight])  # type: ignore[attr-defined]
            self.assertEqual(  # type: ignore[attr-defined]
                layer3.weight.grad,
                rpc.rpc_sync(
                    "worker0",
                    CommonDdpComparisonTest.get_remote_grads,
                    args=(remote_layer3.module_rref, context_id),
                ),
            )
            self.assertEqual(layer4.weight.grad, grads_dict[layer4.weight])  # type: ignore[attr-defined]
