import argparse
import io
import os
import random
import shlex
import subprocess
import time

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.optim as optim
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef, TensorPipeRpcBackendOptions
from torch.distributed.rpc.backend_registry import BackendType
from torch.nn.parallel import DistributedDataParallel as DDP


# Config
NUM_TRAINERS = 8
NUM_PS = 8

NUM_EMBEDDINGS = 300
EMBEDDING_DIM = 64

WARMUP_CYCLES = 5


class HybridModel(torch.nn.Module):
    r"""
   The model consists of a sparse part and a dense part. The dense part is an
   nn.Linear module that is replicated across all trainers using
   DistributedDataParallel. The sparse part has nn.EmbeddingBags stored on multiple
   parameter servers.

   The model holds a Remote Reference to the embedding tables on the parameter
   servers.
   """

    def __init__(self, emb_rref_list, device):
        super().__init__()
        self.emb_rref_list = emb_rref_list
        fc1 = torch.nn.Linear(512, 256)
        fc2 = torch.nn.Linear(256, 128)
        relu = torch.nn.ReLU()
        fc3 = torch.nn.Linear(128, 64)
        fc4 = torch.nn.Linear(64, 32)
        fc5 = torch.nn.Linear(32, 8)
        sec = nn.Sequential(fc1, fc2, relu, fc3, fc4, fc5)
        self.ddp = DDP(sec.to(device), device_ids=[device])
        self.device = device

    def forward(self, indices, offsets):
        emb_lookups = []

        for emb_rref in self.emb_rref_list:
            emb_lookups.append(
                emb_rref.rpc_sync().forward(
                    indices, offsets
                )  # embedding_sum(input, offsets)
            )
            emb_lookups_cat = torch.cat(emb_lookups, dim=1)

        # Make sure combined PS dimension is always bigger or equal than the FC input
        assert NUM_PS * EMBEDDING_DIM >= 512
        dim_normalizer = int(NUM_PS * EMBEDDING_DIM / 512)
        emb_lookups_reshaped = emb_lookups_cat.reshape(
            [emb_lookups_cat.shape[0] * dim_normalizer, 512]
        )

        return self.ddp(emb_lookups_reshaped)


def _retrieve_embedding_parameters(emb_rref):
    return [RRef(p) for p in emb_rref.local_value().parameters()]


def _print_header():
    _print_cont("\n")
    _print_cont("%10s" % "")
    for p in [50, 75, 90, 95]:
        _print_cont("%14s%10s" % ("sec/epoch", "epoch/sec"))
    _print_cont("\n")


def _print_benchmark(prefix, nelem, measurements):
    measurements = sorted(measurements)
    _print_cont("%8s:" % prefix)
    for p in [50, 75, 90, 95]:
        v = np.percentile(measurements, p)
        _print_cont("  p%02d:  %1.3fs  %6d/s" % (p, v, nelem / v))
    _print_cont("\n")


def _print_cont(msg):
    print(msg, end="", flush=True)


def _run_printable(cmd):
    proc = subprocess.run(shlex.split(cmd), capture_output=True)  # type: ignore[call-overload]
    assert proc.returncode == 0

    buffer = io.BytesIO()
    torch.save(proc.stdout.decode("utf-8"), buffer)
    input_tensor = torch.ByteTensor(list(buffer.getvalue()))
    input_length = torch.IntTensor([input_tensor.size(0)])

    output = []
    buffer = io.BytesIO(np.asarray(input_tensor).tobytes())
    output.append(torch.load(buffer))
    return output


def _run_trainer(emb_rref_list, rank):
    r"""
   Each trainer runs a forward pass which involves an embedding lookup on the
   8 parameter servers and running nn.Linear locally. During the backward pass,
   DDP is responsible for aggregating the gradients for the dense part
   (nn.Linear) and distributed autograd ensures gradients updates are
   propagated to the parameter servers.
   """

    # Setup the model.
    model = HybridModel(emb_rref_list, rank)

    # Retrieve all model parameters as rrefs for DistributedOptimizer.

    # Retrieve parameters from all embedding tables for the current trainer.
    model_parameter_rrefs = []
    for ind, emb_rref in enumerate(emb_rref_list):
        ps_name = f"ps{ind}"
        model_parameter_rrefs.extend(
            rpc.rpc_sync(ps_name, _retrieve_embedding_parameters, args=(emb_rref,))
        )

    # model.parameters() only includes local parameters.
    for param in model.parameters():
        model_parameter_rrefs.append(RRef(param))

    # Setup distributed optimizer
    opt = DistributedOptimizer(optim.SGD, model_parameter_rrefs, lr=0.05)

    criterion = torch.nn.CrossEntropyLoss()

    def get_next_batch(rank):
        for _ in range(10):
            num_indices = random.randint(20, 50)
            indices = torch.LongTensor(num_indices).random_(0, NUM_EMBEDDINGS)

            # Generate offsets.
            offsets = []
            start = 0
            batch_size = 0

            while start < num_indices:
                offsets.append(start)
                start += random.randint(1, 10)
                batch_size += 1

            offsets_tensor = torch.LongTensor(offsets)
            target = torch.LongTensor(batch_size).random_(8).cuda(rank)

            yield indices, offsets_tensor, target

    measurements = []
    # Include warm-up cycles during training
    for epoch in range(100 + WARMUP_CYCLES):
        start = time.time()
        batch_size = 0

        # create distributed autograd context
        for indices, offsets, target in get_next_batch(rank):
            batch_size += len(target)

            with dist_autograd.context() as context_id:
                output = model(indices, offsets)
                loss = criterion(output, target)

                # Run distributed backward pass
                dist_autograd.backward(context_id, [loss])

                # Run distributed optimizer. Gradients propagated all the way to the parameter servers
                opt.step(context_id)

                # Not necessary to zero grads as each iteration creates a different
                # distributed autograd context which hosts different grads

        measurements.append(time.time() - start)
        # print("Training done for epoch {}".format(epoch))

    # Throw away warm-up measurements
    measurements = measurements[WARMUP_CYCLES:]
    return rank, measurements, batch_size


def run_worker(rank, world_size):
    r"""
   A wrapper function that initializes RPC, calls the function, and shuts down
   RPC.
   """

    # Using different port numbers in TCP init_method for init_rpc and
    # init_process_group to avoid port conflicts.
    rpc_backend_options = TensorPipeRpcBackendOptions()
    rpc_backend_options.init_method = "tcp://localhost:29500"

    # Rank 16. Master
    if rank == (NUM_TRAINERS + NUM_PS):

        rpc.init_rpc(
            "master", rank=rank,
            backend=BackendType.TENSORPIPE,  # type: ignore[attr-defined]
            world_size=world_size
        )


        # Build the Embedding tables on the Parameter Servers.
        emb_rref_list = []
        index = 0
        while index < NUM_PS:
            ps_name = f"ps{index}"
            emb_rref = rpc.remote(
                ps_name,
                torch.nn.EmbeddingBag,
                args=(NUM_EMBEDDINGS, EMBEDDING_DIM),
                kwargs={"mode": "sum"},
            )
            emb_rref_list.append(emb_rref)
            index += 1

        # Run training loop on the trainers.
        futs = []
        for trainer_rank in range(NUM_TRAINERS):
            trainer_name = f"trainer{trainer_rank}"
            fut = rpc.rpc_async(
                trainer_name, _run_trainer, args=(emb_rref_list, trainer_rank)
            )
            futs.append(fut)

        _print_header()

        measurements_all_trainers = []
        batch_size_all_trainers = 0
        # Wait for all training to finish.
        for fut in futs:
            rank, measurements, batch_size = fut.wait()
            _print_benchmark(f"Trainer{rank}", batch_size, measurements)
            batch_size_all_trainers += batch_size
            measurements_all_trainers.append(measurements)

        _print_benchmark("All", batch_size_all_trainers, measurements_all_trainers)

    # Rank 0-7. Trainers
    elif rank >= 0 and rank < NUM_PS:

        # Initialize process group for Distributed DataParallel on trainers.
        dist.init_process_group(
            backend=dist.Backend.GLOO,
            rank=rank,
            world_size=NUM_TRAINERS,
            init_method="tcp://localhost:29501",
        )

        # Initialize RPC. Trainer just waits for RPCs from master.
        trainer_name = f"trainer{rank}"
        rpc.init_rpc(
            trainer_name,
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )

    # Rank 8-15. Parameter Servers
    elif rank >= NUM_TRAINERS and rank < NUM_TRAINERS + NUM_PS:
        ps_name = f"ps{rank - NUM_TRAINERS}"
        rpc.init_rpc(
            ps_name,
            rank=rank,
            world_size=world_size,
            backend=BackendType.TENSORPIPE,  # type: ignore[attr-defined]
            rpc_backend_options=rpc_backend_options,
        )
        # parameter server do nothing
        pass

    # block until all rpcs finish
    rpc.shutdown()


if __name__ == "__main__":
    """ Initializing the distributed environment. """

    output = _run_printable("nvidia-smi topo -m")
    print("-------------------------------------------")
    print("                  Info                     ")
    print("-------------------------------------------")
    print("")
    print(f"* PyTorch version: {torch.__version__}")
    print(f"* CUDA version: {torch.version.cuda}")
    print("")
    print("------------ nvidia-smi topo -m -----------")
    print("")
    print(output[0])
    print("-------------------------------------------")
    print("PyTorch Distributed Benchmark (DDP and RPC)")
    print("-------------------------------------------")

    # Cmd arguments to enable automated runs (e.g. Chronos, SSH, etc).
    parser = argparse.ArgumentParser(description="PyTorch DDP and RPC Benchmark")
    parser.add_argument(
        "--master-addr", type=str, default="localhost", help="Address of master node."
    )
    parser.add_argument("--master-port", type=str, default="29500", help="Master port.")

    parser.add_argument(
        "--number-trainers",
        type=int,
        default=NUM_TRAINERS,
        help="Number of Trainer Nodes.",
    )
    parser.add_argument(
        "--number-ps", type=int, default=NUM_PS, help="Number of Parameter Servers."
    )
    parser.add_argument(
        "--number-embeddings",
        type=int,
        default=NUM_EMBEDDINGS,
        help="Number of test embeddings to be generated.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=EMBEDDING_DIM,
        help="Number of embedding dimensions.",
    )
    parser.add_argument(
        "--warmup-cycles",
        type=int,
        default=WARMUP_CYCLES,
        help="Number of cycles to warm-up each process before running the benchmark.",
    )

    args = parser.parse_args()

    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port

    NUM_TRAINERS = args.number_trainers
    NUM_PS = args.number_ps

    NUM_EMBEDDINGS = args.number_embeddings
    EMBEDDING_DIM = args.embedding_dim

    WARMUP_CYCLES = args.warmup_cycles

    # Defaults:
    #  8 trainers (rank 0-7),
    #  8 parameter servers (rank 8-15),
    #  1 master (rank 16).
    world_size = NUM_TRAINERS + NUM_PS + 1  # Trainers + PS + Master
    mp.spawn(run_worker, args=(world_size,), nprocs=world_size, join=True)
