import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

import argparse
import os
import time


N_BUCKETS = 10
N_FEATURES = 1024
BATCH_SIZE = 64

parser = argparse.ArgumentParser(description='C++ PS Demo')
parser.add_argument('--world_size', type=int, default=3)
parser.add_argument('--use_rpc', action="store_true", default=False)

args = parser.parse_args()


@rpc.functions.async_execution
def average_gradients(ps_rref, bucket, bucket_id):
    return ps_rref.local_value().add_grad_bucket(bucket, bucket_id)


def ps_hook(ps_rref, bucket):
    bucket_tensor = bucket.get_tensor()
    bucket_id = bucket.get_index()

    return rpc.rpc_async(
        ps_rref.owner(),
        average_gradients,
        args=(ps_rref, bucket_tensor, bucket_id)
    ).then(lambda fut: [fut.wait()])  # DDP hook expects a list of tensors


class PyParameterServer(rpc.ParameterServer):
    def __init__(self, num_trainers, num_buckets):
        super().__init__(num_trainers, num_buckets)

    def __getstate__(self):
        return {}


class Trainer:
    def __init__(self, ps_rref, hook=None):
        self.ps_rref = ps_rref
        model = nn.Sequential(
            *[nn.Linear(N_FEATURES, N_FEATURES, bias=False) for _ in range(N_BUCKETS)]
        ).cuda(0)
        # CUDA_VISIBLE_DEVICES is set
        self.ddp = DDP(model, device_ids=[0], bucket_cap_mb=4 * N_FEATURES * N_FEATURES / (1024 * 1024))
        if hook is not None:
            self.ddp.register_comm_hook(ps_rref, hook)


    def run(self):
        inputs = torch.zeros(BATCH_SIZE, N_FEATURES).cuda(0)
        # warmup
        for _ in range(40):
            self.ddp(inputs).sum().backward()

        # measure
        delays = []
        for _ in range(40):
            torch.cuda.current_stream(0).synchronize()
            tik = time.time()
            self.ddp(inputs).sum().backward()
            torch.cuda.current_stream(0).synchronize()
            tok = time.time()
            delays.append(tok - tik)

        print(f"{rpc.get_worker_info().name} delay: {1000 * sum(delays) / len(delays)}")


def run(rank, world_size, num_gpu_per_node=8):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{rank % num_gpu_per_node}"
    assert torch.cuda.device_count() == 1

    # init RPC, worker0 hosts PS and serves as a coordinator
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=128)
    for r in range(world_size):
        options.set_device_map(f"worker{r}", {0: 0})

    rpc.init_rpc(
        f"worker{rank}",
        rank=rank,
        world_size=world_size,
        rpc_backend_options=options
    )

    if rank != 0:
        # create process group across DDP processes
        torch.distributed.init_process_group(
            "nccl",
            init_method="tcp://localhost:29501",
            rank=rank - 1,
            world_size=world_size - 1
        )

    rpc.api._barrier([f"worker{r}" for r in range(world_size)])

    if rank == 0:
        print(f"{args.world_size - 1} trainers, using {'RPC' if args.use_rpc else 'NCCL'}")
        ps = PyParameterServer(world_size - 1, N_BUCKETS)
        ps_rref = rpc.RRef(ps)

        hook = ps_hook if args.use_rpc else None
        trainers = [
            rpc.remote(f"worker{i}", Trainer, args=(ps_rref, hook))
            for i in range(1, world_size)
        ]

        futs = [trainer.rpc_async().run() for trainer in trainers]
        torch.futures.wait_all(futs)

    rpc.shutdown()


if __name__=="__main__":
    mp.spawn(run, args=(args.world_size,), nprocs=args.world_size, join=True)
