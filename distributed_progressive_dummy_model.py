import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import CPUOffload, BackwardPrefetch
from torch.distributed.fsdp.wrap import wrap
import time
import os


# Setup distributed
def setup():
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup():
    dist.destroy_process_group()


# Simple Module
class Foo(nn.Module):
    def __init__(self):
        super().__init__()

    @torch._dynamo.dont_skip_tracing
    def forward(self, x):
        x = (x @ x).sin()
        return x


def main():
    setup()

    foo = Foo().to(torch.cuda.current_device())
    foo = FSDP(foo)

    foo_c = torch.compile(foo)

    start_time = time.time()
    batch_times = []
    report_interval = 100

    for i in range(30):
        batch_start = time.time()

        if i % report_interval == 0:
            print(f"Rank {dist.get_rank()} - Batch {i}")
            if i > 0:
                last_batches_time = sum(batch_times[-report_interval:])
                if last_batches_time > 0:
                    qps = min(report_interval, len(batch_times[-report_interval:])) / last_batches_time
                    print(f"Average QPS (last {report_interval} batches): {qps:.2f}")

        x = torch.randn(1000, 1000, requires_grad=True, device=torch.cuda.current_device())
        out = foo_c(x).sum()
        out.backward()

        batch_times.append(time.time() - batch_start)

    cleanup()


if __name__ == "__main__":
    main()
