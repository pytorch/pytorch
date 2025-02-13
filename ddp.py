import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
if rank == 0:
    torch.manual_seed(123)
else:
    torch.manual_seed(1234)
torch.cuda.set_device(rank)

# @torch.compile(fullgraph=True)
def fn(model, x):
    return model(x).sum()

x = torch.randn(1000, 1000, device="cuda")
model = torch.nn.Sequential(
    torch.nn.Linear(1000, 1000, device="cuda"),
    torch.nn.Linear(1000, 1000, device="cuda"),
    torch.nn.Linear(1000, 1000, device="cuda"),
    torch.nn.Linear(1000, 1000, device="cuda"),
    torch.nn.Linear(1000, 1000, device="cuda"),
    torch.nn.Linear(1000, 1000, device="cuda"),
    torch.nn.Linear(1000, 1000, device="cuda"),
    torch.nn.Linear(1000, 1000, device="cuda"),
    torch.nn.Linear(1000, 1000, device="cuda"),
    torch.nn.Linear(1000, 1000, device="cuda"),
    torch.nn.Linear(1000, 1000, device="cuda"),
)
model = DDP(model, bucket_cap_mb=1)

prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
with prof:
    for i in range(3):
        print(f"RUNNING ITERATION {i}")
        loss = model(x).sum()
        with torch._dynamo.compiled_autograd._enable(torch.compile):
            loss.backward()
            print("done bwd")
        # loss.backward()
        # for i,param in enumerate(model.parameters()):
        #     assert param.grad is not None
            # print(f"rank={rank}, param[{i}].grad = {param.grad}")
        model.zero_grad()

print("exporting")
prof.export_chrome_trace(f"ddp_trace_rank_{rank}.json")
print("destroying")
dist.destroy_process_group()
print("done")
