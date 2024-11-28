import torch
import torch._dynamo.config



torch._inductor.config.aoti_wrapper = True
torch._inductor.config.cpp_wrapper = True
torch._inductor.config.aot_inductor.package = True

# Artificially generate lots of small kernels
torch._inductor.config.realize_reads_threshold = 1
torch._inductor.config.realize_opcount_threshold = 1
torch._inductor.config.max_fusion_size = 1


@torch.compile(fullgraph=True)
def f(a, b):
    for i in range(1000):
        a = a + b * i
    return a


f(torch.randn(2, device="cuda", requires_grad=True), torch.randn(2, device="cuda")).sum().backward()
