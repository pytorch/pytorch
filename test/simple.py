import torch

# _100M = 100 * 1024 ** 2
# r = torch.randn(_100M, dtype=torch.float32, device='cuda')
# d = torch.randn(_100M, dtype=torch.float64, device='cuda')
# torch.cuda.synchronize()
# torch.cuda.profiler.start()
# r.add_(d)
# torch.cuda.profiler.stop()
# torch.cuda.synchronize()


# a = torch.zeros(1, dtype=torch.bfloat16)
# a + 1e-6
x = torch.rand(10, 1, dtype=torch.float)
xq = torch.quantize_per_tensor(x, 0.01, 30, torch.quint8)
# xq.int_repr().to(torch.int32)
print(xq.int_repr())