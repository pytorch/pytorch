import torch

first_activation_dim = 1000
num_batches = 1000
output_channel = 512
input_channel = 2048
activation = torch.rand(num_batches, first_activation_dim, input_channel, device="cuda", dtype=torch.float16)
weight = torch.rand(num_batches, output_channel, input_channel, device="cuda", dtype=torch.float16)
res = torch.empty(num_batches, first_activation_dim, output_channel, device="cuda", dtype=torch.float16)
bias = torch.rand(output_channel, device="cuda", dtype=torch.float16)

cusparse_linear = torch.classes.cusparselt.CusparseLtLinear(weight)
cusparse_linear.init(activation, res, bias)
cusparse_linear.prune()
cusparse_linear.compress()
cusparse_linear.search_matmul_algo()

num_warmup_iters = 10
for i in range(num_warmup_iters):
    cusparse_linear.masked_mm()
torch.cuda.synchronize()

num_active_iters = 100
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
for i in range(num_active_iters):
    cusparse_linear.masked_mm()
end_event.record()
torch.cuda.synchronize()
sparse_time_per_iter = start_event.elapsed_time(end_event) / num_active_iters
print(sparse_time_per_iter)

cusparselt_time = torch.cusparselt_spmma2(torch.rand(2, 2, device='cuda'), output_channel, first_activation_dim, input_channel)
print(cusparselt_time)
