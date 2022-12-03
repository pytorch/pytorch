import torch

first_activation_dim = 1000
num_batches = 1000
output_channels = [1024, 512, 2048, 512]
input_channels = [512, 512, 512, 2048]
dtype = torch.float16

for output_channel, input_channel in zip(output_channels, input_channels):
    activation = torch.rand(num_batches, first_activation_dim, input_channel, device="cuda", dtype=dtype)
    weight_batched = torch.rand(1, output_channel, input_channel, device="cuda", dtype=dtype)
    weight = torch.rand(output_channel, input_channel, device="cuda", dtype=dtype)
    res = torch.empty(num_batches, first_activation_dim, output_channel, device="cuda", dtype=dtype)
    bias = torch.rand(output_channel, device="cuda", dtype=dtype)

    num_warmup_iters = 10
    num_active_iters = 100

    # benchmark cusparselt linear
    cusparse_linear = torch.classes.cusparselt.CusparseLtLinear(weight_batched)
    cusparse_linear.init(activation, res, bias)
    cusparse_linear.prune()
    cusparse_linear.compress()
    cusparse_linear.search_matmul_algo()

    for i in range(num_warmup_iters):
        cusparse_linear.masked_mm()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for i in range(num_active_iters):
        cusparse_linear.masked_mm()
    end_event.record()
    torch.cuda.synchronize()
    sparse_time = start_event.elapsed_time(end_event) / num_active_iters

    # benchmark dense linear
    for i in range(num_warmup_iters):
        torch.nn.functional.linear(activation, weight, bias=None)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for i in range(num_active_iters):
        torch.nn.functional.linear(activation, weight, bias=bias)
    end_event.record()
    torch.cuda.synchronize()
    dense_time = start_event.elapsed_time(end_event) / num_active_iters

    print("input channel =", input_channel, "; output channel =", output_channel, "speedup:", dense_time/sparse_time)
