import torch

first_activation_dim = 1000
num_batches = 1000

output_channels = [1024, 512, 2048, 512]
input_channels = [512, 512, 512, 2048]
for output_channel, input_channel in zip(output_channels, input_channels):
    activation = torch.rand(num_batches, first_activation_dim, input_channel, device="cuda", dtype=torch.float16)
    weight = torch.rand(output_channel, input_channel, device="cuda", dtype=torch.float16)
    bias = torch.rand(output_channel, device="cuda", dtype=torch.float16)
    for i in range(10):
        torch.nn.functional.linear(activation, weight, bias=None)


    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    num_iters = 100
    for i in range(num_iters):
        torch.nn.functional.linear(activation, weight, bias=bias)
    end_event.record()
    torch.cuda.synchronize()
    dense_time = start_event.elapsed_time(end_event) / num_iters

    # cusparselt kernel benchmark of spmma2 -- timing is done in C++ backend
    cusparselt_time = torch.cusparselt_spmma2(torch.rand(2, 2, device='cuda'), output_channel, first_activation_dim, input_channel)
    # cusparselt_time = torch.cusparselt_spmma(torch.rand(2, 2, device='cuda'), output_channel, first_activation_dim, input_channel)
    print(cusparselt_time, dense_time)
    print("output_channel =", output_channel, "; input_channel =", input_channel, "; speedup =", cusparselt_time / dense_time)



# using pytorch profiler
    # from torch.profiler import profile, ProfilerActivity
    # def trace_handler(p):
    #     output = p.key_averages().table(sort_by="self_cpu_time_total", row_limit=10)
    #     p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")

    # my_schedule = torch.profiler.schedule(
    #     wait=5,
    #     warmup=5,
    #     active=20)
    # with profile(
    #         activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #         schedule=my_schedule,
    #         on_trace_ready=trace_handler) as prof:
    #     for i in range(30):
    #         print(i)
    #         torch.nn.functional.linear(activation, weight, bias=bias)
    #         prof.step()
    # print("fp32 benchmark result:")
    # print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=100))
