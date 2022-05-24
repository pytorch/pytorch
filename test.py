import torch
import copy
import timeit, time
from torchvision import models
from torchvision.models import quantization as quantized_models
from torch.ao.quantization import (
    quantize,
    prepare,
    convert,
    QConfig,
)

num_wait_iters = 10 # 10
num_warmup_iters = 10 # 10
num_active_iters = 200# 200
num_tot_iters = num_wait_iters + num_warmup_iters + num_active_iters
my_schedule = torch.profiler.schedule(
    wait=num_wait_iters,
    warmup=num_warmup_iters,
    active=num_active_iters)

name = "resnet18"
input_value_original = torch.rand(1, 3, 224, 224).to(memory_format=torch.channels_last)
device = "cuda"

# # # FP16 cuda resnet18 benchmark
# input_value = input_value_original.half().to(device=device)
model = models.__dict__[name](pretrained=False).eval().float().to(device=device)

# # Eager mode quantized cuda resnet18 benchmark
qeager = quantized_models.__dict__[name](pretrained=False, quantize=False).eval().float().to(device=device)
qeager.qconfig = torch.ao.quantization.QConfig(
    activation=torch.ao.quantization.observer.HistogramObserver.with_args(
        qscheme=torch.per_tensor_symmetric, dtype=torch.qint8
    ),
    weight=torch.ao.quantization.default_weight_observer
)
qeager.fuse_model()
prepare(qeager, inplace=True)
convert(qeager, inplace=True)

def benchmark_func(model, name, input_value):
    for i in range(num_wait_iters):
        model_out = model(input_value)
    for i in range(num_warmup_iters):
        model_out = model(input_value)
    torch.cuda.synchronize()
    start = time.time()
    for i in range(num_active_iters):
        model_out = model(input_value)
    torch.cuda.synchronize()
    end = time.time()
    print("Runtime for " + name + ": ", ((end - start) * 1000) / num_active_iters, " ms/iter", flush=True)

benchmark_func(model, "fp32", input_value_original.to(device=device))
benchmark_func(model.to(torch.float16), "fp16", input_value_original.to(torch.float16).to(device=device))
benchmark_func(qeager, "Quantized int8", input_value_original.to(device=device))

