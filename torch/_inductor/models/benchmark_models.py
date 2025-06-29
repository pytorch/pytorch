from mm_kernel_prediction_model import ModelWrapper, get_model
from torch._inductor.kernel_lut import MMProblem
import torch
import time
import triton
import dataclasses

start_time = time.time()
num_itrs = 1
model = ModelWrapper()
for i in range(num_itrs):
    model = ModelWrapper()
end_time = time.time()
print(f"Time to load model: {(end_time - start_time) / num_itrs} seconds")

m = 1024
n = 1024
k = 1024
M_dtype = torch.float16
K_dtype = torch.float16
out_dtype = torch.float16

problem = MMProblem(
    B=1,
    M=m,
    N=n,
    K=k,
    M_dtype=M_dtype,
    K_dtype=K_dtype,
    out_dtype=out_dtype,
    out_size=(1, m, n),
    out_stride=(1, m, 1),
)
myconfig = triton.runtime.autotuner.Config(
    kwargs = {
        "BLOCK_M": 128,
        "BLOCK_N": 128,
        "BLOCK_K": 32,
        "num_stages": 3,
        "num_warps": 4,
        "GROUP_M": 8,
    }
)

# itrs = 20
# encoding_time = []
# for j in range(0, 2000, 5):
#     print(f"encoding num configs: {j}")
#     choices = [myconfig] * j
#     total_time = 0
#     for i in range(itrs):
#         start_time = time.time()
#         encoded = model.encode(m, n, k, M_dtype, choices)
#         end_time = time.time()
#         total_time += end_time - start_time
#     foo = (end_time - start_time) / itrs
#     encoding_time.append(foo)
import copy
import random

itrs = 20
inference_time = []
for j in range(0, 2000, 5):
    print(f"inference num configs: {j + 1}")
    choices = []
    for arst in range(j + 1):
        foo = copy.copy(myconfig)
        foo.kwargs["BLOCK_M"] += random.randint(1, 3000)
        choices.append(foo)
    total_time = 0
    for i in range(itrs):
        encoded = model.encode(m, n, k, M_dtype, choices)
        start_time = time.time()
        inference = model.inference(encoded)
        end_time = time.time()
        total_time += end_time - start_time
    foo = (end_time - start_time) / itrs
    inference_time.append(foo)
with open("inference_time.txt", "w") as f:
    print("Num configs, Encoding time, Inference time")
    # for i, (en, inf) in enumerate(zip(encoding_time, inference_time)):
    #     print(f"{i * 5}, {en * 1000}, {inf * 1000}"
    for i, inf in enumerate(inference_time):
        print(f"{i * 5 + 1}, {inf * 1000}")
breakpoint()
