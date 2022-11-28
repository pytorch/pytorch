import torch

import torch._dynamo
import torch._inductor.config as inductor_config
from benchmark_helper import time_with_torch_timer

inductor_config.triton.mm = "triton"


@torch._dynamo.optimize("inductor", nopython=True)
def inductor_mm(a, b):
    return torch.mm(a, b)


def torch_mm_relu(a, b):
    return torch.nn.functional.relu(torch.mm(a, b))


def torch_mm(a, b):
    return torch.mm(a, b)


if __name__ == "__main__":
    # Real shapes from torchbench
    a_shapes = [
        [2048, 768],
        [64, 1280],
        [2048, 768],
        [32, 2048],
        [1, 39200],
        [128, 3072],
        [16, 1280],
    ]
    b_shapes = [
        [768, 3072],
        [1280, 1000],
        [768, 768],
        [2048, 1000],
        [39200, 50],
        [3072, 1000],
        [1280, 1000],
    ]

    # Artificial larger shapes
    a_shapes += [[10240, 512], [10240, 1024]]
    b_shapes += [[512, 10240], [1024, 10240]]

    for i in range(len(a_shapes)):
        a_shape = a_shapes[i]
        b_shape = b_shapes[i]
        print("Shape:", a_shape, "x", b_shape)
        a = torch.randn(a_shape, device="cuda", dtype=torch.float16)
        b = torch.randn(b_shape, device="cuda", dtype=a.dtype)

        time_with_torch_timer(torch_mm, (a, b), string_id="torch mm")
        time_with_torch_timer(torch_mm_relu, (a, b), string_id="torch mm + relu")
        time_with_torch_timer(inductor_mm, (a, b), string_id="inductor mm")


# Results obtained on the AWS AI cluster
# CPU: Intel(R) Xeon(R) Platinum 8275CL CPU @ 3.00GHz
# GPU: NVIDIA A100-SXM 40GB memory
"""
Shape: [2048, 768] x [768, 3072]
torch mm         mean: 0.0592 ms
torch mm + relu  mean: 0.0759 ms
inductor mm      mean: 0.0653 ms
Shape: [64, 1280] x [1280, 1000]
torch mm         mean: 0.0231 ms
torch mm + relu  mean: 0.0316 ms
inductor mm      mean: 0.0252 ms
Shape: [2048, 768] x [768, 768]
torch mm         mean: 0.0190 ms
torch mm + relu  mean: 0.0277 ms
inductor mm      mean: 0.0274 ms
Shape: [32, 2048] x [2048, 1000]
torch mm         mean: 0.0188 ms
torch mm + relu  mean: 0.0290 ms
inductor mm      mean: 0.0244 ms
Shape: [1, 39200] x [39200, 50]
torch mm         mean: 0.0134 ms
torch mm + relu  mean: 0.0234 ms
inductor mm      mean: 0.0290 ms
Shape: [128, 3072] x [3072, 1000]
torch mm         mean: 0.0181 ms
torch mm + relu  mean: 0.0322 ms
inductor mm      mean: 0.0319 ms
Shape: [16, 1280] x [1280, 1000]
torch mm         mean: 0.0188 ms
torch mm + relu  mean: 0.0289 ms
inductor mm      mean: 0.0255 ms
Shape: [10240, 512] x [512, 10240]
torch mm         mean: 0.4589 ms
torch mm + relu  mean: 0.7896 ms
inductor mm      mean: 0.5090 ms
Shape: [10240, 1024] x [1024, 10240]
torch mm         mean: 0.9152 ms
torch mm + relu  mean: 1.2124 ms
inductor mm      mean: 0.9462 ms
"""
