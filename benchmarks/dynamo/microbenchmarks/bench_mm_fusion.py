# flake8: noqa

import triton
from prettytable import PrettyTable

import torch
import torch._dynamo
import torch._inductor.config
from torch._inductor.runtime.benchmarking import benchmarker


# torch._inductor.config.debug = True
torch._inductor.config.triton.dense_indexing = True
torch.manual_seed(0)


# The flag below controls whether to allow TF32 on matmul.
torch.backends.cuda.matmul.allow_tf32 = True


class Func(object):
    # mm
    @torch._dynamo.optimize("inductor")
    def mm(a, b, bias):
        y = torch.mm(a, b)
        return y

    # mm+bias
    @torch._dynamo.optimize("inductor")
    def mm_add(a, b, bias):
        y = torch.mm(a, b)
        return y + bias

    # relu(mm)
    @torch._dynamo.optimize("inductor")
    def mm_relu(a, b, bias):
        y = torch.mm(a, b)
        return torch.relu(y)

    # relu(mm+bias)
    @torch._dynamo.optimize("inductor")
    def mm_add_relu(a, b, bias):
        y = torch.mm(a, b)
        y += bias
        return torch.relu(y)


def bench(shape, layer_id, p, fusion_types=[""]):
    dtype = torch.float16
    M, K = shape[0]
    _, N = shape[1]
    torch.manual_seed(0)
    # allocate inputs
    a = torch.randn(shape[0], device="cuda", dtype=dtype)
    b = torch.randn(shape[1], device="cuda", dtype=dtype)

    def tflops(ms):
        return M * K * N / ms * 1e-9

    row = [layer_id]
    for fusion_type in fusion_types:
        if fusion_type == "":
            fn_mm = getattr(Func, "mm")
        else:
            fn_mm = getattr(Func, f"mm_{fusion_type}")

        if "add" in fusion_type:
            bias = torch.randn((M, N), dtype=dtype, device="cuda")
        else:
            bias = None

        args = (a, b, bias)

        def fn():
            return fn_mm(*args)

        torch._inductor.config.triton.mm = "aten"
        torch_mm_ms, _, _ = benchmarker.benchmark_gpu(fn)
        torch._inductor.config.triton.mm = "triton"
        # reset to force code gen new python code
        torch._dynamo.reset()
        torch._inductor.metrics.reset()
        triton_mm_ms, _, _ = benchmarker.benchmark_gpu(fn)
        assert (
            torch._inductor.metrics.generated_kernel_count == 1
        ), "codegen #kernel != 1"
        row.extend([tflops(torch_mm_ms), tflops(triton_mm_ms)])

    p.add_row(row)


fusion_types = ["", "add", "relu", "add_relu"]
shapes = [
    # alexnet
    ([128, 9216], [9216, 4096]),
    ([128, 4096], [4096, 4096]),
    ([128, 4096], [4096, 1000]),
    # BERT
    ([2048, 768], [768, 768]),
    ([2048, 768], [768, 3072]),
    ([2048, 3072], [3072, 768]),
    # hf_GPT2
    ([1024, 768], [768, 768]),
    ([1024, 768], [768, 3072]),
    ([1024, 3072], [3072, 768]),
    ([1024, 768], [768, 2304]),
]
p = PrettyTable()
field_names = ["layer"]
for fusion_type in fusion_types:
    if fusion_type == "":
        field_names.append("torch mm")
        field_names.append("triton mm")
    else:
        field_names.append(f"torch mm+{fusion_type}")
        field_names.append(f"triton mm+{fusion_type}")

p.field_names = field_names
p.float_format = ".3"
for id, shape in enumerate(shapes):
    bench(shape, id, p, fusion_types)

print(p)
