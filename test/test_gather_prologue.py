import torch
import torch._inductor
from torch._inductor import config

config.fx_graph_cache = False
config.max_autotune_gemm_backends = "TRITON"
config.use_mixed_mm = False
config.mixed_mm_choice = "default"
config.prologue_fusion = True
config.benchmark_epilogue_fusion = False

from torch._inductor.utils import fresh_inductor_cache

def fn(x, y, index=None):
    return (x)[index] @ y

# assertion failure with torch.float16, works with torch.float
dtype = torch.float16
size = 2048
fn_c = torch.compile(mode="max-autotune-no-cudagraphs")(fn)
x = torch.rand([size, size], device="cuda", dtype=dtype)
y = torch.rand([size, size], device="cuda", dtype=dtype)
index = torch.randperm(size, device="cuda")

torch.testing.assert_allclose(fn(x, y, index), fn_c(x, y, index))
