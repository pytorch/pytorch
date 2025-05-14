from triton.testing import do_bench

import torch
from torch._inductor.utils import fresh_inductor_cache


@torch.compile(
    options={
        "max_autotune": True,
        "max_autotune_gemm_backends": "TRITON",
    },
    dynamic=False,
)
def inductor_matmul(m, a, b):
    torch._check(a.shape[0] == b.shape[1])
    # passing in m to have different compiled regions
    return (m, torch.mm(a, b))


# for m in [2, 4, 8, 16]:
for m in [16]:
    with fresh_inductor_cache():
        print(f"m={m}")
        k = 1280
        dynamic_a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        static_a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        dynamic_b = torch.randn(k, m, device="cuda", dtype=torch.bfloat16)
        static_b = torch.randn(k, m, device="cuda", dtype=torch.bfloat16)
        torch._dynamo.decorators.mark_dynamic(
            dynamic_a,  # s0
            0,
            specialize_on=[lambda x: x == 8, lambda x: x == 16],
        )
        torch._dynamo.decorators.mark_dynamic(
            dynamic_b,
            1,
        )
        inductor_matmul(m, static_a, static_b)
        ms = do_bench(lambda: inductor_matmul(m, static_a, static_b))
        print("static ms taken:", ms)
        inductor_matmul(m, dynamic_a, dynamic_b)
        ms = do_bench(lambda: inductor_matmul(m, dynamic_a, dynamic_b))
        print("dynamic ms taken:", ms)
