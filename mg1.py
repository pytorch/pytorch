import torch
from triton.testing import do_bench
from torch._inductor.utils import fresh_inductor_cache

@torch.compile(
    options={
        "max_autotune": True,
        "max_autotune_gemm_backends": "TRITON",
    },
    dynamic=False,
)
def inductor_matmul(m, a, b):
	# passing in m to have different compiled regions
	return (m, torch.mm(a, b))

# for m in [6152, 16, 32]:
for m in [16]:
	with fresh_inductor_cache():
		print(f"m={m}")
		n, k = 1024, 1280
		dynamic_a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
		static_a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
		dynamic_b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
		static_b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
		torch._dynamo.decorators.mark_dynamic(
			dynamic_a, # s0
			0,
			backend_specializations=[
				# hint, specialization
				(16, lambda x0: x0 == 16),
				(16, lambda x0: x0 % 16 == 0),
			]
		)
		torch._dynamo.decorators.mark_dynamic(
			dynamic_a, # also s0 due to duck typing
			1,
		)
		torch._dynamo.decorators.mark_dynamic(
			dynamic_b, # also s0 due to duck typing
			0,
		)
		torch._dynamo.decorators.mark_dynamic(
			dynamic_b, # also s0 due to duck typing
			1,
		)
		inductor_matmul(m, static_a, static_b)
		ms = do_bench(lambda: inductor_matmul(m, static_a, static_b))
		print("static ms taken:", ms)
		inductor_matmul(m, dynamic_a, dynamic_b)
		ms = do_bench(lambda: inductor_matmul(m, dynamic_a, dynamic_b))
		print("dynamic ms taken:", ms)
