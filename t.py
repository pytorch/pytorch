import torch
from triton.testing import do_bench
import torch._inductor.config as inductor_config

inductor_config.coordinate_descent_tuning = True

torch.set_default_device("cuda")
torch.set_default_dtype(torch.bfloat16)

N = 32 * 1024
# V = 50257
V = 50304

def f(x):
    # return torch.softmax(x, dim=-1)
    return torch.log_softmax(x, dim=-1)

x = torch.randn(N, V)
opt_f = torch.compile(f)
expected = f(x)
actual = opt_f(x)

assert torch.allclose(expected, actual, atol=1e-2, rtol=1e-2)

eager_ms = do_bench(lambda: f(x))
opt_ms = do_bench(lambda: opt_f(x))
print(f"{eager_ms=}")
print(f"{opt_ms=}")
print("bye")
