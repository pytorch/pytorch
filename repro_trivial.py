import torch
from torch._dynamo.utils import same
import torch.nn.functional as F
from torch._inductor import metrics

torch._inductor.config.split_reductions = False

def f(x):
    return x.sum(dim=1), x.sum(dim=0)

M, N = 32768, 768
dtype = torch.float
x = torch.randn(M, N, dtype=dtype, device="cuda")

opt_f = torch.compile(f)

ref_outs = f(x)
act_outs = opt_f(x)

assert same(ref_outs, act_outs, tol=1e-3), f"ref:\n{ref_outs}\nact:\n{act_outs}"
print(f"#kernel {metrics.generated_kernel_count}")
print("PASS")
