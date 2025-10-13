import torch
from torch._dynamo.utils import same
import torch.nn.functional as F

torch._inductor.config.split_reductions = False

def ref_fwd(x, w, eps):
    # return F.rms_norm(x, x.shape[-1:], w, eps)
    orig_dtype = x.dtype

    x = x.float()
    rsqrt = torch.rsqrt((x * x).sum(dim=-1) / x.shape[-1] + eps)
    y = (x * rsqrt[:, None] * w).to(dtype=orig_dtype)
    return y

def fwd_bwd(f):
    x.grad = None
    w.grad = None
    out = f(x, w, eps)
    out.backward(dy)
    return x.grad, w.grad

# M, N = 1152 * 500, 384
M, N = 32768, 768
x = torch.randn(M, N, dtype=torch.bfloat16, device="cuda", requires_grad=True)
w = torch.randn(N, dtype=torch.float, device="cuda", requires_grad=True)
dy = torch.randn_like(x)
eps = 1e-5

opt_fwd = torch.compile(ref_fwd)

ref_outs = fwd_bwd(ref_fwd)
act_outs = fwd_bwd(opt_fwd)

assert same(ref_outs, act_outs, tol=1e-2), f"ref:\n{ref_outs}\nact:\n{act_outs}"
print("PASS")
