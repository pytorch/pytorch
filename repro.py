import torch
from torch._dynamo.utils import same
import torch.nn.functional as F
from torch._inductor import metrics

torch._inductor.config.split_reductions = False

def f(x, w, b, eps):
    return F.layer_norm(x, x.shape[-1:], w, b, eps)
    
def fwd_bwd(f):
    x.grad = None
    w.grad = None
    b.grad = None
    out = f(x, w, b, eps)
    out.backward(dy)
    return x.grad, w.grad, b.grad

torch.manual_seed(1337)

# M, N = 1152 * 500, 384
M, N = 32768, 768
xdtype = torch.float
wbdtype = torch.float
x = torch.randn(M, N, dtype=xdtype, device="cuda", requires_grad=True)
w = torch.randn(N, dtype=wbdtype, device="cuda", requires_grad=True)
b = torch.randn(N, dtype=wbdtype, device="cuda", requires_grad=True)
dy = torch.randn_like(x)
eps = 1e-5

opt_f = torch.compile(f)

ref = fwd_bwd(f)
act = fwd_bwd(opt_f)

assert same(ref, act, tol=1e-2), f"ref:\n{ref[2][:64]}\nact:\n{act[2][:64]}"
print("PASS")
print(f"#kernel {metrics.generated_kernel_count}")
