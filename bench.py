from torch.fx.experimental.proxy_tensor import make_fx
from torch._inductor.compile_fx import compile_fx

import torch._inductor.config
torch._inductor.config.debug = True

def fn(x1, x2):
    y = torch.sin(x1 * x2)
    y = x1 * y
    return y,

x1 = torch.randn((20, 20), device="cuda", requires_grad=True)
x2 = torch.randn((20, 20), device="cuda", requires_grad=True)



traced = make_fx(fn)(x1, x2)
compiled = compile_fx(traced, [x1, x2])
o = compiled(x1, x2)[0]

print("Expected kernel launch here")

o.sum().backward()
