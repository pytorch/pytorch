import torch


fn = vmap(
    jacfwd(jacrev(torch.matmul, argnums=1, has_aux=True), argnums=1, has_aux=True),
    in_dims=(None, 0),
)

fn_compiled = torch.compile(fn)
x = torch.randn(16, 16, requires_grad=True)
y = torch.randn(16, 16, requires_grad=True)

out = fn_compiled(x, y)
