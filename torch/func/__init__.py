from torch._functorch.eager_transforms import (
    grad,
    grad_and_value,
    vjp,
    jvp,
    jacrev,
    jacfwd,
    hessian,
    functionalize,
)
from torch._functorch.vmap import vmap
from torch.fx.experimental.proxy_tensor import make_fx
