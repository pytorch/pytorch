from torch._functorch.eager_transforms import (
    vjp,
    jvp,
    jacrev,
    jacfwd,
    hessian,
    functionalize,
    linearize
)
from torch._functorch.apis import grad, grad_and_value
from torch._functorch.functional_call import functional_call, stack_module_state
from torch._functorch.batch_norm_replacement import replace_all_batch_norm_modules_
from torch._functorch.apis import vmap
