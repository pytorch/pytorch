# PyTorch forward-mode is not mature yet
try:
    from functorch import functionalize  # deprecation warnings
except ImportError:
    from torch._functorch.deprecated import functionalize  # Development environment

from torch._functorch.apis import chunk_vmap
from torch._functorch.batch_norm_replacement import replace_all_batch_norm_modules_
from torch._functorch.eager_transforms import hessian, jacfwd, jvp
