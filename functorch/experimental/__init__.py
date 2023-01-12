# PyTorch forward-mode is not mature yet
from torch._functorch.eager_transforms import hessian, jacfwd, jvp
from torch._functorch.vmap import chunk_vmap
from .batch_norm_replacement import replace_all_batch_norm_modules_
from functorch import functionalize
from torch._functorch.modules_as_pytree import modules_as_pytrees, enable_modules_as_pytrees
