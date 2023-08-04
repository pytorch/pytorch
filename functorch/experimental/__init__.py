# PyTorch forward-mode is not mature yet
from torch._functorch.batch_norm_replacement import replace_all_batch_norm_modules_
from torch._functorch.eager_transforms import hessian, jacfwd, jvp
from torch._functorch.vmap import chunk_vmap

from functorch import functionalize
