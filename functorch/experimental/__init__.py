from .batch_norm_replacement import replace_all_batch_norm_modules_
# PyTorch forward-mode is not mature yet
from .._src.eager_transforms import jvp, jacfwd, hessian
from .._src.vmap import chunk_vmap
from functorch import functionalize
