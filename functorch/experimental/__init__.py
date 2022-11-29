# PyTorch forward-mode is not mature yet
from .._src.eager_transforms import hessian, jacfwd, jvp
from .._src.vmap import chunk_vmap
from .batch_norm_replacement import replace_all_batch_norm_modules_
from functorch import functionalize
