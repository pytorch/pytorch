import torch
from . import _C

from ._src.vmap import vmap
from ._src.eager_transforms import grad, grad_with_value, vjp, jacrev
from ._src.make_functional import make_functional, make_functional_with_buffers
