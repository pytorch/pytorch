# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
from . import _C

# Top-level APIs. Please think carefully before adding something to the
# top-level namespace:
# - private helper functions should go into torch._functorch
# - very experimental things should go into functorch.experimental
# - compilation related things should go into functorch.compile

# functorch transforms
from torch._functorch.vmap import vmap
from torch._functorch.eager_transforms import (
    grad, grad_and_value, vjp, jacrev, jvp, jacfwd, hessian, functionalize
)
from torch._functorch.python_key import make_fx

# utilities. Maybe these should go in their own namespace in the future?
from torch._functorch.make_functional import (
    make_functional_with_buffers,
    make_functional,
    combine_state_for_ensemble,
    FunctionalModule,
    FunctionalModuleWithBuffers,
)

__version__ = torch.__version__
