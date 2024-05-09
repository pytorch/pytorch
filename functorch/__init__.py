# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch

from torch._functorch.apis import grad, grad_and_value, vmap

from torch._functorch.eager_transforms import (
    functionalize,
    hessian,
    jacfwd,
    jacrev,
    jvp,
    vjp,
)

# utilities. Maybe these should go in their own namespace in the future?
from torch._functorch.make_functional import (
    combine_state_for_ensemble,
    FunctionalModule,
    FunctionalModuleWithBuffers,
    make_functional,
    make_functional_with_buffers,
)

# Top-level APIs. Please think carefully before adding something to the
# top-level namespace:
# - private helper functions should go into torch._functorch
# - very experimental things should go into functorch.experimental
# - compilation related things should go into functorch.compile

# Was never documented
from torch._functorch.python_key import make_fx

__version__ = torch.__version__
