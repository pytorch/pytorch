# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
from torch._functorch.deprecated import (
    combine_state_for_ensemble,
    functionalize,
    grad,
    grad_and_value,
    hessian,
    jacfwd,
    jacrev,
    jvp,
    make_functional,
    make_functional_with_buffers,
    vjp,
    vmap,
)

# utilities. Maybe these should go in their own namespace in the future?
from torch._functorch.make_functional import (
    FunctionalModule,
    FunctionalModuleWithBuffers,
)

# Was never documented
from torch._functorch.python_key import make_fx


# Top-level APIs. Please think carefully before adding something to the
# top-level namespace:
# - private helper functions should go into torch._functorch
# - very experimental things should go into functorch.experimental
# - compilation related things should go into functorch.compile


__version__ = torch.__version__
