# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
import functools as _functools
from . import _C

# Top-level APIs. Please think carefully before adding something to the
# top-level namespace:
# - private helper functions should go into torch._functorch
# - very experimental things should go into functorch.experimental
# - compilation related things should go into functorch.compile

# functorch transforms
from torch._functorch.vmap import vmap
import torch.func as _func
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


def _alias_api(fn):
    # Copies over everything aside from __module__ and __doc__
    wrapper_updates = ('__name__', '__qualname__', '__annotations__')

    @_functools.wraps(fn, wrapper_updates)
    def inner(*args, **kwargs):
        return fn(*args, **kwargs)

    # NB: By the next release of PyTorch we just want this to say
    # e.g. "See torch.func.grad (url)". The URLs are not ready yet
    # (functorch's doc build is separate from PyTorch's, so we need to
    # manually write in the URL).
    # we're also not done moving everything over yet, so the easiest
    # intermediate solution is to keep the doc the same as before.
    inner.__doc__ = fn.__doc__.replace('torch.func', 'functorch')

    return inner

grad = _alias_api(_func.grad)
grad_and_value = _alias_api(_func.grad_and_value)
vjp = _alias_api(_func.vjp)
jvp = _alias_api(_func.jvp)
jacfwd = _alias_api(_func.jacfwd)
jacrev = _alias_api(_func.jacrev)
hessian = _alias_api(_func.hessian)
functionalize = _alias_api(_func.functionalize)
