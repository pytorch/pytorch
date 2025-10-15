# mypy: ignore-errors

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This module contains pre-dispatch wrappers for functorch operations
that enable proper tracing in PT2 non-strict export/compile fx graph.
"""

import torch
from torch._C._functorch import (
    _add_batch_dim as _add_batch_dim_impl,
    _remove_batch_dim as _remove_batch_dim_impl,
    _vmap_decrement_nesting as _vmap_decrement_nesting_impl,
    _vmap_increment_nesting as _vmap_increment_nesting_impl,
)


def _add_batch_dim(self, batch_dim, level):
    """
    Thin wrapper around torch._C._add_batch_dim that is used to proxy in
    PT2 export/compile fx graph
    """
    from torch._export.utils import _maybe_find_pre_dispatch_tf_mode_for_export

    mode = _maybe_find_pre_dispatch_tf_mode_for_export()

    if mode:
        return torch.overrides.handle_torch_function(
            _add_batch_dim, (self,), self, batch_dim, level
        )

    res = _add_batch_dim_impl(self, batch_dim, level)
    return res


def _remove_batch_dim(self, level, batch_size, out_dim):
    """
    Thin wrapper around torch._C._remove_batch_dim that is used to proxy in
    PT2 export/compile fx graph
    """
    from torch._export.utils import _maybe_find_pre_dispatch_tf_mode_for_export

    mode = _maybe_find_pre_dispatch_tf_mode_for_export()

    if mode:
        return torch.overrides.handle_torch_function(
            _remove_batch_dim, (self,), self, level, batch_size, out_dim
        )

    res = _remove_batch_dim_impl(self, level, batch_size, out_dim)
    return res


def _vmap_increment_nesting(batch_size, randomness):
    """
    Thin wrapper around torch._C._vmap_increment_nesting that is used
    to proxy in export/compile graph
    """
    from torch._export.utils import _maybe_find_pre_dispatch_tf_mode_for_export

    mode = _maybe_find_pre_dispatch_tf_mode_for_export()

    if mode:
        return torch.overrides.handle_torch_function(
            _vmap_increment_nesting, (batch_size,), batch_size, randomness
        )
    res = _vmap_increment_nesting_impl(batch_size, randomness)
    return res


def _vmap_decrement_nesting():
    """
    Thin wrapper around torch._C._vmap_increment_nesting that is used
    to proxy in export/compile graph
    """
    from torch._export.utils import _maybe_find_pre_dispatch_tf_mode_for_export

    mode = _maybe_find_pre_dispatch_tf_mode_for_export()

    if mode:
        return torch.overrides.handle_torch_function(
            _vmap_decrement_nesting,
            (),
        )
    return _vmap_decrement_nesting_impl()


# Global variables for lazy_load_decompositions
DECOMPOSITIONS_LOADED = False
DECOMPOSITIONS_LOCK = None  # Will be initialized when needed
VMAP_DECOMPOSITIONS_LIB = None


def lazy_load_decompositions():
    """
    Lazy loading of vmap decompositions with pre-dispatch support.
    """
    from torch._export.utils import _maybe_find_pre_dispatch_tf_mode_for_export

    mode = _maybe_find_pre_dispatch_tf_mode_for_export()

    if mode:
        return torch.overrides.handle_torch_function(lazy_load_decompositions, ())

    global DECOMPOSITIONS_LOADED, DECOMPOSITIONS_LOCK, VMAP_DECOMPOSITIONS_LIB

    if DECOMPOSITIONS_LOADED:
        return

    # Initialize lock if needed
    if DECOMPOSITIONS_LOCK is None:
        import threading

        DECOMPOSITIONS_LOCK = threading.Lock()

    with DECOMPOSITIONS_LOCK:
        if DECOMPOSITIONS_LOADED:
            return

        import os

        if not (os.environ.get("PYTORCH_JIT", "1") == "1" and __debug__):
            DECOMPOSITIONS_LOADED = True
            return

        # use an alternate way to register an operator into the decomposition table
        # _register_jit_decomposition doesn't work for some operators, e.g. addr,
        #  because the Tensor types generated cannot be unioned by torchscript
        # decomp should be type OpOverload
        VMAP_DECOMPOSITIONS_LIB = torch.library.Library(
            "aten", "IMPL", "FuncTorchBatched"
        )

        from torch._decomp import decomposition_table

        def _register_python_decomposition_vmap(decomp):
            if decomp in decomposition_table:
                VMAP_DECOMPOSITIONS_LIB.impl(decomp, decomposition_table[decomp])
            else:
                raise RuntimeError(f"could not find decomposition for {decomp}")

        _register_python_decomposition_vmap(torch.ops.aten.mse_loss_backward.default)
        _register_python_decomposition_vmap(
            torch.ops.aten.smooth_l1_loss_backward.default
        )
        _register_python_decomposition_vmap(torch.ops.aten.huber_loss_backward.default)
        _register_python_decomposition_vmap(torch.ops.aten.nll_loss_forward.default)
        _register_python_decomposition_vmap(torch.ops.aten.nll_loss2d_forward.default)
        _register_python_decomposition_vmap(torch.ops.aten.nll_loss_backward.default)
        _register_python_decomposition_vmap(torch.ops.aten.nll_loss2d_backward.default)
        _register_python_decomposition_vmap(torch.ops.aten.addr.default)

        DECOMPOSITIONS_LOADED = True
