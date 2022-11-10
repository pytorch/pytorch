import contextlib
import warnings
from typing import Generator, List

import torch
import torch.nn as nn
from torch.distributed.fsdp._common_utils import (
    _FSDPState,
    _has_fsdp_params,
    _module_handles,
    HandleTrainingState,
)
from torch.distributed.fsdp._runtime_utils import (
    _clear_grads_if_needed,
    _reshard,
    _reshard_grads,
    _unshard,
    _unshard_grads,
)
from ._utils import p_assert
from .flat_param import FlatParamHandle

FLAT_PARAM = "_flat_param"


@torch.no_grad()
def _writeback_to_local_shard(
    handles: List[FlatParamHandle],
    writeback_grad: bool,
):
    """
    For each handle, writes back the this rank's shard of the unsharded
    flattened parameter to the sharded flattened parameter. If
    ``writeback_grad=True``, then writes back to the sharded gradient as
    well.

    Precondition: Each handle's ``FlatParameter`` 's data points to the
    padded unsharded flattened parameter.
    """
    for handle in handles:
        # For `NO_SHARD`, `_local_shard` is the unsharded flattened
        # parameter and `grad` is the unsharded gradient, so there is no
        # need to writeback for either
        if not handle.uses_sharded_strategy:
            continue
        assert (
            handle.flat_param.ndim == 1
        ), f"Expects `flat_param` to be flattened but got {handle.flat_param.shape}"

        # Get the unpadded shard instead of the padded shard to persist
        # user changes to the padding (though FSDP does not explicitly
        # support this)
        param_shard, _ = FlatParamHandle._get_unpadded_shard(
            handle.flat_param,
            handle.rank,
            handle.world_size,
        )
        handle.flat_param._local_shard[: param_shard.numel()].copy_(param_shard)  # type: ignore[attr-defined]
        if writeback_grad:
            existing_grad = handle.sharded_grad
            if existing_grad is not None:
                assert handle.flat_param.grad is not None
                grad_shard, _ = FlatParamHandle._get_unpadded_shard(
                    handle.flat_param.grad,
                    handle.rank,
                    handle.world_size,
                )
                existing_grad[: grad_shard.numel()].copy_(grad_shard)


def _deregister_flat_param(state: _FSDPState, module: nn.Module) -> None:
    """
    De-registers the flattened parameter from the wrapped module, hiding it
    from ``nn.Module`` methods.

    We do not use ``del`` because we want ``FLAT_PARAM`` to always be an
    attribute but dynamically change whether it is visible to ``nn.Module``
    methods.
    """
    if _has_fsdp_params(state, module):
        # TODO: figure out the case for the composable APIs.
        module.module._parameters.pop(FLAT_PARAM, None)


def _register_flat_param(state: _FSDPState, module: nn.Module) -> None:
    """
    Registers the flattened parameter to the wrapped module, making it
    visible to ``nn.Module`` methods.

    We do not use :meth:`nn.Module.register_parameter` because we want
    ``FLAT_PARAM`` to always be an attribute but dynamically change whether
    it is visible to ``nn.Module`` methods.
    """
    handles = _module_handles(state, module)
    if _has_fsdp_params(state, module):
        # TODO: figure out the case for the composable APIs.
        module.module._parameters[FLAT_PARAM] = handles[0].flat_param


@contextlib.contextmanager
def _unflatten_as_params(state: _FSDPState, module: nn.Module) -> Generator:
    """
    Assumes that the flattened parameter is unsharded. When in the context,
    de-registers the flattened parameter and unflattens the original
    parameters as ``nn.Parameter`` views into the flattened parameter.
    After the context, re-registers the flattened parameter and restores
    the original parameters as ``Tensor`` views into the flattened
    parameter.
    """
    handles = _module_handles(state, module)
    if not handles:
        yield
    else:
        _deregister_flat_param(state, module)
        try:
            with handles[0].unflatten_as_params():
                yield
        finally:
            if not handles[0]._use_orig_params:
                _register_flat_param(state, module)


@contextlib.contextmanager
def _unshard_params(
    module: nn.Module,
    state: _FSDPState,
    writeback: bool = True,
    rank0_only: bool = False,
    offload_to_cpu: bool = False,
    with_grads: bool = False,
):
    if with_grads and (offload_to_cpu or not state._use_orig_params):
        raise NotImplementedError(
            f"with_grads={with_grads} "
            f"use_orig_params={state._use_orig_params} "
            f"offload_to_cpu={offload_to_cpu} "
            f"is not supported yet"
        )
    if writeback and rank0_only:
        raise ValueError(
            "writeback=True and rank0_only=True is not supported, as model "
            "parameter shapes will be different across ranks, and writing "
            "to them can lead to inconsistencies across ranks when the "
            "context is exited."
        )
    if offload_to_cpu and not rank0_only:
        warnings.warn(
            "offload_to_cpu and rank0_only=False will result in "
            "full parameters being redundantly copied to CPU memory for "
            "GPUs that reside on the same machine, which may incur the risk of "
            "CPU OOM. It is recommended to use ``offload_to_cpu`` with "
            "rank0_only=True."
        )

    torch.cuda.synchronize()
    # If handles are shared by other module(s), the handle may be already unsharded.
    handles = [
        handle
        for handle in _module_handles(state, module)
        if handle._training_state != HandleTrainingState.SUMMON_FULL_PARAMS
    ]
    if not handles:
        yield
        return

    for handle in handles:
        if handle._training_state != HandleTrainingState.IDLE:
            raise ValueError(f"Current handle state is {handle._training_state}")

    for handle in handles:
        handle._training_state = HandleTrainingState.SUMMON_FULL_PARAMS

    _clear_grads_if_needed(handles)
    free_unsharded_flat_params = [handle.needs_unshard() for handle in handles]
    # No need to call `wait_stream()` since we unshard in the computation
    # stream directly
    computation_stream = torch.cuda.current_stream()
    _unshard(state, handles, computation_stream, computation_stream)
    if with_grads:
        _unshard_grads(handles)

    if rank0_only and state.rank != 0:
        # Free the unsharded flattened parameter early
        _reshard(state, handles, free_unsharded_flat_params)
        if with_grads:
            _reshard_grads(handles)
        try:
            yield
        finally:
            for handle in handles:
                handle._training_state = HandleTrainingState.IDLE
    else:
        # Unflatten the unsharded flattened parameters
        with contextlib.ExitStack() as stack:
            # Invariant: rank == 0 or !rank0_only
            for handle in handles:
                if offload_to_cpu and handle.uses_sharded_strategy:
                    stack.enter_context(handle.to_cpu())
                    # TODO (awgu): Since PyTorch enforces that a parameter
                    # and its gradients need to match metadata (e.g.
                    # device), we must move gradients to CPU *after* we
                    # move parameters.
            # TODO (awgu): This FPW call assumes 1 `FlatParameter`
            if not state._use_orig_params:
                stack.enter_context(_unflatten_as_params(state, module))
            try:
                yield
            finally:
                stack.close()
                if writeback:
                    _writeback_to_local_shard(handles, with_grads)
                _reshard(state, handles, free_unsharded_flat_params)
                if with_grads:
                    _reshard_grads(handles)
                for handle in handles:
                    handle._training_state = HandleTrainingState.IDLE


def _deregister_orig_params(state: _FSDPState, module: nn.Module) -> None:
    """
    Deregisters the original parameters; registers the ``FlatParameter``.
    """
    handles = _module_handles(state, module)
    p_assert(
        len(handles) <= 1,
        "Expects <=1 handle per FSDP instance; needs to be refactored "
        "for >1 handle (e.g. non-recursive wrapping)",
    )
    if not handles:
        return
    handle = handles[0]
    p_assert(
        handle._use_orig_params,
        f"Inconsistent `_use_orig_params` -- FSDP: {state._use_orig_params} "
        f"handle: {handle._use_orig_params}",
    )
    handle._deregister_orig_params()
    _register_flat_param(state, module)


def _register_orig_params(state: _FSDPState, module: nn.Module) -> None:
    """
    Deregisters the ``FlatParameter``; registers the original parameters.
    """
    handles = _module_handles(state, module)
    if not handles:
        return
    handle = handles[0]
    _deregister_flat_param(state, module)
    if handle.is_sharded(handle.flat_param):
        handle._use_sharded_views()
        handle._use_sharded_grad_views()
    else:
        handle._use_unsharded_views(as_params=True)
