"""Lightweight memory component tracker for CUDA memory attribution.

This module provides the implementation for memory component tracking that
auto-detects models and optimizers via global hooks. The public API is
:mod:`torch.accelerator.memory.component_tracking`.

Internal module — use ``from torch.accelerator.memory import component_tracking``
for the public interface.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.cuda.memory import (
    _disable_module_tracking,
    _enable_module_tracking,
    _set_component_type,
    _set_memory_metadata,
    _tag_block,
    _tag_module_block,
    ComponentType,
)


class ComponentTracker:
    """Manages the lifecycle of memory attribution hooks.

    Tracks global hooks, auto-detected roots, and provides teardown.
    """

    def __init__(self) -> None:
        self._hook_handles: list[Any] = []
        self._enabled = True
        self._depth = 0
        self._current_root: nn.Module | None = None
        self._tagged_modules: set[int] = set()
        self._tagged_params: set[int] = set()
        self._grad_hooked_params: set[int] = set()

    def _add_handle(self, handle: Any) -> None:
        self._hook_handles.append(handle)

    def disable(self) -> None:
        """Remove all hooks and reset tracking state."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        _set_component_type(ComponentType.OTHER)
        _set_memory_metadata("")
        _disable_module_tracking()
        self._enabled = False
        self._depth = 0
        self._current_root = None
        self._tagged_modules.clear()
        self._tagged_params.clear()
        self._grad_hooked_params.clear()

    @property
    def enabled(self) -> bool:
        return self._enabled


def _enable_auto() -> ComponentTracker:
    """Enable component tracking with auto-detection of models and optimizers.

    Installs global hooks that:
    - Auto-detect root models on first forward (builds FQN cache)
    - Tag allocations as ACTIVATION during forward, TEMP during backward
    - Tag gradient storage via post-accumulate-grad hooks (lazy)
    - Tag optimizer allocations via class-level optimizer step hooks
    - Tag parameters/buffers lazily on first module forward
    - Enable per-module tracking and collective buffer detection
    """
    tracker = ComponentTracker()

    # --- (a) Global forward hooks: auto-detect root, tag ACTIVATION ---

    def _forward_pre_hook(module: nn.Module, args: Any) -> None:
        if tracker._depth == 0:
            if not hasattr(module, "_fqn_cache"):
                module._fqn_cache = {}  # type: ignore[attr-defined]
                for name, mod in module.named_modules():
                    module._fqn_cache[id(mod)] = name  # type: ignore[attr-defined]
            tracker._current_root = module

        tracker._depth += 1

        fqn = ""
        if tracker._current_root is not None:
            cache = getattr(tracker._current_root, "_fqn_cache", {})
            fqn = cache.get(id(module), "")

        # All memory allocated during a module's forward pass is an activation
        # (intermediate tensors, attention scores, etc.)
        _set_component_type(ComponentType.ACTIVATION)
        _set_memory_metadata(fqn)

        if id(module) in tracker._tagged_modules:
            return
        tracker._tagged_modules.add(id(module))

        # Tag this module's parameters and buffers on first visit
        for p in module.parameters(recurse=False):
            if p.data.data_ptr() != 0:
                if id(p) not in tracker._tagged_params:
                    _tag_block(p.data.data_ptr(), ComponentType.PARAMETER)
                    tracker._tagged_params.add(id(p))
                if fqn:
                    _tag_module_block(
                        p.data.data_ptr(), fqn, ComponentType.PARAMETER
                    )
        for b in module.buffers(recurse=False):
            if b.data.data_ptr() != 0:
                _tag_block(b.data.data_ptr(), ComponentType.BUFFER)
                if fqn:
                    _tag_module_block(
                        b.data.data_ptr(), fqn, ComponentType.BUFFER
                    )

        # Register gradient hooks — after backward accumulates gradients,
        # re-tag the grad tensor from OTHER/TEMP to GRADIENT
        for p in module.parameters(recurse=False):
            if p.requires_grad and id(p) not in tracker._grad_hooked_params:
                tracker._grad_hooked_params.add(id(p))

                def _make_grad_hook(param: torch.Tensor) -> Any:
                    def _grad_hook(grad: torch.Tensor) -> None:
                        if (
                            param.grad is not None
                            and param.grad.data.data_ptr() != 0
                        ):
                            _tag_block(
                                param.grad.data.data_ptr(),
                                ComponentType.GRADIENT,
                            )

                    return _grad_hook

                h = p.register_post_accumulate_grad_hook(_make_grad_hook(p))
                tracker._add_handle(h)

    def _forward_hook(module: nn.Module, args: Any, output: Any) -> None:
        tracker._depth -= 1
        if tracker._depth == 0:
            # Forward pass complete — reset to OTHER so allocations outside
            # model forward (e.g. loss computation) aren't tagged as ACTIVATION
            _set_component_type(ComponentType.OTHER)
            _set_memory_metadata("")

    h_fwd_pre = nn.modules.module.register_module_forward_pre_hook(_forward_pre_hook)
    h_fwd_post = nn.modules.module.register_module_forward_hook(_forward_hook)
    tracker._add_handle(h_fwd_pre)
    tracker._add_handle(h_fwd_post)

    # --- (b) Global backward hooks: tag allocations as TEMP ---
    # During backward, intermediate tensors (e.g. recomputed activations in
    # checkpointing, gradient intermediates) are tagged as TEMP.

    def _backward_pre_hook(module: nn.Module, grad_output: Any) -> None:
        fqn = ""
        if tracker._current_root is not None:
            cache = getattr(tracker._current_root, "_fqn_cache", {})
            fqn = cache.get(id(module), "")
        # Allocations during backward are temporary intermediates
        _set_component_type(ComponentType.TEMP)
        _set_memory_metadata(fqn)

    def _backward_post_hook(
        module: nn.Module, grad_input: Any, grad_output: Any
    ) -> None:
        # Module's backward complete — reset so allocations between modules
        # (e.g. autograd engine intermediates) fall back to OTHER
        _set_component_type(ComponentType.OTHER)
        _set_memory_metadata("")

    h_bwd_pre = nn.modules.module.register_module_full_backward_pre_hook(
        _backward_pre_hook
    )
    h_bwd_post = nn.modules.module.register_module_full_backward_hook(
        _backward_post_hook
    )
    tracker._add_handle(h_bwd_pre)
    tracker._add_handle(h_bwd_post)

    # --- (c) Class-level optimizer hooks: tag OPTIMIZER_STATE ---
    # Optimizer.step() allocates momentum/variance buffers on first call
    # and updates them on subsequent calls. Tag all allocations during step
    # as OPTIMIZER_STATE.

    def _optimizer_pre_hook(
        opt: torch.optim.Optimizer, args: Any, kwargs: Any
    ) -> None:
        _set_component_type(ComponentType.OPTIMIZER_STATE)
        _set_memory_metadata("optimizer")

    def _optimizer_post_hook(
        opt: torch.optim.Optimizer, args: Any, kwargs: Any
    ) -> None:
        # Step complete — reset so allocations after optimizer.step()
        # (e.g. zero_grad, next iteration) aren't tagged as optimizer state
        _set_component_type(ComponentType.OTHER)
        _set_memory_metadata("")

    from torch.optim.optimizer import (
        register_optimizer_step_post_hook,
        register_optimizer_step_pre_hook,
    )

    h_opt_pre = register_optimizer_step_pre_hook(_optimizer_pre_hook)
    h_opt_post = register_optimizer_step_post_hook(_optimizer_post_hook)
    tracker._add_handle(h_opt_pre)
    tracker._add_handle(h_opt_post)

    # --- (d) Enable per-module tracking via AllocatorTraceTracker ---
    _enable_module_tracking()

    return tracker
