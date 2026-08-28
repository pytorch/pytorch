# mypy: allow-untyped-defs
"""
State dict utilities for torch.export.

This module provides utilities for restoring state dicts to traced modules,
ensuring that FQNs (Fully Qualified Names) match the original module structure.
"""

from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.fx


def _get_underlying_module(
    module_or_method: torch.nn.Module | Callable[..., Any],
) -> torch.nn.Module:
    """Extract the underlying nn.Module from either a module or a bound method.

    Args:
        module_or_method: Either an nn.Module or a bound method of an nn.Module.

    Returns:
        The underlying nn.Module.

    Raises:
        TypeError: If module_or_method is neither an nn.Module nor a bound method.
    """
    if isinstance(module_or_method, torch.nn.Module):
        return module_or_method
    # Handle bound methods (e.g., module.method)
    if (
        mod_self := getattr(module_or_method, "__self__", None)
    ) is not None and isinstance(mod_self, torch.nn.Module):
        return mod_self
    raise TypeError(
        f"Expected nn.Module or bound method of nn.Module, got {type(module_or_method)}"
    )


def _clear_traced_params_buffers(
    traced_module: torch.fx.GraphModule, const_keys: Sequence[str]
) -> None:
    """Remove all parameters and buffers from traced module before restoring.

    For constants (parameters/buffers that don't need FQN mapping), this function
    removes them from the _buffers dict and re-assigns them as direct attributes.
    This ensures constants don't show up as buffers in the state dict.

    Args:
        traced_module: The traced GraphModule to clean up.
        const_keys: List of keys that represent constants to be cleared.
    """
    for key in const_keys:
        if key not in traced_module._buffers:
            raise AssertionError(f"Key {key} not found in traced_module._buffers")
        # We don't want constants to show up as a buffer in the state dict.
        # Instead they should just be a direct attribute.
        buffer = traced_module._buffers[key]
        del traced_module._buffers[key]
        # Note: setattr will register the value per nn.Module rules:
        # - If it's a Tensor, it'll be re-registered as a buffer (ends up back in _buffers).
        # - Otherwise, it becomes a plain attribute (not part of state_dict).
        setattr(traced_module, key, buffer)


def _restore_state_dict(
    original_module: torch.nn.Module | Callable[..., Any],
    traced_module: torch.fx.GraphModule,
) -> None:
    """
    Restores the state dict of the traced module to match the original module exactly.

    This function ensures that:
    1. Parameters and buffers in the traced module use the same FQNs (Fully Qualified Names)
       as the original module.
    2. The ordering of parameters/buffers matches the original module.
    3. Graph nodes referencing the old names are updated to use the correct FQNs.

    This is useful after using functional tracing APIs (like dynamo_graph_capture_for_export)
    that may flatten parameter/buffer names.

    Args:
        original_module: The original nn.Module (or a bound method of one) that was traced.
        traced_module: The traced fx.GraphModule whose state dict needs to be restored.

    Example::

        import torch
        from torch._dynamo.functional_export import _dynamo_graph_capture_for_export
        from torch.export import _restore_state_dict


        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(10, 10)

            def forward(self, x):
                return self.layer(x)


        model = Model()
        gm = _dynamo_graph_capture_for_export(model)(torch.randn(1, 10))

        # Before: gm may have flattened names like "p_layer_weight"
        # After: gm will have proper FQNs like "layer.weight"
        _restore_state_dict(model, gm)
    """
    # Extract the underlying module if a bound method was passed
    module = _get_underlying_module(original_module)

    # Build ID-based lookups for traced module params/buffers
    # Collect all data first to avoid modifying during iteration
    traced_params: dict[int, tuple[str, torch.nn.Parameter]] = {}
    for name, param in traced_module.named_parameters(remove_duplicate=False):
        traced_params[id(param)] = (name, param)

    traced_buffers: dict[int, tuple[str, torch.Tensor]] = {}
    for name, buffer in traced_module.named_buffers(remove_duplicate=False):
        traced_buffers[id(buffer)] = (name, buffer)

    # Collect original module's parameters and buffers upfront to avoid
    # issues with shared tensor objects during iteration
    orig_params_list: list[tuple[str, torch.nn.Parameter]] = list(
        module.named_parameters(remove_duplicate=False)
    )
    orig_buffers_list: list[tuple[str, torch.Tensor]] = list(
        module.named_buffers(remove_duplicate=False)
    )

    # Build mapping from old names to new names for graph node updates
    name_mapping: dict[str, str] = {}

    # Track which traced names we've processed
    processed_traced_names: set[str] = set()

    # Restore parameters in the order they appear in original module
    for orig_name, orig_param in orig_params_list:
        if id(orig_param) in traced_params:
            # This param exists in traced module - restore it with original FQN
            traced_name, traced_param = traced_params[id(orig_param)]
            processed_traced_names.add(traced_name)
            if traced_name != orig_name:
                # Only reassign if the name is different
                torch.fx.graph_module._assign_attr(
                    traced_param, traced_module, orig_name
                )
                torch.fx.graph_module._del_attr(traced_module, traced_name)
                name_mapping[traced_name] = orig_name
        else:
            # This param doesn't exist in traced module - add it
            torch.fx.graph_module._assign_attr(orig_param, traced_module, orig_name)

    # Restore buffers in the order they appear in original module
    for orig_name, orig_buffer in orig_buffers_list:
        if id(orig_buffer) in traced_buffers:
            # This buffer exists in traced module - restore it with original FQN
            traced_name, traced_buffer = traced_buffers[id(orig_buffer)]
            processed_traced_names.add(traced_name)
            if traced_name != orig_name:
                # Only reassign if the name is different
                torch.fx.graph_module._assign_attr(
                    orig_buffer, traced_module, orig_name
                )
                torch.fx.graph_module._del_attr(traced_module, traced_name)
                name_mapping[traced_name] = orig_name
        else:
            # This buffer doesn't exist in traced module - add it
            torch.fx.graph_module._assign_attr(orig_buffer, traced_module, orig_name)

    param_names = [v[0] for v in traced_params.values()]
    buffer_names = [v[0] for v in traced_buffers.values()]
    # Constants are traced params/buffers that weren't matched to any original param/buffer
    const_keys = list(
        set(param_names + buffer_names).difference(processed_traced_names)
    )

    _clear_traced_params_buffers(traced_module, const_keys)

    # Update get_attr nodes in the graph to use the correct FQNs
    for node in traced_module.graph.nodes:
        if node.op == "get_attr" and node.target in name_mapping:
            node.target = name_mapping[node.target]

    traced_module.recompile()
