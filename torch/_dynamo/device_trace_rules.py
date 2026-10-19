"""Stream-runtime trace rules for registered DeviceInterface backends."""

import sys
from typing import Any


_DEVICE_TRACE_RULE_METHODS = (
    "stream",
    "current_stream",
    "set_stream",
    "default_stream",
)


def registered_device_trace_fns() -> tuple[Any, ...]:
    """Stream-runtime callables overridden on the current device registry.

    Indexed keys such as ``npu:0`` stay in the scan. A backend may register
    only that key. Each function object is returned once. Replacing an
    interface drops callables that are no longer on the registry.
    """
    from .device_interface import DeviceInterface, get_registered_device_interfaces

    found: list[Any] = []
    seen: set[Any] = set()
    for _, interface in get_registered_device_interfaces():
        for name in _DEVICE_TRACE_RULE_METHODS:
            fn = getattr(interface, name, None)
            base = getattr(DeviceInterface, name, None)
            if fn is None or fn is base or not callable(fn):
                continue
            if fn in seen:
                continue
            seen.add(fn)
            found.append(fn)
    return tuple(found)


def register_device_trace_rules(interface: type) -> None:
    """Invalidate Dynamo maps after a DeviceInterface registration.

    Lookup reads ``registered_device_trace_fns()`` from the live registry, so
    a replaced interface loses its previous callables on the next trace.
    ``interface`` is the class just stored by ``register_interface_for_device``.
    """
    if not isinstance(interface, type):
        return
    # trace_rules imports inductor, and that import registers the in-tree
    # interfaces. This function runs before trace_rules has finished loading,
    # so the caches below may not exist yet.
    trace_rules = sys.modules.get("torch._dynamo.trace_rules")
    if trace_rules is None:
        return
    rule_map = getattr(trace_rules, "get_torch_obj_rule_map", None)
    if rule_map is not None:
        rule_map.cache_clear()
    variable_cls = getattr(trace_rules, "TorchInGraphFunctionVariable", None)
    if variable_cls is not None:
        variable_cls._get_handlers.cache_clear()
