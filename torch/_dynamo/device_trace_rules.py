"""Stream-runtime trace rules for registered DeviceInterface backends."""

import sys
from typing import Any


_registered_device_trace_fns: set[Any] = set()
_DEVICE_TRACE_RULE_METHODS = (
    "stream",
    "current_stream",
    "set_stream",
    "default_stream",
)


def register_device_trace_rules(interface: type) -> None:
    """Mark one backend's stream runtime functions as in-graph.

    Out-of-tree backends are not in the static torch name lists. Registration
    records the interface's overridden stream methods so lookup treats them
    like torch.cuda.stream / current_stream / set_stream. Handlers are rebuilt
    on the next trace.
    """
    from .device_interface import DeviceInterface

    added = False
    for name in _DEVICE_TRACE_RULE_METHODS:
        fn = getattr(interface, name, None)
        base = getattr(DeviceInterface, name, None)
        if fn is None or fn is base or not callable(fn):
            continue
        if fn not in _registered_device_trace_fns:
            _registered_device_trace_fns.add(fn)
            added = True
    if not added:
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


def overridden_device_trace_fns(name: str) -> list[Any]:
    """Return stream-runtime callables overridden by registered interfaces.

    Indexed keys such as ``npu:0`` stay in the scan. A backend may register
    only that key. ``dict.fromkeys`` at the handler drops duplicate functions.
    """
    from .device_interface import DeviceInterface, get_registered_device_interfaces

    base = getattr(DeviceInterface, name, None)
    found: list[Any] = []
    for _, interface in get_registered_device_interfaces():
        fn = getattr(interface, name, None)
        if fn is None or fn is base or not callable(fn):
            continue
        found.append(fn)
    return found
