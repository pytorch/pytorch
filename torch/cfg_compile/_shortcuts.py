"""What ``nn.Module`` comes down to when nothing special is going on.

Two functions of ``nn.Module`` sit in front of nearly everything a model does:
``Module.__call__``, which calls ``forward`` unless the module has hooks, and
``Module.__getattr__``, which finds parameters, buffers and submodules in
three dicts. The runtime of ``torch.cfg_compile`` can skip both: this module
tells it under which conditions and to what effect.

What is said here restates the source of those functions, so it is only
registered while that source is what it was when this was written: the
functions are compared with a hash of their text. After a change to them,
read it, bring the conditions below in line, and update the hashes.
"""

import hashlib
import inspect
from typing import Any

from torch.nn.modules import module as _module


# sha256 of the source of the functions the shortcuts stand in for.
_SOURCES = {
    "_wrapped_call_impl": "b11681464e3ee0eba9b806c3b503dfaba804fc28ade233e74998fc3104dcac91",
    "_call_impl": "3dd0447cad1ac3cb08f3d932c041652ec8ba8ccf48b53ad580494f514f48510f",
    "__getattr__": "fe51052b67121828f9ddeb118211e13cfe30435ecb6f1eb96d16efb9110d8acf",
}

_registered: bool | None = None


def unchanged() -> bool:
    """Whether ``nn.Module`` still has the source the shortcuts were written for."""
    members = vars(_module.Module)
    try:
        return all(
            hashlib.sha256(inspect.getsource(members[name]).encode()).hexdigest()
            == digest
            for name, digest in _SOURCES.items()
        )
    except (OSError, TypeError, KeyError):
        return False


def register(runtime: Any) -> bool:
    """Registers the shortcuts through ``nn.Module`` with the runtime, once.

    Returns whether they are registered: they are not if ``nn.Module`` has
    changed since they were written.
    """
    global _registered
    if _registered is not None:
        return _registered
    _registered = unchanged()
    if not _registered:
        return False
    members = vars(_module.Module)
    hooks = (
        "_backward_hooks",
        "_backward_pre_hooks",
        "_forward_hooks",
        "_forward_pre_hooks",
    )
    global_hooks = (
        "_global_backward_pre_hooks",
        "_global_backward_hooks",
        "_global_forward_hooks",
        "_global_forward_pre_hooks",
    )
    # _wrapped_call_impl and _call_impl: `module(*args, **kwargs)` is
    # `module.forward(*args, **kwargs)` unless the module was compiled, is
    # being traced, or it or all modules have hooks.
    runtime.register_call_shortcut(
        members["__call__"],
        "forward",
        [
            ("attr_none", "_compiled_call_impl", None),
            *[("instance_falsy", name, None) for name in hooks],
            *[("global_falsy", vars(_module), name) for name in global_hooks],
            ("not_tracing", None, None),
        ],
    )
    # __getattr__: what the normal lookup does not find is a parameter, a
    # buffer or a submodule, in that order.
    runtime.register_getattr_shortcut(
        members["__getattr__"], ["_parameters", "_buffers", "_modules"]
    )
    return True
