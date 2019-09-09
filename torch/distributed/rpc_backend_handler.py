from __future__ import absolute_import, division, print_function, unicode_literals


_rpc_init_handlers = {}


def register_rpc_backend(backend_str, handler):
    """Registers a new rpc backend.

    Arguments:
        backend (str): backend string to identify the handler.
        handler (function): Handler that is invoked when the
            `_init_rpc()` function is called with a backend.
             This returns the agent.
    """
    global _rpc_init_handlers
    if backend_str in _rpc_init_handlers:
        raise RuntimeError(
            "Rpc backend {}: already registered".format(backend_str)
        )
    _rpc_init_handlers[backend_str] = handler


def registered_init_rpc(backend_str, **kwargs):
    if backend_str not in _rpc_init_handlers:
        raise RuntimeError("No rpc_init handler for {}.".format(backend_str))
    return _rpc_init_handlers[backend_str](**kwargs)


def is_backend_registered(backend_str):
    return backend_str in _rpc_init_handlers
