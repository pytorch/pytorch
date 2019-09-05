from __future__ import absolute_import, division, print_function, unicode_literals


_rpc_init_handlers = {}


def register_rpc_handler(backend_str, handler):
    """Registers a new rpc handler.

    Arguments:
        backend (str): backend to identify the handler.
        handler (function): Handler that is invoked when the
            `init_rpc()` function is called with a backend. This sets the global agent.
    """
    global _rpc_init_handlers
    if backend_str in _rpc_init_handlers:
        raise RuntimeError(
            "Init handler for {}: already registered".format(backend_str)
        )
    _rpc_init_handlers[backend_str] = handler


def registered_init_rpc(backend_str, **kwargs):
    global _rpc_init_handlers
    if backend_str not in _rpc_init_handlers:
        raise RuntimeError("No rpc_init handler for {}.".format(backend_str))
    return _rpc_init_handlers[backend_str](**kwargs)


def is_backend_registered(backend_str):
    global _rpc_init_handlers
    return backend_str in _rpc_init_handlers
