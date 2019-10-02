from __future__ import absolute_import, division, print_function, unicode_literals


_RPC_BACKEND_REGISTRY = {}


def _get_rpc_backend_registry():
    return _RPC_BACKEND_REGISTRY


def is_rpc_backend_registered(backend_name):
    return backend_name in _get_rpc_backend_registry()


def register_rpc_backend(backend_name, init_rpc_backend_handler):
    """Registers a new rpc backend.

    Arguments:
        backend (str): backend string to identify the handler.
        handler (function): Handler that is invoked when the
            `_init_rpc()` function is called with a backend.
             This returns the agent.
    """
    rpc_backend_registry = _get_rpc_backend_registry()
    if backend_name in rpc_backend_registry:
        raise RuntimeError("Rpc backend {}: already registered".format(backend_name))
    rpc_backend_registry[backend_name] = init_rpc_backend_handler


def init_rpc_backend(backend_name, *args, **kwargs):
    rpc_backend_registry = _get_rpc_backend_registry()
    if backend_name not in rpc_backend_registry:
        raise RuntimeError("No rpc_init handler for {}.".format(backend_name))
    return rpc_backend_registry[backend_name](*args, **kwargs)
