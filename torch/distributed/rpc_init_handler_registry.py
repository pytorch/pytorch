from __future__ import absolute_import, division, print_function, unicode_literals


_RPC_INIT_HANDLER_REGISTRY = {}


def _get_rpc_init_handler_registry():
    return _RPC_INIT_HANDLER_REGISTRY


def register_rpc_init_handler(backend_name, rpc_init_handler):
    """Registers a new rpc backend.

    Arguments:
        backend (str): backend string to identify the handler.
        handler (function): Handler that is invoked when the
            `_init_rpc()` function is called with a backend.
             This returns the agent.
    """
    rpc_init_handler_registry = _get_rpc_init_handler_registry()
    if backend_name in rpc_init_handler_registry:
        raise RuntimeError("Rpc backend {}: already registered".format(backend_name))
    rpc_init_handler_registry[backend_name] = rpc_init_handler


def rpc_init(backend_name, **kwargs):
    rpc_init_handler_registry = _get_rpc_init_handler_registry()
    if backend_name not in rpc_init_handler_registry:
        raise RuntimeError("No rpc_init handler for {}.".format(backend_name))
    return rpc_init_handler_registry[backend_name](**kwargs)


def is_backend_registered(backend_name):
    return backend_name in _get_rpc_init_handler_registry()
