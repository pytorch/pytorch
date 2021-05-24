from typing import Type

from ..._C import _suspend_backend_dispatch, _restore_backend_dispatch
from ..modules import Module


def no_init(module_cls: Type[Module], *args, **kwargs) -> Module:
    """Constructs a module without initializing its parameters and buffers.

    This function is meant to be used if the size of the specified module is big
    and you want to inspect it without fully instantiating. For instance an
    automated sharding system can construct a module with ``no_init()``, inspect
    its architecture, and materialize its layers at a later time.

    Internally ``no_init()`` forces all parameters and buffers of the module and
    its sub-modules to use a meta device regardless of the requested device
    type. The returned module can be used for inspection purposes, but has to be
    materialized by calling its ``materialize()`` method before doing actual
    work.

    Args:
        module_cls:
            The type of the module to construct.
        args:
            The positional arguments to pass to the module's constructor.
        kwargs:
            The keyword arguments to pass to the module's constructor.

    Returns:
        A module instance with uninitialized parameters and buffers.

    Example::
        >>> import torch
        >>> m = torch.nn.utils.no_init(torch.nn.Linear, 5, 1)
        >>> m.weight
        Parameter containing:
        tensor(..., device='meta', requires_grad=True)
        >>> n = m.materialize()
        >>> n.weight
        Parameter containing:
        tensor([[-1.4677e+24,  4.5915e-41,  1.4013e-45,  0.0000e+00, -1.4677e+24,
                  4.5915e-41]], requires_grad=True)

    Note:
        The ``args`` and ``kwargs`` arguments must be treated as immutable since
        they will be used a second time during materialization.
    """
    _suspend_backend_dispatch()

    try:
        module = module_cls(*args, **kwargs)
    finally:
        _restore_backend_dispatch()

    def materialize() -> Module:
        return module_cls(*args, **kwargs)

    module.materialize = materialize

    return module
