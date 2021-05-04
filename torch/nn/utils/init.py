import inspect
import torch


def skip_init(module_cls, device='cpu'):
    r"""
    Given a module class object, returns a function that will instantiate the module on the specified
    device without initializing parameters / buffers. This can be useful if initialization is slow or
    if custom initialization will be performed, making the default initialization unnecessary. There
    are some caveats to this, due to the way this function is implemented:

    1. The module must accept a `device` arg in its constructor that is passed to any parameters
    or buffers created during construction.
    2. The module must not perform any computation on parameters in its constructor except
    initialization (i.e. functions from :mod:`torch.nn.init`).

    If these conditions are satisfied, the module can be instantiated with parameter / buffer values
    uninitialized, as if having been created using :func:`torch.empty`.

    Args:
        module_cls: Class object; should be a subclass of :class:`torch.nn.Module`
        device: Device on which to create the module. Default: 'cpu'

    Returns:
        A function that will instantiate the module without initializing parameters / buffers

    Example::

        >>> import torch
        >>> l = torch.nn.utils.skip_init(torch.nn.Linear)
        >>> m1 = l(5, 1)
        >>> m1.weight
        Parameter containing:
        tensor([[0.0000e+00, 1.5846e+29, 7.8307e+00, 2.5250e-29, 1.1210e-44]],
               requires_grad=True)
        >>> m2 = l(in_features=6, out_features=1)
        >>> m2.weight
        Parameter containing:
        tensor([[-1.4677e+24,  4.5915e-41,  1.4013e-45,  0.0000e+00, -1.4677e+24,
                  4.5915e-41]], requires_grad=True)

    """
    if not issubclass(module_cls, torch.nn.Module):
        raise RuntimeError('Expected a Module; got {}'.format(module_cls))
    if 'device' not in inspect.signature(module_cls).parameters:
        raise RuntimeError('Module must support a \'device\' arg to skip initialization')

    def _instantiate(*args, **kwargs):
        return module_cls(*args, **kwargs, device='meta').to_empty(device=device)
    return _instantiate
