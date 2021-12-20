import contextlib

import torch.jit


def _change_class(module) -> None:
    cls = module.__class__
    func_params = module._functional_parameters

    def _getattribute(self, name):
        if name in func_params:
            return func_params[name]
        return object.__getattribute__(self, name)

    param_cls = type(
        f"StatelessReplacer{cls.__name__}",
        (cls,),
        {
            "__getattribute__": _getattribute,
        },
    )

    module.__class__ = param_cls
    module._orig_class = cls


def _swap_parameters(module, tensor_name, tensor):
    # Changes the module class to get a new __getattr__ dunder method
    # that looks for the reparametrized tensor
    if hasattr(module, "_functional_parameters"):
        # Check if the module has an active reparametrization
        # for this parameter
        module._functional_parameters[tensor_name] = tensor
    else:
        # change module class to set a new __getattr__ function
        # register tensor
        module._functional_parameters = {}
        module._functional_parameters[tensor_name] = tensor
        _change_class(module)


def _remove_swap(module, name):
    if hasattr(module, "_orig_class"):
        module.__class__ = module._orig_class
        delattr(module, "_orig_class")


@contextlib.contextmanager
def reparametrize_module(module, parameters_and_buffers):
    # Parametrization does not support to change submodules directly
    for name, tensor in parameters_and_buffers.items():
        _apply_func_submodules(
            _swap_parameters,
            module, name.split("."), (tensor,))
    yield
    for name in parameters_and_buffers:
        _apply_func_submodules(
            _remove_swap,
            module, name.split("."), ())


def _apply_func_submodules(func, module, path, args):
    if len(path) == 1:
        func(module, path[0], *args)
    else:
        _apply_func_submodules(func, getattr(module, path[0]), path[1:], args)


def functional_call(module, parameters_and_buffers, args, kwargs=None):
    # TODO allow kwargs such as unsafe and others for parametrization
    if (
            torch.jit.is_tracing()
            or torch.jit.is_scripting()
            or isinstance(module, (
                torch.jit.RecursiveScriptModule,
                torch.jit.ScriptModule,
                torch.jit.ScriptFunction)
            )
    ):
        raise RuntimeError("The stateless API can't be used with Jitted modules")
    if kwargs is None:
        kwargs = {}
    with reparametrize_module(module, parameters_and_buffers):
        if isinstance(args, tuple):
            out = module(*args, **kwargs)
        else:
            out = module(args, **kwargs)
    return out
