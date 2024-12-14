from typing import Any, Dict, Iterable, List, no_type_check, Type

import torch


__all__: List[str] = []

# WeakTensorKeyDictionary to store relevant meta-data for the Tensor/Parameter
# without changing it's life-time.
# NOTE: Alternative is to add the meta-data as an attribute to the tensor,
#       but that will serialize the meta-data if Tensor is serialized.
param_to_optim_hook_handle_map = torch.utils.weak.WeakTensorKeyDictionary()
param_to_acc_grad_map = torch.utils.weak.WeakTensorKeyDictionary()


@no_type_check
def _apply_optimizer_in_backward(
    optimizer_class: Type[torch.optim.Optimizer],
    params: Iterable[torch.nn.Parameter],
    optimizer_kwargs: Dict[str, Any],
    register_hook: bool = True,
) -> None:
    """
    Upon ``backward()``, the optimizer specified for each parameter will fire after
    the gradient has been accumulated into the parameter.

    Note - gradients for these parameters will be set to None after ``backward()``.
    This means that any other optimizer not specified via `_apply_optimizer_in_backward`
    over this parameter will be a no-op.

    Args:
        optimizer_class: (Type[torch.optim.Optimizer]): Optimizer to apply to parameter
        params: (Iterator[nn.Parameter]): parameters to apply optimizer state to
        optimizer_kwargs: (Dict[str, Any]): kwargs to pass to optimizer constructor
        register_hook: (bool): whether to register a hook that runs the optimizer
            after gradient for this parameter is accumulated. This is the default
            way that optimizer in backward is implemented, but specific use cases
            (such as DDP) may wish to override this to implement custom behavior.
            (Default = True)

    Example::
        params_generator = model.parameters()
        param_1 = next(params_generator)
        remainder_params = list(params_generator)

        apply_optimizer_in_backward(torch.optim.SGD, [param_1], {"lr": .02})
        apply_optimizer_in_backward(torch.optim.Adam, remainder_params, {"lr": .04})

        model(...).sum().backward() # after backward, parameters will already
        # have their registered optimizer(s) applied.

    """
    torch._C._log_api_usage_once("torch.distributed.optim.apply_optimizer_in_backward")

    @no_type_check
    def _apply_optimizer_in_backward_to_param(param: torch.nn.Parameter) -> None:
        # view_as creates a node in autograd graph that allows us access to the
        # parameter's AccumulateGrad autograd function object. We register a
        # hook on this object to fire the optimizer when the gradient for
        # this parameter is ready (has been accumulated into .grad field)

        # Don't create a new acc_grad if we already have one
        # i.e. for shared parameters or attaching multiple optimizers to a param.
        if param not in param_to_acc_grad_map:
            param_to_acc_grad_map[param] = param.view_as(param).grad_fn.next_functions[
                0
            ][0]

        optimizer = optimizer_class([param], **optimizer_kwargs)

        if not hasattr(param, "_in_backward_optimizers"):
            param._in_backward_optimizers = []
            # TODO: Remove these attributes once we have a better way of accessing
            # optimizer classes and kwargs for a parameter.
            param._optimizer_classes = []
            param._optimizer_kwargs = []

        param._in_backward_optimizers.append(optimizer)
        param._optimizer_classes.append(optimizer_class)
        param._optimizer_kwargs.append(optimizer_kwargs)

        if not register_hook:
            return

        def optimizer_hook(*_unused) -> None:
            for opt in param._in_backward_optimizers:
                opt.step()

            param.grad = None

        handle = param_to_acc_grad_map[param].register_hook(optimizer_hook)
        if param not in param_to_optim_hook_handle_map:
            param_to_optim_hook_handle_map[param] = []
        param_to_optim_hook_handle_map[param].append(handle)

    for param in params:
        _apply_optimizer_in_backward_to_param(param)


def _get_in_backward_optimizers(module: torch.nn.Module) -> List[torch.optim.Optimizer]:
    """
    Return a list of in-backward optimizers applied to ``module``'s parameters. Note that these
    optimizers are not intended to directly have their ``step`` or ``zero_grad`` methods called
    by the user and are intended to be used for things like checkpointing.

    Args:
        module: (torch.nn.Module): model to retrieve in-backward optimizers for

    Returns:
        List[torch.optim.Optimizer]: the in-backward optimizers.

    Example::
        _apply_optimizer_in_backward(torch.optim.SGD, model.parameters(), {'lr': 0.01})
        optims = _get_optimizers_in_backward(model)
    """
    optims: List[torch.optim.Optimizer] = []
    for param in module.parameters():
        optims.extend(getattr(param, "_in_backward_optimizers", []))

    return optims
