from typing import Any, Dict, Iterable, Type, List, no_type_check

import torch

__all__: List[str] = []

@no_type_check
def _apply_optimizer_in_backward(
    optimizer_class: Type[torch.optim.Optimizer],
    params: Iterable[torch.nn.Parameter],
    optimizer_kwargs: Dict[str, Any],
) -> None:
    """
    Upon ``backward()``, parameters will fire the corresponding optimizer.

    Note - gradients for these parameters will be set to None after ``backward()``.
    This means that any other (non applied) optimizer over this parameter will be
    a no-op.

    Args:
        optimizer_class: (Type[torch.optim.Optimizer]): Optimizer to apply to parameter
        params: (Iterator[nn.Parameter]): parameters to apply optimizer state to
        optimizer_kwargs: (Dict[str, Any]): kwargs to pass to optimizer constructor

    Example::
        params_generator = model.parameters()
        param_1 = next(params_generator)
        remainder_params = list(params_generator)

        apply_optimizer_in_backward(torch.optim.SGD, [param_1], {"lr": .02})
        apply_optimizer_in_backward(torch.optim.Adam, remainder_params, {"lr": .04})

        model(...).sum().backward() # after backward, parameters will already
        # have their registered optimizer applied.

    """

    @no_type_check
    def _apply_optimizer_in_backward_to_param(param: torch.nn.Parameter) -> None:
        # view_as creates a node in autograd graph that allows us access to the
        # parameter's AccumulateGrad autograd function object. We register a
        # hook on this object to fire the optimizer when the gradient for
        # this parameter is ready (has been accumulated into .grad field)

        # Don't create a new acc_grad if we already have one
        # i.e.f or shared parameters or attaching multiple optimizers to a param.
        if not hasattr(param, 'acc_grad'):
            acc_grad = param.view_as(param).grad_fn.next_functions[0][0]
        else:
            acc_grad = param._acc_grad

        optimizer = optimizer_class([param], **optimizer_kwargs)

        # Keep the grad accumulator around for the lifetime of the Tensor,
        # store it on the param to avoid uncollectable ref-cycle
        if not hasattr(param, 'acc_grad'):
            param._acc_grad = acc_grad  # type: ignore[attr-defined]

        if not hasattr(param, '_in_backward_optimizers'):
            param._in_backward_optimizers = []  # type: ignore[attr-defined]
            # TODO: investigate whether we really need these attributes.
            param._optimizer_classes = []  # type: ignore[attr-defined]
            param._optimizer_kwargs = []  # type: ignore[attr-defined]

        param._in_backward_optimizers.append(optimizer)  # type: ignore[attr-defined]
        param._optimizer_classes.append(optimizer_class)  # type: ignore[attr-defined]
        param._optimizer_kwargs.append(optimizer_kwargs)  # type: ignore[attr-defined]

        def optimizer_hook(*_unused) -> None:
            for opt in param._in_backward_optimizers:  # type: ignore[attr-defined]
                opt.step()

            param.grad = None

        param._acc_grad.register_hook(optimizer_hook)  # type: ignore[attr-defined]

    for param in params:
        _apply_optimizer_in_backward_to_param(param)
