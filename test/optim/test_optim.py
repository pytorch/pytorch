# Owner(s): ["module: optimizer"]

from __future__ import annotations

from typing import Any

import torch
from torch import nn, Tensor
from torch.optim import Optimizer, SGD
from torch.testing._internal.common_utils import (
    gradcheck,
    load_tests,
    skipIfTorchDynamo,
    TestCase,
)


# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests


def _multistep_backprop_diff_hyperparams_fn(
    params: Tensor,
    grad: Tensor,
    opt_differentiable_state: dict[str, Any],
    opt_class: type[Optimizer],
    kwargs: dict[str, Any],
    *ignored: Any,
) -> tuple[Tensor, ...]:
    assert (
        kwargs["differentiable"] is True
    ), "Only call this test function when differentiable=True"

    params = params.clone()
    params.grad = grad

    opt_differentiable_state = {
        k: v.clone() if isinstance(v, torch.Tensor) else v
        for k, v in opt_differentiable_state.items()
    }

    # This copy is necessary so the update on line 78 doesn't overwrite the original kwargs values
    kwargs = kwargs.copy()

    # Have to pass in beta1 and beta2 separately
    # so they're passed in as Tensors (not a tuple) and recognized by gradcheck
    if "beta1" in kwargs or "beta2" in kwargs:
        # Prevent just one beta kwarg from being passed in
        assert (
            "beta1" in kwargs and "beta2" in kwargs
        ), "Both betas should be defined in kwargs"
        kwargs.update({"betas": (kwargs.pop("beta1"), kwargs.pop("beta2"))})

    kwargs.update(
        {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
    )
    differentiable_kwargs = [
        v for v in kwargs.values() if isinstance(v, torch.Tensor) and v.requires_grad
    ] + (list(kwargs["betas"]) if "betas" in kwargs else [])

    criterion = nn.MSELoss()

    optimizer = opt_class([params], **kwargs)
    optimizer.state[params].update(opt_differentiable_state)

    # Simple x, y pair
    x = torch.tensor([1.0], dtype=torch.float64)
    y = torch.tensor([2.0], dtype=torch.float64)

    for _ in range(2):
        loss = criterion(x * torch.sum(params), y)
        loss.backward(
            inputs=(params,),
            create_graph=True,
        )
        optimizer.step()
        optimizer.zero_grad()

    meta_loss = loss
    meta_loss.backward(inputs=(*differentiable_kwargs,), create_graph=True)

    # Extra check to make sure the test properly computed a gradient for all kwargs
    for kwarg in differentiable_kwargs:
        assert kwarg.grad is not None

    return (
        (meta_loss,)
        + tuple(
            v
            for v in optimizer.state[params].values()
            if isinstance(v, torch.Tensor) and v.requires_grad
        )
        + tuple(differentiable_kwargs)
    )


@skipIfTorchDynamo("Differentiable optimizers not supported")
class TestDifferentiableOptimizer(TestCase):
    def test_differentiable_lr(self):
        params = torch.rand(10, requires_grad=True, dtype=torch.float64)
        grad = torch.rand_like(params, requires_grad=True, dtype=torch.float64)
        lr = torch.tensor(0.001, requires_grad=True, dtype=torch.float64)

        mbuff = torch.rand_like(params, requires_grad=True, dtype=torch.float64)
        state = {"momentum_buffer": mbuff}
        kwargs: dict[str, Any] = {"lr": lr, "differentiable": True}

        gradcheck(
            _multistep_backprop_diff_hyperparams_fn,
            (
                params,
                grad,
                state,
                SGD,
                kwargs,  # includes lr
                *state.values(),
                *kwargs.values(),
            ),
        )

    def test_differentiable_weight_decay(self):
        params = torch.rand(10, requires_grad=True, dtype=torch.float64)
        grad = torch.rand_like(params, requires_grad=True, dtype=torch.float64)
        weight_decay = torch.tensor(0.9, requires_grad=True, dtype=torch.float64)

        mbuff = torch.rand_like(params, requires_grad=True, dtype=torch.float64)
        state = {"momentum_buffer": mbuff}
        kwargs: dict[str, Any] = {"weight_decay": weight_decay, "differentiable": True}

        gradcheck(
            _multistep_backprop_diff_hyperparams_fn,
            (
                params,
                grad,
                state,
                SGD,
                kwargs,  # includes weight_decay
                *state.values(),
                *kwargs.values(),
            ),
        )

    def test_differentiable_weight_decay_and_lr(self):
        params = torch.rand(10, requires_grad=True, dtype=torch.float64)
        grad = torch.rand_like(params, requires_grad=True, dtype=torch.float64)

        weight_decay = torch.tensor(0.9, requires_grad=True, dtype=torch.float64)
        lr = torch.tensor(0.001, requires_grad=True, dtype=torch.float64)

        mbuff = torch.rand_like(params, requires_grad=True, dtype=torch.float64)
        state = {"momentum_buffer": mbuff}

        kwargs: dict[str, Any] = {
            "weight_decay": weight_decay,
            "lr": lr,
            "differentiable": True,
        }

        gradcheck(
            _multistep_backprop_diff_hyperparams_fn,
            (
                params,
                grad,
                state,
                SGD,
                kwargs,  # includes lr & weight_decay
                *state.values(),
                *kwargs.values(),
            ),
        )


if __name__ == "__main__":
    print("These tests should be run through test/test_optim.py instead")
