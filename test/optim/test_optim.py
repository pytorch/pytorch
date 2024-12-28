# Owner(s): ["module: optimizer"]

from __future__ import annotations

from typing import Any

import torch
from torch import nn, Tensor
from torch.optim import Optimizer, SGD
from torch.testing._internal.common_optimizers import optim_db, OptimizerInfo, optims
from torch.testing._internal.common_utils import (
    gradcheck,
    load_tests,
    skipIfTorchDynamo,
    TestCase,
)


# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests


def _diff_fn(p, grad, opt_differentiable_state, opt_class, kwargs, *ignored):
    # Ignored is the list of values in `opt_differentiable_state`, we do this
    # for `gradcheck` to correctly track the state tensors as function inputs
    # because otherwise it can't unpack the values in the `opt_differentiable_state`
    # dict
    p = p.clone()
    p.grad = grad
    opt_differentiable_state = {
        k: v.clone() if isinstance(v, torch.Tensor) else v
        for k, v in opt_differentiable_state.items()
    }
    opt = opt_class([p], **kwargs)
    opt.state[p].update(opt_differentiable_state)
    opt.step()
    return (p,) + tuple(
        v
        for v in opt.state[p].values()
        if isinstance(v, torch.Tensor) and v.requires_grad
    )


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


def _get_optimizer_state(
    optim_cls: type[Optimizer],
    params: Tensor,
    optim_kwargs: dict[str, Any],
) -> dict[str, Any]:
    params.grad = torch.rand_like(params)

    optim = optim_cls([params], **optim_kwargs)
    optim.step()

    states = optim.state_dict()["state"]
    # Accesses state of the first (and only) parameter. When using SGD w/ no momentum, there is no state
    state = states[0] if 0 in states else states
    return state


class TestDifferentiableOptimizerOptimizerInfo(TestCase):
    @optims(optim_db, dtypes=[torch.float64])
    def test_optimizers_differentiable_wrt_params(
        self, device: str, dtype: torch.dtype, optim_info: OptimizerInfo
    ):
        optim_cls = optim_info.optim_cls
        if "differentiable" not in optim_info.supported_impls:
            self.skipTest(f"Differentiable {optim_cls.__name__} not supported")

        for optim_input in optim_info.optim_inputs_func(device=device):
            mock_params = torch.rand(10, requires_grad=True, device=device, dtype=dtype)
            mock_optim_kwargs = optim_input.kwargs.copy()
            optim_state = _get_optimizer_state(
                optim_cls, mock_params, mock_optim_kwargs
            )

            optim_kwargs = optim_input.kwargs
            optim_kwargs.update({"differentiable": True})

            # Cannot be capturable and differentiable
            if "capturable" in optim_kwargs:
                optim_kwargs.update({"capturable": False})

            # sgd defaults to foreach on cuda so must explicitly specify not foreach
            if optim_cls is SGD and device.startswith("cuda"):
                optim_kwargs.update({"foreach": False})

            params = torch.rand(10, requires_grad=True, device=device, dtype=dtype)
            grad = torch.rand(10, requires_grad=True, device=device, dtype=dtype)
            gradcheck(
                _diff_fn,
                (
                    params,
                    grad,
                    optim_state,
                    optim_cls,
                    optim_kwargs,
                    *optim_state.values(),
                ),
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
    A = TestDifferentiableOptimizerOptimizerInfo()
    A.test_optimizers_differentiable_wrt_params("cpu", torch.float64, optim_db[-2])
    print("These tests should be run through test/test_optim.py instead")
