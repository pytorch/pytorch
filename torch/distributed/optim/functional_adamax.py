from typing import Dict, List, Optional, Tuple

import torch
import torch.optim._functional as F

from torch import Tensor

__all__: List[str] = []

# Define a TorchScript compatible Functional Adamax Optimizer
# where we use these optimizer in a functional way.
# Instead of using the `param.grad` when updating parameters,
# we explicitly allow the distributed optimizer pass gradients to
# the `step` function. In this way, we could separate the gradients
# and parameters and allow multithreaded trainer to update the
# parameters without data traces on accumulating to the same .grad.
# NOTE: This should be only used by distributed optimizer internals
# and not meant to expose to the user.
@torch.jit.script
class _FunctionalAdamax:
    def __init__(
        self,
        params: List[Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        foreach: bool = False,
        maximize: bool = False,
        _allow_empty_param_list: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        self.defaults = {
            "lr": lr,
            "eps": eps,
            "beta1": betas[0],
            "beta2": betas[1],
            "weight_decay": weight_decay,
        }
        self.foreach = foreach
        self.maximize = maximize
        self.state = torch.jit.annotate(Dict[torch.Tensor, Dict[str, torch.Tensor]], {})

        if len(params) == 0 and not _allow_empty_param_list:
            raise ValueError("optimizer got an empty parameter list")

        # NOTE: we only have one param_group and don't allow user to add additional
        # param group as it's not a common use case.
        self.param_group = {"params": params}

    def step(self, gradients: List[Optional[Tensor]]):
        params = self.param_group["params"]
        params_with_grad = []
        grads = []
        exp_avgs = []
        exp_infs = []
        state_steps: List[Tensor] = []

        if len(params) != len(gradients):
            raise ValueError(
                "the gradients passed in does not equal to the size of the parameters!"
                + f"Params length: {len(params)}. "
                + f"Gradients length: {len(gradients)}"
            )

        for param, gradient in zip(self.param_group["params"], gradients):
            if gradient is not None:
                params_with_grad.append(param)
                grads.append(gradient)
                # Lazy state initialization
                if param not in self.state:
                    self.state[param] = {}
                    state = self.state[param]
                    state["step"] = torch.tensor(0.0)
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_inf"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )

                state = self.state[param]

                exp_avgs.append(state["exp_avg"])
                exp_infs.append(state["exp_inf"])
                state_steps.append(state["step"])

        with torch.no_grad():
            F.adamax(
                params_with_grad,
                grads,
                exp_avgs,
                exp_infs,
                state_steps,
                eps=self.defaults["eps"],
                beta1=self.defaults["beta1"],
                beta2=self.defaults["beta2"],
                lr=self.defaults["lr"],
                weight_decay=self.defaults["weight_decay"],
                foreach=self.foreach,
                maximize=self.maximize,
            )
