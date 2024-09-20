# mypy: allow-untyped-defs
from typing import Dict, List, Optional, Tuple

import torch
import torch.optim._functional as F
from torch import Tensor


__all__: List[str] = []


# Define a TorchScript compatible Functional AdamW Optimizer
# where we use these optimizer in a functional way.
# Instead of using the `param.grad` when updating parameters,
# we explicitly allow the distributed optimizer pass gradients to
# the `step` function. In this way, we could separate the gradients
# and parameters and allow multithreaded trainer to update the
# parameters without data traces on accumulating to the same .grad.
# NOTE: This should be only used by distributed optimizer internals
# and not meant to expose to the user.
@torch.jit.script
class _FunctionalAdamW:
    def __init__(
        self,
        params: List[Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        maximize: bool = False,
        foreach: bool = False,
        fused: bool = False,
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
        self.amsgrad = amsgrad
        self.maximize = maximize
        self.foreach = foreach
        self.fused = fused
        self.state = torch.jit.annotate(Dict[torch.Tensor, Dict[str, torch.Tensor]], {})

        if len(params) == 0 and not _allow_empty_param_list:
            raise ValueError("optimizer got an empty parameter list")

        # NOTE: we only have one param_group and don't allow user to add additional
        # param group as it's not a common use case.
        self.param_group = {"params": params}

    def step_param(self, param: Tensor, grad: Optional[Tensor]):
        params_with_grad = []
        grads = []
        exp_avgs = []
        exp_avg_sqs = []
        max_exp_avg_sqs = []
        state_steps: List[Tensor] = []
        has_complex = torch.is_complex(param)
        if grad is not None:
            params_with_grad.append(param)
            grads.append(grad)
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
            state["exp_avg_sq"] = torch.zeros_like(
                param, memory_format=torch.preserve_format
            )
            if self.amsgrad:
                # Maintains max of all exp. moving avg. of sq. grad. values
                state["max_exp_avg_sq"] = torch.zeros_like(
                    param, memory_format=torch.preserve_format
                )

        state = self.state[param]

        exp_avgs.append(state["exp_avg"])
        exp_avg_sqs.append(state["exp_avg_sq"])

        if self.amsgrad:
            max_exp_avg_sqs.append(state["max_exp_avg_sq"])

        state_steps.append(state["step"])
        with torch.no_grad():
            F.adamw(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=self.amsgrad,
                maximize=self.maximize,
                beta1=self.defaults["beta1"],
                beta2=self.defaults["beta2"],
                lr=self.defaults["lr"],
                weight_decay=self.defaults["weight_decay"],
                eps=self.defaults["eps"],
                foreach=self.foreach,
                fused=self.fused,
                grad_scale=None,
                found_inf=None,
                has_complex=has_complex,
            )

    def step(self, gradients: List[Optional[Tensor]]):
        params = self.param_group["params"]
        params_with_grad = []
        grads = []
        exp_avgs = []
        exp_avg_sqs = []
        max_exp_avg_sqs = []
        state_steps: List[Tensor] = []

        if len(params) != len(gradients):
            raise ValueError(
                "the gradients passed in does not equal to the size of the parameters!"
                + f"Params length: {len(params)}. "
                + f"Gradients length: {len(gradients)}"
            )

        has_complex = False
        for param, gradient in zip(self.param_group["params"], gradients):
            if gradient is not None:
                has_complex |= torch.is_complex(param)
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
                    state["exp_avg_sq"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )
                    if self.amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            param, memory_format=torch.preserve_format
                        )

                state = self.state[param]

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])

                if self.amsgrad:
                    max_exp_avg_sqs.append(state["max_exp_avg_sq"])

                state_steps.append(state["step"])

        with torch.no_grad():
            F.adamw(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=self.amsgrad,
                maximize=self.maximize,
                beta1=self.defaults["beta1"],
                beta2=self.defaults["beta2"],
                lr=self.defaults["lr"],
                weight_decay=self.defaults["weight_decay"],
                eps=self.defaults["eps"],
                foreach=self.foreach,
                fused=self.fused,
                grad_scale=None,
                found_inf=None,
                has_complex=has_complex,
            )
