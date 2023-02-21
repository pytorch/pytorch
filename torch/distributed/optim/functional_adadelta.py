from typing import Dict, List, Optional

import torch
import torch.optim._functional as F

from torch import Tensor

__all__: List[str] = []

# Define a TorchScript compatible Functional Adadelta Optimizer
# where we use these optimizer in a functional way.
# Instead of using the `param.grad` when updating parameters,
# we explicitly allow the distributed optimizer pass gradients to
# the `step` function. In this way, we could separate the gradients
# and parameters and allow multithreaded trainer to update the
# parameters without data traces on accumulating to the same .grad.
# NOTE: This should be only used by distributed optimizer internals
# and not meant to expose to the user.
@torch.jit.script
class _FunctionalAdadelta:
    def __init__(
        self,
        params: List[Tensor],
        lr: float = 1.0,
        rho: float = 0.9,
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        foreach: bool = False,
        maximize: bool = False,
        _allow_empty_param_list: bool = False,
    ):
        self.defaults = {
            "lr": lr,
            "rho": rho,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        self.foreach = foreach
        self.maximize = maximize

        if len(params) == 0 and not _allow_empty_param_list:
            raise ValueError("optimizer got an empty parameter list")

        # NOTE: we only have one param_group and don't allow user to add additional
        # param group as it's not a common use case.
        self.param_group = {"params": params}

        self.state = torch.jit.annotate(Dict[torch.Tensor, Dict[str, torch.Tensor]], {})

    def step(self, gradients: List[Optional[Tensor]]):
        params = self.param_group["params"]
        params_with_grad = []
        grads = []
        square_avgs = []
        acc_deltas = []
        lr = self.defaults["lr"]
        rho = self.defaults["rho"]
        eps = self.defaults["eps"]
        weight_decay = self.defaults["weight_decay"]

        if len(params) != len(gradients):
            raise ValueError(
                "the gradients passed in does not equal to the size of the parameters!"
                + f"Params length: {len(params)}. "
                + f"Gradients length: {len(gradients)}"
            )

        for param, gradient in zip(params, gradients):
            if gradient is not None:
                params_with_grad.append(param)
                grads.append(gradient)
                # Lazy state initialization
                if param not in self.state:
                    self.state[param] = {}
                    state = self.state[param]
                    state["step"] = torch.tensor(0.0)
                    state["square_avg"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )
                    state["acc_delta"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )

                state = self.state[param]
                square_avgs.append(state["square_avg"])
                acc_deltas.append(state["acc_delta"])

        with torch.no_grad():
            F.adadelta(
                params_with_grad,
                grads,
                square_avgs,
                acc_deltas,
                lr=lr,
                rho=rho,
                eps=eps,
                weight_decay=weight_decay,
                foreach=self.foreach,
                maximize=self.maximize,
            )
