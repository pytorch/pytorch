from typing import List, Optional
import torch
import torch.optim.functional as F

from torch import Tensor

# Define a TorchScript compatible Functional SGD Optimizer
# where we use these optimizer in a functional way.
# Instead of using the `param.grad` when updating parameters,
# we explicitly let the user pass gradients to the `step` function
# this is so that we could separate the gradients and parameters
# and allow multithreaded trainer to update the parameters
# without data traces on accumulating to the same .grad.
# NOTE: This should be only used by distributed optimizer internals
# and not meant to expose to the user.
@torch.jit.script
class _FunctionalSGD(object):
    def __init__(
        self,
        params: List[Tensor],
        lr: float = 1e-2,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False
    ):
        self.defaults = {
            "lr": lr,
            "momentum": momentum,
            "dampening": dampening,
            "weight_decay": weight_decay,
        }
        self.nesterov = nesterov

        if len(params) == 0:
            raise ValueError("optimizer got an empty parameter list")

        # NOTE: we only have one param_group and don't allow user to add additional
        # param group as it's not a common use case.
        self.param_group = {"params": params}

    def step(self, gradients: List[Optional[Tensor]]):
        params = self.param_group['params']
        grads = []
        momentum_buffer_list: List[Optional[Tensor]] = []
        lr = self.defaults['lr']
        weight_decay = self.defaults['weight_decay']
        momentum = self.defaults['momentum']
        dampening = self.defaults['dampening']

        if len(params) != len(gradients):
            raise ValueError(
                "the gradients passed in does not equal to the size of the parameters!"
                + f"Params length: {len(params)}. "
                + f"Gradients length: {len(gradients)}"
            )

        for param, gradient in zip(params, gradients):
            if gradient is not None:
                grads.append(gradient)
                momentum_buffer_list.append(None)

        with torch.no_grad():
            F.sgd(params,
                  grads,
                  momentum_buffer_list,
                  weight_decay,
                  momentum,
                  lr,
                  dampening,
                  self.nesterov)
