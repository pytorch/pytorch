from typing import List, Dict, Optional
import torch
import torch.optim._functional as F

from torch import Tensor

# Define a TorchScript compatible Functional Adagrad Optimizer
# where we use these optimizer in a functional way.
# Instead of using the `param.grad` when updating parameters,
# we explicitly let the user pass gradients to the `step` function
# this is so that we could separate the gradients and parameters
# and allow multithreaded trainer to update the parameters
# without data traces on accumulating to the same .grad.
# NOTE: This should be only used by distributed optimizer internals
# and not meant to expose to the user.
@torch.jit.script
class _FunctionalAdagrad(object):
    def __init__(
        self,
        params: List[Tensor],
        lr: float = 1e-2,
        lr_decay: float = 0.0,
        weight_decay: float = 0.0,
        initial_accumulator_value: float = 0.0,
        warmup_lr_multiplier: float = 1.0,
        warmup_num_iters: float = 0.0,
        eps: float = 1e-10,
        coalesce_grad: bool = True,
        _allow_empty_param_list: bool = False,
    ):
        self.defaults = {
            "lr": lr,
            "lr_decay": lr_decay,
            "eps": eps,
            "weight_decay": weight_decay,
            "initial_accumulator_value": initial_accumulator_value,
            "warmup_lr_multiplier": warmup_lr_multiplier,
            "warmup_num_iters": warmup_num_iters,
        }
        self.coalesce_grad = coalesce_grad
        self.state = torch.jit.annotate(Dict[torch.Tensor, Dict[str, torch.Tensor]], {})

        if len(params) == 0 and not _allow_empty_param_list:
            raise ValueError("optimizer got an empty parameter list")

        # NOTE: we only have one param_group and don't allow user to add additional
        # param group as it's not a common use case.
        self.param_group = {"params": params}

        # TODO: no union or any types in TorchScript, make step a scalar tensor instead
        # This is also needed by if we want to share_memory on the step across processes
        for p in self.param_group["params"]:
            self.state[p] = {
                "sum": torch.full_like(p.data, initial_accumulator_value),
                "step": torch.tensor(0.0),
            }

    def step(self, gradients: List[Optional[Tensor]]):
        params = self.param_group['params']
        params_with_grad = []
        grads = []
        state_sums = []
        state_steps: List[int] = []

        if len(params) != len(gradients):
            raise ValueError(
                "the gradients passed in does not equal to the size of the parameters!"
                + f"Params length: {len(params)}. "
                + f"Gradients length: {len(gradients)}"
            )

        for param, gradient in zip(self.param_group['params'], gradients):
            if gradient is not None:
                params_with_grad.append(param)
                grads.append(gradient)
                state = self.state[param]
                state_sums.append(state['sum'])
                # update the steps for each param group update
                state['step'] += 1
                # record the step after step update
                state_steps.append(state['step'].item())

        with torch.no_grad():
            F.adagrad(params,
                      grads,
                      state_sums,
                      state_steps,
                      lr=self.defaults['lr'],
                      weight_decay=self.defaults['weight_decay'],
                      lr_decay=self.defaults['lr_decay'],
                      eps=self.defaults['eps'])
