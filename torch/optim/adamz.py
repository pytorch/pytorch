import torch
from torch import optim
from collections import deque
import numpy as np

"""
AdamZ Optimizer Implementation

This implementation of the AdamZ optimizer is an extension of the traditional Adam optimizer, 
designed to adjust the learning rate dynamically based on the characteristics of the loss function during training. 
It introduces mechanisms to handle overshooting and stagnation in the optimization process.

Hyperparameters:
- params: Iterable of parameters to optimize or dicts defining parameter groups.
- lr (float, optional): The learning rate (default: 0.01).
- betas (Tuple[float, float], optional): Coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999)).
- eps (float, optional): Term added to the denominator to improve numerical stability (default: 1e-8).
- overshoot_factor (float, optional): Factor by which the learning rate is reduced in case of overshooting (default: 0.5).
- stagnation_factor (float, optional): Factor by which the learning rate is increased in case of stagnation (default: 1.2).
- stagnation_threshold (float, optional): Threshold for detecting stagnation based on the standard deviation of the loss (default: 0.2).
- patience (int, optional): Number of steps to wait before adjusting the learning rate (default: 100).
- stagnation_period (int, optional): Number of steps to consider for stagnation detection (default: 10).
- max_norm (float, optional): Maximum norm for gradient clipping (default: 1.0).
- min_lr (float, optional): Minimum allowable learning rate (default: 1e-7).
- max_lr (float, optional): Maximum allowable learning rate (default: 1).

Important Methods:
- adjust_learning_rate(current_loss): Adjusts the learning rate based on the current loss, overshooting, and stagnation conditions.
- step(closure): Performs a single optimization step. If a closure is provided, it is executed to reevaluate the model and return the loss.

This optimizer is particularly useful for scenarios where the training process experiences frequent overshooting or stagnation, 
allowing for more adaptive learning rate adjustments.

Example Usage:
```python
optimizer = AdamZ(model.parameters(), lr=0.01, overshoot_factor=0.5, stagnation_factor=1.2)
```
"""

class AdamZ(optim.Optimizer):
    def __init__(self, params, lr=0.01, betas=(0.9, 0.999), eps=1e-8, overshoot_factor=0.5, stagnation_factor=1.2, stagnation_threshold=0.2, patience=100, stagnation_period=10, max_norm=1.0, min_lr=0.0000001, max_lr=1):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(AdamZ, self).__init__(params, defaults)
        self.overshoot_factor = overshoot_factor
        self.stagnation_factor = stagnation_factor
        self.patience = patience
        self.loss_history = deque(maxlen=patience) 
        self.stagnation_history = deque(maxlen=min(stagnation_period, patience))
        self.stagnation_threshold = stagnation_threshold
        self.max_norm = max_norm
        self.min_lr = min_lr
        self.max_lr = max_lr

    def adjust_learning_rate(self, current_loss):
        self.loss_history.append(current_loss)
        self.stagnation_history.append(current_loss)
        
        for group in self.param_groups:

            # Check for overshooting
            if len(self.loss_history) >= self.patience and current_loss >= max(self.loss_history):
                #print("Overshooting detected. Reducing learning rate.")
                group['lr'] *= self.overshoot_factor
    
    
            # Check for stagnation
            if len(self.loss_history) >= self.patience and np.std(self.stagnation_history) < self.stagnation_threshold * np.std(self.loss_history):
                #print("Stagnation detected. Increasing learning rate.")
                group['lr'] *= self.stagnation_factor
    
    
            # Ensure learning rate is within bounds
            group['lr'] = max(self.min_lr, min(group['lr'], self.max_lr))


    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        if loss is not None:
            self.adjust_learning_rate(loss.item())
            
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.param_groups[0]['params'], self.max_norm)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * (bias_correction2 ** 0.5) / bias_correction1
                         
                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss