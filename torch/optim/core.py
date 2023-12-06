from typing import List, Optional, Union, Tuple
from math import exp

import torch
from torch import Tensor
from .optimizer import (Optimizer, ParamsT, _use_grad_for_differentiable,
                        _default_to_fused_or_foreach, _differentiable_doc, _foreach_doc,
                        _maximize_doc, _view_as_real)

__all__ = ['CoRe', 'core']


class CoRe(Optimizer):
    def __init__(self,
                 params: ParamsT,
                 lr: float = 1e-3,
                 betas: Tuple[float, float, float, float] = (0.7375, 0.8125, 250.0, 0.99),
                 eps: float = 1e-8,
                 etas: Tuple[float, float] = (0.7375, 1.2),
                 step_sizes: Tuple[float, float] = (1e-6, 1e-2),
                 weight_decay: Union[float, list] = 0.1,
                 score_history: int = 0,
                 frozen: Union[int, list] = 0,
                 *,
                 foreach: Optional[bool] = None,
                 maximize: bool = False,
                 differentiable: bool = False):
        if lr <= 0.0:
            raise ValueError(f'Invalid learning rate: {lr}')
        if (
            not 0.0 <= betas[0] < 1.0
            or not 0.0 <= betas[1] < 1.0
            or betas[2] <= 0.0
            or betas[3] >= 1.0
        ):
            raise ValueError(
                f'Invalid beta values: {betas[0]}, {betas[1]}, {betas[2]}, {betas[3]}')
        if eps < 0.0:
            raise ValueError(f'Invalid epsilon value: {eps}')
        if not 0.0 < etas[0] <= 1.0 <= etas[1]:
            raise ValueError(f'Invalid eta values: {etas[0]}, {etas[1]}')
        if not 0.0 < step_sizes[0] <= step_sizes[1]:
            raise ValueError(f'Invalid min and max step sizes: {step_sizes[0]}, {step_sizes[1]}')
        if isinstance(weight_decay, float):
            weight_decay = [weight_decay]
        elif not isinstance(weight_decay, list):
            raise ValueError(f'Invalid weight decay: {weight_decay}')
        if score_history < 0:
            raise ValueError(f'Invalid score history: {score_history}')
        if isinstance(frozen, int):
            frozen = [frozen]
        elif not isinstance(frozen, list):
            raise ValueError(f'Invalid frozen: {frozen}')

        defaults = {'lr': lr,
                    'betas': betas,
                    'eps': eps,
                    'etas': etas,
                    'step_sizes': step_sizes,
                    'weight_decay': weight_decay,
                    'score_history': score_history,
                    'frozen': frozen,
                    'foreach': foreach,
                    'maximize': maximize,
                    'differentiable': differentiable}
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('foreach', None)
            group.setdefault('maximize', False)
            group.setdefault('differentiable', False)

    def _init_group(self,
                    group,
                    params,
                    grads,
                    prevs_1,
                    prevs_2,
                    step_sizes,
                    scores,
                    steps):
        has_complex = False
        for p in group['params']:
            if p.grad is None:
                continue
            has_complex |= torch.is_complex(p)
            params.append(p)
            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError('CoRe does not support sparse gradients')

            grads.append(grad)
            state = self.state[p]

            # State initialization
            if len(state) == 0:
                state['step'] = 0
                state['prev_1'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['prev_2'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                if p.dtype.is_complex:
                    # Complex Number should be as if they are two independent real numbers.
                    # Hence the step_size shouldn't be zero for imaginary part.
                    state['step_size'] = grad.new().resize_as_(grad).fill_(
                        complex(group['lr'], group['lr']))
                    state['score'] = grad.new().resize_as_(grad).fill_(
                        complex(0.0, 0.0))
                else:
                    state['step_size'] = grad.new().resize_as_(grad).fill_(group['lr'])
                    state['score'] = grad.new().resize_as_(grad).fill_(0.0)

            prevs_1.append(state['prev_1'])
            prevs_2.append(state['prev_2'])
            step_sizes.append(state['step_size'])
            scores.append(state['score'])
            steps.append(state['step'])

            state['step'] += 1
        return has_complex

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = []
            grads = []
            prevs_1 = []
            prevs_2 = []
            step_sizes = []
            scores = []
            steps = []
            beta_1_initial, beta_1_final, beta_1_step, beta_2 = group['betas']
            eps = group['eps']
            eta_minus, eta_plus = group['etas']
            step_size_min, step_size_max = group['step_sizes']
            weight_decay = group['weight_decay']
            score_history = group['score_history']
            frozen = group['frozen']
            foreach = group['foreach']
            maximize = group['maximize']
            differentiable = group['differentiable']

            has_complex = self._init_group(group,
                                           params,
                                           grads,
                                           prevs_1,
                                           prevs_2,
                                           step_sizes,
                                           scores,
                                           steps)

            core(params,
                 grads,
                 prevs_1,
                 prevs_2,
                 step_sizes,
                 beta_1_initial=beta_1_initial,
                 beta_1_final=beta_1_final,
                 beta_1_step=beta_1_step,
                 beta_2=beta_2,
                 eps=eps,
                 eta_minus=eta_minus,
                 eta_plus=eta_plus,
                 step_size_min=step_size_min,
                 step_size_max=step_size_max,
                 weight_decay=weight_decay,
                 scores=scores,
                 score_history=score_history,
                 frozen=frozen,
                 steps=steps,
                 foreach=foreach,
                 maximize=maximize,
                 differentiable=differentiable,
                 has_complex=has_complex)

        return loss


CoRe.__doc__ = r"""Implements the Continual Resilient (CoRe) optimizer.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt} \\
            &\textbf{input} : \theta\ \text{(params)},\ f(\theta)\ \text{(objective)},
                           \ \beta_1^\mathrm{a},\beta_1^\mathrm{b},\beta_1^\mathrm{c},\beta_2
                           \ \text{(betas)},\ \epsilon\ \text{(eps)}, \\
            &\hspace{13mm} \eta_-,\eta_+\ \text{(etas)},\ s_\mathrm{min},s_\mathrm{max}
                           \ \text{(step sizes)},\ d\ \text{(weight decay)}, \\
            &\hspace{13mm} t_\mathrm{hist}\ \text{(score history)},\ n_\mathrm{frozen}\ \text{(frozen)} \\
            &\textbf{initialize} : s_0 \leftarrow \text{lr},\ g_0 \leftarrow 0,\ h_0 \leftarrow 0,\ S_0 \leftarrow 0 \\
            &\rule{110mm}{0.4pt} \\
            &\textbf{for}\ t=1\ \textbf{to}\ \ldots\ \textbf{do} \\
            &\hspace{5mm} G_t \leftarrow \nabla_{\theta}f_t(\theta_{t-1})\\
            &\hspace{5mm} \textbf{if}\ \text{maximize} \\
            &\hspace{10mm} G_t \leftarrow -G_t\\
            &\hspace{5mm} \beta_{1,t} \leftarrow \beta_1^\mathrm{b} + (\beta_1^\mathrm{a}-\beta_1^\mathrm{b})\,
                          \mathrm{exp}\{-[(t-1)/\beta_1^\mathrm{c}]^2\}\\
            &\hspace{5mm} g_t \leftarrow \beta_{1,t} g_{t-1} + (1-\beta_{1,t}) G_t \\
            &\hspace{5mm} \textbf{if}\ \beta_2 \geq 0 \\
            &\hspace{10mm} h_t \leftarrow \beta_2 h_{t-1} + (1-\beta_2) G_t^2 \\
            &\hspace{5mm} P_t \leftarrow 1 \\
            &\hspace{5mm} \textbf{if}\ t_\mathrm{hist}>0 \land t>t_\mathrm{hist}
                          \land S_{t-1}\ \mathrm{top}\text{-}n_\mathrm{frozen}\ \mathrm{in}\ \mathbf{S}_{t-1}\\
            &\hspace{10mm} P_t \leftarrow 0 \\
            &\hspace{5mm} \textbf{if}\ g_{t-1} g_t P_t > 0 \\
            &\hspace{10mm} s_t \leftarrow \mathrm{min}(\eta_+ s_{t-1}, s_\mathrm{max}) \\
            &\hspace{5mm} \textbf{else if}\ g_{t-1} g_t P_t < 0 \\
            &\hspace{10mm} s_t \leftarrow \mathrm{max}(\eta_- s_{t-1}, s_\mathrm{min}) \\
            &\hspace{5mm} \textbf{else} \\
            &\hspace{10mm} s_t \leftarrow s_{t-1} \\
            &\hspace{5mm} \textbf{if}\ \beta_2 \geq 0 \\
            &\hspace{10mm} u_t \leftarrow g_t / (1-\beta_{1,t}^t) / \{[h_t/(1-\beta_2^t)]^{0.5}+\epsilon\} \\
            &\hspace{5mm} \textbf{else} \\
            &\hspace{10mm} u_t \leftarrow \mathrm{sgn}(g_t) \\
            &\hspace{5mm} \textbf{if}\ t_\mathrm{hist} > 0\\
            &\hspace{10mm} \textbf{if}\ t \leq t_\mathrm{hist}\\
            &\hspace{15mm} S_t \leftarrow S_{t-1} + t_\mathrm{hist}^{-1} g_t u_t P_t s_t\\
            &\hspace{10mm} \textbf{else} \\
            &\hspace{15mm} S_t \leftarrow (1-t_\mathrm{hist}^{-1}) S_\xi^{\tau-1}
                           + t_\mathrm{hist}^{-1} g_t u_t P_t s_t\\
            &\hspace{5mm}\theta_t \leftarrow (1-d|u_t|P_ts_t) \theta_{t-1} - u_t P_t s_t \\
            &\rule{110mm}{0.4pt} \\[-1.ex]
            &\bf{return}\ \theta_t \\[-1.ex]
            &\rule{110mm}{0.4pt} \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to the papers
    `Lifelong Machine Learning Potentials <https://doi.org/10.1021/acs.jctc.3c00279>`_
    and `CoRe Optimizer: An All-in-One Solution for Machine Learning <https://arxiv.org/abs/2307.15663>`_.
    """ + fr"""
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float, float, float], optional): coefficients
            (beta1a, beta1b, beta1c, beta2) used for computing running averages of
            gradient and its square (default: (0.7375, 0.8125, 250.0, 0.99))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        etas (Tuple[float, float], optional): pair of (etaminus, etaplus) that
            are multiplicative increase and decrease factors (default: (0.7375, 1.2))
        step_sizes (Tuple[float, float], optional): a pair of minimal and
            maximal allowed step sizes (default: (1e-6, 1e-2))
        weight_decay (float or List[float], optional): weight decay for all
            parameters or list of weight decays for parameter groups (default: 0.1)
        score_history (int, optional): number of epochs to build the score history
            before applying plasticity factors (default: 0)
        frozen (int or List[int], optional): number of parameters frozen by the
            plasticity factors (individual specifications for each parameter group
            is possible by a list (default: 0)
        {_foreach_doc}
        {_maximize_doc}
        {_differentiable_doc}
    """


def core(params: List[Tensor],
         grads: List[Tensor],
         prevs_1: List[Tensor],
         prevs_2: List[Tensor],
         step_sizes: List[Tensor],
         # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
         # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
         foreach: Optional[bool] = None,
         maximize: bool = False,
         differentiable: bool = False,
         has_complex: bool = False,
         *,
         beta_1_initial: float,
         beta_1_final: float,
         beta_1_step: float,
         beta_2: float,
         eps: float,
         eta_minus: float,
         eta_plus: float,
         step_size_min: float,
         step_size_max: float,
         weight_decay: List[float],
         scores: List[Tensor],
         score_history: int,
         frozen: List[int],
         steps: List[int]):
    r"""Functional API that performs core algorithm computation.

    See :class:`~torch.optim.CoRe` for details.
    """

    if foreach is None:
        _, foreach = _default_to_fused_or_foreach(params, differentiable, use_fused=False)

    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_core
    else:
        func = _single_tensor_core

    func(params,
         grads,
         prevs_1,
         prevs_2,
         step_sizes,
         beta_1_initial=beta_1_initial,
         beta_1_final=beta_1_final,
         beta_1_step=beta_1_step,
         beta_2=beta_2,
         eps=eps,
         eta_minus=eta_minus,
         eta_plus=eta_plus,
         step_size_min=step_size_min,
         step_size_max=step_size_max,
         weight_decay=weight_decay,
         scores=scores,
         score_history=score_history,
         frozen=frozen,
         steps=steps,
         maximize=maximize,
         differentiable=differentiable,
         has_complex=has_complex)


def _single_tensor_core(params: List[Tensor],
                        grads: List[Tensor],
                        prevs_1: List[Tensor],
                        prevs_2: List[Tensor],
                        step_sizes: List[Tensor],
                        *,
                        beta_1_initial: float,
                        beta_1_final: float,
                        beta_1_step: float,
                        beta_2: float,
                        eps: float,
                        eta_minus: float,
                        eta_plus: float,
                        step_size_min: float,
                        step_size_max: float,
                        weight_decay: List[float],
                        scores: List[Tensor],
                        score_history: int,
                        frozen: List[int],
                        steps: List[int],
                        maximize: bool,
                        differentiable: bool,
                        has_complex: bool):

    n_frozen = len(frozen)
    n_weight_decay = len(weight_decay)
    for i, param in enumerate(params):
        grad = grads[i]
        prev_1 = prevs_1[i]
        prev_2 = prevs_2[i]
        step_size = step_sizes[i]
        score = scores[i]
        step = steps[i]

        # handle complex params
        if torch.is_complex(param):
            param = torch.view_as_real(param)
            grad = torch.view_as_real(grad)
            prev_1 = torch.view_as_real(prev_1)
            prev_2 = torch.view_as_real(prev_2)
            step_size = torch.view_as_real(step_size)
            score = torch.view_as_real(score)

        # for maximization invert gradient signs
        if maximize:
            grad = -grad

        # exponential moving average of squared gradients
        if beta_2 >= 0:
            prev_2.mul_(beta_2)
            prev_2.addcmul_(grad, grad, value=1.0 - beta_2)

        # adjust fractions of previous gradient information in current gradients
        beta_1 = (beta_1_final + (beta_1_initial - beta_1_final)
                  * exp(-(step / beta_1_step)**2))

        # exponential moving average of gradients
        grad = grad.clone(memory_format=torch.preserve_format)
        grad.lerp_(prev_1, beta_1)

        # stability-plasticity balance
        if score_history > 0:
            plasticity = score.new().resize_as_(score).fill_(1.0)
            if step >= score_history:
                if frozen[i % n_frozen] > 0:
                    score_max = float(torch.topk(torch.flatten(score), frozen[i % n_frozen])
                                      .values[frozen[i % n_frozen] - 1])
                    plasticity[score.ge(0.999999 * score_max)] = 0
                    score.mul_(1.0 - 1.0 / score_history)

        # determine step size updates
        if differentiable:
            step_size_update = grad.mul(prev_1.clone())
        else:
            step_size_update = grad.mul(prev_1)
        if score_history > 0:
            step_size_update.mul_(plasticity)
        step_size_update[step_size_update.gt(0)] = eta_plus
        step_size_update[step_size_update.lt(0)] = eta_minus
        step_size_update[step_size_update.eq(0)] = 1.0

        # update step sizes
        step_size.mul_(step_size_update)
        step_size.clamp_(step_size_min, step_size_max)

        # adjust parameter updates
        if beta_2 >= 0:
            param_update = torch.div(grad / (1.0 - beta_1**(step + 1)),
                                     torch.add(torch.sqrt(prev_2 / (1.0 - beta_2**(step + 1))), eps))
        else:
            param_update = grad.sign()
        param_update.mul_(step_size)
        if score_history > 0:
            param_update.mul_(plasticity)
            score.addcmul_(grad, param_update, value=1.0 / score_history)

        # weight decay
        if differentiable:
            param.add_(-weight_decay[i % n_weight_decay] * param_update.abs() * param.clone())
        else:
            param.add_(-weight_decay[i % n_weight_decay] * param_update.abs() * param)

        # update parameters
        param.add_(param_update, alpha=-1.0)

        # update previous gradients
        prev_1.copy_(grad)


def _multi_tensor_core(params: List[Tensor],
                       grads: List[Tensor],
                       prevs_1: List[Tensor],
                       prevs_2: List[Tensor],
                       step_sizes: List[Tensor],
                       *,
                       beta_1_initial: float,
                       beta_1_final: float,
                       beta_1_step: float,
                       beta_2: float,
                       eps: float,
                       eta_minus: float,
                       eta_plus: float,
                       step_size_min: float,
                       step_size_max: float,
                       weight_decay: List[float],
                       scores: List[Tensor],
                       score_history: int,
                       frozen: List[int],
                       steps: List[int],
                       maximize: bool,
                       differentiable: bool,
                       has_complex: bool):

    if len(params) == 0:
        return

    assert not differentiable, "_foreach ops don't support autograd"

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype([
        params, grads, prevs_1, prevs_2, step_sizes, scores])
    for ((
        grouped_params,
        grouped_grads,
        grouped_prevs_1,
        grouped_prevs_2,
        grouped_step_sizes,
        grouped_scores
    ), _) in grouped_tensors.values():

        # handle complex params
        if has_complex:
            _view_as_real(grouped_params,
                          grouped_grads,
                          grouped_prevs_1,
                          grouped_prevs_2,
                          grouped_step_sizes,
                          grouped_scores)

        # for maximization invert gradient signs
        for i in range(len(grouped_grads)):
            grouped_grads[i] = grouped_grads[i].clone(memory_format=torch.preserve_format)
        if maximize:
            torch._foreach_neg_(grouped_grads)

        # exponential moving average of squared gradients
        if beta_2 >= 0:
            torch._foreach_mul_(grouped_prevs_2, beta_2)
            torch._foreach_addcmul_(grouped_prevs_2, grouped_grads, grouped_grads, value=1.0 - beta_2)

        # adjust fractions of previous gradient information in current gradients
        betas_1 = [grouped_grads[i].new().resize_as_(grouped_grads[i]).fill_(
            beta_1_final + (beta_1_initial - beta_1_final)
            * exp(-(steps[i] / beta_1_step)**2)) for i in range(len(steps))]

        # exponential moving average of gradients
        torch._foreach_lerp_(grouped_grads, grouped_prevs_1, betas_1)

        # stability-plasticity balance
        if score_history > 0:
            n_frozen = len(frozen)
            plasticities = [score.new().resize_as_(score).fill_(1.0) for score in grouped_scores]
            for i in range(len(grouped_scores)):
                if steps[i] >= score_history:
                    if frozen[i % n_frozen] > 0:
                        score_max = float(torch.topk(torch.flatten(grouped_scores[i]), frozen[i % n_frozen])
                                          .values[frozen[i % n_frozen] - 1])
                        plasticities[i][grouped_scores[i].ge(0.999999 * score_max)] = 0
                        grouped_scores[i].mul_(1.0 - 1.0 / score_history)

        # determine step size updates
        step_size_updates = torch._foreach_mul(grouped_grads, grouped_prevs_1)
        if score_history > 0:
            torch._foreach_mul_(step_size_updates, plasticities)
        for step_size_update in step_size_updates:
            step_size_update[step_size_update.gt(0)] = eta_plus
            step_size_update[step_size_update.lt(0)] = eta_minus
            step_size_update[step_size_update.eq(0)] = 1.0

        # update step sizes
        torch._foreach_mul_(grouped_step_sizes, step_size_updates)
        for step_size in grouped_step_sizes:
            step_size.clamp_(step_size_min, step_size_max)

        # adjust parameter updates
        if beta_2 >= 0:
            param_updates = [
                torch.div(grouped_grads[i] / (1.0 - betas_1[i]**(steps[i] + 1)),
                          torch.add(torch.sqrt(grouped_prevs_2[i] / (1.0 - beta_2**(steps[i] + 1))), eps))
                for i in range(len(grouped_grads))]
        else:
            param_updates = [grad.sign() for grad in grouped_grads]
        torch._foreach_mul_(param_updates, grouped_step_sizes)
        if score_history > 0:
            torch._foreach_mul_(param_updates, plasticities)
            torch._foreach_addcmul_(grouped_scores, grouped_grads, param_updates, value=1.0 / score_history)

        # weight decay
        n_weight_decay = len(weight_decay)
        params_decay = [-weight_decay[i % n_weight_decay] * param_updates[i].abs() * grouped_params[i]
                        for i in range(len(grouped_step_sizes))]
        torch._foreach_add_(grouped_params, params_decay)

        # update parameters
        torch._foreach_add_(grouped_params, param_updates, alpha=-1.0)

        # update previous gradients
        for i in range(len(grouped_prevs_1)):
            grouped_prevs_1[i].copy_(grouped_grads[i])
