# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor

from .optimizer import (
    _disable_dynamo_if_unsupported,
    _get_scalar_dtype,
    _maximize_doc,
    Optimizer,
    ParamsT,
)


__all__ = ["Adafactor", "adafactor"]


class Adafactor(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-2,
        beta2_decay: float = -0.8,
        eps: Tuple[Optional[float], float] = (None, 1e-3),
        d: float = 1.0,
        weight_decay: float = 0.0,
        *,
        maximize: bool = False,
    ):
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Learning rate should be >= 0 but is: {lr}")
        if not 0.0 >= beta2_decay:
            raise ValueError(f"beta2_decay should be <= 0 but is: {beta2_decay}")
        if eps[0] is not None and not 0.0 <= eps[0]:
            raise ValueError(f"epsilon1 should be >= 0 but is: {eps[0]}")
        if not 0.0 <= eps[1]:
            raise ValueError(f"epsilon2 should be >= 0 but is: {eps[1]}")
        if not 1.0 <= d:
            raise ValueError(f"Clipping threshold d should be >= 1 but is: {d}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"weight_decay should be >= 0 but is: {weight_decay}")
        defaults = dict(
            lr=lr,
            beta2_decay=beta2_decay,
            eps=eps,
            d=d,
            weight_decay=weight_decay,
            maximize=maximize,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            for p in group["params"]:
                p_state = self.state.get(p, [])
                if len(p_state) != 0 and not torch.is_tensor(p_state["step"]):
                    step_val = float(p_state["step"])
                    p_state["step"] = torch.tensor(step_val, dtype=_get_scalar_dtype())

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        row_vars,
        col_vars,
        variances,
        state_steps,
    ):
        for p in group["params"]:
            if p.grad is None:
                continue
            if torch.is_complex(p):
                raise RuntimeError("Adafactor does not support complex parameters")
            if p.grad.is_sparse:
                raise RuntimeError("Adafactor does not support sparse gradients")

            params_with_grad.append(p)
            grads.append(p.grad)

            state = self.state[p]

            # State initialization
            if len(state) == 0:
                # note(crcrpar): Deliberately host `step` on CPU if both capturable and fused are off.
                # This is because kernel launches are costly on CUDA and XLA.
                state["step"] = torch.tensor(0.0, dtype=_get_scalar_dtype())

                if p.grad.dim() > 1:
                    row_shape = list(p.grad.shape)
                    row_shape[-1] = 1
                    # Row factor of variance, NOT the same shape as grads (will be reduced along last dim)
                    state["row_var"] = p.grad.new_zeros(row_shape)

                    col_shape = list(p.grad.shape)
                    col_shape[-2] = 1
                    # Col factor of variance, NOT the same shape as grads (will be reduced along penultimate dim)
                    state["col_var"] = p.grad.new_zeros(col_shape)
                else:
                    state["variance"] = torch.zeros_like(
                        p.grad, memory_format=torch.preserve_format
                    )

            row_vars.append(state.get("row_var", None))
            col_vars.append(state.get("col_var", None))
            variances.append(state.get("variance", None))
            state_steps.append(state["step"])
        return False  # has_complex

    @torch.no_grad()
    def step(self, closure=None):
        r"""Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad: List[Tensor] = []
            grads: List[Tensor] = []
            row_vars: List[Optional[Tensor]] = []
            col_vars: List[Optional[Tensor]] = []
            variances: List[Optional[Tensor]] = []
            state_steps: List[Tensor] = []
            eps1, eps2 = group["eps"]

            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                row_vars,
                col_vars,
                variances,
                state_steps,
            )

            adafactor(
                params_with_grad,
                grads,
                row_vars,
                col_vars,
                variances,
                state_steps,
                d=group["d"],
                lr=group["lr"],
                beta2_decay=group["beta2_decay"],
                weight_decay=group["weight_decay"],
                eps1=eps1,
                eps2=eps2,
                maximize=group["maximize"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
                has_complex=has_complex,
            )

        return loss


Adafactor.__doc__ = (
    r"""Implements Adafactor algorithm.

    .. math::
        \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{(lr)}, \: \tau
                \text{(}\beta_2\text{ decay)}, \: \theta_0 \text{(params)}, \: f(\theta) \text{(objective)},    \\
            &\hspace{15mm}      \: \epsilon_1, \epsilon_2 \text{ (epsilons)}, \: d \text{(clipping threshold)}, \\
            &\hspace{15mm}      \: \lambda \text{(weight decay)},
                \: \textit{maximize}                                                             \\
            &\textbf{initialize} : \: R_0 \leftarrow 0 \text{ (second moment row factor)},       \\
            &\hspace{23mm} \: C_0 \leftarrow 0 \text{ (second moment col factor)},               \\
            &\hspace{23mm} \: \widehat{V}_0 \leftarrow 0 \text{ (second moment for vectors)}     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\

            &\hspace{5mm}\textbf{if} \: \textit{maximize}:                                       \\
            &\hspace{10mm}G_t           \leftarrow   -\nabla_{\theta} f_t (\theta_{t-1})         \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}G_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})          \\
            &\hspace{5mm}\widehat{\beta}_{2_t} \leftarrow 1 - t^{\tau}                           \\
            &\hspace{5mm}\rho_t         \leftarrow min(lr, \frac{1}{\sqrt{t}})                   \\
            &\hspace{5mm}\alpha_t       \leftarrow max(\epsilon_2,
                \text{RMS}(\theta_{t-1}))\rho_t                                                  \\
            &\hspace{5mm}\theta_t       \leftarrow \theta_{t-1} - \gamma \lambda \theta_{t-1}    \\
            &\hspace{5mm}\textbf{if} \: \text{dim}(G_t) > 1:                                     \\
            &\hspace{10mm}R_t           \leftarrow \widehat{\beta}_{2_t}R_{t-1}+
                (1-\widehat{\beta}_{2_t})(G_t \odot G_t) \cdot 1_m                               \\
            &\hspace{10mm}C_t           \leftarrow \widehat{\beta}_{2_t}C_{t-1}+
                (1-\widehat{\beta}_{2_t}) 1^\top_n \cdot (G_t \odot G_t)                         \\
            &\hspace{10mm}\widehat{V}_t \leftarrow
                \frac{R_t \cdot C_t}{max(1^\top_n \cdot R_t, \epsilon_1)}                        \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}\widehat{V}_t \leftarrow \widehat{\beta}_{2_t}\widehat{V}_{t-1}+
                (1-\widehat{\beta}_{2_t}) \cdot (G_t \odot G_t)                                  \\
            &\hspace{5mm}U_t            \leftarrow
                \frac{G_t}{max(\sqrt{\widehat{V}_t}, \epsilon_1)}                                \\
            &\hspace{5mm}\widehat{U}_t  \leftarrow \frac{U_t}{max(1, \frac{\text{RMS}(U_t)}{d})} \\
            &\hspace{5mm}\theta_t       \leftarrow \theta_{t-1} - \alpha_t \widehat{U}_t         \\

            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
        \end{aligned}

    For further details regarding the algorithm we refer to `Adafactor: Adaptive Learning Rates with Sublinear Memory Cost`_.
    """
    + rf"""
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, Tensor, optional): unlike other optimizers, Adafactor does not require a
            learning rate, and Shazeer, Noam, and Mitchell Stern do not use lr at all.
            Deviating from the paper, this implementation uses lr for applying weight
            decay and as the maximum value for relative step size rho_t. Note that in
            the paper, a constant of 0.01 is used as the maximum value for relative
            step size, and so we set 0.01 as the default value. (default: 1e-2)
        beta2_decay (float, optional): the decay rate of beta2. beta2 standardly refers
            to the coefficient used for computing the running average of the gradient
            squared. (default: -0.8)
        eps (Tuple[float, float], optional): epsilon1 is the term added to the denominator
            of the update calculation to improve numerical stability. This use of epsilon1
            deviates from the algorithm written in the paper! See note below for more details.
            epsilon2 is the term used to avoid having too small a weight update when applying
            parameter scaling. (default: (None, 1e-3))
        d (float, optional): the clipping threshold, used to avoid larger-than-desired
            updates.
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        {_maximize_doc}"""
    + r"""
    .. Note::
        The implementation of Adafactor subtly differs from Shazeer, Noam, and Mitchell Stern
        and implementations in some other frameworks with its use of learning rate and
        :math:`\epsilon_1`.

        Regarding the learning rate hyperparameter: Shazeer, Noam, and Mitchell Stern do not
        use lr at all, as the stated algorithm uses :math:`\rho_t` and update clipping to
        affect the step size.

        This implementation allows `lr` to influence the maximum value for :math:`\rho_t`:

        .. math::
            \begin{aligned}
                &\hspace{5mm}\rho_t \leftarrow min(lr, \frac{1}{\sqrt{t}})
            \end{aligned}

        This differs from Shazeer, Noam, and Mitchell Stern, who use a constant of 0.01 as
        the maximum value of :math:`\rho_t`

        .. math::
            \begin{aligned}
                &\hspace{5mm}\rho_t \leftarrow min(0.01, \frac{1}{\sqrt{t}})
            \end{aligned}

        Shazeer, Noam, and Mitchell Stern do not enforce an opinion on how weight decay should
        be computed, and so we use the learning rate as a coefficient for decoupled weight
        decay, similar to what is suggested in `Decoupled Weight Decay Regularization`_.

        Regarding the use of :math:`\epsilon_1`: The implementation attempts to replicate the
        presumed intention of Shazeer, Noam, and Mitchell Stern to use :math:`\epsilon_1` as
        a stabilizing term when the squared gradient becomes small.

        This stabilization can be written as

        .. math::
            \begin{aligned}
                &\hspace{5mm}R_t \leftarrow \widehat{\beta}_{2_t}R_{t-1}+
                    (1-\widehat{\beta}_{2_t})(G_t \odot G_t + 1_n \cdot 1^\top_m) \cdot 1_m          \\
                &\hspace{5mm}C_t \leftarrow \widehat{\beta}_{2_t}C_{t-1}+
                    (1-\widehat{\beta}_{2_t}) 1^\top_n \cdot (G_t \odot G_t + 1_n \cdot 1^\top_m)    \\
                &\hspace{5mm}\widehat{V}_t \leftarrow
                    \frac{R_t \cdot C_t}{max(1^\top_n \cdot R_t, \epsilon_1)}                        \\
                &\hspace{5mm}U_t \leftarrow \frac{G_t}{max(\sqrt{\widehat{V}_t}, \epsilon_1)}        \\
            \end{aligned}

        where the row and column factors of gradient squared :math:`R_t` and :math:`C_t`
        are left alone, and we apply :math:`\epsilon_1` at the final calculation of
        the variance estimate :math:`\widehat{V}_t` and for the update :math:`U_t`.

        This is in contrast to Shazeer, Noam, and Mitchell Stern and other frameworks which
        apply :math:`\epsilon_1` to both row and column factors of the squared gradient, but
        not in the calculations after:

        .. math::
            \begin{aligned}
                &\hspace{5mm}R_t \leftarrow \widehat{\beta}_{2_t}R_{t-1}+
                            (1-\widehat{\beta}_{2_t})(G_t \odot G_t + \epsilon_1 1_n \cdot 1^\top_m) \cdot 1_m          \\
                &\hspace{5mm}C_t \leftarrow \widehat{\beta}_{2_t}C_{t-1}+
                            (1-\widehat{\beta}_{2_t}) 1^\top_n \cdot (G_t \odot G_t + \epsilon_1 1_n \cdot 1^\top_m)    \\
                &\hspace{5mm}\widehat{V}_t \leftarrow \frac{R_t \cdot C_t}{1^\top_n \cdot R_t}                          \\
                &\hspace{5mm}U_t \leftarrow \frac{G_t}{\sqrt{\widehat{V}_t}}                                            \\
            \end{aligned}


    .. _Adafactor\: Adaptive Learning Rates with Sublinear Memory Cost:
        https://arxiv.org/pdf/1804.04235
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    """
)


def _single_tensor_adafactor(
    params: List[Tensor],
    grads: List[Tensor],
    # If grad is 1-dimensional (aka a vector), there is no factorization necessary
    # so row_var and col_var will be None while variance will be filled.
    # Contrarily, for a grad with multiple dimensions, we will factor along the last
    # 2 dimensions, and so row_var and col_var will be filled and variance will be None.
    row_vars: List[Optional[Tensor]],
    col_vars: List[Optional[Tensor]],
    variances: List[Optional[Tensor]],
    state_steps: List[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    d: float,
    lr: Union[Tensor, float],
    beta2_decay: float,
    weight_decay: float,
    eps1: Optional[float],
    eps2: float,
    maximize: bool,
    has_complex: bool,
):
    assert (
        grad_scale is None and found_inf is None
    ), "Grad scaling should occur outside of optimizer.step()"

    if torch.jit.is_scripting():
        # this assert is due to JIT being dumb and not realizing that the ops below
        # have overloads to handle both float and Tensor lrs, so we just assert it's
        # a float since most people using JIT are using floats
        assert isinstance(lr, float)

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        step_t = state_steps[i]
        row_var = row_vars[i]
        col_var = col_vars[i]
        variance = variances[i]
        if eps1 is None:
            eps1 = torch.finfo(param.dtype).eps

        # update step
        step_t += 1
        step_float = step_t.item()

        beta2_t = 1 - step_float**beta2_decay
        rho_t = min(lr, 1 / (step_float**0.5))
        alpha = max(eps2, param.norm(2).item() / (param.numel() ** 0.5)) * rho_t

        # Perform stepweight decay
        if weight_decay != 0:
            param.mul_(1 - lr * weight_decay)

        if grad.dim() > 1:
            assert (
                row_var is not None and col_var is not None
            ), "row_var and col_var should be defined when grad is multidimensional"
            # same as (g * g).mean(dim=-1) w/o materializing an intermediate size g
            row_mean = (
                torch.norm(grad, dim=-1, keepdim=True).square_().div_(grad.size(-1))
            )
            row_var.lerp_(row_mean, 1 - beta2_t)
            # same as (g * g).mean(dim=-2) w/o materializing an intermediate size g
            col_mean = (
                torch.norm(grad, dim=-2, keepdim=True).square_().div_(grad.size(-2))
            )
            col_var.lerp_(col_mean, 1 - beta2_t)
            var_estimate = row_var @ col_var
            var_estimate.div_(row_var.mean(dim=-2, keepdim=True).clamp_(min=eps1))
        else:
            assert (
                variance is not None
            ), "variance should be defined when grad is a vector"
            grad_squared = grad * grad
            variance.lerp_(grad_squared, 1 - beta2_t)
            # avoid writing into variance during update
            var_estimate = variance.clone()

        # square the eps1 as we sqrt after to keep eps1's magnitude
        update = var_estimate.clamp_(min=eps1 * eps1).rsqrt_()
        update.mul_(grad)
        denom = max(1.0, update.norm(2).item() / ((update.numel() ** 0.5) * d))
        param.add_(update, alpha=-alpha / denom)


@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_adafactor)
def adafactor(
    params: List[Tensor],
    grads: List[Tensor],
    row_vars: List[Optional[Tensor]],
    col_vars: List[Optional[Tensor]],
    variances: List[Optional[Tensor]],
    state_steps: List[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    grad_scale: Optional[Tensor] = None,
    found_inf: Optional[Tensor] = None,
    has_complex: bool = False,
    *,
    d: float,
    lr: Union[float, Tensor],
    beta2_decay: float,
    weight_decay: float,
    eps1: float,
    eps2: float,
    maximize: bool,
):
    r"""Functional API that performs Adafactor algorithm computation.

    See :class:`~torch.optim.Adafactor` for details.
    """
    if not torch._utils.is_compiling() and not all(
        isinstance(t, torch.Tensor) for t in state_steps
    ):
        raise RuntimeError(
            "`state_steps` argument must contain a list of singleton tensors"
        )

    func = _single_tensor_adafactor

    func(
        params,
        grads,
        row_vars,
        col_vars,
        variances,
        state_steps,
        d=d,
        lr=lr,
        beta2_decay=beta2_decay,
        weight_decay=weight_decay,
        eps1=eps1,
        eps2=eps2,
        maximize=maximize,
        grad_scale=grad_scale,
        found_inf=found_inf,
        has_complex=has_complex,
    )
