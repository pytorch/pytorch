from collections import defaultdict

import torch
from torch import Tensor

from ._adafactor import Adafactor as _Adafactor
from ._adafactor import adafactor


class Adafactor(_Adafactor):
    @torch.no_grad()
    def step(self, closure=None):
        self._accelerator_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad: list[Tensor] = []
            grads: list[Tensor] = []
            row_vars: list[Tensor | None] = []
            col_vars: list[Tensor | None] = []
            variances: list[Tensor | None] = []
            state_steps: list[Tensor] = []
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

            if not params_with_grad:
                continue

            dtype = params_with_grad[0].dtype
            if all(param.dtype == dtype for param in params_with_grad[1:]):
                dtype_eps1 = eps1
                if dtype_eps1 is None:
                    dtype_eps1 = torch.finfo(dtype).eps
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
                    eps1=dtype_eps1,
                    eps2=eps2,
                    foreach=group["foreach"],
                    maximize=group["maximize"],
                    grad_scale=getattr(self, "grad_scale", None),
                    found_inf=getattr(self, "found_inf", None),
                    has_complex=has_complex,
                )
                continue

            by_dtype = defaultdict(list)
            for i, param in enumerate(params_with_grad):
                by_dtype[param.dtype].append(i)

            for dtype, indices in by_dtype.items():
                dtype_eps1 = eps1
                if dtype_eps1 is None:
                    dtype_eps1 = torch.finfo(dtype).eps

                adafactor(
                    [params_with_grad[i] for i in indices],
                    [grads[i] for i in indices],
                    [row_vars[i] for i in indices],
                    [col_vars[i] for i in indices],
                    [variances[i] for i in indices],
                    [state_steps[i] for i in indices],
                    d=group["d"],
                    lr=group["lr"],
                    beta2_decay=group["beta2_decay"],
                    weight_decay=group["weight_decay"],
                    eps1=dtype_eps1,
                    eps2=eps2,
                    foreach=group["foreach"],
                    maximize=group["maximize"],
                    grad_scale=getattr(self, "grad_scale", None),
                    found_inf=getattr(self, "found_inf", None),
                    has_complex=has_complex,
                )

        return loss


__all__ = ["Adafactor"]
