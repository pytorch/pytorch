import torch
from . import _functional as F
from .optimizer import _maximize_doc, Optimizer

__all__ = ["SparseAdam"]


class SparseAdam(Optimizer):
    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, maximize: bool = False
    ):
        if not 0.0 < lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 < eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, eps=eps, maximize=maximize)
        super().__init__(params, defaults)

        sparse_params = []
        complex_params = []
        for index, param_group in enumerate(self.param_groups):
            assert isinstance(
                param_group, dict
            ), f"param_groups must be a list of dicts, but got {type(param_group)}"
            # given param group, convert given params to a list first before iterating
            for d_index, d_param in enumerate(param_group["params"]):
                if d_param.is_sparse:
                    sparse_params.append([index, d_index])
                if d_param.is_complex():
                    complex_params.append([index, d_index])
        if sparse_params:
            raise ValueError(
                f"Sparse params at indices {sparse_params}: SparseAdam requires dense parameter tensors"
            )
        if complex_params:
            raise ValueError(
                f"Complex params at indices {complex_params}: SparseAdam does not support complex parameters"
            )

    @torch.no_grad()
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
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_steps = []
            eps = group["eps"]
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            maximize = group.get("maximize", False)

            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if not p.grad.is_sparse:
                        raise RuntimeError(
                            "SparseAdam does not support dense gradients, please consider Adam instead"
                        )
                    grads.append(p.grad)

                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state["step"] = 0
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                    exp_avgs.append(state["exp_avg"])
                    exp_avg_sqs.append(state["exp_avg_sq"])

                    # update the steps for each param group update
                    state["step"] += 1
                    # record the step after step update
                    state_steps.append(state["step"])

            F.sparse_adam(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                eps=group["eps"],
                maximize=maximize,
            )

        return loss


SparseAdam.__doc__ = rf"""SparseAdam implements a masked version of the Adam algorithm
    suitable for sparse gradients. Currently, due to implementation constraints (explained
    below), SparseAdam is only intended for a narrow subset of use cases, specifically
    parameters of a dense layout with gradients of a sparse layout. This occurs in a
    special case where the module backwards produces grads already in a sparse layout.
    One example NN module that behaves as such is ``nn.Embedding(sparse=True)``.

    SparseAdam approximates the Adam algorithm by masking out the parameter and moment
    updates corresponding to the zero values in the gradients. Whereas the Adam algorithm
    will update the first moment, the second moment, and the parameters based on all values
    of the gradients, SparseAdam only updates the moments and parameters corresponding
    to the non-zero values of the gradients.

    A simplified way of thinking about the `intended` implementation is as such:

    1. Create a mask of the non-zero values in the sparse gradients. For example,
       if your gradient looks like [0, 5, 0, 0, 9], the mask would be [0, 1, 0, 0, 1].
    2. Apply this mask over the running moments and do computation on only the
       non-zero values.
    3. Apply this mask over the parameters and only apply an update on non-zero values.

    In actuality, we use sparse layout Tensors to optimize this approximation, which means the
    more gradients that are masked by not being materialized, the more performant the optimization.
    Since we rely on using sparse layout tensors, we infer that any materialized value in the
    sparse layout is non-zero and we do NOT actually verify that all values are not zero!
    It is important to not conflate a semantically sparse tensor (a tensor where many
    of its values are zeros) with a sparse layout tensor (a tensor where ``.is_sparse``
    returns ``True``). The SparseAdam approximation is intended for `semantically` sparse
    tensors and the sparse layout is only a implementation detail. A clearer implementation
    would be to use MaskedTensors, but those are experimental.


    .. note::

        If you suspect your gradients are semantically sparse (but do not have sparse
        layout), this variant may not be the best for you. Ideally, you want to avoid
        materializing anything that is suspected to be sparse in the first place, since
        needing to convert all your grads from dense layout to sparse layout may outweigh
        the performance gain. Here, using Adam may be the best alternative, unless you
        can easily rig up your module to output sparse grads similar to
        ``nn.Embedding(sparse=True)``. If you insist on converting your grads, you can do
        so by manually overriding your parameters' ``.grad`` fields with their sparse
        equivalents before calling ``.step()``.


    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        {_maximize_doc}

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980

    """
