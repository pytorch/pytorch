from .optimizer import Optimizer
from .adam import Adam


class LazyAdam(Adam):
    r"""Implements lazy version of Adam algorithm suitable for sparse tensors.

    In this variant, only moments that show up in the gradient get updated, and
    only those portions of the gradient get applied to the parameters.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    """

    def get_update(self, p, **kwargs):
        return Optimizer.get_update(self, p, **kwargs)

    def get_sparse_update(self, p, betas=(.9, .999), eps=1e-8, weight_decay=0, amsgrad=False, **_):
        if amsgrad:
            raise NotImplementedError("AMSGrad not implemented for sparse gradients")
        if weight_decay > 0:
            raise RuntimeError("weight_decay option is not compatible with sparse gradients")

        grad = p.grad.coalesce()  # the update is non-linear so indices must be unique
        state = self.state[p]

        grad_indices = grad._indices()
        grad_values = grad._values()
        size = grad.size()

        def make_sparse(values):
            constructor = grad.new
            if grad_indices.dim() == 0 or values.dim() == 0:
                return constructor().resize_as_(grad)
            return constructor(grad_indices, values, size)

        state['step'] += 1
        beta1, beta2 = betas
        bias_corr1 = 1 - beta1 ** state['step']
        bias_corr2 = 1 - beta2 ** state['step']

        # Decay the first and second moment running average coefficient
        #      old <- b * old + (1 - b) * new
        # <==> old += (1 - b) * (new - old)
        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        old_exp_avg_values = exp_avg.sparse_mask(grad)._values()
        exp_avg_update_values = grad_values.sub(old_exp_avg_values).mul_(1 - beta1)
        exp_avg.add_(make_sparse(exp_avg_update_values))
        old_exp_avg_sq_values = exp_avg_sq.sparse_mask(grad)._values()
        exp_avg_sq_update_values = grad_values.pow(2).sub_(old_exp_avg_sq_values).mul_(1 - beta2)
        exp_avg_sq.add_(make_sparse(exp_avg_sq_update_values))

        # Dense addition again is intended, avoiding another sparse_mask
        mean = exp_avg_update_values.add_(old_exp_avg_values) / bias_corr1
        var = exp_avg_sq_update_values.add_(old_exp_avg_sq_values) / bias_corr2
        return make_sparse(mean.div_(var.sqrt_().add_(eps)))