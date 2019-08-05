import torch
import warnings
from .adam import Adam
from .sgd import SGD


def extend_with_decoupled_weight_decay(base_optimizer):
    r"""Extends ```base_optimizer``` with decoupled weight decay.

    This factory function returns a new optimizer class, in which the weight
    decay is decoupled from the optimization steps w.r.t. to the loss function,
    as proposed in Decoupled Weight Decay Regularization by Loshchilov & Hutter:
    https://arxiv.org/abs/1711.05101

    For SGD variants, this simplifies hyperparameter search since it decouples
    the settings of weight decay and learning rate.
    For adaptive gradient algorithms, it regularizes variables with large
    gradients more than L2 regularization would, which was shown to yield better
    training loss and generalization error in the paper above.
    
    .. note:: The weight decay is implicitly scheduled by multiplying it with the
        learning rate.

    .. note:: To prevent copying of the parameters, the weight decay is
        applied before the optimizer update. If the update step relies on
        the value of the parameter (and not just the gradient), this function
        should not be used.

    Example:
        >>> AdamW = extend_with_decoupled_weight_decay(torch.optim.Adam)
        >>> optimizer = AdamW(weight_decay=1e-5, lr=1e-3)
    """
    
    class DecoupledWeightDecayOptimizer(base_optimizer):

        __doc__ = \
        """
        Implements the {0}W algorithm.
            
        I.e., this class implements the update step of {0} with weight
        decay, where the weight decay is decoupled from the optimization
        steps w.r.t. to the loss function, as proposed in Decoupled Weight
        Decay Regularization by  Loshchilov & Hutter:
        https://arxiv.org/abs/1711.05101

        .. note:: The weight decay is implicitly scheduled by multiplying it with the
        learning rate.

        
        Arguments:
            weight_decay (float, optional): decoupled weight decay (default: 1e-2).
            *args: arguments passed to the base optimizer.
            **kwargs: keyword arguments passed to the base optimizer.
        
        Details on the update step can be found in the documentation of the
        wrapped {0} optimizer.
        """.format(base_optimizer.__name__)

        def __init__(self, *args, **kwargs):
            # Remove weight decay from kwargs to prevent L2 decay or unexpected
            # keyword argument error
            wd = kwargs.pop("weight_decay", 1e-2)
            super(DecoupledWeightDecayOptimizer, self).__init__(*args, **kwargs)
            self.defaults["decoupled_weight_decay"] = wd
            # update all param groups to contain the default weight decay
            # by re-adding them
            for group in self.param_groups:
                group.setdefault("decoupled_weight_decay", wd)
                group.setdefault("weight_decay", 0)

        def step(self, closure=None):
            # First perform the weight decay, then execute the optimizer step.
            # This way, we don't need to store the parameter value prior to the
            # update. The drawback is, that the optimizer's step function must
            # not depend on the parameter value but only the gradient.
            # This also means that the closure has to be evaluated beforehand,
            # so it won't work with optimizers that depend on closures.
            loss = None
            if closure is not None:
                loss = closure()
            for group in self.param_groups:
                wd = group['decoupled_weight_decay']
                for p in group['params']:
                    if p.grad is None:
                        continue
                    if p.grad.is_sparse:
                        grad = p.grad.data
                        grad = grad.coalesce()
                        sparse_data = p.data.sparse_mask(grad)._values()
                        sparse_data.mul_(group['lr'] * wd)
                        sparse_data = grad.new(grad._indices(), sparse_data,
                                               grad.size())
                        p.data.sub_(sparse_data)
                    else:
                        p.data.mul_(1 - group['lr'] * wd)
            # We don't pass the closure here as the result would be wrong due
            # to the preceding weight decay.
            return loss or super(DecoupledWeightDecayOptimizer, self).step()

        # copy the documentation string of the base optimizer
        step.__doc__ = base_optimizer.step.__doc__

    return DecoupledWeightDecayOptimizer


# Subclassing is necessary to generate the documentation correctly,
# SGDW = extend_with_decoupled_weight_decay(SGD) would result in `...alias of.`
_SGDW = extend_with_decoupled_weight_decay(SGD)
class SGDW(_SGDW):
    __doc__ = _SGDW.__doc__


_AdamW = extend_with_decoupled_weight_decay(Adam)
class AdamW(_AdamW):
    __doc__ = _AdamW.__doc__
