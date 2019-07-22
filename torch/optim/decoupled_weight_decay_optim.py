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
    
    .. note:: The weight decay is not coupled with the learning rate. If a
        scheduler is used for the learning rate, it is reasonable to schedule
        the weight decay as well.

    .. note:: To prevent copying of the parameters, the weight decay is
        applied before the optimizer update. If the update step relies on
        the value of the parameter (and not just the gradient), this function
        should not be used.

    Example:
        >>> AdamW = extend_with_decoupled_weight_decay(torch.optim.Adam)
        >>> optimizer = AdamW(decoupled_weight_decay=1e-5, lr=1e-3)
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

        .. note:: The weight decay is not coupled with the  learning rate.
            If a scheduler is used for the learning  rate, it is reasonable to
            schedule the weight decay as well.
        
        Arguments:
            decoupled_weight_decay (float, optional): weight decay (default: 0)
            *args: arguments passed to the base optimizer.
            **kwargs: keyword arguments passed to the base optimizer.
        
        Details on the update step can be found in the documentation of the
        wrapped {0} optimizer.
        """.format(base_optimizer.__name__)

        def __init__(self, decoupled_weight_decay, *args, **kwargs):
            super(DecoupledWeightDecayOptimizer, self).__init__(*args, **kwargs)
            if self.defaults["weight_decay"] != 0:
                warnings.warn(
                    "The weight decay parameter of the base optimizer is not "
                    "0. This means the weights will be decayed with l2 loss "
                    "AND decoupled weight decay.", UserWarning)
            self.defaults["decoupled_weight_decay"] = decoupled_weight_decay
            # update all param groups to contain the default weight decay
            # by re-adding them
            for group in self.param_groups:
                group.setdefault("decoupled_weight_decay", decoupled_weight_decay)

        def step(self, closure=None):
            # First perform the weight decay, then execute the optimizer step.
            # This way, we don't need to store the parameter value prior to the
            # update. The drawback is, that the optimizer's step function must
            # not depend on the parameter value but only the gradient.
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        p.data.mul_(1 - group['decoupled_weight_decay'])
            return super(DecoupledWeightDecayOptimizer, self).step(closure)

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
