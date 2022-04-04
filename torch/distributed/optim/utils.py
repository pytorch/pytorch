from typing import Type
from torch import optim
from .functional_adagrad import _FunctionalAdagrad
from .functional_adam import _FunctionalAdam
from .functional_adamw import _FunctionalAdamW
from .functional_sgd import _FunctionalSGD
from .functional_adadelta import _FunctionalAdadelta
from .functional_rmsprop import _FunctionalRMSprop
from .functional_rprop import _FunctionalRprop
from .functional_adamax import _FunctionalAdamax

# dict to map a user passed in optimizer_class to a functional
# optimizer class if we have already defined inside the
# distributed.optim package, this is so that we hide the
# functional optimizer to user and still provide the same API.
functional_optim_map = {
    optim.Adagrad: _FunctionalAdagrad,
    optim.Adam: _FunctionalAdam,
    optim.AdamW: _FunctionalAdamW,
    optim.SGD: _FunctionalSGD,
    optim.Adadelta: _FunctionalAdadelta,
    optim.RMSprop: _FunctionalRMSprop,
    optim.Rprop: _FunctionalRprop,
    optim.Adamax: _FunctionalAdamax,
}

def as_functional_optim(optim_cls: Type, *args, **kwargs):
    try:
        functional_cls = functional_optim_map[optim_cls]
    except KeyError:
        raise ValueError(f"Optimizer {optim_cls} does not have a functional counterpart!")

    return _create_functional_optim(functional_cls, *args, **kwargs)

def _create_functional_optim(functional_optim_cls: Type, *args, **kwargs):
    return functional_optim_cls(
        [],
        *args,
        **kwargs,
        _allow_empty_param_list=True,
    )
