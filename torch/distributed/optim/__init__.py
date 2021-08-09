"""
:mod:`torch.distributed.optim` exposes DistributedOptimizer, which takes a list
of remote parameters (:class:`~torch.distributed.rpc.RRef`) and runs the
optimizer locally on the workers where the parameters live.  The distributed
optimizer can use any of the local optimizer :ref:`optimizer-algorithms` to
apply the gradients on each worker.
"""
import torch
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

if hasattr(torch._C, '_rpc_init'):
    from .optimizer import DistributedOptimizer

from .post_localSGD_optimizer import PostLocalSGDOptimizer
from .zero_redundancy_optimizer import ZeroRedundancyOptimizer
