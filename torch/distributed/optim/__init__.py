"""
:mod:`torch.distributed.optim` exposes DistributedOptimizer, which takes a list
of remote parameters (:class:`~torch.distributed.rpc.RRef`) and runs the
optimizer locally on the workers where the parameters live.  The distributed
optimizer can use any of the local optimizer :ref:`optimizer-algorithms` to
apply the gradients on each worker.
"""
import torch
from torch import optim

from .apply_optimizer_in_backward import (
    _apply_optimizer_in_backward,
    _get_in_backward_optimizers,
)
from .functional_adadelta import _FunctionalAdadelta

from .functional_adagrad import _FunctionalAdagrad
from .functional_adam import _FunctionalAdam
from .functional_adamax import _FunctionalAdamax
from .functional_adamw import _FunctionalAdamW
from .functional_rmsprop import _FunctionalRMSprop
from .functional_rprop import _FunctionalRprop
from .functional_sgd import _FunctionalSGD
from .named_optimizer import _NamedOptimizer
from .utils import as_functional_optim


# DistributedOptimizer imports torch.distributed.rpc names, so gate availability
# based on RPC being available.
if hasattr(torch._C, "_rpc_init"):
    from .optimizer import DistributedOptimizer

from .post_localSGD_optimizer import PostLocalSGDOptimizer
from .zero_redundancy_optimizer import ZeroRedundancyOptimizer
