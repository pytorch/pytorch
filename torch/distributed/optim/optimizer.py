from typing import List, Optional
import logging

import torch.distributed.rpc as rpc
import torch.optim as optim
import torch.jit as jit
import torch.nn as nn
from torch import Tensor
from torch.distributed.rpc import RRef
from .functional_adagrad import _FunctionalAdagrad
from .functional_adam import _FunctionalAdam
from .functional_adamw import _FunctionalAdamW
from .functional_sgd import _FunctionalSGD
from .functional_adadelta import _FunctionalAdadelta
from .functional_rmsprop import _FunctionalRMSprop
from .functional_rprop import _FunctionalRprop
from .functional_adamax import _FunctionalAdamax
import torch.distributed.autograd as dist_autograd


from collections import defaultdict
from threading import Lock

logger = logging.getLogger(__name__)

# XXX: we define a _ScriptModuleOptimizer here to explicitly
# compile the FunctionalOptimizer class into TorchScript
# This is because ScriptClass instance still lives in
# python unless you explicitly compile it as an attribute
# in ScriptModule or pass it to a ScriptFunction
# _ScriptLocalOptimizerInterface serves as a common
# interface type for Optimizer ScriptModules.
#
# TODO (wanchaol): remove this once we added TorchScript
# class reference semantics
@jit.interface
class _ScriptLocalOptimizerInterface(object):
    def step(self, autograd_ctx_id: int) -> None:
        pass

class _ScriptLocalOptimizer(nn.Module):
    # TorchScript does not support multithread concurrent compiling.
    # request_callback might invoke concurrent compiling, so we
    # serialize the compiling with a lock
    compile_lock = Lock()

    def __init__(self, optim_cls, local_params_rref, *args, **kwargs):
        super().__init__()
        self._local_params = [rref.local_value() for rref in local_params_rref]
        self.optim = optim_cls(
            self._local_params,
            *args,
            **kwargs)

    @jit.export
    def step(self, autograd_ctx_id: int):
        all_local_grads = dist_autograd.get_gradients(autograd_ctx_id)
        # apply functional optimizer step with a list of gradients
        grads: List[Optional[Tensor]] = [
            all_local_grads[p] if p in all_local_grads else None
            for p in self._local_params
        ]

        self.optim.step(grads)


# TODO (wanchaol): remove/merge this with ScriptLocalOptimizer once
# we have converted all to functional optimizer in distributed.optim
class _LocalOptimizer(object):
    # Ideally we would only need to share a lock for instances of
    # _LocalOptimizer that deal with the same parameters. We are
    # making a simplifying assumption here that if there is more
    # than one instance of _LocalOptimizer per worker, they will
    # be optimizing the same parameters (e.g. each data parallel
    # trainer will create its own instance of _LocalOptimizer but
    # they will all optimize the same parameters on each worker)
    global_lock = Lock()

    def __init__(self, optim_cls, local_params_rref, *args, **kwargs):
        self._local_params = [rref.local_value() for rref in local_params_rref]
        self.optim = optim_cls(
            self._local_params,
            *args,
            **kwargs)

    def step(self, autograd_ctx_id):
        all_local_grads = dist_autograd.get_gradients(autograd_ctx_id)

        with _LocalOptimizer.global_lock:
            for param, grad in all_local_grads.items():
                param.grad = grad
            self.optim.step()


def _new_local_optimizer(optim_cls, local_params_rref, *args, **kwargs):
    return rpc.RRef(
        _LocalOptimizer(optim_cls, local_params_rref, *args, **kwargs))


def _local_optimizer_step(local_optim_rref, autograd_ctx_id):
    local_optim = local_optim_rref.local_value()
    local_optim.step(autograd_ctx_id)


# new/step functions combined with _ScriptLocalOptimizer to provide GIL-free optimizer
def _new_script_local_optimizer(optim_cls, local_params_rref, *args, **kwargs):
    optim = _ScriptLocalOptimizer(optim_cls, local_params_rref, *args, **kwargs)

    with _ScriptLocalOptimizer.compile_lock:
        script_optim = jit.script(optim)
        return rpc.RRef(
            script_optim, _ScriptLocalOptimizerInterface)

@jit.script
def _script_local_optimizer_step(
    local_optim_rref: RRef[_ScriptLocalOptimizerInterface],
    autograd_ctx_id: int
) -> None:
    local_optim = local_optim_rref.local_value()
    local_optim.step(autograd_ctx_id)

def _wait_for_all(rpc_futs):
    # TODO: improve error propagation
    exception = None
    results = []
    for fut in rpc_futs:
        try:
            results.append(fut.wait())
        except Exception as e:
            results.append(e)
            exception = e
    if exception is not None:
        raise exception
    return results


class DistributedOptimizer:
    """
    DistributedOptimizer takes remote references to parameters scattered
    across workers and applies the given optimizer locally for each parameter.

    This class uses :meth:`~torch.distributed.autograd.get_gradients` in order
    to retrieve the gradients for specific parameters.

    Concurrent calls to
    :meth:`~torch.distributed.optim.DistributedOptimizer.step`,
    either from the same or different clients, will
    be serialized on each worker -- as each worker's optimizer can only work
    on one set of gradients at a time. However, there is no guarantee that
    the full forward-backward-optimizer sequence will execute for one client
    at a time. This means that the gradients being applied may not correspond
    to the latest forward pass executed on a given worker. Also, there is no
    guaranteed ordering across workers.

    `DistributedOptimizer` creates the local optimizer with TorchScript enabled
    by default, so that optimizer updates are not blocked by the Python Global
    Interpreter Lock (GIL) in the case of multithreaded training (e.g. Distributed
    Model Parallel). This feature is currently enabled for most optimizers. You
    can also follow `the recipe`__ in PyTorch tutorials to enable TorchScript support
    for your own custom optimizers.

    Args:
        optimizer_class (optim.Optimizer): the class of optimizer to
            instantiate on each worker.
        params_rref (list[RRef]): list of RRefs to local or remote parameters
            to optimize.
        args: arguments to pass to the optimizer constructor on each worker.
        kwargs: arguments to pass to the optimizer constructor on each worker.

    Example::
        >>> import torch.distributed.autograd as dist_autograd
        >>> import torch.distributed.rpc as rpc
        >>> from torch import optim
        >>> from torch.distributed.optim import DistributedOptimizer
        >>>
        >>> with dist_autograd.context() as context_id:
        >>>   # Forward pass.
        >>>   rref1 = rpc.remote("worker1", torch.add, args=(torch.ones(2), 3))
        >>>   rref2 = rpc.remote("worker1", torch.add, args=(torch.ones(2), 1))
        >>>   loss = rref1.to_here() + rref2.to_here()
        >>>
        >>>   # Backward pass.
        >>>   dist_autograd.backward(context_id, [loss.sum()])
        >>>
        >>>   # Optimizer.
        >>>   dist_optim = DistributedOptimizer(
        >>>      optim.SGD,
        >>>      [rref1, rref2],
        >>>      lr=0.05,
        >>>   )
        >>>   dist_optim.step(context_id)

    __ https://github.com/pytorch/tutorials/pull/1465
    """

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

    def __init__(self, optimizer_class, params_rref, *args, **kwargs):
        per_worker_params_rref = defaultdict(list)
        for param in params_rref:
            per_worker_params_rref[param.owner()].append(param)

        if optimizer_class in DistributedOptimizer.functional_optim_map and jit._state._enabled:
            optim_ctor = DistributedOptimizer.functional_optim_map.get(optimizer_class)
        else:
            optim_ctor = optimizer_class
        self.is_functional_optim = (optim_ctor != optimizer_class)

        if self.is_functional_optim:
            optimizer_new_func = _new_script_local_optimizer
        else:
            logger.warn(
                f"Creating the optimizer {optimizer_class} without TorchScript support, "
                "this might result in slow computation time in multithreading environment"
                "(i.e. Distributed Model Parallel training on CPU) due to the Python's "
                "Global Interpreter Lock (GIL). Please file an issue if you need this "
                "optimizer in TorchScript. "
            )
            optimizer_new_func = _new_local_optimizer

        remote_optim_futs = []
        for worker, param_rrefs in per_worker_params_rref.items():
            remote_optim_rref_fut = rpc.rpc_async(
                worker,
                optimizer_new_func,
                args=(optim_ctor, param_rrefs) + args,
                kwargs=kwargs,
            )
            remote_optim_futs.append(remote_optim_rref_fut)

        self.remote_optimizers = _wait_for_all(remote_optim_futs)

    def step(self, context_id):
        """
        Performs a single optimization step.

        This will call :meth:`torch.optim.Optimizer.step` on each worker
        containing parameters to be optimized, and will block until all workers
        return. The provided ``context_id`` will be used to retrieve the
        corresponding :class:`~torch.distributed.autograd.context` that
        contains the gradients that should be applied to the parameters.

        Args:
            context_id: the autograd context id for which we should run the
                optimizer step.
        """
        dist_autograd._is_valid_context(context_id)

        if self.is_functional_optim:
            optimizer_step_func = _script_local_optimizer_step
        else:
            optimizer_step_func = _local_optimizer_step

        rpc_futs = []
        for optimizer in self.remote_optimizers:
            rpc_futs.append(rpc.rpc_async(
                optimizer.owner(),
                optimizer_step_func,
                args=(optimizer, context_id),
            ))
        _wait_for_all(rpc_futs)
