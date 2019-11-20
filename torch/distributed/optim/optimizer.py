import torch.distributed.rpc as rpc
import torch.distributed.autograd as dist_autograd

from collections import defaultdict
from threading import Lock


class _LocalOptimizer:
    def __init__(self, optim_cls, local_params_rref, *args, **kwargs):
        self.optim = optim_cls(
            [rref.local_value() for rref in local_params_rref],
            *args,
            **kwargs)
        self.lock = Lock()

    def step(self, autograd_ctx_id):
        all_local_grads = dist_autograd.get_gradients(autograd_ctx_id)

        with self.lock:
            for param, grad in all_local_grads.items():
                param.grad = grad
            self.optim.step()


def _new_local_optimizer(optim_cls, local_params_rref, *args, **kwargs):
    return rpc.RRef(
        _LocalOptimizer(optim_cls, local_params_rref, *args, **kwargs))


def _local_optimizer_step(local_optim_rref, autograd_ctx_id):
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

    Args:
        optimizer_class (optim.Optimizer): the class of optimizer to
            instantiate on each worker.
        params_rref (list[RRef]): list of RRefs to local or remote parameters
            to optimize.
        args: arguments to pass to the optimizer constructor on each worker.
        kwargs: arguments to pass to the optimizer constructor on each worker.

    Example::

        >> import torch.distributed.autograd as dist_autograd
        >> import torch.distributed.rpc as rpc
        >> from torch import optim
        >> from torch.distributed.optim import DistributedOptimizer
        >>
        >> with dist_autograd.context() as context_id:
        >>   # Forward pass.
        >>   rref1 = rpc.remote("worker1", torch.add, args=(torch.ones(2), 3))
        >>   rref2 = rpc.remote("worker1", torch.add, args=(torch.ones(2), 1))
        >>   loss = rref1.to_here() + rref2.to_here()
        >>
        >>   # Backward pass.
        >>   dist_autograd.backward([loss.sum()])
        >>
        >>   # Optimizer.
        >>   dist_optim = DistributedOptimizer(
        >>      optim.SGD,
        >>      [rref1, rref2],
        >>      lr=0.05,
        >>   )
        >>   dist_optim.step()
    """
    def __init__(self, optimizer_class, params_rref, *args, **kwargs):
        per_worker_params_rref = defaultdict(list)
        for param in params_rref:
            per_worker_params_rref[param.owner()].append(param)

        remote_optim_futs = []
        for worker, param_rrefs in per_worker_params_rref.items():
            remote_optim_rref_fut = rpc.rpc_async(
                worker,
                _new_local_optimizer,
                args=(optimizer_class, param_rrefs) + args,
                kwargs=kwargs,
            )
            remote_optim_futs.append(remote_optim_rref_fut)

        self.remote_optimizers = _wait_for_all(remote_optim_futs)

    def step(self):
        """
        Performs a single optimization step.

        This will call :meth:`torch.optim.Optimizer.step` on each worker
        containing parameters to be optimized, and will block until all workers
        return. The current distributed autograd
        :class:`~torch.distributed.autograd.context` will be used globally.
        """
        autograd_ctx_id = dist_autograd._current_context()._context_id()
        rpc_futs = []
        for optim in self.remote_optimizers:
            rpc_futs.append(rpc.rpc_async(
                optim.owner(),
                _local_optimizer_step,
                args=(optim, autograd_ctx_id),
            ))
        _wait_for_all(rpc_futs)
