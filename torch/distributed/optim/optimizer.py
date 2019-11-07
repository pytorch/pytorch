import torch.distributed.rpc as rpc
import torch.distributed.autograd as dist_autograd

from collections import defaultdict
from threading import Lock


class _LocalOptimizer:
    def __init__(self, optim_cls, local_params_rref, *args, **kwargs):
        self.optim = optim_cls(
            [rref.local_value().wait() for rref in local_params_rref],
            *args,
            **kwargs)
        self.lock = Lock()

    def step(self, autograd_ctx_id):
        all_local_grads = dist_autograd.get_gradients(autograd_ctx_id)

        with self.lock:
            for param, grad in all_local_grads.items():
                param.grad = grad
            self.optim.step()


def _local_optimizer_step(local_optim_rref, autograd_ctx_id):
    local_optim = local_optim_rref.local_value().wait()
    local_optim.step(autograd_ctx_id)


def _wait_for_all(rpc_futs):
    # TODO: improve error propagation
    exception = None
    for fut in rpc_futs:
        try:
            fut.wait()
        except Exception as e:
            exception = e
    if exception is not None:
        raise exception


class DistributedOptimizer:
    """
    DistributedOptimizer takes remote references to parameters scattered
    across workers and applies the given optimizer locally for each parameter.

    This class uses distributed autograd in order to retrieve the gradients for
    specific parameters.

    Concurrent calls to step(), either from the same or different clients, will
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
    """
    def __init__(self, optimizer_class, params_rref, *args, **kwargs):
        per_worker_params_rref = defaultdict(list)
        for param in params_rref:
            per_worker_params_rref[param.owner()].append(param)

        self.remote_optimizers = []
        for worker, param_rrefs in per_worker_params_rref.items():
            remote_optim_rref = rpc.remote(
                worker,
                _LocalOptimizer,
                args=[optimizer_class, param_rrefs] + list(args),
                kwargs=kwargs,
            )
            self.remote_optimizers.append(remote_optim_rref)


    def step(self):
        """
        Performs a single optimization step.

        This will call optimizer.step on each worker containing parameters
        to be optimized, and will block until all workers return. The current
        distributed autograd context will be used globally.
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
