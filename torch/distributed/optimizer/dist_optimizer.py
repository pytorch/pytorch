import torch.distributed.rpc as rpc
import torch.distributed.autograd as dist_autograd

from collections import defaultdict

class FunctionalOptimizer:
    """Base class for functional optimizers.

    Functional optimizers are similar to torch.optim optimizers, but
    instead of using params.grad as input, it takes the gradients as input to
    the step function. This allows to implement multi-threaded versions of
    such as Hogwild!.

    Args:
        params (list): list of parameters to optimize. order of the list
                       matters as the list of gradients passed to the step
                       function must be aligned with this list.
    """
    def __init__(self, params):
        self.params = params

    def step(self, gradients):
        """Performs a single optimization step.

        Arguments:
            gradients (list): a list of gradient tensors to be applied to the
                              parameters. This list must be aligned with
                              the self.params list.
        """
        raise NotImplementedError


class FunctionalSGD(FunctionalOptimizer):
    """Simplistic implementation of Stocastic Gradient Descent optimizer.

    Arguments:
        params (list): list of parameters to optimize
        lr (float): learning rate
    """
    def __init__(self, params, lr=0.01):
        super(FunctionalSGD, self).__init__(params)
        self.lr = lr

    def step(self, gradients):
        for param, grad in zip(self.params, gradients):
            param.data.add_(-self.lr, grad.data)


def _create_local_optimizer(optim_cls, local_params_rref, *args, **kwargs):
    return optim_cls(
        [rref.local_value().wait() for rref in local_params_rref],
        *args,
        **kwargs)


def _local_optimizer_step(local_optim_rref, autograd_ctx_id):
    local_optim = local_optim_rref.local_value().wait()
    all_local_grads = dist_autograd.get_gradients(autograd_ctx_id)
    local_grads = [all_local_grads[param] for param in local_optim.params]
    local_optim.step(local_grads)


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

    Args:
        optimizer_class (FunctionalOptimizer): the class of optimizer to
            instantiate on each worker.
        params_rref (list[RRef]): list of RRefs to local or remote parameters
            to optimize.
        args: arguments to pass to the optimizer constructor on each worker.
        kwargs: arguments to pass to the optimizer constructor on each worker.
    """
    def __init__(self, optimizer_class, params_rref, *args, **kwargs):
        per_worker_params_rref = defaultdict(lambda: [])
        for param in params_rref:
            per_worker_params_rref[param.owner()].append(param)

        self.remote_optimizers = []
        for worker, param_rrefs in per_worker_params_rref.items():
            remote_optim = rpc.remote(
                worker,
                _create_local_optimizer,
                args=[optimizer_class, param_rrefs] + list(args),
                kwargs=kwargs,
            )
            self.remote_optimizers.append(remote_optim)


    def step(self, autograd_ctx_id):
        """
        Performs a single optimization step.

        This will call optimizer.step on each worker containing parameters
        to be optimized, and will block until all workers return.

        Args:
            autograd_ctx_id: the distributed autograd context id. This is
                used by the workers to retrieve gradients for the given
                parameters.
        """
        rpc_futs = []
        for optim in self.remote_optimizers:
            rpc_futs.append(rpc.rpc_async(
                optim.owner(),
                _local_optimizer_step,
                args=(optim, autograd_ctx_id),
            ))
        _wait_for_all(rpc_futs)
