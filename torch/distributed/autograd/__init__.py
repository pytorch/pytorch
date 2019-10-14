from __future__ import absolute_import, division, print_function, unicode_literals

class context(object):
    '''
    Autograd context object to wrap forward and backward passes when using
    distributed autograd. The context_id generated in the 'with' is required
    to uniquely identify a distributed autograd pass on all workers. Each
    worker stores metadata associated with this context_id, which is required
    to correctly execute a distributed autograd pass.

    This is only needed in the "FAST" mode (as described in
    https://github.com/pytorch/pytorch/issues/23110) for distributed autograd,
    where we assume all RPC communication is would also be part of the backward
    pass.

    Example::
        >> import torch.distributed.autograd as dist_autograd
        >> with dist_autograd.context() as context_id:
        >>      forward pass...
        >>      backward pass...
        >>      optimizer step...
    '''
    # TODO: Update the above example to a working solution.
    def __enter__(self):
        self.autograd_context = _new_context()
        return self.autograd_context._context_id()

    def __exit__(self, type, value, traceback):
        _release_context(self.autograd_context._context_id())


def backward(roots):
    '''
    Kicks off the distributed backward pass using the provided roots. This
    currently implements the "FAST" mode
    (see https://github.com/pytorch/pytorch/issues/23110) algorithm which
    assumes all RPC messages sent in the same distributed autograd context
    across workers would be part of the autograd graph during the backward pass.

    We use the provided roots to discover the autograd graph and compute
    appropriate dependencies. This method blocks until the entire
    autograd computation is done.

    We accumulate the gradients in the appropriate "autograd context id" on each
    of the nodes. The autograd context id used is the current autograd context
    id of this node when backward() is called. If there is no valid autograd
    context id, we throw an error.

    Arguments:
        roots: List of tensors which represent the roots of the autograd
            computation. All the tensors should be scalars.

    Example::
        >> import torch.distributed.autograd as dist_autograd
        >> with dist_autograd.context() as context_id:
        >>      pred = model.forward()
        >>      loss = loss_func(pred, loss)
        >>      dist_autograd.backward(loss)
    '''
    _backward(roots)
