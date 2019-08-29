from __future__ import absolute_import, division, print_function, unicode_literals

class context(object):
    '''
    Autograd context object to wrap forward and backward passes when using
    distributed autograd. The context_id generated in the 'with' is required
    to uniquely identify a distributed autograd pass on all workers. Each
    worker stores metadata associated with this context_id, which is required
    to correctly execute a distributed autograd pass.

    This is only needed in the "FAST" mode for distributed autograd, where we
    assume all RPC communication is would also be part of the backward pass.

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
