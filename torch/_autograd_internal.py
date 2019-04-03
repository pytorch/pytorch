import torch


class no_version_update(object):
    r"""
    Context-manager that disables version update when an in-place operation on a
    tensor is performed.

    Without this context, in-place operations on a tensor will update the tensor's
    version counter, which is used for detecting modifications to saved autograd
    variables that can result in incorrect gradient calculations.

    With this context, in-place operations on a tensor will not update the tensor's
    version counter, which is useful when we are aware of the possibly incorrect
    gradient calculations, but still want to prevent version update from happening.
    E.g., `get_numerical_jacobian(...)` makes small finite changes to the input
    tensors of a graph and forward them through the graph to compute numerical
    gradients, before restoring the input tensors to their original values. It wants
    the original version counter values of the input tensors to always be preserved,
    so that making small finite changes to the input tensors (and restoring the
    original values later) doesn't invalidate the input tensors in the autograd graph.

    Example::

        >>> x = torch.tensor(1.)
        >>> x._version
        0
        >>> x.add_(1)
        tensor(2.)
        >>> x._version
        1
        >>> with torch._autograd_internal.no_version_update():
        ...   x.add_(1)
        >>> x._version
        1
    """
    def __enter__(self):
        self.prev = torch.is_version_update_enabled()
        torch._C.set_version_update_enabled(False)

    def __exit__(self, *args):
        torch._C.set_version_update_enabled(self.prev)
        return False
