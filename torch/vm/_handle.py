"""The tensor that the node of a graph holds in the place of an input whose
history a later backward may go through.

Eager's nodes do not hold an input that none of them saved: the caller can
let go of it, and it ends there. A backward that is differentiated again
(``create_graph=True``) still goes through the history of such an input,
which eager's nodes hold by their edges. ``GraphInput`` gives what the node
of a graph holds for that: a tensor in the memory of the input, with its
version counter and with an edge to the node that made the input, which is
no view of the input and so holds neither the input nor its Python object.
"""

import torch


# The method of the class under every tensor: what somebody puts on
# ``torch.Tensor`` under the name is not called here.
_detach = torch._C.TensorBase.detach


class GraphInput(torch.autograd.function._SingleLevelFunction):
    """``GraphInput.apply(tensor)``: a tensor in the memory of ``tensor``
    whose node hands a gradient on to the node that made ``tensor``.

    The engine's own: one is made only where a call enters a graph with
    gradients, for an input that a gradient goes to and that no node of
    eager's keeps. That is never under a transform of ``torch.func``,
    under anomaly mode or under a forward-mode level, where such a graph
    runs on eager's tape. The class is a single-level function, which
    costs a fifth of a ``torch.autograd.Function`` with a
    ``setup_context`` at every call; PyTorch refuses to make one inside a
    transform of ``torch.func`` (an internal assertion on single-level
    functions). A gradient goes through its node under every transform,
    in batches as well.
    """

    @staticmethod
    def forward(ctx, tensor):
        return _detach(tensor)

    @staticmethod
    def backward(ctx, grad):
        return grad

    @staticmethod
    def jvp(ctx, tangent):
        return tangent
