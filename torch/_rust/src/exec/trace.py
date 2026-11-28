# Records the aten ops run below Autograd, for building joint graphs (see
# exec/joint.rs). Loaded by the Rust extension as torch._rust._trace.

import torch
from torch.utils._python_dispatch import TorchDispatchMode


class Tracer(TorchDispatchMode):
    # Each record is (OpOverload, args, kwargs, out). Holding them keeps every
    # tensor alive, so tensor ids stay unique while the trace is in use.
    def __init__(self, records):
        super().__init__()
        self.records = records

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        self.records.append((func, args, kwargs, out))
        return out


def forward(fn, args, records):
    # Runs fn on args with each tensor that requires grad replaced by a
    # detached copy that does, recording the ops. Returns (the copies, by arg
    # position or None, the args fn ran on, the outputs).
    inner = [
        a.detach().requires_grad_()
        if isinstance(a, torch.Tensor) and a.requires_grad
        else None
        for a in args
    ]
    call = [i if i is not None else a for i, a in zip(inner, args)]
    with Tracer(records):
        out = fn(*call)
    return inner, call, out


def backward(outs, inputs, tangents, records=None):
    # torch.autograd.grad, recording the ops into `records` unless it's None.
    # `outs`/`tangents` are the tensor outputs that require grad and their
    # gradients. The graph is kept, as a CfgBackward frees it when autograd
    # releases the node; grad mode on means create_graph.
    grad = lambda: torch.autograd.grad(  # noqa: E731
        outs,
        inputs,
        tangents,
        allow_unused=True,
        retain_graph=True,
        create_graph=torch.is_grad_enabled(),
    )
    if records is None:
        return grad()
    with Tracer(records):
        return grad()
