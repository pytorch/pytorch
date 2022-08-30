import torch
import functorch._C

m = functorch._C._dispatch_library("FRAGMENT", "aten", "")


def custom_vjp(name, filter_fn, fwd_fn, bwd_fn):
    m.def_(f"{name}(Tensor[] args) -> Tensor[]")
    m.impl(f"{name}", "CompositeImplicitAutograd", fwd_fn)

    m.def_(f"{name}_vjp(Tensor[] args) -> Tensor[]")
    m.impl(f"{name}_vjp", "CompositeImplicitAutograd", bwd_fn)

    # TODO: it looks like the autograd alias key doesn't work
    m.gen_backward_binding(f"{name}", "AutogradCPU")
    m.gen_backward_binding(f"{name}", "AutogradCUDA")

    def wrapped(*args):
        return filter_fn(getattr(torch.ops.aten, name)(args))
    return wrapped
