torch.func interaction with torch.compile
==============================================

So you want to use a `torch.func` ("functorch") transform (like `vmap`, `grad`, `jacrev`, etc) with `torch.compile`. Here's a guide to what works today, what doesn't, and how to work around it.

Applying a `torch.func` transform to a `torch.compile`'d function
-----------------------------------------------------------------

This doesn't work and is being tracked by `https://github.com/pytorch/pytorch/issues/100320`.

.. code:: python

    import torch

    @torch.compile
    def f(x):
        return torch.sin(x)

    def g(x):
        return torch.grad(f)(x)

    x = torch.randn(2, 3)
    g(x)

As a workaround, please put the `torch.compile` outside of the `torch.func` transform:

.. code:: python

    import torch

    def f(x):
        return torch.sin(x)

    @torch.compile
    def g(x):
        return torch.vmap(f)(x)

    x = torch.randn(2, 3)
    g(x)

Doesn't work (PT 2.0): calling a `torch.func` transform inside of a `torch.compile`'ed function
------------------------------------------------------------------------------------------------

.. code:: python

    import torch

    @torch.compile
    def f(x):
        return torch.vmap(torch.sum)(x)

    x = torch.randn(2, 3)
    f(x)

This doesn't work yet. Please see the workaround (the next section).

Workaround: use `torch._dynamo.allow_in_graph`
----------------------------------------------

`allow_in_graph` is an escape hatch. If your code does not work with `torch.compile`, which introspects Python bytecode, but you believe it will work via a symbolic tracing approach (like `jax.jit`), then use `allow_in_graph`.

By using `allow_in_graph` to annotate a function, you promise PyTorch a couple of things that we are unable to completely verify:
- Your function is pure. That is, all outputs only depend on the inputs and do not depend on any captured Tensors.
- Your function is functional. That is, it does not mutate any state. This may be relaxed; we actually support functions that appear to be functional from the outside: they may have in-place PyTorch operations, but may not mutate global state or inputs to the function.
- Your function does not raise data-dependent errors.

.. code:: python

    import torch

    @torch.compile
    def f(x):
        return torch._dynamo.allow_in_graph(torch.vmap(torch.sum))(x)

    x = torch.randn(2, 3)
    f(x)

A common pitfall is using `allow_in_graph` to annotate a function that invokes an `nn.Module`. This is because the outputs now depend on the parameters of the `nn.Module`. To actually get this to work, use `torch.func.functional_call` to extract the module state.
