torch.func
==========

.. currentmodule:: torch.func

torch.func, previously known as "functorch", is
`JAX-like <https://github.com/google/jax>`_ composable function transforms for PyTorch.

.. note::
   This library is currently in `beta <https://pytorch.org/blog/pytorch-feature-classification-changes/#beta>`_.
   What this means is that the features generally work (unless otherwise documented)
   and we (the PyTorch team) are committed to bringing this library forward. However, the APIs
   may change under user feedback and we don't have full coverage over PyTorch operations.

   If you have suggestions on the API or use-cases you'd like to be covered, please
   open an GitHub issue or reach out. We'd love to hear about how you're using the library.

What are composable function transforms?
----------------------------------------

- A "function transform" is a higher-order function that accepts a numerical function
  and returns a new function that computes a different quantity.

- :mod:`torch.func` has auto-differentiation transforms (``grad(f)`` returns a function that
  computes the gradient of ``f``), a vectorization/batching transform (``vmap(f)``
  returns a function that computes ``f`` over batches of inputs), and others.

- These function transforms can compose with each other arbitrarily. For example,
  composing ``vmap(grad(f))`` computes a quantity called per-sample-gradients that
  stock PyTorch cannot efficiently compute today.

Why composable function transforms?
-----------------------------------

There are a number of use cases that are tricky to do in PyTorch today:

- computing per-sample-gradients (or other per-sample quantities)
- running ensembles of models on a single machine
- efficiently batching together tasks in the inner-loop of MAML
- efficiently computing Jacobians and Hessians
- efficiently computing batched Jacobians and Hessians

Composing :func:`vmap`, :func:`grad`, and :func:`vjp` transforms allows us to express the above without designing a separate subsystem for each.
This idea of composable function transforms comes from the `JAX framework <https://github.com/google/jax>`_.

Read More
---------

.. toctree::
   :maxdepth: 2

   func.whirlwind_tour
   func.api
   func.ux_limitations
   func.migrating
