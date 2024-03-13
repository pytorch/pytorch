:github_url: https://github.com/pytorch/functorch

functorch
===================================

.. currentmodule:: functorch

functorch is `JAX-like <https://github.com/google/jax>`_ composable function transforms for PyTorch.

.. warning::

   We've integrated functorch into PyTorch. As the final step of the
   integration, the functorch APIs are deprecated as of PyTorch 2.0.
   Please use the torch.func APIs instead and see the
   `migration guide <https://pytorch.org/docs/master/func.migrating.html>`_
   and `docs <https://pytorch.org/docs/master/func.html>`_
   for more details.

What are composable function transforms?
----------------------------------------

- A "function transform" is a higher-order function that accepts a numerical function
  and returns a new function that computes a different quantity.

- functorch has auto-differentiation transforms (``grad(f)`` returns a function that
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

Check out our `whirlwind tour <whirlwind_tour>`_ or some of our tutorials mentioned below.


.. toctree::
   :maxdepth: 2
   :caption: functorch: Getting Started

   install
   notebooks/whirlwind_tour.ipynb
   ux_limitations

.. toctree::
   :maxdepth: 2
   :caption: functorch API Reference and Notes

   functorch
   experimental
   aot_autograd

.. toctree::
   :maxdepth: 1
   :caption: functorch Tutorials

   notebooks/jacobians_hessians.ipynb
   notebooks/ensembling.ipynb
   notebooks/per_sample_grads.ipynb
   notebooks/neural_tangent_kernels.ipynb
   notebooks/aot_autograd_optimizations.ipynb
   notebooks/minifier.ipynb
