:github_url: https://github.com/pytorch/functorch

functorch
===================================

.. currentmodule:: functorch

functorch is `JAX-like <https://github.com/google/jax>`_ composable function transforms for PyTorch.

It aims to provide composable vmap and grad transforms that work with PyTorch modules
and PyTorch autograd with good eager-mode performance.

.. note::
   This library is currently in [beta](https://pytorch.org/blog/pytorch-feature-classification-changes/#beta).
   What this means is that the features generally work (unless otherwise documented)
   and we (the PyTorch team) are committed to bringing this library forward. However, the APIs
   may change under user feedback and we don't have full coverage over PyTorch operations.

   If you have suggestions on the API or use-cases you'd like to be covered, please
   open an github issue or reach out. We'd love to hear about how you're using the library.

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

For a whirlwind tour of how to use the transforms, please check out `this section in our README <https://github.com/pytorch/functorch/blob/main/README.md#what-are-the-transforms>`_. For installation instructions or the API reference, please check below.


.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   install
   notebooks/whirlwind_tour.ipynb
   ux_limitations

.. toctree::
   :maxdepth: 2
   :caption: API Reference and Notes

   functorch
   aot_autograd

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   notebooks/jacobians_hessians.ipynb
   notebooks/ensembling.ipynb
   notebooks/per_sample_grads.ipynb
   notebooks/neural_tangent_kernels.ipynb
   notebooks/aot_autograd_optimizations.ipynb
   notebooks/minifier.ipynb
