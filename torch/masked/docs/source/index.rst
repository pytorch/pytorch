.. MaskedTensor documentation master file, created by
   sphinx-quickstart on Wed Feb 16 11:48:08 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MaskedTensor
========================================

This library is a part of the `PyTorch <http://pytorch.org/>`_ project. Please note that this library is currently
classified as a prototype -- that is, this library is at an early stage for feedback and testing, and we encourage
users to submit any issues they may encounter, feature requests, etc. The Github can be found `here <https://github.com/pytorch/maskedtensor>`_.

The purpose of :mod:`maskedtensor` is to serve as an extension to `torch.Tensor`, especially in cases of:

* using any masked semantics (e.g. variable length tensors, nan* operators, etc.)
* differentiation between 0 and NaN gradients
* various sparse applications (see tutorial)

More details can be found in the Overview tutorial.

.. toctree::
   :maxdepth: 1
   :caption: Installation

   install

.. toctree::
   :maxdepth: 1
   :caption: Tutorials:

   notebooks/overview
   notebooks/sparse
   notebooks/nan_grad
   notebooks/safe_softmax
   notebooks/issue_1369
   notebooks/nan_operators

.. toctree::
   :maxdepth: 1
   :caption: Python API

   unary
   binary
   reductions
   view_and_select
