.. _custom-ops-landing-page:

PyTorch Custom Operators Landing Page
=====================================

PyTorch offers a large library of operators that work on Tensors (e.g. :func:`torch.add`,
:func:`torch.sum`, etc). However, you may wish to bring a new custom operation to PyTorch
and get it to work with subsystems like :func:`torch.compile`, autograd, and :func:`torch.vmap`.
In order to do so, you must register the custom operation with PyTorch via the Python
:ref:`torch-library-docs` or C++ TORCH_LIBRARY APIs.

TL;DR
-----

How do I author a custom op from Python?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

..
  [comment] TODO(rzou): The following will be a link to a tutorial on the PyTorch tutorials site in 2.4

Please see the `Python Custom Operators tutorial <https://colab.research.google.com/drive/1xCh5BNHxGnutqGLMHaHwm47cbDL9CB1g#scrollTo=gg6WorNtKzeh>`_


How do I integrate custom C++ and/or CUDA code with PyTorch?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

..
  [comment] TODO(rzou): The following will be a link to a tutorial on the PyTorch tutorials site in 2.4

Please see the `Custom C++ and CUDA Operators tutorial <https://docs.google.com/document/d/1-LdJZBzlxiF0Tm-8NfbyFvRJaofdwRgLcycXGmlIpS0>`_


For more details
^^^^^^^^^^^^^^^^

Please see `The Custom Operators Manual (gdoc) <https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU>`_
(we're working on moving the information to our docs site). We recommend that you
first read one of the tutorials above and then use the Custom Operators Manual as a reference;
it is not meant to be read head to toe.

When should I create a Custom Operator?
---------------------------------------
If your operation is expressible as a composition of built-in PyTorch operators
then please write it as a Python function and call it instead of creating a
custom operator. Use the operator registration APIs to create a custom op if you
are calling into some library that PyTorch doesn't understand (e.g. custom C/C++ code,
a custom CUDA kernel, or Python bindings to C/C++/CUDA extensions).

Why should I create a Custom Operator?
--------------------------------------

It is possible to use a C/C++/CUDA kernel by grabbing a Tensor's data pointer
and passing it to a pybind'ed kernel. However, this approach doesn't compose with
PyTorch subsystems like autograd, torch.compile, vmap, and more. In order
for an operation to compose with PyTorch subsystems, it must be registered
via the operator registration APIs.
