.. role:: hidden
    :class: hidden-section

Automatic differentiation package - torch.autograd
==================================================

.. automodule:: torch.autograd
.. currentmodule:: torch.autograd

.. autofunction:: backward

.. autofunction:: grad

Variable
--------

API compatibility
^^^^^^^^^^^^^^^^^

Variable API is nearly the same as regular Tensor API (with the exception
of a couple in-place methods, that would overwrite inputs required for
gradient computation). In most cases Tensors can be safely replaced with
Variables and the code will remain to work just fine. Because of this,
we're not documenting all the operations on variables, and you should
refer to :class:`torch.Tensor` docs for this purpose.

In-place operations on Variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Supporting in-place operations in autograd is a hard matter, and we discourage
their use in most cases. Autograd's aggressive buffer freeing and reuse makes
it very efficient and there are very few occasions when in-place operations
actually lower memory usage by any significant amount. Unless you're operating
under heavy memory pressure, you might never need to use them.

In-place correctness checks
^^^^^^^^^^^^^^^^^^^^^^^^^^^

All :class:`Variable` s keep track of in-place operations applied to them, and
if the implementation detects that a variable was saved for backward in one of
the functions, but it was modified in-place afterwards, an error will be raised
once backward pass is started. This ensures that if you're using in-place
functions and not seeing any errors, you can be sure that the computed
gradients are correct.


.. autoclass:: Variable
    :members:

:hidden:`Function`
------------------

.. autoclass:: Function
    :members:

Profiler
--------

Autograd includes a profiler that lets you inspect the cost of different
operators inside your model - both on the CPU and GPU. There are two modes
implemented at the moment - CPU-only using :class:`~torch.autograd.profiler.profile`.
and nvprof based (registers both CPU and GPU activity) using
:class:`~torch.autograd.profiler.emit_nvtx`.

.. autoclass:: torch.autograd.profiler.profile
    :members:

.. autoclass:: torch.autograd.profiler.emit_nvtx
    :members:

.. autofunction:: torch.autograd.profiler.load_nvprof
