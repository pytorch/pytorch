.. role:: hidden
    :class: hidden-section

Automatic differentiation package - torch.autograd
==================================================

.. automodule:: torch.autograd
.. currentmodule:: torch.autograd

.. autofunction:: backward

.. autofunction:: grad

.. _locally-disable-grad:

Locally disabling gradient computation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: no_grad

.. autoclass:: enable_grad

.. autoclass:: set_grad_enabled

In-place operations on Tensors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Supporting in-place operations in autograd is a hard matter, and we discourage
their use in most cases. Autograd's aggressive buffer freeing and reuse makes
it very efficient and there are very few occasions when in-place operations
actually lower memory usage by any significant amount. Unless you're operating
under heavy memory pressure, you might never need to use them.

In-place correctness checks
---------------------------

All :class:`Tensor` s keep track of in-place operations applied to them, and
if the implementation detects that a tensor was saved for backward in one of
the functions, but it was modified in-place afterwards, an error will be raised
once backward pass is started. This ensures that if you're using in-place
functions and not seeing any errors, you can be sure that the computed
gradients are correct.

Variable (deprecated)
^^^^^^^^^^^^^^^^^^^^^

.. warning::
    The Variable API has been deprecated: Variables are no longer necessary to
    use autograd with tensors. Autograd automatically supports Tensors with
    ``requires_grad`` set to ``True``. Below please find a quick guide on what
    has changed:

    - ``Variable(tensor)`` and ``Variable(tensor, requires_grad)`` still work as expected,
      but they return Tensors instead of Variables.
    - ``var.data`` is the same thing as ``tensor.data``.
    - Methods such as ``var.backward(), var.detach(), var.register_hook()`` now work on tensors
      with the same method names.

    In addition, one can now create tensors with ``requires_grad=True`` using factory
    methods such as :func:`torch.randn`, :func:`torch.zeros`, :func:`torch.ones`, and others
    like the following:

    ``autograd_tensor = torch.randn((2, 3, 4), requires_grad=True)``

Tensor autograd functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: torch.Tensor
   :members: grad, requires_grad, is_leaf, backward, detach, detach_, register_hook, retain_grad

:hidden:`Function`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: Function
    :members:

.. _grad-check:

Numerical gradient checking
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: gradcheck

.. autofunction:: gradgradcheck

Profiler
^^^^^^^^

Autograd includes a profiler that lets you inspect the cost of different
operators inside your model - both on the CPU and GPU. There are two modes
implemented at the moment - CPU-only using :class:`~torch.autograd.profiler.profile`.
and nvprof based (registers both CPU and GPU activity) using
:class:`~torch.autograd.profiler.emit_nvtx`.

.. autoclass:: torch.autograd.profiler.profile
    :members:

.. autoclass:: torch.autograd.profiler.record_function
    :members:

.. autoclass:: torch.autograd.profiler.emit_nvtx
    :members:

.. autofunction:: torch.autograd.profiler.load_nvprof

Anomaly detection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: detect_anomaly

.. autoclass:: set_detect_anomaly
