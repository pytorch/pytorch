.. role:: hidden
    :class: hidden-section

Automatic differentiation package - torch.autograd
==================================================

.. automodule:: torch.autograd
.. currentmodule:: torch.autograd

.. autosummary::
    :toctree: generated
    :nosignatures:

    backward
    grad

.. _forward-mode-ad:

Forward-mode Automatic Differentiation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. warning::
    This API is in beta. Even though the function signatures are very unlikely to change, improved
    operator coverage is planned before we consider this stable.

Please see the `forward-mode AD tutorial <https://pytorch.org/tutorials/intermediate/forward_ad_usage.html>`__
for detailed steps on how to use this API.

.. autosummary::
    :toctree: generated
    :nosignatures:

    forward_ad.dual_level
    forward_ad.make_dual
    forward_ad.unpack_dual
    forward_ad.enter_dual_level
    forward_ad.exit_dual_level
    forward_ad.UnpackedDualTensor

.. _functional-api:

Functional higher level API
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. warning::
    This API is in beta. Even though the function signatures are very unlikely to change, major
    improvements to performances are planned before we consider this stable.

This section contains the higher level API for the autograd that builds on the basic API above
and allows you to compute jacobians, hessians, etc.

This API works with user-provided functions that take only Tensors as input and return
only Tensors.
If your function takes other arguments that are not Tensors or Tensors that don't have requires_grad set,
you can use a lambda to capture them.
For example, for a function ``f`` that takes three inputs, a Tensor for which we want the jacobian, another
tensor that should be considered constant and a boolean flag as ``f(input, constant, flag=flag)``
you can use it as ``functional.jacobian(lambda x: f(x, constant, flag=flag), input)``.

.. autosummary::
    :toctree: generated
    :nosignatures:

    functional.jacobian
    functional.hessian
    functional.vjp
    functional.jvp
    functional.vhp
    functional.hvp

.. _locally-disable-grad:

Locally disabling gradient computation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

See :ref:`locally-disable-grad-doc` for more information on the differences
between no-grad and inference mode as well as other related mechanisms that
may be confused with the two. Also see :ref:`torch-rst-local-disable-grad`
for a list of functions that can be used to locally disable gradients.

.. _default-grad-layouts:

Default gradient layouts
^^^^^^^^^^^^^^^^^^^^^^^^

When a non-sparse ``param`` receives a non-sparse gradient during
:func:`torch.autograd.backward` or :func:`torch.Tensor.backward`
``param.grad`` is accumulated as follows.

If ``param.grad`` is initially ``None``:

1. If ``param``'s memory is non-overlapping and dense, ``.grad`` is
   created with strides matching ``param`` (thus matching ``param``'s
   layout).
2. Otherwise, ``.grad`` is created with rowmajor-contiguous strides.

If ``param`` already has a non-sparse ``.grad`` attribute:

3. If ``create_graph=False``, ``backward()`` accumulates into ``.grad``
   in-place, which preserves its strides.
4. If ``create_graph=True``, ``backward()`` replaces ``.grad`` with a
   new tensor ``.grad + new grad``, which attempts (but does not guarantee)
   matching the preexisting ``.grad``'s strides.

The default behavior (letting ``.grad``\ s be ``None`` before the first
``backward()``, such that their layout is created according to 1 or 2,
and retained over time according to 3 or 4) is recommended for best performance.
Calls to ``model.zero_grad()`` or ``optimizer.zero_grad()`` will not affect ``.grad``
layouts.

In fact, resetting all ``.grad``\ s to ``None`` before each
accumulation phase, e.g.::

    for iterations...
        ...
        for param in model.parameters():
            param.grad = None
        loss.backward()

such that they're recreated according to 1 or 2 every time,
is a valid alternative to ``model.zero_grad()`` or ``optimizer.zero_grad()``
that may improve performance for some networks.

Manual gradient layouts
-----------------------

If you need manual control over ``.grad``'s strides,
assign ``param.grad =`` a zeroed tensor with desired strides
before the first ``backward()``, and never reset it to ``None``.
3 guarantees your layout is preserved as long as ``create_graph=False``.
4 indicates your layout is *likely* preserved even if ``create_graph=True``.

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
.. autosummary::
    :nosignatures:

   torch.Tensor.grad
   torch.Tensor.requires_grad
   torch.Tensor.is_leaf
   torch.Tensor.backward
   torch.Tensor.detach
   torch.Tensor.detach_
   torch.Tensor.register_hook
   torch.Tensor.register_post_accumulate_grad_hook
   torch.Tensor.retain_grad

:hidden:`Function`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: Function

.. autosummary::
    :toctree: generated
    :nosignatures:

    Function.forward
    Function.backward
    Function.jvp
    Function.vmap

Context method mixins
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When creating a new :class:`Function`, the following methods are available to `ctx`.

.. autosummary::
    :toctree: generated
    :nosignatures:

    function.FunctionCtx.mark_dirty
    function.FunctionCtx.mark_non_differentiable
    function.FunctionCtx.save_for_backward
    function.FunctionCtx.set_materialize_grads

Custom Function utilities
^^^^^^^^^^^^^^^^^^^^^^^^^
Decorator for backward method.

.. autosummary::
    :toctree: generated
    :nosignatures:

    function.once_differentiable

Base custom :class:`Function` used to build PyTorch utilities

.. autosummary::
    :toctree: generated
    :nosignatures:

    function.BackwardCFunction
    function.InplaceFunction
    function.NestedIOFunction


.. _grad-check:

Numerical gradient checking
^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. automodule:: torch.autograd.gradcheck
.. currentmodule:: torch.autograd.gradcheck

.. autosummary::
    :toctree: generated
    :nosignatures:

    gradcheck
    gradgradcheck
    GradcheckError

.. Just to reset the base path for the rest of this file
.. currentmodule:: torch.autograd

Profiler
^^^^^^^^

Autograd includes a profiler that lets you inspect the cost of different
operators inside your model - both on the CPU and GPU. There are three modes
implemented at the moment - CPU-only using :class:`~torch.autograd.profiler.profile`.
nvprof based (registers both CPU and GPU activity) using
:class:`~torch.autograd.profiler.emit_nvtx`.
and vtune profiler based using
:class:`~torch.autograd.profiler.emit_itt`.

.. autoclass:: torch.autograd.profiler.profile

.. autosummary::
    :toctree: generated
    :nosignatures:

    profiler.profile.export_chrome_trace
    profiler.profile.key_averages
    profiler.profile.self_cpu_time_total
    profiler.profile.total_average
    profiler.parse_nvprof_trace
    profiler.EnforceUnique
    profiler.KinetoStepTracker
    profiler.record_function
    profiler_util.Interval
    profiler_util.Kernel
    profiler_util.MemRecordsAcc
    profiler_util.StringTable

.. autoclass:: torch.autograd.profiler.emit_nvtx
.. autoclass:: torch.autograd.profiler.emit_itt


.. autosummary::
    :toctree: generated
    :nosignatures:

    profiler.load_nvprof

Debugging and anomaly detection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: detect_anomaly

.. autoclass:: set_detect_anomaly

.. autosummary::
    :toctree: generated
    :nosignatures:

    grad_mode.set_multithreading_enabled



Autograd graph
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Autograd exposes methods that allow one to inspect the graph and interpose behavior during
the backward pass.

The ``grad_fn`` attribute of a :class:`torch.Tensor` holds a  :class:`torch.autograd.graph.Node`
if the tensor is the output of a operation that was recorded by autograd (i.e., grad_mode is
enabled and at least one of the inputs required gradients), or ``None`` otherwise.

.. autosummary::
    :toctree: generated
    :nosignatures:

    graph.Node.name
    graph.Node.metadata
    graph.Node.next_functions
    graph.Node.register_hook
    graph.Node.register_prehook
    graph.increment_version

Some operations need intermediary results to be saved during the forward pass
in order to execute the backward pass.
These intermediary results are saved as attributes on the ``grad_fn`` and can be accessed.
For example::

    >>> a = torch.tensor([0., 0., 0.], requires_grad=True)
    >>> b = a.exp()
    >>> print(isinstance(b.grad_fn, torch.autograd.graph.Node))
    True
    >>> print(dir(b.grad_fn))
    ['__call__', '__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '_raw_saved_result', '_register_hook_dict', '_saved_result', 'metadata', 'name', 'next_functions', 'register_hook', 'register_prehook', 'requires_grad']
    >>> print(torch.allclose(b.grad_fn._saved_result, b))
    True

You can also define how these saved tensors should be packed / unpacked using hooks.
A common application is to trade compute for memory by saving those intermediary results
to disk or to CPU instead of leaving them on the GPU. This is especially useful if you
notice your model fits on GPU during evaluation, but not training.
Also see :ref:`saved-tensors-hooks-doc`.

.. autoclass:: torch.autograd.graph.saved_tensors_hooks

.. autoclass:: torch.autograd.graph.save_on_cpu

.. autoclass:: torch.autograd.graph.disable_saved_tensors_hooks

.. autoclass:: torch.autograd.graph.register_multi_grad_hook

.. autoclass:: torch.autograd.graph.allow_mutation_on_saved_tensors

.. autoclass:: torch.autograd.graph.GradientEdge

.. autofunction:: torch.autograd.graph.get_gradient_edge



.. This module needs to be documented. Adding here in the meantime
.. for tracking purposes
.. py:module:: torch.autograd.anomaly_mode
.. py:module:: torch.autograd.forward_ad
.. py:module:: torch.autograd.function
.. py:module:: torch.autograd.functional
.. py:module:: torch.autograd.grad_mode
.. py:module:: torch.autograd.graph
.. py:module:: torch.autograd.profiler
.. py:module:: torch.autograd.profiler_legacy
.. py:module:: torch.autograd.profiler_util
.. py:module:: torch.autograd.variable
