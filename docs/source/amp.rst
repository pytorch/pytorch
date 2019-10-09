.. role:: hidden
    :class: hidden-section

Automatic Mixed Precision package - torch.cuda.amp
==================================================

.. automodule:: torch.cuda.amp
.. currentmodule:: torch.cuda.amp

``torch.cuda.amp`` provides convenience methods for "mixed precision:"  using ``torch.float16`` for some operations
and ``torch.float32`` for others.  TODO discuss how we want to summarize this...right balance of technical and quick.

The two ingredients of the mixed precision recipe are

* `Autocasting`_, to ensure each operation runs in its optimal precision, and
* `Gradient Scaling`_, to mitigate FP16 gradient underflow during the backward pass.

By default, you don't need to call ``.half()`` on your model(s) or data to use the routines below.
In fact, you shouldn't.  Model weights should remain FP32.

Turning on mixed precision should not require retuning any hyperparameters, as long as the conventions shown in the
:ref:`Automatic Mixed Precision Examples<amp-examples>` are obeyed.

.. contents:: :local:

Autocasting
^^^^^^^^^^^

Under construction...

Gradient Scaling
^^^^^^^^^^^^^^^^

For additional guidance, please review whether any of the
:ref:`Gradient Scaling Examples<gradient-scaling-examples>` match your use case.

.. autoclass:: AmpScaler
    :members:

Custom Optimizer Guide
^^^^^^^^^^^^^^^^^^^^^^

Existing optimizers (native and custom) can be used safely with the autocasting and gradient scaling APIs without
any changes.  However, the gradient scaling API gives custom optimizer authors the option to define a scaling-safe
``step`` method, which :meth:`AmpScaler.step` will call directly.  This permits certain ninja performance optimizations
like sync-free stepping.

The :ref:`Custom Optimizer Guide<custom-optimizer-guide>` defines the contract such custom optimizers should obey.
