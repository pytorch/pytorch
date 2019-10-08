.. role:: hidden
    :class: hidden-section

Automatic Mixed Precision package - torch.amp
==================================================

.. automodule:: torch.cuda.amp
.. currentmodule:: torch.cuda.amp

``torch.amp`` provides convenience methods for "mixed precision:"  using ``torch.float16`` for some operations
and ``torch.float32`` for others.  TODO discuss how we want to summarize this...

The two ingredients of the mixed precision recipe are `Autocasting`_ and `Gradient Scaling`_.

By default, you don't need to call ``.half()`` on your model(s) or data to use the routines below.  In fact, you shouldn't:
model params should remain FP32.

Turning on mixed precision should not require retuning any hyperparameters, as long as the conventions shown in the
:ref:`Automatic Mixed Precision Examples` are obeyed.

.. contents:: :local:

Autocasting
^^^^^^^^^^^

Under construction...

Gradient Scaling
^^^^^^^^^^^^^^^^

For additional guidance, please review whether any of the :ref:`Gradient Scaling Examples` match your use case.

.. autoclass:: AmpScaler
    :members:
