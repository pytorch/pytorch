.. role:: hidden
    :class: hidden-section

Automatic Mixed Precision package - torch.cuda.amp
==================================================

.. automodule:: torch.cuda.amp
.. currentmodule:: torch.cuda.amp

``torch.cuda.amp`` provides convenience methods for running networks with mixed precision,
where some operations use the ``torch.float32`` (``float``) datatype and other operations
use ``torch.float16`` (``half``). Some operations, like linear layers and convolutions,
are much faster in ``float16``. Other operations, like reductions, often require the dynamic
range of ``float32``. Networks running in mixed precision try to match each operation to its appropriate datatype.

By default, you don't need to call ``.half()`` on your model(s) or data to use the routines below.
In fact, you shouldn't.  Model weights should remain ``float32``.

Mixed precision should not require retuning any hyperparameters, as long as the conventions shown in the
:ref:`Automatic Mixed Precision Examples<amp-examples>` are obeyed.

.. contents:: :local:

Gradient Scaling
^^^^^^^^^^^^^^^^

When training a network in mixed precision, gradients may become too small to represent as
``torch.float16`` values. Scaling gradients during backward prevents such underflow.
Scaled gradients are unscaled before being applied.

.. autoclass:: AmpScaler
    :members:
