.. role:: hidden
    :class: hidden-section

Automatic Mixed Precision package - torch.cuda.amp
==================================================

.. automodule:: torch.cuda.amp
.. currentmodule:: torch.cuda.amp

``torch.cuda.amp`` provides convenience methods for mixed precision.  Mixed precision uses ``torch.float16``
(a.k.a. ``torch.half``) for some operations like linear layers and convolutions, to improve throughput
and reduce the memory footprint.  Operations that require additional precision and range, like reductions,
are carried out in ``torch.float32`` (a.k.a. ``torch.float``).
On Nvidia GPUs, mixed precision can improve performance.

By default, you don't need to call ``.half()`` on your model(s) or data to use the routines below.
In fact, you shouldn't.  Model weights should remain ``float32``.

Mixed precision should not require retuning any hyperparameters, as long as the conventions shown in the
:ref:`Automatic Mixed Precision Examples<amp-examples>` are obeyed.

.. contents:: :local:

Gradient Scaling
^^^^^^^^^^^^^^^^

.. autoclass:: AmpScaler
    :members:
