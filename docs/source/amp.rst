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

.. contents:: :local:

.. _gradient-scaling:

Gradient Scaling
^^^^^^^^^^^^^^^^

When training a network with mixed precision, gradient magnitudes may become too small to represent
in ``torch.float16`` regions of the backward pass.  "Gradient scaling" multiplies outputs by
a scale factor ``S`` and invokes a backward pass on the scaled outputs.  Gradients flowing
backward through the network are then scaled by ``S``, which prevents such underflow.
Scaled leaf gradients are unscaled (divided by ``S``) before being used to step the parameters.
Unscaling at the leaves will not incur underflow because the leaves are ``torch.float32``.

.. autoclass:: AmpScaler
    :members:
