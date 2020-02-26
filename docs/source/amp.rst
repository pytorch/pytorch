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

When training a network with mixed precision, if the forward pass for a particular op has
``torch.float16`` inputs, the backward pass for that op will produce ``torch.float16`` gradients.
Gradient values with small magnitudes may not be representable in ``torch.float16``.
These values will flush to zero ("underflow"), so the update for the corresponding parameters will be lost.

To prevent underflow, "gradient scaling" multiplies the network's loss(es) by a scale factor and
invokes a backward pass on the scaled loss(es).  Gradients flowing backward through the network are
then scaled by the same factor.  In other words, gradient values have a larger magnitude,
so they don't flush to zero.

The parameters' gradients (``.grad`` attributes) should be unscaled before the optimizer uses them
to update the parameters, so the scale factor does not interfere with the learning rate.

.. autoclass:: GradScaler
    :members:
