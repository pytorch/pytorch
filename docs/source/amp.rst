.. role:: hidden
    :class: hidden-section

Automatic Mixed Precision package - torch.cuda.amp
==================================================

.. automodule:: torch.cuda.amp
.. currentmodule:: torch.cuda.amp

``torch.cuda.amp`` provides convenience methods for mixed precision,
where some operations use the ``torch.float32`` (``float``) datatype and other operations
use ``torch.float16`` (``half``). Some operations, like linear layers and convolutions,
are much faster in ``float16``. Other operations, like reductions, often require the dynamic
range of ``float32``.  Mixed precision tries to match each operation to its appropriate datatype.

When training with a mixture of ``float32`` and ``float16``, you should use
:class:`torch.cuda.amp.autocast` along with :class:`torch.cuda.amp.GradScaler`.

:class:`torch.cuda.amp.autocast` and :class:`torch.cuda.amp.GradScaler` are modular.
When combining them, there are no gotchas:  use each as its individual docs suggest.
See the :ref:`Automatic Mixed Precision examples<amp-examples>` for typical
and more advanced cases.

.. contents:: :local:

.. _autocasting:

Autocasting
^^^^^^^^^^^

.. autoclass:: autocast
    :members:

.. _gradient-scaling:

Gradient Scaling
^^^^^^^^^^^^^^^^

If the forward pass for a particular op has ``float16`` inputs, the backward pass for
that op will produce ``float16`` gradients.
Gradient values with small magnitudes may not be representable in ``float16``.
These values will flush to zero ("underflow"), so the update for the corresponding parameters will be lost.

To prevent underflow, "gradient scaling" multiplies the network's loss(es) by a scale factor and
invokes a backward pass on the scaled loss(es).  Gradients flowing backward through the network are
then scaled by the same factor.  In other words, gradient values have a larger magnitude,
so they don't flush to zero.

The parameters' gradients (``.grad`` attributes) should be unscaled before the optimizer uses them
to update the parameters, so the scale factor does not interfere with the learning rate.

.. autoclass:: GradScaler
    :members:
