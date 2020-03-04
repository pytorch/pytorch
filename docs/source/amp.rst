.. role:: hidden
    :class: hidden-section

Automatic Mixed Precision package - torch.cuda.amp
==================================================

.. automodule:: torch.cuda.amp
.. currentmodule:: torch.cuda.amp

``torch.cuda.amp`` provides convenience methods for mixed precision,
where some operations use the ``torch.float32`` (``float``) datatype and other operations
use ``torch.float16`` (``half``). Some ops, like linear layers and convolutions,
are much faster in ``float16``. Other ops, like reductions, often require the dynamic
range of ``float32``.  Mixed precision tries to match each op to its appropriate datatype.

:class:`torch.cuda.amp.autocast` automatically chooses the precision for GPU operations
in specified regions of your network.

:class:`torch.cuda.amp.GradScaler` improves convergence for networks with ``float16``
gradients by minimizing gradient underflow.

Ordinarily, "automatic mixed precision training" uses :class:`autocast` and :class:`GradScaler` together,
as shown in the :ref:`Automatic Mixed Precision examples<amp-examples>`.
However, :class:`torch.cuda.amp.autocast` and :class:`torch.cuda.amp.GradScaler` are modular, and may be used
separately if desired.

.. contents:: :local:

.. _autocasting:

Autocasting
^^^^^^^^^^^

.. autoclass:: autocast
    :members:

.. autofunction::  custom_fwd

.. autofunction::  custom_bwd

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

The parameters' gradient (``.grad`` attributes) should be unscaled before the optimizer uses them
to update the parameters, so the scale factor does not interfere with the learning rate.

.. autoclass:: GradScaler
    :members:
