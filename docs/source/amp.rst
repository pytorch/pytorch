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

Ordinarily, "automatic mixed precision training" uses :class:`torch.cuda.amp.autocast` and
:class:`torch.cuda.amp.GradScaler` together, as shown in the :ref:`Automatic Mixed Precision examples<amp-examples>`.
However, :class:`autocast` and :class:`GradScaler` are modular, and may be used separately if desired.

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

Each parameter's gradient (``.grad`` attribute) should be unscaled before the optimizer
updates the parameters, so the scale factor does not interfere with the learning rate.

.. autoclass:: GradScaler
    :members:

.. _autocast-policies:

Autocast Op Reference
^^^^^^^^^^^^^^^^^^^^^

For ops listed below, autocast explicitly
chooses a precision in autocast-enabled regions.

Autocast only affects CUDA ops.
Autocast affects only out-of-place ops and Tensor methods.
for each CUDA operation in an autocasted-enabled region
Some ops cast internally to the kernel.
Some ops natively promote.

Unlisted ops do not receive explicit autocasting.  They run in whatever dtype
their input(s) happen to be, and the output dtype matches the input dtype.
An exception is is

Ops that run in `float32`
-------------------------

Ops that run in `float16`
-------------------------
CUDA ops on this list are faster in ``float16`` without sacrificing stability.
Autocast calls ``.half()`` on their inputs to ensure they run in ``float16``.
Their output will always

Ops that run in the widest input type
-------------------------------------
These don't require a particular precision for stability, but take multiple inputs
and require that the inputs' dtypes match.  Because autocast may change the input
