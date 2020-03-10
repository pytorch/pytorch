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

The following lists detail the behavior of particular ops in autocast-enabled regions.

Autocast affects only CUDA ops.

Autocast affects only out-of-place ops and Tensor methods.
In-place variants and calls that explicitly supply an `out=...` Tensor
are allowed in autocast-enabled regions, but won't receive autocasting.
For example, in an autocast-enabled region `a.addmm(b, c)` is guaranteed to run
in ``float16``, but `a.addmm_(b, c)` and `a.addmm(b, c, out=d)` may not.
For best performance and stability, prefer out-of-place ops in autocast-enabled
regions.

Ops not listed below do not receive autocasting.  They run in the type
defined by their inputs.  However, autocasting may still change the type
in which unlisted ops run if they're downstream from autocasted ops.

If an op is unlisted, we assume it's safe to run in ``float16``` without impairing
convergence.  If you encounter an unlisted op that causes convergence problems
in ``float16``, please file an issue.

Ops that run in `float16`
-------------------------
In autocast-enabled regions, these always execute in ``float16`` and produce ``float16`` output.

In module ``torch``
"""""""""""""""""""
.. currentmodule:: torch
:class:`Tensor` methods with the same name, if present, are autocasted equivalently.

In module ``torch.nn.functional``
"""""""""""""""""""""""""""""""""
:func:`linear`

Ops that run in ``float32``
---------------------------
In autocast-enabled regions, these always execute in ``float32`` and produce ``float32`` output.

``torch`` functions
"""""""""""""""""""
.. currentmodule:: torch
:class:`Tensor` methods with the same name, if present, are autocasted equivalently.

``torch.nn.functional`` functions
"""""""""""""""""""""""""""""""""
.. currentmodule:: torch.nn.functional

Ops that run in the widest input type
-------------------------------------
These ops don't require a particular dtype for stability, but take multiple inputs
and require that the inputs' dtypes match.  In autocast-enabled regions, all inputs are automatically
casted to match the widest dtype among the inputs.  The op executes and produces output with that
widest dtype.

In module ``torch``
"""""""""""""""""""
.. currentmodule:: torch
:class:`Tensor` methods with the same name, if present, are autocasted equivalently.

:func:`addcdiv`,
:func:`addcmul`,
:func:`atan2`,
:func:`bilinear`,
:func:`cat`,
:func:`cross`,
:func:`dot`,
:func:`equal`,
:func:`stack`,
:func:`tensordot`
