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
:class:`torch.cuda.amp.GradScaler` together, as shown in the :ref:`Automatic Mixed Precision examples<amp-examples>`
and `Automatic Mixed Precision recipe <https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html>`_.
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

.. _autocast-op-reference:

Autocast Op Reference
^^^^^^^^^^^^^^^^^^^^^

.. _autocast-eligibility:

Op Eligibility
--------------
Only CUDA ops are eligible for autocasting.

Ops that run in ``float64`` or non-floating-point dtypes are not eligible, and will
run in these types whether or not autocast is enabled.

Only out-of-place ops and Tensor methods are eligible.
In-place variants and calls that explicitly supply an ``out=...`` Tensor
are allowed in autocast-enabled regions, but won't go through autocasting.
For example, in an autocast-enabled region ``a.addmm(b, c)`` can autocast,
but ``a.addmm_(b, c)`` and ``a.addmm(b, c, out=d)`` cannot.
For best performance and stability, prefer out-of-place ops in autocast-enabled
regions.

Ops called with an explicit ``dtype=...`` argument are not eligible,
and will produce output that respects the ``dtype`` argument.

Op-Specific Behavior
--------------------
The following lists describe the behavior of eligible ops in autocast-enabled regions.
These ops always go through autocasting whether they are invoked as part of a :class:`torch.nn.Module`,
as a function, or as a :class:`torch.Tensor` method. If functions are exposed in multiple namespaces,
they go through autocasting regardless of the namespace.

Ops not listed below do not go through autocasting.  They run in the type
defined by their inputs.  However, autocasting may still change the type
in which unlisted ops run if they're downstream from autocasted ops.

If an op is unlisted, we assume it's numerically stable in ``float16``.
If you believe an unlisted op is numerically unstable in ``float16``,
please file an issue.

Ops that can autocast to ``float16``
""""""""""""""""""""""""""""""""""""

``__matmul__``,
``addbmm``,
``addmm``,
``addmv``,
``addr``,
``baddbmm``,
``bmm``,
``chain_matmul``,
``multi_dot``,
``conv1d``,
``conv2d``,
``conv3d``,
``conv_transpose1d``,
``conv_transpose2d``,
``conv_transpose3d``,
``GRUCell``,
``linear``,
``LSTMCell``,
``matmul``,
``mm``,
``mv``,
``prelu``,
``RNNCell``

Ops that can autocast to ``float32``
""""""""""""""""""""""""""""""""""""

``__pow__``,
``__rdiv__``,
``__rpow__``,
``__rtruediv__``,
``acos``,
``asin``,
``binary_cross_entropy_with_logits``,
``cosh``,
``cosine_embedding_loss``,
``cdist``,
``cosine_similarity``,
``cross_entropy``,
``cumprod``,
``cumsum``,
``dist``,
``erfinv``,
``exp``,
``expm1``,
``gelu``,
``group_norm``,
``hinge_embedding_loss``,
``kl_div``,
``l1_loss``,
``layer_norm``,
``log``,
``log_softmax``,
``log10``,
``log1p``,
``log2``,
``margin_ranking_loss``,
``mse_loss``,
``multilabel_margin_loss``,
``multi_margin_loss``,
``nll_loss``,
``norm``,
``normalize``,
``pdist``,
``poisson_nll_loss``,
``pow``,
``prod``,
``reciprocal``,
``rsqrt``,
``sinh``,
``smooth_l1_loss``,
``soft_margin_loss``,
``softmax``,
``softmin``,
``softplus``,
``sum``,
``renorm``,
``tan``,
``triplet_margin_loss``

Ops that promote to the widest input type
"""""""""""""""""""""""""""""""""""""""""
These ops don't require a particular dtype for stability, but take multiple inputs
and require that the inputs' dtypes match.  If all of the inputs are
``float16``, the op runs in ``float16``.  If any of the inputs is ``float32``,
autocast casts all inputs to ``float32`` and runs the op in ``float32``.

``addcdiv``,
``addcmul``,
``atan2``,
``bilinear``,
``cat``,
``cross``,
``dot``,
``equal``,
``index_put``,
``scatter_add``,
``stack``,
``tensordot``

Some ops not listed here (e.g., binary ops like ``add``) natively promote
inputs without autocasting's intervention.  If inputs are a mixture of ``float16``
and ``float32``, these ops run in ``float32`` and produce ``float32`` output,
regardless of whether autocast is enabled.

Prefer ``binary_cross_entropy_with_logits`` over ``binary_cross_entropy``
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
The backward passes of :func:`torch.nn.functional.binary_cross_entropy` (and :mod:`torch.nn.BCELoss`, which wraps it)
can produce gradients that aren't representable in ``float16``.  In autocast-enabled regions, the forward input
may be ``float16``, which means the backward gradient must be representable in ``float16`` (autocasting ``float16``
forward inputs to ``float32`` doesn't help, because that cast must be reversed in backward).
Therefore, ``binary_cross_entropy`` and ``BCELoss`` raise an error in autocast-enabled regions.

Many models use a sigmoid layer right before the binary cross entropy layer.
In this case, combine the two layers using :func:`torch.nn.functional.binary_cross_entropy_with_logits`
or :mod:`torch.nn.BCEWithLogitsLoss`.  ``binary_cross_entropy_with_logits`` and ``BCEWithLogits``
are safe to autocast.
