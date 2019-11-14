.. _amp-examples:

Automatic Mixed Precision Examples
==================================

.. currentmodule:: torch.cuda.amp

.. contents:: :local:

.. _gradient-scaling-examples:

Gradient Scaling
^^^^^^^^^^^^^^^^

Gradient scaling helps ensure convergence when training with mixed precision,
as explained :ref:`here<gradient-scaling>`.

The following code snippets show how :class:`torch.cuda.amp.AmpScaler` can be used to
perform dynamic gradient scaling automatically.

Working with a Single Optimizer
-------------------------------

::

    # Create an AmpScaler instance.
    scaler = AmpScaler()
    ...
    for input, target in data:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)

        # Scale the loss, and call backward() on the scaled loss to create scaled gradients.
        scaler.scale(loss).backward()

        # Carry out a scaling-safe step.  scaler.step() unscales the optimizer's gradients
        # and skips optimizer.step() if the gradients contain infs or NaNs.
        scaler.step(optimizer)

        # Update the scale for next iteration.
        scaler.update()


Working with Unscaled Gradients
-------------------------------

Gradients resulting from ``scaler.scale(loss).backward()`` are scaled.  If you wish to modify or inspect
these gradients between ``backward()`` and ``scaler.step(optimizer)``,  you should unscale them first.
For example, gradient clipping manipulates a set of gradients such that their global norm
(see :func:`torch.nn.utils.clip_grad_norm_`) or maximum magnitude (see :func:`torch.nn.utils.clip_grad_value_`)
is :math:`<=` some user-imposed threshold.  If you attempted to clip _without_ unscaling, the gradients' norm/maximum
magnitude would also be scaled, so your requested threshold (which was meant to be the threshold for _unscaled_ gradients)
would be invalid.

``scaler.unscale(optimizer)`` unscales gradients held by ``optimizer``'s assigned parameters.
If your model or models contain other parameters that were assigned to another optimizer
(say ``optimizer1``), you may call ``scaler.unscale(optimizer1)`` separately to unscale those
parameters' gradients as well.

Gradient clipping
"""""""""""""""""

Calling ``scaler.unscale(optimizer)`` before clipping enables you to clip unscaled gradients as usual::

    scaler = AmpScaler()
    ...
    for input, target in data:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        scaler.scale(loss).backward()

        # Unscale the gradients owned by optimizer in-place
        scaler.unscale(optimizer)

        # Since the optimizer's owned gradients are unscaled, clip as usual:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        scaler.step(optimizer)
        scaler.update()

``scaler`` records that ``scaler.unscale(optimizer)`` has been called for this optimizer
this iteration, so ``scaler.step(optimizer)`` knows not to redundantly unscale gradients before
calling ``optimizer.step()``.

.. warning::
    If you (optionally) choose to unscale gradients prior to stepping, ``scaler.unscale(optimizer)``
    should only be invoked once per optimizer per step, and only after all gradients for that optimizer's
    owned parameters have been accumulated.

Working with Scaled Gradients
-----------------------------

For some operations, you may need to work with scaled gradients in a setting where
``scaler.unscale`` is unsuitable.

Gradient penalty
""""""""""""""""

A gradient penalty loss term requires manipulating scaled gradients.
The gradient penalty example below demonstrates

* creating scaled out-of-place gradients with :func:`torch.autograd.grad`, and
* correct interaction of gradient scaling with double-backward.

Here's how that looks for a simple L2 penalty::

    scaler = AmpScaler()
    ...
    for input, target in data:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)

        # We should scale outputs for the out-of-place backward pass
        grad_params = torch.autograd.grad(scaler.scale(loss), model.parameters(), create_graph=True)

        # In general, the penalty term may depend nonlinearly on grad_params, so to be safe,
        # manually unscale them before computing the penalty.  The unscale should be out-of-place
        # and autograd-exposed.  For these reasons, and because grad_params are not owned by any optimizer,
        # calling scaler.unscale(optimizer) here is unsuitable, and we use ordinary ops instead:
        inv_scale = 1./scaler.get_scale()
        grad_params = [p*inv_scale for p in grad_params]

        # Compute the penalty term and add it to the loss
        grad_norm = 0
        for grad in grad_params:
            grad_norm += grad.pow(2).sum()
        grad_norm = grad_norm.sqrt()
        loss = loss + grad_norm

        # The usual scaling for backward will now accumulate leaf gradients that are appropriately scaled.
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


Working with Multiple Optimizers, Models, and Losses
----------------------------------------------------

Working with multiple optimizers is just like working with a single optimizer.
It's important, however, that ``scaler.update()`` only be called after all
optimizers used this iteration have been stepped::

    scaler = torch.cuda.amp.AmpScaler()
    ...
    for input, target in data:
        optimizer0.zero_grad()
        optimizer1.zero_grad()
        output0 = model0(input)
        output1 = model1(input)
        loss0 = loss_fn(2 * output0 + 3 * output1, target)
        loss1 = loss_fn(3 * output0 - 5 * output1, target)

        scaler.scale(loss0).backward(retain_graph=True)
        scaler.scale(loss1).backward()

        # You can choose which optimizers receive explicit unscaling
        scaler.unscale(optimizer0)

        scaler.step(optimizer0)
        scaler.step(optimizer1)

        scaler.update()
