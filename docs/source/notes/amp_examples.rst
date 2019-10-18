.. _amp-examples:

Automatic Mixed Precision Examples
==================================

.. currentmodule:: torch.cuda.amp

.. contents:: :local:

.. _gradient-scaling-examples:

Gradient Scaling
^^^^^^^^^^^^^^^^

The code snippets below demonstrate recommended use of :class:`torch.cuda.amp.AmpScaler`.
The :class:`AmpScaler` instance ``scaler`` performs dynamic gradient scaling
(maintains the scale, helps create scaled gradients to prevent
gradient underflow, and carries out scaling-safe steps).

Typical use (1 loss, 1 optimizer)
---------------------------------

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


Scale-Aware Operations
----------------------

The gradients resulting from ``scaler.scale(loss).backward()`` are scaled.  ``scaler.step(optimizer)``
automatically knows if it must unscale gradients before applying them.  However, networks that directly use
gradients between the backward pass and the optimizer step should also be aware of the scale factor.

Gradient clipping
"""""""""""""""""

Clipping is an operation that directly manipulates the gradients.
:meth:`AmpScaler.unscale` may be used to unscale the gradients in-place before ``scaler.step(optimizer)``,
allowing you to clip unscaled gradients as usual::

    scaler = AmpScaler()
    ...
    for input, target in data:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        scaler.scale(loss).backward()

        # Unscale the gradients owned by optimizer in-place
        scaler.unscale(optimizer)

        # Since the optimizer's owned gradients are unscaled, we can clip as usual:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        scaler.step(optimizer)
        scaler.update()

``scaler`` records that ``scaler.unscale(optimizer)`` has been called for this optimizer
this iteration, so ``scaler.step(optimizer)`` knows not to redundantly unscale gradients before
calling ``optimizer.step()``.

Gradient penalty
""""""""""""""""

A gradient penalty loss term also requires awareness of the scale factor.
Gradient penalty demonstrates the following use cases:

* Creating out-of-place gradients with :func:`torch.autograd.grad`
* Correct interaction of gradient scaling with double-backward

Here's how that looks for a simple L2 penalty::

    scaler = AmpScaler()
    ...
    for input, target in data:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)

        # We should scale outputs for the out-of-place backward pass
        grad_params = torch.autograd.grad(scaler.scale(loss), model.parameters(), create_graph=True)

        # In general, the penalty term may depend nonlinearly on the out-of-place gradients, so to be safe,
        # manually unscale them before computing the penalty.  This unscale should be out-of-place and autograd-exposed.
        grad_params = [p*(1./scaler.get_scale()) for p in grad_params]

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


Gradient accumulation
---------------------

Gradient accumulation across iterations (between steps) can be implemented as follows::

    scaler = AmpScaler()
    ...
    for i, (input, target) in enumerate(data):
        output = model(input)
        loss = loss_fn(output, target)
        loss = loss/iters_to_accumulate
        scaler.scale(loss).backward()
        if (i + 1) % iters_to_accumulate == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

Switching gradient scaling on and off
------------------------------------

You can enable/disable gradient scaling globally (everywhere in the network a given :class:`AmpScaler` instance
is used) by supplying a single ``enabled=True|False`` flag to that instance's constructor call::

    scaler = AmpScaler(enabled=args.use_mixed_precision)
    ...
    for input, target in data:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

If ``enabled=False``, ``scaler.step(optimizer)`` is equivalent to ``optimizer.step()``, and
the other methods (``scaler.scale``, ``scaler.update``) become no-ops.


Working with Multiple Optimizers, Models, and Losses
----------------------------------------------------

A single :class:`AmpScaler` instance should scale all losses and step all optimizers.
The usage is equivalent to prior single-optimizer examples.  The only additional constraint is
that ``scaler.update()`` may only be called after all optimizers used this iteration have been stepped::

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

Note that the decision to invoke an explicit ``unscale`` can be made independently for each optimizer.

.. warning::
    If you (optionally) choose to unscale gradients prior to stepping, ``scaler.unscale(optimizer)``
    should only be invoked once per optimizer per step, and only after all gradients for that optimizer's
    owned parameters have been accumulated.
