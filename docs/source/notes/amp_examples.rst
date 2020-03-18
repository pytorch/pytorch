.. _amp-examples:

Automatic Mixed Precision examples
==================================

.. currentmodule:: torch.cuda.amp

.. contents:: :local:

.. _gradient-scaling-examples:

Gradient Scaling
^^^^^^^^^^^^^^^^

Gradient scaling helps prevent gradient underflow when training with mixed precision,
as explained :ref:`here<gradient-scaling>`.

Instances of :class:`torch.cuda.amp.GradScaler` help perform the steps of
gradient scaling conveniently, as shown in the following code snippets.


Typical Use
-----------

::

    # Creates a GradScaler once at the beginning of training.
    scaler = GradScaler()

    for epoch in epochs:
        for input, target in data:
            optimizer.zero_grad()
            output = model(input)
            loss = loss_fn(output, target)

            # Scales the loss, and calls backward() on the scaled loss to create scaled gradients.
            scaler.scale(loss).backward()

            # scaler.step() first unscales the gradients of the optimizer's assigned params.
            # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            scaler.step(optimizer)

            # Updates the scale for next iteration.
            scaler.update()

.. _working-with-unscaled-gradients:

Working with Unscaled Gradients
-------------------------------

All gradients produced by ``scaler.scale(loss).backward()`` are scaled.  If you wish to modify or inspect
the parameters' ``.grad`` attributes between ``backward()`` and ``scaler.step(optimizer)``,  you should
unscale them first.  For example, gradient clipping manipulates a set of gradients such that their global norm
(see :func:`torch.nn.utils.clip_grad_norm_`) or maximum magnitude (see :func:`torch.nn.utils.clip_grad_value_`)
is :math:`<=` some user-imposed threshold.  If you attempted to clip *without* unscaling, the gradients' norm/maximum
magnitude would also be scaled, so your requested threshold (which was meant to be the threshold for *unscaled*
gradients) would be invalid.

``scaler.unscale_(optimizer)`` unscales gradients held by ``optimizer``'s assigned parameters.
If your model or models contain other parameters that were assigned to another optimizer
(say ``optimizer2``), you may call ``scaler.unscale_(optimizer2)`` separately to unscale those
parameters' gradients as well.

Gradient clipping
"""""""""""""""""

Calling ``scaler.unscale_(optimizer)`` before clipping enables you to clip unscaled gradients as usual::

    scaler = GradScaler()

    for epoch in epochs:
        for input, target in data:
            optimizer.zero_grad()
            output = model(input)
            loss = loss_fn(output, target)
            scaler.scale(loss).backward()

            # Unscales the gradients of optimizer's assigned params in-place
            scaler.unscale_(optimizer)

            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
            # although it still skips optimizer.step() if the gradients contain infs or NaNs.
            scaler.step(optimizer)

            # Updates the scale for next iteration.
            scaler.update()

``scaler`` records that ``scaler.unscale_(optimizer)`` was already called for this optimizer
this iteration, so ``scaler.step(optimizer)`` knows not to redundantly unscale gradients before
(internally) calling ``optimizer.step()``.

.. warning::
    :meth:`unscale_` should only be called once per optimizer per :meth:`step` call,
    and only after all gradients for that optimizer's assigned parameters have been accumulated.
    Calling :meth:`unscale_` twice for a given optimizer between each :meth:`step` triggers a RuntimeError.

Working with Scaled Gradients
-----------------------------

For some operations, you may need to work with scaled gradients in a setting where
``scaler.unscale_`` is unsuitable.

Gradient penalty
""""""""""""""""

A gradient penalty implementation typically creates gradients out-of-place using
:func:`torch.autograd.grad`, combines them to create the penalty value,
and adds the penalty value to the loss.

Here's an ordinary example of an L2 penalty without gradient scaling::

    for epoch in epochs:
        for input, target in data:
            optimizer.zero_grad()
            output = model(input)
            loss = loss_fn(output, target)

            # Creates some gradients out-of-place
            grad_params = torch.autograd.grad(loss, model.parameters(), create_graph=True)

            # Computes the penalty term and adds it to the loss
            grad_norm = 0
            for grad in grad_params:
                grad_norm += grad.pow(2).sum()
            grad_norm = grad_norm.sqrt()
            loss = loss + grad_norm

            loss.backward()
            optimizer.step()

To implement a gradient penalty *with* gradient scaling, the loss passed to
:func:`torch.autograd.grad` should be scaled.  The resulting out-of-place gradients
will therefore be scaled, and should be unscaled before being combined to create the
penalty value.

Here's how that looks for the same L2 penalty::

    scaler = GradScaler()

    for epoch in epochs:
        for input, target in data:
            optimizer.zero_grad()
            output = model(input)
            loss = loss_fn(output, target)

            # Scales the loss for the out-of-place backward pass, resulting in scaled grad_params
            scaled_grad_params = torch.autograd.grad(scaler.scale(loss), model.parameters(), create_graph=True)

            # Unscales grad_params before computing the penalty.  grad_params are not owned
            # by any optimizer, so ordinary division is used instead of scaler.unscale_:
            inv_scale = 1./scaler.get_scale()
            grad_params = [p*inv_scale for p in scaled_grad_params]

            # Computes the penalty term and adds it to the loss
            grad_norm = 0
            for grad in grad_params:
                grad_norm += grad.pow(2).sum()
            grad_norm = grad_norm.sqrt()
            loss = loss + grad_norm

            # Applies scaling to the backward call as usual.  Accumulates leaf gradients that are correctly scaled.
            scaler.scale(loss).backward()

            # step() and update() proceed as usual.
            scaler.step(optimizer)
            scaler.update()


Working with Multiple Losses and Optimizers
-------------------------------------------

If your network has multiple losses, you must call ``scaler.scale`` on each of them individually.
If your network has multiple optimizers, you may call ``scaler.unscale_`` on any of them individually,
and you must call ``scaler.step`` on each of them individually.

However, ``scaler.update()`` should only be called once,
after all optimizers used this iteration have been stepped::

    scaler = torch.cuda.amp.GradScaler()

    for epoch in epochs:
        for input, target in data:
            optimizer0.zero_grad()
            optimizer1.zero_grad()
            output0 = model0(input)
            output1 = model1(input)
            loss0 = loss_fn(2 * output0 + 3 * output1, target)
            loss1 = loss_fn(3 * output0 - 5 * output1, target)

            scaler.scale(loss0).backward(retain_graph=True)
            scaler.scale(loss1).backward()

            # You can choose which optimizers receive explicit unscaling, if you
            # want to inspect or modify the gradients of the params they own.
            scaler.unscale_(optimizer0)

            scaler.step(optimizer0)
            scaler.step(optimizer1)

            scaler.update()

Each optimizer independently checks its gradients for infs/NaNs, and therefore makes an independent decision
whether or not to skip the step.  This may result in one optimizer skipping the step
while the other one does not.  Since step skipping occurs rarely (every several hundred iterations)
this should not impede convergence.  If you observe poor convergence after adding gradient scaling
to a multiple-optimizer model, please file an issue.
