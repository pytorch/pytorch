Automatic Mixed Precision Examples
==================================

When following the examples below, you don't need to call ``.half()`` on your model(s) or data.
In fact, you shouldn't:  model weights should remain FP32.                                                                                        
You also don't need to retune any hyperparameters.

.. contents:: :local:

Autocasting Examples
^^^^^^^^^^^^^^^^^^^^

Under construction...

Gradient Scaling Examples
^^^^^^^^^^^^^^^^^^^^^^^^^

Typical use (1 loss, 1 optimizer)
---------------------------------

::

    scaler = AmpScaler()
    ...
    for input, target in data:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

Gradient clipping
-----------------

Gradient clipping requires awareness that the gradients resulting from ``scaler.scale(loss).backward()`` are scaled.
One simple way to account for the scale factor is by clipping to ``max_norm*scaler.get_scale()`` instead of ``max_norm``::

    scaler = AmpScaler()
    ...
    for input, target in data:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        scaler.scale(loss).backward()

        # Gradients are scaled, so we clip to max_norm*scale
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm*scaler.get_scale())

        scaler.step(optimizer)
        scaler.update()


Gradient clipping with separate unscaling
-----------------------------------------

In the above example, the *scaled* gradients were clipped.
The specific case of clipping scaled gradients isn't so hard (all you have to do is clip to ``max_norm*scaler.get_scale()``).
However, in general, between the backward pass and the optimizer step you may wish to manipulate gradients in some way that's not
so easy to translate to scaled gradients.  In such cases, you can unscale and step separately.  Here's how that looks,
using gradient clipping as an example once more::

    scaler = AmpScaler()
    ...
    for input, target in data:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        scaler.scale(loss).backward()

        scaler.unscale(optimizer)
        # Since the optimizer's owned gradients are unscaled, we can clip to max_norm directly:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        scaler.step(optimizer)
        scaler.update()


Gradient penalty
----------------

A gradient penalty loss term also requires awareness of the scale factor.
Gradient penalty demonstrates some subtle/nonstandard use cases:

* Creating out-of-place gradients with ``torch.autograd.grad``
* Correct interaction of gradient scaling with double-backward

Here's how that looks::

    scaler = AmpScaler()
    ...
    for input, target in data:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)

        # We should scale outputs for the out-of-place backward pass
        grad_params = torch.autograd.grad(scaler.scale(loss), model.parameters(), create_graph=True)

        # In general, the penalty term may depend nonlinearly on the out-of-place gradients, so to be safe,
        # manually unscale them before computing the penalty term.  This unscale should be autograd-exposed.
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

Gradient accumulation across iterations (between steps) is a common use case.
:class:`torch.cuda.amp.AmpScaler` accommodates gradient accumulation without trouble::

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

The ``enabled`` kwarg to :class:`torch.cuda.amp.AmpScaler` allows gradient scaling to be globally enabled/disabled without script-side if statements::

    scaler = AmpScaler(enabled=args.use_mixed_precision)
    ...
    for input, target in data:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

If ``enabled=False``, ``scaler.step(optimizer)`` directly invokes ``optimizer.step()`` without any wrapping logic, and
the other methods (``scaler.scale``, ``scaler.update``) become no-ops.


Multiple models/optimizers/losses
---------------------------------

Make sure to call ``scaler.update()`` only at the end of the iteration, after ``scaler.step(optimizer)`` has
been called for all optimizers used this iteration.

The decision to invoke an explicit ``unscale`` can be made independently for each optimizer.

.. warning::
    If you (optionally) choose to unscale gradients prior to stepping, ``scaler.unscale(optimizer)``
    should only be invoked once per optimizer per step, and only after all gradients for that optimizer's
    owned parameters have been accumulated.

::
    scaler = torch.cuda.amp.AmpScaler()

    for input, target in data:
        optimizer0.zero_grad()
        optimizer1.zero_grad()
        output0 = model0(input)
        output1 = model1(input)
        loss0 = loss_fn(2 * output0 + 3 * output1, target)
        loss1 = loss_fn(3 * output0 - 5 * output1, target)

        scaler.scale(loss0).backward(retain_graph=True)
        scaler.scale(loss1).backward()

        scaler.unscale(optimizer0)

        scaler.step(optimizer0)
        scaler.step(optimizer1)
        scaler.update()

