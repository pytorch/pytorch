.. role:: hidden
    :class: hidden-section

Automatic Mixed Precision package - torch.amp
==================================================

.. automodule:: torch.amp
.. currentmodule:: torch.amp

``torch.amp`` provides convenience methods for mixed precision:  using fast FP16 arithmetic where you can and
stable FP32 arithmetic where you must to optimize speed while maintaining accuracy and stability.

The two ingredients of the mixed precision recipe are `Autocasting`_ and `Gradient Scaling`_.

By default, you don't need to call ``.half()`` on your model(s) or data to use the routines below.  In fact, you shouldn't:
model params should remain FP32.

Turning on mixed precision should not require retuning any hyperparameters, as long as the conventions shown in the `Examples`_
below are obeyed.

.. contents:: :local:

Autocasting
^^^^^^^^^^^

Under construction...

Gradient scaling
^^^^^^^^^^^^^^^^

Late in training, FP16 gradients can underflow, halting convergence and in some cases causing destabilization.
Amp mitigates underflow via "dynamic gradient scaling."  Gradients are scaled by multiplying the network's output(s)
by some scale factor S, then invoking a backward pass on the *scaled* output(s).
The chain rule then ensures that all gradients flowing backward through are scaled by S.

Amp attempts to maximize use of FP16's full dynamic range by choosing the highest S that can be used without incurring
inf/nan gradients, which is accomplished as follows: Initially, a high value for S is chosen.
Each iteration, after the backward() calls have created all the gradients for a particular optimizer, that optimizer
checks its gradients for infs/nans.  If any infs/nans are found, the optimizer skips the step,
and S is reduced. S is also gradually increased for each successful (inf/nan free) iteration that occurs.
In this way, over time, S rides the edge of the highest value that can be used without causing overflow.

The model parameters are always the same parameters that are owned and stepped by the optimizer.  For typical use,
the model parameters remain FP32, which means the gradients and parameters the optimizer sees are FP32.
These FP32 gradients are unscaled in FP32 before being applied to FP32 model parameters.

Gradient scaling requires a few changes to your script.  Specifically

1. The backward pass(es) should act on scaled output(s).
2. The optimizer step must be able to skip applying gradients if infs/nans are found.
3. Gradients must be unscaled before stepping, or unscaled and stepped in a single optimizer call.

Here's how that looks in a simple example::

    S = add_amp_attributes(optimizer)
    ...
    for input, target in data:
        optimizer.zero_grad()
        with enable_autocasting():
            output = model(input)
            loss = loss_fn(output, target)
        amp.scale_outputs(loss, S).backward()
        _, found_inf, S = optimizer.unscale_and_step(current_scale=S)

Points 2. and 3. require that the ``step()`` call be changed to :func:`unscale_and_step`.
ALternatively, you may separate the unscaling and stepping into :func:`unscale` and :func:`step_after_unscale`
calls, as shown in the `Examples`_.  These new methods are patched onto each optimizer instance by the initial call to
:func:`add_amp_attributes`.    :func:`add_amp_attributes` also returns a one-element ``torch.cuda.FloatTensor`` S
to use as the recommended initial scale, and :func:`unscale_and_step` returns a new S to serve as the recommended
scale for next iteration.  For any iteration, you are free to ignore the recommended scale and supply your own
value to :func:`amp.scale_outputs`.

Each patched method includes an optional ``scaling_enabled=False`` argument, so gradient
scaling can be switched on and off without any code divergence/if statements.

The `Examples`_ below show proper use of the gradient scaling API for a variety of common cases.

The interface for gradient scaling is purely functional (internally stateless).
Persistent values (the current scale, and the optional scale growth and backoff factors) are stored
explicitly in your script and passed to the relevant scaling and unscaling functions.
This makes it easy to manually alter the values between iterations, and also easy to save and load them,
because they can be stashed directly as part of a checkpoint.

Custom Optimizers (Customizing scaling behavior)
------------------------------------------------

The gradient scaling API requires passing optimizers through :func:`add_amp_attributes` to patch on unscaling and safe
stepping methods.  Currently, the methods added are :func:`unscale`, :func:`check_inf`, :func:`step_after_unscale`,
and :func:`unscale_and_step`.  The default implementations of these methods should work with any optimizer that defines ``step``.
In other words, existing optimizers should work properly with the gradient scaling API without needing to add or reimplement
any methods.

However, if some or all of these methods are already members of the optimizer instance, :func:`add_amp_attributes` won't touch
them.  When writing a custom optimizer, if you wish to customize the behavior of the :func:`unscale`, :func:`check_inf`,
:func:`step_after_unscale`, and/or :func:`unscale_and_step` control points, you may (but are not required to) define some or all of
these methods as part of your optimizer class.  User scripts that obey the API will then invoke your custom behavior without
further changes on their part.

Gotchas
-------

Any gradients produced by a backward pass from the scaled outputs (leaf gradients produced by ``backward`` or ``torch.autograd.backward``, or out-of-place gradients produced by ``torch.autograd.grad``) will be scaled. Therefore, anything that manipulates the gradients between the backward pass and a call to :func:`unscale` or :func:`unscale_and_step` will require proper awareness of the scale factor. Examples of operations that require scale-factor-aware treatment are

* gradient clipping
* gradient penalty computations

Both are demonstrated below.

Examples
^^^^^^^^

Typical use (1 loss, 1 optimizer)
---------------------------------

::

    S = add_amp_attributes(optimizer)
    ...
    for input, target in data:
        optimizer.zero_grad()
        with enable_autocasting():
            output = model(input)
            loss = loss_fn(output, target)
        # backward() need not run within the context manager.
        scale_outputs(loss, S).backward()
        _, found_inf, S = optimizer.unscale_and_step(current_scale=S)


Gradient clipping
-----------------

Gradient clipping requires awareness that the gradients resulting from ``scale_outputs(loss, S).backward()`` are scaled.
One simple way to account for the scale factor is by clipping to ``max_norm*S`` instead of ``max_norm``::

    S = add_amp_attributes(optimizer)
    ...
    for input, target in data:
        optimizer.zero_grad()
        with enable_autocasting():
            output = model(input)
            loss = loss_fn(output, target)
        scale_outputs(loss, S).backward()

        # Gradients are scaled, so we clip to max_norm*scale
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm*S)

        _, found_inf, S = optimizer.unscale_and_step(current_scale=S)


Gradient clipping with separate unscaling
-----------------------------------------

In the above example, the *scaled* gradients were clipped, and the optimizer was told to unscale and step in a single call.
The specific case of clipping scaled gradients isn't so hard (all you have to do is clip to ``max_norm*S``).
However, in general, between the backward pass and the optimizer step you may wish to manipulate gradients in some way that's not
so easy to translate to scaled gradients.  In such cases, you can unscale and step separately.  Here's how that looks,
using gradient clipping as an example once more::

    S = add_amp_attributes(optimizer)
    ...
    for input, target in data:
        optimizer.zero_grad()
        with enable_autocasting():
            output = model(input)
            loss = loss_fn(output, target)
        scale_outputs(loss, S).backward()
        found_inf, S = optimizer.unscale(S)

        # Since we've already unscaled, we can clip to max_norm directly, instead of max_norm*S
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # Manipulate unscaled gradients here however you choose

        # Since we've already unscaled, invoke step_after_unscale
        optimizer.step_after_unscale(found_inf=found_inf)

Note that if you unscale and step separately, you must call :func:`step_after_unscale` instead of :func:`unscale_and_step`.
This contract makes sure the optimizer knows what to do.


Gradient penalty
----------------

A gradient penalty loss term also requires awareness of the scale factor.
Gradient penalty demonstrates some subtle/nonstandard use cases:

* Creating out-of-place gradients with ``torch.autograd.grad``
* Correct interation of gradient scaling with double-backward

Here's how that looks::

    S = add_amp_attributes(optimizer)
    ...
    for input, target in data:
        optimizer.zero_grad()
        with enable_autocasting():
            output = model(input)
            loss = loss_fn(output, target)

        # We should scale_outputs for the out-of-place backward pass
        grad_params = torch.autograd.grad(scale_outputs(loss, S), model.parameters(), create_graph=True)

        # In general, the penalty term may depend nonlinearly on the out-of-place gradients, so to be safe,
        # manually unscale them before computing the penalty term.  This unscale should be autograd-exposed.
        grad_params = [p*(1./S) for p in grad_params]

        # Compute the penalty term and add it to the loss
        with enable_autocasting():
            grad_norm = 0
            for grad in grad_params:
                grad_norm += grad.pow(2).sum()
            grad_norm = grad_norm.sqrt()
            loss = loss + grad_norm

        # The usual scaling for backward will now accumulate leaf gradients appropriately scaled by S
        scale_outputs(loss, S).backward()
        _, found_inf, S = optimizer.unscale_and_step(current_scale=S)


Gradient accumulation
---------------------

Gradient accumulation across iterations (between steps) is a common use case. The scaling API accommodates gradient accumulation without trouble::

    S = add_amp_attributes(optimizer)
    ...
    for i, (input, target) in enumerate(data):
        with enable_autocasting():
            output = model(input)
            loss = loss_fn(output, target)
            loss = loss/iters_to_accumulate
        scale_outputs(loss, S).backward()
        if (i + 1) % iters_to_accumulate == 0:
            # Clip gradients here if desired.
            # You may also use the separate unscale() + step_after_unscale() pattern.
            _, found_inf, S = optimizer.unscale_and_step(current_scale=S)
            optimizer.zero_grad()

Batch replay
------------

Sometimes every iteration/data batch is valuable enough that you don't want to skip any. Instead, it's preferable to replay the batch with a reduced loss scale until gradients do not contain infs/nans. Batch replay control flow is not provided by the API alone, but it's straightforward to rig.

The simplest approach is to use the separate ``unscale()`` + ``step_after_unscale()`` pattern::

    S = add_amp_attributes(optimizer)
    ...
    for input, target in data:
        # Replay this batch until inf/nan-free gradients are produced
        while True:
            optimizer.zero_grad()
            with enable_autocasting():
                output = model(input)
                loss = loss_fn(output, target)
            scale_outputs(loss, S).backward()
            found_inf, S = optimizer.unscale(S)

            # If we didn't find any infs/nans, stop replaying and step.
            if not found_inf.item():  break

        # manipulate unscaled gradients here

        optimizer.step_after_unscale(found_inf=found_inf,
                                     skip_if_inf=False) # we know the gradients don't contain infs at this point

``skip_if_inf=False`` tells :func:`step_after_unscale` that the gradients are known not to contain infs/nans at this point,
which gives the optimizer leeway to take a faster code path.  ``skip_if_inf=False`` is purely a performance optimization;
if not supplied, :func:`step_after_unscale` will use the default code path that checks the value contained by ``found_inf``.
The check is redundant here, but it will still do the right math.

Batch replay may also be implemented using ``unscale_and_step``::

    S = add_amp_attributes(optimizer)
    ...
    for input, target in data:
        # Replay this batch until inf/nan-free gradients are produced
        while True:
            optimizer.zero_grad()
            with enable_autocasting():
                output = model(input)
                loss = loss_fn(output, target)
            scale_outputs(loss, S).backward()
            found_inf = optimizer.check_inf()
            if found_inf.item():
                S = S*0.5 # If gradients contained inf, reduce S and replay.
            else:
                break # If gradients did not contain inf, break the loop and step.
        optimizer.unscale_and_step(current_scale=S,
                                   skip_if_inf=False) # we know the gradients don't contain infs at this point

Note the use of the :func:`check_inf` utility function.  Only functions that consume the current loss scale to unscale the gradients
(:func:`unscale` and :func:`unscale_and_step`) have the authority to recommend a new loss scale.  :func:`check_inf` does not
unscale the gradients, so it does not recommened a new scale.  Therefore, if gradients contained inf/nan, S is reduced manually (``S = S*0.5``) for the next iteration of the replay loop.

Once again, the optional ``skip_if_inf=False`` arg to :func:`unscale_and_step` gives the optimizer leeway to take a faster code path.

Switching mixed precision on and off
------------------------------------

The ``scaling_enabled`` kwarg allows mixed precision to be globally enabled/disabled without script-side if statements::

    S = add_amp_attributes(optimizer)
    enabled = args.use_mixed_precision

    for input, target in data:
        optimizer.zero_grad()
        with enable_autocasting(enabled=enabled):
            output = model(input)
            loss = loss_fn(output, target)
        scale_outputs(loss, S, scaling_enabled=enabled).backward()
        _, found_inf, S = optimizer.unscale_and_step(current_scale=S, scaling_enabled=enabled)

Multiple models/optimizers/losses
---------------------------------

If your script contains multiple backward passes, you should use the same scale factor for all of them, otherwise you open
the door to nasty accumulation corner cases (for example, if two backward passes with different scales accumulate into the
same parameter's gradient).

If multiple calls to :func:`unscale` or :func:`unscale_and_step` return different recommended scales, best practice is to choose
the minimum among these as the scale factor for next iteration::

    # add_amp_attributes may accept an interable of optimizers
    S = add_amp_attributes([optimizer0, optimizer1])
    ...
    for input, target in data:
        optimizer0.zero_grad()
        optimizer1.zero_grad()
        with enable_autocasting():
            output0 = model0(input)
            output1 = model1(input)
            loss0 = loss_fn0(output0, output1, target)
            loss1 = loss_fn1(output0, output1, target)

        # Both losses accumulate into some parameters of model0 and some parameters of model1.
        # They may also be accumulating into many of the same parameters,
        # but that's ok, because they use the same gradient scale.
        scale_outputs(loss0, S).backward(retain_graph=True)
        scale_outputs(loss1, S).backward()

        _, found_inf0, S0 = optimizer0.unscale_and_step(current_scale=S)
        _, found_inf1, S1 = optimizer1.unscale_and_step(current_scale=S)

        # Select the minimum of the two recommended scales to use next iteration
        S = min(S0, S1)

Closures
--------

Closure use is currently not supported.  :func:`unscale_and_step` and :func:`step_after_unscale` accept ``closure`` arguments, but
only so they can throw an explanatory error.

Closure use with dynamic loss scaling is tricky (but not impossible) to support, and we're trying to decide if it's
worth implementing.  If you require closures, please comment on https://github.com/pytorch/pytorch/issues/25081,
which will help us gauge demand.

.. autofunction:: add_amp_attributes

.. autofunction:: unscale_and_step

.. autofunction:: unscale

.. autofunction:: step_after_unscale

.. autofunction:: check_inf
