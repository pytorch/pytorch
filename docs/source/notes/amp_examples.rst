.. _amp-examples:

Automatic Mixed Precision examples
==================================

.. currentmodule:: torch.cuda.amp

Ordinarily, "automatic mixed precision training" means training with
:class:`torch.cuda.amp.autocast` and :class:`torch.cuda.amp.GradScaler` together.

Instances of :class:`torch.cuda.amp.autocast` enable autocasting for chosen regions.
Autocasting automatically chooses the precision for GPU operations to improve performance
while maintaining accuracy.

Instances of :class:`torch.cuda.amp.GradScaler` help perform the steps of
gradient scaling conveniently.  Gradient scaling improves convergence for networks with ``float16``
gradients by minimizing gradient underflow, as explained :ref:`here<gradient-scaling>`.

:class:`torch.cuda.amp.autocast` and :class:`torch.cuda.amp.GradScaler` are modular.
In the samples below, each is used as its individual documentation suggests.

.. contents:: :local:

Typical Mixed Precision Training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    # Creates model and optimizer in default precision
    model = Net().cuda()
    optimizer = optim.SGD(model.parameters(), ...)

    # Creates a GradScaler once at the beginning of training.
    scaler = GradScaler()

    for epoch in epochs:
        for input, target in data:
            optimizer.zero_grad()

            # Runs the forward pass with autocasting.
            with autocast():
                output = model(input)
                loss = loss_fn(output, target)

            # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
            # Backward passes under autocast are not necessary or recommended.
            # Backward ops run in the same precision that autocast used for corresponding forward ops.
            scaler.scale(loss).backward()

            # scaler.step() first unscales the gradients of the optimizer's assigned params.
            # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            scaler.step(optimizer)

            # Updates the scale for next iteration.
            scaler.update()

.. _working-with-unscaled-gradients:

Working with Unscaled Gradients
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
-----------------

Calling ``scaler.unscale_(optimizer)`` before clipping enables you to clip unscaled gradients as usual::

    scaler = GradScaler()

    for epoch in epochs:
        for input, target in data:
            optimizer.zero_grad()
            with autocast():
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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For some operations, you may need to work with scaled gradients in a setting where
``scaler.unscale_`` is unsuitable.

Gradient penalty
----------------

A gradient penalty implementation commonly creates gradients using
:func:`torch.autograd.grad`, combines them to create the penalty value,
and adds the penalty value to the loss.

Here's an ordinary example of an L2 penalty without gradient scaling or autocasting::

    for epoch in epochs:
        for input, target in data:
            optimizer.zero_grad()
            output = model(input)
            loss = loss_fn(output, target)

            # Creates gradients
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
:func:`torch.autograd.grad` should be scaled.  The resulting gradients
will therefore be scaled, and should be unscaled before being combined to create the
penalty value.

Also, the penalty term computation is part of the forward pass, and therefore should be
inside an :class:`autocast` context manager.

Here's how that looks for the same L2 penalty::

    scaler = GradScaler()

    for epoch in epochs:
        for input, target in data:
            optimizer.zero_grad()
            with autocast():
                output = model(input)
                loss = loss_fn(output, target)

            # Scales the loss for autograd.grad's backward pass, resulting in scaled grad_params
            scaled_grad_params = torch.autograd.grad(scaler.scale(loss), model.parameters(), create_graph=True)

            # Unscales grad_params before computing the penalty.  grad_params are not owned
            # by any optimizer, so ordinary division is used instead of scaler.unscale_:
            inv_scale = 1./scaler.get_scale()
            grad_params = [p*inv_scale for p in scaled_grad_params]

            # Computes the penalty term and adds it to the loss
            with autocast():
                grad_norm = 0
                for grad in grad_params:
                    grad_norm += grad.pow(2).sum()
                grad_norm = grad_norm.sqrt()
                loss = loss + grad_norm

            # Applies scaling to the backward call as usual.
            # Accumulates leaf gradients that are correctly scaled.
            scaler.scale(loss).backward()

            # step() and update() proceed as usual.
            scaler.step(optimizer)
            scaler.update()


Working with Multiple Losses and Optimizers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
            with autocast():
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

Each optimizer checks its gradients for infs/NaNs and makes an independent decision
whether or not to skip the step.  This may result in one optimizer skipping the step
while the other one does not.  Since step skipping occurs rarely (every several hundred iterations)
this should not impede convergence.  If you observe poor convergence after adding gradient scaling
to a multiple-optimizer model, please file a bug report.

Working with Multiple GPUs
^^^^^^^^^^^^^^^^^^^^^^^^^^

The issues described here only affect :class:`autocast`.  :class:`GradScaler`\ 's usage is unchanged.

.. _amp-dataparallel:

DataParallel within a single process
------------------------------------

:class:`torch.nn.DataParallel` spawns threads to run the forward pass on each device.
The autocast state is thread local, so the following will not work::

    model = MyModel()
    dp_model = nn.DataParallel(model)

    # Sets autocast in the main thread
    with autocast():
        # dp_model's internal threads won't autocast.  The main thread's autocast state has no effect.
        output = dp_model(input)
        # loss_fn still autocasts, but it's too late...
        loss = loss_fn(output)

The fix is simple.  Enable autocast as part of ``MyModel.forward``::

    MyModel(nn.Module):
        ...
        @autocast()
        def forward(self, input):
           ...

    # Alternatively
    MyModel(nn.Module):
        ...
        def forward(self, input):
            with autocast():
                ...

The following now autocasts in ``dp_model``'s threads (which execute ``forward``) and the main thread
(which executes ``loss_fn``)::

    model = MyModel()
    dp_model = nn.DataParallel(model)

    with autocast():
        output = dp_model(input)
        loss = loss_fn(output)

DistributedDataParallel, one GPU per process
--------------------------------------------

:class:`torch.nn.parallel.DistributedDataParallel`'s documentation recommends one GPU per process for best
performance.  In this case, ``DistributedDataParallel`` does not spawn threads internally,
so usages of :class:`autocast` and :class:`GradScaler` are not affected.

DistributedDataParallel, multiple GPUs per process
--------------------------------------------------

Here :class:`torch.nn.parallel.DistributedDataParallel` may spawn a side thread to run the forward pass on each
device, like :class:`torch.nn.DataParallel`.  :ref:`The fix is the same<amp-dataparallel>`:
apply autocast as part of your model's ``forward`` method to ensure it's enabled in threads.

.. _amp-custom-examples:

Autocast and Custom Autograd Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If your network uses :ref:`custom autograd functions<extending-autograd>`
(subclasses of :class:`torch.autograd.Function`), :class:`autocast` may work out of the box.

However, if any of the following is true
* Your function takes multiple floating-point Tensor inputs

* Your function wraps any of the ops :ref:`Autocast Op Reference<autocast-policies>`
* You want to ensure the function runs in a particular dtype.

Functions that should allow autocasting
---------------------------------------

The :func:`torch.cuda.amp.custom_fwd` and :func:`torch.cuda.amp.custom_bwd` decorators (with no arguments)
ensure that ``forward`` executes with whatever autocast state surrounds the point of use, and that
``backward`` executes with the same autocast state as ``forward`` (which can prevent type mismatch errors)::

    class MyMM(torch.autograd.Function):
        @staticmethod
        @custom_fwd
        def forward(ctx, a, b):
            ctx.save_for_backward(a, b)
            return a.mm(b)
        @staticmethod
        @custom_bwd
        def backward(ctx, grad):
            a, b = ctx.saved_tensors
            return grad.mm(b.t()), a.t().mm(grad)

Functions that need a particular dtype
--------------------------------------

If you know your function requires a certain precision, or if it wraps a backend with limited datatype support,
you may want it to disallow autocasting.

Consider a custom autograd function wrapping
`CUDA extensions <https://pytorch.org/tutorials/advanced/cpp_extension.html>`_
that were only compiled for ``float32``.  You can force the function to run in ``float32`` at the point of use,
as you would for any explicitly ``float32`` subregion::

    with autocast():
        ...
        with autocast(enabled=False):
            output = float32_function(input.float())

Alternatively, you can use the :func:`custom_fwd` decorator with ``cast_inputs=torch.float32`` in the function's
definition.  The ``cast_inputs=torch.float32`` argument locally disables autocast in ``forward`` and ``backward``,
and casts incoming floating-point CUDA Tensors to ``float32``::

    class Float32Function(torch.autograd.Function):
        @staticmethod
        @custom_fwd(cast_inputs=torch.float32)
        def forward(ctx, input):
            ctx.save_for_backward(input)
            ...
            return fwd_output
        @staticmethod
        @custom_bwd
        def backward(ctx, grad):
            ...

Now ``Float32Function`` can be invoked anywhere, without disabling autocast at the point of use::

    func = Float32Function.apply

    with autocast():
        # func will run in float32, regardless of the surrounding autocast state
        output = func(input)
