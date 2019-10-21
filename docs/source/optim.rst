torch.optim
===================================

.. automodule:: torch.optim

How to use an optimizer
-----------------------

To use :mod:`torch.optim` you have to construct an optimizer object, that will hold
the current state and will update the parameters based on the computed gradients.

Constructing it
^^^^^^^^^^^^^^^

To construct an :class:`Optimizer` you have to give it an iterable containing the
parameters (all should be :class:`~torch.autograd.Variable` s) to optimize. Then,
you can specify optimizer-specific options such as the learning rate, weight decay, etc.

.. note::

    If you need to move a model to GPU via ``.cuda()``, please do so before
    constructing optimizers for it. Parameters of a model after ``.cuda()`` will
    be different objects with those before the call.

    In general, you should make sure that optimized parameters live in
    consistent locations when optimizers are constructed and used.

Example::

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam([var1, var2], lr=0.0001)

Per-parameter options
^^^^^^^^^^^^^^^^^^^^^

:class:`Optimizer` s also support specifying per-parameter options. To do this, instead
of passing an iterable of :class:`~torch.autograd.Variable` s, pass in an iterable of
:class:`dict` s. Each of them will define a separate parameter group, and should contain
a ``params`` key, containing a list of parameters belonging to it. Other keys
should match the keyword arguments accepted by the optimizers, and will be used
as optimization options for this group.

.. note::

    You can still pass options as keyword arguments. They will be used as
    defaults, in the groups that didn't override them. This is useful when you
    only want to vary a single option, while keeping all others consistent
    between parameter groups.


For example, this is very useful when one wants to specify per-layer learning rates::

    optim.SGD([
                    {'params': model.base.parameters()},
                    {'params': model.classifier.parameters(), 'lr': 1e-3}
                ], lr=1e-2, momentum=0.9)

This means that ``model.base``'s parameters will use the default learning rate of ``1e-2``,
``model.classifier``'s parameters will use a learning rate of ``1e-3``, and a momentum of
``0.9`` will be used for all parameters.

Taking an optimization step
^^^^^^^^^^^^^^^^^^^^^^^^^^^

All optimizers implement a :func:`~Optimizer.step` method, that updates the
parameters. It can be used in two ways:

``optimizer.step()``
~~~~~~~~~~~~~~~~~~~~

This is a simplified version supported by most optimizers. The function can be
called once the gradients are computed using e.g.
:func:`~torch.autograd.Variable.backward`.

Example::

    for input, target in dataset:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

``optimizer.step(closure)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some optimization algorithms such as Conjugate Gradient and LBFGS need to
reevaluate the function multiple times, so you have to pass in a closure that
allows them to recompute your model. The closure should clear the gradients,
compute the loss, and return it.

Example::

    for input, target in dataset:
        def closure():
            optimizer.zero_grad()
            output = model(input)
            loss = loss_fn(output, target)
            loss.backward()
            return loss
        optimizer.step(closure)

Algorithms
----------

.. autoclass:: Optimizer
    :members:
.. autoclass:: Adadelta
    :members:
.. autoclass:: Adagrad
    :members:
.. autoclass:: Adam
    :members:
.. autoclass:: AdamW
    :members:
.. autoclass:: SparseAdam
    :members:
.. autoclass:: Adamax
    :members:
.. autoclass:: ASGD
    :members:
.. autoclass:: LBFGS
    :members:
.. autoclass:: RMSprop
    :members:
.. autoclass:: Rprop
    :members:
.. autoclass:: SGD
    :members:

How to adjust Learning Rate
---------------------------

:mod:`torch.optim.lr_scheduler` provides several methods to adjust the learning
rate based on the number of epochs. :class:`torch.optim.lr_scheduler.ReduceLROnPlateau`
allows dynamic learning rate reducing based on some validation measurements.

Learning rate scheduling should be applied after optimizer's update; e.g., you
should write your code this way:

    >>> scheduler = ...
    >>> for epoch in range(100):
    >>>     train(...)
    >>>     validate(...)
    >>>     scheduler.step()

.. warning::
  Prior to PyTorch 1.1.0, the learning rate scheduler was expected to be called before
  the optimizer's update; 1.1.0 changed this behavior in a BC-breaking way.  If you use
  the learning rate scheduler (calling ``scheduler.step()``) before the optimizer's update
  (calling ``optimizer.step()``), this will skip the first value of the learning rate schedule.
  If you are unable to reproduce results after upgrading to PyTorch 1.1.0, please check
  if you are calling ``scheduler.step()`` at the wrong time.


.. autoclass:: torch.optim.lr_scheduler.LambdaLR
    :members:
.. autoclass:: torch.optim.lr_scheduler.StepLR
    :members:
.. autoclass:: torch.optim.lr_scheduler.MultiStepLR
    :members:
.. autoclass:: torch.optim.lr_scheduler.ExponentialLR
    :members:
.. autoclass:: torch.optim.lr_scheduler.CosineAnnealingLR
    :members:
.. autoclass:: torch.optim.lr_scheduler.ReduceLROnPlateau
    :members:
.. autoclass:: torch.optim.lr_scheduler.CyclicLR
    :members:
.. autoclass:: torch.optim.lr_scheduler.OneCycleLR
    :members:
.. autoclass:: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
    :members:
