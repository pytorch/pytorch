torch.optim
===================================

.. automodule:: torch.optim

How to use an optimizer
-----------------------

To use :mod:`torch.optim` you have to construct an optimizer object that will hold
the current state and will update the parameters based on the computed gradients.

Constructing it
^^^^^^^^^^^^^^^

To construct an :class:`Optimizer` you have to give it an iterable containing the
parameters (all should be :class:`~torch.nn.Parameter` s) or named parameters
(tuples of (str, :class:`~torch.nn.Parameter`)) to optimize. Then,
you can specify optimizer-specific options such as the learning rate, weight decay, etc.

Example::

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam([var1, var2], lr=0.0001)

Named parameters example::

    optimizer = optim.SGD(model.named_parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam([('layer0', var1), ('layer1', var2)], lr=0.0001)

Per-parameter options
^^^^^^^^^^^^^^^^^^^^^

:class:`Optimizer` s also support specifying per-parameter options. To do this, instead
of passing an iterable of :class:`~torch.autograd.Variable` s, pass in an iterable of
:class:`dict` s. Each of them will define a separate parameter group, and should contain
a ``params`` key, containing a list of parameters belonging to it. Other keys
should match the keyword arguments accepted by the optimizers, and will be used
as optimization options for this group.

For example, this is very useful when one wants to specify per-layer learning rates::

    optim.SGD([
                    {'params': model.base.parameters(), 'lr': 1e-2},
                    {'params': model.classifier.parameters()}
                ], lr=1e-3, momentum=0.9)

    optim.SGD([
                    {'params': model.base.named_parameters(), 'lr': 1e-2},
                    {'params': model.classifier.named_parameters()}
                ], lr=1e-3, momentum=0.9)

This means that ``model.base``'s parameters will use a learning rate of ``1e-2``, whereas
``model.classifier``'s parameters will stick to the default learning rate of ``1e-3``.
Finally a momentum of ``0.9`` will be used for all parameters.

.. note::

    You can still pass options as keyword arguments. They will be used as
    defaults, in the groups that didn't override them. This is useful when you
    only want to vary a single option, while keeping all others consistent
    between parameter groups.

Also consider the following example related to the distinct penalization of parameters.
Remember that :func:`~torch.nn.Module.parameters` returns an iterable that
contains all learnable parameters, including biases and other
parameters that may prefer distinct penalization. To address this, one can specify
individual penalization weights for each parameter group::

    bias_params = [p for name, p in self.named_parameters() if 'bias' in name]
    others = [p for name, p in self.named_parameters() if 'bias' not in name]

    optim.SGD([
                    {'params': others},
                    {'params': bias_params, 'weight_decay': 0}
                ], weight_decay=1e-2, lr=1e-2)

In this manner, bias terms are isolated from non-bias terms, and a ``weight_decay``
of ``0`` is set specifically for the bias terms, as to avoid any penalization for
this group.


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

.. _optimizer-algorithms:

Base class
----------

.. autoclass:: Optimizer

.. autosummary::
    :toctree: generated
    :nosignatures:

    Optimizer.add_param_group
    Optimizer.load_state_dict
    Optimizer.register_load_state_dict_pre_hook
    Optimizer.register_load_state_dict_post_hook
    Optimizer.state_dict
    Optimizer.register_state_dict_pre_hook
    Optimizer.register_state_dict_post_hook
    Optimizer.step
    Optimizer.register_step_pre_hook
    Optimizer.register_step_post_hook
    Optimizer.zero_grad

Algorithms
----------

.. autosummary::
    :toctree: generated
    :nosignatures:

    Adadelta
    Adafactor
    Adagrad
    Adam
    AdamW
    SparseAdam
    Adamax
    ASGD
    BFGS
    LBFGS
    NAdam
    RAdam
    RMSprop
    Rprop
    SGD

Many of our algorithms have various implementations optimized for performance,
readability and/or generality, so we attempt to default to the generally fastest
implementation for the current device if no particular implementation has been
specified by the user.

We have 3 major categories of implementations: for-loop, foreach (multi-tensor), and
fused. The most straightforward implementations are for-loops over the parameters with
big chunks of computation. For-looping is usually slower than our foreach
implementations, which combine parameters into a multi-tensor and run the big chunks
of computation all at once, thereby saving many sequential kernel calls. A few of our
optimizers have even faster fused implementations, which fuse the big chunks of
computation into one kernel. We can think of foreach implementations as fusing
horizontally and fused implementations as fusing vertically on top of that.

In general, the performance ordering of the 3 implementations is fused > foreach > for-loop.
So when applicable, we default to foreach over for-loop. Applicable means the foreach
implementation is available, the user has not specified any implementation-specific kwargs
(e.g., fused, foreach, differentiable), and all tensors are native. Note that while fused
should be even faster than foreach, the implementations are newer and we would like to give
them more bake-in time before flipping the switch everywhere. We summarize the stability status
for each implementation on the second table below, you are welcome to try them out though!

Below is a table showing the available and default implementations of each algorithm:

.. csv-table::
    :header: "Algorithm", "Default", "Has foreach?", "Has fused?"
    :widths: 25, 25, 25, 25
    :delim: ;

    :class:`Adadelta`;foreach;yes;no
    :class:`Adafactor`;for-loop;no;no
    :class:`Adagrad`;foreach;yes;yes (cpu only)
    :class:`Adam`;foreach;yes;yes
    :class:`AdamW`;foreach;yes;yes
    :class:`SparseAdam`;for-loop;no;no
    :class:`Adamax`;foreach;yes;no
    :class:`ASGD`;foreach;yes;no
    :class:`BFGS`;for-loop;no;no
    :class:`LBFGS`;for-loop;no;no
    :class:`NAdam`;foreach;yes;no
    :class:`RAdam`;foreach;yes;no
    :class:`RMSprop`;foreach;yes;no
    :class:`Rprop`;foreach;yes;no
    :class:`SGD`;foreach;yes;yes

Below table is showing the stability status for fused implementations:

.. csv-table::
    :header: "Algorithm", "CPU", "CUDA", "MPS"
    :widths: 25, 25, 25, 25
    :delim: ;

    :class:`Adadelta`;unsupported;unsupported;unsupported
    :class:`Adafactor`;unsupported;unsupported;unsupported
    :class:`Adagrad`;beta;unsupported;unsupported
    :class:`Adam`;beta;stable;beta
    :class:`AdamW`;beta;stable;beta
    :class:`SparseAdam`;unsupported;unsupported;unsupported
    :class:`Adamax`;unsupported;unsupported;unsupported
    :class:`ASGD`;unsupported;unsupported;unsupported
    :class:`BFGS`;unsupported;unsupported;unsupported
    :class:`LBFGS`;unsupported;unsupported;unsupported
    :class:`NAdam`;unsupported;unsupported;unsupported
    :class:`RAdam`;unsupported;unsupported;unsupported
    :class:`RMSprop`;unsupported;unsupported;unsupported
    :class:`Rprop`;unsupported;unsupported;unsupported
    :class:`SGD`;beta;beta;beta

How to adjust learning rate
---------------------------

:class:`torch.optim.lr_scheduler.LRScheduler` provides several methods to adjust the learning
rate based on the number of epochs. :class:`torch.optim.lr_scheduler.ReduceLROnPlateau`
allows dynamic learning rate reducing based on some validation measurements.

Learning rate scheduling should be applied after optimizer's update; e.g., you
should write your code this way:

Example::

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = ExponentialLR(optimizer, gamma=0.9)

    for epoch in range(20):
        for input, target in dataset:
            optimizer.zero_grad()
            output = model(input)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
        scheduler.step()

Most learning rate schedulers can be called back-to-back (also referred to as
chaining schedulers). The result is that each scheduler is applied one after the
other on the learning rate obtained by the one preceding it.

Example::

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler1 = ExponentialLR(optimizer, gamma=0.9)
    scheduler2 = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)

    for epoch in range(20):
        for input, target in dataset:
            optimizer.zero_grad()
            output = model(input)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
        scheduler1.step()
        scheduler2.step()

In many places in the documentation, we will use the following template to refer to schedulers
algorithms.

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


.. autosummary::
    :toctree: generated
    :nosignatures:

    lr_scheduler.LRScheduler
    lr_scheduler.LambdaLR
    lr_scheduler.MultiplicativeLR
    lr_scheduler.StepLR
    lr_scheduler.MultiStepLR
    lr_scheduler.ConstantLR
    lr_scheduler.LinearLR
    lr_scheduler.ExponentialLR
    lr_scheduler.PolynomialLR
    lr_scheduler.CosineAnnealingLR
    lr_scheduler.ChainedScheduler
    lr_scheduler.SequentialLR
    lr_scheduler.ReduceLROnPlateau
    lr_scheduler.CyclicLR
    lr_scheduler.OneCycleLR
    lr_scheduler.CosineAnnealingWarmRestarts

How to utilize named parameters to load optimizer state dict
------------------------------------------------------------

The function :func:`~Optimizer.load_state_dict` stores the optional ``param_names`` content from the
loaded state dict if present. However, the process of loading the optimizer state is not affected,
as the order of the parameters matters to maintain compatibility (in case of different ordering).
To utilize the loaded parameters names from the loaded state dict, a custom ``register_load_state_dict_pre_hook``
needs to be implemented according to the desired behavior.

This can be useful, for instance, when the model architecture changes, but the weights and optimizer states need to
remain unchanged. The following example demonstrates how to implement this customization.

Example::

    class OneLayerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3, 4)

        def forward(self, x):
            return self.fc(x)

    model = OneLayerModel()
    optimizer = optim.SGD(model.named_parameters(), lr=0.01, momentum=0.9)
    # training..
    torch.save(optimizer.state_dict(), PATH)

Let's say that ``model`` implements an expert (MoE), and we want to duplicate it and resume training
for two experts, both initialized the same way as the ``fc`` layer. For the following ``model2`` we create two layers identical to ``fc`` and resume training by loading the model weights and optimizer states from ``model`` into both ``fc1`` and ``fc2`` of ``model2`` (and adjust them accordingly)::

    class TwoLayerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(3, 4)
            self.fc2 = nn.Linear(3, 4)

        def forward(self, x):
            return (self.fc1(x) + self.fc2(x)) / 2

    model2 = TwoLayerModel()
    # adapt and load model weights..
    optimizer2 = optim.SGD(model2.named_parameters(), lr=0.01, momentum=0.9)

To load the state dict for ``optimizer2`` with the state dict of the previous optimizer such that both
``fc1`` and ``fc2`` will be initialized with a copy of ``fc`` optimizer states
(to resume training for each layer from ``fc``), we can use the following hook::

    def adapt_state_dict_ids(optimizer, state_dict):
        adapted_state_dict = deepcopy(optimizer.state_dict())
        # Copy setup parameters (lr, weight_decay, etc.), in case they differ in the loaded state dict.
        for k, v in state_dict['param_groups'][0].items():
            if k not in ['params', 'param_names']:
                adapted_state_dict['param_groups'][0][k] = v

        lookup_dict = {
            'fc1.weight': 'fc.weight',
            'fc1.bias': 'fc.bias',
            'fc2.weight': 'fc.weight',
            'fc2.bias': 'fc.bias'
        }
        clone_deepcopy = lambda d: {k: (v.clone() if isinstance(v, torch.Tensor) else deepcopy(v)) for k, v in d.items()}
        for param_id, param_name in zip(
                optimizer.state_dict()['param_groups'][0]['params'],
                optimizer.state_dict()['param_groups'][0]['param_names']):
            name_in_loaded = lookup_dict[param_name]
            index_in_loaded_list = state_dict['param_groups'][0]['param_names'].index(name_in_loaded)
            id_in_loaded = state_dict['param_groups'][0]['params'][index_in_loaded_list]
            # Copy the state of the corresponding parameter
            if id_in_loaded in state_dict['state']:
                adapted_state_dict['state'][param_id] = clone_deepcopy(state_dict['state'][id_in_loaded])

        return adapted_state_dict

    optimizer2.register_load_state_dict_pre_hook(adapt_state_dict_ids)
    optimizer2.load_state_dict(torch.load(PATH)) # The previous optimizer saved state_dict

This ensures that the adapted state_dict with the correct states for the layers of ``model2`` will be used
during model loading.
Note that this code is designed specifically for this example (e.g., assuming a single parameter group),
and other cases might require different adaptations.

The following example shows how to handle missing parameters in a loaded
``state dict`` when the model structure changes.
The ``Model_bypass`` adds a new ``bypass`` layer, which is not present in the original ``Model1``.
To resume training, a custom ``adapt_state_dict_missing_param`` hook is used to adapt the optimizer's ``state_dict``,
ensuring existing parameters are mapped correctly, while missing ones (like the bypass layer) remain unchanged
(as initialized in this example).
This approach enables smooth loading and resuming of the optimizer state despite model changes.
The new bypass layer will be trained from scratch::

    class Model1(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(5, 5)

        def forward(self, x):
            return self.fc(x) + x


    model = Model1()
    optimizer = optim.SGD(model.named_parameters(), lr=0.01, momentum=0.9)
    # training..
    torch.save(optimizer.state_dict(), PATH)

    class Model_bypass(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(5, 5)
            self.bypass = nn.Linear(5, 5, bias=False)
            torch.nn.init.eye_(self.bypass.weight)

        def forward(self, x):
            return self.fc(x) + self.bypass(x)

    model2 = Model_bypass()
    optimizer2 = optim.SGD(model2.named_parameters(), lr=0.01, momentum=0.9)

    def adapt_state_dict_missing_param(optimizer, state_dict):
        adapted_state_dict = deepcopy(optimizer.state_dict())
        # Copy setup parameters (lr, weight_decay, etc.), in case they differ in the loaded state dict.
        for k, v in state_dict['param_groups'][0].items():
            if k not in ['params', 'param_names']:
                adapted_state_dict['param_groups'][0][k] = v

        lookup_dict = {
            'fc.weight': 'fc.weight',
            'fc.bias': 'fc.bias',
            'bypass.weight': None,
        }

        clone_deepcopy = lambda d: {k: (v.clone() if isinstance(v, torch.Tensor) else deepcopy(v)) for k, v in d.items()}
        for param_id, param_name in zip(
                optimizer.state_dict()['param_groups'][0]['params'],
                optimizer.state_dict()['param_groups'][0]['param_names']):
            name_in_loaded = lookup_dict[param_name]
            if name_in_loaded in state_dict['param_groups'][0]['param_names']:
                index_in_loaded_list = state_dict['param_groups'][0]['param_names'].index(name_in_loaded)
                id_in_loaded = state_dict['param_groups'][0]['params'][index_in_loaded_list]
                # Copy the state of the corresponding parameter
                if id_in_loaded in state_dict['state']:
                    adapted_state_dict['state'][param_id] = clone_deepcopy(state_dict['state'][id_in_loaded])

        return adapted_state_dict

    optimizer2.register_load_state_dict_pre_hook(adapt_state_dict_ids)
    optimizer2.load_state_dict(torch.load(PATH)) # The previous optimizer saved state_dict



As a third example, instead of loading a state according to the order of parameters (the default approach),
this hook can be used to load according to the parameters' names::

    def names_matching(optimizer, state_dict):
        assert len(state_dict['param_groups']) == len(optimizer.state_dict()['param_groups'])
        adapted_state_dict = deepcopy(optimizer.state_dict())
        for g_ind in range(len(state_dict['param_groups'])):
            assert len(state_dict['param_groups'][g_ind]['params']) == len(
                optimizer.state_dict()['param_groups'][g_ind]['params'])

            for k, v in state_dict['param_groups'][g_ind].items():
                if k not in ['params', 'param_names']:
                    adapted_state_dict['param_groups'][g_ind][k] = v

            for param_id, param_name in zip(
                    optimizer.state_dict()['param_groups'][g_ind]['params'],
                    optimizer.state_dict()['param_groups'][g_ind]['param_names']):
                index_in_loaded_list = state_dict['param_groups'][g_ind]['param_names'].index(param_name)
                id_in_loaded = state_dict['param_groups'][g_ind]['params'][index_in_loaded_list]
                # Copy the state of the corresponding parameter
                if id_in_loaded in state_dict['state']:
                    adapted_state_dict['state'][param_id] = deepcopy(state_dict['state'][id_in_loaded])

        return adapted_state_dict



Weight Averaging (SWA and EMA)
------------------------------

:class:`torch.optim.swa_utils.AveragedModel` implements Stochastic Weight Averaging (SWA) and Exponential Moving Average (EMA),
:class:`torch.optim.swa_utils.SWALR` implements the SWA learning rate scheduler and
:func:`torch.optim.swa_utils.update_bn` is a utility function used to update SWA/EMA batch
normalization statistics at the end of training.

SWA has been proposed in `Averaging Weights Leads to Wider Optima and Better Generalization`_.

EMA is a widely known technique to reduce the training time by reducing the number of weight updates needed. It is a variation of `Polyak averaging`_, but using exponential weights instead of equal weights across iterations.

.. _`Averaging Weights Leads to Wider Optima and Better Generalization`: https://arxiv.org/abs/1803.05407

.. _`Polyak averaging`: https://paperswithcode.com/method/polyak-averaging

Constructing averaged models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `AveragedModel` class serves to compute the weights of the SWA or EMA model.

You can create an SWA averaged model by running:

>>> averaged_model = AveragedModel(model)

EMA models are constructed by specifying the ``multi_avg_fn`` argument as follows:

>>> decay = 0.999
>>> averaged_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay))

Decay is a parameter between 0 and 1 that controls how fast the averaged parameters are decayed. If not provided to :func:`torch.optim.swa_utils.get_ema_multi_avg_fn`, the default is 0.999. Decay value should be close to 1.0, as smaller values can cause optimization convergence issues.

:func:`torch.optim.swa_utils.get_ema_multi_avg_fn` returns a function that applies the following EMA equation to the weights:

.. math:: W^\textrm{EMA}_{t+1} = \alpha W^\textrm{EMA}_{t} + (1 - \alpha) W^\textrm{model}_t

where alpha is the EMA decay.

Here the model ``model`` can be an arbitrary :class:`torch.nn.Module` object. ``averaged_model``
will keep track of the running averages of the parameters of the ``model``. To update these
averages, you should use the :func:`update_parameters` function after the `optimizer.step()`:

>>> averaged_model.update_parameters(model)

For SWA and EMA, this call is usually done right after the optimizer ``step()``. In the case of SWA, this is usually skipped for some numbers of steps at the beginning of the training.

Custom averaging strategies
^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, :class:`torch.optim.swa_utils.AveragedModel` computes a running equal average of
the parameters that you provide, but you can also use custom averaging functions with the
``avg_fn`` or ``multi_avg_fn`` parameters:

- ``avg_fn`` allows defining a function operating on each parameter tuple (averaged parameter, model parameter) and should return the new averaged parameter.
- ``multi_avg_fn`` allows defining more efficient operations acting on a tuple of parameter lists, (averaged parameter list, model parameter list), at the same time, for example using the ``torch._foreach*`` functions. This function must update the averaged parameters in-place.

In the following example ``ema_model`` computes an exponential moving average using the ``avg_fn`` parameter:

>>> ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged:\
>>>         0.9 * averaged_model_parameter + 0.1 * model_parameter
>>> ema_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=ema_avg)


In the following example ``ema_model`` computes an exponential moving average using the more efficient ``multi_avg_fn`` parameter:

>>> ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.9))


SWA learning rate schedules
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Typically, in SWA the learning rate is set to a high constant value. :class:`SWALR` is a
learning rate scheduler that anneals the learning rate to a fixed value, and then keeps it
constant. For example, the following code creates a scheduler that linearly anneals the
learning rate from its initial value to 0.05 in 5 epochs within each parameter group:

>>> swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, \
>>>         anneal_strategy="linear", anneal_epochs=5, swa_lr=0.05)

You can also use cosine annealing to a fixed value instead of linear annealing by setting
``anneal_strategy="cos"``.


Taking care of batch normalization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`update_bn` is a utility function that allows to compute the batchnorm statistics for the SWA model
on a given dataloader ``loader`` at the end of training:

>>> torch.optim.swa_utils.update_bn(loader, swa_model)

:func:`update_bn` applies the ``swa_model`` to every element in the dataloader and computes the activation
statistics for each batch normalization layer in the model.

.. warning::
    :func:`update_bn` assumes that each batch in the dataloader ``loader`` is either a tensors or a list of
    tensors where the first element is the tensor that the network ``swa_model`` should be applied to.
    If your dataloader has a different structure, you can update the batch normalization statistics of the
    ``swa_model`` by doing a forward pass with the ``swa_model`` on each element of the dataset.




Putting it all together: SWA
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the example below, ``swa_model`` is the SWA model that accumulates the averages of the weights.
We train the model for a total of 300 epochs and we switch to the SWA learning rate schedule
and start to collect SWA averages of the parameters at epoch 160:

>>> loader, optimizer, model, loss_fn = ...
>>> swa_model = torch.optim.swa_utils.AveragedModel(model)
>>> scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
>>> swa_start = 160
>>> swa_scheduler = SWALR(optimizer, swa_lr=0.05)
>>>
>>> for epoch in range(300):
>>>       for input, target in loader:
>>>           optimizer.zero_grad()
>>>           loss_fn(model(input), target).backward()
>>>           optimizer.step()
>>>       if epoch > swa_start:
>>>           swa_model.update_parameters(model)
>>>           swa_scheduler.step()
>>>       else:
>>>           scheduler.step()
>>>
>>> # Update bn statistics for the swa_model at the end
>>> torch.optim.swa_utils.update_bn(loader, swa_model)
>>> # Use swa_model to make predictions on test data
>>> preds = swa_model(test_input)


Putting it all together: EMA
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the example below, ``ema_model`` is the EMA model that accumulates the exponentially-decayed averages of the weights with a decay rate of 0.999.
We train the model for a total of 300 epochs and start to collect EMA averages immediately.

>>> loader, optimizer, model, loss_fn = ...
>>> ema_model = torch.optim.swa_utils.AveragedModel(model, \
>>>             multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
>>>
>>> for epoch in range(300):
>>>       for input, target in loader:
>>>           optimizer.zero_grad()
>>>           loss_fn(model(input), target).backward()
>>>           optimizer.step()
>>>           ema_model.update_parameters(model)
>>>
>>> # Update bn statistics for the ema_model at the end
>>> torch.optim.swa_utils.update_bn(loader, ema_model)
>>> # Use ema_model to make predictions on test data
>>> preds = ema_model(test_input)

.. autosummary::
    :toctree: generated
    :nosignatures:

    swa_utils.AveragedModel
    swa_utils.SWALR


.. autofunction:: torch.optim.swa_utils.get_ema_multi_avg_fn
.. autofunction:: torch.optim.swa_utils.update_bn


.. This module needs to be documented. Adding here in the meantime
.. for tracking purposes
.. py:module:: torch.optim.adadelta
.. py:module:: torch.optim.adagrad
.. py:module:: torch.optim.adam
.. py:module:: torch.optim.adamax
.. py:module:: torch.optim.adamw
.. py:module:: torch.optim.asgd
.. py:module:: torch.optim.bfgs
.. py:module:: torch.optim.lbfgs
.. py:module:: torch.optim.lr_scheduler
.. py:module:: torch.optim.nadam
.. py:module:: torch.optim.optimizer
.. py:module:: torch.optim.radam
.. py:module:: torch.optim.rmsprop
.. py:module:: torch.optim.rprop
.. py:module:: torch.optim.sgd
.. py:module:: torch.optim.sparse_adam
.. py:module:: torch.optim.swa_utils
