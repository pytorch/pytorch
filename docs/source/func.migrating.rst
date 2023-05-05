Migrating from functorch to torch.func
======================================

torch.func, previously known as "functorch", is
`JAX-like <https://github.com/google/jax>`_ composable function transforms for PyTorch.

functorch started as an out-of-tree library over at
the `pytorch/functorch <https://github.com/pytorch/functorch>`_ repository.
Our goal has always been to upstream functorch directly into PyTorch and provide
it as a core PyTorch library.

As the final step of the upstream, we've decided to migrate from being a top level package
(``functorch``) to being a part of PyTorch to reflect how the function transforms are
integrated directly into PyTorch core. As of PyTorch 2.0, we are deprecating
``import functorch`` and ask that users migrate to the newest APIs, which we
will maintain going forward. ``import functorch`` will be kept around to maintain
backwards compatibility for a couple of releases.

function transforms
-------------------

The following APIs are a drop-in replacement for the following
`functorch APIs <https://pytorch.org/functorch/1.13/functorch.html>`_.
They are fully backwards compatible.


==============================  =======================================
functorch API                    PyTorch API (as of PyTorch 2.0)
==============================  =======================================
functorch.vmap                  :func:`torch.vmap` or :func:`torch.func.vmap`
functorch.grad                  :func:`torch.func.grad`
functorch.vjp                   :func:`torch.func.vjp`
functorch.jvp                   :func:`torch.func.jvp`
functorch.jacrev                :func:`torch.func.jacrev`
functorch.jacfwd                :func:`torch.func.jacfwd`
functorch.hessian               :func:`torch.func.hessian`
functorch.functionalize         :func:`torch.func.functionalize`
==============================  =======================================

Furthermore, if you are using torch.autograd.functional APIs, please try out
the :mod:`torch.func` equivalents instead. :mod:`torch.func` function
transforms are more composable and more performant in many cases.

=========================================== =======================================
torch.autograd.functional API               torch.func API (as of PyTorch 2.0)
=========================================== =======================================
:func:`torch.autograd.functional.vjp`       :func:`torch.func.grad` or :func:`torch.func.vjp`
:func:`torch.autograd.functional.jvp`       :func:`torch.func.jvp`
:func:`torch.autograd.functional.jacobian`  :func:`torch.func.jacrev` or :func:`torch.func.jacfwd`
:func:`torch.autograd.functional.hessian`   :func:`torch.func.hessian`
=========================================== =======================================

NN module utilities
-------------------

We've changed the APIs to apply function transforms over NN modules to make them
fit better into the PyTorch design philosophy. The new API is different, so
please read this section carefully.

functorch.make_functional
^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`torch.func.functional_call` is the replacement for
`functorch.make_functional <https://pytorch.org/functorch/1.13/generated/functorch.make_functional.html#functorch.make_functional>`_
and
`functorch.make_functional_with_buffers <https://pytorch.org/functorch/1.13/generated/functorch.make_functional_with_buffers.html#functorch.make_functional_with_buffers>`_.
However, it is not a drop-in replacement.

If you're in a hurry, you can use
`helper functions in this gist <https://gist.github.com/zou3519/7769506acc899d83ef1464e28f22e6cf>`_
that emulate the behavior of functorch.make_functional and functorch.make_functional_with_buffers.
We recommend using :func:`torch.func.functional_call` directly because it is a more explicit
and flexible API.

Concretely, functorch.make_functional returns a functional module and parameters.
The functional module accepts parameters and inputs to the model as arguments.
:func:`torch.func.functional_call` allows one to call the forward pass of an existing
module using new parameters and buffers and inputs.

Here's an example of how to compute gradients of parameters of a model using functorch
vs :mod:`torch.func`::

    # ---------------
    # using functorch
    # ---------------
    import torch
    import functorch
    inputs = torch.randn(64, 3)
    targets = torch.randn(64, 3)
    model = torch.nn.Linear(3, 3)

    fmodel, params = functorch.make_functional(model)

    def compute_loss(params, inputs, targets):
        prediction = fmodel(params, inputs)
        return torch.nn.functional.mse_loss(prediction, targets)

    grads = functorch.grad(compute_loss)(params, inputs, targets)

    # ------------------------------------
    # using torch.func (as of PyTorch 2.0)
    # ------------------------------------
    import torch
    inputs = torch.randn(64, 3)
    targets = torch.randn(64, 3)
    model = torch.nn.Linear(3, 3)

    params = dict(model.named_parameters())

    def compute_loss(params, inputs, targets):
        prediction = torch.func.functional_call(model, params, (inputs,))
        return torch.nn.functional.mse_loss(prediction, targets)

    grads = torch.func.grad(compute_loss)(params, inputs, targets)

And here's an example of how to compute jacobians of model parameters::

    # ---------------
    # using functorch
    # ---------------
    import torch
    import functorch
    inputs = torch.randn(64, 3)
    model = torch.nn.Linear(3, 3)

    fmodel, params = functorch.make_functional(model)
    jacobians = functorch.jacrev(fmodel)(params, inputs)

    # ------------------------------------
    # using torch.func (as of PyTorch 2.0)
    # ------------------------------------
    import torch
    from torch.func import jacrev, functional_call
    inputs = torch.randn(64, 3)
    model = torch.nn.Linear(3, 3)

    params = dict(model.named_parameters())
    # jacrev computes jacobians of argnums=0 by default.
    # We set it to 1 to compute jacobians of params
    jacobians = jacrev(functional_call, argnums=1)(model, params, (inputs,))

Note that it is important for memory consumption that you should only carry
around a single copy of your parameters. ``model.named_parameters()`` does not copy
the parameters. If in your model training you update the parameters of the model
in-place, then the ``nn.Module`` that is your model has the single copy of the
parameters and everything is OK.

However, if you want to carry your parameters around in a dictionary and update
them out-of-place, then there are two copies of parameters: the one in the
dictionary and the one in the ``model``. In this case, you should change
``model`` to not hold memory by converting it to the meta device via
``model.to('meta')``.

functorch.combine_state_for_ensemble
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Please use :func:`torch.func.stack_module_state` instead of
`functorch.combine_state_for_ensemble <https://pytorch.org/functorch/1.13/generated/functorch.combine_state_for_ensemble.html>`_
:func:`torch.func.stack_module_state` returns two dictionaries, one of stacked parameters, and
one of stacked buffers, that can then be used with :func:`torch.vmap` and :func:`torch.func.functional_call`
for ensembling.

For example, here is an example of how to ensemble over a very simple model::

    import torch
    num_models = 5
    batch_size = 64
    in_features, out_features = 3, 3
    models = [torch.nn.Linear(in_features, out_features) for i in range(num_models)]
    data = torch.randn(batch_size, 3)

    # ---------------
    # using functorch
    # ---------------
    import functorch
    fmodel, params, buffers = functorch.combine_state_for_ensemble(models)
    output = functorch.vmap(fmodel, (0, 0, None))(params, buffers, data)
    assert output.shape == (num_models, batch_size, out_features)

    # ------------------------------------
    # using torch.func (as of PyTorch 2.0)
    # ------------------------------------
    import copy

    # Construct a version of the model with no memory by putting the Tensors on
    # the meta device.
    base_model = copy.deepcopy(models[0])
    base_model.to('meta')

    params, buffers = torch.func.stack_module_state(models)

    # It is possible to vmap directly over torch.func.functional_call,
    # but wrapping it in a function makes it clearer what is going on.
    def call_single_model(params, buffers, data):
        return torch.func.functional_call(base_model, (params, buffers), (data,))

    output = torch.vmap(call_single_model, (0, 0, None))(params, buffers, data)
    assert output.shape == (num_models, batch_size, out_features)


functorch.compile
-----------------

We are no longer supporting functorch.compile (also known as AOTAutograd)
as a frontend for compilation in PyTorch; we have integrated AOTAutograd
into PyTorch's compilation story. If you are a user, please use
:func:`torch.compile` instead.
