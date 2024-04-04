torch.func API Reference
========================

.. currentmodule:: torch.func

.. automodule:: torch.func

Function Transforms
-------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

     vmap
     grad
     grad_and_value
     vjp
     jvp
     linearize
     jacrev
     jacfwd
     hessian
     functionalize

Utilities for working with torch.nn.Modules
-------------------------------------------

In general, you can transform over a function that calls a ``torch.nn.Module``.
For example, the following is an example of computing a jacobian of a function
that takes three values and returns three values:

.. code-block:: python

    model = torch.nn.Linear(3, 3)

    def f(x):
        return model(x)

    x = torch.randn(3)
    jacobian = jacrev(f)(x)
    assert jacobian.shape == (3, 3)

However, if you want to do something like compute a jacobian over the parameters
of the model, then there needs to be a way to construct a function where the
parameters are the inputs to the function.
That's what :func:`functional_call` is for:
it accepts an nn.Module, the transformed ``parameters``, and the inputs to the
Module's forward pass. It returns the value of running the Module's forward pass
with the replaced parameters.

Here's how we would compute the Jacobian over the parameters

.. code-block:: python

    model = torch.nn.Linear(3, 3)

    def f(params, x):
        return torch.func.functional_call(model, params, x)

    x = torch.randn(3)
    jacobian = jacrev(f)(dict(model.named_parameters()), x)


.. autosummary::
    :toctree: generated
    :nosignatures:

    functional_call
    stack_module_state
    replace_all_batch_norm_modules_

If you're looking for information on fixing Batch Norm modules, please follow the
guidance here

.. toctree::
   :maxdepth: 1

   func.batch_norm
