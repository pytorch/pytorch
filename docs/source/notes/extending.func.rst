.. _func-autograd-function:

Extending torch.func with autograd.Function
===========================================

.. currentmodule:: torch.autograd

So you'd like to use :class:`torch.autograd.Function` with the :mod:`torch.func`
transforms like :func:`torch.vmap`, :func:`torch.func.grad`, etc.

There are two main use cases:

- you wish to call code that does not contain PyTorch operations and
  have it work with function transforms. That is, the :class:`torch.autograd.Function`'s
  forward/backward/etc calls into functions from other systems like C++, CUDA, numpy.
- you wish to specify custom gradient rules, like
  JAX's `custom_vjp/custom_jvp <https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html>`_

PyTorch combines both of these concepts into :class:`torch.autograd.Function`.

Basic Usage
-----------

This guide assumes you are familiar with :ref:`extending-autograd`,
which explains how to use :class:`torch.autograd.Function`.

:class:`torch.autograd.Function` can either have a :meth:`~Function.forward` that accepts a ctx object,
or it can have separate :meth:`~Function.forward` (that does not accept ``ctx``) and a :meth:`~Function.setup_context`
staticmethod that modifies the ``ctx`` object.

Only the latter is supported with function transforms:

- :meth:`~Function.forward` is the code that performs the operation and it should not accept
  a ``ctx`` object.
- ``setup_context(ctx, inputs, output)`` is the code where you can
  call methods on ``ctx``. Here is where you should save Tensors for backward
  (by calling ``ctx.save_for_backward(*tensors)``), or save non-Tensors
  (by assigning them to the ``ctx`` object).

Because :meth:`~Function.setup_context` accepts only ``inputs`` and ``output``,
the only quantities that can be saved are either objects (such as Tensors) in
the inputs or outputs or quantities (like ``Tensor.shape``) derived from them.
If you wish to save a non-input intermediate activation from
:meth:`Function.forward` for backward, then you'll need to return it as an
output from :meth:`~Function.forward` so that it gets passed to
:meth:`~Function.setup_context`.

Depending on the transform,

- to support reverse-mode AD (:func:`torch.func.grad`, :func:`torch.func.vjp`),
  the :class:`torch.autograd.Function` needs a :meth:`~Function.backward` staticmethod.
- to support :func:`torch.vmap`, the :class:`torch.autograd.Function` needs a :meth:`~Function.vmap` staticmethod.
- to support :func:`torch.func.jvp`, the :class:`torch.autograd.Function` needs a :meth:`~Function.jvp` staticmethod.
- to support compositions of transforms (like :func:`torch.func.jacrev`,
  :func:`torch.func.jacfwd`, :func:`torch.func.hessian`) -- you may need multiple
  of the above.

In order for the :class:`torch.autograd.Function` to be arbitrarily composable with function
transforms, we recommend that all other staticmethods other than :meth:`~Function.forward` and
:meth:`~Function.setup_context` must be transformable: that is, they must consist of only PyTorch
operators or call other :class:`torch.autograd.Function` (that may call into C++/CUDA/etc).

Let's go over some examples of common use cases.

Example 1: autograd.Function calls into another system
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A common case is a :class:`torch.autograd.Function` with both forward() and backward() calling
into another system (like C++, CUDA, numpy, triton).

::

    import torch
    import numpy as np

    def to_numpy(tensor):
        return tensor.cpu().numpy()

    class NumpySort(torch.autograd.Function):
        # Note that forward does not take ctx
        @staticmethod
        def forward(x, dim):
            device = x.device
            x = to_numpy(x)
            ind = np.argsort(x, axis=dim)
            ind_inv = np.argsort(ind, axis=dim)
            result = np.take_along_axis(x, ind, axis=dim)
            # Any intermediates to be saved in backward must be returned as
            # outputs.
            return (
                # The desired output
                torch.tensor(result, device=device),
                # intermediate to save for backward
                torch.tensor(ind, device=device),
                # intermediate to save for backward
                torch.tensor(ind_inv, device=device),
            )

        # setup_context is responsible for calling methods and/or assigning to
        # the ctx object. Please do not do additional compute (e.g. add
        # Tensors together) in setup_context.
        @staticmethod
        def setup_context(ctx, inputs, output):
            x, dim = inputs
            # Note that output is whatever you returned from forward.
            # If you returned multiple values, then output is a Tuple of multiple values.
            # If you returned a single Tensor, then output is a Tensor.
            # If you returned a Tuple with a single Tensor, then output is a
            # Tuple with a single Tensor.
            _, ind, ind_inv = output
            ctx.mark_non_differentiable(ind, ind_inv)
            # Tensors must be saved via ctx.save_for_backward. Please do not
            # assign them directly onto the ctx object.
            ctx.save_for_backward(ind, ind_inv)
            # Non-tensors may be saved by assigning them as attributes on the ctx object.
            ctx.dim = dim

        @staticmethod
        def backward(ctx, grad_output, _0, _1):
            # For the autograd.Function to be arbitrarily composable with function
            # transforms, all staticmethod other than forward and setup_context
            # must be implemented in a "transformable" way; that is, they must
            # only consist of PyTorch operations or autograd.Function.
            #
            # For example, this allows us to do double backwards and/or compute
            # second order gradients.
            #
            # We've written the backward pass of NumpySort in terms of another
            # autograd.Function, NumpyTake.
            ind, ind_inv = ctx.saved_tensors
            return NumpyTake.apply(grad_output, ind_inv, ind, ctx.dim), None

    class NumpyTake(torch.autograd.Function):
        @staticmethod
        def forward(x, ind, ind_inv, dim):
            device = x.device
            x = to_numpy(x)
            ind = to_numpy(ind)
            return torch.tensor(np.take_along_axis(x, ind, dim), device=device)

        @staticmethod
        def setup_context(ctx, inputs, output):
            x, ind, ind_inv, dim = inputs
            ctx.save_for_backward(ind, ind_inv)
            ctx.dim = dim

        @staticmethod
        def backward(ctx, grad_output):
            ind, ind_inv = ctx.saved_tensors
            result = NumpyTake.apply(grad_output, ind_inv, ind, ctx.dim)
            return result, None, None, None


Now, to make it easier to use ``NumpySort`` (to hide away the intermediates we
returned as outputs, as well as allow default args and kwargs), we create a new
function that invokes it::

    def numpy_sort(x, dim=-1):
        result, _, _ = NumpySort.apply(x, dim)
        return result

And here's a sanity check::

    x = torch.randn(2, 3)
    grad_x = torch.func.grad(lambda x: numpy_sort(x).sum())(x)
    assert torch.allclose(grad_x, torch.ones_like(x))



Example 2: autograd.Function specifies custom gradient rules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Another common case is an :class:`torch.autograd.Function` that is implemented with PyTorch
operations. PyTorch is able to compute gradients for PyTorch operations automatically,
but perhaps we wish to customize how the gradients are computed. Some reasons why
we may want a custom backward different from the one PyTorch gives us are:

- improving numeric stability
- changing the performance characteristics of the backward
- changing how edge cases are handled (e.g. nans, inf)
- modifying the gradient (e.g. gradient clipping)

Here's an example of an :class:`torch.autograd.Function` for the function ``y = x ** 3`` where we
change the performance characteristics (some computation that would normally happen
during the backward pass, computing dx, happens in the forward pass).

::

  class MyCube(torch.autograd.Function):
      @staticmethod
      def forward(x):
          result = x ** 3
          # In regular PyTorch, if we had just run y = x ** 3, then the backward
          # pass computes dx = 3 * x ** 2. In this autograd.Function, we've done
          # that computation here in the forward pass instead.
          dx = 3 * x ** 2
          return result, dx

      @staticmethod
      def setup_context(ctx, inputs, output):
          x, = inputs
          result, dx = output
          ctx.save_for_backward(x, dx)

      @staticmethod
      def backward(ctx, grad_output, grad_dx):
          x, dx = ctx.saved_tensors
          # In order for the autograd.Function to work with higher-order
          # gradients, we must add the gradient contribution of `dx`.
          result = grad_output * dx + grad_dx * 6 * x
          return result

Now, to make it easier to use ``NumpySort`` (and hide away the intermediates we
returned as outputs) we create a new function that invokes it::

    def my_cube(x):
        result, _ = MyCube.apply(x)
        return result

Here's a sanity check computing the second-order gradients::

    x = torch.randn([])
    ggx = torch.func.grad(torch.func.grad(my_cube))(x)
    assert torch.allclose(ggx, 6 * x)

Limitations and gotchas
^^^^^^^^^^^^^^^^^^^^^^^

.. warning::

    Please read these limitations of :class:`torch.autograd.Function` with torch.func transforms
    carefully. We are not able to catch many of these situations and error out
    gracefully so they will lead to undefined behavior.

Please do not capture Tensors that are being transformed over, have
requires_grad=True, or are dual tensors, into the methods of the
:class:`torch.autograd.Function`. The way to be completely safe is to ensure that the only
Tensors being used inside any method of the :class:`torch.autograd.Function` must be directly
passed as inputs (or via the ctx object) rather than come from outside
the :class:`torch.autograd.Function`.

:class:`torch.autograd.Function` does not handle Tensors in pytrees (arbitrary nested
Python data structures that may or may not contain Tensors). For
those Tensors to be tracked by autograd, they must be passed directly as
an argument to :class:`torch.autograd.Function`. This is in contrast to
jax.{custom_vjp, custom_jvp}, which do accept pytrees.

Please only use :meth:`~torch.autograd.function.FunctionCtx.save_for_backward` or
:meth:`~torch.autograd.function.FunctionCtx.save_for_forward` to save Tensors.
Please do not assign Tensors or collections of Tensors directly onto the ctx object -
these Tensors will not get tracked


:func:`torch.vmap` Support
--------------------------

To use an :class:`torch.autograd.Function` with :func:`torch.vmap`, you must either:

- provide a :meth:`~Function.vmap` staticmethod that tells us the behavior of the :class:`torch.autograd.Function`
  under :func:`torch.vmap`
- ask us to autogenerate it by setting ``generate_vmap_rule=True``.

Automatically generate a vmap rule
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If your :class:`torch.autograd.Function` fulfills the following additional constraints, then we
are able to generate a vmap rule for it. If it doesn't fulfill the constraints or if you
want custom behavior under vmap, please manually define a vmap staticmethod (see next section).

.. warning::

     We are not easily able to check for the following constraints and error
     out gracefully. Violation of the constraints may lead to undefined
     behavior.

- The :class:`torch.autograd.Function`'s :meth:`~Function.forward`, :meth:`~Function.backward` (if it exists) and :meth:`~Function.jvp`
  (if it exists) staticmethods must be transformable via :func:`torch.vmap`. That
  is, they must consist of only PyTorch operations (as opposed to e.g. NumPy or custom
  CUDA kernels).

Example::

    class MyCube(torch.autograd.Function):
        # Set generate_vmap_rule to True to ask PyTorch to automatically generate
        # a vmap rule.
        generate_vmap_rule = True

        @staticmethod
        def forward(x):
            result = x ** 3
            dx = 3 * x ** 2
            return result, dx

        @staticmethod
        def setup_context(ctx, inputs, output):
            x, = inputs
            result, dx = output
            ctx.save_for_backward(x, dx)

        @staticmethod
        def backward(ctx, grad_output, grad_dx):
            x, dx = ctx.saved_tensors
            result = grad_output * dx + grad_dx * 6 * x
            return result

    def my_cube(x):
        result, dx = MyCube.apply(x)
        return result

    x = torch.randn(3)
    result = torch.vmap(my_cube)(x)
    assert torch.allclose(result, x ** 3)


Defining the vmap staticmethod
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If your :class:`torch.autograd.Function` calls into another system (like NumPy, C++, CUDA, triton),
then to get it to work with :func:`torch.vmap` or transforms that use it, you'll
need to manually define a :meth:`~Function.vmap` staticmethod.

Depending on what transforms you want to use and your use case, you may not need
to add a :meth:`~Function.vmap` staticmethod to all of your :class:`torch.autograd.Function`:

- For example, :func:`torch.func.jacrev` performs :func:`~torch.vmap` over the backward pass.
  So if you're only interested in using :func:`torch.func.jacrev`, only
  the :meth:`~Function.backward` staticmethod needs to be vmappable.

We do recommend ensuring all of your :class:`torch.autograd.Function` have support for
:func:`torch.vmap` though, especially if you are writing a third-party library and you want your
:class:`torch.autograd.Function` to work with all combinations of :func:`torch.func` transforms.

Conceptually, the vmap staticmethod is responsible for defining how the :meth:`~Function.forward`
should behave under :func:`torch.vmap`. That is, it defines how to transform
the :meth:`~Function.forward` to run over inputs with an additional dimension (the dimension
being vmapped over). This is similar to how :func:`torch.vmap` is implemented over
PyTorch operations: for each operation, we define a vmap rule (sometimes also
referred to as a "batching rule").

Here's how to define the :meth:`~Function.vmap` staticmethod:

- the signature is ``vmap(info, in_dims: Tuple[Optional[int]], *args)``, where
  ``*args`` is the same as the args to :meth:`~Function.forward`.
- The vmap staticmethod is responsible for defining how the :meth:`~Function.forward` should behave
  under :func:`torch.vmap`. That is, given inputs with an additional dimension
  (specified by ``in_dims``), how do we compute the batched version of :meth:`~Function.forward`?
- For each arg in ``args``, ``in_dims`` has a corresponding ``Optional[int]``.
  It is ``None`` if the arg is not a Tensor or if the arg is not being vmapped over,
  otherwise, it is an integer specifying what dimension of the Tensor is being vmapped
  over.
- ``info`` is a collection of additional metadata that may be helpful:
  ``info.batch_size`` specifies the size of the dimension being vmapped over, while
  ``info.randomness`` is the ``randomness`` option that was passed to :func:`torch.vmap`.
- The return of the vmap staticmethod is a tuple of ``(output, out_dims)``. Similar
  to ``in_dims``, ``out_dims`` should be of the same structure as ``output`` and contain
  one ``out_dim`` per output that specifies if the output has the vmapped
  dimension and what index it is in.


Example::

    def to_numpy(tensor):
        return tensor.cpu().numpy()

    class NumpySort(torch.autograd.Function):
        @staticmethod
        def forward(x, dim):
            device = x.device
            x = to_numpy(x)
            ind = np.argsort(x, axis=dim)
            ind_inv = np.argsort(ind, axis=dim)
            result = np.take_along_axis(x, ind, axis=dim)
            return (
                torch.tensor(result, device=device),
                torch.tensor(ind, device=device),
                torch.tensor(ind_inv, device=device),
            )

        @staticmethod
        def setup_context(ctx, inputs, output):
            x, dim = inputs
            _, ind, ind_inv = output
            ctx.mark_non_differentiable(ind, ind_inv)
            ctx.save_for_backward(ind, ind_inv)
            ctx.dim = dim

        @staticmethod
        def backward(ctx, grad_output, _0, _1):
            ind, ind_inv = ctx.saved_tensors
            return NumpyTake.apply(grad_output, ind_inv, ind, ctx.dim), None

        # The signature of the vmap staticmethod is:
        # vmap(info, in_dims: Tuple[Optional[int]], *args)
        # where *args is the same as the arguments to `forward`.
        @staticmethod
        def vmap(info, in_dims, x, dim):
            # For every input (x and dim), in_dims stores an Optional[int]
            # that is:
            # - None if the input is not being vmapped over or if the input
            #   is not a Tensor
            # - an integer if the input is being vmapped over that represents
            #   the index of the dimension being vmapped over.
            x_bdim, _ = in_dims

            # A "vmap rule" is the logic of how to perform the operation given
            # inputs with one additional dimension. In NumpySort, x has an
            # additional dimension (x_bdim). The vmap rule is simply
            # to call NumpySort again but pass it a different `dim`.
            x = x.movedim(x_bdim, 0)
            # Handle negative dims correctly
            dim = dim if dim >= 0 else dim + x.dim() - 1
            result = NumpySort.apply(x, dim + 1)

            # The vmap rule must return a tuple of two things
            # 1. the output. Should be the same amount of things
            #    as returned by the forward().
            # 2. one Optional[int] for each output specifying if each output
            # is being vmapped over, and if so, the index of the
            # dimension being vmapped over.
            #
            # NumpySort.forward returns a Tuple of 3 Tensors. Since we moved the
            # dimension being vmapped over to the front of `x`, that appears at
            # dimension 0 of all outputs.
            # The return is (output, out_dims) -- output is a tuple of 3 Tensors
            # and out_dims is a Tuple of 3 Optional[int]
            return NumpySort.apply(x, dim + 1), (0, 0, 0)

    class NumpyTake(torch.autograd.Function):
        @staticmethod
        def forward(x, ind, ind_inv, dim):
            device = x.device
            x = to_numpy(x)
            ind = to_numpy(ind)
            return torch.tensor(np.take_along_axis(x, ind, dim), device=device)

        @staticmethod
        def setup_context(ctx, inputs, output):
            x, ind, ind_inv, dim = inputs
            ctx.save_for_backward(ind, ind_inv)
            ctx.dim = dim

        @staticmethod
        def backward(ctx, grad_output):
            ind, ind_inv = ctx.saved_tensors
            result = NumpyTake.apply(grad_output, ind_inv, ind, ctx.dim)
            return result, None, None, None

        @staticmethod
        def vmap(info, in_dims, x, ind, ind_inv, dim):
            x_bdim, ind_bdim, ind_inv_bdim, _ = in_dims

            # The strategy is: expand {x, ind, ind_inv} to all have the dimension
            # being vmapped over.
            # Then, call back into NumpyTake(expanded_x, expanded_ind, expanded_ind_inv, new_dim).

            # Handle negative dims by wrapping them to be positive
            logical_dim = x.dim() if x_bdim is None else x_bdim - 1
            dim = dim if dim >= 0 else dim + logical_dim

            def maybe_expand_bdim_at_front(x, x_bdim):
                if x_bdim is None:
                    return x.expand(info.batch_size, *x.shape)
                return x.movedim(x_bdim, 0)

            # If the Tensor doesn't have the dimension being vmapped over,
            # expand it out. Otherwise, move it to the front of the Tensor
            x = maybe_expand_bdim_at_front(x, x_bdim)
            ind = maybe_expand_bdim_at_front(ind, ind_bdim)
            ind_inv = maybe_expand_bdim_at_front(ind_inv, ind_inv_bdim)

            # The return is a tuple (output, out_dims). Since output is a Tensor,
            # then out_dims is an Optional[int] (instead of being a Tuple).
            return NumpyTake.apply(x, ind, ind_inv, dim + 1), 0

    def numpy_sort(x, dim=-1):
        result, _, _ = NumpySort.apply(x, dim)
        return result

    x = torch.randn(2, 3)
    result = torch.vmap(numpy_sort)(x)
    assert torch.allclose(result, numpy_sort(result, 1))


.. note::

    The vmap staticmethod should aim to preserve the semantics of the
    entire :class:`~torch.autograd.Function`. That is, (pseudocode) ``grad(vmap(MyFunc))``
    should be replaceable with a ``grad(map(MyFunc))``.

    If your autograd.Function has any custom behavior in the backward pass, please
    keep this in mind.

.. note::

    It is a legitimate use case to write a custom vmap staticmethod for a
    :class:`~torch.autograd.Function` that PyTorch is able to generate a vmap
    rule for via ``generate_vmap_rule=True``. You may wish to do this if the
    generated vmap rule doesn't have the semantics you're looking for.

:func:`torch.func.jvp` Support
------------------------------

To support forward-mode AD, a :class:`torch.autograd.Function` must have a :meth:`~Function.jvp` staticmethod.
Please see :ref:`forward-ad-autograd-function` for details.
