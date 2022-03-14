.. currentmodule:: functorch

UX Limitations
==============

functorch, like `JAX <https://github.com/google/jax>`_, has restrictions around
what can be transformed. In general, JAX’s limitations are that transforms
only work with pure functions: that is, functions where the output is completely
determined by the input and that do not involve side effects (like mutation).

We have a similar guarantee: our transforms work well with pure functions.
However, we do support certain in-place operations. On one hand, writing code
compatible with functorch transforms may involve changing how you write PyTorch
code, on the other hand, you may find that our transforms let you express things
that were previously difficult to express in PyTorch.

General limitations
-------------------

All functorch transforms share a limitation in that a function should not
assign to global variables. Instead, all outputs to a function must be returned
from the function. This restriction comes from how functorch is implemented:
each transform wraps Tensor inputs in special functorch Tensor subclasses
that facilitate the transform.

So, instead of the following:

::

  import torch
  from functorch import grad

  # Don't do this
  intermediate = None

  def f(x):
    global intermediate
    intermediate = x.sin()
    z = intermediate.sin()
    return z

  x = torch.randn([])
  grad_x = grad(f)(x)

Please rewrite ``f`` to return ``intermediate``:

::

  def f(x):
    intermediate = x.sin()
    z = intermediate.sin()
    return z, intermediate

  grad_x, intermediate = grad(f, has_aux=True)(x)

vmap limitations
----------------

.. note::
  :func:`vmap` is our most restrictive transform.
  The grad-related transforms (:func:`grad`, :func:`vjp`, :func:`jvp`) do not
  have these limitations. :func:`jacfwd` (and :func:`hessian`, which is
  implemented with :func:`jacfwd`) is a composition of :func:`vmap` and
  :func:`jvp` so it also has these limitations.

``vmap(func)`` is a transform that returns a function that maps ``func`` over
some new dimension of each input Tensor. The mental model for vmap is that it is
like running a for-loop: for pure functions (i.e. in the absence of side
effects), ``vmap(f)(x)`` is equivalent to:

::

  torch.stack([f(x_i) for x_i in x.unbind(0)])

Mutation: Arbitrary mutation of Python data structures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the presence of side effects, :func:`vmap` no longer acts like it is running
a for-loop. For example, the following function:

::

  def f(x, list):
    list.pop()
    print("hello!")
    return x.sum(0)

  x = torch.randn(3, 1)
  lst = [0, 1, 2, 3]

  result = vmap(f, in_dims=(0, None))(x, lst)

will print "hello!" once and pop only one element from ``lst``.


:func:`vmap` executes `f` a single time, so all side effects only happen once.

This is a consequence of how vmap is implemented. functorch has a special,
internal BatchedTensor class. ``vmap(f)(*inputs)`` takes all Tensor inputs,
turns them into BatchedTensors, and calls ``f(*batched_tensor_inputs)``.
BatchedTensor overrides the PyTorch API to produce batched (i.e. vectorized)
behavior for each PyTorch operator.


Mutation: in-place PyTorch Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`vmap` will raise an error if it encounters an unsupported PyTorch
in-place operation and it will succeed otherwise. Unsupported operations
are those that would cause a Tensor with more memory to be written to a
Tensor with less memory. Here's an example of how this can occur:

::

  def f(x, y):
    x.add_(y)
  return x

  x = torch.randn(1)
  y = torch.randn(3)

  # Raises an error
  vmap(f, in_dims=(None, 0))(x, y)

``x`` is a Tensor with one element, ``y`` is a Tensor with three elements.
``x + y`` has three elements (due to broadcasting), but attempting to write
three elements back into ``x``, which only has one element, raises an error
due to there not being enough memory to hold three elements.

There is no problem if there is sufficient memory for the in-place operations
to occur:

::

  def f(x, y):
    x.add_(y)
    return x

  x = torch.randn(3)
  y = torch.randn(3)
  expected = x + y

  # Raises an error
  vmap(f, in_dims=(0, 0))(x, y)
  assert torch.allclose(x, expected)

Data-dependent Python control flow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We don't yet support ``vmap`` over data-dependent control flow. Data-dependent
control flow is when the condition of an if-statement, while-loop, or
for-loop is a Tensor that is being ``vmap``'ed over. For example, the
following will raise an error message:

::

  def relu(x):
    if x > 0:
      return x
    return 0

  x = torch.randn(3)
  vmap(relu)(x)

However, any control flow that is not dependent on the values in ``vmap``'ed
tensors will work:

::

  def custom_dot(x):
    if x.dim() == 1:
      return torch.dot(x, x)
    return (x * x).sum()

  x = torch.randn(3)
  vmap(custom_dot)(x)

JAX supports transforming over
`data-dependent control flow <https://jax.readthedocs.io/en/latest/jax.lax.html#control-flow-operators>`_
using special control flow operators (e.g. ``jax.lax.cond``, ``jax.lax.while_loop``).
We're investigating adding equivalents of those to functorch
(open an issue on `GitHub <https://github.com/pytorch/functorch>`_ to voice your support!).

Randomness
----------
The user's intention when calling a random operation can be unclear. Specifically, some users may want
the random behavior to be the same across batches while others may want it to differ across batches.
To address this, ``vmap`` takes a randomness flag.

The flag can only be passed to vmap and can take on 3 values, "error," "different," or "same," defaulting
to error. Under "error" mode, any call to a random function will produce an error asking the user to use
one of the other two flags based on their use case.

Under "different" randomness, elements in a batch produce different random values. For instance,

::

  def add_noise(x):
    y = torch.randn(())  # y will be different across the batch
    return x + y

  x = torch.ones(3)
  result = vmap(add_noise, randomness="different")(x)  # we get 3 different values

Under "same" randomness, elements in a batch produce same random values. For instance,

::

  def add_noise(x):
    y = torch.randn(())  # y will be the same across the batch
    return x + y

  x = torch.ones(3)
  result = vmap(add_noise, randomness="same")(x)  # we get the same value, repeated 3 times


.. warning::
    Our system only determine the randomness behavior of PyTorch operators and cannot control the
    behavior of other libraries, like numpy. This is similar to JAX's limitations with their solutions

.. note::
    Multiple vmap calls using either type of supported randomness will not produce
    the same results. Like with standard PyTorch, a user can get randomness reproducibility through
    either using ``torch.manual_seed()`` outside of vmap or by using generators.

.. note::
    Finally, our randomness differs from JAX because we aren't using a stateless PRNG, in part because PyTorch
    doesn't have full support for a stateless PRNG. Instead, we've introduced a flag system to allow for the
    most common forms of randmoness that we see. If your use case does not fit these forms of randomness, please
    file an issue.
