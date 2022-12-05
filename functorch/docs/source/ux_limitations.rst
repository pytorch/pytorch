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

torch.autograd APIs
-------------------

If you are trying to use a ``torch.autograd`` API like ``torch.autograd.grad``
or ``torch.autograd.backward`` inside of a function being transformed by
:func:`vmap` or one of functorch's AD transforms (:func:`vjp`, :func:`jvp`,
:func:`jacrev`, :func:`jacfwd`), the transform may not be able to transform over it.
If it is unable to do so, you'll receive an error message.

This is a fundamental design limitation in how PyTorch's AD support is implemented
and the reason why we designed the functorch library. Please instead use the functorch
equivalents of the ``torch.autograd`` APIs:
- ``torch.autograd.grad``, ``Tensor.backward`` -> ``functorch.vjp`` or ``functorch.grad``
- ``torch.autograd.functional.jvp`` -> ``functorch.jvp``
- ``torch.autograd.functional.jacobian`` -> ``functorch.jacrev`` or ``functorch.jacfwd``
- ``torch.autograd.functional.hessian`` -> ``functorch.hessian``

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
are those that would cause a Tensor with more elements to be written to a
Tensor with fewer elements. Here's an example of how this can occur:

::

  def f(x, y):
    x.add_(y)
    return x

  x = torch.randn(1)
  y = torch.randn(3)

  # Raises an error because `y` has fewer elements than `x`.
  vmap(f, in_dims=(None, 0))(x, y)

``x`` is a Tensor with one element, ``y`` is a Tensor with three elements.
``x + y`` has three elements (due to broadcasting), but attempting to write
three elements back into ``x``, which only has one element, raises an error
due to attempting to write three elements into a Tensor with a single element.

There is no problem if the Tensor being written to has the same number of
elements (or more):

::

  def f(x, y):
    x.add_(y)
    return x

  x = torch.randn(3)
  y = torch.randn(3)
  expected = x + y

  # Does not raise an error because x and y have the same number of elements.
  vmap(f, in_dims=(0, 0))(x, y)
  assert torch.allclose(x, expected)

Mutation: out= PyTorch Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
:func:`vmap` doesn't support the ``out=`` keyword argument in PyTorch operations.
It will error out gracefully if it encounters that in your code.

This is not a fundamental limitation; we could theoretically support this in the
future but we have chosen not to for now.

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

Data-dependent operations (.item())
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We do not (and will not) support vmap over a user-defined function that calls
``.item()`` on a Tensor. For example, the following will raise an error message:

::

  def f(x):
    return x.item()

  x = torch.randn(3)
  vmap(f)(x)

Please try to rewrite your code to not use ``.item()`` calls.

You may also encounter an error message about using ``.item()`` but you might
not have used it. In those cases, it is possible that PyTorch internally is
calling ``.item()`` -- please file an issue on GitHub and we'll fix
PyTorch internals.

Dynamic shape operations (nonzero and friends)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``vmap(f)`` requires that ``f`` applied to every "example" in your input
returns a Tensor with the same shape. Operations such as ``torch.nonzero``,
``torch.is_nonzero`` are not supported and will error as a result.

To see why, consider the following example:

::

  xs = torch.tensor([[0, 1, 2], [0, 0, 3]])
  vmap(torch.nonzero)(xs)

``torch.nonzero(xs[0])`` returns a Tensor of shape 2;
but ``torch.nonzero(xs[1])`` returns a Tensor of shape 1.
We are unable to construct a single Tensor as an output;
the output would need to be a ragged Tensor (and PyTorch does not yet have
the concept of a ragged Tensor).


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
    most common forms of randomness that we see. If your use case does not fit these forms of randomness, please
    file an issue.
