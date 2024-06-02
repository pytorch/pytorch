Extending PyTorch
=================

In this note we'll cover ways of extending :mod:`torch.nn`,
:mod:`torch.autograd`, :mod:`torch`, and writing custom C++ extensions.

Adding new operators
--------------------

PyTorch offers a large library of operators that work on Tensors (e.g. :func:`torch.add`,
:func:`torch.sum`, etc). However, you may wish to bring a new custom operation to PyTorch
and have it behave like PyTorch's built-in operators. In order to do so, you must
register the custom operation with PyTorch via the Python :ref:`torch-library-docs` or C++ TORCH_LIBRARY
APIs.


Please see :ref:`custom-ops-landing-page` for more details.

.. _extending-autograd:

Extending :mod:`torch.autograd`
-------------------------------

.. currentmodule:: torch.autograd

Adding operations to :mod:`~torch.autograd` requires implementing a new
:class:`Function` subclass for each operation. Recall that Functions
are what :mod:`~torch.autograd` uses to encode the operation history and compute
gradients.

The first part of this doc is focused on backward mode AD as it is the most widely used
feature. A section at the end discusses the extensions for forward mode AD.

When to use
^^^^^^^^^^^
In general, implement a custom function if you want to perform computations in your model
that are not differentiable or rely on non-PyTorch libraries (e.g., NumPy), but
still wish for your operation to chain with other ops and work with the autograd engine.

In some situations, custom functions can also be used to improve performance and
memory usage: If you implemented your forward and backward passes using a
`C++ extension <https://pytorch.org/tutorials/advanced/cpp_extension.html>`_,
you can wrap them in :class:`~Function` to interface with the autograd
engine. If you'd like to reduce the number of buffers saved for the backward pass,
custom functions can be used to combine ops together.

When not to use
^^^^^^^^^^^^^^^
If you can already write your function in terms of PyTorch's built-in ops, its
backward graph is (most likely) already able to be recorded by autograd. In this case, you do
not need to implement the backward function yourself. Consider using a plain
old Python function.

If you need to maintain state, i.e., trainable parameters, you should (also) use a
custom module. See the section below for more information on extending :mod:`torch.nn`.

If you'd like to alter the gradients during the backward pass or perform a side
effect, consider registering a
`tensor <https://pytorch.org/docs/stable/generated/torch.Tensor.register_hook.html#torch.Tensor.register_hook>`_ or
`Module <https://pytorch.org/docs/stable/notes/modules.html#module-hooks>`_ hook.

How to use
^^^^^^^^^^
Take the following steps:
1. Subclass :class:`~Function` and implement the :meth:`~Function.forward`,
(optional) :meth:`~Function.setup_context` and
:meth:`~Function.backward` methods.
2. Call the proper methods on the `ctx` argument.
3. Declare whether your function supports
`double backward <https://pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html>`_.
4. Validate whether your gradients are correct using gradcheck.

**Step 1:** After subclassing :class:`Function`, you'll need to define 3 methods:

- :meth:`~Function.forward` is the code that performs the operation. It can take
  as many arguments as you want, with some of them being optional, if you
  specify the default values. All kinds of Python objects are accepted here.
  :class:`Tensor` arguments that track history (i.e., with
  ``requires_grad=True``) will be converted to ones that don't track history
  before the call, and their use will be registered in the graph. Note that this
  logic won't traverse lists/dicts/any other data structures and will only
  consider tensors that are direct arguments to the call. You can
  return either a single :class:`Tensor` output, or a :class:`tuple` of
  tensors if there are multiple outputs. Also, please refer to the
  docs of :class:`Function` to find descriptions of useful methods that can be
  called only from :meth:`~Function.forward`.
- :meth:`~Function.setup_context` (optional). One can either write a "combined" :meth:`~Function.forward` that
  accepts a ``ctx`` object or (as of PyTorch 2.0) a separate :meth:`~Function.forward` that does
  not accept ``ctx`` and a :meth:`~Function.setup_context` method where the ``ctx`` modification happens.
  The :meth:`~Function.forward` should have the compute and :meth:`~Function.setup_context` should
  only be responsible for the ``ctx`` modification (and not have any compute).
  In general the separate :meth:`~Function.forward` and :meth:`~Function.setup_context` is closer to how
  PyTorch native operations work and therefore more composable with various PyTorch subsystems.
  See :ref:`combining-forward-context` for more details.
- :meth:`~Function.backward` (or :meth:`~Function.vjp`) defines the gradient formula.
  It will be given as many :class:`Tensor` arguments as there were outputs, with each
  of them representing gradient w.r.t. that output. It is important NEVER to modify
  these in-place. It should return as many tensors as there
  were inputs, with each of them containing the gradient w.r.t. its
  corresponding input. If your inputs didn't require gradient
  (:attr:`~ctx.needs_input_grad` is a tuple of booleans indicating
  whether each input needs gradient computation), or were non-:class:`Tensor`
  objects, you can return :class:`python:None`. Also, if you have optional
  arguments to :meth:`~Function.forward` you can return more gradients than there
  were inputs, as long as they're all :any:`python:None`.

**Step 2:** It is your responsibility to use the functions in ``ctx``
properly in order to ensure that the new :class:`Function` works properly with
the autograd engine.

- :meth:`~torch.autograd.function.FunctionCtx.save_for_backward` must be
  used to save any tensors to be used in the backward pass. Non-tensors should
  be stored directly on `ctx`. If tensors that are neither input nor output
  are saved for backward your :class:`~Function` may not support double backward
  (see step 3).
- :meth:`~torch.autograd.function.FunctionCtx.mark_dirty` must be used to
  mark any input that is modified inplace by the forward function.
- :meth:`~torch.autograd.function.FunctionCtx.mark_non_differentiable` must
  be used to tell the engine if an output is not differentiable. By
  default all output tensors that are of differentiable type will be set
  to require gradient. Tensors of non-differentiable type (i.e., integral types)
  are never marked as requiring gradients.
- :meth:`~torch.autograd.function.FunctionCtx.set_materialize_grads` can be
  used to tell the autograd engine to optimize gradient computations in the cases where
  the output does not depend on the input by not materializing grad tensors given to backward
  function. That is, if set to False, None object in Python or "undefined tensor" (tensor x for
  which x.defined() is False) in C++ will not be converted to a tensor filled with zeros prior
  to calling backward, and so your code will need to handle such objects as if they were
  tensors filled with zeros. The default value of this setting is True.

**Step 3:** If your :class:`~Function` does not support double backward
you should explicitly declare this by decorating backward with the
:func:`~function.once_differentiable`. With this decorator, attempts to
perform double backward through your function will produce an error.
See our double backward tutorial for more information on double backward.

**Step 4:** It is recommended that you use :func:`torch.autograd.gradcheck`
to check whether your backward function correctly computes gradients of the
forward by computing the Jacobian matrix using your backward function and
comparing the value element-wise with the Jacobian computed numerically using
finite-differencing.

Example
^^^^^^^

Below you can find code for a ``Linear`` function, with
additional comments::

    # Inherit from Function
    class LinearFunction(Function):

        # Note that forward, setup_context, and backward are @staticmethods
        @staticmethod
        def forward(input, weight, bias):
            output = input.mm(weight.t())
            if bias is not None:
                output += bias.unsqueeze(0).expand_as(output)
            return output

        @staticmethod
        # inputs is a Tuple of all of the inputs passed to forward.
        # output is the output of the forward().
        def setup_context(ctx, inputs, output):
            input, weight, bias = inputs
            ctx.save_for_backward(input, weight, bias)

        # This function has only a single output, so it gets only one gradient
        @staticmethod
        def backward(ctx, grad_output):
            # This is a pattern that is very convenient - at the top of backward
            # unpack saved_tensors and initialize all gradients w.r.t. inputs to
            # None. Thanks to the fact that additional trailing Nones are
            # ignored, the return statement is simple even when the function has
            # optional inputs.
            input, weight, bias = ctx.saved_tensors
            grad_input = grad_weight = grad_bias = None

            # These needs_input_grad checks are optional and there only to
            # improve efficiency. If you want to make your code simpler, you can
            # skip them. Returning gradients for inputs that don't require it is
            # not an error.
            if ctx.needs_input_grad[0]:
                grad_input = grad_output.mm(weight)
            if ctx.needs_input_grad[1]:
                grad_weight = grad_output.t().mm(input)
            if bias is not None and ctx.needs_input_grad[2]:
                grad_bias = grad_output.sum(0)

            return grad_input, grad_weight, grad_bias

Now, to make it easier to use these custom ops, we recommend either aliasing
them or wrapping them in a function. Wrapping in a function lets us support
default arguments and keyword arguments::

    # Option 1: alias
    linear = LinearFunction.apply

    # Option 2: wrap in a function, to support default args and keyword args.
    def linear(input, weight, bias=None):
        return LinearFunction.apply(input, weight, bias)

Here, we give an additional example of a function that is parametrized by
non-Tensor arguments::

    class MulConstant(Function):
        @staticmethod
        def forward(tensor, constant):
            return tensor * constant

        @staticmethod
        def setup_context(ctx, inputs, output):
            # ctx is a context object that can be used to stash information
            # for backward computation
            tensor, constant = inputs
            ctx.constant = constant

        @staticmethod
        def backward(ctx, grad_output):
            # We return as many input gradients as there were arguments.
            # Gradients of non-Tensor arguments to forward must be None.
            return grad_output * ctx.constant, None

And here, we optimize the above example by calling set_materialize_grads(False)::

    class MulConstant(Function):
        @staticmethod
        def forward(tensor, constant):
            return tensor * constant

        @staticmethod
        def setup_context(ctx, inputs, output):
            tensor, constant = inputs
            ctx.set_materialize_grads(False)
            ctx.constant = constant

        @staticmethod
        def backward(ctx, grad_output):
            # Here we must handle None grad_output tensor. In this case we
            # can skip unnecessary computations and just return None.
            if grad_output is None:
                return None, None

            # We return as many input gradients as there were arguments.
            # Gradients of non-Tensor arguments to forward must be None.
            return grad_output * ctx.constant, None

If you need any "intermediate" Tensors computed in :meth:`~Function.forward` to be saved,
either they must be returned as outputs, or combine ``forward`` and :meth:`~Function.setup_context`
(see :ref:`combining-forward-context`).
Note that this means if you want gradients to flow through those intermediate values, you
need to define the gradient formula for them (see also
`the double backward tutorial <https://pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html>`_
)::

    class MyCube(torch.autograd.Function):
        @staticmethod
        def forward(x):
            # We wish to save dx for backward. In order to do so, it must
            # be returned as an output.
            dx = 3 * x ** 2
            result = x ** 3
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
            # gradients, we must add the gradient contribution of `dx`,
            # which is grad_dx * 6 * x.
            result = grad_output * dx + grad_dx * 6 * x
            return result

    # Wrap MyCube in a function so that it is clearer what the output is
    def my_cube(x):
        result, dx = MyCube.apply(x)
        return result

.. note::
    Inputs to ``backward``, i.e., :attr:`grad_output`, can also be tensors that
    track history. So if ``backward`` is implemented with differentiable
    operations, (e.g., invocation of another custom
    :class:`~torch.autograd.Function`), higher order derivatives will work.
    In this case, the tensors saved with ``save_for_backward`` can also be used
    in the backward and have gradients flowing back but tensors saved in the ``ctx``
    won't have gradients flowing back for them.
    If you need gradients to flow back for a Tensor saved in the ``ctx``, you should
    make it an output of the custom ``Function`` and save it with ``save_for_backward``.

You probably want to check if the backward method you implemented actually
computes the derivatives of your function. It is possible by comparing with
numerical approximations using small finite differences::

    from torch.autograd import gradcheck

    # gradcheck takes a tuple of tensors as input, check if your gradient
    # evaluated with these tensors are close enough to numerical
    # approximations and returns True if they all verify this condition.
    input = (torch.randn(20,20,dtype=torch.double,requires_grad=True), torch.randn(30,20,dtype=torch.double,requires_grad=True))
    test = gradcheck(linear, input, eps=1e-6, atol=1e-4)
    print(test)

See :ref:`grad-check` for more details on finite-difference gradient comparisons.
If your function is used in higher order derivatives (differentiating the backward pass) you
can use the ``gradgradcheck`` function from the same package to check higher order derivatives.

.. _combining-forward-context:

Combined or separate :meth:`~Function.forward` and :meth:`~Function.setup_context`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are two main ways to define :class:`~Function`. Either:

- define a :meth:`~Function.forward` that combines the forward compute logic with :meth:`~Function.setup_context`
- (as of PyTorch 2.0) define a separate :meth:`~Function.forward` and :meth:`~Function.setup_context`

We recommend the second option (separate :meth:`~Function.forward` and :meth:`~Function.setup_context`)
because that is closer to how PyTorch native operations are implemented and it composes
with :mod:`torch.func` transforms. However, we plan to support both approaches going forward;
combining :meth:`~Function.forward` with :meth:`~Function.setup_context`: leads to more flexibility since
you are able to save intermediates without returning them as output.

Please see the previous section for how to define :class:`~Function` with separate
:meth:`~Function.forward` and :meth:`~Function.setup_context`.

Here is an example of how to define a :class:`Function` with combined :meth:`~Function.forward` and
:meth:`~Function.setup_context`::

    class LinearFunction(Function):
        @staticmethod
        # ctx is the first argument to forward
        def forward(ctx, input, weight, bias=None):
            # The forward pass can use ctx.
            ctx.save_for_backward(input, weight, bias)
            output = input.mm(weight.t())
            if bias is not None:
                output += bias.unsqueeze(0).expand_as(output)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            input, weight, bias = ctx.saved_tensors
            grad_input = grad_weight = grad_bias = None

            if ctx.needs_input_grad[0]:
                grad_input = grad_output.mm(weight)
            if ctx.needs_input_grad[1]:
                grad_weight = grad_output.t().mm(input)
            if bias is not None and ctx.needs_input_grad[2]:
                grad_bias = grad_output.sum(0)

            return grad_input, grad_weight, grad_bias

.. _forward-ad-autograd-function:

Forward mode AD
^^^^^^^^^^^^^^^

Overriding the forward mode AD formula has a very similar API with some different subtleties.
You can implement the :meth:`~Function.jvp` function.

It will be given as many :class:`Tensor` arguments as there were inputs, with each
of them representing gradient w.r.t. that input. It should return as many tensors as there
were outputs, with each of them containing the gradient w.r.t. its corresponding output.
The :meth:`~Function.jvp` will be called just after the :meth:`~Function.forward`
method, before the :meth:`~Function.apply` returns.

:meth:`~Function.jvp` has a few subtle differences with the :meth:`~Function.backward` function:

- You can use the `ctx` to pass any data from the :meth:`~Function.forward` to the :meth:`~Function.jvp` function.
  If that state will not be needed for the :meth:`~Function.backward`,
  you can explicitly free it by doing ``del ctx.foo`` at the end of the :meth:`~Function.jvp` function.
- The implementation of :meth:`~Function.jvp` must be backward differentiable or explicitly check that
  none of the given forward mode gradient has ``requires_grad`` set.
- The :meth:`~Function.jvp` function must match the view/inplace behavior of :meth:`~Function.forward`.
  For example, if the ``i`` th input is modified inplace, then the ``i`` th gradient must be updated inplace.
  Similarly, if the ``j`` th output is a view of the ``k`` th input. Then the returned ``j`` th output gradient must be
  a view of the given ``k`` th input gradient.
- Because the user cannot specify which gradient needs to be computed, the :meth:`~Function.jvp` function should
  always compute gradients for all the outputs.
- The forward mode gradients do respect the flag set by :meth:`~torch.autograd.function.FunctionCtx.set_materialize_grads`
  and you can get `None` input gradients when this is disabled.

:mod:`torch.func` transforms and/or :func:`torch.vmap`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Please see :ref:`func-autograd-function` for details.


Extending :mod:`torch.nn`
-------------------------

.. currentmodule:: torch.nn

:mod:`~torch.nn` exports two kinds of interfaces - modules and their functional
versions. You can extend it in both ways, but we recommend using modules for
all kinds of layers, that hold any parameters or buffers, and recommend using
a functional form parameter-less operations like activation functions, pooling,
etc.

Adding a functional version of an operation is already fully covered in the
section above.

Adding a :class:`Module`
^^^^^^^^^^^^^^^^^^^^^^^^

Since :mod:`~torch.nn` heavily utilizes :mod:`~torch.autograd`, adding a new
:class:`Module` requires implementing a :class:`~torch.autograd.Function`
that performs the operation and can compute the gradient. From now on let's
assume that we want to implement a ``Linear`` module and we have the function
implemented as in the listing above. There's very little code required to
add this. Now, there are two functions that need to be implemented:

- ``__init__`` (*optional*) - takes in arguments such as kernel sizes, numbers
  of features, etc. and initializes parameters and buffers.
- :meth:`~Module.forward` - instantiates a :class:`~torch.autograd.Function` and
  uses it to perform the operation. It's very similar to a functional wrapper
  shown above.

This is how a ``Linear`` module can be implemented::

    class Linear(nn.Module):
        def __init__(self, input_features, output_features, bias=True):
            super().__init__()
            self.input_features = input_features
            self.output_features = output_features

            # nn.Parameter is a special kind of Tensor, that will get
            # automatically registered as Module's parameter once it's assigned
            # as an attribute. Parameters and buffers need to be registered, or
            # they won't appear in .parameters() (doesn't apply to buffers), and
            # won't be converted when e.g. .cuda() is called. You can use
            # .register_buffer() to register buffers.
            # nn.Parameters require gradients by default.
            self.weight = nn.Parameter(torch.empty(output_features, input_features))
            if bias:
                self.bias = nn.Parameter(torch.empty(output_features))
            else:
                # You should always register all possible parameters, but the
                # optional ones can be None if you want.
                self.register_parameter('bias', None)

            # Not a very smart way to initialize weights
            nn.init.uniform_(self.weight, -0.1, 0.1)
            if self.bias is not None:
                nn.init.uniform_(self.bias, -0.1, 0.1)

        def forward(self, input):
            # See the autograd section for explanation of what happens here.
            return LinearFunction.apply(input, self.weight, self.bias)

        def extra_repr(self):
            # (Optional)Set the extra information about this module. You can test
            # it by printing an object of this class.
            return 'input_features={}, output_features={}, bias={}'.format(
                self.input_features, self.output_features, self.bias is not None
            )

.. _extending-torch-python:

Extending :mod:`torch` Python API
---------------------------------

You can create custom types that emulate :class:`Tensor` by defining a custom
class with methods that match :class:`Tensor`. But what if you want to be able
to pass these types to functions like :func:`torch.add` in the top-level
:mod:`torch` namespace that accept :class:`Tensor` operands?

If your custom Python type defines a method named ``__torch_function__``, PyTorch
will invoke your ``__torch_function__`` implementation when an instance of your
custom class is passed to a function in the :mod:`torch` namespace. This makes
it possible to define custom implementations for any of the functions in the
:mod:`torch` namespace which your ``__torch_function__`` implementation can call,
allowing your users to make use of your custom type with existing PyTorch
workflows that they have already written for :class:`Tensor`. This works with
"duck" types that are unrelated to :class:`Tensor` as well as user-defined
subclasses of :class:`Tensor`.

Extending :mod:`torch` with a :class:`Tensor`-like type
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note:: This functionality is inspired by the NumPy ``__array_function__``
          protocol. See `the NumPy documentation
          <https://numpy.org/doc/stable/user/basics.dispatch.html#basics-dispatch>`_
          and `NEP-0018
          <https://numpy.org/neps/nep-0018-array-function-protocol.html>`_ for
          more details.

To make this concrete, let's begin with a simple example that illustrates the
API dispatch mechanism. We'll create a custom type that represents a 2D scalar
tensor, parametrized by the order ``N`` and value along the diagonal entries,
``value``::

     class ScalarTensor(object):
        def __init__(self, N, value):
            self._N = N
            self._value = value

        def __repr__(self):
            return "ScalarTensor(N={}, value={})".format(self._N, self._value)

        def tensor(self):
            return self._value * torch.eye(self._N)

This first iteration of the design isn't very useful. The main functionality of
``ScalarTensor`` is to provide a more compact string representation of a scalar
tensor than in the base tensor class::

  >>> d = ScalarTensor(5, 2)
  >>> d
  ScalarTensor(N=5, value=2)
  >>> d.tensor()
  tensor([[2., 0., 0., 0., 0.],
          [0., 2., 0., 0., 0.],
          [0., 0., 2., 0., 0.],
          [0., 0., 0., 2., 0.],
          [0., 0., 0., 0., 2.]])

If we try to use this object with the :mod:`torch` API, we will run
into issues::

  >>> import torch
  >>> torch.mean(d)
  TypeError: mean(): argument 'input' (position 1) must be Tensor, not ScalarTensor

Adding a ``__torch_function__`` implementation to ``ScalarTensor`` makes it
possible for the above operation to succeed. Let's re-do our implementation,
this time adding a ``__torch_function__`` implementation::

  HANDLED_FUNCTIONS = {}
  class ScalarTensor(object):
      def __init__(self, N, value):
          self._N = N
          self._value = value

      def __repr__(self):
          return "ScalarTensor(N={}, value={})".format(self._N, self._value)

      def tensor(self):
          return self._value * torch.eye(self._N)

      @classmethod
      def __torch_function__(cls, func, types, args=(), kwargs=None):
          if kwargs is None:
              kwargs = {}
          if func not in HANDLED_FUNCTIONS or not all(
              issubclass(t, (torch.Tensor, ScalarTensor))
              for t in types
          ):
              return NotImplemented
          return HANDLED_FUNCTIONS[func](*args, **kwargs)

The ``__torch_function__`` method takes four arguments: ``func``, a reference
to the torch API function that is being overridden, ``types``, the list of
types of Tensor-likes that implement ``__torch_function__``, ``args``, the
tuple of arguments passed to the function, and ``kwargs``, the dict of keyword
arguments passed to the function. It uses a global dispatch table named
``HANDLED_FUNCTIONS`` to store custom implementations. The keys of this
dictionary are functions in the ``torch`` namespace and the values are
implementations for ``ScalarTensor``.

.. note:: Using a global dispatch table is not a mandated part of the
          ``__torch_function__`` API, it is just a useful design pattern for
          structuring your override implementations.

This class definition isn't quite enough to make ``torch.mean`` do the right
thing when we pass it a ``ScalarTensor`` -- we also need to define an
implementation for ``torch.mean`` for ``ScalarTensor`` operands and add the
implementation to the ``HANDLED_FUNCTIONS`` dispatch table dictionary. One way
of doing this is to define a decorator::

  import functools
  def implements(torch_function):
      """Register a torch function override for ScalarTensor"""
      def decorator(func):
          functools.update_wrapper(func, torch_function)
          HANDLED_FUNCTIONS[torch_function] = func
          return func
      return decorator

which can be applied to the implementation of our override::

  @implements(torch.mean)
  def mean(input):
      return float(input._value) / input._N

With this change we can now use ``torch.mean`` with ``ScalarTensor``::

  >>> d = ScalarTensor(5, 2)
  >>> torch.mean(d)
  0.4

Of course ``torch.mean`` is an example of the simplest kind of function to
override since it only takes one operand. We can use the same machinery to
override a function that takes more than one operand, any one of which might be
a tensor or tensor-like that defines ``__torch_function__``, for example for
:func:`torch.add`::

  def ensure_tensor(data):
      if isinstance(data, ScalarTensor):
          return data.tensor()
      return torch.as_tensor(data)

  @implements(torch.add)
  def add(input, other):
     try:
         if input._N == other._N:
             return ScalarTensor(input._N, input._value + other._value)
         else:
             raise ValueError("Shape mismatch!")
     except AttributeError:
         return torch.add(ensure_tensor(input), ensure_tensor(other))

This version has a fast path for when both operands are ``ScalarTensor``
instances and also a slower path which degrades to converting the data to
tensors when either operand is not a ``ScalarTensor``. That makes the override
function correctly when either operand is a ``ScalarTensor`` or a regular
:class:`Tensor`::

  >>> s = ScalarTensor(2, 2)
  >>> torch.add(s, s)
  ScalarTensor(N=2, value=4)
  >>> t = torch.tensor([[1, 1,], [1, 1]])
  >>> torch.add(s, t)
  tensor([[3., 1.],
          [1., 3.]])

Note that our implementation of ``add`` does not take ``alpha`` or ``out`` as
keyword arguments like :func:`torch.add` does::

  >>> torch.add(s, s, alpha=2)
  TypeError: add() got an unexpected keyword argument 'alpha'

For speed and flexibility the ``__torch_function__`` dispatch mechanism does not
check that the signature of an override function matches the signature of the
function being overrided in the :mod:`torch` API. For some applications ignoring
optional arguments would be fine but to ensure full compatibility with
:class:`Tensor`, user implementations of torch API functions should take care to
exactly emulate the API of the function that is being overrided.

Functions in the :mod:`torch` API that do not have explicit overrides will
return ``NotImplemented`` from ``__torch_function__``. If all operands with
``__torch_function__`` defined on them return ``NotImplemented``, PyTorch will
raise a ``TypeError``. This means that most of the time operations that do not
have explicit overrides for a type will raise a ``TypeError`` when an instance
of such a type is passed::

  >>> torch.mul(s, 3)
  TypeError: no implementation found for 'torch.mul' on types that
  implement __torch_function__: [ScalarTensor]

In practice this means that if you would like to implement your overrides using
a ``__torch_function__`` implementation along these lines, you will need to
explicitly implement the full :mod:`torch` API or the entire subset of the API
that you care about for your use case. This may be a tall order as the full
:mod:`torch` API is quite extensive.

Another option is to not return ``NotImplemented`` for operations that are not
handled but to instead pass a :class:`Tensor` to the original :mod:`torch`
function when no override is available. For example, if we change our
implementation of ``__torch_function__`` for ``ScalarTensor`` to the one below::

  @classmethod
  def __torch_function__(cls, func, types, args=(), kwargs=None):
      if kwargs is None:
          kwargs = {}
      if func not in HANDLED_FUNCTIONS or not all(
              issubclass(t, (torch.Tensor, ScalarTensor))
              for t in types
          ):
          args = [a.tensor() if hasattr(a, 'tensor') else a for a in args]
          return func(*args, **kwargs)
      return HANDLED_FUNCTIONS[func](*args, **kwargs)

Then :func:`torch.mul` will work correctly, although the return type will always
be a :class:`Tensor` rather than a :class:`ScalarTensor`, even if both operands
are :class:`ScalarTensor` instances::

  >>> s = ScalarTensor(2, 2)
  >>> torch.mul(s, s)
  tensor([[4., 0.],
          [0., 4.]])

Also see the ``MetadataTensor`` example below for another variation on this
pattern but instead always returns a ``MetadataTensor`` to propagate metadata
through operations in the :mod:`torch` API.

The ``__torch_function__`` protocol is designed for full coverage of the API,
partial coverage may lead to undesirable results, in particular, certain
functions raising a ``TypeError``. This is especially true for subclasses,
where all three of `torch.add`, `torch.Tensor.__add__` and `torch.Tensor.add`
must be covered, even if they return exactly the same result. Failing to do
this may also lead to infinite recursion. If one requires the implementation
of a function from ``torch.Tensor`` subclasses, they must use
``super().__torch_function__`` inside their implementation.


Subclassing ``torch.Tensor``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
As of version 1.7.0, methods on ``torch.Tensor`` and functions in public
``torch.*`` namespaces applied on ``torch.Tensor`` subclasses
will return subclass instances instead of ``torch.Tensor`` instances::

  >>> class SubTensor(torch.Tensor):
  ...     pass
  >>> type(torch.add(SubTensor([0]), SubTensor([1]))).__name__
  'SubTensor'
  >>> type(torch.add(SubTensor([0]), torch.tensor([1]))).__name__
  'SubTensor'

If multiple subclasses exist, the lowest one in the hierarchy will be chosen by
default. If there is no unique way to determine such a case, then a
``TypeError`` is raised::

  >>> type(torch.add(SubTensor2([0]), SubTensor([1]))).__name__
  'SubTensor2'
  >>> type(torch.add(SubTensor2([0]), torch.tensor([1]))).__name__
  'SubTensor2'
  >>> torch.add(SubTensor([0]), OtherSubTensor([1]))
  Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
  TypeError: no implementation found for 'torch.add' on types that implement __torch_function__: [SubTensor, OtherSubTensor]

If one wishes to have a global override for all tensor methods, one can use
``__torch_function__``. Here is an example that logs all function/method
calls::

  class LoggingTensor(torch.Tensor):
      @classmethod
      def __torch_function__(cls, func, types, args=(), kwargs=None):
          # NOTE: Logging calls Tensor.__repr__, so we can't log __repr__ without infinite recursion
          if func is not torch.Tensor.__repr__:
              logging.info(f"func: {func.__name__}, args: {args!r}, kwargs: {kwargs!r}")
          if kwargs is None:
              kwargs = {}
          return super().__torch_function__(func, types, args, kwargs)

However, if one instead wishes to override a method on the Tensor subclass,
there one can do so either by directly overriding the method (by defining
it for a subclass), or by using ``__torch_function__`` and matching with
``func``.

One should be careful within ``__torch_function__`` for subclasses to always
call ``super().__torch_function__(func, ...)`` instead of ``func`` directly,
as was the case before version 1.7.0. Failing to do this may cause ``func``
to recurse back into ``__torch_function__`` and therefore cause infinite
recursion.

Extending :mod:`torch` with a :class:`Tensor` wrapper type
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Another useful case is a type that wraps a :class:`Tensor`, either as an
attribute or via subclassing. Below we implement a special case of this sort of
type, a ``MetadataTensor`` that attaches a dictionary of metadata to a
:class:`Tensor` that is propagated through :mod:`torch` operations. Since this
is a generic sort of wrapping for the full :mod:`torch` API, we do not need to
individually implement each override so we can make the ``__torch_function__``
implementation more permissive about what operations are allowed::

  class MetadataTensor(object):
      def __init__(self, data, metadata=None, **kwargs):
          self._t = torch.as_tensor(data, **kwargs)
          self._metadata = metadata

      def __repr__(self):
          return "Metadata:\n{}\n\ndata:\n{}".format(self._metadata, self._t)

      @classmethod
      def __torch_function__(cls, func, types, args=(), kwargs=None):
          if kwargs is None:
              kwargs = {}
          metadatas = tuple(a._metadata for a in args if hasattr(a, '_metadata'))
          args = [getattr(a, '_t', a) for a in args]
          assert len(metadatas) > 0
          ret = func(*args, **kwargs)
          return MetadataTensor(ret, metadata=metadatas[0])

This simple implementation won't necessarily work with every function in the
:mod:`torch` API but it is good enough to capture most common operations::

  >>> metadata = {'owner': 'Ministry of Silly Walks'}
  >>> m = MetadataTensor([[1, 2], [3, 4]], metadata=metadata)
  >>> t = torch.tensor([[1, 2], [1, 2]])
  >>> torch.add(t, m)
  Metadata:
  {'owner': 'Ministry of Silly Walks'}

  data:
  tensor([[2, 4],
          [4, 6]])
  >>> torch.mul(t, m)
  Metadata:
  {'owner': 'Ministry of Silly Walks'}

  data:
  tensor([[1, 4],
          [3, 8]])

Operations on multiple types that define ``__torch_function__``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is possible to use the torch API with multiple distinct types that each have
a ``__torch_function__`` implementation, but special care must be taken. In such
a case the rules are:

* The dispatch operation gathers all distinct implementations of
  ``__torch_function__`` for each operand and calls them in order: subclasses
  before superclasses, and otherwise left to right in the operator expression.
* If any value other than ``NotImplemented`` is returned, that value is
  returned as the result. Implementations can register that they do not
  implement an operation by returning ``NotImplemented``.
* If all of the ``__torch_function__`` implementations return
  ``NotImplemented``, PyTorch raises a ``TypeError``.

Testing Coverage of Overrides for the PyTorch API
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One troublesome aspect of implementing ``__torch_function__`` is that if some
operations do and others do not have overrides, users will at best see an
inconsistent experience, or at worst will see errors raised at runtime when they
use a function that does not have an override. To ease this process, PyTorch
provides a developer-facing API for ensuring full support for
``__torch_function__`` overrides. This API is private and may be subject to
changes without warning in the future.

First, to get a listing of all overridable functions, use
``torch.overrides._get_overridable_functions``. This returns a dictionary whose
keys are namespaces in the ``PyTorch`` Python API and whose values are a list of
functions in that namespace that can be overridden. For example, let's print the
names of the first 5 functions in ``torch.nn.functional`` that can be
overridden::

  >>> from torch.overrides import get_overridable_functions
  >>> func_dict = get_overridable_functions()
  >>> nn_funcs = func_dict[torch.nn.functional]
  >>> print([f.__name__ for f in nn_funcs[:5])
  ['adaptive_avg_pool1d', 'adaptive_avg_pool2d', 'adaptive_avg_pool3d',
   'adaptive_max_pool1d', 'adaptive_max_pool1d_with_indices']

This listing of functions makes it possible to iterate over all overridable
functions, however in practice this is not enough to write tests for all of
these functions without laboriously and manually copying the signature of each
function for each test. To ease this process, the
``torch.overrides._get_testing_overrides`` function returns a dictionary mapping
overridable functions in the ``PyTorch`` API to dummy lambda functions that have
the same signature as the original function but unconditionally return -1. These
functions are most useful to use with ``inspect`` to analyze the function
signature of the original ``PyTorch`` function::

  >>> import inspect
  >>> from torch.overrides import get_testing_overrides
  >>> override_dict = get_testing_overrides()
  >>> dummy_add = override_dict[torch.add]
  >>> inspect.signature(dummy_add)
  <Signature (input, other, out=None)>

Finally, ``torch.overrides.get_ignored_functions`` returns a tuple of functions
that explicitly cannot be overrided by ``__torch_function__``. This list can be
useful to confirm that a function that isn't present in the dictionary returned
by ``get_overridable_functions`` cannot be overridden.


.. _extending-torch-c++:

Extending :mod:`torch` native API
---------------------------------

While ``__torch_function__`` allows one to effectively extend PyTorch's pure Python
components' behavior, it does not allow one to extend the parts of
PyTorch implemented in C++. To that end, a :class:`Tensor` subclass can also
define ``__torch_dispatch__`` which will be able to override the behavior at the
C++ level.

To effectively use this feature, it is important to know how the native part of
PyTorch is implemented. The most important component there is what we call the
"dispatcher" (the best description can be found in this `blog post <http://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/>`_ even though it is slightly outdated). As
hinted by its name, it is responsible for calling the right backend
function for a specific call of a function. For example, when calling
``torch.add(a, b)``, the dispatcher will inspect both arguments, figure out which
"feature" (autograd, autocast, functionalization, etc) and which "backend" (CPU,
CUDA, MPS, etc) should be used for this specific call and finally call all the
right kernels.
A very common thing done by a kernel is to "redispatch". For example, when running your
neural network on GPU with autocast, the first call will be the autocast kernel that
will handle any potential autocast logic and redispatch down. The next feature in line
will be autograd that will properly create the autograd graph and then redispatch down.
Finally, we reach the backend kernel for CUDA which will launch the right CUDA kernel
and return the final result. On the way out, autograd will attach the graph to the
output and, finally, autocast will have a chance to do any update it needs on exit.

One configuration of the dispatcher is the order in which all these feature and backend keys are called. The latest list and their order can be found in ``DispatchKey.h`` inside the ``DispatchKey`` enum. For the purpose of extending torch, the important subset of the ordering for this discussion is:

vmap -> Autocast -> Autograd -> ZeroTensor -> Neg/Conj -> Functionalize -> Python -> Backends

The most important key for the purpose of this discussion is ``Python`` as every Tensor subclass with the ``__torch_dispatch__`` method defined will call into this feature. It is from there that the user-defined method is called and where the behavior can be overwritten arbitrarily. From there, calling the provided ``func`` again will perform a "redispatch".

Some important implications of this implementation are:

- This code runs "below all features". It is thus only responsible, like a regular backend, for generating the output value of each Tensor (and can, and should, ignore all advanced features like autograd, autocast, etc).
- If any high level feature implements a given function without redispatching, it will never reach the ``Python`` key and so the ``__torch_dispatch__`` callback will never be triggered. This happens in particular for CompositeImplicitAutograd functions which are evaluated at the Autograd level without redispatching. This is because a CompositeImplicitAutograd function specifies its autograd formula by implicitly calling other native ops, so at the Autograd level, the function is decomposed into its native ops and those are evaluated instead.
- When calling back to Python and when wrapping the results, the same conversions are used as the regular PyTorch Python/C++ binding. In particular, some objects cannot be represented in Python and need special handling (undefined Tensors for example become None).
- Our native functions are lazily populated as ``torch.ops.{namespace}.{func_name}.{overload_name}`` as callable Python objects to enable easily interacting with them from Python. The ``func`` object given to ``__torch_dispatch__`` is always an entry from this namespace. This namespace can be used to directly call native ops and bypass the usual Python API and binding code.

In a similar way where ``__torch_function__`` is able to interpose on all of torch's Python API and Tensor methods, ``__torch_dispatch__`` is able intercepting all calls into the aten native API. Note that all methods on Tensors are converted into function calls before entering the dispatcher and thus will appear as function calls here: ``torch.add(a, 2)`` and ``a + 2`` will lead to exactly the same aten call.
Most of these functions are defined in ``native_functions.yaml`` which specifies the properties of these functions as well as their backend implementation. Their implementation alongside specified features are then automatically registered via codegen.
Some more exotic functions or features are also registered in other places in the C++ codebase or in user-defined C++ extensions.

It is also possible to add `new` native functions using :mod:`torch.library`. This Python feature allows defining and/or adding new implementations to native functions. This can be used to add missing kernels, replace existing ones or define brand new native functions.

You can find many examples of ``__torch_dispatch__``-based subclasses in the `subclass zoo <https://github.com/albanD/subclass_zoo>`_ repo.

Extending all :mod:`torch` API with Modes
-----------------------------------------

Unfortunately, there are functions that do not take Tensor inputs. This means that the subclass approach described above cannot be used to override the behavior of all of PyTorch's functions. Also, if the use case requires to intercept every function call, changing every Tensor to be a subclass can be overly intrusive.

To address this use case, we introduced the concept of "Mode". These exist for ``__torch_function__`` and ``__torch_dispatch__`` overrides, are created by subclassing respectively :class:`torch.overrides.TorchFunctionMode` and :class:`torch.utils._python_dispatch.TorchDispatchMode`, and are used as a context manager.

To simplify the description of how it interacts with subclasses and other modes, whenever the context manager for a mode is entered, every function behaves as if there was an extra Tensor argument at the beginning of the argument list with the mode as a subclass.
This means in particular that all modes handlers will be called before any subclass handler and that modes corresponding to the inner context manager will always run first.

It is also important to note that within a given mode handler, this specific mode is disabled and can be re-enabled manually by doing ``with self:``.

Here is an example that shows logging modes of each type::

  import torch
  from torch.overrides import TorchFunctionMode, resolve_name
  from torch.utils._python_dispatch import TorchDispatchMode

  class FunctionLog(TorchFunctionMode):
      def __torch_function__(self, func, types, args, kwargs=None):
          print(f"Function Log: {resolve_name(func)}(*{args}, **{kwargs})")
          return func(*args, **(kwargs or {}))

  class DispatchLog(TorchDispatchMode):
      def __torch_dispatch__(self, func, types, args, kwargs=None):
          print(f"Dispatch Log: {func}(*{args}, **{kwargs})")
          return func(*args, **(kwargs or {}))

  def f():
      a = torch.rand(10, requires_grad=True)
      b = a * 2
      b.sum().backward()

  print("TorchFunctionMode logging:")
  with FunctionLog():
      f()

  print("TorchDispatchMode logging:")
  with DispatchLog():
      f()

Which prints the following, with extra comments::

  TorchFunctionMode logging:
  Function Log: torch.rand(*(10,), **{'requires_grad': True})
  Function Log: torch.Tensor.mul(*(tensor([0.7164, 0.9897, 0.1745, 0.9336, 0.4287, 0.7989, 0.2169, 0.7474, 0.5624,
          0.5970], requires_grad=True), 2), **None)
  Function Log: torch.Tensor.sum(*(tensor([1.4328, 1.9794, 0.3490, 1.8671, 0.8573, 1.5977, 0.4338, 1.4948, 1.1249,
          1.1939], grad_fn=<MulBackward0>),), **None)
  # Note that at the python level, we only see the call to backward but not what happens in the autograd engine.
  Function Log: torch.Tensor.backward(*(tensor(12.3307, grad_fn=<SumBackward0>),), **{'gradient': None, 'retain_graph': None, 'create_graph': False, 'inputs': None})

  TorchDispatchMode logging:
  # Here the requires_grad flag from autograd is removed while default arguments were populated.
  Dispatch Log: aten.rand.default(*([10],), **{'device': device(type='cpu'), 'pin_memory': False})
  Dispatch Log: aten.mul.Tensor(*(tensor([0.2151, 0.6018, 0.8415, 0.9060, 0.2974, 0.7708, 0.6668, 0.0352, 0.7948,
          0.6023], requires_grad=True), 2), **{})
  Dispatch Log: aten.sum.default(*(tensor([0.4303, 1.2036, 1.6831, 1.8120, 0.5949, 1.5416, 1.3335, 0.0705, 1.5897,
          1.2046], grad_fn=<MulBackward0>),), **{})
  # Here we don't see the call to backward itself, but its constituents. Starting here with the factory function that creates the initial gradient.
  Dispatch Log: aten.ones_like.default(*(tensor(11.4637, grad_fn=<SumBackward0>),), **{'pin_memory': False, 'memory_format': torch.preserve_format})
  # This is the backward of the sum
  Dispatch Log: aten.expand.default(*(tensor(1.), [10]), **{})
  Dispatch Log: aten.mul.Tensor(*(tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]), 2), **{})
  Dispatch Log: aten.detach.default(*(tensor([2., 2., 2., 2., 2., 2., 2., 2., 2., 2.]),), **{})
  Dispatch Log: aten.detach.default(*(tensor([2., 2., 2., 2., 2., 2., 2., 2., 2., 2.]),), **{})
