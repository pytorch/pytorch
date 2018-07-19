Extending PyTorch
=================

In this note we'll cover ways of extending :mod:`torch.nn`,
:mod:`torch.autograd`, and writing custom C extensions utilizing our C
libraries.

Extending :mod:`torch.autograd`
-------------------------------

.. currentmodule:: torch.autograd

Adding operations to :mod:`~torch.autograd` requires implementing a new
:class:`Function` subclass for each operation. Recall that :class:`Function` s
are what :mod:`~torch.autograd` uses to compute the results and gradients, and
encode the operation history. Every new function requires you to implement 2
methods:

- :meth:`~Function.forward` - the code that performs the operation. It can take
  as many arguments as you want, with some of them being optional, if you
  specify the default values. All kinds of Python objects are accepted here.
  :class:`Tensor` arguments that track history (i.e., with
  ``requires_grad=True``) will be converted to ones that don't track history
  before the call, and their use will be registered in the graph. Note that this
  logic won't traverse lists/dicts/any other data structures and will only
  consider :class:`Tensor` s that are direct arguments to the call. You can
  return either a single :class:`Tensor` output, or a :class:`tuple` of
  :class:`Tensor` s if there are multiple outputs. Also, please refer to the
  docs of :class:`Function` to find descriptions of useful methods that can be
  called only from :meth:`~Function.forward`.
- :meth:`~Function.backward` - gradient formula. It will be given
  as many :class:`Tensor` arguments as there were outputs, with each of them
  representing gradient w.r.t. that output. It should return as many
  :class:`Tensor` s as there were inputs, with each of them containing the
  gradient w.r.t. its corresponding input. If your inputs didn't require
  gradient (:attr:`~ctx.needs_input_grad` is a tuple of booleans indicating
  whether each input needs gradient computation), or were non-:class:`Tensor`
  objects, you can return :class:`python:None`. Also, if you have optional
  arguments to :meth:`~Function.forward` you can return more gradients than there
  were inputs, as long as they're all :any:`python:None`.

Below you can find code for a ``Linear`` function from :mod:`torch.nn`, with
additional comments::

    # Inherit from Function
    class LinearFunction(Function):

        # Note that both forward and backward are @staticmethods
        @staticmethod
        # bias is an optional argument
        def forward(ctx, input, weight, bias=None):
            ctx.save_for_backward(input, weight, bias)
            output = input.mm(weight.t())
            if bias is not None:
                output += bias.unsqueeze(0).expand_as(output)
            return output

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
                grad_bias = grad_output.sum(0).squeeze(0)

            return grad_input, grad_weight, grad_bias

Now, to make it easier to use these custom ops, we recommend aliasing their
``apply`` method::

    linear = LinearFunction.apply

Here, we give an additional example of a function that is parametrized by
non-Tensor arguments::

    class MulConstant(Function):
        @staticmethod
        def forward(ctx, tensor, constant):
            # ctx is a context object that can be used to stash information
            # for backward computation
            ctx.constant = constant
            return tensor * constant

        @staticmethod
        def backward(ctx, grad_output):
            # We return as many input gradients as there were arguments.
            # Gradients of non-Tensor arguments to forward must be None.
            return grad_output * ctx.constant, None

.. note::
    Inputs to ``backward``, i.e., :attr:`grad_output`, can also be Tensors that
    track history. So if ``backward`` is implemented with differentiable
    operations, (e.g., invocation of another custom
    :class:`~torch.autograd.function`), higher order derivatives will work.

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
            super(Linear, self).__init__()
            self.input_features = input_features
            self.output_features = output_features

            # nn.Parameter is a special kind of Tensor, that will get
            # automatically registered as Module's parameter once it's assigned
            # as an attribute. Parameters and buffers need to be registered, or
            # they won't appear in .parameters() (doesn't apply to buffers), and
            # won't be converted when e.g. .cuda() is called. You can use
            # .register_buffer() to register buffers.
            # nn.Parameters require gradients by default.
            self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
            if bias:
                self.bias = nn.Parameter(torch.Tensor(output_features))
            else:
                # You should always register all possible parameters, but the
                # optional ones can be None if you want.
                self.register_parameter('bias', None)

            # Not a very smart way to initialize weights
            self.weight.data.uniform_(-0.1, 0.1)
            if bias is not None:
                self.bias.data.uniform_(-0.1, 0.1)

        def forward(self, input):
            # See the autograd section for explanation of what happens here.
            return LinearFunction.apply(input, self.weight, self.bias)

        def extra_repr(self):
            # (Optional)Set the extra information about this module. You can test
            # it by printing an object of this class.
            return 'in_features={}, out_features={}, bias={}'.format(
                self.in_features, self.out_features, self.bias is not None
            )


Writing custom C++ extensions
-----------------------------

See this
`PyTorch tutorial <https://pytorch.org/tutorials/advanced/cpp_extension.html>`_
for a detailed explanation and examples.

Documentations are available at :doc:`../cpp_extension`.


Writing custom C extensions
---------------------------

Example available at
`this GitHub repository <https://github.com/pytorch/extension-ffi>`_.
