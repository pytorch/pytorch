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
encode the operation history. Every new function requires you to implement 3
methods:

- ``__init__`` (*optional*) - if your operation is parametrized by/uses
  objects different than :class:`Variable` s, you should pass them as arguments
  to ``__init__``. For example, ``AddConstant`` function takes a scalar to add,
  while ``Transpose`` requires specifying which two dimensions to swap. If your
  function doesn't require any additional parameters, you can skip it.
- :meth:`~Function.forward` - the code that performs the operation. It can take
  as many arguments as you want, with some of them being
  optional, if you specify the default values. Keep in mind that only
  :class:`Variable` s will be passed in here. You can return either a single
  :class:`Variable` output, or a :class:`tuple` of :class:`Variable` s if there
  are multiple. Also, please refer to the docs of :class:`Function` to find
  descriptions of useful methods that can be called only from
  :meth:`~Function.forward`.
- :meth:`~Function.backward` - gradient formula. It will be given
  as many arguments as there were outputs, with each of them representing
  gradient w.r.t. that output. It should return as many :class:`Tensor` s as
  there were inputs, with each of them containing the gradient w.r.t.
  corresponding input. If you inputs didn't require gradient (see
  :attr:`~Variable.needs_input_grad`), or it was non-differentiable, you
  can return :class:`None`. Also, if you have optional arguments to
  :meth:`~Variable.forward` you can return more gradients than there were
  inputs, as long as they're all :any:`python:None`.

Below you can find code for a ``Linear`` function from :mod:`torch.nn`, with
additional comments::

    # Inherit from Function
    class Linear(Function):

        # bias is an optional argument
        def forward(self, input, weight, bias=None):
            self.save_for_backward(input, weight, bias)
            output = input.mm(weight.t())
            if bias is not None:
                output += bias.unsqueeze(0).expand_as(output)
            return output

        # This function has only a single output, so it gets only one gradient
        def backward(self, grad_output):
            # This is a pattern that is very convenient - at the top of backward
            # unpack saved_tensors and initialize all gradients w.r.t. inputs to
            # None. Thanks to the fact that additional trailing Nones are
            # ignored, the return statement is simple even when the function has
            # optional inputs.
            input, weight, bias = self.saved_tensors
            grad_input = grad_weight = grad_bias = None

            # These needs_input_grad checks are optional and there only to
            # improve efficiency. If you want to make your code simpler, you can
            # skip them. Returning gradients for inputs that don't require it is
            # not an error.
            if self.needs_input_grad[0]:
                grad_input = grad_output.mm(weight)
            if self.needs_input_grad[1]:
                grad_weight = grad_output.t().mm(input)
            if bias is not None and self.needs_input_grad[2]:
                grad_bias = grad_output.sum(0).squeeze(0)

            return grad_input, grad_weight, grad_bias

Now, to make it easier to use these custom ops, we recommend wrapping them in
small helper functions::

    def linear(input, weight, bias=None):
        # First braces create a Function object. Any arguments given here
        # will be passed to __init__. Second braces will invoke the __call__
        # operator, that will then use forward() to compute the result and
        # return it.
        return Linear()(input, weight, bias)

You probably want to check if the backward method you implemented actually
computes the derivatives of your function. It is possible by comparing with
numerical approximations using small finite differences::

    from torch.autograd import gradcheck
   
    # gradchek takes a tuple of tensor as input, check if your gradient
    # evaluated with these tensors are close enough to numerical
    # approximations and returns True if they all verify this condition.
    input = (Variable(torch.randn(20,20).double(), requires_grad=True), Variable(torch.randn(30,20).double(), requires_grad=True),)
    test = gradcheck(Linear(), input, eps=1e-6, atol=1e-4)
    print(test)

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
implementated as in the listing above. There's very little code required to
add this. Now, there are two functions that need to be implemented:

- ``__init__`` (*optional*) - takes in arguments such as kernel sizes, numbers
  of features, etc. and initializes parameters and buffers.
- :meth:`~Module.forward` - instantiates a :class:`~torch.autograd.Function` and
  uses it to perform the operation. It's very similar to a functional wrapper
  shown above.

This is how a ``Linear`` module can be implemented::

    class Linear(nn.Module):
        def __init__(self, input_features, output_features, bias=True):
            self.input_features = input_features
            self.output_features = output_features

            # nn.Parameter is a special kind of Variable, that will get
            # automatically registered as Module's parameter once it's assigned
            # as an attribute. Parameters and buffers need to be registered, or
            # they won't appear in .parameters() (doesn't apply to buffers), and
            # won't be converted when e.g. .cuda() is called. You can use
            # .register_buffer() to register buffers.
            # nn.Parameters can never be volatile and, different than Variables,
            # they require gradients by default.
            self.weight = nn.Parameter(torch.Tensor(input_features, output_features))
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
            return Linear()(input, self.weight, self.bias)


Writing custom C extensions
---------------------------

Coming soon. For now you can find an example at
`GitHub <https://github.com/pytorch/extension-ffi>`_.
