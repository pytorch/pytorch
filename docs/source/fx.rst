.. currentmodule:: torch.fx

torch.fx
=============

Overview
--------
.. automodule:: torch.fx

Limitations of Symbolic Tracing
-------------------------------

FX uses a system of **symbolic tracing** \*\* (a.k.a `abstract
interpretation <https://en.wikipedia.org/wiki/Abstract_interpretation>`__)
to capture the semantics of programs in a transformable/analyzable form.
The system is **tracing** in that it executes the program (really an
``nn.Module`` or function) to gather this information. It is
**symbolic** in that the data flowing through the program during this
execution is not real data, but rather symbols (“Proxy” in FX parlance).

Although symbolic tracing works for most neural net code, it has some
limitations.

Dynamic Control Flow
^^^^^^^^^^^^^^^^^^^^

The main limitation of symbolic tracing is it does not currently support
*dynamic control flow*. That is, loops or ``if`` statements where the
condition may depend on the input values of the program.

For example, let’s examine the following program:

::

    def func_to_trace(x):
        dim0 = x.size[0]
        if dim0 == 3:
            return torch.relu(x)
        else:
            return torch.neg(x)

    traced = torch.fx.symbolic_trace(func_to_trace)
    """
      <...>
      File "dyn.py", line 6, in func_to_trace
        if dim0 == 3:
      File "pytorch/torch/fx/proxy.py", line 155, in __bool__
        return self.tracer.to_bool(self)
      File "pytorch/torch/fx/proxy.py", line 85, in to_bool
        raise TraceError('symbolically traced variables cannot be used as inputs to control flow')
    torch.fx.proxy.TraceError: symbolically traced variables cannot be used as inputs to control flow
    """

The condition to the ``if`` statement relies on the value of ``dim0``,
which eventually relies on the value of ``x``, a function input. Since
``x`` can change (i.e. if you pass a new input tensor to the traced
function), this is *dynamic control flow*. The traceback walks back up
through your code to show you where this situation happens.

Static Control Flow
~~~~~~~~~~~~~~~~~~~

On the other hand, so-called *static control flow* is supported. Static
control flow is loops or ``if`` statements whose value cannot change
across invocations. Typically, in PyTorch programs, this control flow
arises for code making decisions about a model’s architecture based on
hyper-parameters. As a concrete example:

::

    import torch
    import torch.fx

    class MyModule(torch.nn.Module):
        def __init__(self, do_activation : bool = False):
            super().__init__()
            self.do_activation = do_activation
            self.linear = torch.nn.Linear(512, 512)

        def forward(self, x):
            x = self.linear(x)
            # This if-statement is so-called static control flow.
            # Its condition does not depend on any input values
            if self.do_activation:
                x = torch.relu(x)
            return x

    without_activation = MyModule(do_activation=False)
    with_activation = MyModule(do_activation=True)

    traced_without_activation = torch.fx.symbolic_trace(without_activation)
    print(traced_without_activation.code)
    """
    def forward(self, x):
        linear_1 = self.linear(x);  x = None
        return linear_1
    """

    traced_with_activation = torch.fx.symbolic_trace(with_activation)
    print(traced_with_activation.code)
    """
    import torch
    def forward(self, x):
        linear_1 = self.linear(x);  x = None
        relu_1 = torch.relu(linear_1);  linear_1 = None
        return relu_1
    """

The if-statement ``if self.do_activation`` does not depend on any
function inputs, thus it is static. ``do_activation`` can be considered
to be a hyper-parameter, and the traces of different instances of
``MyModule`` with different values for that parameter have different
code. This is a valid pattern that is supported by symbolic tracing.

Many instances of dynamic control flow are semantically static control
flow. These instances can be made to support symbolic tracing by
removing the data dependencies on input values, for example by moving
values to ``Module`` attributes or by passing constant values during
symbolic tracing:

::

        def f(x, flag):
            if flag: return x
            else: return x*2

        fx.symbolic_trace(f) # Fails!

        def g(flag):
            return lambda x: f(x, flag)

        new_f = g(flag=True)
        fx.symbolic_trace(new_f)

In the case of truly dynamic control flow, the sections of the program
that contain this code can be traced as calls to the Method (see
Customizing Tracing with the ``Tracer`` class) or function (see
wrap\ **TODO: link**) rather than tracing through them.

Non-\ ``torch`` Functions
^^^^^^^^^^^^^^^^^^^^^^^^^

FX uses ``__torch_function__`` as the mechanism by which it intercepts
calls (see the `technical
overview <https://github.com/pytorch/pytorch/blob/master/torch/fx/OVERVIEW.md#technical-details>`__
for more information about this). Some functions, such as builtin Python
functions or those in the ``math`` module, are things that are not
covered by ``__torch_function__``, but we would still like to capture
them in symbolic tracing. For example:

::

    from math import sqrt

    def normalize(x):
        """
        Normalize `x` by the size of the batch dimension
        """
        return x / sqrt(len(x))

    # It's valid Python code
    normalize(torch.rand(3, 4))

    traced = torch.fx.symbolic_trace(normalize)
    """
      <...>
      File "sqrt.py", line 9, in normalize
        return x / sqrt(len(x))
      File "pytorch/torch/fx/proxy.py", line 161, in __len__
        raise RuntimeError("'len' is not supported in symbolic tracing by default. If you want "
    RuntimeError: 'len' is not supported in symbolic tracing by default. If you want this call to be recorded, please call torch.fx.wrap('len') at module scope
    """

The error tells us that the built-in function ``len`` is not supported.
We can make it so that functions like this are recorded in the trace as
direct calls using the :func:`wrap` API:

::

    torch.fx.wrap('len')
    torch.fx.wrap('sqrt')

    traced = torch.fx.symbolic_trace(normalize)

    print(traced.code)
    """
    import math
    def forward(self, x):
        len_1 = len(x)
        sqrt_1 = math.sqrt(len_1);  len_1 = None
        truediv = x / sqrt_1;  x = sqrt_1 = None
        return truediv
    """

Customizing Tracing with the ``Tracer`` class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`Tracer` class is the class that underlies the
implementation of ``symbolic_trace``. The behavior of tracing can be
customized by subclassing Tracer, like so:

::

    class MyCustomTracer(torch.fx.Tracer):
        # Inside here you can override various methods
        # to customize tracing. See the `Tracer` API
        # reference
        pass


    # Let's use this custom tracer to trace through this module
    class MyModule(torch.nn.Module):
        def forward(self, x):
            return torch.relu(x) + torch.ones(3, 4)

    mod = MyModule()

    traced_graph = MyCustomTracer().trace(mod)
    # trace() returns a Graph. Let's wrap it up in a
    # GraphModule to make it runnable
    traced = torch.fx.GraphModule(mod, traced_graph)

Leaf Modules
~~~~~~~~~~~~

Leaf Modules are the modules that appear as calls in the symbolic trace
rather than being traced through. The default set of leaf modules is the
set of standard ``torch.nn`` module instances. For example:

::

    class MySpecialSubmodule(torch.nn.Module):
        def forward(self, x):
            return torch.neg(x)

    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 4)
            self.submod = MySpecialSubmodule()

        def forward(self, x):
            return self.submod(self.linear(x))

    traced = torch.fx.symbolic_trace(MyModule())
    print(traced.code)
    # `linear` is preserved as a call, yet `submod` is traced though.
    # This is because the default set of "Leaf Modules" includes all
    # standard `torch.nn` modules.
    """
    import torch
    def forward(self, x):
        linear_1 = self.linear(x);  x = None
        neg_1 = torch.neg(linear_1);  linear_1 = None
        return neg_1
    """

The set of leaf modules can be customized by overriding
:meth:`Tracer.is_leaf_module`.

Miscellanea
^^^^^^^^^^^

-  Tensor constructors (e.g. ``torch.zeros``, ``torch.ones``,
   ``torch.rand``, ``torch.randn``) are currently not traceable.

   -  The deterministic constructors (``zeros``, ``ones``) can be used
      and the value they produce will be embedded in the trace as a
      constant. This is only problematic if the arguments to these
      constructors refers to dynamic input sizes. In this case,
      ``ones_like`` or ``zeros_like`` may be a viable substitute.
   -  Nondeterministic constructors (``rand``, ``randn``) will have a
      single random value embedded in the trace. This is likely not the
      intended behavior.
   -  This behavior may be fixed in a future release.

-  Type annotations

   -  Python 3-style type annotations (e.g.
      ``func(x : torch.Tensor, y : int) -> torch.Tensor``) are supported
      and will be preserved by symbolic tracing.
   -  Python 2-style comment type annotations
      ``# type: (torch.Tensor, int) -> torch.Tensor`` are not currently
      supported.
   -  Annotations on local names within a function are not currently
      supported.

Writing Transformations
-----------------------

TODO

Debugging Transformations
-------------------------

TODO

API Reference
-------------

.. autofunction:: torch.fx.symbolic_trace

.. autofunction:: torch.fx.wrap

.. autoclass:: torch.fx.GraphModule
  :members:

  .. automethod:: __init__

.. autoclass:: torch.fx.Graph
  :members:

  .. automethod:: __init__

.. autoclass:: torch.fx.Node
  :members:

.. autoclass:: torch.fx.Tracer
  :members:

.. autoclass:: torch.fx.Proxy
