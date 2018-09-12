Torch Script
============

.. contents:: :local:

.. automodule:: torch.jit
.. currentmodule:: torch.jit

Torch Script is a way to create serializable and optimizable models from PyTorch code.
Anything code written in Torch Script can be saved from your Python
process and loaded/run a process where there is no python dependency.

We provide tools to incrementally transition a model from being a pure Python program
to a Torch Script program that can be run independently from python, for instance, in a standalone C++ process.
This makes it possible to train models in PyTorch using familiar tools and then export
the model to a production environment where it is not a good idea to run models as python programs
for performance and multi-threading reasons.

Creating Torch Script Code
--------------------------


.. autoclass:: ScriptModule
    :members:

    .. method:: save(filename)

       Save an offline version of this module for use in a separate process. The saved
       module serializes all of the methods and parameters of this module. It can be
       loaded into the C++ API using ``torch::jit::load(filename)`` or into the Python
       API with ``torch.jit.load(filename)``.

       To be able to save a module, it must not make any calls to native python functions.
       This means that all submodules must be subclasses of ScriptModules as well.

       .. DANGER::
          All modules, no matter their device, are always loaded onto the CPU during loading.
          This is different from :func:`torch.load`'s semantics and may change in the future.


.. autofunction:: load

.. autofunction:: trace


Mixing Tracing and Scripting
----------------------------

In many cases either tracing or script is an easier approach for converting a model.
We allow you to compose tracing and scripting to suite the particular requirements
of a part of a model.

Scripted functions can call traced ones. This is particularly useful when you need
to use control-flow around a simple feed-forward model. For instance the beam search
of a sequence to sequence model will typically be written in script but can call an
encoder module generated using tracing.

Example:

::

    import torch

    def foo(x, y):
        return 2 * x + y
    traced_foo = torch.jit.trace(foo, (torch.rand(3), torch.rand(3)))

    @torch.jit.script
    def bar(x):
        return traced_foo(x, x)

Traced functions can call script functions. This is useful when a small part of
a model requires some control-flow even though most of the model is just a feed-forward
network. Control-flow inside of a script function called by a traced function is
preserved correctly:

Example:

::

    import torch
    @torch.jit.script
    def foo(x, y):
        if x.max() > y.max():
            r = x
        else:
            r = y
        return r


    def bar(x, y, z):
        return foo(x, y) + z

    traced_bar = torch.jit.trace(bar, (torch.rand(3), torch.rand(3), torch.rand(3))

This composition also works for modules as well, where it can be used to generate
a submodule using tracing that can be called from the methods of a script module:

Example:

::

    import torch
    import torchvision

    class MyScriptModule(torch.jit.ScriptModule):
        def __init__(self):
            super(MyScriptModule, self).__init__()
            self.means = torch.nn.Parameter(torch.tensor([103.939, 116.779, 123.68])
                                            .resize_(1, 3, 1, 1))
            self.resnet = torch.jit.trace(torchvision.models.resnet18(),
                                          torch.rand(1, 3, 224, 224))

        @torch.jit.script_method
        def forward(self, input):
            return self.resnet(input - self.means)


Torch Script Language Reference
-------------------------------

Torch Script is a subset of Python that can either be written directly (using
the @script annotations) or generated automatically from Python code via
tracing. When using tracing, code is automatically converted into this subset of
Python by recording only the actual operators on tensors and simply executing and
discarding the other surrounding Python code.

When writing Torch Script directly using @script annotations, the programmer must
only use the subset of Python supported in Torch Script. This section documents
what is supported in Torch Script as if it were a language reference for a stand
alone language. Any features of Python not mentioned in this reference are not
part of Torch Script.

As a subset of Python any valid Torch Script function is also a valid Python
function. This makes it possible to remove the @script annotations and debug the
function using standard Python tools like pdb. The reverse is not true: there
are many valid python programs that are not valid Torch Script programs.
Instead, Torch Script focuses specifically on the features of Python that are
needed to represent neural network models in Torch.

.. envvar:: PYTORCH_JIT=1

    Setting the environment variable ``PYTORCH_JIT=0`` will disable all script
    and tracing annotations. If there is hard-to-debug error in one of your
    ScriptModules, you can use this flag to force everything to run using native
    Python. This allows the use of tools like ``pdb`` to debug code.


Types
~~~~~

The largest difference between Torch Script and the full Python language is that
Torch Script only support a small set of types that are needed to express neural
net models. In particular Torch Script supports:

``Tensor``
    A PyTorch tensor of any dtype, dimension, or backend.

``Tuple[T0, T1, ...]``
    A tuple containing subtypes ``T0``, ``T1``, etc. (e.g. ``Tuple[Tensor, Tensor]``)

``int``
    A scalar integer

``float``
    A scalar floating point number

``List[T]``
    A list of which all members are type ``T``

Unlike Python, each variable in Torch Script function must have a single static type.
This makes it easier to optimize Torch Script functions.

Example::

    @torch.jit.script
    def an_error(x):
        if x:
            r = torch.rand(1)
        else:
            r = 4
        return r # Type mismatch: r is set to type Tensor in the true branch
                 # and type int in the false branch

By default, all parameters to a Torch Script function are assumed to be Tensor
because this is the most common type used in modules. To specify that an
argument to a Torch Script function is another type, it is possible to use
MyPy-style type annotations using the types listed above:

Example::

    @torch.jit.script
    def foo(x, tup):
        # type: (int, Tuple[Tensor, Tensor]) -> Tensor
        t0, t1 = tup
        return t0 + t1 + x

    print(foo(3, (torch.rand(3), torch.rand(3))))

.. note::
  It is also possible to annotate types with Python 3 type annotations.
  In our examples, we use comment-based annotations to ensure Python 2
  compatibility as well.

Expressions
~~~~~~~~~~~

The following Python Expressions are supported

Literals
    ``True``, ``False``, ``None``, ``'string literals'``, ``"string literals"``,
    number literals ``3`` (interpreted as int) ``3.4`` (interpreter as a float)

Variables
  ``a``

  .. note::
      See `Variable Resolution`_ for how variables are resolved.

Tuple Construction
    ``(3, 4)``, ``(3,)``

List Construction
    ``[3, 4]``, ``[]``, ``[torch.rand(3), torch.rand(4)]``

    .. note::
        an empty list is assumed have type ``List[Tensor]``.
        The types of other list literals are derived from the type of the members.

Arithmetic Operators
  ``a + b``
  ``a - b``
  ``a * b``
  ``a / b``
  ``a ^ b``
  ``a @ b``

Comparison Operators
  ``a == b``
  ``a != b``
  ``a < b``
  ``a > b``
  ``a <= b``
  ``a >= b``

Logical Operators
  ``a and b``
  ``a or b``
  ``not b``

Subscripts
  ``t[0]``
  ``t[-1]``
  ``t[0:2]``
  ``t[1:]``
  ``t[:1]``
  ``t[:]``
  ``t[0, 1]``
  ``t[0, 1:2]``
  ``t[0, :1]``
  ``t[-1, 1:, 0]``
  ``t[1:, -1, 0]``
  ``t[i:j, i]``

  .. note::
    Torch Script currently does not support mutating tensors in place, so any
    tensor indexing can only appear on the right-hand size of an expression.

Function calls
   Calls to built-in functions: ``torch.rand(3, dtype=torch.int)``

   Calls to other script functions:

   ::

        import torch

        @torch.jit.script
        def foo(x):
          return x + 1

        @torch.jit.script
        def bar(x):
          return foo(x)

Method calls
    Calls to methods of builtin types like tensor: ``x.mm(y)``


    When defining a Script method inside of a ScriptModule, the ``@script_method``
    annotation is used. Inside of these methods it is possible to call other methods
    of this class or access methods on the submodules.

    Calling a submodule directly (e.g. ``self.resnet(input)``) is equivalent to
    calling its ``forward`` method (e.g. ``self.resnet.forward(input)``)

    ::

        import torch

        class MyScriptModule(torch.jit.ScriptModule):
            def __init__(self):
                super(MyScriptModule, self).__init__()
                self.means = torch.nn.Parameter(torch.tensor([103.939, 116.779, 123.68])
                                                .resize_(1, 3, 1, 1))
                self.resnet = torch.jit.trace(torchvision.models.resnet18(),
                                              torch.rand(1, 3, 224, 224))

            @torch.jit.script_method
            def helper(self, input):
              return self.resnet(input - self.means)

            @torch.jit.script_method
            def forward(self, input):
                return self.helper(input)

If expressions
    ``x if x > y else y``

Casts
    ``float(ten)``, ``int(3.5)``, ``bool(ten)``

Accessing Module Parameters
    ``self.my_parameter`` ``self.my_submodule.my_parameter``


Statements
~~~~~~~~~~

Torch Script supports the following types of statements:

Simple Assignments

    ::

        a = b
        a += b # short-hand for a = a + b, does not operate in-place on a
        a -= b

Pattern Matching Assignments

    ::

        a, b = tuple_or_list
        a, b, *c = a_tuple

Print Statements

  ``print("the result of an add:", a + b)``

If Statements

    ::

        if a < 4:
            r = -a
        elif a < 3:
            r = a + a
        else:
            r = 3 * a

While Loops

  ::

      a = 0
      while a < 4:
          print(a)
          a += 1


For loops with ``range``

    ::

        x = 0
        for i in range(0, 10):
            x *= i

    .. note::
      Script currently does not support iterating over generic iterable
      objects like lists or tensors. This will be added in a future version.

For loops over tuples:

    ::

        tup = (3, torch.rand(4))
        for x in tup:
            print(x)

    .. note::
      for loops over tuples will unroll the loop, generating a body for
      each member of the tuple. The body must type-check correctly for each member.

For loops over constant ``torch.nn.ModuleList``

      ::

          class SubModule(torch.jit.ScriptModule):
              def __init__(self):
                  super(Sub, self).__init__()
                  self.weight = nn.Parameter(torch.randn(2))

              @torch.jit.script_method
              def forward(self, input):
                  return self.weight + input

          class MyModule(torch.jit.ScriptModule):
              __constants__ = ['mods']

              def __init__(self):
                  super(MyModule, self).__init__()
                  self.mods = torch.nn.ModuleList([SubModule() for i in range(10)])

              @torch.jit.script_method
              def forward(self, v):
                  for module in self.mods:
                      v = m(v)
                  return v

      .. note::
          To use a module list inside a ``@script_method`` it must be marked
          constant by adding the name of the attribute to the ``__constants__``
          list for the type. For loops over a ModuleList will unroll the body of the
          loop at compile time, with each member of the constant module list.

Return
    ``return a, b``

    .. note::
        there must be a return statement as the last member of the function
        and return statements cannot appear anywhere else in the function. This
        restriction will be removed in the future.

Variable Resolution
~~~~~~~~~~~~~~~~~~~

Torch Script supports a subset of Python's variable resolution (i.e. scoping)
rules. Local variables behave the same as in Python, except for the restriction
that a variable must have the same type along all paths through a function.
If a variable has a different type on different sides of an if statement, it
is an error to use it after the end of the if statement.

Similarly, a variable is not allowed to be used if it is only *defined* along some
paths through the function.

Example::

    @torch.jit.script
    def foo(x):
        if x < 0:
            y = 4
        print(y) # Error: undefined value y

Non-local variables are resolved to Python values at compile time when the
function is defined. These values are then converted into Torch Script values using
the rules described in `Use of Python Values`_.

Use of Python Values
~~~~~~~~~~~~~~~~~~~~

To make writing Torch Script more convenient, we allow script code to refer
to Python values in the surrounding scope. For instance, any time there is a
reference to ``torch``, the Torch Script compiler is actually resolving it to the
``torch`` Python module when the function is declared.  These Python values are
not a first class part of Torch Script. Instead they are desugared at compile-time
into the primitive types that Torch Script supports. This section describes the
rules that are used when accessing Python values in Torch Script. They depend
on the dynamic type of the python valued referenced.

Functions
  Torch Script can call python functions. This functionality is very useful when
  incrementally converting a model into script. The model can be moved function-by-function
  to script, leaving calls to Python functions in place. This way you can incrementally
  check the correctness of the model as you go.

  Example::

      def foo(x):
        print("I am called with {}".format(x))
        import pdb; pdb.set_trace()
        return x

      @torch.jit.script
      def bar(x)
        return foo(x + 1)

  .. note::
    Attempting to call ``save`` on a ScriptModule that contains calls to Python
    functions will fail. The intention is that this pathway is used for debugging
    and the calls removed or turned into script functions before saving.


Attribute Lookup On Python Modules
    Torch Script can lookup attributes on modules. Builtin functions like ``torch.add``
    are accessed this way. This allows Torch Script to call functions defined in
    other modules.

Python-defined Constants
    Torch Script also provides a way to use constants that are defined in Python.
    These can be used to hard-code hyper-parameters into the function, or to
    define universal constants. There are two ways of specifying that a Python
    value should be treated as a constant.

    1. Values looked up as attributes of a module are assumed to be constant.
       Example: ``math.pi``
    2. Attributes of a ScriptModule can be marked constant by listing them
       as a member of the ``__constants__`` property of the class:

       Example::

           class Foo(torch.jit.ScriptModule):
               __constants__ = ['a']

               def __init__(self):
                   super(Foo, self).__init__(False)
                   self.a = 1 + 4

              @torch.jit.ScriptModule
              def forward(self, input):
                  return self.a + input

    Supported constant Python Values are

    * ``int``
    * ``bool``
    * ``torch.device``
    * ``torch.layout``
    * ``torch.dtype``
    * tuples containing supported types
    * ``torch.nn.ModuleList`` which can be used in a TorchScript for loop


Debugging
~~~~~~~~~

Print things

Use ``USE_PYTHON=0`` to debug in normal python mode

Look at the graph

Pay attention to tracer warnings


Builtin Functions
~~~~~~~~~~~~~~~~~

Torch Script supports a subset of the builtin tensor and neural network functions that
PyTorch provides. Most methods on Tensor as well as functions in the ``torch``
namespace are available. Many functions in ``torch.nn.functional`` are also availiable.


We currently do not provide any builtin ScriptModules e.g. a ``Linear`` or
``Conv`` module. This functionality is something that will be developed in the future.
For now we suggest using ``torch.jit.trace`` to transform standard ``torch.nn``
modules into ScriptModules on construction.

.. automodule:: torch.jit.supported_ops
