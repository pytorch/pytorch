TorchScript
============

.. contents:: :local:

.. automodule:: torch.jit
.. currentmodule:: torch.jit

TorchScript is a way to create serializable and optimizable models from PyTorch code.
Any code written in TorchScript can be saved from your Python
process and loaded in a process where there is no Python dependency.

We provide tools to incrementally transition a model from being a pure Python program
to a TorchScript program that can be run independently from Python, for instance, in a standalone C++ program.
This makes it possible to train models in PyTorch using familiar tools and then export
the model to a production environment where it is not a good idea to run models as Python programs
for performance and multi-threading reasons.

Creating TorchScript Code
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
We allow you to compose tracing and scripting to suit the particular requirements
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


TorchScript Language Reference
-------------------------------

TorchScript is a subset of Python that can either be written directly (using
the @script annotations) or generated automatically from Python code via
tracing. When using tracing, code is automatically converted into this subset of
Python by recording only the actual operators on tensors and simply executing and
discarding the other surrounding Python code.

When writing TorchScript directly using @script annotations, the programmer must
only use the subset of Python supported in TorchScript. This section documents
what is supported in TorchScript as if it were a language reference for a stand
alone language. Any features of Python not mentioned in this reference are not
part of TorchScript.

As a subset of Python any valid TorchScript function is also a valid Python
function. This makes it possible to remove the @script annotations and debug the
function using standard Python tools like pdb. The reverse is not true: there
are many valid python programs that are not valid TorchScript programs.
Instead, TorchScript focuses specifically on the features of Python that are
needed to represent neural network models in Torch.

.. envvar:: PYTORCH_JIT=1

    Setting the environment variable ``PYTORCH_JIT=0`` will disable all script
    and tracing annotations. If there is hard-to-debug error in one of your
    ScriptModules, you can use this flag to force everything to run using native
    Python. This allows the use of tools like ``pdb`` to debug code.


Types
~~~~~

The largest difference between TorchScript and the full Python language is that
TorchScript only support a small set of types that are needed to express neural
net models. In particular TorchScript supports:

``Tensor``
    A PyTorch tensor of any dtype, dimension, or backend.

``Tuple[T0, T1, ...]``
    A tuple containing subtypes ``T0``, ``T1``, etc. (e.g. ``Tuple[Tensor, Tensor]``)

``bool``
    A boolean value

``int``
    A scalar integer

``float``
    A scalar floating point number

``List[T]``
    A list of which all members are type ``T``

``Optional[T]``
    A value which is either None or type ``T``

Unlike Python, each variable in TorchScript function must have a single static type.
This makes it easier to optimize TorchScript functions.

Example::

    @torch.jit.script
    def an_error(x):
        if x:
            r = torch.rand(1)
        else:
            r = 4
        return r # Type mismatch: r is set to type Tensor in the true branch
                 # and type int in the false branch


There are 2 scenarios in which you can annotate:

1. Function Argument Type annotation

By default, all parameters to a TorchScript function are assumed to be Tensor
because this is the most common type used in modules. To specify that an
argument to a TorchScript function is another type, it is possible to use
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


2. Variable Type Annotation

For example, a list by default is assumed to be List[Tensor]. If you would like to
have a list of other types. PyTorch provides annotation functions.

Example::

    import torch
    from torch.jit import Tensor
    from typing import List, Tuple

    class ListOfTupleOfTensor(torch.jit.ScriptModule):
        def __init__(self):
            super(ListOfTupleOfTensor, self).__init__()

        @torch.jit.script_method
        def forward(self, x):
            # type: (Tensor) -> List[Tuple[Tensor, Tensor]]

            # This annotates the list to be a List[Tuple[Tensor, Tensor]]
            returns = torch.jit.annotate(List[Tuple[Tensor, Tensor]], [])
            for i in range(10):
                returns.append((x, x))

            return returns


Optional Type Refinement:

TorchScript will refine the type of a variable of type Optional[T] when
a comparison to None is made inside the conditional of an if statement.
The compiler can reason about multiple None checks that are combined with
AND, OR, or NOT. Refinement will also occur for else blocks of if statements
that are not explicitly written.

The expression must be emitted within the conditional; assigning
a None check to a variable and using it in the conditional will not refine types.


Example::

  @torch.jit.script
  def opt_unwrap(x, y, z):
    # type: (Optional[int], Optional[int], Optional[int]) -> int
    if x is None:
      x = 1
    x = x + 1

    if y is not None and z is not None:
      x = y + z
    return x


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
    TorchScript currently does not support mutating tensors in place, so any
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

TorchScript supports the following types of statements:

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
        for i in range(10):
            x *= i

    .. note::
      Script currently does not support iterating over generic iterable
      objects like lists or tensors. Script currently does not support start or
      increment parameters to range. These will be added in a future version.

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
        TorchScript allows returns in the following circumstances:
           1. At the end of a function
           2. In an if-statement where <true> and <false> both return
           3. In an if-statement where <true> returns and <false> is empty (an early return)

Variable Resolution
~~~~~~~~~~~~~~~~~~~

TorchScript supports a subset of Python's variable resolution (i.e. scoping)
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
function is defined. These values are then converted into TorchScript values using
the rules described in `Use of Python Values`_.

Use of Python Values
~~~~~~~~~~~~~~~~~~~~

To make writing TorchScript more convenient, we allow script code to refer
to Python values in the surrounding scope. For instance, any time there is a
reference to ``torch``, the TorchScript compiler is actually resolving it to the
``torch`` Python module when the function is declared.  These Python values are
not a first class part of TorchScript. Instead they are desugared at compile-time
into the primitive types that TorchScript supports. This section describes the
rules that are used when accessing Python values in TorchScript. They depend
on the dynamic type of the python valued referenced.

Functions
  TorchScript can call python functions. This functionality is very useful when
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
    TorchScript can lookup attributes on modules. Builtin functions like ``torch.add``
    are accessed this way. This allows TorchScript to call functions defined in
    other modules.

Python-defined Constants
    TorchScript also provides a way to use constants that are defined in Python.
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

Disable JIT for Debugging
    If you want to disable all JIT modes (tracing and scripting) so you can
    debug your program in raw Python, you can use the ``PYTORCH_JIT`` environment
    variable. ``PYTORCH_JIT`` can be used to globally disable the
    JIT by setting its value to ``0``. Given an example script::

        @torch.jit.script
        def scripted_fn(x : torch.Tensor):
            for i in range(12):
                x = x + x
            return x


        def fn(x):
            x = torch.neg(x)
            import pdb; pdb.set_trace()
            return scripted_fn(x)

        traced_fn = torch.jit.trace(fn, (torch.rand(4, 5),))

        traced_fn(torch.rand(3, 4))

    Debugging this script with PDB works except for when we invoke the @script
    function. We can globally disable JIT, so that we can call the @script
    function as a normal python function and not compile it. If the above script
    is called ``disable_jit_example.py``, we can invoke it like so::

        $ PYTORCH_JIT=0 python disable_jit_example.py

    and we will be able to step into the @script function as a normal Python
    function.


Interpreting Graphs
    TorchScript uses a static single assignment (SSA) intermediate representation
    (IR) to represent computation. The instructions in this format consist of
    ATen (the C++ backend of PyTorch) operators and other primitive operators,
    including control flow operators for loops and conditionals. As an example::

        @torch.jit.script
        def foo(len):
          # type: (int) -> torch.Tensor
          rv = torch.zeros(3, 4)
          for i in range(len):
            if i < 10:
                rv = rv - 1.0
            else:
                rv = rv + 1.0
          return rv

        print(foo.graph)

    A ``ScriptModule`` with a single ``forward`` method will have an attribute
    ``graph``, which you can use to inspect the IR representing the computation.
    If the ScriptModule has more than one method, you will need to access
    ``.graph`` on the method itself and not the module. We can inspect the
    graph of a method named ``bar`` on a ScriptModule by accessing ``.bar.graph``.

    The example script above produces the graph::

	graph(%len : int) {
	  %15 : int = prim::Constant[value=1]()
	  %9 : bool = prim::Constant[value=1]()
	  %7 : Device = prim::Constant[value="cpu"]()
	  %6 : int = prim::Constant[value=0]()
	  %5 : int = prim::Constant[value=6]()
	  %1 : int = prim::Constant[value=3]()
	  %2 : int = prim::Constant[value=4]()
	  %11 : int = prim::Constant[value=10]()
	  %14 : float = prim::Constant[value=1]()
	  %4 : int[] = prim::ListConstruct(%1, %2)
	  %rv.1 : Tensor = aten::zeros(%4, %5, %6, %7)
	  %rv : Tensor = prim::Loop(%len, %9, %rv.1)
	    block0(%i : int, %13 : Tensor) {
	      %12 : bool = aten::lt(%i, %11)
	      %rv.4 : Tensor = prim::If(%12)
		block0() {
		  %rv.2 : Tensor = aten::sub(%13, %14, %15)
		  -> (%rv.2)
		}
		block1() {
		  %rv.3 : Tensor = aten::add(%13, %14, %15)
		  -> (%rv.3)
		}
	      -> (%9, %rv.4)
	    }
	  return (%rv);
	}


    Take the instruction ``%rv.1 : Dynamic = aten::zeros(%3, %4, %5, %6)`` for
    example. ``%rv.1 : Dynamic`` means we assign the output to a (unique)
    value named ``rv.1``, and that value is of ``Dynamic`` type, i.e. we do
    not know its concrete shape. ``aten::zeros`` is the operator (equivalent
    to ``torch.zeros``) and the input list ``(%3, %4, %5, %6)`` specifies which
    values in scope should be passed as inputs. The schema for built-in functions
    like ``aten::zeros`` can be found at `Builtin Functions`_.

    Notice that operators can also have associated ``blocks``, namely the
    ``prim::Loop`` and ``prim::If`` operators. In the graph print-out, these
    operators are formatted to reflect their equivalent source code forms
    to facilitate easy debugging.

    Graphs can be inspected as shown to confirm that the computation described
    by a ``ScriptModule`` is correct, in both automated and manual fashion, as
    described below.


Tracing Edge Cases
    There are some edge cases that exist where the trace of a given Python
    function/module will not be representative of the underlying code. These
    cases can include:

    * Tracing of control flow that is dependent on inputs (e.g. tensor shapes)
    * Tracing of in-place operations of tensor views (e.g. indexing on the
      left-hand side of an assignment)

    Note that these cases may in fact be traceable in the future.


Automatic Trace Checking
    One way to automatically catch many errors in traces is by using ``check_inputs``
    on the ``torch.jit.trace()`` API. ``check_inputs`` takes a list of tuples
    of inputs that will be used to re-trace the computation and verify the
    results. For example::

        def loop_in_traced_fn(x):
            result = x[0]
            for i in range(x.size(0)):
                result = result * x[i]
            return result

        inputs = (torch.rand(3, 4, 5),)
        check_inputs = [(torch.rand(4, 5, 6),), (torch.rand(2, 3, 4),)]

        traced = torch.jit.trace(loop_in_traced_fn, inputs, check_inputs=check_inputs)

    Gives us the following diagnostic information::
	ERROR: Graphs differed across invocations!
	Graph diff::

		  graph(%x : Tensor) {
		    %1 : int = prim::Constant[value=0]()
		    %2 : int = prim::Constant[value=0]()
		    %result.1 : Tensor = aten::select(%x, %1, %2)
		    %4 : int = prim::Constant[value=0]()
		    %5 : int = prim::Constant[value=0]()
		    %6 : Tensor = aten::select(%x, %4, %5)
		    %result.2 : Tensor = aten::mul(%result.1, %6)
		    %8 : int = prim::Constant[value=0]()
		    %9 : int = prim::Constant[value=1]()
		    %10 : Tensor = aten::select(%x, %8, %9)
		-   %result : Tensor = aten::mul(%result.2, %10)
		+   %result.3 : Tensor = aten::mul(%result.2, %10)
		?          ++
		    %12 : int = prim::Constant[value=0]()
		    %13 : int = prim::Constant[value=2]()
		    %14 : Tensor = aten::select(%x, %12, %13)
		+   %result : Tensor = aten::mul(%result.3, %14)
		+   %16 : int = prim::Constant[value=0]()
		+   %17 : int = prim::Constant[value=3]()
		+   %18 : Tensor = aten::select(%x, %16, %17)
		-   %15 : Tensor = aten::mul(%result, %14)
		?     ^                                 ^
		+   %19 : Tensor = aten::mul(%result, %18)
		?     ^                                 ^
		-   return (%15);
		?             ^
		+   return (%19);
		?             ^
		  }


    This message indicates to us that the computation differed between when
    we first traced it and when we traced it with the ``check_inputs``. Indeed,
    the loop within the body of ``loop_in_traced_fn`` depends on the shape
    of the input ``x``, and thus when we try another ``x`` with a different
    shape, the trace differs.

    In this case, data-dependent control flow like this can be captured using
    script instead::

        def fn(x):
            result = x[0]
            for i in range(x.size(0)):
                result = result * x[i]
            return result

        inputs = (torch.rand(3, 4, 5),)
        check_inputs = [(torch.rand(4, 5, 6),), (torch.rand(2, 3, 4),)]

        scripted_fn = torch.jit.script(fn)
        print(scripted_fn.graph)

        for input_tuple in [inputs] + check_inputs:
            torch.testing.assert_allclose(fn(*input_tuple), scripted_fn(*input_tuple))


    Which produces::

	graph(%x : Tensor) {
	  %5 : bool = prim::Constant[value=1]()
	  %1 : int = prim::Constant[value=0]()
	  %result.1 : Tensor = aten::select(%x, %1, %1)
	  %4 : int = aten::size(%x, %1)
	  %result : Tensor = prim::Loop(%4, %5, %result.1)
	    block0(%i : int, %7 : Tensor) {
	      %10 : Tensor = aten::select(%x, %1, %i)
	      %result.2 : Tensor = aten::mul(%7, %10)
	      -> (%5, %result.2)
	    }
	  return (%result);
	}

Tracer Warnings
    The tracer produces warnings for several problematic patterns in traced
    computation. As an example, take a trace of a function that contains an
    in-place assignment on a slice (a view) of a Tensor::

        def fill_row_zero(x):
            x[0] = torch.rand(*x.shape[1:2])
            return x

        traced = torch.jit.trace(fill_row_zero, (torch.rand(3, 4),))
        print(traced.graph)


    Produces several warnings and a graph which simply returns the input::

        fill_row_zero.py:4: TracerWarning: There are 2 live references to the data region being modified when tracing in-place operator copy_ (possibly due to an assignment). This might cause the trace to be incorrect, because all other views that also reference this data will not not reflect this change in the trace! On the other hand, if all other views use the same memory chunk, but are disjoint (e.g. are outputs of torch.split), this might still be safe.
          x[0] = torch.rand(*x.shape[1:2])
        fill_row_zero.py:6: TracerWarning: Output nr 1. of the traced function does not match the corresponding output of the Python function. Detailed error:
        Not within tolerance rtol=1e-05 atol=1e-05 at input[0, 1] (0.09115803241729736 vs. 0.6782537698745728) and 3 other locations (33.00%)
          traced = torch.jit.trace(fill_row_zero, (torch.rand(3, 4),))
        graph(%0 : Float(3, 4)) {
          return (%0);
        }

    We can fix this by modifying the code to not use the in-place update, but
    rather build up the result tensor out-of-place with `torch.cat`::

        def fill_row_zero(x):
            x = torch.cat((torch.rand(1, *x.shape[1:2]), x[1:2]), dim=0)
            return x

        traced = torch.jit.trace(fill_row_zero, (torch.rand(3, 4),))
        print(traced.graph)


Builtin Functions
~~~~~~~~~~~~~~~~~

TorchScript supports a subset of the builtin tensor and neural network
functions that PyTorch provides. Most methods on Tensor as well as functions in
the ``torch`` namespace, all functions in ``torch.nn.functional`` and all
modules from ``torch.nn`` are supported in TorchScript, excluding those in the
table below. For unsupported modules, we suggest using :meth:`torch.jit.trace`.

Unsupported ``torch.nn`` Modules  ::

    torch.nn.modules.adaptive.AdaptiveLogSoftmaxWithLoss
    torch.nn.modules.normalization.CrossMapLRN2d
    torch.nn.modules.fold.Fold
    torch.nn.modules.fold.Unfold
    torch.nn.modules.rnn.GRU
    torch.nn.modules.rnn.LSTM
    torch.nn.modules.rnn.RNN
    torch.nn.modules.rnn.GRUCell
    torch.nn.modules.rnn.LSTMCell
    torch.nn.modules.rnn.RNNCell


.. automodule:: torch.jit.supported_ops

Frequently Asked Questions
--------------------------

Q: I would like to train a model on GPU and do inference on CPU. What are the
best practices?
   First convert your model from GPU to CPU and then save it, like so: ::

      cpu_model = gpu_model.cpu()
      sample_input_cpu = sample_input_gpu.cpu()
      traced_cpu = torch.jit.trace(traced_cpu, sample_input_cpu)
      torch.jit.save(traced_cpu, "cpu.pth")

      traced_gpu = torch.jit.trace(traced_gpu, sample_input_gpu)
      torch.jit.save(traced_gpu, "gpu.pth")

      # ... later, when using the model:

      if use_gpu:
         model = torch.jit.load("gpu.pth")
      else:
         model = torch.jit.load("cpu.pth")

      model(input)

   This is recommended because the tracer may witness tensor creation on a
   specific device, so casting an already-loaded model may have unexpected
   effects. Casting the model *before* saving it ensures that the tracer has
   the correct device information.
