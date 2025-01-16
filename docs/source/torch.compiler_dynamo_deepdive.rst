.. _torch.compiler_dynamo_deepdive:

Dynamo Deep-Dive
================

TorchDynamo (or simply Dynamo) is the tracer within ``torch.compile``,
and it is, more often than not, the one to blame for those insane
backtraces. However, we cannot blindly blame Dynamo for these errors. In
order to provide the user with the flexibility it does, Dynamo is given
the arduous task of understanding any Python program. In particular,
Dynamo has to implement a good part of the Python programming language
internally!

In this post, we will go over the internal design of Dynamo from the
ground up. We will discuss the functionality it provides, and how it is
implemented. By the end of this post, you will have a better
understanding of what went wrong when you ``torch.compiled`` a PyTorch
program and the compilation errored out, or succeeded but the speed-up
was not what you expected.

A Gentle Introduction to Dynamo
-------------------------------

Before getting our hands dirty with all the implementation details,
let’s start by discussing what it is that Dynamo does.

Dynamo is a tracer. This means, given and function and inputs to it, it
executes the function and records a linear sequence of instructions
(without control flow) into a graph. For example, consider the following
program:

.. code:: python

   import torch

   @torch.compile
   def mse(x, y):
       z = (x - y) ** 2
       return z.sum()

   x = torch.randn(200)
   y = torch.randn(200)
   mse(x, y)

If we save this program into the file ``example.py`` and we run

.. code:: bash

   TORCH_LOGS=graph_code python example.py

we see the output that Dynamo traced

.. code:: python

   def forward(l_x_: torch.Tensor, l_y_: torch.Tensor):
       # File: example.py:5, code: z = (x - y) ** 2
       sub = l_x_ - l_y_
       z = sub ** 2
       # File: example.py:6, code: return z.sum()
       sum_1 = z.sum()
       return (sum_1,)

We call this a **graph (or trace) of the function for the given
inputs**. This is represented via an `FX
graph <https://pytorch.org/docs/main/fx.html>`__. We will simply think
of an FX graph as a container that stores a list of function calls.

The first thing we should notice is that the graph is a linear sequence
of PyTorch operations. [1]_ Dynamo records all the PyTorch operations
and stores them sequentially. For example, it split ``z = (x - y) ** 2``
into its two constituting operations, ``sub = l_x_ - l_y_`` and
``z = sub ** 2``.

When we say that the trace is linear, we mean that there is no branching
or any control flow. To see this, consider

.. code:: python

   import torch

   @torch.compile
   def fn(x, n):
       y = x ** 2
       if n >= 0:
           return (n + 1) * y
       else:
           return y / n

   x = torch.randn(200)
   fn(x, 2)

which, when executed with ``TORCH_LOGS=graph_code``, returns

.. code:: python

   def forward(l_x_: torch.Tensor):
       # File: example.py:5, code: y = x ** 2
       y = l_x_ ** 2
       # File: example.py:7, code: return (n + 1) * y
       mul = 3 * y
       return (mul,)

We see that Dynamo completely removed the ``if`` statement from the
trace and just recorded the operations that were executed with the
inputs.

As such, it should be clear that **the trace of a function depends on
the inputs**. In particular, this means that the trace is not generated
when we write ``@torch.compile``, but when we execute the function
``fn(x, 2)`` with the actual arguments.

The other interesting thing to note here is that Dynamo removed the
second argument to the function. Instead, it treated it as a constant
and recorded the result of the operation ``n + 1`` in the graph. This is
another feature of Dynamo: Dynamo will treat as constant any non-tensor
value… other than ints. Let’s see now how are ints special.

The last defining property of Dynamo is that it knows how to handle
dynamic shapes. Symbolic shapes refer to Dynamo’s ability of tracing
shapes, and more generally, integers, rather than leaving them as
constants. This allows for avoiding recompilations and deploying generic
models that work for any size in production. The main examples of places
where dynamic shapes appear are the batch size, where we might train a
model with a fixed batch size but then perform inference for an
arbitrary batch size, or the variable sequence length that one
encounters when processing text or audio.

We can see this by executing a few more times the example above

.. code:: python

   import torch

   @torch.compile
   def fn(x, n):
       y = x ** 2
       if n >= 0:
           return (n + 1) * y
       else:
           return y / n

   x = torch.randn(200)
   fn(x, 2)
   fn(x, 3)
   fn(x, -2)

In this case, ``TORCH_LOGS=graph_code`` generates two more graphs

.. code:: python

   # Graph for n==2 omitted

   def forward(self, l_x_: torch.Tensor, l_n_: torch.SymInt):
       # File: a.py:5, code: y = x ** 2
       y = l_x_ ** 2

       # File: a.py:7, code: return (n + 1) * y
       add = l_n_ + 1
       mul = add * y
       return (mul,)

.. code:: python

   def forward(self, l_x_: torch.Tensor, l_n_: torch.SymInt):
       # File: a.py:5, code: y = x ** 2
       y = l_x_ ** 2

       # File: a.py:9, code: return y / n
       truediv = y / l_n_
       return (truediv,)

Dynamo detected that one integer changed its value after the first call
and started tracing it. We see that these graphs are generic, and trace
the variable ``n`` symbolically via an object of type ``SymInt``.

If after these calls we call ``fn(x, 4)``, Dynamo would not recompile,
but rather reuse the graph that was already traced.

To summarize: 1. Dynamo is a Python tracer 2. Given some inputs, it
returns an FX graph with the PyTorch functions that were executed 3. It
can also trace integers if it detects that they changed between calls 4.
It specializes any other value that is not a tensor or a scalar

Of course, Dynamo does many more things, like figuring out when it needs
to retrace, rewriting the bytecode of the function, implementing graph
breaks… To keep the introduction short, we will incrementally discuss
all these in the sequel.

PEP 523: Adding a frame evaluation API to CPython
-------------------------------------------------

Imagine now that we are given the task to implement Dynamo. Where would
we even start? Rather conveniently for us, `PEP
523 <https://peps.python.org/pep-0523/>`__ was released with Python 3.6.
This PEP `was
designed <https://peps.python.org/pep-0523/#a-jit-for-cpython>`__ to
allow third parties to create JIT compilers for Python. Let’s see how.

**A note on CPython**: CPython is internally implemented as a `stack
machine <https://en.wikipedia.org/wiki/Stack_machine>`__. A Python
program is compiled into
`bytecodes <https://en.wikipedia.org/wiki/Bytecode>`__ that then are
executed by this interpreter. To learn more about these bytecodes, see
the `dis module <https://docs.python.org/3/library/dis.html>`__ from the
standard library. See also `the developer
docs <https://devguide.python.org/internals/interpreter/>`__ for an
introduction to CPython’s interpreter. We will assume that the reader is
familiar with the notion of a stack machine.

PEP 523 exposes an API where a user can add a custom per-function
interpreter. Then, CPython will use this interpreter rather than its own
to execute the function. In order to be able to execute the function, on
entry, CPython provides the custom interpreter with things like - The
bytecode of the function - The value of the arguments of the function
(i.e., the local variables) and their names - The value of the global
variables and their names - The builtin functions like ``abs`` or
``print``

You can see all the fields
`here <https://github.com/pytorch/pytorch/blob/e891a3bba9f05697d72776f6e89347231a141f03/torch/csrc/dynamo/eval_frame.c#L50-L59>`__. [2]_

In summary, CPython provides the user’s interpreter with all the
information necessary to execute the function. [3]_

With this API, we can implement a tracer by implementing an interpreter
that runs the code and records in a graph all the PyTorch operations
that occur during this execution. This is exactly what Dynamo does.

Dynamo uses this CPython API to parse all these objects and packs them
into `a Python
structure <https://github.com/pytorch/pytorch/blob/e891a3bba9f05697d72776f6e89347231a141f03/torch/csrc/dynamo/eval_frame.c#L93-L108>`__.
After it has done so… it goes back from C to python. Other than for this
piece of code that communicates with CPython, Dynamo is fully
implemented in Python.

It should be clear that it is the decorator ``@torch.compile``\ ’s job
to install the necessary scaffolding that will pass the bytecode, the
args, global variables and so on to Dynamo when the function is called.
Again, ``@torch.compile`` does not actually compile anything.

Implementing CPython in Python
------------------------------

So, we are back in the Python world. We have the bytecode of a function,
and all the context necessary to execute it. In particular, we have
landed at
`_convert_frame_assert <https://github.com/pytorch/pytorch/blob/b6df8414601e1e086e830ca9e919e7fdc8874e71/torch/_dynamo/convert_frame.py#L272-L274>`__.
This is the function that the decorator ``torch.compile`` returns! We
get to this function from
`_dynamo.optimize <https://github.com/pytorch/pytorch/blob/b6df8414601e1e086e830ca9e919e7fdc8874e71/torch/_dynamo/eval_frame.py#L715-L727>`__.
The decorator ``torch.compile`` is just a nice API around
``_dynamo.optimize``.

Before getting into implementing a Python interpreter, we want to define
an `IR <https://en.wikipedia.org/wiki/Intermediate_representation>`__.
In particular, we want to wrap all the local and global variables in our
own internal classes. This allows us to better track these objects and
group together objects that can be treated in the same way to the eyes
of Dynamo.

The parent class of the internal class structure is ``VariableTracker``
and represents the different objects that Dynamo understands. For
example, ``ListVariable``, represents a ``list`` object, and keeps
internally a `list of VariableTrackers <https://github.com/pytorch/pytorch/blob/e38a3a6079a3861b4bc9f256120ec661f34e726d/torch/_dynamo/variables/lists.py#L48-L56>`__.
Another example of ``VariableTracker`` is
`ConstantVariable <https://github.com/pytorch/pytorch/blob/83c0763dda1f93c6cf552ba88260a0dc7a3ecb70/torch/_dynamo/variables/constant.py#L30>`__.
ConstantVariable wraps all the `objects considered constant by
Dynamo <https://github.com/pytorch/pytorch/blob/83c0763dda1f93c6cf552ba88260a0dc7a3ecb70/torch/_dynamo/variables/constant.py#L98-L107>`__.
We also have special subclasses for objects that require special
attention, like
`TensorVariable <https://github.com/pytorch/pytorch/blob/83c0763dda1f93c6cf552ba88260a0dc7a3ecb70/torch/_dynamo/variables/tensor.py#L68-L69>`__.
All these internal classes are defined in the
`torch/_dynamo/variables <https://github.com/pytorch/pytorch/tree/83c0763dda1f93c6cf552ba88260a0dc7a3ecb70/torch/_dynamo/variables>`__
folder.

Python objects are wrapped into their corresponding ``VariableTracker``
class in
`VariableBuilder._wrap <https://github.com/pytorch/pytorch/blob/83c0763dda1f93c6cf552ba88260a0dc7a3ecb70/torch/_dynamo/variables/builder.py#L365>`__.
This function is just a very long chain of ``elif``\ s that tries to
recursively pattern-match the Python inputs into the appropriate type of
``VariableTracker``.

**Debugging tip**. When we get unexpected results from dynamo, it is
sometimes caused by the builder. If the logic of the builder is wrong,
sometimes Dynamo may wrap a variable in the incorrect
``VariableTracker`` type, and this may cause issues later on. It is
rather useful to have a look at the ``VariableTracker`` types that
appear in the errors, and the ``VariableTracker`` method that throws the
exception when you encounter a Dynamo error. In particular, sometimes we
find that an object is tracked as a ``UserDefinedObjectVariable`` (this
is Dynamo’s catch-all class), when it should have been tracked as
something more specific. In these cases, the ``SourceBuilder.__call__``
logic is often to blame.

**Debugging tip**. When running a program with ``TORCH_LOGS=dynamo``,
one of the artifacts that are printed out is lines of the form

::

   TRACE LOAD_GLOBAL y [TorchInGraphFunctionVariable(<built-in method any>), TensorVariable()]

This is the bytecode for the original program and the state of the stack
at that point. This is very useful to find where an object was not
traced into the right ``VariableTracker``.

Ok, so we have an IR for our tracer, now we *just* need to reimplement
CPython’s stack machine. This is implemented by
`InstructorTranslatorBase <https://github.com/pytorch/pytorch/blob/69f112d5867f785a3a090a0c6d6644ae047033ac/torch/_dynamo/symbolic_convert.py#L576-L594>`__
in
`symbolic_convert.py <https://github.com/pytorch/pytorch/blob/69f112d5867f785a3a090a0c6d6644ae047033ac/torch/_dynamo/symbolic_convert.py>`__.

``InstructionTranslatorBase`` has about 200 methods, implementing almost
all of Python bytecodes. As an example, we can see the implementation of
``BUILD_LIST``

.. code:: python

   def BUILD_LIST(self, inst):
       items = self.popn(inst.argval)
       self.push(ListVariable(items, mutation_type=ValueMutationNew()))

This is the bytecode generated by constructions like ``l = [2, 3, 4]``.
In this case, since there are three elements, the generated bytecode is
``BUILD_LIST 3``. This means that we pop the top ``3`` elements of the
stack and push a new list object to the top of the stack formed by these
three elements.

Generating the Output Graph
---------------------------

With a way to symbolically execute Python code, we are set to extract
the PyTorch operations that happen during the symbolic execution of a
program given some inputs. This is implemented in Dynamo via the
`OutputGraph <https://github.com/pytorch/pytorch/blob/69f112d5867f785a3a090a0c6d6644ae047033ac/torch/_dynamo/output_graph.py#L221-L230>`__
object. The ``OutputGraph`` object is `bound to an
`InstructionTranslator object <https://github.com/pytorch/pytorch/blob/69f112d5867f785a3a090a0c6d6644ae047033ac/torch/_dynamo/symbolic_convert.py#L2060-L2071>`__
and it tracks all the data necessary to create the FX graph which will
be returned by Dynamo.

All the inputs and intermediary elements of the FX graph are
``fx.Node``\ s. In Dynamo, ``fx.Node``\ s are wrapped in
``fx.Proxy``\ s. ``fx.Proxy``\ s are used to build the FX graph.
In particular, they record every PyTorch operation performed on them
into the graph. You can create a new operation to be added to
the graph by calling `create_proxy <https://github.com/pytorch/pytorch/blob/fb80f05ee2e1cba17892980701bfd5dbce58349f/torch/_dynamo/output_graph.py#L430-L431>`__.
Then, we can add it to the graph through the function
`wrap_fx_proxy <https://github.com/pytorch/pytorch/blob/fb80f05ee2e1cba17892980701bfd5dbce58349f/torch/_dynamo/variables/builder.py#L1311>`__.

A graph stores operations on tensors… and operations on symbolic
integers. We will discuss symbolic integers later on, but first we will
discuss how Dynamo addresses a rather important correctness issue.

.. _making-dynamo-sound-guards:

Making Dynamo Sound: Guards
---------------------------

At this point, we have a way to trace programs completely disregarding control flow.
And for that, we have reimplemented all of CPython… If this sounds like a bit of an
overkill, that is because it is.
`torch.jit.trace <https://pytorch.org/docs/main/generated/torch.jit.trace.html>`__
already implements this without all this machinery, so what gives?

The issue with ``torch.jit.trace``, as it is warned in its docs, is that
it just works if the traced program is not data dependent. In other
words, it will just work if the program itself is linear. This means
writing our program without using if-elses, for-while loops, exceptions.
Even more, none of the libraries that we use can use any control flow!
All in all, not using control flow in a language as dynamic as Python
is, in fact, a huge constraint.

JAX solves this problem by always retracing and caching the graph after
retracing. Dynamo, on the other hand, uses guards to avoid retracing the
whole program every time.

A **guard** is an assumption (a boolean expression on an input) made in
order to specialize a frame for one set of example inputs. Reusing the
graph is only valid if these assumptions hold on the new inputs.

For example, any constant input to a function, like a string, installs a
guard stating that that input should be of type ``str`` and equal to the
string we passed. Running

.. code:: python

   import torch

   @torch.compile
   def fn(a, b):
       return a * len(b)

   fn(torch.arange(10), "Hello")

with ``TORCH_LOGS=guards`` prints (among other guards)

.. code:: python

   ___check_type_id(L['b'], 94334122025024)
   L['b'] == 'Hello'

This reads as “the local variable ``b`` should have a specific type
(``str`` in this case, represented by the constant ``9433...``) and
its value should be ``'Hello'``”. If we then execute the function
again passing a different argument

.. code:: python

   import torch

   @torch.compile
   def fn(a, b):
       return a * len(b)

   fn(torch.arange(10), "Hello")
   fn(torch.arange(10), "Hi")

we can see the guard that failed by running ``TORCH_LOGS=recompiles``

.. code:: python

   Recompiling function fn in script.py:3
   triggered by the following guard failure(s):
        - L['b'] == 'Hello'

Guards are accumulated while `the inputs to the function are wrapped in
the
builder <https://github.com/pytorch/pytorch/blob/69f112d5867f785a3a090a0c6d6644ae047033ac/torch/_dynamo/variables/builder.py#L808-L810>`__
and `during the execution of the
program <https://github.com/pytorch/pytorch/blob/69f112d5867f785a3a090a0c6d6644ae047033ac/torch/_dynamo/variables/dicts.py#L763-L769>`__.
We will show many more examples of guards in the next section, but first
let us discuss sources.

A **source** tracks how to reconstruct a variable from the original
local or global variables present when entering the current frame. In
particular, it tracks the original local and global objects and any of
the objects they contain. In

.. code:: python

   def foo(x: Tensor, y: List[Tensor]):
       a = x * y[0]
       return a * x

``x`` and ``y`` have
`LocalSource <https://github.com/pytorch/pytorch/blob/40dc0580a69565b06ec5263efe5d87cecc8200f7/torch/_dynamo/source.py#L80-L92>`__
as their source, and ``y[0]`` has
`GetItemSource <https://github.com/pytorch/pytorch/blob/40dc0580a69565b06ec5263efe5d87cecc8200f7/torch/_dynamo/source.py#L302>`__,
which stores a ``LocalSource`` inside. On the other hand, ``a`` will not
have a source as it is an intermediate variable that only exists within
the fx graph.

All these are defined in
`torch/_dynamo/source.py <https://github.com/pytorch/pytorch/blob/main/torch/_dynamo/source.py>`__.
We can see the guard generated by ``GetItemSource`` in the following
example:

.. code:: python

   import torch

   @torch.compile
   def fn(x, l):
       return x * len(l[0])

   fn(torch.randn(8), ["Hi", "Hello"])

generates the following guards

.. code:: python

   ___check_type_id(L['l'], 94439025877664)
   len(L['l']) == 2
   ___check_type_id(L['l'][0], 94439025840192)
   L['l'][0] == 'Hi'
   ___check_type_id(L['l'][1], 94439025840192)
   L['l'][1] == 'Hello'

Here, we see the code generated by ``GetItemSource`` (``[0]`` and
``[1]``) wrapping a ``LocalSource`` (``L['l']``).

At this point, with sources and guards, we are able to implement a
caching system to avoid recompilation without having to retrace every
time. We will discuss a bit more in detail this caching system in the
sequel.

The attentive reader will have noticed that this does not explain yet
why we need to have such fine control over the Python interpreter as to
having to reimplement it. The examples of guards that we have shown
depend on the input objects, so we could still compute these before
executing the function. In other words, we could implement this guard
system on top of ``torch.jit.trace`` and get the same functionality with
much less effort… Enter symbolic shapes.

Symbolic Shapes
---------------

Another point we discussed in the introduction is that Dynamo knows how
to trace integers. In order to implement this, we use a symbolic class
`torch.SymInt <https://github.com/pytorch/pytorch/blob/fb80f05ee2e1cba17892980701bfd5dbce58349f/torch/__init__.py#L244-L249>`__
that acts like an ``int`` but it records all the operations performed on
it in the output FX graph. [4]_ We already saw this class in the introduction
when introducing symbolic integer tracing.

Let us now discuss the three properties that define symbolic shape
tracing in Dynamo, and how to implement them.

Static by default
^^^^^^^^^^^^^^^^^

Dynamo assumes that every integer, let that be an input or the shape of
a tensor, is static by default. In other words, no integers will be
traced on the first execution of a function. Then, only if it detects
that an integer or a shape changed value during the execution, it will
trace it and generate a graph generic on that variable.

We already saw this behavior in the introduction using integers. Let us
now look at an example using shapes of tensors.

.. code:: python

   import torch

   @torch.compile
   def fn(a, b):
       return a.shape[0] * a * b

   fn(torch.randn(4, 3), torch.randn(4, 3))
   fn(torch.randn(8, 3), torch.randn(8, 3))

Running this program with ``TORCH_LOGS=graph_code`` we see that these
two calls are traced as

.. code:: python

   def forward(self, l_a_: torch.Tensor, l_b_: torch.Tensor):
       mul = 4 * l_a_
       mul_1 = mul * l_b_
       return (mul_1,)

   def forward(self, s0: torch.SymInt, l_a_: torch.Tensor, l_b_: torch.Tensor):
       size = l_a_.size()
       getitem = size[0]
       mul = getitem * l_a_
       mul_1 = mul * l_b_
       return (mul_1,)

In the first graph the shape is traced as a constant, but once it
changes, it traces it symbolically using a ``SymInt``\ s. In general, a
simpler way to see the shapes of the intermediary values is by running
the program with ``TORCH_LOGS=graph_sizes``

::

   TRACED GRAPH TENSOR SIZES
   ===== __compiled_fn_1 =====
   l_a_: (s0, 3)
   l_a_ (concrete): (8, 3)
   l_b_: (s0, 3)
   l_b_ (concrete): (8, 3)
   mul: (s0, 3)
   mul (concrete): (8, 3)
   mul_1: (s0, 3)
   mul_1 (concrete): (8, 3)

where we can see that the first dimension of the two tensor args is
dynamic, given that it is represented by the ``s0`` variable.

We can find how Dynamo implements this by running ``TORCH_LOGS=guards``

.. code:: python

   # Guards first call
   check_tensor(L['a'], torch.float32, device=None, requires_grad=False, size=[4, 3], stride=[3, 1])
   check_tensor(L['b'], torch.float32, device=None, requires_grad=False, size=[4, 3], stride=[3, 1])

   # Guards second call
   check_tensor(L['a'], torch.float32, device=None, requires_grad=False, size=[None, 3], stride=[3, 1])
   check_tensor(L['b'], torch.float32, device=None, requires_grad=False, size=[None, 3], stride=[3, 1])

   L['b'].size()[0] == L['a'].size()[0]
   2 <= L['a'].size()[0]

We see that on the first call, the guards check that the tensors have
some fixed sizes and strides. These guards fail in the second execution,
so it retraces. Since it was an ``int`` guard that failed, in this
second iteration it traces this ``int`` symbolically and it installs
more general guards on this more generic kernel.

**Compilation performance tip**. If you know that a dimension will vary
in size, you can mark it as dynamic by calling
`torch._dynamo.mark_dynamic <https://github.com/pytorch/pytorch/blob/66a76516bfc341b2b55bb2056d2faa9c2de46d69/torch/_dynamo/decorators.py#L176>`__
before calling ``torch.compile``. This will avoid the first compilation
with a static shape. There are other useful utility functions like
``maybe_mark_dynamic`` or ``mark_static``. You can also have all
integers and shapes traced by calling ``torch.compile(dynamic=True)``.
This is mostly useful for debugging purposes.

0, 1 are always specialized
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Regardless of whether we mark a dimension as dynamic, if we pass an input
where that dimension is 0 or 1, Dynamo will trace it as non-dynamic and it
will generate a specific graph for it. This is the reason why in the example
above we find guards of the form ``2 <= L['a'].size()[0]``.

There are several reasons for this choice. There are two particularly
important - A tensor is empty if and only if any of its dimensions is
zero - A tensor can only be contiguous if one of the strides is one

This policy decision does NOT apply to plain Python ints; if we think a Python
int should be compiled dynamically, we won't specialize them by default;
instead, whether or not it gets specialized depends on its usage.

Duck shaping
^^^^^^^^^^^^

Dynamo performs what we call “duck shaping”. If two dynamic integers
have the same value at trace time, we will assume that they are equal
and guard on it. Effectively, this means that rather than having two
symbols ``s0``, ``s1`` in the example above, we just unified them to
``s0`` and had the guard ``L['b'].size()[0] == L['a'].size()[0]``. This
enables performing fusions within the compiler while being able to
generate kernels that are generic enough.

Guards on symbolic ints
^^^^^^^^^^^^^^^^^^^^^^^

We now understand how symbolic shapes are implemented at a high level
and the properties they have. Now, why is that symbolic shapes forced us
through the tricky route of getting control of the CPython interpreter?
Consider the following example:

.. code:: python

   import torch

   @torch.compile(dynamic=True)
   def fn(a):
       if a.shape[0] * 2 < 16:
           return a
       else:
           return a + 1

   fn(torch.randn(8))

This code has a guard of the form ``2*L['a'].size()[0] >= 16``. This is
a non-trivial guard in terms of the inputs of the function, but it is
registered in the middle of the execution of the program. Even more so,
we cannot know this guard is needed until we see the ``if`` statement
conditional on a ``SymNodeVariable`` argument. Such conditions are
invisible to ``torch.jit.trace`` and require deep analysis of the python
code.

**Debugging tip** Running this code with ``TORCH_LOGS=dynamo`` tells us
where this guard was added

::

   eval 2*s0 >= 16 [guard added] at script.py:5 in fn (_dynamo/variables/tensor.py:812 in evaluate_expr)

Placing a breakpoint there and looking at the backtrace is rather useful
to understand where a guard came from.

Making Dynamo Complete: Graph Breaks
------------------------------------

With all the tools we have discussed, we have a tracer that can trace
PyTorch operations on tensors and integers and has a caching system that
knows when it can reuse a previously traced graph and when it needs to
retrace. All this executing arbitrary Python code!

There is just one small issue with this. The statement “executing
arbitrary Python code” is perhaps a bit too general. Dynamo implements a
good part of Python, but does it implement the more complex parts, like
coroutines or async? Does it implement the whole Python standard
library? NumPy also has a Python API. Does ``torch.compile`` also
understand NumPy? and Django? [5]_

Python’s ecosystem is massive, and a good part of it is written in other
more performant languages like C++ or Rust, and it just exposes Python
bindings. There is no hope in Dynamo tracing through Python objects that
are implemented in C++. What can a tracer do when it finds an operation
that it does not understand?

The usual way machine learning tracers handle this issue is by informing
the user that the operation they choked on and giving up tracing
altogether. This would pose a real usability issue in the case of
PyTorch, where its users are used to the flexibility it gives them. As a
real-world example the ``doctr_det_predictor`` model uses NumPy and the
``cv2`` library to `postprocess the model’s
result <https://github.com/mindee/doctr/blob/f2114758d529ed8d3d0030581638f0520b6b98d8/doctr/models/detection/core.py#L86>`__.

Here is another place where having access to CPython is interesting.
Rather than erroring out, Dynamo can let CPython run that problematic
code! To do this, Dynamo generates at trace time one graph with all the
operations before the problematic code, and one with all the operations
after. [6]_ Then, at runtime, it will delegate to CPython to execute the
first graph, then the problematic code, and then the second graph. This
process of stopping the tracing and generating multiple graphs is called
a **graph break**.

A small confession: I lied all throughout the introduction and the first
sections. Dynamo does not generate one graph, but **multiple graphs**!
For all practical purposes, starting retracing after a second graph can
be thought of as starting tracing a new function. The new graph after
the graph break will have its own guards, its new set of local
variables, and so on.

To discuss how to implement graph breaks, we need to first revisit how
Dynamo interacts with CPython. Using PEP 523, CPython allows a user to
use their own frame evaluation mechanism. What we had not discussed is
that CPython also exposes its own frame evaluation for others to use.
Dynamo leverages this to let the fast CPython interpreter run the
compiled code. For a function without graph breaks, the whole tracing /
execution process of a program that calls the function 2 times with the
same arguments looks like this:

1. In the first call to the function

   1. Dynamo traces the function into an FX graph

      1. The FX graph is compiled by the compiler (Inductor) into
         efficient low-level code… but that’s a story for another day

   2. It rewrites the bytecode of the function so that it simply calls
      the compiled function
   3. It gives CPython this new bytecode and asks it to run it
      [`here <https://github.com/pytorch/pytorch/blob/e891a3bba9f05697d72776f6e89347231a141f03/torch/csrc/dynamo/eval_frame.c#L1006>`__]

2. In the second call to the function

   1. It checks the guards from the first call against the new arguments
      [`here <https://github.com/pytorch/pytorch/blob/e891a3bba9f05697d72776f6e89347231a141f03/torch/csrc/dynamo/eval_frame.c#L658>`__].
      Since they are the same arguments as before, they pass
   2. It asks CPython to run the bytecode associated to those guards
      [`here <https://github.com/pytorch/pytorch/blob/e891a3bba9f05697d72776f6e89347231a141f03/torch/csrc/dynamo/eval_frame.c#L972-L975>`__]

This process on its own looks overly complicated. Why generate new
bytecode and ask CPython to run it rather than simply creating a C++
binding to the compiled function and executing it? Well, this pattern
allows us to implement graph breaks! The bytecode generated by a graph
break has the following structure:

1. Bytecode that executes the first graph
2. Bytecode that leaves the stack as it would be if CPython would have
   executed the first graph. It also replays any modifications to local
   or global variables that would be visible at this point
3. The bytecode that made Dynamo graph break
4. Bytecode that executes the second graph

Let us see this in a simple example

.. code:: python

   import torch

   @torch.compile
   def fn(a):
       b = a + 2
       print("Hi")
       return b + a

   fn(torch.randn(4))

Running this with ``TORCH_LOGS=bytecode`` shows us the initial bytecode
and the modified bytecode

.. code:: python

   MODIFIED BYTECODE fn script.py line 3
    0 LOAD_GLOBAL              1 (__compiled_fn_0)
    2 LOAD_FAST                0 (a)
    4 CALL_FUNCTION            1
    6 STORE_FAST               3 (graph_out_0)
    8 LOAD_GLOBAL              0 (print)
   10 LOAD_CONST               2 ('Hi')
   12 LOAD_FAST                3 (graph_out_0)
   14 LOAD_CONST               3 (0)
   16 BINARY_SUBSCR
   18 STORE_FAST               1 (b)

   20 CALL_FUNCTION            1
   22 LOAD_GLOBAL              2 (__resume_at_14_1)
   24 ROT_TWO
   26 LOAD_FAST                0 (a)
   28 LOAD_FAST                1 (b)
   30 CALL_FUNCTION            3
   32 RETURN_VALUE

   MODIFIED BYTECODE resume_in_fn script.py line 6
    0 LOAD_GLOBAL              1 (__compiled_fn_2)
    2 LOAD_FAST                2 (b)
    4 LOAD_FAST                1 (a)
    6 CALL_FUNCTION            2
    8 UNPACK_SEQUENCE          1
   10 RETURN_VALUE

We can see that the modified bytecode is split into two functions,
``fn``, the original function, and a function called ``resume_in_fn``.
This second function is a function created by Dynamo to implement the
execution of the program starting at the graph break. This is often
called a `continuation
function <https://en.wikipedia.org/wiki/Continuation>`__. This
continuation function simply calls the second compiled function with the
right arguments. The code for the initial function is rewritten
implementing the strategy that we described before

-  L0-4. Call the compiled function (``a + 2``).
-  L6. Store its result in a local variable called ``graph_out_0``.
   ``graph_out_0`` is a tuple
-  L8-18. Leave the stack as it would be at the point of the graph break
-  L20. Execute the code that caused the graph break
-  L22-32. Call the compiled continuation function (``a + b``)

The code generation of the stack in Dynamo is delegated to
``VariableTracker`` subclasses. Every ``VariableTracker`` object in
Dynamo has a `reconstruct <https://github.com/pytorch/pytorch/blob/e891a3bba9f05697d72776f6e89347231a141f03/torch/_dynamo/variables/lists.py#L307-L309>`__
method that generates the necessary bytecode to create the python object
it represents on the stack.

**Debugging tip**. Graph breaks hamper performance, and as such, it is
best to avoid them. Running a program with ``TORCH_LOGS=graph_breaks``
is a great way to find how many graph breaks did our program hit. The
information it returns is in terms of ``VariableTracker`` objects, so
the debugging tips above are sometimes also helpful to figure out what
caused that graph break.

Conclusion
----------

Dynamo is a complex piece of software. Once you sign up to implement a
CPython interpreter you know you are in for a ride. That being said, we
hope that this post helps demystify it a bit.

Dynamo is (mostly) implemented in Python. We left plenty of links to the
pieces of the code that we discussed. We hope that reading those pieces
of code and grepping for the places that call them, or putting
breakpoints on them and looking at the call stack helps understanding
the rest of the code base.

Of course, the best way to learn how a piece of software works is by
extending it. In this case, the best way is to have a look at the `open
dynamo issues on
github <https://github.com/pytorch/pytorch/issues?q=is%3Aissue+is%3Aopen+label%3A%22module%3A+dynamo%22+>`__.
Many of them require very minor changes in the code, once you find where
you need to make those changes.

Footnotes
---------

.. [1] In the literature, this is called a Directed Acyclical Graph (DAG).

.. [2] All this binding code lives in ``torch/csrc/dynamo/eval_frame.c``.

.. [3] In CPython lingo, the set of all these objects are called `a
   frame <https://github.com/python/cpython/blob/f26bfe4b25f7e5a4f68fcac26207b7175abad208/Include/internal/pycore_frame.h#L57-L71>`__.

.. [4] There are also ``SymBool`` and ``SymFloat`` classes. The latter one
   is not used all that much at the time of this writing.

.. [5] Interestingly enough, it does understand NumPy code! Have a look at
   `this blogpost <https://pytorch.org/blog/compiling-numpy-code/>`__
   and `the docs <https://pytorch.org/docs/main/torch.compiler_faq.html#does-numpy-work-with-torch-compile>`__.
   Now, this is just possible because we reimplemented NumPy using
   PyTorch. Good luck implementing Django in PyTorch though…

.. [6] Assuming there is just one piece of problematic code. If there are
   more, Dynamo can split the code into as many graphs as it needs.
