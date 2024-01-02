Guards Overview
===============

From a UX perspective, TorchDynamo is very easy to use. The user invokes
``torchdynamo.optimize`` as an annotation:

.. code-block:: python

   @torchdynamo.optimize(my_compiler)
   def fn_foo(bar):

Where a complete example looks like this:

.. code-block:: python

   from typing import List
   import torch
   from torch import _dynamo as torchdynamo

   def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
       print("my_compiler() called with FX graph:")
       gm.graph.print_tabular()
       return gm.forward  # return a python callable

   @torchdynamo.optimize(my_compiler)
   def toy_example(a, b):
       x = a / (torch.abs(a) + 1)
       if b.sum() < 0:
           b = b * -1
       return x * b

   for _ in range(100):
       toy_example(torch.randn(10), torch.randn(10))

This allows TorchDynamo to capture the interpreted Python frames, grab
any and all relevant information, and speed things up wherever it can.
The speedup comes from a few places, and can be rather dependent on the
backend (`my_compiler` in the example above) provided, but the one speedup
that is important in this section is **caching**. Caching itself is not
a direct speedup but a critical enablement that prevents
recompilation. We dig a hole with dynamo, and caching allows us to get
out. It enables us to hold perf
neutrality while then enabling backends - the true source of our
speedups.

With even a pass-through no-op backend provided:

.. code-block:: python

   def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
       return gm.forward

We can see TorchDynamo speeding up Python execution even on
regular Python, not just PyTorch.

Caching and Guards Overview
---------------------------

TorchDynamo operates through caching transformed (by TorchDynamo) user
bytecode. When TorchDynamo receives a frame for evaluation, it checks if the
**objects referenced in the frame have changed** in certain ways, and if
not, TorchDynamo reads the previously transformed user bytecode to evaluate it.
In this section, we will focus on how we can identify whether or not the
**objects referenced in the frame have changed**. This is a critical
piece of functionality in TorchDynamo, because it drives the entire
invalidation lifecycle. This functionality is called **guards**.

At a very high level, the flow can be summarized like this:

1. TorchDynamo receives a Python frame.
2. It converts the frame (1) passing it through instruction
   translation.
3. For the objects captured in (2), TorchDynamo creates tracking objects that
   are:

   - tracked on an output graph, which is an internal specialization of a `torch.fx.Tracer`
   - guards

4. TorchDynamo processes the guard objects created in (3), turning them into a
   generated Python function, `check_fn`, associated with a piece of code.
5. The `check_fn` is evaluated whenever we encounter this code a
   subsequent time - if a `check_fn` passes and evaluates to `True`, TorchDynamo
   identifies the code in the cache and the code encountered here as same, and
   can be safely used. If it fails and evaluates to `False`, TorchDynamo
   identifies the code in the cache as not valid, and can be thrown out in
   favor of a new entry, through recompilation or a graph break.

Python Frame Evaluation and PEP 523
-----------------------------------

The functionality of TorchDynamo is based on
`PEP 523 <https://peps.python.org/pep-0523/>`__.

TorchDynamo installs a frame evaluation function on Python by using
`_PyInterpreterState_SetEvalFrameFunc`. TorchDynamo has a hook where
Python can hand control back to us during evaluation.

The function we have installed is ``convert_frame`` or
``convert_frame_assert`` in the ``nopython=True`` case, but glossing
over that nuance for now, let’s take a look at ``convert_frame_assert``,
as ``convert_frame`` proxies to it.

We can find the function in ``torch/_dynamo/convert_frame.py`` with a signature
as follows:

.. code-block:: python

   def  convert_frame_assert(compiler_fn: Callable, one_graph=True):

This function wraps the entry point of where Python invokes TorchDynamo
with a frame:

.. code-block:: python

   def  _convert_frame_assert(frame: types.FrameType, cache_size: int):

Here is what this function does:

1. Checks if it has seen this ``code``\ (see: f_code `here
   <https://docs.python.org/3/library/inspect.html>`__) before and exits
   early if it did.
2. Checks if the code is an unsupported case.
3. Checks if the ``cache_size`` (second arg above) crosses the limit
   defined in the config, ``cache_size_limit``. If it has, the function
   drops the frame and logs warnings. This helps to avoid constant
   recompilation of a frame as it generally means that the frame is hot
   in an unexpected way and caching it produces needless overhead,
   as it is likely to get evicted the next time it is encountered.
4. Passes the frame, alongside a function that creates an
   ``InstructionTranslator`` through bytecode
   transformation, via ``transform_code_object``. A few crucial things
   happen under the hood here:

   1. New code is produced through ``transform_code_object``.

   2. An FX tracer named ``output`` is produced through
      ``InstructionTranslator``. This can be a bit confusing,
      as ``InstructionTranslator`` is not an `fx` tracer, but its stored
      in a variable named tracer, and its output **is** an `fx` tracer.

   3. The function produces guards and stores them on ``output`` above.

   4. The function produces ``output_instructions`` and stores them on
      ``output`` above.

   5. The function maps the newly produced transformed code to the initial code it
      read off the frame. This mapping is worth remembering, we will
      refer to it much later on below where we cover guard failures.

5. Using the transformed code from 4.1 and the guards from 4.3,
   the function produces a `GuardedCode`.

Now that we have learned about frame evaluation, let’s review
``InstructionTranslator``, and see how it turns the frame we handed
it over into TorchDynamo internal types.

InstructionTranslator
---------------------

`InstructionTranslator` does a lot! We won’t cover the details of
everything it does, but most importantly for this document, it produces
a mapping of ``symbolic_locals`` which maintains a mapping from the
frame’s ``f_locals`` to TorchDynamo internal Variable objects (more on these
in a moment. ``symbolic_locals`` is filled via traversing the frame’s
locals:

.. code-block:: python

   self.symbolic_locals = collections.OrderedDict(
       (k, VariableBuilder(self, LocalSource(k))(f_locals[k]))
       for k in vars
       if k in f_locals
   )

The important component here  is the invocation of a call
into ``VariableBuilder``. ``VariableBuilder``\ ’s call implementation
proxies into a function called ``_wrap``, which in turn both constructs
instances of ``VariableTracker`` and calls ``make_guards`` on them. More
on that later.

This mapping, in turn, is critical as each Variable has associated
guards, which are then passed to ``self.output``, the instance of
``OutputGraph``, an fx tracer, mentioned in 4.2 of the section above. If
you recall, this ``OutputGraph``, stored in a variable called ``output``
is where our guards are stored before being passed on to become
``GuardedCode``

How does ``InstructionTranslator`` do this? At the heart of it, there is
a loop that is pumped, which drives a function ``step``.

``step`` is just that - a single processing step, taking exactly one
instruction and doing *something* with it.

.. note:: These are real instructions processed by TorchDynamo’s
   ``transform_code_object``, and it is pretty cool.

.. note:: This section purposely skips the details of
   `dis.get_instructions <https://docs.python.org/3/library/dis.html>`__.

For the example above, here is a snippet of a what a few
``Instruction``\'s may look like:

.. code-block:: python

   Instruction(opcode=124, opname='LOAD_FAST', arg=0, argval='b', offset=32, starts_line=8, is_jump_target=True, target=None)
   Instruction(opcode=100, opname='LOAD_CONST', arg=3, argval=-1, offset=34, starts_line=None, is_jump_target=False, target=None)
   Instruction(opcode=20, opname='BINARY_MULTIPLY', arg=None, argval=None, offset=36, starts_line=None, is_jump_target=False, target=None)

This is the core functionality of this function. Take a look at the ``opname``,
and then take a look at this little snippet from inside ``step``;

.. code-block:: python

   if not hasattr(self, inst.opname):
       unimplemented(f"missing: {inst.opname}")
   getattr(self, inst.opname)(inst)

As we can see, the function checks if the current class, the
``InstructionTranslator`` has an attribute set matching the operator name
(for example, ``LOAD_CONST``). If it does, the function invokes it, passing the
whole instruction object in. If it does not, the function drops the frame as
unimplemented.

For the ``LOAD_CONST`` example, we can see that we do indeed support it,
with a relatively straightforward definition:

.. code-block:: python

   def LOAD_CONST(self, inst):
       self.push(ConstantVariable(value=inst.argval))

We can see that this function creates a new instance of the class
``ConstantVariable`` , with a value, in our example case, -1, and then
pushes it onto the stack.

There are dozens of such methods - see ``symbolic_convert.py`` for all of
them. Generally, we implement as many matching methods to Python
bytecode instructions as possible.

Across both the logic downstream of ``step`` and the logic from invoking
``VariableBuilder`` - we now have a lot of ``VariableTracker``\ s and of
course, we’ve spoken about creating guards quiet a bit. Let’s dig into
what Variables are, and get a little closer to understanding guards.

Variables
---------

A ``ConstantVariable`` is an instance of ``VariableTracker``.
``VariableTracker`` represents a tracked Python local or stack value.

When it comes to representing an object inside TorchDynamo, a
``VariableTracker`` does exactly what it says - it tracks a given variable.
It is an extremely flexible class, but there are a few points to keep in
mind:

-  It manages the ``guard`` relationship around the underlying object
   through:

   -  ``make_guard``
   -  ``replace_guards``
   -  ``add_guard(s)``
   -  ``propagate`` - ``propagate(*vars: List[List["VariableTracker"]])`` -
      Perhaps the most important of all, in that it combines guards from
      all the provided ``VariableTracker`` instances passed in. It visits
      the guards and combines the guards from these onto itself.

-  It acts as a proxy on behalf of the underlying object, implementing
   methods for the rest of TorchDynamo to get information about the
   tracked object:

   -  ``call_method``
   -  ``call_function``
   -  ``python_type``
   -  ``as_proxy``
   -  ``is/as_python_proxy``

-  It stores the variable ``source`` of type ``Source``, from
   ``torchdynamo/source.py``. This source type is a relatively self
   contained class that helps us organize and bookkeep where the original
   source came from, and helps provide convenience methods for things
   like getting the name, and importantly for us, producing guards.

And this class (``VariableTracker``) is built around subclassing,
somewhere between a full Abstract Base Class and fully fleshed out class
- it leaves many methods raising ``NotImplementedError`` - with reliance on
subclasses. See ``torchdynamo/variables/`` for all subclasses to fulfill
contracts and custom behaviors.

Knowing what we know now, we can see an example of how an instruction
from ``dis``, ``BUILD_TUPLE``:

   ``BUILD_TUPLE(count)`` Creates a tuple consuming count items from the
   stack, and pushes the resulting tuple onto the stack.

In our case, our signature will be a *little* different due to the way
we create ``Instruction`` objects, but the gist of it will be the same.
Instead of passing in ``count``, we pass in an object with a little
extra bookkeeping, and of course, we deal with turning regular old
python objects into TorchDynamo notions:

.. code-block:: python

   def BUILD_TUPLE(self, inst):
       items = self.popn(inst.argval)
       options = VariableTracker.propagate(items)
       self.push(TupleVariable(items, **options))

Here is what this code does:

1. The function reads ``argval``, which in this case, is
   analogous to ``counts`` in the pydoc for the equivalent instruction.

2. The function ``popn`` the items, in this case, the signature is
   ``def  popn(self, n: int) -> List[TensorVariable]:`` this hints at an
   underlying contract - we are returning ``TensorVariables``. If we
   take a closer look at ``symbolic_convert.py`` and
   ``InstructionTranslatorBase``/``InstructionTranslator``\ we see that
   the only thing pushed onto and popped from our stack are
   ``VariableTracker``\ s.

3) The function calls ``VariableTracker.propagate``. This
   takes the guards from every single item popped off the stack in 2,
   and recursively traverses it and combines all the guards into
   ``options``: ``py  return {      "guards": guards,  }``

4) The function then makes a new instance of a ``VariableTracker``,
   ``TupleVariable``\ out of the ``items`` and ``options``. This then
   allows us to install all the appropriate guards from the ``items``
   that make up the new ``TupleVariable``

.. note:: Where did the first guards come from? Propagation
   is a good technique, but we need something created before it can be
   propagated. ``VariableBuilder`` calls
   ``make_guards`` as it creates ``VariableTracker`` instances, from
   ``f_locals``. This in turn calls into the ``source``, to have it create
   guards.

After all this, bytecode translation is done and we are one step closer
to producing ``GuardedCode``. We now understand how locals become
``VariableTracker``\ s, how instructions are handled, and where guards
are called on for creation. Before we can go into seeing how code and
guards are combined into a GuardedCode object, we need to dig a little
bit into those ``make_guard`` and ``source.make_guard`` calls above. We
can then understand, what was going on when we made guards
alongside, and on, ``VariableTracker`` instances.

Making Guards
-------------

Guards are just Python objects, of the class ``Guard``. Let's look at them
in more detail.

Looking at the definition of the dataclass (and therefore, ctor
signature), we see that it has a name, a source, and a create function.

.. code-block:: python

   @dataclasses.dataclass
   class Guard:
       name: str
       source: GuardSource
       create_fn: Callable

The name should be the name of the variable.

The source here is an enum indicating what *kind* of source the guard
belongs to.

.. note:: Not to be confused with ``Source`` and the other types
   in ``source.py``, as stored on ``VariableTracker``.

``create_fn`` provides the main functionality to transition from a simple
dataclass to actually producing valid Python code to be invoked for
knowing whether or not things have changed in between invocations, and
whether we can safely read from the code cache or not.

The most common code paths for getting an instance of a guard are
through ``make_guards`` on ``VariableTracker``.
``make_guards`` -> ``source.make_guard`` -> ``return Guard(self.name(), self.guard_source(), fn)``

Or, in a concrete example:

.. code-block:: python

   ...
   elif istype(value, range):
       guards = self.make_guards(GuardBuilder.EQUALS_MATCH)
       return RangeVariable(value=value, guards=guards)

Since ``source`` was set at the construction time of this
``VariableTracker``, all that was needed here was to provide the ``fn``,
``GuardBuilder.EQUALS_MATCH`` to the ``create_fn`` field.

This ``create_fn`` must be a method on ``GuardBuilder``. The reason for
this becomes apparent in our next step. Once we have all the guards
created for a frame, we move on to ``CheckFunctionManager`` and
``compile_check_fn``.

Before the ``convert_frame`` function can produce a ``GuardedCode``,
it needs to run the ``CheckFunctionManager``, with all the guards, to
produce a ``check_fn`` which will then, in turn get passed in alongside
the code into ``GuardedCode``. This is the same ``check_fn`` that we store in our
cache entry, and the same one we run to know whether or not to retrieve
the code stored alongside. For reference, here is that code:

.. code-block:: cpp

   static CacheEntry *create_cache_entry(CacheEntry *next,
                                         PyObject *guarded_code) {
     CacheEntry *e = (CacheEntry *)malloc(sizeof(CacheEntry));
     DEBUG_NULL_CHECK(e);
     e->check_fn = PyObject_GetAttrString(guarded_code, "check_fn");
     NULL_CHECK(e->check_fn);
     e->code = (PyCodeObject *)PyObject_GetAttrString(guarded_code, "code");
     NULL_CHECK(e->code);
     e->next = next;
     return e;
   }

We now know how a ``check_fn`` function is used, and who makes it, and
what it is composed of, but what we do not yet know is how. How does a
list of ``Guard`` objects become a function we can run later on?

First, we iterate these guards:

.. code-block:: python

   for guard in sorted(guards or [], key=Guard.sort_key):
       if not config.guard_nn_modules and guard.is_nn_module():
           continue
       guard.create(local_builder, global_builder)

Calling ``guard.create`` runs that ``create_fn`` we set on the ``Guard``
class above (don’t confuse it with the ``check_fn`` we are working on
producing, the names are similar, so it can get a little confusing). In
our example above, our ``create_fn`` is ``GuardBuilder.EQUALS_MATCH``.
So we are now invoking it, passing in the ``self``, the guard itself,
in.

The signature is: ``def EQUALS_MATCH(self, guard: Guard):``

And internally to that function, we can use the ``name`` on the guard to
get back our original object, querying it for data and type information,
which in turn gets us to the most important bit: appending code.

At its simplest, ``EQUALS_MATCH`` appends just one line of code:
``self.code.append(f"{ref} == {val!r}")``. Where ``ref`` is the name of
the variable, and ``val`` is the value. It might produce code like this:

.. code-block:: python

   y == 2

This is a basic example. But if we append a few other kinds of ``GuardBuilder``
functions and then combine them all with
``and`` in between each statement (as we do), we might get something
like this:

.. code-block:: python

   ___guarded_code.valid and ___check_type_id(y, 94367738391392) and y == 2 and ___check_tensors(x)

Here is what this code performs:

1. A check for ``.valid``
2. A type ID check
3. A value check
4. A tensor check

This becomes the heart of the code our ``check_fn``, which in turn
is evaluated the **next** time we encounter this code. It
will then check:

1. Is this code still valid?
2. If (1), Does ``y`` still have a type of ``94367738391392``?
3. If (2), is ``y`` still 2?
4. If (3), let’s check on if tensor ``x`` changed in some specific ways.

If all of these are still true, then we can use the code cached
alongside this ``check_fn``.

.. note:: For a deeper dive for how and where this happens
   you can read ``static PyCodeObject *lookup(CacheEntry *e, PyObject *f_locals) {`` of
   ``_eval_frame.c``.

If not, then, we can move on to recompiling the code anew, and storing
that in the cache alongside this code, and a whole new ``check_fn``,
again to be checked on yet another subsequent frame.

There are lots of other such functions on ``GuardBuilder`` which get
coalesced into, at times massive, strings which then get evaluated as
Python code and stored into ``check_fn``. The example above
illustrates of a simple case. To understand this functionality better, read
the other functions on ``GuardBuilder``, or better yet, dump the ``code`` variable
in ``compile_check_fn`` to see what is getting produced,
especially on larger, real models.

Summary
-------

In this section, we have reviewed:

- The role of ``.valid`` and invalidation around weak references (and potentially soon to be NN Moduleinvalidations).
- How the C++ side of guard functions (``___check_type_id``, ``___check_tensors``, etc) operate.
- What happens when guards fail.
- What happens if we produce invalid guard code.

We covered how user provided code wrapped in a TorchDynamo context
goes on to get traced and tracked internally, organized into ``VariableTracker``\ s
``Source``\ s and subsequently ``Guard``\ s, and how those ``Guards`` in
turn guide cache entry selection and invalidation when handing Python
code.
