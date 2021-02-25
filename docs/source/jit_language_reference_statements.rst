.. contents::
    :local:
    :depth: 2


.. testsetup::

    # These are hidden from the docs, but these are necessary for `doctest`
    # since the `inspect` module doesn't play nicely with the execution
    # environment for `doctest`
    import torch

    original_script = torch.jit.script
    def script_wrapper(obj, *args, **kwargs):
        obj.__module__ = 'FakeMod'
        return original_script(obj, *args, **kwargs)

    torch.jit.script = script_wrapper

    original_trace = torch.jit.trace
    def trace_wrapper(obj, *args, **kwargs):
        obj.__module__ = 'FakeMod'
        return original_trace(obj, *args, **kwargs)

    torch.jit.trace = trace_wrapper

.. _language-reference:

TorchScript Language Reference
==============================

.. statements:

Simple Statements
~~~~~~~~~~~~~~~~~

The following section describes the syntax of simple statements that are supported in TorchScript.
It is modeled after `the simple statements chapter of the Python language reference <https://docs.python.org/3/reference/simple_stmts.html>`_.

Expression statements:
^^^^^^^^^^^^^^^^^^^^^^

::

    expression_stmt ::=  starred_expression
    starred_expression ::=  expression | (starred_item ",")* [starred_item]
    starred_item       ::=  assignment_expression | "*" or_expr

Assignment Statements:
^^^^^^^^^^^^^^^^^^^^^^

::

    assignment_stmt ::=  (target_list "=")+ (starred_expression)
    target_list     ::=  target ("," target)* [","]
    target          ::=  identifier
                     | "(" [target_list] ")"
                     | "[" [target_list] "]"
                     | attributeref
                     | subscription
                     | slicing
                     | "*" target

Augmented assignment statements:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    augmented_assignment_stmt ::=  augtarget augop (expression_list)
    augtarget ::=  identifier | attributeref | subscription
    augop ::=  "+=" | "-=" | "*=" | "/=" | "//=" | "%=" |
                "**="| ">>=" | "<<=" | "&=" | "^=" | "|="


Annotated assignment statements:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

    annotated_assignment_stmt ::=  augtarget ":" expression
                               ["=" (starred_expression)]

The ``raise`` statement:
^^^^^^^^^^^^^^^^^^^^^^^^

::

    raise_stmt ::=  "raise" [expression ["from" expression]]

* Raise statements in TorchScript do not support ``try\except\finally``.

The ``assert`` statement:
^^^^^^^^^^^^^^^^^^^^^^^^^

::

    assert_stmt ::=  "assert" expression ["," expression]

* Assert statements in TorchScript do not support ``try\except\finally``.

The ``return`` statement:
^^^^^^^^^^^^^^^^^^^^^^^^^

::

    return_stmt ::=  "return" [expression_list]

* Return statements in TorchScript do not support ``try\except\finally``.

The ``del`` statement:
^^^^^^^^^^^^^^^^^^^^^^

::

    del_stmt ::=  "del" target_list

The ``pass`` statement:
^^^^^^^^^^^^^^^^^^^^^^

::

    pass_stmt ::= "pass"

The ``print`` statement:
^^^^^^^^^^^^^^^^^^^^^^^^

::

    print_stmt ::= "print" "(" expression  [, expression] [.format{expression_list}] ")"

The ``break`` statement:
^^^^^^^^^^^^^^^^^^^^^^^^

::

    break_stmt ::= "break"

The ``continue`` statement:
^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    continue_stmt ::= "continue"

Compound Statements
~~~~~~~~~~~~~~~~~~~

The following section describes the syntax of compound statements that are supported in TorchScript.
The section also highlights how Torchscript differs from regular Python statements.
It is modeled after `the compound statements chapter of the Python language reference <https://docs.python.org/3/reference/compound_stmts.html>`_.

The ``if`` statement:
^^^^^^^^^^^^^^^^^^^^^
* Torchscript supports both basic ``if/else`` and ternary ``if/else``.

Basic ``if/else`` statement:
""""""""""""""""""""""""""""

::

    if_stmt ::=  "if" assignment_expression ":" suite
             ("elif" assignment_expression ":" suite)
             ["else" ":" suite]

Ternary ``if/else`` statement:
""""""""""""""""""""""""""""""

::

    if_stmt ::=  return [expression_list] "if" assignment_expression "else" [expression_list]

Example

* ``Tensor`` with 1 dimension is promoted to ``bool``

.. testcode::

    import torch

    @torch.jit.script
    def fn(x: torch.tensor):
        if x: # The tensor gets promoted to bool
            return True
        return False

.. testoutput::

    >> print(fn(torch.rand(1)))
    >> True

* ``Tensor`` with multi dimensions are not promoted to ``bool``

Example

.. testcode::

    import torch

    # Multi dimensional Tensors error out.
    # This below code gives RuntimeError:

    @torch.jit.script
    def fn():
        if torch.rand(2):
            print("Tensor is available")

        if torch.rand(4,5,6):
            print("Tensor is available")

    >> print(fn())

* If a conditional variable is annotated as Final, either true or false branch is
* evaluated depending on the evaluation of the conditional variable.

Example

.. testcode::

    import torch

    a : torch.jit.final[Bool] = True

    if a:
        return torch.Tensor(2,3)
    else:
        return []

    Here, only True branch is evaluated, since ``a`` is annotated as ``final`` and set to ``True``

The ``while`` statement:
^^^^^^^^^^^^^^^^^^^^^^^^
::

    while_stmt ::=  "while" assignment_expression ":" suite

* `while...else` statements are not supported in Torchscript. It results in a `RuntimeError`

The ``for-in`` statement:
^^^^^^^^^^^^^^^^^^^^^^^^
::

    for_stmt ::=  "for" target_list "in" expression_list ":" suite
                  ["else" ":" suite]

* ``for...else`` statements are not supported in Torchscript. It results in a ``RuntimeError``

* for loops on tuples: These unroll the loop, generating a body for each member of the tuple.The body must type-check correctly for each member.

Example

.. testcode::

    tup = (3, torch.rand(4))
    for x in tup:
        print(x)

*  for loops on lists: For loops over a ``nn.ModuleList`` will unroll the body of the loop at compile time, with each member of the module list.

Example

.. testcode::

    class SubModule(torch.nn.Module):
        def __init__(self):
            super(SubModule, self).__init__()
            self.weight = nn.Parameter(torch.randn(2))

        def forward(self, input):
            return self.weight + input

    class MyModule(torch.nn.Module):

        def __init__(self):
            super(MyModule, self).init()
            self.mods = torch.nn.ModuleList([SubModule() for i in range(10)])

        def forward(self, v):
            for module in self.mods:
                v = module(v)
            return v

    model = torch.jit.script(MyModule())

The ``with`` statement:
^^^^^^^^^^^^^^^^^^^^^^^
* The ``with`` statement is used to wrap the execution of a block with methods defined by a context manager

::

    with_stmt ::=  "with" with_item ("," with_item) ":" suite
    with_item ::=  expression ["as" target]

* If a target was included in the ``with`` statement, the return value from context manager’s ``__enter__()``
* is assigned to it. Unlike python, if an exception caused the suite to be exited, its type, value, and traceback are
* not passed as arguments to ``__exit__()``. Three ``None`` arguments are supplied.

* ``try/except/finally`` statements are not supported inside ``with`` blocks.
*  Exceptions raised within ``with`` block cannot be suppressed.

The ``tuple`` statement:
^^^^^^^^^^^^^^^^^^^^^^^

::

    tuple_stmt ::= tuple([iterables])

* Iterable types in TorchScript include ``Tensors``, ``lists``,``tuples``, ``dictionaries``, ``strings``,``torch.nn.ModuleList`` and ``torch.nn.ModuleDict``.
* Cannot convert a List to Tuple by using this built-in function.
* Unpacking all outputs into a tuple is covered by:

..testcode::

    abc = func() # Function that returns a tuple
    a,b = func()

The ``getattr`` statement:
^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    getattr_stmt ::= getattr(object, name[, default])

* Attribute name must be a literal string.
* Module type object is not supported for example, torch._C
* Custom Class object is not supported for example, torch.classes.*

The ``getattr`` statement:
^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    hasattr_stmt ::= hasattr(object, name)

* Attribute name must be a literal string.
* Module type object is not supported for example, torch._C
* Custom Class object is not supported for example, torch.classes.*

The ``zip`` statement:
^^^^^^^^^^^^^^^^^^^^^^

::

    zip_stmt ::= zip(iterable1, iterable2)

* Arguments must be iterables.
* Two iterables of same outer container type but different length are supported.

Example

..testcode::

    a = [1, 2] # List
    b = [2, 3, 4] # List
    zip(a, b) # works

* Both the iterables must be of the same container type - (List here).

Example

..testcode::

    a = (1, 2) # Tuple
    b = [2, 3, 4] # List
    zip(a, b) # Runtime error

..testoutput::

    >> RuntimeError: Can not iterate over a module list or
        tuple with a value that does not have a statically determinable length.

* Two iterables of same container Type but different data type is supported

Example

..testcode::

    a = [1.3, 2.4]
    b = [2, 3, 4]
    zip(a, b) # Works

* Iterable types in TorchScript include ``Tensors``, ``lists``, ``tuples``, ``dictionaries``, ``strings``, ``torch.nn.ModuleList`` and ``torch.nn.ModuleDict``.

The ``enumerate`` statement:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    enumerate_stmt ::= enumerate([iterable])

* Arguments must be iterables.
* Iterable types in TorchScript include ``Tensors``, ``lists``, ``tuples``, ``dictionaries``, ``strings``, ``torch.nn.ModuleList`` and ``torch.nn.ModuleDict``.
