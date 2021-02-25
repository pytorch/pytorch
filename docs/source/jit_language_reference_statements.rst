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
    `starred_expression <https://docs.python.org/3/reference/expressions.html#grammar-token-starred-expression>`

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
    `target <https://docs.python.org/3/reference/simple_stmts.html#grammar-token-target>`
    `identifier <https://docs.python.org/3/reference/lexical_analysis.html#grammar-token-identifier>`
    `attributeref <https://docs.python.org/3/reference/expressions.html#grammar-token-attributeref>`
    `subscription <https://docs.python.org/3/reference/expressions.html#grammar-token-subscription>`
    `slicing <https://docs.python.org/3/reference/expressions.html#grammar-token-slicing>`
    `target_list <https://docs.python.org/3/reference/simple_stmts.html#grammar-token-target-list>`
    `starred_expression <https://docs.python.org/3/reference/expressions.html#grammar-token-starred-expression>`

* `yield expression` is not supported in Torchscript.

Augmented assignment statements:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::
    augmented_assignment_stmt ::=  augtarget augop (expression_list)
    augtarget ::=  identifier | attributeref | subscription
    augop ::=  "+=" | "-=" | "*=" | "/=" | "//=" | "%=" |
                "**="| ">>=" | "<<=" | "&=" | "^=" | "|="
    `augtarget <https://docs.python.org/3/reference/simple_stmts.html#grammar-token-augtarget>`
    `augop <https://docs.python.org/3/reference/simple_stmts.html#grammar-token-augop>`
    `identifier <https://docs.python.org/3/reference/lexical_analysis.html#grammar-token-identifier>`
    `attributeref <https://docs.python.org/3/reference/expressions.html#grammar-token-attributeref>`
    `subscription <https://docs.python.org/3/reference/expressions.html#grammar-token-subscription>`
    `expression_list <https://docs.python.org/3/reference/expressions.html#grammar-token-expression-list>`

* `yield expression` is not supported in Torchscript.

Annotated assignment statements:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::
    annotated_assignment_stmt ::=  augtarget ":" expression
                               ["=" (starred_expression)]
    `augtarget <https://docs.python.org/3/reference/simple_stmts.html#grammar-token-augtarget>`
    `expression <https://docs.python.org/3/reference/expressions.html#grammar-token-expression>`
    `starred_expression <https://docs.python.org/3/reference/expressions.html#grammar-token-starred-expression>`

* `yield expression` is not supported in Torchscript.

The ``raise`` statement:
^^^^^^^^^^^^^^^^^^^^^^^^
::
    raise_stmt ::=  "raise" [expression ["from" expression]]
    `expression <https://docs.python.org/3/reference/expressions.html#grammar-token-expression>`

* Raise statements in TorchScript do not support ``try\finally``.

The ``assert`` statement:
^^^^^^^^^^^^^^^^^^^^^^^^^
::
    assert_stmt ::=  "assert" expression ["," expression]
    `expression <https://docs.python.org/3/reference/expressions.html#grammar-token-expression>`

* Assert statements in TorchScript do not support ``try\finally``.

The ``return`` statement:
^^^^^^^^^^^^^^^^^^^^^^^^^
::
    return_stmt ::=  "return" [expression_list]
    `expression_list <https://docs.python.org/3/reference/expressions.html#grammar-token-expression-list>`

* Return statements in TorchScript do not support ``try\except\finally``.

The ``del`` statement:
^^^^^^^^^^^^^^^^^^^^^^
::
    del_stmt ::=  "del" target_list
    `target_list <https://docs.python.org/3/reference/simple_stmts.html#grammar-token-target-list>`

The ``pass`` statement:
^^^^^^^^^^^^^^^^^^^^^^
::
    pass_stmt ::= "pass"

The ``print`` statement:
^^^^^^^^^^^^^^^^^^^^^^^^
::
    print_stmt ::= "print"

The ``break`` statement:
^^^^^^^^^^^^^^^^^^^^^^^^
::
    break_stmt ::= "break"

The ``continue`` statement:
^^^^^^^^^^^^^^^^^^^^^^^^^^^
::
    continue_stmt ::= "continue"

The ``import`` statement:
^^^^^^^^^^^^^^^^^^^^^^^^^
::
    import_stmt     ::=  "import" module ["as" identifier] ("," module ["as" identifier])*
                     | "from" relative_module "import" identifier ["as" identifier]
                     ("," identifier ["as" identifier])*
                     | "from" relative_module "import" "(" identifier ["as" identifier]
                     ("," identifier ["as" identifier])* [","] ")"
                     | "from" module "import" "*"
    module          ::=  (identifier ".")* identifier
    relative_module ::=  "."* module | "."+
    `identifier_list <https://docs.python.org/3/reference/lexical_analysis.html#grammar-token-identifier>`

The ``future`` statement:
^^^^^^^^^^^^^^^^^^^^^^^^^
::
    future_stmt ::=  "from" "__future__" "import" feature ["as" identifier]
                 ("," feature ["as" identifier])*
                 | "from" "__future__" "import" "(" feature ["as" identifier]
                 ("," feature ["as" identifier])* [","] ")"
    feature     ::=  identifier
    `identifier <https://docs.python.org/3/reference/lexical_analysis.html#grammar-token-identifier>`


Compound Statements
~~~~~~~~~~~~~~~~~~~

The following section describes the syntax of Compound statements that are supported in TorchScript.
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
    `assignment_expression <https://docs.python.org/3/reference/expressions.html#grammar-token-assignment-expression>`
    `suite <https://docs.python.org/3/reference/compound_stmts.html#grammar-token-suite>`

Ternary ``if/else`` statement:
""""""""""""""""""""""""""""""

::
    if_stmt ::=  return [expression_list] "if" assignment_expression "else" [expression_list]
    `assignment_expression <https://docs.python.org/3/reference/expressions.html#grammar-token-assignment-expression>`
    `expression_list <https://docs.python.org/3/reference/expressions.html#grammar-token-expression-list>`

* ``Tensor`` with 1 dimension is promoted to ``bool``
    For eg.,
    import torch

    @torch.jit.script
    def fn(x: torch.tensor):
        if x: # The tensor gets promoted to bool
            return True
        return False

    >> print(fn(torch.rand(1)))
    >> True

* ``Tensor`` with multi dimensions are not promoted to ``bool``
    For eg.,:

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

* If a condition variable is annotated as Final, either true or false branch is
* evaluated depending on the  evaluation of the conditional variable.
    For eg.,:

    a : torch.jit.final[Bool] = True

    if a:
        return torch.Tensor(2,3)
    else:
        return []

    Here, only True branch is evaluated, since a is annotated as final and set to True

The ``while`` statement:
^^^^^^^^^^^^^^^^^^^^^^^^
::
    while_stmt ::=  "while" assignment_expression ":" suite
    `assignment_expression <https://docs.python.org/3/reference/expressions.html#grammar-token-assignment-expression>`
    `suite <https://docs.python.org/3/reference/compound_stmts.html#grammar-token-suite>`

* `while...else` statements are not supported in Torchscript. It results in a `RuntimeError`

The ``for-in`` statement:
^^^^^^^^^^^^^^^^^^^^^^^^
::
    for_stmt ::=  "for" target_list "in" expression_list ":" suite
                  ["else" ":" suite]
    `assignment_expression <https://docs.python.org/3/reference/expressions.html#grammar-token-assignment-expression>`
    `expression_list <https://docs.python.org/3/reference/expressions.html#grammar-token-expression-list>`
    `suite <https://docs.python.org/3/reference/compound_stmts.html#grammar-token-suite>`

* `for...else` statements are not supported in Torchscript. It results in a `RuntimeError`

* for loops on tuples: These unroll the loop,  generating a body for each member of the tuple.
    * The body must type-check correctly for each member.

    For eg.,
    tup = (3, torch.rand(4))
    for x in tup:
        print(x)

*  for loops on lists: For loops over a ``nn.ModuleList`` will unroll the body of the loop at compile time,
    * with each member of the module list.

    For eg.,

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
* The `with <https://docs.python.org/3/reference/compound_stmts.html#with>` statement is used to wrap the
* execution of a block with methods defined by a context manager

::
    with_stmt ::=  "with" with_item ("," with_item) ":" suite
    with_item ::=  expression ["as" target]
    `with_item <https://docs.python.org/3/reference/compound_stmts.html#grammar-token-with-item>`
    `suite <https://docs.python.org/3/reference/compound_stmts.html#grammar-token-suite>`

* If a target was included in the `with <https://docs.python.org/3/reference/compound_stmts.html#with>` statement,
* the return value from context manager’s `__enter__() <https://docs.python.org/3/reference/datamodel.html#object.__enter__>`
* is assigned to it. If an exception caused the suite to be exited, its type, value, and traceback are passed as
* arguments to `__exit__() <https://docs.python.org/3/reference/datamodel.html#object.__exit__>. Otherwise, three
* `None <https://docs.python.org/3/library/constants.html#None>` arguments are supplied.

* `try/except/finally` statements are not supported inside `with` blocks.
*  Exceptions raised within `with` block cannot be suppressed.

The ``tuple`` statement:
^^^^^^^^^^^^^^^^^^^^^^^

::
    tuple_stmt ::= tuple([iterables])

* Iterable types in TorchScript include `Tensors`, `lists`, `tuples`, `dictionaries`, `strings`,
    * `torch.nn.ModuleList` and `torch.nn.ModuleDict`.
* Cannot convert a List to Tuple by using this built-in function.
* Unpacking all outputs into a tuple is covered by:
    abc = func() # Function that returns a tuple
    a,b = func()

The ``getattr`` statement:
^^^^^^^^^^^^^^^^^^^^^^^^^^
::
    getattr_stmt ::= getattr(object, name[, default])

* Attribute name must be a literal string.
* Module type object is not supported for eg., torch._C
* Custom Class object is not supported for eg., torch.classes.*

The ``getattr`` statement:
^^^^^^^^^^^^^^^^^^^^^^^^^^
::
    hasattr_stmt ::= hasattr(object, name)

* Attribute name must be a literal string.
* Module type object is not supported for eg., torch._C
* Custom Class object is not supported for eg., torch.classes.*

The ``zip`` statement:
^^^^^^^^^^^^^^^^^^^^^^
::
    zip_stmt ::= zip(iterable1, iterable2)

* Arguments must be iterables.
* Two iterables of same outer container type but different length are supported.

    For eg:
        a = [1, 2]
        b = [2, 3, 4]
        zip(a, b) # works

* Both the iterables must be of the same type - ( list here).

    For eg:
        a = (1, 2) # Tuple
        b = [2, 3, 4] # List
        zip(a, b) # Runtime error

    # RuntimeError: Can not iterate over a module list or
        # tuple with a value that does not have a statically determinable length.

* Two iterables of same container Type but different data type is supported

    For eg:
        a = [1.3, 2.4]
        b = [2, 3, 4]
        zip(a, b) # Works

* Iterable types in TorchScript include `Tensors`, `lists`, `tuples`, `dictionaries`, `strings`,
* `torch.nn.ModuleList` and `torch.nn.ModuleDict`.

The ``enumerate`` statement:
^^^^^^^^^^^^^^^^^^^^^^

::
    enumerate_stmt ::= enumerate([iterable])

* Arguments must be iterables.
* Iterable types in TorchScript include `Tensors`, `lists`, `tuples`, `dictionaries`, `strings`,
    * `torch.nn.ModuleList` and `torch.nn.ModuleDict`.
