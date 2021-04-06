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

.. _language-reference-v2:

TorchScript Language Reference
==============================

.. _expressions:


Expressions
~~~~~~~~~~~

The following section describes the grammar of expressions that are supported in TorchScript.
It is modeled after `the expressions chapter of the Python language reference <https://docs.python.org/3/reference/expressions.html>`_.

Arithmetic Conversions
^^^^^^^^^^^^^^^^^^^^^^
There are a number of implicit type conversions that are performed in TorchScript:


* a ``Tensor`` with a ``float`` or ``int`` datatype can be implicitly converted to an instance of ``FloatType`` or ``IntType`` provided that it has a size of 0, and does not have ``require_grad`` set to ``True`` and will not require narrowing.
* instances of ``StringType`` can be implicitly converted to ``DeviceType``
* the above implicit conversion rules can be applied to instances of ``TupleType`` to produce instances of ``ListType`` with the appropriate contained type


Explicit conversions can be invoked using the ``float``, ``int``, ``bool``, ``str`` built-in functions
that accept primitive data types as arguments and can accept user-defined types if they implement
``__bool__``, ``__str__``, etc.


Atoms
^^^^^
Atoms are the most basic elements of expressions.

::

    atom      ::=  identifier | literal | enclosure
    enclosure ::=  parenth_form | list_display | dict_display

Identifiers
"""""""""""
The rules that dictate what is a legal identifer in TorchScript are the same as
their `Python counterparts <https://docs.python.org/3/reference/lexical_analysis.html#identifiers>`_.

Literals
""""""""

::

    literal ::=  stringliteral | integer | floatnumber

Evaluation of a literal yields an object of the appropriate type with the specific value
(with approximations applied as necessary for floats). Literals are immutable, and multiple evaluations
of identical literals may obtain the same object or distinct objects with the same value.
`stringliteral <https://docs.python.org/3/reference/lexical_analysis.html#string-and-bytes-literals>`_,
`integer <https://docs.python.org/3/reference/lexical_analysis.html#integer-literals>`_, and
`floatnumber <https://docs.python.org/3/reference/lexical_analysis.html#floating-point-literals>`_
are defined in the same way as their Python counterparts.

Parenthesized Forms
"""""""""""""""""""

::

    parenth_form ::=  '(' [expression_list] ')'

A parenthesized expression list yields whatever the expression list yields. If the list contains at least one
comma, it yields a ``Tuple``; otherwise, it yields the single expression inside the expression list. An empty
pair of parentheses yields an empty ``Tuple`` object (``Tuple[]``).

List and Dictionary Displays
""""""""""""""""""""""""""""

::

    list_comprehension ::=  expression comp_for
    comp_for           ::=  'for' target_list 'in' or_expr
    list_display       ::=  '[' [expression_list | list_comprehension] ']'
    dict_display       ::=  '{' [key_datum_list | dict_comprehension] '}'
    key_datum_list     ::=  key_datum (',' key_datum)*
    key_datum          ::=  expression ':' expression
    dict_comprehension ::=  key_datum comp_for

Lists and dicts can be constructed by either listing the container contents explicitly or providing
instructions on how to compute them via a set of looping instructions (i.e. a *comprehension*). A comprehension
is semantically equivalent to using a for loop and appending to an ongoing list the expression of the comprehension.
Comprehensions implicitly create their own scope to make sure the items of the target list do not leak into the
enclosing scope. In the case that container items are explicitly listed, the expressions in the expression list
are evaluated left-to-right. If a key is repeated in a ``dict_display`` that has a ``key_datum_list``, then, the
resultant dictionary uses the value from the rightmost datum in the list that uses the repeated key.

Primaries
^^^^^^^^^

::

    primary ::=  atom | attributeref | subscription | slicing | call


Attribute References
""""""""""""""""""""

::

    attributeref ::=  primary '.' identifier


``primary`` must evaluate to an object of a type that supports attribute references that has an attribute named
``identifier``.

Subscriptions
"""""""""""""

::

    subscription ::=  primary '[' expression_list ']'


The primary must evaluate to an object that supports subscription. If it is a ``List`` , ``Tuple``, or ``str``,
the expression list must evaluate to an integer or slice. If it is a ``Dict``, the expression list must evaluate
to an object of the same type as the key type of the ``Dict``. If the primary is a ``ModuleList``, the expression
list must be an ``integer`` literal. If the primary is a ``ModuleDict``, the expression must be a ``stringliteral``.


Slicings
""""""""
A slicing selects a range of items in a ``str``, ``Tuple``, ``List`` or ``Tensor``. Slicings may be used as
expressions or targets in assignment or ``del`` statements.

::

    slicing      ::=  primary '[' slice_list ']'
    slice_list   ::=  slice_item (',' slice_item)* [',']
    slice_item   ::=  expression | proper_slice
    proper_slice ::=  [expression] ':' [expression] [':' [expression] ]

Slicings with more than one slice item in their slice lists can only be used with primaries that evaluate to an
object of type ``Tensor``.


Calls
"""""

::

    call          ::=  primary '(' argument_list ')'
    argument_list ::=  args [',' kwargs] | kwargs
    args          ::=  [arg (',' arg)*]
    kwargs        ::=  [kwarg (',' kwarg)*]
    kwarg         ::=  arg '=' expression
    arg           ::=  identifier


`primary` must desugar or evaluate to a callable object. All argument expressions are evaluated
before the call is attempted.

Power Operator
^^^^^^^^^^^^^^

::

    power ::=  primary ['**' u_expr]


The power operator has the same semantics as the built-in pow function (not supported); it computes its
left argument raised to the power of its right argument. It binds more tightly than unary operators on the
left, but less tightly than unary operators on the right; i.e. ``-2 ** -3 == -(2 ** (-3))``.  The left and right
operands can be ``int``, ``float`` or ``Tensor``. Scalars are broadcast in the case of scalar-tensor/tensor-scalar
exponentiation operations, and tensor-tensor exponentiation is done elementwise without any broadcasting.

Unary and Arithmetic Bitwise Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    u_expr ::=  power | '-' power | '~' power

The unary ``-`` operator yields the negation of its argument. The unary ``~`` operator yields the bitwise inversion
of its argument. ``-`` can be used with ``int``, ``float``, and ``Tensor`` of ``int`` and ``float``.
``~`` can only be used with ``int`` and ``Tensor`` of ``int``.

Binary Arithmetic Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    m_expr ::=  u_expr | m_expr '*' u_expr | m_expr '@' m_expr | m_expr '//' u_expr | m_expr '/' u_expr | m_expr '%' u_expr
    a_expr ::=  m_expr | a_expr '+' m_expr | a_expr '-' m_expr

The binary arithmetic operators can operate on ``Tensor``, ``int``, and ``float``. For tensor-tensor ops, both arguments must
have the same shape. For scalar-tensor or tensor-scalar ops, the scalar is usually broadcast to the size of the
tensor. Division ops can only accept scalars as their right-hand side argument, and do not support broadcasting.
The ``@ ``operator is for matrix multiplication and only operates on ``Tensor`` arguments. The multiplication operator
(``*``) can be used with a list and integer in order to get a result that is the original list repeated a certain
number of times.

Shifting Operations
^^^^^^^^^^^^^^^^^^^

::

    shift_expr ::=  a_expr | shift_expr ( '<<' | '>>' ) a_expr


These operators accept two ``int`` arguments, two ``Tensor`` arguments, or a ``Tensor`` argument and an ``int`` or
``float`` argument. In all cases, a right shift by ``n`` is defined as floor division by ``pow(2, n)`` and a left shift
by ``n`` is defined as multiplication by ``pow(2, n)``. When both arguments are ``Tensors``, they must have the same
shape. When one is a scalar and the other is a ``Tensor``, the scalar is logically broadcast to match the size of
the ``Tensor``.

Binary Bitwise Operations
^^^^^^^^^^^^^^^^^^^^^^^^^

::

    and_expr ::=  shift_expr | and_expr '&' shift_expr
    xor_expr ::=  and_expr | xor_expr '^' and_expr
    or_expr  ::=  xor_expr | or_expr '|' xor_expr


The ``&`` operator computes the bitwise AND of its arguments, the ``^`` the bitwise XOR and ``|`` the bitwise OR.
Both operands must be ``int`` or ``Tensor``, or the left operand must be ``Tensor`` and the right operand must be
``int``. When both operands are ``Tensor``, they must have the same shape. When the right operand is ``int``, and
the left operand is ``Tensor`` , the right operand is logically broadcast to match the shape of the ``Tensor``.

Comparisons
^^^^^^^^^^^

::

    comparison    ::=  or_expr (comp_operator or_expr)*
    comp_operator ::=  '<' | '>' | '==' | '>=' | '<=' | '!=' | 'is' ['not'] | ['not'] 'in'

A comparison yields a boolean values (``True`` or ``False``) or, if one of the operands is a ``Tensor``, a boolean
``Tensor``. Comparisons can be chained arbitrarily as long as they do not yield boolean ``Tensors`` that have more
than one element. ``a op1 b op2 c ...`` is equivalent to ``a op1 b and b op2 c and ...``.

Value Comparisons
"""""""""""""""""
The operators ``<``, ``>``, ``==``, ``>=``, ``<=``, and ``!=`` compare the values of two objects. The two objects generally need to be of
the same type, unless there is an implicit type conversion available between the objects. User-defined types can
be compared if rich comparison methods ( ``__lt__`` etc.) are defined on them. Built-in type comparison works like
Python:

* numbers are compared mathematically
* strings are compared lexicographically
* lists, tuples, and dicts can be compared only to other lists, tuples, and dicts of the same type and are compared using the comparison operator of corresponding elements

Membership Test Operations
""""""""""""""""""""""""""
The operators ``in`` and ``not in`` test for membership. ``x in s`` evaluates to ``True`` if ``x`` is a member of ``s`` and ``False```` otherwise.
``x not in s`` is equivalent to ``not x in s``. This operator is supported for lists, dicts, and tuples, and can be used with
user-defined types if they implement the ``__contains__`` method.

Identity Comparisons
""""""""""""""""""""
For all types except ``int``, ``double``, ``bool``, and ``torch.device``, operators ``is`` and ``is not`` test for the object’s identity;
``x is y`` is ``True`` if and and only if ``x`` and ``y`` are the same object. For all other types, ``is`` is equivalent to
comparing them using ``==``. ``x is not y`` yields the inverse of ``x is y``.

Boolean Operations
^^^^^^^^^^^^^^^^^^

::

    or_test  ::=  and_test | or_test 'or' and_test
    and_test ::=  not_test | and_test 'and' not_test
    not_test ::=  'bool' '(' or_expr ')' | comparison | 'not' not_test

User-defined objects can customize their conversion to ``bool`` by implementing a ``__bool__`` method. The operator ``not``
yields ``True`` if its operand is false, ``False`` otherwise. The expression ``x`` and ``y`` first evaluates ``x``; if it is ``False``, its
value (``False``) is returned; otherwise, ``y`` is evaluated and its value is returned (``False`` or ``True``). The expression ``x`` or ``y``
first evaluates ``x``; if it is ``True``, its value (``True``) is returned; otherwise, ``y`` is evaluated and its value is returned
(``False`` or ``True``).

Conditional Expressions
^^^^^^^^^^^^^^^^^^^^^^^

::

   conditional_expression ::=  or_expr ['if' or_test 'else' conditional_expression]
    expression            ::=  conditional_expression

The expression ``x if c else y`` first evaluates the condition ``c`` rather than x. If ``c`` is ``True``, ``x`` is
evaluated and its value is returned; otherwise, ``y`` is evaluated and its value is returned. As with if-statements,
``x`` and ``y`` must evaluate to a value of the same type.

Expression Lists
^^^^^^^^^^^^^^^^

::

    expression_list ::=  expression (',' expression)* [',']
    starred_item    ::=  '*' primary

A starred item can only appear on the left-hand side of an assignment statement, e.g. ``a, *b, c = ...``.
