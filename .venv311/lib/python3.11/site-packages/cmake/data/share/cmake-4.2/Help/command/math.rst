math
----

Evaluate a mathematical expression.

.. code-block:: cmake

  math(EXPR <variable> "<expression>" [OUTPUT_FORMAT <format>])

Evaluates a mathematical ``<expression>`` and sets ``<variable>`` to the
resulting value.  The result of the expression must be representable as a
64-bit signed integer. Floating point inputs are invalid e.g. ``1.1 * 10``.
Non-integer results e.g. ``3 / 2`` are truncated.

The mathematical expression must be given as a string (i.e. enclosed in
double quotation marks). An example is ``"5 * (10 + 13)"``.
Supported operators are ``+``, ``-``, ``*``, ``/``, ``%``, ``|``, ``&``,
``^``, ``~``, ``<<``, ``>>``, and ``(...)``; they have the same meaning
as in C code.

.. versionadded:: 3.13
  Hexadecimal numbers are recognized when prefixed with ``0x``, as in C code.

.. versionadded:: 3.13
  The result is formatted according to the option ``OUTPUT_FORMAT``,
  where ``<format>`` is one of

  ``HEXADECIMAL``
    Hexadecimal notation as in C code, i. e. starting with "0x".
  ``DECIMAL``
    Decimal notation. Which is also used if no ``OUTPUT_FORMAT`` option
    is specified.

For example

.. code-block:: cmake

  math(EXPR value "100 * 0xA" OUTPUT_FORMAT DECIMAL)      # value is set to "1000"
  math(EXPR value "100 * 0xA" OUTPUT_FORMAT HEXADECIMAL)  # value is set to "0x3e8"
