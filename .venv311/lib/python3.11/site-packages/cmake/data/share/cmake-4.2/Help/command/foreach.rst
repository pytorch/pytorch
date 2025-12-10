foreach
-------

Evaluate a group of commands for each value in a list.

.. code-block:: cmake

  foreach(<loop_var> <items>)
    <commands>
  endforeach()

where ``<items>`` is a list of items that are separated by
semicolon or whitespace.
All commands between ``foreach`` and the matching ``endforeach`` are recorded
without being invoked.  Once the ``endforeach`` is evaluated, the recorded
list of commands is invoked once for each item in ``<items>``.
At the beginning of each iteration the variable ``<loop_var>`` will be set
to the value of the current item.

The scope of ``<loop_var>`` is restricted to the loop scope. See policy
:policy:`CMP0124` for details.

The commands :command:`break` and :command:`continue` provide means to
escape from the normal control flow.

Per legacy, the :command:`endforeach` command admits
an optional ``<loop_var>`` argument.
If used, it must be a verbatim
repeat of the argument of the opening
``foreach`` command.

.. code-block:: cmake

  foreach(<loop_var> RANGE <stop>)

In this variant, ``foreach`` iterates over the numbers
0, 1, ... up to (and including) the nonnegative integer ``<stop>``.

.. code-block:: cmake

  foreach(<loop_var> RANGE <start> <stop> [<step>])

In this variant, ``foreach`` iterates over the numbers from
``<start>`` up to at most ``<stop>`` in steps of ``<step>``.
If ``<step>`` is not specified, then the step size is 1.
The three arguments ``<start>`` ``<stop>`` ``<step>`` must
all be nonnegative integers, and ``<stop>`` must not be
smaller than ``<start>``; otherwise you enter the danger zone
of undocumented behavior that may change in future releases.

.. code-block:: cmake

  foreach(<loop_var> IN [LISTS [<lists>]] [ITEMS [<items>]])

In this variant, ``<lists>`` is a whitespace or semicolon
separated list of list-valued variables. The ``foreach``
command iterates over each item in each given list.
The ``<items>`` following the ``ITEMS`` keyword are processed
as in the first variant of the ``foreach`` command.
The forms ``LISTS A`` and ``ITEMS ${A}`` are
equivalent. If no ``<lists>`` or ``<items>`` are given, the body
of the loop will never be executed (i.e., it is processed as empty).

The following example shows how the ``LISTS`` option is
processed:

.. code-block:: cmake

  set(A 0;1)
  set(B 2 3)
  set(C "4 5")
  set(D 6;7 8)
  set(E "")
  foreach(X IN LISTS A B C D E)
      message(STATUS "X=${X}")
  endforeach()

yields::

  -- X=0
  -- X=1
  -- X=2
  -- X=3
  -- X=4 5
  -- X=6
  -- X=7
  -- X=8


.. code-block:: cmake

  foreach(<loop_var>... IN ZIP_LISTS <lists>)

.. versionadded:: 3.17

In this variant, ``<lists>`` is a whitespace or semicolon
separated list of list-valued variables. The ``foreach``
command iterates over each list simultaneously setting the
iteration variables as follows:

- if a single ``loop_var`` is given, then it sets a series of
  ``loop_var_N`` variables to the current item from the
  corresponding list;
- if multiple variable names are passed, it sets each variable to the
  current item from the corresponding list. The number of iteration
  variables must match the number of list variables.

If no ``<lists>`` are given, the body of the loop will never be executed
(i.e., it is processed as empty).

The following example shows how the ``ZIP_LISTS`` option is
processed:

.. noqa: spellcheck off

.. code-block:: cmake

  list(APPEND English one two three four)
  list(APPEND Bahasa satu dua tiga)

  foreach(num IN ZIP_LISTS English Bahasa)
      message(STATUS "num_0=${num_0}, num_1=${num_1}")
  endforeach()

  foreach(en ba IN ZIP_LISTS English Bahasa)
      message(STATUS "en=${en}, ba=${ba}")
  endforeach()

yields::

  -- num_0=one, num_1=satu
  -- num_0=two, num_1=dua
  -- num_0=three, num_1=tiga
  -- num_0=four, num_1=
  -- en=one, ba=satu
  -- en=two, ba=dua
  -- en=three, ba=tiga
  -- en=four, ba=

.. noqa: spellcheck on

See Also
^^^^^^^^

* :command:`break`
* :command:`continue`
* :command:`endforeach`
* :command:`while`
