list
----

Operations on :ref:`semicolon-separated lists <CMake Language Lists>`.

Synopsis
^^^^^^^^

.. parsed-literal::

  `Reading`_
    list(`LENGTH`_ <list> <out-var>)
    list(`GET`_ <list> <element index> [<index> ...] <out-var>)
    list(`JOIN`_ <list> <glue> <out-var>)
    list(`SUBLIST`_ <list> <begin> <length> <out-var>)

  `Search`_
    list(`FIND`_ <list> <value> <out-var>)

  `Modification`_
    list(`APPEND`_ <list> [<element>...])
    list(`FILTER`_ <list> {INCLUDE | EXCLUDE} REGEX <regex>)
    list(`INSERT`_ <list> <index> [<element>...])
    list(`POP_BACK`_ <list> [<out-var>...])
    list(`POP_FRONT`_ <list> [<out-var>...])
    list(`PREPEND`_ <list> [<element>...])
    list(`REMOVE_ITEM`_ <list> <value>...)
    list(`REMOVE_AT`_ <list> <index>...)
    list(`REMOVE_DUPLICATES`_ <list>)
    list(`TRANSFORM`_ <list> <ACTION> [...])

  `Ordering`_
    list(`REVERSE`_ <list>)
    list(`SORT`_ <list> [...])

Introduction
^^^^^^^^^^^^

The list subcommands :cref:`APPEND`, :cref:`INSERT`, :cref:`FILTER`,
:cref:`PREPEND`, :cref:`POP_BACK`, :cref:`POP_FRONT`, :cref:`REMOVE_AT`,
:cref:`REMOVE_ITEM`, :cref:`REMOVE_DUPLICATES`, :cref:`REVERSE` and
:cref:`SORT` may create new values for the list within the current CMake
variable scope.  Similar to the :command:`set` command, the ``list`` command
creates new variable values in the current scope, even if the list itself is
actually defined in a parent scope.  To propagate the results of these
operations upwards, use :command:`set` with ``PARENT_SCOPE``,
:command:`set` with ``CACHE INTERNAL``, or some other means of value
propagation.

.. note::

  A list in cmake is a ``;`` separated group of strings.  To create a
  list, the :command:`set` command can be used.  For example,
  ``set(var a b c d e)`` creates a list with ``a;b;c;d;e``, and
  ``set(var "a b c d e")`` creates a string or a list with one item in it.
  (Note that macro arguments are not variables, and therefore cannot be used
  in ``LIST`` commands.)

  Individual elements may not contain an unequal number of ``[`` and ``]``
  characters, and may not end in a backslash (``\``).
  See :ref:`semicolon-separated lists <CMake Language Lists>` for details.

.. note::

  When specifying index values, if ``<element index>`` is 0 or greater, it
  is indexed from the beginning of the list, with 0 representing the
  first list element.  If ``<element index>`` is -1 or lesser, it is indexed
  from the end of the list, with -1 representing the last list element.
  Be careful when counting with negative indices: they do not start from
  0.  -0 is equivalent to 0, the first list element.

Reading
^^^^^^^

.. signature::
  list(LENGTH <list> <output variable>)

  Returns the list's length.

.. signature::
  list(GET <list> <element index> [<element index> ...] <output variable>)

  Returns the list of elements specified by indices from the list.

.. signature:: list(JOIN <list> <glue> <output variable>)

  .. versionadded:: 3.12

  Returns a string joining all list's elements using the glue string.
  To join multiple strings, which are not part of a list,
  use :command:`string(JOIN)`.

.. signature::
  list(SUBLIST <list> <begin> <length> <output variable>)

  .. versionadded:: 3.12

  Returns a sublist of the given list.
  If ``<length>`` is 0, an empty list will be returned.
  If ``<length>`` is -1 or the list is smaller than ``<begin>+<length>`` then
  the remaining elements of the list starting at ``<begin>`` will be returned.

Search
^^^^^^

.. signature::
  list(FIND <list> <value> <output variable>)

  Returns the index of the element specified in the list
  or ``-1`` if it wasn't found.

Modification
^^^^^^^^^^^^

.. signature::
  list(APPEND <list> [<element> ...])

  Appends elements to the list. If no variable named ``<list>`` exists in the
  current scope its value is treated as empty and the elements are appended to
  that empty list.

.. signature::
  list(FILTER <list> <INCLUDE|EXCLUDE> REGEX <regular_expression>)

.. versionadded:: 3.6

Includes or removes items from the list that match the mode's pattern.
In ``REGEX`` mode, items will be matched against the given regular expression.

For more information on regular expressions look under
:ref:`string(REGEX) <Regex Specification>`.

.. signature::
  list(INSERT <list> <element_index> <element> [<element> ...])

  Inserts elements to the list to the specified index. It is an
  error to specify an out-of-range index. Valid indexes are *0* to *N*
  where *N* is the length of the list, inclusive. An empty list
  has length 0. If no variable named ``<list>`` exists in the
  current scope its value is treated as empty and the elements are
  inserted in that empty list.

.. signature::
  list(POP_BACK <list> [<out-var>...])

  .. versionadded:: 3.15

  If no variable name is given, removes exactly one element. Otherwise,
  with *N* variable names provided, assign the last *N* elements' values
  to the given variables and then remove the last *N* values from
  ``<list>``.

.. signature::
  list(POP_FRONT <list> [<out-var>...])

  .. versionadded:: 3.15

  If no variable name is given, removes exactly one element. Otherwise,
  with *N* variable names provided, assign the first *N* elements' values
  to the given variables and then remove the first *N* values from
  ``<list>``.

.. signature::
  list(PREPEND <list> [<element> ...])

  .. versionadded:: 3.15

  Insert elements to the 0th position in the list. If no variable named
  ``<list>`` exists in the current scope its value is treated as empty and
  the elements are prepended to that empty list.

.. signature::
  list(REMOVE_ITEM <list> <value> [<value> ...])

  Removes all instances of the given items from the list.

.. signature::
  list(REMOVE_AT <list> <index> [<index> ...])

  Removes items at given indices from the list.

.. signature::
  list(REMOVE_DUPLICATES <list>)

  Removes duplicated items in the list. The relative order of items
  is preserved, but if duplicates are encountered,
  only the first instance is preserved.

.. signature::
  list(TRANSFORM <list> <ACTION> [<SELECTOR>]
       [OUTPUT_VARIABLE <output variable>])

  .. versionadded:: 3.12

  Transforms the list by applying an ``<ACTION>`` to all or, by specifying a
  ``<SELECTOR>``, to the selected elements of the list, storing the result
  in-place or in the specified output variable.

  .. note::

    The ``TRANSFORM`` sub-command does not change the number of elements in the
    list. If a ``<SELECTOR>`` is specified, only some elements will be changed,
    the other ones will remain the same as before the transformation.

  ``<ACTION>`` specifies the action to apply to the elements of the list.
  The actions have exactly the same semantics as sub-commands of the
  :command:`string` command.  ``<ACTION>`` must be one of the following:

    :command:`APPEND <string(APPEND)>`, :command:`PREPEND <string(PREPEND)>`
      Append, prepend specified value to each element of the list.

      .. signature::
        list(TRANSFORM <list> (APPEND|PREPEND) <value> ...)
        :target: TRANSFORM_APPEND

    :command:`TOLOWER <string(TOLOWER)>`, :command:`TOUPPER <string(TOUPPER)>`
      Convert each element of the list to lower, upper characters.

      .. signature::
        list(TRANSFORM <list> (TOLOWER|TOUPPER) ...)
        :target: TRANSFORM_TOLOWER

    :command:`STRIP <string(STRIP)>`
      Remove leading and trailing spaces from each element of the list.

      .. signature::
        list(TRANSFORM <list> STRIP ...)
        :target: TRANSFORM_STRIP

    :command:`GENEX_STRIP <string(GENEX_STRIP)>`
      Strip any
      :manual:`generator expressions <cmake-generator-expressions(7)>`
      from each element of the list.

      .. signature::
        list(TRANSFORM <list> GENEX_STRIP ...)
        :target: TRANSFORM_GENEX_STRIP

    :command:`REPLACE <string(REGEX REPLACE)>`
      Match the regular expression as many times as possible and substitute
      the replacement expression for the match for each element of the list
      (same semantic as :command:`string(REGEX REPLACE)`).

      .. signature::
        list(TRANSFORM <list> REPLACE <regular_expression>
                                      <replace_expression> ...)
        :target: TRANSFORM_REPLACE

      .. versionchanged:: 4.1
        The ``^`` anchor now matches only at the beginning of the input
        element instead of the beginning of each repeated search.
        See policy :policy:`CMP0186`.

  ``<SELECTOR>`` determines which elements of the list will be transformed.
  Only one type of selector can be specified at a time.
  When given, ``<SELECTOR>`` must be one of the following:

    ``AT``
      Specify a list of indexes.

      .. code-block:: cmake

        list(TRANSFORM <list> <ACTION> AT <index> [<index> ...] ...)

    ``FOR``
      Specify a range with, optionally,
      an increment used to iterate over the range.

      .. code-block:: cmake

        list(TRANSFORM <list> <ACTION> FOR <start> <stop> [<step>] ...)

    ``REGEX``
      Specify a regular expression.
      Only elements matching the regular expression will be transformed.

      .. code-block:: cmake

        list(TRANSFORM <list> <ACTION> REGEX <regular_expression> ...)


Ordering
^^^^^^^^

.. signature::
  list(REVERSE <list>)

  Reverses the contents of the list in-place.

.. signature::
  list(SORT <list> [COMPARE <compare>] [CASE <case>] [ORDER <order>])

  Sorts the list in-place alphabetically.

  .. versionadded:: 3.13
    Added the ``COMPARE``, ``CASE``, and ``ORDER`` options.

  .. versionadded:: 3.18
    Added the ``COMPARE NATURAL`` option.

  Use the ``COMPARE`` keyword to select the comparison method for sorting.
  The ``<compare>`` option should be one of:

    ``STRING``
      Sorts a list of strings alphabetically.
      This is the default behavior if the ``COMPARE`` option is not given.

    ``FILE_BASENAME``
      Sorts a list of pathnames of files by their basenames.

    ``NATURAL``
      Sorts a list of strings using natural order
      (see ``strverscmp(3)`` manual), i.e. such that contiguous digits
      are compared as whole numbers.
      For example: the following list *10.0 1.1 2.1 8.0 2.0 3.1*
      will be sorted as *1.1 2.0 2.1 3.1 8.0 10.0* if the ``NATURAL``
      comparison is selected where it will be sorted as
      *1.1 10.0 2.0 2.1 3.1 8.0* with the ``STRING`` comparison.

  Use the ``CASE`` keyword to select a case sensitive or case insensitive
  sort mode.  The ``<case>`` option should be one of:

    ``SENSITIVE``
      List items are sorted in a case-sensitive manner.
      This is the default behavior if the ``CASE`` option is not given.

    ``INSENSITIVE``
      List items are sorted case insensitively.  The order of
      items which differ only by upper/lowercase is not specified.

  To control the sort order, the ``ORDER`` keyword can be given.
  The ``<order>`` option should be one of:

    ``ASCENDING``
      Sorts the list in ascending order.
      This is the default behavior when the ``ORDER`` option is not given.

    ``DESCENDING``
      Sorts the list in descending order.
