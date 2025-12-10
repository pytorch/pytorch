.. cmake-manual-description: CMake Generator Expressions

cmake-generator-expressions(7)
******************************

.. only:: html

   .. contents::

Introduction
============

Generator expressions are evaluated during build system generation to produce
information specific to each build configuration.  They have the form
``$<...>``.  For example:

.. code-block:: cmake

  target_include_directories(tgt PRIVATE /opt/include/$<CXX_COMPILER_ID>)

This would expand to ``/opt/include/GNU``, ``/opt/include/Clang``, etc.
depending on the C++ compiler used.

Generator expressions are allowed in the context of many target properties,
such as :prop_tgt:`LINK_LIBRARIES`, :prop_tgt:`INCLUDE_DIRECTORIES`,
:prop_tgt:`COMPILE_DEFINITIONS` and others.  They may also be used when using
commands to populate those properties, such as :command:`target_link_libraries`,
:command:`target_include_directories`, :command:`target_compile_definitions`
and others.  They enable conditional linking, conditional definitions used when
compiling, conditional include directories, and more.  The conditions may be
based on the build configuration, target properties, platform information,
or any other queryable information.

Generator expressions can be nested:

.. code-block:: cmake

  target_compile_definitions(tgt PRIVATE
    $<$<VERSION_LESS:$<CXX_COMPILER_VERSION>,4.2.0>:OLD_COMPILER>
  )

The above would expand to ``OLD_COMPILER`` if the
:variable:`CMAKE_CXX_COMPILER_VERSION <CMAKE_<LANG>_COMPILER_VERSION>` is less
than 4.2.0.

Whitespace And Quoting
======================

Generator expressions are typically parsed after command arguments.
If a generator expression contains spaces, new lines, semicolons or
other characters that may be interpreted as command argument separators,
the whole expression should be surrounded by quotes when passed to a
command.  Failure to do so may result in the expression being split and
it may no longer be recognized as a generator expression.

When using :command:`add_custom_command` or :command:`add_custom_target`,
use the ``VERBATIM`` and ``COMMAND_EXPAND_LISTS`` options to obtain robust
argument splitting and quoting.

.. code-block:: cmake

  # WRONG: Embedded space will be treated as an argument separator.
  # This ends up not being seen as a generator expression at all.
  add_custom_target(run_some_tool
    COMMAND some_tool -I$<JOIN:$<TARGET_PROPERTY:tgt,INCLUDE_DIRECTORIES>, -I>
    VERBATIM
  )

.. code-block:: cmake

  # Better, but still not robust. Quotes prevent the space from splitting the
  # expression. However, the tool will receive the expanded value as a single
  # argument.
  add_custom_target(run_some_tool
    COMMAND some_tool "-I$<JOIN:$<TARGET_PROPERTY:tgt,INCLUDE_DIRECTORIES>, -I>"
    VERBATIM
  )

.. code-block:: cmake

  # Nearly correct. Using a semicolon to separate arguments and adding the
  # COMMAND_EXPAND_LISTS option means that paths with spaces will be handled
  # correctly. Quoting the whole expression ensures it is seen as a generator
  # expression. But if the target property is empty, we will get a bare -I
  # with nothing after it.
  add_custom_target(run_some_tool
    COMMAND some_tool "-I$<JOIN:$<TARGET_PROPERTY:tgt,INCLUDE_DIRECTORIES>,;-I>"
    COMMAND_EXPAND_LISTS
    VERBATIM
  )

Using variables to build up a more complex generator expression is also a
good way to reduce errors and improve readability.  The above example can be
improved further like so:

.. code-block:: cmake

  # The $<BOOL:...> check prevents adding anything if the property is empty,
  # assuming the property value cannot be one of CMake's false constants.
  set(prop "$<TARGET_PROPERTY:tgt,INCLUDE_DIRECTORIES>")
  add_custom_target(run_some_tool
    COMMAND some_tool "$<$<BOOL:${prop}>:-I$<JOIN:${prop},;-I>>"
    COMMAND_EXPAND_LISTS
    VERBATIM
  )

Finally, the above example can be expressed in a more simple and robust way
using an alternate generator expression:

.. code-block:: cmake

  add_custom_target(run_some_tool
    COMMAND some_tool "$<LIST:TRANSFORM,$<TARGET_PROPERTY:tgt,INCLUDE_DIRECTORIES>,PREPEND,-I>"
    COMMAND_EXPAND_LISTS
    VERBATIM
  )

A common mistake is to try to split a generator expression across multiple
lines with indenting:

.. code-block:: cmake

  # WRONG: New lines and spaces all treated as argument separators, so the
  # generator expression is split and not recognized correctly.
  target_compile_definitions(tgt PRIVATE
    $<$<AND:
        $<CXX_COMPILER_ID:GNU>,
        $<VERSION_GREATER_EQUAL:$<CXX_COMPILER_VERSION>,5>
      >:HAVE_5_OR_LATER>
  )

Again, use helper variables with well-chosen names to build up a readable
expression instead:

.. code-block:: cmake

  set(is_gnu "$<CXX_COMPILER_ID:GNU>")
  set(v5_or_later "$<VERSION_GREATER_EQUAL:$<CXX_COMPILER_VERSION>,5>")
  set(meet_requirements "$<AND:${is_gnu},${v5_or_later}>")
  target_compile_definitions(tgt PRIVATE
    "$<${meet_requirements}:HAVE_5_OR_LATER>"
  )

Debugging
=========

Since generator expressions are evaluated during generation of the buildsystem,
and not during processing of ``CMakeLists.txt`` files, it is not possible to
inspect their result with the :command:`message()` command.  One possible way
to generate debug messages is to add a custom target:

.. code-block:: cmake

  add_custom_target(genexdebug COMMAND ${CMAKE_COMMAND} -E echo "$<...>")

After running :program:`cmake`, you can then build the ``genexdebug`` target to print
the result of the ``$<...>`` expression (i.e. run the command
:option:`cmake --build ... --target genexdebug <cmake--build --target>`).

Another way is to write debug messages to a file with :command:`file(GENERATE)`:

.. code-block:: cmake

  file(GENERATE OUTPUT filename CONTENT "$<...>")

Generator Expression Reference
==============================

.. note::

  This reference deviates from most of the CMake documentation in that it
  omits angular brackets ``<...>`` around placeholders like ``condition``,
  ``string``, ``target``, etc.  This is to prevent an opportunity for those
  placeholders to be misinterpreted as generator expressions.

.. _`Conditional Generator Expressions`:

Conditional Expressions
-----------------------

A fundamental category of generator expressions relates to conditional logic.
Two forms of conditional generator expressions are supported:

.. genex:: $<condition:true_string>

  Evaluates to ``true_string`` if ``condition`` is ``1``, or an empty string
  if ``condition`` evaluates to ``0``.  Any other value for ``condition``
  results in an error.

.. genex:: $<IF:condition,true_string,false_string>

  .. versionadded:: 3.8

  Evaluates to ``true_string`` if ``condition`` is ``1``, or ``false_string``
  if ``condition`` is ``0``.  Any other value for ``condition`` results in an
  error.

  .. versionadded:: 3.28

    This generator expression short-circuits such that generator expressions in
    ``false_string`` will not evaluate when ``condition`` is ``1``, and generator
    expressions in ``true_string`` will not evaluate when condition is ``0``.

Typically, the ``condition`` is itself a generator expression.  For instance,
the following expression expands to ``DEBUG_MODE`` when the ``Debug``
configuration is used, and the empty string for all other configurations:

.. code-block:: cmake

  $<$<CONFIG:Debug>:DEBUG_MODE>

Boolean-like ``condition`` values other than ``1`` or ``0`` can be handled
by wrapping them with the ``$<BOOL:...>`` generator expression:

.. genex:: $<BOOL:string>

  Converts ``string`` to ``0`` or ``1``. Evaluates to ``0`` if any of the
  following is true:

  * ``string`` is empty,
  * ``string`` is a case-insensitive equal of
    ``0``, ``FALSE``, ``OFF``, ``N``, ``NO``, ``IGNORE``, or ``NOTFOUND``, or
  * ``string`` ends in the suffix ``-NOTFOUND`` (case-sensitive).

  Otherwise evaluates to ``1``.

The ``$<BOOL:...>`` generator expression is often used when a ``condition``
is provided by a CMake variable:

.. code-block:: cmake

  $<$<BOOL:${HAVE_SOME_FEATURE}>:-DENABLE_SOME_FEATURE>


.. _`Boolean Generator Expressions`:

Logical Operators
-----------------

The common boolean logic operators are supported:

.. genex:: $<AND:conditions>

  where ``conditions`` is a comma-separated list of boolean expressions,
  all of which must evaluate to either ``1`` or ``0``.  The whole expression
  evaluates to ``1`` if all conditions are ``1``.  If any condition is ``0``,
  the whole expression evaluates to ``0``.

.. genex:: $<OR:conditions>

  where ``conditions`` is a comma-separated list of boolean expressions.
  all of which must evaluate to either ``1`` or ``0``.  The whole expression
  evaluates to ``1`` if at least one of the ``conditions`` is ``1``.  If all
  ``conditions`` evaluate to ``0``, the whole expression evaluates to ``0``.

.. genex:: $<NOT:condition>

  ``condition`` must be ``0`` or ``1``.  The result of the expression is
  ``0`` if ``condition`` is ``1``, else ``1``.

.. versionadded:: 3.28

  Logical operators short-circuit such that generator expressions in the
  arguments list will not be evaluated once a return value can be determined.

.. _`Comparison Expressions`:

Primary Comparison Expressions
------------------------------

CMake supports a variety of generator expressions that compare things.
This section covers the primary and most widely used comparison types.
Other more specific comparison types are documented in their own separate
sections further below.

String Comparisons
^^^^^^^^^^^^^^^^^^

.. genex:: $<STREQUAL:string1,string2>

  ``1`` if ``string1`` and ``string2`` are equal, else ``0``.
  The comparison is case-sensitive.  For a case-insensitive comparison,
  combine with a :ref:`string transforming generator expression
  <String Transforming Generator Expressions>`.  For example, the following
  evaluates to ``1`` if ``${foo}`` is any of ``BAR``, ``Bar``, ``bar``, etc.

  .. code-block:: cmake

    $<STREQUAL:$<UPPER_CASE:${foo}>,BAR>

.. genex:: $<EQUAL:value1,value2>

  ``1`` if ``value1`` and ``value2`` are numerically equal, else ``0``.

Version Comparisons
^^^^^^^^^^^^^^^^^^^

.. genex:: $<VERSION_LESS:v1,v2>

  ``1`` if ``v1`` is a version less than ``v2``, else ``0``.

.. genex:: $<VERSION_GREATER:v1,v2>

  ``1`` if ``v1`` is a version greater than ``v2``, else ``0``.

.. genex:: $<VERSION_EQUAL:v1,v2>

  ``1`` if ``v1`` is the same version as ``v2``, else ``0``.

.. genex:: $<VERSION_LESS_EQUAL:v1,v2>

  .. versionadded:: 3.7

  ``1`` if ``v1`` is a version less than or equal to ``v2``, else ``0``.

.. genex:: $<VERSION_GREATER_EQUAL:v1,v2>

  .. versionadded:: 3.7

  ``1`` if ``v1`` is a version greater than or equal to ``v2``, else ``0``.

.. _`String Transforming Generator Expressions`:

String Transformations
----------------------

.. genex:: $<LOWER_CASE:string>

  Content of ``string`` converted to lower case.

.. genex:: $<UPPER_CASE:string>

  Content of ``string`` converted to upper case.

.. genex:: $<MAKE_C_IDENTIFIER:...>

  Content of ``...`` converted to a C identifier.  The conversion follows the
  same behavior as :command:`string(MAKE_C_IDENTIFIER)`.

List Expressions
----------------

Most of the expressions in this section are closely associated with the
:command:`list` command, providing the same capabilities, but in
the form of a generator expression.

In each of the following list-related generator expressions, the ``list``
must not contain any commas if that generator expression expects something to
be provided after the ``list``.  For example, the expression
``$<LIST:FIND,list,value>`` requires a ``value`` after the ``list``.
Since a comma is used to separate the ``list`` and the ``value``, the ``list``
cannot itself contain a comma.  This restriction does not apply to the
:command:`list` command, it is specific to the list-handling generator
expressions only.

.. _GenEx List Comparisons:

List Comparisons
^^^^^^^^^^^^^^^^

.. genex:: $<IN_LIST:string,list>

  .. versionadded:: 3.12

  ``1`` if ``string`` is an item in the semicolon-separated ``list``, else ``0``.
  It uses case-sensitive comparisons.

.. _GenEx List Queries:

List Queries
^^^^^^^^^^^^

.. genex:: $<LIST:LENGTH,list>

  .. versionadded:: 3.27

  The number of items in the ``list``.

.. genex:: $<LIST:GET,list,index,...>

  .. versionadded:: 3.27

  Expands to the list of items specified by indices from the ``list``.

.. genex:: $<LIST:SUBLIST,list,begin,length>

  .. versionadded:: 3.27

  A sublist of the given ``list``.  If ``length`` is 0, an empty list
  will be returned.  If ``length`` is -1 or the list is smaller than
  ``begin + length``, the remaining items of the list starting at
  ``begin`` will be returned.

.. genex:: $<LIST:FIND,list,value>

  .. versionadded:: 3.27

  The index of the first item in ``list`` with the specified ``value``,
  or -1 if ``value`` is not in the ``list``.

.. _GenEx List Transformations:

List Transformations
^^^^^^^^^^^^^^^^^^^^

.. _GenEx LIST-JOIN:

.. genex:: $<LIST:JOIN,list,glue>

  .. versionadded:: 3.27

  Converts ``list`` to a single string with the content of the ``glue`` string
  inserted between each item.  This is conceptually the same operation as
  :genex:`$<JOIN:list,glue>`, but the two have different behavior with regard
  to empty items.  ``$<LIST:JOIN,list,glue>`` preserves all empty items,
  whereas :genex:`$<JOIN:list,glue>` drops all empty items from the list.

.. genex:: $<LIST:APPEND,list,item,...>

  .. versionadded:: 3.27

  The ``list`` with each ``item`` appended.  Multiple items should be
  separated by commas.

.. genex:: $<LIST:PREPEND,list,item,...>

  .. versionadded:: 3.27

  The ``list`` with each ``item`` inserted at the beginning.  If there are
  multiple items, they should be separated by commas, and the order of the
  prepended items will be preserved.

.. genex:: $<LIST:INSERT,list,index,item,...>

  .. versionadded:: 3.27

  The ``list`` with the ``item`` (or multiple items) inserted at the specified
  ``index``.  Multiple items should be separated by commas.

  It is an error to specify an out-of-range ``index``. Valid indexes are 0 to N,
  where N is the length of the list, inclusive. An empty list has length 0.

.. genex:: $<LIST:POP_BACK,list>

  .. versionadded:: 3.27

  The ``list`` with the last item removed.

.. genex:: $<LIST:POP_FRONT,list>

  .. versionadded:: 3.27

  The ``list`` with the first item removed.

.. genex:: $<LIST:REMOVE_ITEM,list,value,...>

  .. versionadded:: 3.27

  The ``list`` with all instances of the given ``value`` (or values) removed.
  If multiple values are given, they should be separated by commas.

.. genex:: $<LIST:REMOVE_AT,list,index,...>

  .. versionadded:: 3.27

  The ``list`` with the item at each given ``index`` removed.

.. _GenEx LIST-REMOVE_DUPLICATES:

.. genex:: $<LIST:REMOVE_DUPLICATES,list>

  .. versionadded:: 3.27

  The ``list`` with all duplicated items removed.  The relative order of
  items is preserved, but if duplicates are encountered, only the first
  instance is preserved.  The result is the same as
  :genex:`$<REMOVE_DUPLICATES:list>`.

.. _GenEx LIST-FILTER:

.. genex:: $<LIST:FILTER,list,INCLUDE|EXCLUDE,regex>

  .. versionadded:: 3.27

  A list of items from the ``list`` which match (``INCLUDE``) or do not match
  (``EXCLUDE``) the regular expression ``regex``.  The result is the same as
  :genex:`$<FILTER:list,INCLUDE|EXCLUDE,regex>`.

.. genex:: $<LIST:TRANSFORM,list,ACTION[,SELECTOR]>

  .. versionadded:: 3.27

  The ``list`` transformed by applying an ``ACTION`` to all or, by
  specifying a ``SELECTOR``, to the selected list items.

  .. note::

    The ``TRANSFORM`` sub-command does not change the number of items in the
    list. If a ``SELECTOR`` is specified, only some items will be changed,
    the other ones will remain the same as before the transformation.

  ``ACTION`` specifies the action to apply to the items of the list.
  The actions have exactly the same semantics as for the
  :command:`list(TRANSFORM)` command.  ``ACTION`` must be one of the following:

    :command:`APPEND <list(TRANSFORM_APPEND)>`, :command:`PREPEND <list(TRANSFORM_APPEND)>`
      Append, prepend specified value to each item of the list.

      .. code-block:: cmake

        $<LIST:TRANSFORM,list,(APPEND|PREPEND),value[,SELECTOR]>

    :command:`TOLOWER <list(TRANSFORM_TOLOWER)>`, :command:`TOUPPER <list(TRANSFORM_TOLOWER)>`
      Convert each item of the list to lower, upper characters.

      .. code-block:: cmake

        $<LIST:TRANSFORM,list,(TOLOWER|TOUPPER)[,SELECTOR]>

    :command:`STRIP <list(TRANSFORM_STRIP)>`
      Remove leading and trailing spaces from each item of the list.

      .. code-block:: cmake

        $<LIST:TRANSFORM,list,STRIP[,SELECTOR]>

    :command:`REPLACE <list(TRANSFORM_REPLACE)>`:
      Match the regular expression as many times as possible and substitute
      the replacement expression for the match for each item of the list.

      .. code-block:: cmake

        $<LIST:TRANSFORM,list,REPLACE,regular_expression,replace_expression[,SELECTOR]>

      .. versionchanged:: 4.1
        The ``^`` anchor now matches only at the beginning of the input
        element instead of the beginning of each repeated search.
        See policy :policy:`CMP0186`.

  ``SELECTOR`` determines which items of the list will be transformed.
  Only one type of selector can be specified at a time. When given,
  ``SELECTOR`` must be one of the following:

    ``AT``
      Specify a list of indexes.

      .. code-block:: cmake

        $<LIST:TRANSFORM,list,ACTION,AT,index[,index...]>

    ``FOR``
      Specify a range with, optionally, an increment used to iterate over the
      range.

      .. code-block:: cmake

        $<LIST:TRANSFORM,list,ACTION,FOR,start,stop[,step]>

    ``REGEX``
      Specify a regular expression.
      Only items matching the regular expression will be transformed.

      .. code-block:: cmake

        $<LIST:TRANSFORM,list,ACTION,REGEX,regular_expression>

.. genex:: $<JOIN:list,glue>

  Joins the ``list`` with the content of the ``glue`` string inserted between
  each item.  This is conceptually the same operation as
  :ref:`$\<LIST:JOIN,list,glue\> <GenEx LIST-JOIN>`, but the two have
  different behavior with regard to empty items.
  :ref:`$\<LIST:JOIN,list,glue\> <GenEx LIST-JOIN>` preserves all empty items,
  whereas ``$<JOIN:list,glue>`` drops all empty items from the list.

.. genex:: $<REMOVE_DUPLICATES:list>

  .. versionadded:: 3.15

  Removes duplicated items in the given ``list``. The relative order of items
  is preserved, and if duplicates are encountered, only the first instance is
  retained.  The result is the same as
  :ref:`$\<LIST:REMOVE_DUPLICATES,list\> <GenEx LIST-REMOVE_DUPLICATES>`.

.. genex:: $<FILTER:list,INCLUDE|EXCLUDE,regex>

  .. versionadded:: 3.15

  Includes or removes items from ``list`` that match the regular expression
  ``regex``.  The result is the same as
  :ref:`$\<LIST:FILTER,list,INCLUDE|EXCLUDE,regex\> <GenEx LIST-FILTER>`.

.. _GenEx List Ordering:

List Ordering
^^^^^^^^^^^^^

.. genex:: $<LIST:REVERSE,list>

  .. versionadded:: 3.27

  The ``list`` with the items in reverse order.

.. genex:: $<LIST:SORT,list[,(COMPARE:option|CASE:option|ORDER:option)]...>

  .. versionadded:: 3.27

  The ``list`` sorted according to the specified options.

  Use one of the ``COMPARE`` options to select the comparison method
  for sorting:

    ``STRING``
      Sorts a list of strings alphabetically.
      This is the default behavior if the ``COMPARE`` option is not given.

    ``FILE_BASENAME``
      Sorts a list of file paths by their basenames.

    ``NATURAL``
      Sorts a list of strings using natural order (see the man page for
      ``strverscmp(3)``), such that contiguous digits are compared as whole
      numbers.  For example, the following list ``10.0 1.1 2.1 8.0 2.0 3.1``
      will be sorted as ``1.1 2.0 2.1 3.1 8.0 10.0`` if the ``NATURAL``
      comparison is selected, whereas it will be sorted as
      ``1.1 10.0 2.0 2.1 3.1 8.0`` with the ``STRING`` comparison.

  Use one of the ``CASE`` options to select a case-sensitive or
  case-insensitive sort mode:

    ``SENSITIVE``
      List items are sorted in a case-sensitive manner.
      This is the default behavior if the ``CASE`` option is not given.

    ``INSENSITIVE``
      List items are sorted in a case-insensitive manner.  The order of
      items which differ only by upper/lowercase is not specified.

  To control the sort order, one of the ``ORDER`` options can be given:

    ``ASCENDING``
      Sorts the list in ascending order.
      This is the default behavior when the ``ORDER`` option is not given.

    ``DESCENDING``
      Sorts the list in descending order.

  Options can be specified in any order, but it is an error to specify the
  same option multiple times.

  .. code-block:: cmake

    $<LIST:SORT,list,CASE:SENSITIVE,COMPARE:STRING,ORDER:DESCENDING>

Path Expressions
----------------

Most of the expressions in this section are closely associated with the
:command:`cmake_path` command, providing the same capabilities, but in
the form of a generator expression.

For all generator expressions in this section, paths are expected to be in
cmake-style format. The :ref:`$\<PATH:CMAKE_PATH\> <GenEx PATH-CMAKE_PATH>`
generator expression can be used to convert a native path to a cmake-style
one.

.. _GenEx Path Comparisons:

Path Comparisons
^^^^^^^^^^^^^^^^

.. genex:: $<PATH_EQUAL:path1,path2>

  .. versionadded:: 3.24

  Compares the lexical representations of two paths. No normalization is
  performed on either path. Returns ``1`` if the paths are equal, ``0``
  otherwise.

  See :ref:`cmake_path(COMPARE) <Path Comparison>` for more details.

.. _GenEx Path Queries:

Path Queries
^^^^^^^^^^^^

These expressions provide the generation-time capabilities equivalent to the
:ref:`Query <Path Query>` options of the :command:`cmake_path` command.
All paths are expected to be in cmake-style format.

.. genex:: $<PATH:HAS_*,path>

  .. versionadded:: 3.24

  The following operations return ``1`` if the particular path component is
  present, ``0`` otherwise. See :ref:`Path Structure And Terminology` for the
  meaning of each path component.

  .. code-block:: cmake

    $<PATH:HAS_ROOT_NAME,path>
    $<PATH:HAS_ROOT_DIRECTORY,path>
    $<PATH:HAS_ROOT_PATH,path>
    $<PATH:HAS_FILENAME,path>
    $<PATH:HAS_EXTENSION,path>
    $<PATH:HAS_STEM,path>
    $<PATH:HAS_RELATIVE_PART,path>
    $<PATH:HAS_PARENT_PATH,path>

  Note the following special cases:

  * For ``HAS_ROOT_PATH``, a true result will only be returned if at least one
    of ``root-name`` or ``root-directory`` is non-empty.

  * For ``HAS_PARENT_PATH``, the root directory is also considered to have a
    parent, which will be itself.  The result is true except if the path
    consists of just a :ref:`filename <FILENAME_DEF>`.

.. genex:: $<PATH:IS_ABSOLUTE,path>

  .. versionadded:: 3.24

  Returns ``1`` if the path is absolute according to
  :command:`cmake_path(IS_ABSOLUTE)`, ``0`` otherwise.

.. genex:: $<PATH:IS_RELATIVE,path>

  .. versionadded:: 3.24

  Returns ``1`` if the path is relative according to
  :command:`cmake_path(IS_RELATIVE)`, ``0`` otherwise.

.. genex:: $<PATH:IS_PREFIX[,NORMALIZE],path,input>

  .. versionadded:: 3.24

  Returns ``1`` if ``path`` is the prefix of ``input``, ``0`` otherwise.

  When the ``NORMALIZE`` option is specified, ``path`` and ``input`` are
  :ref:`normalized <Normalization>` before the check.

.. _GenEx Path Decomposition:

Path Decomposition
^^^^^^^^^^^^^^^^^^

These expressions provide the generation-time capabilities equivalent to the
:ref:`Decomposition <Path Decomposition>` options of the :command:`cmake_path`
command.  All paths are expected to be in cmake-style format.

.. genex:: $<PATH:GET_*,...>

  .. versionadded:: 3.24

  The following operations retrieve a different component or group of
  components from a path. See :ref:`Path Structure And Terminology` for the
  meaning of each path component.

  .. versionchanged:: 3.27
    All operations now accept a list of paths as argument. When a list of paths
    is specified, the operation will be applied to each path.

  .. code-block:: cmake

    $<PATH:GET_ROOT_NAME,path...>
    $<PATH:GET_ROOT_DIRECTORY,path...>
    $<PATH:GET_ROOT_PATH,path...>
    $<PATH:GET_FILENAME,path...>
    $<PATH:GET_EXTENSION[,LAST_ONLY],path...>
    $<PATH:GET_STEM[,LAST_ONLY],path...>
    $<PATH:GET_RELATIVE_PART,path...>
    $<PATH:GET_PARENT_PATH,path...>

  If a requested component is not present in the path, an empty string is
  returned.

.. _GenEx Path Transformations:

Path Transformations
^^^^^^^^^^^^^^^^^^^^

These expressions provide the generation-time capabilities equivalent to the
:ref:`Modification <Path Modification>` and :ref:`Generation <Path Generation>`
options of the :command:`cmake_path` command.  All paths are expected to be
in cmake-style format.

.. versionchanged:: 3.27
  All operations now accept a list of paths as argument. When a list of paths
  is specified, the operation will be applied to each path.


.. _GenEx PATH-CMAKE_PATH:

.. genex:: $<PATH:CMAKE_PATH[,NORMALIZE],path...>

  .. versionadded:: 3.24

  Returns ``path``. If ``path`` is a native path, it is converted into a
  cmake-style path with forward-slashes (``/``). On Windows, the long filename
  marker is taken into account.

  When the ``NORMALIZE`` option is specified, the path is :ref:`normalized
  <Normalization>` after the conversion.

.. genex:: $<PATH:NATIVE_PATH[,NORMALIZE],path...>

  .. versionadded:: 4.0

  Returns ``path`` converted into a native format with platform-specific
  slashes (``\`` on Windows hosts and ``/`` elsewhere).

  When the ``NORMALIZE`` option is specified, the path is :ref:`normalized
  <Normalization>` before the conversion.

.. genex:: $<PATH:APPEND,path...,input,...>

  .. versionadded:: 3.24

  Returns all the ``input`` arguments appended to ``path`` using ``/`` as the
  ``directory-separator``. Depending on the ``input``, the value of ``path``
  may be discarded.

  See :command:`cmake_path(APPEND)` for more details.

.. genex:: $<PATH:REMOVE_FILENAME,path...>

  .. versionadded:: 3.24

  Returns ``path`` with filename component (as returned by
  ``$<PATH:GET_FILENAME>``) removed. After removal, any trailing
  ``directory-separator`` is left alone, if present.

  See :command:`cmake_path(REMOVE_FILENAME)` for more details.

.. genex:: $<PATH:REPLACE_FILENAME,path...,input>

  .. versionadded:: 3.24

  Returns ``path`` with the filename component replaced by ``input``. If
  ``path`` has no filename component (i.e. ``$<PATH:HAS_FILENAME>`` returns
  ``0``), ``path`` is unchanged.

  See :command:`cmake_path(REPLACE_FILENAME)` for more details.

.. genex:: $<PATH:REMOVE_EXTENSION[,LAST_ONLY],path...>

  .. versionadded:: 3.24

  Returns ``path`` with the :ref:`extension <EXTENSION_DEF>` removed, if any.

  See :command:`cmake_path(REMOVE_EXTENSION)` for more details.

.. genex:: $<PATH:REPLACE_EXTENSION[,LAST_ONLY],path...,input>

  .. versionadded:: 3.24

  Returns ``path`` with the :ref:`extension <EXTENSION_DEF>` replaced by
  ``input``, if any.

  See :command:`cmake_path(REPLACE_EXTENSION)` for more details.

.. genex:: $<PATH:NORMAL_PATH,path...>

  .. versionadded:: 3.24

  Returns ``path`` normalized according to the steps described in
  :ref:`Normalization`.

.. genex:: $<PATH:RELATIVE_PATH,path...,base_directory>

  .. versionadded:: 3.24

  Returns ``path``, modified to make it relative to the ``base_directory``
  argument.

  See :command:`cmake_path(RELATIVE_PATH)` for more details.

.. genex:: $<PATH:ABSOLUTE_PATH[,NORMALIZE],path...,base_directory>

  .. versionadded:: 3.24

  Returns ``path`` as absolute. If ``path`` is a relative path
  (``$<PATH:IS_RELATIVE>`` returns ``1``), it is evaluated relative to the
  given base directory specified by ``base_directory`` argument.

  When the ``NORMALIZE`` option is specified, the path is
  :ref:`normalized <Normalization>` after the path computation.

  See :command:`cmake_path(ABSOLUTE_PATH)` for more details.

Shell Paths
^^^^^^^^^^^

.. genex:: $<SHELL_PATH:...>

  .. versionadded:: 3.4

  Content of ``...`` converted to shell path style. For example, slashes are
  converted to backslashes in Windows shells and drive letters are converted
  to posix paths in MSYS shells. The ``...`` must be an absolute path.

  .. versionadded:: 3.14
    The ``...`` may be a :ref:`semicolon-separated list <CMake Language Lists>`
    of paths, in which case each path is converted individually and a result
    list is generated using the shell path separator (``:`` on POSIX and
    ``;`` on Windows).  Be sure to enclose the argument containing this genex
    in double quotes in CMake source code so that ``;`` does not split arguments.

Configuration Expressions
-------------------------

.. genex:: $<CONFIG>

  Configuration name. Use this instead of the deprecated :genex:`CONFIGURATION`
  generator expression.

.. genex:: $<CONFIG:cfgs>

  ``1`` if config is any one of the entries in comma-separated list
  ``cfgs``, else ``0``. This is a case-insensitive comparison. The mapping in
  :prop_tgt:`MAP_IMPORTED_CONFIG_<CONFIG>` is also considered by this
  expression when it is evaluated on a property of an :prop_tgt:`IMPORTED`
  target.

  .. versionchanged:: 3.19
    Multiple configurations can be specified for ``cfgs``.
    CMake 3.18 and earlier only accepted a single configuration.

.. genex:: $<OUTPUT_CONFIG:...>

  .. versionadded:: 3.20

  Only valid in :command:`add_custom_command` and :command:`add_custom_target`
  as the outer-most generator expression in an argument.
  With the :generator:`Ninja Multi-Config` generator, generator expressions
  in ``...`` are evaluated using the custom command's "output config".
  With other generators, the content of ``...`` is evaluated normally.

.. genex:: $<COMMAND_CONFIG:...>

  .. versionadded:: 3.20

  Only valid in :command:`add_custom_command` and :command:`add_custom_target`
  as the outer-most generator expression in an argument.
  With the :generator:`Ninja Multi-Config` generator, generator expressions
  in ``...`` are evaluated using the custom command's "command config".
  With other generators, the content of ``...`` is evaluated normally.

Toolchain And Language Expressions
----------------------------------

Platform
^^^^^^^^

.. genex:: $<PLATFORM_ID>

  The current system's CMake platform id.
  See also the :variable:`CMAKE_SYSTEM_NAME` variable.

.. genex:: $<PLATFORM_ID:platform_ids>

  ``1`` if CMake's platform id matches any one of the entries in
  comma-separated list ``platform_ids``, otherwise ``0``.
  See also the :variable:`CMAKE_SYSTEM_NAME` variable.

Compiler Version
^^^^^^^^^^^^^^^^

See also the :variable:`CMAKE_<LANG>_COMPILER_VERSION` variable, which is
closely related to the expressions in this sub-section.

.. genex:: $<C_COMPILER_VERSION>

  The version of the C compiler used.

.. genex:: $<C_COMPILER_VERSION:version>

  ``1`` if the version of the C compiler matches ``version``, otherwise ``0``.

.. genex:: $<CXX_COMPILER_VERSION>

  The version of the CXX compiler used.

.. genex:: $<CXX_COMPILER_VERSION:version>

  ``1`` if the version of the C++ compiler matches ``version``, otherwise ``0``.

.. genex:: $<CUDA_COMPILER_VERSION>

  .. versionadded:: 3.15

  The version of the CUDA compiler used.

.. genex:: $<CUDA_COMPILER_VERSION:version>

  .. versionadded:: 3.15

  ``1`` if the version of the C++ compiler matches ``version``, otherwise ``0``.

.. genex:: $<OBJC_COMPILER_VERSION>

  .. versionadded:: 3.16

  The version of the Objective-C compiler used.

.. genex:: $<OBJC_COMPILER_VERSION:version>

  .. versionadded:: 3.16

  ``1`` if the version of the Objective-C compiler matches ``version``, otherwise ``0``.

.. genex:: $<OBJCXX_COMPILER_VERSION>

  .. versionadded:: 3.16

  The version of the Objective-C++ compiler used.

.. genex:: $<OBJCXX_COMPILER_VERSION:version>

  .. versionadded:: 3.16

  ``1`` if the version of the Objective-C++ compiler matches ``version``, otherwise ``0``.

.. genex:: $<Fortran_COMPILER_VERSION>

  The version of the Fortran compiler used.

.. genex:: $<Fortran_COMPILER_VERSION:version>

  ``1`` if the version of the Fortran compiler matches ``version``, otherwise ``0``.

.. genex:: $<HIP_COMPILER_VERSION>

  .. versionadded:: 3.21

  The version of the HIP compiler used.

.. genex:: $<HIP_COMPILER_VERSION:version>

  .. versionadded:: 3.21

  ``1`` if the version of the HIP compiler matches ``version``, otherwise ``0``.

.. genex:: $<ISPC_COMPILER_VERSION>

  .. versionadded:: 3.19

  The version of the ISPC compiler used.

.. genex:: $<ISPC_COMPILER_VERSION:version>

  .. versionadded:: 3.19

  ``1`` if the version of the ISPC compiler matches ``version``, otherwise ``0``.

Compiler Language, ID, and Frontend-Variant
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

See also the :variable:`CMAKE_<LANG>_COMPILER_ID` and
:variable:`CMAKE_<LANG>_COMPILER_FRONTEND_VARIANT` variables, which are closely
related to most of the expressions in this sub-section.

.. genex:: $<C_COMPILER_ID>

  CMake's compiler id of the C compiler used.

.. genex:: $<C_COMPILER_ID:compiler_ids>

  where ``compiler_ids`` is a comma-separated list.
  ``1`` if CMake's compiler id of the C compiler matches any one
  of the entries in ``compiler_ids``, otherwise ``0``.

  .. versionchanged:: 3.15
    Multiple ``compiler_ids`` can be specified.
    CMake 3.14 and earlier only accepted a single compiler ID.

.. genex:: $<CXX_COMPILER_ID>

  CMake's compiler id of the C++ compiler used.

.. genex:: $<CXX_COMPILER_ID:compiler_ids>

  where ``compiler_ids`` is a comma-separated list.
  ``1`` if CMake's compiler id of the C++ compiler matches any one
  of the entries in ``compiler_ids``, otherwise ``0``.

  .. versionchanged:: 3.15
    Multiple ``compiler_ids`` can be specified.
    CMake 3.14 and earlier only accepted a single compiler ID.

.. genex:: $<CUDA_COMPILER_ID>

  .. versionadded:: 3.15

  CMake's compiler id of the CUDA compiler used.

.. genex:: $<CUDA_COMPILER_ID:compiler_ids>

  .. versionadded:: 3.15

  where ``compiler_ids`` is a comma-separated list.
  ``1`` if CMake's compiler id of the CUDA compiler matches any one
  of the entries in ``compiler_ids``, otherwise ``0``.

.. genex:: $<OBJC_COMPILER_ID>

  .. versionadded:: 3.16

  CMake's compiler id of the Objective-C compiler used.

.. genex:: $<OBJC_COMPILER_ID:compiler_ids>

  .. versionadded:: 3.16

  where ``compiler_ids`` is a comma-separated list.
  ``1`` if CMake's compiler id of the Objective-C compiler matches any one
  of the entries in ``compiler_ids``, otherwise ``0``.

.. genex:: $<OBJCXX_COMPILER_ID>

  .. versionadded:: 3.16

  CMake's compiler id of the Objective-C++ compiler used.

.. genex:: $<OBJCXX_COMPILER_ID:compiler_ids>

  .. versionadded:: 3.16

  where ``compiler_ids`` is a comma-separated list.
  ``1`` if CMake's compiler id of the Objective-C++ compiler matches any one
  of the entries in ``compiler_ids``, otherwise ``0``.

.. genex:: $<Fortran_COMPILER_ID>

  CMake's compiler id of the Fortran compiler used.

.. genex:: $<Fortran_COMPILER_ID:compiler_ids>

  where ``compiler_ids`` is a comma-separated list.
  ``1`` if CMake's compiler id of the Fortran compiler matches any one
  of the entries in ``compiler_ids``, otherwise ``0``.

  .. versionchanged:: 3.15
    Multiple ``compiler_ids`` can be specified.
    CMake 3.14 and earlier only accepted a single compiler ID.

.. genex:: $<HIP_COMPILER_ID>

  .. versionadded:: 3.21

  CMake's compiler id of the HIP compiler used.

.. genex:: $<HIP_COMPILER_ID:compiler_ids>

  .. versionadded:: 3.21

  where ``compiler_ids`` is a comma-separated list.
  ``1`` if CMake's compiler id of the HIP compiler matches any one
  of the entries in ``compiler_ids``, otherwise ``0``.

.. genex:: $<ISPC_COMPILER_ID>

  .. versionadded:: 3.19

  CMake's compiler id of the ISPC compiler used.

.. genex:: $<ISPC_COMPILER_ID:compiler_ids>

  .. versionadded:: 3.19

  where ``compiler_ids`` is a comma-separated list.
  ``1`` if CMake's compiler id of the ISPC compiler matches any one
  of the entries in ``compiler_ids``, otherwise ``0``.

.. genex:: $<C_COMPILER_FRONTEND_VARIANT>

  .. versionadded:: 3.30

  CMake's compiler frontend variant of the C compiler used.

.. genex:: $<C_COMPILER_FRONTEND_VARIANT:compiler_ids>

  .. versionadded:: 3.30

  where ``compiler_ids`` is a comma-separated list.
  ``1`` if CMake's compiler frontend variant of the C compiler matches any one
  of the entries in ``compiler_ids``, otherwise ``0``.

.. genex:: $<CXX_COMPILER_FRONTEND_VARIANT>

  .. versionadded:: 3.30

  CMake's compiler frontend variant of the C++ compiler used.

.. genex:: $<CXX_COMPILER_FRONTEND_VARIANT:compiler_ids>

  .. versionadded:: 3.30

  where ``compiler_ids`` is a comma-separated list.
  ``1`` if CMake's compiler frontend variant of the C++ compiler matches any one
  of the entries in ``compiler_ids``, otherwise ``0``.

.. genex:: $<CUDA_COMPILER_FRONTEND_VARIANT>

  .. versionadded:: 3.30

  CMake's compiler id of the CUDA compiler used.

.. genex:: $<CUDA_COMPILER_FRONTEND_VARIANT:compiler_ids>

  .. versionadded:: 3.30

  where ``compiler_ids`` is a comma-separated list.
  ``1`` if CMake's compiler frontend variant of the CUDA compiler matches any one
  of the entries in ``compiler_ids``, otherwise ``0``.

.. genex:: $<OBJC_COMPILER_FRONTEND_VARIANT>

  .. versionadded:: 3.30

  CMake's compiler frontend variant of the Objective-C compiler used.

.. genex:: $<OBJC_COMPILER_FRONTEND_VARIANT:compiler_ids>

  .. versionadded:: 3.30

  where ``compiler_ids`` is a comma-separated list.
  ``1`` if CMake's compiler frontend variant of the Objective-C compiler matches any one
  of the entries in ``compiler_ids``, otherwise ``0``.

.. genex:: $<OBJCXX_COMPILER_FRONTEND_VARIANT>

  .. versionadded:: 3.30

  CMake's compiler frontend variant of the Objective-C++ compiler used.

.. genex:: $<OBJCXX_COMPILER_FRONTEND_VARIANT:compiler_ids>

  .. versionadded:: 3.30

  where ``compiler_ids`` is a comma-separated list.
  ``1`` if CMake's compiler frontend variant of the Objective-C++ compiler matches any one
  of the entries in ``compiler_ids``, otherwise ``0``.

.. genex:: $<Fortran_COMPILER_FRONTEND_VARIANT>

  .. versionadded:: 3.30

  CMake's compiler id of the Fortran compiler used.

.. genex:: $<Fortran_COMPILER_FRONTEND_VARIANT:compiler_ids>

  .. versionadded:: 3.30

  where ``compiler_ids`` is a comma-separated list.
  ``1`` if CMake's compiler frontend variant of the Fortran compiler matches any one
  of the entries in ``compiler_ids``, otherwise ``0``.

.. genex:: $<HIP_COMPILER_FRONTEND_VARIANT>

  .. versionadded:: 3.30

  CMake's compiler id of the HIP compiler used.

.. genex:: $<HIP_COMPILER_FRONTEND_VARIANT:compiler_ids>

  .. versionadded:: 3.30

  where ``compiler_ids`` is a comma-separated list.
  ``1`` if CMake's compiler frontend variant of the HIP compiler matches any one
  of the entries in ``compiler_ids``, otherwise ``0``.

.. genex:: $<ISPC_COMPILER_FRONTEND_VARIANT>

  .. versionadded:: 3.30

  CMake's compiler id of the ISPC compiler used.

.. genex:: $<ISPC_COMPILER_FRONTEND_VARIANT:compiler_ids>

  .. versionadded:: 3.30

  where ``compiler_ids`` is a comma-separated list.
  ``1`` if CMake's compiler frontend variant of the ISPC compiler matches any one
  of the entries in ``compiler_ids``, otherwise ``0``.

.. genex:: $<COMPILE_LANGUAGE>

  .. versionadded:: 3.3

  The compile language of source files when evaluating compile options.
  See the related boolean expression
  :genex:`$<COMPILE_LANGUAGE:languages> <COMPILE_LANGUAGE:languages>`
  for notes about the portability of this generator expression.

.. genex:: $<COMPILE_LANGUAGE:languages>
  :target: COMPILE_LANGUAGE:languages

  .. versionadded:: 3.3

  .. versionchanged:: 3.15
    Multiple languages can be specified for ``languages``.
    CMake 3.14 and earlier only accepted a single language.

  ``1`` when the language used for compilation unit matches any of the
  comma-separated entries in ``languages``, otherwise ``0``. This expression
  may be used to specify compile options, compile definitions, and include
  directories for source files of a particular language in a target. For
  example:

  .. code-block:: cmake

    add_executable(myapp main.cpp foo.c bar.cpp zot.cu)
    target_compile_options(myapp
      PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions>
    )
    target_compile_definitions(myapp
      PRIVATE $<$<COMPILE_LANGUAGE:CXX>:COMPILING_CXX>
              $<$<COMPILE_LANGUAGE:CUDA>:COMPILING_CUDA>
    )
    target_include_directories(myapp
      PRIVATE $<$<COMPILE_LANGUAGE:CXX,CUDA>:/opt/foo/headers>
    )

  This specifies the use of the ``-fno-exceptions`` compile option,
  ``COMPILING_CXX`` compile definition, and ``cxx_headers`` include
  directory for C++ only (compiler id checks elided).  It also specifies
  a ``COMPILING_CUDA`` compile definition for CUDA.

  Note that with :ref:`Visual Studio Generators` and :generator:`Xcode` there
  is no way to represent target-wide compile definitions or include directories
  separately for ``C`` and ``CXX`` languages.
  Also, with :ref:`Visual Studio Generators` there is no way to represent
  target-wide flags separately for ``C`` and ``CXX`` languages.  Under these
  generators, expressions for both C and C++ sources will be evaluated
  using ``CXX`` if there are any C++ sources and otherwise using ``C``.
  A workaround is to create separate libraries for each source file language
  instead:

  .. code-block:: cmake

    add_library(myapp_c foo.c)
    add_library(myapp_cxx bar.cpp)
    target_compile_options(myapp_cxx PUBLIC -fno-exceptions)
    add_executable(myapp main.cpp)
    target_link_libraries(myapp myapp_c myapp_cxx)

.. genex:: $<COMPILE_LANG_AND_ID:language,compiler_ids>

  .. versionadded:: 3.15

  ``1`` when the language used for compilation unit matches ``language`` and
  CMake's compiler id of the ``language`` compiler matches any one of the
  comma-separated entries in ``compiler_ids``, otherwise ``0``. This expression
  is a short form for the combination of ``$<COMPILE_LANGUAGE:language>`` and
  ``$<LANG_COMPILER_ID:compiler_ids>``. This expression may be used to specify
  compile options, compile definitions, and include directories for source
  files of a particular language and compiler combination in a target.
  For example:

  .. code-block:: cmake

    add_executable(myapp main.cpp foo.c bar.cpp zot.cu)
    target_compile_definitions(myapp
      PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,AppleClang,Clang>:COMPILING_CXX_WITH_CLANG>
              $<$<COMPILE_LANG_AND_ID:CXX,Intel>:COMPILING_CXX_WITH_INTEL>
              $<$<COMPILE_LANG_AND_ID:C,Clang>:COMPILING_C_WITH_CLANG>
    )

  This specifies the use of different compile definitions based on both
  the compiler id and compilation language. This example will have a
  ``COMPILING_CXX_WITH_CLANG`` compile definition when Clang is the CXX
  compiler, and ``COMPILING_CXX_WITH_INTEL`` when Intel is the CXX compiler.
  Likewise, when the C compiler is Clang, it will only see the
  ``COMPILING_C_WITH_CLANG`` definition.

  Without the ``COMPILE_LANG_AND_ID`` generator expression, the same logic
  would be expressed as:

  .. code-block:: cmake

    target_compile_definitions(myapp
      PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:AppleClang,Clang>>:COMPILING_CXX_WITH_CLANG>
              $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:Intel>>:COMPILING_CXX_WITH_INTEL>
              $<$<AND:$<COMPILE_LANGUAGE:C>,$<C_COMPILER_ID:Clang>>:COMPILING_C_WITH_CLANG>
    )

Compile Features
^^^^^^^^^^^^^^^^

.. genex:: $<COMPILE_FEATURES:features>

  .. versionadded:: 3.1

  where ``features`` is a comma-separated list.
  Evaluates to ``1`` if all of the ``features`` are available for the 'head'
  target, and ``0`` otherwise. If this expression is used while evaluating
  the link implementation of a target and if any dependency transitively
  increases the required :prop_tgt:`C_STANDARD` or :prop_tgt:`CXX_STANDARD`
  for the 'head' target, an error is reported.  See the
  :manual:`cmake-compile-features(7)` manual for information on
  compile features and a list of supported compilers.

Compile Context
^^^^^^^^^^^^^^^

.. genex:: $<COMPILE_ONLY:...>

  .. versionadded:: 3.27

  Content of ``...``, when collecting
  :ref:`transitive compile properties <Transitive Compile Properties>`,
  otherwise it is the empty string.  This is intended for use in an
  :prop_tgt:`INTERFACE_LINK_LIBRARIES` and :prop_tgt:`LINK_LIBRARIES` target
  properties, typically populated via the :command:`target_link_libraries` command.
  Provides compilation usage requirements without any linking requirements.

  Use cases include header-only usage where all usages are known to not have
  linking requirements (e.g., all-``inline`` or C++ template libraries).

  Note that for proper evaluation of this expression requires policy :policy:`CMP0099`
  to be set to ``NEW``.

Link Language and ID
^^^^^^^^^^^^^^^^^^^^

.. genex:: $<LINK_LANGUAGE>

  .. versionadded:: 3.18

  The link language of the target when evaluating link options.
  See :ref:`the related boolean expression
  <Boolean LINK_LANGUAGE Generator Expression>` ``$<LINK_LANGUAGE:languages>``
  for notes about the portability of this generator expression.

  .. note::

    This generator expression is not supported by the link libraries
    properties to avoid side-effects due to the double evaluation of
    these properties.


.. _`Boolean LINK_LANGUAGE Generator Expression`:

.. genex:: $<LINK_LANGUAGE:languages>

  .. versionadded:: 3.18

  ``1`` when the language used for link step matches any of the comma-separated
  entries in ``languages``, otherwise ``0``.  This expression may be used to
  specify link libraries, link options, link directories and link dependencies
  of a particular language in a target. For example:

  .. code-block:: cmake

    add_library(api_C ...)
    add_library(api_CXX ...)
    add_library(api INTERFACE)
    target_link_options(api   INTERFACE $<$<LINK_LANGUAGE:C>:-opt_c>
                                        $<$<LINK_LANGUAGE:CXX>:-opt_cxx>)
    target_link_libraries(api INTERFACE $<$<LINK_LANGUAGE:C>:api_C>
                                        $<$<LINK_LANGUAGE:CXX>:api_CXX>)

    add_executable(myapp1 main.c)
    target_link_options(myapp1 PRIVATE api)

    add_executable(myapp2 main.cpp)
    target_link_options(myapp2 PRIVATE api)

  This specifies to use the ``api`` target for linking targets ``myapp1`` and
  ``myapp2``. In practice, ``myapp1`` will link with target ``api_C`` and
  option ``-opt_c`` because it will use ``C`` as link language. And ``myapp2``
  will link with ``api_CXX`` and option ``-opt_cxx`` because ``CXX`` will be
  the link language.

  .. _`Constraints LINK_LANGUAGE Generator Expression`:

  .. note::

    To determine the link language of a target, it is required to collect,
    transitively, all the targets which will be linked to it. So, for link
    libraries properties, a double evaluation will be done. During the first
    evaluation, ``$<LINK_LANGUAGE:..>`` expressions will always return ``0``.
    The link language computed after this first pass will be used to do the
    second pass. To avoid inconsistency, it is required that the second pass
    do not change the link language. Moreover, to avoid unexpected
    side-effects, it is required to specify complete entities as part of the
    ``$<LINK_LANGUAGE:..>`` expression. For example:

    .. code-block:: cmake

      add_library(lib STATIC file.cxx)
      add_library(libother STATIC file.c)

      # bad usage
      add_executable(myapp1 main.c)
      target_link_libraries(myapp1 PRIVATE lib$<$<LINK_LANGUAGE:C>:other>)

      # correct usage
      add_executable(myapp2 main.c)
      target_link_libraries(myapp2 PRIVATE $<$<LINK_LANGUAGE:C>:libother>)

    In this example, for ``myapp1``, the first pass will, unexpectedly,
    determine that the link language is ``CXX`` because the evaluation of the
    generator expression will be an empty string so ``myapp1`` will depends on
    target ``lib`` which is ``C++``. On the contrary, for ``myapp2``, the first
    evaluation will give ``C`` as link language, so the second pass will
    correctly add target ``libother`` as link dependency.

.. genex:: $<LINK_LANG_AND_ID:language,compiler_ids>

  .. versionadded:: 3.18

  ``1`` when the language used for link step matches ``language`` and the
  CMake's compiler id of the language linker matches any one of the comma-separated
  entries in ``compiler_ids``, otherwise ``0``. This expression is a short form
  for the combination of ``$<LINK_LANGUAGE:language>`` and
  ``$<LANG_COMPILER_ID:compiler_ids>``. This expression may be used to specify
  link libraries, link options, link directories and link dependencies of a
  particular language and linker combination in a target. For example:

  .. code-block:: cmake

    add_library(libC_Clang ...)
    add_library(libCXX_Clang ...)
    add_library(libC_Intel ...)
    add_library(libCXX_Intel ...)

    add_executable(myapp main.c)
    if (CXX_CONFIG)
      target_sources(myapp PRIVATE file.cxx)
    endif()
    target_link_libraries(myapp
      PRIVATE $<$<LINK_LANG_AND_ID:CXX,Clang,AppleClang>:libCXX_Clang>
              $<$<LINK_LANG_AND_ID:C,Clang,AppleClang>:libC_Clang>
              $<$<LINK_LANG_AND_ID:CXX,Intel>:libCXX_Intel>
              $<$<LINK_LANG_AND_ID:C,Intel>:libC_Intel>)

  This specifies the use of different link libraries based on both the
  compiler id and link language. This example will have target ``libCXX_Clang``
  as link dependency when ``Clang`` or ``AppleClang`` is the ``CXX``
  linker, and ``libCXX_Intel`` when ``Intel`` is the ``CXX`` linker.
  Likewise when the ``C`` linker is ``Clang`` or ``AppleClang``, target
  ``libC_Clang`` will be added as link dependency and ``libC_Intel`` when
  ``Intel`` is the ``C`` linker.

  See :ref:`the note related to
  <Constraints LINK_LANGUAGE Generator Expression>`
  ``$<LINK_LANGUAGE:language>`` for constraints about the usage of this
  generator expression.

Link Features
^^^^^^^^^^^^^

.. genex:: $<LINK_LIBRARY:feature,library-list>

  .. versionadded:: 3.24

  Specify a set of libraries to link to a target, along with a ``feature``
  which provides details about *how* they should be linked.  For example:

  .. code-block:: cmake

    add_library(lib1 STATIC ...)
    add_library(lib2 ...)
    target_link_libraries(lib2 PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,lib1>")

  This specifies that ``lib2`` should link to ``lib1`` and use the
  ``WHOLE_ARCHIVE`` feature when doing so.

  Feature names are case-sensitive and may only contain letters, numbers and
  underscores.  Feature names defined in all uppercase are reserved for CMake's
  own built-in features.  The pre-defined built-in library features are:

  .. include:: ../variable/include/LINK_LIBRARY_PREDEFINED_FEATURES.rst

  Built-in and custom library features are defined in terms of the following
  variables:

  * :variable:`CMAKE_<LANG>_LINK_LIBRARY_USING_<FEATURE>_SUPPORTED`
  * :variable:`CMAKE_<LANG>_LINK_LIBRARY_USING_<FEATURE>`
  * :variable:`CMAKE_LINK_LIBRARY_USING_<FEATURE>_SUPPORTED`
  * :variable:`CMAKE_LINK_LIBRARY_USING_<FEATURE>`

  The value used for each of these variables is the value as set at the end of
  the directory scope in which the target was created.  The usage is as follows:

  1. If the language-specific
     :variable:`CMAKE_<LANG>_LINK_LIBRARY_USING_<FEATURE>_SUPPORTED` variable
     is true, the ``feature`` must be defined by the corresponding
     :variable:`CMAKE_<LANG>_LINK_LIBRARY_USING_<FEATURE>` variable.
  2. If no language-specific ``feature`` is supported, then the
     :variable:`CMAKE_LINK_LIBRARY_USING_<FEATURE>_SUPPORTED` variable must be
     true and the ``feature`` must be defined by the corresponding
     :variable:`CMAKE_LINK_LIBRARY_USING_<FEATURE>` variable.

  The following limitations should be noted:

  * The ``library-list`` can specify CMake targets or libraries.
    Any CMake target of type :ref:`OBJECT <Object Libraries>`
    or :ref:`INTERFACE <Interface Libraries>` will ignore the feature aspect
    of the expression and instead be linked in the standard way.

  * The ``$<LINK_LIBRARY:...>`` generator expression can only be used to
    specify link libraries.  In practice, this means it can appear in the
    :prop_tgt:`LINK_LIBRARIES`, :prop_tgt:`INTERFACE_LINK_LIBRARIES`, and
    :prop_tgt:`INTERFACE_LINK_LIBRARIES_DIRECT`  target properties, and be
    specified in :command:`target_link_libraries` and :command:`link_libraries`
    commands.

  * If a ``$<LINK_LIBRARY:...>`` generator expression appears in the
    :prop_tgt:`INTERFACE_LINK_LIBRARIES` property of a target, it will be
    included in the imported target generated by a :command:`install(EXPORT)`
    command.  It is the responsibility of the environment consuming this
    import to define the link feature used by this expression.

  * Each target or library involved in the link step must have at most only
    one kind of library feature.  The absence of a feature is also incompatible
    with all other features.  For example:

    .. code-block:: cmake

      add_library(lib1 ...)
      add_library(lib2 ...)
      add_library(lib3 ...)

      # lib1 will be associated with feature1
      target_link_libraries(lib2 PUBLIC "$<LINK_LIBRARY:feature1,lib1>")

      # lib1 is being linked with no feature here. This conflicts with the
      # use of feature1 in the line above and would result in an error.
      target_link_libraries(lib3 PRIVATE lib1 lib2)

    Where it isn't possible to use the same feature throughout a build for a
    given target or library, the :prop_tgt:`LINK_LIBRARY_OVERRIDE` and
    :prop_tgt:`LINK_LIBRARY_OVERRIDE_<LIBRARY>` target properties can be
    used to resolve such incompatibilities.

  * The ``$<LINK_LIBRARY:...>`` generator expression does not guarantee
    that the list of specified targets and libraries will be kept grouped
    together.  To manage constructs like ``--start-group`` and ``--end-group``,
    as supported by the GNU ``ld`` linker, use the :genex:`LINK_GROUP`
    generator expression instead.

.. genex:: $<LINK_GROUP:feature,library-list>

  .. versionadded:: 3.24

  Specify a group of libraries to link to a target, along with a ``feature``
  which defines how that group should be linked.  For example:

  .. code-block:: cmake

    add_library(lib1 STATIC ...)
    add_library(lib2 ...)
    target_link_libraries(lib2 PRIVATE "$<LINK_GROUP:RESCAN,lib1,external>")

  This specifies that ``lib2`` should link to ``lib1`` and ``external``, and
  that both of those two libraries should be included on the linker command
  line according to the definition of the ``RESCAN`` feature.

  Feature names are case-sensitive and may only contain letters, numbers and
  underscores.  Feature names defined in all uppercase are reserved for CMake's
  own built-in features.  Currently, there is only one pre-defined built-in
  group feature:

  .. include:: ../variable/include/LINK_GROUP_PREDEFINED_FEATURES.rst

  Built-in and custom group features are defined in terms of the following
  variables:

  * :variable:`CMAKE_<LANG>_LINK_GROUP_USING_<FEATURE>_SUPPORTED`
  * :variable:`CMAKE_<LANG>_LINK_GROUP_USING_<FEATURE>`
  * :variable:`CMAKE_LINK_GROUP_USING_<FEATURE>_SUPPORTED`
  * :variable:`CMAKE_LINK_GROUP_USING_<FEATURE>`

  The value used for each of these variables is the value as set at the end of
  the directory scope in which the target was created.  The usage is as follows:

  1. If the language-specific
     :variable:`CMAKE_<LANG>_LINK_GROUP_USING_<FEATURE>_SUPPORTED` variable
     is true, the ``feature`` must be defined by the corresponding
     :variable:`CMAKE_<LANG>_LINK_GROUP_USING_<FEATURE>` variable.
  2. If no language-specific ``feature`` is supported, then the
     :variable:`CMAKE_LINK_GROUP_USING_<FEATURE>_SUPPORTED` variable must be
     true and the ``feature`` must be defined by the corresponding
     :variable:`CMAKE_LINK_GROUP_USING_<FEATURE>` variable.

  The ``LINK_GROUP`` generator expression is compatible with the
  :genex:`LINK_LIBRARY` generator expression. The libraries involved in a
  group can be specified using the :genex:`LINK_LIBRARY` generator expression.

  Each target or external library involved in the link step is allowed to be
  part of multiple groups, but only if all the groups involved specify the
  same ``feature``.  Such groups will not be merged on the linker command line,
  the individual groups will still be preserved.  Mixing different group
  features for the same target or library is forbidden.

  .. code-block:: cmake

    add_library(lib1 ...)
    add_library(lib2 ...)
    add_library(lib3 ...)
    add_library(lib4 ...)
    add_library(lib5 ...)

    target_link_libraries(lib3 PUBLIC  "$<LINK_GROUP:feature1,lib1,lib2>")
    target_link_libraries(lib4 PRIVATE "$<LINK_GROUP:feature1,lib1,lib3>")
    # lib4 will be linked with the groups {lib1,lib2} and {lib1,lib3}.
    # Both groups specify the same feature, so this is fine.

    target_link_libraries(lib5 PRIVATE "$<LINK_GROUP:feature2,lib1,lib3>")
    # An error will be raised here because both lib1 and lib3 are part of two
    # groups with different features.

  When a target or an external library is involved in the link step as part of
  a group and also as not part of any group, any occurrence of the non-group
  link item will be replaced by the groups it belongs to.

  .. code-block:: cmake

    add_library(lib1 ...)
    add_library(lib2 ...)
    add_library(lib3 ...)
    add_library(lib4 ...)

    target_link_libraries(lib3 PUBLIC lib1)

    target_link_libraries(lib4 PRIVATE lib3 "$<LINK_GROUP:feature1,lib1,lib2>")
    # lib4 will only be linked with lib3 and the group {lib1,lib2}

  Because ``lib1`` is part of the group defined for ``lib4``, that group then
  gets applied back to the use of ``lib1`` for ``lib3``.  The end result will
  be as though the linking relationship for ``lib3`` had been specified as:

  .. code-block:: cmake

    target_link_libraries(lib3 PUBLIC "$<LINK_GROUP:feature1,lib1,lib2>")

  Be aware that the precedence of the group over the non-group link item can
  result in circular dependencies between groups.  If this occurs, a fatal
  error is raised because circular dependencies are not allowed for groups.

  .. code-block:: cmake

    add_library(lib1A ...)
    add_library(lib1B ...)
    add_library(lib2A ...)
    add_library(lib2B ...)
    add_library(lib3 ...)

    # Non-group linking relationships, these are non-circular so far
    target_link_libraries(lib1A PUBLIC lib2A)
    target_link_libraries(lib2B PUBLIC lib1B)

    # The addition of these groups creates circular dependencies
    target_link_libraries(lib3 PRIVATE
      "$<LINK_GROUP:feat,lib1A,lib1B>"
      "$<LINK_GROUP:feat,lib2A,lib2B>"
    )

  Because of the groups defined for ``lib3``, the linking relationships for
  ``lib1A`` and ``lib2B`` effectively get expanded to the equivalent of:

  .. code-block:: cmake

    target_link_libraries(lib1A PUBLIC "$<LINK_GROUP:feat,lib2A,lib2B>")
    target_link_libraries(lib2B PUBLIC "$<LINK_GROUP:feat,lib1A,lib1B>")

  This creates a circular dependency between groups:
  ``lib1A --> lib2B --> lib1A``.

  The following limitations should also be noted:

  * The ``library-list`` can specify CMake targets or libraries.
    Any CMake target of type :ref:`OBJECT <Object Libraries>`
    or :ref:`INTERFACE <Interface Libraries>` will ignore the feature aspect
    of the expression and instead be linked in the standard way.

  * The ``$<LINK_GROUP:...>`` generator expression can only be used to
    specify link libraries.  In practice, this means it can appear in the
    :prop_tgt:`LINK_LIBRARIES`, :prop_tgt:`INTERFACE_LINK_LIBRARIES`,and
    :prop_tgt:`INTERFACE_LINK_LIBRARIES_DIRECT` target properties, and be
    specified in :command:`target_link_libraries` and :command:`link_libraries`
    commands.

  * If a ``$<LINK_GROUP:...>`` generator expression appears in the
    :prop_tgt:`INTERFACE_LINK_LIBRARIES` property of a target, it will be
    included in the imported target generated by a :command:`install(EXPORT)`
    command.  It is the responsibility of the environment consuming this
    import to define the link feature used by this expression.

Link Context
^^^^^^^^^^^^

.. genex:: $<LINK_ONLY:...>

  .. versionadded:: 3.1

  Content of ``...``, except while collecting usage requirements from
  :ref:`transitive compile properties <Transitive Compile Properties>`,
  in which case it is the empty string.  This is intended for use in an
  :prop_tgt:`INTERFACE_LINK_LIBRARIES` target property, typically populated
  via the :command:`target_link_libraries` command, to specify private link
  dependencies without other usage requirements such as include directories or
  compile options.

  .. versionadded:: 3.24
    ``LINK_ONLY`` may also be used in a :prop_tgt:`LINK_LIBRARIES` target
    property.  See policy :policy:`CMP0131`.

.. genex:: $<DEVICE_LINK:list>

  .. versionadded:: 3.18

  Returns the list if it is the device link step, an empty list otherwise.
  The device link step is controlled by :prop_tgt:`CUDA_SEPARABLE_COMPILATION`
  and :prop_tgt:`CUDA_RESOLVE_DEVICE_SYMBOLS` properties and
  policy :policy:`CMP0105`. This expression can only be used to specify link
  options.

.. genex:: $<HOST_LINK:list>

  .. versionadded:: 3.18

  Returns the list if it is the normal link step, an empty list otherwise.
  This expression is mainly useful when a device link step is also involved
  (see :genex:`$<DEVICE_LINK:list>` generator expression). This expression can
  only be used to specify link options.

Linker ID and Frontend-Variant
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

See also the :variable:`CMAKE_<LANG>_COMPILER_LINKER_ID` and
:variable:`CMAKE_<LANG>_COMPILER_LINKER_FRONTEND_VARIANT` variables, which are
closely related to most of the expressions in this sub-section.

.. genex:: $<C_COMPILER_LINKER_ID>

  .. versionadded:: 4.2

  CMake's linker id of the C linker used.

.. genex:: $<C_COMPILER_LINKER_ID:linker_ids>

  .. versionadded:: 4.2

  where ``linker_ids`` is a comma-separated list.
  ``1`` if CMake's linker id of the C linker matches any one
  of the entries in ``linker_ids``, otherwise ``0``.

.. genex:: $<CXX_COMPILER_LINKER_ID>

  .. versionadded:: 4.2

  CMake's linker id of the C++ linker used.

.. genex:: $<CXX_COMPILER_LINKER_ID:linker_ids>

  .. versionadded:: 4.2

  where ``linker_ids`` is a comma-separated list.
  ``1`` if CMake's linker id of the C++ linker matches any one
  of the entries in ``linker_ids``, otherwise ``0``.

.. genex:: $<CUDA_COMPILER_LINKER_ID>

  .. versionadded:: 4.2

  CMake's linker id of the CUDA linker used.

.. genex:: $<CUDA_COMPILER_LINKER_ID:linker_ids>

  .. versionadded:: 4.2

  where ``linker_ids`` is a comma-separated list.
  ``1`` if CMake's linker id of the CUDA linker matches any one
  of the entries in ``linker_ids``, otherwise ``0``.

.. genex:: $<OBJC_COMPILER_LINKER_ID>

  .. versionadded:: 4.2

  CMake's linker id of the Objective-C linker used.

.. genex:: $<OBJC_COMPILER_LINKER_ID:linker_ids>

  .. versionadded:: 4.2

  where ``linker_ids`` is a comma-separated list.
  ``1`` if CMake's linker id of the Objective-C linker matches any one
  of the entries in ``linker_ids``, otherwise ``0``.

.. genex:: $<OBJCXX_COMPILER_LINKER_ID>

  .. versionadded:: 4.2

  CMake's linker id of the Objective-C++ linker used.

.. genex:: $<OBJCXX_COMPILER_LINKER_ID:linker_ids>

  .. versionadded:: 4.2

  where ``linker_ids`` is a comma-separated list.
  ``1`` if CMake's linker id of the Objective-C++ linker matches any one
  of the entries in ``linker_ids``, otherwise ``0``.

.. genex:: $<Fortran_COMPILER_LINKER_ID>

  .. versionadded:: 4.2

  CMake's linker id of the Fortran linker used.

.. genex:: $<Fortran_COMPILER_LINKER_ID:linker_ids>

  .. versionadded:: 4.2

  where ``linker_ids`` is a comma-separated list.
  ``1`` if CMake's linker id of the Fortran linker matches any one
  of the entries in ``linker_ids``, otherwise ``0``.

.. genex:: $<HIP_COMPILER_LINKER_ID>

  .. versionadded:: 4.2

  CMake's linker id of the HIP linker used.

.. genex:: $<HIP_COMPILER_LINKER_ID:linker_ids>

  .. versionadded:: 4.2

  where ``linker_ids`` is a comma-separated list.
  ``1`` if CMake's linker id of the HIP linker matches any one
  of the entries in ``linker_ids``, otherwise ``0``.

.. genex:: $<C_COMPILER_LINKER_FRONTEND_VARIANT>

  .. versionadded:: 4.2

  CMake's linker frontend variant of the C linker used.

.. genex:: $<C_COMPILER_LINKER_FRONTEND_VARIANT:variant_ids>

  .. versionadded:: 4.2

  where ``variant_ids`` is a comma-separated list.
  ``1`` if CMake's linker frontend variant of the C linker matches any one
  of the entries in ``variant_ids``, otherwise ``0``.

.. genex:: $<CXX_COMPILER_LINKER_FRONTEND_VARIANT>

  .. versionadded:: 4.2

  CMake's linker frontend variant of the C++ linker used.

.. genex:: $<CXX_COMPILER_LINKER_FRONTEND_VARIANT:variant_ids>

  .. versionadded:: 4.2

  where ``variant_ids`` is a comma-separated list.
  ``1`` if CMake's linker frontend variant of the C++ linker matches any one
  of the entries in ``variant_ids``, otherwise ``0``.

.. genex:: $<CUDA_COMPILER_LINKER_FRONTEND_VARIANT>

  .. versionadded:: 4.2

  CMake's linker frontend variant of the CUDA linker used.

.. genex:: $<CUDA_COMPILER_LINKER_FRONTEND_VARIANT:variant_ids>

  .. versionadded:: 4.2

  where ``variant_ids`` is a comma-separated list.
  ``1`` if CMake's linker frontend variant of the CUDA linker matches any one
  of the entries in ``variant_ids``, otherwise ``0``.

.. genex:: $<OBJC_COMPILER_LINKER_FRONTEND_VARIANT>

  .. versionadded:: 4.2

  CMake's linker frontend variant of the Objective-C linker used.

.. genex:: $<OBJC_COMPILER_LINKER_FRONTEND_VARIANT:variant_ids>

  .. versionadded:: 4.2

  where ``variant_ids`` is a comma-separated list.
  ``1`` if CMake's linker frontend variant of the Objective-C linker matches
  any one of the entries in ``variant_ids``, otherwise ``0``.

.. genex:: $<OBJCXX_COMPILER_LINKER_FRONTEND_VARIANT>

  .. versionadded:: 4.2

  CMake's linker frontend variant of the Objective-C++ linker used.

.. genex:: $<OBJCXX_COMPILER_LINKER_FRONTEND_VARIANT:variant_ids>

  .. versionadded:: 4.2

  where ``variant_ids`` is a comma-separated list.
  ``1`` if CMake's linker frontend variant of the Objective-C++ linker matches
  any one of the entries in ``variant_ids``, otherwise ``0``.

.. genex:: $<Fortran_COMPILER_LINKER_FRONTEND_VARIANT>

  .. versionadded:: 4.2

  CMake's linker frontend variant of the Fortran linker used.

.. genex:: $<Fortran_COMPILER_LINKER_FRONTEND_VARIANT:variant_ids>

  .. versionadded:: 4.2

  where ``variant_ids`` is a comma-separated list.
  ``1`` if CMake's linker frontend variant of the Fortran linker matches
  any one of the entries in ``variant_ids``, otherwise ``0``.

.. genex:: $<HIP_COMPILER_LINKER_FRONTEND_VARIANT>

  .. versionadded:: 4.2

  CMake's linker frontend variant of the HIP linker used.

.. genex:: $<HIP_COMPILER_LINKER_FRONTEND_VARIANT:variant_ids>

  .. versionadded:: 4.2

  where ``variant_ids`` is a comma-separated list.
  ``1`` if CMake's linker frontend variant of the HIP linker matches
  any one of the entries in ``variant_ids``, otherwise ``0``.


.. _`Target-Dependent Expressions`:

Target-Dependent Expressions
----------------------------

Target Meta-Data
^^^^^^^^^^^^^^^^

These expressions look up information about a target.

.. genex:: $<TARGET_EXISTS:tgt>

  .. versionadded:: 3.12

  ``1`` if ``tgt`` exists as a CMake target, else ``0``.

.. genex:: $<TARGET_NAME_IF_EXISTS:tgt>

  .. versionadded:: 3.12

  The target name ``tgt`` if the target exists, an empty string otherwise.

  Note that ``tgt`` is not added as a dependency of the target this
  expression is evaluated on.

.. genex:: $<TARGET_NAME:tgt>

  The target name ``tgt`` as written.  This marks ``tgt`` as being the name
  of a target inside a larger expression, which is required if exporting
  targets to multiple dependent export sets.  The ``tgt`` text must be a
  literal name of a target; it may not contain generator expressions.
  The target does not have to exist.

.. genex:: $<TARGET_POLICY:policy>

  ``1`` if the ``policy`` was ``NEW`` when the 'head' target was created,
  else ``0``.  If the ``policy`` was not set, the warning message for the policy
  will be emitted. This generator expression only works for a subset of
  policies.


Target Properties
^^^^^^^^^^^^^^^^^

These expressions look up the values of
:ref:`target properties <Target Properties>`.

.. genex:: $<TARGET_PROPERTY:tgt,prop>

  Value of the property ``prop`` on the target ``tgt``, or empty if
  the property is not set.

  Note that ``tgt`` is not added as a dependency of the target this
  expression is evaluated on.

  .. versionchanged:: 3.26
    When encountered during evaluation of :ref:`Target Usage Requirements`,
    typically in an ``INTERFACE_*`` target property, lookup of the ``tgt``
    name occurs in the directory of the target specifying the requirement,
    rather than the directory of the consuming target for which the
    expression is being evaluated.

.. genex:: $<TARGET_PROPERTY:prop>
  :target: TARGET_PROPERTY:prop

  Value of the property ``prop`` on the target for which the expression
  is being evaluated, or empty if the property is not set.
  Note that for generator expressions in :ref:`Target Usage Requirements`
  this is the consuming target rather than the target specifying the
  requirement.

The expressions have special evaluation rules for some properties:

:ref:`Target Build Specification Properties <Target Build Specification>`
  These evaluate as a :ref:`semicolon-separated list <CMake Language Lists>`
  representing the union of the value on the target itself with the values
  of the corresponding :ref:`Target Usage Requirements` on targets named by
  the target's :prop_tgt:`LINK_LIBRARIES`:

  * For :ref:`Target Compile Properties`, evaluation of corresponding usage
    requirements is transitive over the closure of the linked targets'
    :prop_tgt:`INTERFACE_LINK_LIBRARIES` *excluding* entries guarded by the
    :genex:`LINK_ONLY` generator expression.

  * For :ref:`Target Link Properties`, evaluation of corresponding usage
    requirements is transitive over the closure of the linked targets'
    :prop_tgt:`INTERFACE_LINK_LIBRARIES` *including* entries guarded by the
    :genex:`LINK_ONLY` generator expression.  See policy :policy:`CMP0166`.

  .. versionchanged:: 4.1

    Evaluation of :prop_tgt:`LINK_LIBRARIES` itself is now transitive.
    See policy :policy:`CMP0189`.

:ref:`Target Usage Requirement Properties <Target Usage Requirements>`
  These evaluate as a :ref:`semicolon-separated list <CMake Language Lists>`
  representing the union of the value on the target itself with the values
  of the same properties on targets named by the target's
  :prop_tgt:`INTERFACE_LINK_LIBRARIES`:

  * For :ref:`Transitive Compile Properties`, evaluation is transitive over
    the closure of the target's :prop_tgt:`INTERFACE_LINK_LIBRARIES`
    *excluding* entries guarded by the :genex:`LINK_ONLY` generator expression.

  * For :ref:`Transitive Link Properties`, evaluation is transitive over
    the closure of the target's :prop_tgt:`INTERFACE_LINK_LIBRARIES`
    *including* entries guarded by the :genex:`LINK_ONLY` generator expression.
    See policy :policy:`CMP0166`.

  .. versionchanged:: 4.1

    Evaluation of :prop_tgt:`INTERFACE_LINK_LIBRARIES` itself is now
    transitive.  See policy :policy:`CMP0189`.

:ref:`Custom Transitive Properties`
  .. versionadded:: 3.30

  These are processed during evaluation as follows:

  * Evaluation of :genex:`$<TARGET_PROPERTY:tgt,PROP>` for some property
    ``PROP``, named without an ``INTERFACE_`` prefix,
    checks the :prop_tgt:`TRANSITIVE_COMPILE_PROPERTIES`
    and :prop_tgt:`TRANSITIVE_LINK_PROPERTIES` properties on target ``tgt``,
    on targets named by its :prop_tgt:`LINK_LIBRARIES`, and on the
    transitive closure of targets named by the linked targets'
    :prop_tgt:`INTERFACE_LINK_LIBRARIES`.

    If ``PROP`` is listed by one of those properties, then it evaluates as
    a :ref:`semicolon-separated list <CMake Language Lists>` representing
    the union of the value on the target itself with the values of the
    corresponding ``INTERFACE_PROP`` on targets named by the target's
    :prop_tgt:`LINK_LIBRARIES`:

    * If ``PROP`` is named by :prop_tgt:`TRANSITIVE_COMPILE_PROPERTIES`,
      evaluation of the corresponding ``INTERFACE_PROP`` is transitive over
      the closure of the linked targets' :prop_tgt:`INTERFACE_LINK_LIBRARIES`,
      excluding entries guarded by the :genex:`LINK_ONLY` generator expression.

    * If ``PROP`` is named by :prop_tgt:`TRANSITIVE_LINK_PROPERTIES`,
      evaluation of the corresponding ``INTERFACE_PROP`` is transitive over
      the closure of the linked targets' :prop_tgt:`INTERFACE_LINK_LIBRARIES`,
      including entries guarded by the :genex:`LINK_ONLY` generator expression.

  * Evaluation of :genex:`$<TARGET_PROPERTY:tgt,INTERFACE_PROP>` for some
    property ``INTERFACE_PROP``, named with an ``INTERFACE_`` prefix,
    checks the :prop_tgt:`TRANSITIVE_COMPILE_PROPERTIES`
    and :prop_tgt:`TRANSITIVE_LINK_PROPERTIES` properties on target ``tgt``,
    and on the transitive closure of targets named by its
    :prop_tgt:`INTERFACE_LINK_LIBRARIES`.

    If the corresponding ``PROP`` is listed by one of those properties,
    then ``INTERFACE_PROP`` evaluates as a
    :ref:`semicolon-separated list <CMake Language Lists>` representing the
    union of the value on the target itself with the value of the same
    property on targets named by the target's
    :prop_tgt:`INTERFACE_LINK_LIBRARIES`:

    * If ``PROP`` is named by :prop_tgt:`TRANSITIVE_COMPILE_PROPERTIES`,
      evaluation of the corresponding ``INTERFACE_PROP`` is transitive over
      the closure of the target's :prop_tgt:`INTERFACE_LINK_LIBRARIES`,
      excluding entries guarded by the :genex:`LINK_ONLY` generator expression.

    * If ``PROP`` is named by :prop_tgt:`TRANSITIVE_LINK_PROPERTIES`,
      evaluation of the corresponding ``INTERFACE_PROP`` is transitive over
      the closure of the target's :prop_tgt:`INTERFACE_LINK_LIBRARIES`,
      including entries guarded by the :genex:`LINK_ONLY` generator expression.

  If a ``PROP`` is named by both :prop_tgt:`TRANSITIVE_COMPILE_PROPERTIES`
  and :prop_tgt:`TRANSITIVE_LINK_PROPERTIES`, the latter takes precedence.

:ref:`Compatible Interface Properties`
  These evaluate as a single value combined from the target itself,
  from targets named by the target's :prop_tgt:`LINK_LIBRARIES`, and
  from the transitive closure of the linked targets'
  :prop_tgt:`INTERFACE_LINK_LIBRARIES`.  Values of a compatible
  interface property from multiple targets combine based on the type
  of compatibility required by the ``COMPATIBLE_INTERFACE_*`` property
  defining it.


Target Artifacts
^^^^^^^^^^^^^^^^

These expressions look up information about artifacts associated with
a given target ``tgt``.  Unless otherwise stated, this can be any
runtime artifact, namely:

* An executable target created by :command:`add_executable`.
* A shared library target (``.so``, ``.dll`` but not their ``.lib`` import
  library) created by :command:`add_library`.
* A static library target created by :command:`add_library`.

In the following, the phrase "the ``tgt`` filename" means the name of the
``tgt`` binary file. This has to be distinguished from the phrase
"the target name", which is just the string ``tgt``.

.. genex:: $<TARGET_FILE:tgt>

  Full path to the ``tgt`` binary file.

  Note that ``tgt`` is not added as a dependency of the target this
  expression is evaluated on, unless the expression is being used in
  :command:`add_custom_command` or :command:`add_custom_target`.

.. genex:: $<TARGET_FILE_BASE_NAME:tgt[,POSTFIX:(INCLUDE|EXCLUDE)]>

  .. versionadded:: 3.15

  .. versionadded:: 4.2
    The option ``POSTFIX``, which can be used to control the inclusion or not
    of the :prop_tgt:`<CONFIG>_POSTFIX` target property value as part of the
    base name. The default is ``POSTFIX:INCLUDE``.

  Base name of ``tgt``, i.e. ``$<TARGET_FILE_NAME:tgt>`` without prefix and
  suffix and, optionally, postfix.
  For example, if the ``tgt`` filename is ``libbase_postfix.so``, the base name
  is:

    * ``base_postfix`` for ``$<TARGET_FILE_BASE_NAME:tgt>`` or
      ``$<TARGET_FILE_BASE_NAME:tgt,POSTFIX:INCLUDE>``.
    * ``base`` for ``$<TARGET_FILE_BASE_NAME:tgt,POSTFIX:EXCLUDE>``.

  See also the :prop_tgt:`OUTPUT_NAME`, :prop_tgt:`ARCHIVE_OUTPUT_NAME`,
  :prop_tgt:`LIBRARY_OUTPUT_NAME` and :prop_tgt:`RUNTIME_OUTPUT_NAME`
  target properties, their configuration-specific variants
  :prop_tgt:`OUTPUT_NAME_<CONFIG>`, :prop_tgt:`ARCHIVE_OUTPUT_NAME_<CONFIG>`,
  :prop_tgt:`LIBRARY_OUTPUT_NAME_<CONFIG>` and
  :prop_tgt:`RUNTIME_OUTPUT_NAME_<CONFIG>`, and
  the :prop_tgt:`<CONFIG>_POSTFIX` and :prop_tgt:`DEBUG_POSTFIX` target
  properties.

  Note that ``tgt`` is not added as a dependency of the target this
  expression is evaluated on.

.. genex:: $<TARGET_FILE_PREFIX:tgt>

  .. versionadded:: 3.15

  Prefix of the ``tgt`` filename (such as ``lib``).

  See also the :prop_tgt:`PREFIX` target property.

  Note that ``tgt`` is not added as a dependency of the target this
  expression is evaluated on.

.. genex:: $<TARGET_FILE_SUFFIX:tgt>

  .. versionadded:: 3.15

  Suffix of the ``tgt`` filename (extension such as ``.so`` or ``.exe``).

  See also the :prop_tgt:`SUFFIX` target property.

  Note that ``tgt`` is not added as a dependency of the target this
  expression is evaluated on.

.. genex:: $<TARGET_FILE_NAME:tgt>

  The ``tgt`` filename.

  Note that ``tgt`` is not added as a dependency of the target this
  expression is evaluated on (see policy :policy:`CMP0112`).

.. genex:: $<TARGET_FILE_DIR:tgt>

  Directory of the ``tgt`` binary file.

  Note that ``tgt`` is not added as a dependency of the target this
  expression is evaluated on (see policy :policy:`CMP0112`).

.. genex:: $<TARGET_IMPORT_FILE:tgt>

  .. versionadded:: 3.27

  Full path to the linker import file. On DLL platforms, it would be the
  ``.lib`` file. For executables on AIX, and for shared libraries on macOS,
  it could be, respectively, the ``.imp`` or ``.tbd`` import file,
  depending on the value of the :prop_tgt:`ENABLE_EXPORTS` property.

  This expands to an empty string when there is no import file associated
  with the target.

.. genex:: $<TARGET_IMPORT_FILE_BASE_NAME:tgt[,POSTFIX:(INCLUDE|EXCLUDE)]>

  .. versionadded:: 3.27

  .. versionadded:: 4.2
    The option ``POSTFIX``, which can be used to control the inclusion or not
    of the :prop_tgt:`<CONFIG>_POSTFIX` target property value as part of the
    base name. The default is ``POSTFIX:INCLUDE``.

  Base name of the linker import file of the target ``tgt`` without prefix or
  suffix and, optionally, postfix.
  For example, if the target file name is ``libbase_postfix.tbd``, the base
  name is:

    * ``base_postfix`` for ``$<TARGET_IMPORT_FILE_BASE_NAME:tgt>`` or
      ``$<TARGET_IMPORT_FILE_BASE_NAME:tgt,POSTFIX:INCLUDE>``.
    * ``base`` for ``$<TARGET_IMPORT_FILE_BASE_NAME:tgt,POSTFIX:EXCLUDE>``.

  See also the :prop_tgt:`OUTPUT_NAME` and :prop_tgt:`ARCHIVE_OUTPUT_NAME`
  target properties, their configuration-specific variants
  :prop_tgt:`OUTPUT_NAME_<CONFIG>` and
  :prop_tgt:`ARCHIVE_OUTPUT_NAME_<CONFIG>`, and
  the :prop_tgt:`<CONFIG>_POSTFIX` and :prop_tgt:`DEBUG_POSTFIX` target
  properties.

  Note that ``tgt`` is not added as a dependency of the target this
  expression is evaluated on.

.. genex:: $<TARGET_IMPORT_FILE_PREFIX:tgt>

  .. versionadded:: 3.27

  Prefix of the import file of the target ``tgt``.

  See also the :prop_tgt:`IMPORT_PREFIX` target property.

  Note that ``tgt`` is not added as a dependency of the target this
  expression is evaluated on.

.. genex:: $<TARGET_IMPORT_FILE_SUFFIX:tgt>

  .. versionadded:: 3.27

  Suffix of the import file of the target ``tgt``.

  The suffix corresponds to the file extension (such as ``.lib`` or ``.tbd``).

  See also the :prop_tgt:`IMPORT_SUFFIX` target property.

  Note that ``tgt`` is not added as a dependency of the target this
  expression is evaluated on.

.. genex:: $<TARGET_IMPORT_FILE_NAME:tgt>

  .. versionadded:: 3.27

  Name of the import file of the target ``tgt``.

  Note that ``tgt`` is not added as a dependency of the target this
  expression is evaluated on.

.. genex:: $<TARGET_IMPORT_FILE_DIR:tgt>

  .. versionadded:: 3.27

  Directory of the import file of the target ``tgt``.

  Note that ``tgt`` is not added as a dependency of the target this
  expression is evaluated on.

.. genex:: $<TARGET_LINKER_FILE:tgt>

  File used when linking to the ``tgt`` target.  This will usually
  be the library that ``tgt`` represents (``.a``, ``.lib``, ``.so``),
  but for a shared library on DLL platforms, it would be the ``.lib``
  import library associated with the DLL.

  .. versionadded:: 3.27
    On macOS, it could be the ``.tbd`` import file associated with the shared
    library, depending on the value of the :prop_tgt:`ENABLE_EXPORTS` property.

  This generator expression is equivalent to
  :genex:`$<TARGET_LINKER_LIBRARY_FILE>` or
  :genex:`$<TARGET_LINKER_IMPORT_FILE>` generator expressions, depending on the
  characteristics of the target and the platform.

.. genex:: $<TARGET_LINKER_FILE_BASE_NAME:tgt[,POSTFIX:(INCLUDE|EXCLUDE)]>

  .. versionadded:: 3.15

  .. versionadded:: 4.2
    The option ``POSTFIX``, which can be used to control the inclusion or not
    of the :prop_tgt:`<CONFIG>_POSTFIX` target property value as part of the
    base name. The default is ``POSTFIX:INCLUDE``.

  Base name of file used to link the target ``tgt``, i.e.
  :genex:`$<TARGET_LINKER_FILE_NAME:tgt>` without prefix and suffix, and,
  optionally, postfix.
  For example, if target file name is ``libbase_postfix.a``, the base name is:

    * ``base_postfix`` for ``$<TARGET_LINKER_FILE_BASE_NAME:tgt>`` or
      ``$<TARGET_LINKER_FILE_BASE_NAME:tgt,POSTFIX:INCLUDE>``.
    * ``base`` for ``$<TARGET_LINKER_FILE_BASE_NAME:tgt,POSTFIX:EXCLUDE>``.

  See also the :prop_tgt:`OUTPUT_NAME`, :prop_tgt:`ARCHIVE_OUTPUT_NAME`,
  and :prop_tgt:`LIBRARY_OUTPUT_NAME` target properties, their
  configuration-specific variants :prop_tgt:`OUTPUT_NAME_<CONFIG>`,
  :prop_tgt:`ARCHIVE_OUTPUT_NAME_<CONFIG>` and
  :prop_tgt:`LIBRARY_OUTPUT_NAME_<CONFIG>`, and
  the :prop_tgt:`<CONFIG>_POSTFIX` and :prop_tgt:`DEBUG_POSTFIX` target
  properties.

  Note that ``tgt`` is not added as a dependency of the target this
  expression is evaluated on.

.. genex:: $<TARGET_LINKER_FILE_PREFIX:tgt>

  .. versionadded:: 3.15

  Prefix of file used to link target ``tgt``.

  See also the :prop_tgt:`PREFIX` and :prop_tgt:`IMPORT_PREFIX` target
  properties.

  Note that ``tgt`` is not added as a dependency of the target this
  expression is evaluated on.

.. genex:: $<TARGET_LINKER_FILE_SUFFIX:tgt>

  .. versionadded:: 3.15

  Suffix of file used to link where ``tgt`` is the name of a target.

  The suffix corresponds to the file extension (such as ".so" or ".lib").

  See also the :prop_tgt:`SUFFIX` and :prop_tgt:`IMPORT_SUFFIX` target
  properties.

  Note that ``tgt`` is not added as a dependency of the target this
  expression is evaluated on.

.. genex:: $<TARGET_LINKER_FILE_NAME:tgt>

  Name of file used to link target ``tgt``.

  Note that ``tgt`` is not added as a dependency of the target this
  expression is evaluated on (see policy :policy:`CMP0112`).

.. genex:: $<TARGET_LINKER_FILE_DIR:tgt>

  Directory of file used to link target ``tgt``.

  Note that ``tgt`` is not added as a dependency of the target this
  expression is evaluated on (see policy :policy:`CMP0112`).

.. genex:: $<TARGET_LINKER_LIBRARY_FILE:tgt>

  .. versionadded:: 3.27

  File used when linking o the ``tgt`` target is done using directly the
  library, and not an import file. This will usually be the library that
  ``tgt`` represents (``.a``, ``.so``, ``.dylib``). So, on DLL platforms, it
  will be an empty string.

.. genex:: $<TARGET_LINKER_LIBRARY_FILE_BASE_NAME:tgt[,POSTFIX:(INCLUDE|EXCLUDE)]>

  .. versionadded:: 3.27

  .. versionadded:: 4.2
    The option ``POSTFIX``, which can be used to control the inclusion or not
    of the :prop_tgt:`<CONFIG>_POSTFIX` target property value as part of the
    base name. The default is ``POSTFIX:INCLUDE``.

  Base name of library file used to link the target ``tgt``, i.e.
  :genex:`$<TARGET_LINKER_LIBRARY_FILE_NAME:tgt>` without prefix and
  suffix,and, optionally, postfix.
  For example, if target file name is ``libbase_postfix.a``, the base name is:

    * ``base_postfix`` for ``$<TARGET_LINKER_LIBRARY_FILE_BASE_NAME:tgt>`` or
      ``$<TARGET_LINKER_LIBRARY_FILE_BASE_NAME:tgt,POSTFIX:INCLUDE>``.
    * ``base`` for
      ``$<TARGET_LINKER_LIBRARY_FILE_BASE_NAME:tgt,POSTFIX:EXCLUDE>``.

  See also the :prop_tgt:`OUTPUT_NAME`, :prop_tgt:`ARCHIVE_OUTPUT_NAME`,
  and :prop_tgt:`LIBRARY_OUTPUT_NAME` target properties, their
  configuration-specific variants :prop_tgt:`OUTPUT_NAME_<CONFIG>`,
  :prop_tgt:`ARCHIVE_OUTPUT_NAME_<CONFIG>` and
  :prop_tgt:`LIBRARY_OUTPUT_NAME_<CONFIG>`, and
  the :prop_tgt:`<CONFIG>_POSTFIX` and :prop_tgt:`DEBUG_POSTFIX` target
  properties.

  Note that ``tgt`` is not added as a dependency of the target this
  expression is evaluated on.

.. genex:: $<TARGET_LINKER_LIBRARY_FILE_PREFIX:tgt>

  .. versionadded:: 3.27

  Prefix of the library file used to link target ``tgt``.

  See also the :prop_tgt:`PREFIX` target property.

  Note that ``tgt`` is not added as a dependency of the target this
  expression is evaluated on.

.. genex:: $<TARGET_LINKER_LIBRARY_FILE_SUFFIX:tgt>

  .. versionadded:: 3.27

  Suffix of the library file used to link target ``tgt``.

  The suffix corresponds to the file extension (such as ".a" or ".dylib").

  See also the :prop_tgt:`SUFFIX` target property.

  Note that ``tgt`` is not added as a dependency of the target this
  expression is evaluated on.

.. genex:: $<TARGET_LINKER_LIBRARY_FILE_NAME:tgt>

  .. versionadded:: 3.27

  Name of the library file used to link target ``tgt``.

  Note that ``tgt`` is not added as a dependency of the target this
  expression is evaluated on.

.. genex:: $<TARGET_LINKER_LIBRARY_FILE_DIR:tgt>

  .. versionadded:: 3.27

  Directory of the library file used to link target ``tgt``.

  Note that ``tgt`` is not added as a dependency of the target this
  expression is evaluated on.

.. genex:: $<TARGET_LINKER_IMPORT_FILE:tgt>

  .. versionadded:: 3.27

  File used when linking to the ``tgt`` target is done using an import
  file.  This will usually be the import file that ``tgt`` represents
  (``.lib``, ``.tbd``). So, when no import file is involved in the link step,
  an empty string is returned.

.. genex:: $<TARGET_LINKER_IMPORT_FILE_BASE_NAME:tgt[,POSTFIX:(INCLUDE|EXCLUDE)]>

  .. versionadded:: 3.27

  .. versionadded:: 4.2
    The option ``POSTFIX``, which can be used to control the inclusion or not
    of the :prop_tgt:`<CONFIG>_POSTFIX` target property value as part of the
    base name. The default is ``POSTFIX:INCLUDE``.

  Base name of the import file used to link the target ``tgt``, i.e.
  :genex:`$<TARGET_LINKER_IMPORT_FILE_NAME:tgt>` without prefix and suffix,
  and, optionally, postfix.
  For example, if target file name is ``libbase_postfix.tbd``, the base name is

    * ``base_postfix`` for ``$<TARGET_LINKER_IMPORT_FILE_BASE_NAME:tgt>`` or
      ``$<TARGET_LINKER_IMPORT_FILE_BASE_NAME:tgt,POSTFIX:INCLUDE>``.
    * ``base`` for
      ``$<TARGET_LINKER_IMPORT_FILE_BASE_NAME:tgt,POSTFIX:EXCLUDE>``.

  See also the :prop_tgt:`OUTPUT_NAME` and :prop_tgt:`ARCHIVE_OUTPUT_NAME`,
  target properties, their configuration-specific variants
  :prop_tgt:`OUTPUT_NAME_<CONFIG>` and
  :prop_tgt:`ARCHIVE_OUTPUT_NAME_<CONFIG>`, and
  the :prop_tgt:`<CONFIG>_POSTFIX` and :prop_tgt:`DEBUG_POSTFIX` target
  properties.

  Note that ``tgt`` is not added as a dependency of the target this
  expression is evaluated on.

.. genex:: $<TARGET_LINKER_IMPORT_FILE_PREFIX:tgt>

  .. versionadded:: 3.27

  Prefix of the import file used to link target ``tgt``.

  See also the :prop_tgt:`IMPORT_PREFIX` target property.

  Note that ``tgt`` is not added as a dependency of the target this
  expression is evaluated on.

.. genex:: $<TARGET_LINKER_IMPORT_FILE_SUFFIX:tgt>

  .. versionadded:: 3.27

  Suffix of the import file used to link target ``tgt``.

  The suffix corresponds to the file extension (such as ".lib" or ".tbd").

  See also the :prop_tgt:`IMPORT_SUFFIX` target property.

  Note that ``tgt`` is not added as a dependency of the target this
  expression is evaluated on.

.. genex:: $<TARGET_LINKER_IMPORT_FILE_NAME:tgt>

  .. versionadded:: 3.27

  Name of the import file used to link target ``tgt``.

  Note that ``tgt`` is not added as a dependency of the target this
  expression is evaluated on.

.. genex:: $<TARGET_LINKER_IMPORT_FILE_DIR:tgt>

  .. versionadded:: 3.27

  Directory of the import file used to link target ``tgt``.

  Note that ``tgt`` is not added as a dependency of the target this
  expression is evaluated on.

.. genex:: $<TARGET_SONAME_FILE:tgt>

  File with soname (``.so.3``) where ``tgt`` is the name of a target.

.. genex:: $<TARGET_SONAME_FILE_NAME:tgt>

  Name of file with soname (``.so.3``).

  Note that ``tgt`` is not added as a dependency of the target this
  expression is evaluated on (see policy :policy:`CMP0112`).

.. genex:: $<TARGET_SONAME_FILE_DIR:tgt>

  Directory of file with soname (``.so.3``).

  Note that ``tgt`` is not added as a dependency of the target this
  expression is evaluated on (see policy :policy:`CMP0112`).

.. genex:: $<TARGET_SONAME_IMPORT_FILE:tgt>

  .. versionadded:: 3.27

  Import file with soname (``.3.tbd``) where ``tgt`` is the name of a target.

.. genex:: $<TARGET_SONAME_IMPORT_FILE_NAME:tgt>

  .. versionadded:: 3.27

  Name of the import file with soname (``.3.tbd``).

  Note that ``tgt`` is not added as a dependency of the target this
  expression is evaluated on.

.. genex:: $<TARGET_SONAME_IMPORT_FILE_DIR:tgt>

  .. versionadded:: 3.27

  Directory of the import file with soname (``.3.tbd``).

  Note that ``tgt`` is not added as a dependency of the target this
  expression is evaluated on.

.. genex:: $<TARGET_PDB_FILE:tgt>

  .. versionadded:: 3.1

  Full path to the linker generated program database file (.pdb)
  where ``tgt`` is the name of a target.

  .. versionchanged:: 4.2
    The postfix, as specified by :prop_tgt:`DEBUG_POSTFIX` or
    :prop_tgt:`<CONFIG>_POSTFIX` target properties, is always included in the
    ``PDB`` file name. See the policy :policy:`CMP0202`.

  See also the :prop_tgt:`PDB_NAME` and :prop_tgt:`PDB_OUTPUT_DIRECTORY`
  target properties and their configuration specific variants
  :prop_tgt:`PDB_NAME_<CONFIG>` and :prop_tgt:`PDB_OUTPUT_DIRECTORY_<CONFIG>`.

.. genex:: $<TARGET_PDB_FILE_BASE_NAME:tgt[,POSTFIX:(INCLUDE|EXCLUDE)]>

  .. versionadded:: 3.15

  Base name of the linker generated program database file (.pdb)
  where ``tgt`` is the name of a target.

  .. versionadded:: 4.2
    The option ``POSTFIX``, which can be used to control the inclusion or not
    of the :prop_tgt:`<CONFIG>_POSTFIX` target property value as part of the
    base name. The default is ``POSTFIX:INCLUDE``.

  .. versionchanged:: 4.2
    The postfix, as specified by :prop_tgt:`DEBUG_POSTFIX` or
    :prop_tgt:`<CONFIG>_POSTFIX` target properties, is always included in the
    ``PDB`` base name, except if option ``POSTFIX`` has value ``EXCLUDE``.
    See the policy :policy:`CMP0202`.

  The base name corresponds to the target PDB file name (see
  ``$<TARGET_PDB_FILE_NAME:tgt>``) without prefix and suffix, and, optionally,
  postfix. For example, if target file name is ``base_postfix.pdb``, the base
  name is

    * ``base_postfix`` for ``$<TARGET_PDB_FILE_BASE_NAME:tgt>`` or
      ``$<TARGET_PDB_FILE_BASE_NAME:tgt,POSTFIX:INCLUDE>``.
    * ``base`` for ``$<TARGET_PDB_FILE_BASE_NAME:tgt,POSTFIX:EXCLUDE>``.

  See also the :prop_tgt:`OUTPUT_NAME`, :prop_tgt:`PDB_NAME` target properties,
  and their configuration-specific variants :prop_tgt:`OUTPUT_NAME_<CONFIG>`
  and :prop_tgt:`PDB_NAME_<CONFIG>`, and the :prop_tgt:`<CONFIG>_POSTFIX` and
  :prop_tgt:`DEBUG_POSTFIX` target properties.

  Note that ``tgt`` is not added as a dependency of the target this
  expression is evaluated on.

.. genex:: $<TARGET_PDB_FILE_NAME:tgt>

  .. versionadded:: 3.1

  Name of the linker generated program database file (.pdb).

  Note that ``tgt`` is not added as a dependency of the target this
  expression is evaluated on (see policy :policy:`CMP0112`).

.. genex:: $<TARGET_PDB_FILE_DIR:tgt>

  .. versionadded:: 3.1

  Directory of the linker generated program database file (.pdb).

  Note that ``tgt`` is not added as a dependency of the target this
  expression is evaluated on (see policy :policy:`CMP0112`).

.. genex:: $<TARGET_BUNDLE_DIR:tgt>

  .. versionadded:: 3.9

  Full path to the bundle directory (``/path/to/my.app``,
  ``/path/to/my.framework``, or ``/path/to/my.bundle``),
  where ``tgt`` is the name of a target.

  Note that ``tgt`` is not added as a dependency of the target this
  expression is evaluated on (see policy :policy:`CMP0112`).

.. genex:: $<TARGET_BUNDLE_DIR_NAME:tgt>

  .. versionadded:: 3.24

  Name of the bundle directory (``my.app``, ``my.framework``, or
  ``my.bundle``), where ``tgt`` is the name of a target.

  Note that ``tgt`` is not added as a dependency of the target this
  expression is evaluated on (see policy :policy:`CMP0112`).

.. genex:: $<TARGET_BUNDLE_CONTENT_DIR:tgt>

  .. versionadded:: 3.9

  Full path to the bundle content directory where ``tgt`` is the name of a
  target.  For the macOS SDK it leads to ``/path/to/my.app/Contents``,
  ``/path/to/my.framework``, or ``/path/to/my.bundle/Contents``.
  For all other SDKs (e.g. iOS) it leads to ``/path/to/my.app``,
  ``/path/to/my.framework``, or ``/path/to/my.bundle`` due to the flat
  bundle structure.

  Note that ``tgt`` is not added as a dependency of the target this
  expression is evaluated on (see policy :policy:`CMP0112`).

.. genex:: $<TARGET_OBJECTS:tgt>

  .. versionadded:: 3.1

  List of objects resulting from building ``tgt``.  This would typically be
  used on :ref:`object library <Object Libraries>` targets.

.. genex:: $<TARGET_RUNTIME_DLLS:tgt>

  .. versionadded:: 3.21

  List of DLLs that the target depends on at runtime. This is determined by
  the locations of all the ``SHARED`` targets in the target's transitive
  dependencies. If only the directories of the DLLs are needed, see the
  :genex:`TARGET_RUNTIME_DLL_DIRS` generator expression.
  Using this generator expression on targets other than
  executables, ``SHARED`` libraries, and ``MODULE`` libraries is an error.
  **On non-DLL platforms, this expression always evaluates to an empty string**.

  This generator expression can be used to copy all of the DLLs that a target
  depends on into its output directory in a ``POST_BUILD`` custom command using
  the :option:`cmake -E copy -t <cmake-E copy>` command. For example:

  .. code-block:: cmake

    find_package(foo CONFIG REQUIRED) # package generated by install(EXPORT)

    add_executable(exe main.c)
    target_link_libraries(exe PRIVATE foo::foo foo::bar)
    add_custom_command(TARGET exe POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy -t $<TARGET_FILE_DIR:exe> $<TARGET_RUNTIME_DLLS:exe>
      COMMAND_EXPAND_LISTS
    )

  .. note::

    :ref:`Imported Targets` are supported only if they know the location
    of their ``.dll`` files.  An imported ``SHARED`` library must have
    :prop_tgt:`IMPORTED_LOCATION` set to its ``.dll`` file.  See the
    :ref:`add_library imported libraries <add_library imported libraries>`
    section for details.  Many :ref:`Find Modules` produce imported targets
    with the ``UNKNOWN`` type and therefore will be ignored.

  On platforms that support runtime paths (``RPATH``), refer to the
  :prop_tgt:`INSTALL_RPATH` target property.
  On Apple platforms, refer to the :prop_tgt:`INSTALL_NAME_DIR` target property.

.. genex:: $<TARGET_RUNTIME_DLL_DIRS:tgt>

  .. versionadded:: 3.27

  List of the directories which contain the DLLs that the target depends on at
  runtime (see :genex:`TARGET_RUNTIME_DLLS`). This is determined by
  the locations of all the ``SHARED`` targets in the target's transitive
  dependencies. Using this generator expression on targets other than
  executables, ``SHARED`` libraries, and ``MODULE`` libraries is an error.
  **On non-DLL platforms, this expression always evaluates to an empty string**.

  This generator expression can e.g. be used to create a batch file using
  :command:`file(GENERATE)` which sets the PATH environment variable accordingly.

.. genex:: $<TARGET_INTERMEDIATE_DIR:tgt>

  .. versionadded:: 4.2

  The full path to the directory where intermediate target files, such as
  object and dependency files, are stored.

Export And Install Expressions
------------------------------

.. genex:: $<INSTALL_INTERFACE:...>

  Content of ``...`` when the property is exported using
  :command:`install(EXPORT)`, and empty otherwise.

.. genex:: $<BUILD_INTERFACE:...>

  Content of ``...`` when the property is exported using :command:`export`, or
  when the target is used by another target in the same buildsystem. Expands to
  the empty string otherwise.

.. genex:: $<BUILD_LOCAL_INTERFACE:...>

  .. versionadded:: 3.26

  Content of ``...`` when the target is used by another target in the same
  buildsystem. Expands to the empty string otherwise.

.. genex:: $<INSTALL_PREFIX>

  Content of the install prefix when the target is exported via
  :command:`install(EXPORT)`, or when evaluated in the
  :prop_tgt:`INSTALL_NAME_DIR` property or the ``INSTALL_NAME_DIR`` argument of
  :command:`install(RUNTIME_DEPENDENCY_SET)`, and empty otherwise.

  .. versionchanged:: 3.27
    Evaluates to the content of the install prefix
    in the code argument of :command:`install(CODE)` or
    the file argument of :command:`install(SCRIPT)`.

Multi-level Expression Evaluation
---------------------------------

.. genex:: $<GENEX_EVAL:expr>

  .. versionadded:: 3.12

  Content of ``expr`` evaluated as a generator expression in the current
  context. This enables consumption of generator expressions whose
  evaluation results itself in generator expressions.

.. genex:: $<TARGET_GENEX_EVAL:tgt,expr>

  .. versionadded:: 3.12

  Content of ``expr`` evaluated as a generator expression in the context of
  ``tgt`` target. This enables consumption of custom target properties that
  themselves contain generator expressions.

  Having the capability to evaluate generator expressions is very useful when
  you want to manage custom properties supporting generator expressions.
  For example:

  .. code-block:: cmake

    add_library(foo ...)

    set_property(TARGET foo PROPERTY
      CUSTOM_KEYS $<$<CONFIG:DEBUG>:FOO_EXTRA_THINGS>
    )

    add_custom_target(printFooKeys
      COMMAND ${CMAKE_COMMAND} -E echo $<TARGET_PROPERTY:foo,CUSTOM_KEYS>
    )

  This naive implementation of the ``printFooKeys`` custom command is wrong
  because ``CUSTOM_KEYS`` target property is not evaluated and the content
  is passed as is (i.e. ``$<$<CONFIG:DEBUG>:FOO_EXTRA_THINGS>``).

  To have the expected result (i.e. ``FOO_EXTRA_THINGS`` if config is
  ``Debug``), it is required to evaluate the output of
  ``$<TARGET_PROPERTY:foo,CUSTOM_KEYS>``:

  .. code-block:: cmake

    add_custom_target(printFooKeys
      COMMAND ${CMAKE_COMMAND} -E
        echo $<TARGET_GENEX_EVAL:foo,$<TARGET_PROPERTY:foo,CUSTOM_KEYS>>
    )

Escaped Characters
------------------

These expressions evaluate to specific string literals. Use them in place of
the actual string literal where you need to prevent them from having their
special meaning.

.. genex:: $<ANGLE-R>

  A literal ``>``. Used for example to compare strings that contain a ``>``.

.. genex:: $<COMMA>

  A literal ``,``. Used for example to compare strings which contain a ``,``.

.. genex:: $<SEMICOLON>

  A literal ``;``. Used to prevent list expansion on an argument with ``;``.

.. genex:: $<QUOTE>

  .. versionadded:: 3.30

  A literal ``"``. Used to allow string literal quotes inside a generator expression.


Deprecated Expressions
----------------------

.. genex:: $<CONFIGURATION>

  Configuration name. Deprecated since CMake 3.0. Use :genex:`CONFIG` instead.
