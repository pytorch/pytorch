if
--

Conditionally execute a group of commands.

Synopsis
^^^^^^^^

.. code-block:: cmake

  if(<condition>)
    <commands>
  elseif(<condition>) # optional block, can be repeated
    <commands>
  else()              # optional block
    <commands>
  endif()

Evaluates the ``condition`` argument of the ``if`` clause according to the
`Condition syntax`_ described below. If the result is true, then the
``commands`` in the ``if`` block are executed.
Otherwise, optional ``elseif`` blocks are processed in the same way.
Finally, if no ``condition`` is true, ``commands`` in the optional ``else``
block are executed.

Per legacy, the :command:`else` and :command:`endif` commands admit
an optional ``<condition>`` argument.
If used, it must be a verbatim
repeat of the argument of the opening
``if`` command.

.. _`Condition Syntax`:

Condition Syntax
^^^^^^^^^^^^^^^^

The following syntax applies to the ``condition`` argument of
the ``if``, ``elseif`` and :command:`while` clauses.

Compound conditions are evaluated in the following order of precedence:

1. `Parentheses`_.

2. Unary tests such as `COMMAND`_, `POLICY`_, `TARGET`_, `TEST`_,
   `EXISTS`_, `IS_READABLE`_, `IS_WRITABLE`_, `IS_EXECUTABLE`_,
   `IS_DIRECTORY`_, `IS_SYMLINK`_, `IS_ABSOLUTE`_, and `DEFINED`_.

3. Binary tests such as `EQUAL`_, `LESS`_, `LESS_EQUAL`_, `GREATER`_,
   `GREATER_EQUAL`_, `STREQUAL`_, `STRLESS`_, `STRLESS_EQUAL`_,
   `STRGREATER`_, `STRGREATER_EQUAL`_, `VERSION_EQUAL`_, `VERSION_LESS`_,
   `VERSION_LESS_EQUAL`_, `VERSION_GREATER`_, `VERSION_GREATER_EQUAL`_,
   `PATH_EQUAL`_, `IN_LIST`_, `IS_NEWER_THAN`_, and `MATCHES`_.

4. Unary logical operator `NOT`_.

5. Binary logical operators `AND`_ and `OR`_, from left to right,
   without any short-circuit.

Basic Expressions
"""""""""""""""""

.. signature:: if(<constant>)
  :target: constant

  True if the constant is ``1``, ``ON``, ``YES``, ``TRUE``, ``Y``,
  or a non-zero number (including floating point numbers).
  False if the constant is ``0``, ``OFF``,
  ``NO``, ``FALSE``, ``N``, ``IGNORE``, ``NOTFOUND``, the empty string,
  or ends in the suffix ``-NOTFOUND``.  Named boolean constants are
  case-insensitive.  If the argument is not one of these specific
  constants, it is treated as a variable or string (see `Variable Expansion`_
  further below) and one of the following two forms applies.

.. signature:: if(<variable>)
  :target: variable

  True if given a variable that is defined to a value that is not a false
  constant.  False otherwise, including if the variable is undefined.
  Note that macro arguments are not variables.
  :ref:`Environment Variables <CMake Language Environment Variables>` also
  cannot be tested this way, e.g. ``if(ENV{some_var})`` will always evaluate
  to false.

.. signature:: if(<string>)
  :target: string

  A quoted string always evaluates to false unless:

  * The string's value is one of the true constants, or
  * in CMake versions prior to 4.0, policy :policy:`CMP0054` is not set
    to ``NEW`` and the string's value happens to be a variable name that
    is affected by :policy:`CMP0054`'s behavior.

Logic Operators
"""""""""""""""

.. signature:: if(NOT <condition>)

  True if the condition is not true.

.. signature:: if(<cond1> AND <cond2>)
  :target: AND

  True if both conditions would be considered true individually.

.. signature:: if(<cond1> OR <cond2>)
  :target: OR

  True if either condition would be considered true individually.

.. signature:: if((condition) AND (condition OR (condition)))
  :target: parentheses

  The conditions inside the parenthesis are evaluated first and then
  the remaining condition is evaluated as in the other examples.
  Where there are nested parenthesis the innermost are evaluated as part
  of evaluating the condition that contains them.

Existence Checks
""""""""""""""""

.. signature:: if(COMMAND <command-name>)

  True if the given name is a command, macro or function that can be
  invoked.

.. signature:: if(POLICY <policy-id>)

  True if the given name is an existing policy (of the form ``CMP<NNNN>``).

.. signature:: if(TARGET <target-name>)

  True if the given name is an existing logical target name created
  by a call to the :command:`add_executable`, :command:`add_library`,
  or :command:`add_custom_target` command that has already been invoked
  (in any directory).

.. signature:: if(TEST <test-name>)

  .. versionadded:: 3.3

  True if the given name is an existing test name created by the
  :command:`add_test` command.

.. signature:: if(DEFINED <name>|CACHE{<name>}|ENV{<name>})

  True if a variable, cache variable or environment variable
  with given ``<name>`` is defined. The value of the variable
  does not matter. Note the following caveats:

  * Macro arguments are not variables.
  * It is not possible to test directly whether a ``<name>`` is a non-cache
    variable.  The expression ``if(DEFINED someName)`` will evaluate to true
    if either a cache or non-cache variable ``someName`` exists.  In
    comparison, the expression ``if(DEFINED CACHE{someName})`` will only
    evaluate to true if a cache variable ``someName`` exists.  Both expressions
    need to be tested if you need to know whether a non-cache variable exists:
    ``if(DEFINED someName AND NOT DEFINED CACHE{someName})``.

 .. versionadded:: 3.14
  Added support for ``CACHE{<name>}`` variables.

.. signature:: if(<variable|string> IN_LIST <variable>)
  :target: IN_LIST

  .. versionadded:: 3.3

  True if the given element is contained in the named list variable.

File Operations
"""""""""""""""

.. signature:: if(EXISTS <path-to-file-or-directory>)

  True if the named file or directory exists and is readable.  Behavior
  is well-defined only for explicit full paths (a leading ``~/`` is not
  expanded as a home directory and is considered a relative path).
  Resolves symbolic links, i.e. if the named file or directory is a
  symbolic link, returns true if the target of the symbolic link exists.

  False if the given path is an empty string.

  .. note::
    Prefer ``if(IS_READABLE)`` to check file readability.  ``if(EXISTS)``
    may be changed in the future to only check file existence.

.. signature:: if(IS_READABLE <path-to-file-or-directory>)

  .. versionadded:: 3.29

  True if the named file or directory is readable.  Behavior
  is well-defined only for explicit full paths (a leading ``~/`` is not
  expanded as a home directory and is considered a relative path).
  Resolves symbolic links, i.e. if the named file or directory is a
  symbolic link, returns true if the target of the symbolic link is readable.

  False if the given path is an empty string.

.. signature:: if(IS_WRITABLE <path-to-file-or-directory>)

  .. versionadded:: 3.29

  True if the named file or directory is writable.  Behavior
  is well-defined only for explicit full paths (a leading ``~/`` is not
  expanded as a home directory and is considered a relative path).
  Resolves symbolic links, i.e. if the named file or directory is a
  symbolic link, returns true if the target of the symbolic link is writable.

  False if the given path is an empty string.

.. signature:: if(IS_EXECUTABLE <path-to-file-or-directory>)

  .. versionadded:: 3.29

  True if the named file or directory is executable.  Behavior
  is well-defined only for explicit full paths (a leading ``~/`` is not
  expanded as a home directory and is considered a relative path).
  Resolves symbolic links, i.e. if the named file or directory is a
  symbolic link, returns true if the target of the symbolic link is executable.

  False if the given path is an empty string.

.. signature:: if(<file1> IS_NEWER_THAN <file2>)
  :target: IS_NEWER_THAN

  True if ``file1`` is newer than ``file2`` or if one of the two files doesn't
  exist.  Behavior is well-defined only for full paths.  If the file
  time stamps are exactly the same, an ``IS_NEWER_THAN`` comparison returns
  true, so that any dependent build operations will occur in the event
  of a tie.  This includes the case of passing the same file name for
  both file1 and file2.

.. signature:: if(IS_DIRECTORY <path>)

  True if ``path`` is a directory.  Behavior is well-defined only
  for full paths.

  False if the given path is an empty string.

.. signature:: if(IS_SYMLINK <path>)

  True if the given path is a symbolic link.  Behavior is well-defined
  only for full paths.

.. signature:: if(IS_ABSOLUTE <path>)

  True if the given path is an absolute path.  Note the following special
  cases:

  * An empty ``path`` evaluates to false.
  * On Windows hosts, any ``path`` that begins with a drive letter and colon
    (e.g. ``C:``), a forward slash or a backslash will evaluate to true.
    This means a path like ``C:no\base\dir`` will evaluate to true, even
    though the non-drive part of the path is relative.
  * On non-Windows hosts, any ``path`` that begins with a tilde (``~``)
    evaluates to true.

Comparisons
"""""""""""

.. signature:: if(<variable|string> MATCHES <regex>)
  :target: MATCHES

  True if the given string or variable's value matches the given regular
  expression.  See :ref:`Regex Specification` for regex format.

  .. versionadded:: 2.6
   ``()`` groups are captured in :variable:`CMAKE_MATCH_<n>` variables.

.. signature:: if(<variable|string> LESS <variable|string>)
  :target: LESS

  True if the given string or variable's value parses as a real number
  (like a C ``double``) and less than that on the right.

.. signature:: if(<variable|string> GREATER <variable|string>)
  :target: GREATER

  True if the given string or variable's value parses as a real number
  (like a C ``double``) and greater than that on the right.

.. signature:: if(<variable|string> EQUAL <variable|string>)
  :target: EQUAL

  True if the given string or variable's value parses as a real number
  (like a C ``double``) and equal to that on the right.

.. signature:: if(<variable|string> LESS_EQUAL <variable|string>)
  :target: LESS_EQUAL

  .. versionadded:: 3.7

  True if the given string or variable's value parses as a real number
  (like a C ``double``) and less than or equal to that on the right.

.. signature:: if(<variable|string> GREATER_EQUAL <variable|string>)
  :target: GREATER_EQUAL

  .. versionadded:: 3.7

  True if the given string or variable's value parses as a real number
  (like a C ``double``) and greater than or equal to that on the right.

.. signature:: if(<variable|string> STRLESS <variable|string>)
  :target: STRLESS

  True if the given string or variable's value is lexicographically less
  than the string or variable on the right.

.. signature:: if(<variable|string> STRGREATER <variable|string>)
  :target: STRGREATER

  True if the given string or variable's value is lexicographically greater
  than the string or variable on the right.

.. signature:: if(<variable|string> STREQUAL <variable|string>)
  :target: STREQUAL

  True if the given string or variable's value is lexicographically equal
  to the string or variable on the right.

.. signature:: if(<variable|string> STRLESS_EQUAL <variable|string>)
  :target: STRLESS_EQUAL

  .. versionadded:: 3.7

  True if the given string or variable's value is lexicographically less
  than or equal to the string or variable on the right.

.. signature:: if(<variable|string> STRGREATER_EQUAL <variable|string>)
  :target: STRGREATER_EQUAL

  .. versionadded:: 3.7

  True if the given string or variable's value is lexicographically greater
  than or equal to the string or variable on the right.

Version Comparisons
"""""""""""""""""""

.. signature:: if(<variable|string> VERSION_LESS <variable|string>)
  :target: VERSION_LESS

  Component-wise integer version number comparison (version format is
  ``major[.minor[.patch[.tweak]]]``, omitted components are treated as zero).
  Any non-integer version component or non-integer trailing part of a version
  component effectively truncates the string at that point.

.. signature:: if(<variable|string> VERSION_GREATER <variable|string>)
  :target: VERSION_GREATER

  Component-wise integer version number comparison (version format is
  ``major[.minor[.patch[.tweak]]]``, omitted components are treated as zero).
  Any non-integer version component or non-integer trailing part of a version
  component effectively truncates the string at that point.

.. signature:: if(<variable|string> VERSION_EQUAL <variable|string>)
  :target: VERSION_EQUAL

  Component-wise integer version number comparison (version format is
  ``major[.minor[.patch[.tweak]]]``, omitted components are treated as zero).
  Any non-integer version component or non-integer trailing part of a version
  component effectively truncates the string at that point.

.. signature:: if(<variable|string> VERSION_LESS_EQUAL <variable|string>)
  :target: VERSION_LESS_EQUAL

  .. versionadded:: 3.7

  Component-wise integer version number comparison (version format is
  ``major[.minor[.patch[.tweak]]]``, omitted components are treated as zero).
  Any non-integer version component or non-integer trailing part of a version
  component effectively truncates the string at that point.

.. signature:: if(<variable|string> VERSION_GREATER_EQUAL <variable|string>)
  :target: VERSION_GREATER_EQUAL

  .. versionadded:: 3.7

  Component-wise integer version number comparison (version format is
  ``major[.minor[.patch[.tweak]]]``, omitted components are treated as zero).
  Any non-integer version component or non-integer trailing part of a version
  component effectively truncates the string at that point.

Path Comparisons
""""""""""""""""

.. signature:: if(<variable|string> PATH_EQUAL <variable|string>)
  :target: PATH_EQUAL

  .. versionadded:: 3.24

  Lexicographically compares two CMake paths component-by-component without
  accessing the filesystem. Only if every component of both paths match will
  the two paths compare equal.  Multiple path separators are effectively
  collapsed into a single separator, but note that backslashes are not
  converted to forward slashes.
  No other :ref:`path normalization <Normalization>` is performed.
  Trailing slashes are preserved, thus ``/a/b`` and ``/a/b/`` are not equal.

  Component-wise comparison is superior to string-based comparison due to the
  handling of multiple path separators.  In the following example, the
  expression evaluates to true using ``PATH_EQUAL``, but false with
  ``STREQUAL``:

  .. code-block:: cmake

    # comparison is TRUE
    if ("/a//b/c" PATH_EQUAL "/a/b/c")
       ...
    endif()

    # comparison is FALSE
    if ("/a//b/c" STREQUAL "/a/b/c")
       ...
    endif()

  See :ref:`cmake_path(COMPARE) <Path Comparison>` for more details.

Variable Expansion
^^^^^^^^^^^^^^^^^^

The if command was written very early in CMake's history, predating
the ``${}`` variable evaluation syntax, and for convenience evaluates
variables named by its arguments as shown in the above signatures.
Note that normal variable evaluation with ``${}`` applies before the if
command even receives the arguments.  Therefore code like

.. code-block:: cmake

 set(var1 OFF)
 set(var2 "var1")
 if(${var2})

appears to the if command as

.. code-block:: cmake

  if(var1)

and is evaluated according to the ``if(<variable>)`` case documented
above.  The result is ``OFF`` which is false.  However, if we remove the
``${}`` from the example then the command sees

.. code-block:: cmake

  if(var2)

which is true because ``var2`` is defined to ``var1`` which is not a false
constant.

Automatic evaluation applies in the other cases whenever the
above-documented condition syntax accepts ``<variable|string>``:

* The left hand argument to `MATCHES`_ is first checked to see if it is
  a defined variable.  If so, the variable's value is used, otherwise the
  original value is used.

* If the left hand argument to `MATCHES`_ is missing it returns false
  without error

* Both left and right hand arguments to `LESS`_, `GREATER`_, `EQUAL`_,
  `LESS_EQUAL`_, and `GREATER_EQUAL`_, are independently tested to see if
  they are defined variables.  If so, their defined values are used otherwise
  the original value is used.

* Both left and right hand arguments to `STRLESS`_, `STRGREATER`_,
  `STREQUAL`_, `STRLESS_EQUAL`_, and `STRGREATER_EQUAL`_ are independently
  tested to see if they are defined variables.  If so, their defined values are
  used otherwise the original value is used.

* Both left and right hand arguments to `VERSION_LESS`_,
  `VERSION_GREATER`_, `VERSION_EQUAL`_, `VERSION_LESS_EQUAL`_, and
  `VERSION_GREATER_EQUAL`_ are independently tested to see if they are defined
  variables.  If so, their defined values are used otherwise the original value
  is used.

* The left hand argument to `IN_LIST`_ is tested to see if it is a defined
  variable.  If so, the variable's value is used, otherwise the original
  value is used.

* The right hand argument to `NOT`_ is tested to see if it is a boolean
  constant.  If so, the value is used, otherwise it is assumed to be a
  variable and it is dereferenced.

* The left and right hand arguments to `AND`_ and `OR`_ are independently
  tested to see if they are boolean constants.  If so, they are used as
  such, otherwise they are assumed to be variables and are dereferenced.

.. versionchanged:: 3.1
  To prevent ambiguity, potential variable or keyword names can be
  specified in a :ref:`Quoted Argument` or a :ref:`Bracket Argument`.
  A quoted or bracketed variable or keyword will be interpreted as a
  string and not dereferenced or interpreted.
  See policy :policy:`CMP0054`.

There is no automatic evaluation for environment or cache
:ref:`Variable References`.  Their values must be referenced as
``$ENV{<name>}`` or ``$CACHE{<name>}`` wherever the above-documented
condition syntax accepts ``<variable|string>``.

See also
^^^^^^^^

* :command:`else`
* :command:`elseif`
* :command:`endif`
