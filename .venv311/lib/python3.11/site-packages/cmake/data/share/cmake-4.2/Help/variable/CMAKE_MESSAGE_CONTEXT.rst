CMAKE_MESSAGE_CONTEXT
---------------------

.. versionadded:: 3.17

When enabled by the :option:`cmake --log-context` command line
option or the :variable:`CMAKE_MESSAGE_CONTEXT_SHOW` variable, the
:command:`message` command converts the ``CMAKE_MESSAGE_CONTEXT`` list into a
dot-separated string surrounded by square brackets and prepends it to each line
for messages of log levels ``NOTICE`` and below.

For logging contexts to work effectively, projects should generally
``APPEND`` and ``POP_BACK`` an item to the current value of
``CMAKE_MESSAGE_CONTEXT`` rather than replace it.
Projects should not assume the message context at the top of the source tree
is empty, as there are scenarios where the context might have already been set
(e.g. hierarchical projects).

.. warning::

  Valid context names are restricted to anything that could be used
  as a CMake variable name.  All names that begin with an underscore
  or the string ``cmake_`` are also reserved for use by CMake and
  should not be used by projects.

Example:

.. code-block:: cmake

  function(bar)
    list(APPEND CMAKE_MESSAGE_CONTEXT "bar")
    message(VERBOSE "bar VERBOSE message")
  endfunction()

  function(baz)
    list(APPEND CMAKE_MESSAGE_CONTEXT "baz")
    message(DEBUG "baz DEBUG message")
  endfunction()

  function(foo)
    list(APPEND CMAKE_MESSAGE_CONTEXT "foo")
    bar()
    message(TRACE "foo TRACE message")
    baz()
  endfunction()

  list(APPEND CMAKE_MESSAGE_CONTEXT "top")

  message(VERBOSE "Before `foo`")
  foo()
  message(VERBOSE "After `foo`")

  list(POP_BACK CMAKE_MESSAGE_CONTEXT)


Which results in the following output:

.. code-block:: none

  -- [top] Before `foo`
  -- [top.foo.bar] bar VERBOSE message
  -- [top.foo] foo TRACE message
  -- [top.foo.baz] baz DEBUG message
  -- [top] After `foo`
