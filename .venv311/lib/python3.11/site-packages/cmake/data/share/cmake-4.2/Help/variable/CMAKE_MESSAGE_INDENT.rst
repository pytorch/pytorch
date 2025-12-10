CMAKE_MESSAGE_INDENT
--------------------

.. versionadded:: 3.16

The :command:`message` command joins the strings from this list and for
log levels of ``NOTICE`` and below, it prepends the resultant string to
each line of the message.

Example:

.. code-block:: cmake

  list(APPEND listVar one two three)

  message(VERBOSE [[Collected items in the "listVar":]])
  list(APPEND CMAKE_MESSAGE_INDENT "  ")

  foreach(item IN LISTS listVar)
    message(VERBOSE ${item})
  endforeach()

  list(POP_BACK CMAKE_MESSAGE_INDENT)
  message(VERBOSE "No more indent")

Which results in the following output:

.. code-block:: none

  -- Collected items in the "listVar":
  --   one
  --   two
  --   three
  -- No more indent
