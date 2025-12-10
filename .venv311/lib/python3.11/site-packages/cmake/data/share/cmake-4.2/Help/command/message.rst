message
-------

Log a message.

Synopsis
^^^^^^^^

.. parsed-literal::

  `General messages`_
    message([<mode>] "message text" ...)

  `Reporting checks`_
    message(<checkState> "message text" ...)

  `Configure Log`_
    message(CONFIGURE_LOG <text>...)

General messages
^^^^^^^^^^^^^^^^

.. code-block:: cmake

  message([<mode>] "message text" ...)

Record the specified message text in the log.  If more than one message
string is given, they are concatenated into a single message with no
separator between the strings.

The optional ``<mode>`` keyword determines the type of message, which
influences the way the message is handled:

``FATAL_ERROR``
  CMake Error, stop processing and generation.

  The :manual:`cmake(1)` executable will return a non-zero
  :ref:`exit code <CMake Exit Code>`.

``SEND_ERROR``
  CMake Error, continue processing, but skip generation.

``WARNING``
  CMake Warning, continue processing.

``AUTHOR_WARNING``
  CMake Warning (dev), continue processing.

``DEPRECATION``
  CMake Deprecation Error or Warning if variable
  :variable:`CMAKE_ERROR_DEPRECATED` or :variable:`CMAKE_WARN_DEPRECATED`
  is enabled, respectively, else no message.

(none) or ``NOTICE``
  Important message printed to stderr to attract user's attention.

``STATUS``
  The main interesting messages that project users might be interested in.
  Ideally these should be concise, no more than a single line, but still
  informative.

``VERBOSE``
  Detailed informational messages intended for project users.  These messages
  should provide additional details that won't be of interest in most cases,
  but which may be useful to those building the project when they want deeper
  insight into what's happening.

``DEBUG``
  Detailed informational messages intended for developers working on the
  project itself as opposed to users who just want to build it.  These messages
  will not typically be of interest to other users building the project and
  will often be closely related to internal implementation details.

``TRACE``
  Fine-grained messages with very low-level implementation details.  Messages
  using this log level would normally only be temporary and would expect to be
  removed before releasing the project, packaging up the files, etc.

.. versionadded:: 3.15
  Added the ``NOTICE``, ``VERBOSE``, ``DEBUG``, and ``TRACE`` levels.

The CMake command-line tool displays ``STATUS`` to ``TRACE`` messages on stdout
with the message preceded by two hyphens and a space.  All other message types
are sent to stderr and are not prefixed with hyphens.  The
:manual:`CMake GUI <cmake-gui(1)>` displays all messages in its log area.
The :manual:`curses interface <ccmake(1)>` shows ``STATUS`` to ``TRACE``
messages one at a time on a status line and other messages in an
interactive pop-up box.  The :option:`--log-level <cmake --log-level>`
command-line option to each of these tools can be used to control which
messages will be shown.

.. versionadded:: 3.17
  To make a log level persist between CMake runs, the
  :variable:`CMAKE_MESSAGE_LOG_LEVEL` variable can be set instead.
  Note that the command line option takes precedence over the cache variable.

.. versionadded:: 3.16
  Messages of log levels ``NOTICE`` and below will have each line preceded
  by the content of the :variable:`CMAKE_MESSAGE_INDENT` variable (converted to
  a single string by concatenating its list items).  For ``STATUS`` to ``TRACE``
  messages, this indenting content will be inserted after the hyphens.

.. versionadded:: 3.17
  Messages of log levels ``NOTICE`` and below can also have each line preceded
  with context of the form ``[some.context.example]``.  The content between the
  square brackets is obtained by converting the :variable:`CMAKE_MESSAGE_CONTEXT`
  list variable to a dot-separated string.  The message context will always
  appear before any indenting content but after any automatically added leading
  hyphens. By default, message context is not shown, it has to be explicitly
  enabled by giving the :option:`cmake --log-context`
  command-line option or by setting the :variable:`CMAKE_MESSAGE_CONTEXT_SHOW`
  variable to true.  See the :variable:`CMAKE_MESSAGE_CONTEXT` documentation for
  usage examples.

CMake Warning and Error message text displays using a simple markup
language.  Non-indented text is formatted in line-wrapped paragraphs
delimited by newlines.  Indented text is considered pre-formatted.


Reporting checks
^^^^^^^^^^^^^^^^

.. versionadded:: 3.17

A common pattern in CMake output is a message indicating the start of some
sort of check, followed by another message reporting the result of that check.
For example:

.. code-block:: cmake

  message(STATUS "Looking for someheader.h")
  #... do the checks, set checkSuccess with the result
  if(checkSuccess)
    message(STATUS "Looking for someheader.h - found")
  else()
    message(STATUS "Looking for someheader.h - not found")
  endif()

This can be more robustly and conveniently expressed using the ``CHECK_...``
keyword form of the ``message()`` command:

.. code-block:: cmake

  message(<checkState> "message" ...)

where ``<checkState>`` must be one of the following:

  ``CHECK_START``
    Record a concise message about the check about to be performed.

  ``CHECK_PASS``
    Record a successful result for a check.

  ``CHECK_FAIL``
    Record an unsuccessful result for a check.

When recording a check result, the command repeats the message from the most
recently started check for which no result has yet been reported, then some
separator characters and then the message text provided after the
``CHECK_PASS`` or ``CHECK_FAIL`` keyword.  Check messages are always reported
at ``STATUS`` log level.

Checks may be nested and every ``CHECK_START`` should have exactly one
matching ``CHECK_PASS`` or ``CHECK_FAIL``.
The :variable:`CMAKE_MESSAGE_INDENT` variable can also be used to add
indenting to nested checks if desired.  For example:

.. code-block:: cmake

  message(CHECK_START "Finding my things")
  list(APPEND CMAKE_MESSAGE_INDENT "  ")
  unset(missingComponents)

  message(CHECK_START "Finding partA")
  # ... do check, assume we find A
  message(CHECK_PASS "found")

  message(CHECK_START "Finding partB")
  # ... do check, assume we don't find B
  list(APPEND missingComponents B)
  message(CHECK_FAIL "not found")

  list(POP_BACK CMAKE_MESSAGE_INDENT)
  if(missingComponents)
    message(CHECK_FAIL "missing components: ${missingComponents}")
  else()
    message(CHECK_PASS "all components found")
  endif()

Output from the above would appear something like the following::

  -- Finding my things
  --   Finding partA
  --   Finding partA - found
  --   Finding partB
  --   Finding partB - not found
  -- Finding my things - missing components: B

Configure Log
^^^^^^^^^^^^^

.. versionadded:: 3.26

.. code-block:: cmake

  message(CONFIGURE_LOG <text>...)

Record a :ref:`configure-log message event <message configure-log event>`
with the specified ``<text>``.  By convention, if the text contains more
than one line, the first line should be a summary of the event.

This mode is intended to record the details of a system inspection check
or other one-time operation guarded by a cache entry, but that is not
performed using :command:`try_compile` or :command:`try_run`, which
automatically log their details.  Projects should avoid calling it every
time CMake runs.  For example:

.. code-block:: cmake

  if (NOT DEFINED MY_CHECK_RESULT)
    # Print check summary in configure output.
    message(CHECK_START "My Check")

    # ... perform system inspection, e.g., with execute_process ...

    # Cache the result so we do not run the check again.
    set(MY_CHECK_RESULT "${MY_CHECK_RESULT}" CACHE INTERNAL "My Check")

    # Record the check details in the cmake-configure-log.
    message(CONFIGURE_LOG
      "My Check Result: ${MY_CHECK_RESULT}\n"
      "${details}"
    )

    # Print check result in configure output.
    if(MY_CHECK_RESULT)
      message(CHECK_PASS "passed")
    else()
      message(CHECK_FAIL "failed")
    endif()
  endif()

If no project is currently being configured, such as in
:ref:`cmake -P <Script Processing Mode>` script mode,
this command does nothing.

See Also
^^^^^^^^

* :command:`cmake_language(GET_MESSAGE_LOG_LEVEL)`
