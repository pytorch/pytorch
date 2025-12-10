CMAKE_MESSAGE_CONTEXT_SHOW
--------------------------

.. versionadded:: 3.17

Setting this variable to true enables showing a context with each line
logged by the :command:`message` command (see :variable:`CMAKE_MESSAGE_CONTEXT`
for how the context itself is specified).

This variable is an alternative to providing the ``--log-context`` option
on the :manual:`cmake <cmake(1)>` command line.  Whereas the command line
option will apply only to that one CMake run, setting
``CMAKE_MESSAGE_CONTEXT_SHOW`` to true as a cache variable will ensure that
subsequent CMake runs will continue to show the message context.

Projects should not set ``CMAKE_MESSAGE_CONTEXT_SHOW``.  It is intended for
users so that they may control whether or not to include context with messages.
