include_guard
-------------

.. versionadded:: 3.10

Provides an include guard for the file currently being processed by CMake.

.. code-block:: cmake

  include_guard([DIRECTORY|GLOBAL])

Sets up an include guard for the current CMake file (see the
:variable:`CMAKE_CURRENT_LIST_FILE` variable documentation).

CMake will end its processing of the current file at the location of the
``include_guard`` command if the current file has already been
processed for the applicable scope (see below). This provides functionality
similar to the include guards commonly used in source headers or to the
``#pragma once`` directive. If the current file has been processed previously
for the applicable scope, the effect is as though :command:`return` had been
called. Do not call this command from inside a function being defined within
the current file.

An optional argument specifying the scope of the guard may be provided.
Possible values for the option are:

``DIRECTORY``
  The include guard applies within the current directory and below. The file
  will only be included once within this directory scope, but may be included
  again by other files outside of this directory (i.e. a parent directory or
  another directory not pulled in by :command:`add_subdirectory` or
  :command:`include` from the current file or its children).

``GLOBAL``
  The include guard applies globally to the whole build. The current file
  will only be included once regardless of the scope.

If no arguments given, ``include_guard`` has the same scope as a variable,
meaning that the include guard effect is isolated by the most recent
function scope or current directory if no inner function scopes exist.
In this case the command behavior is the same as:

.. code-block:: cmake

  if(__CURRENT_FILE_VAR__)
    return()
  endif()
  set(__CURRENT_FILE_VAR__ TRUE)
