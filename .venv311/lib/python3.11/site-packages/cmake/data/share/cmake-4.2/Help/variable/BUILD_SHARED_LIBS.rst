BUILD_SHARED_LIBS
-----------------

Tell :command:`add_library` to default to ``SHARED`` libraries,
instead of ``STATIC`` libraries, when called with no explicit library type.

Calls to :command:`add_library` without any explicit library type check
the current ``BUILD_SHARED_LIBS`` variable value.  If it is true, then the
default library type is ``SHARED``.  Otherwise, the default is ``STATIC``.

For example, the code:

.. code-block:: cmake

  add_library(example ${sources})

behaves as if written

.. code-block:: cmake

  if(BUILD_SHARED_LIBS)
    add_library(example SHARED ${sources})
  else()
    add_library(example STATIC ${sources})
  endif()

CMake does not define ``BUILD_SHARED_LIBS`` by default, but projects
often create a cache entry for it using the :command:`option` command:

.. code-block:: cmake

  option(BUILD_SHARED_LIBS "Build using shared libraries" ON)

This provides a switch that users can control, e.g., with :option:`cmake -D`.
If adding such an option to the project, do so in the top level
``CMakeLists.txt`` file, before any :command:`add_library` calls.
Note that if bringing external dependencies directly into the build, such as
with :module:`FetchContent` or a direct call to :command:`add_subdirectory`,
and one of those dependencies has such a call to
:command:`option(BUILD_SHARED_LIBS ...) <option>`, the top level project must
also call :command:`option(BUILD_SHARED_LIBS ...) <option>` before bringing in
its dependencies.  Failure to do so can lead to different behavior between the
first and subsequent CMake runs.
