cmake_language
--------------

.. versionadded:: 3.18

Call meta-operations on CMake commands.

Synopsis
^^^^^^^^

.. parsed-literal::

  cmake_language(`CALL`_ <command> [<arg>...])
  cmake_language(`EVAL`_ CODE <code>...)
  cmake_language(`DEFER`_ <options>... CALL <command> [<arg>...])
  cmake_language(`SET_DEPENDENCY_PROVIDER`_ <command> SUPPORTED_METHODS <methods>...)
  cmake_language(`GET_MESSAGE_LOG_LEVEL`_ <out-var>)
  cmake_language(`EXIT`_ <exit-code>)
  cmake_language(`TRACE`_ <boolean> ...)

Introduction
^^^^^^^^^^^^

This command will call meta-operations on built-in CMake commands or
those created via the :command:`macro` or :command:`function` commands.

``cmake_language`` does not introduce a new variable or policy scope.

Calling Commands
^^^^^^^^^^^^^^^^

.. signature::
  cmake_language(CALL <command> [<arg>...])

  Calls the named ``<command>`` with the given arguments (if any).
  For example, the code:

  .. code-block:: cmake

    set(message_command "message")
    cmake_language(CALL ${message_command} STATUS "Hello World!")

  is equivalent to

  .. code-block:: cmake

    message(STATUS "Hello World!")

  .. note::
    To ensure consistency of the code, the following commands are not allowed:

    * ``if`` / ``elseif`` / ``else`` / ``endif``
    * ``block`` / ``endblock``
    * ``while`` / ``endwhile``
    * ``foreach`` / ``endforeach``
    * ``function`` / ``endfunction``
    * ``macro`` / ``endmacro``

Evaluating Code
^^^^^^^^^^^^^^^

.. signature::
  cmake_language(EVAL CODE <code>...)
  :target: EVAL

  Evaluates the ``<code>...`` as CMake code.

  For example, the code:

  .. code-block:: cmake

    set(A TRUE)
    set(B TRUE)
    set(C TRUE)
    set(condition "(A AND B) OR C")

    cmake_language(EVAL CODE "
      if (${condition})
        message(STATUS TRUE)
      else()
        message(STATUS FALSE)
      endif()"
    )

  is equivalent to

  .. code-block:: cmake

    set(A TRUE)
    set(B TRUE)
    set(C TRUE)
    set(condition "(A AND B) OR C")

    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/eval.cmake "
      if (${condition})
        message(STATUS TRUE)
      else()
        message(STATUS FALSE)
      endif()"
    )

    include(${CMAKE_CURRENT_BINARY_DIR}/eval.cmake)

Deferring Calls
^^^^^^^^^^^^^^^

.. versionadded:: 3.19

.. signature::
  cmake_language(DEFER <options>... CALL <command> [<arg>...])

  Schedules a call to the named ``<command>`` with the given arguments (if any)
  to occur at a later time.  By default, deferred calls are executed as if
  written at the end of the current directory's ``CMakeLists.txt`` file,
  except that they run even after a :command:`return` call.  Variable
  references in arguments are evaluated at the time the deferred call is
  executed.

  The options are:

  ``DIRECTORY <dir>``
    Schedule the call for the end of the given directory instead of the
    current directory.  The ``<dir>`` may reference either a source
    directory or its corresponding binary directory.  Relative paths are
    treated as relative to the current source directory.

    The given directory must be known to CMake, being either the top-level
    directory or one added by :command:`add_subdirectory`.  Furthermore,
    the given directory must not yet be finished processing.  This means
    it can be the current directory or one of its ancestors.

  ``ID <id>``
    Specify an identification for the deferred call.
    The ``<id>`` may not be empty and may not begin with a capital letter ``A-Z``.
    The ``<id>`` may begin with an underscore (``_``) only if it was generated
    automatically by an earlier call that used ``ID_VAR`` to get the id.

  ``ID_VAR <var>``
    Specify a variable in which to store the identification for the
    deferred call.  If ``ID <id>`` is not given, a new identification
    will be generated and the generated id will start with an underscore (``_``).

  The currently scheduled list of deferred calls may be retrieved:

  .. code-block:: cmake

    cmake_language(DEFER [DIRECTORY <dir>] GET_CALL_IDS <var>)

  This will store in ``<var>`` a :ref:`semicolon-separated list <CMake Language
  Lists>` of deferred call ids.  The ids are for the directory scope in which
  the calls have been deferred to (i.e. where they will be executed), which can
  be different to the scope in which they were created.  The ``DIRECTORY``
  option can be used to specify the scope for which to retrieve the call ids.
  If that option is not given, the call ids for the current directory scope
  will be returned.

  Details of a specific call may be retrieved from its id:

  .. code-block:: cmake

    cmake_language(DEFER [DIRECTORY <dir>] GET_CALL <id> <var>)

  This will store in ``<var>`` a :ref:`semicolon-separated list <CMake Language
  Lists>` in which the first element is the name of the command to be
  called, and the remaining elements are its unevaluated arguments (any
  contained ``;`` characters are included literally and cannot be distinguished
  from multiple arguments).  If multiple calls are scheduled with the same id,
  this retrieves the first one.  If no call is scheduled with the given id in
  the specified ``DIRECTORY`` scope (or the current directory scope if no
  ``DIRECTORY`` option is given), this stores an empty string in the variable.

  Deferred calls may be canceled by their id:

  .. code-block:: cmake

    cmake_language(DEFER [DIRECTORY <dir>] CANCEL_CALL <id>...)

  This cancels all deferred calls matching any of the given ids in the specified
  ``DIRECTORY`` scope (or the current directory scope if no ``DIRECTORY`` option
  is given).  Unknown ids are silently ignored.

Deferred Call Examples
""""""""""""""""""""""

For example, the code:

.. code-block:: cmake

  cmake_language(DEFER CALL message "${deferred_message}")
  cmake_language(DEFER ID_VAR id CALL message "Canceled Message")
  cmake_language(DEFER CANCEL_CALL ${id})
  message("Immediate Message")
  set(deferred_message "Deferred Message")

prints::

  Immediate Message
  Deferred Message

The ``Canceled Message`` is never printed because its command is
canceled.  The ``deferred_message`` variable reference is not evaluated
until the call site, so it can be set after the deferred call is scheduled.

In order to evaluate variable references immediately when scheduling a
deferred call, wrap it using ``cmake_language(EVAL)``.  However, note that
arguments will be re-evaluated in the deferred call, though that can be
avoided by using bracket arguments.  For example:

.. code-block:: cmake

  set(deferred_message "Deferred Message 1")
  set(re_evaluated [[${deferred_message}]])
  cmake_language(EVAL CODE "
    cmake_language(DEFER CALL message [[${deferred_message}]])
    cmake_language(DEFER CALL message \"${re_evaluated}\")
  ")
  message("Immediate Message")
  set(deferred_message "Deferred Message 2")

also prints::

  Immediate Message
  Deferred Message 1
  Deferred Message 2

.. _dependency_providers:

Dependency Providers
^^^^^^^^^^^^^^^^^^^^

.. versionadded:: 3.24

.. note:: A high-level introduction to this feature can be found in the
          :ref:`Using Dependencies Guide <dependency_providers_overview>`.

.. signature::
  cmake_language(SET_DEPENDENCY_PROVIDER <command>
                 SUPPORTED_METHODS <methods>...)

  When a call is made to :command:`find_package` or
  :command:`FetchContent_MakeAvailable`, the call may be forwarded to a
  dependency provider which then has the opportunity to fulfill the request.
  If the request is for one of the ``<methods>`` specified when the provider
  was set, CMake calls the provider's ``<command>`` with a set of
  method-specific arguments.  If the provider does not fulfill the request,
  or if the provider doesn't support the request's method, or no provider
  is set, the built-in :command:`find_package` or
  :command:`FetchContent_MakeAvailable` implementation is used to fulfill
  the request in the usual way.

  One or more of the following values can be specified for the ``<methods>``
  when setting the provider:

  ``FIND_PACKAGE``
    The provider command accepts :command:`find_package` requests.

  ``FETCHCONTENT_MAKEAVAILABLE_SERIAL``
    The provider command accepts :command:`FetchContent_MakeAvailable`
    requests.  It expects each dependency to be fed to the provider command
    one at a time, not the whole list in one go.

  Only one provider can be set at any point in time.  If a provider is already
  set when ``cmake_language(SET_DEPENDENCY_PROVIDER)`` is called, the new
  provider replaces the previously set one.  The specified ``<command>`` must
  already exist when ``cmake_language(SET_DEPENDENCY_PROVIDER)`` is called.
  As a special case, providing an empty string for the ``<command>`` and no
  ``<methods>`` will discard any previously set provider.

  The dependency provider can only be set while processing one of the files
  specified by the :variable:`CMAKE_PROJECT_TOP_LEVEL_INCLUDES` variable.
  Thus, dependency providers can only be set as part of the first call to
  :command:`project`.  Calling ``cmake_language(SET_DEPENDENCY_PROVIDER)``
  outside of that context will result in an error.

  .. versionadded:: 3.30
    The :prop_gbl:`PROPAGATE_TOP_LEVEL_INCLUDES_TO_TRY_COMPILE` global
    property can be set if the dependency provider also wants to be enabled
    in whole-project calls to :command:`try_compile`.

  .. note::
    The choice of dependency provider should always be under the user's control.
    As a convenience, a project may choose to provide a file that users can
    list in their :variable:`CMAKE_PROJECT_TOP_LEVEL_INCLUDES` variable, but
    the use of such a file should always be the user's choice.

Provider commands
"""""""""""""""""

Providers define a single ``<command>`` to accept requests.  The name of
the command should be specific to that provider, not something overly
generic that another provider might also use.  This enables users to compose
different providers in their own custom provider.  The recommended form is
``xxx_provide_dependency()``, where ``xxx`` is the provider-specific part
(e.g. ``vcpkg_provide_dependency()``, ``conan_provide_dependency()``,
``ourcompany_provide_dependency()``, and so on).

.. code-block:: cmake

  xxx_provide_dependency(<method> [<method-specific-args>...])

Because some methods expect certain variables to be set in the calling scope,
the provider command should typically be implemented as a macro rather than a
function.  This ensures it does not introduce a new variable scope.

The arguments CMake passes to the dependency provider depend on the type of
request.  The first argument is always the method, and it will only ever
be one of the ``<methods>`` that was specified when setting the provider.

``FIND_PACKAGE``
  The ``<method-specific-args>`` will be everything passed to the
  :command:`find_package` call that requested the dependency.  The first of
  these ``<method-specific-args>`` will therefore always be the name of the
  dependency.  Dependency names are case-sensitive for this method because
  :command:`find_package` treats them case-sensitively too.

  If the provider command fulfills the request, it must set the same variable
  that :command:`find_package` expects to be set.  For a dependency named
  ``depName``, the provider must set ``depName_FOUND`` to true if it fulfilled
  the request.  If the provider returns without setting this variable, CMake
  will assume the request was not fulfilled and will fall back to the
  built-in implementation.

  If the provider needs to call the built-in :command:`find_package`
  implementation as part of its processing, it can do so by including the
  ``BYPASS_PROVIDER`` keyword as one of the arguments.

``FETCHCONTENT_MAKEAVAILABLE_SERIAL``
  The ``<method-specific-args>`` will be everything passed to the
  :command:`FetchContent_Declare` call that corresponds to the requested
  dependency, with the following exceptions:

  * If ``SOURCE_DIR`` or ``BINARY_DIR`` were not part of the original
    declared arguments, they will be added with their default values.
  * If :variable:`FETCHCONTENT_TRY_FIND_PACKAGE_MODE` is set to ``NEVER``,
    any ``FIND_PACKAGE_ARGS`` will be omitted.
  * The ``OVERRIDE_FIND_PACKAGE`` keyword is always omitted.

  The first of the ``<method-specific-args>`` will always be the name of the
  dependency.  Dependency names are case-insensitive for this method because
  :module:`FetchContent` also treats them case-insensitively.

  If the provider fulfills the request, it should call
  :command:`FetchContent_SetPopulated`, passing the name of the dependency as
  the first argument.  The ``SOURCE_DIR`` and ``BINARY_DIR`` arguments to that
  command should only be given if the provider makes the dependency's source
  and build directories available in exactly the same way as the built-in
  :command:`FetchContent_MakeAvailable` command.

  If the provider returns without calling :command:`FetchContent_SetPopulated`
  for the named dependency, CMake will assume the request was not fulfilled
  and will fall back to the built-in implementation.

  Note that empty arguments may be significant for this method (e.g. an empty
  string following a ``GIT_SUBMODULES`` keyword).  Therefore, if forwarding
  these arguments on to another command, extra care must be taken to avoid such
  arguments being silently dropped.

  If ``FETCHCONTENT_SOURCE_DIR_<uppercaseDepName>`` is set, then the
  dependency provider will never see requests for the ``<depName>`` dependency
  for this method. When the user sets such a variable, they are explicitly
  overriding where to get that dependency from and are taking on the
  responsibility that their overriding version meets any requirements for that
  dependency and is compatible with whatever else in the project uses it.
  Depending on the value of :variable:`FETCHCONTENT_TRY_FIND_PACKAGE_MODE`
  and whether the ``OVERRIDE_FIND_PACKAGE`` option was given to
  :command:`FetchContent_Declare`, having
  ``FETCHCONTENT_SOURCE_DIR_<uppercaseDepName>`` set may also prevent the
  dependency provider from seeing requests for a ``find_package(depName)``
  call too.

Provider Examples
"""""""""""""""""

This first example only intercepts :command:`find_package` calls.  The
provider command runs an external tool which copies the relevant artifacts
into a provider-specific directory, if that tool knows about the dependency.
It then relies on the built-in implementation to then find those artifacts.
:command:`FetchContent_MakeAvailable` calls would not go through the provider.

.. code-block:: cmake
  :caption: mycomp_provider.cmake

  # Always ensure we have the policy settings this provider expects
  cmake_minimum_required(VERSION 3.24)

  set(MYCOMP_PROVIDER_INSTALL_DIR ${CMAKE_BINARY_DIR}/mycomp_packages
    CACHE PATH "The directory this provider installs packages to"
  )
  # Tell the built-in implementation to look in our area first, unless
  # the find_package() call uses NO_..._PATH options to exclude it
  list(APPEND CMAKE_MODULE_PATH ${MYCOMP_PROVIDER_INSTALL_DIR}/cmake)
  list(APPEND CMAKE_PREFIX_PATH ${MYCOMP_PROVIDER_INSTALL_DIR})

  macro(mycomp_provide_dependency method package_name)
    execute_process(
      COMMAND some_tool ${package_name} --installdir ${MYCOMP_PROVIDER_INSTALL_DIR}
      COMMAND_ERROR_IS_FATAL ANY
    )
  endmacro()

  cmake_language(
    SET_DEPENDENCY_PROVIDER mycomp_provide_dependency
    SUPPORTED_METHODS FIND_PACKAGE
  )

The user would then typically use the above file like so::

  cmake -DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=/path/to/mycomp_provider.cmake ...

The next example demonstrates a provider that accepts both methods, but
only handles one specific dependency.  It enforces providing Google Test
using :module:`FetchContent`, but leaves all other dependencies to be
fulfilled by CMake's built-in implementation.  It accepts a few different
names, which demonstrates one way of working around projects that hard-code
an unusual or undesirable way of adding this particular dependency to the
build.  The example also demonstrates how to use the :command:`list` command
to preserve variables that may be overwritten by a call to
:command:`FetchContent_MakeAvailable`.

.. code-block:: cmake
  :caption: mycomp_provider.cmake

  cmake_minimum_required(VERSION 3.24)

  # Because we declare this very early, it will take precedence over any
  # details the project might declare later for the same thing
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        e2239ee6043f73722e7aa812a459f54a28552929 # release-1.11.0
  )

  # Both FIND_PACKAGE and FETCHCONTENT_MAKEAVAILABLE_SERIAL methods provide
  # the package or dependency name as the first method-specific argument.
  macro(mycomp_provide_dependency method dep_name)
    if("${dep_name}" MATCHES "^(gtest|googletest)$")
      # Save our current command arguments in case we are called recursively
      list(APPEND mycomp_provider_args ${method} ${dep_name})

      # This will forward to the built-in FetchContent implementation,
      # which detects a recursive call for the same thing and avoids calling
      # the provider again if dep_name is the same as the current call.
      FetchContent_MakeAvailable(googletest)

      # Restore our command arguments
      list(POP_BACK mycomp_provider_args dep_name method)

      # Tell the caller we fulfilled the request
      if("${method}" STREQUAL "FIND_PACKAGE")
        # We need to set this if we got here from a find_package() call
        # since we used a different method to fulfill the request.
        # This example assumes projects only use the gtest targets,
        # not any of the variables the FindGTest module may define.
        set(${dep_name}_FOUND TRUE)
      elseif(NOT "${dep_name}" STREQUAL "googletest")
        # We used the same method, but were given a different name to the
        # one we populated with. Tell the caller about the name it used.
        FetchContent_SetPopulated(${dep_name}
          SOURCE_DIR "${googletest_SOURCE_DIR}"
          BINARY_DIR "${googletest_BINARY_DIR}"
        )
      endif()
    endif()
  endmacro()

  cmake_language(
    SET_DEPENDENCY_PROVIDER mycomp_provide_dependency
    SUPPORTED_METHODS
      FIND_PACKAGE
      FETCHCONTENT_MAKEAVAILABLE_SERIAL
  )

The final example demonstrates how to modify arguments to a
:command:`find_package` call.  It forces all such calls to have the
``QUIET`` keyword.  It uses the ``BYPASS_PROVIDER`` keyword to prevent
calling the provider command recursively for the same dependency.

.. code-block:: cmake
  :caption: mycomp_provider.cmake

  cmake_minimum_required(VERSION 3.24)

  macro(mycomp_provide_dependency method)
    find_package(${ARGN} BYPASS_PROVIDER QUIET)
  endmacro()

  cmake_language(
    SET_DEPENDENCY_PROVIDER mycomp_provide_dependency
    SUPPORTED_METHODS FIND_PACKAGE
  )

Getting current message log level
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. versionadded:: 3.25

.. _query_message_log_level:

.. signature::
  cmake_language(GET_MESSAGE_LOG_LEVEL <output_variable>)

  Writes the current :command:`message` logging level
  into the given ``<output_variable>``.

  See :command:`message` for the possible logging levels.

  The current message logging level can be set either using the
  :option:`--log-level <cmake --log-level>`
  command line option of the :manual:`cmake(1)` program or using
  the :variable:`CMAKE_MESSAGE_LOG_LEVEL` variable.

  If both the command line option and the variable are set, the command line
  option takes precedence. If neither are set, the default logging level
  is returned.

Terminating Scripts
^^^^^^^^^^^^^^^^^^^

.. versionadded:: 3.29

.. signature::
  cmake_language(EXIT <exit-code>)

  Terminate the current :option:`cmake -P` script and exit with ``<exit-code>``.

  This command works only in :ref:`script mode <Script Processing Mode>`.
  If used outside of that context, it will cause a fatal error.

  The ``<exit-code>`` should be non-negative.
  If ``<exit-code>`` is negative, then the behavior
  is unspecified (e.g., on Windows the error code -1
  becomes ``0xffffffff``, and on Linux it becomes 255).
  Exit codes above 255 may not be supported by the underlying
  shell or platform, and some shells may interpret values
  above 125 specially.  Therefore, it is advisable to only
  specify an ``<exit-code>`` in the range 0 to 125.

Trace Control
^^^^^^^^^^^^^

.. versionadded:: 4.2

.. signature::
  cmake_language(TRACE ON [EXPAND])
  cmake_language(TRACE OFF)
  :target:
    TRACE
    TRACE-OFF

  The TRACE subcommand controls runtime tracing of executed CMake commands and
  macros within the current process. When enabled, trace output is written
  in the same format as if CMake had been started with the
  :option:`cmake --trace` or :option:`cmake --trace-expand` command line options.

  Tracing scopes are nestable. Multiple ``TRACE ON`` calls may be active at the
  same time, and each ``TRACE OFF`` deactivates one nesting level.

  If CMake is run with :option:`cmake --trace` or :option:`cmake --trace-expand`,
  those options override and force tracing globally, regardless of
  ``cmake_language(TRACE OFF)`` calls. In such cases, the command may still
  be invoked but has no effect on the trace state.
