# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FetchContent
------------------

.. versionadded:: 3.11

.. only:: html

  .. contents::

This module provides commands to populate content at configure time or as
part of the calling script.

Load this module in CMake with:

.. code-block:: cmake

  include(FetchContent)

.. note:: The :guide:`Using Dependencies Guide` provides a high-level
  introduction to this general topic. It provides a broader overview of
  where the ``FetchContent`` module fits into the bigger picture,
  including its relationship to the :command:`find_package` command.
  The guide is recommended pre-reading before moving on to the details below.

Overview
^^^^^^^^

This module enables populating content at configure time via any method
supported by the :module:`ExternalProject` module.  Whereas
:command:`ExternalProject_Add` downloads at build time, the
``FetchContent`` module makes content available immediately, allowing the
configure step to use the content in commands like :command:`add_subdirectory`,
:command:`include` or :command:`file` operations.

Content population details should be defined separately from the command that
performs the actual population.  This separation ensures that all the
dependency details are defined before anything might try to use them to
populate content.  This is particularly important in more complex project
hierarchies where dependencies may be shared between multiple projects.

The following shows a typical example of declaring content details for some
dependencies and then ensuring they are populated with a separate call:

.. code-block:: cmake

  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        703bd9caab50b139428cea1aaff9974ebee5742e # release-1.10.0
  )
  FetchContent_Declare(
    myCompanyIcons
    URL      https://intranet.mycompany.com/assets/iconset_1.12.tar.gz
    URL_HASH MD5=5588a7b18261c20068beabfb4f530b87
  )

  FetchContent_MakeAvailable(googletest myCompanyIcons)

The :command:`FetchContent_MakeAvailable` command ensures the named
dependencies have been populated, either by an earlier call, or by populating
them itself.  When performing the population, it will also add them to the
main build, if possible, so that the main build can use the populated
projects' targets, etc.  See the command's documentation for how these steps
are performed.

When using a hierarchical project arrangement, projects at higher levels in
the hierarchy are able to override the declared details of content specified
anywhere lower in the project hierarchy.  The first details to be declared
for a given dependency take precedence, regardless of where in the project
hierarchy that occurs.  Similarly, the first call that tries to populate a
dependency "wins", with subsequent populations reusing the result of the
first instead of repeating the population again.
See the :ref:`Examples <fetch-content-examples>` which demonstrate
this scenario.

The ``FetchContent`` module also supports defining and populating
content in a single call, with no check for whether the content has been
populated elsewhere already.  This should not be done in projects, but may
be appropriate for populating content in
:ref:`CMake script mode <Script Processing Mode>`.
See :command:`FetchContent_Populate` for details.

Commands
^^^^^^^^

.. command:: FetchContent_Declare

  .. code-block:: cmake

    FetchContent_Declare(
      <name>
      <contentOptions>...
      [EXCLUDE_FROM_ALL]
      [SYSTEM]
      [OVERRIDE_FIND_PACKAGE |
       FIND_PACKAGE_ARGS args...]
    )

  The ``FetchContent_Declare()`` function records the options that describe
  how to populate the specified content.  If such details have already
  been recorded earlier in this project (regardless of where in the project
  hierarchy), this and all later calls for the same content ``<name>`` are
  ignored.  This "first to record, wins" approach is what allows hierarchical
  projects to have parent projects override content details of child projects.

  The content ``<name>`` can be any string without spaces, but good practice
  would be to use only letters, numbers, and underscores.  The name will be
  treated case-insensitively, and it should be obvious for the content it
  represents. It is often the name of the child project, or the value given
  to its top level :command:`project` command (if it is a CMake project).
  For well-known public projects, the name should generally be the official
  name of the project.  Choosing an unusual name makes it unlikely that other
  projects needing that same content will use the same name, leading to
  the content being populated multiple times.

  The ``<contentOptions>`` can be any of the download, update, or patch options
  that the :command:`ExternalProject_Add` command understands.  The configure,
  build, install, and test steps are explicitly disabled, so options related
  to those steps are prohibited and will be discarded if given.
  The ``SOURCE_SUBDIR`` option is an exception, see
  :command:`FetchContent_MakeAvailable` for details on how that affects
  behavior.

  .. versionchanged:: 3.30
    When policy :policy:`CMP0168` is set to ``NEW``, some output-related and
    directory-related options are ignored.  See the policy documentation for
    details.

  In most cases, ``<contentOptions>`` will just be a couple of options defining
  the download method and method-specific details like a commit tag or archive
  hash.  For example:

  .. code-block:: cmake

    FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG        703bd9caab50b139428cea1aaff9974ebee5742e # release-1.10.0
    )

    FetchContent_Declare(
      myCompanyIcons
      URL      https://intranet.mycompany.com/assets/iconset_1.12.tar.gz
      URL_HASH MD5=5588a7b18261c20068beabfb4f530b87
    )

    FetchContent_Declare(
      myCompanyCertificates
      SVN_REPOSITORY svn+ssh://svn.mycompany.com/srv/svn/trunk/certs
      SVN_REVISION   -r12345
    )

  Where contents are being fetched from a remote location and you do not
  control that server, it is advisable to use a hash for ``GIT_TAG`` rather
  than a branch or tag name.  A commit hash is more secure and helps to
  confirm that the downloaded contents are what you expected.

  .. versionchanged:: 3.14
    Commands for the download, update, or patch steps can access the terminal.
    This may be needed for things like password prompts or real-time display
    of command progress.

  .. versionadded:: 3.22
    The :variable:`CMAKE_TLS_VERIFY`, :variable:`CMAKE_TLS_CAINFO`,
    :variable:`CMAKE_NETRC`, and :variable:`CMAKE_NETRC_FILE` variables now
    provide the defaults for their corresponding content options, just like
    they do for :command:`ExternalProject_Add`. Previously, these variables
    were ignored by the ``FetchContent`` module.

  .. versionadded:: 3.24

    ``FIND_PACKAGE_ARGS``
      This option is for scenarios where the
      :command:`FetchContent_MakeAvailable` command may first try a call to
      :command:`find_package` to satisfy the dependency for ``<name>``.
      By default, such a call would be simply ``find_package(<name>)``, but
      ``FIND_PACKAGE_ARGS`` can be used to provide additional arguments to be
      appended after the ``<name>``.  ``FIND_PACKAGE_ARGS`` can also be given
      with nothing after it, which indicates that :command:`find_package` can
      still be called if :variable:`FETCHCONTENT_TRY_FIND_PACKAGE_MODE` is
      set to ``OPT_IN``, or is not set.

      It would not normally be appropriate to specify ``REQUIRED`` as one of
      the additional arguments after ``FIND_PACKAGE_ARGS``.  Doing so would
      mean the :command:`find_package` call must succeed, so none of the other
      details specified in the ``FetchContent_Declare()`` call would get a
      chance to be used as a fall-back.

      Everything after the ``FIND_PACKAGE_ARGS`` keyword is appended to the
      :command:`find_package` call, so all other ``<contentOptions>`` must
      come before the ``FIND_PACKAGE_ARGS`` keyword.  If the
      :variable:`CMAKE_FIND_PACKAGE_TARGETS_GLOBAL` variable is set to true
      at the time ``FetchContent_Declare()`` is called, a ``GLOBAL`` keyword
      will be appended to the :command:`find_package` arguments if it was
      not already specified.  It will also be appended if
      ``FIND_PACKAGE_ARGS`` was not given, but
      :variable:`FETCHCONTENT_TRY_FIND_PACKAGE_MODE` was set to ``ALWAYS``.

      ``OVERRIDE_FIND_PACKAGE`` cannot be used when ``FIND_PACKAGE_ARGS`` is
      given.

      :ref:`dependency_providers` discusses another way that
      :command:`FetchContent_MakeAvailable` calls can be redirected.
      ``FIND_PACKAGE_ARGS`` is intended for project control, whereas
      dependency providers allow users to override project behavior.

    ``OVERRIDE_FIND_PACKAGE``
      When a ``FetchContent_Declare(<name> ...)`` call includes this option,
      subsequent calls to ``find_package(<name> ...)`` will ensure that
      ``FetchContent_MakeAvailable(<name>)`` has been called, then use the
      config package files in the :variable:`CMAKE_FIND_PACKAGE_REDIRECTS_DIR`
      directory (which are usually created by ``FetchContent_MakeAvailable()``).
      This effectively makes :command:`FetchContent_MakeAvailable` override
      :command:`find_package` for the named dependency, allowing the former to
      satisfy the package requirements of the latter.  ``FIND_PACKAGE_ARGS``
      cannot be used when ``OVERRIDE_FIND_PACKAGE`` is given.

      If a :ref:`dependency provider <dependency_providers>` has been set
      and the project calls :command:`find_package` for the ``<name>``
      dependency, ``OVERRIDE_FIND_PACKAGE`` will not prevent the provider
      from seeing that call.  Dependency providers always have the opportunity
      to intercept any direct call to :command:`find_package`, except if that
      call contains the ``BYPASS_PROVIDER`` option.

  .. versionadded:: 3.25

    ``SYSTEM``
      If the ``SYSTEM`` argument is provided, the :prop_dir:`SYSTEM` directory
      property of a subdirectory added by
      :command:`FetchContent_MakeAvailable` will be set to true.  This will
      affect non-imported targets created as part of that command.
      See the :prop_tgt:`SYSTEM` target property documentation for a more
      detailed discussion of the effects.

  .. versionadded:: 3.28

    ``EXCLUDE_FROM_ALL``
      If the ``EXCLUDE_FROM_ALL`` argument is provided, then targets in the
      subdirectory added by :command:`FetchContent_MakeAvailable` will not be
      included in the ``ALL`` target by default, and may be excluded from IDE
      project files. See the documentation for the directory property
      :prop_dir:`EXCLUDE_FROM_ALL` for a detailed discussion of the effects.

.. command:: FetchContent_MakeAvailable

  .. versionadded:: 3.14

  .. code-block:: cmake

    FetchContent_MakeAvailable(<name1> [<name2>...])

  This command ensures that each of the named dependencies are made available
  to the project by the time it returns.  There must have been a call to
  :command:`FetchContent_Declare` for each dependency, and the first such call
  will control how that dependency will be made available, as described below.

  If ``<lowercaseName>_SOURCE_DIR`` is not set:

  * .. versionadded:: 3.24

      If a :ref:`dependency provider <dependency_providers>` is set, call the
      provider's command with ``FETCHCONTENT_MAKEAVAILABLE_SERIAL`` as the
      first argument, followed by the arguments of the first call to
      :command:`FetchContent_Declare` for ``<name>``.  If ``SOURCE_DIR`` or
      ``BINARY_DIR`` were not part of the original declared arguments, they
      will be added with their default values.
      If :variable:`FETCHCONTENT_TRY_FIND_PACKAGE_MODE` was set to ``NEVER``
      when the details were declared, any ``FIND_PACKAGE_ARGS`` will be
      omitted.  The ``OVERRIDE_FIND_PACKAGE`` keyword is also always omitted.
      If the provider fulfilled the request, ``FetchContent_MakeAvailable()``
      will consider that dependency handled, skip the remaining steps below,
      and move on to the next dependency in the list.

  * .. versionadded:: 3.24

      If permitted, :command:`find_package(<name> [<args>...]) <find_package>`
      will be called, where ``<args>...`` may be provided by the
      ``FIND_PACKAGE_ARGS`` option in :command:`FetchContent_Declare`.
      The value of the :variable:`FETCHCONTENT_TRY_FIND_PACKAGE_MODE` variable
      at the time :command:`FetchContent_Declare` was called determines whether
      ``FetchContent_MakeAvailable()`` can call :command:`find_package`.
      If the :variable:`CMAKE_FIND_PACKAGE_TARGETS_GLOBAL` variable is set to
      true when ``FetchContent_MakeAvailable()`` is called, it still affects
      any imported targets created when that in turn calls
      :command:`find_package`, even if that variable was false when the
      corresponding details were declared.

  If the dependency was not satisfied by a provider or a
  :command:`find_package` call, ``FetchContent_MakeAvailable()`` then uses
  the following logic to make the dependency available:

  * If the dependency has already been populated earlier in this run, set
    the ``<lowercaseName>_POPULATED``, ``<lowercaseName>_SOURCE_DIR``, and
    ``<lowercaseName>_BINARY_DIR`` variables in the same way as a call to
    :command:`FetchContent_GetProperties`, then skip the remaining steps
    below and move on to the next dependency in the list.

  * Populate the dependency using the details recorded by an earlier call
    to :command:`FetchContent_Declare`.
    Halt with a fatal error if no such details have been recorded.
    :variable:`FETCHCONTENT_SOURCE_DIR_<uppercaseName>` can be used to override
    the declared details and use content provided at the specified location
    instead.

  * .. versionadded:: 3.24

      Ensure the :variable:`CMAKE_FIND_PACKAGE_REDIRECTS_DIR` directory
      contains a ``<lowercaseName>-config.cmake`` and a
      ``<lowercaseName>-config-version.cmake`` file (or equivalently,
      ``<name>Config.cmake`` and ``<name>ConfigVersion.cmake``).
      The directory that the :variable:`CMAKE_FIND_PACKAGE_REDIRECTS_DIR`
      variable points to is cleared at the start of every CMake run.
      If no config file exists after populating the dependency in the previous
      step, a minimal one will be written which :command:`includes <include>`
      any ``<lowercaseName>-extra.cmake`` or ``<name>Extra.cmake`` file with
      the ``OPTIONAL`` flag (so the files can be missing and won't generate a
      warning).  Similarly, if no config version file exists, a very simple
      one will be written which sets ``PACKAGE_VERSION_COMPATIBLE`` and
      ``PACKAGE_VERSION_EXACT`` to true.  This ensures all future calls to
      :command:`find_package()` for the dependency will use the redirected
      config file, regardless of any version requirements.
      CMake cannot automatically determine an arbitrary dependency's version,
      so it cannot set ``PACKAGE_VERSION``.
      When a dependency is pulled in via :command:`add_subdirectory` in the
      next step, it may choose to overwrite the generated config version file
      in :variable:`CMAKE_FIND_PACKAGE_REDIRECTS_DIR` with one that also sets
      ``PACKAGE_VERSION``.
      The dependency may also write a ``<lowercaseName>-extra.cmake`` or
      ``<name>Extra.cmake`` file to perform custom processing, or define any
      variables that their normal (installed) package config file would
      otherwise usually define (many projects don't do any custom processing
      or set any variables and therefore have no need to do this).
      If required, the main project can write these files instead if the
      dependency project doesn't do so.  This allows the main project to
      add missing details from older dependencies that haven't or can't be
      updated to support this functionality.
      See `Integrating With find_package()`_ for examples.

  * If the top directory of the populated content contains a ``CMakeLists.txt``
    file, call :command:`add_subdirectory` to add it to the main build.
    It is not an error for there to be no ``CMakeLists.txt`` file, which
    allows the command to be used for dependencies that make downloaded
    content available at a known location, but which do not need or support
    being added directly to the build.

    .. versionadded:: 3.18
      The ``SOURCE_SUBDIR`` option can be given in the declared details to
      look somewhere below the top directory instead (i.e. the same way that
      ``SOURCE_SUBDIR`` is used by the :command:`ExternalProject_Add`
      command).  The path provided with ``SOURCE_SUBDIR`` must be relative,
      and it will be treated as relative to the top directory.  It can also
      point to a directory that does not contain a ``CMakeLists.txt`` file,
      or even to a directory that doesn't exist.  This can be used to avoid
      adding a project that contains a ``CMakeLists.txt`` file in its top
      directory.

    .. versionadded:: 3.25
      If the ``SYSTEM`` keyword was included in the call to
      :command:`FetchContent_Declare`, the ``SYSTEM`` keyword will be
      added to the :command:`add_subdirectory` command.

    .. versionadded:: 3.28
      If the ``EXCLUDE_FROM_ALL`` keyword was included in the call to
      :command:`FetchContent_Declare`, the ``EXCLUDE_FROM_ALL`` keyword will
      be added to the :command:`add_subdirectory` command.

    .. versionadded:: 3.29
      :variable:`CMAKE_EXPORT_FIND_PACKAGE_NAME` is set to the dependency name
      before calling :command:`add_subdirectory`.

  Projects should aim to declare the details of all dependencies they might
  use before they call ``FetchContent_MakeAvailable()`` for any of them.
  This ensures that if any of the dependencies are also sub-dependencies of
  one or more of the others, the main project still controls the details
  that will be used (because it will declare them first before the
  dependencies get a chance to).  In the following code samples, assume that
  the ``uses_other`` dependency also uses ``FetchContent`` to add the ``other``
  dependency internally:

  .. code-block:: cmake

    # WRONG: Should declare all details first
    FetchContent_Declare(uses_other ...)
    FetchContent_MakeAvailable(uses_other)

    FetchContent_Declare(other ...)    # Will be ignored, uses_other beat us to it
    FetchContent_MakeAvailable(other)  # Would use details declared by uses_other

  .. code-block:: cmake

    # CORRECT: All details declared first, so they will take priority
    FetchContent_Declare(uses_other ...)
    FetchContent_Declare(other ...)
    FetchContent_MakeAvailable(uses_other other)

  Note that :variable:`CMAKE_VERIFY_INTERFACE_HEADER_SETS` is explicitly set
  to false upon entry to ``FetchContent_MakeAvailable()``, and is restored to
  its original value before the command returns.  Developers typically only
  want to verify header sets from the main project, not those from any
  dependencies.  This local manipulation of the
  :variable:`CMAKE_VERIFY_INTERFACE_HEADER_SETS` variable provides that
  intuitive behavior.  You can use variables like
  :variable:`CMAKE_PROJECT_INCLUDE` or
  :variable:`CMAKE_PROJECT_<PROJECT-NAME>_INCLUDE` to turn verification back
  on for all or some dependencies.  You can also set the
  :prop_tgt:`VERIFY_INTERFACE_HEADER_SETS` property of individual targets.

.. command:: FetchContent_Populate

  The ``FetchContent_Populate()`` command is a self-contained call which can
  be used to perform content population as an isolated operation.
  It is rarely the right command to use, projects should almost always use
  :command:`FetchContent_Declare` and :command:`FetchContent_MakeAvailable`
  instead. The main use case for ``FetchContent_Populate()`` is in
  :ref:`CMake script mode <Script Processing Mode>` as part of implementing
  some other higher level custom feature.

  .. code-block:: cmake

    FetchContent_Populate(
      <name>
      [QUIET]
      [SUBBUILD_DIR <subBuildDir>]
      [SOURCE_DIR <srcDir>]
      [BINARY_DIR <binDir>]
      ...
    )

  At least one option must be specified after `<name>`, otherwise the call
  is interpreted differently (see :ref:`below <FetchContent_Populate-depName>`).
  The supported options for ``FetchContent_Populate()`` are the same as those
  for :command:`FetchContent_Declare()`, with a few exceptions. The following
  do not relate to populating content with ``FetchContent_Populate()`` and
  therefore are not supported:

  * ``EXCLUDE_FROM_ALL``
  * ``SYSTEM``
  * ``OVERRIDE_FIND_PACKAGE``
  * ``FIND_PACKAGE_ARGS``

  The few options shown in the signature above are either specific to
  ``FetchContent_Populate()``, or their behavior is slightly modified from how
  :command:`ExternalProject_Add` treats them:

  ``QUIET``
    The ``QUIET`` option can be given to hide the output associated with
    populating the specified content.  If the population fails, the output will
    be shown regardless of whether this option was given or not so that the
    cause of the failure can be diagnosed.  The :variable:`FETCHCONTENT_QUIET`
    variable has no effect on ``FetchContent_Populate()`` calls of this form
    where the content details are provided directly.

    .. versionchanged:: 3.30
      The ``QUIET`` option and :variable:`FETCHCONTENT_QUIET` variable have no
      effect when policy :policy:`CMP0168` is set to ``NEW``. The output is
      still quiet by default in that case, but verbosity is controlled by the
      message logging level (see :variable:`CMAKE_MESSAGE_LOG_LEVEL` and
      :option:`--log-level <cmake --log-level>`).

  ``SUBBUILD_DIR``
    The ``SUBBUILD_DIR`` argument can be provided to change the location of the
    sub-build created to perform the population.  The default value is
    ``${CMAKE_CURRENT_BINARY_DIR}/<lowercaseName>-subbuild``, and it would be
    unusual to need to override this default.  If a relative path is specified,
    it will be interpreted as relative to :variable:`CMAKE_CURRENT_BINARY_DIR`.
    This option should not be confused with the ``SOURCE_SUBDIR`` option, which
    only affects the :command:`FetchContent_MakeAvailable` command.

    .. versionchanged:: 3.30
      ``SUBBUILD_DIR`` is ignored when policy :policy:`CMP0168` is set to
      ``NEW``, since there is no sub-build in that case.

  ``SOURCE_DIR``, ``BINARY_DIR``
    The ``SOURCE_DIR`` and ``BINARY_DIR`` arguments are supported by
    :command:`ExternalProject_Add`, but different default values are used by
    ``FetchContent_Populate()``.  ``SOURCE_DIR`` defaults to
    ``${CMAKE_CURRENT_BINARY_DIR}/<lowercaseName>-src``, and ``BINARY_DIR``
    defaults to ``${CMAKE_CURRENT_BINARY_DIR}/<lowercaseName>-build``.
    If a relative path is specified, it will be interpreted as relative to
    :variable:`CMAKE_CURRENT_BINARY_DIR`.

  In addition to the above explicit options, any other unrecognized options are
  passed through unmodified to :command:`ExternalProject_Add` to set up the
  download, patch, and update steps.  The following options are explicitly
  prohibited (they are disabled by the ``FetchContent_Populate()`` command):

  - ``CONFIGURE_COMMAND``
  - ``BUILD_COMMAND``
  - ``INSTALL_COMMAND``
  - ``TEST_COMMAND``

  With this form, the :variable:`FETCHCONTENT_FULLY_DISCONNECTED` and
  :variable:`FETCHCONTENT_UPDATES_DISCONNECTED` variables and policy
  :policy:`CMP0170` are ignored.

  When this form of ``FetchContent_Populate()`` returns, the following
  variables will be set in the scope of the caller:

  ``<lowercaseName>_SOURCE_DIR``
    The location where the populated content can be found upon return.

  ``<lowercaseName>_BINARY_DIR``
    A directory originally intended for use as a corresponding build directory,
    but is unlikely to be relevant when using this form of the command.

  If using ``FetchContent_Populate()`` within
  :ref:`CMake script mode <Script Processing Mode>`, be aware that the
  implementation sets up a sub-build which therefore requires a CMake
  generator and build tool to be available. If these cannot be found by
  default, then the :variable:`CMAKE_GENERATOR` and potentially the
  :variable:`CMAKE_MAKE_PROGRAM` variables will need to be set appropriately
  on the command line invoking the script.

  .. versionchanged:: 3.30
    If policy :policy:`CMP0168` is set to ``NEW``, no sub-build is used.
    Within :ref:`CMake script mode <Script Processing Mode>`, that allows
    ``FetchContent_Populate()`` to be called without any build tool or
    CMake generator.

  .. versionadded:: 3.18
    Added support for the ``DOWNLOAD_NO_EXTRACT`` option.

.. _`FetchContent_Populate-depName`:

  The command supports another form, although it should no longer be used:

  .. code-block:: cmake

    FetchContent_Populate(<name>)

  .. versionchanged:: 3.30
    This form is deprecated. Policy :policy:`CMP0169` provides backward
    compatibility for projects that still need to use this form, but projects
    should be updated to use :command:`FetchContent_MakeAvailable` instead.

  In this form, the only argument given to ``FetchContent_Populate()`` is the
  ``<name>``.  When used this way, the command assumes the content details have
  been recorded by an earlier call to :command:`FetchContent_Declare`.  The
  details are stored in a global property, so they are unaffected by things
  like variable or directory scope.  Therefore, it doesn't matter where in the
  project the details were previously declared, as long as they have been
  declared before the call to ``FetchContent_Populate()``.  Those saved details
  are then used to populate the content using a method based on
  :command:`ExternalProject_Add` (see policy :policy:`CMP0168` for important
  behavioral aspects of how that is done).

  When this form of ``FetchContent_Populate()`` returns, the following
  variables will be set in the scope of the caller:

  ``<lowercaseName>_POPULATED``
    This will always be set to ``TRUE`` by the call.

  ``<lowercaseName>_SOURCE_DIR``
    The location where the populated content can be found upon return.

  ``<lowercaseName>_BINARY_DIR``
    A directory intended for use as a corresponding build directory.

  The values of the three variables can also be retrieved from anywhere in the
  project hierarchy using the :command:`FetchContent_GetProperties` command.

  The implementation ensures that if the content has already been populated
  in a previous CMake run, that content will be reused rather than repopulating
  again.  For the common case where population involves downloading content,
  the cost of the download is only paid once. But note that it is an error to
  call ``FetchContent_Populate(<name>)`` with the same ``<name>`` more than
  once within a single CMake run. See :command:`FetchContent_GetProperties`
  for how to determine if population of a ``<name>`` has already been
  performed in the current run.

.. command:: FetchContent_GetProperties

  When using saved content details, a call to
  :command:`FetchContent_MakeAvailable` or :command:`FetchContent_Populate`
  records information in global properties which can be queried at any time.
  This information may include the source and binary directories associated with
  the content, and also whether or not the content population has been processed
  during the current configure run.

  .. code-block:: cmake

    FetchContent_GetProperties(
      <name>
      [SOURCE_DIR <srcDirVar>]
      [BINARY_DIR <binDirVar>]
      [POPULATED <doneVar>]
    )

  The ``SOURCE_DIR``, ``BINARY_DIR``, and ``POPULATED`` options can be used to
  specify which properties should be retrieved.  Each option accepts a value
  which is the name of the variable in which to store that property.  Most of
  the time though, only ``<name>`` is given, in which case the call will then
  set the same variables as a call to
  :command:`FetchContent_MakeAvailable(name) <FetchContent_MakeAvailable>` or
  :command:`FetchContent_Populate(name) <FetchContent_Populate>`.
  Note that the ``SOURCE_DIR`` and ``BINARY_DIR`` values can be empty if the
  call is fulfilled by a :ref:`dependency provider <dependency_providers>`.

  This command is rarely needed when using
  :command:`FetchContent_MakeAvailable`.  It is more commonly used as part of
  implementing the deprecated pattern with :command:`FetchContent_Populate`,
  which ensures that the relevant variables will always be defined regardless
  of whether or not the population has been performed elsewhere in the project
  already:

  .. code-block:: cmake

    # WARNING: This pattern is deprecated, don't use it!
    #
    # Check if population has already been performed
    FetchContent_GetProperties(depname)
    if(NOT depname_POPULATED)
      # Fetch the content using previously declared details
      FetchContent_Populate(depname)

      # Set custom variables, policies, etc.
      # ...

      # Bring the populated content into the build
      add_subdirectory(${depname_SOURCE_DIR} ${depname_BINARY_DIR})
    endif()

.. command:: FetchContent_SetPopulated

  .. versionadded:: 3.24

  .. note::
    This command should only be called by
    :ref:`dependency providers <dependency_providers>`.  Calling it in any
    other context is unsupported and future CMake versions may halt with a
    fatal error in such cases.

  .. code-block:: cmake

    FetchContent_SetPopulated(
      <name>
      [SOURCE_DIR <srcDir>]
      [BINARY_DIR <binDir>]
    )

  If a provider command fulfills a ``FETCHCONTENT_MAKEAVAILABLE_SERIAL``
  request, it must call this function before returning.  The ``SOURCE_DIR``
  and ``BINARY_DIR`` arguments can be used to specify the values that
  :command:`FetchContent_GetProperties` should return for its corresponding
  arguments.  Only provide ``SOURCE_DIR`` and ``BINARY_DIR`` if they have
  the same meaning as if they had been populated by the built-in
  :command:`FetchContent_MakeAvailable` implementation.


Variables
^^^^^^^^^

A number of cache variables can influence the behavior where details from a
:command:`FetchContent_Declare` call are used to populate content.

.. note::
  All of these variables are intended for the developer to customize behavior.
  They should not normally be set by the project.

.. variable:: FETCHCONTENT_BASE_DIR

  In most cases, the saved details do not specify any options relating to the
  directories to use for the internal sub-build, final source, and build areas.
  It is generally best to leave these decisions up to the ``FetchContent``
  module to handle on the project's behalf.  The ``FETCHCONTENT_BASE_DIR``
  cache variable controls the point under which all content population
  directories are collected, but in most cases, developers would not need to
  change this.  The default location is ``${CMAKE_BINARY_DIR}/_deps``, but if
  developers change this value, they should aim to keep the path short and
  just below the top level of the build tree to avoid running into path
  length problems on Windows.

.. variable:: FETCHCONTENT_QUIET

  The logging output during population can be quite verbose, making the
  configure stage quite noisy.  This cache option (``ON`` by default) hides
  all population output unless an error is encountered.  If experiencing
  problems with hung downloads, temporarily switching this option off may
  help diagnose which content population is causing the issue.

  .. versionchanged:: 3.30
    ``FETCHCONTENT_QUIET`` is ignored if policy :policy:`CMP0168` is set to
    ``NEW``.  The output is still quiet by default in that case, but verbosity
    is controlled by the message logging level (see
    :variable:`CMAKE_MESSAGE_LOG_LEVEL` and
    :option:`--log-level <cmake --log-level>`).

.. variable:: FETCHCONTENT_FULLY_DISCONNECTED

  When this option is enabled, no attempt is made to download or update
  any content.  It is assumed that all content has already been populated in
  a previous run, or the source directories have been pointed at existing
  contents the developer has provided manually (using options described
  further below).  When the developer knows that no changes have been made to
  any content details, turning this option ``ON`` can speed up
  the configure stage.  It is ``OFF`` by default.

  .. note::

    The ``FETCHCONTENT_FULLY_DISCONNECTED`` variable is not an appropriate way
    to prevent any network access on the first run in a build directory.
    Doing so can break projects, lead to misleading error messages, and hide
    subtle population failures.  This variable is specifically intended to
    only be turned on *after* the first time CMake has been run.
    If you want to prevent network access even on the first run, use a
    :ref:`dependency provider <dependency_providers>` and populate the
    dependency from local content instead.

  .. versionchanged:: 3.30
    The constraint that the source directory has already been populated when
    ``FETCHCONTENT_FULLY_DISCONNECTED`` is true is now enforced.
    See policy :policy:`CMP0170`.

.. variable:: FETCHCONTENT_UPDATES_DISCONNECTED

  This is a less severe download/update control compared to
  :variable:`FETCHCONTENT_FULLY_DISCONNECTED`.  Instead of bypassing all
  download and update logic, ``FETCHCONTENT_UPDATES_DISCONNECTED`` only
  prevents the update step from making connections to remote servers
  when using the git or hg download methods.  Updates still occur if details
  about the update step change, but the update is attempted with only the
  information already available locally (so switching to a different tag or
  commit that is already fetched locally will succeed, but switching to an
  unknown commit hash will fail).  The download step is not affected, so if
  content has not been downloaded previously, it will still be downloaded
  when this option is enabled.  This can speed up the configure step, but
  not as much as :variable:`FETCHCONTENT_FULLY_DISCONNECTED`.
  ``FETCHCONTENT_UPDATES_DISCONNECTED`` is ``OFF`` by default.

.. variable:: FETCHCONTENT_TRY_FIND_PACKAGE_MODE

  .. versionadded:: 3.24

  This variable modifies the details that :command:`FetchContent_Declare`
  records for a given dependency.  While it ultimately controls the behavior
  of :command:`FetchContent_MakeAvailable`, it is the variable's value when
  :command:`FetchContent_Declare` is called that gets used.  It makes no
  difference what the variable is set to when
  :command:`FetchContent_MakeAvailable` is called.  Since the variable should
  only be set by the user and not by projects directly, it will typically have
  the same value throughout anyway, so this distinction is not usually
  noticeable.

  ``FETCHCONTENT_TRY_FIND_PACKAGE_MODE`` ultimately controls whether
  :command:`FetchContent_MakeAvailable` is allowed to call
  :command:`find_package` to satisfy a dependency.  The variable can be set
  to one of the following values:

  ``OPT_IN``
    :command:`FetchContent_MakeAvailable` will only call
    :command:`find_package` if the :command:`FetchContent_Declare` call
    included a ``FIND_PACKAGE_ARGS`` keyword.  This is also the default
    behavior if ``FETCHCONTENT_TRY_FIND_PACKAGE_MODE`` is not set.

  ``ALWAYS``
    :command:`find_package` can be called by
    :command:`FetchContent_MakeAvailable` regardless of whether the
    :command:`FetchContent_Declare` call included a ``FIND_PACKAGE_ARGS``
    keyword or not.  If no ``FIND_PACKAGE_ARGS`` keyword was given, the
    behavior will be as though ``FIND_PACKAGE_ARGS`` had been provided,
    with no additional arguments after it.

  ``NEVER``
    :command:`FetchContent_MakeAvailable` will not call
    :command:`find_package`.  Any ``FIND_PACKAGE_ARGS`` given to the
    :command:`FetchContent_Declare` call will be ignored.

  As a special case, if the :variable:`FETCHCONTENT_SOURCE_DIR_<uppercaseName>`
  variable has a non-empty value for a dependency, it is assumed that the
  user is overriding all other methods of making that dependency available.
  ``FETCHCONTENT_TRY_FIND_PACKAGE_MODE`` will have no effect on that
  dependency and :command:`FetchContent_MakeAvailable` will not try to call
  :command:`find_package` for it.

In addition to the above, the following variables are also defined for each
content name:

.. variable:: FETCHCONTENT_SOURCE_DIR_<uppercaseName>

  If this is set, no download or update steps are performed for the specified
  content and the ``<lowercaseName>_SOURCE_DIR`` variable returned to the
  caller is pointed at this location.  This gives developers a way to have a
  separate checkout of the content that they can modify freely without
  interference from the build.  The build simply uses that existing source,
  but it still defines ``<lowercaseName>_BINARY_DIR`` to point inside its own
  build area.  Developers are strongly encouraged to use this mechanism rather
  than editing the sources populated in the default location, as changes to
  sources in the default location can be lost when content population details
  are changed by the project.

.. variable:: FETCHCONTENT_UPDATES_DISCONNECTED_<uppercaseName>

  This is the per-content equivalent of
  :variable:`FETCHCONTENT_UPDATES_DISCONNECTED`.  If the global option or
  this option is ``ON``, then updates for the git and hg methods will not
  contact any remote for the named content.  They will only use information
  already available locally.  Disabling updates for individual content can
  be useful for content whose details rarely change, while still leaving
  other frequently changing content with updates enabled.

.. _`fetch-content-examples`:

Examples
^^^^^^^^

Typical Case
""""""""""""

This first fairly straightforward example ensures that some popular testing
frameworks are available to the main build:

.. code-block:: cmake

  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        703bd9caab50b139428cea1aaff9974ebee5742e # release-1.10.0
  )
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        605a34765aa5d5ecbf476b4598a862ada971b0cc # v3.0.1
  )

  # After the following call, the CMake targets defined by googletest and
  # Catch2 will be available to the rest of the build
  FetchContent_MakeAvailable(googletest Catch2)

.. _FetchContent-find_package-integration-examples:

Integrating With find_package()
"""""""""""""""""""""""""""""""

For the previous example, if the user wanted to try to find ``googletest``
and ``Catch2`` via :command:`find_package` first before trying to download
and build them from source, they could set the
:variable:`FETCHCONTENT_TRY_FIND_PACKAGE_MODE` variable to ``ALWAYS``.
This would also affect any other calls to :command:`FetchContent_Declare`
throughout the project, which might not be acceptable.  The behavior can be
enabled for just these two dependencies instead by adding ``FIND_PACKAGE_ARGS``
to the declared details and leaving
:variable:`FETCHCONTENT_TRY_FIND_PACKAGE_MODE` unset, or set to ``OPT_IN``:

.. code-block:: cmake

  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        703bd9caab50b139428cea1aaff9974ebee5742e # release-1.10.0
    FIND_PACKAGE_ARGS NAMES GTest
  )
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        605a34765aa5d5ecbf476b4598a862ada971b0cc # v3.0.1
    FIND_PACKAGE_ARGS
  )

  # This will try calling find_package() first for both dependencies
  FetchContent_MakeAvailable(googletest Catch2)

For ``Catch2``, no additional arguments to :command:`find_package` are needed,
so no additional arguments are provided after the ``FIND_PACKAGE_ARGS``
keyword.  For ``googletest``, its package is more commonly called ``GTest``,
so arguments are added to support it being found by that name.

If the user wanted to disable :command:`FetchContent_MakeAvailable` from
calling :command:`find_package` for any dependency, even if it provided
``FIND_PACKAGE_ARGS`` in its declared details, they could set
:variable:`FETCHCONTENT_TRY_FIND_PACKAGE_MODE` to ``NEVER``.

If the project wanted to indicate that these two dependencies should be
downloaded and built from source and that :command:`find_package` calls
should be redirected to use the built dependencies, the
``OVERRIDE_FIND_PACKAGE`` option should be used when declaring the content
details:

.. code-block:: cmake

  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        703bd9caab50b139428cea1aaff9974ebee5742e # release-1.10.0
    OVERRIDE_FIND_PACKAGE
  )
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        605a34765aa5d5ecbf476b4598a862ada971b0cc # v3.0.1
    OVERRIDE_FIND_PACKAGE
  )

  # The following will automatically forward through to FetchContent_MakeAvailable()
  find_package(googletest)
  find_package(Catch2)

CMake provides a FindGTest module which defines some variables that older
projects may use instead of linking to the imported targets.  To support
those cases, we can provide an extra file.  In keeping with the
"first to define, wins" philosophy of ``FetchContent``, we only write out
that file if something else hasn't already done so.

.. code-block:: cmake

  FetchContent_MakeAvailable(googletest)

  if(NOT EXISTS ${CMAKE_FIND_PACKAGE_REDIRECTS_DIR}/googletest-extra.cmake AND
     NOT EXISTS ${CMAKE_FIND_PACKAGE_REDIRECTS_DIR}/googletestExtra.cmake)
    file(WRITE ${CMAKE_FIND_PACKAGE_REDIRECTS_DIR}/googletest-extra.cmake
  [=[
  if("${GTEST_LIBRARIES}" STREQUAL "" AND TARGET GTest::gtest)
    set(GTEST_LIBRARIES GTest::gtest)
  endif()
  if("${GTEST_MAIN_LIBRARIES}" STREQUAL "" AND TARGET GTest::gtest_main)
    set(GTEST_MAIN_LIBRARIES GTest::gtest_main)
  endif()
  if("${GTEST_BOTH_LIBRARIES}" STREQUAL "")
    set(GTEST_BOTH_LIBRARIES ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES})
  endif()
  ]=])
  endif()

Projects will also likely be using ``find_package(GTest)`` rather than
``find_package(googletest)``, but it is possible to make use of the
:variable:`CMAKE_FIND_PACKAGE_REDIRECTS_DIR` area to pull in the latter as
a dependency of the former.  This is likely to be sufficient to satisfy
a typical ``find_package(GTest)`` call.

.. code-block:: cmake

  FetchContent_MakeAvailable(googletest)

  if(NOT EXISTS ${CMAKE_FIND_PACKAGE_REDIRECTS_DIR}/gtest-config.cmake AND
     NOT EXISTS ${CMAKE_FIND_PACKAGE_REDIRECTS_DIR}/GTestConfig.cmake)
    file(WRITE ${CMAKE_FIND_PACKAGE_REDIRECTS_DIR}/gtest-config.cmake
  [=[
  include(CMakeFindDependencyMacro)
  find_dependency(googletest)
  ]=])
  endif()

  if(NOT EXISTS ${CMAKE_FIND_PACKAGE_REDIRECTS_DIR}/gtest-config-version.cmake AND
     NOT EXISTS ${CMAKE_FIND_PACKAGE_REDIRECTS_DIR}/GTestConfigVersion.cmake)
    file(WRITE ${CMAKE_FIND_PACKAGE_REDIRECTS_DIR}/gtest-config-version.cmake
  [=[
  include(${CMAKE_FIND_PACKAGE_REDIRECTS_DIR}/googletest-config-version.cmake OPTIONAL)
  if(NOT PACKAGE_VERSION_COMPATIBLE)
    include(${CMAKE_FIND_PACKAGE_REDIRECTS_DIR}/googletestConfigVersion.cmake OPTIONAL)
  endif()
  ]=])
  endif()

Overriding Where To Find CMakeLists.txt
"""""""""""""""""""""""""""""""""""""""

If the sub-project's ``CMakeLists.txt`` file is not at the top level of its
source tree, the ``SOURCE_SUBDIR`` option can be used to tell ``FetchContent``
where to find it.  The following example shows how to use that option, and
it also sets a variable which is meaningful to the subproject before pulling
it into the main build (set as an ``INTERNAL`` cache variable to avoid
problems with policy :policy:`CMP0077`):

.. code-block:: cmake

  include(FetchContent)
  FetchContent_Declare(
    protobuf
    GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
    GIT_TAG        ae50d9b9902526efd6c7a1907d09739f959c6297 # v3.15.0
    SOURCE_SUBDIR  cmake
  )
  set(protobuf_BUILD_TESTS OFF CACHE INTERNAL "")
  FetchContent_MakeAvailable(protobuf)

Complex Dependency Hierarchies
""""""""""""""""""""""""""""""

In more complex project hierarchies, the dependency relationships can be more
complicated.  Consider a hierarchy where ``projA`` is the top level project and
it depends directly on projects ``projB`` and ``projC``.  Both ``projB`` and
``projC`` can be built standalone and they also both depend on another project
``projD``.  ``projB`` additionally depends on ``projE``.  This example assumes
that all five projects are available on a company git server.  The
``CMakeLists.txt`` of each project might have sections like the following:

.. code-block:: cmake
  :caption: *projA*

  include(FetchContent)
  FetchContent_Declare(
    projB
    GIT_REPOSITORY git@mycompany.com:git/projB.git
    GIT_TAG        4a89dc7e24ff212a7b5167bef7ab079d
  )
  FetchContent_Declare(
    projC
    GIT_REPOSITORY git@mycompany.com:git/projC.git
    GIT_TAG        4ad4016bd1d8d5412d135cf8ceea1bb9
  )
  FetchContent_Declare(
    projD
    GIT_REPOSITORY git@mycompany.com:git/projD.git
    GIT_TAG        origin/integrationBranch
  )
  FetchContent_Declare(
    projE
    GIT_REPOSITORY git@mycompany.com:git/projE.git
    GIT_TAG        v2.3-rc1
  )

  # Order is important, see notes in the discussion further below
  FetchContent_MakeAvailable(projD projB projC)


.. code-block:: cmake
  :caption: *projB*

  include(FetchContent)
  FetchContent_Declare(
    projD
    GIT_REPOSITORY git@mycompany.com:git/projD.git
    GIT_TAG        20b415f9034bbd2a2e8216e9a5c9e632
  )
  FetchContent_Declare(
    projE
    GIT_REPOSITORY git@mycompany.com:git/projE.git
    GIT_TAG        68e20f674a48be38d60e129f600faf7d
  )

  FetchContent_MakeAvailable(projD projE)


.. code-block:: cmake
  :caption: *projC*

  include(FetchContent)
  FetchContent_Declare(
    projD
    GIT_REPOSITORY git@mycompany.com:git/projD.git
    GIT_TAG        7d9a17ad2c962aa13e2fbb8043fb6b8a
  )

  FetchContent_MakeAvailable(projD)

A few key points should be noted in the above:

- ``projB`` and ``projC`` define different content details for ``projD``,
  but ``projA`` also defines a set of content details for ``projD``.
  Because ``projA`` will define them first, the details from ``projB`` and
  ``projC`` will not be used.  The override details defined by ``projA``
  are not required to match either of those from ``projB`` or ``projC``, but
  it is up to the higher level project to ensure that the details it does
  define still make sense for the child projects.
- In the ``projA`` call to :command:`FetchContent_MakeAvailable`, ``projD``
  is listed ahead of ``projB`` and ``projC``, so it will be populated before
  either ``projB`` or ``projC``. It isn't required for ``projA`` to do this,
  doing so ensures that ``projA`` fully controls the environment in which
  ``projD`` is brought into the build (directory properties are particularly
  relevant).
- While ``projA`` defines content details for ``projE``, it does not need
  to explicitly call ``FetchContent_MakeAvailable(projE)`` or
  ``FetchContent_Populate(projD)`` itself.  Instead, it leaves that to the
  child ``projB``.  For higher level projects, it is often enough to just
  define the override content details and leave the actual population to the
  child projects.  This saves repeating the same thing at each level of the
  project hierarchy unnecessarily, but it should only be done if directory
  properties set by dependencies are not expected to influence the population
  of the shared dependency (``projE`` in this case).

Populating Content Without Adding It To The Build
"""""""""""""""""""""""""""""""""""""""""""""""""

Projects don't always need to add the populated content to the build.
Sometimes the project just wants to make the downloaded content available at
a predictable location.  The next example ensures that a set of standard
company toolchain files (and potentially even the toolchain binaries
themselves) is available early enough to be used for that same build.

.. code-block:: cmake

  cmake_minimum_required(VERSION 3.14)

  include(FetchContent)
  FetchContent_Declare(
    mycom_toolchains
    URL  https://intranet.mycompany.com//toolchains_1.3.2.tar.gz
  )
  FetchContent_MakeAvailable(mycom_toolchains)

  project(CrossCompileExample)

The project could be configured to use one of the downloaded toolchains like
so:

.. code-block:: shell

  cmake -DCMAKE_TOOLCHAIN_FILE=_deps/mycom_toolchains-src/toolchain_arm.cmake /path/to/src

When CMake processes the ``CMakeLists.txt`` file, it will download and unpack
the tarball into ``_deps/mycompany_toolchains-src`` relative to the build
directory.  The :variable:`CMAKE_TOOLCHAIN_FILE` variable is not used until
the :command:`project` command is reached, at which point CMake looks for the
named toolchain file relative to the build directory.  Because the tarball has
already been downloaded and unpacked by then, the toolchain file will be in
place, even the very first time that :program:`cmake` is run in the build directory.

Populating Content In CMake Script Mode
"""""""""""""""""""""""""""""""""""""""

This last example demonstrates how one might download and unpack a
firmware tarball using CMake's :ref:`script mode <Script Processing Mode>`.
The call to :command:`FetchContent_Populate` specifies all the content details
and the unpacked firmware will be placed in a ``firmware`` directory below the
current working directory.

.. code-block:: cmake
  :caption: :file:`getFirmware.cmake`

  # NOTE: Intended to be run in script mode with cmake -P
  include(FetchContent)
  FetchContent_Populate(
    firmware
    URL        https://mycompany.com/assets/firmware-1.23-arm.tar.gz
    URL_HASH   MD5=68247684da89b608d466253762b0ff11
    SOURCE_DIR firmware
  )

#]=======================================================================]

# Control policies for most of the things defined by this module. Only a few
# FetchContent_MakeAvailable() implementation details are excluded for
# backward compatibility reasons (see just after the endblock()).
block(SCOPE_FOR POLICIES)
cmake_policy(VERSION 4.1)

include(${CMAKE_CURRENT_LIST_DIR}/ExternalProject/shared_internal_commands.cmake)

#=======================================================================
# Recording and retrieving content details for later population
#=======================================================================

# Internal use, projects must not call this directly. It is
# intended for use by FetchContent_Declare() only.
#
# Sets a content-specific global property (not meant for use
# outside of functions defined here in this file) which can later
# be retrieved using __FetchContent_getSavedDetails() with just the
# same content name. If there is already a value stored in the
# property, it is left unchanged and this call has no effect.
# This allows parent projects to define the content details,
# overriding anything a child project may try to set (properties
# are not cached between runs, so the first thing to set it in a
# build will be in control).
function(__FetchContent_declareDetails contentName)

  string(TOLOWER ${contentName} contentNameLower)
  set(savedDetailsPropertyName "_FetchContent_${contentNameLower}_savedDetails")
  get_property(alreadyDefined GLOBAL PROPERTY ${savedDetailsPropertyName} DEFINED)
  if(alreadyDefined)
    return()
  endif()

  if("${FETCHCONTENT_TRY_FIND_PACKAGE_MODE}" STREQUAL "ALWAYS")
    set(__tryFindPackage TRUE)
    set(__tryFindPackageAllowed TRUE)
  elseif("${FETCHCONTENT_TRY_FIND_PACKAGE_MODE}" STREQUAL "NEVER")
    set(__tryFindPackage FALSE)
    set(__tryFindPackageAllowed FALSE)
  elseif("${FETCHCONTENT_TRY_FIND_PACKAGE_MODE}" STREQUAL "OPT_IN" OR
         NOT DEFINED FETCHCONTENT_TRY_FIND_PACKAGE_MODE)
    set(__tryFindPackage FALSE)
    set(__tryFindPackageAllowed TRUE)
  else()
    message(FATAL_ERROR
      "Unsupported value for FETCHCONTENT_TRY_FIND_PACKAGE_MODE: "
      "${FETCHCONTENT_TRY_FIND_PACKAGE_MODE}"
    )
  endif()

  set(__cmdArgs)
  set(__findPackageArgs)
  set(__sawQuietKeyword NO)
  set(__sawGlobalKeyword NO)
  set(__direct_population NO)
  foreach(__item IN LISTS ARGN)
    if(__item STREQUAL "__DIRECT_POPULATION")
      set(__direct_population YES)
      continue()
    endif()

    if(DEFINED __findPackageArgs)
      # All remaining args are for find_package()
      string(APPEND __findPackageArgs " [==[${__item}]==]")
      if(__item STREQUAL "QUIET")
        set(__sawQuietKeyword YES)
      elseif(__item STREQUAL "GLOBAL")
        set(__sawGlobalKeyword YES)
      endif()
      continue()
    endif()

    # Still processing non-find_package() args
    if(__item STREQUAL "FIND_PACKAGE_ARGS")
      if(__tryFindPackageAllowed)
        set(__tryFindPackage TRUE)
      endif()
      # All arguments after this keyword are for find_package(). Define the
      # variable but with an empty value initially. This allows us to check
      # at the start of the loop whether to store remaining items in this
      # variable or not. Note that there could be no more args, which is still
      # a valid case because we automatically provide ${contentName} as the
      # package name and there may not need to be any further arguments.
      set(__findPackageArgs "")
      continue()  # Don't store this item
    elseif(__item STREQUAL "OVERRIDE_FIND_PACKAGE")
      set(__tryFindPackageAllowed FALSE)
      # Define a separate dedicated property for find_package() to check
      # in its implementation. This will be a placeholder until FetchContent
      # actually does the population. After that, we will have created a
      # stand-in config file that find_package() will pick up instead.
      set(propertyName "_FetchContent_${contentNameLower}_override_find_package")
      define_property(GLOBAL PROPERTY ${propertyName})
      set_property(GLOBAL PROPERTY ${propertyName} TRUE)
    endif()

    string(APPEND __cmdArgs " [==[${__item}]==]")
  endforeach()

  set_property(GLOBAL PROPERTY
    "_FetchContent_${contentNameLower}_direct_population" ${__direct_population}
  )

  define_property(GLOBAL PROPERTY ${savedDetailsPropertyName})
  cmake_language(EVAL CODE
    "set_property(GLOBAL PROPERTY ${savedDetailsPropertyName} ${__cmdArgs})"
  )

  if(__tryFindPackage AND __tryFindPackageAllowed)
    set(propertyName "_FetchContent_${contentNameLower}_find_package_args")
    define_property(GLOBAL PROPERTY ${propertyName})
    if(NOT __sawQuietKeyword)
      string(PREPEND __findPackageArgs "QUIET ")
    endif()
    if(CMAKE_FIND_PACKAGE_TARGETS_GLOBAL AND NOT __sawGlobalKeyword)
      string(APPEND __findPackageArgs " GLOBAL")
    endif()
    cmake_language(EVAL CODE
      "set_property(GLOBAL PROPERTY ${propertyName} ${__findPackageArgs})"
    )
  endif()

endfunction()


# Internal use, projects must not call this directly. It is
# intended for use by the FetchContent_Declare() function.
#
# Retrieves details saved for the specified content in an
# earlier call to __FetchContent_declareDetails().
function(__FetchContent_getSavedDetails contentName outVar)

  string(TOLOWER ${contentName} contentNameLower)
  set(propertyName "_FetchContent_${contentNameLower}_savedDetails")
  get_property(alreadyDefined GLOBAL PROPERTY ${propertyName} DEFINED)
  if(NOT alreadyDefined)
    message(FATAL_ERROR "No content details recorded for ${contentName}")
  endif()
  get_property(propertyValue GLOBAL PROPERTY ${propertyName})
  set(${outVar} "${propertyValue}" PARENT_SCOPE)

endfunction()


# Saves population details of the content, sets defaults for the
# SOURCE_DIR and BUILD_DIR.
function(FetchContent_Declare contentName)

  # Always check this even if we won't save these details.
  # This helps projects catch errors earlier.
  # Avoid using if(... IN_LIST ...) so we don't have to alter policy settings
  list(FIND ARGN OVERRIDE_FIND_PACKAGE index_OVERRIDE_FIND_PACKAGE)
  list(FIND ARGN FIND_PACKAGE_ARGS index_FIND_PACKAGE_ARGS)
  if(index_OVERRIDE_FIND_PACKAGE GREATER_EQUAL 0 AND
     index_FIND_PACKAGE_ARGS GREATER_EQUAL 0)
    message(FATAL_ERROR
      "Cannot specify both OVERRIDE_FIND_PACKAGE and FIND_PACKAGE_ARGS "
      "when declaring details for ${contentName}"
    )
  endif()

  # Because we are only looking for a subset of the supported keywords, we
  # cannot check for multi-value arguments with this method. We will have to
  # handle the URL keyword differently.
  set(oneValueArgs
    GIT_REPOSITORY
    SVN_REPOSITORY
    DOWNLOAD_NO_EXTRACT
    DOWNLOAD_EXTRACT_TIMESTAMP
    BINARY_DIR
    SOURCE_DIR
  )

  cmake_parse_arguments(PARSE_ARGV 1 ARG "" "${oneValueArgs}" "")

  string(TOLOWER ${contentName} contentNameLower)

  if(NOT ARG_BINARY_DIR)
    set(ARG_BINARY_DIR "${FETCHCONTENT_BASE_DIR}/${contentNameLower}-build")
  endif()

  if(NOT ARG_SOURCE_DIR)
    set(ARG_SOURCE_DIR "${FETCHCONTENT_BASE_DIR}/${contentNameLower}-src")
  endif()

  if(ARG_GIT_REPOSITORY)
    # We resolve the GIT_REPOSITORY here so that we get the right parent in the
    # remote selection logic. In the sub-build, ExternalProject_Add() would see
    # the private sub-build directory as the parent project, but the parent
    # project should be the one that called FetchContent_Declare(). We resolve
    # a relative repo here so that the sub-build's ExternalProject_Add() only
    # ever sees a non-relative repo.
    # Since these checks may be non-trivial on some platforms (notably Windows),
    # don't perform them if we won't be using these details. This also allows
    # projects to override calls with relative URLs when they have checked out
    # the parent project in an unexpected way, such as from a mirror or fork.
    set(savedDetailsPropertyName "_FetchContent_${contentNameLower}_savedDetails")
    get_property(alreadyDefined GLOBAL PROPERTY ${savedDetailsPropertyName} DEFINED)
    if(NOT alreadyDefined)
      cmake_policy(GET CMP0150 cmp0150
        PARENT_SCOPE # undocumented, do not use outside of CMake
      )
      _ep_resolve_git_remote(_resolved_git_repository
        "${ARG_GIT_REPOSITORY}" "${cmp0150}" "${FETCHCONTENT_BASE_DIR}"
      )
      set(ARG_GIT_REPOSITORY "${_resolved_git_repository}")
    endif()
  endif()

  if(ARG_SVN_REPOSITORY)
    # Add a hash of the svn repository URL to the source dir. This works
    # around the problem where if the URL changes, the download would
    # fail because it tries to checkout/update rather than switch the
    # old URL to the new one. We limit the hash to the first 7 characters
    # so that the source path doesn't get overly long (which can be a
    # problem on windows due to path length limits).
    string(SHA1 urlSHA ${ARG_SVN_REPOSITORY})
    string(SUBSTRING ${urlSHA} 0 7 urlSHA)
    string(APPEND ARG_SOURCE_DIR "-${urlSHA}")
  endif()

  # The ExternalProject_Add() call in the sub-build won't see the CMP0135
  # policy setting of our caller. Work out if that policy will be needed and
  # explicitly set the relevant option if not already provided. The condition
  # here is essentially an abbreviated version of the logic in
  # ExternalProject's _ep_add_download_command() function.
  if(NOT ARG_DOWNLOAD_NO_EXTRACT AND
     NOT DEFINED ARG_DOWNLOAD_EXTRACT_TIMESTAMP)
    list(FIND ARGN URL urlIndex)
    if(urlIndex GREATER_EQUAL 0)
      math(EXPR urlIndex "${urlIndex} + 1")
      list(LENGTH ARGN numArgs)
      if(urlIndex GREATER_EQUAL numArgs)
        message(FATAL_ERROR
          "URL keyword needs to be followed by at least one URL"
        )
      endif()
      # If we have multiple URLs, none of them are allowed to be local paths.
      # Therefore, we can test just the first URL, and if it is non-local, so
      # will be the others if there are more.
      list(GET ARGN ${urlIndex} firstUrl)
      if(NOT IS_DIRECTORY "${firstUrl}")
        cmake_policy(GET CMP0135 _FETCHCONTENT_CMP0135
          PARENT_SCOPE # undocumented, do not use outside of CMake
        )
        if(_FETCHCONTENT_CMP0135 STREQUAL "")
          message(AUTHOR_WARNING
            "The DOWNLOAD_EXTRACT_TIMESTAMP option was not given and policy "
            "CMP0135 is not set. The policy's OLD behavior will be used. "
            "When using a URL download, the timestamps of extracted files "
            "should preferably be that of the time of extraction, otherwise "
            "code that depends on the extracted contents might not be "
            "rebuilt if the URL changes. The OLD behavior preserves the "
            "timestamps from the archive instead, but this is usually not "
            "what you want. Update your project to the NEW behavior or "
            "specify the DOWNLOAD_EXTRACT_TIMESTAMP option with a value of "
            "true to avoid this robustness issue."
          )
          set(ARG_DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
        elseif(_FETCHCONTENT_CMP0135 STREQUAL "NEW")
          set(ARG_DOWNLOAD_EXTRACT_TIMESTAMP FALSE)
        else()
          set(ARG_DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
        endif()
      endif()
    endif()
  endif()

  # Add back in the keyword args we pulled out and potentially tweaked/added
  set(forward_args "${ARG_UNPARSED_ARGUMENTS}")
  set(sep EXTERNALPROJECT_INTERNAL_ARGUMENT_SEPARATOR)
  foreach(key IN LISTS oneValueArgs)
    if(DEFINED ARG_${key})
      list(PREPEND forward_args ${key} "${ARG_${key}}" ${sep})
      set(sep "")
    endif()
  endforeach()

  cmake_policy(GET CMP0168 cmp0168
    PARENT_SCOPE # undocumented, do not use outside of CMake
  )
  if(cmp0168 STREQUAL "NEW")
    list(PREPEND forward_args __DIRECT_POPULATION ${sep})
  endif()

  set(__argsQuoted)
  foreach(__item IN LISTS forward_args)
    string(APPEND __argsQuoted " [==[${__item}]==]")
  endforeach()
  cmake_language(EVAL CODE
    "__FetchContent_declareDetails(${contentNameLower} ${__argsQuoted})"
  )

endfunction()


#=======================================================================
# Set/get whether the specified content has been populated yet.
# The setter also records the source and binary dirs used.
#=======================================================================

# Semi-internal use. Projects must not call this directly. Dependency
# providers must call it if they satisfy a request made with the
# FETCHCONTENT_MAKEAVAILABLE_SERIAL method (that is the only permitted
# place to call it outside of the FetchContent module).
function(FetchContent_SetPopulated contentName)

  cmake_parse_arguments(PARSE_ARGV 1 arg
    ""
    "SOURCE_DIR;BINARY_DIR"
    ""
  )
  if(NOT "${arg_UNPARSED_ARGUMENTS}" STREQUAL "")
    message(FATAL_ERROR "Unsupported arguments: ${arg_UNPARSED_ARGUMENTS}")
  endif()

  string(TOLOWER ${contentName} contentNameLower)
  set(prefix "_FetchContent_${contentNameLower}")

  set(propertyName "${prefix}_sourceDir")
  define_property(GLOBAL PROPERTY ${propertyName})
  if("${arg_SOURCE_DIR}" STREQUAL "")
    # Don't discard a previously provided SOURCE_DIR
    get_property(arg_SOURCE_DIR GLOBAL PROPERTY ${propertyName})
  endif()
  set_property(GLOBAL PROPERTY ${propertyName} "${arg_SOURCE_DIR}")

  set(propertyName "${prefix}_binaryDir")
  define_property(GLOBAL PROPERTY ${propertyName})
  if("${arg_BINARY_DIR}" STREQUAL "")
    # Don't discard a previously provided BINARY_DIR
    get_property(arg_BINARY_DIR GLOBAL PROPERTY ${propertyName})
  endif()
  set_property(GLOBAL PROPERTY ${propertyName} "${arg_BINARY_DIR}")

  set(propertyName "${prefix}_populated")
  define_property(GLOBAL PROPERTY ${propertyName})
  set_property(GLOBAL PROPERTY ${propertyName} TRUE)

endfunction()


# Set variables in the calling scope for any of the retrievable
# properties. If no specific properties are requested, variables
# will be set for all retrievable properties.
#
# This function is intended to also be used by projects as the canonical
# way to detect whether they should call FetchContent_Populate()
# and pull the populated source into the build with add_subdirectory(),
# if they are using the populated content in that way.
function(FetchContent_GetProperties contentName)

  string(TOLOWER ${contentName} contentNameLower)

  set(options "")
  set(oneValueArgs SOURCE_DIR BINARY_DIR POPULATED)
  set(multiValueArgs "")

  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(NOT ARG_SOURCE_DIR AND
     NOT ARG_BINARY_DIR AND
     NOT ARG_POPULATED)
    # No specific properties requested, provide them all
    set(ARG_SOURCE_DIR ${contentNameLower}_SOURCE_DIR)
    set(ARG_BINARY_DIR ${contentNameLower}_BINARY_DIR)
    set(ARG_POPULATED  ${contentNameLower}_POPULATED)
  endif()

  set(prefix "_FetchContent_${contentNameLower}")

  if(ARG_SOURCE_DIR)
    set(propertyName "${prefix}_sourceDir")
    get_property(value GLOBAL PROPERTY ${propertyName})
    if(value)
      set(${ARG_SOURCE_DIR} ${value} PARENT_SCOPE)
    endif()
  endif()

  if(ARG_BINARY_DIR)
    set(propertyName "${prefix}_binaryDir")
    get_property(value GLOBAL PROPERTY ${propertyName})
    if(value)
      set(${ARG_BINARY_DIR} ${value} PARENT_SCOPE)
    endif()
  endif()

  if(ARG_POPULATED)
    set(propertyName "${prefix}_populated")
    get_property(value GLOBAL PROPERTY ${propertyName} DEFINED)
    set(${ARG_POPULATED} ${value} PARENT_SCOPE)
  endif()

endfunction()


#=======================================================================
# Performing the population
#=======================================================================

# The value of contentName will always have been lowercased by the caller.
# All other arguments are assumed to be options that are understood by
# ExternalProject_Add(), except for QUIET and SUBBUILD_DIR.
function(__FetchContent_doPopulation contentName)

  set(options
      QUIET
      # EXCLUDE_FROM_ALL and SYSTEM have no meaning for ExternalProject, they
      # are only used by us in FetchContent_MakeAvailable(). We need to parse
      # and discard them here.
      EXCLUDE_FROM_ALL
      SYSTEM
  )
  set(oneValueArgs
      SUBBUILD_DIR
      SOURCE_DIR
      BINARY_DIR
      # We need special processing if DOWNLOAD_NO_EXTRACT is true
      DOWNLOAD_NO_EXTRACT
      # Prevent the following from being passed through
      CONFIGURE_COMMAND
      BUILD_COMMAND
      INSTALL_COMMAND
      TEST_COMMAND
      # We force these to be ON since we are always executing serially
      # and we want all steps to have access to the terminal in case they
      # need input from the command line (e.g. ask for a private key password)
      # or they want to provide timely progress. We silently absorb and
      # discard these if they are set by the caller.
      USES_TERMINAL_DOWNLOAD
      USES_TERMINAL_UPDATE
      USES_TERMINAL_PATCH
      # Internal options, may change at any time
      __DIRECT_POPULATION
  )
  set(multiValueArgs "")

  cmake_parse_arguments(PARSE_ARGV 1 ARG
    "${options}" "${oneValueArgs}" "${multiValueArgs}")

  if(DEFINED ARG___DIRECT_POPULATION)
    # Direct call to FetchContent_Populate() with full details. The policy
    # setting of its caller is included in the function arguments.
    set(direct_population ${ARG___DIRECT_POPULATION})
  else()
    # FetchContent_Populate() called with only the name of a dependency.
    # We need the policy setting of the corresponding FetchContent_Declare().
    get_property(direct_population GLOBAL PROPERTY
      "_FetchContent_${contentNameLower}_direct_population"
    )
  endif()

  if(NOT ARG_SUBBUILD_DIR)
    if(NOT direct_population)
      message(FATAL_ERROR "Internal error: SUBBUILD_DIR not set")
    endif()
  elseif(NOT IS_ABSOLUTE "${ARG_SUBBUILD_DIR}")
    set(ARG_SUBBUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/${ARG_SUBBUILD_DIR}")
  endif()

  if(NOT ARG_SOURCE_DIR)
    message(FATAL_ERROR "Internal error: SOURCE_DIR not set")
  elseif(NOT IS_ABSOLUTE "${ARG_SOURCE_DIR}")
    set(ARG_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/${ARG_SOURCE_DIR}")
  endif()

  if(NOT ARG_BINARY_DIR)
    message(FATAL_ERROR "Internal error: BINARY_DIR not set")
  elseif(NOT IS_ABSOLUTE "${ARG_BINARY_DIR}")
    set(ARG_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/${ARG_BINARY_DIR}")
  endif()

  # Ensure the caller can know where to find the source and build directories
  # with some convenient variables. Doing this here ensures the caller sees
  # the correct result in the case where the default values are overridden by
  # the content details set by the project.
  set(${contentName}_SOURCE_DIR "${ARG_SOURCE_DIR}" PARENT_SCOPE)
  set(${contentName}_BINARY_DIR "${ARG_BINARY_DIR}" PARENT_SCOPE)

  if(direct_population)
    __FetchContent_populateDirect()
  else()
    __FetchContent_populateSubbuild()
  endif()
endfunction()


function(__FetchContent_populateDirect)
  # Policies CMP0097, CMP0135 and CMP0150 are handled in FetchContent_Declare()
  # and the stored arguments already account for them.
  # For CMP0097, the arguments will always assume NEW behavior by the time
  # we get to here, so ensure ExternalProject sees that.
  set(_EP_CMP0097 NEW)

  set(args_to_parse
    "${ARG_UNPARSED_ARGUMENTS}"
    SOURCE_DIR "${ARG_SOURCE_DIR}"
    BINARY_DIR "${ARG_BINARY_DIR}"
  )
  if(ARG_DOWNLOAD_NO_EXTRACT)
    list(APPEND args_to_parse DOWNLOAD_NO_EXTRACT YES)
  endif()

  get_property(cmake_role GLOBAL PROPERTY CMAKE_ROLE)
  if(cmake_role STREQUAL "PROJECT")
    # We don't support direct population where a project makes a direct call
    # to FetchContent_Populate(). That always goes through ExternalProject and
    # will soon be deprecated anyway.
    set(function_for_args FetchContent_Declare)
  elseif(cmake_role STREQUAL "SCRIPT")
    set(function_for_args FetchContent_Populate)
  else()
    message(FATAL_ERROR "Unsupported context for direct population")
  endif()

  _ep_get_add_keywords(keywords)
  _ep_parse_arguments_to_vars(
    ${function_for_args}
    "${keywords}"
    ${contentName}
    _EP_
    "${args_to_parse}"
  )

  # We use a simplified set of directories here. We do not need the full set
  # of directories that ExternalProject supports, and we don't need the
  # extensive customization options it supports either. Note that
  # _EP_SOURCE_DIR and _EP_BINARY_DIR are always included in the saved args,
  # so we must not set them here.
  if(cmake_role STREQUAL "PROJECT")
    # Put these under CMakeFiles so that they are removed by "cmake --fresh",
    # which will cause the steps to re-run.
    set(_EP_STAMP_DIR "${CMAKE_BINARY_DIR}/CMakeFiles/fc-stamp/${contentNameLower}")
    set(_EP_TMP_DIR   "${CMAKE_BINARY_DIR}/CMakeFiles/fc-tmp/${contentNameLower}")
  else()
    # We have no CMakeFiles in script mode, so keep everything together.
    set(_EP_STAMP_DIR "${FETCHCONTENT_BASE_DIR}/${contentNameLower}-stamp")
    set(_EP_TMP_DIR   "${FETCHCONTENT_BASE_DIR}/${contentNameLower}-tmp")
  endif()
  # Always put downloaded things under FETCHCONTENT_BASE_DIR so that we can
  # reuse previously downloaded content, even after a "cmake --fresh".
  set(_EP_DOWNLOAD_DIR "${FETCHCONTENT_BASE_DIR}/${contentNameLower}-tmp")

  # If CMAKE_DISABLE_SOURCE_CHANGES is set to true and _EP_SOURCE_DIR is an
  # existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
  # would cause a fatal error, even though it would be a no-op.
  if(NOT EXISTS "${_EP_SOURCE_DIR}")
    file(MAKE_DIRECTORY "${_EP_SOURCE_DIR}")
  endif()
  file(MAKE_DIRECTORY
    "${_EP_BINARY_DIR}"
    "${_EP_STAMP_DIR}"
    "${_EP_TMP_DIR}"
    "${_EP_DOWNLOAD_DIR}"
  )

  # We take over the stamp files and use our own for detecting whether each
  # step is up-to-date. The method used by ExternalProject is specific to
  # using a sub-build and is not appropriate for us here.

  set(download_script ${_EP_TMP_DIR}/download.cmake)
  set(update_script   ${_EP_TMP_DIR}/update.cmake)
  set(patch_script    ${_EP_TMP_DIR}/patch.cmake)
  _ep_add_download_command(${contentName}
    SCRIPT_FILE ${download_script}
    DEPENDS_VARIABLE download_depends
  )
  _ep_add_update_command(${contentName}
    SCRIPT_FILE ${update_script}
    DEPENDS_VARIABLE update_depends
  )
  _ep_add_patch_command(${contentName}
    SCRIPT_FILE ${patch_script}
    # No additional dependencies for the patch step
  )

  set(download_stamp ${_EP_STAMP_DIR}/download.stamp)
  set(update_stamp   ${_EP_STAMP_DIR}/update.stamp)
  set(patch_stamp    ${_EP_STAMP_DIR}/patch.stamp)
  __FetchContent_doStepDirect(
    SCRIPT_FILE ${download_script}
    STAMP_FILE  ${download_stamp}
    DEPENDS     ${download_depends}
  )
  __FetchContent_doStepDirect(
    SCRIPT_FILE ${update_script}
    STAMP_FILE  ${update_stamp}
    DEPENDS     ${update_depends} ${download_stamp}
  )
  __FetchContent_doStepDirect(
    SCRIPT_FILE ${patch_script}
    STAMP_FILE  ${patch_stamp}
    DEPENDS     ${update_stamp}
  )

endfunction()


function(__FetchContent_doStepDirect)
  set(noValueOptions )
  set(singleValueOptions
    SCRIPT_FILE
    STAMP_FILE
  )
  set(multiValueOptions
    DEPENDS
  )
  cmake_parse_arguments(PARSE_ARGV 0 arg
    "${noValueOptions}" "${singleValueOptions}" "${multiValueOptions}"
  )

  if(NOT EXISTS ${arg_STAMP_FILE})
    set(do_step YES)
  else()
    set(do_step NO)
    foreach(dep_file IN LISTS arg_DEPENDS arg_SCRIPT_FILE)
      if(NOT EXISTS "${arg_STAMP_FILE}" OR
        NOT EXISTS "${dep_file}" OR
        NOT "${arg_STAMP_FILE}" IS_NEWER_THAN "${dep_file}")
        set(do_step YES)
        break()
      endif()
    endforeach()
  endif()

  if(do_step)
    include(${arg_SCRIPT_FILE})
    file(TOUCH "${arg_STAMP_FILE}")
  endif()
endfunction()


function(__FetchContent_populateSubbuild)
  # All argument parsing is done in __FetchContent_doPopulate(), since it is
  # common to both the subbuild and direct population strategies.
  # Parsed arguments are in ARG_... variables.

  # The unparsed arguments may contain spaces, so build up ARG_EXTRA
  # in such a way that it correctly substitutes into the generated
  # CMakeLists.txt file with each argument quoted.
  unset(ARG_EXTRA)
  foreach(arg IN LISTS ARG_UNPARSED_ARGUMENTS)
    set(ARG_EXTRA "${ARG_EXTRA} \"${arg}\"")
  endforeach()

  if(ARG_DOWNLOAD_NO_EXTRACT)
    set(ARG_EXTRA "${ARG_EXTRA} DOWNLOAD_NO_EXTRACT YES")
    set(__FETCHCONTENT_COPY_FILE
"
ExternalProject_Get_Property(${contentName}-populate DOWNLOADED_FILE)
get_filename_component(dlFileName \"\${DOWNLOADED_FILE}\" NAME)

ExternalProject_Add_Step(${contentName}-populate copyfile
  COMMAND    \"${CMAKE_COMMAND}\" -E copy_if_different
             \"<DOWNLOADED_FILE>\" \"${ARG_SOURCE_DIR}\"
  DEPENDEES  patch
  DEPENDERS  configure
  BYPRODUCTS \"${ARG_SOURCE_DIR}/\${dlFileName}\"
  COMMENT    \"Copying file to SOURCE_DIR\"
)
")
  else()
    unset(__FETCHCONTENT_COPY_FILE)
  endif()

  # Hide output if requested, but save it to a variable in case there's an
  # error so we can show the output upon failure. When not quiet, don't
  # capture the output to a variable because the user may want to see the
  # output as it happens (e.g. progress during long downloads). Combine both
  # stdout and stderr in the one capture variable so the output stays in order.
  if (ARG_QUIET)
    set(outputOptions
        OUTPUT_VARIABLE capturedOutput
        ERROR_VARIABLE  capturedOutput
    )
  else()
    set(capturedOutput)
    set(outputOptions)
    message(STATUS "Populating ${contentName}")
  endif()

  if(CMAKE_GENERATOR)
    set(subCMakeOpts "-G${CMAKE_GENERATOR}")
    if(CMAKE_GENERATOR_PLATFORM)
      list(APPEND subCMakeOpts "-A${CMAKE_GENERATOR_PLATFORM}")
    endif()
    if(CMAKE_GENERATOR_TOOLSET)
      list(APPEND subCMakeOpts "-T${CMAKE_GENERATOR_TOOLSET}")
    endif()
    if(CMAKE_GENERATOR_INSTANCE)
      list(APPEND subCMakeOpts "-DCMAKE_GENERATOR_INSTANCE:INTERNAL=${CMAKE_GENERATOR_INSTANCE}")
    endif()
    if(CMAKE_MAKE_PROGRAM)
      list(APPEND subCMakeOpts "-DCMAKE_MAKE_PROGRAM:FILEPATH=${CMAKE_MAKE_PROGRAM}")
    endif()

    # GreenHills needs to know about the compiler and toolset to run the
    # subbuild commands. Be sure to update the similar section in
    # ExternalProject.cmake:_ep_extract_configure_command()
    if(CMAKE_GENERATOR MATCHES "Green Hills MULTI")
      list(APPEND subCMakeOpts
        "-DGHS_TARGET_PLATFORM:STRING=${GHS_TARGET_PLATFORM}"
        "-DGHS_PRIMARY_TARGET:STRING=${GHS_PRIMARY_TARGET}"
        "-DGHS_TOOLSET_ROOT:STRING=${GHS_TOOLSET_ROOT}"
        "-DGHS_OS_ROOT:STRING=${GHS_OS_ROOT}"
        "-DGHS_OS_DIR:STRING=${GHS_OS_DIR}"
        "-DGHS_BSP_NAME:STRING=${GHS_BSP_NAME}"
      )
    endif()

    # Override the sub-build's configuration types for multi-config generators.
    # This ensures we are not affected by any custom setting from the project
    # and can always request a known configuration further below.
    get_property(is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    if(is_multi_config)
      list(APPEND subCMakeOpts "-DCMAKE_CONFIGURATION_TYPES:STRING=Debug")
    endif()

  else()
    # Likely we've been invoked via CMake's script mode where no
    # generator is set (and hence CMAKE_MAKE_PROGRAM could not be
    # trusted even if provided). We will have to rely on being
    # able to find the default generator and build tool.
    unset(subCMakeOpts)
  endif()

  set(__FETCHCONTENT_CACHED_INFO "")
  set(__passthrough_vars
    CMAKE_EP_GIT_REMOTE_UPDATE_STRATEGY
    CMAKE_TLS_VERSION
    CMAKE_TLS_VERIFY
    CMAKE_TLS_CAINFO
    CMAKE_NETRC
    CMAKE_NETRC_FILE
  )
  foreach(var IN LISTS __passthrough_vars)
    if(DEFINED ${var})
      # Embed directly in the generated CMakeLists.txt file to avoid making
      # the cmake command line excessively long. It also makes debugging and
      # testing easier.
      string(APPEND __FETCHCONTENT_CACHED_INFO "set(${var} [==[${${var}}]==])\n")
    endif()
  endforeach()

  # Avoid using if(... IN_LIST ...) so we don't have to alter policy settings
  list(FIND ARG_UNPARSED_ARGUMENTS GIT_REPOSITORY indexResult)
  if(indexResult GREATER_EQUAL 0)
    find_package(Git QUIET)
    string(APPEND __FETCHCONTENT_CACHED_INFO "
# Pass through things we've already detected in the main project to avoid
# paying the cost of redetecting them again in ExternalProject_Add()
set(GIT_EXECUTABLE [==[${GIT_EXECUTABLE}]==])
set(Git_VERSION [==[${Git_VERSION}]==])
set_property(GLOBAL PROPERTY _CMAKE_FindGit_GIT_EXECUTABLE_VERSION
  [==[${GIT_EXECUTABLE};${Git_VERSION}]==]
)
")
  endif()

  # Create and build a separate CMake project to carry out the population.
  # If we've already previously done these steps, they will not cause
  # anything to be updated, so extra rebuilds of the project won't occur.
  # Make sure to pass through CMAKE_MAKE_PROGRAM in case the main project
  # has this set to something not findable on the PATH. We also ensured above
  # that the Debug config will be defined for multi-config generators.
  configure_file("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/FetchContent/CMakeLists.cmake.in"
                 "${ARG_SUBBUILD_DIR}/CMakeLists.txt"
                 @ONLY
  )
  execute_process(
    COMMAND ${CMAKE_COMMAND} ${subCMakeOpts} .
    RESULT_VARIABLE result
    ${outputOptions}
    WORKING_DIRECTORY "${ARG_SUBBUILD_DIR}"
  )
  if(result)
    if(capturedOutput)
      message("${capturedOutput}")
    endif()
    message(FATAL_ERROR "CMake step for ${contentName} failed: ${result}")
  endif()
  execute_process(
    COMMAND ${CMAKE_COMMAND} --build . --config Debug
    RESULT_VARIABLE result
    ${outputOptions}
    WORKING_DIRECTORY "${ARG_SUBBUILD_DIR}"
  )
  if(result)
    if(capturedOutput)
      message("${capturedOutput}")
    endif()
    message(FATAL_ERROR "Build step for ${contentName} failed: ${result}")
  endif()

endfunction()


option(FETCHCONTENT_FULLY_DISCONNECTED   "Disables all attempts to download or update content and assumes source dirs already exist")
option(FETCHCONTENT_UPDATES_DISCONNECTED "Enables UPDATE_DISCONNECTED behavior for all content population")
option(FETCHCONTENT_QUIET                "Enables QUIET option for all content population" ON)
set(FETCHCONTENT_BASE_DIR "${CMAKE_BINARY_DIR}/_deps" CACHE PATH "Directory under which to collect all populated content")

# Populate the specified content using details stored from
# an earlier call to FetchContent_Declare().
function(FetchContent_Populate contentName)

  if(NOT contentName)
    message(FATAL_ERROR "Empty contentName not allowed for FetchContent_Populate()")
  endif()

  if(ARGC EQUAL 1)
    cmake_policy(GET CMP0169 cmp0169
      PARENT_SCOPE # undocumented, do not use outside of CMake
    )
    if(NOT cmp0169 STREQUAL "OLD")
      string(CONCAT msg
        "Calling FetchContent_Populate(${contentName}) is deprecated, call "
        "FetchContent_MakeAvailable(${contentName}) instead. "
        "Policy CMP0169 can be set to OLD to allow "
        "FetchContent_Populate(${contentName}) to be called directly for now, "
        "but the ability to call it with declared details will be removed "
        "completely in a future version."
      )
      if(cmp0169 STREQUAL "NEW")
        message(FATAL_ERROR "${msg}")
      else()
        message(AUTHOR_WARNING "${msg}")
      endif()
    endif()
    set(__doDirectArgs)
  else()
    cmake_policy(GET CMP0168 cmp0168
      PARENT_SCOPE # undocumented, do not use outside of CMake
    )
    if(cmp0168 STREQUAL "NEW")
      set(__doDirectArgs __DIRECT_POPULATION YES)
    else()
      set(__doDirectArgs __DIRECT_POPULATION NO)
    endif()
  endif()

  cmake_policy(GET CMP0170 cmp0170
    PARENT_SCOPE # undocumented, do not use outside of CMake
  )

  cmake_parse_arguments(PARSE_ARGV 1 __arg "" "" "")
  set(__argsQuoted "[==[${contentName}]==] [==[${cmp0170}]==]")
  foreach(__item IN LISTS __arg_UNPARSED_ARGUMENTS __doDirectArgs)
    string(APPEND __argsQuoted " [==[${__item}]==]")
  endforeach()

  cmake_language(EVAL CODE "__FetchContent_Populate(${__argsQuoted})")

  string(TOLOWER ${contentName} contentNameLower)
  foreach(var IN ITEMS SOURCE_DIR BINARY_DIR POPULATED)
    set(var "${contentNameLower}_${var}")
    if(DEFINED ${var})
      set(${var} "${${var}}" PARENT_SCOPE)
    endif()
  endforeach()

endfunction()

function(__FetchContent_Populate contentName cmp0170)

  string(TOLOWER ${contentName} contentNameLower)

  if(ARGN)
    # This is the direct population form with details fully specified
    # as part of the call, so we already have everything we need
    __FetchContent_doPopulation(
      ${contentNameLower}
      SUBBUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/${contentNameLower}-subbuild"
      SOURCE_DIR   "${CMAKE_CURRENT_BINARY_DIR}/${contentNameLower}-src"
      BINARY_DIR   "${CMAKE_CURRENT_BINARY_DIR}/${contentNameLower}-build"
      ${ARGN}  # Could override any of the above ..._DIR variables
    )

    # Pass source and binary dir variables back to the caller
    set(${contentNameLower}_SOURCE_DIR "${${contentNameLower}_SOURCE_DIR}" PARENT_SCOPE)
    set(${contentNameLower}_BINARY_DIR "${${contentNameLower}_BINARY_DIR}" PARENT_SCOPE)

    # Don't set global properties, or record that we did this population, since
    # this was a direct call outside of the normal declared details form.
    # We only want to save values in the global properties for content that
    # honors the hierarchical details mechanism so that projects are not
    # robbed of the ability to override details set in nested projects.
    return()
  endif()

  # No details provided, so assume they were saved from an earlier call
  # to FetchContent_Declare(). Do a check that we haven't already
  # populated this content before in case the caller forgot to check.
  FetchContent_GetProperties(${contentName})
  if(${contentNameLower}_POPULATED)
    if("${${contentNameLower}_SOURCE_DIR}" STREQUAL "")
      message(FATAL_ERROR
        "Content ${contentName} already populated by find_package() or a "
        "dependency provider"
      )
    else()
      message(FATAL_ERROR
        "Content ${contentName} already populated in ${${contentNameLower}_SOURCE_DIR}"
      )
    endif()
  endif()

  __FetchContent_getSavedDetails(${contentName} contentDetails)
  if("${contentDetails}" STREQUAL "")
    message(FATAL_ERROR "No details have been set for content: ${contentName}")
  endif()

  string(TOUPPER ${contentName} contentNameUpper)
  set(FETCHCONTENT_SOURCE_DIR_${contentNameUpper}
      "${FETCHCONTENT_SOURCE_DIR_${contentNameUpper}}"
      CACHE PATH "When not empty, overrides where to find pre-populated content for ${contentName}")

  if(FETCHCONTENT_SOURCE_DIR_${contentNameUpper})
    # The source directory has been explicitly provided in the cache,
    # so no population is required. The build directory may still be specified
    # by the declared details though.

    if(NOT IS_ABSOLUTE "${FETCHCONTENT_SOURCE_DIR_${contentNameUpper}}")
      # Don't check this directory because we don't know what location it is
      # expected to be relative to. We can't make this a hard error for backward
      # compatibility reasons.
      message(WARNING "Relative source directory specified. This is not safe, "
        "as it depends on the calling directory scope.\n"
        "  FETCHCONTENT_SOURCE_DIR_${contentNameUpper} --> ${FETCHCONTENT_SOURCE_DIR_${contentNameUpper}}")
    elseif(NOT EXISTS "${FETCHCONTENT_SOURCE_DIR_${contentNameUpper}}")
      message(FATAL_ERROR "Manually specified source directory is missing:\n"
        "  FETCHCONTENT_SOURCE_DIR_${contentNameUpper} --> ${FETCHCONTENT_SOURCE_DIR_${contentNameUpper}}")
    endif()

    set(${contentNameLower}_SOURCE_DIR "${FETCHCONTENT_SOURCE_DIR_${contentNameUpper}}")

    cmake_parse_arguments(savedDetails "" "BINARY_DIR" "" ${contentDetails})

    if(savedDetails_BINARY_DIR)
      set(${contentNameLower}_BINARY_DIR ${savedDetails_BINARY_DIR})
    else()
      set(${contentNameLower}_BINARY_DIR "${FETCHCONTENT_BASE_DIR}/${contentNameLower}-build")
    endif()

  elseif(FETCHCONTENT_FULLY_DISCONNECTED)
    # Bypass population and assume source is already there from a previous run.
    # Declared details may override the default source or build directories.

    cmake_parse_arguments(savedDetails "" "SOURCE_DIR;BINARY_DIR" "" ${contentDetails})

    if(savedDetails_SOURCE_DIR)
      set(${contentNameLower}_SOURCE_DIR ${savedDetails_SOURCE_DIR})
    else()
      set(${contentNameLower}_SOURCE_DIR "${FETCHCONTENT_BASE_DIR}/${contentNameLower}-src")
    endif()
    if(NOT IS_ABSOLUTE "${${contentNameLower}_SOURCE_DIR}")
      message(WARNING
        "Relative source directory specified. This is not safe, as it depends "
        "on the calling directory scope.\n"
        "  ${${contentNameLower}_SOURCE_DIR}"
      )
      set(${contentNameLower}_SOURCE_DIR
        "${CMAKE_CURRENT_BINARY_DIR}/${${contentNameLower}_SOURCE_DIR}"
      )
    endif()
    if(NOT EXISTS "${${contentNameLower}_SOURCE_DIR}")
      if(cmp0170 STREQUAL "")
        set(cmp0170 WARN)
      endif()
      string(CONCAT msg
        "FETCHCONTENT_FULLY_DISCONNECTED is set to true, which requires the "
        "source directory for dependency ${contentName} to already be populated. "
        "This generally means it must not be set to true the first time CMake "
        "is run in a build directory. The following source directory should "
        "already be populated, but it doesn't exist:\n"
        "  ${${contentNameLower}_SOURCE_DIR}\n"
        "Policy CMP0170 controls enforcement of this requirement."
      )
      if(cmp0170 STREQUAL "NEW")
        message(FATAL_ERROR "${msg}")
      elseif(NOT cmp0170 STREQUAL "OLD")
        # Note that this is a user warning, not a project author warning.
        # The user has set FETCHCONTENT_FULLY_DISCONNECTED in a scenario
        # where that is not allowed.
        message(WARNING "${msg}")
      endif()
    endif()

    if(savedDetails_BINARY_DIR)
      set(${contentNameLower}_BINARY_DIR ${savedDetails_BINARY_DIR})
    else()
      set(${contentNameLower}_BINARY_DIR "${FETCHCONTENT_BASE_DIR}/${contentNameLower}-build")
    endif()

  else()
    # Support both a global "disconnect all updates" and a per-content
    # update test (either one being set disables updates for this content).
    option(FETCHCONTENT_UPDATES_DISCONNECTED_${contentNameUpper}
           "Enables UPDATE_DISCONNECTED behavior just for population of ${contentName}")
    if(FETCHCONTENT_UPDATES_DISCONNECTED OR
       FETCHCONTENT_UPDATES_DISCONNECTED_${contentNameUpper})
      set(disconnectUpdates True)
    else()
      set(disconnectUpdates False)
    endif()

    if(FETCHCONTENT_QUIET)
      set(quietFlag QUIET)
    else()
      unset(quietFlag)
    endif()

    set(__detailsQuoted)
    foreach(__item IN LISTS contentDetails)
      if(NOT __item STREQUAL "OVERRIDE_FIND_PACKAGE")
        string(APPEND __detailsQuoted " [==[${__item}]==]")
      endif()
    endforeach()
    cmake_language(EVAL CODE "
      __FetchContent_doPopulation(
        ${contentNameLower}
        ${quietFlag}
        UPDATE_DISCONNECTED ${disconnectUpdates}
        SUBBUILD_DIR \"${FETCHCONTENT_BASE_DIR}/${contentNameLower}-subbuild\"
        SOURCE_DIR   \"${FETCHCONTENT_BASE_DIR}/${contentNameLower}-src\"
        BINARY_DIR   \"${FETCHCONTENT_BASE_DIR}/${contentNameLower}-build\"
        # Put the saved details last so they can override any of the
        # the options we set above (this can include SOURCE_DIR or
        # BUILD_DIR)
        ${__detailsQuoted}
      )"
    )
  endif()

  FetchContent_SetPopulated(
    ${contentName}
    SOURCE_DIR "${${contentNameLower}_SOURCE_DIR}"
    BINARY_DIR "${${contentNameLower}_BINARY_DIR}"
  )

  # Pass variables back to the caller. The variables passed back here
  # must match what FetchContent_GetProperties() sets when it is called
  # with just the content name.
  set(${contentNameLower}_SOURCE_DIR "${${contentNameLower}_SOURCE_DIR}" PARENT_SCOPE)
  set(${contentNameLower}_BINARY_DIR "${${contentNameLower}_BINARY_DIR}" PARENT_SCOPE)
  set(${contentNameLower}_POPULATED  True PARENT_SCOPE)

endfunction()

function(__FetchContent_setupFindPackageRedirection contentName)

  __FetchContent_getSavedDetails(${contentName} contentDetails)

  string(TOLOWER ${contentName} contentNameLower)
  get_property(wantFindPackage GLOBAL PROPERTY
    _FetchContent_${contentNameLower}_find_package_args
    DEFINED
  )

  # Avoid using if(... IN_LIST ...) so we don't have to alter policy settings
  list(FIND contentDetails OVERRIDE_FIND_PACKAGE indexResult)
  if(NOT wantFindPackage AND indexResult EQUAL -1)
    # No find_package() redirection allowed
    return()
  endif()

  # We write out dep-config.cmake and dep-config-version.cmake file name
  # forms here because they are forced to lowercase. FetchContent
  # dependency names are case-insensitive, but find_package() config files
  # are only case-insensitive for the -config and -config-version forms,
  # not the Config and ConfigVersion forms.
  set(inFileDir ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/FetchContent)
  set(configFilePrefix1 "${CMAKE_FIND_PACKAGE_REDIRECTS_DIR}/${contentName}Config")
  set(configFilePrefix2 "${CMAKE_FIND_PACKAGE_REDIRECTS_DIR}/${contentNameLower}-config")
  if(NOT EXISTS "${configFilePrefix1}.cmake" AND
    NOT EXISTS "${configFilePrefix2}.cmake")
    configure_file(${inFileDir}/package-config.cmake.in
      "${configFilePrefix2}.cmake" @ONLY
    )
  endif()
  if(NOT EXISTS "${configFilePrefix1}Version.cmake" AND
    NOT EXISTS "${configFilePrefix2}-version.cmake")
    configure_file(${inFileDir}/package-config-version.cmake.in
      "${configFilePrefix2}-version.cmake" @ONLY
    )
  endif()

  # Now that we've created the redirected package config files, prevent
  # find_package() from delegating to FetchContent and let it find these
  # config files through its normal processing.
  set(propertyName "${prefix}_override_find_package")
  set(GLOBAL PROPERTY ${propertyName} FALSE)
  set(${contentName}_DIR "${CMAKE_FIND_PACKAGE_REDIRECTS_DIR}"
    CACHE INTERNAL "Redirected by FetchContent"
  )

endfunction()

# Arguments are assumed to be the names of dependencies that have been
# declared previously and should be populated. It is not an error if
# any of them have already been populated (they will just be skipped in
# that case). The command is implemented as a macro so that the variables
# defined by the FetchContent_GetProperties() and FetchContent_Populate()
# calls will be available to the caller.
macro(FetchContent_MakeAvailable)

  # We must append an item, even if the variable is unset, so prefix its value.
  # We will strip that prefix when we pop the value at the end of the macro.
  list(APPEND __cmake_fcCurrentVarsStack
    "__fcprefix__${CMAKE_VERIFY_INTERFACE_HEADER_SETS}"
  )
  set(CMAKE_VERIFY_INTERFACE_HEADER_SETS FALSE)

  get_property(__cmake_providerCommand GLOBAL PROPERTY
    __FETCHCONTENT_MAKEAVAILABLE_SERIAL_PROVIDER
  )
  foreach(__cmake_contentName IN ITEMS ${ARGV})
    string(TOLOWER ${__cmake_contentName} __cmake_contentNameLower)

    # If user specified FETCHCONTENT_SOURCE_DIR_... for this dependency, that
    # overrides everything else and we shouldn't try to use find_package() or
    # a dependency provider.
    string(TOUPPER ${__cmake_contentName} __cmake_contentNameUpper)
    if("${FETCHCONTENT_SOURCE_DIR_${__cmake_contentNameUpper}}" STREQUAL "")
      # Dependency provider gets first opportunity, but prevent infinite
      # recursion if we are called again for the same thing
      if(NOT "${__cmake_providerCommand}" STREQUAL "" AND
        NOT DEFINED __cmake_fcProvider_${__cmake_contentNameLower})
        message(VERBOSE
          "Trying FETCHCONTENT_MAKEAVAILABLE_SERIAL dependency provider for "
          "${__cmake_contentName}"
        )

        if(DEFINED CMAKE_EXPORT_FIND_PACKAGE_NAME)
          list(APPEND __cmake_fcCurrentVarsStack "${CMAKE_EXPORT_FIND_PACKAGE_NAME}")
        else()
          # This just needs to be something that can't be a real package name
          list(APPEND __cmake_fcCurrentVarsStack "<<::VAR_NOT_SET::>>")
        endif()
        set(CMAKE_EXPORT_FIND_PACKAGE_NAME "${__cmake_contentName}")

        # It's still valid if there are no saved details. The project may have
        # been written to assume a dependency provider is always set and will
        # provide dependencies without having any declared details for them.
        __FetchContent_getSavedDetails(${__cmake_contentName} __cmake_contentDetails)
        set(__cmake_providerArgs
          "FETCHCONTENT_MAKEAVAILABLE_SERIAL"
          "${__cmake_contentName}"
        )
        # Empty arguments must be preserved because of things like
        # GIT_SUBMODULES (see CMP0097)
        foreach(__cmake_item IN LISTS __cmake_contentDetails)
          string(APPEND __cmake_providerArgs " [==[${__cmake_item}]==]")
        endforeach()

        # This property might be defined but empty. As long as it is defined,
        # find_package() can be called.
        get_property(__cmake_addfpargs GLOBAL PROPERTY
          _FetchContent_${__cmake_contentNameLower}_find_package_args
          DEFINED
        )
        if(__cmake_addfpargs)
          get_property(__cmake_fpargs GLOBAL PROPERTY
            _FetchContent_${__cmake_contentNameLower}_find_package_args
          )
          string(APPEND __cmake_providerArgs " FIND_PACKAGE_ARGS")
          foreach(__cmake_item IN LISTS __cmake_fpargs)
            string(APPEND __cmake_providerArgs " [==[${__cmake_item}]==]")
          endforeach()
        endif()

        # Calling the provider could lead to FetchContent_MakeAvailable() being
        # called for a nested dependency. That nested call may occur in the
        # current variable scope. We have to save and restore the variables we
        # need preserved.
        list(APPEND __cmake_fcCurrentVarsStack
          ${__cmake_contentName}
          ${__cmake_contentNameLower}
        )

        set(__cmake_fcProvider_${__cmake_contentNameLower} YES)

        # The provider needs to see policies from our caller, so we need a
        # helper macro defined outside our policy block. We pass through a
        # variable name rather than variable contents to avoid any potential
        # problems with parsing macro arguments.
        set(__cmake_fcCode "${__cmake_providerCommand}(${__cmake_providerArgs})")
        __FetchContent_MakeAvailable_eval_code(__cmake_fcCode)

        list(POP_BACK __cmake_fcCurrentVarsStack
          __cmake_contentNameLower
          __cmake_contentName
          CMAKE_EXPORT_FIND_PACKAGE_NAME
        )
        if(CMAKE_EXPORT_FIND_PACKAGE_NAME STREQUAL "<<::VAR_NOT_SET::>>")
          unset(CMAKE_EXPORT_FIND_PACKAGE_NAME)
        endif()

        unset(__cmake_fcCode)
        unset(__cmake_fcProvider_${__cmake_contentNameLower})
        unset(__cmake_providerArgs)
        unset(__cmake_addfpargs)
        unset(__cmake_fpargs)
        unset(__cmake_item)
        unset(__cmake_contentDetails)

        FetchContent_GetProperties(${__cmake_contentName})
        if(${__cmake_contentNameLower}_POPULATED)
          continue()
        endif()
      endif()

      # Check if we've been asked to try find_package() first, even if we
      # have already populated this dependency. If we previously tried to
      # use find_package() for this and it succeeded, those things might
      # no longer be in scope, so we have to do it again.
      get_property(__cmake_haveFpArgs GLOBAL PROPERTY
        _FetchContent_${__cmake_contentNameLower}_find_package_args DEFINED
      )
      if(__cmake_haveFpArgs)
        unset(__cmake_haveFpArgs)
        message(VERBOSE "Trying find_package(${__cmake_contentName} ...) before FetchContent")
        get_property(__cmake_fpArgs GLOBAL PROPERTY
          _FetchContent_${__cmake_contentNameLower}_find_package_args
        )

        # This call could lead to FetchContent_MakeAvailable() being called for
        # a nested dependency and it may occur in the current variable scope.
        # We have to save/restore the variables we need to preserve.
        list(APPEND __cmake_fcCurrentNameStack
          ${__cmake_contentName}
          ${__cmake_contentNameLower}
        )
        # We pass variable names rather than their contents so as to avoid any
        # potential problems with macro argument parsing
        __FetchContent_MakeAvailable_find_package(__cmake_contentName __cmake_fpArgs)
        list(POP_BACK __cmake_fcCurrentNameStack
          __cmake_contentNameLower
          __cmake_contentName
        )
        unset(__cmake_fpArgs)

        if(${__cmake_contentName}_FOUND)
          FetchContent_SetPopulated(${__cmake_contentName})
          FetchContent_GetProperties(${__cmake_contentName})
          continue()
        endif()
      endif()
    else()
      unset(__cmake_haveFpArgs)
    endif()

    FetchContent_GetProperties(${__cmake_contentName})
    if(NOT ${__cmake_contentNameLower}_POPULATED)
      cmake_policy(GET CMP0170 __cmake_fc_cmp0170
        PARENT_SCOPE # undocumented, do not use outside of CMake
      )
      __FetchContent_Populate(${__cmake_contentName} "${__cmake_fc_cmp0170}")
      unset(__cmake_fc_cmp0170)
      __FetchContent_setupFindPackageRedirection(${__cmake_contentName})

      # Only try to call add_subdirectory() if the populated content
      # can be treated that way. Protecting the call with the check
      # allows this function to be used for projects that just want
      # to ensure the content exists, such as to provide content at
      # a known location. We check the saved details for an optional
      # SOURCE_SUBDIR which can be used in the same way as its meaning
      # for ExternalProject. It won't matter if it was passed through
      # to the ExternalProject sub-build, since it would have been
      # ignored there.
      set(__cmake_srcdir "${${__cmake_contentNameLower}_SOURCE_DIR}")
      __FetchContent_getSavedDetails(${__cmake_contentName} __cmake_contentDetails)
      if("${__cmake_contentDetails}" STREQUAL "")
        message(FATAL_ERROR "No details have been set for content: ${__cmake_contentName}")
      endif()
      cmake_parse_arguments(__cmake_arg "EXCLUDE_FROM_ALL;SYSTEM" "SOURCE_SUBDIR" "" ${__cmake_contentDetails})
      if(NOT "${__cmake_arg_SOURCE_SUBDIR}" STREQUAL "")
        string(APPEND __cmake_srcdir "/${__cmake_arg_SOURCE_SUBDIR}")
      endif()

      if(EXISTS ${__cmake_srcdir}/CMakeLists.txt)
        if(DEFINED CMAKE_EXPORT_FIND_PACKAGE_NAME)
          list(APPEND __cmake_fcCurrentVarsStack "${CMAKE_EXPORT_FIND_PACKAGE_NAME}")
        else()
          # This just needs to be something that can't be a real package name
          list(APPEND __cmake_fcCurrentVarsStack "<<::VAR_NOT_SET::>>")
        endif()
        set(CMAKE_EXPORT_FIND_PACKAGE_NAME "${__cmake_contentName}")

        set(__cmake_add_subdirectory_args ${__cmake_srcdir} ${${__cmake_contentNameLower}_BINARY_DIR})
        if(__cmake_arg_EXCLUDE_FROM_ALL)
          list(APPEND __cmake_add_subdirectory_args EXCLUDE_FROM_ALL)
        endif()
        if(__cmake_arg_SYSTEM)
          list(APPEND __cmake_add_subdirectory_args SYSTEM)
        endif()

        # We pass a variable name rather than its contents so as to avoid any
        # potential problems with macro argument parsing. It's highly unlikely
        # in this case, but still theoretically possible someone might try to
        # use a directory name that looks like a CMake variable evaluation.
        __FetchContent_MakeAvailable_add_subdirectory(__cmake_add_subdirectory_args)

        list(POP_BACK __cmake_fcCurrentVarsStack CMAKE_EXPORT_FIND_PACKAGE_NAME)
        if(CMAKE_EXPORT_FIND_PACKAGE_NAME STREQUAL "<<::VAR_NOT_SET::>>")
          unset(CMAKE_EXPORT_FIND_PACKAGE_NAME)
        endif()
      endif()

      unset(__cmake_srcdir)
      unset(__cmake_contentDetails)
      unset(__cmake_arg_EXCLUDE_FROM_ALL)
      unset(__cmake_arg_SYSTEM)
      unset(__cmake_arg_SOURCE_SUBDIR)
      unset(__cmake_add_subdirectory_args)
    endif()
  endforeach()

  # Prefix will be "__fcprefix__"
  list(POP_BACK __cmake_fcCurrentVarsStack __cmake_original_verify_setting)
  string(SUBSTRING "${__cmake_original_verify_setting}"
    12 -1 __cmake_original_verify_setting
  )
  set(CMAKE_VERIFY_INTERFACE_HEADER_SETS ${__cmake_original_verify_setting})

  # clear local variables to prevent leaking into the caller's scope
  unset(__cmake_contentName)
  unset(__cmake_contentNameLower)
  unset(__cmake_contentNameUpper)
  unset(__cmake_providerCommand)
  unset(__cmake_original_verify_setting)

endmacro()

endblock()   # End of FetchContent module's policy scope

# These are factored out here outside our policies block to preserve policy
# settings of the scope from which FetchContent was included. Any project or
# user code that actually relies on this is fragile and should enforce its own
# policies instead, but we keep these here to preserve backward compatibility.
macro(__FetchContent_MakeAvailable_eval_code code_var)
  cmake_language(EVAL CODE "${${code_var}}")
endmacro()

macro(__FetchContent_MakeAvailable_find_package first_arg_var remaining_args_var)
  find_package(${${first_arg_var}} ${${remaining_args_var}})
endmacro()

macro(__FetchContent_MakeAvailable_add_subdirectory args_var)
  add_subdirectory(${${args_var}})
endmacro()
