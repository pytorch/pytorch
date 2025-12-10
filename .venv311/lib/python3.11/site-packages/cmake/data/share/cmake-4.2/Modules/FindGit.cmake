# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindGit
-------

Finds the Git distributed version control system:

.. code-block:: cmake

  find_package(Git [<version>] [...])

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following :ref:`Imported Targets` when the
:prop_gbl:`CMAKE_ROLE` is ``PROJECT``:

``Git::Git``
  .. versionadded:: 3.14

  Target that encapsulates Git command-line client executable.  It can be used
  in :manual:`generator expressions <cmake-generator-expressions(7)>`, and
  commands like :command:`add_custom_target` and :command:`add_custom_command`.
  This target is available only if Git is found.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``Git_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether (the requested version of) Git was found.

``Git_VERSION``
  .. versionadded:: 4.2

  The version of Git found.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``GIT_EXECUTABLE``
  Path to the ``git`` command-line client executable.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``GIT_FOUND``
  .. deprecated:: 4.2
    Use ``Git_FOUND``, which has the same value.

  Boolean indicating whether (the requested version of) Git was found.

``GIT_VERSION_STRING``
  .. deprecated:: 4.2
    Use ``Git_VERSION``, which has the same value.

  The version of Git found.

Examples
^^^^^^^^

Finding Git and retrieving the latest commit from the project repository:

.. code-block:: cmake

  find_package(Git)
  if(Git_FOUND)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} --no-pager log -n 1 HEAD "--pretty=format:%h %s"
      OUTPUT_VARIABLE output
      RESULT_VARIABLE result
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(result EQUAL 0)
      message(STATUS "Last Git commit: ${output}")
    endif()
  endif()
#]=======================================================================]

# Look for 'git'
#
set(git_names git)

# Prefer .cmd variants on Windows unless running in a Makefile
# in the MSYS shell.
#
if(CMAKE_HOST_WIN32)
  if(NOT CMAKE_GENERATOR MATCHES "MSYS")
    set(git_names git.cmd git)
    # GitHub search path for Windows
    file(GLOB github_path
      "$ENV{LOCALAPPDATA}/Github/PortableGit*/cmd"
      "$ENV{LOCALAPPDATA}/Github/PortableGit*/bin"
      )
    # SourceTree search path for Windows
    set(_git_sourcetree_path "$ENV{LOCALAPPDATA}/Atlassian/SourceTree/git_local/bin")
  endif()
endif()

# First search the PATH and specific locations.
find_program(GIT_EXECUTABLE
  NAMES ${git_names}
  PATHS ${github_path} ${_git_sourcetree_path}
  DOC "Git command line client"
  )

if(CMAKE_HOST_WIN32)
  # Now look for installations in Git/ directories under typical installation
  # prefixes on Windows.  Exclude PATH from this search because VS 2017's
  # command prompt happens to have a PATH entry with a Git/ subdirectory
  # containing a minimal git not meant for general use.
  find_program(GIT_EXECUTABLE
    NAMES ${git_names}
    PATH_SUFFIXES Git/cmd Git/bin
    NO_SYSTEM_ENVIRONMENT_PATH
    DOC "Git command line client"
    )
endif()

mark_as_advanced(GIT_EXECUTABLE)

unset(git_names)
unset(_git_sourcetree_path)

if(GIT_EXECUTABLE)
  # Avoid querying the version if we've already done that this run. For
  # projects that use things like ExternalProject or FetchContent heavily,
  # this saving can be measurable on some platforms.
  #
  # This is an internal property, projects must not try to use it.
  # We don't want this stored in the cache because it might still change
  # between CMake runs, but it shouldn't change during a run for a given
  # git executable location.
  set(__doGitVersionCheck TRUE)
  get_property(__gitVersionProp GLOBAL
    PROPERTY _CMAKE_FindGit_GIT_EXECUTABLE_VERSION
  )
  if(__gitVersionProp)
    list(GET __gitVersionProp 0 __gitExe)
    list(GET __gitVersionProp 1 __gitVersion)
    if(__gitExe STREQUAL GIT_EXECUTABLE AND NOT __gitVersion STREQUAL "")
      set(Git_VERSION "${__gitVersion}")
      set(GIT_VERSION_STRING "${Git_VERSION}")
      set(__doGitVersionCheck FALSE)
    endif()
    unset(__gitExe)
    unset(__gitVersion)
  endif()
  unset(__gitVersionProp)

  if(__doGitVersionCheck)
    execute_process(COMMAND ${GIT_EXECUTABLE} --version
                    OUTPUT_VARIABLE git_version
                    ERROR_QUIET
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    if (git_version MATCHES "^git version [0-9]")
      string(REPLACE "git version " "" Git_VERSION "${git_version}")
      set(GIT_VERSION_STRING "${Git_VERSION}")
      set_property(GLOBAL PROPERTY _CMAKE_FindGit_GIT_EXECUTABLE_VERSION
        "${GIT_EXECUTABLE};${Git_VERSION}"
      )
    endif()
    unset(git_version)
  endif()
  unset(__doGitVersionCheck)

  get_property(_findgit_role GLOBAL PROPERTY CMAKE_ROLE)
  if(_findgit_role STREQUAL "PROJECT" AND NOT TARGET Git::Git)
    add_executable(Git::Git IMPORTED)
    set_property(TARGET Git::Git PROPERTY IMPORTED_LOCATION "${GIT_EXECUTABLE}")
  endif()
  unset(_findgit_role)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Git
                                  REQUIRED_VARS GIT_EXECUTABLE
                                  VERSION_VAR Git_VERSION)
