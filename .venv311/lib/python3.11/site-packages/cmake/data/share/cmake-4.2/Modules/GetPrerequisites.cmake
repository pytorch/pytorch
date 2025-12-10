# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
GetPrerequisites
----------------

.. deprecated:: 3.16

  Use :command:`file(GET_RUNTIME_DEPENDENCIES)` instead.

This module provides commands to analyze and list the dependencies
(prerequisites) of executable or shared library files.  These commands list
the shared libraries (``.dll``, ``.dylib``, or ``.so`` files) required by an
executable or shared library.

Load this module in CMake with:

.. code-block:: cmake

  include(GetPrerequisites)

This module determines dependencies using the following platform-specific
tools:

* ``dumpbin`` (Windows)
* ``objdump`` (MinGW on Windows)
* ``ldd`` (Linux/Unix)
* ``otool`` (Apple operating systems)

.. versionchanged:: 3.16
  The tool specified by the :variable:`CMAKE_OBJDUMP` variable will be used, if
  set.

Commands
^^^^^^^^

This module provides the following commands:

* :command:`get_prerequisites`
* :command:`list_prerequisites`
* :command:`list_prerequisites_by_glob`
* :command:`gp_append_unique`
* :command:`is_file_executable`
* :command:`gp_item_default_embedded_path`
  (projects can override it with ``gp_item_default_embedded_path_override()``)
* :command:`gp_resolve_item`
  (projects can override it with ``gp_resolve_item_override()``)
* :command:`gp_resolved_file_type`
  (projects can override it with ``gp_resolved_file_type_override()``)
* :command:`gp_file_type`

.. command:: get_prerequisites

  Gets the list of shared library files required by specified target:

  .. code-block:: cmake

    get_prerequisites(<target> <prerequisites-var> <exclude-system> <recurse>
                      <exepath> <dirs> [<rpaths>])

  The list in the variable named ``<prerequisites-var>`` should be empty on
  first entry to this command.  On exit, ``<prerequisites-var>`` will contain
  the list of required shared library files.

  The arguments are:

  ``<target>``
    The full path to an executable or shared library file.
  ``<prerequisites-var>``
    The name of a CMake variable to contain the results.
  ``<exclude-system>``
    If set to 1 system prerequisites will be excluded, if set to 0 they will be
    included.
  ``<recurse>``
    If set to 1 all prerequisites will be found recursively, if set to 0 only
    direct prerequisites are listed.
  ``<exepath>``
    The path to the top level executable used for ``@executable_path``
    replacement on Apple operating systems.
  ``<dirs>``
    A list of paths where libraries might be found: these paths are searched
    first when a target without any path info is given.  Then standard system
    locations are also searched: PATH, Framework locations, /usr/lib...
  ``<rpaths>``
    Optional run-time search paths for an executable file or library to help
    find files.

  .. versionadded:: 3.14
    The variable ``GET_PREREQUISITES_VERBOSE`` can be set to true before calling
    this command to enable verbose output.

.. command:: list_prerequisites

  Prints a message listing the prerequisites of the specified target:

  .. code-block:: cmake

    list_prerequisites(<target> [<recurse> [<exclude-system> [<verbose>]]])

  The arguments are:

  ``<target>``
    The name of a shared library or executable target or the full path to a
    shared library or executable file.
  ``<recurse>``
    If set to 1 all prerequisites will be found recursively, if set to 0 only
    direct prerequisites are listed.
  ``<exclude-system>``
    If set to 1 system prerequisites will be excluded, if set to 0 they will be
    included.
  ``<verbose>``
    If set to 0 only the full path names of the prerequisites are printed. If
    set to 1 extra information will be displayed.

.. command:: list_prerequisites_by_glob

  Prints the prerequisites of shared library and executable files matching a
  globbing pattern:

  .. code-block:: cmake

    list_prerequisites_by_glob(<GLOB|GLOB_RECURSE>
                               <glob-exp>
                               [<optional-args>...])

  The arguments are:

  ``GLOB`` or ``GLOB_RECURSE``
    The globbing mode, whether to traverse only the match or also its
    subdirectories recursively.
  ``<glob-exp>``
    A globbing expression used with :command:`file(GLOB)` or
    :command:`file(GLOB_RECURSE)` to retrieve a list of matching files.  If a
    matching file is executable, its prerequisites are listed.
  ``<optional-args>...``
    Any additional (optional) arguments provided are passed along as the
    optional arguments to the ``list_prerequisite()`` calls.

.. command:: gp_append_unique

  Appends the value to the list only if it is not already in the list:

  .. code-block:: cmake

    gp_append_unique(<list-var> <value>)

  The arguments are:

  ``<value>``
    The value to be appended to the list.
  ``<list-var>``
    The list variable name that will have the value appended only if it is
    not already in the list.

.. command:: is_file_executable

  Checks if given file is a binary executable:

  .. code-block:: cmake

    is_file_executable(<file> <result-var>)

  This command sets the ``<result-var>`` to 1 if ``<file>`` is a binary
  executable; otherwise it sets it to 0.

.. command:: gp_item_default_embedded_path

  Determines the reference path for the specified item:

  .. code-block:: cmake

    gp_item_default_embedded_path(<item> <default-embedded-path-var>)

  This command determines the reference path for ``<item>`` when it is
  embedded inside a bundle and stores it to a variable
  ``<default-embedded-path-var>``.

  Projects can override this command by defining a custom
  ``gp_item_default_embedded_path_override()`` command.

.. command:: gp_resolve_item

  Resolves a given item into an existing full path file and stores it to a
  variable:

  .. code-block:: cmake

    gp_resolve_item(<context> <item> <exepath> <dirs> <resolved-item-var>
                    [<rpaths>])

  The arguments are:

  ``<context>``
    The path to the top level loading path used for ``@loader_path`` replacement
    on Apple operating systems.  When resolving item, ``@loader_path``
    references will be resolved relative to the directory of the given context
    value (presumably another library).
  ``<item>``
    The item to resolve.
  ``<exepath>``
    See the argument description in :command:`get_prerequisites`.
  ``<dirs>``
    See the argument description in :command:`get_prerequisites`.
  ``<resolved-item-var>``
    The result variable where the resolved item is stored into.
  ``<rpaths>``
    See the argument description in :command:`get_prerequisites`.

  Projects can override this command by defining a custom
  ``gp_resolve_item_override()`` command.

.. command:: gp_resolved_file_type

  Determines the type of a given file:

  .. code-block:: cmake

    gp_resolved_file_type(<original-file> <file> <exepath> <dirs> <type-var>
                          [<rpaths>])

  This command determines the type of ``<file>`` with respect to the
  ``<original-file>``.  The resulting type of prerequisite is stored in the
  ``<type-var>`` variable.

  Use ``<exepath>`` and ``<dirs>`` if necessary to resolve non-absolute
  ``<file>`` values -- but only for non-embedded items.

  ``<rpaths>``
    See the argument description in :command:`get_prerequisites`.

  The ``<type-var>`` variable will be set to one of the following values:

  * ``system``
  * ``local``
  * ``embedded``
  * ``other``

  Projects can override this command by defining a custom
  ``gp_resolved_file_type_override()`` command.

.. command:: gp_file_type

  Determines the type of a given file:

  .. code-block:: cmake

    gp_file_type(<original-file> <file> <type-var>)

  This command determines the type of ``<file>`` with respect to the
  ``<original-file>``.  The resulting type of prerequisite is stored in the
  ``<type-var>`` variable.

  The ``<type-var>`` variable will be set to one of the following values:

  * ``system``
  * ``local``
  * ``embedded``
  * ``other``

Examples
^^^^^^^^

Example: Basic Usage
""""""""""""""""""""

Printing all dependencies of a shared library, including system libraries, with
verbose output:

.. code-block:: cmake

  include(GetPrerequisites)
  list_prerequisites("path/to/libfoo.dylib" 1 0 1)

Example: Upgrading Code
"""""""""""""""""""""""

For example:

.. code-block:: cmake

  include(GetPrerequisites)
  # ...
  gp_append_unique(keys "${key}")

the ``gp_append_unique()`` can be in new code replaced with:

.. code-block:: cmake

  if(NOT key IN_LIST keys)
    list(APPEND keys "${key}")
  endif()
#]=======================================================================]

function(gp_append_unique list_var value)
  if(NOT value IN_LIST ${list_var})
    set(${list_var} ${${list_var}} "${value}" PARENT_SCOPE)
  endif()
endfunction()


function(is_file_executable file result_var)
  #
  # A file is not executable until proven otherwise:
  #
  set(${result_var} 0 PARENT_SCOPE)

  get_filename_component(file_full "${file}" ABSOLUTE)
  string(TOLOWER "${file_full}" file_full_lower)

  # If file name ends in .exe on Windows, *assume* executable:
  #
  if(WIN32 AND NOT UNIX)
    if("${file_full_lower}" MATCHES "\\.exe$")
      set(${result_var} 1 PARENT_SCOPE)
      return()
    endif()

    # A clause could be added here that uses output or return value of dumpbin
    # to determine ${result_var}. In 99%+? practical cases, the exe name
    # match will be sufficient...
    #
  endif()

  # Use the information returned from the Unix shell command "file" to
  # determine if ${file_full} should be considered an executable file...
  #
  # If the file command's output contains "executable" and does *not* contain
  # "text" then it is likely an executable suitable for prerequisite analysis
  # via the get_prerequisites macro.
  #
  if(UNIX)
    if(NOT file_cmd)
      find_program(file_cmd "file")
      mark_as_advanced(file_cmd)
    endif()

    if(file_cmd)
      execute_process(COMMAND "${file_cmd}" "${file_full}"
        RESULT_VARIABLE file_rv
        OUTPUT_VARIABLE file_ov
        ERROR_VARIABLE file_ev
        OUTPUT_STRIP_TRAILING_WHITESPACE
        )
      if(NOT file_rv STREQUAL "0")
        message(FATAL_ERROR "${file_cmd} failed: ${file_rv}\n${file_ev}")
      endif()

      # Replace the name of the file in the output with a placeholder token
      # (the string " _file_full_ ") so that just in case the path name of
      # the file contains the word "text" or "executable" we are not fooled
      # into thinking "the wrong thing" because the file name matches the
      # other 'file' command output we are looking for...
      #
      string(REPLACE "${file_full}" " _file_full_ " file_ov "${file_ov}")
      string(TOLOWER "${file_ov}" file_ov)

      #message(STATUS "file_ov='${file_ov}'")
      if("${file_ov}" MATCHES "executable")
        #message(STATUS "executable!")
        if("${file_ov}" MATCHES "text")
          #message(STATUS "but text, so *not* a binary executable!")
        else()
          set(${result_var} 1 PARENT_SCOPE)
          return()
        endif()
      endif()

      # Also detect position independent executables on Linux,
      # where "file" gives "shared object ... (uses shared libraries)"
      if("${file_ov}" MATCHES "shared object.*\(uses shared libs\)")
        set(${result_var} 1 PARENT_SCOPE)
        return()
      endif()

      # "file" version 5.22 does not print "(used shared libraries)"
      # but uses "interpreter"
      if("${file_ov}" MATCHES "shared object.*interpreter")
        set(${result_var} 1 PARENT_SCOPE)
        return()
      endif()

    else()
      message(STATUS "warning: No 'file' command, skipping execute_process...")
    endif()
  endif()
endfunction()


function(gp_item_default_embedded_path item default_embedded_path_var)

  # On Windows and Linux, "embed" prerequisites in the same directory
  # as the executable by default:
  #
  set(path "@executable_path")

  # On the Mac, relative to the executable depending on the type
  # of the thing we are embedding:
  #
  if(APPLE)
    #
    # The assumption here is that all executables in the bundle will be
    # in same-level-directories inside the bundle. The parent directory
    # of an executable inside the bundle should be MacOS or a sibling of
    # MacOS and all embedded paths returned from here will begin with
    # "@executable_path/../" and will work from all executables in all
    # such same-level-directories inside the bundle.
    #

    # By default, embed things right next to the main bundle executable:
    #
    set(path "@executable_path/../../Contents/MacOS")

    # Embed frameworks and .dylibs in the embedded "Frameworks" directory
    # (sibling of MacOS):
    #
    if(item MATCHES "[^/]+\\.framework/" OR item MATCHES "\\.dylib$")
      set(path "@executable_path/../Frameworks")
    endif()
  endif()

  # Provide a hook so that projects can override the default embedded location
  # of any given library by whatever logic they choose:
  #
  if(COMMAND gp_item_default_embedded_path_override)
    gp_item_default_embedded_path_override("${item}" path)
  endif()

  set(${default_embedded_path_var} "${path}" PARENT_SCOPE)
endfunction()


function(gp_resolve_item context item exepath dirs resolved_item_var)
  set(resolved 0)
  set(resolved_item "${item}")
  if(ARGC GREATER 5)
    set(rpaths "${ARGV5}")
  else()
    set(rpaths "")
  endif()

  # Is it already resolved?
  #
  if(IS_ABSOLUTE "${resolved_item}" AND EXISTS "${resolved_item}")
    set(resolved 1)
  endif()

  if(NOT resolved)
    if(item MATCHES "^@executable_path")
      #
      # @executable_path references are assumed relative to exepath
      #
      string(REPLACE "@executable_path" "${exepath}" ri "${item}")
      get_filename_component(ri "${ri}" ABSOLUTE)

      if(EXISTS "${ri}")
        #message(STATUS "info: embedded item exists (${ri})")
        set(resolved 1)
        set(resolved_item "${ri}")
      else()
        message(STATUS "warning: embedded item does not exist '${ri}'")
      endif()
    endif()
  endif()

  if(NOT resolved)
    if(item MATCHES "^@loader_path")
      #
      # @loader_path references are assumed relative to the
      # PATH of the given "context" (presumably another library)
      #
      get_filename_component(contextpath "${context}" PATH)
      string(REPLACE "@loader_path" "${contextpath}" ri "${item}")
      get_filename_component(ri "${ri}" ABSOLUTE)

      if(EXISTS "${ri}")
        #message(STATUS "info: embedded item exists (${ri})")
        set(resolved 1)
        set(resolved_item "${ri}")
      else()
        message(STATUS "warning: embedded item does not exist '${ri}'")
      endif()
    endif()
  endif()

  if(NOT resolved)
    if(item MATCHES "^@rpath")
      #
      # @rpath references are relative to the paths built into the binaries with -rpath
      # We handle this case like we do for other Unixes
      #
      string(REPLACE "@rpath/" "" norpath_item "${item}")

      set(ri "ri-NOTFOUND")
      find_file(ri "${norpath_item}" ${exepath} ${dirs} ${rpaths} NO_DEFAULT_PATH)
      if(ri)
        #message(STATUS "info: 'find_file' in exepath/dirs/rpaths (${ri})")
        set(resolved 1)
        set(resolved_item "${ri}")
        set(ri "ri-NOTFOUND")
      endif()

    endif()
  endif()

  if(NOT resolved)
    set(ri "ri-NOTFOUND")
    find_file(ri "${item}" ${exepath} ${dirs} NO_DEFAULT_PATH)
    find_file(ri "${item}" ${exepath} ${dirs} /usr/lib)

    get_filename_component(basename_item "${item}" NAME)
    find_file(ri "${basename_item}" PATHS ${exepath} ${dirs} NO_DEFAULT_PATH)
    find_file(ri "${basename_item}" PATHS /usr/lib)

    if(ri)
      #message(STATUS "info: 'find_file' in exepath/dirs (${ri})")
      set(resolved 1)
      set(resolved_item "${ri}")
      set(ri "ri-NOTFOUND")
    endif()
  endif()

  if(NOT resolved)
    if(item MATCHES "[^/]+\\.framework/")
      set(fw "fw-NOTFOUND")
      find_file(fw "${item}"
        "~/Library/Frameworks"
        "/Library/Frameworks"
        "/System/Library/Frameworks"
      )
      if(fw)
        #message(STATUS "info: 'find_file' found framework (${fw})")
        set(resolved 1)
        set(resolved_item "${fw}")
        set(fw "fw-NOTFOUND")
      endif()
    endif()
  endif()

  # Using find_program on Windows will find dll files that are in the PATH.
  # (Converting simple file names into full path names if found.)
  #
  if(WIN32 AND NOT UNIX)
  if(NOT resolved)
    set(ri "ri-NOTFOUND")
    find_program(ri "${item}" PATHS ${exepath} ${dirs} NO_DEFAULT_PATH)
    find_program(ri "${item}" PATHS ${exepath} ${dirs})
    if(ri)
      #message(STATUS "info: 'find_program' in exepath/dirs (${ri})")
      set(resolved 1)
      set(resolved_item "${ri}")
      set(ri "ri-NOTFOUND")
    endif()
  endif()
  endif()

  # Provide a hook so that projects can override item resolution
  # by whatever logic they choose:
  #
  if(COMMAND gp_resolve_item_override)
    gp_resolve_item_override("${context}" "${item}" "${exepath}" "${dirs}" resolved_item resolved)
  endif()

  if(NOT resolved)
    message(STATUS "
warning: cannot resolve item '${item}'

  possible problems:
    need more directories?
    need to use InstallRequiredSystemLibraries?
    run in install tree instead of build tree?
")
#    message(STATUS "
#******************************************************************************
#warning: cannot resolve item '${item}'
#
#  possible problems:
#    need more directories?
#    need to use InstallRequiredSystemLibraries?
#    run in install tree instead of build tree?
#
#    context='${context}'
#    item='${item}'
#    exepath='${exepath}'
#    dirs='${dirs}'
#    resolved_item_var='${resolved_item_var}'
#******************************************************************************
#")
  endif()

  set(${resolved_item_var} "${resolved_item}" PARENT_SCOPE)
endfunction()


function(gp_resolved_file_type original_file file exepath dirs type_var)
  if(ARGC GREATER 5)
    set(rpaths "${ARGV5}")
  else()
    set(rpaths "")
  endif()
  #message(STATUS "**")

  if(NOT IS_ABSOLUTE "${original_file}")
    message(STATUS "warning: gp_resolved_file_type expects absolute full path for first arg original_file")
  endif()
  if(IS_ABSOLUTE "${original_file}")
    get_filename_component(original_file "${original_file}" ABSOLUTE) # canonicalize path
  endif()

  set(is_embedded 0)
  set(is_local 0)
  set(is_system 0)

  set(resolved_file "${file}")

  if("${file}" MATCHES "^@(executable|loader)_path")
    set(is_embedded 1)
  endif()

  if(NOT is_embedded)
    if(NOT IS_ABSOLUTE "${file}")
      gp_resolve_item("${original_file}" "${file}" "${exepath}" "${dirs}" resolved_file "${rpaths}")
    endif()
    if(IS_ABSOLUTE "${resolved_file}")
      get_filename_component(resolved_file "${resolved_file}" ABSOLUTE) # canonicalize path
    endif()

    string(TOLOWER "${original_file}" original_lower)
    string(TOLOWER "${resolved_file}" lower)

    if(UNIX)
      if(resolved_file MATCHES "^/*(/lib/|/lib32/|/libx32/|/lib64/|/usr/lib/|/usr/lib32/|/usr/libx32/|/usr/lib64/|/usr/X11R6/|/usr/bin/)" OR
         resolved_file MATCHES "/cce/.*/lib/lib[^/]+\.so\\.[0-9][^/]*$")
        set(is_system 1)
      endif()
    endif()

    if(APPLE)
      if(resolved_file MATCHES "^(/System/Library/|/usr/lib/)")
        set(is_system 1)
      endif()
    endif()

    if(WIN32)
      string(TOLOWER "$ENV{SystemRoot}" sysroot)
      file(TO_CMAKE_PATH "${sysroot}" sysroot)

      string(TOLOWER "$ENV{windir}" windir)
      file(TO_CMAKE_PATH "${windir}" windir)

      if(lower MATCHES "^(${sysroot}/sys(tem|wow)|${windir}/sys(tem|wow)|(.*/)*(msvc|api-ms-win-|vcruntime)[^/]+dll)")
        set(is_system 1)
      endif()

      if(UNIX)
        # if cygwin, we can get the properly formed windows paths from cygpath
        find_program(CYGPATH_EXECUTABLE cygpath)

        if(CYGPATH_EXECUTABLE)
          execute_process(COMMAND ${CYGPATH_EXECUTABLE} -W
                          RESULT_VARIABLE env_rv
                          OUTPUT_VARIABLE env_windir
                          ERROR_VARIABLE env_ev
                          OUTPUT_STRIP_TRAILING_WHITESPACE)
          if(NOT env_rv STREQUAL "0")
            message(FATAL_ERROR "${CYGPATH_EXECUTABLE} -W failed: ${env_rv}\n${env_ev}")
          endif()
          execute_process(COMMAND ${CYGPATH_EXECUTABLE} -S
                          RESULT_VARIABLE env_rv
                          OUTPUT_VARIABLE env_sysdir
                          ERROR_VARIABLE env_ev
                          OUTPUT_STRIP_TRAILING_WHITESPACE)
          if(NOT env_rv STREQUAL "0")
            message(FATAL_ERROR "${CYGPATH_EXECUTABLE} -S failed: ${env_rv}\n${env_ev}")
          endif()
          string(TOLOWER "${env_windir}" windir)
          string(TOLOWER "${env_sysdir}" sysroot)

          if(lower MATCHES "^(${sysroot}/sys(tem|wow)|${windir}/sys(tem|wow)|(.*/)*(msvc|api-ms-win-|vcruntime)[^/]+dll)")
            set(is_system 1)
          endif()
        endif()
      endif()
    endif()

    if(NOT is_system)
      get_filename_component(original_path "${original_lower}" PATH)
      get_filename_component(path "${lower}" PATH)
      if(original_path STREQUAL path)
        set(is_local 1)
      else()
        string(LENGTH "${original_path}/" original_length)
        string(LENGTH "${lower}" path_length)
        if(${path_length} GREATER ${original_length})
          string(SUBSTRING "${lower}" 0 ${original_length} path)
          if("${original_path}/" STREQUAL path)
            set(is_embedded 1)
          endif()
        endif()
      endif()
    endif()
  endif()

  # Return type string based on computed booleans:
  #
  set(type "other")

  if(is_system)
    set(type "system")
  elseif(is_embedded)
    set(type "embedded")
  elseif(is_local)
    set(type "local")
  endif()

  #message(STATUS "gp_resolved_file_type: '${file}' '${resolved_file}'")
  #message(STATUS "                type: '${type}'")

  if(NOT is_embedded)
    if(NOT IS_ABSOLUTE "${resolved_file}")
      if(lower MATCHES "^(msvc|api-ms-win-|vcruntime)[^/]+dll" AND is_system)
        message(STATUS "info: non-absolute msvc file '${file}' returning type '${type}'")
      else()
        message(STATUS "warning: gp_resolved_file_type non-absolute file '${file}' returning type '${type}' -- possibly incorrect")
      endif()
    endif()
  endif()

  # Provide a hook so that projects can override the decision on whether a
  # library belongs to the system or not by whatever logic they choose:
  #
  if(COMMAND gp_resolved_file_type_override)
    gp_resolved_file_type_override("${resolved_file}" type)
  endif()

  set(${type_var} "${type}" PARENT_SCOPE)

  #message(STATUS "**")
endfunction()


function(gp_file_type original_file file type_var)
  if(NOT IS_ABSOLUTE "${original_file}")
    message(STATUS "warning: gp_file_type expects absolute full path for first arg original_file")
  endif()

  get_filename_component(exepath "${original_file}" PATH)

  set(type "")
  gp_resolved_file_type("${original_file}" "${file}" "${exepath}" "" type)

  set(${type_var} "${type}" PARENT_SCOPE)
endfunction()


function(get_prerequisites target prerequisites_var exclude_system recurse exepath dirs)
  set(verbose 0)
  set(eol_char "E")
  if(ARGC GREATER 6)
    set(rpaths "${ARGV6}")
  else()
    set(rpaths "")
  endif()

  if(GET_PREREQUISITES_VERBOSE)
    set(verbose 1)
  endif()

  if(NOT IS_ABSOLUTE "${target}")
    message("warning: target '${target}' is not absolute...")
  endif()

  if(NOT EXISTS "${target}")
    message("warning: target '${target}' does not exist...")
    return()
  endif()

  # Check for a script by extension (.bat,.sh,...) or if the file starts with "#!" (shebang)
  file(READ ${target} file_contents LIMIT 5)
  if(target MATCHES "\\.(bat|c?sh|bash|ksh|cmd)$" OR file_contents MATCHES "^#!")
    message(STATUS "GetPrerequisites(${target}) : ignoring script file")
    # Clear var
    set(${prerequisites_var} "" PARENT_SCOPE)
    return()
  endif()

  set(gp_cmd_paths ${gp_cmd_paths}
    "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\VisualStudio\\14.0;InstallDir]/../../VC/bin"
    "$ENV{VS140COMNTOOLS}/../../VC/bin"
    "C:/Program Files (x86)/Microsoft Visual Studio 14.0/VC/bin"
    "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\VisualStudio\\12.0;InstallDir]/../../VC/bin"
    "$ENV{VS120COMNTOOLS}/../../VC/bin"
    "C:/Program Files (x86)/Microsoft Visual Studio 12.0/VC/bin"
    "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\VisualStudio\\11.0;InstallDir]/../../VC/bin"
    "$ENV{VS110COMNTOOLS}/../../VC/bin"
    "C:/Program Files (x86)/Microsoft Visual Studio 11.0/VC/bin"
    "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\VisualStudio\\10.0;InstallDir]/../../VC/bin"
    "$ENV{VS100COMNTOOLS}/../../VC/bin"
    "C:/Program Files (x86)/Microsoft Visual Studio 10.0/VC/bin"
    "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\VisualStudio\\9.0;InstallDir]/../../VC/bin"
    "$ENV{VS90COMNTOOLS}/../../VC/bin"
    "C:/Program Files/Microsoft Visual Studio 9.0/VC/bin"
    "C:/Program Files (x86)/Microsoft Visual Studio 9.0/VC/bin"
    "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\VisualStudio\\8.0;InstallDir]/../../VC/bin"
    "$ENV{VS80COMNTOOLS}/../../VC/bin"
    "C:/Program Files/Microsoft Visual Studio 8/VC/BIN"
    "C:/Program Files (x86)/Microsoft Visual Studio 8/VC/BIN"
    "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\VisualStudio\\7.1;InstallDir]/../../VC7/bin"
    "$ENV{VS71COMNTOOLS}/../../VC7/bin"
    "C:/Program Files/Microsoft Visual Studio .NET 2003/VC7/BIN"
    "C:/Program Files (x86)/Microsoft Visual Studio .NET 2003/VC7/BIN"
    )

  # <setup-gp_tool-vars>
  #
  # Try to choose the right tool by default. Caller can set gp_tool prior to
  # calling this function to force using a different tool.
  #
  if(NOT gp_tool)
    set(gp_tool "ldd")

    if(APPLE)
      set(gp_tool "otool")
    endif()

    if(WIN32 AND NOT UNIX) # This is how to check for cygwin, har!
      find_program(gp_dumpbin "dumpbin" PATHS ${gp_cmd_paths})
      if(gp_dumpbin)
        set(gp_tool "dumpbin")
      elseif(CMAKE_OBJDUMP) # Try harder. Maybe we're on MinGW
        set(gp_tool "${CMAKE_OBJDUMP}")
      else()
        set(gp_tool "objdump")
      endif()
    endif()
  endif()

  find_program(gp_cmd ${gp_tool} PATHS ${gp_cmd_paths})

  if(NOT gp_cmd)
    message(STATUS "warning: could not find '${gp_tool}' - cannot analyze prerequisites...")
    return()
  endif()

  set(gp_cmd_maybe_filter)      # optional command to pre-filter gp_tool results

  if(gp_tool MATCHES "ldd$")
    set(gp_cmd_args "")
    set(gp_regex "^[\t ]*[^\t ]+ =>[\t ]+([^\t\(]+)( \(.+\))?${eol_char}$")
    set(gp_regex_error "not found${eol_char}$")
    set(gp_regex_fallback "^[\t ]*([^\t ]+) => ([^\t ]+).*${eol_char}$")
    set(gp_regex_cmp_count 1)
  elseif(gp_tool MATCHES "otool$")
    set(gp_cmd_args "-L")
    set(gp_regex "^\t([^\t]+) \\(compatibility version ([0-9]+.[0-9]+.[0-9]+), current version ([0-9]+.[0-9]+.[0-9]+)(, weak)?\\)${eol_char}$")
    set(gp_regex_error "")
    set(gp_regex_fallback "")
    set(gp_regex_cmp_count 3)
  elseif(gp_tool MATCHES "dumpbin$")
    set(gp_cmd_args "/dependents")
    set(gp_regex "^    ([^ ].*[Dd][Ll][Ll])${eol_char}$")
    set(gp_regex_error "")
    set(gp_regex_fallback "")
    set(gp_regex_cmp_count 1)
  elseif(gp_tool MATCHES "objdump(\\.exe)?$")
    set(gp_cmd_args "-p")
    set(gp_regex "^[\t ]*DLL Name: (.*\\.[Dd][Ll][Ll])${eol_char}$")
    set(gp_regex_error "")
    set(gp_regex_fallback "")
    set(gp_regex_cmp_count 1)
    # objdump generates copious output so we create a grep filter to pre-filter results
    if(WIN32)
      find_program(gp_grep_cmd findstr)
      set(gp_grep_cmd_arg "")
    else()
      find_program(gp_grep_cmd grep)
      set(gp_grep_cmd_arg "-a")
    endif()
    if(gp_grep_cmd)
      set(gp_cmd_maybe_filter COMMAND ${gp_grep_cmd} "${gp_grep_cmd_arg}" "^[[:blank:]]*DLL Name: ")
    endif()
  else()
    message(STATUS "warning: gp_tool='${gp_tool}' is an unknown tool...")
    message(STATUS "CMake function get_prerequisites needs more code to handle '${gp_tool}'")
    message(STATUS "Valid gp_tool values are dumpbin, ldd, objdump and otool.")
    return()
  endif()


  if(gp_tool MATCHES "dumpbin$")
    # When running dumpbin, it also needs the "Common7/IDE" directory in the
    # PATH. It will already be in the PATH if being run from a Visual Studio
    # command prompt. Add it to the PATH here in case we are running from a
    # different command prompt.
    #
    get_filename_component(gp_cmd_dir "${gp_cmd}" PATH)
    get_filename_component(gp_cmd_dlls_dir "${gp_cmd_dir}/../../Common7/IDE" ABSOLUTE)
    # Use cmake paths as a user may have a PATH element ending with a backslash.
    # This will escape the list delimiter and create havoc!
    if(EXISTS "${gp_cmd_dlls_dir}")
      # only add to the path if it is not already in the path
      set(gp_found_cmd_dlls_dir 0)
      file(TO_CMAKE_PATH "$ENV{PATH}" env_path)
      foreach(gp_env_path_element ${env_path})
        if(gp_env_path_element STREQUAL gp_cmd_dlls_dir)
          set(gp_found_cmd_dlls_dir 1)
        endif()
      endforeach()

      if(NOT gp_found_cmd_dlls_dir)
        file(TO_NATIVE_PATH "${gp_cmd_dlls_dir}" gp_cmd_dlls_dir)
        set(ENV{PATH} "$ENV{PATH};${gp_cmd_dlls_dir}")
      endif()
    endif()
  endif()
  #
  # </setup-gp_tool-vars>

  if(gp_tool MATCHES "ldd$")
    set(old_ld_env "$ENV{LD_LIBRARY_PATH}")
    set(new_ld_env "${exepath}")
    foreach(dir ${dirs})
      string(APPEND new_ld_env ":${dir}")
    endforeach()
    set(ENV{LD_LIBRARY_PATH} "${new_ld_env}:$ENV{LD_LIBRARY_PATH}")
  endif()


  # Track new prerequisites at each new level of recursion. Start with an
  # empty list at each level:
  #
  set(unseen_prereqs)

  # Run gp_cmd on the target:
  #
  execute_process(
    COMMAND ${gp_cmd} ${gp_cmd_args} ${target}
    ${gp_cmd_maybe_filter}
    RESULT_VARIABLE gp_rv
    OUTPUT_VARIABLE gp_cmd_ov
    ERROR_VARIABLE gp_ev
    )

  if(gp_tool MATCHES "dumpbin$")
    # Exclude delay load dependencies under windows (they are listed in dumpbin output after the message below)
    string(FIND "${gp_cmd_ov}" "Image has the following delay load dependencies" gp_delayload_pos)
    if (${gp_delayload_pos} GREATER -1)
      string(SUBSTRING "${gp_cmd_ov}" 0 ${gp_delayload_pos} gp_cmd_ov_no_delayload_deps)
      string(SUBSTRING "${gp_cmd_ov}" ${gp_delayload_pos} -1 gp_cmd_ov_delayload_deps)
      if (verbose)
        message(STATUS "GetPrerequisites(${target}) : ignoring the following delay load dependencies :\n ${gp_cmd_ov_delayload_deps}")
      endif()
      set(gp_cmd_ov ${gp_cmd_ov_no_delayload_deps})
    endif()
  endif()

  if(NOT gp_rv STREQUAL "0")
    if(gp_tool MATCHES "dumpbin$")
      # dumpbin error messages seem to go to stdout
      message(FATAL_ERROR "${gp_cmd} failed: ${gp_rv}\n${gp_ev}\n${gp_cmd_ov}")
    else()
      message(FATAL_ERROR "${gp_cmd} failed: ${gp_rv}\n${gp_ev}")
    endif()
  endif()

  if(gp_tool MATCHES "ldd$")
    set(ENV{LD_LIBRARY_PATH} "${old_ld_env}")
  endif()

  if(verbose)
    message(STATUS "<RawOutput cmd='${gp_cmd} ${gp_cmd_args} ${target}'>")
    message(STATUS "gp_cmd_ov='${gp_cmd_ov}'")
    message(STATUS "</RawOutput>")
  endif()

  get_filename_component(target_dir "${target}" PATH)

  # Convert to a list of lines:
  #
  string(REPLACE ";" "\\;" candidates "${gp_cmd_ov}")
  string(REPLACE "\n" "${eol_char};" candidates "${candidates}")

  # check for install id and remove it from list, since otool -L can include a
  # reference to itself
  set(gp_install_id)
  if(gp_tool MATCHES "otool$")
    execute_process(
      COMMAND ${gp_cmd} -D ${target}
      RESULT_VARIABLE otool_rv
      OUTPUT_VARIABLE gp_install_id_ov
      ERROR_VARIABLE otool_ev
      )
    if(NOT otool_rv STREQUAL "0")
      message(FATAL_ERROR "otool -D failed: ${otool_rv}\n${otool_ev}")
    endif()
    # second line is install name
    string(REGEX REPLACE ".*:\n" "" gp_install_id "${gp_install_id_ov}")
    if(gp_install_id)
      # trim
      string(REGEX MATCH "[^\n ].*[^\n ]" gp_install_id "${gp_install_id}")
      #message("INSTALL ID is \"${gp_install_id}\"")
    endif()
  endif()

  # Analyze each line for file names that match the regular expression:
  #
  foreach(candidate ${candidates})
  if("${candidate}" MATCHES "${gp_regex}")

    # Extract information from each candidate:
    if(gp_regex_error AND "${candidate}" MATCHES "${gp_regex_error}")
      string(REGEX REPLACE "${gp_regex_fallback}" "\\1" raw_item "${candidate}")
    else()
      string(REGEX REPLACE "${gp_regex}" "\\1" raw_item "${candidate}")
    endif()

    if(gp_regex_cmp_count GREATER 1)
      string(REGEX REPLACE "${gp_regex}" "\\2" raw_compat_version "${candidate}")
      string(REGEX REPLACE "^([0-9]+)\\.([0-9]+)\\.([0-9]+)$" "\\1" compat_major_version "${raw_compat_version}")
      string(REGEX REPLACE "^([0-9]+)\\.([0-9]+)\\.([0-9]+)$" "\\2" compat_minor_version "${raw_compat_version}")
      string(REGEX REPLACE "^([0-9]+)\\.([0-9]+)\\.([0-9]+)$" "\\3" compat_patch_version "${raw_compat_version}")
    endif()

    if(gp_regex_cmp_count GREATER 2)
      string(REGEX REPLACE "${gp_regex}" "\\3" raw_current_version "${candidate}")
      string(REGEX REPLACE "^([0-9]+)\\.([0-9]+)\\.([0-9]+)$" "\\1" current_major_version "${raw_current_version}")
      string(REGEX REPLACE "^([0-9]+)\\.([0-9]+)\\.([0-9]+)$" "\\2" current_minor_version "${raw_current_version}")
      string(REGEX REPLACE "^([0-9]+)\\.([0-9]+)\\.([0-9]+)$" "\\3" current_patch_version "${raw_current_version}")
    endif()

    # Use the raw_item as the list entries returned by this function. Use the
    # gp_resolve_item function to resolve it to an actual full path file if
    # necessary.
    #
    set(item "${raw_item}")

    # Add each item unless it is excluded:
    #
    set(add_item 1)

    if(item STREQUAL gp_install_id)
      set(add_item 0)
    endif()

    if(add_item AND ${exclude_system})
      set(type "")
      gp_resolved_file_type("${target}" "${item}" "${exepath}" "${dirs}" type "${rpaths}")

      if(type STREQUAL "system")
        set(add_item 0)
      endif()
    endif()

    if(add_item)
      list(LENGTH ${prerequisites_var} list_length_before_append)
      gp_append_unique(${prerequisites_var} "${item}")
      list(LENGTH ${prerequisites_var} list_length_after_append)

      if(${recurse})
        # If item was really added, this is the first time we have seen it.
        # Add it to unseen_prereqs so that we can recursively add *its*
        # prerequisites...
        #
        # But first: resolve its name to an absolute full path name such
        # that the analysis tools can simply accept it as input.
        #
        if(NOT list_length_before_append EQUAL list_length_after_append)
          gp_resolve_item("${target}" "${item}" "${exepath}" "${dirs}" resolved_item "${rpaths}")
          if(EXISTS "${resolved_item}")
            # Recurse only if we could resolve the item.
            # Otherwise the prerequisites_var list will be cleared
            set(unseen_prereqs ${unseen_prereqs} "${resolved_item}")
          endif()
        endif()
      endif()
    endif()
  else()
    if(verbose)
      message(STATUS "ignoring non-matching line: '${candidate}'")
    endif()
  endif()
  endforeach()

  list(LENGTH ${prerequisites_var} prerequisites_var_length)
  if(prerequisites_var_length GREATER 0)
    list(SORT ${prerequisites_var})
  endif()
  if(${recurse})
    set(more_inputs ${unseen_prereqs})
    foreach(input ${more_inputs})
      get_prerequisites("${input}" ${prerequisites_var} ${exclude_system} ${recurse} "${exepath}" "${dirs}" "${rpaths}")
    endforeach()
  endif()

  set(${prerequisites_var} ${${prerequisites_var}} PARENT_SCOPE)
endfunction()


function(list_prerequisites target)
  if(ARGC GREATER 1 AND NOT "${ARGV1}" STREQUAL "")
    set(all "${ARGV1}")
  else()
    set(all 1)
  endif()

  if(ARGC GREATER 2 AND NOT "${ARGV2}" STREQUAL "")
    set(exclude_system "${ARGV2}")
  else()
    set(exclude_system 0)
  endif()

  if(ARGC GREATER 3 AND NOT "${ARGV3}" STREQUAL "")
    set(verbose "${ARGV3}")
  else()
    set(verbose 0)
  endif()

  set(count 0)
  set(count_str "")
  set(print_count "${verbose}")
  set(print_prerequisite_type "${verbose}")
  set(print_target "${verbose}")
  set(type_str "")

  get_filename_component(exepath "${target}" PATH)

  set(prereqs "")
  get_prerequisites("${target}" prereqs ${exclude_system} ${all} "${exepath}" "")

  if(print_target)
    message(STATUS "File '${target}' depends on:")
  endif()

  foreach(d ${prereqs})
    math(EXPR count "${count} + 1")

    if(print_count)
      set(count_str "${count}. ")
    endif()

    if(print_prerequisite_type)
      gp_file_type("${target}" "${d}" type)
      set(type_str " (${type})")
    endif()

    message(STATUS "${count_str}${d}${type_str}")
  endforeach()
endfunction()


function(list_prerequisites_by_glob glob_arg glob_exp)
  message(STATUS "=============================================================================")
  message(STATUS "List prerequisites of executables matching ${glob_arg} '${glob_exp}'")
  message(STATUS "")
  file(${glob_arg} file_list ${glob_exp})
  foreach(f ${file_list})
    is_file_executable("${f}" is_f_executable)
    if(is_f_executable)
      message(STATUS "=============================================================================")
      list_prerequisites("${f}" ${ARGN})
      message(STATUS "")
    endif()
  endforeach()
endfunction()
