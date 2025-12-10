# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
GNUInstallDirs
--------------

This module defines the installation directory variables according to the
`GNU Coding Standards`_ and provides a command to compute
installation-related absolute paths.

Load this module in a CMake project with:

.. code-block:: cmake

  include(GNUInstallDirs)

.. _`GNU Coding Standards`: https://www.gnu.org/prep/standards/html_node/Directory-Variables.html

Result Variables
^^^^^^^^^^^^^^^^

Inclusion of this module defines the following variables:

``CMAKE_INSTALL_<dir>``
  Destination for files of a given type.  This value may be passed to
  the ``DESTINATION`` options of  :command:`install` commands for the
  corresponding file type.  It should be a path relative to the installation
  prefix so that it can be converted to an absolute path in a relocatable way.
  However, there are some `special cases`_ as documented below.

  While absolute paths are allowed, they are not recommended as they
  do not work with the ``cmake --install`` command's
  :option:`--prefix <cmake--install --prefix>` option, or with the
  :manual:`cpack <cpack(1)>` installer generators. In particular, there is no
  need to make paths absolute by prepending :variable:`CMAKE_INSTALL_PREFIX`;
  this prefix is used by default if the DESTINATION is a relative path.

``CMAKE_INSTALL_FULL_<dir>``
  The absolute path generated from the corresponding ``CMAKE_INSTALL_<dir>``
  value.  If the value is not already an absolute path, an absolute path
  is constructed typically by prepending the value of the
  :variable:`CMAKE_INSTALL_PREFIX` variable, except in `special cases`_
  as documented below.

  These variables shouldn't be used in :command:`install` commands
  as they do not work with the ``cmake --install`` command's
  :option:`--prefix <cmake--install --prefix>` option, or with the
  :manual:`cpack <cpack(1)>` installer generators.

where ``<dir>`` is one of:

``BINDIR``
  user executables (``bin``)
``SBINDIR``
  system admin executables (``sbin``)
``LIBEXECDIR``
  program executables (``libexec``)
``SYSCONFDIR``
  read-only single-machine data (``etc``)

  .. versionchanged:: 4.1
    If the :variable:`CMAKE_INSTALL_PREFIX` falls into the
    `special cases`_,  the default paths for are the absolute
    path variants as described there. See policy :policy:`CMP0192`.
``SHAREDSTATEDIR``
  modifiable architecture-independent data (``com``)
``LOCALSTATEDIR``
  modifiable single-machine data (``var``)

  .. versionchanged:: 4.1
    If the :variable:`CMAKE_INSTALL_PREFIX` falls into the
    `special cases`_,  the default paths for are the absolute
    path variants as described there. See policy :policy:`CMP0192`.
``RUNSTATEDIR``
  run-time variable data (``LOCALSTATEDIR/run``)

  .. versionadded:: 3.9

  .. versionchanged:: 4.1
    If the :variable:`CMAKE_INSTALL_PREFIX` falls into the
    `special cases`_,  the default paths for are the absolute
    path variants as described there. See policy :policy:`CMP0192`.
``LIBDIR``
  object code libraries (``lib`` or ``lib64``)

  On Debian, this may be ``lib/<multiarch-tuple>`` when
  :variable:`CMAKE_INSTALL_PREFIX` is ``/usr``.
``INCLUDEDIR``
  C header files (``include``)
``OLDINCLUDEDIR``
  C header files for non-gcc (``/usr/include``)
``DATAROOTDIR``
  read-only architecture-independent data root (``share``)
``DATADIR``
  read-only architecture-independent data (``DATAROOTDIR``)
``INFODIR``
  info documentation (``DATAROOTDIR/info``)
``LOCALEDIR``
  locale-dependent data (``DATAROOTDIR/locale``)
``MANDIR``
  man documentation (``DATAROOTDIR/man``)
``DOCDIR``
  documentation root (``DATAROOTDIR/doc/PROJECT_NAME``)

If the includer does not define a value the above-shown default will be
used and the value will appear in the cache for editing by the user.

If a default value for the ``CMAKE_INSTALL_<dir>`` is used and the
:variable:`CMAKE_INSTALL_PREFIX` is changed, the new default value will
be used calculated on the new :variable:`CMAKE_INSTALL_PREFIX` value.
Using :option:`--prefix <cmake--install --prefix>` in ``cmake --install``
will not alter these values.

.. _`GNUInstallDirs special cases`:

Special Cases
^^^^^^^^^^^^^

.. versionadded:: 3.4

The following values of :variable:`CMAKE_INSTALL_PREFIX` are special:

``/``

  For ``<dir>`` other than the ``SYSCONFDIR``, ``LOCALSTATEDIR`` and
  ``RUNSTATEDIR``, the value of ``CMAKE_INSTALL_<dir>`` is prefixed
  with ``usr/`` if it is not user-specified as an absolute path.
  For example, the ``INCLUDEDIR`` value ``include`` becomes ``usr/include``.
  This is required by the `GNU Coding Standards`_, which state:

    When building the complete GNU system, the prefix will be empty
    and ``/usr`` will be a symbolic link to ``/``.

  .. versionchanged:: 4.1
    The ``CMAKE_INSTALL_<dir>`` variables are cached with the ``usr/`` prefix.
    See policy :policy:`CMP0193`.

``/usr``

  For ``<dir>`` equal to ``SYSCONFDIR``, ``LOCALSTATEDIR`` or
  ``RUNSTATEDIR``, the ``CMAKE_INSTALL_FULL_<dir>`` is computed by
  prepending just ``/`` to the value of ``CMAKE_INSTALL_<dir>``
  if it is not already an absolute path.
  For example, the ``SYSCONFDIR`` value ``etc`` becomes ``/etc``.
  This is required by the `GNU Coding Standards`_.

  .. versionchanged:: 4.1
    The default values of ``CMAKE_INSTALL_<dir>`` for ``<dir>`` equal
    to ``SYSCONFDIR``, ``LOCALSTATEDIR`` and ``RUNSTATEDIR`` are the
    absolute paths ``/etc``, ``/var`` and ``/var/run`` respectively.
    See policy :policy:`CMP0192`.

``/opt/...``

  For ``<dir>`` equal to ``SYSCONFDIR``, ``LOCALSTATEDIR`` or
  ``RUNSTATEDIR``, the ``CMAKE_INSTALL_FULL_<dir>`` is computed by
  *appending* the prefix to the value of ``CMAKE_INSTALL_<dir>``
  if it is not already an absolute path.
  For example, the ``SYSCONFDIR`` value ``etc`` becomes ``/etc/opt/...``.
  This is defined by the `Filesystem Hierarchy Standard`_.

  This behavior does not apply to paths under ``/opt/homebrew/...``.

  .. versionchanged:: 4.1
    The default values of ``CMAKE_INSTALL_<dir>`` for ``<dir>`` equal
    to ``SYSCONFDIR``, ``LOCALSTATEDIR`` and ``RUNSTATEDIR`` are the
    absolute paths ``/etc/opt/...``, ``/var/opt/...`` and
    ``/var/run/opt/...`` respectively. See policy :policy:`CMP0192`.

.. _`Filesystem Hierarchy Standard`: https://refspecs.linuxfoundation.org/FHS_3.0/fhs/index.html

Commands
^^^^^^^^

This module provides the following command:

.. command:: GNUInstallDirs_get_absolute_install_dir

  .. versionadded:: 3.7

  Computes an absolute installation path from a given relative path:

  .. code-block:: cmake

    GNUInstallDirs_get_absolute_install_dir(<result-var> <input-var> <dir>)

  This command takes the value from the variable ``<input-var>`` and
  computes its absolute path according to GNU standard installation
  directories.  If the input path is relative, it is prepended with
  :variable:`CMAKE_INSTALL_PREFIX` and may be adjusted for the
  `special cases`_ described above.

  The arguments are:

  ``<result-var>``
    Name of the variable in which to store the computed absolute path.

  ``<input-var>``
    Name of the variable containing the path that will be used to compute
    its associated absolute installation path.

    .. versionchanged:: 4.1
      This variable is no longer altered.  See policy :policy:`CMP0193`.
      In previous CMake versions, this command modified the ``<input-var>``
      variable value based on the `special cases`_.

  ``<dir>``
    .. versionadded:: 3.20

    The directory type name, e.g., ``SYSCONFDIR``, ``LOCALSTATEDIR``,
    ``RUNSTATEDIR``, etc.  This argument determines whether `special cases`_
    apply when computing the absolute path.

    .. versionchanged:: 3.20

      Before the ``<dir>`` argument was introduced, the directory type
      could be specified by setting the ``dir`` variable prior to calling
      this command.  As of CMake 3.20, if the ``<dir>`` argument is provided
      explicitly, the ``dir`` variable is ignored.

  While this command is used internally by this module to compute the
  ``CMAKE_INSTALL_FULL_<dir>`` variables, it is also exposed publicly for
  users to create additional custom installation path variables and compute
  absolute paths where necessary, using the same logic.

See Also
^^^^^^^^

* The :command:`install` command.
#]=======================================================================]

cmake_policy(SET CMP0140 NEW)

# Note that even though we read the policy every time this file is `include`
# only the first occurrence has effect because it is used for the initialization
# of cache variables
cmake_policy(GET CMP0192 _GNUInstallDirs_CMP0192)

# Convert a cache variable to PATH type

function(_GNUInstallDirs_cache_convert_to_path var description)
  get_property(cache_type CACHE ${var} PROPERTY TYPE)
  if(cache_type STREQUAL "UNINITIALIZED")
    file(TO_CMAKE_PATH "${${var}}" cmakepath)
    set_property(CACHE ${var} PROPERTY TYPE PATH)
    set_property(CACHE ${var} PROPERTY VALUE "${cmakepath}")
    set_property(CACHE ${var} PROPERTY HELPSTRING "${description}")
  endif()
endfunction()

# Create a cache variable with default for a path.
function(_GNUInstallDirs_cache_path var description)
  set(cmake_install_var "CMAKE_INSTALL_${var}")
  set(default "${_GNUInstallDirs_${var}_DEFAULT}")
  # Check if we have a special way to calculate the defaults
  if(COMMAND _GNUInstallDirs_${var}_get_default)
    # Check if the current CMAKE_INSTALL_PREFIX is the same as before
    set(install_prefix_is_same TRUE)
    unset(last_default)
    if(DEFINED _GNUInstallDirs_LAST_CMAKE_INSTALL_PREFIX
        AND NOT _GNUInstallDirs_LAST_CMAKE_INSTALL_PREFIX STREQUAL CMAKE_INSTALL_PREFIX)
      set(install_prefix_is_same FALSE)
      # Recalculate what the last default would have been
      cmake_language(CALL _GNUInstallDirs_${var}_get_default
        last_default
        "${_GNUInstallDirs_LAST_CMAKE_INSTALL_PREFIX}")
    endif()

    if(DEFINED CACHE{${cmake_install_var}} AND install_prefix_is_same)
      # If the cache variable was already set from a previous run and the
      # install prefix has not changed, we don't need to do anything
      return()
    else()
      # Otherwise get the new default
      cmake_language(CALL _GNUInstallDirs_${var}_get_default
        default
        "${CMAKE_INSTALL_PREFIX}")
      # if the current value is the same as the cache value and the same as
      # the old default, reset the value to the new default
      if(${cmake_install_var} STREQUAL "$CACHE{${cmake_install_var}}"
          AND DEFINED last_default
          AND ${cmake_install_var} STREQUAL last_default)
        set(full_description "${description} (${default})")
        set_property(CACHE ${cmake_install_var} PROPERTY TYPE PATH)
        set_property(CACHE ${cmake_install_var} PROPERTY VALUE "${default}")
        set_property(CACHE ${cmake_install_var} PROPERTY HELPSTRING "${full_description}")
      endif()
      # Continue to normal flow
    endif()
  endif()

  # Normal flow
  set(full_description "${description} (${default})")
  if(NOT DEFINED ${cmake_install_var})
    set(${cmake_install_var} "${default}" CACHE PATH "${full_description}")
  endif()
  _GNUInstallDirs_cache_convert_to_path("${cmake_install_var}" "${full_description}")
endfunction()

# Create a cache variable with not default for a path, with a fallback
# when unset; used for entries slaved to other entries such as
# DATAROOTDIR.
function(_GNUInstallDirs_cache_path_fallback var description)
  set(cmake_install_var "CMAKE_INSTALL_${var}")
  set(default "${_GNUInstallDirs_${var}_DEFAULT}")
  # Check if there is a more special way to handle the default
  if(COMMAND _GNUInstallDirs_${var}_get_default)
    cmake_language(CALL _GNUInstallDirs_${var}_get_default
      default
      "${CMAKE_INSTALL_PREFIX}")
  endif()
  if(NOT ${cmake_install_var})
    set(${cmake_install_var} "" CACHE PATH "${description}")
    set(${cmake_install_var} "${default}")
  endif()
  _GNUInstallDirs_cache_convert_to_path("${cmake_install_var}" "${description}")
  return(PROPAGATE ${cmake_install_var})
endfunction()

# Other helpers
# Check what system we are on for LIBDIR formatting
function(_GNUInstallDirs_get_system_type_for_install out_var)
  unset(${out_var})
  # Check if we are building for conda
  if(DEFINED ENV{CONDA_BUILD} AND DEFINED ENV{PREFIX})
    set(conda_prefix "$ENV{PREFIX}")
    cmake_path(ABSOLUTE_PATH conda_prefix NORMALIZE)
    if("${CMAKE_INSTALL_PREFIX}" STREQUAL conda_prefix)
      set(${out_var} "conda")
    endif()
  elseif(DEFINED ENV{CONDA_PREFIX})
    set(conda_prefix "$ENV{CONDA_PREFIX}")
    cmake_path(ABSOLUTE_PATH conda_prefix NORMALIZE)
    if("${CMAKE_INSTALL_PREFIX}" STREQUAL conda_prefix AND
        NOT ("${CMAKE_INSTALL_PREFIX}" MATCHES "^/usr/?$" OR
             "${CMAKE_INSTALL_PREFIX}" MATCHES "^/usr/local/?$"))
      set(${out_var} "conda")
    endif()
  endif()
  # If we didn't detect conda from the previous step, check
  # for the linux variant
  if(NOT ${out_var})
    if (EXISTS "/etc/alpine-release")
      set(${out_var} "alpine")
    elseif (EXISTS "/etc/arch-release")
      set(${out_var} "arch linux")
    elseif (EXISTS "/etc/debian_version")
      set(${out_var} "debian")
    endif()
  endif()
  return(PROPAGATE ${out_var})
endfunction()

# Special handler for `/`, `/usr`, `/opt/...` install prefixes
# Used for SYSCONFDIR, LOCALSTATEDIR and RUNSTATEDIR paths
function(_GNUInstallDirs_special_absolute out_var original_path install_prefix)
  set(${out_var} "${original_path}")

  if(install_prefix MATCHES "^/usr/?$")
    set(${out_var} "/${original_path}")
  elseif(install_prefix MATCHES "^/opt/" AND NOT install_prefix MATCHES "^/opt/homebrew/")
    set(${out_var} "/${original_path}/${install_prefix}")
  endif()

  return(PROPAGATE ${out_var})
endfunction()

# Common handler for defaults that should be in /<dir>
# i.e. SYSCONFDIR and LOCALSTATEDIR
function(__GNUInstallDirs_default_in_root out_var original_path install_prefix)
  if(_GNUInstallDirs_CMP0192 STREQUAL "NEW")
    _GNUInstallDirs_special_absolute(${out_var}
      "${original_path}" "${install_prefix}")
  endif()
  cmake_path(NORMAL_PATH ${out_var})
  return(PROPAGATE ${out_var})
endfunction()

# Common handler for defaults that should be in usr/<dir>
function(__GNUInstallDirs_default_in_usr out_var initial_value install_prefix)
  set(${out_var} "${initial_value}")
  if(install_prefix STREQUAL "/")
    cmake_policy(GET CMP0193 cmp0193
        PARENT_SCOPE # undocumented, do not use outside of CMake
    )
    if(cmp0193 STREQUAL "NEW")
      set(${out_var} "usr/${${out_var}}")
    endif()
  endif()
  return(PROPAGATE ${out_var})
endfunction()

# Installation directories
#

# Set the standard default values before any special handling
set(_GNUInstallDirs_BINDIR_DEFAULT "bin")
set(_GNUInstallDirs_SBINDIR_DEFAULT "sbin")
set(_GNUInstallDirs_LIBEXECDIR_DEFAULT "libexec")
set(_GNUInstallDirs_SYSCONFDIR_DEFAULT "etc")
set(_GNUInstallDirs_SHAREDSTATEDIR_DEFAULT "com")
set(_GNUInstallDirs_LOCALSTATEDIR_DEFAULT "var")
set(_GNUInstallDirs_LIBDIR_DEFAULT "lib")
set(_GNUInstallDirs_INCLUDEDIR_DEFAULT "include")
set(_GNUInstallDirs_OLDINCLUDEDIR_DEFAULT "/usr/include")
set(_GNUInstallDirs_DATAROOTDIR_DEFAULT "share")

# Define the special defaults handling
# Signature
#   _GNUInstallDirs_<Dir>_get_default(out_var install_prefix)
#
# ``out_var``
#   Output variable with the calculated default
#
# ``install_prefix``
#   The CMAKE_INSTALL_PREFIX used to calculate the default

function(_GNUInstallDirs_LIBDIR_get_default out_var install_prefix)
  set(${out_var} "${_GNUInstallDirs_LIBDIR_DEFAULT}")

  # Override this default 'lib' with 'lib64' iff:
  #  - we are on Linux system but NOT cross-compiling
  #  - we are NOT on debian
  #  - we are NOT building for conda
  #  - we are on a 64 bits system
  # reason is: amd64 ABI: https://github.com/hjl-tools/x86-psABI/wiki/X86-psABI
  # For Debian with multiarch, use 'lib/${CMAKE_LIBRARY_ARCHITECTURE}' if
  # CMAKE_LIBRARY_ARCHITECTURE is set (which contains e.g. "i386-linux-gnu"
  # and CMAKE_INSTALL_PREFIX is "/usr"
  # See http://wiki.debian.org/Multiarch
  if (NOT DEFINED CMAKE_SYSTEM_NAME OR NOT DEFINED CMAKE_SIZEOF_VOID_P)
    message(AUTHOR_WARNING
      "Unable to determine default CMAKE_INSTALL_LIBDIR directory because no target architecture is known. "
      "Please enable at least one language before including GNUInstallDirs.")
  endif()
  if(CMAKE_SYSTEM_NAME MATCHES "^(Linux|GNU)$" AND NOT CMAKE_CROSSCOMPILING)
    _GNUInstallDirs_get_system_type_for_install(system_type)
    if(system_type STREQUAL "debian")
      if(CMAKE_LIBRARY_ARCHITECTURE)
        if("${install_prefix}" MATCHES "^/usr/?$")
          set(${out_var} "lib/${CMAKE_LIBRARY_ARCHITECTURE}")
        endif()
      endif()
    elseif(NOT DEFINED system_type)
      # not debian, alpine, arch, or conda so rely on CMAKE_SIZEOF_VOID_P:
      if("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
        set(${out_var} "lib64")
      endif()
    endif()
  endif()
  __GNUInstallDirs_default_in_usr(${out_var} "${${out_var}}" "${install_prefix}")

  return(PROPAGATE ${out_var})
endfunction()

foreach(dir IN ITEMS
    SYSCONFDIR
    LOCALSTATEDIR
)
  # Cannot call function() directly because `dir` would not be accessible inside the function
  # Using cmake_language(EVAL) to call a short wrapper function instead
  cmake_language(EVAL CODE "
    function(_GNUInstallDirs_${dir}_get_default out_var install_prefix)
      set(\${out_var} \"\${_GNUInstallDirs_${dir}_DEFAULT}\")
      __GNUInstallDirs_default_in_root(\${out_var} \"\${\${out_var}}\" \"\${install_prefix}\")
      return(PROPAGATE \${out_var})
    endfunction()
  "
  )
endforeach()

# Depends on current CMAKE_INSTALL_LOCALSTATEDIR value
function(_GNUInstallDirs_RUNSTATEDIR_get_default out_var install_prefix)
  set(${out_var} "${_GNUInstallDirs_RUNSTATEDIR_DEFAULT}")
  if(_GNUInstallDirs_CMP0192 STREQUAL "NEW")
    # In the /opt/ case we want the install_prefix to be appended as
    # LOCALSTATEDIR/run/PREFIX
    if(install_prefix MATCHES "^/opt/" AND NOT install_prefix MATCHES "^/opt/homebrew/")
        string(REPLACE "${install_prefix}" "/run${install_prefix}"
          ${out_var} "${CMAKE_INSTALL_LOCALSTATEDIR}"
        )
    endif()
  endif()
  return(PROPAGATE ${out_var})
endfunction()

# All of the other (primitive) dirs are typically in usr/<dir>.
# A special handling is needed for the `/` install_prefix
foreach(dir IN ITEMS
    BINDIR
    SBINDIR
    LIBEXECDIR
    SHAREDSTATEDIR
    INCLUDEDIR
    OLDINCLUDEDIR
    DATAROOTDIR
    # Except all the previous ones that had a special handling:
    # LIBDIR, SYSCONFDIR, LOCALSTATEDIR, OLDINCLUDEDIR
)
  # Cannot call function() directly because `dir` would not be accessible inside the function
  # Using cmake_language(EVAL) to call a short wrapper function instead
  cmake_language(EVAL CODE "
    function(_GNUInstallDirs_${dir}_get_default out_var install_prefix)
      set(\${out_var} \"\${_GNUInstallDirs_${dir}_DEFAULT}\")
      __GNUInstallDirs_default_in_usr(\${out_var} \"\${\${out_var}}\" \"\${install_prefix}\")
      return(PROPAGATE \${out_var})
    endfunction()
  "
  )
endforeach()

_GNUInstallDirs_cache_path(BINDIR
  "User executables")
_GNUInstallDirs_cache_path(SBINDIR
  "System admin executables")
_GNUInstallDirs_cache_path(LIBEXECDIR
  "Program executables")
_GNUInstallDirs_cache_path(SYSCONFDIR
  "Read-only single-machine data")
_GNUInstallDirs_cache_path(SHAREDSTATEDIR
  "Modifiable architecture-independent data")
_GNUInstallDirs_cache_path(LOCALSTATEDIR
  "Modifiable single-machine data")
_GNUInstallDirs_cache_path(LIBDIR
  "Object code libraries")
_GNUInstallDirs_cache_path(INCLUDEDIR
  "C header files")
_GNUInstallDirs_cache_path(OLDINCLUDEDIR
  "C header files for non-gcc")
_GNUInstallDirs_cache_path(DATAROOTDIR
  "Read-only architecture-independent data root")

#-----------------------------------------------------------------------------
# Values whose defaults are relative to DATAROOTDIR.  Store empty values in
# the cache and store the defaults in local variables if the cache values are
# not set explicitly.  This auto-updates the defaults as DATAROOTDIR changes.

if(CMAKE_SYSTEM_NAME MATCHES "^(([^kF].*)?BSD|DragonFly)$")
  set(_GNUInstallDirs_INFODIR_DEFAULT "info")
  _GNUInstallDirs_cache_path(INFODIR
    "Info documentation")
else()
  set(_GNUInstallDirs_INFODIR_DEFAULT "${CMAKE_INSTALL_DATAROOTDIR}/info")
  _GNUInstallDirs_cache_path_fallback(INFODIR
    "Info documentation (DATAROOTDIR/info)")
endif()

if(CMAKE_SYSTEM_NAME MATCHES "^(([^k].*)?BSD|DragonFly)$" AND NOT CMAKE_SYSTEM_NAME MATCHES "^(FreeBSD)$")
  set(_GNUInstallDirs_MANDIR_DEFAULT "man")
  _GNUInstallDirs_cache_path(MANDIR
    "Man documentation")
else()
  set(_GNUInstallDirs_MANDIR_DEFAULT "${CMAKE_INSTALL_DATAROOTDIR}/man")
  _GNUInstallDirs_cache_path_fallback(MANDIR
    "Man documentation (DATAROOTDIR/man)")
endif()

set(_GNUInstallDirs_DATADIR_DEFAULT "${CMAKE_INSTALL_DATAROOTDIR}")
set(_GNUInstallDirs_LOCALEDIR_DEFAULT "${CMAKE_INSTALL_DATAROOTDIR}/locale")
set(_GNUInstallDirs_DOCDIR_DEFAULT "${CMAKE_INSTALL_DATAROOTDIR}/doc/${PROJECT_NAME}")
set(_GNUInstallDirs_RUNSTATEDIR_DEFAULT "${CMAKE_INSTALL_LOCALSTATEDIR}/run")

_GNUInstallDirs_cache_path_fallback(DATADIR
  "Read-only architecture-independent data (DATAROOTDIR)")
_GNUInstallDirs_cache_path_fallback(LOCALEDIR
  "Locale-dependent data (DATAROOTDIR/locale)")
_GNUInstallDirs_cache_path_fallback(DOCDIR
  "Documentation root (DATAROOTDIR/doc/PROJECT_NAME)")
_GNUInstallDirs_cache_path_fallback(RUNSTATEDIR
  "Run-time variable data (LOCALSTATEDIR/run)")

# Unset all the defaults used
foreach(dir IN ITEMS
    BINDIR
    SBINDIR
    LIBEXECDIR
    SYSCONFDIR
    SHAREDSTATEDIR
    LOCALSTATEDIR
    LIBDIR
    INCLUDEDIR
    OLDINCLUDEDIR
    DATAROOTDIR
    DATADIR
    INFODIR
    MANDIR
    LOCALEDIR
    DOCDIR
    RUNSTATEDIR
)
  unset(_GNUInstallDirs_${dir}_DEFAULT)
endforeach()

# Save for next run
set(_GNUInstallDirs_LAST_CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}" CACHE INTERNAL "CMAKE_INSTALL_PREFIX during last run")

#-----------------------------------------------------------------------------

mark_as_advanced(
  CMAKE_INSTALL_BINDIR
  CMAKE_INSTALL_SBINDIR
  CMAKE_INSTALL_LIBEXECDIR
  CMAKE_INSTALL_SYSCONFDIR
  CMAKE_INSTALL_SHAREDSTATEDIR
  CMAKE_INSTALL_LOCALSTATEDIR
  CMAKE_INSTALL_RUNSTATEDIR
  CMAKE_INSTALL_LIBDIR
  CMAKE_INSTALL_INCLUDEDIR
  CMAKE_INSTALL_OLDINCLUDEDIR
  CMAKE_INSTALL_DATAROOTDIR
  CMAKE_INSTALL_DATADIR
  CMAKE_INSTALL_INFODIR
  CMAKE_INSTALL_LOCALEDIR
  CMAKE_INSTALL_MANDIR
  CMAKE_INSTALL_DOCDIR
  )

function(GNUInstallDirs_get_absolute_install_dir absvar var)
  set(GGAID_extra_args ${ARGN})
  list(LENGTH GGAID_extra_args GGAID_extra_arg_count)
  if(GGAID_extra_arg_count GREATER "0")
    list(GET GGAID_extra_args 0 GGAID_dir)
  else()
    # Historical behavior: use ${dir} from caller's scope
    set(GGAID_dir "${dir}")
    message(AUTHOR_WARNING
      "GNUInstallDirs_get_absolute_install_dir called without third argument. "
      "Using \${dir} from the caller's scope for compatibility with CMake 3.19 and below.")
  endif()

  if(NOT IS_ABSOLUTE "${${var}}")
    # Handle special cases:
    # - CMAKE_INSTALL_PREFIX == /
    # - CMAKE_INSTALL_PREFIX == /usr
    # - CMAKE_INSTALL_PREFIX == /opt/...
    if("${GGAID_dir}" STREQUAL "SYSCONFDIR" OR "${GGAID_dir}" STREQUAL "LOCALSTATEDIR" OR "${GGAID_dir}" STREQUAL "RUNSTATEDIR")
      _GNUInstallDirs_special_absolute(${absvar} "${${var}}" "${CMAKE_INSTALL_PREFIX}")
      # If the CMAKE_INSTALL_PREFIX was not special, the output
      # is still not absolute, so use the default logic.
      if(NOT IS_ABSOLUTE "${${absvar}}")
        # Make sure we account for any trailing `/`
        if(CMAKE_INSTALL_PREFIX MATCHES "/$")
          set(${absvar} "${CMAKE_INSTALL_PREFIX}${${var}}")
        else()
          set(${absvar} "${CMAKE_INSTALL_PREFIX}/${${var}}")
        endif()
      endif()
    elseif("${CMAKE_INSTALL_PREFIX}" STREQUAL "/")
      if (NOT "${${var}}" MATCHES "^usr/")
        set(${var} "usr/${${var}}")
      endif()
      set(${absvar} "/${${var}}")
    else()
      set(${absvar} "${CMAKE_INSTALL_PREFIX}/${${var}}")
    endif()
  else()
    set(${absvar} "${${var}}")
  endif()

  set(return_vars ${absvar})
  cmake_policy(GET CMP0193 cmp0193
    PARENT_SCOPE # undocumented, do not use outside of CMake
  )
  if(NOT cmp0193 STREQUAL "NEW")
    list(APPEND return_vars ${var})
  endif()
  return(PROPAGATE ${return_vars})
endfunction()

# Result directories
#
foreach(dir
    BINDIR
    SBINDIR
    LIBEXECDIR
    SYSCONFDIR
    SHAREDSTATEDIR
    LOCALSTATEDIR
    RUNSTATEDIR
    LIBDIR
    INCLUDEDIR
    OLDINCLUDEDIR
    DATAROOTDIR
    DATADIR
    INFODIR
    LOCALEDIR
    MANDIR
    DOCDIR
    )
  GNUInstallDirs_get_absolute_install_dir(CMAKE_INSTALL_FULL_${dir} CMAKE_INSTALL_${dir} ${dir})
endforeach()

unset(_GNUInstallDirs_CMP0192)
