# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindEnvModules
--------------

.. versionadded:: 3.15

Finds an Environment Modules implementation and provides commands for use in
CMake scripts:

.. code-block:: cmake

  find_package(EnvModules [...])

The Environment Modules system is a command-line tool that manages Unix-like
shell environments by dynamically modifying environment variables.
It is commonly used in High-Performance Computing (HPC) environments to
support multiple software versions or configurations.

This module is compatible with the two most common implementations:

* Lua-based Lmod
* TCL-based Environment Modules

This module is primarily intended for setting up compiler and library
environments within a :ref:`CTest Script <CTest Script>` (``ctest -S``).
It may also be used in a :ref:`CMake Script <Script Processing Mode>`
(``cmake -P``).

.. note::

  The loaded environment will not persist beyond the end of the calling
  process.  Do not use this module in CMake project code (such as
  ``CMakeLists.txt``) to load compiler environments, as the environment
  changes will not be available during the build phase.  In such a case, load
  the desired environment before invoking CMake or the generated build system.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``EnvModules_FOUND``
  Boolean indicating whether a compatible Environment Modules framework was
  found.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``EnvModules_COMMAND``
  The path to a low level module command to use.

Hints
^^^^^

This module accepts the following variables before calling the
``find_package(EnvModules)``:

``ENV{MODULESHOME}``
  This environment variable is usually set by the Environment Modules
  implementation, and can be used as a hint to locate the module command to
  execute.

Commands
^^^^^^^^

This module provides the following commands for interacting with the
Environment Modules system, if found:

.. command:: env_module

  Executes an arbitrary module command:

  .. code-block:: cmake

    env_module(<command> <args>...)
    env_module(
      COMMAND <command> <args>...
      [OUTPUT_VARIABLE <out-var>]
      [RESULT_VARIABLE <ret-var>]
    )

  The options are:

  ``COMMAND <command> <args>...``
    The module sub-command and arguments to execute as if passed directly to
    the module command in the shell environment.

  ``OUTPUT_VARIABLE <out-var>``
    Stores the standard output of the executed module command in the specified
    variable.

  ``RESULT_VARIABLE <ret-var>``
    Stores the return code of the executed module command in the specified
    variable.

.. command:: env_module_swap

  Swaps one module for another:

  .. code-block:: cmake

    env_module_swap(
      <out-mod>
      <in-mod>
      [OUTPUT_VARIABLE <out-var>]
      [RESULT_VARIABLE <ret-var>]
    )

  This is functionally equivalent to the ``module swap <out-mod> <in-mod>``
  shell command.

  The options are:

  ``OUTPUT_VARIABLE <out-var>``
    Stores the standard output of the executed module command in the specified
    variable.

  ``RESULT_VARIABLE <ret-var>``
    Stores the return code of the executed module command in the specified
    variable.

.. command:: env_module_list

  Retrieves the list of currently loaded modules:

  .. code-block:: cmake

    env_module_list(<out-var>)

  This is functionally equivalent to the ``module list`` shell command.
  The result is stored in ``<out-var>`` as a properly formatted CMake
  :ref:`semicolon-separated list <CMake Language Lists>` variable.

.. command:: env_module_avail

  Retrieves the list of available modules:

  .. code-block:: cmake

    env_module_avail([<mod-prefix>] <out-var>)

  This is functionally equivalent to the ``module avail <mod-prefix>`` shell
  command.  The result is stored in ``<out-var>`` as a properly formatted
  CMake :ref:`semicolon-separated list <CMake Language Lists>` variable.

Examples
^^^^^^^^

In the following example, this module is used in a CTest script to configure
the compiler and libraries for a Cray Programming Environment.
After the Environment Modules system is found, the ``env_module()`` command is
used to load the necessary compiler, MPI, and scientific libraries to set up
the build environment.  The ``CRAYPE_LINK_TYPE`` environment variable is set
to ``dynamic`` to specify dynamic linking.  This instructs the Cray Linux
Environment compiler driver to link against dynamic libraries at runtime,
rather than linking static libraries at compile time.  As a result, the
compiler produces dynamically linked executable files.

.. code-block:: cmake
  :caption: ``example-script.cmake``

  set(CTEST_BUILD_NAME "CrayLinux-CrayPE-Cray-dynamic")
  set(CTEST_BUILD_CONFIGURATION Release)
  set(CTEST_BUILD_FLAGS "-k -j8")
  set(CTEST_CMAKE_GENERATOR "Unix Makefiles")

  # ...

  find_package(EnvModules REQUIRED)

  # Clear all currently loaded Environment Modules to start with a clean state
  env_module(purge)

  # Load the base module-handling system to use other modules
  env_module(load modules)

  # Load Cray Programming Environment (Cray PE) support, which manages
  # platform-specific optimizations and architecture selection
  env_module(load craype)

  # Load the Cray programming environment
  env_module(load PrgEnv-cray)

  # Load settings targeting the Intel Knights Landing (KNL) CPU architecture
  env_module(load craype-knl)

  # Load the Cray MPI (Message Passing Interface) library, needed for
  # distributed computing
  env_module(load cray-mpich)

  # Load Cray's scientific library package, which includes optimized math
  # libraries (like BLAS, LAPACK)
  env_module(load cray-libsci)

  set(ENV{CRAYPE_LINK_TYPE} dynamic)

  # ...
#]=======================================================================]

function(env_module)
  if(NOT EnvModules_COMMAND)
    message(FATAL_ERROR "Failed to process module command.  EnvModules_COMMAND not found")
    return()
  endif()

  set(options)
  set(oneValueArgs OUTPUT_VARIABLE RESULT_VARIABLE)
  set(multiValueArgs COMMAND)
  cmake_parse_arguments(MOD_ARGS
    "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGV}
  )
  if(NOT MOD_ARGS_COMMAND)
    # If no explicit command argument was given, then treat the calling syntax
    # as: module(cmd args...)
    set(exec_cmd ${ARGV})
  else()
    set(exec_cmd ${MOD_ARGS_COMMAND})
  endif()

  if(MOD_ARGS_OUTPUT_VARIABLE)
    set(err_var_args ERROR_VARIABLE err_var)
  endif()

  execute_process(
    COMMAND mktemp -t module.cmake.XXXXXXXXXXXX
    OUTPUT_VARIABLE tempfile_name
  )
  string(STRIP "${tempfile_name}" tempfile_name)

  # If the $MODULESHOME/init/cmake file exists then assume that the CMake
  # "shell" functionality exits
  if(EXISTS "$ENV{MODULESHOME}/init/cmake")
    execute_process(
      COMMAND ${EnvModules_COMMAND} cmake ${exec_cmd}
      OUTPUT_FILE ${tempfile_name}
      ${err_var_args}
      RESULT_VARIABLE ret_var
    )

  else() # fallback to the sh shell and manually convert to CMake
    execute_process(
      COMMAND ${EnvModules_COMMAND} sh ${exec_cmd}
      OUTPUT_VARIABLE out_var
      ${err_var_args}
      RESULT_VARIABLE ret_var
    )
  endif()

  # If we executed successfully then process and cleanup the temp file
  if(ret_var EQUAL 0)
    # No CMake shell so we need to process the sh output into CMake code
    if(NOT EXISTS "$ENV{MODULESHOME}/init/cmake")
      file(WRITE ${tempfile_name} "")
      string(REPLACE "\n" ";" out_var "${out_var}")
      foreach(sh_cmd IN LISTS out_var)
        if(sh_cmd MATCHES "^ *unset *([^ ]*)")
          set(cmake_cmd "unset(ENV{${CMAKE_MATCH_1}})")
        elseif(sh_cmd MATCHES "^ *export *([^ ]*)")
          set(cmake_cmd "set(ENV{${CMAKE_MATCH_1}} \"\${${CMAKE_MATCH_1}}\")")
        elseif(sh_cmd MATCHES " *([^ =]*) *= *(.*)")
          set(var_name "${CMAKE_MATCH_1}")
          set(var_value "${CMAKE_MATCH_2}")
          if(var_value MATCHES "^\"(.*[^\\])\"")
            # If it's in quotes, take the value as is
            set(var_value "${CMAKE_MATCH_1}")
          else()
            # Otherwise, strip trailing spaces
            string(REGEX REPLACE "([^\\])? +$" "\\1" var_value "${var_value}")
          endif()
          string(REPLACE "\\ " " " var_value "${var_value}")
          set(cmake_cmd "set(${var_name} \"${var_value}\")")
        else()
          continue()
        endif()
        file(APPEND ${tempfile_name} "${cmake_cmd}\n")
      endforeach()
    endif()

    # Process the change in environment variables
    include(${tempfile_name})
    file(REMOVE ${tempfile_name})
  endif()

  # Push the output back out to the calling scope
  if(MOD_ARGS_OUTPUT_VARIABLE)
    set(${MOD_ARGS_OUTPUT_VARIABLE} "${err_var}" PARENT_SCOPE)
  endif()
  if(MOD_ARGS_RESULT_VARIABLE)
    set(${MOD_ARGS_RESULT_VARIABLE} ${ret_var} PARENT_SCOPE)
  endif()
endfunction()

#------------------------------------------------------------------------------
function(env_module_swap out_mod in_mod)
  set(options)
  set(oneValueArgs OUTPUT_VARIABLE RESULT_VARIABLE)
  set(multiValueArgs)

  cmake_parse_arguments(MOD_ARGS
    "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGV}
  )

  env_module(COMMAND -t swap ${out_mod} ${in_mod}
    OUTPUT_VARIABLE tmp_out
    RETURN_VARIABLE tmp_ret
  )

  if(MOD_ARGS_OUTPUT_VARIABLE)
    set(${MOD_ARGS_OUTPUT_VARIABLE} "${tmp_out}" PARENT_SCOPE)
  endif()
  if(MOD_ARGS_RESULT_VARIABLE)
    set(${MOD_ARGS_RESULT_VARIABLE} ${tmp_ret} PARENT_SCOPE)
  endif()
endfunction()

#------------------------------------------------------------------------------
function(env_module_list out_var)
  env_module(COMMAND -t list OUTPUT_VARIABLE tmp_out)

  # Convert output into a CMake list
  string(REPLACE "\n" ";" ${out_var} "${tmp_out}")

  # Remove title headers and empty entries
  list(REMOVE_ITEM ${out_var} "No modules loaded")
  if(${out_var})
    list(FILTER ${out_var} EXCLUDE REGEX "^(.*:)?$")
  endif()
  list(FILTER ${out_var} EXCLUDE REGEX "^(.*:)?$")

  set(${out_var} ${${out_var}} PARENT_SCOPE)
endfunction()

#------------------------------------------------------------------------------
function(env_module_avail)
  if(ARGC EQUAL 1)
    set(mod_prefix)
    set(out_var ${ARGV0})
  elseif(ARGC EQUAL 2)
    set(mod_prefix ${ARGV0})
    set(out_var ${ARGV1})
  else()
    message(FATAL_ERROR "Usage: env_module_avail([mod_prefix] out_var)")
  endif()
  env_module(COMMAND -t avail ${mod_prefix} OUTPUT_VARIABLE tmp_out)

  # Convert output into a CMake list
  string(REPLACE "\n" ";" tmp_out "${tmp_out}")

  set(${out_var})
  foreach(MOD IN LISTS tmp_out)
    # Remove directory entries and empty values
    if(MOD MATCHES "^(.*:)?$")
      continue()
    endif()

    # Convert default modules
    if(MOD MATCHES "^(.*)/$" ) # "foo/"
      list(APPEND ${out_var} ${CMAKE_MATCH_1})
    elseif(MOD MATCHES "^((.*)/.*)\\(default\\)$") # "foo/1.2.3(default)"
      list(APPEND ${out_var} ${CMAKE_MATCH_2})
      list(APPEND ${out_var} ${CMAKE_MATCH_1})
    else()
      list(APPEND ${out_var} ${MOD})
    endif()
  endforeach()

  set(${out_var} ${${out_var}} PARENT_SCOPE)
endfunction()

#------------------------------------------------------------------------------
# Make sure we know where the underlying module command is
find_program(EnvModules_COMMAND
  NAMES lmod modulecmd
  HINTS ENV MODULESHOME
  PATH_SUFFIXES libexec
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(EnvModules DEFAULT_MSG EnvModules_COMMAND)
