# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
CPackIFWConfigureFile
---------------------

.. versionadded:: 3.8

This module defines :command:`configure_file` similar command to
configure file templates prepared in QtIFW/SDK/Creator style.

Load this module in a CMake project with:

.. code-block:: cmake

  include(CPackIFWConfigureFile)

Commands
^^^^^^^^

The module defines the following commands:

.. command:: cpack_ifw_configure_file

  Copy a file to another location and modify its contents.

  .. code-block:: cmake

    cpack_ifw_configure_file(<input> <output>)

  Copies an ``<input>`` file to an ``<output>`` file and substitutes variable
  values referenced as ``%{VAR}`` or ``%VAR%`` in the input file content.
  Each variable reference will be replaced with the current value of the
  variable, or the empty string if the variable is not defined.

#]=======================================================================]

if(NOT DEFINED CPackIFWConfigureFile_CMake_INCLUDED)
set(CPackIFWConfigureFile_CMake_INCLUDED 1)

macro(cpack_ifw_configure_file INPUT OUTPUT)
  file(READ "${INPUT}" _tmp)
  foreach(_tmp_regex "%{([^%}]+)}" "%([^%]+)%")
    string(REGEX MATCHALL "${_tmp_regex}" _tmp_vars "${_tmp}")
    while(_tmp_vars)
      foreach(_tmp_var ${_tmp_vars})
        string(REGEX REPLACE "${_tmp_regex}" "\\1"
          _tmp_var_name "${_tmp_var}")
        if(DEFINED ${_tmp_var_name})
          set(_tmp_var_value "${${_tmp_var_name}}")
        elseif(NOT "$ENV{${_tmp_var_name}}" STREQUAL "")
          set(_tmp_var_value "$ENV{${_tmp_var_name}}")
        else()
          set(_tmp_var_value "")
        endif()
        string(REPLACE "${_tmp_var}" "${_tmp_var_value}" _tmp "${_tmp}")
      endforeach()
      string(REGEX MATCHALL "${_tmp_regex}" _tmp_vars "${_tmp}")
    endwhile()
  endforeach()
  if(IS_ABSOLUTE "${OUTPUT}")
    file(WRITE "${OUTPUT}" "${_tmp}")
  else()
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT}" "${_tmp}")
  endif()
endmacro()

endif() # NOT DEFINED CPackIFWConfigureFile_CMake_INCLUDED
