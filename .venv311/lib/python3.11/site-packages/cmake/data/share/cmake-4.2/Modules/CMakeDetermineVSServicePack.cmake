# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
CMakeDetermineVSServicePack
---------------------------

.. versionchanged:: 4.1

  This module is available only if policy :policy:`CMP0196` is not set to ``NEW``.

.. deprecated:: 3.0

  This module should no longer be used.  The functionality of this module has
  been superseded by the :variable:`CMAKE_<LANG>_COMPILER_VERSION` variable that
  contains the compiler version number.

This module provides a command to determine the installed Visual Studio
service pack version for Visual Studio 2012 and earlier.

Load this module in a CMake project with:

.. code-block:: cmake

  include(CMakeDetermineVSServicePack)

Commands
^^^^^^^^

This module provides the following command:

.. command:: DetermineVSServicePack

  Determines the Visual Studio service pack version of the ``cl`` compiler
  in use:

  .. code-block:: cmake

    DetermineVSServicePack(<variable>)

  The result is stored in the specified internal cache variable ``<variable>``,
  which is set to one of the following values, or to an empty string if the
  service pack cannot be determined:

  * ``vc80``, ``vc80sp1``
  * ``vc90``, ``vc90sp1``
  * ``vc100``, ``vc100sp1``
  * ``vc110``, ``vc110sp1``, ``vc110sp2``, ``vc110sp3``, ``vc110sp4``

Examples
^^^^^^^^

Determining the Visual Studio service pack version in a project:

.. code-block:: cmake

  if(MSVC)
    include(CMakeDetermineVSServicePack)
    DetermineVSServicePack(my_service_pack)
    if(my_service_pack)
      message(STATUS "Detected: ${my_service_pack}")
    endif()
  endif()
#]=======================================================================]

cmake_policy(GET CMP0196 _CMakeDetermineVSServicePack_CMP0196)
if(_CMakeDetermineVSServicePack_CMP0196 STREQUAL "NEW")
  message(FATAL_ERROR "The CMakeDetermineVSServicePack module has been removed by policy CMP0196.")
endif()

if(_CMakeDetermineVSServicePack_testing)
  set(_CMakeDetermineVSServicePack_included TRUE)
  return()
endif()

if(NOT CMAKE_MINIMUM_REQUIRED_VERSION VERSION_LESS 2.8.8)
  message(DEPRECATION
    "This module is deprecated and should not be used.  "
    "Use the CMAKE_<LANG>_COMPILER_VERSION variable instead."
    )
endif()

# [INTERNAL]
# Please do not call this function directly
function(_DetermineVSServicePackFromCompiler _OUT_VAR _cl_version)
  if    (${_cl_version} VERSION_EQUAL "14.00.50727.42")
    set(_version "vc80")
  elseif(${_cl_version} VERSION_EQUAL "14.00.50727.762")
    set(_version "vc80sp1")
  elseif(${_cl_version} VERSION_EQUAL "15.00.21022.08")
    set(_version "vc90")
  elseif(${_cl_version} VERSION_EQUAL "15.00.30729.01")
    set(_version "vc90sp1")
  elseif(${_cl_version} VERSION_EQUAL "16.00.30319.01")
    set(_version "vc100")
  elseif(${_cl_version} VERSION_EQUAL "16.00.40219.01")
    set(_version "vc100sp1")
  elseif(${_cl_version} VERSION_EQUAL "17.00.50727.1")
    set(_version "vc110")
  elseif(${_cl_version} VERSION_EQUAL "17.00.51106.1")
    set(_version "vc110sp1")
  elseif(${_cl_version} VERSION_EQUAL "17.00.60315.1")
    set(_version "vc110sp2")
  elseif(${_cl_version} VERSION_EQUAL "17.00.60610.1")
    set(_version "vc110sp3")
  elseif(${_cl_version} VERSION_EQUAL "17.00.61030")
    set(_version "vc110sp4")
  else()
    set(_version "")
  endif()
  set(${_OUT_VAR} ${_version} PARENT_SCOPE)
endfunction()


############################################################
# [INTERNAL]
# Please do not call this function directly
function(_DetermineVSServicePack_FastCheckVersionWithCompiler _SUCCESS_VAR  _VERSION_VAR)
    if(EXISTS ${CMAKE_CXX_COMPILER})
      execute_process(
          COMMAND ${CMAKE_CXX_COMPILER} -?
          ERROR_VARIABLE _output
          OUTPUT_QUIET
        )

      if(_output MATCHES "Compiler Version (([0-9]+)\\.([0-9]+)\\.([0-9]+)(\\.([0-9]+))?)")
        set(_cl_version ${CMAKE_MATCH_1})
        set(_major ${CMAKE_MATCH_2})
        set(_minor ${CMAKE_MATCH_3})
        if("${_major}${_minor}" STREQUAL "${MSVC_VERSION}")
          set(${_SUCCESS_VAR} true PARENT_SCOPE)
          set(${_VERSION_VAR} ${_cl_version} PARENT_SCOPE)
        endif()
      endif()
    endif()
endfunction()

############################################################
# [INTERNAL]
# Please do not call this function directly
function(_DetermineVSServicePack_CheckVersionWithTryCompile _SUCCESS_VAR  _VERSION_VAR)
    file(WRITE "${CMAKE_BINARY_DIR}/return0.cc"
      "int main() { return 0; }\n")

    try_compile(
      _CompileResult
      SOURCES "${CMAKE_BINARY_DIR}/return0.cc"
      OUTPUT_VARIABLE _output
      COPY_FILE "${CMAKE_BINARY_DIR}/return0.cc")

    file(REMOVE "${CMAKE_BINARY_DIR}/return0.cc")

    if(_output MATCHES "Compiler Version (([0-9]+)\\.([0-9]+)\\.([0-9]+)(\\.([0-9]+))?)")
      set(${_SUCCESS_VAR} true PARENT_SCOPE)
      set(${_VERSION_VAR} "${CMAKE_MATCH_1}" PARENT_SCOPE)
    endif()
endfunction()

############################################################
# [INTERNAL]
# Please do not call this function directly
function(_DetermineVSServicePack_CheckVersionWithTryRun _SUCCESS_VAR  _VERSION_VAR)
    file(WRITE "${CMAKE_BINARY_DIR}/return0.cc"
        "#include <stdio.h>\n\nconst unsigned int CompilerVersion=_MSC_FULL_VER;\n\nint main(int argc, char* argv[])\n{\n  int M( CompilerVersion/10000000);\n  int m((CompilerVersion%10000000)/100000);\n  int b(CompilerVersion%100000);\n\n  printf(\"%d.%02d.%05d.01\",M,m,b);\n return 0;\n}\n")

    try_run(
        _RunResult
        _CompileResult
        SOURCES "${CMAKE_BINARY_DIR}/return0.cc"
        RUN_OUTPUT_VARIABLE  _runoutput
        )

    file(REMOVE "${CMAKE_BINARY_DIR}/return0.cc")

    string(REGEX MATCH "[0-9]+.[0-9]+.[0-9]+.[0-9]+"
        _cl_version "${_runoutput}")

    if(_cl_version)
      set(${_SUCCESS_VAR} true PARENT_SCOPE)
      set(${_VERSION_VAR} ${_cl_version} PARENT_SCOPE)
    endif()
endfunction()


#
# A function to call to determine the Visual Studio service pack
# in use.  See documentation above.
function(DetermineVSServicePack _pack)
    if(NOT DETERMINED_VS_SERVICE_PACK OR NOT ${_pack})

        _DetermineVSServicePack_FastCheckVersionWithCompiler(DETERMINED_VS_SERVICE_PACK _cl_version)
        if(NOT DETERMINED_VS_SERVICE_PACK)
            _DetermineVSServicePack_CheckVersionWithTryCompile(DETERMINED_VS_SERVICE_PACK _cl_version)
            if(NOT DETERMINED_VS_SERVICE_PACK)
                _DetermineVSServicePack_CheckVersionWithTryRun(DETERMINED_VS_SERVICE_PACK _cl_version)
            endif()
        endif()

        if(DETERMINED_VS_SERVICE_PACK)

            if(_cl_version)
                # Call helper function to determine VS version
                _DetermineVSServicePackFromCompiler(_sp "${_cl_version}")
                if(_sp)
                    set(${_pack} ${_sp} CACHE INTERNAL
                        "The Visual Studio Release with Service Pack")
                endif()
            endif()
        endif()
    endif()
endfunction()
