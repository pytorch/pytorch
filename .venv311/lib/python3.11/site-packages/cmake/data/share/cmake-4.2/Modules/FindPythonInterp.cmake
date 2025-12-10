# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindPythonInterp
----------------

.. versionchanged:: 3.27
  This module is available only if policy :policy:`CMP0148` is not set to ``NEW``.

.. deprecated:: 3.12

  Use :module:`FindPython3`, :module:`FindPython2`, or :module:`FindPython`
  instead.

Finds the Python interpreter and determines the location of its executable:

.. code-block:: cmake

  find_package(PythonInterp [<version>] [...])

.. note::

  When using both this and the :module:`FindPythonLibs` module, call
  ``find_package(PythonInterp)`` before ``find_package(PythonLibs)``.  This
  ensures that the detected interpreter version is used to guide the selection
  of compatible libraries, resulting in a consistent ``PYTHON_LIBRARIES`` value.

.. note::

  A call to ``find_package(PythonInterp ${V})`` for Python version ``V`` may
  find a ``python`` executable with no version suffix.  In this case no attempt
  is made to avoid Python executables from other versions.  Use
  :module:`FindPython3`, :module:`FindPython2`, or :module:`FindPython` instead.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``PythonInterp_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether the (requested version of) Python executable
  was found.

``PYTHON_VERSION_STRING``
  Python version found (e.g., ``2.5.2``).
``PYTHON_VERSION_MAJOR``
  Python major version found (e.g., ``2``).
``PYTHON_VERSION_MINOR``
  Python minor version found (e.g., ``5``).
``PYTHON_VERSION_PATCH``
  Python patch version found (e.g., ``2``).

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``PYTHON_EXECUTABLE``
  The path to the Python interpreter.

Hints
^^^^^

This module accepts the following variables before calling
``find_package(PythonInterp)``:

``Python_ADDITIONAL_VERSIONS``
  This variable can be used to specify a list of version numbers that should be
  taken into account when searching for Python.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``PYTHONINTERP_FOUND``
  .. deprecated:: 3.12
    Use ``PythonInterp_FOUND``, which has the same value.

  Boolean indicating whether the (requested version of) Python executable
  was found.

Examples
^^^^^^^^

Finding the Python interpreter in earlier versions of CMake:

.. code-block:: cmake

  find_package(PythonInterp)
  execute_process(COMMAND ${PYTHON_EXECUTABLE} --help)

Starting with CMake 3.12, the Python interpreter can be found using the
:module:`FindPython` module.  The equivalent example using the modern approach
is:

.. code-block:: cmake

  find_package(Python)
  execute_process(COMMAND ${Python_EXECUTABLE} --help)
#]=======================================================================]

cmake_policy(GET CMP0148 _FindPythonInterp_CMP0148)
if(_FindPythonInterp_CMP0148 STREQUAL "NEW")
  message(FATAL_ERROR "The FindPythonInterp module has been removed by policy CMP0148.")
endif()

if(_FindPythonInterp_testing)
  set(_FindPythonInterp_included TRUE)
  return()
endif()

unset(_Python_NAMES)

set(_PYTHON1_VERSIONS 1.6 1.5)
set(_PYTHON2_VERSIONS 2.7 2.6 2.5 2.4 2.3 2.2 2.1 2.0)
set(_PYTHON3_VERSIONS 3.15 3.14 3.13 3.12 3.11 3.10 3.9 3.8 3.7 3.6 3.5 3.4 3.3 3.2 3.1 3.0)

if(PythonInterp_FIND_VERSION)
    if(PythonInterp_FIND_VERSION_COUNT GREATER 1)
        set(_PYTHON_FIND_MAJ_MIN "${PythonInterp_FIND_VERSION_MAJOR}.${PythonInterp_FIND_VERSION_MINOR}")
        list(APPEND _Python_NAMES
             python${_PYTHON_FIND_MAJ_MIN}
             python${PythonInterp_FIND_VERSION_MAJOR})
        unset(_PYTHON_FIND_OTHER_VERSIONS)
        if(NOT PythonInterp_FIND_VERSION_EXACT)
            foreach(_PYTHON_V ${_PYTHON${PythonInterp_FIND_VERSION_MAJOR}_VERSIONS})
                if(NOT _PYTHON_V VERSION_LESS _PYTHON_FIND_MAJ_MIN)
                    list(APPEND _PYTHON_FIND_OTHER_VERSIONS ${_PYTHON_V})
                endif()
             endforeach()
        endif()
        unset(_PYTHON_FIND_MAJ_MIN)
    else()
        list(APPEND _Python_NAMES python${PythonInterp_FIND_VERSION_MAJOR})
        set(_PYTHON_FIND_OTHER_VERSIONS ${_PYTHON${PythonInterp_FIND_VERSION_MAJOR}_VERSIONS})
    endif()
else()
    set(_PYTHON_FIND_OTHER_VERSIONS ${_PYTHON3_VERSIONS} ${_PYTHON2_VERSIONS} ${_PYTHON1_VERSIONS})
endif()
find_program(PYTHON_EXECUTABLE NAMES ${_Python_NAMES})

# Set up the versions we know about, in the order we will search. Always add
# the user supplied additional versions to the front.
set(_Python_VERSIONS ${Python_ADDITIONAL_VERSIONS})
# If FindPythonInterp has already found the major and minor version,
# insert that version next to get consistent versions of the interpreter and
# library.
if(DEFINED PYTHONLIBS_VERSION_STRING)
  string(REPLACE "." ";" _PYTHONLIBS_VERSION "${PYTHONLIBS_VERSION_STRING}")
  list(GET _PYTHONLIBS_VERSION 0 _PYTHONLIBS_VERSION_MAJOR)
  list(GET _PYTHONLIBS_VERSION 1 _PYTHONLIBS_VERSION_MINOR)
  list(APPEND _Python_VERSIONS ${_PYTHONLIBS_VERSION_MAJOR}.${_PYTHONLIBS_VERSION_MINOR})
endif()
# Search for the current active python version first
list(APPEND _Python_VERSIONS ";")
list(APPEND _Python_VERSIONS ${_PYTHON_FIND_OTHER_VERSIONS})

unset(_PYTHON_FIND_OTHER_VERSIONS)
unset(_PYTHON1_VERSIONS)
unset(_PYTHON2_VERSIONS)
unset(_PYTHON3_VERSIONS)

# Search for newest python version if python executable isn't found
if(NOT PYTHON_EXECUTABLE)
    foreach(_CURRENT_VERSION IN LISTS _Python_VERSIONS)
      set(_Python_NAMES python${_CURRENT_VERSION})
      if(CMAKE_HOST_WIN32)
        list(APPEND _Python_NAMES python)
      endif()
      find_program(PYTHON_EXECUTABLE
        NAMES ${_Python_NAMES}
        PATHS
            [HKEY_LOCAL_MACHINE\\SOFTWARE\\Python\\PythonCore\\${_CURRENT_VERSION}\\InstallPath]
            [HKEY_LOCAL_MACHINE\\SOFTWARE\\Python\\PythonCore\\${_CURRENT_VERSION}-32\\InstallPath]
            [HKEY_LOCAL_MACHINE\\SOFTWARE\\Python\\PythonCore\\${_CURRENT_VERSION}-64\\InstallPath]
            [HKEY_CURRENT_USER\\SOFTWARE\\Python\\PythonCore\\${_CURRENT_VERSION}\\InstallPath]
            [HKEY_CURRENT_USER\\SOFTWARE\\Python\\PythonCore\\${_CURRENT_VERSION}-32\\InstallPath]
            [HKEY_CURRENT_USER\\SOFTWARE\\Python\\PythonCore\\${_CURRENT_VERSION}-64\\InstallPath]
        )
    endforeach()
endif()

# determine python version string
if(PYTHON_EXECUTABLE)
    execute_process(COMMAND "${PYTHON_EXECUTABLE}" -c
                            "import sys; sys.stdout.write(';'.join([str(x) for x in sys.version_info[:3]]))"
                    OUTPUT_VARIABLE _VERSION
                    RESULT_VARIABLE _PYTHON_VERSION_RESULT
                    ERROR_QUIET)
    if(NOT _PYTHON_VERSION_RESULT)
        string(REPLACE ";" "." PYTHON_VERSION_STRING "${_VERSION}")
        list(GET _VERSION 0 PYTHON_VERSION_MAJOR)
        list(GET _VERSION 1 PYTHON_VERSION_MINOR)
        list(GET _VERSION 2 PYTHON_VERSION_PATCH)
        if(PYTHON_VERSION_PATCH EQUAL 0)
            # it's called "Python 2.7", not "2.7.0"
            string(REGEX REPLACE "\\.0$" "" PYTHON_VERSION_STRING "${PYTHON_VERSION_STRING}")
        endif()
    else()
        # sys.version predates sys.version_info, so use that
        # sys.version was first documented for Python 1.5, so assume version 1.4
        # if retrieving sys.version fails.
        execute_process(COMMAND "${PYTHON_EXECUTABLE}" -c "try: import sys; sys.stdout.write(sys.version)\nexcept: sys.stdout.write(\"1.4.0\")"
                        OUTPUT_VARIABLE _VERSION
                        RESULT_VARIABLE _PYTHON_VERSION_RESULT
                        ERROR_QUIET)
        if(NOT _PYTHON_VERSION_RESULT)
            string(REGEX REPLACE " .*" "" PYTHON_VERSION_STRING "${_VERSION}")
            string(REGEX REPLACE "^([0-9]+)\\.[0-9]+.*" "\\1" PYTHON_VERSION_MAJOR "${PYTHON_VERSION_STRING}")
            string(REGEX REPLACE "^[0-9]+\\.([0-9])+.*" "\\1" PYTHON_VERSION_MINOR "${PYTHON_VERSION_STRING}")
            if(PYTHON_VERSION_STRING MATCHES "^[0-9]+\\.[0-9]+\\.([0-9]+)")
                set(PYTHON_VERSION_PATCH "${CMAKE_MATCH_1}")
            else()
                set(PYTHON_VERSION_PATCH "0")
            endif()
        else()
            unset(PYTHON_VERSION_STRING)
            unset(PYTHON_VERSION_MAJOR)
            unset(PYTHON_VERSION_MINOR)
            unset(PYTHON_VERSION_PATCH)
        endif()
    endif()
    unset(_PYTHON_VERSION_RESULT)
    unset(_VERSION)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PythonInterp REQUIRED_VARS PYTHON_EXECUTABLE VERSION_VAR PYTHON_VERSION_STRING)

mark_as_advanced(PYTHON_EXECUTABLE)
