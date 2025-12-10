# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindPythonLibs
--------------

.. versionchanged:: 3.27
  This module is available only if policy :policy:`CMP0148` is not set to ``NEW``.

.. deprecated:: 3.12

  Use :module:`FindPython3`, :module:`FindPython2`, or :module:`FindPython`
  instead.

Finds the Python installation and determines the location of its include
directories and libraries, as well as the name of the Python library to
link against:

.. code-block:: cmake

  find_package(PythonLibs [<version>] [...])

.. note::

  When using both this and the :module:`FindPythonInterp` module, call
  ``find_package(PythonInterp)`` before ``find_package(PythonLibs)``.  This
  ensures that the detected interpreter version is used to guide the selection
  of compatible libraries, resulting in a consistent ``PYTHON_LIBRARIES`` value.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``PythonLibs_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether the (requested version of) Python libraries
  were found.

``PYTHONLIBS_VERSION_STRING``
  The version of the Python libraries found.

``PYTHON_LIBRARIES``
  Libraries needed to link against to use Python.

``PYTHON_INCLUDE_DIRS``
  Include directories needed to use Python.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set to specify the Python installation
to use:

``PYTHON_LIBRARY``
  The path to the Python library.

``PYTHON_INCLUDE_DIR``
  The directory containing the ``Python.h`` header file.

Hints
^^^^^

This module accepts the following variables before calling
``find_package(PythonLibs)``:

``Python_ADDITIONAL_VERSIONS``
  This variable can be used to specify a list of version numbers that should be
  taken into account when searching for Python.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``PYTHONLIBS_FOUND``
  .. deprecated:: 3.12
    Use ``PythonLibs_FOUND``, which has the same value.

  Boolean indicating whether the (requested version of) Python libraries
  were found.

``PYTHON_DEBUG_LIBRARIES``
  .. deprecated:: 2.8.8
    Use ``PYTHON_LIBRARIES`` instead.

  Result variable that holds the path to the debug library.

``PYTHON_INCLUDE_PATH``
  .. deprecated:: 2.8.0
    Use ``PYTHON_INCLUDE_DIR`` or ``PYTHON_INCLUDE_DIRS`` instead.

  Result variable that holds the path to the directory containing the
  ``Python.h`` header file.

Examples
^^^^^^^^

In earlier versions of CMake, Python libraries were found and used in a project
like this:

.. code-block:: cmake

  find_package(PythonLibs)
  target_link_libraries(app PRIVATE ${PYTHON_LIBRARIES})
  target_include_directories(app PRIVATE ${PYTHON_INCLUDE_DIRS})

Starting with CMake 3.12, Python libraries can be found using the
:module:`FindPython` module.  The equivalent example using the modern approach
with an imported target is:

.. code-block:: cmake

  find_package(Python COMPONENTS Development)
  target_link_libraries(app PRIVATE Python::Python)
#]=======================================================================]

cmake_policy(PUSH)
cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>

cmake_policy(GET CMP0148 _FindPythonLibs_CMP0148)
if(_FindPythonLibs_CMP0148 STREQUAL "NEW")
  message(FATAL_ERROR "The FindPythonLibs module has been removed by policy CMP0148.")
endif()

if(_FindPythonLibs_testing)
  set(_FindPythonLibs_included TRUE)
  cmake_policy(POP)
  return()
endif()

# Use the executable's path as a hint
set(_Python_LIBRARY_PATH_HINT)
if(IS_ABSOLUTE "${PYTHON_EXECUTABLE}")
  if(WIN32)
    get_filename_component(_Python_PREFIX "${PYTHON_EXECUTABLE}" PATH)
    if(_Python_PREFIX)
      set(_Python_LIBRARY_PATH_HINT ${_Python_PREFIX}/libs)
    endif()
    unset(_Python_PREFIX)
  else()
    get_filename_component(_Python_PREFIX "${PYTHON_EXECUTABLE}" PATH)
    get_filename_component(_Python_PREFIX "${_Python_PREFIX}" PATH)
    if(_Python_PREFIX)
      set(_Python_LIBRARY_PATH_HINT ${_Python_PREFIX}/lib)
    endif()
    unset(_Python_PREFIX)
  endif()
endif()

block(SCOPE_FOR POLICIES)
  cmake_policy(SET CMP0173 OLD)
  include(${CMAKE_CURRENT_LIST_DIR}/CMakeFindFrameworks.cmake)
endblock()
# Search for the python framework on Apple.
CMAKE_FIND_FRAMEWORKS(Python)

# Save CMAKE_FIND_FRAMEWORK
if(DEFINED CMAKE_FIND_FRAMEWORK)
  set(_PythonLibs_CMAKE_FIND_FRAMEWORK ${CMAKE_FIND_FRAMEWORK})
else()
  unset(_PythonLibs_CMAKE_FIND_FRAMEWORK)
endif()
# To avoid picking up the system Python.h pre-maturely.
set(CMAKE_FIND_FRAMEWORK LAST)

set(_PYTHON1_VERSIONS 1.6 1.5)
set(_PYTHON2_VERSIONS 2.7 2.6 2.5 2.4 2.3 2.2 2.1 2.0)
set(_PYTHON3_VERSIONS 3.15 3.14 3.13 3.12 3.11 3.10 3.9 3.8 3.7 3.6 3.5 3.4 3.3 3.2 3.1 3.0)

if(PythonLibs_FIND_VERSION)
    if(PythonLibs_FIND_VERSION_COUNT GREATER 1)
        set(_PYTHON_FIND_MAJ_MIN "${PythonLibs_FIND_VERSION_MAJOR}.${PythonLibs_FIND_VERSION_MINOR}")
        unset(_PYTHON_FIND_OTHER_VERSIONS)
        if(PythonLibs_FIND_VERSION_EXACT)
            if(_PYTHON_FIND_MAJ_MIN STREQUAL PythonLibs_FIND_VERSION)
                set(_PYTHON_FIND_OTHER_VERSIONS "${PythonLibs_FIND_VERSION}")
            else()
                set(_PYTHON_FIND_OTHER_VERSIONS "${PythonLibs_FIND_VERSION}" "${_PYTHON_FIND_MAJ_MIN}")
            endif()
        else()
            foreach(_PYTHON_V ${_PYTHON${PythonLibs_FIND_VERSION_MAJOR}_VERSIONS})
                if(NOT _PYTHON_V VERSION_LESS _PYTHON_FIND_MAJ_MIN)
                    list(APPEND _PYTHON_FIND_OTHER_VERSIONS ${_PYTHON_V})
                endif()
             endforeach()
        endif()
        unset(_PYTHON_FIND_MAJ_MIN)
    else()
        set(_PYTHON_FIND_OTHER_VERSIONS ${_PYTHON${PythonLibs_FIND_VERSION_MAJOR}_VERSIONS})
    endif()
else()
    set(_PYTHON_FIND_OTHER_VERSIONS ${_PYTHON3_VERSIONS} ${_PYTHON2_VERSIONS} ${_PYTHON1_VERSIONS})
endif()

# Set up the versions we know about, in the order we will search. Always add
# the user supplied additional versions to the front.
# If FindPythonInterp has already found the major and minor version,
# insert that version between the user supplied versions and the stock
# version list.
set(_Python_VERSIONS ${Python_ADDITIONAL_VERSIONS})
if(DEFINED PYTHON_VERSION_MAJOR AND DEFINED PYTHON_VERSION_MINOR)
  list(APPEND _Python_VERSIONS ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR})
endif()
list(APPEND _Python_VERSIONS ${_PYTHON_FIND_OTHER_VERSIONS})

unset(_PYTHON_FIND_OTHER_VERSIONS)
unset(_PYTHON1_VERSIONS)
unset(_PYTHON2_VERSIONS)
unset(_PYTHON3_VERSIONS)

# Python distribution: define which architectures can be used
if (CMAKE_SIZEOF_VOID_P)
  # In this case, search only for 64bit or 32bit
  math (EXPR _PYTHON_ARCH "${CMAKE_SIZEOF_VOID_P} * 8")
  set (_PYTHON_ARCH2 _PYTHON_PREFIX_ARCH})
else()
  if (PYTHON_EXECUTABLE)
    # determine interpreter architecture
    execute_process (COMMAND "${PYTHON_EXECUTABLE}" -c "import sys; print(sys.maxsize > 2**32)"
                     RESULT_VARIABLE _PYTHON_RESULT
                     OUTPUT_VARIABLE _PYTHON_IS64BIT
                     ERROR_VARIABLE _PYTHON_IS64BIT)
      if (NOT _PYTHON_RESULT)
        if (_PYTHON_IS64BIT)
          set (_PYTHON_ARCH 64)
          set (_PYTHON_ARCH2 64)
        else()
          set (_PYTHON_ARCH 32)
          set (_PYTHON_ARCH2 32)
        endif()
      endif()
  else()
    # architecture unknown, search for both 64bit and 32bit
    set (_PYTHON_ARCH 64)
    set (_PYTHON_ARCH2 32)
  endif()
endif()

foreach(_CURRENT_VERSION ${_Python_VERSIONS})
  string(REPLACE "." "" _CURRENT_VERSION_NO_DOTS ${_CURRENT_VERSION})
  if(WIN32)
    find_library(PYTHON_DEBUG_LIBRARY
      NAMES python${_CURRENT_VERSION_NO_DOTS}_d python
      NAMES_PER_DIR
      HINTS ${_Python_LIBRARY_PATH_HINT}
      PATHS
      [HKEY_LOCAL_MACHINE\\SOFTWARE\\Python\\PythonCore\\${_CURRENT_VERSION}\\InstallPath]/libs/Debug
      [HKEY_LOCAL_MACHINE\\SOFTWARE\\Python\\PythonCore\\${_CURRENT_VERSION}-${_PYTHON_ARCH}\\InstallPath]/libs/Debug
      [HKEY_LOCAL_MACHINE\\SOFTWARE\\Python\\PythonCore\\${_CURRENT_VERSION}-${_PYTHON_ARCH2}\\InstallPath]/libs/Debug
      [HKEY_CURRENT_USER\\SOFTWARE\\Python\\PythonCore\\${_CURRENT_VERSION}\\InstallPath]/libs/Debug
      [HKEY_CURRENT_USER\\SOFTWARE\\Python\\PythonCore\\${_CURRENT_VERSION}-${_PYTHON_ARCH}\\InstallPath]/libs/Debug
      [HKEY_CURRENT_USER\\SOFTWARE\\Python\\PythonCore\\${_CURRENT_VERSION}-${_PYTHON_ARCH2}\\InstallPath]/libs/Debug
      [HKEY_LOCAL_MACHINE\\SOFTWARE\\Python\\PythonCore\\${_CURRENT_VERSION}\\InstallPath]/libs
      [HKEY_LOCAL_MACHINE\\SOFTWARE\\Python\\PythonCore\\${_CURRENT_VERSION}-${_PYTHON_ARCH}\\InstallPath]/libs
      [HKEY_LOCAL_MACHINE\\SOFTWARE\\Python\\PythonCore\\${_CURRENT_VERSION}-${_PYTHON_ARCH2}\\InstallPath]/libs
      [HKEY_CURRENT_USER\\SOFTWARE\\Python\\PythonCore\\${_CURRENT_VERSION}\\InstallPath]/libs
      [HKEY_CURRENT_USER\\SOFTWARE\\Python\\PythonCore\\${_CURRENT_VERSION}-${_PYTHON_ARCH}\\InstallPath]/libs
      [HKEY_CURRENT_USER\\SOFTWARE\\Python\\PythonCore\\${_CURRENT_VERSION}-${_PYTHON_ARCH2}\\InstallPath]/libs
      )
  endif()

  set(PYTHON_FRAMEWORK_LIBRARIES)
  if(Python_FRAMEWORKS AND NOT PYTHON_LIBRARY)
    foreach(dir ${Python_FRAMEWORKS})
      list(APPEND PYTHON_FRAMEWORK_LIBRARIES
           ${dir}/Versions/${_CURRENT_VERSION}/lib)
    endforeach()
  endif()
  find_library(PYTHON_LIBRARY
    NAMES
      python${_CURRENT_VERSION_NO_DOTS}
      python${_CURRENT_VERSION}mu
      python${_CURRENT_VERSION}m
      python${_CURRENT_VERSION}u
      python${_CURRENT_VERSION}
    NAMES_PER_DIR
    HINTS
      ${_Python_LIBRARY_PATH_HINT}
    PATHS
      ${PYTHON_FRAMEWORK_LIBRARIES}
      [HKEY_LOCAL_MACHINE\\SOFTWARE\\Python\\PythonCore\\${_CURRENT_VERSION}\\InstallPath]/libs
      [HKEY_LOCAL_MACHINE\\SOFTWARE\\Python\\PythonCore\\${_CURRENT_VERSION}-${_PYTHON_ARCH}\\InstallPath]/libs
      [HKEY_LOCAL_MACHINE\\SOFTWARE\\Python\\PythonCore\\${_CURRENT_VERSION}-${_PYTHON_ARCH2}\\InstallPath]/libs
      [HKEY_CURRENT_USER\\SOFTWARE\\Python\\PythonCore\\${_CURRENT_VERSION}\\InstallPath]/libs
      [HKEY_CURRENT_USER\\SOFTWARE\\Python\\PythonCore\\${_CURRENT_VERSION}-${_PYTHON_ARCH}\\InstallPath]/libs
      [HKEY_CURRENT_USER\\SOFTWARE\\Python\\PythonCore\\${_CURRENT_VERSION}-${_PYTHON_ARCH2}\\InstallPath]/libs
  )
  # Look for the static library in the Python config directory
  find_library(PYTHON_LIBRARY
    NAMES python${_CURRENT_VERSION_NO_DOTS} python${_CURRENT_VERSION}
    NAMES_PER_DIR
    # This is where the static library is usually located
    PATH_SUFFIXES python${_CURRENT_VERSION}/config
  )

  # Don't search for include dir until library location is known
  if(PYTHON_LIBRARY)

    # Use the library's install prefix as a hint
    set(_Python_INCLUDE_PATH_HINT)
    # PYTHON_LIBRARY may contain a list because of SelectLibraryConfigurations
    # which may have been run previously. If it is the case, the list can be:
    #   optimized;<FILEPATH_TO_RELEASE_LIBRARY>;debug;<FILEPATH_TO_DEBUG_LIBRARY>
    foreach(lib ${PYTHON_LIBRARY} ${PYTHON_DEBUG_LIBRARY})
      if(IS_ABSOLUTE "${lib}")
        get_filename_component(_Python_PREFIX "${lib}" PATH)
        get_filename_component(_Python_PREFIX "${_Python_PREFIX}" PATH)
        if(_Python_PREFIX)
          list(APPEND _Python_INCLUDE_PATH_HINT ${_Python_PREFIX}/include)
        endif()
        unset(_Python_PREFIX)
      endif()
    endforeach()

    # Add framework directories to the search paths
    set(PYTHON_FRAMEWORK_INCLUDES)
    if(Python_FRAMEWORKS AND NOT PYTHON_INCLUDE_DIR)
      foreach(dir ${Python_FRAMEWORKS})
        list(APPEND PYTHON_FRAMEWORK_INCLUDES
          ${dir}/Versions/${_CURRENT_VERSION}/include)
      endforeach()
    endif()

    find_path(PYTHON_INCLUDE_DIR
      NAMES Python.h
      HINTS
        ${_Python_INCLUDE_PATH_HINT}
      PATHS
        ${PYTHON_FRAMEWORK_INCLUDES}
        [HKEY_LOCAL_MACHINE\\SOFTWARE\\Python\\PythonCore\\${_CURRENT_VERSION}\\InstallPath]/include
        [HKEY_LOCAL_MACHINE\\SOFTWARE\\Python\\PythonCore\\${_CURRENT_VERSION}-${_PYTHON_ARCH}\\InstallPath]/include
        [HKEY_LOCAL_MACHINE\\SOFTWARE\\Python\\PythonCore\\${_CURRENT_VERSION}-${_PYTHON_ARCH2}\\InstallPath]/include
        [HKEY_CURRENT_USER\\SOFTWARE\\Python\\PythonCore\\${_CURRENT_VERSION}\\InstallPath]/include
        [HKEY_CURRENT_USER\\SOFTWARE\\Python\\PythonCore\\${_CURRENT_VERSION}-${_PYTHON_ARCH}\\InstallPath]/include
        [HKEY_CURRENT_USER\\SOFTWARE\\Python\\PythonCore\\${_CURRENT_VERSION}-${_PYTHON_ARCH2}\\InstallPath]/include
      PATH_SUFFIXES
        python${_CURRENT_VERSION}mu
        python${_CURRENT_VERSION}m
        python${_CURRENT_VERSION}u
        python${_CURRENT_VERSION}
    )
  endif()

  # For backward compatibility, set PYTHON_INCLUDE_PATH.
  set(PYTHON_INCLUDE_PATH "${PYTHON_INCLUDE_DIR}")

  if(PYTHON_INCLUDE_DIR AND EXISTS "${PYTHON_INCLUDE_DIR}/patchlevel.h")
    file(STRINGS "${PYTHON_INCLUDE_DIR}/patchlevel.h" python_version_str
         REGEX "^#define[ \t]+PY_VERSION[ \t]+\"[^\"]+\"")
    string(REGEX REPLACE "^#define[ \t]+PY_VERSION[ \t]+\"([^\"]+)\".*" "\\1"
                         PYTHONLIBS_VERSION_STRING "${python_version_str}")
    unset(python_version_str)
  endif()

  if(PYTHON_LIBRARY AND PYTHON_INCLUDE_DIR)
    break()
  endif()
endforeach()

unset(_Python_INCLUDE_PATH_HINT)
unset(_Python_LIBRARY_PATH_HINT)

mark_as_advanced(
  PYTHON_DEBUG_LIBRARY
  PYTHON_LIBRARY
  PYTHON_INCLUDE_DIR
)

# We use PYTHON_INCLUDE_DIR, PYTHON_LIBRARY and PYTHON_DEBUG_LIBRARY for the
# cache entries because they are meant to specify the location of a single
# library. We now set the variables listed by the documentation for this
# module.
set(PYTHON_INCLUDE_DIRS "${PYTHON_INCLUDE_DIR}")
set(PYTHON_DEBUG_LIBRARIES "${PYTHON_DEBUG_LIBRARY}")

# These variables have been historically named in this module different from
# what select_library_configurations() expects.
set(PYTHON_LIBRARY_DEBUG "${PYTHON_DEBUG_LIBRARY}")
set(PYTHON_LIBRARY_RELEASE "${PYTHON_LIBRARY}")
include(${CMAKE_CURRENT_LIST_DIR}/SelectLibraryConfigurations.cmake)
select_library_configurations(PYTHON)
# select_library_configurations() sets ${PREFIX}_FOUND if it has a library.
# Unset this, this prefix doesn't match the module prefix, they are different
# for historical reasons.
unset(PYTHON_FOUND)

# Restore CMAKE_FIND_FRAMEWORK
if(DEFINED _PythonLibs_CMAKE_FIND_FRAMEWORK)
  set(CMAKE_FIND_FRAMEWORK ${_PythonLibs_CMAKE_FIND_FRAMEWORK})
  unset(_PythonLibs_CMAKE_FIND_FRAMEWORK)
else()
  unset(CMAKE_FIND_FRAMEWORK)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PythonLibs
                                  REQUIRED_VARS PYTHON_LIBRARIES PYTHON_INCLUDE_DIRS
                                  VERSION_VAR PYTHONLIBS_VERSION_STRING)

# PYTHON_ADD_MODULE(<name> src1 src2 ... srcN) is used to build modules for python.
# PYTHON_WRITE_MODULES_HEADER(<filename>) writes a header file you can include
# in your sources to initialize the static python modules
function(PYTHON_ADD_MODULE _NAME )
  get_property(_TARGET_SUPPORTS_SHARED_LIBS
    GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS)
  option(PYTHON_ENABLE_MODULE_${_NAME} "Add module ${_NAME}" TRUE)
  option(PYTHON_MODULE_${_NAME}_BUILD_SHARED
    "Add module ${_NAME} shared" ${_TARGET_SUPPORTS_SHARED_LIBS})

  # Mark these options as advanced
  mark_as_advanced(PYTHON_ENABLE_MODULE_${_NAME}
    PYTHON_MODULE_${_NAME}_BUILD_SHARED)

  if(PYTHON_ENABLE_MODULE_${_NAME})
    if(PYTHON_MODULE_${_NAME}_BUILD_SHARED)
      set(PY_MODULE_TYPE MODULE)
    else()
      set(PY_MODULE_TYPE STATIC)
      set_property(GLOBAL  APPEND  PROPERTY  PY_STATIC_MODULES_LIST ${_NAME})
    endif()

    set_property(GLOBAL  APPEND  PROPERTY  PY_MODULES_LIST ${_NAME})
    add_library(${_NAME} ${PY_MODULE_TYPE} ${ARGN})
#    target_link_libraries(${_NAME} ${PYTHON_LIBRARIES})

    if(PYTHON_MODULE_${_NAME}_BUILD_SHARED)
      set_target_properties(${_NAME} PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}")
      if(WIN32 AND NOT CYGWIN)
        set_target_properties(${_NAME} PROPERTIES SUFFIX ".pyd")
      endif()
    endif()

  endif()
endfunction()

function(PYTHON_WRITE_MODULES_HEADER _filename)

  get_property(PY_STATIC_MODULES_LIST  GLOBAL  PROPERTY PY_STATIC_MODULES_LIST)

  get_filename_component(_name "${_filename}" NAME)
  string(REPLACE "." "_" _name "${_name}")
  string(TOUPPER ${_name} _nameUpper)
  set(_filename ${CMAKE_CURRENT_BINARY_DIR}/${_filename})

  set(_filenameTmp "${_filename}.in")
  file(WRITE ${_filenameTmp} "/*Created by cmake, do not edit, changes will be lost*/\n")
  file(APPEND ${_filenameTmp}
"#ifndef ${_nameUpper}
#define ${_nameUpper}

#include <Python.h>

#ifdef __cplusplus
extern \"C\" {
#endif /* __cplusplus */

")

  foreach(_currentModule ${PY_STATIC_MODULES_LIST})
    file(APPEND ${_filenameTmp} "extern void init${PYTHON_MODULE_PREFIX}${_currentModule}(void);\n\n")
  endforeach()

  file(APPEND ${_filenameTmp}
"#ifdef __cplusplus
}
#endif /* __cplusplus */

")


  foreach(_currentModule ${PY_STATIC_MODULES_LIST})
    file(APPEND ${_filenameTmp} "int ${_name}_${_currentModule}(void) \n{\n  static char name[]=\"${PYTHON_MODULE_PREFIX}${_currentModule}\"; return PyImport_AppendInittab(name, init${PYTHON_MODULE_PREFIX}${_currentModule});\n}\n\n")
  endforeach()

  file(APPEND ${_filenameTmp} "void ${_name}_LoadAllPythonModules(void)\n{\n")
  foreach(_currentModule ${PY_STATIC_MODULES_LIST})
    file(APPEND ${_filenameTmp} "  ${_name}_${_currentModule}();\n")
  endforeach()
  file(APPEND ${_filenameTmp} "}\n\n")
  file(APPEND ${_filenameTmp} "#ifndef EXCLUDE_LOAD_ALL_FUNCTION\nvoid CMakeLoadAllPythonModules(void)\n{\n  ${_name}_LoadAllPythonModules();\n}\n#endif\n\n#endif\n")

# with configure_file() cmake complains that you may not use a file created using file(WRITE) as input file for configure_file()
  execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_filenameTmp}" "${_filename}" OUTPUT_QUIET ERROR_QUIET)

endfunction()

cmake_policy(POP)
