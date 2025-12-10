# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindFLTK
--------

Finds the Fast Light Toolkit (FLTK), a cross-platform toolkit for GUI
development:

.. code-block:: cmake

  find_package(FLTK [...])

FLTK uses CMake-based build system and provides a package configuration file for
projects to find it.  As of its 1.4.0 version it also provides
:ref:`Imported Targets` that encapsulate usage requirements.  For example,
``fltk::fltk``, which can be linked to project targets where FLTK is needed.
This module takes that into account and first attempts to find FLTK in
*config mode*.  If the configuration file is not available, it falls back to
*module mode* and searches standard locations. See also to the official FLTK
documentation for more information, how to use FLTK with CMake.

.. versionadded:: 3.11
  Debug and release (optimized) library variants are found separately and use
  :ref:`per-configuration <Build Configurations>` variables.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``FLTK_FOUND``
  Boolean indicating whether FLTK was found.

``FLTK_LIBRARIES``
  Libraries needed to link against to use FLTK.

``FLTK_WRAP_UI``
  Boolean indicating whether the ``fluid`` executable is found.  This variable
  is available only if FLTK is found in *module mode* and can be used, for
  example, to conditionally invoke the :command:`fltk_wrap_ui` command if it is
  needed and available.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables are also available to set or use:

``FLTK_FLUID_EXECUTABLE``
  The path to the ``fluid`` binary tool.

``FLTK_INCLUDE_DIR``
  The include directory containing header files needed to use FLTK.

``FLTK_BASE_LIBRARY_RELEASE``
  .. versionadded:: 3.11

  The path to the release (optimized) FLTK base library.

``FLTK_BASE_LIBRARY_DEBUG``
  .. versionadded:: 3.11

  The path to the debug FLTK base library.

``FLTK_GL_LIBRARY_RELEASE``
  .. versionadded:: 3.11

  The path to the release (optimized) FLTK GL library.

``FLTK_GL_LIBRARY_DEBUG``
  .. versionadded:: 3.11

  The path to the debug FLTK GL library.

``FLTK_FORMS_LIBRARY_RELEASE``
  .. versionadded:: 3.11

  The path to the release (optimized) FLTK Forms library.

``FLTK_FORMS_LIBRARY_DEBUG``
  .. versionadded:: 3.11

  The path to the debug FLTK Forms library.

``FLTK_IMAGES_LIBRARY_RELEASE``
  .. versionadded:: 3.11

  The path to the release (optimized) FLTK Images protobuf library.

``FLTK_IMAGES_LIBRARY_DEBUG``
  .. versionadded:: 3.11

  The path to the debug FLTK Images library.

Input Variables
^^^^^^^^^^^^^^^

By default, this module searches for all FLTK libraries and its ``fluid``
executable.  The following variables can be set before calling
``find_package(FLTK)`` to indicate which elements are optional for a successful
configuration:

``FLTK_SKIP_FLUID``
  Set to boolean true to mark the ``fluid`` executable as optional.

``FLTK_SKIP_FORMS``
  Set to boolean true to mark the FLTK Forms library as optional; it will
  therefore not be included in the ``FLTK_LIBRARIES`` result variable.

``FLTK_SKIP_IMAGES``
  Set to boolean true to mark the FLTK Image library as optional; it will
  therefore not be included in the ``FLTK_LIBRARIES`` result variable.

``FLTK_SKIP_OPENGL``
  Set to boolean true to mark the FLTK OpenGL library as optional; it will
  therefore not be included in the ``FLTK_LIBRARIES`` result variable.

Examples
^^^^^^^^

Finding FLTK and conditionally creating a ``fltk::fltk`` imported interface
target, if it is not provided by the upstream FLTK package.  Imported target can
then be linked to a project target:

.. code-block:: cmake

  find_package(FLTK)

  if(FLTK_FOUND AND NOT TARGET fltk::fltk)
    add_library(fltk::fltk INTERFACE IMPORTED)
    set_target_properties(
      fltk::fltk
      PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${FLTK_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${FLTK_LIBRARIES}"
    )
  endif()

  target_link_libraries(project_target PRIVATE fltk::fltk)
#]=======================================================================]

if(NOT FLTK_SKIP_OPENGL)
  find_package(OpenGL)
endif()

#  Platform dependent libraries required by FLTK
if(WIN32)
  if(NOT CYGWIN)
    if(BORLAND)
      set( FLTK_PLATFORM_DEPENDENT_LIBS import32 )
    else()
      set( FLTK_PLATFORM_DEPENDENT_LIBS wsock32 comctl32 )
    endif()
  endif()
endif()

if(UNIX)
  include(${CMAKE_CURRENT_LIST_DIR}/FindX11.cmake)
  find_library(FLTK_MATH_LIBRARY m)
  set( FLTK_PLATFORM_DEPENDENT_LIBS ${X11_LIBRARIES} ${FLTK_MATH_LIBRARY})
endif()

if(APPLE)
  set( FLTK_PLATFORM_DEPENDENT_LIBS  "-framework Carbon -framework Cocoa -framework ApplicationServices -lz")
endif()

# If FLTK_INCLUDE_DIR is already defined we assign its value to FLTK_DIR.
if(FLTK_INCLUDE_DIR)
  set(FLTK_DIR ${FLTK_INCLUDE_DIR})
endif()


# If FLTK has been built using CMake we try to find everything directly
set(FLTK_DIR_STRING "directory containing FLTKConfig.cmake.  This is either the root of the build tree, or PREFIX/lib/fltk for an installation.")

# Search only if the location is not already known.
if(NOT FLTK_DIR)
  # Get the system search path as a list.
  file(TO_CMAKE_PATH "$ENV{PATH}" FLTK_DIR_SEARCH2)

  # Construct a set of paths relative to the system search path.
  set(FLTK_DIR_SEARCH "")
  foreach(dir ${FLTK_DIR_SEARCH2})
    set(FLTK_DIR_SEARCH ${FLTK_DIR_SEARCH} "${dir}/../lib/fltk")
  endforeach()
  string(REPLACE "//" "/" FLTK_DIR_SEARCH "${FLTK_DIR_SEARCH}")

  #
  # Look for an installation or build tree.
  #
  find_path(FLTK_DIR FLTKConfig.cmake
    # Look for an environment variable FLTK_DIR.
    HINTS
      ENV FLTK_DIR

    # Look in places relative to the system executable search path.
    ${FLTK_DIR_SEARCH}

    PATH_SUFFIXES
    fltk
    fltk/include
    lib/fltk
    lib/fltk/include

    # Help the user find it if we cannot.
    DOC "The ${FLTK_DIR_STRING}"
    )
endif()

# Check if FLTK was built using CMake
foreach(fltk_include IN LISTS FLTK_DIR)
  if(EXISTS "${fltk_include}/FLTKConfig.cmake")
    set(FLTK_BUILT_WITH_CMAKE 1)
    set(FLTK_CMAKE_PATH "${fltk_include}/FLTKConfig.cmake")
    break()
  endif()
endforeach()

if(FLTK_BUILT_WITH_CMAKE)
  set(FLTK_FOUND 1)
  include("${FLTK_CMAKE_PATH}")

  # Fluid
  if(FLUID_COMMAND)
    set(FLTK_FLUID_EXECUTABLE ${FLUID_COMMAND} CACHE FILEPATH "Fluid executable")
  else()
    find_program(FLTK_FLUID_EXECUTABLE fluid PATHS
      ${FLTK_EXECUTABLE_DIRS}
      ${FLTK_EXECUTABLE_DIRS}/RelWithDebInfo
      ${FLTK_EXECUTABLE_DIRS}/Debug
      ${FLTK_EXECUTABLE_DIRS}/Release
      NO_SYSTEM_PATH)
  endif()
  # mark_as_advanced(FLTK_FLUID_EXECUTABLE)

  set(FLTK_INCLUDE_DIR ${FLTK_DIR})
  link_directories(${FLTK_LIBRARY_DIRS})

  set(FLTK_BASE_LIBRARY fltk)
  set(FLTK_GL_LIBRARY fltk_gl)
  set(FLTK_FORMS_LIBRARY fltk_forms)
  set(FLTK_IMAGES_LIBRARY fltk_images)

  # Add the extra libraries
  load_cache(${FLTK_DIR}
    READ_WITH_PREFIX FL
    FLTK_USE_SYSTEM_JPEG
    FLTK_USE_SYSTEM_PNG
    FLTK_USE_SYSTEM_ZLIB
    )

  set(FLTK_IMAGES_LIBS "")
  if(FLFLTK_USE_SYSTEM_JPEG)
    set(FLTK_IMAGES_LIBS ${FLTK_IMAGES_LIBS} fltk_jpeg)
  endif()
  if(FLFLTK_USE_SYSTEM_PNG)
    set(FLTK_IMAGES_LIBS ${FLTK_IMAGES_LIBS} fltk_png)
  endif()
  if(FLFLTK_USE_SYSTEM_ZLIB)
    set(FLTK_IMAGES_LIBS ${FLTK_IMAGES_LIBS} fltk_zlib)
  endif()
  set(FLTK_IMAGES_LIBS "${FLTK_IMAGES_LIBS}" CACHE INTERNAL
    "Extra libraries for fltk_images library.")

else()

  # if FLTK was not built using CMake
  # Find fluid executable.
  find_program(FLTK_FLUID_EXECUTABLE fluid ${FLTK_INCLUDE_DIR}/fluid)

  # Use location of fluid to help find everything else.
  set(FLTK_INCLUDE_SEARCH_PATH "")
  set(FLTK_LIBRARY_SEARCH_PATH "")
  if(FLTK_FLUID_EXECUTABLE)
    get_filename_component(FLTK_BIN_DIR "${FLTK_FLUID_EXECUTABLE}" PATH)
    set(FLTK_INCLUDE_SEARCH_PATH ${FLTK_INCLUDE_SEARCH_PATH}
      ${FLTK_BIN_DIR}/../include ${FLTK_BIN_DIR}/..)
    set(FLTK_LIBRARY_SEARCH_PATH ${FLTK_LIBRARY_SEARCH_PATH}
      ${FLTK_BIN_DIR}/../lib)
    set(FLTK_WRAP_UI 1)
  endif()

  #
  # Try to find FLTK include dir using fltk-config
  #
  if(UNIX)
    # Use fltk-config to generate a list of possible include directories
    find_program(FLTK_CONFIG_SCRIPT fltk-config PATHS ${FLTK_BIN_DIR})
    if(FLTK_CONFIG_SCRIPT)
      if(NOT FLTK_INCLUDE_DIR)
        execute_process(COMMAND ${FLTK_CONFIG_SCRIPT} --cxxflags OUTPUT_VARIABLE FLTK_CXXFLAGS)
        if(FLTK_CXXFLAGS)
          string(REGEX MATCHALL "-I[^ ]*" _fltk_temp_dirs ${FLTK_CXXFLAGS})
          string(REPLACE "-I" "" _fltk_temp_dirs "${_fltk_temp_dirs}")
          foreach(_dir ${_fltk_temp_dirs})
            string(STRIP ${_dir} _output)
            list(APPEND _FLTK_POSSIBLE_INCLUDE_DIRS ${_output})
          endforeach()
        endif()
      endif()
    endif()
  endif()

  list(APPEND FLTK_INCLUDE_SEARCH_PATH ${_FLTK_POSSIBLE_INCLUDE_DIRS})

  find_path(FLTK_INCLUDE_DIR
      NAMES FL/Fl.h FL/Fl.H    # fltk 1.1.9 has Fl.H (#8376)
      PATH_SUFFIXES fltk fltk/include
      PATHS ${FLTK_INCLUDE_SEARCH_PATH})

  #
  # Try to find FLTK library
  if(UNIX)
    if(FLTK_CONFIG_SCRIPT)
      execute_process(COMMAND ${FLTK_CONFIG_SCRIPT} --libs OUTPUT_VARIABLE _FLTK_POSSIBLE_LIBS)
      if(_FLTK_POSSIBLE_LIBS)
        get_filename_component(_FLTK_POSSIBLE_LIBRARY_DIR ${_FLTK_POSSIBLE_LIBS} PATH)
      endif()
    endif()
  endif()

  list(APPEND FLTK_LIBRARY_SEARCH_PATH ${FLTK_INCLUDE_DIR}/lib ${_FLTK_POSSIBLE_LIBRARY_DIR})

  include(${CMAKE_CURRENT_LIST_DIR}/SelectLibraryConfigurations.cmake)

  # Allow libraries to be set manually
  if(NOT FLTK_BASE_LIBRARY)
      find_library(FLTK_BASE_LIBRARY_RELEASE NAMES fltk PATHS ${FLTK_LIBRARY_SEARCH_PATH} PATH_SUFFIXES fltk fltk/lib)
      find_library(FLTK_BASE_LIBRARY_DEBUG NAMES fltkd PATHS ${FLTK_LIBRARY_SEARCH_PATH} PATH_SUFFIXES fltk fltk/lib)
      select_library_configurations(FLTK_BASE)
  endif()
  if(NOT FLTK_GL_LIBRARY)
      find_library(FLTK_GL_LIBRARY_RELEASE NAMES fltkgl fltk_gl PATHS ${FLTK_LIBRARY_SEARCH_PATH} PATH_SUFFIXES fltk fltk/lib)
      find_library(FLTK_GL_LIBRARY_DEBUG NAMES fltkgld fltk_gld PATHS ${FLTK_LIBRARY_SEARCH_PATH} PATH_SUFFIXES fltk fltk/lib)
      select_library_configurations(FLTK_GL)
  endif()
  if(NOT FLTK_FORMS_LIBRARY)
      find_library(FLTK_FORMS_LIBRARY_RELEASE NAMES fltkforms fltk_forms PATHS ${FLTK_LIBRARY_SEARCH_PATH} PATH_SUFFIXES fltk fltk/lib)
      find_library(FLTK_FORMS_LIBRARY_DEBUG NAMES fltkformsd fltk_formsd PATHS ${FLTK_LIBRARY_SEARCH_PATH} PATH_SUFFIXES fltk fltk/lib)
      select_library_configurations(FLTK_FORMS)
  endif()
  if(NOT FLTK_IMAGES_LIBRARY)
      find_library(FLTK_IMAGES_LIBRARY_RELEASE NAMES fltkimages fltk_images PATHS ${FLTK_LIBRARY_SEARCH_PATH} PATH_SUFFIXES fltk fltk/lib)
      find_library(FLTK_IMAGES_LIBRARY_DEBUG NAMES fltkimagesd fltk_imagesd PATHS ${FLTK_LIBRARY_SEARCH_PATH} PATH_SUFFIXES fltk fltk/lib)
      select_library_configurations(FLTK_IMAGES)
  endif()

  # Find the extra libraries needed for the fltk_images library.
  if(UNIX)
    if(FLTK_CONFIG_SCRIPT)
      execute_process(COMMAND ${FLTK_CONFIG_SCRIPT} --use-images --ldflags
        OUTPUT_VARIABLE FLTK_IMAGES_LDFLAGS)
      set(FLTK_LIBS_EXTRACT_REGEX ".*-lfltk_images (.*) -lfltk.*")
      if("${FLTK_IMAGES_LDFLAGS}" MATCHES "${FLTK_LIBS_EXTRACT_REGEX}")
        string(REGEX REPLACE " +" ";" FLTK_IMAGES_LIBS "${CMAKE_MATCH_1}")
        # The execute_process() will not be inherited into subdirectories from
        # the file that originally included this module.  Save the answer.
        set(FLTK_IMAGES_LIBS "${FLTK_IMAGES_LIBS}" CACHE INTERNAL
          "Extra libraries for fltk_images library.")
      endif()
    endif()
  endif()

endif()

# Append all of the required libraries together (by default, everything)
set(FLTK_LIBRARIES)
if(NOT FLTK_SKIP_IMAGES)
  list(APPEND FLTK_LIBRARIES ${FLTK_IMAGES_LIBRARY})
endif()
if(NOT FLTK_SKIP_FORMS)
  list(APPEND FLTK_LIBRARIES ${FLTK_FORMS_LIBRARY})
endif()
if(NOT FLTK_SKIP_OPENGL)
  list(APPEND FLTK_LIBRARIES ${FLTK_GL_LIBRARY} ${OPENGL_gl_LIBRARY})
  list(APPEND FLTK_INCLUDE_DIR ${OPENGL_INCLUDE_DIR})
  list(REMOVE_DUPLICATES FLTK_INCLUDE_DIR)
endif()
list(APPEND FLTK_LIBRARIES ${FLTK_BASE_LIBRARY})

include(FindPackageHandleStandardArgs)
if(FLTK_SKIP_FLUID)
  find_package_handle_standard_args(FLTK DEFAULT_MSG FLTK_LIBRARIES FLTK_INCLUDE_DIR)
else()
  find_package_handle_standard_args(FLTK DEFAULT_MSG FLTK_LIBRARIES FLTK_INCLUDE_DIR FLTK_FLUID_EXECUTABLE)
endif()

if(FLTK_FOUND)
  if(APPLE)
    set(FLTK_LIBRARIES ${FLTK_PLATFORM_DEPENDENT_LIBS} ${FLTK_LIBRARIES})
  else()
    set(FLTK_LIBRARIES ${FLTK_LIBRARIES} ${FLTK_PLATFORM_DEPENDENT_LIBS})
  endif()

  # The following deprecated settings are for compatibility with CMake 1.4
  set (HAS_FLTK ${FLTK_FOUND})
  set (FLTK_INCLUDE_PATH ${FLTK_INCLUDE_DIR})
  set (FLTK_FLUID_EXE ${FLTK_FLUID_EXECUTABLE})
  set (FLTK_LIBRARY ${FLTK_LIBRARIES})
endif()
