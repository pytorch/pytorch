# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindImageMagick
---------------

Finds ImageMagick, a software suite for displaying, converting, and manipulating
raster images:

.. code-block:: cmake

  find_package(ImageMagick [<version>] [COMPONENTS <components>...] [...])

.. versionadded:: 3.9
  Support for ImageMagick 7.

Components
^^^^^^^^^^

This module supports components and searches for a set of ImageMagick tools.
Typical components include the names of ImageMagick executables, but are not
limited to the following (future versions of ImageMagick may provide additional
components not listed here):

* ``animate``
* ``compare``
* ``composite``
* ``conjure``
* ``convert``
* ``display``
* ``identify``
* ``import``
* ``mogrify``
* ``montage``
* ``stream``

There are also components for the following ImageMagick APIs:

``Magick++``
  Finds the ImageMagick C++ API.
``MagickWand``
  Finds the ImageMagick MagickWand C API.
``MagickCore``
  Finds the ImageMagick MagickCore low-level C API.

Components can be specified using the :command:`find_package` command:

.. code-block:: cmake

  find_package(ImageMagick [COMPONENTS <components>...])

If no components are specified, the module only searches for the ImageMagick
executable directory.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following :ref:`Imported Targets`:

``ImageMagick::Magick++``
  .. versionadded:: 3.26

  Target encapsulating the ImageMagick C++ API usage requirements, available if
  ImageMagick C++ is found.

``ImageMagick::MagickWand``
  .. versionadded:: 3.26

  Target encapsulating the ImageMagick MagickWand C API usage requirements,
  available if MagickWand is found.

``ImageMagick::MagickCore``
  .. versionadded:: 3.26

  Target encapsulating the ImageMagick MagickCore low-level C API usage
  requirements, available if MagickCore is found.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``ImageMagick_FOUND``
  Boolean indicating whether (the requested version of) ImageMagick and all
  its requested components were found.

``ImageMagick_VERSION``
  .. versionadded:: 4.2

  The version of ImageMagick found in form of
  ``<major>.<minor>.<patch>-<addendum>`` (e.g., ``6.9.12-98``, where ``98``
  is the addendum release number).

  .. note::

    Version detection is available only for ImageMagick 6 and later.

``ImageMagick_INCLUDE_DIRS``
  All include directories needed to use ImageMagick.

``ImageMagick_LIBRARIES``
  Libraries needed to link against to use ImageMagick.

``ImageMagick_COMPILE_OPTIONS``
  .. versionadded:: 3.26

  Compile options of all libraries.

``ImageMagick_<component>_FOUND``
  Boolean indicating whether the ImageMagick ``<component>`` is found.

``ImageMagick_<component>_EXECUTABLE``
  The full path to ``<component>`` executable.

``ImageMagick_<component>_INCLUDE_DIRS``
  Include directories containing headers needed to use the ImageMagick
  ``<component>``.

``ImageMagick_<component>_COMPILE_OPTIONS``
  .. versionadded:: 3.26

  Compile options of the ImageMagick ``<component>``.

``ImageMagick_<component>_LIBRARIES``
  .. versionadded:: 3.31

  Libraries needed to link against to use the ImageMagick ``<component>``.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``ImageMagick_EXECUTABLE_DIR``
  The full path to directory containing ImageMagick executables.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``ImageMagick_VERSION_STRING``
  .. deprecated:: 4.2
    Use ``ImageMagick_VERSION``, which has the same value.

  The version of ImageMagick found.

Examples
^^^^^^^^

Finding ImageMagick with its component ``Magick++``  and linking it to a project
target:

.. code-block:: cmake

  find_package(ImageMagick COMPONENTS Magick++)
  target_link_libraries(example PRIVATE ImageMagick::Magick++)
#]=======================================================================]

cmake_policy(PUSH)
cmake_policy(SET CMP0140 NEW)
cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>

find_package(PkgConfig QUIET)

#---------------------------------------------------------------------
# Helper functions
#---------------------------------------------------------------------
function(FIND_IMAGEMAGICK_API component header)
  set(ImageMagick_${component}_FOUND FALSE PARENT_SCOPE)

  if(PkgConfig_FOUND)
    pkg_check_modules(PC_${component} QUIET ${component})
  endif()

  find_path(ImageMagick_${component}_INCLUDE_DIR
    NAMES ${header}
    HINTS
      ${PC_${component}_INCLUDEDIR}
      ${PC_${component}_INCLUDE_DIRS}
    PATHS
      ${ImageMagick_INCLUDE_DIRS}
      "[HKEY_LOCAL_MACHINE\\SOFTWARE\\ImageMagick\\Current;BinPath]/include"
    PATH_SUFFIXES
      ImageMagick ImageMagick-6 ImageMagick-7
    DOC "Path to the ImageMagick arch-independent include dir."
    NO_DEFAULT_PATH
    )
  find_path(ImageMagick_${component}_ARCH_INCLUDE_DIR
    NAMES
      magick/magick-baseconfig.h
      MagickCore/magick-baseconfig.h
    HINTS
      ${PC_${component}_INCLUDEDIR}
      ${PC_${component}_INCLUDE_DIRS}
    PATHS
      ${ImageMagick_INCLUDE_DIRS}
      "[HKEY_LOCAL_MACHINE\\SOFTWARE\\ImageMagick\\Current;BinPath]/include"
    PATH_SUFFIXES
      ImageMagick ImageMagick-6 ImageMagick-7
    DOC "Path to the ImageMagick arch-specific include dir."
    NO_DEFAULT_PATH
    )
  find_library(ImageMagick_${component}_LIBRARY
    NAMES ${ARGN}
    HINTS
      ${PC_${component}_LIBDIR}
      ${PC_${component}_LIB_DIRS}
    PATHS
      "[HKEY_LOCAL_MACHINE\\SOFTWARE\\ImageMagick\\Current;BinPath]/lib"
    DOC "Path to the ImageMagick Magick++ library."
    NO_DEFAULT_PATH
    )

  # old version have only indep dir
  if(ImageMagick_${component}_INCLUDE_DIR AND ImageMagick_${component}_LIBRARY)
    set(ImageMagick_${component}_FOUND TRUE PARENT_SCOPE)

    # Construct per-component include directories.
    set(ImageMagick_${component}_INCLUDE_DIRS
      ${ImageMagick_${component}_INCLUDE_DIR}
      )
    if(ImageMagick_${component}_ARCH_INCLUDE_DIR)
      list(APPEND ImageMagick_${component}_INCLUDE_DIRS
        ${ImageMagick_${component}_ARCH_INCLUDE_DIR})
    endif()
    list(REMOVE_DUPLICATES ImageMagick_${component}_INCLUDE_DIRS)
    set(ImageMagick_${component}_INCLUDE_DIRS
      ${ImageMagick_${component}_INCLUDE_DIRS} PARENT_SCOPE)

    set(ImageMagick_${component}_LIBRARIES
      ${ImageMagick_${component}_LIBRARY}
      )
    set(ImageMagick_${component}_LIBRARIES
      ${ImageMagick_${component}_LIBRARIES} PARENT_SCOPE)

    set(ImageMagick_${component}_COMPILE_OPTIONS ${PC_${component}_CFLAGS_OTHER})

    # Add the per-component include directories to the full include dirs.
    list(APPEND ImageMagick_INCLUDE_DIRS ${ImageMagick_${component}_INCLUDE_DIRS})
    list(REMOVE_DUPLICATES ImageMagick_INCLUDE_DIRS)
    set(ImageMagick_INCLUDE_DIRS ${ImageMagick_INCLUDE_DIRS} PARENT_SCOPE)

    list(APPEND ImageMagick_LIBRARIES
      ${ImageMagick_${component}_LIBRARY}
      )
    set(ImageMagick_LIBRARIES ${ImageMagick_LIBRARIES} PARENT_SCOPE)

    list(APPEND ImageMagick_COMPILE_OPTIONS
      ${ImageMagick_${component}_COMPILE_OPTIONS}
      )
    set(ImageMagick_COMPILE_OPTIONS ${ImageMagick_COMPILE_OPTIONS} PARENT_SCOPE)

    if(NOT TARGET ImageMagick::${component})
      add_library(ImageMagick::${component} UNKNOWN IMPORTED)
      set_target_properties(ImageMagick::${component} PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${ImageMagick_${component}_INCLUDE_DIRS}"
        INTERFACE_COMPILE_OPTIONS "${ImageMagick_${component}_COMPILE_OPTIONS}"
        IMPORTED_LOCATION "${ImageMagick_${component}_LIBRARY}")
    endif()
  endif()
endfunction()

function(FIND_IMAGEMAGICK_EXE component)
  set(_IMAGEMAGICK_EXECUTABLE
    ${ImageMagick_EXECUTABLE_DIR}/${component}${CMAKE_EXECUTABLE_SUFFIX})
  if(EXISTS ${_IMAGEMAGICK_EXECUTABLE})
    set(ImageMagick_${component}_EXECUTABLE
      ${_IMAGEMAGICK_EXECUTABLE}
       PARENT_SCOPE
       )
    set(ImageMagick_${component}_FOUND TRUE PARENT_SCOPE)
  else()
    set(ImageMagick_${component}_FOUND FALSE PARENT_SCOPE)
  endif()
endfunction()

function(_ImageMagick_GetVersion)
  unset(version)

  if(ImageMagick_mogrify_EXECUTABLE)
    execute_process(
      COMMAND ${ImageMagick_mogrify_EXECUTABLE} -version
      OUTPUT_VARIABLE version
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(version MATCHES "^Version: ImageMagick ([-0-9.]+)")
      set(version "${CMAKE_MATCH_1}")
    endif()
  elseif(ImageMagick_INCLUDE_DIRS)
    # MagickLibSubversion was used in ImageMagick <= 6.5.
    set(
      regex
      "^[\t ]*#[\t ]*define[\t ]+(MagickLibVersionText|MagickLibAddendum|MagickLibSubversion)[\t ]+\"([-0-9.]+)\""
    )

    foreach(dir IN LISTS ImageMagick_INCLUDE_DIRS)
      foreach(subdir IN ITEMS MagickCore magick)
        if(EXISTS ${dir}/${subdir}/version.h)
          file(STRINGS "${dir}/${subdir}/version.h" results REGEX "${regex}")

          foreach(line ${results})
            if(line MATCHES "${regex}")
              if(DEFINED version)
                string(APPEND version "${CMAKE_MATCH_2}")
              else()
                set(version "${CMAKE_MATCH_2}")
              endif()

              if(CMAKE_MATCH_1 STREQUAL "MagickLibAddendum")
                break()
              endif()
            endif()
          endforeach()
        endif()

        if(DEFINED version)
          break()
        endif()
      endforeach()

      if(DEFINED version)
        break()
      endif()
    endforeach()
  endif()

  if(DEFINED version)
    set(ImageMagick_VERSION "${version}")
    set(ImageMagick_VERSION_STRING "${ImageMagick_VERSION}")
  endif()

  return(PROPAGATE ImageMagick_VERSION ImageMagick_VERSION_STRING)
endfunction()

#---------------------------------------------------------------------
# Start Actual Work
#---------------------------------------------------------------------
# Try to find a ImageMagick installation binary path.
find_path(ImageMagick_EXECUTABLE_DIR
  NAMES mogrify${CMAKE_EXECUTABLE_SUFFIX}
  PATHS
    "[HKEY_LOCAL_MACHINE\\SOFTWARE\\ImageMagick\\Current;BinPath]"
  DOC "Path to the ImageMagick binary directory."
  NO_DEFAULT_PATH
  )
find_path(ImageMagick_EXECUTABLE_DIR
  NAMES mogrify${CMAKE_EXECUTABLE_SUFFIX}
  )

# Find each component. Search for all tools in same dir
# <ImageMagick_EXECUTABLE_DIR>; otherwise they should be found
# independently and not in a cohesive module such as this one.
unset(ImageMagick_REQUIRED_VARS)
unset(ImageMagick_DEFAULT_EXECUTABLES)
foreach(component ${ImageMagick_FIND_COMPONENTS}
    # DEPRECATED: forced components for backward compatibility
    convert mogrify import montage composite
    )
  if(component STREQUAL "Magick++")
    FIND_IMAGEMAGICK_API(Magick++ Magick++.h
      Magick++ CORE_RL_Magick++_
      Magick++-6 Magick++-7
      Magick++-Q8 Magick++-Q16 Magick++-Q16HDRI Magick++-Q8HDRI
      Magick++-6.Q64 Magick++-6.Q32 Magick++-6.Q64HDRI Magick++-6.Q32HDRI
      Magick++-6.Q16 Magick++-6.Q8 Magick++-6.Q16HDRI Magick++-6.Q8HDRI
      Magick++-7.Q64 Magick++-7.Q32 Magick++-7.Q64HDRI Magick++-7.Q32HDRI
      Magick++-7.Q16 Magick++-7.Q8 Magick++-7.Q16HDRI Magick++-7.Q8HDRI
      )
    list(APPEND ImageMagick_REQUIRED_VARS ImageMagick_Magick++_LIBRARY)
  elseif(component STREQUAL "MagickWand")
    FIND_IMAGEMAGICK_API(MagickWand "wand/MagickWand.h;MagickWand/MagickWand.h"
      Wand MagickWand CORE_RL_wand_ CORE_RL_MagickWand_
      MagickWand-6 MagickWand-7
      MagickWand-Q16 MagickWand-Q8 MagickWand-Q16HDRI MagickWand-Q8HDRI
      MagickWand-6.Q64 MagickWand-6.Q32 MagickWand-6.Q64HDRI MagickWand-6.Q32HDRI
      MagickWand-6.Q16 MagickWand-6.Q8 MagickWand-6.Q16HDRI MagickWand-6.Q8HDRI
      MagickWand-7.Q64 MagickWand-7.Q32 MagickWand-7.Q64HDRI MagickWand-7.Q32HDRI
      MagickWand-7.Q16 MagickWand-7.Q8 MagickWand-7.Q16HDRI MagickWand-7.Q8HDRI
      )
    list(APPEND ImageMagick_REQUIRED_VARS ImageMagick_MagickWand_LIBRARY)
  elseif(component STREQUAL "MagickCore")
    FIND_IMAGEMAGICK_API(MagickCore "magick/MagickCore.h;MagickCore/MagickCore.h"
      Magick MagickCore CORE_RL_magick_ CORE_RL_MagickCore_
      MagickCore-6 MagickCore-7
      MagickCore-Q16 MagickCore-Q8 MagickCore-Q16HDRI MagickCore-Q8HDRI
      MagickCore-6.Q64 MagickCore-6.Q32 MagickCore-6.Q64HDRI MagickCore-6.Q32HDRI
      MagickCore-6.Q16 MagickCore-6.Q8 MagickCore-6.Q16HDRI MagickCore-6.Q8HDRI
      MagickCore-7.Q64 MagickCore-7.Q32 MagickCore-7.Q64HDRI MagickCore-7.Q32HDRI
      MagickCore-7.Q16 MagickCore-7.Q8 MagickCore-7.Q16HDRI MagickCore-7.Q8HDRI
      )
    list(APPEND ImageMagick_REQUIRED_VARS ImageMagick_MagickCore_LIBRARY)
  else()
    if(ImageMagick_EXECUTABLE_DIR)
      FIND_IMAGEMAGICK_EXE(${component})
    endif()

    if(ImageMagick_FIND_COMPONENTS)
      list(FIND ImageMagick_FIND_COMPONENTS ${component} is_requested)
      if(is_requested GREATER -1)
        list(APPEND ImageMagick_REQUIRED_VARS ImageMagick_${component}_EXECUTABLE)
      endif()
    elseif(ImageMagick_${component}_EXECUTABLE)
      # if no components were requested explicitly put all (default) executables
      # in the list
      list(APPEND ImageMagick_DEFAULT_EXECUTABLES ImageMagick_${component}_EXECUTABLE)
    endif()
  endif()
endforeach()

if(NOT ImageMagick_FIND_COMPONENTS AND NOT ImageMagick_DEFAULT_EXECUTABLES)
  # No components were requested, and none of the default components were
  # found. Just insert mogrify into the list of the default components to
  # find so FPHSA below has something to check
  list(APPEND ImageMagick_REQUIRED_VARS ImageMagick_mogrify_EXECUTABLE)
elseif(ImageMagick_DEFAULT_EXECUTABLES)
  list(APPEND ImageMagick_REQUIRED_VARS ${ImageMagick_DEFAULT_EXECUTABLES})
endif()

set(ImageMagick_INCLUDE_DIRS ${ImageMagick_INCLUDE_DIRS})
set(ImageMagick_LIBRARIES ${ImageMagick_LIBRARIES})

_ImageMagick_GetVersion()

#---------------------------------------------------------------------
# Standard Package Output
#---------------------------------------------------------------------
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ImageMagick
                                  REQUIRED_VARS ${ImageMagick_REQUIRED_VARS}
                                  VERSION_VAR ImageMagick_VERSION
  )

#---------------------------------------------------------------------
# DEPRECATED: Setting variables for backward compatibility.
#---------------------------------------------------------------------
set(IMAGEMAGICK_BINARY_PATH          ${ImageMagick_EXECUTABLE_DIR}
    CACHE PATH "Path to the ImageMagick binary directory.")
set(IMAGEMAGICK_CONVERT_EXECUTABLE   ${ImageMagick_convert_EXECUTABLE}
    CACHE FILEPATH "Path to ImageMagick's convert executable.")
set(IMAGEMAGICK_MOGRIFY_EXECUTABLE   ${ImageMagick_mogrify_EXECUTABLE}
    CACHE FILEPATH "Path to ImageMagick's mogrify executable.")
set(IMAGEMAGICK_IMPORT_EXECUTABLE    ${ImageMagick_import_EXECUTABLE}
    CACHE FILEPATH "Path to ImageMagick's import executable.")
set(IMAGEMAGICK_MONTAGE_EXECUTABLE   ${ImageMagick_montage_EXECUTABLE}
    CACHE FILEPATH "Path to ImageMagick's montage executable.")
set(IMAGEMAGICK_COMPOSITE_EXECUTABLE ${ImageMagick_composite_EXECUTABLE}
    CACHE FILEPATH "Path to ImageMagick's composite executable.")
mark_as_advanced(
  IMAGEMAGICK_BINARY_PATH
  IMAGEMAGICK_CONVERT_EXECUTABLE
  IMAGEMAGICK_MOGRIFY_EXECUTABLE
  IMAGEMAGICK_IMPORT_EXECUTABLE
  IMAGEMAGICK_MONTAGE_EXECUTABLE
  IMAGEMAGICK_COMPOSITE_EXECUTABLE
  )

cmake_policy(POP)
