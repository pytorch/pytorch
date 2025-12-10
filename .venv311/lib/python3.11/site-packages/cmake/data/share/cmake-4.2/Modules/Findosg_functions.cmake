# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
Findosg_functions
-----------------

.. note::

  This module is not intended to be included or invoked directly by project code
  during typical use of :command:`find_package` command.  It is internally used
  by OpenSceneGraph (OSG) find modules to assist with searching for OSG
  libraries and NodeKits.  For usage details refer to the
  :module:`FindOpenSceneGraph` module.
#]=======================================================================]

include(${CMAKE_CURRENT_LIST_DIR}/SelectLibraryConfigurations.cmake)

#
# OSG_FIND_PATH
#
function(OSG_FIND_PATH module header)
  string(TOUPPER ${module} module_uc)

  # Try the user's environment request before anything else.
  find_path(${module_uc}_INCLUDE_DIR ${header}
    HINTS
      ENV ${module_uc}_DIR
      ENV OSG_DIR
      ENV OSGDIR
      ENV OSG_ROOT
      ${${module_uc}_DIR}
      ${OSG_DIR}
    PATH_SUFFIXES include
  )
endfunction()


#
# OSG_FIND_LIBRARY
#
function(OSG_FIND_LIBRARY module library)
  string(TOUPPER ${module} module_uc)

  find_library(${module_uc}_LIBRARY_RELEASE
    NAMES ${library}
    HINTS
      ENV ${module_uc}_DIR
      ENV OSG_DIR
      ENV OSGDIR
      ENV OSG_ROOT
      ${${module_uc}_DIR}
      ${OSG_DIR}
    PATH_SUFFIXES lib
  )

  find_library(${module_uc}_LIBRARY_DEBUG
    NAMES ${library}d
    HINTS
      ENV ${module_uc}_DIR
      ENV OSG_DIR
      ENV OSGDIR
      ENV OSG_ROOT
      ${${module_uc}_DIR}
      ${OSG_DIR}
    PATH_SUFFIXES lib
  )

  select_library_configurations(${module_uc})

  # the variables set by select_library_configurations go out of scope
  # here, so we need to set them again
  set(${module_uc}_LIBRARY ${${module_uc}_LIBRARY} PARENT_SCOPE)
  set(${module_uc}_LIBRARIES ${${module_uc}_LIBRARIES} PARENT_SCOPE)
endfunction()

#
# OSG_MARK_AS_ADVANCED
# Just a convenience function for calling MARK_AS_ADVANCED
#
function(OSG_MARK_AS_ADVANCED _module)
  string(TOUPPER ${_module} _module_UC)
  mark_as_advanced(${_module_UC}_INCLUDE_DIR)
  mark_as_advanced(${_module_UC}_LIBRARY)
  mark_as_advanced(${_module_UC}_LIBRARY_DEBUG)
endfunction()
