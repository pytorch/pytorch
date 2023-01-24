# ---[ glog

# We will try to use the config mode first, and then manual find.
find_package(glog CONFIG QUIET)
if(NOT TARGET glog::glog)
  find_package(glog MODULE QUIET)
endif()

if(TARGET glog::glog)
  message(STATUS "Caffe2: Found glog with new-style glog target.")
elseif(GLOG_FOUND)
  message(
      STATUS
      "Caffe2: Found glog with old-style glog starget. Glog never shipped "
      "old style glog targets, so somewhere in your cmake path there might "
      "be a custom Findglog.cmake file that got triggered. We will make a "
      "best effort to create the new style glog target for you.")
  add_library(glog::glog UNKNOWN IMPORTED)
  set_property(
      TARGET glog::glog PROPERTY IMPORTED_LOCATION ${GLOG_LIBRARY})
  set_property(
      TARGET glog::glog PROPERTY INTERFACE_INCLUDE_DIRECTORIES
      ${GLOG_INCLUDE_DIR})
else()
  message(STATUS "Caffe2: Cannot find glog automatically. Using legacy find.")

  # - Try to find Glog
  #
  # The following variables are optionally searched for defaults
  #  GLOG_ROOT_DIR: Base directory where all GLOG components are found
  #
  # The following are set after configuration is done:
  #  GLOG_FOUND
  #  GLOG_INCLUDE_DIRS
  #  GLOG_LIBRARIES
  #  GLOG_LIBRARYRARY_DIRS

  include(FindPackageHandleStandardArgs)
  set(GLOG_ROOT_DIR "" CACHE PATH "Folder contains Google glog")
  if(NOT WIN32)
      find_path(GLOG_INCLUDE_DIR glog/logging.h
          PATHS ${GLOG_ROOT_DIR})
  endif()

  find_library(GLOG_LIBRARY glog
      PATHS ${GLOG_ROOT_DIR}
      PATH_SUFFIXES lib lib64)

  find_package_handle_standard_args(glog DEFAULT_MSG GLOG_INCLUDE_DIR GLOG_LIBRARY)

  if(GLOG_FOUND)
    message(STATUS
        "Caffe2: Found glog (include: ${GLOG_INCLUDE_DIR}, "
        "library: ${GLOG_LIBRARY})")
    add_library(glog::glog UNKNOWN IMPORTED)
    set_property(
        TARGET glog::glog PROPERTY IMPORTED_LOCATION ${GLOG_LIBRARY})
    set_property(
        TARGET glog::glog PROPERTY INTERFACE_INCLUDE_DIRECTORIES
        ${GLOG_INCLUDE_DIR})
  endif()
endif()

# After above, we should have the glog::glog target now.
if(NOT TARGET glog::glog)
  message(WARNING
      "Caffe2: glog cannot be found. Depending on whether you are building "
      "Caffe2 or a Caffe2 dependent library, the next warning / error will "
      "give you more info.")
endif()
