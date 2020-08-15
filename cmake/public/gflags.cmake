# ---[ gflags

# We will try to use the config mode first, and then manual find.
find_package(gflags CONFIG QUIET)
if(NOT TARGET gflags)
  find_package(gflags MODULE QUIET)
endif()

if(TARGET gflags)
  message(STATUS "Caffe2: Found gflags with new-style gflags target.")
elseif(GFLAGS_FOUND)
  message(STATUS "Caffe2: Found gflags with old-style gflag starget.")
  add_library(gflags UNKNOWN IMPORTED)
  set_property(
      TARGET gflags PROPERTY IMPORTED_LOCATION ${GFLAGS_LIBRARY})
  set_property(
      TARGET gflags PROPERTY INTERFACE_INCLUDE_DIRECTORIES
      ${GFLAGS_INCLUDE_DIR})
else()
  message(STATUS
      "Caffe2: Cannot find gflags automatically. Using legacy find.")

  # - Try to find GFLAGS in the legacy way.
  #
  # The following variables are optionally searched for defaults
  #  GFLAGS_ROOT_DIR: Base directory where all GFLAGS components are found
  #
  # The following are set after configuration is done:
  #  GFLAGS_FOUND
  #  GFLAGS_INCLUDE_DIRS
  #  GFLAGS_LIBRARIES
  #  GFLAGS_LIBRARYRARY_DIRS
  include(FindPackageHandleStandardArgs)
  set(GFLAGS_ROOT_DIR "" CACHE PATH "Folder contains Gflags")

  # We are testing only a couple of files in the include directories
  if(WIN32)
    find_path(GFLAGS_INCLUDE_DIR gflags/gflags.h
        PATHS ${GFLAGS_ROOT_DIR}/src/windows)
  else()
    find_path(GFLAGS_INCLUDE_DIR gflags/gflags.h
        PATHS ${GFLAGS_ROOT_DIR})
  endif()

  if(WIN32)
    find_library(GFLAGS_LIBRARY_RELEASE
        NAMES libgflags
        PATHS ${GFLAGS_ROOT_DIR}
        PATH_SUFFIXES Release)

    find_library(GFLAGS_LIBRARY_DEBUG
        NAMES libgflags-debug
        PATHS ${GFLAGS_ROOT_DIR}
        PATH_SUFFIXES Debug)
    set(GFLAGS_LIBRARY optimized ${GFLAGS_LIBRARY_RELEASE} debug ${GFLAGS_LIBRARY_DEBUG})
  else()
    find_library(GFLAGS_LIBRARY gflags)
  endif()

  find_package_handle_standard_args(
      gflags DEFAULT_MSG GFLAGS_INCLUDE_DIR GFLAGS_LIBRARY)

  if(GFLAGS_FOUND)
    message(
        STATUS
        "Caffe2: Found gflags  (include: ${GFLAGS_INCLUDE_DIR}, "
        "library: ${GFLAGS_LIBRARY})")
    add_library(gflags UNKNOWN IMPORTED)
    set_property(
        TARGET gflags PROPERTY IMPORTED_LOCATION ${GFLAGS_LIBRARY})
    set_property(
        TARGET gflags PROPERTY INTERFACE_INCLUDE_DIRECTORIES
        ${GFLAGS_INCLUDE_DIR})
  endif()
endif()

# After above, we should have the gflags target now.
if(NOT TARGET gflags)
  message(WARNING
      "Caffe2: gflags cannot be found. Depending on whether you are building "
      "Caffe2 or a Caffe2 dependent library, the next warning / error will "
      "give you more info.")
endif()

