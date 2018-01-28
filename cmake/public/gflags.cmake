# ---[ gflags
find_package(gflags)

if (TARGET gflags)
  message(STATUS "Found gflags with new-style gflags target.")
elseif(GFLAGS_FOUND)
  message(STATUS "Found gflags with old-style gflag starget.")
  add_library(gflags UNKNOWN IMPORTED)
  set_property(
      TARGET gflags PROPERTY IMPORTED_LOCATION ${GFLAGS_LIBRARY})
  set_property(
      TARGET gflags PROPERTY INTERFACE_INCLUDE_DIRECTORIES
      ${GFLAGS_INCLUDE_DIR})
else()
  message(STATUS "Cannot find gflags with config files. Using legacy find.")

  # - Try to find GFLAGS in the legacy way.
  #
  # The following variables are optionally searched for defaults
  #  GFLAGS_ROOT_DIR:            Base directory where all GFLAGS components are found
  #
  # The following are set after configuration is done:
  #  GFLAGS_FOUND
  #  GFLAGS_INCLUDE_DIRS
  #  GFLAGS_LIBRARIES
  #  GFLAGS_LIBRARYRARY_DIRS
  include(FindPackageHandleStandardArgs)
  set(GFLAGS_ROOT_DIR "" CACHE PATH "Folder contains Gflags")

  # We are testing only a couple of files in the include directories
  if(NOT WIN32)
      find_path(GFLAGS_INCLUDE_DIR gflags/gflags.h
          PATHS ${GFLAGS_ROOT_DIR})
  endif()

  if(MSVC)
      find_package(gflags NO_MODULE)
      set(GFLAGS_LIBRARY ${gflags_LIBRARIES})
  else()
      find_library(GFLAGS_LIBRARY gflags)
  endif()

  find_package_handle_standard_args(
      gflags DEFAULT_MSG GFLAGS_INCLUDE_DIR GFLAGS_LIBRARY)

  if(GFLAGS_FOUND)
    message(
        STATUS
        "Found gflags  (include: ${GFLAGS_INCLUDE_DIR}, "
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
if (NOT TARGET gflags)
  message(WARNING
      "gflags cannot be found. Depending on whether you are building Caffe2 "
      "or a Caffe2 dependent library, the next warning / error will give you "
      "more info.")
endif()

