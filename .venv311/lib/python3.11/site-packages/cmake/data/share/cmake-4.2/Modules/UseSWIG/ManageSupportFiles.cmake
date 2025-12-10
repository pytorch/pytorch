# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


if (ACTION STREQUAL "CLEAN")
  # Collect current list of generated files
  file (GLOB_RECURSE files LIST_DIRECTORIES TRUE RELATIVE "${SUPPORT_FILES_WORKING_DIRECTORY}" "${SUPPORT_FILES_WORKING_DIRECTORY}/*")

  if (files)
    # clean-up the output directory
    ## compute full paths
    list (TRANSFORM files PREPEND "${SUPPORT_FILES_OUTPUT_DIRECTORY}/")
    ## remove generated files from the output directory
    file (REMOVE ${files})

    # clean-up working directory
    file (REMOVE_RECURSE "${SUPPORT_FILES_WORKING_DIRECTORY}")
  endif()

  file (MAKE_DIRECTORY "${SUPPORT_FILES_WORKING_DIRECTORY}")
endif()

if (ACTION STREQUAL "COPY")
  # Collect current list of generated files
  file (GLOB files LIST_DIRECTORIES TRUE "${SUPPORT_FILES_WORKING_DIRECTORY}/*")

  if (files)
    # copy files to the output directory
    file (COPY ${files} DESTINATION "${SUPPORT_FILES_OUTPUT_DIRECTORY}")
  endif()
endif()
