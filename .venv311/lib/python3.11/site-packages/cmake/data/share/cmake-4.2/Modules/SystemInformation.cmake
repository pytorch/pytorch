# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(DumpInformation)

# first get the standard information for the platform
include_directories("This does not exist")
get_directory_property(incl INCLUDE_DIRECTORIES)
set_directory_properties(PROPERTIES INCLUDE_DIRECTORIES "${DumpInformation_BINARY_DIR};${DumpInformation_SOURCE_DIR}")

configure_file("${CMAKE_ROOT}/Modules/SystemInformation.in" "${RESULT_FILE}")


file(APPEND "${RESULT_FILE}"
  "\n=================================================================\n")
file(APPEND "${RESULT_FILE}"
  "=== VARIABLES\n")
file(APPEND "${RESULT_FILE}"
  "=================================================================\n")
get_cmake_property(res VARIABLES)
foreach(var ${res})
  file(APPEND "${RESULT_FILE}" "${var} \"${${var}}\"\n")
endforeach()

file(APPEND "${RESULT_FILE}"
  "\n=================================================================\n")
file(APPEND "${RESULT_FILE}"
  "=== COMMANDS\n")
file(APPEND "${RESULT_FILE}"
  "=================================================================\n")
get_cmake_property(res COMMANDS)
foreach(var ${res})
  file(APPEND "${RESULT_FILE}" "${var}\n")
endforeach()

file(APPEND "${RESULT_FILE}"
  "\n=================================================================\n")
file(APPEND "${RESULT_FILE}"
  "=== MACROS\n")
file(APPEND "${RESULT_FILE}"
  "=================================================================\n")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/AllMacros.txt "")
get_cmake_property(res MACROS)
foreach(var ${res})
  file(APPEND "${RESULT_FILE}" "${var}\n")
endforeach()

file(APPEND "${RESULT_FILE}"
  "\n=================================================================\n")
file(APPEND "${RESULT_FILE}"
  "=== OTHER\n")
file(APPEND "${RESULT_FILE}"
  "=================================================================\n")
get_directory_property(res INCLUDE_DIRECTORIES)
foreach(var ${res})
  file(APPEND "${RESULT_FILE}" "INCLUDE_DIRECTORY: ${var}\n")
endforeach()

get_directory_property(res LINK_DIRECTORIES)
foreach(var ${res})
  file(APPEND "${RESULT_FILE}" "LINK_DIRECTORIES: ${var}\n")
endforeach()

get_directory_property(res INCLUDE_REGULAR_EXPRESSION)
file(APPEND "${RESULT_FILE}" "INCLUDE_REGULAR_EXPRESSION: ${res}\n")

# include other files if they are present, such as when run from within the
# binary tree
macro(DUMP_FILE THE_FILE)
  if (EXISTS "${THE_FILE}")
    file(APPEND "${RESULT_FILE}"
      "\n=================================================================\n")
    file(APPEND "${RESULT_FILE}"
      "=== ${THE_FILE}\n")
    file(APPEND "${RESULT_FILE}"
      "=================================================================\n")

    file(READ "${THE_FILE}" FILE_CONTENTS LIMIT 50000)
    file(APPEND "${RESULT_FILE}" "${FILE_CONTENTS}")
  endif ()
endmacro()

DUMP_FILE("../CMakeCache.txt")
DUMP_FILE("../CMakeFiles/CMakeSystem.cmake")

foreach (EXTRA_FILE ${EXTRA_DUMP_FILES})
  DUMP_FILE("${EXTRA_FILE}")
endforeach ()
