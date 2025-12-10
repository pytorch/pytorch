# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

if(NOT CPACK_WIX_ROOT)
  string(REPLACE "\\" "/" CPACK_WIX_ROOT "$ENV{WIX}")
endif()

if(CPACK_WIX_VERSION VERSION_GREATER_EQUAL 4)
  find_program(CPACK_WIX_EXECUTABLE NAMES wix
    PATHS "${CPACK_WIX_ROOT}" PATH_SUFFIXES "bin")
  if(NOT CPACK_WIX_EXECUTABLE)
    message(FATAL_ERROR "Could not find the 'wix' executable.")
  endif()

  if(NOT DEFINED CPACK_WIX_INSTALL_SCOPE)
    set(CPACK_WIX_INSTALL_SCOPE "perMachine")
  endif()
else()
  find_program(CPACK_WIX_CANDLE_EXECUTABLE candle
    PATHS "${CPACK_WIX_ROOT}" PATH_SUFFIXES "bin")
  if(NOT CPACK_WIX_CANDLE_EXECUTABLE)
    message(FATAL_ERROR "Could not find the WiX candle executable.")
  endif()

  find_program(CPACK_WIX_LIGHT_EXECUTABLE light
    PATHS "${CPACK_WIX_ROOT}" PATH_SUFFIXES "bin")
  if(NOT CPACK_WIX_LIGHT_EXECUTABLE)
    message(FATAL_ERROR "Could not find the WiX light executable.")
  endif()

  if(NOT DEFINED CPACK_WIX_INSTALL_SCOPE)
    set(CPACK_WIX_INSTALL_SCOPE "NONE")
  endif()
endif()
