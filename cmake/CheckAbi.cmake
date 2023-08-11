if(DEFINED GLIBCXX_USE_CXX11_ABI)
  message(STATUS "_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI} is already defined as a cmake variable")
  return()
endif()

# XXX This ABI check cannot be run with arm-linux-androideabi-g++
message(STATUS "${CMAKE_CXX_COMPILER} ${PROJECT_SOURCE_DIR}/torch/abi-check.cpp -o ${CMAKE_BINARY_DIR}/abi-check")
execute_process(
  COMMAND
  "${CMAKE_CXX_COMPILER}"
  "${PROJECT_SOURCE_DIR}/torch/abi-check.cpp"
  "-o"
  "${CMAKE_BINARY_DIR}/abi-check"
  RESULT_VARIABLE ABI_CHECK_COMPILE_RESULT)
if(ABI_CHECK_COMPILE_RESULT)
  message(FATAL_ERROR "Could not compile ABI Check: ${ABI_CHECK_COMPILE_RESULT}")
  set(GLIBCXX_USE_CXX11_ABI 0)
endif()
execute_process(
  COMMAND "${CMAKE_BINARY_DIR}/abi-check"
  RESULT_VARIABLE ABI_CHECK_RESULT
  OUTPUT_VARIABLE GLIBCXX_USE_CXX11_ABI)
if(ABI_CHECK_RESULT)
  message(WARNING "Could not run ABI Check: ${ABI_CHECK_RESULT}")
  set(GLIBCXX_USE_CXX11_ABI 0)
endif()
message(STATUS "Determined _GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI}")
