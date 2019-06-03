add_custom_target(test_cmake_build)

if(CMAKE_VERSION VERSION_LESS 3.1)
  # 3.0 needed for interface library for subdirectory_target/installed_target
  # 3.1 needed for cmake -E env for testing
  return()
endif()

include(CMakeParseArguments)
function(pybind11_add_build_test name)
  cmake_parse_arguments(ARG "INSTALL" "" "" ${ARGN})

  set(build_options "-DCMAKE_PREFIX_PATH=${PROJECT_BINARY_DIR}/mock_install"
                    "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
                    "-DPYTHON_EXECUTABLE:FILEPATH=${PYTHON_EXECUTABLE}"
                    "-DPYBIND11_CPP_STANDARD=${PYBIND11_CPP_STANDARD}")
  if(NOT ARG_INSTALL)
    list(APPEND build_options "-DPYBIND11_PROJECT_DIR=${PROJECT_SOURCE_DIR}")
  endif()

  add_custom_target(test_${name} ${CMAKE_CTEST_COMMAND}
    --quiet --output-log ${name}.log
    --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/${name}"
                     "${CMAKE_CURRENT_BINARY_DIR}/${name}"
    --build-config Release
    --build-noclean
    --build-generator ${CMAKE_GENERATOR}
    $<$<BOOL:${CMAKE_GENERATOR_PLATFORM}>:--build-generator-platform> ${CMAKE_GENERATOR_PLATFORM}
    --build-makeprogram ${CMAKE_MAKE_PROGRAM}
    --build-target check
    --build-options ${build_options}
  )
  if(ARG_INSTALL)
    add_dependencies(test_${name} mock_install)
  endif()
  add_dependencies(test_cmake_build test_${name})
endfunction()

pybind11_add_build_test(subdirectory_function)
pybind11_add_build_test(subdirectory_target)
if(NOT ${PYTHON_MODULE_EXTENSION} MATCHES "pypy")
  pybind11_add_build_test(subdirectory_embed)
endif()

if(PYBIND11_INSTALL)
  add_custom_target(mock_install ${CMAKE_COMMAND}
    "-DCMAKE_INSTALL_PREFIX=${PROJECT_BINARY_DIR}/mock_install"
    -P "${PROJECT_BINARY_DIR}/cmake_install.cmake"
  )

  pybind11_add_build_test(installed_function INSTALL)
  pybind11_add_build_test(installed_target INSTALL)
  if(NOT ${PYTHON_MODULE_EXTENSION} MATCHES "pypy")
    pybind11_add_build_test(installed_embed INSTALL)
  endif()
endif()

add_dependencies(check test_cmake_build)
