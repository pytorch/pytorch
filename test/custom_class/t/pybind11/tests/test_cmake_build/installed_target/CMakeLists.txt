cmake_minimum_required(VERSION 3.0)
project(test_installed_target CXX)

set(CMAKE_MODULE_PATH "")

find_package(pybind11 CONFIG REQUIRED)
message(STATUS "Found pybind11 v${pybind11_VERSION}: ${pybind11_INCLUDE_DIRS}")

add_library(test_cmake_build MODULE ../main.cpp)

target_link_libraries(test_cmake_build PRIVATE pybind11::module)

# make sure result is, for example, test_installed_target.so, not libtest_installed_target.dylib
set_target_properties(test_cmake_build PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                                  SUFFIX "${PYTHON_MODULE_EXTENSION}")

# Do not treat includes from IMPORTED target as SYSTEM (Python headers in pybind11::module).
# This may be needed to resolve header conflicts, e.g. between Python release and debug headers.
set_target_properties(test_cmake_build PROPERTIES NO_SYSTEM_FROM_IMPORTED ON)

add_custom_target(check ${CMAKE_COMMAND} -E env PYTHONPATH=$<TARGET_FILE_DIR:test_cmake_build>
                  ${PYTHON_EXECUTABLE} ${PROJECT_SOURCE_DIR}/../test.py ${PROJECT_NAME})
