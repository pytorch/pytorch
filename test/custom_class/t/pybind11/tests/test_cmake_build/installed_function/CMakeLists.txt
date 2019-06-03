cmake_minimum_required(VERSION 2.8.12)
project(test_installed_module CXX)

set(CMAKE_MODULE_PATH "")

find_package(pybind11 CONFIG REQUIRED)
message(STATUS "Found pybind11 v${pybind11_VERSION}: ${pybind11_INCLUDE_DIRS}")

pybind11_add_module(test_cmake_build SHARED NO_EXTRAS ../main.cpp)

add_custom_target(check ${CMAKE_COMMAND} -E env PYTHONPATH=$<TARGET_FILE_DIR:test_cmake_build>
                  ${PYTHON_EXECUTABLE} ${PROJECT_SOURCE_DIR}/../test.py ${PROJECT_NAME})
