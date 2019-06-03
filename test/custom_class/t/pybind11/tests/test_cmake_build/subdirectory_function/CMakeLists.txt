cmake_minimum_required(VERSION 2.8.12)
project(test_subdirectory_module CXX)

add_subdirectory(${PYBIND11_PROJECT_DIR} pybind11)
pybind11_add_module(test_cmake_build THIN_LTO ../main.cpp)

add_custom_target(check ${CMAKE_COMMAND} -E env PYTHONPATH=$<TARGET_FILE_DIR:test_cmake_build>
                  ${PYTHON_EXECUTABLE} ${PROJECT_SOURCE_DIR}/../test.py ${PROJECT_NAME})
