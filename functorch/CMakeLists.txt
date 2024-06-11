cmake_minimum_required(VERSION 3.18)
project(functorch)
set(CMAKE_CXX_STANDARD 17)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(FT_DIR csrc)
file(GLOB_RECURSE FT_SOURCES ${FT_DIR}/*.cpp ${FT_DIR}/*.c)

add_library(${PROJECT_NAME} MODULE ${FT_SOURCES})
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(${PROJECT_NAME} PRIVATE FUNCTORCH_BUILD_MAIN_LIB)
target_compile_definitions(${PROJECT_NAME} PRIVATE TORCH_EXTENSION_NAME=_C)
target_compile_definitions(${PROJECT_NAME} PRIVATE TORCH_API_INCLUDE_EXTENSION_H)
target_compile_options(${PROJECT_NAME} PRIVATE ${TORCH_PYTHON_COMPILE_OPTIONS})
target_compile_options_if_supported(${PROJECT_NAME} "-Wmissing-prototypes")
target_compile_options_if_supported(${PROJECT_NAME} "-Werror=missing-prototypes")
target_link_libraries(${PROJECT_NAME} PRIVATE torch torch_python)
target_link_libraries(${PROJECT_NAME} PRIVATE pybind::pybind11)

set_target_properties(${PROJECT_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY
      ${CMAKE_BINARY_DIR}/functorch)
set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH "${_rpath_portable_origin}/../torch/lib")

# Copy-pasted prefix/suffix logic for Python extensions from
# https://github.com/pytorch/pytorch/blob/33bb8ae350611760139457b85842b1d7edf9aa11/caffe2/CMakeLists.txt#L1975
# https://github.com/pytorch/pytorch/blob/33bb8ae350611760139457b85842b1d7edf9aa11/caffe2/CMakeLists.txt#L2022
# TODO: It would be good to be able to use Python3_add_library target, but it does not work in many cases
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" DEBUG_POSTFIX "")
if(WIN32)
  set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".pyd")
else()
  set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".so")
endif()
# Needed to link functorch on MacOS
if(NOT ${TORCH_PYTHON_LINK_FLAGS} STREQUAL "")
  set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS ${TORCH_PYTHON_LINK_FLAGS})
endif()
install(TARGETS ${PROJECT_NAME} DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}")
