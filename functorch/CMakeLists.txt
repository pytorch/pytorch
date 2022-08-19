cmake_minimum_required(VERSION 3.12)
project(functorch)
set(CMAKE_CXX_STANDARD 14)
file(STRINGS version.txt FUNCTORCH_VERSION)
find_package(Python3 REQUIRED COMPONENTS Development)
if(NOT BUILD_FUNCTORCH)
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
endif()


include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(FT_DIR csrc)
file(GLOB_RECURSE FT_SOURCES ${FT_DIR}/*.cpp)

Python3_add_library(${PROJECT_NAME} MODULE ${FT_SOURCES})
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(${PROJECT_NAME} PRIVATE FUNCTORCH_BUILD_MAIN_LIB)
target_compile_definitions(${PROJECT_NAME} PRIVATE TORCH_EXTENSION_NAME=_C)
target_compile_definitions(${PROJECT_NAME} PRIVATE TORCH_API_INCLUDE_EXTENSION_H)
if(BUILD_FUNCTORCH)
  target_compile_options(${PROJECT_NAME} PRIVATE ${TORCH_PYTHON_COMPILE_OPTIONS})
  target_link_libraries(${PROJECT_NAME} PRIVATE torch torch_python)
  target_link_libraries(${PROJECT_NAME} PRIVATE pybind::pybind11)

  set_target_properties(${PROJECT_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY
        ${CMAKE_BINARY_DIR}/functorch)
  install(TARGETS ${PROJECT_NAME} DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}")
else()
  target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
endif()
