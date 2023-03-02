cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(test)

set(Torch_DIR "../../../../torch/share/cmake/Torch")
find_package(Torch REQUIRED)

add_executable(test test.cpp ${CMAKE_BINARY_DIR}/aot_inductor_output.h)

add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/aot_inductor_output.h
        COMMAND python ${CMAKE_SOURCE_DIR}/test.py
        DEPENDS ${CMAKE_SOURCE_DIR}/test.py
)
add_custom_target(generate_header ALL
    DEPENDS ${CMAKE_BINARY_DIR}/aot_inductor_output.h)

add_library(aot_inductor_output SHARED IMPORTED)
set_property(TARGET aot_inductor_output PROPERTY
             IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/aot_inductor_output.so)

target_link_libraries(test "${TORCH_LIBRARIES}" aot_inductor_output)

set_property(TARGET test PROPERTY CXX_STANDARD 17)
