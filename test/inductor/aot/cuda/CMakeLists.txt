cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(test)

set(Torch_DIR "../../../../torch/share/cmake/Torch")
find_package(Torch REQUIRED)

add_library(aot_inductor_output SHARED IMPORTED)
set_property(TARGET aot_inductor_output PROPERTY
             IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/aot_inductor_output.so)
add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/aot_inductor_output.so
        COMMAND python ${CMAKE_SOURCE_DIR}/test.py
        DEPENDS ${CMAKE_SOURCE_DIR}/test.py
)
add_custom_target(aot_inductor_output_target ALL
        DEPENDS ${CMAKE_BINARY_DIR}/aot_inductor_output.so)

add_executable(test test.cpp)
target_link_libraries(test ${TORCH_LIBRARIES} aot_inductor_output)
add_dependencies(test aot_inductor_output_target)
set_property(TARGET test PROPERTY CXX_STANDARD 17)