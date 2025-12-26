include(CMakePrintHelpers)

# Generate AITER/CK Asm code
find_package(Python3 REQUIRED)
execute_process(
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/third_party/aiter/csrc/py_itfs_cu/fmha_v3_bwd_kernel_generate.py --receipt 1 --output_dir ${CMAKE_CURRENT_LIST_DIR}
    RESULT_VARIABLE ret
)

if(ret AND NOT ret EQUAL 0)
    message( FATAL_ERROR "Failed to generate FAv3 CK Kernels")
endif()

execute_process(
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/third_party/aiter/csrc/cpp_itfs/mha_bwd_generate.py --receipt 3 --output_dir ${CMAKE_CURRENT_LIST_DIR}
    RESULT_VARIABLE ret
)


# Change file extensions from .cpp to .hip using CMake native commands
file(GLOB FAV3_CPP_FILES "${CMAKE_CURRENT_LIST_DIR}/*.cpp")
foreach(cpp_file ${FAV3_CPP_FILES})
  string(REGEX REPLACE "\\.cpp$" ".hip" hip_file "${cpp_file}")
  file(RENAME "${cpp_file}" "${hip_file}")
endforeach()
