# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

# In CMakeDetermineCUDACompiler and CMakeTestCUDACompiler we detect
# libraries that the CUDA compiler implicitly passes to the host linker.
# CMake invokes the host linker directly and so needs to pass these libraries.
# Filter out implicit link libraries that should not be passed unconditionally.
macro(cmake_cuda_filter_implicit_libs _var_CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES)
  list(REMOVE_ITEM "${_var_CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES}"
    # The CUDA runtime libraries are controlled by CMAKE_CUDA_RUNTIME_LIBRARY.
    cudart        cudart.lib
    cudart_static cudart_static.lib
    cudadevrt     cudadevrt.lib

    # Dependencies of the CUDA static runtime library on Linux hosts.
    rt
    pthread
    dl
    )
endmacro()
