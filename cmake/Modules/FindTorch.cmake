# FindTorch
# -------
#
# Finds the Torch library
#
# This will define the following variables:
#
#   TORCH_FOUND        -- True if the system has the Torch library
#   TORCH_INCLUDE_DIRS -- The include directories for torch
#   TORCH_LIBRARIES    -- Libraries to link to
#
# and the following imported targets:
#
#   Torch
#
# and the following functions:
#
#   torch_add_custom_op_library(<name> <source_files>)

include(FindPackageHandleStandardArgs)

get_filename_component(PYTORCH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../" ABSOLUTE)
set(PYTORCH_BUILD_PATH "${PYTORCH_ROOT}/tools/cpp_build/build")

set(TORCH_INCLUDE_DIRS
  "${PYTORCH_ROOT}"
  "${PYTORCH_ROOT}/aten/src"
  "${PYTORCH_BUILD_PATH}/install/include"
  "${PYTORCH_BUILD_PATH}/install/include/ATen"
  "${PYTORCH_BUILD_PATH}/install/include/TH"
)

find_library(TORCH_LIBRARY torch
  PATHS "${PYTORCH_BUILD_PATH}/install/lib" NO_DEFAULT_PATH)

find_library(CAFFE2_LIBRARY caffe2
  PATHS "${PYTORCH_BUILD_PATH}/install/lib" NO_DEFAULT_PATH)

mark_as_advanced(TORCH_LIBRARY TORCH_INCLUDE_DIRS)

# If possible, link CUDA.
find_package(CUDA)
if (CUDA_FOUND)
  find_library(CAFFE2_CUDA_LIBRARY caffe2_gpu
    PATHS "${PYTORCH_BUILD_PATH}/install/lib" NO_DEFAULT_PATH)
  set(TORCH_CUDA_LIBRARIES -L${CUDA_TOOLKIT_ROOT_DIR}/lib64 cuda nvrtc cudart nvToolsExt)
  list(APPEND TORCH_INCLUDE_DIRS ${CUDA_TOOLKIT_INCLUDE})
endif()

add_library(Torch SHARED IMPORTED)
set_target_properties(Torch PROPERTIES
  IMPORTED_LOCATION "${TORCH_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${TORCH_INCLUDE_DIRS}"
  INTERFACE_LINK_LIBRARIES "${CAFFE2_LIBRARY};${CAFFE2_CUDA_LIBRARY};${TORCH_CUDA_LIBRARIES}"
  CXX_STANDARD 11
)

set(TORCH_LIBRARIES Torch)

find_package_handle_standard_args(
  TORCH
  DEFAULT_MSG
  TORCH_LIBRARY
  TORCH_INCLUDE_DIRS)

# Creates a shared library <name> with the correct include directories
# and linker flags set to include Torch header files and link with Torch
# libraries. Also sets the C++ standard version to C++11. All options
# can be override by specifying further options on the `<name>` CMake target.
function(torch_add_custom_op_library name source_files)
  add_library(${name} SHARED ${source_files})
  target_include_directories(${name} PUBLIC "${TORCH_INCLUDE_DIRS}")
  target_link_libraries(${name} "${TORCH_LIBRARIES}")
  target_compile_options(${name} PUBLIC -std=c++11)
endfunction(torch_add_custom_op_library)
