# This file used to build libtorch.so.
# Now it only builds the Torch python bindings.

if (NOT CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO)
  cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
  project(torch CXX C)
  find_package(torch REQUIRED)
  option(USE_CUDA "Use CUDA" ON)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

if (NOT BUILD_PYTHON)
  return()
endif()

if (USE_TBB)
include_directories(${TBB_ROOT_DIR}/include)
endif()

set(TORCH_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(TORCH_ROOT "${TORCH_SRC_DIR}/..")

if(NOT TORCH_INSTALL_LIB_DIR)
  set(TORCH_INSTALL_LIB_DIR lib)
endif()

if (MSVC)
    set(LIBSHM_SUBDIR libshm_windows)
else()
    set(LIBSHM_SUBDIR libshm)
endif()

set(LIBSHM_SRCDIR ${TORCH_SRC_DIR}/lib/${LIBSHM_SUBDIR})
add_subdirectory(${LIBSHM_SRCDIR})


# Generate files
set(TOOLS_PATH "${TORCH_ROOT}/tools")


set(TORCH_PYTHON_SRCS
    ${GENERATED_THNN_CXX}
    ${GENERATED_CXX_PYTHON}
    ${TORCH_SRC_DIR}/csrc/CudaIPCTypes.cpp
    ${TORCH_SRC_DIR}/csrc/DataLoader.cpp
    ${TORCH_SRC_DIR}/csrc/Device.cpp
    ${TORCH_SRC_DIR}/csrc/Dtype.cpp
    ${TORCH_SRC_DIR}/csrc/DynamicTypes.cpp
    ${TORCH_SRC_DIR}/csrc/Exceptions.cpp
    ${TORCH_SRC_DIR}/csrc/TypeInfo.cpp
    ${TORCH_SRC_DIR}/csrc/Generator.cpp
    ${TORCH_SRC_DIR}/csrc/Layout.cpp
    ${TORCH_SRC_DIR}/csrc/MemoryFormat.cpp
    ${TORCH_SRC_DIR}/csrc/Module.cpp
    ${TORCH_SRC_DIR}/csrc/PtrWrapper.cpp
    ${TORCH_SRC_DIR}/csrc/Size.cpp
    ${TORCH_SRC_DIR}/csrc/Storage.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/python/init.cpp
    ${TORCH_SRC_DIR}/csrc/autograd/functions/init.cpp
    ${TORCH_SRC_DIR}/csrc/autograd/init.cpp
    ${TORCH_SRC_DIR}/csrc/autograd/python_anomaly_mode.cpp
    ${TORCH_SRC_DIR}/csrc/autograd/python_cpp_function.cpp
    ${TORCH_SRC_DIR}/csrc/autograd/python_engine.cpp
    ${TORCH_SRC_DIR}/csrc/autograd/python_function.cpp
    ${TORCH_SRC_DIR}/csrc/autograd/python_hook.cpp
    ${TORCH_SRC_DIR}/csrc/autograd/python_legacy_variable.cpp
    ${TORCH_SRC_DIR}/csrc/autograd/python_variable.cpp
    ${TORCH_SRC_DIR}/csrc/autograd/python_variable_indexing.cpp
    ${TORCH_SRC_DIR}/csrc/byte_order.cpp
    ${TORCH_SRC_DIR}/csrc/jit/init.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/onnx.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/onnx/fixup_onnx_loop.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/onnx/prepare_division_for_onnx.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/onnx/peephole.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/onnx/constant_fold.cpp
    ${TORCH_SRC_DIR}/csrc/jit/python_arg_flatten.cpp
    ${TORCH_SRC_DIR}/csrc/jit/python_interpreter.cpp
    ${TORCH_SRC_DIR}/csrc/jit/python_ir.cpp
    ${TORCH_SRC_DIR}/csrc/jit/python_tracer.cpp
    ${TORCH_SRC_DIR}/csrc/jit/script/init.cpp
    ${TORCH_SRC_DIR}/csrc/jit/script/python_sugared_value.cpp
    ${TORCH_SRC_DIR}/csrc/jit/script/python_tree_views.cpp
    ${TORCH_SRC_DIR}/csrc/multiprocessing/init.cpp
    ${TORCH_SRC_DIR}/csrc/onnx/init.cpp
    ${TORCH_SRC_DIR}/csrc/serialization.cpp
    ${TORCH_SRC_DIR}/csrc/tensor/python_tensor.cpp
    ${TORCH_SRC_DIR}/csrc/utils.cpp
    ${TORCH_SRC_DIR}/csrc/utils/cuda_lazy_init.cpp
    ${TORCH_SRC_DIR}/csrc/utils/invalid_arguments.cpp
    ${TORCH_SRC_DIR}/csrc/utils/object_ptr.cpp
    ${TORCH_SRC_DIR}/csrc/utils/python_arg_parser.cpp
    ${TORCH_SRC_DIR}/csrc/utils/structseq.cpp
    ${TORCH_SRC_DIR}/csrc/utils/tensor_apply.cpp
    ${TORCH_SRC_DIR}/csrc/utils/tensor_dtypes.cpp
    ${TORCH_SRC_DIR}/csrc/utils/tensor_layouts.cpp
    ${TORCH_SRC_DIR}/csrc/utils/tensor_list.cpp
    ${TORCH_SRC_DIR}/csrc/utils/tensor_memoryformats.cpp
    ${TORCH_SRC_DIR}/csrc/utils/tensor_new.cpp
    ${TORCH_SRC_DIR}/csrc/utils/tensor_numpy.cpp
    ${TORCH_SRC_DIR}/csrc/utils/tensor_types.cpp
    ${TORCH_SRC_DIR}/csrc/utils/tuple_parser.cpp
    )

set(TORCH_PYTHON_INCLUDE_DIRECTORIES
    ${PYTHON_INCLUDE_DIR}

    ${TORCH_ROOT}
    ${TORCH_ROOT}/aten/src
    ${TORCH_ROOT}/aten/src/TH

    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/aten/src
    ${CMAKE_BINARY_DIR}/caffe2/aten/src
    ${CMAKE_BINARY_DIR}/third_party
    ${CMAKE_BINARY_DIR}/third_party/onnx

    ${TORCH_ROOT}/third_party/build/nccl/include
    ${TORCH_ROOT}/third_party/gloo
    ${TORCH_ROOT}/third_party/onnx
    ${TORCH_ROOT}/third_party/pybind11/include

    ${TORCH_SRC_DIR}/csrc
    ${TORCH_SRC_DIR}/csrc/api/include
    ${TORCH_SRC_DIR}/lib
    )


list(APPEND TORCH_PYTHON_INCLUDE_DIRECTORIES ${LIBSHM_SRCDIR})

set(TORCH_PYTHON_LINK_LIBRARIES
    torch_library
    shm)

set(TORCH_PYTHON_COMPILE_DEFINITIONS)

set(TORCH_PYTHON_COMPILE_OPTIONS)

set(TORCH_PYTHON_LINK_FLAGS "")

if (MSVC)
    string(APPEND TORCH_PYTHON_LINK_FLAGS " /NODEFAULTLIB:LIBCMT.LIB")
    list(APPEND TORCH_PYTHON_LINK_LIBRARIES ${PYTHON_LIBRARIES})
    if (NOT ${CMAKE_BUILD_TYPE} MATCHES "Release")
      string(APPEND TORCH_PYTHON_LINK_FLAGS " /DEBUG:FULL")
    endif()
elseif (APPLE)
    string(APPEND TORCH_PYTHON_LINK_FLAGS " -undefined dynamic_lookup")
else()
    list(APPEND TORCH_PYTHON_COMPILE_OPTIONS
      -fno-strict-aliasing
      -Wno-write-strings
      -Wno-strict-aliasing)
endif()

if (USE_CUDA)
    list(APPEND TORCH_PYTHON_SRCS
      ${TORCH_SRC_DIR}/csrc/cuda/Module.cpp
      ${TORCH_SRC_DIR}/csrc/cuda/Storage.cpp
      ${TORCH_SRC_DIR}/csrc/cuda/Stream.cpp
      ${TORCH_SRC_DIR}/csrc/cuda/Event.cpp
      ${TORCH_SRC_DIR}/csrc/cuda/utils.cpp
      ${TORCH_SRC_DIR}/csrc/cuda/python_comm.cpp
      ${TORCH_SRC_DIR}/csrc/cuda/serialization.cpp
      ${GENERATED_THNN_CXX_CUDA}
      )
    list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_CUDA)

    if(MSVC)
      list(APPEND TORCH_PYTHON_LINK_LIBRARIES ${NVTOOLEXT_HOME}/lib/x64/nvToolsExt64_1.lib)
      list(APPEND TORCH_PYTHON_INCLUDE_DIRECTORIES "${NVTOOLEXT_HOME}/include")
    elseif(APPLE)
      list(APPEND TORCH_PYTHON_LINK_LIBRARIES ${CUDA_TOOLKIT_ROOT_DIR}/lib/libnvToolsExt.dylib)
    else()
      find_library(LIBNVTOOLSEXT libnvToolsExt.so PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64/)
      list(APPEND TORCH_PYTHON_LINK_LIBRARIES ${LIBNVTOOLSEXT})
    endif()

endif()

if (USE_CUDNN)
    list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_CUDNN)

    # NOTE: these are at the front, in case there's another cuDNN in
    # CUDA path.
    # Basically, this is the case where $CUDA_HOME/lib64 has an old or
    # incompatible libcudnn.so, which we can inadvertently link to if
    # we're not careful.
    list(INSERT 0 TORCH_PYTHON_LINK_LIBRARIES ${CUDNN_LIBRARY})
    list(INSERT 0 TORCH_PYTHON_INCLUDE_DIRECTORIES ${CUDNN_INCLUDE_DIR})
endif()

if (USE_MIOPEN)
    list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_MIOPEN)
    list(APPEND TORCH_PYTHON_INCLUDE_DIRECTORIES ${MIOPEN_INCLUDE_DIR})
    list(APPEND TORCH_PYTHON_LINK_LIBRARIES ${MIOPEN_LIBRARY})
endif()

if (USE_NUMPY)
    list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_NUMPY)
endif()

if (USE_ROCM)
    list(APPEND TORCH_PYTHON_SRCS
      ${TORCH_SRC_DIR}/csrc/cuda/Module.cpp
      ${TORCH_SRC_DIR}/csrc/cuda/Storage.cpp
      ${TORCH_SRC_DIR}/csrc/cuda/Stream.cpp
      ${TORCH_SRC_DIR}/csrc/cuda/Event.cpp
      ${TORCH_SRC_DIR}/csrc/cuda/utils.cpp
      ${TORCH_SRC_DIR}/csrc/cuda/python_comm.cpp
      ${TORCH_SRC_DIR}/csrc/cuda/serialization.cpp
      ${GENERATED_THNN_CXX_CUDA}
      )

    list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS
      USE_ROCM
      __HIP_PLATFORM_HCC__
      )
endif()

if (USE_DISTRIBUTED)
    list(APPEND TORCH_PYTHON_SRCS ${TORCH_SRC_DIR}/csrc/distributed/Module.cpp)
    list(APPEND TORCH_PYTHON_INCLUDE_DIRECTORIES ${TORCH_SRC_DIR}/lib/THD)
    list(APPEND TORCH_PYTHON_LINK_LIBRARIES THD)
    list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_DISTRIBUTED)
    if (NOT MSVC AND NOT APPLE)
      list(APPEND TORCH_PYTHON_SRCS
        ${TORCH_SRC_DIR}/csrc/distributed/c10d/comm.cpp
        ${TORCH_SRC_DIR}/csrc/distributed/c10d/init.cpp
        ${TORCH_SRC_DIR}/csrc/distributed/c10d/reducer.cpp
        )
      list(APPEND TORCH_PYTHON_LINK_LIBRARIES c10d)
      list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_C10D)
      if (USE_CUDA OR USE_ROCM)
        list(APPEND TORCH_PYTHON_SRCS ${TORCH_SRC_DIR}/csrc/distributed/c10d/ddp.cpp)
      endif()
    endif()
endif()

if (USE_NCCL)
    list(APPEND TORCH_PYTHON_SRCS
      ${TORCH_SRC_DIR}/csrc/cuda/nccl.cpp
      ${TORCH_SRC_DIR}/csrc/cuda/python_nccl.cpp)
    list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS USE_NCCL)
    list(APPEND TORCH_PYTHON_LINK_LIBRARIES __caffe2_nccl)
    if (USE_SYSTEM_NCCL)
    endif()
endif()

add_custom_target(torch_python_stubs DEPENDS "${TORCH_SRC_DIR}/__init__.pyi")
# For Declarations.yaml dependency
add_dependencies(torch_python_stubs ATEN_CPU_FILES_GEN_TARGET)
add_custom_command(
    OUTPUT
    "${TORCH_SRC_DIR}/__init__.pyi"
    COMMAND
    "${PYTHON_EXECUTABLE}" -mtools.pyi.gen_pyi
      --declarations-path "${CMAKE_BINARY_DIR}/aten/src/ATen/Declarations.yaml"
    DEPENDS
    "${CMAKE_BINARY_DIR}/aten/src/ATen/Declarations.yaml"
    "${TORCH_SRC_DIR}/__init__.pyi.in"
    WORKING_DIRECTORY
    "${TORCH_ROOT}"
    )

add_library(torch_python SHARED ${TORCH_PYTHON_SRCS})

# Required workaround for generated sources
# See https://samthursfield.wordpress.com/2015/11/21/cmake-dependencies-between-targets-and-files-and-custom-commands/#custom-commands-in-different-directories
add_dependencies(torch_python generate-torch-sources)
set_source_files_properties(
    ${GENERATED_THNN_SOURCES}
    ${GENERATED_CXX_PYTHON}
    PROPERTIES GENERATED TRUE
    )

target_compile_definitions(torch_python PUBLIC _THP_CORE)

add_dependencies(torch_python torch_python_stubs)

target_link_libraries(torch_python ${TORCH_PYTHON_LINK_LIBRARIES})

target_compile_definitions(torch_python PRIVATE ${TORCH_PYTHON_COMPILE_DEFINITIONS})

target_compile_options(torch_python PRIVATE ${TORCH_PYTHON_COMPILE_OPTIONS})

target_include_directories(torch_python PUBLIC ${TORCH_PYTHON_INCLUDE_DIRECTORIES})


if (NOT TORCH_PYTHON_LINK_FLAGS STREQUAL "")
    set_target_properties(torch_python PROPERTIES LINK_FLAGS ${TORCH_PYTHON_LINK_FLAGS})
endif()

install(TARGETS torch_python DESTINATION "${TORCH_INSTALL_LIB_DIR}")
