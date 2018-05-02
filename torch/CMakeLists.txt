if (CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO)
  if (NOT BUILD_TORCH)
    return()
  endif()
else()
  cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
  project(Torch CXX C)
  include(CMakeDependentOption)
  option(BUILD_SHARED_LIBS "Build shared libs." ON)
  option(BUILD_PYTHON "Build Python binaries" ON)
  option(USE_CUDA "Use CUDA" ON)
  option(USE_CUDNN "Use cuDNN" ON)  # New option
  option(USE_NCCL "Use NCCL" ON)
  option(USE_MKLML "Use MKLML interface in MKL BLAS" ON)
  option(USE_NNPACK "Use NNPACK" ON)
  option(USE_DISTRIBUTED "Use THD (distributed)" OFF)  # New option
  option(USE_DISTRIBUTED_MW "Use THD (distributed) MW" OFF)  # New option
  option(USE_SYSTEM_NCCL "Use system-wide NCCL" OFF)  # New option
  option(USE_GLOO_IBVERBS "Use Gloo IB verbs for distributed support" OFF)  # New option

  # Legacy options, which we will eventually remove
  cmake_dependent_option(
      WITH_CUDA "Legacy CUDA" ON
      "USE_CUDA" OFF)
  cmake_dependent_option(
      NO_CUDA "Legacy no CUDA" OFF
      "USE_CUDA" ON)
  cmake_dependent_option(
      NO_PYTHON "Legacy Python" OFF
      "BUILD_PYTHON" ON)
  cmake_dependent_option(
      WITH_CUDNN "Legacy cuDNN" ON
      "USE_CUDNN" OFF)
  cmake_dependent_option(
      WITH_NCCL "Legacy NCCL" ON
      "USE_NCCL" OFF)
  cmake_dependent_option(
      WITH_DISTRIBUTED "Legacy THD (distributed)" ON
      "USE_DISTRIBUTED" OFF)
  cmake_dependent_option(
      WITH_DISTRIBUTED_MW "Legacy THD (distributed) MW" ON
      "USE_DISTRIBUTED_MW" OFF)
  cmake_dependent_option(
      WITH_GLOO_IBVERBS "Legacy Gloo IB verbs for distributed support" ON
      "USE_GLOO_IBVERBS" OFF)

  # ---[ CMake scripts + modules
  list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../cmake/Modules)

  if (MSVC AND ${BUILD_SHARED_LIBS})
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
  endif()

  # ---[ ATen dependency
  add_subdirectory(../aten aten)
  #include_directories(${ATEN_INCLUDE_DIR})
  #find_path(ATEN_INCLUDE_DIR ATen/ATen.h PATHS "${ATEN_PATH}/src/" NO_DEFAULT_PATH)
  #find_path(ATEN_BUILD_INCLUDE_DIR ATen/Type.h PATHS "${ATEN_BUILD_PATH}/src/ATen" NO_DEFAULT_PATH)
  #find_library(ATEN_LIBRARY ATen PATHS "${ATEN_BUILD_PATH}/src/ATen" NO_DEFAULT_PATH)

  # ---[ Misc checks to cope with various compiler modes
  #include(../cmake/MiscCheck.cmake)
  #include(../cmake/BuildVariables.cmake)

  # ---[ Dependencies
  #include(../cmake/Dependencies.cmake)

  # ---[ Utils
  # TODO: merge the following 3 files into cmake/public/utils.cmake.
  include(../cmake/Utils.cmake)
  include(../cmake/public/utils.cmake)

  # ---[ Python + Numpy
  if (BUILD_PYTHON)
    #set(Python_ADDITIONAL_VERSIONS 2.8 2.7 2.6)
    find_package(PythonInterp 3)
    find_package(PythonLibs 3)
    #find_package(PythonInterp 2.7)
    #find_package(PythonLibs 2.7)
    find_package(NumPy REQUIRED)
    if (PYTHONINTERP_FOUND AND PYTHONLIBS_FOUND AND NUMPY_FOUND)
      include_directories(${PYTHON_INCLUDE_DIR} ${NUMPY_INCLUDE_DIR})
    else()
      message(WARNING "Python dependencies not met. Not compiling with python. Suppress this warning with -DBUILD_PYTHON=OFF")
      caffe2_update_option(BUILD_PYTHON OFF)
    endif()
  endif()

  # ---[ pybind11
  if (BUILD_PYTHON)
    find_package(pybind11)
    if (pybind11_FOUND)
      include_directories(${pybind11_INCLUDE_DIRS})
    else()
      include_directories(${PROJECT_SOURCE_DIR}/third_party/pybind11/include)
    endif()
  endif()

  # Set default build type
  if (NOT CMAKE_BUILD_TYPE)
      message(STATUS "Build type not set - defaulting to Release")
      set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build from: Debug Release RelWithDebInfo MinSizeRel Coverage." FORCE)
  endif()
endif()

# ---[ Set link flag, handle additional deps for gcc 4.8 and above
if (CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.8.0 AND NOT ANDROID)
  message(STATUS "GCC ${CMAKE_CXX_COMPILER_VERSION}: Adding gcc and gcc_s libs to link line")
  list(APPEND Caffe2_DEPENDENCY_LIBS gcc_s gcc)
endif()

# ---[ Build flags
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)
if (NOT MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -fPIC")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-narrowing")
  # Additional PyTorch requirements
  # Python 2.6 requires -fno-strict-aliasing, see
  # http://legacy.python.org/dev/peps/pep-3123/
  # We also depend on it in our code (even Python 3).
  # Clang has an unfixed bug leading to spurious missing
  # braces warnings, see
  # https://bugs.llvm.org/show_bug.cgi?id=21629
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers -Wno-write-strings -Wno-zero-length-array -fno-strict-aliasing -Wno-missing-braces")

# TODO
#  if check_env_flag('WERROR')
#    extra_compile_args.append('-Werror')
#  endif()

else()
  foreach(flag_var
      CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
      CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
    if (${CAFFE2_USE_MSVC_STATIC_RUNTIME})
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif(${flag_var} MATCHES "/MD")
    else()
      if (${flag_var} MATCHES "/MT")
        string(REGEX REPLACE "/MT" "/MD" ${flag_var} "${${flag_var}}")
      endif()
    endif()
    # /bigobj increases number of sections in .obj file, which is needed to link
    # against libraries in Python 2.7 under Windows
    set(${flag_var} "${${flag_var}} /MP /bigobj")
    # Additional PyTorch requirements
    # /Z7 turns on symbolic debugging information in .obj files
    # /EHa is about native C++ catch support for asynchronous
    # structured exception handling (SEH)
    # /DNOMINMAX removes builtin min/max functions
    # /wdXXXX disables warning no. XXXX
    set(${flag_var} "${${flag_var}} /Z7 /EHa /DNOMINMAX /wd4267 /wd4251 /wd4522 /wd4522 /wd4838 /wd4305 /wd4244 /wd4190 /wd4101 /wd4996 /wd4275")

# TODO
#    if (${CMAKE_BUILD_TYPE} MATCHES "Debug")
#        if (MSVC)
#            extra_link_args.append('/DEBUG:FULL')
#        else()
#            extra_compile_args += ['-O0', '-g']
#            extra_link_args += ['-O0', '-g']
#        endif()
#    endif()

  endforeach(flag_var)
endif()

# Prefix path to Caffe2 headers.
# If a directory containing installed Caffe2 headers was inadvertently
# added to the list of include directories, prefixing
# PROJECT_SOURCE_DIR means this source tree always takes precedence.
include_directories(BEFORE ${PROJECT_SOURCE_DIR})

# Prefix path to generated Caffe2 headers.
# These need to take precedence over their empty counterparts located
# in PROJECT_SOURCE_DIR.
include_directories(BEFORE ${PROJECT_BINARY_DIR})

# Add PyTorch include paths
#include_directories(${PROJECT_SOURCE_DIR}/../torch/lib)
include_directories(${PROJECT_SOURCE_DIR}/..)
#tmp_install_path = lib_path + "/tmp_install"
#include_dirs += [
#    cwd,
#    os.path.join(cwd, "torch", "csrc"),
#    third_party_path + "/pybind11/include",
#    tmp_install_path + "/include",
#    tmp_install_path + "/include/TH",
#    tmp_install_path + "/include/THNN",
#    tmp_install_path + "/include/ATen",
#]

# we specify exact lib names to avoid conflict with lua-torch installs
#ATEN_LIB = os.path.join(lib_path, 'libATen.so')
#THD_LIB = os.path.join(lib_path, 'libTHD.a')
#NCCL_LIB = os.path.join(lib_path, 'libnccl.so.1')

# static library only
#NANOPB_STATIC_LIB = os.path.join(lib_path, 'libprotobuf-nanopb.a')

#if (APPLE)
#    ATEN_LIB = os.path.join(lib_path, 'libATen.dylib')
#    NCCL_LIB = os.path.join(lib_path, 'libnccl.1.dylib')
#endif()
#if (MSVC)
#    ATEN_LIB = os.path.join(lib_path, 'ATen.lib')
#    if (BUILD_DEBUG)
#        NANOPB_STATIC_LIB = os.path.join(lib_path, 'protobuf-nanopbd.lib')
#    else()
#        NANOPB_STATIC_LIB = os.path.join(lib_path, 'protobuf-nanopb.lib')
#    endif()
#endif()
#
#MAIN_COMPILE_ARGS = ['-D_THP_CORE']
#main_libraries = ['shm']
#main_link_args = [ATEN_LIB, NANOPB_STATIC_LIB]


# ---[ torch main source files
set(TORCH_SRCS
    "csrc/assertions.cpp"
    "csrc/byte_order.cpp"
    "csrc/torch.cpp"
    "csrc/utils.cpp"
    "csrc/utils/cuda_lazy_init.cpp"
    "csrc/utils/device.cpp"
    "csrc/utils/invalid_arguments.cpp"
    "csrc/utils/object_ptr.cpp"
    "csrc/utils/python_arg_parser.cpp"
    "csrc/utils/tensor_list.cpp"
    "csrc/utils/tensor_new.cpp"
    "csrc/utils/tensor_numpy.cpp"
    "csrc/utils/tensor_dtypes.cpp"
    "csrc/utils/tensor_layouts.cpp"
    "csrc/utils/tensor_types.cpp"
    "csrc/utils/tuple_parser.cpp"
    "csrc/utils/tensor_apply.cpp"
    "csrc/utils/tensor_conversion_dispatch.cpp"
    "csrc/utils/tensor_flatten.cpp"
    "csrc/utils/variadic.cpp"
    "csrc/allocators.cpp"
    "csrc/serialization.cpp"
    "csrc/jit/init.cpp"
    "csrc/jit/interpreter.cpp"
    "csrc/jit/ir.cpp"
    "csrc/jit/fusion_compiler.cpp"
    "csrc/jit/graph_executor.cpp"
    "csrc/jit/python_ir.cpp"
    "csrc/jit/test_jit.cpp"
    "csrc/jit/tracer.cpp"
    "csrc/jit/tracer_state.cpp"
    "csrc/jit/python_tracer.cpp"
    "csrc/jit/passes/shape_analysis.cpp"
    "csrc/jit/interned_strings.cpp"
    "csrc/jit/type.cpp"
    "csrc/jit/export.cpp"
    "csrc/jit/autodiff.cpp"
    "csrc/jit/interpreter_autograd_function.cpp"
    "csrc/jit/python_arg_flatten.cpp"
    "csrc/jit/python_compiled_function.cpp"
    "csrc/jit/variable_flags.cpp"
    "csrc/jit/passes/create_autodiff_subgraphs.cpp"
    "csrc/jit/passes/graph_fuser.cpp"
    "csrc/jit/passes/onnx.cpp"
    "csrc/jit/passes/dead_code_elimination.cpp"
    "csrc/jit/passes/lower_tuples.cpp"
    "csrc/jit/passes/common_subexpression_elimination.cpp"
    "csrc/jit/passes/peephole.cpp"
    "csrc/jit/passes/inplace_check.cpp"
    "csrc/jit/passes/canonicalize.cpp"
    "csrc/jit/passes/batch_mm.cpp"
    "csrc/jit/passes/onnx/peephole.cpp"
    "csrc/jit/generated/aten_dispatch.cpp"
    "csrc/jit/script/lexer.cpp"
    "csrc/jit/script/compiler.cpp"
    "csrc/jit/script/module.cpp"
    "csrc/jit/script/init.cpp"
    "csrc/jit/script/python_tree_views.cpp"
    "csrc/autograd/init.cpp"
    "csrc/autograd/grad_mode.cpp"
    "csrc/autograd/engine.cpp"
    "csrc/autograd/function.cpp"
    "csrc/autograd/variable.cpp"
    "csrc/autograd/saved_variable.cpp"
    "csrc/autograd/input_buffer.cpp"
    "csrc/autograd/profiler.cpp"
    "csrc/autograd/python_function.cpp"
    "csrc/autograd/python_cpp_function.cpp"
    "csrc/autograd/python_variable.cpp"
    "csrc/autograd/python_variable_indexing.cpp"
    "csrc/autograd/python_legacy_variable.cpp"
    "csrc/autograd/python_engine.cpp"
    "csrc/autograd/python_hook.cpp"
    "csrc/autograd/generated/VariableType.cpp"
    "csrc/autograd/generated/Functions.cpp"
    "csrc/autograd/generated/python_torch_functions.cpp"
    "csrc/autograd/generated/python_variable_methods.cpp"
    "csrc/autograd/generated/python_functions.cpp"
    "csrc/autograd/generated/python_nn_functions.cpp"
    "csrc/autograd/functions/basic_ops.cpp"
    "csrc/autograd/functions/tensor.cpp"
    "csrc/autograd/functions/accumulate_grad.cpp"
    "csrc/autograd/functions/special.cpp"
    "csrc/autograd/functions/utils.cpp"
    "csrc/autograd/functions/init.cpp"
    "csrc/nn/THNN.cpp"
    "csrc/tensor/python_tensor.cpp"
    "csrc/onnx/onnx.pb.cpp"
    "csrc/onnx/onnx.cpp"
    "csrc/onnx/init.cpp"
    )

if (BUILD_PYTHON)
    list(APPEND TORCH_SRCS
        "csrc/PtrWrapper.cpp"
        "csrc/Module.cpp"
        "csrc/Generator.cpp"
        "csrc/Size.cpp"
        "csrc/Dtype.cpp"
        "csrc/Device.cpp"
        "csrc/Exceptions.cpp"
        "csrc/Layout.cpp"
        "csrc/Storage.cpp"
        "csrc/DataLoader.cpp"
        "csrc/DynamicTypes.cpp"
        )
endif()

if (USE_DISTRIBUTED)
    list(APPEND TORCH_SRCS
        "csrc/distributed/Module.cpp"
        )
    if (USE_DISTRIBUTED_MW)
        list(APPEND TORCH_SRCS
            "csrc/distributed/Tensor.cpp"
            "csrc/distributed/Storage.cpp"
            )
    endif()
    #include_dirs += [tmp_install_path + "/include/THD"]
    #main_link_args += [THD_LIB]
endif()

if (USE_CUDA)
    # Add to the core library source
    list(APPEND TORCH_SRCS
        "csrc/cuda/Module.cpp"
        "csrc/cuda/Storage.cpp"
        "csrc/cuda/Stream.cpp"
        "csrc/cuda/utils.cpp"
        "csrc/cuda/comm.cpp"
        "csrc/cuda/python_comm.cpp"
        "csrc/cuda/serialization.cpp"
        "csrc/nn/THCUNN.cpp"
        )

    # Add to the library references
    # TODO
endif()

if (USE_NCCL)
    #if (USE_SYSTEM_NCCL)
    #    main_link_args += [NCCL_SYSTEM_LIB]
    #    include_dirs.append(NCCL_INCLUDE_DIR)
    #else()
    #    main_link_args += [NCCL_LIB]
    #endif()
    list(APPEND TORCH_SRCS
        "csrc/cuda/nccl.cpp"
        "csrc/cuda/python_nccl.cpp"
        )
endif()

if (USE_CUDNN)
# TODO
    # main_libraries += [CUDNN_LIBRARY]
    # # NOTE: these are at the front, in case there's another cuDNN in CUDA path
    # include_dirs.insert(0, CUDNN_INCLUDE_DIR)
    #if (NOT MSVC)
    #    extra_link_args.insert(0, '-Wl,-rpath,' + CUDNN_LIB_DIR)
    #endif()
endif()

# TODO
#if os.getenv('PYTORCH_BINARY_BUILD') and platform.system() == 'Linux':
#    print('PYTORCH_BINARY_BUILD found. Static linking libstdc++ on Linux')
#    # get path of libstdc++ and link manually.
#    # for reasons unknown, -static-libstdc++ doesn't fully link some symbols
#    CXXNAME = os.getenv('CXX', 'g++')
#    STDCPP_LIB = subprocess.check_output([CXXNAME, '-print-file-name=libstdc++.a'])
#    STDCPP_LIB = STDCPP_LIB[:-1]
#    if type(STDCPP_LIB) != str:  # python 3
#        STDCPP_LIB = STDCPP_LIB.decode(sys.stdout.encoding)
#    main_link_args += [STDCPP_LIB]
#    version_script = os.path.abspath("tools/pytorch.version")
#    extra_link_args += ['-Wl,--version-script=' + version_script]

# TODO
#if (USE_CUDA)
#    nvtoolext_lib_name = None
#    if (MSVC)
#        cuda_lib_path = CUDA_HOME + '/lib/x64/'
#        nvtoolext_lib_path = NVTOOLEXT_HOME + '/lib/x64/'
#        nvtoolext_include_path = os.path.join(NVTOOLEXT_HOME, 'include')
#
#        library_dirs.append(nvtoolext_lib_path)
#        include_dirs.append(nvtoolext_include_path)
#
#        nvtoolext_lib_name = 'nvToolsExt64_1'
#
#        # MSVC doesn't support runtime symbol resolving, `nvrtc` and `cuda` should be linked
#        main_libraries += ['nvrtc', 'cuda']
#    else()
#        cuda_lib_dirs = ['lib64', 'lib']
#
#        for lib_dir in cuda_lib_dirs:
#            cuda_lib_path = os.path.join(CUDA_HOME, lib_dir)
#            if os.path.exists(cuda_lib_path):
#                break
#        extra_link_args.append('-Wl,-rpath,' + cuda_lib_path)
#
#        nvtoolext_lib_name = 'nvToolsExt'
#    endif()
#
#    library_dirs.append(cuda_lib_path)
#    cuda_include_path = os.path.join(CUDA_HOME, 'include')
#    include_dirs.append(cuda_include_path)
#    include_dirs.append(tmp_install_path + "/include/THCUNN")
#    extra_compile_args += ['-DWITH_CUDA']
#    extra_compile_args += ['-DCUDA_LIB_PATH=' + cuda_lib_path]
#    main_libraries += ['cudart', nvtoolext_lib_name]
#endif()

# ---[ torch._nvrtc library build
if (USE_CUDA)
# TODO
#    thnvrtc_link_flags = extra_link_args + [make_relative_rpath('lib')]
#    if (LINUX)
#        thnvrtc_link_flags = thnvrtc_link_flags + ['-Wl,--no-as-needed']
#    endif()
#    # these have to be specified as -lcuda in link_flags because they
#    # have to come right after the `no-as-needed` option
#    if (MSVC)
#        thnvrtc_link_flags += ['cuda.lib', 'nvrtc.lib']
#    else()
#        thnvrtc_link_flags += ['-lcuda', '-lnvrtc']
#    endif()
#    cuda_stub_path = [cuda_lib_path + '/stubs']
#    if (APPLE)
#        # on macOS this is where the CUDA stub is installed according to the manual
#        cuda_stub_path = ["/usr/local/cuda/lib"]
#    endif()
# TODO
#    THNVRTC = Extension("torch._nvrtc",
#                        sources=['torch/csrc/nvrtc.cpp'],
#                        language='c++',
#                        include_dirs=include_dirs,
#                        library_dirs=library_dirs + cuda_stub_path,
#                        extra_link_args=thnvrtc_link_flags,
#                        )

    set(TORCH_NVRTC_SRCS "csrc/nvrtc.cpp")
    add_library(torch_nvrtc SHARED ${TORCH_NVRTC_SRCS})
    # target_link_libraries(torch_nvrtc PUBLIC caffe2_library)
    target_include_directories(torch_nvrtc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
    install(TARGETS torch_nvrtc DESTINATION lib)
    caffe2_interface_library(torch_nvrtc torch_nvrtc_library)

endif()

if (BUILD_PYTHON)
    if (NOT MSVC)
        # ---[ torch._dl library build
        set(TORCH_DL_SRCS "csrc/dl.c")
        add_library(torch_dl ${TORCH_DL_SRCS})
        # target_link_libraries(torch_dl PUBLIC caffe2_library)
        # target_include_directories(torch_dl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
        install(TARGETS torch_dl DESTINATION lib)
        caffe2_interface_library(torch_dl torch_dl_library)
    endif()
endif()

# ---[ torch._C library build
add_library(torch ${TORCH_SRCS})
# target_link_libraries(torch PUBLIC caffe2_library)
#target_include_directories(torch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(COMMON_INCLUDES
  "${ATEN_INCLUDE_DIR}"
  "${ATEN_INCLUDE_DIR}/TH"
#  "${ATEN_BUILD_INCLUDE_DIR}"
#  "${ATEN_BUILD_PATH}/src/TH"
#  "${ATEN_INCLUDE_DIR}/src/TH"
  "${CMAKE_CURRENT_SOURCE_DIR}/.."
  "${CUDA_INCLUDE_DIRS}")
target_include_directories(
  torch
  PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/../aten/src"
  "${CMAKE_CURRENT_SOURCE_DIR}/../aten/src/TH"
  "${ATen_BUILD_DIR}/src/TH"
#  "${ATen_SOURCE_DIR}/src/TH"
#  "${ATEN_INCLUDE_DIR}/TH"
#  "${CMAKE_CURRENT_SOURCE_DIR}/../aten/src/TH"
##  "${COMMON_INCLUDES}"
#  "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/nanopb"
  )
#target_compile_options(torch PRIVATE -Wall -Wextra)
target_link_libraries(torch ATen)

install(TARGETS torch DESTINATION lib)
caffe2_interface_library(torch torch_library)
