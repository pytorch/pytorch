# ---[ Generate and install header and cpp files
include(../cmake/Codegen.cmake)

# ---[ Declare source file lists

# ---[ ATen build
if (NOT BUILD_ATEN_MOBILE)
  set(__caffe2_CMAKE_POSITION_INDEPENDENT_CODE ${CMAKE_POSITION_INDEPENDENT_CODE})
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  set(AT_LINK_STYLE INTERFACE)
  add_subdirectory(../aten aten)
  set(CMAKE_POSITION_INDEPENDENT_CODE ${__caffe2_CMAKE_POSITION_INDEPENDENT_CODE})

  # Generate the headers wrapped by our operator
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/contrib/aten/aten_op.h
  COMMAND
  ${PYCMD} ${CMAKE_CURRENT_SOURCE_DIR}/contrib/aten/gen_op.py
    --aten_root=${CMAKE_CURRENT_SOURCE_DIR}/../aten
    --template_dir=${CMAKE_CURRENT_SOURCE_DIR}/contrib/aten
    --yaml_dir=${CMAKE_CURRENT_BINARY_DIR}/../aten/src/ATen
    --install_dir=${CMAKE_CURRENT_BINARY_DIR}/contrib/aten
  DEPENDS
  ATEN_CPU_FILES_GEN_TARGET
  ${CMAKE_CURRENT_SOURCE_DIR}/contrib/aten/gen_op.py
  ${CMAKE_CURRENT_SOURCE_DIR}/contrib/aten/aten_op_template.h)

  add_custom_target(__aten_op_header_gen
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/contrib/aten/aten_op.h)
  add_library(aten_op_header_gen INTERFACE)
  add_dependencies(aten_op_header_gen __aten_op_header_gen)

  # Add source, includes, and libs to lists
  list(APPEND Caffe2_CPU_SRCS ${ATen_CPU_SRCS})
  list(APPEND Caffe2_GPU_SRCS ${ATen_CUDA_SRCS})
  list(APPEND Caffe2_CPU_TEST_SRCS ${ATen_CPU_TEST_SRCS})
  list(APPEND Caffe2_GPU_TEST_SRCS ${ATen_CUDA_TEST_SRCS})
  list(APPEND Caffe2_CPU_TEST_SRCS ${ATen_CORE_TEST_SRCS})
  list(APPEND Caffe2_CPU_INCLUDE ${ATen_CPU_INCLUDE})
  list(APPEND Caffe2_GPU_INCLUDE ${ATen_CUDA_INCLUDE})
  list(APPEND Caffe2_DEPENDENCY_LIBS ${ATen_CPU_DEPENDENCY_LIBS})
  list(APPEND Caffe2_CUDA_DEPENDENCY_LIBS ${ATen_CUDA_DEPENDENCY_LIBS})
  list(APPEND Caffe2_DEPENDENCY_INCLUDE ${ATen_THIRD_PARTY_INCLUDE})

  IF(USE_ROCM)
    # Set the HIP Variables
    set(Caffe2_HIP_SRCS ${Caffe2_HIP_SRCS} ${ATen_CUDA_SRCS})
    set(Caffe2_HIP_INCLUDES ${Caffe2_HIP_INCLUDES} ${Caffe2_GPU_INCLUDE})
  ENDIF(USE_ROCM)
else()
  # Only add "ATen Core", a minimal, easy-to-compile fragment of ATen.
  # This codepath should only be exercised by the Android build.
  add_subdirectory(../aten/src/ATen/core ATen_core)
  list(APPEND Caffe2_CPU_SRCS ${ATen_CORE_SRCS})
  list(APPEND Caffe2_CPU_INCLUDE ${ATen_CORE_INCLUDE})
  list(APPEND Caffe2_CPU_TEST_SRCS ${ATen_CORE_TEST_SRCS})
  # See cmake/Codegen.cmake for header installation
endif()

# ---[ Torch build
if(BUILD_TORCH)
  add_subdirectory(../torch torch)
endif()

# ---[ Caffe2 build
# Note: the folders that are being commented out have not been properly
# addressed yet.

if(NOT BUILD_ATEN_ONLY)
  add_subdirectory(proto)
  add_subdirectory(contrib)
  add_subdirectory(core)
  add_subdirectory(utils)
  add_subdirectory(predictor)
  add_subdirectory(core/nomnigraph)
  add_subdirectory(core/nomscheduler)
  add_subdirectory(core/dispatch)
  add_subdirectory(serialize)
  if (USE_NVRTC)
    add_subdirectory(cuda_rtc)
  endif()
  add_subdirectory(db)
  add_subdirectory(distributed)
  # add_subdirectory(experiments) # note, we may remove this folder at some point
  add_subdirectory(ideep)
  add_subdirectory(image)
  add_subdirectory(video)
  add_subdirectory(mobile)
  add_subdirectory(mpi)
  add_subdirectory(observers)
  add_subdirectory(onnx)
  if (BUILD_CAFFE2_OPS)
    add_subdirectory(operators)
    add_subdirectory(operators/rnn)
    if (USE_FBGEMM)
      add_subdirectory(quantization)
      add_subdirectory(quantization/server)
    endif()
    if (USE_QNNPACK)
      add_subdirectory(operators/quantized)
    endif()
  endif()
  add_subdirectory(opt)
  add_subdirectory(perfkernels)
  add_subdirectory(python)
  add_subdirectory(queue)
  add_subdirectory(sgd)
  add_subdirectory(share)
  # add_subdirectory(test) # todo: use caffe2_gtest_main instead of gtest_main because we will need to call GlobalInit
  add_subdirectory(transforms)
endif()

# Advanced: if we have white list specified, we will do intersections for all
# main lib srcs.
if (CAFFE2_WHITELISTED_FILES)
  caffe2_do_whitelist(Caffe2_CPU_SRCS CAFFE2_WHITELISTED_FILES)
  caffe2_do_whitelist(Caffe2_GPU_SRCS CAFFE2_WHITELISTED_FILES)
  caffe2_do_whitelist(Caffe2_HIP_SRCS CAFFE2_WHITELISTED_FILES)
endif()

# Debug messages - if you want to get a list of source files, enable the
# following.
if (FALSE)
  message(STATUS "CPU sources: ")
  foreach(tmp ${Caffe2_CPU_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "GPU sources: ")
  foreach(tmp ${Caffe2_GPU_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "CPU include: ")
  foreach(tmp ${Caffe2_CPU_INCLUDE})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "GPU include: ")
  foreach(tmp ${Caffe2_GPU_INCLUDE})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "CPU test sources: ")
  foreach(tmp ${Caffe2_CPU_TEST_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "GPU test sources: ")
  foreach(tmp ${Caffe2_GPU_TEST_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "HIP sources: ")
  foreach(tmp ${Caffe2_HIP_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "HIP test sources: ")
  foreach(tmp ${Caffe2_HIP_TEST_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "ATen CPU test sources: ")
  foreach(tmp ${ATen_CPU_TEST_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "ATen CUDA test sources: ")
  foreach(tmp ${ATen_CUDA_TEST_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()
endif()

# ---[ List of libraries to link with
if(NOT BUILD_ATEN_ONLY)
  add_library(caffe2_protos STATIC $<TARGET_OBJECTS:Caffe2_PROTO>)
  add_dependencies(caffe2_protos Caffe2_PROTO)
  # If we are going to link protobuf locally inside caffe2 libraries, what we will do is
  # to create a helper static library that always contains libprotobuf source files, and
  # link the caffe2 related dependent libraries to it.
  target_include_directories(caffe2_protos INTERFACE $<INSTALL_INTERFACE:include>)
  # Reason for this public dependency is as follows:
  # (1) Strictly speaking, we should not expose any Protobuf related functions. We should
  #     only use function interfaces wrapped with our own public API, and link protobuf
  #     locally.
  # (2) However, currently across the Caffe2 codebase, we have extensive use of protobuf
  #     functionalities. For example, not only libcaffe2.so uses it, but also other
  #     binaries such as python extensions etc. As a result, we will have to have a
  #     transitive dependency to libprotobuf.
  #
  # Good thing is that, if we specify CAFFE2_LINK_LOCAL_PROTOBUF, then we do not need to
  # separately deploy protobuf binaries - libcaffe2.so will contain all functionalities
  # one needs. One can verify this via ldd.
  #
  # TODO item in the future includes:
  # (1) Enable using lite protobuf
  # (2) Properly define public API that do not directly depend on protobuf itself.
  # (3) Expose the libprotobuf.a file for dependent libraries to link to.
  #
  # What it means for users/developers?
  # (1) Users: nothing affecting the users, other than the fact that CAFFE2_LINK_LOCAL_PROTOBUF
  #     avoids the need to deploy protobuf.
  # (2) Developers: if one simply uses core caffe2 functionality without using protobuf,
  #     nothing changes. If one has a dependent library that uses protobuf, then one needs to
  #     have the right protobuf version as well as linking to libprotobuf.a.
  target_link_libraries(caffe2_protos PUBLIC protobuf::libprotobuf)
endif()

# Compile exposed libraries.
add_library(caffe2 ${Caffe2_CPU_SRCS})
if (NOT WIN32)
  target_compile_options(caffe2 PRIVATE "-fvisibility=hidden")
endif()

if(NOT BUILD_ATEN_ONLY)
  caffe2_interface_library(caffe2_protos caffe2_protos_whole)
  target_link_libraries(caffe2 PRIVATE caffe2_protos_whole)
  if (${CAFFE2_LINK_LOCAL_PROTOBUF})
    target_link_libraries(caffe2 INTERFACE protobuf::libprotobuf)
  else()
    target_link_libraries(caffe2 PUBLIC protobuf::libprotobuf)
  endif()
endif()

target_link_libraries(caffe2 PUBLIC c10)
target_link_libraries(caffe2 PUBLIC ${Caffe2_PUBLIC_DEPENDENCY_LIBS})
target_link_libraries(caffe2 PRIVATE ${Caffe2_DEPENDENCY_LIBS})
target_link_libraries(caffe2 PRIVATE ${Caffe2_DEPENDENCY_WHOLE_LINK_LIBS})
target_include_directories(caffe2 INTERFACE $<INSTALL_INTERFACE:include>)
target_include_directories(caffe2 PRIVATE ${Caffe2_CPU_INCLUDE})
target_include_directories(caffe2 SYSTEM PRIVATE "${Caffe2_DEPENDENCY_INCLUDE}")
# Set standard properties on the target
torch_set_target_props(caffe2)

if (MSVC)
target_compile_options(caffe2 INTERFACE "-std=c++11")
else()
target_compile_options(caffe2 INTERFACE "$<$<COMPILE_LANGUAGE:CXX>:-std=c++11>")
endif()

target_compile_options(caffe2 PRIVATE "-DCAFFE2_BUILD_MAIN_LIB")
if (MSVC AND NOT BUILD_SHARED_LIBS)
  # Note [Supporting both static and dynamic libraries on Window]
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # A Windows library may be distributed as either a static or dynamic
  # library.  The chosen distribution mechanism affects how you setup
  # the headers for the library: if you statically link a function,
  # all you need is an ordinary signature:
  #
  #     void f();
  #
  # But if you *dynamically* link it, then you must provide a __declspec
  # specifying that it should be imported from a DLL:
  #
  #     __declspec(dllimport) void f();
  #
  # Mixing the two situations will not work: if you specify dllimport
  # while statically linking, the linker will complain it cannot find
  # the __imp_f symbol (which serve as the DLL entrypoint); if you
  # fail to specify dllimport for a symbol that's coming from a DLL,
  # the linker will complain that it can't find f.  Joy!
  #
  # Most places on the Internet, you will find people have written
  # their headers under the assumption that the application will
  # only ever be dynamically linked, as they define a macro which
  # tags a function as __declspec(dllexport) if you are actually
  # building the library, and __declspec(dllimport) otherwise.  But
  # if you want these headers to also work if you are linking against
  # a static library, you need a way to avoid adding these __declspec's
  # at all.  And that "mechanism" needs to apply to any downstream
  # libraries/executables which are going to link against your library.
  #
  #   As an aside, why do we need to support both modes?
  #   For historical reasons, PyTorch ATen on Windows is built dynamically,
  #   while Caffe2 on Windows is built statically (mostly because if
  #   we build it dynamically, we are over the DLL exported symbol limit--and
  #   that is because Caffe2 hasn't comprehensively annotated all symbols
  #   which cross the DLL boundary with CAFFE_API).  So any code
  #   which is used by both PyTorch and Caffe2 needs to support both
  #   modes of linking.
  #
  # So, you have a macro (call it AT_CORE_STATIC_WINDOWS) which you need to have
  # set for any downstream library/executable that transitively includes your
  # headers.  How are you going to do this?  You have two options:
  #
  #   1. Write out a config.h header which stores whether or not
  #      you are linking statically or dynamically.
  #
  #   2. Force all of users to set the the macro themselves.  If they
  #      use cmake, you can set -DAT_CORE_STATIC_WINDOWS=1 as a PUBLIC
  #      compile option, in which case cmake will automatically
  #      add the macro for you.
  #
  # Which one is better? Well, it depends: they trade off implementor
  # ease versus user ease: (1) is more work for the library author
  # but the user doesn't have to worry about it; (2) requires the user
  # to set the macro themselves... but only if they don't use cmake.
  #
  # So, which is appropriate in our situation?  In my mind, here is
  # the distinguishing factor: it is more common to distribute
  # DLLs, since they don't require you to line up the CRT version
  # (/MD, /MDd, /MT, /MTd) and MSVC version at the use site.  So,
  # if a user is already in the business of static linkage, they're
  # already in "expert user" realm.  So, I've decided that at this
  # point in time, the simplicity of implementation of (2) wins out.
  #
  # NB: This must be target_compile_definitions, not target_compile_options,
  # as the latter is not respected by nvcc
  target_compile_definitions(caffe2 PUBLIC "AT_CORE_STATIC_WINDOWS=1")
endif()
if (MSVC AND BUILD_SHARED_LIBS)
  # ONNX is linked statically and needs to be exported from this library
  # to be used externally. Make sure that references match the export.
  target_compile_options(caffe2 PRIVATE "-DONNX_BUILD_MAIN_LIB")
endif()
# Use -O2 for release builds (-O3 doesn't improve perf, and -Os results in perf regression)
target_compile_options(caffe2 PRIVATE "$<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:-O2>")
install(TARGETS caffe2 EXPORT Caffe2Targets DESTINATION lib)
caffe2_interface_library(caffe2 caffe2_library)
list(APPEND Caffe2_MAIN_LIBS caffe2_library)

# ---[ CUDA library.
if(USE_CUDA)
  set(CUDA_LINK_LIBRARIES_KEYWORD PRIVATE)
  torch_cuda_based_add_library(caffe2_gpu ${Caffe2_GPU_SRCS})
  set(CUDA_LINK_LIBRARIES_KEYWORD)
  target_link_libraries(caffe2_gpu INTERFACE caffe2::cudart)

  target_include_directories(
      caffe2_gpu INTERFACE $<INSTALL_INTERFACE:include>)
  target_include_directories(
      caffe2_gpu PRIVATE ${Caffe2_GPU_INCLUDE})
  target_link_libraries(
      caffe2_gpu PRIVATE ${Caffe2_CUDA_DEPENDENCY_LIBS})

  # These public dependencies must go after the previous dependencies, as the
  # order of the libraries in the linker call matters here when statically
  # linking; libculibos and cublas must be last.
  target_link_libraries(
      caffe2_gpu PUBLIC caffe2 ${Caffe2_PUBLIC_CUDA_DEPENDENCY_LIBS})

  # See Note [Supporting both static and dynamic libraries on Window]
  # TODO: I'm actually not sure why this is necessary, because caffe2_gpu
  # should depend on caffe2 (which itself would give us the necessary
  # macro definition).
  if (MSVC AND NOT BUILD_SHARED_LIBS)
      target_compile_options(caffe2_gpu PUBLIC "-DAT_CORE_STATIC_WINDOWS=1")
  endif()

  # NB: This must be target_compile_definitions, not target_compile_options,
  # as the latter is not respected by nvcc
  if (MSVC)
    target_compile_definitions(caffe2_gpu PRIVATE "-DCAFFE2_CUDA_BUILD_MAIN_LIB")
  endif()

  # Set standard properties on the target
  torch_set_target_props(caffe2_gpu)

  install(TARGETS caffe2_gpu EXPORT Caffe2Targets DESTINATION lib)
  caffe2_interface_library(caffe2_gpu caffe2_gpu_library)
  list(APPEND Caffe2_MAIN_LIBS caffe2_gpu_library)
endif()

# ---[ Caffe2 HIP sources.
if(USE_ROCM)
  # Call again since Caffe2_HIP_INCLUDES is extended with ATen include dirs.
  # Get Compile Definitions from the directory (FindHIP.cmake bug)
  get_directory_property(MY_DEFINITIONS COMPILE_DEFINITIONS)
  if(MY_DEFINITIONS)
    foreach(_item ${MY_DEFINITIONS})
      LIST(APPEND HIP_HCC_FLAGS "-D${_item}")
    endforeach()
  endif()

  # Call again since Caffe2_HIP_INCLUDES is extended with ATen include dirs.
  hip_include_directories(${Caffe2_HIP_INCLUDES})

  filter_list(__caffe2_hip_srcs_cpp Caffe2_HIP_SRCS "\\.(cc|cpp|cu)$")
  set_source_files_properties(${__caffe2_hip_srcs_cpp} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)

  # FindHIP.CMake checks if the SHARED flag is set and adds extra logic accordingly.
  hip_add_library(caffe2_hip ${Caffe2_HIP_SRCS})

  # Since PyTorch files contain HIP headers, these flags are required for the necessary definitions to be added.
  target_compile_options(caffe2_hip PRIVATE ${HIP_HCC_FLAGS})
  target_link_libraries(caffe2_hip PUBLIC caffe2)
  target_link_libraries(caffe2_hip PUBLIC ${Caffe2_HIP_DEPENDENCY_LIBS})

  # Since PyTorch files contain HIP headers, this is also needed to capture the includes.
  target_include_directories(caffe2_hip PRIVATE ${Caffe2_HIP_INCLUDES})
  target_include_directories(caffe2_hip INTERFACE $<INSTALL_INTERFACE:include>)

  # Set standard properties on the target
  torch_set_target_props(caffe2_hip)

  caffe2_interface_library(caffe2_hip caffe2_hip_library)
  list(APPEND Caffe2_MAIN_LIBS caffe2_hip_library)
  install(TARGETS caffe2_hip EXPORT Caffe2Targets DESTINATION lib)
endif()

# ---[ Check if warnings should be errors.
if ($ENV{WERROR})
  target_compile_options(caffe2 PRIVATE -Werror)
  if(USE_CUDA)
    target_compile_options(caffe2_gpu PRIVATE -Werror)
  endif()
endif()

# ---[ Test binaries.
if (BUILD_TEST)
  foreach(test_src ${Caffe2_CPU_TEST_SRCS})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} "${test_src}")
    target_link_libraries(${test_name} ${Caffe2_MAIN_LIBS} gtest_main)
    target_include_directories(${test_name} PRIVATE $<INSTALL_INTERFACE:include>)
    target_include_directories(${test_name} PRIVATE ${Caffe2_CPU_INCLUDE})
    add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
    if (INSTALL_TEST)
      install(TARGETS ${test_name} DESTINATION test)
    endif()
  endforeach()

  if (USE_CUDA)
    foreach(test_src ${Caffe2_GPU_TEST_SRCS})
      get_filename_component(test_name ${test_src} NAME_WE)
      cuda_add_executable(${test_name} "${test_src}")
      target_link_libraries(${test_name} ${Caffe2_MAIN_LIBS} gtest_main)
      target_include_directories(${test_name} PRIVATE $<INSTALL_INTERFACE:include>)
      target_include_directories(${test_name} PRIVATE ${Caffe2_CPU_INCLUDE})
      add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
      if (INSTALL_TEST)
        install(TARGETS ${test_name} DESTINATION test)
      endif()
    endforeach()
  endif()

  if(USE_ROCM)
    foreach(test_src ${Caffe2_HIP_TEST_SRCS})
      set_source_files_properties(${test_src} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
      get_filename_component(test_name ${test_src} NAME_WE)
      add_executable(${test_name} "${test_src}")
      target_link_libraries(${test_name} ${Caffe2_MAIN_LIBS} gtest_main)
      target_include_directories(${test_name} PRIVATE $<INSTALL_INTERFACE:include>)
      target_include_directories(${test_name} PRIVATE ${Caffe2_CPU_INCLUDE} ${Caffe2_HIP_INCLUDES})
      target_compile_options(${test_name} PRIVATE ${HIP_CXX_FLAGS})
      add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
      if (INSTALL_TEST)
        install(TARGETS ${test_name} DESTINATION test)
      endif()
    endforeach()
  endif()

  # For special tests that explicitly uses dependencies, we add them here
  if (USE_MPI)
    target_link_libraries(mpi_test ${MPI_CXX_LIBRARIES})
    if (USE_CUDA)
      target_link_libraries(mpi_gpu_test ${MPI_CXX_LIBRARIES})
    endif()
  endif()
endif()

if (BUILD_PYTHON)
  # Python site-packages
  # Get canonical directory for python site packages (relative to install
  # location).  It varies from system to system.
  # We should pin the path separator to the forward slash on Windows. 
  # More details can be seen at
  # https://github.com/pytorch/pytorch/tree/master/tools/build_pytorch_libs.bat#note-backslash-munging-on-windows
  pycmd(PYTHON_SITE_PACKAGES "
      import os
      from distutils import sysconfig
      print(sysconfig.get_python_lib(prefix='').replace(os.sep, '/'))
  ")
  SET(PYTHON_SITE_PACKAGES ${PYTHON_SITE_PACKAGES} PARENT_SCOPE) # for Summary
  # ---[ Options.
  SET(PYTHON_LIB_REL_PATH "${PYTHON_SITE_PACKAGES}" CACHE STRING "Python installation path (relative to CMake installation prefix)")
  message(STATUS "Using ${PYTHON_LIB_REL_PATH} as python relative installation path")
  # Python extension suffix
  # Try to get from python through sysconfig.get_env_var('EXT_SUFFIX') first,
  # fallback to ".pyd" if windows and ".so" for all others.
  pycmd(PY_EXT_SUFFIX "
      from distutils import sysconfig
      ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
      print(ext_suffix if ext_suffix else '')
  ")
  if("${PY_EXT_SUFFIX}" STREQUAL "")
    if (MSVC)
      set(PY_EXT_SUFFIX ".pyd")
    else()
      set(PY_EXT_SUFFIX ".so")
    endif()
  endif()

  # Allow different install locations for libcaffe2
  # For setuptools installs (that all build Python), install libcaffe2 into
  # site-packages, alongside the torch libraries. The pybind11 library needs
  # an rpath to the torch library folder
  # For cmake installs, including c++ only installs, install libcaffe2 into
  # CMAKE_INSTALL_PREFIX/lib . The pybind11 library can have a hardcoded
  # rpath
  if(APPLE)
    set(_rpath_portable_origin "@loader_path")
  else()
    set(_rpath_portable_origin $ORIGIN)
  endif(APPLE)
  set(caffe2_pybind11_rpath "${_rpath_portable_origin}")
  if(${BUILDING_WITH_TORCH_LIBS})
    # site-packages/caffe2/python/caffe2_pybind11_state
    # site-packages/torch/lib
    set(caffe2_pybind11_rpath "${_rpath_portable_origin}/../../torch/lib")
  endif(${BUILDING_WITH_TORCH_LIBS})


  # ---[ Python.
  add_library(caffe2_pybind11_state MODULE ${Caffe2_CPU_PYTHON_SRCS})
  set_target_properties(caffe2_pybind11_state PROPERTIES COMPILE_FLAGS "-fvisibility=hidden")
  set_target_properties(caffe2_pybind11_state PROPERTIES PREFIX "" DEBUG_POSTFIX "")
  set_target_properties(caffe2_pybind11_state PROPERTIES SUFFIX ${PY_EXT_SUFFIX})
  if (APPLE)
    set_target_properties(caffe2_pybind11_state PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
  endif()
  target_include_directories(caffe2_pybind11_state PRIVATE $<INSTALL_INTERFACE:include>)
  target_include_directories(caffe2_pybind11_state PRIVATE ${Caffe2_CPU_INCLUDE})

  target_link_libraries(
      caffe2_pybind11_state caffe2_library)
  if (WIN32)
    target_link_libraries(caffe2_pybind11_state ${PYTHON_LIBRARIES})
  endif(WIN32)

  # Install caffe2_pybind11_state(_gpu|hip) in site-packages/caffe2/python,
  # so it needs an rpath to find libcaffe2
  set_target_properties(
      caffe2_pybind11_state PROPERTIES LIBRARY_OUTPUT_DIRECTORY
      ${CMAKE_BINARY_DIR}/caffe2/python)
  install(TARGETS caffe2_pybind11_state DESTINATION "${PYTHON_LIB_REL_PATH}/caffe2/python")
  set_target_properties(caffe2_pybind11_state PROPERTIES INSTALL_RPATH "${caffe2_pybind11_rpath}")

  if(USE_CUDA)
    add_library(caffe2_pybind11_state_gpu MODULE ${Caffe2_GPU_PYTHON_SRCS})
    set_target_properties(caffe2_pybind11_state_gpu PROPERTIES COMPILE_FLAGS "-fvisibility=hidden")
    set_target_properties(caffe2_pybind11_state_gpu PROPERTIES PREFIX "" DEBUG_POSTFIX "")
    set_target_properties(caffe2_pybind11_state_gpu PROPERTIES SUFFIX ${PY_EXT_SUFFIX})
    if (APPLE)
      set_target_properties(caffe2_pybind11_state_gpu PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
    endif()
    target_include_directories(caffe2_pybind11_state_gpu PRIVATE $<INSTALL_INTERFACE:include>)
    target_include_directories(caffe2_pybind11_state_gpu PRIVATE ${Caffe2_CPU_INCLUDE})
    target_link_libraries(
        caffe2_pybind11_state_gpu caffe2_library caffe2_gpu_library)
    if (WIN32)
      target_link_libraries(caffe2_pybind11_state_gpu ${PYTHON_LIBRARIES})
    endif(WIN32)

    # Install with same rpath as non-gpu caffe2_pybind11_state
    set_target_properties(
        caffe2_pybind11_state_gpu PROPERTIES LIBRARY_OUTPUT_DIRECTORY
        ${CMAKE_BINARY_DIR}/caffe2/python)
    install(TARGETS caffe2_pybind11_state_gpu DESTINATION "${PYTHON_LIB_REL_PATH}/caffe2/python")
    set_target_properties(caffe2_pybind11_state_gpu PROPERTIES INSTALL_RPATH "${caffe2_pybind11_rpath}")
  endif()

  if(USE_ROCM)
    add_library(caffe2_pybind11_state_hip MODULE ${Caffe2_HIP_PYTHON_SRCS})
    target_compile_options(caffe2_pybind11_state_hip PRIVATE ${HIP_CXX_FLAGS} -fvisibility=hidden)
    set_target_properties(caffe2_pybind11_state_hip PROPERTIES PREFIX "")
    set_target_properties(caffe2_pybind11_state_hip PROPERTIES SUFFIX ${PY_EXT_SUFFIX})
    if (APPLE)
      set_target_properties(caffe2_pybind11_state_hip PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
    endif()
    target_include_directories(caffe2_pybind11_state_hip PRIVATE $<INSTALL_INTERFACE:include>)
    target_include_directories(caffe2_pybind11_state_hip PRIVATE ${Caffe2_CPU_INCLUDE} ${Caffe2_HIP_INCLUDES})
    target_link_libraries(
        caffe2_pybind11_state_hip caffe2_library caffe2_hip_library)
    if (WIN32)
      target_link_libraries(caffe2_pybind11_state_hip ${PYTHON_LIBRARIES})
    endif(WIN32)

    # Install with same rpath as non-hip caffe2_pybind11_state
    set_target_properties(
        caffe2_pybind11_state_hip PROPERTIES LIBRARY_OUTPUT_DIRECTORY
        ${CMAKE_BINARY_DIR}/caffe2/python)
    install(TARGETS caffe2_pybind11_state_hip DESTINATION "${PYTHON_LIB_REL_PATH}/caffe2/python")
    set_target_properties(caffe2_pybind11_state_hip PROPERTIES INSTALL_RPATH "${caffe2_pybind11_rpath}")
  endif()

  if (MSVC AND CMAKE_GENERATOR MATCHES "Visual Studio")
    # If we are building under windows, we will copy the file from
    # build/caffe2/python/{Debug,Release}/caffe2_pybind11_state.pyd
    # to its parent folder so that we can do in-build execution.
    add_custom_target(windows_python_copy_lib ALL)
    add_dependencies(windows_python_copy_lib caffe2_pybind11_state)
    add_custom_command(
        TARGET windows_python_copy_lib POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:caffe2_pybind11_state>
        ${CMAKE_BINARY_DIR}/caffe2/python)
    if (USE_CUDA)
      add_dependencies(windows_python_copy_lib caffe2_pybind11_state_gpu)
      add_custom_command(
          TARGET windows_python_copy_lib POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy
          $<TARGET_FILE:caffe2_pybind11_state_gpu>
          ${CMAKE_BINARY_DIR}/caffe2/python)
    endif()
    if (USE_ROCM)
      add_dependencies(windows_python_copy_lib caffe2_pybind11_state_hip)
      add_custom_command(
          TARGET windows_python_copy_lib POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy
          $<TARGET_FILE:caffe2_pybind11_state_hip>
          ${CMAKE_BINARY_DIR}/caffe2/python)
    endif()
  endif()

  # Finally, Copy all python files to build directory
  # Generate and create all needed __init__.py files, if they aren't already
  # present in the current source tree.
  message(STATUS "Automatically generating missing __init__.py files.")
  caffe_autogen_init_py_files()

  # Create a custom target that copies all python files.
  file(GLOB_RECURSE PYTHON_SRCS RELATIVE ${PROJECT_SOURCE_DIR}
       "${PROJECT_SOURCE_DIR}/caffe2/*.py")
  add_custom_target(python_copy_files ALL)
  if(MSVC OR CMAKE_GENERATOR MATCHES "Ninja")
    # ninja fails when the command line is too long so we split
    # the target into several. This would be beneficial for VS also
    # since it build targets in parallel but not custom commands
    foreach(python_src ${PYTHON_SRCS})
      get_filename_component(dir ${python_src} DIRECTORY)
      string(SHA1 name_hash "${python_src}")
      # get_filename_component(name_we ${python_src} NAME_WE)
      add_custom_target(python_copy_files_${name_hash}
          COMMAND ${CMAKE_COMMAND} -E copy
          ${PROJECT_SOURCE_DIR}/${python_src} ${CMAKE_BINARY_DIR}/${dir})
      add_dependencies(python_copy_files python_copy_files_${name_hash})
    endforeach()
  else()
    foreach(python_src ${PYTHON_SRCS})
      get_filename_component(dir ${python_src} DIRECTORY)
      add_custom_command(
          TARGET python_copy_files PRE_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy
          ${PROJECT_SOURCE_DIR}/${python_src} ${CMAKE_BINARY_DIR}/${dir})
    endforeach()
  endif()

  # Install commands
  # Pick up static python files
  install(DIRECTORY ${CMAKE_BINARY_DIR}/caffe2 DESTINATION ${PYTHON_LIB_REL_PATH}
          FILES_MATCHING PATTERN "*.py")
  # Caffe proto files
  install(DIRECTORY ${CMAKE_BINARY_DIR}/caffe DESTINATION ${PYTHON_LIB_REL_PATH}
          FILES_MATCHING PATTERN "*.py")
  # Caffe2 proto files
  install(DIRECTORY ${CMAKE_BINARY_DIR}/caffe2 DESTINATION ${PYTHON_LIB_REL_PATH}
          FILES_MATCHING PATTERN "*.py")
endif()

# Finally, set the Caffe2_MAIN_LIBS variable in the parent scope.
set(Caffe2_MAIN_LIBS ${Caffe2_MAIN_LIBS} PARENT_SCOPE)
