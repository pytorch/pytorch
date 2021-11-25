set(TORCH_API_TEST_DIR "${TORCH_ROOT}/test/cpp/api")
set(TORCH_API_TEST_SOURCES
  ${TORCH_ROOT}/test/cpp/common/main.cpp
  ${TORCH_API_TEST_DIR}/autograd.cpp
  ${TORCH_API_TEST_DIR}/any.cpp
  ${TORCH_API_TEST_DIR}/dataloader.cpp
  ${TORCH_API_TEST_DIR}/enum.cpp
  ${TORCH_API_TEST_DIR}/expanding-array.cpp
  ${TORCH_API_TEST_DIR}/fft.cpp
  ${TORCH_API_TEST_DIR}/functional.cpp
  ${TORCH_API_TEST_DIR}/integration.cpp
  ${TORCH_API_TEST_DIR}/init.cpp
  ${TORCH_API_TEST_DIR}/jit.cpp
  ${TORCH_API_TEST_DIR}/memory.cpp
  ${TORCH_API_TEST_DIR}/meta_tensor.cpp
  ${TORCH_API_TEST_DIR}/misc.cpp
  ${TORCH_API_TEST_DIR}/module.cpp
  ${TORCH_API_TEST_DIR}/moduledict.cpp
  ${TORCH_API_TEST_DIR}/modulelist.cpp
  ${TORCH_API_TEST_DIR}/modules.cpp
  ${TORCH_API_TEST_DIR}/parameterdict.cpp
  ${TORCH_API_TEST_DIR}/parameterlist.cpp
  ${TORCH_API_TEST_DIR}/namespace.cpp
  ${TORCH_API_TEST_DIR}/nn_utils.cpp
  ${TORCH_API_TEST_DIR}/optim.cpp
  ${TORCH_API_TEST_DIR}/ordered_dict.cpp
  ${TORCH_API_TEST_DIR}/rnn.cpp
  ${TORCH_API_TEST_DIR}/sequential.cpp
  ${TORCH_API_TEST_DIR}/transformer.cpp
  ${TORCH_API_TEST_DIR}/serialize.cpp
  ${TORCH_API_TEST_DIR}/special.cpp
  ${TORCH_API_TEST_DIR}/static.cpp
  ${TORCH_API_TEST_DIR}/support.cpp
  ${TORCH_API_TEST_DIR}/tensor_cuda.cpp
  ${TORCH_API_TEST_DIR}/tensor_indexing.cpp
  ${TORCH_API_TEST_DIR}/tensor_options_cuda.cpp
  ${TORCH_API_TEST_DIR}/tensor_options.cpp
  ${TORCH_API_TEST_DIR}/tensor.cpp
  ${TORCH_API_TEST_DIR}/torch_include.cpp
  ${TORCH_API_TEST_DIR}/inference_mode.cpp
  ${TORCH_API_TEST_DIR}/grad_mode.cpp
  ${TORCH_API_TEST_DIR}/operations.cpp
)

if(USE_DEPLOY)
  list(APPEND TORCH_API_TEST_SOURCES ${TORCH_API_TEST_DIR}/imethod.cpp)
endif()

if(USE_CUDA)
  list(APPEND TORCH_API_TEST_SOURCES ${TORCH_API_TEST_DIR}/parallel.cpp)
endif()

add_executable(test_api ${TORCH_API_TEST_SOURCES})
target_include_directories(test_api PRIVATE ${ATen_CPU_INCLUDE})
target_link_libraries(test_api PRIVATE torch gtest)
if(NOT MSVC)
  target_compile_options(test_api PRIVATE -Wno-unused-variable)
endif()

if(USE_CUDA)
  target_link_libraries(test_api PRIVATE
    ${CUDA_LIBRARIES}
    ${CUDA_NVRTC_LIB}
    ${CUDA_CUDA_LIB}
    ${TORCH_CUDA_LIBRARIES})

  # Make sure the CUDA stubs folder doesn't end up in the RPATH of test_api:
  set(TEST_API_LIBS ${CUDA_LIBRARIES} ${CUDA_NVRTC_LIB} ${CUDA_CUDA_LIB} ${TORCH_CUDA_LIBRARIES})
  foreach(LIB IN LISTS TEST_API_LIBS)
    message("LIB: ${LIB}")
    if(LIB MATCHES "stubs")
      message("Filtering ${LIB} from being set in caffe2_nvrtc's RPATH, because it appears to point to the CUDA stubs directory, which should not be RPATHed.")
    else()
      cmake_path(GET LIB PARENT_PATH LIB_PATH)
      message("LIBPATH: ${LIB_PATH}")
      list(APPEND TEST_API_RPATH ${LIB_PATH})
    endif()
  endforeach()
  message("TEST_API_RPATH: ${TEST_API_RPATH}")
  set_target_properties(test_api PROPERTIES INSTALL_RPATH_USE_LINK_PATH FALSE)
  set_target_properties(test_api PROPERTIES INSTALL_RPATH "${TEST_API_RPATH}")

  target_compile_definitions(test_api PRIVATE "USE_CUDA")
endif()

if(USE_DEPLOY)
  target_link_libraries(test_api PRIVATE torch_deploy)
endif()

# Workaround for https://github.com/pytorch/pytorch/issues/40941
if(USE_OPENMP AND CMAKE_COMPILER_IS_GNUCXX AND (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.0.0))
  # Compiling transformer.cpp or pow_test.cpp with -O2+ and both -fuse-openmp and -faligned-newout any optimization
  # Fails with internal compiler error in gcc-7.5 or older
  # Workaround by compiling the tests without openmp (which they are not using anyway)
  set_property(TARGET test_api APPEND_STRING PROPERTY COMPILE_FLAGS "-fno-openmp")
endif()

if(NOT MSVC)
  if(APPLE)
    target_compile_options(test_api PRIVATE
      # Clang has an unfixed bug leading to spurious missing braces
      # warnings, see https://bugs.llvm.org/show_bug.cgi?id=21629
      -Wno-missing-braces)
  else()
    target_compile_options(test_api PRIVATE
      # Considered to be flaky.  See the discussion at
      # https://github.com/pytorch/pytorch/pull/9608
      -Wno-maybe-uninitialized
      # gcc gives nonsensical warnings about variadic.h
      -Wno-unused-but-set-parameter)
  endif()
endif()

if(INSTALL_TEST)
  install(TARGETS test_api DESTINATION bin)
  # Install PDB files for MSVC builds
  if(MSVC AND BUILD_SHARED_LIBS)
    install(FILES $<TARGET_PDB_FILE:test_api> DESTINATION bin OPTIONAL)
  endif()
endif()

add_executable(parallel_benchmark ${TORCH_API_TEST_DIR}/parallel_benchmark.cpp)
target_include_directories(parallel_benchmark PRIVATE ${ATen_CPU_INCLUDE})
target_link_libraries(parallel_benchmark PRIVATE torch)
