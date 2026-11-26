# Skip on windows
if(WIN32)
  return()
endif()

set(AOTI_ABI_CHECK_TEST_ROOT ${TORCH_ROOT}/test/cpp/aoti_abi_check)

# Build the cpp gtest binary containing the cpp-only tests.
set(AOTI_ABI_CHECK_TEST_SRCS
  ${AOTI_ABI_CHECK_TEST_ROOT}/main.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_accessor.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_accumulatetype.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_cast.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_copysign.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_devicetype.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_dispatch.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_dispatch_v2.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_dtype.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_exception.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_headeronlyarrayref.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_irange.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_jiterator_macros.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_layout.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_load.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_macros.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_math.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_mathconstants.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_memoryformat.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_metaprogramming.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_model_pool.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_native_math.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_overflows.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_rand.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_scalartype.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_transposetype.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_typelist.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_typetraits.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_vec.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_vec_half.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_zmath.cpp
)

# The below are tests that require CPU_CAPABILITY setup
# You may think test_vec.cpp needs to be in there, but it does not.
set(AOTI_ABI_CHECK_VEC_TEST_SRCS
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_vec_half.cpp
)

if(USE_CUDA)
  list(APPEND AOTI_ABI_CHECK_TEST_SRCS
    ${AOTI_ABI_CHECK_TEST_ROOT}/cuda/test_cast.cu
    ${AOTI_ABI_CHECK_TEST_ROOT}/cuda/test_kernel_utils.cu
  )
endif()
# This directory is not hipified. The CUDA file above stays CUDA-only.
# The ROCm file forces ROCM_VERSION 0 so the packed-atomic branch is compiled.
if(USE_ROCM)
  list(APPEND AOTI_ABI_CHECK_TEST_SRCS
    ${AOTI_ABI_CHECK_TEST_ROOT}/cuda/test_rocm_packed_atomic.cu
  )
  set_source_files_properties(
    ${AOTI_ABI_CHECK_TEST_ROOT}/cuda/test_rocm_packed_atomic.cu
    PROPERTIES LANGUAGE HIP COMPILE_DEFINITIONS USE_ROCM)
endif()

# test_model_pool.cpp follows the build's device macro, like the pool's code.
# ROCm builds hipify the pool's USE_CUDA into USE_ROCM.
if(USE_CUDA)
  set_source_files_properties(${AOTI_ABI_CHECK_TEST_ROOT}/test_model_pool.cpp
    PROPERTIES COMPILE_DEFINITIONS USE_CUDA)
elseif(USE_ROCM)
  set_source_files_properties(${AOTI_ABI_CHECK_TEST_ROOT}/test_model_pool.cpp
    PROPERTIES COMPILE_DEFINITIONS USE_ROCM)
elseif(USE_XPU)
  set_source_files_properties(${AOTI_ABI_CHECK_TEST_ROOT}/test_model_pool.cpp
    PROPERTIES COMPILE_DEFINITIONS USE_XPU)
endif()

add_executable(test_aoti_abi_check
  ${AOTI_ABI_CHECK_TEST_SRCS}
)
# A HIP source makes CMake link this target with ROCm clang. CMake builds gtest
# without PIC flags, and in manywheel builds that HIP link produces a PIE, which
# ld.lld rejects for those objects. Keep the host linker.
if(USE_ROCM)
  set_target_properties(test_aoti_abi_check PROPERTIES
    LINKER_LANGUAGE CXX
    HIP_ARCHITECTURES "${PYTORCH_ROCM_ARCH}")
endif()

# TODO temporary until we can delete the old gtest polyfills.
target_compile_definitions(test_aoti_abi_check PRIVATE USE_GTEST)

# WARNING: DO NOT LINK torch!!!
# The purpose is to check if the used aten/c10 headers are written in a header-only way
target_link_libraries(test_aoti_abi_check PRIVATE gtest_main sleef)
if(USE_CUDA)
  target_link_libraries(test_aoti_abi_check PRIVATE torch::cudart)
elseif(USE_ROCM)
  target_link_libraries(test_aoti_abi_check PRIVATE hip::host)
endif()
target_include_directories(test_aoti_abi_check PRIVATE ${ATen_CPU_INCLUDE})
if(NOT USE_SYSTEM_SLEEF)
  target_include_directories(test_aoti_abi_check PRIVATE ${CMAKE_BINARY_DIR}/include)
endif()

# Disable unused-variable warnings for variables that are only used to test compilation
target_compile_options_if_supported(test_aoti_abi_check -Wno-unused-variable)
target_compile_options_if_supported(test_aoti_abi_check -Wno-unused-but-set-variable)
# Add -Wno-dangling-pointer for GCC 13. C++ only: ROCm clang compiles the HIP
# source and does not know the flag.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13)
  target_compile_options(test_aoti_abi_check PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-dangling-pointer>)
endif()

foreach(test_src ${AOTI_ABI_CHECK_VEC_TEST_SRCS})
  foreach(i RANGE ${NUM_CPU_CAPABILITY_NAMES})
    get_filename_component(test_name ${test_src} NAME_WE)
    list(GET CPU_CAPABILITY_NAMES ${i} CPU_CAPABILITY)
    list(GET CPU_CAPABILITY_FLAGS ${i} FLAGS)
    separate_arguments(FLAGS UNIX_COMMAND "${FLAGS}")
    add_executable(${test_name}_${CPU_CAPABILITY} "${test_src}")

    target_link_libraries(${test_name}_${CPU_CAPABILITY} PRIVATE gtest_main sleef)
    target_include_directories(${test_name}_${CPU_CAPABILITY} PRIVATE ${ATen_CPU_INCLUDE})
    if(NOT USE_SYSTEM_SLEEF)
      target_include_directories(${test_name}_${CPU_CAPABILITY} PRIVATE ${CMAKE_BINARY_DIR}/include)
    endif()

    # Define CPU_CAPABILITY and CPU_CAPABILITY_XXX macros for conditional compilation
    target_compile_definitions(${test_name}_${CPU_CAPABILITY} PRIVATE CPU_CAPABILITY=${CPU_CAPABILITY} CPU_CAPABILITY_${CPU_CAPABILITY})
    target_compile_options(${test_name}_${CPU_CAPABILITY} PRIVATE ${FLAGS})
    target_compile_options_if_supported(${test_name}_${CPU_CAPABILITY} -Wno-unused-variable)
    target_compile_options_if_supported(${test_name}_${CPU_CAPABILITY} -Wno-unused-but-set-variable)
  endforeach()
endforeach()

if(INSTALL_TEST)
  install(TARGETS test_aoti_abi_check DESTINATION bin)
  # Install PDB files for MSVC builds
  if(MSVC AND BUILD_SHARED_LIBS)
    install(FILES $<TARGET_PDB_FILE:test_aoti_abi_check> DESTINATION bin OPTIONAL)
  endif()
endif()
