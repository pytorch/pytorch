# Skip on windows
if(WIN32)
  return()
endif()

set(AOTI_ABI_CHECK_TEST_ROOT ${TORCH_ROOT}/test/cpp/aoti_abi_check)

# Build the cpp gtest binary containing the cpp-only tests.
set(AOTI_ABI_CHECK_TEST_SRCS
  ${AOTI_ABI_CHECK_TEST_ROOT}/main.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_cast.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_devicetype.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_dispatch.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_dispatch_v2.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_dtype.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_exception.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_headeronlyarrayref.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_macros.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_math.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_rand.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_scalartype.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_vec.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_vec_half.cpp
)

# The below are tests that require CPU_CAPABILITY setup
# You may think test_vec.cpp needs to be in there, but it does not.
set(AOTI_ABI_CHECK_VEC_TEST_SRCS
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_vec_half.cpp
)

add_executable(test_aoti_abi_check
  ${AOTI_ABI_CHECK_TEST_SRCS}
)

# TODO temporary until we can delete the old gtest polyfills.
target_compile_definitions(test_aoti_abi_check PRIVATE USE_GTEST)

# WARNING: DO NOT LINK torch!!!
# The purpose is to check if the used aten/c10 headers are written in a header-only way
target_link_libraries(test_aoti_abi_check PRIVATE gtest_main sleef)
target_include_directories(test_aoti_abi_check PRIVATE ${ATen_CPU_INCLUDE})
if(NOT USE_SYSTEM_SLEEF)
  target_include_directories(test_aoti_abi_check PRIVATE ${CMAKE_BINARY_DIR}/include)
endif()

# Disable unused-variable warnings for variables that are only used to test compilation
target_compile_options_if_supported(test_aoti_abi_check -Wno-unused-variable)
target_compile_options_if_supported(test_aoti_abi_check -Wno-unused-but-set-variable)
# Add -Wno-dangling-pointer for GCC 13
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13)
  target_compile_options_if_supported(test_aoti_abi_check -Wno-dangling-pointer)
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
