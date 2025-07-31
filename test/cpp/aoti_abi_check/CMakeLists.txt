set(AOTI_ABI_CHECK_TEST_ROOT ${TORCH_ROOT}/test/cpp/aoti_abi_check)

# Build the cpp gtest binary containing the cpp-only tests.
set(AOTI_ABI_CHECK_TEST_SRCS
  ${AOTI_ABI_CHECK_TEST_ROOT}/main.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_cast.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_dtype.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_macros.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_math.cpp
  ${AOTI_ABI_CHECK_TEST_ROOT}/test_rand.cpp
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
# The purpose is to check if the used aten/c10 headers are writtern in a header-only way
target_link_libraries(test_aoti_abi_check PRIVATE gtest_main)
target_include_directories(test_aoti_abi_check PRIVATE ${ATen_CPU_INCLUDE})

foreach(test_src ${AOTI_ABI_CHECK_VEC_TEST_SRCS})
  foreach(i RANGE ${NUM_CPU_CAPABILITY_NAMES})
    get_filename_component(test_name ${test_src} NAME_WE)
    list(GET CPU_CAPABILITY_NAMES ${i} CPU_CAPABILITY)
    list(GET CPU_CAPABILITY_FLAGS ${i} FLAGS)
    separate_arguments(FLAGS UNIX_COMMAND "${FLAGS}")
    add_executable(${test_name}_${CPU_CAPABILITY} "${test_src}")

    target_link_libraries(${test_name}_${CPU_CAPABILITY} PRIVATE gtest_main)
    target_include_directories(${test_name}_${CPU_CAPABILITY} PRIVATE ${ATen_CPU_INCLUDE})

    # Define CPU_CAPABILITY and CPU_CAPABILITY_XXX macros for conditional compilation
    target_compile_definitions(${test_name}_${CPU_CAPABILITY} PRIVATE CPU_CAPABILITY=${CPU_CAPABILITY} CPU_CAPABILITY_${CPU_CAPABILITY})
    target_compile_options(${test_name}_${CPU_CAPABILITY} PRIVATE ${FLAGS})
  endforeach()
endforeach()

if(INSTALL_TEST)
  install(TARGETS test_aoti_abi_check DESTINATION bin)
  # Install PDB files for MSVC builds
  if(MSVC AND BUILD_SHARED_LIBS)
    install(FILES $<TARGET_PDB_FILE:test_aoti_abi_check> DESTINATION bin OPTIONAL)
  endif()
endif()
