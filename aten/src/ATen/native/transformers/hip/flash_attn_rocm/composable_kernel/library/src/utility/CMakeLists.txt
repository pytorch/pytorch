## utility
set(UTILITY_SOURCE
    device_memory.cpp
    host_tensor.cpp
    convolution_parameter.cpp
)

add_library(utility STATIC ${UTILITY_SOURCE})
add_library(composable_kernel::utility ALIAS utility)

target_include_directories(utility PUBLIC
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/library/utility>"
)

rocm_install(
    TARGETS utility
    EXPORT utilityTargets
)

rocm_install(
    EXPORT utilityTargets
    FILE composable_kernelutilityTargets.cmake
    NAMESPACE composable_kernel::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/composable_kernel
)

clang_tidy_check(utility)
