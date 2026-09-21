# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

include(GoogleTest)
set(CMAKE_CXX_STANDARD 20)

if(KINETO_BACKEND STREQUAL "xpu")
    set(XPU_XPUPTI_LIBRARY ${PTI_LIBRARY} ${SYCL_LIBRARY})
    add_subdirectory(xpupti)
else()
    set(XPU_XPUPTI_LIBRARY "")
endif()

include_directories(${LIBKINETO_DIR})
link_libraries(fmt::fmt-header-only)
# ApproximateClockTest
add_executable(ApproximateClockTest ApproximateClockTest.cpp)
target_link_libraries(ApproximateClockTest PRIVATE
    gtest_main
    kineto_base kineto_api
    ${XPU_XPUPTI_LIBRARY})
gtest_discover_tests(ApproximateClockTest)

# ConfigTest
add_executable(ConfigTest ConfigTest.cpp)
target_link_libraries(ConfigTest PRIVATE
    gtest_main
    kineto_base kineto_api
    ${XPU_XPUPTI_LIBRARY})
gtest_discover_tests(ConfigTest)

# ConfigLoaderTest
add_executable(ConfigLoaderTest ConfigLoaderTest.cpp)
target_link_libraries(ConfigLoaderTest PRIVATE
    gtest_main
    kineto_base kineto_api
    ${XPU_XPUPTI_LIBRARY})
gtest_discover_tests(ConfigLoaderTest)

# ActivityProfilerControllerTest
add_executable(ActivityProfilerControllerTest
    ActivityProfilerControllerTest.cpp
    TestUtils.cpp)
target_link_libraries(ActivityProfilerControllerTest PRIVATE
    gtest_main
    kineto_base kineto_api
    ${XPU_XPUPTI_LIBRARY})
gtest_discover_tests(ActivityProfilerControllerTest)

# AsyncActivityProfilerHandlerTest
# Builds on Windows, but one of its cases does not: MetadataJsonFormatingTest
# sets the environment with unix setenv, so the source compiles that case out
# under _WIN32. The exclusion is per test rather than per target here because
# the other fifteen cases run there fine.
add_executable(AsyncActivityProfilerHandlerTest
    AsyncActivityProfilerHandlerTest.cpp
    TestUtils.cpp)
target_link_libraries(AsyncActivityProfilerHandlerTest PRIVATE
    gtest_main
    kineto_base kineto_api
    nlohmann_json::nlohmann_json
    ${XPU_XPUPTI_LIBRARY})
gtest_discover_tests(AsyncActivityProfilerHandlerTest)

# ConfigLoaderPollThreadExceptionTest
# Not supported on Windows: sets the environment with unix setenv.
if(NOT WIN32)
    add_executable(ConfigLoaderPollThreadExceptionTest
        ConfigLoaderPollThreadExceptionTest.cpp)
    target_link_libraries(ConfigLoaderPollThreadExceptionTest PRIVATE
        gtest_main
        kineto_base kineto_api
        ${XPU_XPUPTI_LIBRARY})
    gtest_discover_tests(ConfigLoaderPollThreadExceptionTest)
endif()

# AsyncE2ECpuTraceTest
# Not supported on Windows: it drives an on-demand config, and Config only
# accepts on-demand trace files under a directory that defaults to /tmp.
if(NOT WIN32)
    add_executable(AsyncE2ECpuTraceTest
        AsyncE2ECpuTraceTest.cpp
        MockCpuActivityBuffer.cpp
        TestUtils.cpp)
    target_link_libraries(AsyncE2ECpuTraceTest PRIVATE
        gtest_main
        kineto_base kineto_api
        nlohmann_json::nlohmann_json
        ${XPU_XPUPTI_LIBRARY})
    gtest_discover_tests(AsyncE2ECpuTraceTest)
endif()

# SyncActivityProfilerHandlerTest and GenericActivityProfilerTeardownTest
# use nothing unix-specific, but have never been built on Windows. Enabling
# them is a separate change that needs its own Windows run to confirm.
if(NOT WIN32)
    add_executable(SyncActivityProfilerHandlerTest
        SyncActivityProfilerHandlerTest.cpp)
    target_link_libraries(SyncActivityProfilerHandlerTest PRIVATE
        gtest_main
        kineto_base kineto_api
        ${XPU_XPUPTI_LIBRARY})
    gtest_discover_tests(SyncActivityProfilerHandlerTest)

    add_executable(GenericActivityProfilerTeardownTest
        GenericActivityProfilerTeardownTest.cpp)
    target_link_libraries(GenericActivityProfilerTeardownTest PRIVATE
        gtest_main
        kineto_base kineto_api
        ${XPU_XPUPTI_LIBRARY})
    gtest_discover_tests(GenericActivityProfilerTeardownTest)
endif()

if(KINETO_BACKEND STREQUAL "cuda")
# CuptiActivityProfilerTest
add_executable(CuptiActivityProfilerTest
    CuptiActivityProfilerTest.cpp
    MockActivitySubProfiler.cpp
    MockCpuActivityBuffer.cpp
    TestUtils.cpp)
target_link_libraries(CuptiActivityProfilerTest PRIVATE
    gtest_main
    gmock
    kineto_base kineto_api
    nlohmann_json::nlohmann_json)
target_include_directories(CuptiActivityProfilerTest PRIVATE
    "${LIBKINETO_DIR}"
    "${LIBKINETO_DIR}/include"
    "${LIBKINETO_DIR}/src"
    "${CUDA_SOURCE_DIR}/include"
    "${CUPTI_INCLUDE_DIR}")
gtest_discover_tests(CuptiActivityProfilerTest)

if(KINETO_ENABLE_CUPTI_PM_SAMPLING)
# CuptiPMSamplingApiTest
add_executable(CuptiPMSamplingApiTest
    CuptiPMSamplingApiTest.cpp)
target_link_libraries(CuptiPMSamplingApiTest PRIVATE
    gtest_main
    kineto_base kineto_api)
target_include_directories(CuptiPMSamplingApiTest PRIVATE
    "${LIBKINETO_DIR}"
    "${LIBKINETO_DIR}/include"
    "${LIBKINETO_DIR}/src")
gtest_discover_tests(CuptiPMSamplingApiTest)

# CuptiPMSamplingProfilerTest
add_executable(CuptiPMSamplingProfilerTest
    CuptiPMSamplingProfilerTest.cpp)
target_link_libraries(CuptiPMSamplingProfilerTest PRIVATE
    gtest_main
    kineto_base kineto_api)
target_include_directories(CuptiPMSamplingProfilerTest PRIVATE
    "${LIBKINETO_DIR}"
    "${LIBKINETO_DIR}/include"
    "${LIBKINETO_DIR}/src")
gtest_discover_tests(CuptiPMSamplingProfilerTest)
endif()

# CuptiCallbackApiTest
add_executable(CuptiCallbackApiTest CuptiCallbackApiTest.cpp)
target_link_libraries(CuptiCallbackApiTest PRIVATE
    gtest_main
    kineto_base kineto_api)
gtest_discover_tests(CuptiCallbackApiTest)

# CuptiStringsTest
add_executable(CuptiStringsTest CuptiStringsTest.cpp)
target_link_libraries(CuptiStringsTest PRIVATE
    gtest_main
    kineto_base kineto_api)
gtest_discover_tests(CuptiStringsTest)

# DevicePropertiesTest
add_executable(DevicePropertiesTest DevicePropertiesTest.cpp)
target_link_libraries(DevicePropertiesTest PRIVATE
    gtest_main
    kineto_base kineto_api)
gtest_discover_tests(DevicePropertiesTest)
endif()

if(KINETO_BACKEND STREQUAL "rocm")
# RocmActivityProfilerTest
add_executable(RocmActivityProfilerTest
    RocmActivityProfilerTest.cpp
    MockActivitySubProfiler.cpp
    MockCpuActivityBuffer.cpp
    TestUtils.cpp)
target_compile_definitions(RocmActivityProfilerTest PRIVATE
    "__HIP_PLATFORM_HCC__"
    "__HIP_PLATFORM_AMD__")
target_link_libraries(RocmActivityProfilerTest PRIVATE
    gtest_main
    gmock
    kineto_base kineto_api
    nlohmann_json::nlohmann_json)
target_include_directories(RocmActivityProfilerTest PRIVATE
    "${LIBKINETO_DIR}"
    "${LIBKINETO_DIR}/include"
    "${LIBKINETO_DIR}/src"
    "${ROCM_INCLUDE_DIRS}")
gtest_discover_tests(RocmActivityProfilerTest)
endif()

# LoggerObserverTest
add_executable(LoggerObserverTest LoggerObserverTest.cpp)
target_link_libraries(LoggerObserverTest PRIVATE
    gtest_main
    kineto_base kineto_api
    ${XPU_XPUPTI_LIBRARY})
gtest_discover_tests(LoggerObserverTest)

# RegisterLoggerFactoryTest
add_executable(RegisterLoggerFactoryTest RegisterLoggerFactoryTest.cpp)
target_link_libraries(RegisterLoggerFactoryTest PRIVATE
    gtest_main
    kineto_base kineto_api
    ${XPU_XPUPTI_LIBRARY})
target_include_directories(RegisterLoggerFactoryTest PRIVATE
    "${LIBKINETO_DIR}"
    "${LIBKINETO_DIR}/include"
    "${LIBKINETO_DIR}/src")
gtest_discover_tests(RegisterLoggerFactoryTest)

# PidInfoTest
add_executable(PidInfoTest PidInfoTest.cpp)
target_link_libraries(PidInfoTest PRIVATE
    gtest_main
    kineto_base kineto_api
    ${XPU_XPUPTI_LIBRARY})
gtest_discover_tests(PidInfoTest)

# TypedMetadataTest
add_executable(TypedMetadataTest TypedMetadataTest.cpp)
target_link_libraries(TypedMetadataTest PRIVATE
    gtest_main
    kineto_base kineto_api
    ${XPU_XPUPTI_LIBRARY})
target_include_directories(TypedMetadataTest PRIVATE
    "${LIBKINETO_DIR}"
    "${LIBKINETO_DIR}/include"
    "${LIBKINETO_DIR}/src")
gtest_discover_tests(TypedMetadataTest)

# GenericTraceActivityMetadataTest
add_executable(GenericTraceActivityMetadataTest
    GenericTraceActivityMetadataTest.cpp)
target_link_libraries(GenericTraceActivityMetadataTest PRIVATE
    gtest_main
    kineto_base kineto_api
    ${XPU_XPUPTI_LIBRARY})
target_include_directories(GenericTraceActivityMetadataTest PRIVATE
    "${LIBKINETO_DIR}"
    "${LIBKINETO_DIR}/include"
    "${LIBKINETO_DIR}/src")
gtest_discover_tests(GenericTraceActivityMetadataTest)

# OutputJsonTest
add_executable(OutputJsonTest
    OutputJsonTest.cpp
    TestUtils.cpp)
target_link_libraries(OutputJsonTest PRIVATE
    gtest_main
    kineto_base kineto_api
    nlohmann_json::nlohmann_json
    ${XPU_XPUPTI_LIBRARY})
target_include_directories(OutputJsonTest PRIVATE
    "${LIBKINETO_DIR}"
    "${LIBKINETO_DIR}/include"
    "${LIBKINETO_DIR}/src")
gtest_discover_tests(OutputJsonTest)

# Every target defined in this directory is a test executable, so the
# directory's own target list is the set to install. Reading it here, at the
# end of the file, is what makes a newly added test install itself.
if(KINETO_INSTALL_TESTS)
    get_property(KINETO_TEST_TARGETS DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
    install(TARGETS ${KINETO_TEST_TARGETS}
        DESTINATION "${KINETO_TEST_INSTALL_DIR}")
endif()
