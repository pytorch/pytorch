# NCCL EP Example/Library CMake Configuration

# NCCL_EP requires sm_90+. If the caller passed -DCMAKE_CUDA_ARCHITECTURES use
# it as-is and let nvcc fail if it's below sm_90. Otherwise mirror the
# per-CUDA-version block in the root CMakeLists.txt trimmed to Hopper+.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    if(CUDA_MAJOR EQUAL 11)
        if(CUDA_MINOR LESS 8)
            message(FATAL_ERROR "NCCL_EP requires CUDA 11.8+ for sm_90 support. Got CUDA ${CUDAToolkit_VERSION}")
        else()
            set(CMAKE_CUDA_ARCHITECTURES "90")
        endif()
    elseif(CUDA_MAJOR EQUAL 12)
        if(CUDA_MINOR LESS 8)
            set(CMAKE_CUDA_ARCHITECTURES "90")
        else()
            set(CMAKE_CUDA_ARCHITECTURES "90;100;120")
        endif()
    elseif(CUDA_MAJOR GREATER_EQUAL 13)
        set(CMAKE_CUDA_ARCHITECTURES "90;100;110;120")
    else()
        message(FATAL_ERROR "NCCL_EP requires CUDA 11.8+ for sm_90 support. Got CUDA ${CUDAToolkit_VERSION}")
    endif()
endif()
message(STATUS "NCCL_EP using CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

# Parse library version from include/nccl_ep.h (single source of truth).
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/include/nccl_ep.h _NCCL_EP_HEADER_CONTENT)
string(REGEX REPLACE ".*#define NCCL_EP_MAJOR ([0-9]+).*" "\\1" NCCL_EP_MAJOR "${_NCCL_EP_HEADER_CONTENT}")
string(REGEX REPLACE ".*#define NCCL_EP_MINOR ([0-9]+).*" "\\1" NCCL_EP_MINOR "${_NCCL_EP_HEADER_CONTENT}")
string(REGEX REPLACE ".*#define NCCL_EP_PATCH ([0-9]+).*" "\\1" NCCL_EP_PATCH "${_NCCL_EP_HEADER_CONTENT}")
unset(_NCCL_EP_HEADER_CONTENT)
message(STATUS "NCCL EP version: ${NCCL_EP_MAJOR}.${NCCL_EP_MINOR}.${NCCL_EP_PATCH}")

# Build layout:
#   NCCL_HOME          — input: NCCL core headers + libnccl.so
#   NCCL_EP_BUILDDIR   — output: NCCL EP artifacts
# Defaults preserve the legacy single-tree layout under CMAKE_BINARY_DIR.

set(NCCL_HOME "${CMAKE_BINARY_DIR}" CACHE PATH "NCCL core build/install directory used by NCCL EP")
set(NCCL_EP_BUILDDIR "${CMAKE_BINARY_DIR}" CACHE PATH "NCCL EP build output directory")
set(NCCL_INCDIR "${NCCL_HOME}/include")
set(NCCL_LIBDIR "${NCCL_HOME}/lib")
set(NCCL_EP_INCDIR "${NCCL_EP_BUILDDIR}/include")
set(NCCL_EP_LIBDIR "${NCCL_EP_BUILDDIR}/lib")

# Source files
set(NCCL_EP_HOST_SRC
    nccl_ep.cc
    device/jit/jit_utils.cc
    device/jit/jit_cache.cc
    device/jit/nvcc_compiler.cc
    device/jit/jit_runtime.cc
)

set(NCCL_EP_DEVICE_SRC
    device/low_latency.cu
    device/hybridep_adapter.cu
)

set(NCCL_EP_SRC
    ${NCCL_EP_HOST_SRC}
    ${NCCL_EP_DEVICE_SRC}
)

add_library(nccl_ep_shared SHARED
    ${NCCL_EP_SRC}
)
add_library(nccl_ep_static STATIC
    ${NCCL_EP_SRC}
)
add_custom_target(nccl_ep_lib DEPENDS nccl_ep_shared nccl_ep_static)

set(NCCL_EP_TARGETS nccl_ep_shared nccl_ep_static)

set_target_properties(nccl_ep_shared nccl_ep_static PROPERTIES
    OUTPUT_NAME "nccl_ep"
    ARCHIVE_OUTPUT_DIRECTORY "${NCCL_EP_LIBDIR}"
    LIBRARY_OUTPUT_DIRECTORY "${NCCL_EP_LIBDIR}"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

# Version the shared library:
#   libnccl_ep.so.MAJOR.MINOR.PATCH (actual file)
#   libnccl_ep.so.MAJOR             (SONAME; linker symlink)
#   libnccl_ep.so                   (unversioned symlink for build/runtime loading)
set_target_properties(nccl_ep_shared PROPERTIES
    VERSION   ${NCCL_EP_MAJOR}.${NCCL_EP_MINOR}.${NCCL_EP_PATCH}
    SOVERSION ${NCCL_EP_MAJOR}
)

# No RUNPATH baked into the artifact (matches libnccl.so's convention).
# BUILD_WITH_INSTALL_RPATH=TRUE ensures the build-tree binary also has no
# embedded paths, so wheel staging that copies from the build dir is safe.
set_target_properties(nccl_ep_shared PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH ""
)

set(NCCL_EP_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/device
    ${NCCL_INCDIR}
    ${NCCL_INCDIR}/nccl_device
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}/cccl
)

set(NCCL_EP_JIT_SOURCE_DIR "${NCCL_EP_INCDIR}/nccl_ep")
set(NCCL_EP_JIT_BUILD_INCLUDE_DIR "${NCCL_EP_INCDIR}")
set(NCCL_EP_JIT_CUDA_INCLUDE_DIR "")
if(CUDAToolkit_INCLUDE_DIRS)
    list(GET CUDAToolkit_INCLUDE_DIRS 0 NCCL_EP_JIT_CUDA_INCLUDE_DIR)
endif()
set(NCCL_EP_JIT_CONFIG_HEADER "${NCCL_EP_JIT_SOURCE_DIR}/config.h")
set(NCCL_EP_JIT_HEADERS
    device/hybrid_ep.cuh
    device/device_primitives.cuh
    device/hybridep_configs.cuh
    device/cuda_compat_shims.cuh
    device/scan_kernel.cuh
    device/macros.cuh
)

file(MAKE_DIRECTORY "${NCCL_EP_JIT_SOURCE_DIR}")
file(WRITE "${NCCL_EP_JIT_CONFIG_HEADER}"
    "#pragma once\n"
    "#define NCCL_EP_JIT_SOURCE_DIR \"${NCCL_EP_JIT_SOURCE_DIR}\"\n"
    "#define NCCL_EP_JIT_BUILD_INCLUDE_DIR \"${NCCL_EP_JIT_BUILD_INCLUDE_DIR}\"\n"
    "#define NCCL_EP_JIT_CUDA_INCLUDE_DIR \"${NCCL_EP_JIT_CUDA_INCLUDE_DIR}\"\n"
)

foreach(header IN LISTS NCCL_EP_JIT_HEADERS)
    get_filename_component(header_dir "${header}" DIRECTORY)
    file(MAKE_DIRECTORY "${NCCL_EP_JIT_SOURCE_DIR}/${header_dir}")
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/${header}" "${NCCL_EP_JIT_SOURCE_DIR}/${header}" COPYONLY)
endforeach()
configure_file(include/common.hpp "${NCCL_EP_JIT_SOURCE_DIR}/common.hpp" COPYONLY)

# Allow device code to call host constexpr with --expt-relaxed-constexpr
# Hides all symbols by default in the host compiled object files with -fvisibility=hidden
foreach(t IN LISTS NCCL_EP_TARGETS)
    target_compile_options(${t} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-fvisibility=hidden>
        $<$<COMPILE_LANGUAGE:CXX>:-include>
        $<$<COMPILE_LANGUAGE:CXX>:${NCCL_EP_JIT_CONFIG_HEADER}>
    )
    if(TARGET CUDA::cuda_driver)
        target_link_libraries(${t} PUBLIC CUDA::cuda_driver)
    else()
        target_link_libraries(${t} PUBLIC cuda)
    endif()
    target_include_directories(${t} PUBLIC ${NCCL_EP_INCLUDE_DIRS})
    target_link_directories(${t} PUBLIC ${NCCL_LIBDIR})
    target_link_libraries(${t} PUBLIC -lnccl)
    target_link_libraries(${t} PUBLIC ${CUDAToolkit_LIBRARIES})
endforeach()

# Copy headers to NCCL_EP_BUILDDIR/include (runs at configure)
file(MAKE_DIRECTORY "${NCCL_EP_INCDIR}")
configure_file(include/nccl_ep.h    "${NCCL_EP_INCDIR}/nccl_ep.h"    COPYONLY)
configure_file(include/ep_enums.h   "${NCCL_EP_INCDIR}/ep_enums.h"   COPYONLY)

# Install to NCCL_EP_BUILDDIR.
install(FILES
    include/nccl_ep.h
    include/ep_enums.h
    DESTINATION "${NCCL_EP_INCDIR}"
)

foreach(header IN LISTS NCCL_EP_JIT_HEADERS)
    get_filename_component(header_dir "${header}" DIRECTORY)
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/${header}"
        DESTINATION "${NCCL_EP_INCDIR}/nccl_ep/${header_dir}"
    )
endforeach()
install(FILES include/common.hpp
    DESTINATION "${NCCL_EP_INCDIR}/nccl_ep"
)

install(TARGETS ${NCCL_EP_TARGETS}
    LIBRARY DESTINATION "${NCCL_EP_LIBDIR}"
    ARCHIVE DESTINATION "${NCCL_EP_LIBDIR}"
)
