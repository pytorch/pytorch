cmake_minimum_required(VERSION 3.22)

# Our module dir, include that now so that we can get the version automatically
# from git describe
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
include(GitVersion)
get_git_version(
    SLANG_VERSION_NUMERIC
    SLANG_VERSION_FULL
    "${CMAKE_CURRENT_LIST_DIR}"
)

#
# Our project
#
project(slang VERSION "${SLANG_VERSION_NUMERIC}" LANGUAGES)
set(PROJECT_VERSION "${SLANG_VERSION_FULL}")

#
# Global CMake options
#
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.24")
    cmake_policy(SET CMP0135 OLD)
endif()
cmake_policy(SET CMP0077 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.25")
    cmake_policy(SET CMP0141 NEW)
endif()
cmake_policy(SET CMP0091 NEW)

# Don't use absolute paths to the build tree in RPATH, this makes the build
# tree relocatable
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)

# Export the compile datebase as a json file, this can be used by VIM language server
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Enable placing targets into a hierarchy for IDE generators
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

#
# CMake-supplied modules and our utils
#
enable_language(C CXX)

include(FindPackageHandleStandardArgs)
include(CMakeDependentOption)
include(GNUInstallDirs)

include(CCacheDebugInfoWorkaround)
include(CompilerFlags)
include(Glob)
include(LLVM)
include(SlangTarget)
include(AutoOption)
include(GitHubRelease)
include(FetchedSharedLibrary)

#
# Options
#

auto_option(
    SLANG_ENABLE_CUDA
    CUDAToolkit
    "Enable CUDA tests using CUDA found in CUDA_PATH"
)
auto_option(
    SLANG_ENABLE_OPTIX
    OptiX
    "Enable OptiX build/tests, requires SLANG_ENABLE_CUDA"
)
auto_option(
    SLANG_ENABLE_NVAPI
    NVAPI
    "Enable NVAPI usage (Only available for builds targeting Windows)"
)
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    auto_option(
        SLANG_ENABLE_XLIB
        X11
        "Build gfx and platform with Xlib to support windowed apps on Linux"
    )
else()
    set(SLANG_ENABLE_XLIB OFF)
endif()
auto_option(
    SLANG_ENABLE_AFTERMATH
    Aftermath
    "Enable Aftermath in GFX, and add aftermath crash example to project"
)
advanced_option(
    SLANG_ENABLE_DX_ON_VK
    "Use dxvk and vkd3d-proton for DirectX support"
    OFF
)
advanced_option(SLANG_ENABLE_SLANG_RHI "Use slang-rhi as dependency" ON)

option(
    SLANG_EMBED_CORE_MODULE_SOURCE
    "Embed core module source in the binary"
    ON
)
option(
    SLANG_EMBED_CORE_MODULE
    "Build slang with an embedded version of the core module"
    ON
)

option(SLANG_ENABLE_DXIL "Enable generating DXIL with DXC" ON)

option(SLANG_ENABLE_FULL_IR_VALIDATION "Enable full IR validation (SLOW!)")
option(SLANG_ENABLE_IR_BREAK_ALLOC, "Enable _debugUID on IR allocation")
option(SLANG_ENABLE_ASAN "Enable ASAN (address sanitizer)")

option(SLANG_ENABLE_PREBUILT_BINARIES "Enable using prebuilt binaries" ON)
option(SLANG_ENABLE_GFX "Enable gfx targets" ON)
option(SLANG_ENABLE_SLANGD "Enable language server target" ON)
option(SLANG_ENABLE_SLANGC "Enable standalone compiler target" ON)
option(SLANG_ENABLE_SLANGI "Enable Slang interpreter target" ON)
option(SLANG_ENABLE_SLANGRT "Enable runtime target" ON)
option(
    SLANG_ENABLE_SLANG_GLSLANG
    "Enable glslang dependency and slang-glslang wrapper target"
    ON
)
option(
    SLANG_ENABLE_TESTS
    "Enable test targets, some tests may require SLANG_ENABLE_GFX, SLANG_ENABLE_SLANGD or SLANG_ENABLE_SLANGRT"
    ON
)
option(
    SLANG_ENABLE_EXAMPLES
    "Enable example targets, requires SLANG_ENABLE_GFX"
    ON
)
option(SLANG_ENABLE_REPLAYER "Enable slang-replay tool" ON)

option(
    SLANG_GITHUB_TOKEN
    "Use a given token value for accessing Github REST API"
    ""
)

advanced_option(SLANG_USE_SYSTEM_MINIZ "Build using system Miniz library" OFF)
advanced_option(SLANG_USE_SYSTEM_LZ4 "Build using system LZ4 library" OFF)
advanced_option(
    SLANG_USE_SYSTEM_VULKAN_HEADERS
    "Build using system Vulkan headers"
    OFF
)
advanced_option(
    SLANG_USE_SYSTEM_SPIRV_HEADERS
    "Build using system SPIR-V headers"
    OFF
)
advanced_option(
    SLANG_USE_SYSTEM_UNORDERED_DENSE
    "Build using system unordered dense"
    OFF
)

option(
    SLANG_SPIRV_HEADERS_INCLUDE_DIR
    "Provide a specific path for the SPIR-V headers and grammar files"
)
mark_as_advanced(SLANG_SPIRV_HEADERS_INCLUDE_DIR)

# Options for user defined paths for external modules.
advanced_option(
    SLANG_OVERRIDE_LZ4_PATH
    "Build using user defined path for LZ4"
    OFF
)
advanced_option(
    SLANG_OVERRIDE_MINIZ_PATH
    "Build using user defined path for Miniz"
    OFF
)
advanced_option(
    SLANG_OVERRIDE_UNORDERED_DENSE_PATH
    "Build using user defined path for unordered_dense"
    OFF
)
advanced_option(
    SLANG_OVERRIDE_VULKAN_HEADERS_PATH
    "Build using user defined path for Vulkan headers"
    OFF
)
advanced_option(
    SLANG_OVERRIDE_SPIRV_HEADERS_PATH
    "Build using user defined path for SPIR-V headers"
    OFF
)
advanced_option(
    SLANG_OVERRIDE_SPIRV_TOOLS_PATH
    "Build using user defined path for SPIR-V tools"
    OFF
)
advanced_option(
    SLANG_OVERRIDE_GLSLANG_PATH
    "Build using user defined path for glslang"
    OFF
)

advanced_option(
    SLANG_EXCLUDE_DAWN
    "Optionally exclude webgpu_dawn from the build"
    OFF
)

advanced_option(
    SLANG_EXCLUDE_TINT
    "Optionally exclude slang-tint from the build"
    OFF
)

if(${SLANG_USE_SYSTEM_LZ4})
    add_compile_definitions(SLANG_USE_SYSTEM_LZ4_HEADER)
endif()

if(${SLANG_USE_SYSTEM_SPIRV_HEADERS})
    add_compile_definitions(SLANG_USE_SYSTEM_SPIRV_HEADER)
endif()

if(${SLANG_USE_SYSTEM_UNORDERED_DENSE})
    add_compile_definitions(SLANG_USE_SYSTEM_UNORDERED_DENSE_HEADER)
endif()

if(SLANG_OVERRIDE_SPIRV_HEADERS_PATH)
    add_compile_definitions(SLANG_USE_SYSTEM_SPIRV_HEADER)
endif()

enum_option(
    SLANG_LIB_TYPE
    # Default
    SHARED
    "How to build the slang lib:"
    # Options
    SHARED
    "Build slang as a shared library (default)"
    STATIC
    "Build slang as a static library"
)

option(
    SLANG_ENABLE_RELEASE_DEBUG_INFO
    "Generate debug info for Release builds"
    ON
)

option(SLANG_ENABLE_RELEASE_LTO "Enable LTO for Release builds" ON)

option(
    SLANG_ENABLE_SPLIT_DEBUG_INFO
    "Generate split debug info for debug builds"
    ON
)

set(SLANG_GENERATORS_PATH
    ""
    CACHE PATH
    "An optional path to the outputs of the all-generators target compiled for the build platform, used when cross-compiling"
)

enum_option(
    SLANG_SLANG_LLVM_FLAVOR
    # Default
    FETCH_BINARY_IF_POSSIBLE
    "How to get or build slang-llvm:"
    # Options
    FETCH_BINARY
    "Use a binary distribution of the slang-llvm library instead of building or using LLVM"
    FETCH_BINARY_IF_POSSIBLE
    "Like FETCH_BINARY, except falls back to DISABLE if a prebuilt slang-llvm can't be downloaded"
    USE_SYSTEM_LLVM
    "Build slang-llvm using system-provided LLVM and Clang binaries"
    DISABLE
    "Do not build llvm or fetch slang-llvm"
)

if(SLANG_SLANG_LLVM_FLAVOR MATCHES FETCH_BINARY)
    # If the user didn't specify a URL, find the best one now
    if(NOT SLANG_SLANG_LLVM_BINARY_URL)
        get_best_slang_binary_release_url("${SLANG_GITHUB_TOKEN}" url)
        if(NOT DEFINED url)
            if(SLANG_SLANG_LLVM_FLAVOR STREQUAL FETCH_BINARY_IF_POSSIBLE)
                message(
                    WARNING
                    "Unable to find a prebuilt binary for slang-llvm, Slang will be built without LLVM support. Please consider setting SLANG_SLANG_LLVM_BINARY_URL manually"
                )
            else()
                message(
                    FATAL_ERROR
                    "Unable to find binary release for slang-llvm, please set a different SLANG_SLANG_LLVM_FLAVOR or set SLANG_SLANG_LLVM_BINARY_URL manually"
                )
            endif()
        endif()
    endif()
    set(SLANG_SLANG_LLVM_BINARY_URL
        ${url}
        CACHE STRING
        "URL specifying the location of the slang-llvm prebuilt library"
    )
endif()

if(NOT SLANG_EXCLUDE_DAWN)
    set(webgpu_dawn_release_tag "webgpu_dawn-0")
    if(
        CMAKE_SYSTEM_NAME MATCHES "Windows"
        AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64"
    )
        set(SLANG_WEBGPU_DAWN_BINARY_URL
            "https://github.com/shader-slang/dawn/releases/download/${webgpu_dawn_release_tag}/webgpu_dawn-windows-x64.zip"
        )
    endif()
endif()

if(NOT SLANG_EXCLUDE_TINT)
    set(slang_tint_release_tag "slang-tint-0")
    if(
        CMAKE_SYSTEM_NAME MATCHES "Windows"
        AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64"
    )
        set(SLANG_SLANG_TINT_BINARY_URL
            "https://github.com/shader-slang/dawn/releases/download/${slang_tint_release_tag}/slang-tint-windows-x64.zip"
        )
    endif()
endif()

#
# Option validation
#

if(NOT SLANG_EMBED_CORE_MODULE AND NOT SLANG_EMBED_CORE_MODULE_SOURCE)
    message(
        SEND_ERROR
        "One of SLANG_EMBED_CORE_MODULE and SLANG_EMBED_CORE_MODULE_SOURCE must be enabled"
    )
endif()

if(SLANG_ENABLE_OPTIX AND NOT SLANG_ENABLE_CUDA)
    message(
        SEND_ERROR
        "SLANG_ENABLE_OPTIX is not supported without SLANG_ENABLE_CUDA"
    )
endif()

if(SLANG_ENABLE_NVAPI AND NOT CMAKE_SYSTEM_NAME MATCHES "Windows")
    message(SEND_ERROR "SLANG_ENABLE_NVAPI is only supported on Windows")
endif()

if(SLANG_ENABLE_TESTS AND NOT SLANG_ENABLE_GFX)
    message(SEND_ERROR "SLANG_ENABLE_TESTS requires SLANG_ENABLE_GFX")
endif()

#
# Dependencies, most of these are however handled inside the "auto_option"
# calls above
#

find_package(Threads REQUIRED)

if(${SLANG_USE_SYSTEM_UNORDERED_DENSE})
    find_package(unordered_dense CONFIG QUIET)
endif()

add_subdirectory(external)

# webgpu_dawn is only available as a fetched shared library, since Dawn's nested source
# trees are too large and onerous for us to depend on.
# We ignore the failure to fetch the library, since it's not required for the build to succeed.
if(SLANG_WEBGPU_DAWN_BINARY_URL)
    copy_fetched_shared_library(
        "webgpu_dawn"
        "${SLANG_WEBGPU_DAWN_BINARY_URL}"
        IGNORE_FAILURE
        SLANG_GITHUB_TOKEN ${SLANG_GITHUB_TOKEN}
    )
endif()

# slang-tint is only available as a fetched shared library, since it's hosted in the Dawn
# repository, and Dawn's nested source trees are too large and onerous for us to depend
# on.
# We ignore the failure to fetch the library, since it's not required for the build to succeed.
if(SLANG_SLANG_TINT_BINARY_URL)
    copy_fetched_shared_library(
        "slang-tint"
        "${SLANG_SLANG_TINT_BINARY_URL}"
        IGNORE_FAILURE
        SLANG_GITHUB_TOKEN ${SLANG_GITHUB_TOKEN}
    )
endif()

fetch_or_build_slang_llvm()

#
# Our targets
#

add_subdirectory(source/core)
add_subdirectory(source/slang-rt)
add_subdirectory(source/compiler-core)
add_subdirectory(source/slang-wasm)
add_subdirectory(source/slang-glslang)
add_subdirectory(tools)
add_subdirectory(prelude)
add_subdirectory(source/slang-core-module)
add_subdirectory(source/slang-glsl-module)
add_subdirectory(source/slang)
add_subdirectory(source/slangc)
add_subdirectory(examples)

#
# Packaging
#
set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
set(CPACK_COMPONENTS_ALL_IN_ONE_PACKAGE ON)
set(CPACK_STRIP_FILES FALSE)
install(
    FILES "${slang_SOURCE_DIR}/README.md" "${slang_SOURCE_DIR}/LICENSE"
    DESTINATION .
    COMPONENT metadata
    EXCLUDE_FROM_ALL
)
install(
    DIRECTORY "${slang_SOURCE_DIR}/docs/"
    DESTINATION share/doc/slang
    PATTERN ".*" EXCLUDE
)
install(
    DIRECTORY "${slang_SOURCE_DIR}/include"
    DESTINATION .
    PATTERN ".*" EXCLUDE
)
include(CPack)

# Write basic package config version file using standard CMakePackageConfigHelpers utility
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${PROJECT_NAME}ConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

# Write SlangConfig.cmake which should allow find_pacakage(SLANG) to work correctly
# SlangConfig.cmake will define slang::slang target that can be linked with using
# target_link_libraries. It will also define SLANG_EXECUTABLE export variable that
# should point to slangc if SLANG_ENABLE_SLANGC is ON.
configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/SlangConfig.cmake.in"
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    INSTALL_DESTINATION cmake
)

# Conditionally handle the case for Emscripten where slang does not create
# linkable targets. In this case do not export the targets. Otherwise, just
# export the slang targets. 
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    if(NOT ${SLANG_LIB_TYPE} STREQUAL "STATIC")
        install(
            EXPORT SlangTargets
            FILE ${PROJECT_NAME}Targets.cmake
            NAMESPACE ${PROJECT_NAME}::
            DESTINATION cmake
        )
    endif()
endif()

install(
    FILES
        "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
        "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    DESTINATION cmake
)
