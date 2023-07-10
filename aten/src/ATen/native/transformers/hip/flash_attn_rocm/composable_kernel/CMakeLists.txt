cmake_minimum_required(VERSION 3.14)

# Check support for CUDA/HIP in Cmake
project(composable_kernel)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

enable_testing()

set(ROCM_SYMLINK_LIBS OFF)
find_package(ROCM REQUIRED PATHS /opt/rocm)

include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
include(ROCMSetupVersion)
include(ROCMInstallSymlinks)
include(ROCMCreatePackage)
include(CheckCXXCompilerFlag)

rocm_setup_version(VERSION 0.2.0)
include(TargetFlags)
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_INSTALL_PREFIX} ${CMAKE_INSTALL_PREFIX}/llvm ${CMAKE_INSTALL_PREFIX}/hip /opt/rocm /opt/rocm/llvm /opt/rocm/hip)

option(USE_BITINT_EXTENSION_INT4, "Whether to enable clang's BitInt extension to provide int4 data type." OFF)

if(USE_BITINT_EXTENSION_INT4)
    add_compile_definitions(CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4)
    add_compile_options(-Wno-bit-int-extension)
    message("CK compiled with USE_BITINT_EXTENSION_INT4 set to ${USE_BITINT_EXTENSION_INT4}")
endif()

## Threads
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
link_libraries(Threads::Threads)

## C++
enable_language(CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
message("CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}")

## OpenMP
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
	# workaround issue hipcc in rocm3.5 cannot find openmp
	set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
	set(OpenMP_CXX_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument")
	set(OpenMP_CXX_LIB_NAMES "libomp" "libgomp" "libiomp5")
	set(OpenMP_libomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
	set(OpenMP_libgomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
	set(OpenMP_libiomp5_LIBRARY ${OpenMP_CXX_LIB_NAMES})
else()
	find_package(OpenMP REQUIRED)
endif()

message("OpenMP_CXX_LIB_NAMES: ${OpenMP_CXX_LIB_NAMES}")
message("OpenMP_gomp_LIBRARY: ${OpenMP_gomp_LIBRARY}")
message("OpenMP_pthread_LIBRARY: ${OpenMP_pthread_LIBRARY}")
message("OpenMP_CXX_FLAGS: ${OpenMP_CXX_FLAGS}")

link_libraries(${OpenMP_gomp_LIBRARY})
link_libraries(${OpenMP_pthread_LIBRARY})

## HIP
find_package(HIP REQUIRED)
# Override HIP version in config.h, if necessary.
# The variables set by find_package() can't be overwritten,
# therefore let's use intermediate variables.
set(CK_HIP_VERSION_MAJOR "${HIP_VERSION_MAJOR}")
set(CK_HIP_VERSION_MINOR "${HIP_VERSION_MINOR}")
set(CK_HIP_VERSION_PATCH "${HIP_VERSION_PATCH}")
if( DEFINED CK_OVERRIDE_HIP_VERSION_MAJOR )
    set(CK_HIP_VERSION_MAJOR "${CK_OVERRIDE_HIP_VERSION_MAJOR}")
    message(STATUS "CK_HIP_VERSION_MAJOR overriden with ${CK_OVERRIDE_HIP_VERSION_MAJOR}")
endif()
if( DEFINED CK_OVERRIDE_HIP_VERSION_MINOR )
    set(CK_HIP_VERSION_MINOR "${CK_OVERRIDE_HIP_VERSION_MINOR}")
    message(STATUS "CK_HIP_VERSION_MINOR overriden with ${CK_OVERRIDE_HIP_VERSION_MINOR}")
endif()
if( DEFINED CK_OVERRIDE_HIP_VERSION_PATCH )
    set(CK_HIP_VERSION_PATCH "${CK_OVERRIDE_HIP_VERSION_PATCH}")
    message(STATUS "CK_HIP_VERSION_PATCH overriden with ${CK_OVERRIDE_HIP_VERSION_PATCH}")
endif()
message(STATUS "Build with HIP ${HIP_VERSION}")
link_libraries(hip::device)
add_compile_definitions(__HIP_PLATFORM_HCC__=1)

## tidy
include(EnableCompilerWarnings)
set(CK_TIDY_ERRORS ERRORS * -readability-inconsistent-declaration-parameter-name)
if(CMAKE_CXX_COMPILER MATCHES ".*hcc" OR CMAKE_CXX_COMPILER MATCHES ".*clang\\+\\+")
    set(CK_TIDY_CHECKS -modernize-use-override -readability-non-const-parameter)
# Enable tidy on hip
elseif(CK_BACKEND STREQUAL "HIP" OR CK_BACKEND STREQUAL "HIPNOGPU")
    set(CK_TIDY_ERRORS ALL)
endif()


include(ClangTidy)
enable_clang_tidy(
    CHECKS
        *
        -abseil-*
        -android-cloexec-fopen
        # Yea we shouldn't be using rand()
        -cert-msc30-c
        -bugprone-exception-escape
        -bugprone-macro-parentheses
        -cert-env33-c
        -cert-msc32-c
        -cert-msc50-cpp
        -cert-msc51-cpp
        -cert-dcl37-c
        -cert-dcl51-cpp
        -clang-analyzer-alpha.core.CastToStruct
        -clang-analyzer-optin.performance.Padding
        -clang-diagnostic-deprecated-declarations
        -clang-diagnostic-extern-c-compat
        -clang-diagnostic-unused-command-line-argument
        -cppcoreguidelines-avoid-c-arrays
        -cppcoreguidelines-avoid-magic-numbers
        -cppcoreguidelines-explicit-virtual-functions
        -cppcoreguidelines-init-variables
        -cppcoreguidelines-macro-usage
        -cppcoreguidelines-non-private-member-variables-in-classes
        -cppcoreguidelines-pro-bounds-array-to-pointer-decay
        -cppcoreguidelines-pro-bounds-constant-array-index
        -cppcoreguidelines-pro-bounds-pointer-arithmetic
        -cppcoreguidelines-pro-type-member-init
        -cppcoreguidelines-pro-type-reinterpret-cast
        -cppcoreguidelines-pro-type-union-access
        -cppcoreguidelines-pro-type-vararg
        -cppcoreguidelines-special-member-functions
        -fuchsia-*
        -google-explicit-constructor
        -google-readability-braces-around-statements
        -google-readability-todo
        -google-runtime-int
        -google-runtime-references
        -hicpp-vararg
        -hicpp-braces-around-statements
        -hicpp-explicit-conversions
        -hicpp-named-parameter
        -hicpp-no-array-decay
        # We really shouldn't use bitwise operators with signed integers, but
        # opencl leaves us no choice
        -hicpp-avoid-c-arrays
        -hicpp-signed-bitwise
        -hicpp-special-member-functions
        -hicpp-uppercase-literal-suffix
        -hicpp-use-auto
        -hicpp-use-equals-default
        -hicpp-use-override
        -llvm-header-guard
        -llvm-include-order
        #-llvmlibc-*
        -llvmlibc-restrict-system-libc-headers
        -llvmlibc-callee-namespace
        -llvmlibc-implementation-in-namespace
        -llvm-else-after-return
        -llvm-qualified-auto
        -misc-misplaced-const
        -misc-non-private-member-variables-in-classes
        -misc-no-recursion
        -modernize-avoid-bind
        -modernize-avoid-c-arrays
        -modernize-pass-by-value
        -modernize-use-auto
        -modernize-use-default-member-init
        -modernize-use-equals-default
        -modernize-use-trailing-return-type
        -modernize-use-transparent-functors
        -performance-unnecessary-value-param
        -readability-braces-around-statements
        -readability-else-after-return
        # we are not ready to use it, but very useful
        -readability-function-cognitive-complexity
        -readability-isolate-declaration
        -readability-magic-numbers
        -readability-named-parameter
        -readability-uppercase-literal-suffix
        -readability-convert-member-functions-to-static
        -readability-qualified-auto
        -readability-redundant-string-init
        # too many narrowing conversions in our code
        -bugprone-narrowing-conversions
        -cppcoreguidelines-narrowing-conversions
        -altera-struct-pack-align
        -cppcoreguidelines-prefer-member-initializer
        ${CK_TIDY_CHECKS}
        ${CK_TIDY_ERRORS}
    HEADER_FILTER
        "\.hpp$"
    EXTRA_ARGS
        -DCK_USE_CLANG_TIDY
)

include(CppCheck)
enable_cppcheck(
    CHECKS
        warning
        style
        performance
        portability
    SUPPRESS
        ConfigurationNotChecked
        constStatement
        duplicateCondition
        noExplicitConstructor
        passedByValue
        preprocessorErrorDirective
        shadowVariable
        unusedFunction
        unusedPrivateFunction
        unusedStructMember
        unmatchedSuppression
    FORCE
    SOURCES
        library/src
    INCLUDE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/library/include
    DEFINE
        CPPCHECK=1
        __linux__=1
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

include_directories(BEFORE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/library/include
    ${HIP_INCLUDE_DIRS}
)


SET(BUILD_DEV ON CACHE BOOL "BUILD_DEV")
if(BUILD_DEV)
    add_compile_options(-Werror)
    add_compile_options(-Weverything)
endif()
message("CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR})

file(GLOB_RECURSE INSTANCE_FILES "${PROJECT_SOURCE_DIR}/*/device_*_instance.cpp")
file(GLOB dir_list RELATIVE ${PROJECT_SOURCE_DIR}/library/src/tensor_operation_instance/gpu ${PROJECT_SOURCE_DIR}/library/src/tensor_operation_instance/gpu/*)
set(CK_DEVICE_INSTANCES)
FOREACH(subdir_path ${dir_list})
    IF(IS_DIRECTORY "${PROJECT_SOURCE_DIR}/library/src/tensor_operation_instance/gpu/${subdir_path}")
       list(APPEND CK_DEVICE_INSTANCES device_${subdir_path}_instance)
    ENDIF()
ENDFOREACH()
add_custom_target(instances DEPENDS utility;${CK_DEVICE_INSTANCES}  SOURCES ${INSTANCE_FILES})

rocm_package_setup_component(tests
        LIBRARY_NAME composablekernel
        PACKAGE_NAME tests # Prevent -static suffix on package name
)

rocm_package_setup_component(examples
        LIBRARY_NAME composablekernel
        PACKAGE_NAME examples
)

rocm_package_setup_component(profiler
        LIBRARY_NAME composablekernel
        PACKAGE_NAME ckProfiler
)

add_subdirectory(library)
add_subdirectory(example)
add_subdirectory(test)
add_subdirectory(profiler)

#Create an interface target for the include only files and call it "composablekernels"
include(CMakePackageConfigHelpers)

set(version 1.0.0)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/composable_kernelConfigVersion.cmake"
    VERSION "${version}"
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/composable_kernelConfig.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/composable_kernel
        NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

rocm_install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/composable_kernelConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/composable_kernelConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/composable_kernel
)

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_RPM_PACKAGE_LICENSE "MIT")

rocm_create_package(
    NAME composablekernel
    DESCRIPTION "High Performance Composable Kernel for AMD GPUs"
    MAINTAINER "MIOpen Kernels Dev Team <dl.MIOpen@amd.com>"
    LDCONFIG
    HEADER_ONLY
)
