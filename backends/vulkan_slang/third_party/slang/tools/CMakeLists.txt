#
# Tools used to generate source during the build:
#

add_custom_target(
    all-generators
    COMMENT "meta target which depends on all generators"
)
set_target_properties(all-generators PROPERTIES FOLDER generators)

if(CMAKE_HOST_WIN32)
    set(CMAKE_HOST_EXECUTABLE_SUFFIX ".exe")
else()
    set(CMAKE_HOST_EXECUTABLE_SUFFIX "")
endif()
function(generator dir)
    if(SLANG_GENERATORS_PATH)
        cmake_parse_arguments(ARG "" "TARGET_NAME" "" ${ARGN})
        if(ARG_TARGET_NAME)
            set(target ${ARG_TARGET_NAME})
        else()
            get_filename_component(target ${dir} NAME)
        endif()
        add_executable(${target} IMPORTED GLOBAL)
        set_property(
            TARGET ${target}
            PROPERTY
                IMPORTED_LOCATION
                    "${SLANG_GENERATORS_PATH}/${target}${CMAKE_HOST_EXECUTABLE_SUFFIX}"
        )
    else()
        slang_add_target(
            ${dir}
            EXECUTABLE
            EXCLUDE_FROM_ALL
            USE_FEWER_WARNINGS
            LINK_WITH_PRIVATE core
            OUTPUT_DIR generators
            REQUIRED_BY all-generators
            FOLDER generators
            INSTALL_COMPONENT generators
            EXPORT_SET_NAME SlangGeneratorTargets
            ${ARGN}
        )
    endif()
endfunction()

generator(slang-fiddle LINK_WITH_PRIVATE compiler-core)
generator(slang-embed)
generator(slang-generate USE_FEWER_WARNINGS)
generator(slang-lookup-generator LINK_WITH_PRIVATE compiler-core)
generator(slang-capability-generator LINK_WITH_PRIVATE compiler-core)
generator(slang-spirv-embed-generator LINK_WITH_PRIVATE compiler-core)
generator(
    ${slang_SOURCE_DIR}/source/slangc
    TARGET_NAME
    slang-bootstrap
    USE_FEWER_WARNINGS
    EXTRA_COMPILE_DEFINITIONS_PRIVATE
    SLANG_BOOTSTRAP=1
    LINK_WITH_PUBLIC
    slang-without-embedded-core-module
    LINK_WITH_PRIVATE
    prelude
    slang-capability-lookup
    slang-lookup-tables
    Threads::Threads
)

slang_add_target(
    slang-cpp-parser
    STATIC
    EXPORT_MACRO_PREFIX SLANG
    USE_FEWER_WARNINGS
    LINK_WITH_PRIVATE core compiler-core
    INCLUDE_DIRECTORIES_PUBLIC .
    FOLDER generators
)

#
# Language Server
#
if(SLANG_ENABLE_SLANGD)
    slang_add_target(
        slangd
        EXECUTABLE
        LINK_WITH_PRIVATE
            core
            compiler-core
            slang
            slang-fiddle-output
            slang-capability-defs
            Threads::Threads
        INSTALL
        EXPORT_SET_NAME SlangTargets
    )
endif()

#
# Slang Interpreter
#
if(SLANG_ENABLE_SLANGI)
    slang_add_target(
        slangi
        EXECUTABLE
        LINK_WITH_PRIVATE core compiler-core slang
        INSTALL
        EXPORT_SET_NAME SlangTargets
    )
endif()

if(SLANG_ENABLE_GFX)
    #
    # `platform` contains all the platform abstractions for a GUI application.
    #
    slang_add_target(
        platform
        ${SLANG_LIB_TYPE}
        EXCLUDE_FROM_ALL
        USE_FEWER_WARNINGS
        LINK_WITH_PRIVATE
            core
            imgui
            stb
            $<$<BOOL:${SLANG_ENABLE_XLIB}>:X11::X11>
            "$<$<PLATFORM_ID:Darwin>:-framework Cocoa>"
            "$<$<PLATFORM_ID:Darwin>:-framework QuartzCore>"
            ${CMAKE_DL_LIBS}
        LINK_WITH_FRAMEWORK Foundation Cocoa QuartzCore
        EXTRA_COMPILE_DEFINITIONS_PRIVATE
            $<$<BOOL:${SLANG_ENABLE_XLIB}>:SLANG_ENABLE_XLIB=1>
        INCLUDE_FROM_PRIVATE gfx
        INCLUDE_DIRECTORIES_PUBLIC . platform
        EXPORT_MACRO_PREFIX SLANG_PLATFORM
    )

    #
    # GFX
    #
    slang_add_target(
        gfx
        ${SLANG_LIB_TYPE}
        USE_FEWER_WARNINGS
        LINK_WITH_PRIVATE
            core
            slang
            Vulkan-Headers
            metal-cpp
            stb
            $<$<BOOL:${SLANG_ENABLE_XLIB}>:X11::X11>
            $<$<BOOL:${SLANG_ENABLE_CUDA}>:CUDA::cuda_driver>
            $<$<BOOL:${SLANG_ENABLE_NVAPI}>:${NVAPI_LIBRARIES}>
        LINK_WITH_FRAMEWORK Foundation Cocoa QuartzCore Metal
        EXTRA_COMPILE_DEFINITIONS_PRIVATE
            $<$<BOOL:${SLANG_ENABLE_CUDA}>:GFX_ENABLE_CUDA>
            $<$<BOOL:${SLANG_ENABLE_OPTIX}>:GFX_OPTIX>
            $<$<BOOL:${SLANG_ENABLE_NVAPI}>:GFX_NVAPI>
            $<$<BOOL:${SLANG_ENABLE_XLIB}>:SLANG_ENABLE_XLIB>
        # This is a shared library, so we need to set a preprocessor macro to mark
        # exported symbols
        EXPORT_MACRO_PREFIX SLANG_GFX
        # slang-gfx is in this directory, anything which depends on gfx should include
        # this
        INCLUDE_DIRECTORIES_PUBLIC
            ${slang_SOURCE_DIR}
            ${slang_SOURCE_DIR}/include
        INCLUDE_DIRECTORIES_PRIVATE ${NVAPI_INCLUDE_DIRS}
        INSTALL
        EXPORT_SET_NAME SlangTargets
        FOLDER gfx
    )
    if(SLANG_ENABLE_TESTS)
        set(modules_dest_dir $<TARGET_FILE_DIR:slang-test>)
        add_custom_target(
            copy-gfx-slang-modules
            COMMAND ${CMAKE_COMMAND} -E make_directory ${modules_dest_dir}
            COMMAND
                ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_LIST_DIR}/gfx/gfx.slang
                ${modules_dest_dir}/gfx.slang
            COMMAND
                ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_LIST_DIR}/gfx/slang.slang
                ${modules_dest_dir}/slang.slang
            WORKING_DIRECTORY ${slang_SOURCE_DIR}
            VERBATIM
        )
        set_target_properties(
            copy-gfx-slang-modules
            PROPERTIES FOLDER generators
        )
        install(
            FILES ${modules_dest_dir}/gfx.slang ${modules_dest_dir}/slang.slang
            DESTINATION ${runtime_subdir}
        )
        add_dependencies(gfx copy-gfx-slang-modules)
    endif()

    slang_add_target(
        gfx-util
        STATIC
        LINK_WITH_PRIVATE core
        INCLUDE_FROM_PRIVATE gfx
        # The headers are included with 'include "gfx-util/blah.h"' which is found
        # in the tools directory
        INCLUDE_DIRECTORIES_PUBLIC .
        FOLDER gfx
    )
endif()

#
# The test executables and runtime-loaded modules
#
if(SLANG_ENABLE_TESTS)
    slang_add_target(
        test-server
        EXECUTABLE
        EXCLUDE_FROM_ALL
        LINK_WITH_PRIVATE core compiler-core slang
        INCLUDE_FROM_PRIVATE unit-test
        FOLDER test
    )
    slang_add_target(
        test-process
        EXECUTABLE
        EXCLUDE_FROM_ALL
        LINK_WITH_PRIVATE core compiler-core
        FOLDER test
    )

    slang_add_target(
        slang-test
        EXECUTABLE
        USE_FEWER_WARNINGS
        INCLUDE_DIRECTORIES_PUBLIC .
        LINK_WITH_PRIVATE
            core
            compiler-core
            slang
            stb
            ${CMAKE_DL_LIBS}
            Threads::Threads
        REQUIRES
            # Shared libraries dlopened by slang-test
            slang-reflection-test
            slang-unit-test
            # Used by some tests when they run
            slangd
            test-server
            test-process
        OPTIONAL_REQUIRES
            gfx
            slang-rt
            slang-glslang
            slang-llvm
            copy-webgpu_dawn
            copy-slang-tint
            # See issue 5305.
            copy-prebuilt-binaries
        FOLDER test
        DEBUG_DIR ${slang_SOURCE_DIR}
    )
    set_property(
        DIRECTORY ${slang_SOURCE_DIR}
        PROPERTY VS_STARTUP_PROJECT slang-test
    )
    include(CTest)
    add_test(
        NAME slang-test
        COMMAND
            slang-test -bindir
            ${slang_SOURCE_DIR}/build/$<CONFIG>/${runtime_subdir}
            -expected-failure-list
            ${slang_SOURCE_DIR}/tests/expected-failure.txt
            -expected-failure-list
            ${slang_SOURCE_DIR}/tests/expected-failure-github.txt
        WORKING_DIRECTORY ${slang_SOURCE_DIR}
    )

    slang_add_target(
        unit-test
        OBJECT
        EXCLUDE_FROM_ALL
        INCLUDE_DIRECTORIES_PUBLIC .
        INCLUDE_FROM_PRIVATE slang core
        INCLUDE_DIRECTORIES_PRIVATE ${slang_SOURCE_DIR}/include
        FOLDER test
    )

    # These are libraries loaded at runtime from the test executable:
    if(SLANG_ENABLE_GFX)
        slang_add_target(
            gfx-unit-test
            MODULE
            EXCLUDE_FROM_ALL
            EXTRA_COMPILE_DEFINITIONS_PRIVATE SLANG_SHARED_LIBRARY_TOOL
            USE_FEWER_WARNINGS
            LINK_WITH_PRIVATE core slang unit-test gfx gfx-util platform stb
            OUTPUT_NAME gfx-unit-test-tool
            REQUIRED_BY slang-test
            FOLDER test/tools
        )
        if(SLANG_ENABLE_SLANG_RHI)
            slang_add_target(
                render-test
                MODULE
                EXCLUDE_FROM_ALL
                EXTRA_COMPILE_DEFINITIONS_PRIVATE SLANG_SHARED_LIBRARY_TOOL
                USE_FEWER_WARNINGS
                LINK_WITH_PRIVATE
                    core
                    compiler-core
                    slang
                    slang-rhi
                    platform
                    stb
                    $<$<BOOL:${SLANG_ENABLE_CUDA}>:CUDA::cuda_driver>
                EXTRA_COMPILE_DEFINITIONS_PRIVATE
                    $<$<BOOL:${SLANG_ENABLE_CUDA}>:RENDER_TEST_CUDA>
                    $<$<BOOL:${SLANG_ENABLE_OPTIX}>:RENDER_TEST_OPTIX>
                EXTRA_COMPILE_OPTIONS_PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/EHa>
                OUTPUT_NAME render-test-tool
                REQUIRED_BY slang-test
                FOLDER test/tools
            )
        endif()
    endif()
    slang_add_target(
        slang-unit-test
        MODULE
        EXCLUDE_FROM_ALL
        EXTRA_COMPILE_DEFINITIONS_PRIVATE SLANG_SHARED_LIBRARY_TOOL
        USE_FEWER_WARNINGS
        LINK_WITH_PRIVATE core compiler-core unit-test slang Threads::Threads
        OUTPUT_NAME slang-unit-test-tool
        FOLDER test/tools
    )
    slang_add_target(
        slang-reflection-test
        MODULE
        EXCLUDE_FROM_ALL
        EXTRA_COMPILE_DEFINITIONS_PRIVATE SLANG_SHARED_LIBRARY_TOOL
        USE_FEWER_WARNINGS
        LINK_WITH_PRIVATE core slang Threads::Threads
        OUTPUT_NAME slang-reflection-test-tool
        FOLDER test/tools
    )

    slang_add_target(
        slang-profile
        EXECUTABLE
        EXCLUDE_FROM_ALL
        LINK_WITH_PRIVATE core slang
        FOLDER test
    )
endif()

#
# slang-replay tool for replaying the record files
#
if(SLANG_ENABLE_REPLAYER)
    slang_add_target(
        slang-replay
        EXECUTABLE
        EXTRA_SOURCE_DIRS
            ${slang_SOURCE_DIR}/source/slang-record-replay/replay
            ${slang_SOURCE_DIR}/source/slang-record-replay/util
        LINK_WITH_PRIVATE core compiler-core slang
        INCLUDE_DIRECTORIES_PUBLIC
            ${slang_SOURCE_DIR}/source/slang-record-replay
        FOLDER test
    )
endif()
