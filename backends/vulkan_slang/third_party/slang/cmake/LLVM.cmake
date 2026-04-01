# A convenience on top of the llvm package's cmake files, this creates a target
# to pass to target_link_libraries which correctly pulls in the llvm include
# dir and other compile dependencies
function(llvm_target_from_components target_name)
    set(components ${ARGN})
    llvm_map_components_to_libnames(llvm_libs
        ${components}
    )
    add_library(${target_name} INTERFACE)
    target_link_libraries(${target_name} INTERFACE ${llvm_libs})
    target_include_directories(
        ${target_name}
        SYSTEM
        INTERFACE ${LLVM_INCLUDE_DIRS}
    )
    target_compile_definitions(${target_name} INTERFACE ${LLVM_DEFINITIONS})
    if(NOT LLVM_ENABLE_RTTI)
        # Make sure that we don't disable rtti if this library wasn't compiled with
        # support
        add_supported_cxx_flags(${target_name} INTERFACE -fno-rtti /GR-)
    endif()
endfunction()

# The same for clang
function(clang_target_from_libs target_name)
    set(clang_libs ${ARGN})
    add_library(${target_name} INTERFACE)
    target_link_libraries(${target_name} INTERFACE ${clang_libs})
    target_include_directories(
        ${target_name}
        SYSTEM
        INTERFACE ${CLANG_INCLUDE_DIRS}
    )
    target_compile_definitions(${target_name} INTERFACE ${CLANG_DEFINITIONS})
    if(NOT LLVM_ENABLE_RTTI)
        # Make sure that we don't disable rtti if this library wasn't compiled with
        # support
        add_supported_cxx_flags(${target_name} INTERFACE -fno-rtti /GR-)
    endif()
endfunction()

function(fetch_or_build_slang_llvm)
    if(SLANG_SLANG_LLVM_FLAVOR STREQUAL "FETCH_BINARY")
        install_fetched_shared_library(
            "slang-llvm"
            "${SLANG_SLANG_LLVM_BINARY_URL}"
        )
    elseif(SLANG_SLANG_LLVM_FLAVOR STREQUAL "FETCH_BINARY_IF_POSSIBLE")
        if(SLANG_SLANG_LLVM_BINARY_URL)
            install_fetched_shared_library(
                "slang-llvm"
                "${SLANG_SLANG_LLVM_BINARY_URL}"
                IGNORE_FAILURE
            )
            if(NOT TARGET slang-llvm)
                message(
                    WARNING
                    "Unable to fetch slang-llvm prebuilt binary, configuring without LLVM support"
                )
            endif()
        endif()
    elseif(SLANG_SLANG_LLVM_FLAVOR STREQUAL "USE_SYSTEM_LLVM")
        find_package(LLVM 13.0 REQUIRED CONFIG)
        find_package(Clang REQUIRED CONFIG)

        llvm_target_from_components(llvm-dep filecheck native orcjit)
        clang_target_from_libs(
            clang-dep
            clangBasic
            clangCodeGen
            clangDriver
            clangLex
            clangFrontend
            clangFrontendTool
        )
        slang_add_target(
            source/slang-llvm
            MODULE
            LINK_WITH_PRIVATE core compiler-core llvm-dep clang-dep
            # We include slang.h, but don't need to link with it
            INCLUDE_FROM_PRIVATE slang
            # We include tools/slang-test/filecheck.h, but don't need to link
            # with it and it might not be a target if SLANG_ENABLE_TESTS is
            # false, so just include the directory manually here
            INCLUDE_DIRECTORIES_PRIVATE ${slang_SOURCE_DIR}/tools
            # This uses the SLANG_DLL_EXPORT macro from slang.h, so make sure to set
            # SLANG_DYNAMIC and SLANG_DYNAMIC_EXPORT
            EXPORT_MACRO_PREFIX SLANG
            INSTALL
            INSTALL_COMPONENT slang-llvm
            EXPORT_SET_NAME SlangTargets
        )
        # If we don't include this, then the symbols in the LLVM linked here may
        # conflict with those of other LLVMs linked at runtime, for instance in mesa.
        add_supported_cxx_linker_flags(
            slang-llvm
            PRIVATE
            "-Wl,--exclude-libs,ALL"
        )

        # The LLVM headers need a warning disabling, which somehow slips through \external
        if(MSVC)
            target_compile_options(slang-llvm PRIVATE -wd4244)
        endif()

        # TODO: Put a check here that libslang-llvm.so doesn't have a 'NEEDED'
        # directive for libLLVM-13.so, it's almost certainly going to break at
        # runtime in surprising ways when linked alongside Mesa (or anything else
        # pulling in libLLVM.so)
    endif()

    if(SLANG_ENABLE_PREBUILT_BINARIES)
        if(CMAKE_SYSTEM_NAME MATCHES "Windows")
            # DX Agility SDK requires the D3D12*.DLL files to be placed under a sub-directory, "D3D12".
            # https://devblogs.microsoft.com/directx/gettingstarted-dx12agility/#d3d12sdkpath-should-not-be-the-same-directory-as-the-application-exe
            file(
                GLOB prebuilt_binaries
                "${slang_SOURCE_DIR}/external/slang-binaries/bin/windows-x64/*"
            )
            file(
                GLOB prebuilt_d3d12_binaries
                "${slang_SOURCE_DIR}/external/slang-binaries/bin/windows-x64/[dD]3[dD]12*"
            )
            list(REMOVE_ITEM prebuilt_binaries ${prebuilt_d3d12_binaries})
            add_custom_target(
                copy-prebuilt-binaries
                ALL
                COMMAND
                    ${CMAKE_COMMAND} -E make_directory
                    ${CMAKE_BINARY_DIR}/$<CONFIG>/${runtime_subdir}
                COMMAND
                    ${CMAKE_COMMAND} -E copy_if_different ${prebuilt_binaries}
                    ${CMAKE_BINARY_DIR}/$<CONFIG>/${runtime_subdir}
                COMMAND
                    ${CMAKE_COMMAND} -E make_directory
                    ${CMAKE_BINARY_DIR}/$<CONFIG>/${runtime_subdir}/D3D12
                COMMAND
                    ${CMAKE_COMMAND} -E copy_if_different
                    ${prebuilt_d3d12_binaries}
                    ${CMAKE_BINARY_DIR}/$<CONFIG>/${runtime_subdir}/D3D12
                VERBATIM
            )
            set_target_properties(
                copy-prebuilt-binaries
                PROPERTIES FOLDER external
            )
        endif()
    endif()
endfunction()
