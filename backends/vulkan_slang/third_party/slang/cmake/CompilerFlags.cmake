#
# Given a list of flags, add those which the C++ compiler supports to the target
#
include(CheckCXXCompilerFlag)
function(add_supported_cxx_flags target)
    cmake_parse_arguments(ARG "PRIVATE;PUBLIC;INTERFACE" "" "" ${ARGN})
    set(flags ${ARG_UNPARSED_ARGUMENTS})
    if(ARG_PRIVATE)
        set(private PRIVATE)
    endif()
    if(ARG_PUBLIC)
        set(public PUBLIC)
    endif()
    if(ARG_INTERFACE)
        set(interface INTERFACE)
    endif()

    foreach(flag ${flags})
        # remove the `no-` prefix from warnings because gcc doesn't treat it as an
        # error on its own
        string(REGEX REPLACE "\\-Wno\\-(.+)" "-W\\1" flag_to_test "${flag}")
        string(
            REGEX REPLACE
            "[^a-zA-Z0-9]+"
            "_"
            test_name
            "CXXFLAG_${flag_to_test}"
        )
        check_cxx_compiler_flag("${flag_to_test}" ${test_name})
        if(${test_name})
            target_compile_options(
                ${target}
                ${private}
                ${public}
                ${interface}
                ${flag}
            )
        endif()
    endforeach()
endfunction()

#
# Given a list of linker flags, add those which the compiler supports to the
# target
#
include(CheckLinkerFlag)
function(add_supported_cxx_linker_flags target)
    cmake_parse_arguments(ARG "PRIVATE;PUBLIC;INTERFACE;BEFORE" "" "" ${ARGN})
    set(flags ${ARG_UNPARSED_ARGUMENTS})
    if(ARG_BEFORE)
        set(before BEFORE)
    endif()
    if(ARG_PRIVATE)
        set(private PRIVATE)
    endif()
    if(ARG_PUBLIC)
        set(public PUBLIC)
    endif()
    if(ARG_INTERFACE)
        set(interface INTERFACE)
    endif()

    foreach(flag ${flags})
        string(
            REGEX REPLACE
            "[^a-zA-Z0-9]+"
            "_"
            test_name
            "CXXLINKFLAG_${flag}"
        )
        check_linker_flag(CXX "${flag}" ${test_name})
        if(${test_name})
            target_link_options(
                ${target}
                ${before}
                ${private}
                ${public}
                ${interface}
                ${flag}
            )
        endif()
    endforeach()
endfunction()

#
# Add our default compiler flags to a target
#
# Pass USE_EXTRA_WARNINGS to enable -WExtra or /W3
#
function(set_default_compile_options target)
    cmake_parse_arguments(
        ARG
        "USE_EXTRA_WARNINGS;USE_FEWER_WARNINGS"
        ""
        ""
        ${ARGN}
    )

    set(warning_flags)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        list(
            APPEND
            warning_flags
            -Wall
            # Disabled warnings:
            -Wno-switch
            -Wno-parentheses
            -Wno-unused-local-typedefs
            -Wno-class-memaccess
            -Wno-assume
            -Wno-reorder
            -Wno-invalid-offsetof
            -Wno-newline-eof
            -Wno-return-std-move
            # Enabled warnings:
            # If a function returns an address/reference to a local, we want it to
            # produce an error, because it probably means something very bad.
            -Werror=return-local-addr
            # Some warnings which are on by default in MSVC
            -Wnarrowing
        )
        if(ARG_USE_EXTRA_WARNINGS)
            list(APPEND warning_flags -Wextra)
        endif()
        if(ARG_USE_FEWER_WARNINGS)
            list(
                APPEND
                warning_flags
                -Wno-class-memaccess
                -Wno-unused-variable
                -Wno-unused-parameter
                -Wno-sign-compare
                -Wno-unused-function
                -Wno-unused-value
                -Wno-unused-but-set-variable
                -Wno-implicit-fallthrough
                -Wno-missing-field-initializers
                -Wno-strict-aliasing
                -Wno-maybe-uninitialized
            )
        endif()
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        list(APPEND warning_flags)
        if(ARG_USE_EXTRA_WARNINGS)
            list(APPEND warning_flags /W4)
        elseif(ARG_USE_FEWER_WARNINGS)
            list(APPEND warning_flags /W0)
        else()
            list(APPEND warning_flags /W2)
        endif()
    endif()

    add_supported_cxx_flags(${target} PRIVATE ${warning_flags})

    if(NOT WIN32)
        # these options are for ELF specific and not for Windows
        add_supported_cxx_linker_flags(
            ${target}
            PRIVATE
            # Don't assume that symbols will be resolved at runtime
            "-Wl,--no-undefined"
            # No reason not to do this? Useful when using split debug info
            "-Wl,--build-id"
        )
    endif()

    set_target_properties(
        ${target}
        PROPERTIES # -fvisibility=hidden
            CXX_VISIBILITY_PRESET
            hidden
            C_VISIBILITY_PRESET
            hidden
            VISIBILITY_INLINES_HIDDEN
            ON
            # C++ standard
            CXX_STANDARD
            20
            # pic
            POSITION_INDEPENDENT_CODE
            ON
    )

    target_compile_definitions(
        ${target}
        PRIVATE # Add _DEBUG depending on the build configuration
            $<$<CONFIG:Debug>:_DEBUG>
            # For including windows.h in a way that minimized namespace
            # pollution. Although we define these here, we still set them
            # manually in any header files which may be included by another
            # project
            WIN32_LEAN_AND_MEAN
            VC_EXTRALEAN
            NOMINMAX
            # Use multi-byte character set on Windows
            UNICODE
            _UNICODE
    )

    #
    # Settings dependent on config options
    #

    target_compile_definitions(
        ${target}
        PRIVATE
            SLANG_ENABLE_DXIL_SUPPORT=$<BOOL:${SLANG_ENABLE_DXIL}>
            $<$<BOOL:${SLANG_ENABLE_FULL_DEBUG_VALIDATION}>:SLANG_ENABLE_FULL_IR_VALIDATION>
            $<$<BOOL:${SLANG_ENABLE_IR_BREAK_ALLOC}>:SLANG_ENABLE_IR_BREAK_ALLOC>
            $<$<BOOL:${SLANG_ENABLE_DX_ON_VK}>:SLANG_CONFIG_DX_ON_VK>
            $<$<STREQUAL:${SLANG_LIB_TYPE},STATIC>:STB_IMAGE_STATIC>
    )

    if(SLANG_ENABLE_ASAN)
        add_supported_cxx_flags(
            ${target}
            PRIVATE
            /fsanitize=address
            -fsanitize=address
        )
        add_supported_cxx_linker_flags(
            ${target}
            BEFORE
            PUBLIC
            /INCREMENTAL:NO
            -fsanitize=address
        )
    endif()
endfunction()
