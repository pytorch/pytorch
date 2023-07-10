################################################################################
#
# MIT License
#
# Copyright (c) 2017 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################
include(CMakeParseArguments)
include(Analyzers)

get_filename_component(CLANG_TIDY_EXE_HINT "${CMAKE_CXX_COMPILER}" PATH)

find_program(CLANG_TIDY_EXE
    NAMES
        clang-tidy
        clang-tidy-5.0
        clang-tidy-4.0
        clang-tidy-3.9
        clang-tidy-3.8
        clang-tidy-3.7
        clang-tidy-3.6
        clang-tidy-3.5
    HINTS
        ${CLANG_TIDY_EXE_HINT}
    PATH_SUFFIXES
        compiler/bin
    PATHS
        /opt/rocm/llvm/bin
        /opt/rocm/hcc
        /usr/local/opt/llvm/bin
)

function(find_clang_tidy_version VAR)
    execute_process(COMMAND ${CLANG_TIDY_EXE} -version OUTPUT_VARIABLE VERSION_OUTPUT)
    separate_arguments(VERSION_OUTPUT_LIST UNIX_COMMAND "${VERSION_OUTPUT}")
    list(FIND VERSION_OUTPUT_LIST "version" VERSION_INDEX)
    if(VERSION_INDEX GREATER 0)
        math(EXPR VERSION_INDEX "${VERSION_INDEX} + 1")
        list(GET VERSION_OUTPUT_LIST ${VERSION_INDEX} VERSION)
        set(${VAR} ${VERSION} PARENT_SCOPE)
    else()
        set(${VAR} "0.0" PARENT_SCOPE)
    endif()

endfunction()

if( NOT CLANG_TIDY_EXE )
    message( STATUS "Clang tidy not found" )
    set(CLANG_TIDY_VERSION "0.0")
else()
    find_clang_tidy_version(CLANG_TIDY_VERSION)
    message( STATUS "Clang tidy found: ${CLANG_TIDY_VERSION}")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CLANG_TIDY_FIXIT_DIR ${CMAKE_BINARY_DIR}/fixits)
file(MAKE_DIRECTORY ${CLANG_TIDY_FIXIT_DIR})
set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${CLANG_TIDY_FIXIT_DIR})

macro(enable_clang_tidy)
    set(options ANALYZE_TEMPORARY_DTORS ALL)
    set(oneValueArgs HEADER_FILTER)
    set(multiValueArgs CHECKS ERRORS EXTRA_ARGS)

    cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    string(REPLACE ";" "," CLANG_TIDY_CHECKS "${PARSE_CHECKS}")
    string(REPLACE ";" "," CLANG_TIDY_ERRORS "${PARSE_ERRORS}")
    set(CLANG_TIDY_EXTRA_ARGS)
    foreach(ARG ${PARSE_EXTRA_ARGS})
        list(APPEND CLANG_TIDY_EXTRA_ARGS "-extra-arg=${ARG}")
    endforeach()

    set(CLANG_TIDY_ALL)
    if(PARSE_ALL)
        set(CLANG_TIDY_ALL ALL)
    endif()

    message(STATUS "Clang tidy checks: ${CLANG_TIDY_CHECKS}")

    if (${PARSE_ANALYZE_TEMPORARY_DTORS})
        set(CLANG_TIDY_ANALYZE_TEMPORARY_DTORS "-analyze-temporary-dtors")
    endif()

    if (${CLANG_TIDY_VERSION} VERSION_LESS "3.9.0")
        set(CLANG_TIDY_ERRORS_ARG "")
    else()
        set(CLANG_TIDY_ERRORS_ARG "-warnings-as-errors='${CLANG_TIDY_ERRORS}'")
    endif()

    if (${CLANG_TIDY_VERSION} VERSION_LESS "3.9.0")
        set(CLANG_TIDY_QUIET_ARG "")
    else()
        set(CLANG_TIDY_QUIET_ARG "-quiet")
    endif()

    if(PARSE_HEADER_FILTER)
        string(REPLACE "$" "$$" CLANG_TIDY_HEADER_FILTER "${PARSE_HEADER_FILTER}")
    else()
        set(CLANG_TIDY_HEADER_FILTER ".*")
    endif()

    set(CLANG_TIDY_COMMAND
        ${CLANG_TIDY_EXE}
        ${CLANG_TIDY_QUIET_ARG}
        -p ${CMAKE_BINARY_DIR}
        -checks='${CLANG_TIDY_CHECKS}'
        ${CLANG_TIDY_ERRORS_ARG}
        ${CLANG_TIDY_EXTRA_ARGS}
        ${CLANG_TIDY_ANALYZE_TEMPORARY_DTORS}
        -header-filter='${CLANG_TIDY_HEADER_FILTER}'
    )
    add_custom_target(tidy ${CLANG_TIDY_ALL})
    mark_as_analyzer(tidy)
    add_custom_target(tidy-base)
    add_custom_target(tidy-make-fixit-dir COMMAND ${CMAKE_COMMAND} -E make_directory ${CLANG_TIDY_FIXIT_DIR})
    add_custom_target(tidy-rm-fixit-dir COMMAND ${CMAKE_COMMAND} -E remove_directory ${CLANG_TIDY_FIXIT_DIR})
    add_dependencies(tidy-make-fixit-dir tidy-rm-fixit-dir)
    add_dependencies(tidy-base tidy-make-fixit-dir)
endmacro()

function(clang_tidy_check TARGET)
    get_target_property(SOURCES ${TARGET} SOURCES)
    # TODO: Use generator expressions instead
    # COMMAND ${CLANG_TIDY_COMMAND} $<TARGET_PROPERTY:${TARGET},SOURCES>
    # COMMAND ${CLANG_TIDY_COMMAND} $<JOIN:$<TARGET_PROPERTY:${TARGET},SOURCES>, >
    foreach(SOURCE ${SOURCES})
        if((NOT "${SOURCE}" MATCHES "(h|hpp|hxx)$") AND (NOT "${SOURCE}" MATCHES "TARGET_OBJECTS"))
            string(MAKE_C_IDENTIFIER "${SOURCE}" tidy_file)
            set(tidy_target tidy-target-${TARGET}-${tidy_file})
            add_custom_target(${tidy_target}
                # for some targets clang-tidy not able to get information from .clang-tidy
                DEPENDS ${SOURCE}
                COMMAND ${CLANG_TIDY_COMMAND} "-config=\{CheckOptions: \[\{key: bugprone-reserved-identifier.AllowedIdentifiers,value: __HIP_PLATFORM_HCC__\; __HIP_ROCclr__\}\]\}" ${SOURCE} "-export-fixes=${CLANG_TIDY_FIXIT_DIR}/${TARGET}-${tidy_file}.yaml"
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                COMMENT "clang-tidy: Running clang-tidy on target ${SOURCE}..."
            )
            add_dependencies(${tidy_target} ${TARGET})
            add_dependencies(${tidy_target} tidy-base)
            add_dependencies(tidy ${tidy_target})
        endif()
    endforeach()
endfunction()

