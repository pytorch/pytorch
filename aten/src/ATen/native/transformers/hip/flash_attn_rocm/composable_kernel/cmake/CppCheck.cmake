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
include(ProcessorCount)
include(Analyzers)

find_program(CPPCHECK_EXE 
    NAMES 
        cppcheck
    PATHS
        /opt/rocm/bin
)

ProcessorCount(CPPCHECK_JOBS)

set(CPPCHECK_BUILD_DIR ${CMAKE_BINARY_DIR}/cppcheck-build)
file(MAKE_DIRECTORY ${CPPCHECK_BUILD_DIR})
set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${CPPCHECK_BUILD_DIR})

macro(enable_cppcheck)
    set(options FORCE)
    set(oneValueArgs)
    set(multiValueArgs CHECKS SUPPRESS DEFINE UNDEFINE INCLUDE SOURCES)

    cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    string(REPLACE ";" "," CPPCHECK_CHECKS "${PARSE_CHECKS}")
    string(REPLACE ";" "\n" CPPCHECK_SUPPRESS "${PARSE_SUPPRESS};*:/usr/*")
    file(WRITE ${CMAKE_BINARY_DIR}/cppcheck-supressions "${CPPCHECK_SUPPRESS}")
    set(CPPCHECK_DEFINES)
    foreach(DEF ${PARSE_DEFINE})
        set(CPPCHECK_DEFINES "${CPPCHECK_DEFINES} -D${DEF}")
    endforeach()

    set(CPPCHECK_UNDEFINES)
    foreach(DEF ${PARSE_UNDEFINE})
        set(CPPCHECK_UNDEFINES "${CPPCHECK_UNDEFINES} -U${DEF}")
    endforeach()

    set(CPPCHECK_INCLUDES)
    foreach(INC ${PARSE_INCLUDE})
        set(CPPCHECK_INCLUDES "${CPPCHECK_INCLUDES} -I${INC}")
    endforeach()

    # set(CPPCHECK_FORCE)
    set(CPPCHECK_FORCE "--project=${CMAKE_BINARY_DIR}/compile_commands.json")
    if(PARSE_FORCE)
        set(CPPCHECK_FORCE --force)
    endif()

    set(SOURCES)
    set(GLOBS)
    foreach(SOURCE ${PARSE_SOURCES})
        get_filename_component(ABS_SOURCE ${SOURCE} ABSOLUTE)
        if(EXISTS ${ABS_SOURCE})
            if(IS_DIRECTORY ${ABS_SOURCE})
                set(GLOBS "${GLOBS} ${ABS_SOURCE}/*.cpp ${ABS_SOURCE}/*.hpp ${ABS_SOURCE}/*.cxx ${ABS_SOURCE}/*.c ${ABS_SOURCE}/*.h")
            else()
                set(SOURCES "${SOURCES} ${ABS_SOURCE}")
            endif()
        else()
            set(GLOBS "${GLOBS} ${ABS_SOURCE}")
        endif()
    endforeach()

    file(WRITE ${CMAKE_BINARY_DIR}/cppcheck.cmake "
        file(GLOB_RECURSE GSRCS ${GLOBS})
        set(CPPCHECK_COMMAND
            ${CPPCHECK_EXE}
            -q
            # -v
            # --report-progress
            ${CPPCHECK_FORCE}
            --cppcheck-build-dir=${CPPCHECK_BUILD_DIR}
            --platform=native
            --template=gcc
            --error-exitcode=1
            -j ${CPPCHECK_JOBS}
            ${CPPCHECK_DEFINES}
            ${CPPCHECK_UNDEFINES}
            ${CPPCHECK_INCLUDES}
            --enable=${CPPCHECK_CHECKS}
            --inline-suppr
            --suppressions-list=${CMAKE_BINARY_DIR}/cppcheck-supressions
            ${SOURCES} \${GSRCS}
        )
        string(REPLACE \";\" \" \" CPPCHECK_SHOW_COMMAND \"\${CPPCHECK_COMMAND}\")
        message(\"\${CPPCHECK_SHOW_COMMAND}\")
        execute_process(
            COMMAND \${CPPCHECK_COMMAND}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            RESULT_VARIABLE RESULT
        )
        if(NOT RESULT EQUAL 0)
            message(FATAL_ERROR \"Cppcheck failed\")
        endif()
")

    add_custom_target(cppcheck
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_BINARY_DIR}/cppcheck.cmake
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "cppcheck: Running cppcheck..."
    )
    mark_as_analyzer(cppcheck)
endmacro()


