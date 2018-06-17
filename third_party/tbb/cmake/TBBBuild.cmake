# Copyright (c) 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
#
#

#
# Usage:
#  include(TBBBuild.cmake)
#  tbb_build(ROOT <tbb_root> MAKE_ARGS <arg1> [... <argN>])
#  find_package(TBB <options>)
#

include(CMakeParseArguments)

# Save the location of Intel TBB CMake modules here, as it will not be possible to do inside functions,
# see for details: https://cmake.org/cmake/help/latest/variable/CMAKE_CURRENT_LIST_DIR.html
set(_tbb_cmake_module_path ${CMAKE_CURRENT_LIST_DIR})

##
# Builds Intel TBB.
#
# Parameters:
#  TBB_ROOT   <directory> - path to Intel TBB root directory (with sources);
#  MAKE_ARGS  <list>      - user-defined arguments to be passed to make-tool;
#  CONFIG_DIR <variable>  - store location of the created TBBConfig if the build was ok, store <variable>-NOTFOUND otherwise.
#
function(tbb_build)
    # NOTE: internal function are used to hide them from user.

    ##
    # Provides arguments for make-command to build Intel TBB.
    #
    # Following arguments are provided automatically if they are not defined by user:
    #  compiler=<value>
    #  tbb_build_dir=<value>
    #  tbb_build_prefix=<value>
    #  -j<n>
    #
    # Parameters:
    #  USER_DEFINED_ARGS <list> - list of user-defined arguments;
    #  RESULT <variable> - resulting list of 'make' arguments.
    #
    function(tbb_get_make_args)
        set(oneValueArgs RESULT)
        set(multiValueArgs USER_DEFINED_ARGS)
        cmake_parse_arguments(tbb_GMA "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

        set(result ${tbb_GMA_USER_DEFINED_ARGS})

        if (NOT tbb_GMA_USER_DEFINED_ARGS MATCHES "compiler=")
            # TODO: add other supported compilers.
            if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
                set(compiler gcc)
            elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
                set(compiler icc)
                if (CMAKE_SYSTEM_NAME MATCHES "Windows")
                    set(compiler icl)
                endif()
            elseif (MSVC)
                set(compiler cl)
            elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
                set(compiler clang)
            endif()

            set(result "compiler=${compiler}" ${result})
        endif()

        if (NOT tbb_GMA_USER_DEFINED_ARGS MATCHES "tbb_build_dir=")
            set(result "tbb_build_dir=${CMAKE_CURRENT_BINARY_DIR}/tbb_cmake_build" ${result})
        endif()

        if (NOT tbb_GMA_USER_DEFINED_ARGS MATCHES "tbb_build_prefix=")
            set(result "tbb_build_prefix=tbb_cmake_build_subdir" ${result})
        endif()

        if (NOT tbb_GMA_USER_DEFINED_ARGS MATCHES "(;|^) *\\-j[0-9]* *(;|$)")
            include(ProcessorCount)
            ProcessorCount(num_of_cores)
            if (NOT num_of_cores EQUAL 0)
                set(result "-j${num_of_cores}" ${result})
            endif()
        endif()

        if (CMAKE_SYSTEM_NAME MATCHES "Android")
            set(result target=android ${result})
        endif()

        set(${tbb_GMA_RESULT} ${result} PARENT_SCOPE)
    endfunction()

    ##
    # Provides release and debug directories basing on 'make' arguments.
    #
    # Following 'make' arguments are parsed: tbb_build_dir, tbb_build_prefix
    #
    # Parameters:
    #  MAKE_ARGS   <list>     - 'make' arguments (tbb_build_dir and tbb_build_prefix are required)
    #  RELEASE_DIR <variable> - store normalized (CMake) path to release directory
    #  DEBUG_DIR   <variable> - store normalized (CMake) path to debug directory
    #
    function(tbb_get_build_paths_from_make_args)
        set(oneValueArgs RELEASE_DIR DEBUG_DIR)
        set(multiValueArgs MAKE_ARGS)
        cmake_parse_arguments(tbb_GBPFMA "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

        foreach(arg ${tbb_GBPFMA_MAKE_ARGS})
            if (arg MATCHES "tbb_build_dir=")
                string(REPLACE "tbb_build_dir=" "" tbb_build_dir "${arg}")
            elseif (arg MATCHES "tbb_build_prefix=")
                string(REPLACE "tbb_build_prefix=" "" tbb_build_prefix "${arg}")
            endif()
        endforeach()

        set(tbb_release_dir "${tbb_build_dir}/${tbb_build_prefix}_release")
        set(tbb_debug_dir "${tbb_build_dir}/${tbb_build_prefix}_debug")

        file(TO_CMAKE_PATH "${tbb_release_dir}" tbb_release_dir)
        file(TO_CMAKE_PATH "${tbb_debug_dir}" tbb_debug_dir)

        set(${tbb_GBPFMA_RELEASE_DIR} ${tbb_release_dir} PARENT_SCOPE)
        set(${tbb_GBPFMA_DEBUG_DIR} ${tbb_debug_dir} PARENT_SCOPE)
    endfunction()

    # -------------------- #
    # Function entry point #
    # -------------------- #
    set(oneValueArgs TBB_ROOT CONFIG_DIR)
    set(multiValueArgs MAKE_ARGS)
    cmake_parse_arguments(tbb_build "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT EXISTS "${tbb_build_TBB_ROOT}/Makefile" OR NOT EXISTS "${tbb_build_TBB_ROOT}/src")
        message(STATUS "Intel TBB can not be built: Makefile or src directory was not found in ${tbb_build_TBB_ROOT}")
        set(${tbb_build_CONFIG_DIR} ${tbb_build_CONFIG_DIR}-NOTFOUND PARENT_SCOPE)
        return()
    endif()

    set(make_tool_name make)
    if (CMAKE_SYSTEM_NAME MATCHES "Windows")
        set(make_tool_name gmake)
    elseif (CMAKE_SYSTEM_NAME MATCHES "Android")
        set(make_tool_name ndk-build)
    endif()

    find_program(TBB_MAKE_TOOL ${make_tool_name} DOC "Make-tool to build Intel TBB.")
    mark_as_advanced(TBB_MAKE_TOOL)

    if (NOT TBB_MAKE_TOOL)
        message(STATUS "Intel TBB can not be built: required make-tool (${make_tool_name}) was not found")
        set(${tbb_build_CONFIG_DIR} ${tbb_build_CONFIG_DIR}-NOTFOUND PARENT_SCOPE)
        return()
    endif()

    tbb_get_make_args(USER_DEFINED_ARGS ${tbb_build_MAKE_ARGS} RESULT tbb_make_args)

    set(tbb_build_cmd ${TBB_MAKE_TOOL} ${tbb_make_args})

    string(REPLACE ";" " " tbb_build_cmd_str "${tbb_build_cmd}")
    message(STATUS "Building Intel TBB: ${tbb_build_cmd_str}")
    execute_process(COMMAND ${tbb_build_cmd}
                    WORKING_DIRECTORY ${tbb_build_TBB_ROOT}
                    RESULT_VARIABLE tbb_build_result
                    ERROR_VARIABLE tbb_build_error_output
                    OUTPUT_QUIET)

    if (NOT tbb_build_result EQUAL 0)
        message(STATUS "Building is unsuccessful (${tbb_build_result}): ${tbb_build_error_output}")
        set(${tbb_build_CONFIG_DIR} ${tbb_build_CONFIG_DIR}-NOTFOUND PARENT_SCOPE)
        return()
    endif()

    tbb_get_build_paths_from_make_args(MAKE_ARGS ${tbb_make_args}
                                       RELEASE_DIR tbb_release_dir
                                       DEBUG_DIR tbb_debug_dir)

    include(${_tbb_cmake_module_path}/TBBMakeConfig.cmake)
    tbb_make_config(TBB_ROOT ${tbb_build_TBB_ROOT}
                    SYSTEM_NAME ${CMAKE_SYSTEM_NAME}
                    CONFIG_DIR tbb_config_dir
                    CONFIG_FOR_SOURCE
                    TBB_RELEASE_DIR ${tbb_release_dir}
                    TBB_DEBUG_DIR ${tbb_debug_dir})

    set(${tbb_build_CONFIG_DIR} ${tbb_config_dir} PARENT_SCOPE)
endfunction()
