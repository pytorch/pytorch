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

include(CMakeParseArguments)

# Save the location of Intel TBB CMake modules here, as it will not be possible to do inside functions,
# see for details: https://cmake.org/cmake/help/latest/variable/CMAKE_CURRENT_LIST_DIR.html
set(_tbb_cmake_module_path ${CMAKE_CURRENT_LIST_DIR})

##
# Downloads file.
#
# Parameters:
#  URL     <url>      - URL to download data from;
#  SAVE_AS <filename> - filename there to save downloaded data;
#  INFO    <string>   - text description of content to be downloaded;
#                       will be printed as message in format is "Downloading <INFO>: <URL>;
#  FORCE              - option to delete local file from SAVE_AS if it exists;
#
function(_tbb_download_file)
    set(options FORCE)
    set(oneValueArgs URL RELEASE SAVE_AS INFO)
    cmake_parse_arguments(tbb_df "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (tbb_df_FORCE AND EXISTS "${tbb_df_SAVE_AS}")
        file(REMOVE ${tbb_df_SAVE_AS})
    endif()

    if (NOT EXISTS "${tbb_df_SAVE_AS}")
        set(_show_progress)
        if (TBB_DOWNLOADING_PROGRESS)
            set(_show_progress SHOW_PROGRESS)
        endif()

        message(STATUS "Downloading ${tbb_df_INFO}: ${tbb_df_URL}")
        file(DOWNLOAD ${tbb_df_URL} ${tbb_df_SAVE_AS} ${_show_progress} STATUS download_status)

        list(GET download_status 0 download_status_num)
        if (NOT download_status_num EQUAL 0)
            message(STATUS "Unsuccessful downloading: ${download_status}")
            file(REMOVE ${tbb_df_SAVE_AS})
            return()
        endif()
    else()
        message(STATUS "Needed file was found locally ${tbb_df_SAVE_AS}. Remove it if you still want to download a new one")
    endif()
endfunction()

##
# Checks if specified Intel TBB release is available on GitHub.
#
# tbb_check_git_release(<release> <result>)
# Parameters:
#  <release_tag> - release to be checked;
#  <result>  - store result (TRUE/FALSE).
#
function(_tbb_check_git_release_tag _tbb_release_tag _tbb_release_tag_avail)
    if (_tbb_release_tag STREQUAL LATEST)
        set(${_tbb_release_tag_avail} TRUE PARENT_SCOPE)
        return()
    endif()

    set(tbb_releases_file "${CMAKE_CURRENT_BINARY_DIR}/tbb_releases.json")

    _tbb_download_file(URL     "${tbb_github_api}/releases"
                       SAVE_AS ${tbb_releases_file}
                       INFO    "information from GitHub about Intel TBB releases"
                       FORCE)

    if (NOT EXISTS "${tbb_releases_file}")
        set(${_tbb_release_tag_avail} FALSE PARENT_SCOPE)
        return()
    endif()

    file(READ ${tbb_releases_file} tbb_releases)

    string(REPLACE "\"" "" tbb_releases ${tbb_releases})
    string(REGEX MATCHALL "tag_name: *([A-Za-z0-9_\\.]+)" tbb_releases ${tbb_releases})

    set(_release_available FALSE)
    foreach(tbb_rel ${tbb_releases})
        string(REGEX REPLACE "tag_name: *" "" tbb_rel_cut ${tbb_rel})
        list(REMOVE_ITEM tbb_releases ${tbb_rel})
        list(APPEND tbb_releases ${tbb_rel_cut})
        if (_tbb_release_tag STREQUAL tbb_rel_cut)
            set(_release_available TRUE)
            break()
        endif()
    endforeach()

    if (NOT _release_available)
        string(REPLACE ";" ", " tbb_releases_str "${tbb_releases}")
        message(STATUS "Requested release tag ${_tbb_release_tag} is not available. Available Intel TBB release tags: ${tbb_releases_str}")
    endif()

    set(${_tbb_release_tag_avail} ${_release_available} PARENT_SCOPE)
endfunction()

##
# Compares two Intel TBB releases and provides result
# TRUE if the first release is less than the second, FALSE otherwise.
#
# tbb_is_release_less(<rel1> <rel2> <result>)
#
function(_tbb_is_release_less rel1 rel2 result)
    # Convert release to numeric representation to compare it using "if" with VERSION_LESS.
    string(REGEX REPLACE "[A-Za-z]" "" rel1 "${rel1}")
    string(REPLACE "_" "." rel1 "${rel1}")
    string(REGEX REPLACE "[A-Za-z]" "" rel2 "${rel2}")
    string(REPLACE "_" "." rel2 "${rel2}")

    if (${rel1} VERSION_LESS ${rel2})
        set(${result} TRUE PARENT_SCOPE)
        return()
    endif()

    set(${result} FALSE PARENT_SCOPE)
endfunction()

##
# Finds exact URL to download Intel TBB basing on provided parameters.
#
# Usage:
#  _tbb_get_url(URL <var_to_save_url> RELEASE_TAG <release_tag|LATEST> OS <os> [SOURCE_CODE])
#
function(_tbb_get_url)
    set(oneValueArgs URL RELEASE_TAG OS)
    set(options SOURCE_CODE)
    cmake_parse_arguments(tbb_get_url "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(tbb_github_api "https://api.github.com/repos/01org/tbb")

    _tbb_check_git_release_tag(${tbb_get_url_RELEASE_TAG} tbb_release_available)
    if (NOT tbb_release_available)
        set(${tbb_download_FULL_PATH} ${tbb_download_FULL_PATH}-NOTFOUND PARENT_SCOPE)
        return()
    endif()

    if (tbb_get_url_RELEASE_TAG STREQUAL LATEST)
        set(tbb_rel_info_api_url "${tbb_github_api}/releases/latest")
    else()
        set(tbb_rel_info_api_url "${tbb_github_api}/releases/tags/${tbb_get_url_RELEASE_TAG}")
    endif()

    set(tbb_release_info_file "${CMAKE_CURRENT_BINARY_DIR}/tbb_${tbb_get_url_RELEASE_TAG}_info.json")

    _tbb_download_file(URL     ${tbb_rel_info_api_url}
                       SAVE_AS ${tbb_release_info_file}
                       INFO    "information from GitHub about packages for Intel TBB ${tbb_get_url_RELEASE_TAG}"
                       FORCE)

    if (NOT EXISTS "${tbb_release_info_file}")
        set(${tbb_get_url_URL} ${tbb_get_url_URL}-NOTFOUND PARENT_SCOPE)
        return()
    endif()

    file(STRINGS ${tbb_release_info_file} tbb_release_info)

    if (tbb_get_url_SOURCE_CODE)
        # Find name of the latest release to get link to source archive.
        if (tbb_get_url_RELEASE_TAG STREQUAL LATEST)
            string(REPLACE "\"" "" tbb_release_info ${tbb_release_info})
            string(REGEX REPLACE ".*tag_name: *([A-Za-z0-9_\\.]+).*" "\\1" tbb_get_url_RELEASE_TAG "${tbb_release_info}")
        endif()

        set(${tbb_get_url_URL} "https://github.com/01org/tbb/archive/${tbb_get_url_RELEASE_TAG}.tar.gz" PARENT_SCOPE)
    else()
        if (tbb_get_url_OS MATCHES "Linux")
            set(tbb_lib_archive_suffix lin.tgz)
        elseif (tbb_get_url_OS MATCHES "Windows")
            set(tbb_lib_archive_suffix win.zip)
        elseif (tbb_get_url_OS MATCHES "Darwin")
            set(tbb_lib_archive_suffix mac.tgz)

            # Since 2017_U4 release archive for Apple has suffix "mac.tgz" instead of "osx.tgz".
            if (NOT tbb_get_url_RELEASE_TAG STREQUAL "LATEST")
                _tbb_is_release_less(${tbb_get_url_RELEASE_TAG} 2017_U4 release_less)
                if (release_less)
                    set(tbb_lib_archive_suffix osx.tgz)
                endif()
            endif()
        elseif (tbb_get_url_OS MATCHES "Android")
            set(tbb_lib_archive_suffix and.tgz)
        else()
            message(STATUS "Currently prebuilt Intel TBB is not available for your OS (${tbb_get_url_OS})")
            set(${tbb_get_url_URL} ${tbb_get_url_URL}-NOTFOUND PARENT_SCOPE)
            return()
        endif()

        string(REGEX REPLACE ".*(https.*oss_${tbb_lib_archive_suffix}).*" "\\1" tbb_bin_url "${tbb_release_info}")

        set(${tbb_get_url_URL} ${tbb_bin_url} PARENT_SCOPE)
    endif()
endfunction()

function(tbb_get)
    set(oneValueArgs RELEASE_TAG SYSTEM_NAME SAVE_TO TBB_ROOT CONFIG_DIR)
    set(options SOURCE_CODE)
    cmake_parse_arguments(tbb_get "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(tbb_os ${CMAKE_SYSTEM_NAME})
    if (tbb_get_SYSTEM_NAME)
        set(tbb_os ${tbb_get_SYSTEM_NAME})
    endif()

    set(tbb_release_tag LATEST)
    if (tbb_get_RELEASE_TAG)
        set(tbb_release_tag ${tbb_get_RELEASE_TAG})
    endif()

    set(tbb_save_to ${CMAKE_CURRENT_BINARY_DIR}/tbb_downloaded)
    if (tbb_get_SAVE_TO)
        set(tbb_save_to ${tbb_get_SAVE_TO})
    endif()

    if (tbb_get_SOURCE_CODE)
        _tbb_get_url(URL tbb_url RELEASE_TAG ${tbb_release_tag} OS ${tbb_os} SOURCE_CODE)
    else()
        _tbb_get_url(URL tbb_url RELEASE_TAG ${tbb_release_tag} OS ${tbb_os})
    endif()

    if (NOT tbb_url)
        message(STATUS "URL to download Intel TBB has not been found")
        set(${tbb_get_TBB_ROOT} ${tbb_get_TBB_ROOT}-NOTFOUND PARENT_SCOPE)
        return()
    endif()

    get_filename_component(filename ${tbb_url} NAME)
    set(local_file "${CMAKE_CURRENT_BINARY_DIR}/${filename}")

    _tbb_download_file(URL     ${tbb_url}
                       SAVE_AS ${local_file}
                       INFO    "Intel TBB library")

    if (NOT EXISTS "${local_file}")
        set(${tbb_get_TBB_ROOT} ${tbb_get_TBB_ROOT}-NOTFOUND PARENT_SCOPE)
        return()
    endif()

    get_filename_component(subdir_name ${filename} NAME_WE)
    file(MAKE_DIRECTORY ${tbb_save_to}/${subdir_name})
    if (NOT EXISTS "${tbb_save_to}/${subdir_name}")
        message(STATUS "${tbb_save_to}/${subdir_name} can not be created")
        set(${tbb_get_TBB_ROOT} ${tbb_get_TBB_ROOT}-NOTFOUND PARENT_SCOPE)
        return()
    endif()

    message(STATUS "Unpacking ${local_file} to ${tbb_save_to}/${subdir_name}")
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf ${local_file}
                    WORKING_DIRECTORY ${tbb_save_to}/${subdir_name}
                    RESULT_VARIABLE unpacking_result)

    if (NOT unpacking_result EQUAL 0)
        message(STATUS "Unsuccessful unpacking: ${unpacking_result}")
        set(${tbb_get_TBB_ROOT} ${tbb_get_TBB_ROOT}-NOTFOUND PARENT_SCOPE)
        return()
    endif()

    file(GLOB_RECURSE tbb_h ${tbb_save_to}/${subdir_name}/*/include/tbb/tbb.h)
    list(GET tbb_h 0 tbb_h)

    if (NOT EXISTS "${tbb_h}")
        message(STATUS "tbb/tbb.h has not been found in the downloaded package")
        set(${tbb_get_TBB_ROOT} ${tbb_get_TBB_ROOT}-NOTFOUND PARENT_SCOPE)
        return()
    endif()

    get_filename_component(tbb_root "${tbb_h}" PATH)
    get_filename_component(tbb_root "${tbb_root}" PATH)
    get_filename_component(tbb_root "${tbb_root}" PATH)

    if (NOT tbb_get_SOURCE_CODE)
        set(tbb_config_dir ${tbb_root}/cmake)

        if (NOT EXISTS "${tbb_config_dir}")
            tbb_make_config(TBB_ROOT ${tbb_root} CONFIG_DIR tbb_config_dir)
        endif()

        set(${tbb_get_CONFIG_DIR} ${tbb_config_dir} PARENT_SCOPE)
    endif()

    set(${tbb_get_TBB_ROOT} ${tbb_root} PARENT_SCOPE)
endfunction()
