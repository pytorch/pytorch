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

function(tbb_conf_gen_print_help)
    message("Usage: cmake -DTBB_ROOT=<tbb_root> -DTBB_OS=Linux|Windows|Darwin [-DSAVE_TO=<path>] -P tbb_config_generator.cmake")
endfunction()

if (NOT DEFINED TBB_ROOT)
    tbb_conf_gen_print_help()
    message(FATAL_ERROR "Required parameter TBB_ROOT is not defined")
endif()

if (NOT EXISTS "${TBB_ROOT}")
    tbb_conf_gen_print_help()
    message(FATAL_ERROR "TBB_ROOT=${TBB_ROOT} does not exist")
endif()

if (NOT DEFINED TBB_OS)
    tbb_conf_gen_print_help()
    message(FATAL_ERROR "Required parameter TBB_OS is not defined")
endif()

if (DEFINED SAVE_TO)
    set(tbb_conf_gen_save_to_param SAVE_TO ${SAVE_TO})
endif()

include(${CMAKE_CURRENT_LIST_DIR}/TBBMakeConfig.cmake)
tbb_make_config(TBB_ROOT ${TBB_ROOT} CONFIG_DIR tbb_config_dir SYSTEM_NAME ${TBB_OS} ${tbb_conf_gen_save_to_param})

message(STATUS "TBBConfig files were created in ${tbb_config_dir}")
