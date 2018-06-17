/*
    Copyright (c) 2017-2018 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.




*/

#ifndef __IPC_UTILS_H
#define __IPC_UTILS_H

namespace tbb {
namespace internal {
namespace rml {

char* get_shared_name(const char* prefix);
int get_num_threads(const char* env_var);
bool get_enable_flag(const char* env_var);

}}} //tbb::internal::rml

#endif
