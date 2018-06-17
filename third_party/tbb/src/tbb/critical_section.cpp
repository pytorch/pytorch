/*
    Copyright (c) 2005-2018 Intel Corporation

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

#include "tbb/critical_section.h"
#include "itt_notify.h"

namespace tbb {
    namespace internal {

void critical_section_v4::internal_construct() {
    ITT_SYNC_CREATE(&my_impl, _T("ppl::critical_section"), _T(""));
}
}  // namespace internal
}  // namespace tbb
