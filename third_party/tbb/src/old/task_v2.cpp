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

/*  This compilation unit provides definition of task::destroy( task& )
    that is binary compatible with TBB 2.x. In TBB 3.0, the method became
    static, and its name decoration changed, though the definition remained.

    The macro switch should be set prior to including task.h
    or any TBB file that might bring task.h up.
*/
#define __TBB_DEPRECATED_TASK_INTERFACE 1
#include "tbb/task.h"

namespace tbb {

void task::destroy( task& victim ) {
    // Forward to static version
    task_base::destroy( victim );
}

} // namespace tbb
