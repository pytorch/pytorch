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

#include "tbb/tbb_machine.h"
#include "tbb/spin_mutex.h"
#include "itt_notify.h"
#include "tbb_misc.h"

namespace tbb {

void spin_mutex::scoped_lock::internal_acquire( spin_mutex& m ) {
    __TBB_ASSERT( !my_mutex, "already holding a lock on a spin_mutex" );
    ITT_NOTIFY(sync_prepare, &m);
    __TBB_LockByte(m.flag);
    my_mutex = &m;
    ITT_NOTIFY(sync_acquired, &m);
}

void spin_mutex::scoped_lock::internal_release() {
    __TBB_ASSERT( my_mutex, "release on spin_mutex::scoped_lock that is not holding a lock" );

    ITT_NOTIFY(sync_releasing, my_mutex);
    __TBB_UnlockByte(my_mutex->flag);
    my_mutex = NULL;
}

bool spin_mutex::scoped_lock::internal_try_acquire( spin_mutex& m ) {
    __TBB_ASSERT( !my_mutex, "already holding a lock on a spin_mutex" );
    bool result = bool( __TBB_TryLockByte(m.flag) );
    if( result ) {
        my_mutex = &m;
        ITT_NOTIFY(sync_acquired, &m);
    }
    return result;
}

void spin_mutex::internal_construct() {
    ITT_SYNC_CREATE(this, _T("tbb::spin_mutex"), _T(""));
}

} // namespace tbb
