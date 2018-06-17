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

#include "tbb/queuing_mutex.h"
#include "tbb/tbb_machine.h"
#include "tbb/tbb_stddef.h"
#include "tbb_misc.h"
#include "itt_notify.h"

namespace tbb {

using namespace internal;

//! A method to acquire queuing_mutex lock
void queuing_mutex::scoped_lock::acquire( queuing_mutex& m )
{
    __TBB_ASSERT( !this->mutex, "scoped_lock is already holding a mutex");

    // Must set all fields before the fetch_and_store, because once the
    // fetch_and_store executes, *this becomes accessible to other threads.
    mutex = &m;
    next  = NULL;
    going = 0;

    // The fetch_and_store must have release semantics, because we are
    // "sending" the fields initialized above to other processors.
    scoped_lock* pred = m.q_tail.fetch_and_store<tbb::release>(this);
    if( pred ) {
        ITT_NOTIFY(sync_prepare, mutex);
#if TBB_USE_ASSERT
        __TBB_control_consistency_helper(); // on "m.q_tail"
        __TBB_ASSERT( !pred->next, "the predecessor has another successor!");
#endif
        pred->next = this;
        spin_wait_while_eq( going, 0ul );
    }
    ITT_NOTIFY(sync_acquired, mutex);

    // Force acquire so that user's critical section receives correct values
    // from processor that was previously in the user's critical section.
    __TBB_load_with_acquire(going);
}

//! A method to acquire queuing_mutex if it is free
bool queuing_mutex::scoped_lock::try_acquire( queuing_mutex& m )
{
    __TBB_ASSERT( !this->mutex, "scoped_lock is already holding a mutex");

    // Must set all fields before the fetch_and_store, because once the
    // fetch_and_store executes, *this becomes accessible to other threads.
    next  = NULL;
    going = 0;

    // The CAS must have release semantics, because we are
    // "sending" the fields initialized above to other processors.
    if( m.q_tail.compare_and_swap<tbb::release>(this, NULL) )
        return false;

    // Force acquire so that user's critical section receives correct values
    // from processor that was previously in the user's critical section.
    __TBB_load_with_acquire(going);
    mutex = &m;
    ITT_NOTIFY(sync_acquired, mutex);
    return true;
}

//! A method to release queuing_mutex lock
void queuing_mutex::scoped_lock::release( )
{
    __TBB_ASSERT(this->mutex!=NULL, "no lock acquired");

    ITT_NOTIFY(sync_releasing, mutex);
    if( !next ) {
        if( this == mutex->q_tail.compare_and_swap<tbb::release>(NULL, this) ) {
            // this was the only item in the queue, and the queue is now empty.
            goto done;
        }
        // Someone in the queue
        spin_wait_while_eq( next, (scoped_lock*)0 );
    }
    __TBB_ASSERT(next,NULL);
    __TBB_store_with_release(next->going, 1);
done:
    initialize();
}

void queuing_mutex::internal_construct() {
    ITT_SYNC_CREATE(this, _T("tbb::queuing_mutex"), _T(""));
}

} // namespace tbb
