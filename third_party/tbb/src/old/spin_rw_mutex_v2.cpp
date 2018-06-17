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

#include "spin_rw_mutex_v2.h"
#include "tbb/tbb_machine.h"
#include "../tbb/itt_notify.h"
#include "tbb/atomic.h"

namespace tbb {

using namespace internal;

static inline bool CAS(volatile uintptr_t &addr, uintptr_t newv, uintptr_t oldv) {
    return as_atomic(addr).compare_and_swap(newv, oldv) == oldv;
}

//! Signal that write lock is released
void spin_rw_mutex::internal_itt_releasing(spin_rw_mutex *mutex) {
    __TBB_ASSERT_EX(mutex, NULL); // To prevent compiler warnings
    ITT_NOTIFY(sync_releasing, mutex);
}

//! Acquire write (exclusive) lock on the given mutex.
bool spin_rw_mutex::internal_acquire_writer(spin_rw_mutex *mutex)
{
    ITT_NOTIFY(sync_prepare, mutex);
    for( atomic_backoff backoff;;backoff.pause() ) {
        state_t s = mutex->state;
        if( !(s & BUSY) ) { // no readers, no writers
            if( CAS(mutex->state, WRITER, s) )
                break; // successfully stored writer flag
            backoff.reset(); // we could be very close to complete op.
        } else if( !(s & WRITER_PENDING) ) { // no pending writers
            __TBB_AtomicOR(&mutex->state, WRITER_PENDING);
        }
    }
    ITT_NOTIFY(sync_acquired, mutex);
    __TBB_ASSERT( (mutex->state & BUSY)==WRITER, "invalid state of a write lock" );
    return false;
}

//! Release write lock on the given mutex
void spin_rw_mutex::internal_release_writer(spin_rw_mutex *mutex) {
    __TBB_ASSERT( (mutex->state & BUSY)==WRITER, "invalid state of a write lock" );
    ITT_NOTIFY(sync_releasing, mutex);
    mutex->state = 0; 
}

//! Acquire read (shared) lock on the given mutex.
void spin_rw_mutex::internal_acquire_reader(spin_rw_mutex *mutex) {
    ITT_NOTIFY(sync_prepare, mutex);
    for( atomic_backoff backoff;;backoff.pause() ) {
        state_t s = mutex->state;
        if( !(s & (WRITER|WRITER_PENDING)) ) { // no writer or write requests
            if( CAS(mutex->state, s+ONE_READER, s) )
                break; // successfully stored increased number of readers
            backoff.reset(); // we could be very close to complete op.
        }
    }
    ITT_NOTIFY(sync_acquired, mutex);
    __TBB_ASSERT( mutex->state & READERS, "invalid state of a read lock: no readers" );
    __TBB_ASSERT( !(mutex->state & WRITER), "invalid state of a read lock: active writer" );
}

//! Upgrade reader to become a writer.
/** Returns whether the upgrade happened without releasing and re-acquiring the lock */
bool spin_rw_mutex::internal_upgrade(spin_rw_mutex *mutex) {
    state_t s = mutex->state;
    __TBB_ASSERT( s & READERS, "invalid state before upgrade: no readers " );
    __TBB_ASSERT( !(s & WRITER), "invalid state before upgrade: active writer " );
    // check and set writer-pending flag
    // required conditions: either no pending writers, or we are the only reader
    // (with multiple readers and pending writer, another upgrade could have been requested)
    while( (s & READERS)==ONE_READER || !(s & WRITER_PENDING) ) {
        if( CAS(mutex->state, s | WRITER_PENDING, s) )
        {
            ITT_NOTIFY(sync_prepare, mutex);
            atomic_backoff backoff;
            while( (mutex->state & READERS) != ONE_READER ) backoff.pause();
            __TBB_ASSERT(mutex->state == (ONE_READER | WRITER_PENDING),"invalid state when upgrading to writer");
            // both new readers and writers are blocked at this time
            mutex->state = WRITER;
            ITT_NOTIFY(sync_acquired, mutex);
            __TBB_ASSERT( (mutex->state & BUSY) == WRITER, "invalid state after upgrade" );
            return true; // successfully upgraded
        } else {
            s = mutex->state; // re-read
        }
    }
    // slow reacquire
    internal_release_reader(mutex);
    return internal_acquire_writer(mutex); // always returns false
}

//! Downgrade writer to a reader
void spin_rw_mutex::internal_downgrade(spin_rw_mutex *mutex) {
    __TBB_ASSERT( (mutex->state & BUSY) == WRITER, "invalid state before downgrade" );
    ITT_NOTIFY(sync_releasing, mutex);
    mutex->state = ONE_READER;
    __TBB_ASSERT( mutex->state & READERS, "invalid state after downgrade: no readers" );
    __TBB_ASSERT( !(mutex->state & WRITER), "invalid state after downgrade: active writer" );
}

//! Release read lock on the given mutex
void spin_rw_mutex::internal_release_reader(spin_rw_mutex *mutex)
{
    __TBB_ASSERT( mutex->state & READERS, "invalid state of a read lock: no readers" );
    __TBB_ASSERT( !(mutex->state & WRITER), "invalid state of a read lock: active writer" );
    ITT_NOTIFY(sync_releasing, mutex); // release reader
    __TBB_FetchAndAddWrelease((volatile void *)&(mutex->state),-(intptr_t)ONE_READER);
}

//! Try to acquire write lock on the given mutex
bool spin_rw_mutex::internal_try_acquire_writer( spin_rw_mutex * mutex )
{
    // for a writer: only possible to acquire if no active readers or writers
    state_t s = mutex->state; // on IA-64 architecture, this volatile load has acquire semantic
    if( !(s & BUSY) ) // no readers, no writers; mask is 1..1101
        if( CAS(mutex->state, WRITER, s) ) {
            ITT_NOTIFY(sync_acquired, mutex);
            return true; // successfully stored writer flag
        }
    return false;
}

//! Try to acquire read lock on the given mutex
bool spin_rw_mutex::internal_try_acquire_reader( spin_rw_mutex * mutex )
{
    // for a reader: acquire if no active or waiting writers
    state_t s = mutex->state;    // on IA-64 architecture, a load of volatile variable has acquire semantic
    while( !(s & (WRITER|WRITER_PENDING)) ) // no writers
        if( CAS(mutex->state, s+ONE_READER, s) ) {
            ITT_NOTIFY(sync_acquired, mutex);
            return true; // successfully stored increased number of readers
        }
    return false;
}

} // namespace tbb
