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

#ifndef __TBB_spin_rw_mutex_H
#define __TBB_spin_rw_mutex_H

#include "tbb/tbb_stddef.h"

namespace tbb {

//! Fast, unfair, spinning reader-writer lock with backoff and writer-preference
/** @ingroup synchronization */
class spin_rw_mutex {
    //! @cond INTERNAL

    //! Present so that 1.0 headers work with 1.1 dynamic library.
    static void __TBB_EXPORTED_FUNC internal_itt_releasing(spin_rw_mutex *);

    //! Internal acquire write lock.
    static bool __TBB_EXPORTED_FUNC internal_acquire_writer(spin_rw_mutex *);

    //! Out of line code for releasing a write lock.
    /** This code has debug checking and instrumentation for Intel(R) Thread Checker and Intel(R) Thread Profiler. */
    static void __TBB_EXPORTED_FUNC internal_release_writer(spin_rw_mutex *);

    //! Internal acquire read lock.
    static void __TBB_EXPORTED_FUNC internal_acquire_reader(spin_rw_mutex *);

    //! Internal upgrade reader to become a writer.
    static bool __TBB_EXPORTED_FUNC internal_upgrade(spin_rw_mutex *);

    //! Out of line code for downgrading a writer to a reader.
    /** This code has debug checking and instrumentation for Intel(R) Thread Checker and Intel(R) Thread Profiler. */
    static void __TBB_EXPORTED_FUNC internal_downgrade(spin_rw_mutex *);

    //! Internal release read lock.
    static void __TBB_EXPORTED_FUNC internal_release_reader(spin_rw_mutex *);

    //! Internal try_acquire write lock.
    static bool __TBB_EXPORTED_FUNC internal_try_acquire_writer(spin_rw_mutex *);

    //! Internal try_acquire read lock.
    static bool __TBB_EXPORTED_FUNC internal_try_acquire_reader(spin_rw_mutex *);

    //! @endcond
public:
    //! Construct unacquired mutex.
    spin_rw_mutex() : state(0) {}

#if TBB_USE_ASSERT
    //! Destructor asserts if the mutex is acquired, i.e. state is zero.
    ~spin_rw_mutex() {
        __TBB_ASSERT( !state, "destruction of an acquired mutex");
    };
#endif /* TBB_USE_ASSERT */

    //! The scoped locking pattern
    /** It helps to avoid the common problem of forgetting to release lock.
        It also nicely provides the "node" for queuing locks. */
    class scoped_lock : internal::no_copy {
    public:
        //! Construct lock that has not acquired a mutex.
        /** Equivalent to zero-initialization of *this. */
        scoped_lock() : mutex(NULL) {}

        //! Construct and acquire lock on given mutex.
        scoped_lock( spin_rw_mutex& m, bool write = true ) : mutex(NULL) {
            acquire(m, write);
        }

        //! Release lock (if lock is held).
        ~scoped_lock() {
            if( mutex ) release();
        }

        //! Acquire lock on given mutex.
        void acquire( spin_rw_mutex& m, bool write = true ) {
            __TBB_ASSERT( !mutex, "holding mutex already" );
            mutex = &m;
            is_writer = write;
            if( write ) internal_acquire_writer(mutex);
            else        internal_acquire_reader(mutex);
        }

        //! Upgrade reader to become a writer.
        /** Returns whether the upgrade happened without releasing and re-acquiring the lock */
        bool upgrade_to_writer() {
            __TBB_ASSERT( mutex, "lock is not acquired" );
            __TBB_ASSERT( !is_writer, "not a reader" );
            is_writer = true;
            return internal_upgrade(mutex);
        }

        //! Release lock.
        void release() {
            __TBB_ASSERT( mutex, "lock is not acquired" );
            spin_rw_mutex *m = mutex; 
            mutex = NULL;
            if( is_writer ) {
#if TBB_USE_THREADING_TOOLS||TBB_USE_ASSERT
                internal_release_writer(m);
#else
                m->state = 0; 
#endif /* TBB_USE_THREADING_TOOLS||TBB_USE_ASSERT */
            } else {
                internal_release_reader(m);
            }
        };

        //! Downgrade writer to become a reader.
        bool downgrade_to_reader() {
            __TBB_ASSERT( mutex, "lock is not acquired" );
            __TBB_ASSERT( is_writer, "not a writer" );
#if TBB_USE_THREADING_TOOLS||TBB_USE_ASSERT
            internal_downgrade(mutex);
#else
            mutex->state = 4; // Bit 2 - reader, 00..00100
#endif
            is_writer = false;
            return true;
        }

        //! Try acquire lock on given mutex.
        bool try_acquire( spin_rw_mutex& m, bool write = true ) {
            __TBB_ASSERT( !mutex, "holding mutex already" );
            bool result;
            is_writer = write; 
            result = write? internal_try_acquire_writer(&m)
                          : internal_try_acquire_reader(&m);
            if( result ) mutex = &m;
            return result;
        }

    private:
        //! The pointer to the current mutex that is held, or NULL if no mutex is held.
        spin_rw_mutex* mutex;

        //! If mutex!=NULL, then is_writer is true if holding a writer lock, false if holding a reader lock.
        /** Not defined if not holding a lock. */
        bool is_writer;
    };

private:
    typedef uintptr_t state_t;
    static const state_t WRITER = 1;
    static const state_t WRITER_PENDING = 2;
    static const state_t READERS = ~(WRITER | WRITER_PENDING);
    static const state_t ONE_READER = 4;
    static const state_t BUSY = WRITER | READERS;
    /** Bit 0 = writer is holding lock
        Bit 1 = request by a writer to acquire lock (hint to readers to wait)
        Bit 2..N = number of readers holding lock */
    volatile state_t state;
};

} // namespace tbb

#endif /* __TBB_spin_rw_mutex_H */
