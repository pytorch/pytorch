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

#if _WIN32||_WIN64
#include <errno.h> // EDEADLK
#endif
#include "tbb/mutex.h"
#include "itt_notify.h"
#if __TBB_TSX_AVAILABLE
#include "governor.h" // for speculation_enabled()
#endif

namespace tbb {
    void mutex::scoped_lock::internal_acquire( mutex& m ) {

#if _WIN32||_WIN64
        switch( m.state ) {
        case INITIALIZED:
        case HELD:
            EnterCriticalSection( &m.impl );
            // If a thread comes here, and another thread holds the lock, it will block
            // in EnterCriticalSection.  When it returns from EnterCriticalSection,
            // m.state must be set to INITIALIZED.  If the same thread tries to acquire a lock it
            // already holds, the lock is in HELD state, thus will cause throwing the exception.
            if (m.state==HELD)
                tbb::internal::handle_perror(EDEADLK,"mutex::scoped_lock: deadlock caused by attempt to reacquire held mutex");
            m.state = HELD;
            break;
        case DESTROYED:
            __TBB_ASSERT(false,"mutex::scoped_lock: mutex already destroyed");
            break;
        default:
            __TBB_ASSERT(false,"mutex::scoped_lock: illegal mutex state");
            break;
        }
#else
        int error_code = pthread_mutex_lock(&m.impl);
        if( error_code )
            tbb::internal::handle_perror(error_code,"mutex::scoped_lock: pthread_mutex_lock failed");
#endif /* _WIN32||_WIN64 */
        my_mutex = &m;
    }

void mutex::scoped_lock::internal_release() {
    __TBB_ASSERT( my_mutex, "mutex::scoped_lock: not holding a mutex" );
#if _WIN32||_WIN64
     switch( my_mutex->state ) {
        case INITIALIZED:
            __TBB_ASSERT(false,"mutex::scoped_lock: try to release the lock without acquisition");
            break;
        case HELD:
            my_mutex->state = INITIALIZED;
            LeaveCriticalSection(&my_mutex->impl);
            break;
        case DESTROYED:
            __TBB_ASSERT(false,"mutex::scoped_lock: mutex already destroyed");
            break;
        default:
            __TBB_ASSERT(false,"mutex::scoped_lock: illegal mutex state");
            break;
    }
#else
     int error_code = pthread_mutex_unlock(&my_mutex->impl);
     __TBB_ASSERT_EX(!error_code, "mutex::scoped_lock: pthread_mutex_unlock failed");
#endif /* _WIN32||_WIN64 */
     my_mutex = NULL;
}

bool mutex::scoped_lock::internal_try_acquire( mutex& m ) {
#if _WIN32||_WIN64
    switch( m.state ) {
        case INITIALIZED:
        case HELD:
            break;
        case DESTROYED:
            __TBB_ASSERT(false,"mutex::scoped_lock: mutex already destroyed");
            break;
        default:
            __TBB_ASSERT(false,"mutex::scoped_lock: illegal mutex state");
            break;
    }
#endif /* _WIN32||_WIN64 */

    bool result;
#if _WIN32||_WIN64
    result = TryEnterCriticalSection(&m.impl)!=0;
    if( result ) {
        __TBB_ASSERT(m.state!=HELD, "mutex::scoped_lock: deadlock caused by attempt to reacquire held mutex");
        m.state = HELD;
    }
#else
    result = pthread_mutex_trylock(&m.impl)==0;
#endif /* _WIN32||_WIN64 */
    if( result )
        my_mutex = &m;
    return result;
}

void mutex::internal_construct() {
#if _WIN32||_WIN64
    InitializeCriticalSectionEx(&impl, 4000, 0);
    state = INITIALIZED;
#else
    int error_code = pthread_mutex_init(&impl,NULL);
    if( error_code )
        tbb::internal::handle_perror(error_code,"mutex: pthread_mutex_init failed");
#endif /* _WIN32||_WIN64*/
    ITT_SYNC_CREATE(&impl, _T("tbb::mutex"), _T(""));
}

void mutex::internal_destroy() {
#if _WIN32||_WIN64
    switch( state ) {
      case INITIALIZED:
        DeleteCriticalSection(&impl);
       break;
      case DESTROYED:
        __TBB_ASSERT(false,"mutex: already destroyed");
        break;
      default:
        __TBB_ASSERT(false,"mutex: illegal state for destruction");
        break;
    }
    state = DESTROYED;
#else
    int error_code = pthread_mutex_destroy(&impl);
#if __TBB_TSX_AVAILABLE
    // For processors with speculative execution, skip the error code check due to glibc bug #16657
    if( tbb::internal::governor::speculation_enabled() ) return;
#endif
    __TBB_ASSERT_EX(!error_code,"mutex: pthread_mutex_destroy failed");
#endif /* _WIN32||_WIN64 */
}

} // namespace tbb
