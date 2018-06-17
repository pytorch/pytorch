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

// All platform-specific threading support is encapsulated here. */

#ifndef __RML_thread_monitor_H
#define __RML_thread_monitor_H

#if USE_WINTHREAD
#include <windows.h>
#include <process.h>
#include <malloc.h> //_alloca
#include "tbb/tbb_misc.h" // support for processor groups
#if __TBB_WIN8UI_SUPPORT
#include <thread>
#endif
#elif USE_PTHREAD
#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#else
#error Unsupported platform
#endif
#include <stdio.h>
#include "tbb/itt_notify.h"
#include "tbb/atomic.h"
#include "tbb/semaphore.h"

// All platform-specific threading support is in this header.

#if (_WIN32||_WIN64)&&!__TBB_ipf
// Deal with 64K aliasing.  The formula for "offset" is a Fibonacci hash function,
// which has the desirable feature of spreading out the offsets fairly evenly
// without knowing the total number of offsets, and furthermore unlikely to
// accidentally cancel out other 64K aliasing schemes that Microsoft might implement later.
// See Knuth Vol 3. "Theorem S" for details on Fibonacci hashing.
// The second statement is really does need "volatile", otherwise the compiler might remove the _alloca.
#define AVOID_64K_ALIASING(idx)                       \
    size_t offset = (idx+1) * 40503U % (1U<<16);      \
    void* volatile sink_for_alloca = _alloca(offset); \
    __TBB_ASSERT_EX(sink_for_alloca, "_alloca failed");
#else
// Linux thread allocators avoid 64K aliasing.
#define AVOID_64K_ALIASING(idx) tbb::internal::suppress_unused_warning(idx)
#endif /* _WIN32||_WIN64 */

namespace rml {

namespace internal {

#if DO_ITT_NOTIFY
static const ::tbb::tchar *SyncType_RML = _T("%Constant");
static const ::tbb::tchar *SyncObj_ThreadMonitor = _T("RML Thr Monitor");
#endif /* DO_ITT_NOTIFY */

//! Monitor with limited two-phase commit form of wait.
/** At most one thread should wait on an instance at a time. */
class thread_monitor {
public:
    class cookie {
        friend class thread_monitor;
        tbb::atomic<size_t> my_epoch;
    };
    thread_monitor() : spurious(false), my_sema() {
        my_cookie.my_epoch = 0;
        ITT_SYNC_CREATE(&my_sema, SyncType_RML, SyncObj_ThreadMonitor);
        in_wait = false;
    }
    ~thread_monitor() {}

    //! If a thread is waiting or started a two-phase wait, notify it.
    /** Can be called by any thread. */
    void notify();

    //! Begin two-phase wait.
    /** Should only be called by thread that owns the monitor.
        The caller must either complete the wait or cancel it. */
    void prepare_wait( cookie& c );

    //! Complete a two-phase wait and wait until notification occurs after the earlier prepare_wait.
    void commit_wait( cookie& c );

    //! Cancel a two-phase wait.
    void cancel_wait();

#if USE_WINTHREAD
    typedef HANDLE handle_type;

    #define __RML_DECL_THREAD_ROUTINE unsigned WINAPI
    typedef unsigned (WINAPI *thread_routine_type)(void*);

    //! Launch a thread
    static handle_type launch( thread_routine_type thread_routine, void* arg, size_t stack_size, const size_t* worker_index = NULL );

#elif USE_PTHREAD
    typedef pthread_t handle_type;

    #define __RML_DECL_THREAD_ROUTINE void*
    typedef void*(*thread_routine_type)(void*);

    //! Launch a thread
    static handle_type launch( thread_routine_type thread_routine, void* arg, size_t stack_size );
#endif /* USE_PTHREAD */

    //! Yield control to OS
    /** Affects the calling thread. **/
    static void yield();

    //! Join thread
    static void join(handle_type handle);

    //! Detach thread
    static void detach_thread(handle_type handle);
private:
    cookie my_cookie;
    tbb::atomic<bool>   in_wait;
    bool   spurious;
    tbb::internal::binary_semaphore my_sema;
#if USE_PTHREAD
    static void check( int error_code, const char* routine );
#endif
};

#if USE_WINTHREAD

#ifndef STACK_SIZE_PARAM_IS_A_RESERVATION
#define STACK_SIZE_PARAM_IS_A_RESERVATION 0x00010000
#endif

#if __TBB_WIN8UI_SUPPORT
inline thread_monitor::handle_type thread_monitor::launch( thread_routine_type thread_function, void* arg, size_t, const size_t*) {
//TODO: check that exception thrown from std::thread is not swallowed silently
    std::thread* thread_tmp=new std::thread(thread_function, arg);
    return thread_tmp->native_handle();
}
#else //__TBB_WIN8UI_SUPPORT
inline thread_monitor::handle_type thread_monitor::launch( thread_routine_type thread_routine, void* arg, size_t stack_size, const size_t* worker_index ) {
    unsigned thread_id;
    int number_of_processor_groups = ( worker_index ) ? tbb::internal::NumberOfProcessorGroups() : 0;
    unsigned create_flags = ( number_of_processor_groups > 1 ) ? CREATE_SUSPENDED : 0;
    HANDLE h = (HANDLE)_beginthreadex( NULL, unsigned(stack_size), thread_routine, arg, STACK_SIZE_PARAM_IS_A_RESERVATION | create_flags, &thread_id );
    if( !h ) {
        fprintf(stderr,"thread_monitor::launch: _beginthreadex failed\n");
        exit(1);
    }
    if ( number_of_processor_groups > 1 ) {
        tbb::internal::MoveThreadIntoProcessorGroup( h,
                        tbb::internal::FindProcessorGroupIndex( static_cast<int>(*worker_index) ) );
        ResumeThread( h );
    }
    return h;
}
#endif //__TBB_WIN8UI_SUPPORT

void thread_monitor::join(handle_type handle) {
#if TBB_USE_ASSERT
    DWORD res =
#endif
        WaitForSingleObjectEx(handle, INFINITE, FALSE);
    __TBB_ASSERT( res==WAIT_OBJECT_0, NULL );
#if TBB_USE_ASSERT
    BOOL val =
#endif
        CloseHandle(handle);
    __TBB_ASSERT( val, NULL );
}

void thread_monitor::detach_thread(handle_type handle) {
#if TBB_USE_ASSERT
    BOOL val =
#endif
        CloseHandle(handle);
    __TBB_ASSERT( val, NULL );
}

inline void thread_monitor::yield() {
// TODO: consider unification via __TBB_Yield or tbb::this_tbb_thread::yield
#if !__TBB_WIN8UI_SUPPORT
    SwitchToThread();
#else
    std::this_thread::yield();
#endif
}
#endif /* USE_WINTHREAD */

#if USE_PTHREAD
// TODO: can we throw exceptions instead of termination?
inline void thread_monitor::check( int error_code, const char* routine ) {
    if( error_code ) {
        fprintf(stderr,"thread_monitor %s in %s\n", strerror(error_code), routine );
        exit(1);
    }
}

inline thread_monitor::handle_type thread_monitor::launch( void* (*thread_routine)(void*), void* arg, size_t stack_size ) {
    // FIXME - consider more graceful recovery than just exiting if a thread cannot be launched.
    // Note that there are some tricky situations to deal with, such that the thread is already
    // grabbed as part of an OpenMP team.
    pthread_attr_t s;
    check(pthread_attr_init( &s ), "pthread_attr_init");
    if( stack_size>0 )
        check(pthread_attr_setstacksize( &s, stack_size ), "pthread_attr_setstack_size" );
    pthread_t handle;
    check( pthread_create( &handle, &s, thread_routine, arg ), "pthread_create" );
    check( pthread_attr_destroy( &s ), "pthread_attr_destroy" );
    return handle;
}

void thread_monitor::join(handle_type handle) {
    check(pthread_join(handle, NULL), "pthread_join");
}

void thread_monitor::detach_thread(handle_type handle) {
    check(pthread_detach(handle), "pthread_detach");
}

inline void thread_monitor::yield() {
    sched_yield();
}
#endif /* USE_PTHREAD */

inline void thread_monitor::notify() {
    my_cookie.my_epoch = my_cookie.my_epoch + 1;
    bool do_signal = in_wait.fetch_and_store( false );
    if( do_signal )
        my_sema.V();
}

inline void thread_monitor::prepare_wait( cookie& c ) {
    if( spurious ) {
        spurious = false;
        //  consumes a spurious posted signal. don't wait on my_sema.
        my_sema.P();
    }
    c = my_cookie;
    in_wait = true;
   __TBB_full_memory_fence();
}

inline void thread_monitor::commit_wait( cookie& c ) {
    bool do_it = ( c.my_epoch == my_cookie.my_epoch);
    if( do_it ) my_sema.P();
    else        cancel_wait();
}

inline void thread_monitor::cancel_wait() {
    spurious = ! in_wait.fetch_and_store( false );
}

} // namespace internal
} // namespace rml

#endif /* __RML_thread_monitor_H */
