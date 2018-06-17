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
#include <process.h>        // _beginthreadex()
#endif
#include <errno.h>
#include "tbb_misc.h"       // handle_win_error()
#include "tbb/tbb_stddef.h"
#include "tbb/tbb_thread.h"
#include "tbb/tbb_allocator.h"
#include "tbb/global_control.h" // thread_stack_size
#include "governor.h"       // default_num_threads()
#if __TBB_WIN8UI_SUPPORT
#include <thread>
#endif

namespace tbb {
namespace internal {

//! Allocate a closure
void* allocate_closure_v3( size_t size )
{
    return allocate_via_handler_v3( size );
}

//! Free a closure allocated by allocate_closure_v3
void free_closure_v3( void *ptr )
{
    deallocate_via_handler_v3( ptr );
}

void tbb_thread_v3::join()
{
    if (!joinable())
        handle_perror( EINVAL, "tbb_thread::join" ); // Invalid argument
    if (this_tbb_thread::get_id() == get_id())
        handle_perror( EDEADLK, "tbb_thread::join" ); // Resource deadlock avoided
#if _WIN32||_WIN64
#if __TBB_WIN8UI_SUPPORT
    std::thread* thread_tmp=(std::thread*)my_thread_id;
    thread_tmp->join();
    delete thread_tmp;
#else // __TBB_WIN8UI_SUPPORT
    DWORD status = WaitForSingleObjectEx( my_handle, INFINITE, FALSE );
    if ( status == WAIT_FAILED )
        handle_win_error( GetLastError() );
    BOOL close_stat = CloseHandle( my_handle );
    if ( close_stat == 0 )
        handle_win_error( GetLastError() );
    my_thread_id = 0;
#endif // __TBB_WIN8UI_SUPPORT
#else
    int status = pthread_join( my_handle, NULL );
    if( status )
        handle_perror( status, "pthread_join" );
#endif // _WIN32||_WIN64
    my_handle = 0;
}

void tbb_thread_v3::detach() {
    if (!joinable())
        handle_perror( EINVAL, "tbb_thread::detach" ); // Invalid argument
#if _WIN32||_WIN64
    BOOL status = CloseHandle( my_handle );
    if ( status == 0 )
      handle_win_error( GetLastError() );
    my_thread_id = 0;
#else
    int status = pthread_detach( my_handle );
    if( status )
        handle_perror( status, "pthread_detach" );
#endif // _WIN32||_WIN64
    my_handle = 0;
}

void tbb_thread_v3::internal_start( __TBB_NATIVE_THREAD_ROUTINE_PTR(start_routine),
                                    void* closure ) {
#if _WIN32||_WIN64
#if __TBB_WIN8UI_SUPPORT
    std::thread* thread_tmp=new std::thread(start_routine, closure);
    my_handle  = thread_tmp->native_handle();
//  TODO: to find out the way to find thread_id without GetThreadId and other
//  desktop functions.
//  Now tbb_thread does have its own thread_id that stores std::thread object
    my_thread_id = (size_t)thread_tmp;
#else
    unsigned thread_id;
    // The return type of _beginthreadex is "uintptr_t" on new MS compilers,
    // and 'unsigned long' on old MS compilers.  uintptr_t works for both.
    uintptr_t status = _beginthreadex( NULL, (unsigned)global_control::active_value(global_control::thread_stack_size),
                                       start_routine, closure, 0, &thread_id );
    if( status==0 )
        handle_perror(errno,"__beginthreadex");
    else {
        my_handle = (HANDLE)status;
        my_thread_id = thread_id;
    }
#endif
#else
    pthread_t thread_handle;
    int status;
    pthread_attr_t stack_size;
    status = pthread_attr_init( &stack_size );
    if( status )
        handle_perror( status, "pthread_attr_init" );
    status = pthread_attr_setstacksize( &stack_size, global_control::active_value(global_control::thread_stack_size) );
    if( status )
        handle_perror( status, "pthread_attr_setstacksize" );

    status = pthread_create( &thread_handle, &stack_size, start_routine, closure );
    if( status )
        handle_perror( status, "pthread_create" );
    status = pthread_attr_destroy( &stack_size );
    if( status )
        handle_perror( status, "pthread_attr_destroy" );

    my_handle = thread_handle;
#endif // _WIN32||_WIN64
}

unsigned tbb_thread_v3::hardware_concurrency() __TBB_NOEXCEPT(true) {
    return governor::default_num_threads();
}

tbb_thread_v3::id thread_get_id_v3() {
#if _WIN32||_WIN64
    return tbb_thread_v3::id( GetCurrentThreadId() );
#else
    return tbb_thread_v3::id( pthread_self() );
#endif // _WIN32||_WIN64
}

void move_v3( tbb_thread_v3& t1, tbb_thread_v3& t2 )
{
    if (t1.joinable())
        t1.detach();
    t1.my_handle = t2.my_handle;
    t2.my_handle = 0;
#if _WIN32||_WIN64
    t1.my_thread_id = t2.my_thread_id;
    t2.my_thread_id = 0;
#endif // _WIN32||_WIN64
}

void thread_yield_v3()
{
    __TBB_Yield();
}

void thread_sleep_v3(const tick_count::interval_t &i)
{
#if _WIN32||_WIN64
     tick_count t0 = tick_count::now();
     tick_count t1 = t0;
     for(;;) {
         double remainder = (i-(t1-t0)).seconds()*1e3;  // milliseconds remaining to sleep
         if( remainder<=0 ) break;
         DWORD t = remainder>=INFINITE ? INFINITE-1 : DWORD(remainder);
#if !__TBB_WIN8UI_SUPPORT
         Sleep( t );
#else
         std::chrono::milliseconds sleep_time( t );
         std::this_thread::sleep_for( sleep_time );
#endif
         t1 = tick_count::now();
    }
#else
    struct timespec req;
    double sec = i.seconds();

    req.tv_sec = static_cast<long>(sec);
    req.tv_nsec = static_cast<long>( (sec - req.tv_sec)*1e9 );
    nanosleep(&req, NULL);
#endif // _WIN32||_WIN64
}

} // internal
} // tbb
