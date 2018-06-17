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

#include "tbb/tbb_config.h"
#include "tbb/compat/condition_variable"
#include "tbb/atomic.h"
#include "tbb_misc.h"
#include "dynamic_link.h"
#include "itt_notify.h"

namespace tbb {

namespace internal {

//condition_variable
#if _WIN32||_WIN64
using tbb::interface5::internal::condition_variable_using_event;

static atomic<do_once_state> condvar_api_state;

void WINAPI init_condvar_using_event( condition_variable_using_event* cv_event )
{
    // TODO: For Metro port, we can always use the API for condition variables, without dynamic_link etc.
    cv_event->event = CreateEventEx(NULL, NULL, 0x1 /*CREATE_EVENT_MANUAL_RESET*/, EVENT_ALL_ACCESS );
    InitializeCriticalSectionEx( &cv_event->mutex, 4000, 0 );
    cv_event->n_waiters = 0;
    cv_event->release_count = 0;
    cv_event->epoch = 0;
}

BOOL WINAPI sleep_condition_variable_cs_using_event( condition_variable_using_event* cv_event, LPCRITICAL_SECTION cs, DWORD dwMilliseconds )
{
    EnterCriticalSection( &cv_event->mutex );
    ++cv_event->n_waiters;
    unsigned my_generation = cv_event->epoch;
    LeaveCriticalSection( &cv_event->mutex );
    LeaveCriticalSection( cs );
    for (;;) {
        // should come here at least once
        DWORD rc = WaitForSingleObjectEx( cv_event->event, dwMilliseconds, FALSE );
        EnterCriticalSection( &cv_event->mutex );
        if( rc!=WAIT_OBJECT_0 ) {
            --cv_event->n_waiters;
            LeaveCriticalSection( &cv_event->mutex );
            if( rc==WAIT_TIMEOUT ) {
                SetLastError( WAIT_TIMEOUT );
                EnterCriticalSection( cs );
            }
            return false;
        }
        __TBB_ASSERT( rc==WAIT_OBJECT_0, NULL );
        if( cv_event->release_count>0 && cv_event->epoch!=my_generation )
            break;
        LeaveCriticalSection( &cv_event->mutex );
    }

    // still in the critical section
    --cv_event->n_waiters;
    int count = --cv_event->release_count;
    LeaveCriticalSection( &cv_event->mutex );

    if( count==0 ) {
        __TBB_ASSERT( cv_event->event, "Premature destruction of condition variable?" );
        ResetEvent( cv_event->event );
    }
    EnterCriticalSection( cs );
    return true;
}

void WINAPI wake_condition_variable_using_event( condition_variable_using_event* cv_event )
{
    EnterCriticalSection( &cv_event->mutex );
    if( cv_event->n_waiters>cv_event->release_count ) {
        SetEvent( cv_event->event ); // Signal the manual-reset event.
        ++cv_event->release_count;
        ++cv_event->epoch;
    }
    LeaveCriticalSection( &cv_event->mutex );
}

void WINAPI wake_all_condition_variable_using_event( condition_variable_using_event* cv_event )
{
    EnterCriticalSection( &cv_event->mutex );
    if( cv_event->n_waiters>0 ) {
        SetEvent( cv_event->event );
        cv_event->release_count = cv_event->n_waiters;
        ++cv_event->epoch;
    }
    LeaveCriticalSection( &cv_event->mutex );
}

void WINAPI destroy_condvar_using_event( condition_variable_using_event* cv_event )
{
    HANDLE my_event = cv_event->event;
    EnterCriticalSection( &cv_event->mutex );
    // NULL is an invalid HANDLE value
    cv_event->event = NULL;
    if( cv_event->n_waiters>0 ) {
        LeaveCriticalSection( &cv_event->mutex );
        spin_wait_until_eq( cv_event->n_waiters, 0 );
        // make sure the last thread completes its access to cv
        EnterCriticalSection( &cv_event->mutex );
    }
    LeaveCriticalSection( &cv_event->mutex );
    CloseHandle( my_event );
}

void WINAPI destroy_condvar_noop( CONDITION_VARIABLE* /*cv*/ ) { /*no op*/ }

static void (WINAPI *__TBB_init_condvar)( PCONDITION_VARIABLE ) = (void (WINAPI *)(PCONDITION_VARIABLE))&init_condvar_using_event;
static BOOL (WINAPI *__TBB_condvar_wait)( PCONDITION_VARIABLE, LPCRITICAL_SECTION, DWORD ) = (BOOL (WINAPI *)(PCONDITION_VARIABLE,LPCRITICAL_SECTION, DWORD))&sleep_condition_variable_cs_using_event;
static void (WINAPI *__TBB_condvar_notify_one)( PCONDITION_VARIABLE ) = (void (WINAPI *)(PCONDITION_VARIABLE))&wake_condition_variable_using_event;
static void (WINAPI *__TBB_condvar_notify_all)( PCONDITION_VARIABLE ) = (void (WINAPI *)(PCONDITION_VARIABLE))&wake_all_condition_variable_using_event;
static void (WINAPI *__TBB_destroy_condvar)( PCONDITION_VARIABLE ) = (void (WINAPI *)(PCONDITION_VARIABLE))&destroy_condvar_using_event;

//! Table describing how to link the handlers.
static const dynamic_link_descriptor CondVarLinkTable[] = {
    DLD(InitializeConditionVariable, __TBB_init_condvar),
    DLD(SleepConditionVariableCS,    __TBB_condvar_wait),
    DLD(WakeConditionVariable,       __TBB_condvar_notify_one),
    DLD(WakeAllConditionVariable,    __TBB_condvar_notify_all)
};

void init_condvar_module()
{
    __TBB_ASSERT( (uintptr_t)__TBB_init_condvar==(uintptr_t)&init_condvar_using_event, NULL );
#if __TBB_WIN8UI_SUPPORT
    // We expect condition variables to be always available for Windows* store applications,
    // so there is no need to check presense and use alternative implementation.
    __TBB_init_condvar = (void (WINAPI *)(PCONDITION_VARIABLE))&InitializeConditionVariable;
    __TBB_condvar_wait = (BOOL(WINAPI *)(PCONDITION_VARIABLE, LPCRITICAL_SECTION, DWORD))&SleepConditionVariableCS;
    __TBB_condvar_notify_one = (void (WINAPI *)(PCONDITION_VARIABLE))&WakeConditionVariable;
    __TBB_condvar_notify_all = (void (WINAPI *)(PCONDITION_VARIABLE))&WakeAllConditionVariable;
    __TBB_destroy_condvar = (void (WINAPI *)(PCONDITION_VARIABLE))&destroy_condvar_noop;
#else
    if (dynamic_link("Kernel32.dll", CondVarLinkTable, 4))
        __TBB_destroy_condvar = (void (WINAPI *)(PCONDITION_VARIABLE))&destroy_condvar_noop;
#endif
}
#endif /* _WIN32||_WIN64 */

} // namespace internal

#if _WIN32||_WIN64

namespace interface5 {
namespace internal {

using tbb::internal::condvar_api_state;
using tbb::internal::__TBB_init_condvar;
using tbb::internal::__TBB_condvar_wait;
using tbb::internal::__TBB_condvar_notify_one;
using tbb::internal::__TBB_condvar_notify_all;
using tbb::internal::__TBB_destroy_condvar;
using tbb::internal::init_condvar_module;

void internal_initialize_condition_variable( condvar_impl_t& cv )
{
    atomic_do_once( &init_condvar_module, condvar_api_state );
    __TBB_init_condvar( &cv.cv_native );
}

void internal_destroy_condition_variable( condvar_impl_t& cv )
{
    __TBB_destroy_condvar( &cv.cv_native );
}

void internal_condition_variable_notify_one( condvar_impl_t& cv )
{
    __TBB_condvar_notify_one ( &cv.cv_native );
}

void internal_condition_variable_notify_all( condvar_impl_t& cv )
{
    __TBB_condvar_notify_all( &cv.cv_native );
}

bool internal_condition_variable_wait( condvar_impl_t& cv, mutex* mtx, const tick_count::interval_t* i )
{
    DWORD duration = i ? DWORD((i->seconds()*1000)) : INFINITE;
    mtx->set_state( mutex::INITIALIZED );
    BOOL res = __TBB_condvar_wait( &cv.cv_native, mtx->native_handle(), duration );
    mtx->set_state( mutex::HELD );
    return res?true:false;
}

} // namespace internal
} // nameespace interface5

#endif /* _WIN32||_WIN64 */

} // namespace tbb
