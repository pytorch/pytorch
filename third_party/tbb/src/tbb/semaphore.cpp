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

#include "semaphore.h"
#if __TBB_USE_SRWLOCK
#include "dynamic_link.h" // Refers to src/tbb, not include/tbb
#include "tbb_misc.h"
#endif

namespace tbb {
namespace internal {

// TODO: For new win UI port, we can use SRWLock API without dynamic_link etc.
#if __TBB_USE_SRWLOCK

static atomic<do_once_state> concmon_module_inited;

void WINAPI init_binsem_using_event( SRWLOCK* h_ )
{
    srwl_or_handle* shptr = (srwl_or_handle*) h_;
    shptr->h = CreateEventEx( NULL, NULL, 0, EVENT_ALL_ACCESS|SEMAPHORE_ALL_ACCESS );
}

void WINAPI acquire_binsem_using_event( SRWLOCK* h_ )
{
    srwl_or_handle* shptr = (srwl_or_handle*) h_;
    WaitForSingleObjectEx( shptr->h, INFINITE, FALSE );
}

void WINAPI release_binsem_using_event( SRWLOCK* h_ )
{
    srwl_or_handle* shptr = (srwl_or_handle*) h_;
    SetEvent( shptr->h );
}

static void (WINAPI *__TBB_init_binsem)( SRWLOCK* ) = (void (WINAPI *)(SRWLOCK*))&init_binsem_using_event;
static void (WINAPI *__TBB_acquire_binsem)( SRWLOCK* ) = (void (WINAPI *)(SRWLOCK*))&acquire_binsem_using_event;
static void (WINAPI *__TBB_release_binsem)( SRWLOCK* ) = (void (WINAPI *)(SRWLOCK*))&release_binsem_using_event;

//! Table describing the how to link the handlers.
static const dynamic_link_descriptor SRWLLinkTable[] = {
    DLD(InitializeSRWLock,       __TBB_init_binsem),
    DLD(AcquireSRWLockExclusive, __TBB_acquire_binsem),
    DLD(ReleaseSRWLockExclusive, __TBB_release_binsem)
};

inline void init_concmon_module()
{
    __TBB_ASSERT( (uintptr_t)__TBB_init_binsem==(uintptr_t)&init_binsem_using_event, NULL );
    if( dynamic_link( "Kernel32.dll", SRWLLinkTable, sizeof(SRWLLinkTable)/sizeof(dynamic_link_descriptor) ) ) {
        __TBB_ASSERT( (uintptr_t)__TBB_init_binsem!=(uintptr_t)&init_binsem_using_event, NULL );
        __TBB_ASSERT( (uintptr_t)__TBB_acquire_binsem!=(uintptr_t)&acquire_binsem_using_event, NULL );
        __TBB_ASSERT( (uintptr_t)__TBB_release_binsem!=(uintptr_t)&release_binsem_using_event, NULL );
    }
}

binary_semaphore::binary_semaphore() {
    atomic_do_once( &init_concmon_module, concmon_module_inited );

    __TBB_init_binsem( &my_sem.lock );
    if( (uintptr_t)__TBB_init_binsem!=(uintptr_t)&init_binsem_using_event )
        P();
}

binary_semaphore::~binary_semaphore() {
    if( (uintptr_t)__TBB_init_binsem==(uintptr_t)&init_binsem_using_event )
        CloseHandle( my_sem.h );
}

void binary_semaphore::P() { __TBB_acquire_binsem( &my_sem.lock ); }

void binary_semaphore::V() { __TBB_release_binsem( &my_sem.lock ); }

#endif /* __TBB_USE_SRWLOCK */

} // namespace internal
} // namespace tbb
