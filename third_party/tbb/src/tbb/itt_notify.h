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

#ifndef _TBB_ITT_NOTIFY
#define _TBB_ITT_NOTIFY

#include "tbb/tbb_stddef.h"

#if DO_ITT_NOTIFY

#if _WIN32||_WIN64
    #ifndef UNICODE
        #define UNICODE
    #endif
#endif /* WIN */

#ifndef INTEL_ITTNOTIFY_API_PRIVATE
#define INTEL_ITTNOTIFY_API_PRIVATE
#endif

#include "tools_api/ittnotify.h"
#include "tools_api/legacy/ittnotify.h"
extern "C" void __itt_fini_ittlib(void);

#if _WIN32||_WIN64
    #undef _T
    #undef __itt_event_create
    #define __itt_event_create __itt_event_createA
#endif /* WIN */


#endif /* DO_ITT_NOTIFY */

#if !ITT_CALLER_NULL
#define ITT_CALLER_NULL ((__itt_caller)0)
#endif

namespace tbb {
//! Unicode support
#if (_WIN32||_WIN64) && !__MINGW32__
    //! Unicode character type. Always wchar_t on Windows.
    /** We do not use typedefs from Windows TCHAR family to keep consistence of TBB coding style. **/
    typedef wchar_t tchar;
    //! Standard Windows macro to markup the string literals.
    #define _T(string_literal) L ## string_literal
#else /* !WIN */
    typedef char tchar;
    //! Standard Windows style macro to markup the string literals.
    #define _T(string_literal) string_literal
#endif /* !WIN */
} // namespace tbb

#if DO_ITT_NOTIFY
namespace tbb {
    //! Display names of internal synchronization types
    extern const tchar
            *SyncType_GlobalLock,
            *SyncType_Scheduler;
    //! Display names of internal synchronization components/scenarios
    extern const tchar
            *SyncObj_SchedulerInitialization,
            *SyncObj_SchedulersList,
            *SyncObj_WorkerLifeCycleMgmt,
            *SyncObj_TaskStealingLoop,
            *SyncObj_WorkerTaskPool,
            *SyncObj_MasterTaskPool,
            *SyncObj_TaskPoolSpinning,
            *SyncObj_Mailbox,
            *SyncObj_TaskReturnList,
            *SyncObj_TaskStream,
            *SyncObj_ContextsList
            ;

    namespace internal {
        void __TBB_EXPORTED_FUNC itt_set_sync_name_v3( void* obj, const tchar* name);

    } // namespace internal

} // namespace tbb

// const_cast<void*>() is necessary to cast off volatility
#define ITT_NOTIFY(name,obj)            __itt_notify_##name(const_cast<void*>(static_cast<volatile void*>(obj)))
#define ITT_THREAD_SET_NAME(name)       __itt_thread_set_name(name)
#define ITT_FINI_ITTLIB()               __itt_fini_ittlib()
#define ITT_SYNC_CREATE(obj, type, name) __itt_sync_create((void*)(obj), type, name, 2)
#define ITT_SYNC_RENAME(obj, name)      __itt_sync_rename(obj, name)
#define ITT_STACK_CREATE(obj)           obj = __itt_stack_caller_create()
#if __TBB_TASK_GROUP_CONTEXT
#define ITT_STACK(precond, name, obj)   (precond) ? __itt_stack_##name(obj) : ((void)0);
#else
#define ITT_STACK(precond, name, obj)      ((void)0)
#endif /* !__TBB_TASK_GROUP_CONTEXT */

#else /* !DO_ITT_NOTIFY */

#define ITT_NOTIFY(name,obj)            ((void)0)
#define ITT_THREAD_SET_NAME(name)       ((void)0)
#define ITT_FINI_ITTLIB()               ((void)0)
#define ITT_SYNC_CREATE(obj, type, name) ((void)0)
#define ITT_SYNC_RENAME(obj, name)      ((void)0)
#define ITT_STACK_CREATE(obj)           ((void)0)
#define ITT_STACK(precond, name, obj)   ((void)0)

#endif /* !DO_ITT_NOTIFY */

namespace tbb {
namespace internal {
int __TBB_load_ittnotify();
}}

#endif /* _TBB_ITT_NOTIFY */
