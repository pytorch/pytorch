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

#ifndef _TBB_tbb_main_H
#define _TBB_tbb_main_H

#include "tbb/atomic.h"
#include "governor.h"

namespace tbb {

namespace internal {

void DoOneTimeInitializations ();

//------------------------------------------------------------------------
// __TBB_InitOnce
//------------------------------------------------------------------------

//! Class that supports TBB initialization.
/** It handles acquisition and release of global resources (e.g. TLS) during startup and shutdown,
    as well as synchronization for DoOneTimeInitializations. */
class __TBB_InitOnce {
    friend void DoOneTimeInitializations();
    friend void ITT_DoUnsafeOneTimeInitialization ();

    static atomic<int> count;

    //! Platform specific code to acquire resources.
    static void acquire_resources();

    //! Platform specific code to release resources.
    static void release_resources();

    //! Specifies if the one-time initializations has been done.
    static bool InitializationDone;

    //! Global initialization lock
    /** Scenarios are possible when tools interop has to be initialized before the
        TBB itself. This imposes a requirement that the global initialization lock
        has to support valid static initialization, and does not issue any tool
        notifications in any build mode. **/
    static __TBB_atomic_flag InitializationLock;

public:
    static void lock()   { __TBB_LockByte( InitializationLock ); }

    static void unlock() { __TBB_UnlockByte( InitializationLock ); }

    static bool initialization_done() { return __TBB_load_with_acquire(InitializationDone); }

    //! Add initial reference to resources.
    /** We assume that dynamic loading of the library prevents any other threads
        from entering the library until this constructor has finished running. **/
    __TBB_InitOnce() { add_ref(); }

    //! Remove the initial reference to resources.
    /** This is not necessarily the last reference if other threads are still running. **/
    ~__TBB_InitOnce() {
        governor::terminate_auto_initialized_scheduler(); // TLS dtor not called for the main thread
        remove_ref();
        // We assume that InitializationDone is not set after file-scope destructors
        // start running, and thus no race on InitializationDone is possible.
        if( initialization_done() ) {
            // Remove an extra reference that was added in DoOneTimeInitializations.
            remove_ref();
        }
    }
    //! Add reference to resources.  If first reference added, acquire the resources.
    static void add_ref();

    //! Remove reference to resources.  If last reference removed, release the resources.
    static void remove_ref();
}; // class __TBB_InitOnce


} // namespace internal

} // namespace tbb

#endif /* _TBB_tbb_main_H */
