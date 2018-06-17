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

#ifndef _TBB_governor_H
#define _TBB_governor_H

#include "tbb/task_scheduler_init.h"
#include "../rml/include/rml_tbb.h"

#include "tbb_misc.h" // for AvailableHwConcurrency
#include "tls.h"

#if __TBB_SURVIVE_THREAD_SWITCH
#include "cilk-tbb-interop.h"
#endif /* __TBB_SURVIVE_THREAD_SWITCH */

namespace tbb {
namespace internal {

class market;
class generic_scheduler;
class __TBB_InitOnce;

namespace rml {
class tbb_client;
}

//------------------------------------------------------------------------
// Class governor
//------------------------------------------------------------------------

//! The class handles access to the single instance of market, and to TLS to keep scheduler instances.
/** It also supports automatic on-demand initialization of the TBB scheduler.
    The class contains only static data members and methods.*/
class governor {
private:
    friend class __TBB_InitOnce;
    friend class market;

    //! TLS for scheduler instances associated with individual threads
    static basic_tls<uintptr_t> theTLS;

    //! Caches the maximal level of parallelism supported by the hardware
    static unsigned DefaultNumberOfThreads;

    static rml::tbb_factory theRMLServerFactory;

    static bool UsePrivateRML;

    // Flags for runtime-specific conditions
    static bool is_speculation_enabled;
    static bool is_rethrow_broken;

    //! Create key for thread-local storage and initialize RML.
    static void acquire_resources ();

    //! Destroy the thread-local storage key and deinitialize RML.
    static void release_resources ();

    static rml::tbb_server* create_rml_server ( rml::tbb_client& );

    //! The internal routine to undo automatic initialization.
    /** The signature is written with void* so that the routine
        can be the destructor argument to pthread_key_create. */
    static void auto_terminate(void* scheduler);

public:
    static unsigned default_num_threads () {
        // No memory fence required, because at worst each invoking thread calls AvailableHwConcurrency once.
        return DefaultNumberOfThreads ? DefaultNumberOfThreads :
                                        DefaultNumberOfThreads = AvailableHwConcurrency();
    }
    static void one_time_init();
    //! Processes scheduler initialization request (possibly nested) in a master thread
    /** If necessary creates new instance of arena and/or local scheduler.
        The auto_init argument specifies if the call is due to automatic initialization. **/
    static generic_scheduler* init_scheduler( int num_threads, stack_size_type stack_size, bool auto_init );

    //! Automatic initialization of scheduler in a master thread with default settings without arena
    static generic_scheduler* init_scheduler_weak();

    //! Processes scheduler termination request (possibly nested) in a master thread
    static bool terminate_scheduler( generic_scheduler* s, const task_scheduler_init *tsi_ptr, bool blocking );

    //! Register TBB scheduler instance in thread-local storage.
    static void sign_on( generic_scheduler* s );

    //! Unregister TBB scheduler instance from thread-local storage.
    static void sign_off( generic_scheduler* s );

    //! Used to check validity of the local scheduler TLS contents.
    static bool is_set( generic_scheduler* s );

    //! Temporarily set TLS slot to the given scheduler
    static void assume_scheduler( generic_scheduler* s );

    //! Computes the value of the TLS
    static uintptr_t tls_value_of( generic_scheduler* s );

    // TODO IDEA: refactor bit manipulations over pointer types to a class?
    //! Converts TLS value to the scheduler pointer
    static generic_scheduler* tls_scheduler_of( uintptr_t v ) {
        return (generic_scheduler*)(v & ~uintptr_t(1));
    }

    //! Obtain the thread-local instance of the TBB scheduler.
    /** If the scheduler has not been initialized yet, initialization is done automatically.
        Note that auto-initialized scheduler instance is destroyed only when its thread terminates. **/
    static generic_scheduler* local_scheduler () {
        uintptr_t v = theTLS.get();
        return (v&1) ? tls_scheduler_of(v) : init_scheduler( task_scheduler_init::automatic, 0, /*auto_init=*/true );
    }

    static generic_scheduler* local_scheduler_weak () {
        uintptr_t v = theTLS.get();
        return v ? tls_scheduler_of(v) : init_scheduler_weak();
    }

    static generic_scheduler* local_scheduler_if_initialized () {
        return tls_scheduler_of( theTLS.get() );
    }

    //! Undo automatic initialization if necessary; call when a thread exits.
    static void terminate_auto_initialized_scheduler() {
        auto_terminate( local_scheduler_if_initialized() );
    }

    static void print_version_info ();

    static void initialize_rml_factory ();

    static bool does_client_join_workers (const tbb::internal::rml::tbb_client &client);

#if __TBB_SURVIVE_THREAD_SWITCH
    static __cilk_tbb_retcode stack_op_handler( __cilk_tbb_stack_op op, void* );
#endif /* __TBB_SURVIVE_THREAD_SWITCH */

    static bool speculation_enabled() { return is_speculation_enabled; }
    static bool rethrow_exception_broken() { return is_rethrow_broken; }

}; // class governor

} // namespace internal
} // namespace tbb

#endif /* _TBB_governor_H */
