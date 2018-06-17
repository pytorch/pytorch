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

#ifndef _TBB_scheduler_common_H
#define _TBB_scheduler_common_H

#include "tbb/tbb_machine.h"
#include "tbb/cache_aligned_allocator.h"

#include <string.h>  // for memset, memcpy, memmove

#include "tbb_statistics.h"

#if TBB_USE_ASSERT > 1
#include <stdio.h>
#endif /* TBB_USE_ASSERT > 1 */

/* Temporarily change "private" to "public" while including "tbb/task.h".
   This hack allows us to avoid publishing internal types and methods
   in the public header files just for sake of friend declarations. */
#ifndef private
    #define private public
    #define undef_private
#endif

#include "tbb/task.h"
#include "tbb/tbb_exception.h"

#ifdef undef_private
    #undef private
#endif

#ifndef __TBB_SCHEDULER_MUTEX_TYPE
#define __TBB_SCHEDULER_MUTEX_TYPE tbb::spin_mutex
#endif
// TODO: add conditional inclusion based on specified type
#include "tbb/spin_mutex.h"

// This macro is an attempt to get rid of ugly ifdefs in the shared parts of the code.
// It drops the second argument depending on whether the controlling macro is defined.
// The first argument is just a convenience allowing to keep comma before the macro usage.
#if __TBB_TASK_GROUP_CONTEXT
    #define __TBB_CONTEXT_ARG1(context) context
    #define __TBB_CONTEXT_ARG(arg1, context) arg1, context
#else /* !__TBB_TASK_GROUP_CONTEXT */
    #define __TBB_CONTEXT_ARG1(context)
    #define __TBB_CONTEXT_ARG(arg1, context) arg1
#endif /* !__TBB_TASK_GROUP_CONTEXT */

#if __TBB_TASK_ISOLATION
    #define __TBB_ISOLATION_EXPR(isolation) isolation
    #define __TBB_ISOLATION_ARG(arg1, isolation) arg1, isolation
#else
    #define __TBB_ISOLATION_EXPR(isolation)
    #define __TBB_ISOLATION_ARG(arg1, isolation) arg1
#endif /* __TBB_TASK_ISOLATION */


#if DO_TBB_TRACE
#include <cstdio>
#define TBB_TRACE(x) ((void)std::printf x)
#else
#define TBB_TRACE(x) ((void)(0))
#endif /* DO_TBB_TRACE */

#if !__TBB_CPU_CTL_ENV_PRESENT
#include <fenv.h>
#endif

#if _MSC_VER && !defined(__INTEL_COMPILER)
    // Workaround for overzealous compiler warnings
    // These particular warnings are so ubiquitous that no attempt is made to narrow
    // the scope of the warnings.
    #pragma warning (disable: 4100 4127 4312 4244 4267 4706)
#endif

namespace tbb {
namespace interface7 {
namespace internal {
class task_arena_base;
class delegated_task;
class wait_task;
}}
namespace internal {
using namespace interface7::internal;

class arena;
template<typename SchedulerTraits> class custom_scheduler;
class generic_scheduler;
class governor;
class mail_outbox;
class market;
class observer_proxy;
class task_scheduler_observer_v3;

#if __TBB_TASK_PRIORITY
static const intptr_t num_priority_levels = 3;
static const intptr_t normalized_normal_priority = (num_priority_levels - 1) / 2;

inline intptr_t normalize_priority ( priority_t p ) {
    return intptr_t(p - priority_low) / priority_stride_v4;
}

static const priority_t priority_from_normalized_rep[num_priority_levels] = {
    priority_low, priority_normal, priority_high
};

inline void assert_priority_valid ( intptr_t p ) {
    __TBB_ASSERT_EX( p >= 0 && p < num_priority_levels, NULL );
}

inline intptr_t& priority ( task& t ) {
    return t.prefix().context->my_priority;
}
#else /* __TBB_TASK_PRIORITY */
static const intptr_t num_priority_levels = 1;
#endif /* __TBB_TASK_PRIORITY */

//! Mutex type for global locks in the scheduler
typedef __TBB_SCHEDULER_MUTEX_TYPE scheduler_mutex_type;

#if __TBB_TASK_GROUP_CONTEXT
//! Task group state change propagation global epoch
/** Together with generic_scheduler::my_context_state_propagation_epoch forms
    cross-thread signaling mechanism that allows to avoid locking at the hot path
    of normal execution flow.

    When a descendant task group context is registered or unregistered, the global
    and local epochs are compared. If they differ, a state change is being propagated,
    and thus registration/deregistration routines take slower branch that may block
    (at most one thread of the pool can be blocked at any moment). Otherwise the
    control path is lock-free and fast. **/
extern uintptr_t the_context_state_propagation_epoch;

//! Mutex guarding state change propagation across task groups forest.
/** Also protects modification of related data structures. **/
typedef scheduler_mutex_type context_state_propagation_mutex_type;
extern context_state_propagation_mutex_type the_context_state_propagation_mutex;
#endif /* __TBB_TASK_GROUP_CONTEXT */

//! Alignment for a task object
const size_t task_alignment = 32;

//! Number of bytes reserved for a task prefix
/** If not exactly sizeof(task_prefix), the extra bytes *precede* the task_prefix. */
const size_t task_prefix_reservation_size = ((sizeof(internal::task_prefix)-1)/task_alignment+1)*task_alignment;

//! Definitions for bits in task_prefix::extra_state
enum task_extra_state {
    //! Tag for v1 tasks (i.e. tasks in TBB 1.0 and 2.0)
    es_version_1_task = 0,
    //! Tag for v3 tasks (i.e. tasks in TBB 2.1-2.2)
    es_version_3_task = 1,
    //! Tag for enqueued tasks
    es_task_enqueued = 0x10,
    //! Tag for v3 task_proxy.
    es_task_proxy = 0x20,
    //! Set if ref_count might be changed by another thread.  Used for debugging.
    es_ref_count_active = 0x40,
    //! Set if the task has been stolen
    es_task_is_stolen = 0x80
};

inline void reset_extra_state ( task *t ) {
    t->prefix().extra_state &= ~(es_task_is_stolen | es_task_enqueued);
}

//! Optimization hint to free_task that enables it omit unnecessary tests and code.
enum free_task_hint {
    //! No hint
    no_hint=0,
    //! Task is known to have been allocated by this scheduler
    local_task=1,
    //! Task is known to be a small task.
    /** Task should be returned to the free list of *some* scheduler, possibly not this scheduler. */
    small_task=2,
    //! Bitwise-OR of local_task and small_task.
    /** Task should be returned to free list of this scheduler. */
    small_local_task=3,
    //! Disable caching for a small task.
    no_cache = 4,
    //! Task is known to be a small task and must not be cached.
    no_cache_small_task = no_cache | small_task
};

//------------------------------------------------------------------------
// Debugging support
//------------------------------------------------------------------------

#if TBB_USE_ASSERT

static const uintptr_t venom = tbb::internal::select_size_t_constant<0xDEADBEEFU,0xDDEEAADDDEADBEEFULL>::value;

template <typename T>
void poison_value ( T& val ) { val = * punned_cast<T*>(&venom); }

/** Expected to be used in assertions only, thus no empty form is defined. **/
inline bool is_alive( uintptr_t v ) { return v != venom; }

/** Logically, this method should be a member of class task.
    But we do not want to publish it, so it is here instead. */
inline void assert_task_valid( const task* task ) {
    __TBB_ASSERT( task!=NULL, NULL );
    __TBB_ASSERT( !is_poisoned(&task), NULL );
    __TBB_ASSERT( (uintptr_t)task % task_alignment == 0, "misaligned task" );
#if __TBB_RECYCLE_TO_ENQUEUE
    __TBB_ASSERT( (unsigned)task->state()<=(unsigned)task::to_enqueue, "corrupt task (invalid state)" );
#else
    __TBB_ASSERT( (unsigned)task->state()<=(unsigned)task::recycle, "corrupt task (invalid state)" );
#endif
}

#else /* !TBB_USE_ASSERT */

/** In contrast to debug version poison_value() is a macro here because
    the variable used as its argument may be undefined in release builds. **/
#define poison_value(g) ((void)0)

inline void assert_task_valid( const task* ) {}

#endif /* !TBB_USE_ASSERT */

//------------------------------------------------------------------------
// Helpers
//------------------------------------------------------------------------

#if __TBB_TASK_GROUP_CONTEXT
inline bool ConcurrentWaitsEnabled ( task& t ) {
    return (t.prefix().context->my_version_and_traits & task_group_context::concurrent_wait) != 0;
}

inline bool CancellationInfoPresent ( task& t ) {
    return t.prefix().context->my_cancellation_requested != 0;
}

#if TBB_USE_CAPTURED_EXCEPTION
    inline tbb_exception* TbbCurrentException( task_group_context*, tbb_exception* src) { return src->move(); }
    inline tbb_exception* TbbCurrentException( task_group_context* c, captured_exception* src) {
        if( c->my_version_and_traits & task_group_context::exact_exception )
            runtime_warning( "Exact exception propagation is requested by application but the linked library is built without support for it");
        return src;
    }
    #define TbbRethrowException(TbbCapturedException) (TbbCapturedException)->throw_self()
#else
    // Using macro instead of an inline function here allows to avoid evaluation of the
    // TbbCapturedException expression when exact propagation is enabled for the context.
    #define TbbCurrentException(context, TbbCapturedException) \
        context->my_version_and_traits & task_group_context::exact_exception    \
            ? tbb_exception_ptr::allocate()    \
            : tbb_exception_ptr::allocate( *(TbbCapturedException) );
    #define TbbRethrowException(TbbCapturedException) \
        { \
            if( governor::rethrow_exception_broken() ) fix_broken_rethrow(); \
            (TbbCapturedException)->throw_self(); \
        }
#endif /* !TBB_USE_CAPTURED_EXCEPTION */

#define TbbRegisterCurrentException(context, TbbCapturedException) \
    if ( context->cancel_group_execution() ) {  \
        /* We are the first to signal cancellation, so store the exception that caused it. */  \
        context->my_exception = TbbCurrentException( context, TbbCapturedException ); \
    }

#define TbbCatchAll(context)  \
    catch ( tbb_exception& exc ) {  \
        TbbRegisterCurrentException( context, &exc );   \
    } catch ( std::exception& exc ) {   \
        TbbRegisterCurrentException( context, captured_exception::allocate(typeid(exc).name(), exc.what()) ); \
    } catch ( ... ) {   \
        TbbRegisterCurrentException( context, captured_exception::allocate("...", "Unidentified exception") );\
    }

#else /* !__TBB_TASK_GROUP_CONTEXT */

inline bool ConcurrentWaitsEnabled ( task& t ) { return false; }

#endif /* __TBB_TASK_GROUP_CONTEXT */

inline void prolonged_pause() {
#if defined(__TBB_time_stamp) && !__TBB_STEALING_PAUSE
    // Assumption based on practice: 1000-2000 ticks seems to be a suitable invariant for the
    // majority of platforms. Currently, skip platforms that define __TBB_STEALING_PAUSE
    // because these platforms require very careful tuning.
    machine_tsc_t prev = __TBB_time_stamp();
    const machine_tsc_t finish = prev + 1000;
    atomic_backoff backoff;
    do {
        backoff.bounded_pause();
        machine_tsc_t curr = __TBB_time_stamp();
        if ( curr <= prev )
            // Possibly, the current logical thread is moved to another hardware thread or overflow is occurred.
            break;
        prev = curr;
    } while ( prev < finish );
#else
#ifdef __TBB_STEALING_PAUSE
    static const long PauseTime = __TBB_STEALING_PAUSE;
#elif __TBB_ipf
    static const long PauseTime = 1500;
#else
    static const long PauseTime = 80;
#endif
    // TODO IDEA: Update PauseTime adaptively?
    __TBB_Pause(PauseTime);
#endif
}

//------------------------------------------------------------------------
// arena_slot
//------------------------------------------------------------------------
struct arena_slot_line1 {
    //TODO: make this tbb:atomic<>.
    //! Scheduler of the thread attached to the slot
    /** Marks the slot as busy, and is used to iterate through the schedulers belonging to this arena **/
    generic_scheduler* my_scheduler;

    // Synchronization of access to Task pool
    /** Also is used to specify if the slot is empty or locked:
         0 - empty
        -1 - locked **/
    task* *__TBB_atomic task_pool;

    //! Index of the first ready task in the deque.
    /** Modified by thieves, and by the owner during compaction/reallocation **/
    __TBB_atomic size_t head;
};

struct arena_slot_line2 {
    //! Hint provided for operations with the container of starvation-resistant tasks.
    /** Modified by the owner thread (during these operations). **/
    unsigned hint_for_pop;

    //! Index of the element following the last ready task in the deque.
    /** Modified by the owner thread. **/
    __TBB_atomic size_t tail;

    //! Capacity of the primary task pool (number of elements - pointers to task).
    size_t my_task_pool_size;

    // Task pool of the scheduler that owns this slot
    task* *__TBB_atomic task_pool_ptr;

#if __TBB_STATISTICS
    //! Set of counters to accumulate internal statistics related to this arena
    statistics_counters *my_counters;
#endif /* __TBB_STATISTICS */
};

struct arena_slot : padded<arena_slot_line1>, padded<arena_slot_line2> {
#if TBB_USE_ASSERT
    void fill_with_canary_pattern ( size_t first, size_t last ) {
        for ( size_t i = first; i < last; ++i )
            poison_pointer(task_pool_ptr[i]);
    }
#else
    void fill_with_canary_pattern ( size_t, size_t ) {}
#endif /* TBB_USE_ASSERT */

    void allocate_task_pool( size_t n ) {
        size_t byte_size = ((n * sizeof(task*) + NFS_MaxLineSize - 1) / NFS_MaxLineSize) * NFS_MaxLineSize;
        my_task_pool_size = byte_size / sizeof(task*);
        task_pool_ptr = (task**)NFS_Allocate( 1, byte_size, NULL );
        // No need to clear the fresh deque since valid items are designated by the head and tail members.
        // But fill it with a canary pattern in the high vigilance debug mode.
        fill_with_canary_pattern( 0, my_task_pool_size );
    }

    //! Deallocate task pool that was allocated by means of allocate_task_pool.
    void free_task_pool( ) {
        // TODO: understand the assertion and modify
        // __TBB_ASSERT( !task_pool /*TODO: == EmptyTaskPool*/, NULL);
        if( task_pool_ptr ) {
           __TBB_ASSERT( my_task_pool_size, NULL);
           NFS_Free( task_pool_ptr );
           task_pool_ptr = NULL;
           my_task_pool_size = 0;
        }
    }
};

#if !__TBB_CPU_CTL_ENV_PRESENT
class cpu_ctl_env {
    fenv_t *my_fenv_ptr;
public:
    cpu_ctl_env() : my_fenv_ptr(NULL) {}
    ~cpu_ctl_env() {
        if ( my_fenv_ptr )
            tbb::internal::NFS_Free( (void*)my_fenv_ptr );
    }
    // It is possible not to copy memory but just to copy pointers but the following issues should be addressed:
    //   1. The arena lifetime and the context lifetime are independent;
    //   2. The user is allowed to recapture different FPU settings to context so 'current FPU settings' inside
    //   dispatch loop may become invalid.
    // But do we really want to improve the fenv implementation? It seems to be better to replace the fenv implementation
    // with a platform specific implementation.
    cpu_ctl_env( const cpu_ctl_env &src ) : my_fenv_ptr(NULL) {
        *this = src;
    }
    cpu_ctl_env& operator=( const cpu_ctl_env &src ) {
        __TBB_ASSERT( src.my_fenv_ptr, NULL );
        if ( !my_fenv_ptr )
            my_fenv_ptr = (fenv_t*)tbb::internal::NFS_Allocate(1, sizeof(fenv_t), NULL);
        *my_fenv_ptr = *src.my_fenv_ptr;
        return *this;
    }
    bool operator!=( const cpu_ctl_env &ctl ) const {
        __TBB_ASSERT( my_fenv_ptr, "cpu_ctl_env is not initialized." );
        __TBB_ASSERT( ctl.my_fenv_ptr, "cpu_ctl_env is not initialized." );
        return memcmp( (void*)my_fenv_ptr, (void*)ctl.my_fenv_ptr, sizeof(fenv_t) );
    }
    void get_env () {
        if ( !my_fenv_ptr )
            my_fenv_ptr = (fenv_t*)tbb::internal::NFS_Allocate(1, sizeof(fenv_t), NULL);
        fegetenv( my_fenv_ptr );
    }
    const cpu_ctl_env& set_env () const {
        __TBB_ASSERT( my_fenv_ptr, "cpu_ctl_env is not initialized." );
        fesetenv( my_fenv_ptr );
        return *this;
    }
};
#endif /* !__TBB_CPU_CTL_ENV_PRESENT */

} // namespace internal
} // namespace tbb

#endif /* _TBB_scheduler_common_H */
