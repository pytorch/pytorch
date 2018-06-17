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

#ifndef _TBB_malloc_Customize_H_
#define _TBB_malloc_Customize_H_

// customizing MALLOC_ASSERT macro
#include "tbb/tbb_stddef.h"
#define MALLOC_ASSERT(assertion, message) __TBB_ASSERT(assertion, message)

#ifndef MALLOC_DEBUG
#define MALLOC_DEBUG TBB_USE_DEBUG
#endif

#include "tbb/tbb_machine.h"

#if DO_ITT_NOTIFY
#include "tbb/itt_notify.h"
#define MALLOC_ITT_SYNC_PREPARE(pointer) ITT_NOTIFY(sync_prepare, (pointer))
#define MALLOC_ITT_SYNC_ACQUIRED(pointer) ITT_NOTIFY(sync_acquired, (pointer))
#define MALLOC_ITT_SYNC_RELEASING(pointer) ITT_NOTIFY(sync_releasing, (pointer))
#define MALLOC_ITT_SYNC_CANCEL(pointer) ITT_NOTIFY(sync_cancel, (pointer))
#define MALLOC_ITT_FINI_ITTLIB()        ITT_FINI_ITTLIB()
#else
#define MALLOC_ITT_SYNC_PREPARE(pointer) ((void)0)
#define MALLOC_ITT_SYNC_ACQUIRED(pointer) ((void)0)
#define MALLOC_ITT_SYNC_RELEASING(pointer) ((void)0)
#define MALLOC_ITT_SYNC_CANCEL(pointer) ((void)0)
#define MALLOC_ITT_FINI_ITTLIB()        ((void)0)
#endif

//! Stripped down version of spin_mutex.
/** Instances of MallocMutex must be declared in memory that is zero-initialized.
    There are no constructors.  This is a feature that lets it be
    used in situations where the mutex might be used while file-scope constructors
    are running.

    There are no methods "acquire" or "release".  The scoped_lock must be used
    in a strict block-scoped locking pattern.  Omitting these methods permitted
    further simplification. */
class MallocMutex : tbb::internal::no_copy {
    __TBB_atomic_flag flag;

public:
    class scoped_lock : tbb::internal::no_copy {
        MallocMutex& mutex;
        bool taken;
    public:
        scoped_lock( MallocMutex& m ) : mutex(m), taken(true) { __TBB_LockByte(m.flag); }
        scoped_lock( MallocMutex& m, bool block, bool *locked ) : mutex(m), taken(false) {
            if (block) {
                __TBB_LockByte(m.flag);
                taken = true;
            } else {
                taken = __TBB_TryLockByte(m.flag);
            }
            if (locked) *locked = taken;
        }
        ~scoped_lock() {
            if (taken) __TBB_UnlockByte(mutex.flag);
        }
    };
    friend class scoped_lock;
};

// TODO: use signed/unsigned in atomics more consistently
inline intptr_t AtomicIncrement( volatile intptr_t& counter ) {
    return __TBB_FetchAndAddW( &counter, 1 )+1;
}

inline uintptr_t AtomicAdd( volatile intptr_t& counter, intptr_t value ) {
    return __TBB_FetchAndAddW( &counter, value );
}

inline intptr_t AtomicCompareExchange( volatile intptr_t& location, intptr_t new_value, intptr_t comparand) {
    return __TBB_CompareAndSwapW( &location, new_value, comparand );
}

inline uintptr_t AtomicFetchStore(volatile void* location, uintptr_t value) {
    return __TBB_FetchAndStoreW(location, value);
}

inline void AtomicOr(volatile void *operand, uintptr_t addend) {
    __TBB_AtomicOR(operand, addend);
}

inline void AtomicAnd(volatile void *operand, uintptr_t addend) {
    __TBB_AtomicAND(operand, addend);
}

inline intptr_t FencedLoad( const volatile intptr_t &location ) {
    return __TBB_load_with_acquire(location);
}

inline void FencedStore( volatile intptr_t &location, intptr_t value ) {
    __TBB_store_with_release(location, value);
}

inline void SpinWaitWhileEq(const volatile intptr_t &location, const intptr_t value) {
    tbb::internal::spin_wait_while_eq(location, value);
}

class AtomicBackoff {
    tbb::internal::atomic_backoff backoff;
public:
    AtomicBackoff() {}
    void pause() { backoff.pause(); }
};

inline void SpinWaitUntilEq(const volatile intptr_t &location, const intptr_t value) {
    tbb::internal::spin_wait_until_eq(location, value);
}

inline intptr_t BitScanRev(uintptr_t x) {
    return !x? -1 : __TBB_Log2(x);
}

template<typename T>
static inline bool isAligned(T* arg, uintptr_t alignment) {
    return tbb::internal::is_aligned(arg,alignment);
}

static inline bool isPowerOfTwo(uintptr_t arg) {
    return tbb::internal::is_power_of_two(arg);
}
static inline bool isPowerOfTwoAtLeast(uintptr_t arg, uintptr_t power2) {
    return arg && tbb::internal::is_power_of_two_at_least(arg,power2);
}

#define MALLOC_STATIC_ASSERT(condition,msg) __TBB_STATIC_ASSERT(condition,msg)

#define USE_DEFAULT_MEMORY_MAPPING 1

// To support malloc replacement
#include "proxy.h"

#if MALLOC_UNIXLIKE_OVERLOAD_ENABLED
#define malloc_proxy __TBB_malloc_proxy
extern "C" void * __TBB_malloc_proxy(size_t)  __attribute__ ((weak));
#elif MALLOC_ZONE_OVERLOAD_ENABLED
// as there is no significant overhead, always suppose that proxy can be present
const bool malloc_proxy = true;
#else
const bool malloc_proxy = false;
#endif

namespace rml {
namespace internal {
    void init_tbbmalloc();
} } // namespaces

#define MALLOC_EXTRA_INITIALIZATION rml::internal::init_tbbmalloc()

// Need these to work regardless of tools support.
namespace tbb {
    namespace internal {

        enum notify_type {prepare=0, cancel, acquired, releasing};

#if TBB_USE_THREADING_TOOLS
        inline void call_itt_notify(notify_type t, void *ptr) {
            switch ( t ) {
            case prepare:
                MALLOC_ITT_SYNC_PREPARE( ptr );
                break;
            case cancel:
                MALLOC_ITT_SYNC_CANCEL( ptr );
                break;
            case acquired:
                MALLOC_ITT_SYNC_ACQUIRED( ptr );
                break;
            case releasing:
                MALLOC_ITT_SYNC_RELEASING( ptr );
                break;
            }
        }
#else
        inline void call_itt_notify(notify_type /*t*/, void * /*ptr*/) {}
#endif // TBB_USE_THREADING_TOOLS

        template <typename T>
        inline void itt_store_word_with_release(T& dst, T src) {
#if TBB_USE_THREADING_TOOLS
            call_itt_notify(releasing, &dst);
#endif // TBB_USE_THREADING_TOOLS
            FencedStore(*(intptr_t*)&dst, src);
        }

        template <typename T>
        inline T itt_load_word_with_acquire(T& src) {
            T result = FencedLoad(*(intptr_t*)&src);
#if TBB_USE_THREADING_TOOLS
            call_itt_notify(acquired, &src);
#endif // TBB_USE_THREADING_TOOLS
            return result;

        }
    } // namespace internal
} // namespace tbb

#include "tbb/internal/_aggregator_impl.h"

template <typename OperationType>
struct MallocAggregator {
    typedef tbb::internal::aggregator_generic<OperationType> type;
};

//! aggregated_operation base class
template <typename Derived>
struct MallocAggregatedOperation {
    typedef tbb::internal::aggregated_operation<Derived> type;
};

#endif /* _TBB_malloc_Customize_H_ */
