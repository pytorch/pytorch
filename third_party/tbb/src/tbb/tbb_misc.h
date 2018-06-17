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

#ifndef _TBB_tbb_misc_H
#define _TBB_tbb_misc_H

#include "tbb/tbb_stddef.h"
#include "tbb/tbb_machine.h"
#include "tbb/atomic.h"     // For atomic_xxx definitions

#if __linux__ || __FreeBSD__
#include <sys/param.h>  // __FreeBSD_version
#if __FreeBSD_version >= 701000
#include <sys/cpuset.h>
#endif
#endif

// Does the operating system have a system call to pin a thread to a set of OS processors?
#define __TBB_OS_AFFINITY_SYSCALL_PRESENT ((__linux__ && !__ANDROID__) || (__FreeBSD_version >= 701000))
// On IBM* Blue Gene* CNK nodes, the affinity API has restrictions that prevent its usability for TBB,
// and also sysconf(_SC_NPROCESSORS_ONLN) already takes process affinity into account.
#define __TBB_USE_OS_AFFINITY_SYSCALL (__TBB_OS_AFFINITY_SYSCALL_PRESENT && !__bg__)

namespace tbb {
namespace internal {

const size_t MByte = 1024*1024;

#if __TBB_WIN8UI_SUPPORT
// In Win8UI mode, TBB uses a thread creation API that does not allow to specify the stack size.
// Still, the thread stack size value, either explicit or default, is used by the scheduler.
// So here we set the default value to match the platform's default of 1MB.
const size_t ThreadStackSize = 1*MByte;
#else
const size_t ThreadStackSize = (sizeof(uintptr_t) <= 4 ? 2 : 4 )*MByte;
#endif

#ifndef __TBB_HardwareConcurrency

//! Returns maximal parallelism level supported by the current OS configuration.
int AvailableHwConcurrency();

#else

inline int AvailableHwConcurrency() {
    int n = __TBB_HardwareConcurrency();
    return n > 0 ? n : 1; // Fail safety strap
}
#endif /* __TBB_HardwareConcurrency */


#if _WIN32||_WIN64

//! Returns number of processor groups in the current OS configuration.
/** AvailableHwConcurrency must be called at least once before calling this method. **/
int NumberOfProcessorGroups();

//! Retrieves index of processor group containing processor with the given index
int FindProcessorGroupIndex ( int processorIndex );

//! Affinitizes the thread to the specified processor group
void MoveThreadIntoProcessorGroup( void* hThread, int groupIndex );

#endif /* _WIN32||_WIN64 */

//! Throws std::runtime_error with what() returning error_code description prefixed with aux_info
void handle_win_error( int error_code );

//! True if environment variable with given name is set and not 0; otherwise false.
bool GetBoolEnvironmentVariable( const char * name );

//! Prints TBB version information on stderr
void PrintVersion();

//! Prints arbitrary extra TBB version information on stderr
void PrintExtraVersionInfo( const char* category, const char* format, ... );

//! A callback routine to print RML version information on stderr
void PrintRMLVersionInfo( void* arg, const char* server_info );

// For TBB compilation only; not to be used in public headers
#if defined(min) || defined(max)
#undef min
#undef max
#endif

//! Utility template function returning lesser of the two values.
/** Provided here to avoid including not strict safe <algorithm>.\n
    In case operands cause signed/unsigned or size mismatch warnings it is caller's
    responsibility to do the appropriate cast before calling the function. **/
template<typename T>
T min ( const T& val1, const T& val2 ) {
    return val1 < val2 ? val1 : val2;
}

//! Utility template function returning greater of the two values.
/** Provided here to avoid including not strict safe <algorithm>.\n
    In case operands cause signed/unsigned or size mismatch warnings it is caller's
    responsibility to do the appropriate cast before calling the function. **/
template<typename T>
T max ( const T& val1, const T& val2 ) {
    return val1 < val2 ? val2 : val1;
}

//! Utility helper structure to ease overload resolution
template<int > struct int_to_type {};

//------------------------------------------------------------------------
// FastRandom
//------------------------------------------------------------------------

/** Defined in tbb_main.cpp **/
unsigned GetPrime ( unsigned seed );

//! A fast random number generator.
/** Uses linear congruential method. */
class FastRandom {
private:
#if __TBB_OLD_PRIMES_RNG
    unsigned x, a;
    static const unsigned c = 1;
#else
    unsigned x, c;
    static const unsigned a = 0x9e3779b1; // a big prime number
#endif //__TBB_OLD_PRIMES_RNG
public:
    //! Get a random number.
    unsigned short get() {
        return get(x);
    }
    //! Get a random number for the given seed; update the seed for next use.
    unsigned short get( unsigned& seed ) {
        unsigned short r = (unsigned short)(seed>>16);
        __TBB_ASSERT(c&1, "c must be odd for big rng period");
        seed = seed*a+c;
        return r;
    }
    //! Construct a random number generator.
    FastRandom( void* unique_ptr ) { init(uintptr_t(unique_ptr)); }
    FastRandom( uint32_t seed) { init(seed); }
    FastRandom( uint64_t seed) { init(seed); }
    template <typename T>
    void init( T seed ) {
        init(seed,int_to_type<sizeof(seed)>());
    }
    void init( uint64_t seed , int_to_type<8> ) {
        init(uint32_t((seed>>32)+seed), int_to_type<4>());
    }
    void init( uint32_t seed, int_to_type<4> ) {
#if __TBB_OLD_PRIMES_RNG
        x = seed;
        a = GetPrime( seed );
#else
        // threads use different seeds for unique sequences
        c = (seed|1)*0xba5703f5; // c must be odd, shuffle by a prime number
        x = c^(seed>>1); // also shuffle x for the first get() invocation
#endif
    }
};

//------------------------------------------------------------------------
// Atomic extensions
//------------------------------------------------------------------------

//! Atomically replaces value of dst with newValue if they satisfy condition of compare predicate
/** Return value semantics is the same as for CAS. **/
template<typename T1, typename T2, class Pred>
T1 atomic_update ( tbb::atomic<T1>& dst, T2 newValue, Pred compare ) {
    T1 oldValue = dst;
    while ( compare(oldValue, newValue) ) {
        if ( dst.compare_and_swap((T1)newValue, oldValue) == oldValue )
            break;
        oldValue = dst;
    }
    return oldValue;
}

//! One-time initialization states
enum do_once_state {
    do_once_uninitialized = 0,  ///< No execution attempts have been undertaken yet
    do_once_pending,            ///< A thread is executing associated do-once routine
    do_once_executed,           ///< Do-once routine has been executed
    initialization_complete = do_once_executed  ///< Convenience alias
};

//! One-time initialization function
/** /param initializer Pointer to function without arguments
           The variant that returns bool is used for cases when initialization can fail
           and it is OK to continue execution, but the state should be reset so that
           the initialization attempt was repeated the next time.
    /param state Shared state associated with initializer that specifies its
            initialization state. Must be initially set to #uninitialized value
            (e.g. by means of default static zero initialization). **/
template <typename F>
void atomic_do_once ( const F& initializer, atomic<do_once_state>& state ) {
    // tbb::atomic provides necessary acquire and release fences.
    // The loop in the implementation is necessary to avoid race when thread T2
    // that arrived in the middle of initialization attempt by another thread T1
    // has just made initialization possible.
    // In such a case T2 has to rely on T1 to initialize, but T1 may already be past
    // the point where it can recognize the changed conditions.
    while ( state != do_once_executed ) {
        if( state == do_once_uninitialized ) {
            if( state.compare_and_swap( do_once_pending, do_once_uninitialized ) == do_once_uninitialized ) {
                run_initializer( initializer, state );
                break;
            }
        }
        spin_wait_while_eq( state, do_once_pending );
    }
}

// Run the initializer which can not fail
inline void run_initializer( void (*f)(), atomic<do_once_state>& state ) {
    f();
    state = do_once_executed;
}

// Run the initializer which can require repeated call
inline void run_initializer( bool (*f)(), atomic<do_once_state>& state ) {
    state = f() ? do_once_executed : do_once_uninitialized;
}

#if __TBB_USE_OS_AFFINITY_SYSCALL
  #if __linux__
    typedef cpu_set_t basic_mask_t;
  #elif __FreeBSD_version >= 701000
    typedef cpuset_t basic_mask_t;
  #else
    #error affinity_helper is not implemented in this OS
  #endif
    class affinity_helper : no_copy {
        basic_mask_t* threadMask;
        int is_changed;
    public:
        affinity_helper() : threadMask(NULL), is_changed(0) {}
        ~affinity_helper();
        void protect_affinity_mask( bool restore_process_mask  );
        void dismiss();
    };
    void destroy_process_mask();
#else
    class affinity_helper : no_copy {
    public:
        void protect_affinity_mask( bool ) {}
        void dismiss() {}
    };
    inline void destroy_process_mask(){}
#endif /* __TBB_USE_OS_AFFINITY_SYSCALL */

bool cpu_has_speculation();
bool gcc_rethrow_exception_broken();
void fix_broken_rethrow();

} // namespace internal
} // namespace tbb

#endif /* _TBB_tbb_misc_H */
