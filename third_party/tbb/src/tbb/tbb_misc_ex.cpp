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

// Source file for miscellaneous entities that are infrequently referenced by
// an executing program, and implementation of which requires dynamic linking.

#include "tbb_misc.h"

#if !defined(__TBB_HardwareConcurrency)

#include "dynamic_link.h"
#include <stdio.h>
#include <limits.h>

#if _WIN32||_WIN64
#include "tbb/machine/windows_api.h"
#if __TBB_WIN8UI_SUPPORT
#include <thread>
#endif
#else
#include <unistd.h>
#if __linux__
#include <sys/sysinfo.h>
#include <string.h>
#include <sched.h>
#include <errno.h>
#elif __sun
#include <sys/sysinfo.h>
#elif __FreeBSD__
#include <errno.h>
#include <string.h>
#include <sys/param.h>  // Required by <sys/cpuset.h>
#include <sys/cpuset.h>
#endif
#endif

namespace tbb {
namespace internal {

#if __TBB_USE_OS_AFFINITY_SYSCALL

#if __linux__
// Handlers for interoperation with libiomp
static int (*libiomp_try_restoring_original_mask)();
// Table for mapping to libiomp entry points
static const dynamic_link_descriptor iompLinkTable[] = {
    { "kmp_set_thread_affinity_mask_initial", (pointer_to_handler*)(void*)(&libiomp_try_restoring_original_mask) }
};
#endif

static void set_thread_affinity_mask( size_t maskSize, const basic_mask_t* threadMask ) {
#if __linux__
    if( sched_setaffinity( 0, maskSize, threadMask ) )
#else /* FreeBSD */
    if( cpuset_setaffinity( CPU_LEVEL_WHICH, CPU_WHICH_TID, -1, maskSize, threadMask ) )
#endif
        runtime_warning( "setaffinity syscall failed" );
}

static void get_thread_affinity_mask( size_t maskSize, basic_mask_t* threadMask ) {
#if __linux__
    if( sched_getaffinity( 0, maskSize, threadMask ) )
#else /* FreeBSD */
    if( cpuset_getaffinity( CPU_LEVEL_WHICH, CPU_WHICH_TID, -1, maskSize, threadMask ) )
#endif
        runtime_warning( "getaffinity syscall failed" );
}

static basic_mask_t* process_mask;
static int num_masks;

void destroy_process_mask() {
    if( process_mask ) {
        delete [] process_mask;
    }
}

#define curMaskSize sizeof(basic_mask_t) * num_masks
affinity_helper::~affinity_helper() {
    if( threadMask ) {
        if( is_changed ) {
            set_thread_affinity_mask( curMaskSize, threadMask );
        }
        delete [] threadMask;
    }
}
void affinity_helper::protect_affinity_mask( bool restore_process_mask ) {
    if( threadMask == NULL && num_masks ) { // TODO: assert num_masks validity?
        threadMask = new basic_mask_t [num_masks];
        memset( threadMask, 0, curMaskSize );
        get_thread_affinity_mask( curMaskSize, threadMask );
        if( restore_process_mask ) {
            __TBB_ASSERT( process_mask, "A process mask is requested but not yet stored" );
            is_changed = memcmp( process_mask, threadMask, curMaskSize );
            if( is_changed )
                set_thread_affinity_mask( curMaskSize, process_mask );
        } else {
            // Assume that the mask will be changed by the caller.
            is_changed = 1;
        }
    }
}
void affinity_helper::dismiss() {
    if( threadMask ) {
        delete [] threadMask;
        threadMask = NULL;
    }
    is_changed = 0;
}
#undef curMaskSize

static atomic<do_once_state> hardware_concurrency_info;

static int theNumProcs;

static void initialize_hardware_concurrency_info () {
    int err;
    int availableProcs = 0;
    int numMasks = 1;
#if __linux__
#if __TBB_MAIN_THREAD_AFFINITY_BROKEN
    int maxProcs = INT_MAX; // To check the entire mask.
    int pid = 0; // Get the mask of the calling thread.
#else
    int maxProcs = sysconf(_SC_NPROCESSORS_ONLN);
    int pid = getpid();
#endif
#else /* FreeBSD >= 7.1 */
    int maxProcs = sysconf(_SC_NPROCESSORS_ONLN);
#endif
    basic_mask_t* processMask;
    const size_t BasicMaskSize =  sizeof(basic_mask_t);
    for (;;) {
        const int curMaskSize = BasicMaskSize * numMasks;
        processMask = new basic_mask_t[numMasks];
        memset( processMask, 0, curMaskSize );
#if __linux__
        err = sched_getaffinity( pid, curMaskSize, processMask );
        if ( !err || errno != EINVAL || curMaskSize * CHAR_BIT >= 256 * 1024 )
            break;
#else /* FreeBSD >= 7.1 */
        // CPU_LEVEL_WHICH - anonymous (current) mask, CPU_LEVEL_CPUSET - assigned mask
#if __TBB_MAIN_THREAD_AFFINITY_BROKEN
        err = cpuset_getaffinity( CPU_LEVEL_WHICH, CPU_WHICH_TID, -1, curMaskSize, processMask );
#else
        err = cpuset_getaffinity( CPU_LEVEL_WHICH, CPU_WHICH_PID, -1, curMaskSize, processMask );
#endif
        if ( !err || errno != ERANGE || curMaskSize * CHAR_BIT >= 16 * 1024 )
            break;
#endif /* FreeBSD >= 7.1 */
        delete[] processMask;
        numMasks <<= 1;
    }
    if ( !err ) {
        // We have found the mask size and captured the process affinity mask into processMask.
        num_masks = numMasks; // do here because it's needed for affinity_helper to work
#if __linux__
        // For better coexistence with libiomp which might have changed the mask already,
        // check for its presence and ask it to restore the mask.
        dynamic_link_handle libhandle;
        if ( dynamic_link( "libiomp5.so", iompLinkTable, 1, &libhandle, DYNAMIC_LINK_GLOBAL ) ) {
            // We have found the symbol provided by libiomp5 for restoring original thread affinity.
            affinity_helper affhelp;
            affhelp.protect_affinity_mask( /*restore_process_mask=*/false );
            if ( libiomp_try_restoring_original_mask()==0 ) {
                // Now we have the right mask to capture, restored by libiomp.
                const int curMaskSize = BasicMaskSize * numMasks;
                memset( processMask, 0, curMaskSize );
                get_thread_affinity_mask( curMaskSize, processMask );
            } else
                affhelp.dismiss();  // thread mask has not changed
            dynamic_unlink( libhandle );
            // Destructor of affinity_helper restores the thread mask (unless dismissed).
        }
#endif
        for ( int m = 0; availableProcs < maxProcs && m < numMasks; ++m ) {
            for ( size_t i = 0; (availableProcs < maxProcs) && (i < BasicMaskSize * CHAR_BIT); ++i ) {
                if ( CPU_ISSET( i, processMask + m ) )
                    ++availableProcs;
            }
        }
        process_mask = processMask;
    }
    else {
        // Failed to get the process affinity mask; assume the whole machine can be used.
        availableProcs = (maxProcs == INT_MAX) ? sysconf(_SC_NPROCESSORS_ONLN) : maxProcs;
        delete[] processMask;
    }
    theNumProcs = availableProcs > 0 ? availableProcs : 1; // Fail safety strap
    __TBB_ASSERT( theNumProcs <= sysconf(_SC_NPROCESSORS_ONLN), NULL );
}

int AvailableHwConcurrency() {
    atomic_do_once( &initialize_hardware_concurrency_info, hardware_concurrency_info );
    return theNumProcs;
}

/* End of __TBB_USE_OS_AFFINITY_SYSCALL implementation */
#elif __ANDROID__

// Work-around for Android that reads the correct number of available CPUs since system calls are unreliable.
// Format of "present" file is: ([<int>-<int>|<int>],)+
int AvailableHwConcurrency() {
    FILE *fp = fopen("/sys/devices/system/cpu/present", "r");
    if (fp == NULL) return 1;
    int num_args, lower, upper, num_cpus=0;
    while ((num_args = fscanf(fp, "%u-%u", &lower, &upper)) != EOF) {
        switch(num_args) {
            case 2: num_cpus += upper - lower + 1; break;
            case 1: num_cpus += 1; break;
        }
        fscanf(fp, ",");
    }
    return (num_cpus > 0) ? num_cpus : 1;
}

#elif defined(_SC_NPROCESSORS_ONLN)

int AvailableHwConcurrency() {
    int n = sysconf(_SC_NPROCESSORS_ONLN);
    return (n > 0) ? n : 1;
}

#elif _WIN32||_WIN64

static atomic<do_once_state> hardware_concurrency_info;

static const WORD TBB_ALL_PROCESSOR_GROUPS = 0xffff;

// Statically allocate an array for processor group information.
// Windows 7 supports maximum 4 groups, but let's look ahead a little.
static const WORD MaxProcessorGroups = 64;

struct ProcessorGroupInfo {
    DWORD_PTR   mask;                   ///< Affinity mask covering the whole group
    int         numProcs;               ///< Number of processors in the group
    int         numProcsRunningTotal;   ///< Subtotal of processors in this and preceding groups

    //! Total number of processor groups in the system
    static int NumGroups;

    //! Index of the group with a slot reserved for the first master thread
    /** In the context of multiple processor groups support current implementation
        defines "the first master thread" as the first thread to invoke
        AvailableHwConcurrency().

        TODO:   Implement a dynamic scheme remapping workers depending on the pending
                master threads affinity. **/
    static int HoleIndex;
};

int ProcessorGroupInfo::NumGroups = 1;
int ProcessorGroupInfo::HoleIndex = 0;

ProcessorGroupInfo theProcessorGroups[MaxProcessorGroups];

struct TBB_GROUP_AFFINITY {
    DWORD_PTR Mask;
    WORD   Group;
    WORD   Reserved[3];
};

static DWORD (WINAPI *TBB_GetActiveProcessorCount)( WORD groupIndex ) = NULL;
static WORD (WINAPI *TBB_GetActiveProcessorGroupCount)() = NULL;
static BOOL (WINAPI *TBB_SetThreadGroupAffinity)( HANDLE hThread,
                        const TBB_GROUP_AFFINITY* newAff, TBB_GROUP_AFFINITY *prevAff );
static BOOL (WINAPI *TBB_GetThreadGroupAffinity)( HANDLE hThread, TBB_GROUP_AFFINITY* );

static const dynamic_link_descriptor ProcessorGroupsApiLinkTable[] = {
      DLD(GetActiveProcessorCount, TBB_GetActiveProcessorCount)
    , DLD(GetActiveProcessorGroupCount, TBB_GetActiveProcessorGroupCount)
    , DLD(SetThreadGroupAffinity, TBB_SetThreadGroupAffinity)
    , DLD(GetThreadGroupAffinity, TBB_GetThreadGroupAffinity)
};

static void initialize_hardware_concurrency_info () {
#if __TBB_WIN8UI_SUPPORT
    // For these applications processor groups info is unavailable
    // Setting up a number of processors for one processor group
    theProcessorGroups[0].numProcs = theProcessorGroups[0].numProcsRunningTotal = std::thread::hardware_concurrency();
#else /* __TBB_WIN8UI_SUPPORT */
    dynamic_link( "Kernel32.dll", ProcessorGroupsApiLinkTable,
                  sizeof(ProcessorGroupsApiLinkTable)/sizeof(dynamic_link_descriptor) );
    SYSTEM_INFO si;
    GetNativeSystemInfo(&si);
    DWORD_PTR pam, sam, m = 1;
    GetProcessAffinityMask( GetCurrentProcess(), &pam, &sam );
    int nproc = 0;
    for ( size_t i = 0; i < sizeof(DWORD_PTR) * CHAR_BIT; ++i, m <<= 1 ) {
        if ( pam & m )
            ++nproc;
    }
    __TBB_ASSERT( nproc <= (int)si.dwNumberOfProcessors, NULL );
    // By default setting up a number of processors for one processor group
    theProcessorGroups[0].numProcs = theProcessorGroups[0].numProcsRunningTotal = nproc;
    // Setting up processor groups in case the process does not restrict affinity mask and more than one processor group is present
    if ( nproc == (int)si.dwNumberOfProcessors && TBB_GetActiveProcessorCount ) {
        // The process does not have restricting affinity mask and multiple processor groups are possible
        ProcessorGroupInfo::NumGroups = (int)TBB_GetActiveProcessorGroupCount();
        __TBB_ASSERT( ProcessorGroupInfo::NumGroups <= MaxProcessorGroups, NULL );
        // Fail safety bootstrap. Release versions will limit available concurrency
        // level, while debug ones would assert.
        if ( ProcessorGroupInfo::NumGroups > MaxProcessorGroups )
            ProcessorGroupInfo::NumGroups = MaxProcessorGroups;
        if ( ProcessorGroupInfo::NumGroups > 1 ) {
            TBB_GROUP_AFFINITY ga;
            if ( TBB_GetThreadGroupAffinity( GetCurrentThread(), &ga ) )
                ProcessorGroupInfo::HoleIndex = ga.Group;
            int nprocs = 0;
            for ( WORD i = 0; i < ProcessorGroupInfo::NumGroups; ++i ) {
                ProcessorGroupInfo  &pgi = theProcessorGroups[i];
                pgi.numProcs = (int)TBB_GetActiveProcessorCount(i);
                __TBB_ASSERT( pgi.numProcs <= (int)sizeof(DWORD_PTR) * CHAR_BIT, NULL );
                pgi.mask = pgi.numProcs == sizeof(DWORD_PTR) * CHAR_BIT ? ~(DWORD_PTR)0 : (DWORD_PTR(1) << pgi.numProcs) - 1;
                pgi.numProcsRunningTotal = nprocs += pgi.numProcs;
            }
            __TBB_ASSERT( nprocs == (int)TBB_GetActiveProcessorCount( TBB_ALL_PROCESSOR_GROUPS ), NULL );
        }
    }
#endif /* __TBB_WIN8UI_SUPPORT */

    PrintExtraVersionInfo("Processor groups", "%d", ProcessorGroupInfo::NumGroups);
    if (ProcessorGroupInfo::NumGroups>1)
        for (int i=0; i<ProcessorGroupInfo::NumGroups; ++i)
            PrintExtraVersionInfo( "----- Group", "%d: size %d", i, theProcessorGroups[i].numProcs);
}

int NumberOfProcessorGroups() {
    __TBB_ASSERT( hardware_concurrency_info == initialization_complete, "NumberOfProcessorGroups is used before AvailableHwConcurrency" );
    return ProcessorGroupInfo::NumGroups;
}

// Offset for the slot reserved for the first master thread
#define HoleAdjusted(procIdx, grpIdx) (procIdx + (holeIdx <= grpIdx))

int FindProcessorGroupIndex ( int procIdx ) {
    // In case of oversubscription spread extra workers in a round robin manner
    int holeIdx;
    const int numProcs = theProcessorGroups[ProcessorGroupInfo::NumGroups - 1].numProcsRunningTotal;
    if ( procIdx >= numProcs - 1 ) {
        holeIdx = INT_MAX;
        procIdx = (procIdx - numProcs + 1) % numProcs;
    }
    else
        holeIdx = ProcessorGroupInfo::HoleIndex;
    __TBB_ASSERT( hardware_concurrency_info == initialization_complete, "FindProcessorGroupIndex is used before AvailableHwConcurrency" );
    // Approximate the likely group index assuming all groups are of the same size
    int i = procIdx / theProcessorGroups[0].numProcs;
    // Make sure the approximation is a valid group index
    if (i >= ProcessorGroupInfo::NumGroups) i = ProcessorGroupInfo::NumGroups-1;
    // Now adjust the approximation up or down
    if ( theProcessorGroups[i].numProcsRunningTotal > HoleAdjusted(procIdx, i) ) {
        while ( theProcessorGroups[i].numProcsRunningTotal - theProcessorGroups[i].numProcs > HoleAdjusted(procIdx, i) ) {
            __TBB_ASSERT( i > 0, NULL );
            --i;
        }
    }
    else {
        do {
            ++i;
        } while ( theProcessorGroups[i].numProcsRunningTotal <= HoleAdjusted(procIdx, i) );
    }
    __TBB_ASSERT( i < ProcessorGroupInfo::NumGroups, NULL );
    return i;
}

void MoveThreadIntoProcessorGroup( void* hThread, int groupIndex ) {
    __TBB_ASSERT( hardware_concurrency_info == initialization_complete, "MoveThreadIntoProcessorGroup is used before AvailableHwConcurrency" );
    if ( !TBB_SetThreadGroupAffinity )
        return;
    TBB_GROUP_AFFINITY ga = { theProcessorGroups[groupIndex].mask, (WORD)groupIndex, {0,0,0} };
    TBB_SetThreadGroupAffinity( hThread, &ga, NULL );
}

int AvailableHwConcurrency() {
    atomic_do_once( &initialize_hardware_concurrency_info, hardware_concurrency_info );
    return theProcessorGroups[ProcessorGroupInfo::NumGroups - 1].numProcsRunningTotal;
}

/* End of _WIN32||_WIN64 implementation */
#else
    #error AvailableHwConcurrency is not implemented for this OS
#endif

} // namespace internal
} // namespace tbb

#endif /* !__TBB_HardwareConcurrency */
