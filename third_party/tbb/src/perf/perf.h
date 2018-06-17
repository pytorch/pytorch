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

#ifndef __tbb_perf_h__
#define __tbb_perf_h__

#ifndef TBB_PERF_TYPEINFO
#define TBB_PERF_TYPEINFO 1
#endif

#if TBB_PERF_TYPEINFO
    #include <typeinfo>
    #define __TBB_PERF_TEST_CLASS_NAME(T) typeid(T).name()
#else /* !TBB_PERF_TYPEINFO */
    #define __TBB_PERF_TEST_CLASS_NAME(T) NULL
#endif /* !TBB_PERF_TYPEINFO */


#include "tbb/tick_count.h"

// TODO: Fix build scripts to provide more reliable build phase identification means
#ifndef __TBB_PERF_API
#if _USRDLL
    #if _MSC_VER
        #define __TBB_PERF_API __declspec(dllexport)
    #else /* !_MSC_VER */
        #define __TBB_PERF_API
    #endif /* !_MSC_VER */
#else /* !_USRDLL */
    #if _MSC_VER
        #define __TBB_PERF_API __declspec(dllimport)
    #else /* !_MSC_VER */
        #define __TBB_PERF_API
    #endif /* !_MSC_VER */
#endif /* !_USRDLL */
#endif /* !__TBB_PERF_API */

#if _WIN32||_WIN64

namespace Perf {
    typedef unsigned __int64 tick_t;
    #if defined(_M_X64)
        inline tick_t rdtsc () { return __rdtsc(); }
    #elif _M_IX86
        inline tick_t rdtsc () { __asm { rdtsc } }
    #else
        #error Unsupported ISA
    #endif
} // namespace Perf

#elif __linux__ || __APPLE__

#include <stdint.h>

namespace Perf {
    typedef uint64_t tick_t;
    #if __x86_64__ || __i386__ || __i386
        inline tick_t rdtsc () {
            uint32_t lo, hi;
            __asm__ __volatile__ ( "rdtsc" : "=a" (lo), "=d" (hi) );
            return (tick_t)lo | ((tick_t)hi) << 32;
        }
    #else
        #error Unsupported ISA
    #endif
} // namespace Perf

#else
    #error Unsupported OS
#endif /* OS */

__TBB_PERF_API extern int NumThreads,
                          MaxConcurrency,
                          NumCpus;

// Functions and global variables provided by the benchmarking framework
namespace Perf {

typedef double duration_t;

static const int MaxWorkloadNameLen = 64;

static const char* NoHistogram = (char*)-1;
static const char* DefaultHistogram = (char*)-2;

__TBB_PERF_API void AnchorFunc ( void* );
__TBB_PERF_API void AnchorFunc2 ( void*, void*  );

//! Helper that can be used in the preprocess handler to clean caches
/** Cleaning caches is necessary to obtain reproducible results when a test
    accesses significant ranges of memory. **/
__TBB_PERF_API void WipeCaches ();

//! Specifies the name to be used to designate the current workload in output
/** Should be used from Test::SetWorkload(). If necessary workload name will be
    truncated to MaxWorkloadNameLen characters. **/
__TBB_PERF_API void SetWorkloadName( const char* format, ... );

class __TBB_PERF_API Test {
public:
    virtual ~Test () {}

    //! Struct used by tests running in multiple masters mode
    struct ThreadInfo {
        //! Zero based thread ID
        int     tid;
        //! Pointer to test specific data
        /** If used by the test, should be initialized by OnStartLocal(), and 
            finalized by OnFinishLocal(). **/
        void*   data;
    };

    ////////////////////////////////////////////////////////////////////////////////
    // Mandatory methods
    
    //! Returns the number of workloads supported
    virtual int NumWorkloads () = 0;

    //! Set workload info for the subsequent calls to Run() and RunSerial()
    /** This method can use global helper function Perf::SetWorkloadName() in order
        to specify the name of the current workload, which will be used in output
        to designate the workload. If SetWorkloadName is not called, workloadIndex
        will be used for this purpose.

        When testing task scheduler, make sure that this method does not trigger
        its automatic initialization. **/
    virtual void SetWorkload ( int workloadIndex ) = 0;

    //! Test implementation
    /** Called by the timing framework several times in a loop to achieve approx.
        RunDuration time, and this loop is timed NumRuns times to collect statistics.
        Argument ti specifies information about the master thread calling this method. **/
    virtual void Run ( ThreadInfo& ti ) = 0;

    ////////////////////////////////////////////////////////////////////////////////
    // Optional methods

    //! Returns short title string to be used in the regular output to identify the test
    /** Should uniquely identify the test among other ones in the given benchmark suite.
        If not implemented, the test implementation class' RTTI name is used. **/
    virtual const char* Name () { return NULL; };

    //! Returns minimal number of master threads
    /** Used for task scheduler tests only (when UseTbbScheduler option is specified 
        in session settings). **/
    virtual int MinNumMasters () { return 1; }

    //! Returns maximal number of master threads
    /** Used for task scheduler tests only (when UseTbbScheduler option is specified 
        in session settings). **/
    virtual int MaxNumMasters () { return 1; }

    //! Executes serial workload equivalent to the one processed by Run()
    /** Called by the timing framework several times in a loop to collect statistics. **/
    virtual void RunSerial ( ThreadInfo& ti );

    //! Invoked before each call to Run() 
    /** Can be used to preinitialize data necessary for the test, clean up 
        caches (see Perf::WipeCaches), etc.
        In multiple masters mode this method is called on each thread. **/
    virtual void OnStart ( ThreadInfo& ti );

    //! Invoked after each call to Run() 
    /** Can be used to free resources allocated by OnStart().
        Note that this method must work correctly independently of whether Run(),
        RunSerial() or nothing is called between OnStart() and OnFinish().
        In multiple masters mode this method is called on each thread. **/
    virtual void OnFinish ( ThreadInfo& ti );

    //! Functionality, the cost of which has to be factored out from timing results
    /** Applies to both parallel and serial versions. **/
    virtual void Baseline ( ThreadInfo& );

    //! Returns description string to be used in the benchmark info/summary output
    virtual const char* Description () { return NULL; }

    //! Specifies if the histogram of individual run times in a series
    /** If the method is not overridden, histogramName argument of TestMain is used. **/
    virtual const char* HistogramName () { return DefaultHistogram; }
}; // class Test

namespace internal {
    __TBB_PERF_API void RegisterTest ( Test*, const char* testClassName, bool takeOwnership );
}

template<class T>
void RegisterTest() { internal::RegisterTest( new T, __TBB_PERF_TEST_CLASS_NAME(T), true ); }

template<class T>
void RegisterTest( T& t ) { internal::RegisterTest( &t, __TBB_PERF_TEST_CLASS_NAME(T), false ); }

enum SessionOptions {
    //! Use Test::RunSerial if present
    UseBaseline = 0x01,
    UseSerialBaseline = 0x02,
    UseBaselines = UseBaseline | UseSerialBaseline,
    UseTaskScheduler = 0x10,
    UseAffinityModes = 0x20,
    UseSmallestWorkloadOnly = 0x40
};

struct SessionSettings {
    //! A combination of SessionOptions flags
    uintptr_t my_opts;

    //! Name of a file to store performance results
    /** These results are duplicates of what is printed on the console. **/
    const char* my_resFile;

    //! Output destination for the histogram of individual run times in a series
    /** If it is a string, the histogram is stored in a file with such name. 
        If it is NULL, the histogram is printed on the console. By default histograms
        are suppressed.

        The histogram is formatted as two column table: 
        "time bucket start" "number of tests in this bucket"
        
        When this setting enables histogram generation, an individual test 
        can override it by implementing HistogramName method. **/
    const char* my_histogramName;

    SessionSettings ( uintptr_t opts = 0, const char* resFile = NULL, const char* histogram = NoHistogram )
        : my_opts(opts)
        , my_resFile(resFile)
        , my_histogramName(histogram)
    {}
}; // struct SessionSettings

//! Benchmarking session entry point
/** Executes all the individual tests registered previously by means of 
    RegisterTest<MycrotestImpl> **/
__TBB_PERF_API int TestMain( int argc, char* argv[],
                             const SessionSettings* defaultSettings = NULL );


} // namespace Perf

#endif /* __tbb_perf_h__ */


