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

// Declarations for rock-bottom simple test harness.
// Just include this file to use it.
// Every test is presumed to have a command line of the form "test [-v] [MinThreads[:MaxThreads]]"
// The default for MinThreads is 1, for MaxThreads 4.
// The defaults can be overridden by defining macros HARNESS_DEFAULT_MIN_THREADS
// and HARNESS_DEFAULT_MAX_THREADS before including harness.h

#ifndef tbb_tests_harness_H
#define tbb_tests_harness_H

#include "tbb/tbb_config.h"
#include "harness_defs.h"

namespace Harness {
    enum TestResult {
        Done,
        Skipped,
        Unknown
    };
}

//! Entry point to a TBB unit test application
/** It MUST be defined by the test application.

    If HARNESS_NO_PARSE_COMMAND_LINE macro was not explicitly set before including harness.h,
    then global variables MinThread, and MaxThread will be available and
    initialized when it is called.

    Returns Harness::Done when the tests passed successfully. When the test fail, it must
    not return, calling exit(errcode) or abort() instead. When the test is not supported
    for the given platform/compiler/etc, it should return Harness::Skipped.

    To provide non-standard variant of main() for the test, define HARNESS_CUSTOM_MAIN
    before including harness.h **/
int TestMain ();

#if __SUNPRO_CC
    #include <stdlib.h>
    #include <string.h>
    #include <ucontext.h>
#else /* !__SUNPRO_CC */
    #include <cstdlib>
    #include <cstring>
#endif /* !__SUNPRO_CC */

#include <new>

#if __TBB_MIC_NATIVE
    #include "harness_mic.h"
#else
    #define HARNESS_EXPORT
    #define REPORT_FATAL_ERROR REPORT
#endif /* !__MIC__ */

#if _WIN32||_WIN64
    #include "tbb/machine/windows_api.h"
    #if _WIN32_WINNT > 0x0501 && _MSC_VER && !_M_ARM
        // Suppress "typedef ignored ... when no variable is declared" warning by vc14
        #pragma warning (push)
        #pragma warning (disable: 4091)
        #include <dbghelp.h>
        #pragma warning (pop)
        #pragma comment (lib, "dbghelp.lib")
    #endif
    #if __TBB_WIN8UI_SUPPORT
        #include <thread>
    #endif
    #if _MSC_VER
        #include <crtdbg.h>
    #endif
    #include <process.h>
#else
    #include <pthread.h>
#endif

#if __linux__
    #include <sys/utsname.h> /* for uname */
    #include <errno.h>       /* for use in LinuxKernelVersion() */
    #include <features.h>
#endif
// at least GLIBC 2.1 or OSX 10.5
#if __GLIBC__>2 || ( __GLIBC__==2 && __GLIBC_MINOR__ >= 1) || __APPLE__
    #include <execinfo.h> /*backtrace*/
    #define BACKTRACE_FUNCTION_AVAILABLE 1
#endif

namespace Harness {
    class NativeMutex {
#if _WIN32||_WIN64
        CRITICAL_SECTION my_critical_section;
      public:
        NativeMutex() {
            InitializeCriticalSectionEx(&my_critical_section, 4000, 0);
        }
        void lock() {
            EnterCriticalSection(&my_critical_section);
        }
        void unlock() {
            LeaveCriticalSection(&my_critical_section);
        }
        ~NativeMutex() {
            DeleteCriticalSection(&my_critical_section);
        }
#else
        pthread_mutex_t m_mutex;
    public:
        NativeMutex() {
             pthread_mutex_init(&m_mutex, NULL);
        }
        void lock() {
            pthread_mutex_lock(&m_mutex);
        }
        void unlock() {
            pthread_mutex_unlock(&m_mutex);
        }
        ~NativeMutex() {
            pthread_mutex_destroy(&m_mutex);
        }
#endif
    };
    namespace internal {
        static NativeMutex print_stack_mutex;
    }
}

#include "harness_runtime_loader.h"
#include "harness_report.h"

//! Prints current call stack
void print_call_stack() {
    Harness::internal::print_stack_mutex.lock();
    fflush(stdout); fflush(stderr);
    #if BACKTRACE_FUNCTION_AVAILABLE
        const int sz = 100; // max number of frames to capture
        void *buff[sz];
        int n = backtrace(buff, sz);
        REPORT("Call stack info (%d):\n", n);
        backtrace_symbols_fd(buff, n, fileno(stdout));
    #elif __SUNPRO_CC
        REPORT("Call stack info:\n");
        printstack(fileno(stdout));
    #elif _WIN32_WINNT > 0x0501 && _MSC_VER>=1500 && !__TBB_WIN8UI_SUPPORT
        const int sz = 62; // XP limitation for number of frames
        void *buff[sz];
        int n = CaptureStackBackTrace(0, sz, buff, NULL);
        REPORT("Call stack info (%d):\n", n);
        static LONG once = 0;
        if( !InterlockedExchange(&once, 1) )
            SymInitialize(GetCurrentProcess(), NULL, TRUE);
        const int len = 255; // just some reasonable string buffer size
        union { SYMBOL_INFO sym; char pad[sizeof(SYMBOL_INFO)+len]; };
        sym.MaxNameLen = len;
        sym.SizeOfStruct = sizeof( SYMBOL_INFO );
        DWORD64 offset;
        for(int i = 1; i < n; i++) { // skip current frame
            if(!SymFromAddr( GetCurrentProcess(), DWORD64(buff[i]), &offset, &sym )) {
                sym.Address = ULONG64(buff[i]); offset = 0; sym.Name[0] = 0;
            }
            REPORT("[%d] %016I64X+%04I64X: %s\n", i, sym.Address, offset, sym.Name); //TODO: print module name
        }
    #endif /*BACKTRACE_FUNCTION_AVAILABLE*/
    Harness::internal::print_stack_mutex.unlock();
}

#if !HARNESS_NO_ASSERT
    #include <exception> //for set_terminate
    #include "harness_assert.h"
    #if TEST_USES_TBB
        #include <tbb/tbb_stddef.h> /*set_assertion_handler*/
    #endif

    struct InitReporter {
        void (*default_terminate_handler)() ;
        InitReporter(): default_terminate_handler(NULL) {
            #if TEST_USES_TBB
                #if TBB_USE_ASSERT
                    tbb::set_assertion_handler(ReportError);
                #endif
                ASSERT_WARNING(TBB_INTERFACE_VERSION <= tbb::TBB_runtime_interface_version(), "runtime version mismatch");
            #endif
            #if TBB_USE_EXCEPTIONS
                default_terminate_handler = std::set_terminate(handle_terminate);
            #endif
        }
        static void handle_terminate();
    };
    static InitReporter InitReportError;

    void InitReporter::handle_terminate(){
        REPORT("std::terminate called.\n");
        print_call_stack();
        if (InitReportError.default_terminate_handler){
            InitReportError.default_terminate_handler();
        }
    }

    typedef void (*test_error_extra_t)(void);
    static test_error_extra_t ErrorExtraCall;
    //! Set additional handler to process failed assertions
    void SetHarnessErrorProcessing( test_error_extra_t extra_call ) {
        ErrorExtraCall = extra_call;
    }

    //! Reports errors issued by failed assertions
    void ReportError( const char* filename, int line, const char* expression, const char * message ) {
        print_call_stack();
    #if __TBB_ICL_11_1_CODE_GEN_BROKEN
        printf("%s:%d, assertion %s: %s\n", filename, line, expression, message ? message : "failed" );
    #else
        REPORT_FATAL_ERROR("%s:%d, assertion %s: %s\n", filename, line, expression, message ? message : "failed" );
    #endif

        if( ErrorExtraCall )
            (*ErrorExtraCall)();
        fflush(stdout); fflush(stderr);
    #if HARNESS_TERMINATE_ON_ASSERT
        TerminateProcess(GetCurrentProcess(), 1);
    #elif HARNESS_EXIT_ON_ASSERT
        exit(1);
    #elif HARNESS_CONTINUE_ON_ASSERT
        // continue testing
    #elif _MSC_VER && _DEBUG
        // aligned with tbb_assert_impl.h behavior
        if(1 == _CrtDbgReport(_CRT_ASSERT, filename, line, NULL, "%s\r\n%s", expression, message?message:""))
            _CrtDbgBreak();
    #else
        abort();
    #endif /* HARNESS_EXIT_ON_ASSERT */
    }
    //! Reports warnings issued by failed warning assertions
    void ReportWarning( const char* filename, int line, const char* expression, const char * message ) {
        REPORT("Warning: %s:%d, assertion %s: %s\n", filename, line, expression, message ? message : "failed" );
    }

#else /* !HARNESS_NO_ASSERT */

    #define ASSERT(p,msg) (Harness::suppress_unused_warning(p), (void)0)
    #define ASSERT_WARNING(p,msg) (Harness::suppress_unused_warning(p), (void)0)

#endif /* !HARNESS_NO_ASSERT */

namespace Harness {
    //TODO: unify with utility::internal::array_length from examples common utilities
    template<typename T, size_t N>
    inline size_t array_length(const T(&)[N])
    {
       return N;
    }

    template<typename T, size_t N>
    inline T* end( T(& array)[N])
    {
       return array+ array_length(array) ;
    }

} //namespace Harness

#if TEST_USES_TBB
    #include "tbb/blocked_range.h"

    namespace Harness {
        template<typename T, size_t N>
        tbb::blocked_range<T*> make_blocked_range( T(& array)[N]){ return tbb::blocked_range<T*>(array, array + N);}
    }
#endif

#if !HARNESS_NO_PARSE_COMMAND_LINE

//! Controls level of commentary printed via printf-like REMARK() macro.
/** If true, makes the test print commentary.  If false, test should print "done" and nothing more. */
static bool Verbose;

#ifndef HARNESS_DEFAULT_MIN_THREADS
    #define HARNESS_DEFAULT_MIN_THREADS 1
#endif

//! Minimum number of threads
static int MinThread = HARNESS_DEFAULT_MIN_THREADS;

#ifndef HARNESS_DEFAULT_MAX_THREADS
    #define HARNESS_DEFAULT_MAX_THREADS 4
#endif

//! Maximum number of threads
static int MaxThread = HARNESS_DEFAULT_MAX_THREADS;

//! Parse command line of the form "name [-v] [MinThreads[:MaxThreads]]"
/** Sets Verbose, MinThread, and MaxThread accordingly.
    The nthread argument can be a single number or a range of the form m:n.
    A single number m is interpreted as if written m:m.
    The numbers must be non-negative.
    Clients often treat the value 0 as "run sequentially." */
static void ParseCommandLine( int argc, char* argv[] ) {
    if( !argc ) REPORT("Command line with 0 arguments\n");
    int i = 1;
    if( i<argc ) {
        if( strncmp( argv[i], "-v", 2 )==0 ) {
            Verbose = true;
            ++i;
        }
    }
    if( i<argc ) {
        char* endptr;
        MinThread = strtol( argv[i], &endptr, 0 );
        if( *endptr==':' )
            MaxThread = strtol( endptr+1, &endptr, 0 );
        else if( *endptr=='\0' )
            MaxThread = MinThread;
        if( *endptr!='\0' ) {
            REPORT_FATAL_ERROR("garbled nthread range\n");
            exit(1);
        }
        if( MinThread<0 ) {
            REPORT_FATAL_ERROR("nthread must be nonnegative\n");
            exit(1);
        }
        if( MaxThread<MinThread ) {
            REPORT_FATAL_ERROR("nthread range is backwards\n");
            exit(1);
        }
        ++i;
    }
#if __TBB_STDARGS_BROKEN
    if ( !argc )
        argc = 1;
    else {
        while ( i < argc && argv[i][0] == 0 )
            ++i;
    }
#endif /* __TBB_STDARGS_BROKEN */
    if( i!=argc ) {
        REPORT_FATAL_ERROR("Usage: %s [-v] [nthread|minthread:maxthread]\n", argv[0] );
        exit(1);
    }
}
#endif /* HARNESS_NO_PARSE_COMMAND_LINE */

#if !HARNESS_CUSTOM_MAIN

#if __TBB_MPI_INTEROP
#undef SEEK_SET
#undef SEEK_CUR
#undef SEEK_END
#include "mpi.h"
#endif

#if __TBB_MIC_OFFLOAD && __MIC__
extern "C" int COIProcessProxyFlush();
#endif

HARNESS_EXPORT
#if HARNESS_NO_PARSE_COMMAND_LINE
int main() {
#if __TBB_MPI_INTEROP
    MPI_Init(NULL,NULL);
#endif
#else
int main(int argc, char* argv[]) {
    ParseCommandLine( argc, argv );
#if __TBB_MPI_INTEROP
    MPI_Init(&argc,&argv);
#endif
#endif
#if HARNESS_SKIP_TEST
    REPORT( "skip\n" );
    return 0;
#else
#if __TBB_MPI_INTEROP
    // Simple TBB/MPI interoperability harness for most of tests
    // Worker processes send blocking messages to the master process about their rank and group size
    // Master process receives this info and print it in verbose mode
    int rank, size, myrank;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    if (myrank == 0) {
#if !HARNESS_NO_PARSE_COMMAND_LINE
        REMARK("Hello mpi world. I am %d of %d\n", myrank, size);
#endif
        for ( int i = 1; i < size; i++ ) {
            MPI_Recv (&rank, 1, MPI_INT, i, 1, MPI_COMM_WORLD, &status);
            MPI_Recv (&size, 1, MPI_INT, i, 1, MPI_COMM_WORLD, &status);
#if !HARNESS_NO_PARSE_COMMAND_LINE
            REMARK("Hello mpi world. I am %d of %d\n", rank, size);
#endif
        }
    } else {
        MPI_Send (&myrank, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        MPI_Send (&size, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
    }
#endif

    int res = Harness::Unknown;
#if __TBB_MIC_OFFLOAD
    // "mic:-1" or "mandatory" specifies execution on the target. The runtime
    // system chooses the specific target. Execution on the CPU is not allowed.
#if __INTEL_COMPILER < 1400
    #pragma offload target(mic:-1) out(res)
#else
    #pragma offload target(mic) out(res) mandatory
#endif
#endif
    {
        res = TestMain();
#if __TBB_MIC_OFFLOAD && __MIC__
        // It is recommended not to use the __MIC__ macro directly in the offload block but it is Ok here
        // since it is not lead to an unexpected difference between host and target compilation phases.
        // We need to flush internals COI buffers to order output from the offload part before the host part.
        // Also it is work-around for the issue with missed output.
        COIProcessProxyFlush();
#endif
    }

    ASSERT( res==Harness::Done || res==Harness::Skipped, "Wrong return code by TestMain");
#if __TBB_MPI_INTEROP
    if (myrank == 0) {
        REPORT( res==Harness::Done ? "done\n" : "skip\n" );
    }
    MPI_Finalize();
#else
    REPORT( res==Harness::Done ? "done\n" : "skip\n" );
#endif
    return 0;
#endif /* HARNESS_SKIP_TEST */
}

#endif /* !HARNESS_CUSTOM_MAIN */

//! Base class for prohibiting compiler-generated operator=
class NoAssign {
    //! Assignment not allowed
    void operator=( const NoAssign& );
public:
    NoAssign() {} // explicitly defined to prevent gratuitous warnings
};

//! Base class for prohibiting compiler-generated copy constructor or operator=
class NoCopy: NoAssign {
    //! Copy construction not allowed
    NoCopy( const NoCopy& );
public:
    NoCopy() {}
};

#if __TBB_CPP11_RVALUE_REF_PRESENT
#include <utility>

//! Base class for objects which support move ctors
class Movable {
public:
    Movable() : alive(true) {}
    void Reset() { alive = true; }
    Movable(Movable&& other) {
        ASSERT(other.alive, "Moving from a dead object");
        alive = true;
        other.alive = false;
    }
    Movable& operator=(Movable&& other) {
        ASSERT(alive, "Assignment to a dead object");
        ASSERT(other.alive, "Assignment of a dead object");
        other.alive = false;
        return *this;
    }
    Movable& operator=(const Movable& other) {
        ASSERT(alive, "Assignment to a dead object");
        ASSERT(other.alive, "Assignment of a dead object");
        return *this;
    }
    Movable(const Movable& other) {
        ASSERT(other.alive, "Const reference to a dead object");
        alive = true;
    }
    ~Movable() { alive = false; }
    volatile bool alive;
};

class MoveOnly : Movable, NoCopy {
public:
    MoveOnly() : Movable() {}
    MoveOnly(MoveOnly&& other) : Movable( std::move(other) ) {}
};
#endif /* __TBB_CPP11_RVALUE_REF_PRESENT */

#if HARNESS_TBBMALLOC_THREAD_SHUTDOWN && __TBB_SOURCE_DIRECTLY_INCLUDED && (_WIN32||_WIN64)
#include "../tbbmalloc/tbbmalloc_internal_api.h"
#endif

//! For internal use by template function NativeParallelFor
template<typename Index, typename Body>
class NativeParallelForTask: NoCopy {
public:
    NativeParallelForTask( Index index_, const Body& body_ ) :
        index(index_),
        body(body_)
    {}

    //! Start task
    void start() {
#if _WIN32||_WIN64
        unsigned thread_id;
#if __TBB_WIN8UI_SUPPORT
        std::thread* thread_tmp=new std::thread(thread_function, this);
        thread_handle = thread_tmp->native_handle();
        thread_id = 0;
#else
        unsigned stack_size = 0;
#if HARNESS_THREAD_STACK_SIZE
        stack_size = HARNESS_THREAD_STACK_SIZE;
#endif
        thread_handle = (HANDLE)_beginthreadex( NULL, stack_size, thread_function, this, 0, &thread_id );
#endif
        ASSERT( thread_handle!=0, "NativeParallelFor: _beginthreadex failed" );
#else
#if __ICC==1100
    #pragma warning (push)
    #pragma warning (disable: 2193)
#endif /* __ICC==1100 */
        // Some machines may have very large hard stack limit. When the test is
        // launched by make, the default stack size is set to the hard limit, and
        // calls to pthread_create fail with out-of-memory error.
        // Therefore we set the stack size explicitly (as for TBB worker threads).
#if !defined(HARNESS_THREAD_STACK_SIZE)
#if __i386__||__i386||__arm__
        const size_t stack_size = 1*MByte;
#elif __x86_64__
        const size_t stack_size = 2*MByte;
#else
        const size_t stack_size = 4*MByte;
#endif
#else
        const size_t stack_size = HARNESS_THREAD_STACK_SIZE;
#endif /* HARNESS_THREAD_STACK_SIZE */
        pthread_attr_t attr_stack;
        int status = pthread_attr_init(&attr_stack);
        ASSERT(0==status, "NativeParallelFor: pthread_attr_init failed");
        status = pthread_attr_setstacksize( &attr_stack, stack_size );
        ASSERT(0==status, "NativeParallelFor: pthread_attr_setstacksize failed");
        status = pthread_create(&thread_id, &attr_stack, thread_function, this);
        ASSERT(0==status, "NativeParallelFor: pthread_create failed");
        pthread_attr_destroy(&attr_stack);
#if __ICC==1100
    #pragma warning (pop)
#endif
#endif /* _WIN32||_WIN64 */
    }

    //! Wait for task to finish
    void wait_to_finish() {
#if _WIN32||_WIN64
        DWORD status = WaitForSingleObjectEx( thread_handle, INFINITE, FALSE );
        ASSERT( status!=WAIT_FAILED, "WaitForSingleObject failed" );
        CloseHandle( thread_handle );
#else
        int status = pthread_join( thread_id, NULL );
        ASSERT( !status, "pthread_join failed" );
#endif
#if HARNESS_NO_ASSERT
        (void)status;
#endif
    }

private:
#if _WIN32||_WIN64
    HANDLE thread_handle;
#else
    pthread_t thread_id;
#endif

    //! Range over which task will invoke the body.
    const Index index;

    //! Body to invoke over the range.
    const Body body;

#if _WIN32||_WIN64
    static unsigned __stdcall thread_function( void* object )
#else
    static void* thread_function(void* object)
#endif
    {
        NativeParallelForTask& self = *static_cast<NativeParallelForTask*>(object);
        (self.body)(self.index);
#if HARNESS_TBBMALLOC_THREAD_SHUTDOWN && __TBB_SOURCE_DIRECTLY_INCLUDED && (_WIN32||_WIN64)
        // in those cases can't release per-thread cache automatically,
        // so do it manually
        // TODO: investigate less-intrusive way to do it, for example via FLS keys
        __TBB_mallocThreadShutdownNotification();
#endif
        return 0;
    }
};

//! Execute body(i) in parallel for i in the interval [0,n).
/** Each iteration is performed by a separate thread. */
template<typename Index, typename Body>
void NativeParallelFor( Index n, const Body& body ) {
    typedef NativeParallelForTask<Index,Body> task;

    if( n>0 ) {
        // Allocate array to hold the tasks
        task* array = static_cast<task*>(operator new( n*sizeof(task) ));

        // Construct the tasks
        for( Index i=0; i!=n; ++i )
            new( &array[i] ) task(i,body);

        // Start the tasks
        for( Index i=0; i!=n; ++i )
            array[i].start();

        // Wait for the tasks to finish and destroy each one.
        for( Index i=n; i; --i ) {
            array[i-1].wait_to_finish();
            array[i-1].~task();
        }

        // Deallocate the task array
        operator delete(array);
    }
}

//! The function to zero-initialize arrays; useful to avoid warnings
template <typename T>
void zero_fill(void* array, size_t n) {
    memset(array, 0, sizeof(T)*n);
}

#if __SUNPRO_CC && defined(min)
#undef min
#undef max
#endif

#ifndef min
//! Utility template function returning lesser of the two values.
/** Provided here to avoid including not strict safe <algorithm>.\n
    In case operands cause signed/unsigned or size mismatch warnings it is caller's
    responsibility to do the appropriate cast before calling the function. **/
template<typename T1, typename T2>
T1 min ( const T1& val1, const T2& val2 ) {
    return val1 < val2 ? val1 : val2;
}
#endif /* !min */

#ifndef max
//! Utility template function returning greater of the two values.
/** Provided here to avoid including not strict safe <algorithm>.\n
    In case operands cause signed/unsigned or size mismatch warnings it is caller's
    responsibility to do the appropriate cast before calling the function. **/
template<typename T1, typename T2>
T1 max ( const T1& val1, const T2& val2 ) {
    return val1 < val2 ? val2 : val1;
}
#endif /* !max */

template<typename T>
static inline bool is_aligned(T arg, size_t alignment) {
    return 0==((size_t)arg &  (alignment-1));
}

#if __linux__
inline unsigned LinuxKernelVersion()
{
    unsigned digit1, digit2, digit3;
    struct utsname utsnameBuf;

    if (-1 == uname(&utsnameBuf)) {
        REPORT_FATAL_ERROR("Can't call uname: errno %d\n", errno);
        exit(1);
    }
    if (3 != sscanf(utsnameBuf.release, "%u.%u.%u", &digit1, &digit2, &digit3)) {
        REPORT_FATAL_ERROR("Unable to parse OS release '%s'\n", utsnameBuf.release);
        exit(1);
    }
    return 1000000*digit1+1000*digit2+digit3;
}
#endif

namespace Harness {

#if !HARNESS_NO_ASSERT
//! Base class that asserts that no operations are made with the object after its destruction.
class NoAfterlife {
protected:
    enum state_t {
        LIVE=0x56781234,
        DEAD=0xDEADBEEF
    } m_state;

public:
    NoAfterlife() : m_state(LIVE) {}
    NoAfterlife( const NoAfterlife& src ) : m_state(LIVE) {
        ASSERT( src.IsLive(), "Constructing from the dead source" );
    }
    ~NoAfterlife() {
        ASSERT( IsLive(), "Repeated destructor call" );
        m_state = DEAD;
    }
    const NoAfterlife& operator=( const NoAfterlife& src ) {
        ASSERT( IsLive(), NULL );
        ASSERT( src.IsLive(), NULL );
        return *this;
    }
    void AssertLive() const {
        ASSERT( IsLive(), "Already dead" );
    }
    bool IsLive() const {
        return m_state == LIVE;
    }
}; // NoAfterlife
#endif /* !HARNESS_NO_ASSERT */

#if _WIN32 || _WIN64
    void Sleep ( int ms ) {
#if !__TBB_WIN8UI_SUPPORT
        ::Sleep(ms);
#else
         std::chrono::milliseconds sleep_time( ms );
         std::this_thread::sleep_for( sleep_time );
#endif

    }

    typedef DWORD tid_t;
    tid_t CurrentTid () { return GetCurrentThreadId(); }

#else /* !WIN */

    void Sleep ( int ms ) {
        timespec  requested = { ms / 1000, (ms % 1000)*1000000 };
        timespec  remaining = { 0, 0 };
        nanosleep(&requested, &remaining);
    }

    typedef pthread_t tid_t;
    tid_t CurrentTid () { return pthread_self(); }
#endif /* !WIN */

    static const unsigned Primes[] = {
        0x9e3779b1, 0xffe6cc59, 0x2109f6dd, 0x43977ab5, 0xba5703f5, 0xb495a877, 0xe1626741, 0x79695e6b,
        0xbc98c09f, 0xd5bee2b3, 0x287488f9, 0x3af18231, 0x9677cd4d, 0xbe3a6929, 0xadc6a877, 0xdcf0674b,
        0xbe4d6fe9, 0x5f15e201, 0x99afc3fd, 0xf3f16801, 0xe222cfff, 0x24ba5fdb, 0x0620452d, 0x79f149e3,
        0xc8b93f49, 0x972702cd, 0xb07dd827, 0x6c97d5ed, 0x085a3d61, 0x46eb5ea7, 0x3d9910ed, 0x2e687b5b,
        0x29609227, 0x6eb081f1, 0x0954c4e1, 0x9d114db9, 0x542acfa9, 0xb3e6bd7b, 0x0742d917, 0xe9f3ffa7,
        0x54581edb, 0xf2480f45, 0x0bb9288f, 0xef1affc7, 0x85fa0ca7, 0x3ccc14db, 0xe6baf34b, 0x343377f7,
        0x5ca19031, 0xe6d9293b, 0xf0a9f391, 0x5d2e980b, 0xfc411073, 0xc3749363, 0xb892d829, 0x3549366b,
        0x629750ad, 0xb98294e5, 0x892d9483, 0xc235baf3, 0x3d2402a3, 0x6bdef3c9, 0xbec333cd, 0x40c9520f
    };

    class FastRandom {
        unsigned x, a;
    public:
        unsigned short get() {
            unsigned short r = (unsigned short)(x >> 16);
            x = x*a + 1;
            return r;
        }
        explicit FastRandom( unsigned seed ) {
            x = seed;
            a = Primes[seed % (sizeof(Primes) / sizeof(Primes[0]))];
        }
    };
    template<typename T>
    class FastRandomBody {
        FastRandom r;
    public:
        explicit FastRandomBody( unsigned seed ) : r(seed) {}
        // Depending on the input type T the result distribution formed from this operator()
        // might possess different characteristics than the original one used in FastRandom instance.
        T operator()() { return T(r.get()); }
    };

    int SetEnv( const char *envname, const char *envval ) {
        ASSERT( envname && envval, "Harness::SetEnv() requires two valid C strings" );
#if __TBB_WIN8UI_SUPPORT
        ASSERT( false, "Harness::SetEnv() should not be called in code built for win8ui" );
        return -1;
#elif !(_MSC_VER || __MINGW32__ || __MINGW64__)
        // On POSIX systems use setenv
        return setenv(envname, envval, /*overwrite=*/1);
#elif __STDC_SECURE_LIB__>=200411
        // this macro is set in VC & MinGW if secure API functions are present
        return _putenv_s(envname, envval);
#else
        // If no secure API on Windows, use _putenv
        size_t namelen = strlen(envname), valuelen = strlen(envval);
        char* buf = new char[namelen+valuelen+2];
        strncpy(buf, envname, namelen);
        buf[namelen] = '=';
        strncpy(buf+namelen+1, envval, valuelen);
        buf[namelen+1+valuelen] = char(0);
        int status = _putenv(buf);
        delete[] buf;
        return status;
#endif
    }

    char* GetEnv(const char *envname) {
        ASSERT(envname, "Harness::GetEnv() requires a valid C string");
#if __TBB_WIN8UI_SUPPORT
        return NULL;
#else
        return std::getenv(envname);
#endif
    }

    class DummyBody {
        int m_numIters;
    public:
        explicit DummyBody( int iters ) : m_numIters( iters ) {}
        void operator()( int ) const {
            for ( volatile int i = 0; i < m_numIters; ++i ) {}
        }
    };
} // namespace Harness

#endif /* tbb_tests_harness_H */
