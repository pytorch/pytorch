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

// We want to test waiting for workers feature with non-preview binaries. However,
// we want to have some testing of task_scheduler_init without this macro.
#if !__TBB_CPF_BUILD
#define TBB_PREVIEW_WAITING_FOR_WORKERS 1
#endif

#include "tbb/task_scheduler_init.h"
#include <cstdlib>
#include <cstdio>
#if TBB_USE_EXCEPTIONS
#include <stdexcept>
#endif

#include "harness_assert.h"
#if _MSC_VER
#pragma warning (push)
    // MSVC discovers that ASSERT(false) inside TestBlockingTerminateNS::ExceptionTest2::Body makes the code
    // in parallel_for after the body call unreachable. So supress the warning.
#pragma warning (disable: 4702)
#endif
#include "tbb/parallel_for.h"
#if _MSC_VER
#pragma warning (pop)
#endif

#include "harness_concurrency_tracker.h"
#include "harness_task.h"
#include "harness.h"

const int DefaultThreads = tbb::task_scheduler_init::default_num_threads();

namespace tbb { namespace internal {
size_t __TBB_EXPORTED_FUNC get_initial_auto_partitioner_divisor();
}}

int ArenaConcurrency() {
    return int(tbb::internal::get_initial_auto_partitioner_divisor()/4); // TODO: expose through task_arena interface?
}

// Generally, TBB does not guarantee mandatory parallelism. This test uses some whitebox knowledge about when all the threads can be available
bool test_mandatory_parallelism = true;

//! Test that task::initialize and task::terminate work when doing nothing else.
/** maxthread is treated as the "maximum" number of worker threads. */
void InitializeAndTerminate( int maxthread ) {
    __TBB_TRY {
        for( int i=0; i<256; ++i ) {
            int threads = (std::rand() % maxthread) + 1;
            switch( i&3 ) {
                default: {
                    tbb::task_scheduler_init init( threads );
                    ASSERT(init.is_active(), NULL);
                    ASSERT(ArenaConcurrency()==(threads==1?2:threads), NULL);
                    if (test_mandatory_parallelism)
                        Harness::ExactConcurrencyLevel::check(threads, Harness::ExactConcurrencyLevel::Serialize);
                    if(i&0x20) tbb::task::enqueue( (*new( tbb::task::allocate_root() ) TaskGenerator(2,6)) ); // a work deferred to workers
                    break;
                }
                case 0: {
                    tbb::task_scheduler_init init;
                    ASSERT(init.is_active(), NULL);
                    ASSERT(ArenaConcurrency()==(DefaultThreads==1?2:init.default_num_threads()), NULL);
                    if (test_mandatory_parallelism)
                        Harness::ExactConcurrencyLevel::check(init.default_num_threads(), Harness::ExactConcurrencyLevel::Serialize);
                    if(i&0x40) tbb::task::enqueue( (*new( tbb::task::allocate_root() ) TaskGenerator(3,5)) ); // a work deferred to workers
                    break;
                }
                case 1: {
                    tbb::task_scheduler_init init( tbb::task_scheduler_init::deferred );
                    ASSERT(!init.is_active(), "init should not be active; initialization was deferred");
                    init.initialize( threads );
                    ASSERT(init.is_active(), NULL);
                    ASSERT(ArenaConcurrency()==(threads==1?2:threads), NULL);
                    if (test_mandatory_parallelism)
                        Harness::ExactConcurrencyLevel::check(threads, Harness::ExactConcurrencyLevel::Serialize);
                    init.terminate();
                    ASSERT(!init.is_active(), "init should not be active; it was terminated");
                    break;
                }
                case 2: {
                    tbb::task_scheduler_init init( tbb::task_scheduler_init::automatic );
                    ASSERT(init.is_active(), NULL);
                    ASSERT(ArenaConcurrency()==(DefaultThreads==1?2:init.default_num_threads()), NULL);
                    if (test_mandatory_parallelism)
                        Harness::ExactConcurrencyLevel::check(init.default_num_threads(), Harness::ExactConcurrencyLevel::Serialize);
                    break;
                }
            }
        }
    } __TBB_CATCH( std::runtime_error& error ) {
#if TBB_USE_EXCEPTIONS
        REPORT("ERROR: %s\n", error.what() );
#endif /* TBB_USE_EXCEPTIONS */
    }
}

#if _WIN64
namespace std {      // 64-bit Windows compilers have not caught up with 1998 ISO C++ standard
    using ::srand;
}
#endif /* _WIN64 */

struct ThreadedInit {
    void operator()( int ) const {
        InitializeAndTerminate(MaxThread);
    }
};

#if _MSC_VER
#include "tbb/machine/windows_api.h"
#include <tchar.h>
#endif /* _MSC_VER */

/** The test will fail in particular if task_scheduler_init mistakenly hooks up
    auto-initialization mechanism. **/
void AssertExplicitInitIsNotSupplanted () {
    tbb::task_scheduler_init init(1);

    Harness::ExactConcurrencyLevel::check(1);
}

struct TestNoWorkerSurplusRun {
    void operator() (int) const {
        const unsigned THREADS = tbb::tbb_thread::hardware_concurrency()*2/3;
        for (int j=0; j<10; j++) {
            tbb::task_scheduler_init t(THREADS);
            Harness::ExactConcurrencyLevel::Combinable unique;

            for (int i=0; i<50; i++)
                Harness::ExactConcurrencyLevel::checkLessOrEqual(THREADS, &unique);
        }
    }
};

void TestNoWorkerSurplus () {
    // Run the test in a special thread because otherwise the surplus issue
    // is not observed for some hardware configurations
    NativeParallelFor( 1, TestNoWorkerSurplusRun() );
}

#if TBB_PREVIEW_WAITING_FOR_WORKERS
#include "tbb/task_group.h"
#include "tbb/task_arena.h"

namespace TestBlockingTerminateNS {
    struct EmptyBody {
        void operator()() const {}
        void operator()( int ) const {}
    };

    struct TestAutoInitBody {
        void operator()( int ) const {
            tbb::parallel_for( 0, 100, EmptyBody() );
        }
    };

    static tbb::atomic<int> gSeed;
    static tbb::atomic<int> gNumSuccesses;

    class TestMultpleWaitBody {
        bool myAutoInit;
    public:
        TestMultpleWaitBody( bool autoInit = false ) : myAutoInit( autoInit ) {}
        void operator()( int ) const {
            tbb::task_scheduler_init init( tbb::task_scheduler_init::deferred );
            if ( !myAutoInit )
                init.initialize( tbb::task_scheduler_init::automatic );
            Harness::FastRandom rnd( ++gSeed );
            // In case of auto init sub-tests we skip
            //  - case #4 to avoid recursion
            //  - case #5 because it is explicit initialization
            const int numCases = myAutoInit ? 4 : 6;
            switch ( rnd.get() % numCases ) {
            case 0: {
                tbb::task_arena a;
                a.enqueue( EmptyBody() );
                break;
            }
            case 1: {
                tbb::task_group tg;
                tg.run( EmptyBody() );
                tg.wait();
                break;
            }
            case 2:
                tbb::parallel_for( 0, 100, EmptyBody() );
                break;
            case 3:
                /* do nothing */
                break;
            case 4:
                // Create and join several threads with auto initialized scheduler.
                NativeParallelFor( rnd.get() % 5 + 1, TestMultpleWaitBody( true ) );
                break;
            case 5:
                {
                    tbb::task_scheduler_init init2;
                    bool res = init2.blocking_terminate( std::nothrow );
                    ASSERT( !res, NULL );
                }
                break;
            }
            if ( !myAutoInit && init.blocking_terminate( std::nothrow ) )
                ++gNumSuccesses;
        }
    };

    void TestMultpleWait() {
        const int minThreads = 1;
        const int maxThreads = 16;
        const int numRepeats = 5;
        // Initialize seed with different values on different machines.
        gSeed = tbb::task_scheduler_init::default_num_threads();
        for ( int repeats = 0; repeats<numRepeats; ++repeats ) {
            for ( int threads = minThreads; threads<maxThreads; ++threads ) {
                gNumSuccesses = 0;
                NativeParallelFor( threads, TestMultpleWaitBody() );
                ASSERT( gNumSuccesses > 0, "At least one blocking terminate must return 'true'" );
            }
        }
    }

#if TBB_USE_EXCEPTIONS
    template <typename F>
    void TestException( F &f ) {
        Harness::suppress_unused_warning( f );
        bool caught = false;
        try {
            f();
            ASSERT( false, NULL );
        }
        catch ( const std::runtime_error& ) {
            caught = true;
        }
#if TBB_USE_CAPTURED_EXCEPTION
        catch ( const tbb::captured_exception& ) {
            caught = true;
        }
#endif
        catch ( ... ) {
            ASSERT( false, NULL );
        }
        ASSERT( caught, NULL );
    }

    class ExceptionTest1 {
        tbb::task_scheduler_init tsi1;
        int myIndex;
    public:
        ExceptionTest1( int index ) : myIndex( index ) {}

        void operator()() {
            tbb::task_scheduler_init tsi2;
            (myIndex == 0 ? tsi1 : tsi2).blocking_terminate();
            ASSERT( false, "Blocking terminate did not throw the exception" );
        }
    };

    struct ExceptionTest2 {
        class Body {
            Harness::SpinBarrier& myBarrier;
        public:
            Body( Harness::SpinBarrier& barrier ) : myBarrier( barrier ) {}
            void operator()( int ) const {
                myBarrier.wait();
                tbb::task_scheduler_init init;
                init.blocking_terminate();
                ASSERT( false, "Blocking terminate did not throw the exception inside the parallel region" );
            }
        };
        void operator()() {
            const int numThreads = 4;
            tbb::task_scheduler_init init( numThreads );
            Harness::SpinBarrier barrier( numThreads );
            tbb::parallel_for( 0, numThreads, Body( barrier ) );
            ASSERT( false, "Parallel loop did not throw the exception" );
        }
    };
#endif /* TBB_USE_EXCEPTIONS */

    void TestExceptions() {
        for ( int i = 0; i<2; ++i ) {
            tbb::task_scheduler_init tsi[2];
            bool res1 = tsi[i].blocking_terminate( std::nothrow );
            ASSERT( !res1, NULL );
            bool res2 = tsi[1-i].blocking_terminate( std::nothrow );
            ASSERT( res2, NULL );
        }
#if TBB_USE_EXCEPTIONS
        ExceptionTest1 Test1(0), Test2(1);
        TestException( Test1 );
        TestException( Test2 );
        ExceptionTest2 Test3;
        TestException( Test3 );
#endif
    }
}

void TestBlockingTerminate() {
    TestBlockingTerminateNS::TestExceptions();
    TestBlockingTerminateNS::TestMultpleWait();
}
#endif /* TBB_PREVIEW_WAITING_FOR_WORKERS */

int TestMain () {
    // Do not use tbb::task_scheduler_init directly in the scope of main's body,
    // as a static variable, or as a member of a static variable.
#if _MSC_VER && !__TBB_NO_IMPLICIT_LINKAGE && !defined(__TBB_LIB_NAME)
    #ifdef _DEBUG
        ASSERT(!GetModuleHandle(_T("tbb.dll")) && GetModuleHandle(_T("tbb_debug.dll")),
            "test linked with wrong (non-debug) tbb library");
    #else
        ASSERT(!GetModuleHandle(_T("tbb_debug.dll")) && GetModuleHandle(_T("tbb.dll")),
            "test linked with wrong (debug) tbb library");
    #endif
#endif /* _MSC_VER && !__TBB_NO_IMPLICIT_LINKAGE && !__TBB_LIB_NAME */
    std::srand(2);
    REMARK("testing master thread\n");
    int threads = DefaultThreads*2;
    {   // work-around shared RML
        tbb::task_scheduler_init init( threads );
        if( !Harness::ExactConcurrencyLevel::isEqual( threads ) ) {
            threads = DefaultThreads;
            if( MaxThread > DefaultThreads )
                MaxThread = DefaultThreads;
#if RML_USE_WCRM
            REPORT("Known issue: shared RML for ConcRT does not support oversubscription\n");
            test_mandatory_parallelism = false; // we cannot rely on ConcRT to provide all the requested threads
#else
            REPORT("Known issue: machine is heavy loaded or shared RML which does not support oversubscription is loaded\n");
#endif
        }
    }
    InitializeAndTerminate( threads ); // test initialization of more than default number of threads
    for( int p=MinThread; p<=MaxThread; ++p ) {
        REMARK("testing with %d threads\n", p );
        // protect market with excess threads from default initializations
        // TODO IDEA: enhance task_scheduler_init to serve as global_control setting so that
        // number of threads > default concurrency will be requested from market.
        // Such settings must be aggregated via 'max' function and 'max_allowed_parallelism' control
        // (which has 'min' aggregation) will have precedence over it.
        tbb::task_scheduler_init init( tbb::task_scheduler_init::deferred );
        if( MaxThread > DefaultThreads ) init.initialize( MaxThread );
        NativeParallelFor( p, ThreadedInit() );
    }
    AssertExplicitInitIsNotSupplanted();
#if TBB_PREVIEW_WAITING_FOR_WORKERS
    TestBlockingTerminate();
#endif
    return Harness::Done;
}
