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

#include "tbb/compat/condition_variable"
#include "tbb/mutex.h"
#include "tbb/recursive_mutex.h"
#include "tbb/tick_count.h"
#include "tbb/atomic.h"

#include <stdexcept>

#include "harness.h"

#if TBB_IMPLEMENT_CPP0X
// This test deliberately avoids a "using tbb" statement,
// so that the error of putting types in the wrong namespace will be caught.
using namespace std;
#else
using namespace tbb::interface5;
#endif

#if __TBB_CPP11_RVALUE_REF_PRESENT
template<typename M>
void TestUniqueLockMoveConstructorAndAssignOp(){
    typedef unique_lock<M> unique_lock_t;

    static const bool locked = true;
    static const bool unlocked = false;

    struct Locked{
        bool value;
        Locked(bool a_value) : value(a_value) {}
    };

    typedef Locked destination;
    typedef Locked source;

    struct MutexAndLockFixture{
        M mutex;
        unique_lock_t lock;
        const bool was_locked;

        MutexAndLockFixture(source lckd_src) : lock(mutex), was_locked(lckd_src.value){
            if (!lckd_src.value) lock.unlock();
            ASSERT(was_locked == lock.owns_lock(), "unlock did not release the mutex while should?");
        }
    };

    struct TestCases{
        const char* filename;
        int line;

        TestCases(const char* a_filename, int a_line) : filename(a_filename), line(a_line) {}

        void TestMoveConstructor(source locked_src){
            MutexAndLockFixture src(locked_src);
            unique_lock_t dst_lock(std::move(src.lock));
            AssertOwnershipWasTransfered(dst_lock, src.lock, src.was_locked, &src.mutex);
        }

        void TestMoveAssignment(source locked_src, destination locked_dest){
            MutexAndLockFixture src(locked_src);
            MutexAndLockFixture dst(locked_dest);

            dst.lock = std::move(src.lock);
            ASSERT_CUSTOM(unique_lock_t(dst.mutex, try_to_lock).owns_lock(), "unique_lock should release owned mutex on assignment", filename, line);
            AssertOwnershipWasTransfered(dst.lock, src.lock, src.was_locked, &src.mutex);
        }

        void AssertOwnershipWasTransfered(unique_lock_t const& dest_lock, unique_lock_t const& src_lck, const bool was_locked, const M* mutex) {
            ASSERT_CUSTOM(dest_lock.owns_lock() == was_locked, "moved to lock object should have the same state as source before move", filename, line);
            ASSERT_CUSTOM(dest_lock.mutex() == mutex, "moved to lock object should have the same state as source before move", filename, line);
            ASSERT_CUSTOM(src_lck.owns_lock() == false, "moved from lock object must not left locked", filename, line);
            ASSERT_CUSTOM(src_lck.mutex() == NULL, "moved from lock object must not has mutex", filename, line);
        }
    };
//TODO: to rework this with an assertion binder
#define AT_LOCATION() TestCases( __FILE__, __LINE__) \

        AT_LOCATION().TestMoveConstructor(source(locked));
        AT_LOCATION().TestMoveAssignment (source(locked), destination(locked));
        AT_LOCATION().TestMoveAssignment (source(locked), destination(unlocked));
        AT_LOCATION().TestMoveConstructor(source(unlocked));
        AT_LOCATION().TestMoveAssignment (source(unlocked), destination(locked));
        AT_LOCATION().TestMoveAssignment (source(unlocked), destination(unlocked));

#undef AT_LOCATION

}
#endif //__TBB_CPP11_RVALUE_REF_PRESENT

template<typename M>
struct Counter {
    typedef M mutex_type;
    M mutex;
    volatile long value;
    void flog_once_lock_guard( size_t mode );
    void flog_once_unique_lock( size_t mode );
};

template<typename M>
void Counter<M>::flog_once_lock_guard(size_t mode)
/** Increments counter once for each iteration in the iteration space. */
{
    if( mode&1 ) {
        // Try acquire and release with implicit lock_guard
        // precondition: if mutex_type is not a recursive mutex, the calling thread does not own the mutex m.
        // if the precondition is not met, either dead-lock incorrect 'value' would result in.
        lock_guard<M> lg(mutex);
        value = value+1;
    } else {
        // Try acquire and release with adopt lock_quard
        // precodition: the calling thread owns the mutex m.
        // if the precondition is not met, incorrect 'value' would result in because the thread unlocks
        // mutex that it does not own.
        mutex.lock();
        lock_guard<M> lg( mutex, adopt_lock );
        value = value+1;
    }
}

template<typename M>
void Counter<M>::flog_once_unique_lock(size_t mode)
/** Increments counter once for each iteration in the iteration space. */
{
    switch( mode&7 ) {
    case 0:
        {// implicitly acquire and release mutex with unique_lock
          unique_lock<M> ul( mutex );
          value = value+1;
          ASSERT( ul==true, NULL );
        }
        break;
    case 1:
        {// unique_lock with defer_lock
          unique_lock<M> ul( mutex, defer_lock );
          ASSERT( ul.owns_lock()==false, NULL );
          ul.lock();
          value = value+1;
          ASSERT( ul.owns_lock()==true, NULL );
        }
        break;
    case 2:
        {// unique_lock::try_lock() with try_to_lock
          unique_lock<M> ul( mutex, try_to_lock );
          if( !ul )
              while( !ul.try_lock() )
                  __TBB_Yield();
          value = value+1;
        }
        break;
    case 3:
        {// unique_lock::try_lock_for() with try_to_lock
          unique_lock<M> ul( mutex, defer_lock );
          tbb::tick_count::interval_t i(1.0);
          while( !ul.try_lock_for( i ) )
              ;
          value = value+1;
          ASSERT( ul.owns_lock()==true, NULL );
        }
        break;
    case 4:
        {
          unique_lock<M> ul_o4;
          {// unique_lock with adopt_lock
            mutex.lock();
            unique_lock<M> ul( mutex, adopt_lock );
            value = value+1;
            ASSERT( ul.owns_lock()==true, NULL );
            ASSERT( ul.mutex()==&mutex, NULL );
            ASSERT( ul_o4.owns_lock()==false, NULL );
            ASSERT( ul_o4.mutex()==NULL, NULL );
            swap( ul, ul_o4 );
            ASSERT( ul.owns_lock()==false, NULL );
            ASSERT( ul.mutex()==NULL, NULL );
            ASSERT( ul_o4.owns_lock()==true, NULL );
            ASSERT( ul_o4.mutex()==&mutex, NULL );
            ul_o4.unlock();
          }
          ASSERT( ul_o4.owns_lock()==false, NULL );
        }
        break;
    case 5:
        {
          unique_lock<M> ul_o5;
          {// unique_lock with adopt_lock
            mutex.lock();
            unique_lock<M> ul( mutex, adopt_lock );
            value = value+1;
            ASSERT( ul.owns_lock()==true, NULL );
            ASSERT( ul.mutex()==&mutex, NULL );
            ASSERT( ul_o5.owns_lock()==false, NULL );
            ASSERT( ul_o5.mutex()==NULL, NULL );
            ul_o5.swap( ul );
            ASSERT( ul.owns_lock()==false, NULL );
            ASSERT( ul.mutex()==NULL, NULL );
            ASSERT( ul_o5.owns_lock()==true, NULL );
            ASSERT( ul_o5.mutex()==&mutex, NULL );
            ul_o5.unlock();
          }
          ASSERT( ul_o5.owns_lock()==false, NULL );
        }
        break;
    default:
        {// unique_lock with adopt_lock, and release()
          mutex.lock();
          unique_lock<M> ul( mutex, adopt_lock );
          ASSERT( ul==true, NULL );
          value = value+1;
          M* old_m = ul.release();
          old_m->unlock();
          ASSERT( ul.owns_lock()==false, NULL );
        }
        break;
    }
}

static tbb::atomic<size_t> Order;

template<typename State, long TestSize>
struct WorkForLocks: NoAssign {
    static const size_t chunk = 100;
    State& state;
    WorkForLocks( State& state_ ) : state(state_) {}
    void operator()( int ) const {
        size_t step;
        while( (step=Order.fetch_and_add<tbb::acquire>(chunk))<TestSize ) {
            for( size_t i=0; i<chunk && step<TestSize; ++i, ++step ) {
                state.flog_once_lock_guard(step);
                state.flog_once_unique_lock(step);
            }
        }
    }
};

template<typename M>
void TestLocks( const char* name, int nthread ) {
    REMARK("testing %s in TestLocks\n",name);
    Counter<M> counter;
    counter.value = 0;
    Order = 0;
    // use the macro because of a gcc 4.6 bug
#define TEST_SIZE 100000
    NativeParallelFor( nthread, WorkForLocks<Counter<M>, TEST_SIZE>(counter) );

    if( counter.value!=2*TEST_SIZE )
        REPORT("ERROR for %s in TestLocks: counter.value=%ld != 2 * %ld=test_size\n",name,counter.value,TEST_SIZE);
#undef TEST_SIZE
}

static tbb::atomic<int> barrier;

// Test if the constructor works and if native_handle() works
template<typename M>
struct WorkForCondVarCtor: NoAssign {
    condition_variable& my_cv;
    M& my_mtx;
    WorkForCondVarCtor( condition_variable& cv_, M& mtx_ ) : my_cv(cv_), my_mtx(mtx_) {}
    void operator()( int tid ) const {
        ASSERT( tid<=1, NULL ); // test with 2 threads.
        condition_variable::native_handle_type handle = my_cv.native_handle();
        if( tid&1 ) {
            my_mtx.lock();
            ++barrier;
#if _WIN32||_WIN64
            if( !tbb::interface5::internal::internal_condition_variable_wait( *handle, &my_mtx ) ) {
                int ec = GetLastError();
                ASSERT( ec!=WAIT_TIMEOUT, NULL );
                throw_exception( tbb::internal::eid_condvar_wait_failed );
            }
#else
            if( pthread_cond_wait( handle, my_mtx.native_handle() ) )
                throw_exception( tbb::internal::eid_condvar_wait_failed );
#endif
            ++barrier;
            my_mtx.unlock();
        } else {
            bool res;
            while( (res=my_mtx.try_lock())==true && barrier==0 ) {
                my_mtx.unlock();
                __TBB_Yield();
            }
            if( res ) my_mtx.unlock();
            do {
#if _WIN32||_WIN64
                tbb::interface5::internal::internal_condition_variable_notify_one( *handle );
#else
                pthread_cond_signal( handle );
#endif
                __TBB_Yield();
            } while ( barrier<2 );
        }
    }
};

static condition_variable* test_cv;
static tbb::atomic<int> n_waiters;

// Test if the destructor works
template<typename M>
struct WorkForCondVarDtor: NoAssign {
    int nthread;
    M& my_mtx;
    WorkForCondVarDtor( int n, M& mtx_ ) : nthread(n), my_mtx(mtx_) {}
    void operator()( int tid ) const {
        if( tid==0 ) {
            unique_lock<M> ul( my_mtx, defer_lock );
            test_cv = new condition_variable;

            while( n_waiters<nthread-1 )
                __TBB_Yield();
            ul.lock();
            test_cv->notify_all();
            ul.unlock();
            while( n_waiters>0 )
                __TBB_Yield();
            delete test_cv;
        } else {
            while( test_cv==NULL )
                __TBB_Yield();
            unique_lock<M> ul(my_mtx);
            ++n_waiters;
            test_cv->wait( ul );
            --n_waiters;
        }
    }
};

static const int max_ticket  = 100;
static const int short_delay = 10;
static const int long_delay  = 100;

tbb::atomic<int> n_signaled;
tbb::atomic<int> n_done, n_done_1, n_done_2;
tbb::atomic<int> n_timed_out;

static bool false_to_true;

struct TestPredicateFalseToTrue {
    TestPredicateFalseToTrue() {}
    bool operator()() { return false_to_true; }
};

struct TestPredicateFalse {
    TestPredicateFalse() {}
    bool operator()() { return false; }
};

struct TestPredicateTrue {
    TestPredicateTrue() {}
    bool operator()() { return true; }
};

// Test timed wait and timed wait with pred
template<typename M>
struct WorkForCondVarTimedWait: NoAssign {
    int nthread;
    condition_variable& test_cv;
    M& my_mtx;
    WorkForCondVarTimedWait( int n_, condition_variable& cv_, M& mtx_ ) : nthread(n_), test_cv(cv_), my_mtx(mtx_) {}
    void operator()( int tid ) const {
        tbb::tick_count t1, t2;

        unique_lock<M> ul( my_mtx, defer_lock );

        ASSERT( n_timed_out==0, NULL );
        ++barrier;
        while( barrier<nthread ) __TBB_Yield();

        // test if a thread times out with wait_for()
        for( int i=1; i<10; ++i ) {
            tbb::tick_count::interval_t intv((double)i*0.0999 /*seconds*/);
            ul.lock();
            cv_status st = no_timeout;
            __TBB_TRY {
                /** Some version of glibc return EINVAL instead 0 when spurious wakeup occurs on pthread_cond_timedwait() **/
                st = test_cv.wait_for( ul, intv );
            } __TBB_CATCH( std::runtime_error& ) {}
            ASSERT( ul, "mutex should have been reacquired" );
            ul.unlock();
            if( st==timeout )
                ++n_timed_out;
        }

        ASSERT( n_timed_out>0, "should have been timed-out at least once\n" );
        ++n_done_1;
        while( n_done_1<nthread ) __TBB_Yield();

        for( int i=1; i<10; ++i ) {
            tbb::tick_count::interval_t intv((double)i*0.0001 /*seconds*/);
            ul.lock();
            __TBB_TRY {
                /** Some version of glibc return EINVAL instead 0 when spurious wakeup occurs on pthread_cond_timedwait() **/
                ASSERT( false==test_cv.wait_for( ul, intv, TestPredicateFalse()), "incorrect return value" );
            } __TBB_CATCH( std::runtime_error& ) {}
            ASSERT( ul, "mutex should have been reacquired" );
            ul.unlock();
        }

        if( tid==0 )
            n_waiters = 0;
        // barrier
        ++n_done_2;
        while( n_done_2<nthread ) __TBB_Yield();

        // at this point, we know wait_for() successfully times out.
        // so test if a thread blocked on wait_for() could receive a signal before its waiting time elapses.
        if( tid==0 ) {
            // signaler
            n_signaled = 0;
            ASSERT( n_waiters==0, NULL );
            ++n_done_2; // open gate 1

            while( n_waiters<(nthread-1) ) __TBB_Yield(); // wait until all other threads block on cv. flag_1

            ul.lock();
            test_cv.notify_all();
            n_waiters = 0;
            ul.unlock();

            while( n_done_2<2*nthread ) __TBB_Yield();
            ASSERT( n_signaled>0, "too small an interval?" );
            n_signaled = 0;

        } else {
            while( n_done_2<nthread+1 ) __TBB_Yield(); // gate 1

            // sleeper
            tbb::tick_count::interval_t intv((double)2.0 /*seconds*/);
            ul.lock();
            ++n_waiters; // raise flag 1/(nthread-1)
            t1 = tbb::tick_count::now();
            cv_status st = test_cv.wait_for( ul, intv ); // gate 2
            t2 = tbb::tick_count::now();
            ul.unlock();
            if( st==no_timeout ) {
                ++n_signaled;
                ASSERT( (t2-t1).seconds()<intv.seconds(), "got a signal after timed-out?" );
            }
        }

        ASSERT( n_done==0, NULL );
        ++n_done_2;

        if( tid==0 ) {
            ASSERT( n_waiters==0, NULL );
            ++n_done; // open gate 3

            while( n_waiters<(nthread-1) ) __TBB_Yield(); // wait until all other threads block on cv.
            for( int i=0; i<2*short_delay; ++i ) __TBB_Yield();  // give some time to waiters so that all of them in the waitq
            ul.lock();
            false_to_true = true;
            test_cv.notify_all(); // open gate 4
            ul.unlock();

            while( n_done<nthread ) __TBB_Yield(); // wait until all other threads wake up.
            ASSERT( n_signaled>0, "too small an interval?" );
        } else {

            while( n_done<1 ) __TBB_Yield(); // gate 3

            tbb::tick_count::interval_t intv((double)2.0 /*seconds*/);
            ul.lock();
            ++n_waiters;
            // wait_for w/ predciate
            t1 = tbb::tick_count::now();
            ASSERT( test_cv.wait_for( ul, intv, TestPredicateFalseToTrue())==true, NULL ); // gate 4
            t2 = tbb::tick_count::now();
            ul.unlock();
            if( (t2-t1).seconds()<intv.seconds() )
                ++n_signaled;
            ++n_done;
        }
    }
};

tbb::atomic<int> ticket_for_sleep, ticket_for_wakeup, signaled_ticket, wokeup_ticket;
tbb::atomic<unsigned> n_visit_to_waitq;
unsigned max_waitq_length;

template<typename M>
struct WorkForCondVarWaitAndNotifyOne: NoAssign {
    int nthread;
    condition_variable& test_cv;
    M& my_mtx;
    WorkForCondVarWaitAndNotifyOne( int n_, condition_variable& cv_, M& mtx_ ) : nthread(n_), test_cv(cv_), my_mtx(mtx_) {}
    void operator()( int tid ) const {
        if( tid&1 ) {
            // exercise signal part
            while( ticket_for_wakeup<max_ticket ) {
                int my_ticket = ++ticket_for_wakeup; // atomically grab the next ticket
                if( my_ticket>max_ticket )
                    break;

                for( ;; ) {
                    unique_lock<M> ul( my_mtx, defer_lock );
                    ul.lock();
                    if( n_waiters>0 && my_ticket<=ticket_for_sleep && my_ticket==(wokeup_ticket+1) ) {
                        signaled_ticket = my_ticket;
                        test_cv.notify_one();
                        ++n_signaled;
                        ul.unlock();
                        break;
                    }
                    ul.unlock();
                    __TBB_Yield();
                }

                // give waiters time to go to sleep.
                for( int m=0; m<short_delay; ++m )
                    __TBB_Yield();
            }
        } else {
            while( ticket_for_sleep<max_ticket ) {
                unique_lock<M> ul( my_mtx, defer_lock );
                ul.lock();
                // exercise wait part
                int my_ticket = ++ticket_for_sleep; // grab my ticket
                if( my_ticket>max_ticket ) break;

                // each waiter should go to sleep at least once
                unsigned nw = ++n_waiters;
                for( ;; ) {
                    // update to max_waitq_length
                    if( nw>max_waitq_length ) max_waitq_length = nw;
                    ++n_visit_to_waitq;
                    test_cv.wait( ul );
                    // if( ret==false ) ++n_timedout;
                    ASSERT( ul, "mutex should have been locked" );
                    --n_waiters;
                    if( signaled_ticket==my_ticket ) {
                        wokeup_ticket = my_ticket;
                        break;
                    }
                    if( n_waiters>0 )
                        test_cv.notify_one();
                    nw = ++n_waiters; // update to max_waitq_length occurs above
                }

                ul.unlock();
                __TBB_Yield(); // give other threads chance to run.
            }
        }
        ++n_done;
        spin_wait_until_eq( n_done, nthread );
        ASSERT( n_signaled==max_ticket, "incorrect number of notifications sent" );
    }
};

struct TestPredicate1 {
    int target;
    TestPredicate1( int i_ ) : target(i_) {}
    bool operator()( ) { return signaled_ticket==target; }
};

template<typename M>
struct WorkForCondVarWaitPredAndNotifyAll: NoAssign {
    int nthread;
    condition_variable& test_cv;
    M& my_mtx;
    int multiple;
    WorkForCondVarWaitPredAndNotifyAll( int n_, condition_variable& cv_, M& mtx_, int m_ ) :
        nthread(n_), test_cv(cv_), my_mtx(mtx_), multiple(m_) {}
    void operator()( int tid ) const {
        if( tid&1 ) {
            while( ticket_for_sleep<max_ticket ) {
                unique_lock<M> ul( my_mtx, defer_lock );
                // exercise wait part
                int my_ticket = ++ticket_for_sleep; // grab my ticket
                if( my_ticket>max_ticket )
                    break;

                ul.lock();
                ++n_visit_to_waitq;
                unsigned nw = ++n_waiters;
                if( nw>max_waitq_length ) max_waitq_length = nw;
                test_cv.wait( ul, TestPredicate1( my_ticket ) );
                wokeup_ticket = my_ticket;
                --n_waiters;
                ASSERT( ul, "mutex should have been locked" );
                ul.unlock();

                __TBB_Yield(); // give other threads chance to run.
            }
        } else {
            // exercise signal part
            while( ticket_for_wakeup<max_ticket ) {
                int my_ticket = ++ticket_for_wakeup; // atomically grab the next ticket
                if( my_ticket>max_ticket )
                    break;

                for( ;; ) {
                    unique_lock<M> ul( my_mtx );
                    if( n_waiters>0 && my_ticket<=ticket_for_sleep && my_ticket==(wokeup_ticket+1) ) {
                        signaled_ticket = my_ticket;
                        test_cv.notify_all();
                        ++n_signaled;
                        ul.unlock();
                        break;
                    }
                    ul.unlock();
                    __TBB_Yield();
                }

                // give waiters time to go to sleep.
                for( int m=0; m<long_delay*multiple; ++m )
                    __TBB_Yield();
            }
        }
        ++n_done;
        spin_wait_until_eq( n_done, nthread );
        ASSERT( n_signaled==max_ticket, "incorrect number of notifications sent" );
    }
};

void InitGlobalCounters()
{
      ticket_for_sleep = ticket_for_wakeup = signaled_ticket = wokeup_ticket = 0;
      n_waiters = 0;
      n_signaled = 0;
      n_done = n_done_1 = n_done_2 = 0;
      n_visit_to_waitq = 0;
      n_timed_out = 0;
}

template<typename M>
void TestConditionVariable( const char* name, int nthread )
{
    REMARK("testing %s in TestConditionVariable\n",name);
    Counter<M> counter;
    M mtx;

    ASSERT( nthread>1, "at least two threads are needed for testing condition_variable" );
    REMARK(" - constructor\n" );
    // Test constructor.
    {
      condition_variable cv1;
#if _WIN32||_WIN64
      condition_variable::native_handle_type handle = cv1.native_handle();
      ASSERT( uintptr_t(&handle->cv_event)==uintptr_t(&handle->cv_native), NULL );
#endif
      M mtx1;
      barrier = 0;
      NativeParallelFor( 2, WorkForCondVarCtor<M>( cv1, mtx1 ) );
    }

    REMARK(" - destructor\n" );
    // Test destructor.
    {
      M mtx2;
      test_cv = NULL;
      n_waiters = 0;
      NativeParallelFor( nthread, WorkForCondVarDtor<M>( nthread, mtx2 ) );
    }

    REMARK(" - timed_wait (i.e., wait_for)\n");
    // Test timed wait.
    {
      condition_variable cv_tw;
      M mtx_tw;
      barrier = 0;
      InitGlobalCounters();
      int nthr = nthread>4?4:nthread;
      NativeParallelFor( nthr, WorkForCondVarTimedWait<M>( nthr, cv_tw, mtx_tw ) );
    }

    REMARK(" - wait with notify_one\n");
    // Test wait and notify_one
    do {
        condition_variable cv3;
        M mtx3;
        InitGlobalCounters();
        NativeParallelFor( nthread, WorkForCondVarWaitAndNotifyOne<M>( nthread, cv3, mtx3 ) );
    } while( n_visit_to_waitq==0 || max_waitq_length==0 );

    REMARK(" - predicated wait with notify_all\n");
    // Test wait_pred and notify_all
    int delay_multiple = 1;
    do {
        condition_variable cv4;
        M mtx4;
        InitGlobalCounters();
        NativeParallelFor( nthread, WorkForCondVarWaitPredAndNotifyAll<M>( nthread, cv4, mtx4, delay_multiple ) );
        if( max_waitq_length<unsigned(nthread/2) )
            ++delay_multiple;
    } while( n_visit_to_waitq<=0 || max_waitq_length<unsigned(nthread/2) );
}

#if TBB_USE_EXCEPTIONS && !__TBB_THROW_ACROSS_MODULE_BOUNDARY_BROKEN
static tbb::atomic<int> err_count;

#define TRY_AND_CATCH_RUNTIME_ERROR(op,msg) \
        try {                             \
            op;                           \
            ++err_count;                  \
        } catch( std::runtime_error& e ) {ASSERT( strstr(e.what(), msg) , NULL );} catch(...) {++err_count;}

template<typename M>
void TestUniqueLockException( const char * name ) {
    REMARK("testing %s TestUniqueLockException\n",name);
    M mtx;
    unique_lock<M> ul_0;
    err_count = 0;

    TRY_AND_CATCH_RUNTIME_ERROR( ul_0.lock(), "Operation not permitted" );
    TRY_AND_CATCH_RUNTIME_ERROR( ul_0.try_lock(), "Operation not permitted" );

    unique_lock<M> ul_1( mtx );

    TRY_AND_CATCH_RUNTIME_ERROR( ul_1.lock(), "Resource deadlock" );
    TRY_AND_CATCH_RUNTIME_ERROR( ul_1.try_lock(), "Resource deadlock" );

    ul_1.unlock();
    TRY_AND_CATCH_RUNTIME_ERROR( ul_1.unlock(), "Operation not permitted" );

    ASSERT( !err_count, "Some exceptions are not thrown or incorrect ones are thrown" );
}

template<typename M>
void TestConditionVariableException( const char * name ) {
    REMARK("testing %s in TestConditionVariableException; yet to be implemented\n",name);
}
#endif /* TBB_USE_EXCEPTIONS */

template<typename Mutex, typename RecursiveMutex>
void DoCondVarTest()
{
#if __TBB_CPP11_RVALUE_REF_PRESENT
    TestUniqueLockMoveConstructorAndAssignOp<Mutex>();
    TestUniqueLockMoveConstructorAndAssignOp<RecursiveMutex>();
#endif

    for( int p=MinThread; p<=MaxThread; ++p ) {
        REMARK( "testing with %d threads\n", p );
        TestLocks<Mutex>( "mutex", p );
        TestLocks<RecursiveMutex>( "recursive_mutex", p );

        if( p<=1 ) continue;

        // for testing condition_variable, at least one sleeper and one notifier are needed
        TestConditionVariable<Mutex>( "mutex", p );
    }
#if __TBB_THROW_ACROSS_MODULE_BOUNDARY_BROKEN
    REPORT("Known issue: exception handling tests are skipped.\n");
#elif TBB_USE_EXCEPTIONS
    TestUniqueLockException<Mutex>( "mutex" );
    TestUniqueLockException<RecursiveMutex>( "recursive_mutex" );
    TestConditionVariableException<Mutex>( "mutex" );
#endif /* TBB_USE_EXCEPTIONS */
}
