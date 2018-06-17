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

#define HARNESS_DEFAULT_MIN_THREADS 2
#define HARNESS_DEFAULT_MAX_THREADS 4
#define __TBB_SHUFFLE_PRESENT (_MSC_VER >= 1910 || __cplusplus >= 201103L && (__TBB_GLIBCXX_VERSION >= 40800 || _LIBCPP_VERSION))

#include "harness.h"

#if __TBB_SHUFFLE_PRESENT
#include <random>
#endif

#if __TBB_TASK_GROUP_CONTEXT

#define __TBB_ATOMICS_CODEGEN_BROKEN __SUNPRO_CC

#define private public
#include "tbb/task.h"
#undef private

#include "tbb/task_scheduler_init.h"
#include "tbb/spin_mutex.h"
#include "tbb/tick_count.h"

#include <string>

#define NUM_CHILD_TASKS                 256
#define NUM_ROOT_TASKS                  32
#define NUM_ROOTS_IN_GROUP              8

//! Statistics about number of tasks in different states
class TaskStats {
    typedef tbb::spin_mutex::scoped_lock lock_t;
    //! Number of tasks allocated that was ever allocated
    volatile intptr_t m_Existed;
    //! Number of tasks executed to the moment
    volatile intptr_t m_Executed;
    //! Number of tasks allocated but not yet destroyed to the moment
    volatile intptr_t m_Existing;

    mutable tbb::spin_mutex  m_Mutex;
public:
    //! Assumes that assignment is noncontended for the left-hand operand
    const TaskStats& operator= ( const TaskStats& rhs ) {
        if ( this != &rhs ) {
            lock_t lock(rhs.m_Mutex);
            m_Existed = rhs.m_Existed;
            m_Executed = rhs.m_Executed;
            m_Existing = rhs.m_Existing;
        }
        return *this;
    }
    intptr_t Existed() const { return m_Existed; }
    intptr_t Executed() const { return m_Executed; }
    intptr_t Existing() const { return m_Existing; }
    void IncExisted() { lock_t lock(m_Mutex); ++m_Existed; ++m_Existing; }
    void IncExecuted() { lock_t lock(m_Mutex); ++m_Executed; }
    void DecExisting() { lock_t lock(m_Mutex); --m_Existing; }
    //! Assumed to be used in uncontended manner only
    void Reset() { m_Executed = m_Existing = m_Existed = 0; }
};

TaskStats g_CurStat;

inline intptr_t Existed () { return g_CurStat.Existed(); }

#include "harness_eh.h"

bool g_BoostExecutedCount = true;
volatile bool g_TaskWasCancelled = false;

inline void ResetGlobals () {
    ResetEhGlobals();
    g_BoostExecutedCount = true;
    g_TaskWasCancelled = false;
    g_CurStat.Reset();
}

#define ASSERT_TEST_POSTCOND() \
    ASSERT (g_CurStat.Existed() >= g_CurStat.Executed(), "Total number of tasks is less than executed");  \
    ASSERT (!g_CurStat.Existing(), "Not all task objects have been destroyed"); \
    ASSERT (!tbb::task::self().is_cancelled(), "Scheduler's default context has not been cleaned up properly");

inline void WaitForException () {
    int n = 0;
    while ( ++n < c_Timeout && !__TBB_load_with_acquire(g_ExceptionCaught) )
        __TBB_Yield();
    ASSERT_WARNING( n < c_Timeout, "WaitForException failed" );
}

class TaskBase : public tbb::task {
    tbb::task* execute () __TBB_override {
        tbb::task* t = NULL;
        __TBB_TRY {
            t = do_execute();
        } __TBB_CATCH( ... ) {
            g_CurStat.IncExecuted();
            __TBB_RETHROW();
        }
        g_CurStat.IncExecuted();
        return t;
    }
protected:
    TaskBase ( bool throwException = true ) : m_Throw(throwException) { g_CurStat.IncExisted(); }
    ~TaskBase () { g_CurStat.DecExisting(); }

    virtual tbb::task* do_execute () = 0;

    bool m_Throw;
}; // class TaskBase

class LeafTask : public TaskBase {
    tbb::task* do_execute () __TBB_override {
        Harness::ConcurrencyTracker ct;
        WaitUntilConcurrencyPeaks();
        if ( g_BoostExecutedCount )
            ++g_CurExecuted;
        if ( m_Throw )
            ThrowTestException(NUM_CHILD_TASKS/2);
        if ( !g_ThrowException )
            __TBB_Yield();
        return NULL;
    }
public:
    LeafTask ( bool throw_exception = true ) : TaskBase(throw_exception) {}
};

class SimpleRootTask : public TaskBase {
    tbb::task* do_execute () __TBB_override {
        set_ref_count(NUM_CHILD_TASKS + 1);
        for ( size_t i = 0; i < NUM_CHILD_TASKS; ++i )
            spawn( *new( allocate_child() ) LeafTask(m_Throw) );
        wait_for_all();
        return NULL;
    }
public:
    SimpleRootTask ( bool throw_exception = true ) : TaskBase(throw_exception) {}
};

#if TBB_USE_EXCEPTIONS

class SimpleThrowingTask : public tbb::task {
public:
    tbb::task* execute () __TBB_override { throw 0; }
    ~SimpleThrowingTask() {}
};

//! Checks if innermost running task information is updated correctly during cancellation processing
void Test0 () {
    tbb::task_scheduler_init init (1);
    tbb::empty_task &r = *new( tbb::task::allocate_root() ) tbb::empty_task;
    tbb::task_list tl;
    tl.push_back( *new( r.allocate_child() ) SimpleThrowingTask );
    tl.push_back( *new( r.allocate_child() ) SimpleThrowingTask );
    r.set_ref_count( 3 );
    try {
        r.spawn_and_wait_for_all( tl );
    }
    catch (...) {}
    r.destroy( r );
}

//! Default exception behavior test.
/** Allocates a root task that spawns a bunch of children, one or several of which throw
    a test exception in a worker or master thread (depending on the global setting). **/
void Test1 () {
    ResetGlobals();
    tbb::empty_task &r = *new( tbb::task::allocate_root() ) tbb::empty_task;
    ASSERT (!g_CurStat.Existing() && !g_CurStat.Existed() && !g_CurStat.Executed(),
            "something wrong with the task accounting");
    r.set_ref_count(NUM_CHILD_TASKS + 1);
    for ( int i = 0; i < NUM_CHILD_TASKS; ++i )
        r.spawn( *new( r.allocate_child() ) LeafTask );
    TRY();
        r.wait_for_all();
    CATCH_AND_ASSERT();
    r.destroy(r);
    ASSERT_TEST_POSTCOND();
} // void Test1 ()

//! Default exception behavior test.
/** Allocates and spawns root task that runs a bunch of children, one of which throws
    a test exception in a worker thread. (Similar to Test1, except that the root task
    is spawned by the test function, and children are created by the root task instead
    of the test function body.) **/
void Test2 () {
    ResetGlobals();
    SimpleRootTask &r = *new( tbb::task::allocate_root() ) SimpleRootTask;
    ASSERT (g_CurStat.Existing() == 1 && g_CurStat.Existed() == 1 && !g_CurStat.Executed(),
            "something wrong with the task accounting");
    TRY();
        tbb::task::spawn_root_and_wait(r);
    CATCH_AND_ASSERT();
    ASSERT (g_ExceptionCaught, "no exception occurred");
    ASSERT_TEST_POSTCOND();
} // void Test2 ()

//! The same as Test2() except the root task has explicit context.
/** The context is initialized as bound in order to check correctness of its associating
    with a root task. **/
void Test3 () {
    ResetGlobals();
    tbb::task_group_context  ctx(tbb::task_group_context::bound);
    SimpleRootTask &r = *new( tbb::task::allocate_root(ctx) ) SimpleRootTask;
    ASSERT (g_CurStat.Existing() == 1 && g_CurStat.Existed() == 1 && !g_CurStat.Executed(),
            "something wrong with the task accounting");
    TRY();
        tbb::task::spawn_root_and_wait(r);
    CATCH_AND_ASSERT();
    ASSERT (g_ExceptionCaught, "no exception occurred");
    ASSERT_TEST_POSTCOND();
} // void Test2 ()

class RootLauncherTask : public TaskBase {
    tbb::task_group_context::kind_type m_CtxKind;

    tbb::task* do_execute () __TBB_override {
        tbb::task_group_context  ctx(m_CtxKind);
        SimpleRootTask &r = *new( allocate_root() ) SimpleRootTask;
        r.change_group(ctx);
        TRY();
            spawn_root_and_wait(r);
            // Give a child of our siblings a chance to throw the test exception
            WaitForException();
        CATCH();
        ASSERT (__TBB_EXCEPTION_TYPE_INFO_BROKEN || !g_UnknownException, "unknown exception was caught");
        return NULL;
    }
public:
    RootLauncherTask ( tbb::task_group_context::kind_type ctx_kind = tbb::task_group_context::isolated ) : m_CtxKind(ctx_kind) {}
};

/** Allocates and spawns a bunch of roots, which allocate and spawn new root with
    isolated context, which at last spawns a bunch of children each, one of which
    throws a test exception in a worker thread. **/
void Test4 () {
    ResetGlobals();
    tbb::task_list  tl;
    for ( size_t i = 0; i < NUM_ROOT_TASKS; ++i )
        tl.push_back( *new( tbb::task::allocate_root() ) RootLauncherTask );
    TRY();
        tbb::task::spawn_root_and_wait(tl);
    CATCH_AND_ASSERT();
    ASSERT (!l_ExceptionCaughtAtCurrentLevel, "exception in this scope is unexpected");
    intptr_t  num_tasks_expected = NUM_ROOT_TASKS * (NUM_CHILD_TASKS + 2);
    ASSERT (g_CurStat.Existed() == num_tasks_expected, "Wrong total number of tasks");
    if ( g_SolitaryException )
        ASSERT (g_CurStat.Executed() >= num_tasks_expected - NUM_CHILD_TASKS, "Unexpected number of executed tasks");
    ASSERT_TEST_POSTCOND();
} // void Test4 ()

/** The same as Test4, except the contexts are bound. **/
void Test4_1 () {
    ResetGlobals();
    tbb::task_list  tl;
    for ( size_t i = 0; i < NUM_ROOT_TASKS; ++i )
        tl.push_back( *new( tbb::task::allocate_root() ) RootLauncherTask(tbb::task_group_context::bound) );
    TRY();
        tbb::task::spawn_root_and_wait(tl);
    CATCH_AND_ASSERT();
    ASSERT (!l_ExceptionCaughtAtCurrentLevel, "exception in this scope is unexpected");
    intptr_t  num_tasks_expected = NUM_ROOT_TASKS * (NUM_CHILD_TASKS + 2);
    ASSERT (g_CurStat.Existed() == num_tasks_expected, "Wrong total number of tasks");
    if ( g_SolitaryException )
        ASSERT (g_CurStat.Executed() >= num_tasks_expected - NUM_CHILD_TASKS, "Unexpected number of executed tasks");
    ASSERT_TEST_POSTCOND();
} // void Test4_1 ()


class RootsGroupLauncherTask : public TaskBase {
    tbb::task* do_execute () __TBB_override {
        tbb::task_group_context  ctx (tbb::task_group_context::isolated);
        tbb::task_list  tl;
        for ( size_t i = 0; i < NUM_ROOT_TASKS; ++i )
            tl.push_back( *new( allocate_root(ctx) ) SimpleRootTask );
        TRY();
            spawn_root_and_wait(tl);
            // Give worker a chance to throw exception
            WaitForException();
        CATCH_AND_ASSERT();
        return NULL;
    }
};

/** Allocates and spawns a bunch of roots, which allocate and spawn groups of roots
    with an isolated context shared by all group members, which at last spawn a bunch
    of children each, one of which throws a test exception in a worker thread. **/
void Test5 () {
    ResetGlobals();
    tbb::task_list  tl;
    for ( size_t i = 0; i < NUM_ROOTS_IN_GROUP; ++i )
        tl.push_back( *new( tbb::task::allocate_root() ) RootsGroupLauncherTask );
    TRY();
        tbb::task::spawn_root_and_wait(tl);
    CATCH_AND_ASSERT();
    ASSERT (!l_ExceptionCaughtAtCurrentLevel, "unexpected exception intercepted");
    if ( g_SolitaryException )  {
        intptr_t  num_tasks_expected = NUM_ROOTS_IN_GROUP * (1 + NUM_ROOT_TASKS * (1 + NUM_CHILD_TASKS));
        intptr_t  min_num_tasks_executed = num_tasks_expected - NUM_ROOT_TASKS * (NUM_CHILD_TASKS + 1);
        ASSERT (g_CurStat.Executed() >= min_num_tasks_executed, "Too few tasks executed");
    }
    ASSERT_TEST_POSTCOND();
} // void Test5 ()

class ThrowingRootLauncherTask : public TaskBase {
    tbb::task* do_execute () __TBB_override {
        tbb::task_group_context  ctx (tbb::task_group_context::bound);
        SimpleRootTask &r = *new( allocate_root(ctx) ) SimpleRootTask(false);
        TRY();
            spawn_root_and_wait(r);
        CATCH();
        ASSERT (!l_ExceptionCaughtAtCurrentLevel, "unexpected exception intercepted");
        ThrowTestException(NUM_CHILD_TASKS);
        g_TaskWasCancelled |= is_cancelled();
        return NULL;
    }
};

class BoundHierarchyLauncherTask : public TaskBase {
    bool m_Recover;

    void alloc_roots ( tbb::task_group_context& ctx, tbb::task_list& tl ) {
        for ( size_t i = 0; i < NUM_ROOT_TASKS; ++i )
            tl.push_back( *new( allocate_root(ctx) ) ThrowingRootLauncherTask );
    }

    tbb::task* do_execute () __TBB_override {
        tbb::task_group_context  ctx (tbb::task_group_context::isolated);
        tbb::task_list tl;
        alloc_roots(ctx, tl);
        TRY();
            spawn_root_and_wait(tl);
        CATCH_AND_ASSERT();
        ASSERT (l_ExceptionCaughtAtCurrentLevel, "no exception occurred");
        ASSERT (!tl.empty(), "task list was cleared somehow");
        if ( g_SolitaryException )
            ASSERT (g_TaskWasCancelled, "No tasks were cancelled despite of exception");
        if ( m_Recover ) {
            // Test task_group_context::unbind and task_group_context::reset methods
            g_ThrowException = false;
            l_ExceptionCaughtAtCurrentLevel = false;
            tl.clear();
            alloc_roots(ctx, tl);
            ctx.reset();
            try {
                spawn_root_and_wait(tl);
            }
            catch (...) {
                l_ExceptionCaughtAtCurrentLevel = true;
            }
            ASSERT (!l_ExceptionCaughtAtCurrentLevel, "unexpected exception occurred");
        }
        return NULL;
    }
public:
    BoundHierarchyLauncherTask ( bool recover = false ) : m_Recover(recover) {}

}; // class BoundHierarchyLauncherTask

//! Test for bound contexts forming 2 level tree. Exception is thrown on the 1st (root) level.
/** Allocates and spawns a root that spawns a bunch of 2nd level roots sharing
    the same isolated context, each of which in their turn spawns a single 3rd level
    root with  the bound context, and these 3rd level roots spawn bunches of leaves
    in the end. Leaves do not generate exceptions. The test exception is generated
    by one of the 2nd level roots. **/
void Test6 () {
    ResetGlobals();
    BoundHierarchyLauncherTask &r = *new( tbb::task::allocate_root() ) BoundHierarchyLauncherTask;
    TRY();
        tbb::task::spawn_root_and_wait(r);
    CATCH_AND_ASSERT();
    ASSERT (!l_ExceptionCaughtAtCurrentLevel, "unexpected exception intercepted");
    // After the first of the branches (ThrowingRootLauncherTask) completes,
    // the rest of the task tree may be collapsed before having a chance to execute leaves.
    // A number of branches running concurrently with the first one will be able to spawn leaves though.
    /// \todo: If additional checkpoints are added to scheduler the following assertion must weaken
    intptr_t  num_tasks_expected = 1 + NUM_ROOT_TASKS * (2 + NUM_CHILD_TASKS);
    intptr_t  min_num_tasks_created = 1 + g_NumThreads * 2 + NUM_CHILD_TASKS;
    // 2 stands for BoundHierarchyLauncherTask and SimpleRootTask
    // 1 corresponds to BoundHierarchyLauncherTask
    intptr_t  min_num_tasks_executed = 2 + 1 + NUM_CHILD_TASKS;
    ASSERT (g_CurStat.Existed() <= num_tasks_expected, "Number of expected tasks is calculated incorrectly");
    ASSERT (g_CurStat.Existed() >= min_num_tasks_created, "Too few tasks created");
    ASSERT (g_CurStat.Executed() >= min_num_tasks_executed, "Too few tasks executed");
    ASSERT_TEST_POSTCOND();
} // void Test6 ()

//! Tests task_group_context::unbind and task_group_context::reset methods.
/** Allocates and spawns a root that spawns a bunch of 2nd level roots sharing
    the same isolated context, each of which in their turn spawns a single 3rd level
    root with  the bound context, and these 3rd level roots spawn bunches of leaves
    in the end. Leaves do not generate exceptions. The test exception is generated
    by one of the 2nd level roots. **/
void Test7 () {
    ResetGlobals();
    BoundHierarchyLauncherTask &r = *new( tbb::task::allocate_root() ) BoundHierarchyLauncherTask;
    TRY();
        tbb::task::spawn_root_and_wait(r);
    CATCH_AND_ASSERT();
    ASSERT (!l_ExceptionCaughtAtCurrentLevel, "unexpected exception intercepted");
    ASSERT_TEST_POSTCOND();
} // void Test6 ()

class BoundHierarchyLauncherTask2 : public TaskBase {
    tbb::task* do_execute () __TBB_override {
        tbb::task_group_context  ctx;
        tbb::task_list  tl;
        for ( size_t i = 0; i < NUM_ROOT_TASKS; ++i )
            tl.push_back( *new( allocate_root(ctx) ) RootLauncherTask(tbb::task_group_context::bound) );
        TRY();
            spawn_root_and_wait(tl);
        CATCH_AND_ASSERT();
        // Exception must be intercepted by RootLauncherTask
        ASSERT (!l_ExceptionCaughtAtCurrentLevel, "no exception occurred");
        return NULL;
    }
}; // class BoundHierarchyLauncherTask2

//! Test for bound contexts forming 2 level tree. Exception is thrown in the 2nd (outer) level.
/** Allocates and spawns a root that spawns a bunch of 2nd level roots sharing
    the same isolated context, each of which in their turn spawns a single 3rd level
    root with  the bound context, and these 3rd level roots spawn bunches of leaves
    in the end. The test exception is generated by one of the leaves. **/
void Test8 () {
    ResetGlobals();
    BoundHierarchyLauncherTask2 &r = *new( tbb::task::allocate_root() ) BoundHierarchyLauncherTask2;
    TRY();
        tbb::task::spawn_root_and_wait(r);
    CATCH_AND_ASSERT();
    ASSERT (!l_ExceptionCaughtAtCurrentLevel, "unexpected exception intercepted");
    if ( g_SolitaryException )  {
        intptr_t  num_tasks_expected = 1 + NUM_ROOT_TASKS * (2 + NUM_CHILD_TASKS);
        intptr_t  min_num_tasks_created = 1 + g_NumThreads * (2 + NUM_CHILD_TASKS);
        intptr_t  min_num_tasks_executed = num_tasks_expected - (NUM_CHILD_TASKS + 1);
        ASSERT (g_CurStat.Existed() <= num_tasks_expected, "Number of expected tasks is calculated incorrectly");
        ASSERT (g_CurStat.Existed() >= min_num_tasks_created, "Too few tasks created");
        ASSERT (g_CurStat.Executed() >= min_num_tasks_executed, "Too few tasks executed");
    }
    ASSERT_TEST_POSTCOND();
} // void Test8 ()

template<typename T>
void ThrowMovableException ( intptr_t threshold, const T& data ) {
    if ( !IsThrowingThread() )
        return;
    if ( !g_SolitaryException ) {
#if __TBB_ATOMICS_CODEGEN_BROKEN
        g_ExceptionsThrown = g_ExceptionsThrown + 1;
#else
        ++g_ExceptionsThrown;
#endif
        throw tbb::movable_exception<T>(data);
    }
    while ( g_CurStat.Existed() < threshold )
        __TBB_Yield();
    if ( g_ExceptionsThrown.compare_and_swap(1, 0) == 0 )
        throw tbb::movable_exception<T>(data);
}

const int g_IntExceptionData = -375;
const std::string g_StringExceptionData = "My test string";

// Exception data class implementing minimal requirements of tbb::movable_exception
class ExceptionData {
    const ExceptionData& operator = ( const ExceptionData& src );
    explicit ExceptionData ( int n ) : m_Int(n), m_String(g_StringExceptionData) {}
public:
    ExceptionData ( const ExceptionData& src ) : m_Int(src.m_Int), m_String(src.m_String) {}
    ~ExceptionData () {}

    int m_Int;
    std::string m_String;

    // Simple way to provide an instance when all initializing constructors are private
    // and to avoid memory reclamation problems.
    static ExceptionData s_data;
};

ExceptionData ExceptionData::s_data(g_IntExceptionData);

typedef tbb::movable_exception<int> SolitaryMovableException;
typedef tbb::movable_exception<ExceptionData> MultipleMovableException;

class LeafTaskWithMovableExceptions : public TaskBase {
    tbb::task* do_execute () __TBB_override {
        Harness::ConcurrencyTracker ct;
        WaitUntilConcurrencyPeaks();
        if ( g_SolitaryException )
            ThrowMovableException<int>(NUM_CHILD_TASKS/2, g_IntExceptionData);
        else
            ThrowMovableException<ExceptionData>(NUM_CHILD_TASKS/2, ExceptionData::s_data);
        return NULL;
    }
};

void CheckException ( tbb::tbb_exception& e ) {
    ASSERT (strcmp(e.name(), (g_SolitaryException ? typeid(SolitaryMovableException)
                                                   : typeid(MultipleMovableException)).name() ) == 0,
                                                   "Unexpected original exception name");
    ASSERT (strcmp(e.what(), "tbb::movable_exception") == 0, "Unexpected original exception info ");
    if ( g_SolitaryException ) {
        SolitaryMovableException& me = dynamic_cast<SolitaryMovableException&>(e);
        ASSERT (me.data() == g_IntExceptionData, "Unexpected solitary movable_exception data");
    }
    else {
        MultipleMovableException& me = dynamic_cast<MultipleMovableException&>(e);
        ASSERT (me.data().m_Int == g_IntExceptionData, "Unexpected multiple movable_exception int data");
        ASSERT (me.data().m_String == g_StringExceptionData, "Unexpected multiple movable_exception string data");
    }
}

void CheckException () {
    try {
        throw;
    } catch ( tbb::tbb_exception& e ) {
        CheckException(e);
    }
    catch ( ... ) {
    }
}

//! Test for movable_exception behavior, and external exception recording.
/** Allocates a root task that spawns a bunch of children, one or several of which throw
    a movable exception in a worker or master thread (depending on the global settings).
    The test also checks the correctness of multiple rethrowing of the pending exception. **/
void TestMovableException () {
    REMARK( "TestMovableException\n" );
    ResetGlobals();
    bool bUnsupported = false;
    tbb::task_group_context ctx;
    tbb::empty_task *r = new( tbb::task::allocate_root() ) tbb::empty_task;
    ASSERT (!g_CurStat.Existing() && !g_CurStat.Existed() && !g_CurStat.Executed(),
            "something wrong with the task accounting");
    r->set_ref_count(NUM_CHILD_TASKS + 1);
    for ( int i = 0; i < NUM_CHILD_TASKS; ++i )
        r->spawn( *new( r->allocate_child() ) LeafTaskWithMovableExceptions );
    TRY()
        r->wait_for_all();
    } catch ( ... ) {
        ASSERT (!ctx.is_group_execution_cancelled(), "");
        CheckException();
        try {
            throw;
        } catch ( tbb::tbb_exception& e ) {
            CheckException(e);
            g_ExceptionCaught = l_ExceptionCaughtAtCurrentLevel = true;
        }
        catch ( ... ) {
            g_ExceptionCaught = true;
            g_UnknownException = unknownException = true;
        }
        try {
            ctx.register_pending_exception();
        } catch ( ... ) {
            bUnsupported = true;
            REPORT( "Warning: register_pending_exception() failed. This is expected in case of linking with static msvcrt\n" );
        }
        ASSERT (ctx.is_group_execution_cancelled() || bUnsupported, "After exception registration the context must be in the cancelled state");
    }
    r->destroy(*r);
    ASSERT_EXCEPTION();
    ASSERT_TEST_POSTCOND();

    r = new( tbb::task::allocate_root(ctx) ) tbb::empty_task;
    r->set_ref_count(1);
    g_ExceptionCaught = g_UnknownException = false;
    try {
        r->wait_for_all();
    } catch ( tbb::tbb_exception& e ) {
        CheckException(e);
        g_ExceptionCaught = true;
    }
    catch ( ... ) {
        g_ExceptionCaught = true;
        g_UnknownException = true;
    }
    ASSERT (g_ExceptionCaught || bUnsupported, "no exception occurred");
    ASSERT (__TBB_EXCEPTION_TYPE_INFO_BROKEN || !g_UnknownException  || bUnsupported, "unknown exception was caught");
    r->destroy(*r);
} // void Test10 ()

#endif /* TBB_USE_EXCEPTIONS */

template<class T>
class CtxLauncherTask : public tbb::task {
    tbb::task_group_context &m_Ctx;

    tbb::task* execute () __TBB_override {
        spawn_root_and_wait( *new( allocate_root(m_Ctx) ) T );
        return NULL;
    }
public:
    CtxLauncherTask ( tbb::task_group_context& ctx ) : m_Ctx(ctx) {}
};

//! Test for cancelling a task hierarchy from outside (from a task running in parallel with it).
void TestCancelation () {
    ResetGlobals();
    g_ThrowException = false;
    tbb::task_group_context  ctx;
    tbb::task_list  tl;
    tl.push_back( *new( tbb::task::allocate_root() ) CtxLauncherTask<SimpleRootTask>(ctx) );
    tl.push_back( *new( tbb::task::allocate_root() ) CancellatorTask(ctx, NUM_CHILD_TASKS / 4) );
    TRY();
        tbb::task::spawn_root_and_wait(tl);
    CATCH_AND_FAIL();
    ASSERT (g_CurStat.Executed() <= g_ExecutedAtLastCatch + g_NumThreads, "Too many tasks were executed after cancellation");
    ASSERT_TEST_POSTCOND();
} // void Test9 ()

class CtxDestroyerTask : public tbb::task {
    int m_nestingLevel;

    tbb::task* execute () __TBB_override {
        ASSERT ( m_nestingLevel >= 0 && m_nestingLevel < MaxNestingDepth, "Wrong nesting level. The test is broken" );
        tbb::task_group_context  ctx;
        tbb::task *t = new( allocate_root(ctx) ) tbb::empty_task;
        int level = ++m_nestingLevel;
        if ( level < MaxNestingDepth ) {
            execute();
        }
        else {
            if ( !CancellatorTask::WaitUntilReady() )
                REPORT( "Warning: missing wakeup\n" );
            ++g_CurExecuted;
        }
        if ( ctx.is_group_execution_cancelled() )
            ++s_numCancelled;
        t->destroy(*t);
        return NULL;
    }
public:
    CtxDestroyerTask () : m_nestingLevel(0) { s_numCancelled = 0; }

    static const int MaxNestingDepth = 256;
    static int s_numCancelled;
};

int CtxDestroyerTask::s_numCancelled = 0;

//! Test for data race between cancellation propagation and context destruction.
/** If the data race ever occurs, an assertion inside TBB will be triggered. **/
void TestCtxDestruction () {
    REMARK( "TestCtxDestruction\n" );
    for ( size_t i = 0; i < 10; ++i ) {
        tbb::task_group_context  ctx;
        tbb::task_list  tl;
        ResetGlobals();
        g_BoostExecutedCount = false;
        g_ThrowException = false;
        CancellatorTask::Reset();

        tl.push_back( *new( tbb::task::allocate_root() ) CtxLauncherTask<CtxDestroyerTask>(ctx) );
        tl.push_back( *new( tbb::task::allocate_root() ) CancellatorTask(ctx, 1) );
        tbb::task::spawn_root_and_wait(tl);
        ASSERT( g_CurExecuted == 1, "Test is broken" );
        ASSERT( CtxDestroyerTask::s_numCancelled <= CtxDestroyerTask::MaxNestingDepth, "Test is broken" );
    }
} // void TestCtxDestruction()

#include <algorithm>
#include "harness_barrier.h"

class CtxConcurrentDestroyer : NoAssign, Harness::NoAfterlife {
    static const int ContextsPerThread = 512;

    static int s_Concurrency;
    static int s_NumContexts;
    static tbb::task_group_context** s_Contexts;
    static char* s_Buffer;
    static Harness::SpinBarrier s_Barrier;
    static Harness::SpinBarrier s_ExitBarrier;

    struct Shuffler {
        void operator() () const {
#if __TBB_SHUFFLE_PRESENT
            std::shuffle(s_Contexts, s_Contexts + s_NumContexts, std::mt19937(std::random_device()()));
#else
            std::random_shuffle(s_Contexts, s_Contexts + s_NumContexts);
#endif
        }
    };
public:
    static void Init ( int p ) {
        s_Concurrency = p;
        s_NumContexts = p * ContextsPerThread;
        s_Contexts = new tbb::task_group_context*[s_NumContexts];
        s_Buffer = new char[s_NumContexts * sizeof(tbb::task_group_context)];
        s_Barrier.initialize( p );
        s_ExitBarrier.initialize( p );
    }
    static void Uninit () {
        for ( int i = 0; i < s_NumContexts; ++i ) {
            tbb::internal::context_list_node_t &node = s_Contexts[i]->my_node;
            ASSERT( !node.my_next && !node.my_prev, "Destroyed context was written to during context chain update" );
        }
        delete []s_Contexts;
        delete []s_Buffer;
    }

    void operator() ( int id ) const {
        int begin = ContextsPerThread * id,
            end = begin + ContextsPerThread;
        for ( int i = begin; i < end; ++i )
            s_Contexts[i] = new( s_Buffer + i * sizeof(tbb::task_group_context) ) tbb::task_group_context;
        s_Barrier.wait( Shuffler() );
        for ( int i = begin; i < end; ++i ) {
            s_Contexts[i]->tbb::task_group_context::~task_group_context();
            memset( s_Contexts[i], 0, sizeof(tbb::task_group_context) );
        }
        s_ExitBarrier.wait();
    }
}; // class CtxConcurrentDestroyer

int CtxConcurrentDestroyer::s_Concurrency;
int CtxConcurrentDestroyer::s_NumContexts;
tbb::task_group_context** CtxConcurrentDestroyer::s_Contexts;
char* CtxConcurrentDestroyer::s_Buffer;
Harness::SpinBarrier CtxConcurrentDestroyer::s_Barrier;
Harness::SpinBarrier CtxConcurrentDestroyer::s_ExitBarrier;

void TestConcurrentCtxDestruction () {
    REMARK( "TestConcurrentCtxDestruction\n" );
    CtxConcurrentDestroyer::Init(g_NumThreads);
    NativeParallelFor( g_NumThreads, CtxConcurrentDestroyer() );
    CtxConcurrentDestroyer::Uninit();
}

void RunTests () {
    REMARK ("Number of threads %d\n", g_NumThreads);
    tbb::task_scheduler_init init (g_NumThreads);
    g_Master = Harness::CurrentTid();
#if TBB_USE_EXCEPTIONS
    Test1();
    Test2();
    Test3();
    Test4();
    Test4_1();
    Test5();
    Test6();
    Test7();
    Test8();
    TestMovableException();
#endif /* TBB_USE_EXCEPTIONS */
    TestCancelation();
    TestCtxDestruction();
#if !RML_USE_WCRM
    TestConcurrentCtxDestruction();
#endif
}

int TestMain () {
    REMARK ("Using %s\n", TBB_USE_CAPTURED_EXCEPTION ? "tbb:captured_exception" : "exact exception propagation");
    MinThread = min(NUM_ROOTS_IN_GROUP, min(tbb::task_scheduler_init::default_num_threads(), max(2, MinThread)));
    MaxThread = min(NUM_ROOTS_IN_GROUP, max(MinThread, min(tbb::task_scheduler_init::default_num_threads(), MaxThread)));
    ASSERT (NUM_ROOTS_IN_GROUP < NUM_ROOT_TASKS, "Fix defines");
#if TBB_USE_EXCEPTIONS
    // Test0 always runs on one thread
    Test0();
#endif /* TBB_USE_EXCEPTIONS */
    g_SolitaryException = 0;
    for ( g_NumThreads = MinThread; g_NumThreads <= MaxThread; ++g_NumThreads )
        RunTests();
    return Harness::Done;
}

#else /* !__TBB_TASK_GROUP_CONTEXT */

int TestMain () {
    return Harness::Skipped;
}

#endif /* !__TBB_TASK_GROUP_CONTEXT */
