/*
    Copyright (c) 2017-2018 Intel Corporation

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

#include "rml_tbb.h"
#include "../server/thread_monitor.h"
#include "tbb/atomic.h"
#include "tbb/cache_aligned_allocator.h"
#include "tbb/scheduler_common.h"
#include "tbb/governor.h"
#include "tbb/tbb_misc.h"

#include "ipc_utils.h"

#include <fcntl.h>

namespace rml {
namespace internal {

static const char* IPC_ENABLE_VAR_NAME = "IPC_ENABLE";

typedef versioned_object::version_type version_type;

extern "C" factory::status_type __RML_open_factory(factory& f, version_type& server_version, version_type client_version) {
    if( !tbb::internal::rml::get_enable_flag( IPC_ENABLE_VAR_NAME ) ) {
        return factory::st_incompatible;
    }

    // Hack to keep this library from being closed
    static tbb::atomic<bool> one_time_flag;
    if( one_time_flag.compare_and_swap(true,false)==false ) {
        __TBB_ASSERT( (size_t)f.library_handle!=factory::c_dont_unload, NULL );
#if _WIN32||_WIN64
        f.library_handle = reinterpret_cast<HMODULE>(factory::c_dont_unload);
#else
        f.library_handle = reinterpret_cast<void*>(factory::c_dont_unload);
#endif
    }
    // End of hack

    return factory::st_success;
}

extern "C" void __RML_close_factory(factory& f) {
}

class ipc_thread_monitor : public thread_monitor {
public:
    ipc_thread_monitor() : thread_monitor() {}

#if USE_WINTHREAD
#elif USE_PTHREAD
    static handle_type launch(thread_routine_type thread_routine, void* arg, size_t stack_size);
#endif
};

#if USE_WINTHREAD
#elif USE_PTHREAD
inline ipc_thread_monitor::handle_type ipc_thread_monitor::launch(void* (*thread_routine)(void*), void* arg, size_t stack_size) {
    pthread_attr_t s;
    if( pthread_attr_init( &s ) ) return 0;
    if( stack_size>0 ) {
        if( pthread_attr_setstacksize( &s, stack_size ) ) return 0;
    }
    pthread_t handle;
    if( pthread_create( &handle, &s, thread_routine, arg ) ) return 0;
    if( pthread_attr_destroy( &s ) ) return 0;
    return handle;
}
#endif

}} //rml::internal

using rml::internal::ipc_thread_monitor;

namespace tbb {
namespace internal {
namespace rml {

typedef ipc_thread_monitor::handle_type thread_handle;

class ipc_server;

static const char* IPC_MAX_THREADS_VAR_NAME = "MAX_THREADS";
static const char* IPC_ACTIVE_SEM_PREFIX = "/__IPC_active";
static const char* IPC_STOP_SEM_PREFIX = "/__IPC_stop";
static const char* IPC_ACTIVE_SEM_VAR_NAME = "IPC_ACTIVE_SEMAPHORE";
static const char* IPC_STOP_SEM_VAR_NAME = "IPC_STOP_SEMAPHORE";
static const mode_t IPC_SEM_MODE = 0660;

static tbb::atomic<unsigned> my_global_thread_count;

char* get_active_sem_name() {
    char* value = getenv( IPC_ACTIVE_SEM_VAR_NAME );
    if( value!=NULL && strlen( value )>0 ) {
        char* sem_name = new char[strlen( value ) + 1];
        __TBB_ASSERT( sem_name!=NULL, NULL );
        strcpy( sem_name, value );
        return sem_name;
    } else {
        return get_shared_name( IPC_ACTIVE_SEM_PREFIX );
    }
}

char* get_stop_sem_name() {
    char* value = getenv( IPC_STOP_SEM_VAR_NAME );
    if( value!=NULL && strlen( value )>0 ) {
        char* sem_name = new char[strlen( value ) + 1];
        __TBB_ASSERT( sem_name!=NULL, NULL );
        strcpy( sem_name, value );
        return sem_name;
    } else {
        return get_shared_name( IPC_STOP_SEM_PREFIX );
    }
}

static void release_thread_sem(sem_t* my_sem) {
    int old;
    do {
        old = my_global_thread_count;
        if( old<=0 ) return;
    } while( my_global_thread_count.compare_and_swap(old-1, old)!=old );
    if( old>0 ) {
        sem_post( my_sem );
    }
}

extern "C" void set_active_sem_name() {
    char* templ = new char[strlen( IPC_ACTIVE_SEM_PREFIX ) + strlen( "_XXXXXX" ) + 1];
    __TBB_ASSERT( templ!=NULL, NULL );
    strcpy( templ, IPC_ACTIVE_SEM_PREFIX );
    strcpy( templ + strlen( IPC_ACTIVE_SEM_PREFIX ), "_XXXXXX" );
    char* sem_name = mktemp( templ );
    if( sem_name!=NULL ) {
        int status = setenv( IPC_ACTIVE_SEM_VAR_NAME, sem_name, 1 );
        __TBB_ASSERT( status==0, NULL );
    }
    delete[] templ;
}

extern "C" void set_stop_sem_name() {
    char* templ = new char[strlen( IPC_STOP_SEM_PREFIX ) + strlen( "_XXXXXX" ) + 1];
    __TBB_ASSERT( templ!=NULL, NULL );
    strcpy( templ, IPC_STOP_SEM_PREFIX );
    strcpy( templ + strlen( IPC_STOP_SEM_PREFIX ), "_XXXXXX" );
    char* sem_name = mktemp( templ );
    if( sem_name!=NULL ) {
        int status = setenv( IPC_STOP_SEM_VAR_NAME, sem_name, 1 );
        __TBB_ASSERT( status==0, NULL );
    }
    delete[] templ;
}

extern "C" void release_resources() {
    if( my_global_thread_count!=0 ) {
        char* active_sem_name = get_active_sem_name();
        sem_t* my_active_sem = sem_open( active_sem_name, O_CREAT );
        __TBB_ASSERT( my_active_sem, "Unable to open active threads semaphore" );
        delete[] active_sem_name;

        do {
            release_thread_sem( my_active_sem );
        } while( my_global_thread_count!=0 );
    }
}

extern "C" void release_semaphores() {
    int status = 0;
    char* sem_name = NULL;

    sem_name = get_active_sem_name();
    if( sem_name==NULL ) {
        runtime_warning("Can not get RML semaphore name");
        return;
    }
    status = sem_unlink( sem_name );
    if( status!=0 ) {
        if( errno==ENOENT ) {
            /* There is no semaphore with the given name, nothing to do */
        } else {
            runtime_warning("Can not release RML semaphore");
            return;
        }
    }
    delete[] sem_name;

    sem_name = get_stop_sem_name();
    if( sem_name==NULL ) {
        runtime_warning( "Can not get RML semaphore name" );
        return;
    }
    status = sem_unlink( sem_name );
    if( status!=0 ) {
        if( errno==ENOENT ) {
            /* There is no semaphore with the given name, nothing to do */
        } else {
            runtime_warning("Can not release RML semaphore");
            return;
        }
    }
    delete[] sem_name;
}

class ipc_worker: no_copy {
protected:
    //! State in finite-state machine that controls the worker.
    /** State diagram:
                    /----------stop---\
                    |           ^     |
                    V           |     |
        init --> starting --> normal  |
          |         |           |     |
          |         V           |     |
          \------> quit <-------/<----/
      */
    enum state_t {
        //! *this is initialized
        st_init,
        //! *this has associated thread that is starting up.
        st_starting,
        //! Associated thread is doing normal life sequence.
        st_normal,
        //! Associated thread is stopped but can be started again.
        st_stop,
        //! Associated thread has ended normal life sequence and promises to never touch *this again.
        st_quit
    };
    atomic<state_t> my_state;

    //! Associated server
    ipc_server& my_server;

    //! Associated client
    tbb_client& my_client;

    //! index used for avoiding the 64K aliasing problem
    const size_t my_index;

    //! Monitor for sleeping when there is no work to do.
    /** The invariant that holds for sleeping workers is:
        "my_slack<=0 && my_state==st_normal && I am on server's list of asleep threads" */
    ipc_thread_monitor my_thread_monitor;

    //! Handle of the OS thread associated with this worker
    thread_handle my_handle;

    //! Link for list of workers that are sleeping or have no associated thread.
    ipc_worker* my_next;

    friend class ipc_server;

    //! Actions executed by the associated thread
    void run();

    //! Wake up associated thread (or launch a thread if there is none)
    bool wake_or_launch();

    //! Called by a thread (usually not the associated thread) to commence termination.
    void start_shutdown(bool join);

    //! Called by a thread (usually not the associated thread) to commence stopping.
    void start_stopping(bool join);

    static __RML_DECL_THREAD_ROUTINE thread_routine(void* arg);

    static void release_handle(thread_handle my_handle, bool join);

protected:
    ipc_worker(ipc_server& server, tbb_client& client, const size_t i) :
        my_server(server),
        my_client(client),
        my_index(i)
    {
        my_state = st_init;
    }
};

static const size_t cache_line_size = tbb::internal::NFS_MaxLineSize;

#if _MSC_VER && !defined(__INTEL_COMPILER)
    // Suppress overzealous compiler warnings about uninstantiable class
    #pragma warning(push)
    #pragma warning(disable:4510 4610)
#endif
class padded_ipc_worker: public ipc_worker {
    char pad[cache_line_size - sizeof(ipc_worker)%cache_line_size];
public:
    padded_ipc_worker(ipc_server& server, tbb_client& client, const size_t i)
    : ipc_worker( server,client,i ) { suppress_unused_warning(pad); }
};
#if _MSC_VER && !defined(__INTEL_COMPILER)
    #pragma warning(pop)
#endif

class ipc_waker : public padded_ipc_worker {
private:
    static __RML_DECL_THREAD_ROUTINE thread_routine(void* arg);
    void run();
    bool wake_or_launch();

    friend class ipc_server;

public:
    ipc_waker(ipc_server& server, tbb_client& client, const size_t i)
    : padded_ipc_worker( server, client, i ) {}
};

class ipc_stopper : public padded_ipc_worker {
private:
    static __RML_DECL_THREAD_ROUTINE thread_routine(void* arg);
    void run();
    bool wake_or_launch();

    friend class ipc_server;

public:
    ipc_stopper(ipc_server& server, tbb_client& client, const size_t i)
    : padded_ipc_worker( server, client, i ) {}
};

class ipc_server: public tbb_server, no_copy {
private:
    tbb_client& my_client;
    //! Maximum number of threads to be created.
    /** Threads are created lazily, so maximum might not actually be reached. */
    tbb_client::size_type my_n_thread;

    //! Stack size for each thread. */
    const size_t my_stack_size;

    //! Number of jobs that could use their associated thread minus number of active threads.
    /** If negative, indicates oversubscription.
        If positive, indicates that more threads should run.
        Can be lowered asynchronously, but must be raised only while holding my_asleep_list_mutex,
        because raising it impacts the invariant for sleeping threads. */
    atomic<int> my_slack;

    //! Counter used to determine when to delete this.
    atomic<int> my_ref_count;

    padded_ipc_worker* my_thread_array;

    //! List of workers that are asleep or committed to sleeping until notified by another thread.
    tbb::atomic<ipc_worker*> my_asleep_list_root;

    //! Protects my_asleep_list_root
    typedef scheduler_mutex_type asleep_list_mutex_type;
    asleep_list_mutex_type my_asleep_list_mutex;

    //! Should server wait workers while terminate
    const bool my_join_workers;

    //! Service thread for waking of workers
    ipc_waker* my_waker;

    //! Service thread to stop threads
    ipc_stopper* my_stopper;

    //! Semaphore to account active threads
    sem_t* my_active_sem;

    //! Semaphore to account stop threads
    sem_t* my_stop_sem;

#if TBB_USE_ASSERT
    atomic<int> my_net_slack_requests;
#endif /* TBB_USE_ASSERT */

    //! Wake up to two sleeping workers, if there are any sleeping.
    /** The call is used to propagate a chain reaction where each thread wakes up two threads,
        which in turn each wake up two threads, etc. */
    void propagate_chain_reaction() {
        // First test of a double-check idiom.  Second test is inside wake_some(0).
        if( my_slack>0 ) {
            int active_threads = 0;
            if( try_get_active_thread() ) {
                ++active_threads;
                if( try_get_active_thread() ) {
                    ++active_threads;
                }
                wake_some( 0, active_threads );
            }
        }
    }

    //! Try to add t to list of sleeping workers
    bool try_insert_in_asleep_list(ipc_worker& t);

    //! Try to add t to list of sleeping workers even if there is some work to do
    bool try_insert_in_asleep_list_forced(ipc_worker& t);

    //! Equivalent of adding additional_slack to my_slack and waking up to 2 threads if my_slack permits.
    void wake_some(int additional_slack, int active_threads);

    //! Equivalent of adding additional_slack to my_slack and waking up to 1 thread if my_slack permits.
    void wake_one_forced(int additional_slack);

    //! Stop one thread from asleep list
    bool stop_one();

    //! Wait for active thread
    bool wait_active_thread();

    //! Try to get active thread
    bool try_get_active_thread();

    //! Release active thread
    void release_active_thread();

    //! Wait for thread to stop
    bool wait_stop_thread();

    //! Add thread to stop list
    void add_stop_thread();

    void remove_server_ref() {
        if( --my_ref_count==0 ) {
            my_client.acknowledge_close_connection();
            this->~ipc_server();
            tbb::cache_aligned_allocator<ipc_server>().deallocate( this, 1 );
        }
    }

    friend class ipc_worker;
    friend class ipc_waker;
    friend class ipc_stopper;
public:
    ipc_server(tbb_client& client);
    virtual ~ipc_server();

    version_type version() const __TBB_override {
        return 0;
    }

    void request_close_connection(bool /*exiting*/) __TBB_override {
        my_waker->start_shutdown(false);
        my_stopper->start_shutdown(false);
        for( size_t i=0; i<my_n_thread; ++i )
            my_thread_array[i].start_shutdown( my_join_workers );
        remove_server_ref();
    }

    void yield() __TBB_override {__TBB_Yield();}

    void independent_thread_number_changed(int) __TBB_override { __TBB_ASSERT( false, NULL ); }

    unsigned default_concurrency() const __TBB_override { return my_n_thread - 1; }

    void adjust_job_count_estimate(int delta) __TBB_override;

#if _WIN32||_WIN64
    void register_master(::rml::server::execution_resource_t&) __TBB_override {}
    void unregister_master(::rml::server::execution_resource_t) __TBB_override {}
#endif /* _WIN32||_WIN64 */
};

//------------------------------------------------------------------------
// Methods of ipc_worker
//------------------------------------------------------------------------
#if _MSC_VER && !defined(__INTEL_COMPILER)
    // Suppress overzealous compiler warnings about an initialized variable 'sink_for_alloca' not referenced
    #pragma warning(push)
    #pragma warning(disable:4189)
#endif
#if __MINGW32__ && __GNUC__==4 &&__GNUC_MINOR__>=2 && !__MINGW64__
// ensure that stack is properly aligned
__attribute__((force_align_arg_pointer))
#endif
__RML_DECL_THREAD_ROUTINE ipc_worker::thread_routine(void* arg) {
    ipc_worker* self = static_cast<ipc_worker*>(arg);
    AVOID_64K_ALIASING( self->my_index );
    self->run();
    return 0;
}
#if _MSC_VER && !defined(__INTEL_COMPILER)
    #pragma warning(pop)
#endif

void ipc_worker::release_handle(thread_handle handle, bool join) {
    if( join )
        ipc_thread_monitor::join( handle );
    else
        ipc_thread_monitor::detach_thread( handle );
}

void ipc_worker::start_shutdown(bool join) {
    state_t s;

    do {
        s = my_state;
        __TBB_ASSERT( s!=st_quit, NULL );
    } while( my_state.compare_and_swap( st_quit, s )!=s );
    if( s==st_normal || s==st_starting ) {
        // May have invalidated invariant for sleeping, so wake up the thread.
        // Note that the notify() here occurs without maintaining invariants for my_slack.
        // It does not matter, because my_state==st_quit overrides checking of my_slack.
        my_thread_monitor.notify();
        // Do not need release handle in st_init state,
        // because in this case the thread wasn't started yet.
        // For st_starting release is done at launch site.
        if( s==st_normal )
            release_handle( my_handle, join );
    }
}

void ipc_worker::start_stopping(bool join) {
    state_t s;

    do {
        s = my_state;
    } while( my_state.compare_and_swap( st_stop, s )!=s );
    if( s==st_normal || s==st_starting ) {
        // May have invalidated invariant for sleeping, so wake up the thread.
        // Note that the notify() here occurs without maintaining invariants for my_slack.
        // It does not matter, because my_state==st_quit overrides checking of my_slack.
        my_thread_monitor.notify();
        // Do not need release handle in st_init state,
        // because in this case the thread wasn't started yet.
        // For st_starting release is done at launch site.
        if( s==st_normal )
            release_handle( my_handle, join );
    }
}

void ipc_worker::run() {
    my_server.propagate_chain_reaction();

    // Transiting to st_normal here would require setting my_handle,
    // which would create race with the launching thread and
    // complications in handle management on Windows.

    ::rml::job& j = *my_client.create_one_job();
    state_t state = my_state;
    while( state!=st_quit && state!=st_stop ) {
        if( my_server.my_slack>=0 ) {
            my_client.process(j);
        } else {
            ipc_thread_monitor::cookie c;
            // Prepare to wait
            my_thread_monitor.prepare_wait(c);
            // Check/set the invariant for sleeping
            state = my_state;
            if( state!=st_quit && state!=st_stop && my_server.try_insert_in_asleep_list(*this) ) {
                if( my_server.my_n_thread > 1 ) my_server.release_active_thread();
                my_thread_monitor.commit_wait(c);
                my_server.propagate_chain_reaction();
            } else {
                // Invariant broken
                my_thread_monitor.cancel_wait();
            }
        }
        state = my_state;
    }
    my_client.cleanup(j);

    my_server.remove_server_ref();
}

inline bool ipc_worker::wake_or_launch() {
    if( ( my_state==st_init && my_state.compare_and_swap( st_starting, st_init )==st_init ) ||
        ( my_state==st_stop && my_state.compare_and_swap( st_starting, st_stop )==st_stop ) ) {
        // after this point, remove_server_ref() must be done by created thread
#if USE_WINTHREAD
        my_handle = ipc_thread_monitor::launch( thread_routine, this, my_server.my_stack_size, &this->my_index );
#elif USE_PTHREAD
        {
        affinity_helper fpa;
        fpa.protect_affinity_mask( /*restore_process_mask=*/true );
        my_handle = ipc_thread_monitor::launch( thread_routine, this, my_server.my_stack_size );
        if( my_handle == 0 ) {
            // Unable to create new thread for process
            // However, this is expected situation for the use cases of this coordination server
            state_t s = my_state.compare_and_swap( st_init, st_starting );
            if (st_starting != s) {
                // Do shutdown during startup. my_handle can't be released
                // by start_shutdown, because my_handle value might be not set yet
                // at time of transition from st_starting to st_quit.
                __TBB_ASSERT( s==st_quit, NULL );
                release_handle( my_handle, my_server.my_join_workers );
            }
            return false;
        } else {
            my_server.my_ref_count++;
        }
        // Implicit destruction of fpa resets original affinity mask.
        }
#endif /* USE_PTHREAD */
        state_t s = my_state.compare_and_swap( st_normal, st_starting );
        if( st_starting!=s ) {
            // Do shutdown during startup. my_handle can't be released
            // by start_shutdown, because my_handle value might be not set yet
            // at time of transition from st_starting to st_quit.
            __TBB_ASSERT( s==st_quit, NULL );
            release_handle( my_handle, my_server.my_join_workers );
        }
    }
    else {
        my_thread_monitor.notify();
    }

    return true;
}

//------------------------------------------------------------------------
// Methods of ipc_waker
//------------------------------------------------------------------------
#if _MSC_VER && !defined(__INTEL_COMPILER)
    // Suppress overzealous compiler warnings about an initialized variable 'sink_for_alloca' not referenced
    #pragma warning(push)
    #pragma warning(disable:4189)
#endif
#if __MINGW32__ && __GNUC__==4 &&__GNUC_MINOR__>=2 && !__MINGW64__
// ensure that stack is properly aligned
__attribute__((force_align_arg_pointer))
#endif
__RML_DECL_THREAD_ROUTINE ipc_waker::thread_routine(void* arg) {
    ipc_waker* self = static_cast<ipc_waker*>(arg);
    AVOID_64K_ALIASING( self->my_index );
    self->run();
    return 0;
}
#if _MSC_VER && !defined(__INTEL_COMPILER)
    #pragma warning(pop)
#endif

void ipc_waker::run() {
    // Transiting to st_normal here would require setting my_handle,
    // which would create race with the launching thread and
    // complications in handle management on Windows.

    while( my_state!=st_quit ) {
        bool have_to_sleep = false;
        if( my_server.my_slack>0 ) {
            if( my_server.wait_active_thread() ) {
                if( my_server.my_slack>0 ) {
                    my_server.wake_some( 0, 1 );
                } else {
                    my_server.release_active_thread();
                    have_to_sleep = true;
                }
            }
        } else {
            have_to_sleep = true;
        }
        if( have_to_sleep ) {
            ipc_thread_monitor::cookie c;
            // Prepare to wait
            my_thread_monitor.prepare_wait(c);
            // Check/set the invariant for sleeping
            if( my_state!=st_quit && my_server.my_slack<0 ) {
                my_thread_monitor.commit_wait(c);
            } else {
                // Invariant broken
                my_thread_monitor.cancel_wait();
            }
        }
    }

    my_server.remove_server_ref();
}

inline bool ipc_waker::wake_or_launch() {
    if( my_state==st_init && my_state.compare_and_swap( st_starting, st_init )==st_init ) {
        // after this point, remove_server_ref() must be done by created thread
#if USE_WINTHREAD
        my_handle = ipc_thread_monitor::launch( thread_routine, this, my_server.my_stack_size, &this->my_index );
#elif USE_PTHREAD
        {
        affinity_helper fpa;
        fpa.protect_affinity_mask( /*restore_process_mask=*/true );
        my_handle = ipc_thread_monitor::launch( thread_routine, this, my_server.my_stack_size );
        if( my_handle == 0 ) {
            runtime_warning( "Unable to create new thread for process %d", getpid() );
            state_t s = my_state.compare_and_swap( st_init, st_starting );
            if (st_starting != s) {
                // Do shutdown during startup. my_handle can't be released
                // by start_shutdown, because my_handle value might be not set yet
                // at time of transition from st_starting to st_quit.
                __TBB_ASSERT( s==st_quit, NULL );
                release_handle( my_handle, my_server.my_join_workers );
            }
            return false;
        } else {
            my_server.my_ref_count++;
        }
        // Implicit destruction of fpa resets original affinity mask.
        }
#endif /* USE_PTHREAD */
        state_t s = my_state.compare_and_swap( st_normal, st_starting );
        if( st_starting!=s ) {
            // Do shutdown during startup. my_handle can't be released
            // by start_shutdown, because my_handle value might be not set yet
            // at time of transition from st_starting to st_quit.
            __TBB_ASSERT( s==st_quit, NULL );
            release_handle( my_handle, my_server.my_join_workers );
        }
    }
    else {
        my_thread_monitor.notify();
    }

    return true;
}

//------------------------------------------------------------------------
// Methods of ipc_stopper
//------------------------------------------------------------------------
#if _MSC_VER && !defined(__INTEL_COMPILER)
    // Suppress overzealous compiler warnings about an initialized variable 'sink_for_alloca' not referenced
    #pragma warning(push)
    #pragma warning(disable:4189)
#endif
#if __MINGW32__ && __GNUC__==4 &&__GNUC_MINOR__>=2 && !__MINGW64__
// ensure that stack is properly aligned
__attribute__((force_align_arg_pointer))
#endif
__RML_DECL_THREAD_ROUTINE ipc_stopper::thread_routine(void* arg) {
    ipc_stopper* self = static_cast<ipc_stopper*>(arg);
    AVOID_64K_ALIASING( self->my_index );
    self->run();
    return 0;
}
#if _MSC_VER && !defined(__INTEL_COMPILER)
    #pragma warning(pop)
#endif

void ipc_stopper::run() {
    // Transiting to st_normal here would require setting my_handle,
    // which would create race with the launching thread and
    // complications in handle management on Windows.

    while( my_state!=st_quit ) {
        if( my_server.wait_stop_thread() ) {
            if( my_state!=st_quit ) {
                if( !my_server.stop_one() ) {
                    my_server.add_stop_thread();
                    prolonged_pause();
                }
            }
        }
    }

    my_server.remove_server_ref();
}

inline bool ipc_stopper::wake_or_launch() {
    if( my_state==st_init && my_state.compare_and_swap( st_starting, st_init )==st_init ) {
        // after this point, remove_server_ref() must be done by created thread
#if USE_WINTHREAD
        my_handle = ipc_thread_monitor::launch( thread_routine, this, my_server.my_stack_size, &this->my_index );
#elif USE_PTHREAD
        {
        affinity_helper fpa;
        fpa.protect_affinity_mask( /*restore_process_mask=*/true );
        my_handle = ipc_thread_monitor::launch( thread_routine, this, my_server.my_stack_size );
        if( my_handle == 0 ) {
            runtime_warning( "Unable to create new thread for process %d", getpid() );
            state_t s = my_state.compare_and_swap( st_init, st_starting );
            if (st_starting != s) {
                // Do shutdown during startup. my_handle can't be released
                // by start_shutdown, because my_handle value might be not set yet
                // at time of transition from st_starting to st_quit.
                __TBB_ASSERT( s==st_quit, NULL );
                release_handle( my_handle, my_server.my_join_workers );
            }
            return false;
        } else {
            my_server.my_ref_count++;
        }
        // Implicit destruction of fpa resets original affinity mask.
        }
#endif /* USE_PTHREAD */
        state_t s = my_state.compare_and_swap( st_normal, st_starting );
        if( st_starting!=s ) {
            // Do shutdown during startup. my_handle can't be released
            // by start_shutdown, because my_handle value might be not set yet
            // at time of transition from st_starting to st_quit.
            __TBB_ASSERT( s==st_quit, NULL );
            release_handle( my_handle, my_server.my_join_workers );
        }
    }
    else {
        my_thread_monitor.notify();
    }

    return true;
}

//------------------------------------------------------------------------
// Methods of ipc_server
//------------------------------------------------------------------------
ipc_server::ipc_server(tbb_client& client) :
    my_client( client ),
    my_stack_size( client.min_stack_size() ),
    my_thread_array(NULL),
    my_waker(NULL),
    my_stopper(NULL),
    my_join_workers(false)
{
    my_ref_count = 1;
    my_slack = 0;
#if TBB_USE_ASSERT
    my_net_slack_requests = 0;
#endif /* TBB_USE_ASSERT */
    my_n_thread = get_num_threads(IPC_MAX_THREADS_VAR_NAME);
    if( my_n_thread==0 ) {
        my_n_thread = AvailableHwConcurrency();
        __TBB_ASSERT( my_n_thread>0, NULL );
    }

    my_asleep_list_root = NULL;
    my_thread_array = tbb::cache_aligned_allocator<padded_ipc_worker>().allocate( my_n_thread );
    memset( my_thread_array, 0, sizeof(padded_ipc_worker)*my_n_thread );
    for( size_t i=0; i<my_n_thread; ++i ) {
        ipc_worker* t = new( &my_thread_array[i] ) padded_ipc_worker( *this, client, i );
        t->my_next = my_asleep_list_root;
        my_asleep_list_root = t;
    }

    my_waker = tbb::cache_aligned_allocator<ipc_waker>().allocate(1);
    memset( my_waker, 0, sizeof(ipc_waker) );
    new( my_waker ) ipc_waker( *this, client, my_n_thread );

    my_stopper = tbb::cache_aligned_allocator<ipc_stopper>().allocate(1);
    memset( my_stopper, 0, sizeof(ipc_stopper) );
    new( my_stopper ) ipc_stopper( *this, client, my_n_thread + 1 );

    char* active_sem_name = get_active_sem_name();
    my_active_sem = sem_open( active_sem_name, O_CREAT, IPC_SEM_MODE, my_n_thread - 1 );
    __TBB_ASSERT( my_active_sem, "Unable to open active threads semaphore" );
    delete[] active_sem_name;

    char* stop_sem_name = get_stop_sem_name();
    my_stop_sem = sem_open( stop_sem_name, O_CREAT, IPC_SEM_MODE, 0 );
    __TBB_ASSERT( my_stop_sem, "Unable to open stop threads semaphore" );
    delete[] stop_sem_name;
}

ipc_server::~ipc_server() {
    __TBB_ASSERT( my_net_slack_requests==0, NULL );

    for( size_t i=my_n_thread; i--; )
        my_thread_array[i].~padded_ipc_worker();
    tbb::cache_aligned_allocator<padded_ipc_worker>().deallocate( my_thread_array, my_n_thread );
    tbb::internal::poison_pointer( my_thread_array );

    my_waker->~ipc_waker();
    tbb::cache_aligned_allocator<ipc_waker>().deallocate( my_waker, 1 );
    tbb::internal::poison_pointer( my_waker );

    my_stopper->~ipc_stopper();
    tbb::cache_aligned_allocator<ipc_stopper>().deallocate( my_stopper, 1 );
    tbb::internal::poison_pointer( my_stopper );

    sem_close( my_active_sem );
    sem_close( my_stop_sem );
}

inline bool ipc_server::try_insert_in_asleep_list(ipc_worker& t) {
    asleep_list_mutex_type::scoped_lock lock;
    if( !lock.try_acquire( my_asleep_list_mutex ) )
        return false;
    // Contribute to slack under lock so that if another takes that unit of slack,
    // it sees us sleeping on the list and wakes us up.
    int k = ++my_slack;
    if( k<=0 ) {
        t.my_next = my_asleep_list_root;
        my_asleep_list_root = &t;
        return true;
    } else {
        --my_slack;
        return false;
    }
}

inline bool ipc_server::try_insert_in_asleep_list_forced(ipc_worker& t) {
    asleep_list_mutex_type::scoped_lock lock;
    if( !lock.try_acquire( my_asleep_list_mutex ) )
        return false;
    // Contribute to slack under lock so that if another takes that unit of slack,
    // it sees us sleeping on the list and wakes us up.
    ++my_slack;
    t.my_next = my_asleep_list_root;
    my_asleep_list_root = &t;
    return true;
}

inline bool ipc_server::wait_active_thread() {
    if( sem_wait( my_active_sem ) == 0 ) {
        ++my_global_thread_count;
        return true;
    }
    return false;
}

inline bool ipc_server::try_get_active_thread() {
    if( sem_trywait( my_active_sem ) == 0 ) {
        ++my_global_thread_count;
        return true;
    }
    return false;
}

inline void ipc_server::release_active_thread() {
    release_thread_sem( my_active_sem );
}

inline bool ipc_server::wait_stop_thread() {
    struct timespec ts;
    if( clock_gettime( CLOCK_REALTIME, &ts )==0 ) {
        ts.tv_sec++;
        if( sem_timedwait( my_stop_sem, &ts )==0 ) {
            return true;
        }
    }
    return false;
}

inline void ipc_server::add_stop_thread() {
    sem_post( my_stop_sem );
}

void ipc_server::wake_some( int additional_slack, int active_threads ) {
    __TBB_ASSERT( additional_slack>=0, NULL );
    ipc_worker* wakee[2];
    ipc_worker **w = wakee;
    {
        asleep_list_mutex_type::scoped_lock lock(my_asleep_list_mutex);
        while( active_threads>0 && my_asleep_list_root && w<wakee+2 ) {
            if( additional_slack>0 ) {
                if( additional_slack+my_slack<=0 ) // additional demand does not exceed surplus supply
                    break;
                --additional_slack;
            } else {
                // Chain reaction; Try to claim unit of slack
                int old;
                do {
                    old = my_slack;
                    if( old<=0 ) goto done;
                } while( my_slack.compare_and_swap( old-1, old )!=old );
            }
            // Pop sleeping worker to combine with claimed unit of slack
            my_asleep_list_root = (*w++ = my_asleep_list_root)->my_next;
            --active_threads;
        }
        if( additional_slack ) {
            // Contribute our unused slack to my_slack.
            my_slack += additional_slack;
        }
    }
done:
    while( w>wakee ) {
        if( !(*--w)->wake_or_launch() ) {
            add_stop_thread();
            do {
            } while( !try_insert_in_asleep_list_forced(**w) );
            release_active_thread();
        }
    }
    while( active_threads ) {
        release_active_thread();
        --active_threads;
    }
}

void ipc_server::wake_one_forced( int additional_slack ) {
    __TBB_ASSERT( additional_slack>=0, NULL );
    ipc_worker* wakee[1];
    ipc_worker **w = wakee;
    {
        asleep_list_mutex_type::scoped_lock lock(my_asleep_list_mutex);
        while( my_asleep_list_root && w<wakee+1 ) {
            if( additional_slack>0 ) {
                if( additional_slack+my_slack<=0 ) // additional demand does not exceed surplus supply
                    break;
                --additional_slack;
            } else {
                // Chain reaction; Try to claim unit of slack
                int old;
                do {
                    old = my_slack;
                    if( old<=0 ) goto done;
                } while( my_slack.compare_and_swap( old-1, old )!=old );
            }
            // Pop sleeping worker to combine with claimed unit of slack
            my_asleep_list_root = (*w++ = my_asleep_list_root)->my_next;
        }
        if( additional_slack ) {
            // Contribute our unused slack to my_slack.
            my_slack += additional_slack;
        }
    }
done:
    while( w>wakee ) {
        if( !(*--w)->wake_or_launch() ) {
            add_stop_thread();
            do {
            } while( !try_insert_in_asleep_list_forced(**w) );
        }
    }
}

bool ipc_server::stop_one() {
    ipc_worker* current = NULL;
    ipc_worker* next = NULL;
    {
        asleep_list_mutex_type::scoped_lock lock(my_asleep_list_mutex);
        if( my_asleep_list_root ) {
            current = my_asleep_list_root;
            if( current->my_state==ipc_worker::st_normal ) {
                next = current->my_next;
                while( next!= NULL && next->my_state==ipc_worker::st_normal ) {
                    current = next;
                    next = current->my_next;
                }
                current->start_stopping( my_join_workers );
                return true;
            }
        }
    }
    return false;
}

void ipc_server::adjust_job_count_estimate( int delta ) {
#if TBB_USE_ASSERT
    my_net_slack_requests+=delta;
#endif /* TBB_USE_ASSERT */
    if( my_n_thread > 1 ) {
        if( delta<0 ) {
            my_slack+=delta;
        } else if( delta>0 ) {
            int active_threads = 0;
            if( try_get_active_thread() ) {
                ++active_threads;
                if( try_get_active_thread() ) {
                    ++active_threads;
                }
            }
            wake_some( delta, active_threads );

            if( !my_waker->wake_or_launch() ) {
                add_stop_thread();
            }
            if( !my_stopper->wake_or_launch() ) {
                add_stop_thread();
            }
        }
    } else { // Corner case when RML shouldn't provide any worker thread but client has to have at least one
        if( delta<0 ) {
            my_slack += delta;
        } else {
            wake_one_forced( delta );
        }
    }
}

//------------------------------------------------------------------------
// RML factory methods
//------------------------------------------------------------------------

#if USE_PTHREAD

static tbb_client* my_global_client = NULL;
static tbb_server* my_global_server = NULL;

void rml_atexit() {
    release_resources();
}

void rml_atfork_child() {
    if( my_global_server!=NULL && my_global_client!=NULL ) {
        ipc_server* server = static_cast<ipc_server*>( my_global_server );
        server->~ipc_server();
        memset( server, 0, sizeof(ipc_server) );
        new( server ) ipc_server( *my_global_client );
        pthread_atfork( NULL, NULL, rml_atfork_child );
        atexit( rml_atexit );
    }
}

#endif /* USE_PTHREAD */

extern "C" tbb_factory::status_type __TBB_make_rml_server(tbb_factory& f, tbb_server*& server, tbb_client& client) {
    server = new( tbb::cache_aligned_allocator<ipc_server>().allocate(1) ) ipc_server(client);
#if USE_PTHREAD
    my_global_client = &client;
    my_global_server = server;
    pthread_atfork( NULL, NULL, rml_atfork_child );
    atexit( rml_atexit );
#endif /* USE_PTHREAD */
    if( getenv( "RML_DEBUG" ) ) {
        runtime_warning("IPC server is started");
    }
    return tbb_factory::st_success;
}

extern "C" void __TBB_call_with_my_server_info(::rml::server_info_callback_t cb, void* arg) {
}

} // namespace rml
} // namespace internal

} // namespace tbb
