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

#include "rml_tbb.h"
#include "../server/thread_monitor.h"
#include "tbb/atomic.h"
#include "tbb/cache_aligned_allocator.h"
#include "scheduler_common.h"
#include "governor.h"
#include "tbb_misc.h"

using rml::internal::thread_monitor;

namespace tbb {
namespace internal {
namespace rml {

typedef thread_monitor::handle_type thread_handle;

class private_server;

class private_worker: no_copy {
private:
    //! State in finite-state machine that controls the worker.
    /** State diagram:
        init --> starting --> normal
          |         |           |
          |         V           |
          \------> quit <------/
      */
    enum state_t {
        //! *this is initialized
        st_init,
        //! *this has associated thread that is starting up.
        st_starting,
        //! Associated thread is doing normal life sequence.
        st_normal,
        //! Associated thread has ended normal life sequence and promises to never touch *this again.
        st_quit
    };
    atomic<state_t> my_state;

    //! Associated server
    private_server& my_server;

    //! Associated client
    tbb_client& my_client;

    //! index used for avoiding the 64K aliasing problem
    const size_t my_index;

    //! Monitor for sleeping when there is no work to do.
    /** The invariant that holds for sleeping workers is:
        "my_slack<=0 && my_state==st_normal && I am on server's list of asleep threads" */
    thread_monitor my_thread_monitor;

    //! Handle of the OS thread associated with this worker
    thread_handle my_handle;

    //! Link for list of workers that are sleeping or have no associated thread.
    private_worker* my_next;

    friend class private_server;

    //! Actions executed by the associated thread
    void run();

    //! Wake up associated thread (or launch a thread if there is none)
    void wake_or_launch();

    //! Called by a thread (usually not the associated thread) to commence termination.
    void start_shutdown();

    static __RML_DECL_THREAD_ROUTINE thread_routine( void* arg );

    static void release_handle(thread_handle my_handle, bool join);

protected:
    private_worker( private_server& server, tbb_client& client, const size_t i ) :
        my_server(server), my_client(client), my_index(i),
        my_thread_monitor(), my_handle(), my_next()
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
class padded_private_worker: public private_worker {
    char pad[cache_line_size - sizeof(private_worker)%cache_line_size];
public:
    padded_private_worker( private_server& server, tbb_client& client, const size_t i )
    : private_worker(server,client,i) { suppress_unused_warning(pad); }
};
#if _MSC_VER && !defined(__INTEL_COMPILER)
    #pragma warning(pop)
#endif

class private_server: public tbb_server, no_copy {
private:
    tbb_client& my_client;
    //! Maximum number of threads to be created.
    /** Threads are created lazily, so maximum might not actually be reached. */
    const tbb_client::size_type my_n_thread;

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

    padded_private_worker* my_thread_array;

    //! List of workers that are asleep or committed to sleeping until notified by another thread.
    tbb::atomic<private_worker*> my_asleep_list_root;

    //! Protects my_asleep_list_root
    typedef scheduler_mutex_type asleep_list_mutex_type;
    asleep_list_mutex_type my_asleep_list_mutex;

#if TBB_USE_ASSERT
    atomic<int> my_net_slack_requests;
#endif /* TBB_USE_ASSERT */

    //! Wake up to two sleeping workers, if there are any sleeping.
    /** The call is used to propagate a chain reaction where each thread wakes up two threads,
        which in turn each wake up two threads, etc. */
    void propagate_chain_reaction() {
        // First test of a double-check idiom.  Second test is inside wake_some(0).
        if( my_asleep_list_root )
            wake_some(0);
    }

    //! Try to add t to list of sleeping workers
    bool try_insert_in_asleep_list( private_worker& t );

    //! Equivalent of adding additional_slack to my_slack and waking up to 2 threads if my_slack permits.
    void wake_some( int additional_slack );

    virtual ~private_server();

    void remove_server_ref() {
        if( --my_ref_count==0 ) {
            my_client.acknowledge_close_connection();
            this->~private_server();
            tbb::cache_aligned_allocator<private_server>().deallocate( this, 1 );
        }
    }

    friend class private_worker;
public:
    private_server( tbb_client& client );

    version_type version() const __TBB_override {
        return 0;
    }

    void request_close_connection( bool /*exiting*/ ) __TBB_override {
        for( size_t i=0; i<my_n_thread; ++i )
            my_thread_array[i].start_shutdown();
        remove_server_ref();
    }

    void yield() __TBB_override {__TBB_Yield();}

    void independent_thread_number_changed( int ) __TBB_override {__TBB_ASSERT(false,NULL);}

    unsigned default_concurrency() const __TBB_override { return governor::default_num_threads() - 1; }

    void adjust_job_count_estimate( int delta ) __TBB_override;

#if _WIN32||_WIN64
    void register_master ( ::rml::server::execution_resource_t& ) __TBB_override {}
    void unregister_master ( ::rml::server::execution_resource_t ) __TBB_override {}
#endif /* _WIN32||_WIN64 */
};

//------------------------------------------------------------------------
// Methods of private_worker
//------------------------------------------------------------------------
#if _MSC_VER && !defined(__INTEL_COMPILER)
    // Suppress overzealous compiler warnings about an initialized variable 'sink_for_alloca' not referenced
    #pragma warning(push)
    #pragma warning(disable:4189)
#endif
#if __MINGW32__ && __GNUC__==4 &&__GNUC_MINOR__>=2 && !__MINGW64__
// ensure that stack is properly aligned for TBB threads
__attribute__((force_align_arg_pointer))
#endif
__RML_DECL_THREAD_ROUTINE private_worker::thread_routine( void* arg ) {
    private_worker* self = static_cast<private_worker*>(arg);
    AVOID_64K_ALIASING( self->my_index );
    self->run();
    return 0;
}
#if _MSC_VER && !defined(__INTEL_COMPILER)
    #pragma warning(pop)
#endif

void private_worker::release_handle(thread_handle handle, bool join) {
    if (join)
        thread_monitor::join(handle);
    else
        thread_monitor::detach_thread(handle);
}

void private_worker::start_shutdown() {
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
        if (s==st_normal)
            release_handle(my_handle, governor::does_client_join_workers(my_client));
    } else if( s==st_init ) {
        // Perform action that otherwise would be performed by associated thread when it quits.
        my_server.remove_server_ref();
    }
}

void private_worker::run() {
    my_server.propagate_chain_reaction();

    // Transiting to st_normal here would require setting my_handle,
    // which would create race with the launching thread and
    // complications in handle management on Windows.

    ::rml::job& j = *my_client.create_one_job();
    while( my_state!=st_quit ) {
        if( my_server.my_slack>=0 ) {
            my_client.process(j);
        } else {
            thread_monitor::cookie c;
            // Prepare to wait
            my_thread_monitor.prepare_wait(c);
            // Check/set the invariant for sleeping
            if( my_state!=st_quit && my_server.try_insert_in_asleep_list(*this) ) {
                my_thread_monitor.commit_wait(c);
                my_server.propagate_chain_reaction();
            } else {
                // Invariant broken
                my_thread_monitor.cancel_wait();
            }
        }
    }
    my_client.cleanup(j);

    ++my_server.my_slack;
    my_server.remove_server_ref();
}

inline void private_worker::wake_or_launch() {
    if( my_state==st_init && my_state.compare_and_swap( st_starting, st_init )==st_init ) {
        // after this point, remove_server_ref() must be done by created thread
#if USE_WINTHREAD
        my_handle = thread_monitor::launch( thread_routine, this, my_server.my_stack_size, &this->my_index );
#elif USE_PTHREAD
        {
        affinity_helper fpa;
        fpa.protect_affinity_mask( /*restore_process_mask=*/true );
        my_handle = thread_monitor::launch( thread_routine, this, my_server.my_stack_size );
        // Implicit destruction of fpa resets original affinity mask.
        }
#endif /* USE_PTHREAD */
        state_t s = my_state.compare_and_swap( st_normal, st_starting );
        if (st_starting != s) {
            // Do shutdown during startup. my_handle can't be released
            // by start_shutdown, because my_handle value might be not set yet
            // at time of transition from st_starting to st_quit.
            __TBB_ASSERT( s==st_quit, NULL );
            release_handle(my_handle, governor::does_client_join_workers(my_client));
        }
    }
    else
        my_thread_monitor.notify();
}

//------------------------------------------------------------------------
// Methods of private_server
//------------------------------------------------------------------------
private_server::private_server( tbb_client& client ) :
    my_client(client),
    my_n_thread(client.max_job_count()),
    my_stack_size(client.min_stack_size()),
    my_thread_array(NULL)
{
    my_ref_count = my_n_thread+1;
    my_slack = 0;
#if TBB_USE_ASSERT
    my_net_slack_requests = 0;
#endif /* TBB_USE_ASSERT */
    my_asleep_list_root = NULL;
    my_thread_array = tbb::cache_aligned_allocator<padded_private_worker>().allocate( my_n_thread );
    for( size_t i=0; i<my_n_thread; ++i ) {
        private_worker* t = new( &my_thread_array[i] ) padded_private_worker( *this, client, i );
        t->my_next = my_asleep_list_root;
        my_asleep_list_root = t;
    }
}

private_server::~private_server() {
    __TBB_ASSERT( my_net_slack_requests==0, NULL );
    for( size_t i=my_n_thread; i--; )
        my_thread_array[i].~padded_private_worker();
    tbb::cache_aligned_allocator<padded_private_worker>().deallocate( my_thread_array, my_n_thread );
    tbb::internal::poison_pointer( my_thread_array );
}

inline bool private_server::try_insert_in_asleep_list( private_worker& t ) {
    asleep_list_mutex_type::scoped_lock lock;
    if( !lock.try_acquire(my_asleep_list_mutex) )
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

void private_server::wake_some( int additional_slack ) {
    __TBB_ASSERT( additional_slack>=0, NULL );
    private_worker* wakee[2];
    private_worker**w = wakee;
    {
        asleep_list_mutex_type::scoped_lock lock(my_asleep_list_mutex);
        while( my_asleep_list_root && w<wakee+2 ) {
            if( additional_slack>0 ) {
                if (additional_slack+my_slack<=0) // additional demand does not exceed surplus supply
                    break;
                --additional_slack;
            } else {
                // Chain reaction; Try to claim unit of slack
                int old;
                do {
                    old = my_slack;
                    if( old<=0 ) goto done;
                } while( my_slack.compare_and_swap(old-1,old)!=old );
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
    while( w>wakee )
        (*--w)->wake_or_launch();
}

void private_server::adjust_job_count_estimate( int delta ) {
#if TBB_USE_ASSERT
    my_net_slack_requests+=delta;
#endif /* TBB_USE_ASSERT */
    if( delta<0 ) {
        my_slack+=delta;
    } else if( delta>0 ) {
        wake_some( delta );
    }
}

//! Factory method called from task.cpp to create a private_server.
tbb_server* make_private_server( tbb_client& client ) {
    return new( tbb::cache_aligned_allocator<private_server>().allocate(1) ) private_server(client);
}

} // namespace rml
} // namespace internal

} // namespace tbb
