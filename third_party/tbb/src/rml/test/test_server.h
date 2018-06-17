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

/* This header contains code shared by test_omp_server.cpp and test_tbb_server.cpp 
   There is no ifndef guard - test is supposed to include this file exactly once.
   The test is also exected to have #include of rml_omp.h or rml_tbb.h before 
   including this header. 

   This header should not use any parts of TBB that require linking in the TBB run-time. 
   It uses a few instances of tbb::atomic<T>, all of which are completely inlined. */

#include "tbb/atomic.h"
#include "tbb/tbb_thread.h"
#include "harness.h"
#include "harness_memory.h"
#include "harness_concurrency_tracker.h"

//! Define TRIVIAL as 1 to test only a single client, no nesting, no extra threads.
#define TRIVIAL 0

//! Maximum number of clients 
#if TRIVIAL 
const size_t MaxClient = 1;
#else
const size_t MaxClient = 4;
#endif

const size_t ClientStackSize[MaxClient] = {
    1000000
#if !TRIVIAL
   ,2000000
   ,1000000
   ,4000000
#endif /* TRIVIAL */
};

const size_t OverheadStackSize = 500000;

const size_t JobArraySize = 1000;

static bool TestSingleConnection;

static size_t N_TestConnections;

static int server_concurrency;

class MyJob: public ::rml::job {
public:
    //! Enumeration for tracking states of a job.
    enum state_t {
        //! Job has not yet been allocated.
        unallocated,
        //! Is idle.
        idle,
        //! Has a thread working on it.
        busy,
        //! After call to client::cleanup 
        clean
    };
    tbb::atomic<int> state;
    tbb::atomic<int> processing_count;
    void update( state_t new_state, state_t old_state ) {
        int o = state.compare_and_swap(new_state,old_state);
        ASSERT( o==old_state, "illegal transition" );
    }
    void update_from_either( state_t new_state, state_t old_state1, state_t old_state2 ) {
        int snapshot;
        do {
            snapshot = state;
            ASSERT( snapshot==old_state1||snapshot==old_state2, "illegal transition" );
        } while( state.compare_and_swap(new_state,snapshot)!=snapshot );
    }
    MyJob() {
        state=unallocated;
        processing_count=0;
    }
    ~MyJob() {
        // Overwrite so that accidental use after destruction can be detected.
        memset(this,-1,sizeof(*this));
    }
};

static tbb::atomic<int> ClientConstructions;
static tbb::atomic<int> ClientDestructions;

struct Nesting {
    int level;
    int limit;
    Nesting() : level(0), limit(0) {}
    Nesting( int level_, int limit_ ) : level(level_), limit(limit_) {}
};

template<typename Client>
class ClientBase: public Client {
protected:
    typedef typename Client::size_type size_type;
    typedef typename Client::version_type version_type;
    typedef typename Client::policy_type policy_type;
    typedef typename Client::job job;
private:
    size_type my_max_job_count;
    size_t my_stack_size;
    tbb::atomic<size_t> next_job_index;
    int my_client_id;
    rml::server* my_server;

public:
    enum state_t {
        //! Treat *this as constructed.
        live=0x1234,
        //! Treat *this as destroyed.
        destroyed=0xDEAD
    };

    tbb::atomic<int> state;
    void update( state_t new_state, state_t old_state ) {
        int o = state.compare_and_swap(new_state,old_state);
        ASSERT( o==old_state, NULL );
    }

    tbb::atomic<bool> expect_close_connection;

    MyJob *job_array;

    version_type version() const __TBB_override {
        ASSERT( state==live, NULL );
        return 1;
    }

    size_type max_job_count() const __TBB_override {
        ASSERT( state==live, NULL );
        return my_max_job_count;
    }

    size_t min_stack_size() const __TBB_override {
        ASSERT( state==live, NULL );
        return my_stack_size;
    }

    policy_type policy() const __TBB_override {return Client::throughput;}

    void acknowledge_close_connection() __TBB_override {
        ASSERT( expect_close_connection, NULL );
        for( size_t k=next_job_index; k>0; ) {
            --k;
            ASSERT( job_array[k].state==MyJob::clean, NULL );
        }
        delete[] job_array;
        job_array = NULL;
        ASSERT( my_server, NULL );
        update( destroyed, live );
        delete this;
    }

    void cleanup( job& j_ ) __TBB_override {
        REMARK("client %d: cleanup(%p) called\n",client_id(),&j_);
        ASSERT( state==live, NULL );
        MyJob& j = static_cast<MyJob&>(j_);
        while( j.state==MyJob::busy )
            my_server->yield();
        j.update(MyJob::clean,MyJob::idle);
        REMARK("client %d: cleanup(%p) returns\n",client_id(),&j_);
    }
   
    job* create_one_job();

protected:
    void do_process( job* j_ ) {
        ASSERT( state==live, NULL );
        MyJob& j = static_cast<MyJob&>(*j_);
        ASSERT( j_, NULL );
        j.update(MyJob::busy,MyJob::idle);
        // use of the plain addition (not the atomic increment) is intentonial
        j.processing_count = j.processing_count + 1;
        ASSERT( my_stack_size>OverheadStackSize, NULL ); 
#ifdef __ia64__
        // Half of the stack is reserved for RSE, so test only remaining half.
        UseStackSpace( (my_stack_size-OverheadStackSize)/2 );
#else
        UseStackSpace( my_stack_size-OverheadStackSize );
#endif
        j.update(MyJob::idle,MyJob::busy);
        my_server->yield();
    } 
public:
    ClientBase() : my_server(NULL) {
        my_client_id = ClientConstructions++;
        next_job_index = 0; 
    }
    int client_id() const {return my_client_id;}

    Nesting nesting;

    void initialize( size_type max_job_count, Nesting nesting_, size_t stack_size ) {
        ASSERT( stack_size>0, NULL );
        my_max_job_count = max_job_count;
        nesting = nesting_;
        my_stack_size = stack_size;
        job_array = new MyJob[JobArraySize];
        expect_close_connection = false;
        state = live;
    }

    void set_server( rml::server* s ) {my_server=s;}

    unsigned default_concurrency() const { ASSERT( my_server, NULL); return my_server->default_concurrency(); }

    virtual ~ClientBase() {
        ASSERT( state==destroyed, NULL );
        ++ClientDestructions;
    }
};

template<typename Client>
typename Client::job* ClientBase<Client>::create_one_job() {
    REMARK("client %d: create_one_job() called\n",client_id());
    size_t k = next_job_index++;
    ASSERT( state==live, NULL );
    // Following assertion depends on assumption that implementation does not destroy jobs until 
    // the connection is closed.  If the implementation is changed to destroy jobs sooner, the 
    // test logic in this header will have to be reworked.
    ASSERT( k<my_max_job_count, "RML allocated more than max_job_count jobs simultaneously" );
    ASSERT( k<JobArraySize, "JobArraySize not big enough (problem is in test, not RML)" );
    MyJob& j = job_array[k];
    j.update(MyJob::idle,MyJob::unallocated);
    REMARK("client %d: create_one_job() for k=%d returns %p\n",client_id(),int(k),&j);
    return &j;
}

struct warning_tracker {
    tbb::atomic<int> n_more_than_available;
    tbb::atomic<int> n_too_many_threads;
    tbb::atomic<int> n_system_overload;
    warning_tracker() {
        n_more_than_available = 0;
        n_too_many_threads = 0;
        n_system_overload = 0;
    }
    bool all_set() { return n_more_than_available>0 && n_too_many_threads>0 && n_system_overload>0; }
} tracker;

class Checker {
public:
    int default_concurrency;
    void check_number_of_threads_delivered( int n_delivered, int n_requested, int n_extra ) const;
    Checker( rml::server& server ) : default_concurrency(int(server.default_concurrency())) {}
};

void Checker::check_number_of_threads_delivered( int n_delivered, int n_requested, int n_extra ) const {
    ASSERT( default_concurrency>=0, NULL );
    if( tracker.all_set() ) return;
    // Check that number of threads delivered is reasonable.
    int n_avail = default_concurrency;
    if( n_extra>0 )
        n_avail-=n_extra;
    if( n_avail<0 ) 
        n_avail=0;
    if( n_requested>default_concurrency ) 
        n_avail += n_requested-default_concurrency;
    int n_expected = n_requested;
    if( n_expected>n_avail )
        n_expected=n_avail;
    const char* msg = NULL;
    if( n_delivered>n_avail ) {
        if( ++tracker.n_more_than_available>1 )
            return;
        msg = "server delivered more threads than were theoretically available";
    } else if( n_delivered>n_expected ) {
        if( ++tracker.n_too_many_threads>1 )
            return;
        msg = "server delivered more threads than expected";
    } else if( n_delivered<n_expected ) {
        if( ++tracker.n_system_overload>1 )
            return;
        msg = "server delivered fewer threads than ideal; or, the system is overloaded?";
    }
    if( msg ) {
        REPORT("Warning: %s (n_delivered=%d n_avail=%d n_requested=%d n_extra=%d default_concurrency=%d)\n",
               msg, n_delivered, n_avail, n_requested, n_extra, default_concurrency );
    }
}

template<typename Factory,typename Client>
class DoOneConnection: NoAssign {
    //! Number of threads to request
    const int n_thread;
    //! Nesting 
    const Nesting nesting;
    //! Number of extra threads to pretend having outside the RML
    const int n_extra;
    //! If true, check number of threads actually delivered.
    const bool check_delivered;
public:
    DoOneConnection( int n_thread_, Nesting nesting_, int n_extra_, bool check_delivered_ ) : 
        n_thread(n_thread_), 
        nesting(nesting_), 
        n_extra(n_extra_), 
        check_delivered(check_delivered_)
    {
    }
   
    //! Test ith connection 
    void operator()( size_t i ) const;
};

template<typename Factory,typename Client>
void DoOneConnection<Factory,Client>::operator()( size_t i ) const {
    ASSERT( i<MaxClient, NULL );
    Client* client = new Client;
    client->initialize( Client::is_omp ? JobArraySize : n_thread, nesting, ClientStackSize[i] );
    Factory factory;
    memset( &factory, 0, sizeof(factory) );
    typename Factory::status_type status = factory.open();
    ASSERT( status==Factory::st_success, NULL );

    typename Factory::server_type* server; 
    status = factory.make_server( server, *client );
    ASSERT( status==Factory::st_success, NULL );
    Harness::ConcurrencyTracker ct;
    REMARK("client %d: opened server n_thread=%d nesting=(%d,%d)\n",
               client->client_id(), n_thread, nesting.level, nesting.limit);
    client->set_server( server );
    Checker checker( *server );
 
    FireUpJobs( *server, *client, n_thread, n_extra, check_delivered && !client->is_strict() ? &checker : NULL );

    // Close the connection
    client->expect_close_connection = true;
    REMARK("client %d: calling request_close_connection\n", client->client_id());
#if !RML_USE_WCRM
    int default_concurrency = server->default_concurrency();
#endif
    server->request_close_connection();
    // Client deletes itself when it sees call to acknowledge_close_connection from server.
    factory.close();
#if !RML_USE_WCRM
    if( TestSingleConnection )
        __TBB_ASSERT_EX( uintptr_t(factory.scratch_ptr)==uintptr_t(default_concurrency), "under/over subscription?" );
#endif
}

//! Test with n_threads threads and n_client clients.
template<typename Factory, typename Client>
void SimpleTest() {
    Harness::ConcurrencyTracker::Reset();
    TestSingleConnection = true;
    N_TestConnections = 1;
    for( int n_thread=MinThread; n_thread<=MaxThread; ++n_thread ) {
        // Test a single connection, no nesting, no extra threads
        DoOneConnection<Factory,Client> doc(n_thread,Nesting(0,0),0,false);
        doc(0);
    }
#if !TRIVIAL
    TestSingleConnection = false;
    for( int n_thread=MinThread; n_thread<=MaxThread; ++n_thread ) {
        // Test parallel connections
        for( int n_client=1; n_client<=int(MaxClient); ++n_client ) {
            N_TestConnections = n_client;
            REMARK("SimpleTest: n_thread=%d n_client=%d\n",n_thread,n_client);
            NativeParallelFor( n_client, DoOneConnection<Factory,Client>(n_thread,Nesting(0,0),0,false) );
        }
        // Test server::independent_thread_number_changed
        N_TestConnections = 1;
        for( int n_extra=-4; n_extra<=32; n_extra=n_extra+1+n_extra/5 ) {
            DoOneConnection<Factory,Client> doc(n_thread,Nesting(0,0),n_extra,true);
            doc(0);
        }
#if !RML_USE_WCRM
        // Test nested connections
        DoOneConnection<Factory,Client> doc(n_thread,Nesting(0,2),0,false);
        doc(0);
#endif
    }
    ASSERT( Harness::ConcurrencyTracker::PeakParallelism()>1 || server_concurrency==0, "No multiple connections exercised?" );
#endif /* !TRIVIAL */
    // Let RML catch up.
    while( ClientConstructions!=ClientDestructions )
        Harness::Sleep(1);
}

static void check_server_info( void* arg, const char* server_info )
{
    ASSERT( strstr(server_info, (char*)arg), NULL );
}

template<typename Factory, typename Client>
void VerifyInitialization( int n_thread ) {
    Client* client = new Client;
    client->initialize( Client::is_omp ? JobArraySize : n_thread, Nesting(), ClientStackSize[0] );
    Factory factory;
    memset( &factory, 0, sizeof(factory) );
    typename Factory::status_type status = factory.open();
    ASSERT( status!=Factory::st_not_found, "could not find RML library" );
    ASSERT( status!=Factory::st_incompatible, NULL );
    ASSERT( status==Factory::st_success, NULL );
    factory.call_with_server_info( check_server_info, (void*)"Intel(R) RML library" );
    typename Factory::server_type* server; 
    status = factory.make_server( server, *client );
    ASSERT( status!=Factory::st_incompatible, NULL );
    ASSERT( status!=Factory::st_not_found, NULL );
    ASSERT( status==Factory::st_success, NULL );
    REMARK("client %d: opened server n_thread=%d nesting=(%d,%d)\n",
               client->client_id(), n_thread, 0, 0);
    ASSERT( server, NULL );
    client->set_server( server );
    server_concurrency = server->default_concurrency();

    DoClientSpecificVerification( *server, n_thread );

    // Close the connection
    client->expect_close_connection = true;
    REMARK("client %d: calling request_close_connection\n", client->client_id());
    server->request_close_connection();
    // Client deletes itself when it sees call to acknowledge_close_connection from server.
    factory.close();
}
