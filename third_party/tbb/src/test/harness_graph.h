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

/** @file harness_graph.cpp
    This contains common helper classes and functions for testing graph nodes
**/

#ifndef harness_graph_H
#define harness_graph_H

#include "harness.h"
#include "tbb/flow_graph.h"
#include "tbb/null_rw_mutex.h"
#include "tbb/atomic.h"
#include "tbb/concurrent_unordered_map.h"
#include "tbb/task.h"
#include "tbb/task_scheduler_init.h"

using tbb::flow::internal::SUCCESSFULLY_ENQUEUED;

#define WAIT_MAX 2000000
#define BACKOFF_WAIT(ex,msg) \
{ \
    int wait_cnt = 0; \
    tbb::internal::atomic_backoff backoff; \
    do { \
        backoff.pause(); \
        ++wait_cnt; \
    } \
    while( (ex) && (wait_cnt < WAIT_MAX)); \
    ASSERT(wait_cnt < WAIT_MAX, msg); \
}
#define BACKOFF_WAIT_NOASSERT(ex,msg) \
{ \
    int wait_cnt = 0; \
    tbb::internal::atomic_backoff backoff; \
    do { \
        backoff.pause(); \
        ++wait_cnt; \
    } \
    while( (ex) && (wait_cnt < WAIT_MAX)); \
    if(wait_cnt >= WAIT_MAX) REMARK("%s\n",msg); \
}

// Needed conversion to and from continue_msg, but didn't want to add
// conversion operators to the class, since we don't want it in general,
// only in these tests.
template<typename InputType, typename OutputType>
struct convertor {
    static OutputType convert_value(const InputType &i) {
        return OutputType(i);
    }
};

template<typename InputType>
struct convertor<InputType,tbb::flow::continue_msg> {
    static tbb::flow::continue_msg convert_value(const InputType &/*i*/) {
        return tbb::flow::continue_msg();
    }
};

template<typename OutputType>
struct convertor<tbb::flow::continue_msg,OutputType> {
    static OutputType convert_value(const tbb::flow::continue_msg &/*i*/) {
        return OutputType();
    }
};

// helper for multifunction_node tests.
template<size_t N>
struct mof_helper {
    template<typename InputType, typename ports_type>
    static inline void output_converted_value(const InputType &i, ports_type &p) {
        (void)tbb::flow::get<N-1>(p).try_put(convertor<InputType,typename tbb::flow::tuple_element<N-1,ports_type>::type::output_type>::convert_value(i));
        output_converted_value<N-1>(i, p);
    }
};

template<>
struct mof_helper<1> {
    template<typename InputType, typename ports_type>
    static inline void output_converted_value(const InputType &i, ports_type &p) {
        // just emit a default-constructed object
        (void)tbb::flow::get<0>(p).try_put(convertor<InputType,typename tbb::flow::tuple_element<0,ports_type>::type::output_type>::convert_value(i));
    }
};

template< typename InputType, typename OutputType >
struct harness_graph_default_functor {
    static OutputType construct( InputType v ) {
        return OutputType(v);
    }
};

template< typename OutputType >
struct harness_graph_default_functor< tbb::flow::continue_msg, OutputType > {
    static OutputType construct( tbb::flow::continue_msg ) {
        return OutputType();
    }
};

template< typename InputType >
struct harness_graph_default_functor< InputType, tbb::flow::continue_msg > {
    static tbb::flow::continue_msg construct( InputType ) {
        return tbb::flow::continue_msg();
    }
};

template< >
struct harness_graph_default_functor< tbb::flow::continue_msg, tbb::flow::continue_msg > {
    static tbb::flow::continue_msg construct( tbb::flow::continue_msg ) {
        return tbb::flow::continue_msg();
    }
};

template<typename InputType, typename OutputSet>
struct harness_graph_default_multifunction_functor {
    static const int N = tbb::flow::tuple_size<OutputSet>::value;
    typedef typename tbb::flow::multifunction_node<InputType,OutputSet>::output_ports_type ports_type;
    static void construct(const InputType &i, ports_type &p) {
        mof_helper<N>::output_converted_value(i, p);
    }
};

//! An executor that accepts InputType and generates OutputType
template< typename InputType, typename OutputType >
struct harness_graph_executor {

    typedef OutputType (*function_ptr_type)( InputType v );

    template<typename RW>
    struct mutex_holder { static RW mutex; };

    static function_ptr_type fptr;
    static tbb::atomic<size_t> execute_count;
    static tbb::atomic<size_t> current_executors;
    static size_t max_executors;

    static inline OutputType func( InputType v ) {
        size_t c; // Declaration separate from initialization to avoid ICC internal error on IA-64 architecture
        c = current_executors.fetch_and_increment();
        ASSERT( max_executors == 0 || c <= max_executors, NULL );
        ++execute_count;
        OutputType v2 = (*fptr)(v);
        current_executors.fetch_and_decrement();
        return v2;
    }

    template< typename RW >
    static inline OutputType tfunc( InputType v ) {
        // Invocations allowed to be concurrent, the lock is acquired in shared ("read") mode.
        // A test can take it exclusively, thus creating a barrier for invocations.
        typename RW::scoped_lock l( mutex_holder<RW>::mutex, /*write=*/false );
        return func(v);
    }

    template< typename RW >
    struct tfunctor {
        tbb::atomic<size_t> my_execute_count;
        tfunctor() { my_execute_count = 0; }
        tfunctor( const tfunctor &f ) { my_execute_count = f.my_execute_count; }
        OutputType operator()( InputType i ) {
           typename RW::scoped_lock l( harness_graph_executor::mutex_holder<RW>::mutex, /*write=*/false );
           my_execute_count.fetch_and_increment();
           return harness_graph_executor::func(i);
        }
    };
    typedef tfunctor<tbb::null_rw_mutex> functor;

};

//! A multifunction executor that accepts InputType and has only one Output of OutputType.
template< typename InputType, typename OutputTuple >
struct harness_graph_multifunction_executor {
    typedef typename tbb::flow::multifunction_node<InputType,OutputTuple>::output_ports_type ports_type;
    typedef typename tbb::flow::tuple_element<0,OutputTuple>::type OutputType;

    typedef void (*mfunction_ptr_type)( const InputType& v, ports_type &p );

    template<typename RW>
    struct mutex_holder { static RW mutex; };

    static mfunction_ptr_type fptr;
    static tbb::atomic<size_t> execute_count;
    static tbb::atomic<size_t> current_executors;
    static size_t max_executors;

    static inline void empty_func( const InputType&, ports_type& ) {
    }

    static inline void func( const InputType &v, ports_type &p ) {
        size_t c; // Declaration separate from initialization to avoid ICC internal error on IA-64 architecture
        c = current_executors.fetch_and_increment();
        ASSERT( max_executors == 0 || c <= max_executors, NULL );
        ASSERT(tbb::flow::tuple_size<OutputTuple>::value == 1, NULL);
        ++execute_count;
        (*fptr)(v,p);
        current_executors.fetch_and_decrement();
    }

    template< typename RW >
    static inline void tfunc( const InputType& v, ports_type &p ) {
        // Shared lock in invocations, exclusive in a test; see a comment in harness_graph_executor.
        typename RW::scoped_lock l( mutex_holder<RW>::mutex, /*write=*/false );
        func(v,p);
    }

    template< typename RW >
    struct tfunctor {
        tbb::atomic<size_t> my_execute_count;
        tfunctor() { my_execute_count = 0; }
        tfunctor( const tfunctor &f ) { my_execute_count = f.my_execute_count; }
        void operator()( const InputType &i, ports_type &p ) {
           typename RW::scoped_lock l( harness_graph_multifunction_executor::mutex_holder<RW>::mutex, /*write=*/false );
           my_execute_count.fetch_and_increment();
           harness_graph_multifunction_executor::func(i,p);
        }
    };
    typedef tfunctor<tbb::null_rw_mutex> functor;

};

// static vars for function_node tests
template< typename InputType, typename OutputType >
template< typename RW >
RW harness_graph_executor<InputType, OutputType>::mutex_holder<RW>::mutex;

template< typename InputType, typename OutputType >
tbb::atomic<size_t> harness_graph_executor<InputType, OutputType>::execute_count;

template< typename InputType, typename OutputType >
typename harness_graph_executor<InputType, OutputType>::function_ptr_type harness_graph_executor<InputType, OutputType>::fptr
    = harness_graph_default_functor< InputType, OutputType >::construct;

template< typename InputType, typename OutputType >
tbb::atomic<size_t> harness_graph_executor<InputType, OutputType>::current_executors;

template< typename InputType, typename OutputType >
size_t harness_graph_executor<InputType, OutputType>::max_executors = 0;

// static vars for multifunction_node tests
template< typename InputType, typename OutputTuple >
template< typename RW >
RW harness_graph_multifunction_executor<InputType, OutputTuple>::mutex_holder<RW>::mutex;

template< typename InputType, typename OutputTuple >
tbb::atomic<size_t> harness_graph_multifunction_executor<InputType, OutputTuple>::execute_count;

template< typename InputType, typename OutputTuple >
typename harness_graph_multifunction_executor<InputType, OutputTuple>::mfunction_ptr_type harness_graph_multifunction_executor<InputType, OutputTuple>::fptr
    = harness_graph_default_multifunction_functor< InputType, OutputTuple >::construct;

template< typename InputType, typename OutputTuple >
tbb::atomic<size_t> harness_graph_multifunction_executor<InputType, OutputTuple>::current_executors;

template< typename InputType, typename OutputTuple >
size_t harness_graph_multifunction_executor<InputType, OutputTuple>::max_executors = 0;

//! Counts the number of puts received
template< typename T >
struct harness_counting_receiver : public tbb::flow::receiver<T> {

    tbb::atomic< size_t > my_count;
    T max_value;
    size_t num_copies;
    tbb::flow::graph& my_graph;

    harness_counting_receiver(tbb::flow::graph& g) : num_copies(1), my_graph(g) {
       my_count = 0;
    }

    void initialize_map( const T& m, size_t c ) {
       my_count = 0;
       max_value = m;
       num_copies = c;
    }

    tbb::flow::graph& graph_reference() __TBB_override {
        return my_graph;
    }

    tbb::task *try_put_task( const T & ) __TBB_override {
      ++my_count;
      return const_cast<tbb::task *>(SUCCESSFULLY_ENQUEUED);
    }

    void validate() {
        size_t n = my_count;
        ASSERT( n == num_copies*max_value, NULL );
    }

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    typedef typename tbb::flow::receiver<T>::built_predecessors_type built_predecessors_type;
    built_predecessors_type mbp;
    built_predecessors_type &built_predecessors() __TBB_override { return mbp; }
    typedef typename tbb::flow::receiver<T>::predecessor_list_type predecessor_list_type;
    typedef typename tbb::flow::receiver<T>::predecessor_type predecessor_type;
    void internal_add_built_predecessor(predecessor_type &) __TBB_override {}
    void internal_delete_built_predecessor(predecessor_type &) __TBB_override {}
    void copy_predecessors(predecessor_list_type &) __TBB_override { }
    size_t predecessor_count() __TBB_override { return 0; }
#endif
    void reset_receiver(tbb::flow::reset_flags /*f*/) __TBB_override { my_count = 0; }
};

//! Counts the number of puts received
template< typename T >
struct harness_mapped_receiver : public tbb::flow::receiver<T>, NoCopy {

    tbb::atomic< size_t > my_count;
    T max_value;
    size_t num_copies;
    typedef tbb::concurrent_unordered_map< T, tbb::atomic< size_t > > map_type;
    map_type *my_map;
    tbb::flow::graph& my_graph;

    harness_mapped_receiver(tbb::flow::graph& g) : my_map(NULL), my_graph(g) {
       my_count = 0;
    }

    ~harness_mapped_receiver() {
        if ( my_map ) delete my_map;
    }

    void initialize_map( const T& m, size_t c ) {
       my_count = 0;
       max_value = m;
       num_copies = c;
       if ( my_map ) delete my_map;
       my_map = new map_type;
    }

    tbb::task * try_put_task( const T &t ) __TBB_override {
      if ( my_map ) {
          tbb::atomic<size_t> a;
          a = 1;
          std::pair< typename map_type::iterator, bool > r =  (*my_map).insert( typename map_type::value_type( t, a ) );
          if ( r.second == false ) {
              size_t v = r.first->second.fetch_and_increment();
              ASSERT( v < num_copies, NULL );
          }
      } else {
          ++my_count;
      }
      return const_cast<tbb::task *>(SUCCESSFULLY_ENQUEUED);
    }

    tbb::flow::graph& graph_reference() __TBB_override {
        return my_graph;
    }

    void validate() {
        if ( my_map ) {
            for ( size_t i = 0; i < (size_t)max_value; ++i ) {
                size_t n = (*my_map)[(int)i];
                ASSERT( n == num_copies, NULL );
            }
        } else {
            size_t n = my_count;
            ASSERT( n == num_copies*max_value, NULL );
        }
    }
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    typedef typename tbb::flow::receiver<T>::built_predecessors_type built_predecessors_type;
    built_predecessors_type mbp;
    built_predecessors_type &built_predecessors() __TBB_override { return mbp; }
    typedef typename tbb::flow::receiver<T>::predecessor_list_type predecessor_list_type;
    typedef typename tbb::flow::receiver<T>::predecessor_type predecessor_type;
    void internal_add_built_predecessor(predecessor_type &) __TBB_override {}
    void internal_delete_built_predecessor(predecessor_type &) __TBB_override {}
    void copy_predecessors(predecessor_list_type &) __TBB_override { }
    size_t predecessor_count() __TBB_override { return 0; }
#endif
    void reset_receiver(tbb::flow::reset_flags /*f*/) __TBB_override {
        my_count = 0;
        if(my_map) delete my_map;
        my_map = new map_type;
    }

};

//! Counts the number of puts received
template< typename T >
struct harness_counting_sender : public tbb::flow::sender<T>, NoCopy {

    typedef typename tbb::flow::sender<T>::successor_type successor_type;
    tbb::atomic< successor_type * > my_receiver;
    tbb::atomic< size_t > my_count;
    tbb::atomic< size_t > my_received;
    size_t my_limit;

    harness_counting_sender( ) : my_limit(~size_t(0)) {
       my_receiver = NULL;
       my_count = 0;
       my_received = 0;
    }

    harness_counting_sender( size_t limit ) : my_limit(limit) {
       my_receiver = NULL;
       my_count = 0;
       my_received = 0;
    }

    bool register_successor( successor_type &r ) __TBB_override {
        my_receiver = &r;
        return true;
    }

    bool remove_successor( successor_type &r ) __TBB_override {
        successor_type *s = my_receiver.fetch_and_store( NULL );
        ASSERT( s == &r, NULL );
        return true;
    }

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    typedef typename tbb::flow::sender<T>::successor_list_type successor_list_type;
    typedef typename tbb::flow::sender<T>::built_successors_type built_successors_type;
    built_successors_type bst;
    built_successors_type &built_successors() __TBB_override { return bst; }
    void internal_add_built_successor( successor_type &) __TBB_override {}
    void internal_delete_built_successor( successor_type &) __TBB_override {}
    void copy_successors(successor_list_type &) __TBB_override { }
    size_t successor_count() __TBB_override { return 0; }
#endif

    bool try_get( T & v ) __TBB_override {
        size_t i = my_count.fetch_and_increment();
        if ( i < my_limit ) {
           v = T( i );
           ++my_received;
           return true;
        } else {
           return false;
        }
    }

    bool try_put_once() {
        successor_type *s = my_receiver;
        size_t i = my_count.fetch_and_increment();
        if ( s->try_put( T(i) ) ) {
            ++my_received;
            return true;
        } else {
            return false;
        }
    }

    void try_put_until_false() {
        successor_type *s = my_receiver;
        size_t i = my_count.fetch_and_increment();

        while ( s->try_put( T(i) ) ) {
            ++my_received;
            i = my_count.fetch_and_increment();
        }
    }

    void try_put_until_limit() {
        successor_type *s = my_receiver;

        for ( int i = 0; i < (int)my_limit; ++i ) {
            ASSERT( s->try_put( T(i) ), NULL );
            ++my_received;
        }
        ASSERT( my_received == my_limit, NULL );
    }

};

// test for resets of buffer-type nodes.
tbb::atomic<int> serial_fn_state0;
tbb::atomic<int> serial_fn_state1;
tbb::atomic<int> serial_continue_state0;

template<typename T>
struct serial_fn_body {
    tbb::atomic<int> *_flag;
    serial_fn_body(tbb::atomic<int> &myatomic) : _flag(&myatomic) { }
    T operator()(const T& in) {
        if(*_flag == 0) {
            *_flag = 1;
            // wait until we are released
            tbb::internal::atomic_backoff backoff;
            do {
                backoff.pause();
            } while(*_flag == 1);
        }
        // return value
        return in;
    }
};

template<typename T>
struct serial_continue_body {
    tbb::atomic<int> *_flag;
    serial_continue_body(tbb::atomic<int> &myatomic) : _flag(&myatomic) {}
    T operator()(const tbb::flow::continue_msg& /*in*/) {
        // signal we have received a value
        *_flag = 1;
        // wait until we are released
        tbb::internal::atomic_backoff backoff;
        do {
            backoff.pause();
        } while(*_flag == 1);
        // return value
        return (T)1;
    }
};

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES


// walk two lists via iterator, match elements of each, in possibly-different ordder, and
// return true if all elements of sv appear in tv.
template<typename SV, typename TV>
bool lists_match(SV &sv, TV &tv) {
    if(sv.size() != tv.size()) {
        return false;
    }
    std::vector<bool> bv(sv.size(), false);
    for(typename TV::iterator itv = tv.begin(); itv != tv.end(); ++itv) {
        int ibv = 0;
        for(typename SV::iterator isv = sv.begin(); isv != sv.end(); ++isv) {
            if(!bv[ibv]) {
                if(*itv == *isv) {
                    bv[ibv] = true;
                    goto found_it;;
                }
            }
            ++ibv;
        }
        return false;
found_it:
        continue;
    }
    return true;
}
#endif  /* TBB_PREVIEW_FLOW_GRAPH_FEATURES */

template<typename T, typename BufferType>
void test_resets() {
    const int NN = 3;
    tbb::task_scheduler_init init(4);
    tbb::task_group_context   tgc;
    tbb::flow::graph          g(tgc);
    BufferType                b0(g);
    tbb::flow::queue_node<T>  q0(g);
    T j;
    bool nFound[NN];

    // reset empties buffer
    for(T i = 0; i < NN; ++i) {
        b0.try_put(i);
        nFound[(int)i] = false;
    }
    g.wait_for_all();
    g.reset();
    ASSERT(!b0.try_get(j), "reset did not empty buffer");

    // reset doesn't delete edge

    tbb::flow::make_edge(b0,q0);
    g.reset();
    for(T i = 0; i < NN; ++i) {
        b0.try_put(i);
    }

    g.wait_for_all();
    for( T i = 0; i < NN; ++i) {
        ASSERT(q0.try_get(j), "Missing value from buffer");
        ASSERT(!nFound[(int)j], "Duplicate value found");
        nFound[(int)j] = true;
    }

    for(int ii = 0; ii < NN; ++ii) {
        ASSERT(nFound[ii], "missing value");
    }
    ASSERT(!q0.try_get(j), "Extra values in output");

    // reset reverses a reversed edge.
    // we will use a serial rejecting node to get the edge to reverse.
    tbb::flow::function_node<T, T, tbb::flow::rejecting> sfn(g, tbb::flow::serial, serial_fn_body<T>(serial_fn_state0));
    tbb::flow::queue_node<T> outq(g);
    tbb::flow::remove_edge(b0,q0);
    tbb::flow::make_edge(b0, sfn);
    tbb::flow::make_edge(sfn,outq);
    g.wait_for_all();  // wait for all the tasks started by building the graph are done.
    serial_fn_state0 = 0;

    // b0 ------> sfn ------> outq

    for(int icnt = 0; icnt < 2; ++icnt) {
        g.wait_for_all();
        serial_fn_state0 = 0;
        b0.try_put((T)0);  // will start sfn
        // wait until function_node starts
        BACKOFF_WAIT(serial_fn_state0 == 0,"Timed out waiting for function_node to start");
        // now the function_node is executing.
        // this will start a task to forward the second item
        // to the serial function node
        b0.try_put((T)1);  // first item will be consumed by task completing the execution
        BACKOFF_WAIT_NOASSERT(g.root_task()->ref_count() >= 3,"Timed out waiting try_put task to wind down");
        b0.try_put((T)2);  // second item will remain after cancellation
        // now wait for the task that attempts to forward the buffer item to
        // complete.
        BACKOFF_WAIT_NOASSERT(g.root_task()->ref_count() >= 3,"Timed out waiting for tasks to wind down");
        // now cancel the graph.
        ASSERT(tgc.cancel_group_execution(), "task group already cancelled");
        serial_fn_state0 = 0;  // release the function_node.
        g.wait_for_all();  // wait for all the tasks to complete.
        // check that at most one output reached the queue_node
        T outt;
        T outt2;
        bool got_item1 = outq.try_get(outt);
        bool got_item2 = outq.try_get(outt2);
        // either the output queue was empty (if the function_node tested for cancellation before putting the
        // result to the queue) or there was one element in the queue (the 0).
        ASSERT(!got_item1 || ((int)outt == 0 && !got_item2), "incorrect output from function_node");
        // the edge between the buffer and the function_node should be reversed, and the last
        // message we put in the buffer should still be there.  We can't directly test for the
        // edge reversal.
        got_item1 = b0.try_get(outt);
        ASSERT(got_item1, " buffer lost a message");
        ASSERT(2 == (int)outt || 1 == (int)outt, " buffer had incorrect message");  // the one not consumed by the node.
        ASSERT(g.is_cancelled(), "Graph was not cancelled");
        g.reset();
    }  // icnt

    // reset with remove_edge removes edge.  (icnt ==0 => forward edge, 1 => reversed edge
    for(int icnt = 0; icnt < 2; ++icnt) {
        if(icnt == 1) {
            // set up reversed edge
            tbb::flow::make_edge(b0, sfn);
            tbb::flow::make_edge(sfn,outq);
            serial_fn_state0 = 0;
            b0.try_put((T)0);  // starts up the function node
            b0.try_put((T)1);  // shoyuld reverse the edge
            BACKOFF_WAIT(serial_fn_state0 == 0,"Timed out waiting for edge reversal");
            ASSERT(tgc.cancel_group_execution(), "task group already cancelled");
            serial_fn_state0 = 0;  // release the function_node.
            g.wait_for_all();  // wait for all the tasks to complete.
        }
        g.reset(tbb::flow::rf_clear_edges);
        // test that no one is a successor to the buffer now.
        serial_fn_state0 = 1;  // let the function_node go if it gets an input message
        b0.try_put((T)23);
        g.wait_for_all();
        ASSERT((int)serial_fn_state0 == 1, "function_node executed when it shouldn't");
        T outt;
        ASSERT(b0.try_get(outt) && (T)23 == outt, "node lost its input");
    }
}

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES

template< typename NODE_TYPE >
class test_buffer_base_extract {
protected:
    tbb::flow::graph &g;
    NODE_TYPE &in0;
    NODE_TYPE &in1;
    NODE_TYPE &middle;
    NODE_TYPE &out0;
    NODE_TYPE &out1;
    NODE_TYPE *ins[2];
    NODE_TYPE *outs[2];
    typename NODE_TYPE::successor_type *ms_ptr;
    typename NODE_TYPE::predecessor_type *mp_ptr;

    typename NODE_TYPE::predecessor_list_type in0_p_list;
    typename NODE_TYPE::successor_list_type in0_s_list;
    typename NODE_TYPE::predecessor_list_type in1_p_list;
    typename NODE_TYPE::successor_list_type in1_s_list;
    typename NODE_TYPE::predecessor_list_type out0_p_list;
    typename NODE_TYPE::successor_list_type out0_s_list;
    typename NODE_TYPE::predecessor_list_type out1_p_list;
    typename NODE_TYPE::successor_list_type out1_s_list;
    typename NODE_TYPE::predecessor_list_type mp_list;
    typename NODE_TYPE::predecessor_list_type::iterator mp_list_iter;
    typename NODE_TYPE::successor_list_type ms_list;
    typename NODE_TYPE::successor_list_type::iterator ms_list_iter;

    virtual void set_up_lists() {
        in0_p_list.clear();
        in0_s_list.clear();
        in1_p_list.clear();
        in1_s_list.clear();
        mp_list.clear();
        ms_list.clear();
        out0_p_list.clear();
        out0_s_list.clear();
        out1_p_list.clear();
        out1_s_list.clear();
        in0.copy_predecessors(in0_p_list);
        in0.copy_successors(in0_s_list);
        in1.copy_predecessors(in1_p_list);
        in1.copy_successors(in1_s_list);
        middle.copy_predecessors(mp_list);
        middle.copy_successors(ms_list);
        out0.copy_predecessors(out0_p_list);
        out0.copy_successors(out0_s_list);
        out1.copy_predecessors(out1_p_list);
        out1.copy_successors(out1_s_list);
    }

    void make_and_validate_full_graph() {
        /*     in0           out0  */
        /*         \       /       */
        /*           middle        */
        /*         /       \       */
        /*     in1           out1  */
        tbb::flow::make_edge( in0, middle );
        tbb::flow::make_edge( in1, middle );
        tbb::flow::make_edge( middle, out0 );
        tbb::flow::make_edge( middle, out1 );

        set_up_lists();

        ASSERT( in0.predecessor_count() == 0 && in0_p_list.size() == 0, "expected 0 predecessors" );
        ASSERT( in0.successor_count() == 1 && in0_s_list.size() == 1 && *(in0_s_list.begin()) == ms_ptr, "expected 1 successor" );
        ASSERT( in1.predecessor_count() == 0 && in1_p_list.size() == 0, "expected 0 predecessors" );
        ASSERT( in1.successor_count() == 1 && in1_s_list.size() == 1 && *(in1_s_list.begin()) == ms_ptr, "expected 1 successor" );
        ASSERT( middle.predecessor_count() == 2 && mp_list.size() == 2, "expected 2 predecessors" );
        ASSERT( middle.successor_count() == 2 && ms_list.size() == 2, "expected 2 successors" );
        ASSERT( out0.predecessor_count() == 1 && out0_p_list.size() == 1 && *(out0_p_list.begin()) == mp_ptr, "expected 1 predecessor" );
        ASSERT( out0.successor_count() == 0 && out0_s_list.size() == 0, "expected 0 successors" );
        ASSERT( out1.predecessor_count() == 1 && out1_p_list.size() == 1 && *(out1_p_list.begin()) == mp_ptr, "expected 1 predecessor" );
        ASSERT( out1.successor_count() == 0 && out1_s_list.size() == 0, "expected 0 successors" );

        int first_pred = *(mp_list.begin()) == ins[0] ? 0 : ( *(mp_list.begin()) == ins[1] ? 1 : -1 );
        mp_list_iter = mp_list.begin(); ++mp_list_iter;
        int second_pred = *mp_list_iter == ins[0] ? 0 : ( *mp_list_iter == ins[1] ? 1 : -1 );
        ASSERT( first_pred != -1 && second_pred != -1 && first_pred != second_pred, "bad predecessor(s) for middle" );

        int first_succ = *(ms_list.begin()) == outs[0] ? 0 : ( *(ms_list.begin()) == outs[1] ? 1 : -1 );
        ms_list_iter = ++(ms_list.begin());
        int second_succ = *ms_list_iter == outs[0] ? 0 : ( *ms_list_iter == outs[1] ? 1 : -1 );
        ASSERT( first_succ != -1 && second_succ != -1 && first_succ != second_succ, "bad successor(s) for middle" );

        in0.try_put(1);
        in1.try_put(2);
        g.wait_for_all();

        int r = 0;
        int v = 0;

        ASSERT( in0.try_get(v) == false, "buffer should not have a value" );
        ASSERT( in1.try_get(v) == false, "buffer should not have a value" );
        ASSERT( middle.try_get(v) == false, "buffer should not have a value" );
        while ( out0.try_get(v) ) {
            ASSERT( (v == 1 || v == 2) && (v&r) == 0, "duplicate value" );
            r |= v;
            g.wait_for_all();
        }
        while ( out1.try_get(v) ) {
            ASSERT( (v == 1 || v == 2) && (v&r) == 0, "duplicate value" );
            r |= v;
            g.wait_for_all();
        }
        ASSERT( r == 3, "not all values received" );
        g.wait_for_all();
    }

    void validate_half_graph() {
        /*     in0           out0  */
        /*                         */
        /*           middle        */
        /*         /       \       */
        /*     in1           out1  */
        set_up_lists();

        ASSERT( in0.predecessor_count() == 0 && in0_p_list.size() == 0, "expected 0 predecessors" );
        ASSERT( in0.successor_count() == 0 && in0_s_list.size() == 0, "expected 0 successors" );
        ASSERT( in1.predecessor_count() == 0 && in1_p_list.size() == 0, "expected 0 predecessors" );
        ASSERT( in1.successor_count() == 1 && in1_s_list.size() == 1 && *(in1_s_list.begin()) == ms_ptr, "expected 1 successor" );
        ASSERT( middle.predecessor_count() == 1 && mp_list.size() == 1, "expected 1 predecessor" );
        ASSERT( middle.successor_count() == 1 && ms_list.size() == 1, "expected 1 successor" );
        ASSERT( out0.predecessor_count() == 0 && out0_p_list.size() == 0, "expected 0 predecessors" );
        ASSERT( out0.successor_count() == 0 && out0_s_list.size() == 0, "expected 0 successors" );
        ASSERT( out1.predecessor_count() == 1 && out1_p_list.size() == 1 && *(out1_p_list.begin()) == mp_ptr, "expected 1 predecessor" );
        ASSERT( out1.successor_count() == 0 && out1_s_list.size() == 0, "expected 0 successors" );

        ASSERT( middle.predecessor_count() == 1 && mp_list.size() == 1, "expected two predecessors" );
        ASSERT( middle.successor_count() == 1 && ms_list.size() == 1, "expected two successors" );

        ASSERT( *(mp_list.begin()) == ins[1], "incorrect predecessor" );
        ASSERT( *(ms_list.begin()) == outs[1], "incorrect successor" );

        in0.try_put(1);
        in1.try_put(2);
        g.wait_for_all();

        int v = 0;
        ASSERT( in0.try_get(v) == true && v == 1, "buffer should have a value of 1" );
        ASSERT( in1.try_get(v) == false, "buffer should not have a value" );
        ASSERT( middle.try_get(v) == false, "buffer should not have a value" );
        ASSERT( out0.try_get(v) == false, "buffer should not have a value" );
        ASSERT( out1.try_get(v) == true && v == 2, "buffer should have a value of 2" );
        g.wait_for_all();
    }

    void validate_empty_graph() {
        /*     in0           out0  */
        /*                         */
        /*           middle        */
        /*                         */
        /*     in1           out1  */
        set_up_lists();

        ASSERT( in0.predecessor_count() == 0 && in0_p_list.size() == 0, "expected 0 predecessors" );
        ASSERT( in0.successor_count() == 0 && in0_s_list.size() == 0, "expected 0 successors" );
        ASSERT( in1.predecessor_count() == 0 && in1_p_list.size() == 0, "expected 0 predecessors" );
        ASSERT( in1.successor_count() == 0 && in1_s_list.size() == 0, "expected 0 successors" );
        ASSERT( middle.predecessor_count() == 0 && mp_list.size() == 0, "expected 0 predecessors" );
        ASSERT( middle.successor_count() == 0 && ms_list.size() == 0, "expected 0 successors" );
        ASSERT( out0.predecessor_count() == 0 && out0_p_list.size() == 0, "expected 0 predecessors" );
        ASSERT( out0.successor_count() == 0 && out0_s_list.size() == 0, "expected 0 successors" );
        ASSERT( out1.predecessor_count() == 0 && out1_p_list.size() == 0, "expected 0 predecessors" );
        ASSERT( out1.successor_count() == 0 && out1_s_list.size() == 0, "expected 0 successors" );

        ASSERT( middle.predecessor_count() == 0 && mp_list.size() == 0, "expected 0 predecessors" );
        ASSERT( middle.successor_count() == 0 && ms_list.size() == 0, "expected 0 successors" );

        in0.try_put(1);
        in1.try_put(2);
        g.wait_for_all();

        int v = 0;
        ASSERT( in0.try_get(v) == true && v == 1, "buffer should have a value of 1" );
        ASSERT( in1.try_get(v) == true && v == 2, "buffer should have a value of 2" );
        ASSERT( middle.try_get(v) == false, "buffer should not have a value" );
        ASSERT( out0.try_get(v) == false, "buffer should not have a value" );
        ASSERT( out1.try_get(v) == false, "buffer should not have a value" );
        g.wait_for_all();
    }

    // forbid the ecompiler generation of operator= (VS2012 warning)
    test_buffer_base_extract& operator=(test_buffer_base_extract & /*other*/);

public:

    test_buffer_base_extract(tbb::flow::graph &_g, NODE_TYPE &i0, NODE_TYPE &i1, NODE_TYPE &m, NODE_TYPE &o0, NODE_TYPE &o1) :
        g(_g), in0(i0), in1(i1), middle(m), out0(o0), out1(o1) {
        ins[0] = &in0;
        ins[1] = &in1;
        outs[0] = &out0;
        outs[1] = &out1;
        ms_ptr = static_cast< typename NODE_TYPE::successor_type * >(&middle);
        mp_ptr = static_cast< typename NODE_TYPE::predecessor_type *>(&middle);
    }

    virtual ~test_buffer_base_extract() {}

    void run_tests() {
        make_and_validate_full_graph();

        in0.extract();
        out0.extract();
        validate_half_graph();

        in1.extract();
        out1.extract();
        validate_empty_graph();

        make_and_validate_full_graph();

        middle.extract();
        validate_empty_graph();

        make_and_validate_full_graph();
    }

};

template< typename NODE_TYPE >
class test_buffer_extract : public test_buffer_base_extract<NODE_TYPE> {
protected:
    tbb::flow::graph my_g;
    NODE_TYPE my_in0;
    NODE_TYPE my_in1;
    NODE_TYPE my_middle;
    NODE_TYPE my_out0;
    NODE_TYPE my_out1;
public:
    test_buffer_extract() : test_buffer_base_extract<NODE_TYPE>( my_g, my_in0, my_in1, my_middle, my_out0, my_out1),
                            my_in0(my_g), my_in1(my_g), my_middle(my_g), my_out0(my_g), my_out1(my_g) { }
};

template< >
class test_buffer_extract< tbb::flow::sequencer_node<int> > : public test_buffer_base_extract< tbb::flow::sequencer_node<int> > {
protected:
    typedef tbb::flow::sequencer_node<int> my_node_t;
    tbb::flow::graph my_g;
    my_node_t my_in0;
    my_node_t my_in1;
    my_node_t my_middle;
    my_node_t my_out0;
    my_node_t my_out1;

    typedef tbb::atomic<size_t> count_t;
    count_t middle_count;
    count_t out0_count;
    count_t out1_count;

    struct always_zero { size_t operator()(int) { return 0; } };
    struct always_inc {
        count_t *c;
        always_inc(count_t &_c) : c(&_c) {}
        size_t operator()(int) {
            return c->fetch_and_increment();
        }
    };

    void set_up_lists() __TBB_override {
        middle_count = 0;
        out0_count = 0;
        out1_count = 0;
        my_g.reset(); // reset the sequencer nodes to start at 0 again
        test_buffer_base_extract< my_node_t >::set_up_lists();
    }


public:
    test_buffer_extract() : test_buffer_base_extract<my_node_t>( my_g, my_in0, my_in1, my_middle, my_out0, my_out1),
                            my_in0(my_g, always_zero()), my_in1(my_g, always_zero()), my_middle(my_g, always_inc(middle_count)),
                            my_out0(my_g, always_inc(out0_count)), my_out1(my_g, always_inc(out1_count)) {
    }
};

// test for simple node that has one input, one output (overwrite_node, write_once_node, limiter_node)
// decrement tests have to be done separately.
template<template< class > class NType, typename ItemType>
void test_extract_on_node() {
    tbb::flow::graph g;
    ItemType dont_care;
    NType<ItemType> node0(g);
    tbb::flow::queue_node<ItemType> q0(g);
    tbb::flow::queue_node<ItemType> q1(g);
    tbb::flow::queue_node<ItemType> q2(g);
    for( int i = 0; i < 2; ++i) {
        tbb::flow::make_edge(q0,node0);
        tbb::flow::make_edge(q1,node0);
        tbb::flow::make_edge(node0, q2);
        q0.try_put(ItemType(i));
        g.wait_for_all();

        /* q0               */
        /*   \              */
        /*    \             */
        /*      node0 -- q2 */
        /*    /             */
        /*   /              */
        /* q1               */

        ASSERT(node0.predecessor_count() == 2 && q0.successor_count() == 1 && q1.successor_count() == 1, "bad predecessor count");
        ASSERT(node0.successor_count() == 1 && q2.predecessor_count() == 1, "bad successor count");

        ASSERT(q2.try_get(dont_care) && int(dont_care) == i, "item not forwarded");
        typename NType<ItemType>::successor_list_type sv, sv1;
        typename NType<ItemType>::predecessor_list_type pv, pv1;

        pv1.push_back(&q0);
        pv1.push_back(&q1);
        sv1.push_back(&q2);
        node0.copy_predecessors(pv);
        node0.copy_successors(sv);
        ASSERT(lists_match(pv,pv1), "predecessor vector incorrect");
        ASSERT(lists_match(sv,sv1), "successor vector incorrect");

        if(i == 0) {
            node0.extract();
        }
        else {
            q0.extract();
            q1.extract();
            q2.extract();
        }

        q0.try_put(ItemType(2));
        g.wait_for_all();
        ASSERT(!q2.try_get(dont_care), "node0 not disconnected");
        ASSERT(q0.try_get(dont_care), "q0 empty (should have one item)");

        node0.copy_predecessors(pv);
        node0.copy_successors(sv);
        ASSERT(node0.predecessor_count() == 0 && q0.successor_count() == 0 && q1.successor_count() == 0, "error in pred count after extract");
        ASSERT(pv.size() == 0, "error in pred array count after extract");
        ASSERT(node0.successor_count() == 0 && q2.predecessor_count() == 0, "error in succ count after extract");
        ASSERT(sv.size() == 0, "error in succ array count after extract");
        g.wait_for_all();
    }
}

#endif  // TBB_PREVIEW_FLOW_GRAPH_FEATURES

template<typename NodeType>
void test_input_ports_return_ref(NodeType& mip_node) {
    typename NodeType::input_ports_type& input_ports1 = mip_node.input_ports();
    typename NodeType::input_ports_type& input_ports2 = mip_node.input_ports();
    ASSERT(&input_ports1 == &input_ports2, "input_ports() should return reference");
}

template<typename NodeType>
void test_output_ports_return_ref(NodeType& mop_node) {
    typename NodeType::output_ports_type& output_ports1 = mop_node.output_ports();
    typename NodeType::output_ports_type& output_ports2 = mop_node.output_ports();
    ASSERT(&output_ports1 == &output_ports2, "output_ports() should return reference");
}

template< template <typename> class ReservingNodeType, typename DataType, bool DoClear >
class harness_reserving_body : NoAssign {
    ReservingNodeType<DataType> &my_reserving_node;
    tbb::flow::buffer_node<DataType> &my_buffer_node;
public:
    harness_reserving_body(ReservingNodeType<DataType> &reserving_node, tbb::flow::buffer_node<DataType> &bn) : my_reserving_node(reserving_node), my_buffer_node(bn) {}
    void operator()(DataType i) const {
        my_reserving_node.try_put(i);
        if (DoClear) my_reserving_node.clear();
        my_buffer_node.try_put(i);
        my_reserving_node.try_put(i);
    }
};

template< template <typename> class ReservingNodeType, typename DataType >
void test_reserving_nodes() {
    const int N = 300;
 
    tbb::flow::graph g;

    ReservingNodeType<DataType> reserving_n(g);

    tbb::flow::buffer_node<DataType> buffering_n(g);
    tbb::flow::join_node< tbb::flow::tuple<DataType, DataType>, tbb::flow::reserving > join_n(g);
    harness_counting_receiver< tbb::flow::tuple<DataType, DataType> > end_receiver(g);

    tbb::flow::make_edge(reserving_n, tbb::flow::input_port<0>(join_n));
    tbb::flow::make_edge(buffering_n, tbb::flow::input_port<1>(join_n));
    tbb::flow::make_edge(join_n, end_receiver);

    NativeParallelFor(N, harness_reserving_body<ReservingNodeType, DataType, false>(reserving_n, buffering_n));
    g.wait_for_all();

    ASSERT(end_receiver.my_count == N, NULL);

    // Should not hang
    NativeParallelFor(N, harness_reserving_body<ReservingNodeType, DataType, true>(reserving_n, buffering_n));
    g.wait_for_all();

    ASSERT(end_receiver.my_count == 2 * N, NULL);
}

#endif
