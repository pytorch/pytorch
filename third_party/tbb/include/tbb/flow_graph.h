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

#ifndef __TBB_flow_graph_H
#define __TBB_flow_graph_H

#include "tbb_stddef.h"
#include "atomic.h"
#include "spin_mutex.h"
#include "null_mutex.h"
#include "spin_rw_mutex.h"
#include "null_rw_mutex.h"
#include "task.h"
#include "cache_aligned_allocator.h"
#include "tbb_exception.h"
#include "internal/_template_helpers.h"
#include "internal/_aggregator_impl.h"
#include "tbb_profiling.h"
#include "task_arena.h"

#if __TBB_PREVIEW_ASYNC_MSG
#include <vector>    // std::vector in internal::async_storage
#include <memory>    // std::shared_ptr in async_msg
#endif

#if __TBB_PREVIEW_STREAMING_NODE
// For streaming_node
#include <array>            // std::array
#include <unordered_map>    // std::unordered_map
#include <type_traits>      // std::decay, std::true_type, std::false_type
#endif // __TBB_PREVIEW_STREAMING_NODE

#if TBB_DEPRECATED_FLOW_ENQUEUE
#define FLOW_SPAWN(a) tbb::task::enqueue((a))
#else
#define FLOW_SPAWN(a) tbb::task::spawn((a))
#endif

// use the VC10 or gcc version of tuple if it is available.
#if __TBB_CPP11_TUPLE_PRESENT
    #include <tuple>
namespace tbb {
    namespace flow {
        using std::tuple;
        using std::tuple_size;
        using std::tuple_element;
        using std::get;
    }
}
#else
    #include "compat/tuple"
#endif

#include<list>
#include<queue>

/** @file
  \brief The graph related classes and functions

  There are some applications that best express dependencies as messages
  passed between nodes in a graph.  These messages may contain data or
  simply act as signals that a predecessors has completed. The graph
  class and its associated node classes can be used to express such
  applications.
*/

namespace tbb {
namespace flow {

//! An enumeration the provides the two most common concurrency levels: unlimited and serial
enum concurrency { unlimited = 0, serial = 1 };

namespace interface10 {

//! A generic null type
struct null_type {};

//! An empty class used for messages that mean "I'm done"
class continue_msg {};

//! Forward declaration section
template< typename T > class sender;
template< typename T > class receiver;
class continue_receiver;
template< typename T > class limiter_node;  // needed for resetting decrementer
template< typename R, typename B > class run_and_put_task;

namespace internal {

template<typename T, typename M> class successor_cache;
template<typename T, typename M> class broadcast_cache;
template<typename T, typename M> class round_robin_cache;
template<typename T, typename M> class predecessor_cache;
template<typename T, typename M> class reservable_predecessor_cache;

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
// Holder of edges both for caches and for those nodes which do not have predecessor caches.
// C == receiver< ... > or sender< ... >, depending.
template<typename C>
class edge_container {

public:
    typedef std::list<C *, tbb::tbb_allocator<C *> > edge_list_type;

    void add_edge(C &s) {
        built_edges.push_back(&s);
    }

    void delete_edge(C &s) {
        for (typename edge_list_type::iterator i = built_edges.begin(); i != built_edges.end(); ++i) {
            if (*i == &s) {
                (void)built_edges.erase(i);
                return;  // only remove one predecessor per request
            }
        }
    }

    void copy_edges(edge_list_type &v) {
        v = built_edges;
    }

    size_t edge_count() {
        return (size_t)(built_edges.size());
    }

    void clear() {
        built_edges.clear();
    }

    // methods remove the statement from all predecessors/successors liste in the edge
    // container.
    template< typename S > void sender_extract(S &s);
    template< typename R > void receiver_extract(R &r);

private:
    edge_list_type built_edges;
};  // class edge_container
#endif  /* TBB_PREVIEW_FLOW_GRAPH_FEATURES */

} // namespace internal

} // namespace interface10
} // namespace flow
} // namespace tbb

//! The graph class
#include "internal/_flow_graph_impl.h"

namespace tbb {
namespace flow {
namespace interface10 {

// enqueue left task if necessary. Returns the non-enqueued task if there is one.
static inline tbb::task *combine_tasks(graph& g, tbb::task * left, tbb::task * right) {
    // if no RHS task, don't change left.
    if (right == NULL) return left;
    // right != NULL
    if (left == NULL) return right;
    if (left == SUCCESSFULLY_ENQUEUED) return right;
    // left contains a task
    if (right != SUCCESSFULLY_ENQUEUED) {
        // both are valid tasks
        internal::spawn_in_graph_arena(g, *left);
        return right;
    }
    return left;
}

#if __TBB_PREVIEW_ASYNC_MSG

template < typename T > class async_msg;

namespace internal {

template < typename T > class async_storage;

template< typename T, typename = void >
struct async_helpers {
    typedef async_msg<T> async_type;
    typedef T filtered_type;

    static const bool is_async_type = false;

    static const void* to_void_ptr(const T& t) {
        return static_cast<const void*>(&t);
    }

    static void* to_void_ptr(T& t) {
        return static_cast<void*>(&t);
    }

    static const T& from_void_ptr(const void* p) {
        return *static_cast<const T*>(p);
    }

    static T& from_void_ptr(void* p) {
        return *static_cast<T*>(p);
    }

    static task* try_put_task_wrapper_impl(receiver<T>* const this_recv, const void *p, bool is_async) {
        if (is_async) {
            // This (T) is NOT async and incoming 'A<X> t' IS async
            // Get data from async_msg
            const async_msg<filtered_type>& msg = async_helpers< async_msg<filtered_type> >::from_void_ptr(p);
            task* const new_task = msg.my_storage->subscribe(*this_recv, this_recv->graph_reference());
            // finalize() must be called after subscribe() because set() can be called in finalize()
            // and 'this_recv' client must be subscribed by this moment
            msg.finalize();
            return new_task;
        }
        else {
            // Incoming 't' is NOT async
            return this_recv->try_put_task(from_void_ptr(p));
        }
    }
};

template< typename T >
struct async_helpers< T, typename std::enable_if< std::is_base_of<async_msg<typename T::async_msg_data_type>, T>::value >::type > {
    typedef T async_type;
    typedef typename T::async_msg_data_type filtered_type;

    static const bool is_async_type = true;

    // Receiver-classes use const interfaces
    static const void* to_void_ptr(const T& t) {
        return static_cast<const void*>(&static_cast<const async_msg<filtered_type>&>(t));
    }

    static void* to_void_ptr(T& t) {
        return static_cast<void*>(&static_cast<async_msg<filtered_type>&>(t));
    }

    // Sender-classes use non-const interfaces
    static const T& from_void_ptr(const void* p) {
        return *static_cast<const T*>(static_cast<const async_msg<filtered_type>*>(p));
    }

    static T& from_void_ptr(void* p) {
        return *static_cast<T*>(static_cast<async_msg<filtered_type>*>(p));
    }

    // Used in receiver<T> class
    static task* try_put_task_wrapper_impl(receiver<T>* const this_recv, const void *p, bool is_async) {
        if (is_async) {
            // Both are async
            return this_recv->try_put_task(from_void_ptr(p));
        }
        else {
            // This (T) is async and incoming 'X t' is NOT async
            // Create async_msg for X
            const filtered_type& t = async_helpers<filtered_type>::from_void_ptr(p);
            const T msg(t);
            return this_recv->try_put_task(msg);
        }
    }
};

class untyped_receiver;

class untyped_sender {
    template< typename, typename > friend class internal::predecessor_cache;
    template< typename, typename > friend class internal::reservable_predecessor_cache;
public:
    //! The successor type for this node
    typedef untyped_receiver successor_type;

    virtual ~untyped_sender() {}

    // NOTE: Following part of PUBLIC section is copy-paste from original sender<T> class

    // TODO: Prevent untyped successor registration

    //! Add a new successor to this node
    virtual bool register_successor( successor_type &r ) = 0;

    //! Removes a successor from this node
    virtual bool remove_successor( successor_type &r ) = 0;

    //! Releases the reserved item
    virtual bool try_release( ) { return false; }

    //! Consumes the reserved item
    virtual bool try_consume( ) { return false; }

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    //! interface to record edges for traversal & deletion
    typedef internal::edge_container<successor_type> built_successors_type;
    typedef built_successors_type::edge_list_type successor_list_type;
    virtual built_successors_type &built_successors()                   = 0;
    virtual void    internal_add_built_successor( successor_type & )    = 0;
    virtual void    internal_delete_built_successor( successor_type & ) = 0;
    virtual void    copy_successors( successor_list_type &)             = 0;
    virtual size_t  successor_count()                                   = 0;
#endif
protected:
    //! Request an item from the sender
    template< typename X >
    bool try_get( X &t ) {
        return try_get_wrapper( internal::async_helpers<X>::to_void_ptr(t), internal::async_helpers<X>::is_async_type );
    }

    //! Reserves an item in the sender
    template< typename X >
    bool try_reserve( X &t ) {
        return try_reserve_wrapper( internal::async_helpers<X>::to_void_ptr(t), internal::async_helpers<X>::is_async_type );
    }

    virtual bool try_get_wrapper( void* p, bool is_async ) = 0;
    virtual bool try_reserve_wrapper( void* p, bool is_async ) = 0;
};

class untyped_receiver  {
    template< typename, typename > friend class run_and_put_task;
    template< typename > friend class limiter_node;

    template< typename, typename > friend class internal::broadcast_cache;
    template< typename, typename > friend class internal::round_robin_cache;
    template< typename, typename > friend class internal::successor_cache;

#if __TBB_PREVIEW_OPENCL_NODE
    template< typename, typename > friend class proxy_dependency_receiver;
#endif /* __TBB_PREVIEW_OPENCL_NODE */
public:
    //! The predecessor type for this node
    typedef untyped_sender predecessor_type;

    //! Destructor
    virtual ~untyped_receiver() {}

    //! Put an item to the receiver
    template<typename X>
    bool try_put(const X& t) {
        task *res = try_put_task(t);
        if (!res) return false;
        if (res != SUCCESSFULLY_ENQUEUED) internal::spawn_in_graph_arena(graph_reference(), *res);
        return true;
    }

    // NOTE: Following part of PUBLIC section is copy-paste from original receiver<T> class

    // TODO: Prevent untyped predecessor registration

    //! Add a predecessor to the node
    virtual bool register_predecessor( predecessor_type & ) { return false; }

    //! Remove a predecessor from the node
    virtual bool remove_predecessor( predecessor_type & ) { return false; }

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    typedef internal::edge_container<predecessor_type> built_predecessors_type;
    typedef built_predecessors_type::edge_list_type predecessor_list_type;
    virtual built_predecessors_type &built_predecessors()                  = 0;
    virtual void   internal_add_built_predecessor( predecessor_type & )    = 0;
    virtual void   internal_delete_built_predecessor( predecessor_type & ) = 0;
    virtual void   copy_predecessors( predecessor_list_type & )            = 0;
    virtual size_t predecessor_count()                                     = 0;
#endif
protected:
    template<typename X>
    task *try_put_task(const X& t) {
        return try_put_task_wrapper( internal::async_helpers<X>::to_void_ptr(t), internal::async_helpers<X>::is_async_type );
    }

    virtual task* try_put_task_wrapper( const void* p, bool is_async ) = 0;

    virtual graph& graph_reference() = 0;

    // NOTE: Following part of PROTECTED and PRIVATE sections is copy-paste from original receiver<T> class

    //! put receiver back in initial state
    virtual void reset_receiver(reset_flags f = rf_reset_protocol) = 0;

    virtual bool is_continue_receiver() { return false; }
};

} // namespace internal

//! Pure virtual template class that defines a sender of messages of type T
template< typename T >
class sender : public internal::untyped_sender {
public:
    //! The output type of this sender
    typedef T output_type;

    typedef typename internal::async_helpers<T>::filtered_type filtered_type;

    //! Request an item from the sender
    virtual bool try_get( T & ) { return false; }

    //! Reserves an item in the sender
    virtual bool try_reserve( T & ) { return false; }

protected:
    virtual bool try_get_wrapper( void* p, bool is_async ) __TBB_override {
        // Both async OR both are NOT async
        if ( internal::async_helpers<T>::is_async_type == is_async ) {
            return try_get( internal::async_helpers<T>::from_void_ptr(p) );
        }
        // Else: this (T) is async OR incoming 't' is async
        __TBB_ASSERT(false, "async_msg interface does not support 'pull' protocol in try_get()");
        return false;
    }

    virtual bool try_reserve_wrapper( void* p, bool is_async ) __TBB_override {
        // Both async OR both are NOT async
        if ( internal::async_helpers<T>::is_async_type == is_async ) {
            return try_reserve( internal::async_helpers<T>::from_void_ptr(p) );
        }
        // Else: this (T) is async OR incoming 't' is async
        __TBB_ASSERT(false, "async_msg interface does not support 'pull' protocol in try_reserve()");
        return false;
    }
};  // class sender<T>

//! Pure virtual template class that defines a receiver of messages of type T
template< typename T >
class receiver : public internal::untyped_receiver {
    template< typename > friend class internal::async_storage;
    template< typename, typename > friend struct internal::async_helpers;
public:
    //! The input type of this receiver
    typedef T input_type;

    typedef typename internal::async_helpers<T>::filtered_type filtered_type;

    //! Put an item to the receiver
    bool try_put( const typename internal::async_helpers<T>::filtered_type& t ) {
        return internal::untyped_receiver::try_put(t);
    }

    bool try_put( const typename internal::async_helpers<T>::async_type& t ) {
        return internal::untyped_receiver::try_put(t);
    }

protected:
    virtual task* try_put_task_wrapper( const void *p, bool is_async ) __TBB_override {
        return internal::async_helpers<T>::try_put_task_wrapper_impl(this, p, is_async);
    }

    //! Put item to successor; return task to run the successor if possible.
    virtual task *try_put_task(const T& t) = 0;

}; // class receiver<T>

#else // __TBB_PREVIEW_ASYNC_MSG

//! Pure virtual template class that defines a sender of messages of type T
template< typename T >
class sender {
public:
    //! The output type of this sender
    typedef T output_type;

    //! The successor type for this node
    typedef receiver<T> successor_type;

    virtual ~sender() {}

    // NOTE: Following part of PUBLIC section is partly copy-pasted in sender<T> under #if __TBB_PREVIEW_ASYNC_MSG

    //! Add a new successor to this node
    virtual bool register_successor( successor_type &r ) = 0;

    //! Removes a successor from this node
    virtual bool remove_successor( successor_type &r ) = 0;

    //! Request an item from the sender
    virtual bool try_get( T & ) { return false; }

    //! Reserves an item in the sender
    virtual bool try_reserve( T & ) { return false; }

    //! Releases the reserved item
    virtual bool try_release( ) { return false; }

    //! Consumes the reserved item
    virtual bool try_consume( ) { return false; }

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    //! interface to record edges for traversal & deletion
    typedef typename  internal::edge_container<successor_type> built_successors_type;
    typedef typename  built_successors_type::edge_list_type successor_list_type;
    virtual built_successors_type &built_successors()                   = 0;
    virtual void    internal_add_built_successor( successor_type & )    = 0;
    virtual void    internal_delete_built_successor( successor_type & ) = 0;
    virtual void    copy_successors( successor_list_type &)             = 0;
    virtual size_t  successor_count()                                   = 0;
#endif
};  // class sender<T>

//! Pure virtual template class that defines a receiver of messages of type T
template< typename T >
class receiver {
public:
    //! The input type of this receiver
    typedef T input_type;

    //! The predecessor type for this node
    typedef sender<T> predecessor_type;

    //! Destructor
    virtual ~receiver() {}

    //! Put an item to the receiver
    bool try_put( const T& t ) {
        task *res = try_put_task(t);
        if (!res) return false;
        if (res != SUCCESSFULLY_ENQUEUED) internal::spawn_in_graph_arena(graph_reference(), *res);
        return true;
    }

    //! put item to successor; return task to run the successor if possible.
protected:
    template< typename R, typename B > friend class run_and_put_task;
    template< typename X, typename Y > friend class internal::broadcast_cache;
    template< typename X, typename Y > friend class internal::round_robin_cache;
    virtual task *try_put_task(const T& t) = 0;
    virtual graph& graph_reference() = 0;
public:
    // NOTE: Following part of PUBLIC and PROTECTED sections is copy-pasted in receiver<T> under #if __TBB_PREVIEW_ASYNC_MSG

    //! Add a predecessor to the node
    virtual bool register_predecessor( predecessor_type & ) { return false; }

    //! Remove a predecessor from the node
    virtual bool remove_predecessor( predecessor_type & ) { return false; }

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    typedef typename internal::edge_container<predecessor_type> built_predecessors_type;
    typedef typename built_predecessors_type::edge_list_type predecessor_list_type;
    virtual built_predecessors_type &built_predecessors()                  = 0;
    virtual void   internal_add_built_predecessor( predecessor_type & )    = 0;
    virtual void   internal_delete_built_predecessor( predecessor_type & ) = 0;
    virtual void   copy_predecessors( predecessor_list_type & )            = 0;
    virtual size_t predecessor_count()                                     = 0;
#endif

protected:
    //! put receiver back in initial state
    template<typename U> friend class limiter_node;
    virtual void reset_receiver(reset_flags f = rf_reset_protocol) = 0;

    template<typename TT, typename M> friend class internal::successor_cache;
    virtual bool is_continue_receiver() { return false; }

#if __TBB_PREVIEW_OPENCL_NODE
    template< typename, typename > friend class proxy_dependency_receiver;
#endif /* __TBB_PREVIEW_OPENCL_NODE */
}; // class receiver<T>

#endif // __TBB_PREVIEW_ASYNC_MSG

//! Base class for receivers of completion messages
/** These receivers automatically reset, but cannot be explicitly waited on */
class continue_receiver : public receiver< continue_msg > {
public:

    //! The input type
    typedef continue_msg input_type;

    //! The predecessor type for this node
    typedef receiver<input_type>::predecessor_type predecessor_type;

    //! Constructor
    explicit continue_receiver( int number_of_predecessors = 0 ) {
        my_predecessor_count = my_initial_predecessor_count = number_of_predecessors;
        my_current_count = 0;
    }

    //! Copy constructor
    continue_receiver( const continue_receiver& src ) : receiver<continue_msg>() {
        my_predecessor_count = my_initial_predecessor_count = src.my_initial_predecessor_count;
        my_current_count = 0;
    }

    //! Increments the trigger threshold
    bool register_predecessor( predecessor_type & ) __TBB_override {
        spin_mutex::scoped_lock l(my_mutex);
        ++my_predecessor_count;
        return true;
    }

    //! Decrements the trigger threshold
    /** Does not check to see if the removal of the predecessor now makes the current count
        exceed the new threshold.  So removing a predecessor while the graph is active can cause
        unexpected results. */
    bool remove_predecessor( predecessor_type & ) __TBB_override {
        spin_mutex::scoped_lock l(my_mutex);
        --my_predecessor_count;
        return true;
    }

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    typedef internal::edge_container<predecessor_type> built_predecessors_type;
    typedef built_predecessors_type::edge_list_type predecessor_list_type;
    built_predecessors_type &built_predecessors() __TBB_override { return my_built_predecessors; }

    void internal_add_built_predecessor( predecessor_type &s) __TBB_override {
        spin_mutex::scoped_lock l(my_mutex);
        my_built_predecessors.add_edge( s );
    }

    void internal_delete_built_predecessor( predecessor_type &s) __TBB_override {
        spin_mutex::scoped_lock l(my_mutex);
        my_built_predecessors.delete_edge(s);
    }

    void copy_predecessors( predecessor_list_type &v) __TBB_override {
        spin_mutex::scoped_lock l(my_mutex);
        my_built_predecessors.copy_edges(v);
    }

    size_t predecessor_count() __TBB_override {
        spin_mutex::scoped_lock l(my_mutex);
        return my_built_predecessors.edge_count();
    }

#endif  /* TBB_PREVIEW_FLOW_GRAPH_FEATURES */

protected:
    template< typename R, typename B > friend class run_and_put_task;
    template<typename X, typename Y> friend class internal::broadcast_cache;
    template<typename X, typename Y> friend class internal::round_robin_cache;
    // execute body is supposed to be too small to create a task for.
    task *try_put_task( const input_type & ) __TBB_override {
        {
            spin_mutex::scoped_lock l(my_mutex);
            if ( ++my_current_count < my_predecessor_count )
                return SUCCESSFULLY_ENQUEUED;
            else
                my_current_count = 0;
        }
        task * res = execute();
        return res? res : SUCCESSFULLY_ENQUEUED;
    }

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    // continue_receiver must contain its own built_predecessors because it does
    // not have a node_cache.
    built_predecessors_type my_built_predecessors;
#endif
    spin_mutex my_mutex;
    int my_predecessor_count;
    int my_current_count;
    int my_initial_predecessor_count;
    // the friend declaration in the base class did not eliminate the "protected class"
    // error in gcc 4.1.2
    template<typename U> friend class limiter_node;

    void reset_receiver( reset_flags f ) __TBB_override {
        my_current_count = 0;
        if (f & rf_clear_edges) {
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
            my_built_predecessors.clear();
#endif
            my_predecessor_count = my_initial_predecessor_count;
        }
    }

    //! Does whatever should happen when the threshold is reached
    /** This should be very fast or else spawn a task.  This is
        called while the sender is blocked in the try_put(). */
    virtual task * execute() = 0;
    template<typename TT, typename M> friend class internal::successor_cache;
    bool is_continue_receiver() __TBB_override { return true; }

}; // class continue_receiver

}  // interfaceX

#if __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING
    template <typename K, typename T>
    K key_from_message( const T &t ) {
        return t.key();
    }
#endif /* __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING */

    using interface10::sender;
    using interface10::receiver;
    using interface10::continue_receiver;
}  // flow
}  // tbb

#include "internal/_flow_graph_trace_impl.h"
#include "internal/_tbb_hash_compare_impl.h"

namespace tbb {
namespace flow {
namespace interface10 {

#include "internal/_flow_graph_body_impl.h"
#include "internal/_flow_graph_cache_impl.h"
#include "internal/_flow_graph_types_impl.h"
#if __TBB_PREVIEW_ASYNC_MSG
#include "internal/_flow_graph_async_msg_impl.h"
#endif
using namespace internal::graph_policy_namespace;

template <typename C, typename N>
graph_iterator<C,N>::graph_iterator(C *g, bool begin) : my_graph(g), current_node(NULL)
{
    if (begin) current_node = my_graph->my_nodes;
    //else it is an end iterator by default
}

template <typename C, typename N>
typename graph_iterator<C,N>::reference graph_iterator<C,N>::operator*() const {
    __TBB_ASSERT(current_node, "graph_iterator at end");
    return *operator->();
}

template <typename C, typename N>
typename graph_iterator<C,N>::pointer graph_iterator<C,N>::operator->() const {
    return current_node;
}

template <typename C, typename N>
void graph_iterator<C,N>::internal_forward() {
    if (current_node) current_node = current_node->next;
}

//! Constructs a graph with isolated task_group_context
inline graph::graph() : my_nodes(NULL), my_nodes_last(NULL), my_task_arena(NULL) {
    prepare_task_arena();
    own_context = true;
    cancelled = false;
    caught_exception = false;
    my_context = new task_group_context();
    my_root_task = (new (task::allocate_root(*my_context)) empty_task);
    my_root_task->set_ref_count(1);
    tbb::internal::fgt_graph(this);
    my_is_active = true;
}

inline graph::graph(task_group_context& use_this_context) :
    my_context(&use_this_context), my_nodes(NULL), my_nodes_last(NULL), my_task_arena(NULL) {
    prepare_task_arena();
    own_context = false;
    my_root_task = (new (task::allocate_root(*my_context)) empty_task);
    my_root_task->set_ref_count(1);
    tbb::internal::fgt_graph(this);
    my_is_active = true;
}

inline graph::~graph() {
    wait_for_all();
    my_root_task->set_ref_count(0);
    tbb::task::destroy(*my_root_task);
    if (own_context) delete my_context;
    delete my_task_arena;
}

inline void graph::reserve_wait() {
    if (my_root_task) {
        my_root_task->increment_ref_count();
        tbb::internal::fgt_reserve_wait(this);
    }
}

inline void graph::release_wait() {
    if (my_root_task) {
        tbb::internal::fgt_release_wait(this);
        my_root_task->decrement_ref_count();
    }
}

inline void graph::register_node(graph_node *n) {
    n->next = NULL;
    {
        spin_mutex::scoped_lock lock(nodelist_mutex);
        n->prev = my_nodes_last;
        if (my_nodes_last) my_nodes_last->next = n;
        my_nodes_last = n;
        if (!my_nodes) my_nodes = n;
    }
}

inline void graph::remove_node(graph_node *n) {
    {
        spin_mutex::scoped_lock lock(nodelist_mutex);
        __TBB_ASSERT(my_nodes && my_nodes_last, "graph::remove_node: Error: no registered nodes");
        if (n->prev) n->prev->next = n->next;
        if (n->next) n->next->prev = n->prev;
        if (my_nodes_last == n) my_nodes_last = n->prev;
        if (my_nodes == n) my_nodes = n->next;
    }
    n->prev = n->next = NULL;
}

inline void graph::reset( reset_flags f ) {
    // reset context
    internal::deactivate_graph(*this);

    if(my_context) my_context->reset();
    cancelled = false;
    caught_exception = false;
    // reset all the nodes comprising the graph
    for(iterator ii = begin(); ii != end(); ++ii) {
        graph_node *my_p = &(*ii);
        my_p->reset_node(f);
    }
    // Reattach the arena. Might be useful to run the graph in a particular task_arena
    // while not limiting graph lifetime to a single task_arena::execute() call.
    prepare_task_arena( /*reinit=*/true );
    internal::activate_graph(*this);
    // now spawn the tasks necessary to start the graph
    for(task_list_type::iterator rti = my_reset_task_list.begin(); rti != my_reset_task_list.end(); ++rti) {
        my_task_arena->execute(graph::spawn_functor(*(*rti)));
    }
    my_reset_task_list.clear();
}

inline graph::iterator graph::begin() { return iterator(this, true); }

inline graph::iterator graph::end() { return iterator(this, false); }

inline graph::const_iterator graph::begin() const { return const_iterator(this, true); }

inline graph::const_iterator graph::end() const { return const_iterator(this, false); }

inline graph::const_iterator graph::cbegin() const { return const_iterator(this, true); }

inline graph::const_iterator graph::cend() const { return const_iterator(this, false); }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
inline void graph::set_name(const char *name) {
    tbb::internal::fgt_graph_desc(this, name);
}
#endif

inline graph_node::graph_node(graph& g) : my_graph(g) {
    my_graph.register_node(this);
}

inline graph_node::~graph_node() {
    my_graph.remove_node(this);
}

#include "internal/_flow_graph_node_impl.h"

//! An executable node that acts as a source, i.e. it has no predecessors
template < typename Output >
class source_node : public graph_node, public sender< Output > {
public:
    //! The type of the output message, which is complete
    typedef Output output_type;

    //! The type of successors of this node
    typedef typename sender<output_type>::successor_type successor_type;

    //Source node has no input type
    typedef null_type input_type;

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    typedef typename sender<output_type>::built_successors_type built_successors_type;
    typedef typename sender<output_type>::successor_list_type successor_list_type;
#endif

    //! Constructor for a node with a successor
    template< typename Body >
    source_node( graph &g, Body body, bool is_active = true )
        : graph_node(g), my_active(is_active), init_my_active(is_active),
        my_body( new internal::source_body_leaf< output_type, Body>(body) ),
        my_init_body( new internal::source_body_leaf< output_type, Body>(body) ),
        my_reserved(false), my_has_cached_item(false)
    {
        my_successors.set_owner(this);
        tbb::internal::fgt_node_with_body( tbb::internal::FLOW_SOURCE_NODE, &this->my_graph,
                                           static_cast<sender<output_type> *>(this), this->my_body );
    }

    //! Copy constructor
    source_node( const source_node& src ) :
        graph_node(src.my_graph), sender<Output>(),
        my_active(src.init_my_active),
        init_my_active(src.init_my_active), my_body( src.my_init_body->clone() ), my_init_body(src.my_init_body->clone() ),
        my_reserved(false), my_has_cached_item(false)
    {
        my_successors.set_owner(this);
        tbb::internal::fgt_node_with_body( tbb::internal::FLOW_SOURCE_NODE, &this->my_graph,
                                           static_cast<sender<output_type> *>(this), this->my_body );
    }

    //! The destructor
    ~source_node() { delete my_body; delete my_init_body; }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    void set_name( const char *name ) __TBB_override {
        tbb::internal::fgt_node_desc( this, name );
    }
#endif

    //! Add a new successor to this node
    bool register_successor( successor_type &r ) __TBB_override {
        spin_mutex::scoped_lock lock(my_mutex);
        my_successors.register_successor(r);
        if ( my_active )
            spawn_put();
        return true;
    }

    //! Removes a successor from this node
    bool remove_successor( successor_type &r ) __TBB_override {
        spin_mutex::scoped_lock lock(my_mutex);
        my_successors.remove_successor(r);
        return true;
    }

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES

    built_successors_type &built_successors() __TBB_override { return my_successors.built_successors(); }

    void internal_add_built_successor( successor_type &r) __TBB_override {
        spin_mutex::scoped_lock lock(my_mutex);
        my_successors.internal_add_built_successor(r);
    }

    void internal_delete_built_successor( successor_type &r) __TBB_override {
        spin_mutex::scoped_lock lock(my_mutex);
        my_successors.internal_delete_built_successor(r);
    }

    size_t successor_count() __TBB_override {
        spin_mutex::scoped_lock lock(my_mutex);
        return my_successors.successor_count();
    }

    void copy_successors(successor_list_type &v) __TBB_override {
        spin_mutex::scoped_lock l(my_mutex);
        my_successors.copy_successors(v);
    }
#endif  /* TBB_PREVIEW_FLOW_GRAPH_FEATURES */

    //! Request an item from the node
    bool try_get( output_type &v ) __TBB_override {
        spin_mutex::scoped_lock lock(my_mutex);
        if ( my_reserved )
            return false;

        if ( my_has_cached_item ) {
            v = my_cached_item;
            my_has_cached_item = false;
            return true;
        }
        // we've been asked to provide an item, but we have none.  enqueue a task to
        // provide one.
        spawn_put();
        return false;
    }

    //! Reserves an item.
    bool try_reserve( output_type &v ) __TBB_override {
        spin_mutex::scoped_lock lock(my_mutex);
        if ( my_reserved ) {
            return false;
        }

        if ( my_has_cached_item ) {
            v = my_cached_item;
            my_reserved = true;
            return true;
        } else {
            return false;
        }
    }

    //! Release a reserved item.
    /** true = item has been released and so remains in sender, dest must request or reserve future items */
    bool try_release( ) __TBB_override {
        spin_mutex::scoped_lock lock(my_mutex);
        __TBB_ASSERT( my_reserved && my_has_cached_item, "releasing non-existent reservation" );
        my_reserved = false;
        if(!my_successors.empty())
            spawn_put();
        return true;
    }

    //! Consumes a reserved item
    bool try_consume( ) __TBB_override {
        spin_mutex::scoped_lock lock(my_mutex);
        __TBB_ASSERT( my_reserved && my_has_cached_item, "consuming non-existent reservation" );
        my_reserved = false;
        my_has_cached_item = false;
        if ( !my_successors.empty() ) {
            spawn_put();
        }
        return true;
    }

    //! Activates a node that was created in the inactive state
    void activate() {
        spin_mutex::scoped_lock lock(my_mutex);
        my_active = true;
        if (!my_successors.empty())
            spawn_put();
    }

    template<typename Body>
    Body copy_function_object() {
        internal::source_body<output_type> &body_ref = *this->my_body;
        return dynamic_cast< internal::source_body_leaf<output_type, Body> & >(body_ref).get_body();
    }

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    void extract( ) __TBB_override {
        my_successors.built_successors().sender_extract(*this);   // removes "my_owner" == this from each successor
        my_active = init_my_active;
        my_reserved = false;
        if(my_has_cached_item) my_has_cached_item = false;
    }
#endif

protected:

    //! resets the source_node to its initial state
    void reset_node( reset_flags f) __TBB_override {
        my_active = init_my_active;
        my_reserved =false;
        if(my_has_cached_item) {
            my_has_cached_item = false;
        }
        if(f & rf_clear_edges) my_successors.clear();
        if(f & rf_reset_bodies) {
            internal::source_body<output_type> *tmp = my_init_body->clone();
            delete my_body;
            my_body = tmp;
        }
        if(my_active)
            internal::add_task_to_graph_reset_list(this->my_graph, create_put_task());
    }

private:
    spin_mutex my_mutex;
    bool my_active;
    bool init_my_active;
    internal::source_body<output_type> *my_body;
    internal::source_body<output_type> *my_init_body;
    internal::broadcast_cache< output_type > my_successors;
    bool my_reserved;
    bool my_has_cached_item;
    output_type my_cached_item;

    // used by apply_body_bypass, can invoke body of node.
    bool try_reserve_apply_body(output_type &v) {
        spin_mutex::scoped_lock lock(my_mutex);
        if ( my_reserved ) {
            return false;
        }
        if ( !my_has_cached_item ) {
            tbb::internal::fgt_begin_body( my_body );
            bool r = (*my_body)(my_cached_item);
            tbb::internal::fgt_end_body( my_body );
            if (r) {
                my_has_cached_item = true;
            }
        }
        if ( my_has_cached_item ) {
            v = my_cached_item;
            my_reserved = true;
            return true;
        } else {
            return false;
        }
    }

    // when resetting, and if the source_node was created with my_active == true, then
    // when we reset the node we must store a task to run the node, and spawn it only
    // after the reset is complete and is_active() is again true.  This is why we don't
    // test for is_active() here.
    task* create_put_task() {
        return ( new ( task::allocate_additional_child_of( *(this->my_graph.root_task()) ) )
                        internal:: source_task_bypass < source_node< output_type > >( *this ) );
    }

    //! Spawns a task that applies the body
    void spawn_put( ) {
        if(internal::is_graph_active(this->my_graph)) {
            internal::spawn_in_graph_arena(this->my_graph, *create_put_task());
        }
    }

    friend class internal::source_task_bypass< source_node< output_type > >;
    //! Applies the body.  Returning SUCCESSFULLY_ENQUEUED okay; forward_task_bypass will handle it.
    task * apply_body_bypass( ) {
        output_type v;
        if ( !try_reserve_apply_body(v) )
            return NULL;

        task *last_task = my_successors.try_put_task(v);
        if ( last_task )
            try_consume();
        else
            try_release();
        return last_task;
    }
};  // class source_node

template<typename T>
struct allocate_buffer {
    static const bool value = false;
};

template<>
struct allocate_buffer<queueing> {
    static const bool value = true;
};

//! Implements a function node that supports Input -> Output
template < typename Input, typename Output = continue_msg, typename Policy = queueing, typename Allocator=cache_aligned_allocator<Input> >
class function_node : public graph_node, public internal::function_input<Input,Output,Allocator>, public internal::function_output<Output> {
public:
    typedef Input input_type;
    typedef Output output_type;
    typedef internal::function_input<input_type,output_type,Allocator> fInput_type;
    typedef internal::function_input_queue<input_type, Allocator> input_queue_type;
    typedef internal::function_output<output_type> fOutput_type;
    typedef typename fInput_type::predecessor_type predecessor_type;
    typedef typename fOutput_type::successor_type successor_type;
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    typedef typename fInput_type::predecessor_list_type predecessor_list_type;
    typedef typename fOutput_type::successor_list_type successor_list_type;
#endif
    using fInput_type::my_predecessors;

    //! Constructor
    // input_queue_type is allocated here, but destroyed in the function_input_base.
    // TODO: pass the graph_buffer_policy to the function_input_base so it can all
    // be done in one place.  This would be an interface-breaking change.
    template< typename Body >
    function_node( graph &g, size_t concurrency, Body body ) :
        graph_node(g), fInput_type(g, concurrency, body, allocate_buffer<Policy>::value ?
               new input_queue_type( ) : NULL ) {
        tbb::internal::fgt_node_with_body( tbb::internal::FLOW_FUNCTION_NODE, &this->my_graph,
                static_cast<receiver<input_type> *>(this), static_cast<sender<output_type> *>(this), this->my_body );
    }

    //! Copy constructor
    function_node( const function_node& src ) :
        graph_node(src.my_graph),
        fInput_type(src, allocate_buffer<Policy>::value ? new input_queue_type : NULL),
        fOutput_type() {
        tbb::internal::fgt_node_with_body( tbb::internal::FLOW_FUNCTION_NODE, &this->my_graph,
                static_cast<receiver<input_type> *>(this), static_cast<sender<output_type> *>(this), this->my_body );
    }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    void set_name( const char *name ) __TBB_override {
        tbb::internal::fgt_node_desc( this, name );
    }
#endif

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    void extract( ) __TBB_override {
        my_predecessors.built_predecessors().receiver_extract(*this);
        successors().built_successors().sender_extract(*this);
    }
#endif

protected:
    template< typename R, typename B > friend class run_and_put_task;
    template<typename X, typename Y> friend class internal::broadcast_cache;
    template<typename X, typename Y> friend class internal::round_robin_cache;
    using fInput_type::try_put_task;

    internal::broadcast_cache<output_type> &successors () __TBB_override { return fOutput_type::my_successors; }

    void reset_node(reset_flags f) __TBB_override {
        fInput_type::reset_function_input(f);
        // TODO: use clear() instead.
        if(f & rf_clear_edges) {
            successors().clear();
            my_predecessors.clear();
        }
        __TBB_ASSERT(!(f & rf_clear_edges) || successors().empty(), "function_node successors not empty");
        __TBB_ASSERT(this->my_predecessors.empty(), "function_node predecessors not empty");
    }

};  // class function_node

//! implements a function node that supports Input -> (set of outputs)
// Output is a tuple of output types.
template < typename Input, typename Output, typename Policy = queueing, typename Allocator=cache_aligned_allocator<Input> >
class multifunction_node :
    public graph_node,
    public internal::multifunction_input
    <
        Input,
        typename internal::wrap_tuple_elements<
            tbb::flow::tuple_size<Output>::value,  // #elements in tuple
            internal::multifunction_output,  // wrap this around each element
            Output // the tuple providing the types
        >::type,
        Allocator
    > {
protected:
    static const int N = tbb::flow::tuple_size<Output>::value;
public:
    typedef Input input_type;
    typedef null_type output_type;
    typedef typename internal::wrap_tuple_elements<N,internal::multifunction_output, Output>::type output_ports_type;
    typedef internal::multifunction_input<input_type, output_ports_type, Allocator> fInput_type;
    typedef internal::function_input_queue<input_type, Allocator> input_queue_type;
private:
    typedef typename internal::multifunction_input<input_type, output_ports_type, Allocator> base_type;
    using fInput_type::my_predecessors;
public:
    template<typename Body>
    multifunction_node( graph &g, size_t concurrency, Body body ) :
        graph_node(g), base_type(g,concurrency, body,  allocate_buffer<Policy>::value ? new input_queue_type : NULL) {
        tbb::internal::fgt_multioutput_node_with_body<N>( tbb::internal::FLOW_MULTIFUNCTION_NODE,
                &this->my_graph, static_cast<receiver<input_type> *>(this),
                this->output_ports(), this->my_body );
    }

    multifunction_node( const multifunction_node &other) :
        graph_node(other.my_graph), base_type(other,  allocate_buffer<Policy>::value ? new input_queue_type : NULL) {
        tbb::internal::fgt_multioutput_node_with_body<N>( tbb::internal::FLOW_MULTIFUNCTION_NODE,
                &this->my_graph, static_cast<receiver<input_type> *>(this),
                this->output_ports(), this->my_body );
    }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    void set_name( const char *name ) __TBB_override {
        tbb::internal::fgt_multioutput_node_desc( this, name );
    }
#endif

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    void extract( ) __TBB_override {
        my_predecessors.built_predecessors().receiver_extract(*this);
        base_type::extract();
    }
#endif
    // all the guts are in multifunction_input...
protected:
    void reset_node(reset_flags f) __TBB_override { base_type::reset(f); }
};  // multifunction_node

//! split_node: accepts a tuple as input, forwards each element of the tuple to its
//  successors.  The node has unlimited concurrency, so it does not reject inputs.
template<typename TupleType, typename Allocator=cache_aligned_allocator<TupleType> >
class split_node : public graph_node, public receiver<TupleType> {
    static const int N = tbb::flow::tuple_size<TupleType>::value;
    typedef receiver<TupleType> base_type;
public:
    typedef TupleType input_type;
    typedef Allocator allocator_type;
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    typedef typename base_type::predecessor_type predecessor_type;
    typedef typename base_type::predecessor_list_type predecessor_list_type;
    typedef internal::predecessor_cache<input_type, null_mutex > predecessor_cache_type;
    typedef typename predecessor_cache_type::built_predecessors_type built_predecessors_type;
#endif

    typedef typename internal::wrap_tuple_elements<
            N,  // #elements in tuple
            internal::multifunction_output,  // wrap this around each element
            TupleType // the tuple providing the types
        >::type  output_ports_type;

    explicit split_node(graph &g) : graph_node(g)
    {
        tbb::internal::fgt_multioutput_node<N>(tbb::internal::FLOW_SPLIT_NODE, &this->my_graph,
            static_cast<receiver<input_type> *>(this), this->output_ports());
    }
    split_node( const split_node & other) : graph_node(other.my_graph), base_type(other)
    {
        tbb::internal::fgt_multioutput_node<N>(tbb::internal::FLOW_SPLIT_NODE, &this->my_graph,
            static_cast<receiver<input_type> *>(this), this->output_ports());
    }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    void set_name( const char *name ) __TBB_override {
        tbb::internal::fgt_multioutput_node_desc( this, name );
    }
#endif

    output_ports_type &output_ports() { return my_output_ports; }

protected:
    task *try_put_task(const TupleType& t) __TBB_override {
        // Sending split messages in parallel is not justified, as overheads would prevail.
        // Also, we do not have successors here. So we just tell the task returned here is successful.
        return internal::emit_element<N>::emit_this(this->my_graph, t, output_ports());
    }
    void reset_node(reset_flags f) __TBB_override {
        if (f & rf_clear_edges)
            internal::clear_element<N>::clear_this(my_output_ports);

        __TBB_ASSERT(!(f & rf_clear_edges) || internal::clear_element<N>::this_empty(my_output_ports), "split_node reset failed");
    }
    void reset_receiver(reset_flags /*f*/) __TBB_override {}
    graph& graph_reference() __TBB_override {
        return my_graph;
    }
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
private: //! split_node doesn't use this "predecessors" functionality; so, we have "dummies" here;
    void extract() __TBB_override {}

    //! Adds to list of predecessors added by make_edge
    void internal_add_built_predecessor(predecessor_type&) __TBB_override {}

    //! removes from to list of predecessors (used by remove_edge)
    void internal_delete_built_predecessor(predecessor_type&) __TBB_override {}

    size_t predecessor_count() __TBB_override { return 0; }

    void copy_predecessors(predecessor_list_type&) __TBB_override {}

    built_predecessors_type &built_predecessors() __TBB_override { return my_predessors; }

    //! dummy member
    built_predecessors_type my_predessors;
#endif  /* TBB_PREVIEW_FLOW_GRAPH_FEATURES */

private:
    output_ports_type my_output_ports;
};

//! Implements an executable node that supports continue_msg -> Output
template <typename Output>
class continue_node : public graph_node, public internal::continue_input<Output>, public internal::function_output<Output> {
public:
    typedef continue_msg input_type;
    typedef Output output_type;
    typedef internal::continue_input<Output> fInput_type;
    typedef internal::function_output<output_type> fOutput_type;
    typedef typename fInput_type::predecessor_type predecessor_type;
    typedef typename fOutput_type::successor_type successor_type;

    //! Constructor for executable node with continue_msg -> Output
    template <typename Body >
    continue_node( graph &g, Body body ) :
        graph_node(g), internal::continue_input<output_type>( g, body ) {
        tbb::internal::fgt_node_with_body( tbb::internal::FLOW_CONTINUE_NODE, &this->my_graph,
                                           static_cast<receiver<input_type> *>(this),
                                           static_cast<sender<output_type> *>(this), this->my_body );
    }


    //! Constructor for executable node with continue_msg -> Output
    template <typename Body >
    continue_node( graph &g, int number_of_predecessors, Body body ) :
        graph_node(g), internal::continue_input<output_type>( g, number_of_predecessors, body ) {
        tbb::internal::fgt_node_with_body( tbb::internal::FLOW_CONTINUE_NODE, &this->my_graph,
                                           static_cast<receiver<input_type> *>(this),
                                           static_cast<sender<output_type> *>(this), this->my_body );
    }

    //! Copy constructor
    continue_node( const continue_node& src ) :
        graph_node(src.my_graph), internal::continue_input<output_type>(src),
        internal::function_output<Output>() {
        tbb::internal::fgt_node_with_body( tbb::internal::FLOW_CONTINUE_NODE, &this->my_graph,
                                           static_cast<receiver<input_type> *>(this),
                                           static_cast<sender<output_type> *>(this), this->my_body );
    }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    void set_name( const char *name ) __TBB_override {
        tbb::internal::fgt_node_desc( this, name );
    }
#endif

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    void extract() __TBB_override {
        fInput_type::my_built_predecessors.receiver_extract(*this);
        successors().built_successors().sender_extract(*this);
    }
#endif

protected:
    template< typename R, typename B > friend class run_and_put_task;
    template<typename X, typename Y> friend class internal::broadcast_cache;
    template<typename X, typename Y> friend class internal::round_robin_cache;
    using fInput_type::try_put_task;
    internal::broadcast_cache<output_type> &successors () __TBB_override { return fOutput_type::my_successors; }

    void reset_node(reset_flags f) __TBB_override {
        fInput_type::reset_receiver(f);
        if(f & rf_clear_edges)successors().clear();
        __TBB_ASSERT(!(f & rf_clear_edges) || successors().empty(), "continue_node not reset");
    }
};  // continue_node

template< typename T >
class overwrite_node : public graph_node, public receiver<T>, public sender<T> {
public:
    typedef T input_type;
    typedef T output_type;
    typedef typename receiver<input_type>::predecessor_type predecessor_type;
    typedef typename sender<output_type>::successor_type successor_type;
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    typedef typename receiver<input_type>::built_predecessors_type built_predecessors_type;
    typedef typename sender<output_type>::built_successors_type built_successors_type;
    typedef typename receiver<input_type>::predecessor_list_type predecessor_list_type;
    typedef typename sender<output_type>::successor_list_type successor_list_type;
#endif

    explicit overwrite_node(graph &g) : graph_node(g), my_buffer_is_valid(false) {
        my_successors.set_owner( this );
        tbb::internal::fgt_node( tbb::internal::FLOW_OVERWRITE_NODE, &this->my_graph,
                                 static_cast<receiver<input_type> *>(this), static_cast<sender<output_type> *>(this) );
    }

    //! Copy constructor; doesn't take anything from src; default won't work
    overwrite_node( const overwrite_node& src ) :
        graph_node(src.my_graph), receiver<T>(), sender<T>(), my_buffer_is_valid(false)
    {
        my_successors.set_owner( this );
        tbb::internal::fgt_node( tbb::internal::FLOW_OVERWRITE_NODE, &this->my_graph,
                                 static_cast<receiver<input_type> *>(this), static_cast<sender<output_type> *>(this) );
    }

    ~overwrite_node() {}

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    void set_name( const char *name ) __TBB_override {
        tbb::internal::fgt_node_desc( this, name );
    }
#endif

   bool register_successor( successor_type &s ) __TBB_override {
        spin_mutex::scoped_lock l( my_mutex );
        if (my_buffer_is_valid && internal::is_graph_active( my_graph )) {
            // We have a valid value that must be forwarded immediately.
            bool ret = s.try_put( my_buffer );
#if TBB_PREVIEW_RESERVABLE_OVERWRITE_NODE
            if ( ret ) {
                // We add the successor that accepted our put
                my_successors.register_successor( s );
            } else {
                // In case of reservation a race between the moment of reservation and register_successor can appear,
                // because failed reserve does not mean that register_successor is not ready to put a message immediately. 
                // We have some sort of infinite loop: reserving node tries to set pull state for the edge,
                // but overwrite_node tries to return push state back. That is why we have to break this loop with task creation.
                task *rtask = new ( task::allocate_additional_child_of( *( my_graph.root_task() ) ) )
                    register_predecessor_task( *this, s );
                internal::spawn_in_graph_arena( my_graph, *rtask );
            }
#else
            if ( ret || !s.register_predecessor( *this  ) ) {
                // We add the successor: it accepted our put or it rejected it but won't let us become a predecessor
                my_successors.register_successor( s );
            } else {
                // We don't add the successor: it rejected our put and we became its predecessor instead
                return false;
            }
#endif
        } else {
            // No valid value yet, just add as successor
            my_successors.register_successor( s );
        }
        return true;
    }

    bool remove_successor( successor_type &s ) __TBB_override {
        spin_mutex::scoped_lock l( my_mutex );
        my_successors.remove_successor(s);
        return true;
    }

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    built_predecessors_type &built_predecessors() __TBB_override { return my_built_predecessors; }
    built_successors_type   &built_successors()   __TBB_override { return my_successors.built_successors(); }

    void internal_add_built_successor( successor_type &s) __TBB_override {
        spin_mutex::scoped_lock l( my_mutex );
        my_successors.internal_add_built_successor(s);
    }

    void internal_delete_built_successor( successor_type &s) __TBB_override {
        spin_mutex::scoped_lock l( my_mutex );
        my_successors.internal_delete_built_successor(s);
    }

    size_t successor_count() __TBB_override {
        spin_mutex::scoped_lock l( my_mutex );
        return my_successors.successor_count();
    }

    void copy_successors(successor_list_type &v) __TBB_override {
        spin_mutex::scoped_lock l( my_mutex );
        my_successors.copy_successors(v);
    }

    void internal_add_built_predecessor( predecessor_type &p) __TBB_override {
        spin_mutex::scoped_lock l( my_mutex );
        my_built_predecessors.add_edge(p);
    }

    void internal_delete_built_predecessor( predecessor_type &p) __TBB_override {
        spin_mutex::scoped_lock l( my_mutex );
        my_built_predecessors.delete_edge(p);
    }

    size_t predecessor_count() __TBB_override {
        spin_mutex::scoped_lock l( my_mutex );
        return my_built_predecessors.edge_count();
    }

    void copy_predecessors( predecessor_list_type &v ) __TBB_override {
        spin_mutex::scoped_lock l( my_mutex );
        my_built_predecessors.copy_edges(v);
    }

    void extract() __TBB_override {
        my_buffer_is_valid = false;
        built_successors().sender_extract(*this);
        built_predecessors().receiver_extract(*this);
    }

#endif  /* TBB_PREVIEW_FLOW_GRAPH_FEATURES */

    bool try_get( input_type &v ) __TBB_override {
        spin_mutex::scoped_lock l( my_mutex );
        if ( my_buffer_is_valid ) {
            v = my_buffer;
            return true;
        }
        return false;
    }

#if TBB_PREVIEW_RESERVABLE_OVERWRITE_NODE
    //! Reserves an item
    bool try_reserve( T &v ) __TBB_override {
        return try_get(v);
    }

    //! Releases the reserved item
    bool try_release() __TBB_override { return true; }

    //! Consumes the reserved item
    bool try_consume() __TBB_override { return true; }
#endif
 
    bool is_valid() {
       spin_mutex::scoped_lock l( my_mutex );
       return my_buffer_is_valid;
    }

    void clear() {
       spin_mutex::scoped_lock l( my_mutex );
       my_buffer_is_valid = false;
    }

protected:

    template< typename R, typename B > friend class run_and_put_task;
    template<typename X, typename Y> friend class internal::broadcast_cache;
    template<typename X, typename Y> friend class internal::round_robin_cache;
    task * try_put_task( const input_type &v ) __TBB_override {
        spin_mutex::scoped_lock l( my_mutex );
        return try_put_task_impl(v);
    }

    task * try_put_task_impl(const input_type &v) {
        my_buffer = v;
        my_buffer_is_valid = true;
        task * rtask = my_successors.try_put_task(v);
        if (!rtask) rtask = SUCCESSFULLY_ENQUEUED;
        return rtask;
    }

    graph& graph_reference() __TBB_override {
        return my_graph;
    }

#if TBB_PREVIEW_RESERVABLE_OVERWRITE_NODE
    //! Breaks an infinite loop between the node reservation and register_successor call
    struct register_predecessor_task : public task {
        register_predecessor_task(sender<T>& owner, receiver<T>& succ) :
            o(owner), s(succ) {};

        tbb::task* execute() __TBB_override {
            if (!s.register_predecessor(o)) {
                o.register_successor(s);
            }
            return NULL;
        }

        sender<T>& o;
        receiver<T>& s;
    };
#endif

    spin_mutex my_mutex;
    internal::broadcast_cache< input_type, null_rw_mutex > my_successors;
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    internal::edge_container<predecessor_type> my_built_predecessors;
#endif
    input_type my_buffer;
    bool my_buffer_is_valid;
    void reset_receiver(reset_flags /*f*/) __TBB_override {}

    void reset_node( reset_flags f) __TBB_override {
        my_buffer_is_valid = false;
       if (f&rf_clear_edges) {
           my_successors.clear();
       }
    }
};  // overwrite_node

template< typename T >
class write_once_node : public overwrite_node<T> {
public:
    typedef T input_type;
    typedef T output_type;
    typedef overwrite_node<T> base_type;
    typedef typename receiver<input_type>::predecessor_type predecessor_type;
    typedef typename sender<output_type>::successor_type successor_type;

    //! Constructor
    explicit write_once_node(graph& g) : base_type(g) {
        tbb::internal::fgt_node( tbb::internal::FLOW_WRITE_ONCE_NODE, &(this->my_graph),
                                 static_cast<receiver<input_type> *>(this),
                                 static_cast<sender<output_type> *>(this) );
    }

    //! Copy constructor: call base class copy constructor
    write_once_node( const write_once_node& src ) : base_type(src) {
        tbb::internal::fgt_node( tbb::internal::FLOW_WRITE_ONCE_NODE, &(this->my_graph),
                                 static_cast<receiver<input_type> *>(this),
                                 static_cast<sender<output_type> *>(this) );
    }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    void set_name( const char *name ) __TBB_override {
        tbb::internal::fgt_node_desc( this, name );
    }
#endif

protected:
    template< typename R, typename B > friend class run_and_put_task;
    template<typename X, typename Y> friend class internal::broadcast_cache;
    template<typename X, typename Y> friend class internal::round_robin_cache;
    task *try_put_task( const T &v ) __TBB_override {
        spin_mutex::scoped_lock l( this->my_mutex );
        return this->my_buffer_is_valid ? NULL : this->try_put_task_impl(v);
    }
};

//! Forwards messages of type T to all successors
template <typename T>
class broadcast_node : public graph_node, public receiver<T>, public sender<T> {
public:
    typedef T input_type;
    typedef T output_type;
    typedef typename receiver<input_type>::predecessor_type predecessor_type;
    typedef typename sender<output_type>::successor_type successor_type;
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    typedef typename receiver<input_type>::predecessor_list_type predecessor_list_type;
    typedef typename sender<output_type>::successor_list_type successor_list_type;
#endif
private:
    internal::broadcast_cache<input_type> my_successors;
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    internal::edge_container<predecessor_type> my_built_predecessors;
    spin_mutex pred_mutex;  // serialize accesses on edge_container
#endif
public:

    explicit broadcast_node(graph& g) : graph_node(g) {
        my_successors.set_owner( this );
        tbb::internal::fgt_node( tbb::internal::FLOW_BROADCAST_NODE, &this->my_graph,
                                 static_cast<receiver<input_type> *>(this), static_cast<sender<output_type> *>(this) );
    }

    // Copy constructor
    broadcast_node( const broadcast_node& src ) :
        graph_node(src.my_graph), receiver<T>(), sender<T>()
    {
        my_successors.set_owner( this );
        tbb::internal::fgt_node( tbb::internal::FLOW_BROADCAST_NODE, &this->my_graph,
                                 static_cast<receiver<input_type> *>(this), static_cast<sender<output_type> *>(this) );
    }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    void set_name( const char *name ) __TBB_override {
        tbb::internal::fgt_node_desc( this, name );
    }
#endif

    //! Adds a successor
    bool register_successor( successor_type &r ) __TBB_override {
        my_successors.register_successor( r );
        return true;
    }

    //! Removes s as a successor
    bool remove_successor( successor_type &r ) __TBB_override {
        my_successors.remove_successor( r );
        return true;
    }

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    typedef typename sender<T>::built_successors_type built_successors_type;

    built_successors_type &built_successors() __TBB_override { return my_successors.built_successors(); }

    void internal_add_built_successor(successor_type &r) __TBB_override {
        my_successors.internal_add_built_successor(r);
    }

    void internal_delete_built_successor(successor_type &r) __TBB_override {
        my_successors.internal_delete_built_successor(r);
    }

    size_t successor_count() __TBB_override {
        return my_successors.successor_count();
    }

    void copy_successors(successor_list_type &v) __TBB_override {
        my_successors.copy_successors(v);
    }

    typedef typename receiver<T>::built_predecessors_type built_predecessors_type;

    built_predecessors_type &built_predecessors() __TBB_override { return my_built_predecessors; } 

    void internal_add_built_predecessor( predecessor_type &p) __TBB_override {
        spin_mutex::scoped_lock l(pred_mutex);
        my_built_predecessors.add_edge(p);
    }

    void internal_delete_built_predecessor( predecessor_type &p) __TBB_override {
        spin_mutex::scoped_lock l(pred_mutex);
        my_built_predecessors.delete_edge(p);
    }

    size_t predecessor_count() __TBB_override {
        spin_mutex::scoped_lock l(pred_mutex);
        return my_built_predecessors.edge_count();
    }

    void copy_predecessors(predecessor_list_type &v) __TBB_override {
        spin_mutex::scoped_lock l(pred_mutex);
        my_built_predecessors.copy_edges(v);
    }

    void extract() __TBB_override {
        my_built_predecessors.receiver_extract(*this);
        my_successors.built_successors().sender_extract(*this);
    }
#endif  /* TBB_PREVIEW_FLOW_GRAPH_FEATURES */

protected:
    template< typename R, typename B > friend class run_and_put_task;
    template<typename X, typename Y> friend class internal::broadcast_cache;
    template<typename X, typename Y> friend class internal::round_robin_cache;
    //! build a task to run the successor if possible.  Default is old behavior.
    task *try_put_task(const T& t) __TBB_override {
        task *new_task = my_successors.try_put_task(t);
        if (!new_task) new_task = SUCCESSFULLY_ENQUEUED;
        return new_task;
    }

    graph& graph_reference() __TBB_override {
        return my_graph;
    }

    void reset_receiver(reset_flags /*f*/) __TBB_override {}

    void reset_node(reset_flags f) __TBB_override {
        if (f&rf_clear_edges) {
           my_successors.clear();
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
           my_built_predecessors.clear();
#endif
        }
        __TBB_ASSERT(!(f & rf_clear_edges) || my_successors.empty(), "Error resetting broadcast_node");
    }
};  // broadcast_node

//! Forwards messages in arbitrary order
template <typename T, typename A=cache_aligned_allocator<T> >
class buffer_node : public graph_node, public internal::reservable_item_buffer<T, A>, public receiver<T>, public sender<T> {
public:
    typedef T input_type;
    typedef T output_type;
    typedef typename receiver<input_type>::predecessor_type predecessor_type;
    typedef typename sender<output_type>::successor_type successor_type;
    typedef buffer_node<T, A> class_type;
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    typedef typename receiver<input_type>::predecessor_list_type predecessor_list_type;
    typedef typename sender<output_type>::successor_list_type successor_list_type;
#endif
protected:
    typedef size_t size_type;
    internal::round_robin_cache< T, null_rw_mutex > my_successors;

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    internal::edge_container<predecessor_type> my_built_predecessors;
#endif

    friend class internal::forward_task_bypass< buffer_node< T, A > >;

    enum op_type {reg_succ, rem_succ, req_item, res_item, rel_res, con_res, put_item, try_fwd_task
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
        , add_blt_succ, del_blt_succ,
        add_blt_pred, del_blt_pred,
        blt_succ_cnt, blt_pred_cnt,
        blt_succ_cpy, blt_pred_cpy   // create vector copies of preds and succs
#endif
    };

    // implements the aggregator_operation concept
    class buffer_operation : public internal::aggregated_operation< buffer_operation > {
    public:
        char type;
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
        task * ltask;
        union {
            input_type *elem;
            successor_type *r;
            predecessor_type *p;
            size_t cnt_val;
            successor_list_type *svec;
            predecessor_list_type *pvec;
        };
#else
        T *elem;
        task * ltask;
        successor_type *r;
#endif
        buffer_operation(const T& e, op_type t) : type(char(t))

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
                                                  , ltask(NULL), elem(const_cast<T*>(&e))
#else
                                                  , elem(const_cast<T*>(&e)) , ltask(NULL)
#endif
        {}
        buffer_operation(op_type t) : type(char(t)),  ltask(NULL) {}
    };

    bool forwarder_busy;
    typedef internal::aggregating_functor<class_type, buffer_operation> handler_type;
    friend class internal::aggregating_functor<class_type, buffer_operation>;
    internal::aggregator< handler_type, buffer_operation> my_aggregator;

    virtual void handle_operations(buffer_operation *op_list) {
        handle_operations_impl(op_list, this);
    }

    template<typename derived_type>
    void handle_operations_impl(buffer_operation *op_list, derived_type* derived) {
        __TBB_ASSERT(static_cast<class_type*>(derived) == this, "'this' is not a base class for derived");

        buffer_operation *tmp = NULL;
        bool try_forwarding = false;
        while (op_list) {
            tmp = op_list;
            op_list = op_list->next;
            switch (tmp->type) {
            case reg_succ: internal_reg_succ(tmp); try_forwarding = true; break;
            case rem_succ: internal_rem_succ(tmp); break;
            case req_item: internal_pop(tmp); break;
            case res_item: internal_reserve(tmp); break;
            case rel_res:  internal_release(tmp); try_forwarding = true; break;
            case con_res:  internal_consume(tmp); try_forwarding = true; break;
            case put_item: try_forwarding = internal_push(tmp); break;
            case try_fwd_task: internal_forward_task(tmp); break;
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
            // edge recording
            case add_blt_succ: internal_add_built_succ(tmp); break;
            case del_blt_succ: internal_del_built_succ(tmp); break;
            case add_blt_pred: internal_add_built_pred(tmp); break;
            case del_blt_pred: internal_del_built_pred(tmp); break;
            case blt_succ_cnt: internal_succ_cnt(tmp); break;
            case blt_pred_cnt: internal_pred_cnt(tmp); break;
            case blt_succ_cpy: internal_copy_succs(tmp); break;
            case blt_pred_cpy: internal_copy_preds(tmp); break;
#endif
            }
        }

        derived->order();

        if (try_forwarding && !forwarder_busy) {
            if(internal::is_graph_active(this->my_graph)) {
                forwarder_busy = true;
                task *new_task = new(task::allocate_additional_child_of(*(this->my_graph.root_task()))) internal::
                        forward_task_bypass
                        < buffer_node<input_type, A> >(*this);
                // tmp should point to the last item handled by the aggregator.  This is the operation
                // the handling thread enqueued.  So modifying that record will be okay.
                // workaround for icc bug
                tbb::task *z = tmp->ltask;
                graph &g = this->my_graph;
                tmp->ltask = combine_tasks(g, z, new_task);  // in case the op generated a task
            }
        }
    }  // handle_operations

    inline task *grab_forwarding_task( buffer_operation &op_data) {
        return op_data.ltask;
    }

    inline bool enqueue_forwarding_task(buffer_operation &op_data) {
        task *ft = grab_forwarding_task(op_data);
        if(ft) {
            internal::spawn_in_graph_arena(graph_reference(), *ft);
            return true;
        }
        return false;
    }

    //! This is executed by an enqueued task, the "forwarder"
    virtual task *forward_task() {
        buffer_operation op_data(try_fwd_task);
        task *last_task = NULL;
        do {
            op_data.status = internal::WAIT;
            op_data.ltask = NULL;
            my_aggregator.execute(&op_data);

            // workaround for icc bug
            tbb::task *xtask = op_data.ltask;
            graph& g = this->my_graph;
            last_task = combine_tasks(g, last_task, xtask);
        } while (op_data.status ==internal::SUCCEEDED);
        return last_task;
    }

    //! Register successor
    virtual void internal_reg_succ(buffer_operation *op) {
        my_successors.register_successor(*(op->r));
        __TBB_store_with_release(op->status, internal::SUCCEEDED);
    }

    //! Remove successor
    virtual void internal_rem_succ(buffer_operation *op) {
        my_successors.remove_successor(*(op->r));
        __TBB_store_with_release(op->status, internal::SUCCEEDED);
    }

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    typedef typename sender<T>::built_successors_type built_successors_type;

    built_successors_type &built_successors() __TBB_override { return my_successors.built_successors(); }

    virtual void internal_add_built_succ(buffer_operation *op) {
        my_successors.internal_add_built_successor(*(op->r));
        __TBB_store_with_release(op->status, internal::SUCCEEDED);
    }

    virtual void internal_del_built_succ(buffer_operation *op) {
        my_successors.internal_delete_built_successor(*(op->r));
        __TBB_store_with_release(op->status, internal::SUCCEEDED);
    }

    typedef typename receiver<T>::built_predecessors_type built_predecessors_type;

    built_predecessors_type &built_predecessors() __TBB_override { return my_built_predecessors; } 

    virtual void internal_add_built_pred(buffer_operation *op) {
        my_built_predecessors.add_edge(*(op->p));
        __TBB_store_with_release(op->status, internal::SUCCEEDED);
    }

    virtual void internal_del_built_pred(buffer_operation *op) {
        my_built_predecessors.delete_edge(*(op->p));
        __TBB_store_with_release(op->status, internal::SUCCEEDED);
    }

    virtual void internal_succ_cnt(buffer_operation *op) {
        op->cnt_val = my_successors.successor_count();
        __TBB_store_with_release(op->status, internal::SUCCEEDED);
    }

    virtual void internal_pred_cnt(buffer_operation *op) {
        op->cnt_val = my_built_predecessors.edge_count();
        __TBB_store_with_release(op->status, internal::SUCCEEDED);
    }

    virtual void internal_copy_succs(buffer_operation *op) {
        my_successors.copy_successors(*(op->svec));
        __TBB_store_with_release(op->status, internal::SUCCEEDED);
    }

    virtual void internal_copy_preds(buffer_operation *op) {
        my_built_predecessors.copy_edges(*(op->pvec));
        __TBB_store_with_release(op->status, internal::SUCCEEDED);
    }

#endif  /* TBB_PREVIEW_FLOW_GRAPH_FEATURES */

private:
    void order() {}

    bool is_item_valid() {
        return this->my_item_valid(this->my_tail - 1);
    }

    void try_put_and_add_task(task*& last_task) {
        task *new_task = my_successors.try_put_task(this->back());
        if (new_task) {
            // workaround for icc bug
            graph& g = this->my_graph;
            last_task = combine_tasks(g, last_task, new_task);
            this->destroy_back();
        }
    }

protected:
    //! Tries to forward valid items to successors
    virtual void internal_forward_task(buffer_operation *op) {
        internal_forward_task_impl(op, this);
    }

    template<typename derived_type>
    void internal_forward_task_impl(buffer_operation *op, derived_type* derived) {
        __TBB_ASSERT(static_cast<class_type*>(derived) == this, "'this' is not a base class for derived");

        if (this->my_reserved || !derived->is_item_valid()) {
            __TBB_store_with_release(op->status, internal::FAILED);
            this->forwarder_busy = false;
            return;
        }
        // Try forwarding, giving each successor a chance
        task * last_task = NULL;
        size_type counter = my_successors.size();
        for (; counter > 0 && derived->is_item_valid(); --counter)
            derived->try_put_and_add_task(last_task);

        op->ltask = last_task;  // return task
        if (last_task && !counter) {
            __TBB_store_with_release(op->status, internal::SUCCEEDED);
        }
        else {
            __TBB_store_with_release(op->status, internal::FAILED);
            forwarder_busy = false;
        }
    }

    virtual bool internal_push(buffer_operation *op) {
        this->push_back(*(op->elem));
        __TBB_store_with_release(op->status, internal::SUCCEEDED);
        return true;
    }

    virtual void internal_pop(buffer_operation *op) {
        if(this->pop_back(*(op->elem))) {
            __TBB_store_with_release(op->status, internal::SUCCEEDED);
        }
        else {
            __TBB_store_with_release(op->status, internal::FAILED);
        }
    }

    virtual void internal_reserve(buffer_operation *op) {
        if(this->reserve_front(*(op->elem))) {
            __TBB_store_with_release(op->status, internal::SUCCEEDED);
        }
        else {
            __TBB_store_with_release(op->status, internal::FAILED);
        }
    }

    virtual void internal_consume(buffer_operation *op) {
        this->consume_front();
        __TBB_store_with_release(op->status, internal::SUCCEEDED);
    }

    virtual void internal_release(buffer_operation *op) {
        this->release_front();
        __TBB_store_with_release(op->status, internal::SUCCEEDED);
    }

public:
    //! Constructor
    explicit buffer_node( graph &g ) : graph_node(g), internal::reservable_item_buffer<T>(),
        forwarder_busy(false) {
        my_successors.set_owner(this);
        my_aggregator.initialize_handler(handler_type(this));
        tbb::internal::fgt_node( tbb::internal::FLOW_BUFFER_NODE, &this->my_graph,
                                 static_cast<receiver<input_type> *>(this), static_cast<sender<output_type> *>(this) );
    }

    //! Copy constructor
    buffer_node( const buffer_node& src ) : graph_node(src.my_graph),
        internal::reservable_item_buffer<T>(), receiver<T>(), sender<T>() {
        forwarder_busy = false;
        my_successors.set_owner(this);
        my_aggregator.initialize_handler(handler_type(this));
        tbb::internal::fgt_node( tbb::internal::FLOW_BUFFER_NODE, &this->my_graph,
                                 static_cast<receiver<input_type> *>(this), static_cast<sender<output_type> *>(this) );
    }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    void set_name( const char *name ) __TBB_override {
        tbb::internal::fgt_node_desc( this, name );
    }
#endif

    //
    // message sender implementation
    //

    //! Adds a new successor.
    /** Adds successor r to the list of successors; may forward tasks.  */
    bool register_successor( successor_type &r ) __TBB_override {
        buffer_operation op_data(reg_succ);
        op_data.r = &r;
        my_aggregator.execute(&op_data);
        (void)enqueue_forwarding_task(op_data);
        return true;
    }

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    void internal_add_built_successor( successor_type &r) __TBB_override {
        buffer_operation op_data(add_blt_succ);
        op_data.r = &r;
        my_aggregator.execute(&op_data);
    }

    void internal_delete_built_successor( successor_type &r) __TBB_override {
        buffer_operation op_data(del_blt_succ);
        op_data.r = &r;
        my_aggregator.execute(&op_data);
    }

    void internal_add_built_predecessor( predecessor_type &p) __TBB_override {
        buffer_operation op_data(add_blt_pred);
        op_data.p = &p;
        my_aggregator.execute(&op_data);
    }

    void internal_delete_built_predecessor( predecessor_type &p) __TBB_override {
        buffer_operation op_data(del_blt_pred);
        op_data.p = &p;
        my_aggregator.execute(&op_data);
    }

    size_t predecessor_count() __TBB_override {
        buffer_operation op_data(blt_pred_cnt);
        my_aggregator.execute(&op_data);
        return op_data.cnt_val;
    }

    size_t successor_count() __TBB_override {
        buffer_operation op_data(blt_succ_cnt);
        my_aggregator.execute(&op_data);
        return op_data.cnt_val;
    }

    void copy_predecessors( predecessor_list_type &v ) __TBB_override {
        buffer_operation op_data(blt_pred_cpy);
        op_data.pvec = &v;
        my_aggregator.execute(&op_data);
    }

    void copy_successors( successor_list_type &v ) __TBB_override {
        buffer_operation op_data(blt_succ_cpy);
        op_data.svec = &v;
        my_aggregator.execute(&op_data);
    }

#endif

    //! Removes a successor.
    /** Removes successor r from the list of successors.
        It also calls r.remove_predecessor(*this) to remove this node as a predecessor. */
    bool remove_successor( successor_type &r ) __TBB_override {
        r.remove_predecessor(*this);
        buffer_operation op_data(rem_succ);
        op_data.r = &r;
        my_aggregator.execute(&op_data);
        // even though this operation does not cause a forward, if we are the handler, and
        // a forward is scheduled, we may be the first to reach this point after the aggregator,
        // and so should check for the task.
        (void)enqueue_forwarding_task(op_data);
        return true;
    }

    //! Request an item from the buffer_node
    /**  true = v contains the returned item<BR>
         false = no item has been returned */
    bool try_get( T &v ) __TBB_override {
        buffer_operation op_data(req_item);
        op_data.elem = &v;
        my_aggregator.execute(&op_data);
        (void)enqueue_forwarding_task(op_data);
        return (op_data.status==internal::SUCCEEDED);
    }

    //! Reserves an item.
    /**  false = no item can be reserved<BR>
         true = an item is reserved */
    bool try_reserve( T &v ) __TBB_override {
        buffer_operation op_data(res_item);
        op_data.elem = &v;
        my_aggregator.execute(&op_data);
        (void)enqueue_forwarding_task(op_data);
        return (op_data.status==internal::SUCCEEDED);
    }

    //! Release a reserved item.
    /**  true = item has been released and so remains in sender */
    bool try_release() __TBB_override {
        buffer_operation op_data(rel_res);
        my_aggregator.execute(&op_data);
        (void)enqueue_forwarding_task(op_data);
        return true;
    }

    //! Consumes a reserved item.
    /** true = item is removed from sender and reservation removed */
    bool try_consume() __TBB_override {
        buffer_operation op_data(con_res);
        my_aggregator.execute(&op_data);
        (void)enqueue_forwarding_task(op_data);
        return true;
    }

protected:

    template< typename R, typename B > friend class run_and_put_task;
    template<typename X, typename Y> friend class internal::broadcast_cache;
    template<typename X, typename Y> friend class internal::round_robin_cache;
    //! receive an item, return a task *if possible
    task *try_put_task(const T &t) __TBB_override {
        buffer_operation op_data(t, put_item);
        my_aggregator.execute(&op_data);
        task *ft = grab_forwarding_task(op_data);
        // sequencer_nodes can return failure (if an item has been previously inserted)
        // We have to spawn the returned task if our own operation fails.

        if(ft && op_data.status ==internal::FAILED) {
            // we haven't succeeded queueing the item, but for some reason the
            // call returned a task (if another request resulted in a successful
            // forward this could happen.)  Queue the task and reset the pointer.
            internal::spawn_in_graph_arena(graph_reference(), *ft); ft = NULL;
        }
        else if(!ft && op_data.status ==internal::SUCCEEDED) {
            ft = SUCCESSFULLY_ENQUEUED;
        }
        return ft;
    }

    graph& graph_reference() __TBB_override {
        return my_graph;
    }

    void reset_receiver(reset_flags /*f*/) __TBB_override { }

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
public:
    void extract() __TBB_override {
        my_built_predecessors.receiver_extract(*this);
        my_successors.built_successors().sender_extract(*this);
    }
#endif

protected:
    void reset_node( reset_flags f) __TBB_override {
        internal::reservable_item_buffer<T, A>::reset();
        // TODO: just clear structures
        if (f&rf_clear_edges) {
            my_successors.clear();
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
            my_built_predecessors.clear();
#endif
        }
        forwarder_busy = false;
    }
};  // buffer_node

//! Forwards messages in FIFO order
template <typename T, typename A=cache_aligned_allocator<T> >
class queue_node : public buffer_node<T, A> {
protected:
    typedef buffer_node<T, A> base_type;
    typedef typename base_type::size_type size_type;
    typedef typename base_type::buffer_operation queue_operation;
    typedef queue_node class_type;

private:
    template<typename, typename> friend class buffer_node;

    bool is_item_valid() {
        return this->my_item_valid(this->my_head);
    }

    void try_put_and_add_task(task*& last_task) {
        task *new_task = this->my_successors.try_put_task(this->front());
        if (new_task) {
            // workaround for icc bug
            graph& graph_ref = this->graph_reference();
            last_task = combine_tasks(graph_ref, last_task, new_task);
            this->destroy_front();
        }
    }

protected:
    void internal_forward_task(queue_operation *op) __TBB_override {
        this->internal_forward_task_impl(op, this);
    }

    void internal_pop(queue_operation *op) __TBB_override {
        if ( this->my_reserved || !this->my_item_valid(this->my_head)){
            __TBB_store_with_release(op->status, internal::FAILED);
        }
        else {
            this->pop_front(*(op->elem));
            __TBB_store_with_release(op->status, internal::SUCCEEDED);
        }
    }
    void internal_reserve(queue_operation *op) __TBB_override {
        if (this->my_reserved || !this->my_item_valid(this->my_head)) {
            __TBB_store_with_release(op->status, internal::FAILED);
        }
        else {
            this->reserve_front(*(op->elem));
            __TBB_store_with_release(op->status, internal::SUCCEEDED);
        }
    }
    void internal_consume(queue_operation *op) __TBB_override {
        this->consume_front();
        __TBB_store_with_release(op->status, internal::SUCCEEDED);
    }

public:
    typedef T input_type;
    typedef T output_type;
    typedef typename receiver<input_type>::predecessor_type predecessor_type;
    typedef typename sender<output_type>::successor_type successor_type;

    //! Constructor
    explicit queue_node( graph &g ) : base_type(g) {
        tbb::internal::fgt_node( tbb::internal::FLOW_QUEUE_NODE, &(this->my_graph),
                                 static_cast<receiver<input_type> *>(this),
                                 static_cast<sender<output_type> *>(this) );
    }

    //! Copy constructor
    queue_node( const queue_node& src) : base_type(src) {
        tbb::internal::fgt_node( tbb::internal::FLOW_QUEUE_NODE, &(this->my_graph),
                                 static_cast<receiver<input_type> *>(this),
                                 static_cast<sender<output_type> *>(this) );
    }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    void set_name( const char *name ) __TBB_override {
        tbb::internal::fgt_node_desc( this, name );
    }
#endif

protected:
    void reset_node( reset_flags f) __TBB_override {
        base_type::reset_node(f);
    }
};  // queue_node

//! Forwards messages in sequence order
template< typename T, typename A=cache_aligned_allocator<T> >
class sequencer_node : public queue_node<T, A> {
    internal::function_body< T, size_t > *my_sequencer;
    // my_sequencer should be a benign function and must be callable
    // from a parallel context.  Does this mean it needn't be reset?
public:
    typedef T input_type;
    typedef T output_type;
    typedef typename receiver<input_type>::predecessor_type predecessor_type;
    typedef typename sender<output_type>::successor_type successor_type;

    //! Constructor
    template< typename Sequencer >
    sequencer_node( graph &g, const Sequencer& s ) : queue_node<T, A>(g),
        my_sequencer(new internal::function_body_leaf< T, size_t, Sequencer>(s) ) {
        tbb::internal::fgt_node( tbb::internal::FLOW_SEQUENCER_NODE, &(this->my_graph),
                                 static_cast<receiver<input_type> *>(this),
                                 static_cast<sender<output_type> *>(this) );
    }

    //! Copy constructor
    sequencer_node( const sequencer_node& src ) : queue_node<T, A>(src),
        my_sequencer( src.my_sequencer->clone() ) {
        tbb::internal::fgt_node( tbb::internal::FLOW_SEQUENCER_NODE, &(this->my_graph),
                                 static_cast<receiver<input_type> *>(this),
                                 static_cast<sender<output_type> *>(this) );
    }

    //! Destructor
    ~sequencer_node() { delete my_sequencer; }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    void set_name( const char *name ) __TBB_override {
        tbb::internal::fgt_node_desc( this, name );
    }
#endif

protected:
    typedef typename buffer_node<T, A>::size_type size_type;
    typedef typename buffer_node<T, A>::buffer_operation sequencer_operation;

private:
    bool internal_push(sequencer_operation *op) __TBB_override {
        size_type tag = (*my_sequencer)(*(op->elem));
#if !TBB_DEPRECATED_SEQUENCER_DUPLICATES
        if (tag < this->my_head) {
            // have already emitted a message with this tag
            __TBB_store_with_release(op->status, internal::FAILED);
            return false;
        }
#endif
        // cannot modify this->my_tail now; the buffer would be inconsistent.
        size_t new_tail = (tag+1 > this->my_tail) ? tag+1 : this->my_tail;

        if (this->size(new_tail) > this->capacity()) {
            this->grow_my_array(this->size(new_tail));
        }
        this->my_tail = new_tail;

        const internal::op_stat res = this->place_item(tag, *(op->elem)) ? internal::SUCCEEDED : internal::FAILED;
        __TBB_store_with_release(op->status, res);
        return res ==internal::SUCCEEDED;
    }
};  // sequencer_node

//! Forwards messages in priority order
template< typename T, typename Compare = std::less<T>, typename A=cache_aligned_allocator<T> >
class priority_queue_node : public buffer_node<T, A> {
public:
    typedef T input_type;
    typedef T output_type;
    typedef buffer_node<T,A> base_type;
    typedef priority_queue_node class_type;
    typedef typename receiver<input_type>::predecessor_type predecessor_type;
    typedef typename sender<output_type>::successor_type successor_type;

    //! Constructor
    explicit priority_queue_node( graph &g ) : buffer_node<T, A>(g), mark(0) {
        tbb::internal::fgt_node( tbb::internal::FLOW_PRIORITY_QUEUE_NODE, &(this->my_graph),
                                 static_cast<receiver<input_type> *>(this),
                                 static_cast<sender<output_type> *>(this) );
    }

    //! Copy constructor
    priority_queue_node( const priority_queue_node &src ) : buffer_node<T, A>(src), mark(0) {
        tbb::internal::fgt_node( tbb::internal::FLOW_PRIORITY_QUEUE_NODE, &(this->my_graph),
                                 static_cast<receiver<input_type> *>(this),
                                 static_cast<sender<output_type> *>(this) );
    }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    void set_name( const char *name ) __TBB_override {
        tbb::internal::fgt_node_desc( this, name );
    }
#endif

protected:

    void reset_node( reset_flags f) __TBB_override {
        mark = 0;
        base_type::reset_node(f);
    }

    typedef typename buffer_node<T, A>::size_type size_type;
    typedef typename buffer_node<T, A>::item_type item_type;
    typedef typename buffer_node<T, A>::buffer_operation prio_operation;

    //! Tries to forward valid items to successors
    void internal_forward_task(prio_operation *op) __TBB_override {
        this->internal_forward_task_impl(op, this);
    }

    void handle_operations(prio_operation *op_list) __TBB_override {
        this->handle_operations_impl(op_list, this);
    }

    bool internal_push(prio_operation *op) __TBB_override {
        prio_push(*(op->elem));
        __TBB_store_with_release(op->status, internal::SUCCEEDED);
        return true;
    }

    void internal_pop(prio_operation *op) __TBB_override {
        // if empty or already reserved, don't pop
        if ( this->my_reserved == true || this->my_tail == 0 ) {
            __TBB_store_with_release(op->status, internal::FAILED);
            return;
        }

        *(op->elem) = prio();
        __TBB_store_with_release(op->status, internal::SUCCEEDED);
        prio_pop();

    }

    // pops the highest-priority item, saves copy
    void internal_reserve(prio_operation *op) __TBB_override {
        if (this->my_reserved == true || this->my_tail == 0) {
            __TBB_store_with_release(op->status, internal::FAILED);
            return;
        }
        this->my_reserved = true;
        *(op->elem) = prio();
        reserved_item = *(op->elem);
        __TBB_store_with_release(op->status, internal::SUCCEEDED);
        prio_pop();
    }

    void internal_consume(prio_operation *op) __TBB_override {
        __TBB_store_with_release(op->status, internal::SUCCEEDED);
        this->my_reserved = false;
        reserved_item = input_type();
    }

    void internal_release(prio_operation *op) __TBB_override {
        __TBB_store_with_release(op->status, internal::SUCCEEDED);
        prio_push(reserved_item);
        this->my_reserved = false;
        reserved_item = input_type();
    }

private:
    template<typename, typename> friend class buffer_node;

    void order() {
        if (mark < this->my_tail) heapify();
        __TBB_ASSERT(mark == this->my_tail, "mark unequal after heapify");
    }

    bool is_item_valid() {
        return this->my_tail > 0;
    }

    void try_put_and_add_task(task*& last_task) {
        task * new_task = this->my_successors.try_put_task(this->prio());
        if (new_task) {
            // workaround for icc bug
            graph& graph_ref = this->graph_reference();
            last_task = combine_tasks(graph_ref, last_task, new_task);
            prio_pop();
        }
    }

private:
    Compare compare;
    size_type mark;

    input_type reserved_item;

    // in case a reheap has not been done after a push, check if the mark item is higher than the 0'th item
    bool prio_use_tail() {
        __TBB_ASSERT(mark <= this->my_tail, "mark outside bounds before test");
        return mark < this->my_tail && compare(this->get_my_item(0), this->get_my_item(this->my_tail - 1));
    }

    // prio_push: checks that the item will fit, expand array if necessary, put at end
    void prio_push(const T &src) {
        if ( this->my_tail >= this->my_array_size )
            this->grow_my_array( this->my_tail + 1 );
        (void) this->place_item(this->my_tail, src);
        ++(this->my_tail);
        __TBB_ASSERT(mark < this->my_tail, "mark outside bounds after push");
    }

    // prio_pop: deletes highest priority item from the array, and if it is item
    // 0, move last item to 0 and reheap.  If end of array, just destroy and decrement tail
    // and mark.  Assumes the array has already been tested for emptiness; no failure.
    void prio_pop()  {
        if (prio_use_tail()) {
            // there are newly pushed elements; last one higher than top
            // copy the data
            this->destroy_item(this->my_tail-1);
            --(this->my_tail);
            __TBB_ASSERT(mark <= this->my_tail, "mark outside bounds after pop");
            return;
        }
        this->destroy_item(0);
        if(this->my_tail > 1) {
            // push the last element down heap
            __TBB_ASSERT(this->my_item_valid(this->my_tail - 1), NULL);
            this->move_item(0,this->my_tail - 1);
        }
        --(this->my_tail);
        if(mark > this->my_tail) --mark;
        if (this->my_tail > 1) // don't reheap for heap of size 1
            reheap();
        __TBB_ASSERT(mark <= this->my_tail, "mark outside bounds after pop");
    }

    const T& prio() {
        return this->get_my_item(prio_use_tail() ? this->my_tail-1 : 0);
    }

    // turn array into heap
    void heapify() {
        if(this->my_tail == 0) {
            mark = 0;
            return;
        }
        if (!mark) mark = 1;
        for (; mark<this->my_tail; ++mark) { // for each unheaped element
            size_type cur_pos = mark;
            input_type to_place;
            this->fetch_item(mark,to_place);
            do { // push to_place up the heap
                size_type parent = (cur_pos-1)>>1;
                if (!compare(this->get_my_item(parent), to_place))
                    break;
                this->move_item(cur_pos, parent);
                cur_pos = parent;
            } while( cur_pos );
            (void) this->place_item(cur_pos, to_place);
        }
    }

    // otherwise heapified array with new root element; rearrange to heap
    void reheap() {
        size_type cur_pos=0, child=1;
        while (child < mark) {
            size_type target = child;
            if (child+1<mark &&
                compare(this->get_my_item(child),
                        this->get_my_item(child+1)))
                ++target;
            // target now has the higher priority child
            if (compare(this->get_my_item(target),
                        this->get_my_item(cur_pos)))
                break;
            // swap
            this->swap_items(cur_pos, target);
            cur_pos = target;
            child = (cur_pos<<1)+1;
        }
    }
};  // priority_queue_node

//! Forwards messages only if the threshold has not been reached
/** This node forwards items until its threshold is reached.
    It contains no buffering.  If the downstream node rejects, the
    message is dropped. */
template< typename T >
class limiter_node : public graph_node, public receiver< T >, public sender< T > {
public:
    typedef T input_type;
    typedef T output_type;
    typedef typename receiver<input_type>::predecessor_type predecessor_type;
    typedef typename sender<output_type>::successor_type successor_type;
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    typedef typename receiver<input_type>::built_predecessors_type built_predecessors_type;
    typedef typename sender<output_type>::built_successors_type built_successors_type;
    typedef typename receiver<input_type>::predecessor_list_type predecessor_list_type;
    typedef typename sender<output_type>::successor_list_type successor_list_type;
#endif
    //TODO: There is a lack of predefined types for its controlling "decrementer" port. It should be fixed later.

private:
    size_t my_threshold;
    size_t my_count; //number of successful puts
    size_t my_tries; //number of active put attempts
    internal::reservable_predecessor_cache< T, spin_mutex > my_predecessors;
    spin_mutex my_mutex;
    internal::broadcast_cache< T > my_successors;
    int init_decrement_predecessors;

    friend class internal::forward_task_bypass< limiter_node<T> >;

    // Let decrementer call decrement_counter()
    friend class internal::decrementer< limiter_node<T> >;

    bool check_conditions() {  // always called under lock
        return ( my_count + my_tries < my_threshold && !my_predecessors.empty() && !my_successors.empty() );
    }

    // only returns a valid task pointer or NULL, never SUCCESSFULLY_ENQUEUED
    task *forward_task() {
        input_type v;
        task *rval = NULL;
        bool reserved = false;
            {
                spin_mutex::scoped_lock lock(my_mutex);
                if ( check_conditions() )
                    ++my_tries;
                else
                    return NULL;
            }

        //SUCCESS
        // if we can reserve and can put, we consume the reservation
        // we increment the count and decrement the tries
        if ( (my_predecessors.try_reserve(v)) == true ){
            reserved=true;
            if ( (rval = my_successors.try_put_task(v)) != NULL ){
                {
                    spin_mutex::scoped_lock lock(my_mutex);
                    ++my_count;
                    --my_tries;
                    my_predecessors.try_consume();
                    if ( check_conditions() ) {
                        if ( internal::is_graph_active(this->my_graph) ) {
                            task *rtask = new ( task::allocate_additional_child_of( *(this->my_graph.root_task()) ) )
                                internal::forward_task_bypass< limiter_node<T> >( *this );
                            internal::spawn_in_graph_arena(graph_reference(), *rtask);
                        }
                    }
                }
                return rval;
            }
        }
        //FAILURE
        //if we can't reserve, we decrement the tries
        //if we can reserve but can't put, we decrement the tries and release the reservation
        {
            spin_mutex::scoped_lock lock(my_mutex);
            --my_tries;
            if (reserved) my_predecessors.try_release();
            if ( check_conditions() ) {
                if ( internal::is_graph_active(this->my_graph) ) {
                    task *rtask = new ( task::allocate_additional_child_of( *(this->my_graph.root_task()) ) )
                        internal::forward_task_bypass< limiter_node<T> >( *this );
                    __TBB_ASSERT(!rval, "Have two tasks to handle");
                    return rtask;
                }
            }
            return rval;
        }
    }

    void forward() {
        __TBB_ASSERT(false, "Should never be called");
        return;
    }

    task * decrement_counter() {
        {
            spin_mutex::scoped_lock lock(my_mutex);
            if(my_count) --my_count;
        }
        return forward_task();
    }

public:
    //! The internal receiver< continue_msg > that decrements the count
    internal::decrementer< limiter_node<T> > decrement;

    //! Constructor
    limiter_node(graph &g, size_t threshold, int num_decrement_predecessors=0) :
        graph_node(g), my_threshold(threshold), my_count(0), my_tries(0),
        init_decrement_predecessors(num_decrement_predecessors),
        decrement(num_decrement_predecessors)
    {
        my_predecessors.set_owner(this);
        my_successors.set_owner(this);
        decrement.set_owner(this);
        tbb::internal::fgt_node( tbb::internal::FLOW_LIMITER_NODE, &this->my_graph,
                                 static_cast<receiver<input_type> *>(this), static_cast<receiver<continue_msg> *>(&decrement),
                                 static_cast<sender<output_type> *>(this) );
    }

    //! Copy constructor
    limiter_node( const limiter_node& src ) :
        graph_node(src.my_graph), receiver<T>(), sender<T>(),
        my_threshold(src.my_threshold), my_count(0), my_tries(0),
        init_decrement_predecessors(src.init_decrement_predecessors),
        decrement(src.init_decrement_predecessors)
    {
        my_predecessors.set_owner(this);
        my_successors.set_owner(this);
        decrement.set_owner(this);
        tbb::internal::fgt_node( tbb::internal::FLOW_LIMITER_NODE, &this->my_graph,
                                 static_cast<receiver<input_type> *>(this), static_cast<receiver<continue_msg> *>(&decrement),
                                 static_cast<sender<output_type> *>(this) );
    }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    void set_name( const char *name ) __TBB_override {
        tbb::internal::fgt_node_desc( this, name );
    }
#endif

    //! Replace the current successor with this new successor
    bool register_successor( successor_type &r ) __TBB_override {
        spin_mutex::scoped_lock lock(my_mutex);
        bool was_empty = my_successors.empty();
        my_successors.register_successor(r);
        //spawn a forward task if this is the only successor
        if ( was_empty && !my_predecessors.empty() && my_count + my_tries < my_threshold ) {
            if ( internal::is_graph_active(this->my_graph) ) {
                task* task = new ( task::allocate_additional_child_of( *(this->my_graph.root_task()) ) )
                            internal::forward_task_bypass < limiter_node<T> >( *this );
                internal::spawn_in_graph_arena(graph_reference(), *task);
            }
        }
        return true;
    }

    //! Removes a successor from this node
    /** r.remove_predecessor(*this) is also called. */
    bool remove_successor( successor_type &r ) __TBB_override {
        r.remove_predecessor(*this);
        my_successors.remove_successor(r);
        return true;
    }

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    built_successors_type &built_successors() __TBB_override { return my_successors.built_successors(); }
    built_predecessors_type &built_predecessors() __TBB_override { return my_predecessors.built_predecessors(); }

    void internal_add_built_successor(successor_type &src) __TBB_override {
        my_successors.internal_add_built_successor(src);
    }

    void internal_delete_built_successor(successor_type &src) __TBB_override {
        my_successors.internal_delete_built_successor(src);
    }

    size_t successor_count() __TBB_override { return my_successors.successor_count(); }

    void copy_successors(successor_list_type &v) __TBB_override {
        my_successors.copy_successors(v);
    }

    void internal_add_built_predecessor(predecessor_type &src) __TBB_override {
        my_predecessors.internal_add_built_predecessor(src);
    }

    void internal_delete_built_predecessor(predecessor_type &src) __TBB_override {
        my_predecessors.internal_delete_built_predecessor(src);
    }

    size_t predecessor_count() __TBB_override { return my_predecessors.predecessor_count(); }

    void copy_predecessors(predecessor_list_type &v) __TBB_override {
        my_predecessors.copy_predecessors(v);
    }

    void extract() __TBB_override {
        my_count = 0;
        my_successors.built_successors().sender_extract(*this);
        my_predecessors.built_predecessors().receiver_extract(*this);
        decrement.built_predecessors().receiver_extract(decrement);
    }
#endif  /* TBB_PREVIEW_FLOW_GRAPH_FEATURES */

    //! Adds src to the list of cached predecessors.
    bool register_predecessor( predecessor_type &src ) __TBB_override {
        spin_mutex::scoped_lock lock(my_mutex);
        my_predecessors.add( src );
        if ( my_count + my_tries < my_threshold && !my_successors.empty() && internal::is_graph_active(this->my_graph) ) {
            task* task = new ( task::allocate_additional_child_of( *(this->my_graph.root_task()) ) )
                        internal::forward_task_bypass < limiter_node<T> >( *this );
            internal::spawn_in_graph_arena(graph_reference(), *task);
        }
        return true;
    }

    //! Removes src from the list of cached predecessors.
    bool remove_predecessor( predecessor_type &src ) __TBB_override {
        my_predecessors.remove( src );
        return true;
    }

protected:

    template< typename R, typename B > friend class run_and_put_task;
    template<typename X, typename Y> friend class internal::broadcast_cache;
    template<typename X, typename Y> friend class internal::round_robin_cache;
    //! Puts an item to this receiver
    task *try_put_task( const T &t ) __TBB_override {
        {
            spin_mutex::scoped_lock lock(my_mutex);
            if ( my_count + my_tries >= my_threshold )
                return NULL;
            else
                ++my_tries;
        }

        task * rtask = my_successors.try_put_task(t);

        if ( !rtask ) {  // try_put_task failed.
            spin_mutex::scoped_lock lock(my_mutex);
            --my_tries;
            if (check_conditions() && internal::is_graph_active(this->my_graph)) {
                rtask = new ( task::allocate_additional_child_of( *(this->my_graph.root_task()) ) )
                    internal::forward_task_bypass< limiter_node<T> >( *this );
            }
        }
        else {
            spin_mutex::scoped_lock lock(my_mutex);
            ++my_count;
            --my_tries;
             }
        return rtask;
    }

    graph& graph_reference() __TBB_override {
        return my_graph;
    }

    void reset_receiver(reset_flags /*f*/) __TBB_override {
        __TBB_ASSERT(false,NULL);  // should never be called
    }

    void reset_node( reset_flags f) __TBB_override {
        my_count = 0;
        if(f & rf_clear_edges) {
            my_predecessors.clear();
            my_successors.clear();
        }
        else
        {
            my_predecessors.reset( );
        }
        decrement.reset_receiver(f);
    }
};  // limiter_node

#include "internal/_flow_graph_join_impl.h"

using internal::reserving_port;
using internal::queueing_port;
using internal::key_matching_port;
using internal::input_port;
using internal::tag_value;

template<typename OutputTuple, typename JP=queueing> class join_node;

template<typename OutputTuple>
class join_node<OutputTuple,reserving>: public internal::unfolded_join_node<tbb::flow::tuple_size<OutputTuple>::value, reserving_port, OutputTuple, reserving> {
private:
    static const int N = tbb::flow::tuple_size<OutputTuple>::value;
    typedef typename internal::unfolded_join_node<N, reserving_port, OutputTuple, reserving> unfolded_type;
public:
    typedef OutputTuple output_type;
    typedef typename unfolded_type::input_ports_type input_ports_type;
    explicit join_node(graph &g) : unfolded_type(g) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_JOIN_NODE_RESERVING, &this->my_graph,
                                            this->input_ports(), static_cast< sender< output_type > *>(this) );
    }
    join_node(const join_node &other) : unfolded_type(other) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_JOIN_NODE_RESERVING, &this->my_graph,
                                            this->input_ports(), static_cast< sender< output_type > *>(this) );
    }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    void set_name( const char *name ) __TBB_override {
        tbb::internal::fgt_node_desc( this, name );
    }
#endif

};

template<typename OutputTuple>
class join_node<OutputTuple,queueing>: public internal::unfolded_join_node<tbb::flow::tuple_size<OutputTuple>::value, queueing_port, OutputTuple, queueing> {
private:
    static const int N = tbb::flow::tuple_size<OutputTuple>::value;
    typedef typename internal::unfolded_join_node<N, queueing_port, OutputTuple, queueing> unfolded_type;
public:
    typedef OutputTuple output_type;
    typedef typename unfolded_type::input_ports_type input_ports_type;
    explicit join_node(graph &g) : unfolded_type(g) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_JOIN_NODE_QUEUEING, &this->my_graph,
                                            this->input_ports(), static_cast< sender< output_type > *>(this) );
    }
    join_node(const join_node &other) : unfolded_type(other) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_JOIN_NODE_QUEUEING, &this->my_graph,
                                            this->input_ports(), static_cast< sender< output_type > *>(this) );
    }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    void set_name( const char *name ) __TBB_override {
        tbb::internal::fgt_node_desc( this, name );
    }
#endif

};

// template for key_matching join_node
// tag_matching join_node is a specialization of key_matching, and is source-compatible.
template<typename OutputTuple, typename K, typename KHash>
class join_node<OutputTuple, key_matching<K, KHash> > : public internal::unfolded_join_node<tbb::flow::tuple_size<OutputTuple>::value,
      key_matching_port, OutputTuple, key_matching<K,KHash> > {
private:
    static const int N = tbb::flow::tuple_size<OutputTuple>::value;
    typedef typename internal::unfolded_join_node<N, key_matching_port, OutputTuple, key_matching<K,KHash> > unfolded_type;
public:
    typedef OutputTuple output_type;
    typedef typename unfolded_type::input_ports_type input_ports_type;

#if __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING
    join_node(graph &g) : unfolded_type(g) {}
#endif  /* __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING */

    template<typename __TBB_B0, typename __TBB_B1>
    join_node(graph &g, __TBB_B0 b0, __TBB_B1 b1) : unfolded_type(g, b0, b1) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_JOIN_NODE_TAG_MATCHING, &this->my_graph,
                                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }
    template<typename __TBB_B0, typename __TBB_B1, typename __TBB_B2>
    join_node(graph &g, __TBB_B0 b0, __TBB_B1 b1, __TBB_B2 b2) : unfolded_type(g, b0, b1, b2) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_JOIN_NODE_TAG_MATCHING, &this->my_graph,
                                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }
    template<typename __TBB_B0, typename __TBB_B1, typename __TBB_B2, typename __TBB_B3>
    join_node(graph &g, __TBB_B0 b0, __TBB_B1 b1, __TBB_B2 b2, __TBB_B3 b3) : unfolded_type(g, b0, b1, b2, b3) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_JOIN_NODE_TAG_MATCHING, &this->my_graph,
                                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }
    template<typename __TBB_B0, typename __TBB_B1, typename __TBB_B2, typename __TBB_B3, typename __TBB_B4>
    join_node(graph &g, __TBB_B0 b0, __TBB_B1 b1, __TBB_B2 b2, __TBB_B3 b3, __TBB_B4 b4) :
            unfolded_type(g, b0, b1, b2, b3, b4) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_JOIN_NODE_TAG_MATCHING, &this->my_graph,
                                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }
#if __TBB_VARIADIC_MAX >= 6
    template<typename __TBB_B0, typename __TBB_B1, typename __TBB_B2, typename __TBB_B3, typename __TBB_B4,
        typename __TBB_B5>
    join_node(graph &g, __TBB_B0 b0, __TBB_B1 b1, __TBB_B2 b2, __TBB_B3 b3, __TBB_B4 b4, __TBB_B5 b5) :
            unfolded_type(g, b0, b1, b2, b3, b4, b5) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_JOIN_NODE_TAG_MATCHING, &this->my_graph,
                                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }
#endif
#if __TBB_VARIADIC_MAX >= 7
    template<typename __TBB_B0, typename __TBB_B1, typename __TBB_B2, typename __TBB_B3, typename __TBB_B4,
        typename __TBB_B5, typename __TBB_B6>
    join_node(graph &g, __TBB_B0 b0, __TBB_B1 b1, __TBB_B2 b2, __TBB_B3 b3, __TBB_B4 b4, __TBB_B5 b5, __TBB_B6 b6) :
            unfolded_type(g, b0, b1, b2, b3, b4, b5, b6) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_JOIN_NODE_TAG_MATCHING, &this->my_graph,
                                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }
#endif
#if __TBB_VARIADIC_MAX >= 8
    template<typename __TBB_B0, typename __TBB_B1, typename __TBB_B2, typename __TBB_B3, typename __TBB_B4,
        typename __TBB_B5, typename __TBB_B6, typename __TBB_B7>
    join_node(graph &g, __TBB_B0 b0, __TBB_B1 b1, __TBB_B2 b2, __TBB_B3 b3, __TBB_B4 b4, __TBB_B5 b5, __TBB_B6 b6,
            __TBB_B7 b7) : unfolded_type(g, b0, b1, b2, b3, b4, b5, b6, b7) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_JOIN_NODE_TAG_MATCHING, &this->my_graph,
                                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }
#endif
#if __TBB_VARIADIC_MAX >= 9
    template<typename __TBB_B0, typename __TBB_B1, typename __TBB_B2, typename __TBB_B3, typename __TBB_B4,
        typename __TBB_B5, typename __TBB_B6, typename __TBB_B7, typename __TBB_B8>
    join_node(graph &g, __TBB_B0 b0, __TBB_B1 b1, __TBB_B2 b2, __TBB_B3 b3, __TBB_B4 b4, __TBB_B5 b5, __TBB_B6 b6,
            __TBB_B7 b7, __TBB_B8 b8) : unfolded_type(g, b0, b1, b2, b3, b4, b5, b6, b7, b8) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_JOIN_NODE_TAG_MATCHING, &this->my_graph,
                                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }
#endif
#if __TBB_VARIADIC_MAX >= 10
    template<typename __TBB_B0, typename __TBB_B1, typename __TBB_B2, typename __TBB_B3, typename __TBB_B4,
        typename __TBB_B5, typename __TBB_B6, typename __TBB_B7, typename __TBB_B8, typename __TBB_B9>
    join_node(graph &g, __TBB_B0 b0, __TBB_B1 b1, __TBB_B2 b2, __TBB_B3 b3, __TBB_B4 b4, __TBB_B5 b5, __TBB_B6 b6,
            __TBB_B7 b7, __TBB_B8 b8, __TBB_B9 b9) : unfolded_type(g, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_JOIN_NODE_TAG_MATCHING, &this->my_graph,
                                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }
#endif
    join_node(const join_node &other) : unfolded_type(other) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_JOIN_NODE_TAG_MATCHING, &this->my_graph,
                                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    void set_name( const char *name ) __TBB_override {
        tbb::internal::fgt_node_desc( this, name );
    }
#endif

};

// indexer node
#include "internal/_flow_graph_indexer_impl.h"

// TODO: Implement interface with variadic template or tuple
template<typename T0, typename T1=null_type, typename T2=null_type, typename T3=null_type,
                      typename T4=null_type, typename T5=null_type, typename T6=null_type,
                      typename T7=null_type, typename T8=null_type, typename T9=null_type> class indexer_node;

//indexer node specializations
template<typename T0>
class indexer_node<T0> : public internal::unfolded_indexer_node<tuple<T0> > {
private:
    static const int N = 1;
public:
    typedef tuple<T0> InputTuple;
    typedef typename internal::tagged_msg<size_t, T0> output_type;
    typedef typename internal::unfolded_indexer_node<InputTuple> unfolded_type;
    indexer_node(graph& g) : unfolded_type(g) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }
    // Copy constructor
    indexer_node( const indexer_node& other ) : unfolded_type(other) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
     void set_name( const char *name ) {
        tbb::internal::fgt_node_desc( this, name );
    }
#endif
};

template<typename T0, typename T1>
class indexer_node<T0, T1> : public internal::unfolded_indexer_node<tuple<T0, T1> > {
private:
    static const int N = 2;
public:
    typedef tuple<T0, T1> InputTuple;
    typedef typename internal::tagged_msg<size_t, T0, T1> output_type;
    typedef typename internal::unfolded_indexer_node<InputTuple> unfolded_type;
    indexer_node(graph& g) : unfolded_type(g) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }
    // Copy constructor
    indexer_node( const indexer_node& other ) : unfolded_type(other) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
     void set_name( const char *name ) {
        tbb::internal::fgt_node_desc( this, name );
    }
#endif
};

template<typename T0, typename T1, typename T2>
class indexer_node<T0, T1, T2> : public internal::unfolded_indexer_node<tuple<T0, T1, T2> > {
private:
    static const int N = 3;
public:
    typedef tuple<T0, T1, T2> InputTuple;
    typedef typename internal::tagged_msg<size_t, T0, T1, T2> output_type;
    typedef typename internal::unfolded_indexer_node<InputTuple> unfolded_type;
    indexer_node(graph& g) : unfolded_type(g) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }
    // Copy constructor
    indexer_node( const indexer_node& other ) : unfolded_type(other) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
        void set_name( const char *name ) {
        tbb::internal::fgt_node_desc( this, name );
    }
#endif
};

template<typename T0, typename T1, typename T2, typename T3>
class indexer_node<T0, T1, T2, T3> : public internal::unfolded_indexer_node<tuple<T0, T1, T2, T3> > {
private:
    static const int N = 4;
public:
    typedef tuple<T0, T1, T2, T3> InputTuple;
    typedef typename internal::tagged_msg<size_t, T0, T1, T2, T3> output_type;
    typedef typename internal::unfolded_indexer_node<InputTuple> unfolded_type;
    indexer_node(graph& g) : unfolded_type(g) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }
    // Copy constructor
    indexer_node( const indexer_node& other ) : unfolded_type(other) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    void set_name( const char *name ) __TBB_override {
        tbb::internal::fgt_node_desc( this, name );
    }
#endif
};

template<typename T0, typename T1, typename T2, typename T3, typename T4>
class indexer_node<T0, T1, T2, T3, T4> : public internal::unfolded_indexer_node<tuple<T0, T1, T2, T3, T4> > {
private:
    static const int N = 5;
public:
    typedef tuple<T0, T1, T2, T3, T4> InputTuple;
    typedef typename internal::tagged_msg<size_t, T0, T1, T2, T3, T4> output_type;
    typedef typename internal::unfolded_indexer_node<InputTuple> unfolded_type;
    indexer_node(graph& g) : unfolded_type(g) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }
    // Copy constructor
    indexer_node( const indexer_node& other ) : unfolded_type(other) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    void set_name( const char *name ) __TBB_override {
        tbb::internal::fgt_node_desc( this, name );
    }
#endif
};

#if __TBB_VARIADIC_MAX >= 6
template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5>
class indexer_node<T0, T1, T2, T3, T4, T5> : public internal::unfolded_indexer_node<tuple<T0, T1, T2, T3, T4, T5> > {
private:
    static const int N = 6;
public:
    typedef tuple<T0, T1, T2, T3, T4, T5> InputTuple;
    typedef typename internal::tagged_msg<size_t, T0, T1, T2, T3, T4, T5> output_type;
    typedef typename internal::unfolded_indexer_node<InputTuple> unfolded_type;
    indexer_node(graph& g) : unfolded_type(g) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }
    // Copy constructor
    indexer_node( const indexer_node& other ) : unfolded_type(other) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    void set_name( const char *name ) __TBB_override {
        tbb::internal::fgt_node_desc( this, name );
    }
#endif
};
#endif //variadic max 6

#if __TBB_VARIADIC_MAX >= 7
template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
         typename T6>
class indexer_node<T0, T1, T2, T3, T4, T5, T6> : public internal::unfolded_indexer_node<tuple<T0, T1, T2, T3, T4, T5, T6> > {
private:
    static const int N = 7;
public:
    typedef tuple<T0, T1, T2, T3, T4, T5, T6> InputTuple;
    typedef typename internal::tagged_msg<size_t, T0, T1, T2, T3, T4, T5, T6> output_type;
    typedef typename internal::unfolded_indexer_node<InputTuple> unfolded_type;
    indexer_node(graph& g) : unfolded_type(g) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }
    // Copy constructor
    indexer_node( const indexer_node& other ) : unfolded_type(other) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    void set_name( const char *name ) __TBB_override {
        tbb::internal::fgt_node_desc( this, name );
    }
#endif
};
#endif //variadic max 7

#if __TBB_VARIADIC_MAX >= 8
template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
         typename T6, typename T7>
class indexer_node<T0, T1, T2, T3, T4, T5, T6, T7> : public internal::unfolded_indexer_node<tuple<T0, T1, T2, T3, T4, T5, T6, T7> > {
private:
    static const int N = 8;
public:
    typedef tuple<T0, T1, T2, T3, T4, T5, T6, T7> InputTuple;
    typedef typename internal::tagged_msg<size_t, T0, T1, T2, T3, T4, T5, T6, T7> output_type;
    typedef typename internal::unfolded_indexer_node<InputTuple> unfolded_type;
    indexer_node(graph& g) : unfolded_type(g) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }
    // Copy constructor
    indexer_node( const indexer_node& other ) : unfolded_type(other) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    void set_name( const char *name ) __TBB_override {
        tbb::internal::fgt_node_desc( this, name );
    }
#endif
};
#endif //variadic max 8

#if __TBB_VARIADIC_MAX >= 9
template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
         typename T6, typename T7, typename T8>
class indexer_node<T0, T1, T2, T3, T4, T5, T6, T7, T8> : public internal::unfolded_indexer_node<tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8> > {
private:
    static const int N = 9;
public:
    typedef tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8> InputTuple;
    typedef typename internal::tagged_msg<size_t, T0, T1, T2, T3, T4, T5, T6, T7, T8> output_type;
    typedef typename internal::unfolded_indexer_node<InputTuple> unfolded_type;
    indexer_node(graph& g) : unfolded_type(g) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }
    // Copy constructor
    indexer_node( const indexer_node& other ) : unfolded_type(other) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    void set_name( const char *name ) __TBB_override {
        tbb::internal::fgt_node_desc( this, name );
    }
#endif
};
#endif //variadic max 9

#if __TBB_VARIADIC_MAX >= 10
template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
         typename T6, typename T7, typename T8, typename T9>
class indexer_node/*default*/ : public internal::unfolded_indexer_node<tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> > {
private:
    static const int N = 10;
public:
    typedef tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> InputTuple;
    typedef typename internal::tagged_msg<size_t, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> output_type;
    typedef typename internal::unfolded_indexer_node<InputTuple> unfolded_type;
    indexer_node(graph& g) : unfolded_type(g) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }
    // Copy constructor
    indexer_node( const indexer_node& other ) : unfolded_type(other) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    void set_name( const char *name ) __TBB_override {
        tbb::internal::fgt_node_desc( this, name );
    }
#endif
};
#endif //variadic max 10

#if __TBB_PREVIEW_ASYNC_MSG
inline void internal_make_edge( internal::untyped_sender &p, internal::untyped_receiver &s ) {
#else
template< typename T >
inline void internal_make_edge( sender<T> &p, receiver<T> &s ) {
#endif
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    s.internal_add_built_predecessor(p);
    p.internal_add_built_successor(s);
#endif
    p.register_successor( s );
    tbb::internal::fgt_make_edge( &p, &s );
}

//! Makes an edge between a single predecessor and a single successor
template< typename T >
inline void make_edge( sender<T> &p, receiver<T> &s ) {
    internal_make_edge( p, s );
}

#if __TBB_PREVIEW_ASYNC_MSG
template< typename TS, typename TR,
    typename = typename tbb::internal::enable_if<tbb::internal::is_same_type<TS, internal::untyped_sender>::value
                                              || tbb::internal::is_same_type<TR, internal::untyped_receiver>::value>::type>
inline void make_edge( TS &p, TR &s ) {
    internal_make_edge( p, s );
}

template< typename T >
inline void make_edge( sender<T> &p, receiver<typename T::async_msg_data_type> &s ) {
    internal_make_edge( p, s );
}

template< typename T >
inline void make_edge( sender<typename T::async_msg_data_type> &p, receiver<T> &s ) {
    internal_make_edge( p, s );
}

#endif // __TBB_PREVIEW_ASYNC_MSG

#if __TBB_FLOW_GRAPH_CPP11_FEATURES
//Makes an edge from port 0 of a multi-output predecessor to port 0 of a multi-input successor.
template< typename T, typename V,
          typename = typename T::output_ports_type, typename = typename V::input_ports_type >
inline void make_edge( T& output, V& input) {
    make_edge(get<0>(output.output_ports()), get<0>(input.input_ports()));
}

//Makes an edge from port 0 of a multi-output predecessor to a receiver.
template< typename T, typename R,
          typename = typename T::output_ports_type >
inline void make_edge( T& output, receiver<R>& input) {
     make_edge(get<0>(output.output_ports()), input);
}

//Makes an edge from a sender to port 0 of a multi-input successor.
template< typename S,  typename V,
          typename = typename V::input_ports_type >
inline void make_edge( sender<S>& output, V& input) {
     make_edge(output, get<0>(input.input_ports()));
}
#endif

#if __TBB_PREVIEW_ASYNC_MSG
inline void internal_remove_edge( internal::untyped_sender &p, internal::untyped_receiver &s ) {
#else
template< typename T >
inline void internal_remove_edge( sender<T> &p, receiver<T> &s ) {
#endif
    p.remove_successor( s );
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    // TODO: should we try to remove p from the predecessor list of s, in case the edge is reversed?
    p.internal_delete_built_successor(s);
    s.internal_delete_built_predecessor(p);
#endif
    tbb::internal::fgt_remove_edge( &p, &s );
}

//! Removes an edge between a single predecessor and a single successor
template< typename T >
inline void remove_edge( sender<T> &p, receiver<T> &s ) {
    internal_remove_edge( p, s );
}

#if __TBB_PREVIEW_ASYNC_MSG
template< typename TS, typename TR,
    typename = typename tbb::internal::enable_if<tbb::internal::is_same_type<TS, internal::untyped_sender>::value
                                              || tbb::internal::is_same_type<TR, internal::untyped_receiver>::value>::type>
inline void remove_edge( TS &p, TR &s ) {
    internal_remove_edge( p, s );
}

template< typename T >
inline void remove_edge( sender<T> &p, receiver<typename T::async_msg_data_type> &s ) {
    internal_remove_edge( p, s );
}

template< typename T >
inline void remove_edge( sender<typename T::async_msg_data_type> &p, receiver<T> &s ) {
    internal_remove_edge( p, s );
}
#endif // __TBB_PREVIEW_ASYNC_MSG

#if __TBB_FLOW_GRAPH_CPP11_FEATURES
//Removes an edge between port 0 of a multi-output predecessor and port 0 of a multi-input successor.
template< typename T, typename V,
          typename = typename T::output_ports_type, typename = typename V::input_ports_type >
inline void remove_edge( T& output, V& input) {
    remove_edge(get<0>(output.output_ports()), get<0>(input.input_ports()));
}

//Removes an edge between port 0 of a multi-output predecessor and a receiver.
template< typename T, typename R,
          typename = typename T::output_ports_type >
inline void remove_edge( T& output, receiver<R>& input) {
     remove_edge(get<0>(output.output_ports()), input);
}
//Removes an edge between a sender and port 0 of a multi-input successor.
template< typename S,  typename V,
          typename = typename V::input_ports_type >
inline void remove_edge( sender<S>& output, V& input) {
     remove_edge(output, get<0>(input.input_ports()));
}
#endif

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
template<typename C >
template< typename S >
void internal::edge_container<C>::sender_extract( S &s ) {
    edge_list_type e = built_edges;
    for ( typename edge_list_type::iterator i = e.begin(); i != e.end(); ++i ) {
        remove_edge(s, **i);
    }
}

template<typename C >
template< typename R >
void internal::edge_container<C>::receiver_extract( R &r ) {
    edge_list_type e = built_edges;
    for ( typename edge_list_type::iterator i = e.begin(); i != e.end(); ++i ) {
        remove_edge(**i, r);
    }
}
#endif  /* TBB_PREVIEW_FLOW_GRAPH_FEATURES */

//! Returns a copy of the body from a function or continue node
template< typename Body, typename Node >
Body copy_body( Node &n ) {
    return n.template copy_function_object<Body>();
}

#if __TBB_FLOW_GRAPH_CPP11_FEATURES

//composite_node
template< typename InputTuple, typename OutputTuple > class composite_node;

template< typename... InputTypes, typename... OutputTypes>
class composite_node <tbb::flow::tuple<InputTypes...>, tbb::flow::tuple<OutputTypes...> > : public graph_node{

public:
    typedef tbb::flow::tuple< receiver<InputTypes>&... > input_ports_type;
    typedef tbb::flow::tuple< sender<OutputTypes>&... > output_ports_type;

private:
#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    const char *my_type_name;
#endif
    std::unique_ptr<input_ports_type> my_input_ports;
    std::unique_ptr<output_ports_type> my_output_ports;

    static const size_t NUM_INPUTS = sizeof...(InputTypes);
    static const size_t NUM_OUTPUTS = sizeof...(OutputTypes);

protected:
    void reset_node(reset_flags) __TBB_override {}

public:
#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    composite_node( graph &g, const char *type_name ) : graph_node(g), my_type_name(type_name) {
        tbb::internal::fgt_multiinput_multioutput_node( tbb::internal::FLOW_COMPOSITE_NODE, this, &this->my_graph );
        tbb::internal::fgt_multiinput_multioutput_node_desc( this, my_type_name );
    }
#endif
    composite_node( graph &g ) : graph_node(g) {
        tbb::internal::fgt_multiinput_multioutput_node( tbb::internal::FLOW_COMPOSITE_NODE, this, &this->my_graph );
    }

   template<typename T1, typename T2>
   void set_external_ports(T1&& input_ports_tuple, T2&& output_ports_tuple) {
       __TBB_STATIC_ASSERT(NUM_INPUTS == tbb::flow::tuple_size<input_ports_type>::value, "number of arguments does not match number of input ports");
       __TBB_STATIC_ASSERT(NUM_OUTPUTS == tbb::flow::tuple_size<output_ports_type>::value, "number of arguments does not match number of output ports");
       my_input_ports = tbb::internal::make_unique<input_ports_type>(std::forward<T1>(input_ports_tuple));
       my_output_ports = tbb::internal::make_unique<output_ports_type>(std::forward<T2>(output_ports_tuple));

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
      tbb::internal::fgt_internal_input_alias_helper<T1, NUM_INPUTS>::alias_port( this, input_ports_tuple);
      tbb::internal::fgt_internal_output_alias_helper<T2, NUM_OUTPUTS>::alias_port( this, output_ports_tuple);
#endif
   }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    template< typename... NodeTypes >
    void add_visible_nodes(const NodeTypes&... n) { internal::add_nodes_impl(this, true, n...); }

    template< typename... NodeTypes >
    void add_nodes(const NodeTypes&... n) { internal::add_nodes_impl(this, false, n...); }
#else
    template<typename... Nodes> void add_nodes(Nodes&...) { }
    template<typename... Nodes> void add_visible_nodes(Nodes&...) { }
#endif

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    void set_name( const char *name ) __TBB_override {
        tbb::internal::fgt_multiinput_multioutput_node_desc( this, name );
    }
#endif

    input_ports_type& input_ports() {
         __TBB_ASSERT(my_input_ports, "input ports not set, call set_external_ports to set input ports");
         return *my_input_ports;
    }

    output_ports_type& output_ports() {
         __TBB_ASSERT(my_output_ports, "output ports not set, call set_external_ports to set output ports");
         return *my_output_ports;
    }

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    void extract() __TBB_override {
        __TBB_ASSERT(false, "Current composite_node implementation does not support extract");
    }
#endif
};  // class composite_node

//composite_node with only input ports
template< typename... InputTypes>
class composite_node <tbb::flow::tuple<InputTypes...>, tbb::flow::tuple<> > : public graph_node {
public:
    typedef tbb::flow::tuple< receiver<InputTypes>&... > input_ports_type;

private:
#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    const char *my_type_name;
#endif
    std::unique_ptr<input_ports_type> my_input_ports;
    static const size_t NUM_INPUTS = sizeof...(InputTypes);

protected:
    void reset_node(reset_flags) __TBB_override {}

public:
#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    composite_node( graph &g, const char *type_name = "composite_node") : graph_node(g), my_type_name(type_name) {
        tbb::internal::itt_make_task_group( tbb::internal::ITT_DOMAIN_FLOW, this, tbb::internal::FLOW_NODE, &g, tbb::internal::FLOW_GRAPH, tbb::internal::FLOW_COMPOSITE_NODE );
        tbb::internal::fgt_multiinput_multioutput_node_desc( this, my_type_name );
    }
#else
    composite_node( graph &g) : graph_node(g) {}
#endif

   template<typename T>
   void set_external_ports(T&& input_ports_tuple) {
       __TBB_STATIC_ASSERT(NUM_INPUTS == tbb::flow::tuple_size<input_ports_type>::value, "number of arguments does not match number of input ports");

       my_input_ports = tbb::internal::make_unique<input_ports_type>(std::forward<T>(input_ports_tuple));

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
      tbb::internal::fgt_internal_input_alias_helper<T, NUM_INPUTS>::alias_port( this, std::forward<T>(input_ports_tuple));
#endif
   }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    template< typename... NodeTypes >
    void add_visible_nodes(const NodeTypes&... n) { internal::add_nodes_impl(this, true, n...); }

    template< typename... NodeTypes >
    void add_nodes( const NodeTypes&... n) { internal::add_nodes_impl(this, false, n...); }
#else
    template<typename... Nodes> void add_nodes(Nodes&...) {}
    template<typename... Nodes> void add_visible_nodes(Nodes&...) {}
#endif

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    void set_name( const char *name ) __TBB_override {
        tbb::internal::fgt_multiinput_multioutput_node_desc( this, name );
    }
#endif

    input_ports_type& input_ports() {
         __TBB_ASSERT(my_input_ports, "input ports not set, call set_external_ports to set input ports");
         return *my_input_ports;
    }

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    void extract() __TBB_override {
        __TBB_ASSERT(false, "Current composite_node implementation does not support extract");
    }
#endif

};  // class composite_node

//composite_nodes with only output_ports
template<typename... OutputTypes>
class composite_node <tbb::flow::tuple<>, tbb::flow::tuple<OutputTypes...> > : public graph_node {
public:
    typedef tbb::flow::tuple< sender<OutputTypes>&... > output_ports_type;

private:
#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    const char *my_type_name;
#endif
    std::unique_ptr<output_ports_type> my_output_ports;
    static const size_t NUM_OUTPUTS = sizeof...(OutputTypes);

protected:
    void reset_node(reset_flags) __TBB_override {}

public:
#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    composite_node( graph &g, const char *type_name = "composite_node") : graph_node(g), my_type_name(type_name) {
        tbb::internal::itt_make_task_group( tbb::internal::ITT_DOMAIN_FLOW, this, tbb::internal::FLOW_NODE, &g, tbb::internal::FLOW_GRAPH, tbb::internal::FLOW_COMPOSITE_NODE );
        tbb::internal::fgt_multiinput_multioutput_node_desc( this, my_type_name );
    }
#else
    composite_node( graph &g) : graph_node(g) {}
#endif

   template<typename T>
   void set_external_ports(T&& output_ports_tuple) {
       __TBB_STATIC_ASSERT(NUM_OUTPUTS == tbb::flow::tuple_size<output_ports_type>::value, "number of arguments does not match number of output ports");

       my_output_ports = tbb::internal::make_unique<output_ports_type>(std::forward<T>(output_ports_tuple));

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
      tbb::internal::fgt_internal_output_alias_helper<T, NUM_OUTPUTS>::alias_port( this, std::forward<T>(output_ports_tuple));
#endif
   }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    template<typename... NodeTypes >
    void add_visible_nodes(const NodeTypes&... n) { internal::add_nodes_impl(this, true, n...); }

    template<typename... NodeTypes >
    void add_nodes(const NodeTypes&... n) { internal::add_nodes_impl(this, false, n...); }
#else
    template<typename... Nodes> void add_nodes(Nodes&...) {}
    template<typename... Nodes> void add_visible_nodes(Nodes&...) {}
#endif

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    void set_name( const char *name ) __TBB_override {
        tbb::internal::fgt_multiinput_multioutput_node_desc( this, name );
    }
#endif

    output_ports_type& output_ports() {
         __TBB_ASSERT(my_output_ports, "output ports not set, call set_external_ports to set output ports");
         return *my_output_ports;
    }

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    void extract() __TBB_override {
        __TBB_ASSERT(false, "Current composite_node implementation does not support extract");
    }
#endif

};  // class composite_node

#endif // __TBB_FLOW_GRAPH_CPP11_FEATURES

namespace internal {

template<typename Gateway>
class async_body_base: tbb::internal::no_assign {
public:
    typedef Gateway gateway_type;

    async_body_base(gateway_type *gateway): my_gateway(gateway) { }
    void set_gateway(gateway_type *gateway) {
        my_gateway = gateway;
    }

protected:
    gateway_type *my_gateway;
};

template<typename Input, typename Ports, typename Gateway, typename Body>
class async_body: public async_body_base<Gateway> {
public:
    typedef async_body_base<Gateway> base_type;
    typedef Gateway gateway_type;

    async_body(const Body &body, gateway_type *gateway) 
        : base_type(gateway), my_body(body) { }

    void operator()( const Input &v, Ports & ) {
        my_body(v, *this->my_gateway);
    }

    Body get_body() { return my_body; }

private:
    Body my_body;
};

}

//! Implements async node
template < typename Input, typename Output, typename Policy = queueing, typename Allocator=cache_aligned_allocator<Input> >
class async_node : public multifunction_node< Input, tuple< Output >, Policy, Allocator >, public sender< Output > {
    typedef multifunction_node< Input, tuple< Output >, Policy, Allocator > base_type;
    typedef typename internal::multifunction_input<Input, typename base_type::output_ports_type, Allocator> mfn_input_type;

public:
    typedef Input input_type;
    typedef Output output_type;
    typedef receiver<input_type> receiver_type;
    typedef typename receiver_type::predecessor_type predecessor_type;
    typedef typename sender<output_type>::successor_type successor_type;
    typedef receiver_gateway<output_type> gateway_type;
    typedef internal::async_body_base<gateway_type> async_body_base_type;
    typedef typename base_type::output_ports_type output_ports_type;

private:
    struct try_put_functor {
        typedef internal::multifunction_output<Output> output_port_type;
        output_port_type *port;
        const Output *value;
        bool result;
        try_put_functor(output_port_type &p, const Output &v) : port(&p), value(&v), result(false) { }
        void operator()() {
            result = port->try_put(*value);
        }
    };

    class receiver_gateway_impl: public receiver_gateway<Output> {
    public:
        receiver_gateway_impl(async_node* node): my_node(node) {}
        void reserve_wait() __TBB_override {
            tbb::internal::fgt_async_reserve(static_cast<typename async_node::receiver_type *>(my_node), &my_node->my_graph);
            my_node->my_graph.reserve_wait();
        }

        void release_wait() __TBB_override {
            my_node->my_graph.release_wait();
            tbb::internal::fgt_async_commit(static_cast<typename async_node::receiver_type *>(my_node), &my_node->my_graph);
        }

        //! Implements gateway_type::try_put for an external activity to submit a message to FG
        bool try_put(const Output &i) __TBB_override {
            return my_node->try_put_impl(i);
        }

    private:
        async_node* my_node;
    } my_gateway;

    //The substitute of 'this' for member construction, to prevent compiler warnings
    async_node* self() { return this; }

    //! Implements gateway_type::try_put for an external activity to submit a message to FG
    bool try_put_impl(const Output &i) {
        internal::multifunction_output<Output> &port_0 = internal::output_port<0>(*this);
        tbb::internal::fgt_async_try_put_begin(this, &port_0);
        try_put_functor tpf(port_0, i);
        internal::execute_in_graph_arena(this->my_graph, tpf);
        tbb::internal::fgt_async_try_put_end(this, &port_0);
        return tpf.result;
    }

public:
    template<typename Body>
    async_node( graph &g, size_t concurrency, Body body ) :
        base_type( g, concurrency, internal::async_body<Input, typename base_type::output_ports_type, gateway_type, Body>(body, &my_gateway) ), my_gateway(self()) {
        tbb::internal::fgt_multioutput_node_with_body<1>( tbb::internal::FLOW_ASYNC_NODE,
                &this->my_graph, static_cast<receiver<input_type> *>(this),
                this->output_ports(), this->my_body );
    }

    async_node( const async_node &other ) : base_type(other), sender<Output>(), my_gateway(self()) {
        static_cast<async_body_base_type*>(this->my_body->get_body_ptr())->set_gateway(&my_gateway);
        static_cast<async_body_base_type*>(this->my_init_body->get_body_ptr())->set_gateway(&my_gateway);

        tbb::internal::fgt_multioutput_node_with_body<1>( tbb::internal::FLOW_ASYNC_NODE,
                &this->my_graph, static_cast<receiver<input_type> *>(this),
                this->output_ports(), this->my_body );
    }

    gateway_type& gateway() {
        return my_gateway;
    }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    void set_name( const char *name ) __TBB_override {
            tbb::internal::fgt_multioutput_node_desc( this, name );
    }
#endif

    // Define sender< Output >

    //! Add a new successor to this node
    bool register_successor( successor_type &r ) __TBB_override {
        return internal::output_port<0>(*this).register_successor(r);
    }

    //! Removes a successor from this node
    bool remove_successor( successor_type &r ) __TBB_override {
        return internal::output_port<0>(*this).remove_successor(r);
    }

    template<typename Body>
    Body copy_function_object() {
        typedef internal::multifunction_body<input_type, typename base_type::output_ports_type> mfn_body_type;
        typedef internal::async_body<Input, typename base_type::output_ports_type, gateway_type, Body> async_body_type;
        mfn_body_type &body_ref = *this->my_body;
        async_body_type ab = *static_cast<async_body_type*>(dynamic_cast< internal::multifunction_body_leaf<input_type, typename base_type::output_ports_type, async_body_type> & >(body_ref).get_body_ptr());
        return ab.get_body();
    }

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    //! interface to record edges for traversal & deletion
    typedef typename  internal::edge_container<successor_type> built_successors_type;
    typedef typename  built_successors_type::edge_list_type successor_list_type;
    built_successors_type &built_successors() __TBB_override {
        return internal::output_port<0>(*this).built_successors();
    }

    void internal_add_built_successor( successor_type &r ) __TBB_override {
        internal::output_port<0>(*this).internal_add_built_successor(r);
    }

    void internal_delete_built_successor( successor_type &r ) __TBB_override {
        internal::output_port<0>(*this).internal_delete_built_successor(r);
    }

    void copy_successors( successor_list_type &l ) __TBB_override {
        internal::output_port<0>(*this).copy_successors(l);
    }

    size_t  successor_count() __TBB_override {
        return internal::output_port<0>(*this).successor_count();
    }
#endif

protected:

    void reset_node( reset_flags f) __TBB_override {
       base_type::reset_node(f);
    }
};

#if __TBB_PREVIEW_STREAMING_NODE
#include "internal/_flow_graph_streaming_node.h"
#endif // __TBB_PREVIEW_STREAMING_NODE

} // interfaceX

    using interface10::reset_flags;
    using interface10::rf_reset_protocol;
    using interface10::rf_reset_bodies;
    using interface10::rf_clear_edges;

    using interface10::graph;
    using interface10::graph_node;
    using interface10::continue_msg;

    using interface10::source_node;
    using interface10::function_node;
    using interface10::multifunction_node;
    using interface10::split_node;
    using interface10::internal::output_port;
    using interface10::indexer_node;
    using interface10::internal::tagged_msg;
    using interface10::internal::cast_to;
    using interface10::internal::is_a;
    using interface10::continue_node;
    using interface10::overwrite_node;
    using interface10::write_once_node;
    using interface10::broadcast_node;
    using interface10::buffer_node;
    using interface10::queue_node;
    using interface10::sequencer_node;
    using interface10::priority_queue_node;
    using interface10::limiter_node;
    using namespace interface10::internal::graph_policy_namespace;
    using interface10::join_node;
    using interface10::input_port;
    using interface10::copy_body;
    using interface10::make_edge;
    using interface10::remove_edge;
    using interface10::internal::tag_value;
#if __TBB_FLOW_GRAPH_CPP11_FEATURES
    using interface10::composite_node;
#endif
    using interface10::async_node;
#if __TBB_PREVIEW_ASYNC_MSG
    using interface10::async_msg;
#endif
#if __TBB_PREVIEW_STREAMING_NODE
    using interface10::port_ref;
    using interface10::streaming_node;
#endif // __TBB_PREVIEW_STREAMING_NODE

} // flow
} // tbb

#undef __TBB_PFG_RESET_ARG
#undef __TBB_COMMA

#endif // __TBB_flow_graph_H
