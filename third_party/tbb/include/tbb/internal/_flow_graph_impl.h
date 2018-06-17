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

#ifndef __TBB_flow_graph_impl_H
#define __TBB_flow_graph_impl_H

#include "../tbb_stddef.h"
#include "../task.h"
#include "../task_arena.h"
#include "../flow_graph_abstractions.h"

#include <list>

#if TBB_DEPRECATED_FLOW_ENQUEUE
#define FLOW_SPAWN(a) tbb::task::enqueue((a))
#else
#define FLOW_SPAWN(a) tbb::task::spawn((a))
#endif

namespace tbb {
namespace flow {

namespace internal {
static tbb::task * const SUCCESSFULLY_ENQUEUED = (task *)-1;
}

namespace interface10 {

using tbb::flow::internal::SUCCESSFULLY_ENQUEUED;

class graph;
class graph_node;

template <typename GraphContainerType, typename GraphNodeType>
class graph_iterator {
    friend class graph;
    friend class graph_node;
public:
    typedef size_t size_type;
    typedef GraphNodeType value_type;
    typedef GraphNodeType* pointer;
    typedef GraphNodeType& reference;
    typedef const GraphNodeType& const_reference;
    typedef std::forward_iterator_tag iterator_category;

    //! Default constructor
    graph_iterator() : my_graph(NULL), current_node(NULL) {}

    //! Copy constructor
    graph_iterator(const graph_iterator& other) :
        my_graph(other.my_graph), current_node(other.current_node)
    {}

    //! Assignment
    graph_iterator& operator=(const graph_iterator& other) {
        if (this != &other) {
            my_graph = other.my_graph;
            current_node = other.current_node;
        }
        return *this;
    }

    //! Dereference
    reference operator*() const;

    //! Dereference
    pointer operator->() const;

    //! Equality
    bool operator==(const graph_iterator& other) const {
        return ((my_graph == other.my_graph) && (current_node == other.current_node));
    }

    //! Inequality
    bool operator!=(const graph_iterator& other) const { return !(operator==(other)); }

    //! Pre-increment
    graph_iterator& operator++() {
        internal_forward();
        return *this;
    }

    //! Post-increment
    graph_iterator operator++(int) {
        graph_iterator result = *this;
        operator++();
        return result;
    }

private:
    // the graph over which we are iterating
    GraphContainerType *my_graph;
    // pointer into my_graph's my_nodes list
    pointer current_node;

    //! Private initializing constructor for begin() and end() iterators
    graph_iterator(GraphContainerType *g, bool begin);
    void internal_forward();
};  // class graph_iterator

// flags to modify the behavior of the graph reset().  Can be combined.
enum reset_flags {
    rf_reset_protocol = 0,
    rf_reset_bodies = 1 << 0,  // delete the current node body, reset to a copy of the initial node body.
    rf_clear_edges = 1 << 1   // delete edges
};

namespace internal {

void activate_graph(graph& g);
void deactivate_graph(graph& g);
bool is_graph_active(graph& g);
void spawn_in_graph_arena(graph& g, tbb::task& arena_task);
void add_task_to_graph_reset_list(graph& g, tbb::task *tp);
template<typename F> void execute_in_graph_arena(graph& g, F& f);

}

//! The graph class
/** This class serves as a handle to the graph */
class graph : tbb::internal::no_copy, public tbb::flow::graph_proxy {
    friend class graph_node;

    template< typename Body >
    class run_task : public task {
    public:
        run_task(Body& body) : my_body(body) {}
        tbb::task *execute() __TBB_override {
            my_body();
            return NULL;
        }
    private:
        Body my_body;
    };

    template< typename Receiver, typename Body >
    class run_and_put_task : public task {
    public:
        run_and_put_task(Receiver &r, Body& body) : my_receiver(r), my_body(body) {}
        tbb::task *execute() __TBB_override {
            tbb::task *res = my_receiver.try_put_task(my_body());
            if (res == SUCCESSFULLY_ENQUEUED) res = NULL;
            return res;
        }
    private:
        Receiver &my_receiver;
        Body my_body;
    };
    typedef std::list<tbb::task *> task_list_type;

    class wait_functor {
        tbb::task* graph_root_task;
    public:
        wait_functor(tbb::task* t) : graph_root_task(t) {}
        void operator()() const { graph_root_task->wait_for_all(); }
    };

    //! A functor that spawns a task
    class spawn_functor : tbb::internal::no_assign {
        tbb::task& spawn_task;
    public:
        spawn_functor(tbb::task& t) : spawn_task(t) {}
        void operator()() const {
            FLOW_SPAWN(spawn_task);
        }
    };

    void prepare_task_arena(bool reinit = false) {
        if (reinit) {
            __TBB_ASSERT(my_task_arena, "task arena is NULL");
            my_task_arena->terminate();
            my_task_arena->initialize(tbb::task_arena::attach());
        }
        else {
            __TBB_ASSERT(my_task_arena == NULL, "task arena is not NULL");
            my_task_arena = new tbb::task_arena(tbb::task_arena::attach());
        }
        if (!my_task_arena->is_active()) // failed to attach
            my_task_arena->initialize(); // create a new, default-initialized arena
        __TBB_ASSERT(my_task_arena->is_active(), "task arena is not active");
    }

public:
    //! Constructs a graph with isolated task_group_context
    graph();

    //! Constructs a graph with use_this_context as context
    explicit graph(tbb::task_group_context& use_this_context);

    //! Destroys the graph.
    /** Calls wait_for_all, then destroys the root task and context. */
    ~graph();

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    void set_name(const char *name);
#endif

    void increment_wait_count() {
        reserve_wait();
    }

    void decrement_wait_count() {
        release_wait();
    }

    //! Used to register that an external entity may still interact with the graph.
    /** The graph will not return from wait_for_all until a matching number of decrement_wait_count calls
    is made. */
    void reserve_wait() __TBB_override;

    //! Deregisters an external entity that may have interacted with the graph.
    /** The graph will not return from wait_for_all until all the number of decrement_wait_count calls
    matches the number of increment_wait_count calls. */
    void release_wait() __TBB_override;

    //! Spawns a task that runs a body and puts its output to a specific receiver
    /** The task is spawned as a child of the graph. This is useful for running tasks
    that need to block a wait_for_all() on the graph.  For example a one-off source. */
    template< typename Receiver, typename Body >
    void run(Receiver &r, Body body) {
        if (internal::is_graph_active(*this)) {
            task* rtask = new (task::allocate_additional_child_of(*root_task()))
                run_and_put_task< Receiver, Body >(r, body);
            my_task_arena->execute(spawn_functor(*rtask));
        }
    }

    //! Spawns a task that runs a function object
    /** The task is spawned as a child of the graph. This is useful for running tasks
    that need to block a wait_for_all() on the graph. For example a one-off source. */
    template< typename Body >
    void run(Body body) {
        if (internal::is_graph_active(*this)) {
            task* rtask = new (task::allocate_additional_child_of(*root_task())) run_task< Body >(body);
            my_task_arena->execute(spawn_functor(*rtask));
        }
    }

    //! Wait until graph is idle and decrement_wait_count calls equals increment_wait_count calls.
    /** The waiting thread will go off and steal work while it is block in the wait_for_all. */
    void wait_for_all() {
        cancelled = false;
        caught_exception = false;
        if (my_root_task) {
#if TBB_USE_EXCEPTIONS
            try {
#endif
                my_task_arena->execute(wait_functor(my_root_task));
                cancelled = my_context->is_group_execution_cancelled();
#if TBB_USE_EXCEPTIONS
            }
            catch (...) {
                my_root_task->set_ref_count(1);
                my_context->reset();
                caught_exception = true;
                cancelled = true;
                throw;
            }
#endif
            // TODO: the "if" condition below is just a work-around to support the concurrent wait
            // mode. The cancellation and exception mechanisms are still broken in this mode.
            // Consider using task group not to re-implement the same functionality.
            if (!(my_context->traits() & tbb::task_group_context::concurrent_wait)) {
                my_context->reset();  // consistent with behavior in catch()
                my_root_task->set_ref_count(1);
            }
        }
    }

    //! Returns the root task of the graph
    tbb::task * root_task() {
        return my_root_task;
    }

    // ITERATORS
    template<typename C, typename N>
    friend class graph_iterator;

    // Graph iterator typedefs
    typedef graph_iterator<graph, graph_node> iterator;
    typedef graph_iterator<const graph, const graph_node> const_iterator;

    // Graph iterator constructors
    //! start iterator
    iterator begin();
    //! end iterator
    iterator end();
    //! start const iterator
    const_iterator begin() const;
    //! end const iterator
    const_iterator end() const;
    //! start const iterator
    const_iterator cbegin() const;
    //! end const iterator
    const_iterator cend() const;

    //! return status of graph execution
    bool is_cancelled() { return cancelled; }
    bool exception_thrown() { return caught_exception; }

    // thread-unsafe state reset.
    void reset(reset_flags f = rf_reset_protocol);

private:
    tbb::task *my_root_task;
    tbb::task_group_context *my_context;
    bool own_context;
    bool cancelled;
    bool caught_exception;
    bool my_is_active;
    task_list_type my_reset_task_list;

    graph_node *my_nodes, *my_nodes_last;

    tbb::spin_mutex nodelist_mutex;
    void register_node(graph_node *n);
    void remove_node(graph_node *n);

    tbb::task_arena* my_task_arena;

    friend void internal::activate_graph(graph& g);
    friend void internal::deactivate_graph(graph& g);
    friend bool internal::is_graph_active(graph& g);
    friend void internal::spawn_in_graph_arena(graph& g, tbb::task& arena_task);
    friend void internal::add_task_to_graph_reset_list(graph& g, tbb::task *tp);
    template<typename F> friend void internal::execute_in_graph_arena(graph& g, F& f);

    friend class tbb::interface7::internal::task_arena_base;

};  // class graph

//! The base of all graph nodes.
class graph_node : tbb::internal::no_copy {
    friend class graph;
    template<typename C, typename N>
    friend class graph_iterator;
protected:
    graph& my_graph;
    graph_node *next, *prev;
public:
    explicit graph_node(graph& g);
    
    virtual ~graph_node();

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    virtual void set_name(const char *name) = 0;
#endif

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    virtual void extract() = 0;
#endif

protected:
    // performs the reset on an individual node.
    virtual void reset_node(reset_flags f = rf_reset_protocol) = 0;
};  // class graph_node

namespace internal {

inline void activate_graph(graph& g) {
    g.my_is_active = true;
}

inline void deactivate_graph(graph& g) {
    g.my_is_active = false;
}

inline bool is_graph_active(graph& g) {
    return g.my_is_active;
}

//! Executes custom functor inside graph arena
template<typename F>
inline void execute_in_graph_arena(graph& g, F& f) {
    if (is_graph_active(g)) {
        __TBB_ASSERT(g.my_task_arena && g.my_task_arena->is_active(), NULL);
        g.my_task_arena->execute(f);
    }
}

//! Spawns a task inside graph arena
inline void spawn_in_graph_arena(graph& g, tbb::task& arena_task) {
    graph::spawn_functor s_fn(arena_task);
    execute_in_graph_arena(g, s_fn);
}

inline void add_task_to_graph_reset_list(graph& g, tbb::task *tp) {
    g.my_reset_task_list.push_back(tp);
}

} // namespace internal

} // namespace interface10
} // namespace flow
} // namespace tbb

#endif // __TBB_flow_graph_impl_H
