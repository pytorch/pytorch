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

// Before including pipeline.h, set up the variable to count heap allocated
// filter_node objects, and make it known for the header.
int filter_node_count = 0;
#define __TBB_TEST_FILTER_NODE_COUNT filter_node_count
#include "tbb/pipeline.h"

#include "tbb/atomic.h"
#include "harness.h"
#include <string.h>

#include "tbb/tbb_allocator.h"
#include "tbb/spin_mutex.h"

const unsigned n_tokens = 8;
// we can conceivably have two buffers used in the middle filter for every token in flight, so
// we must allocate two buffers for every token.  Unlikely, but possible.
const unsigned n_buffers = 2*n_tokens;
const int max_counter = 16;
static tbb::atomic<int> output_counter;
static tbb::atomic<int> input_counter;
static tbb::atomic<int> non_pointer_specialized_calls;
static tbb::atomic<int> pointer_specialized_calls;
static tbb::atomic<int> first_pointer_specialized_calls;
static tbb::atomic<int> second_pointer_specialized_calls;
static tbb::spin_mutex buffer_mutex;

static int intbuffer[max_counter];  // store results for <int,int> parallel pipeline test
static bool check_intbuffer;

static void* buffers[n_buffers];
static bool buf_available[n_buffers];

void *fetchNextBuffer() {
    tbb::spin_mutex::scoped_lock sl1(buffer_mutex);
    for(size_t icnt = 0; icnt < n_buffers; ++icnt) {
        if(buf_available[icnt]) {
            buf_available[icnt] = false;
            return buffers[icnt];
        }
    }
    ASSERT(0, "Ran out of buffers");
    return 0;
}
void freeBuffer(void *buf) {
    for(size_t i=0; i < n_buffers;++i) {
        if(buffers[i] == buf) {
            buf_available[i] = true;
            return;
        }
    }
    ASSERT(0, "Tried to free a buffer not in our list");
}

template<typename T>
class free_on_scope_exit {
public:
    free_on_scope_exit(T *p) : my_p(p) {}
    ~free_on_scope_exit() { if(!my_p) return; my_p->~T(); freeBuffer(my_p); }
private:
    T *my_p;
};

#include "harness_checktype.h"

// methods for testing check_type< >, that return okay values for other types.
template<typename T>
bool middle_is_ready(T &/*p*/) { return false; }

template<typename U>
bool middle_is_ready(check_type<U> &p) { return p.is_ready(); }

template<typename T>
bool output_is_ready(T &/*p*/) { return true; }

template<typename U>
bool output_is_ready(check_type<U> &p) { return p.is_ready(); }

template<typename T>
int middle_my_id( T &/*p*/) { return 0; }

template<typename U>
int middle_my_id(check_type<U> &p) { return p.my_id(); }

template<typename T>
int output_my_id( T &/*p*/) { return 1; }

template<typename U>
int output_my_id(check_type<U> &p) { return p.my_id(); }

template<typename T>
void my_function(T &p) { p = 0; }

template<typename U>
void my_function(check_type<U> &p) { p.function(); }

// Filters must be copy-constructible, and be const-qualifiable.
template<typename U>
class input_filter : Harness::NoAfterlife {
public:
    U operator()( tbb::flow_control& control ) const {
        AssertLive();
        if( --input_counter < 0 ) {
            control.stop();
        }
        else  // only count successful reads
            ++non_pointer_specialized_calls;
        return U();  // default constructed
    }

};

// specialization for pointer
template<typename U>
class input_filter<U*> : Harness::NoAfterlife {
public:
    U* operator()(tbb::flow_control& control) const {
        AssertLive();
        int ival = --input_counter;
        if(ival < 0) {
            control.stop();
            return NULL;
        }
        ++pointer_specialized_calls;
        if(ival == max_counter / 2) {
            return NULL;  // non-stop NULL
        }
        U* myReturn = new(fetchNextBuffer()) U();
        return myReturn;
    }
};

template<>
class input_filter<void> : Harness::NoAfterlife {
public:
    void operator()( tbb::flow_control& control ) const {
        AssertLive();
        if( --input_counter < 0 ) {
            control.stop();
        }
        else
            ++non_pointer_specialized_calls;
    }

};

// specialization for int that passes back a sequence of integers
template<>
class input_filter<int> : Harness::NoAfterlife {
public:
    int
    operator()(tbb::flow_control& control ) const {
        AssertLive();
        int oldval = --input_counter;
        if( oldval < 0 ) {
            control.stop();
        }
        else
            ++non_pointer_specialized_calls;
        return oldval+1;
    }
};

template<typename T, typename U>
class middle_filter : Harness::NoAfterlife {
public:
    U operator()(T t) const {
        AssertLive();
        ASSERT(!middle_my_id(t), "bad id value");
        ASSERT(!middle_is_ready(t), "Already ready" );
        U out;
        my_function(out);
        ++non_pointer_specialized_calls;
        return out;
    }
};

template<typename T, typename U>
class middle_filter<T*,U> : Harness::NoAfterlife {
public:
    U operator()(T* my_storage) const {
        free_on_scope_exit<T> my_ptr(my_storage);  // free_on_scope_exit marks the buffer available
        AssertLive();
        if(my_storage) {  // may have been passed in a NULL
            ASSERT(!middle_my_id(*my_storage), "bad id value");
            ASSERT(!middle_is_ready(*my_storage), "Already ready" );
        }
        ++first_pointer_specialized_calls;
        U out;
        my_function(out);
        return out;
    }
};

template<typename T, typename U>
class middle_filter<T,U*> : Harness::NoAfterlife {
public:
    U* operator()(T my_storage) const {
        AssertLive();
        ASSERT(!middle_my_id(my_storage), "bad id value");
        ASSERT(!middle_is_ready(my_storage), "Already ready" );
        // allocate new space from buffers
        U* my_return = new(fetchNextBuffer()) U();
        my_function(*my_return);
        ++second_pointer_specialized_calls;
        return my_return;
    }
};

template<typename T, typename U>
class middle_filter<T*,U*> : Harness::NoAfterlife {
public:
    U* operator()(T* my_storage) const {
        free_on_scope_exit<T> my_ptr(my_storage);  // free_on_scope_exit marks the buffer available
        AssertLive();
        if(my_storage) {
            ASSERT(!middle_my_id(*my_storage), "bad id value");
            ASSERT(!middle_is_ready(*my_storage), "Already ready" );
        }
        // may have been passed a NULL
        ++pointer_specialized_calls;
        if(!my_storage) return NULL;
        ASSERT(!middle_my_id(*my_storage), "bad id value");
        ASSERT(!middle_is_ready(*my_storage), "Already ready" );
        U* my_return = new(fetchNextBuffer()) U();
        my_function(*my_return);
        return my_return;
    }
};

// specialization for int that squares the input and returns that.
template<>
class middle_filter<int,int> : Harness::NoAfterlife {
public:
    int operator()(int my_input) const {
        AssertLive();
        ++non_pointer_specialized_calls;
        return my_input*my_input;
    }
};

// ---------------------------------
template<typename T>
class output_filter : Harness::NoAfterlife {
public:
    void operator()(T c) const {
        AssertLive();
        ASSERT(output_my_id(c), "unset id value");
        ASSERT(output_is_ready(c), "not yet ready");
        ++non_pointer_specialized_calls;
        output_counter++;
    }
};

// specialization for int that puts the received value in an array
template<>
class output_filter<int> : Harness::NoAfterlife {
public:
    void operator()(int my_input) const {
        AssertLive();
        ++non_pointer_specialized_calls;
        int myindx = output_counter++;
        intbuffer[myindx] = my_input;
    }
};


template<typename T>
class output_filter<T*> : Harness::NoAfterlife {
public:
    void operator()(T* c) const {
        free_on_scope_exit<T> my_ptr(c);
        AssertLive();
        if(c) {
            ASSERT(output_my_id(*c), "unset id value");
            ASSERT(output_is_ready(*c), "not yet ready");
        }
        output_counter++;
        ++pointer_specialized_calls;
    }
};

typedef enum {
    no_pointer_counts,
    assert_nonpointer,
    assert_firstpointer,
    assert_secondpointer,
    assert_allpointer
} final_assert_type;

void resetCounters() {
    output_counter = 0;
    input_counter = max_counter;
    non_pointer_specialized_calls = 0;
    pointer_specialized_calls = 0;
    first_pointer_specialized_calls = 0;
    second_pointer_specialized_calls = 0;
    // we have to reset the buffer flags because our input filters return allocated space on end-of-input,
    // (on eof a default-constructed object is returned) and they do not pass through the filter further.
    for(size_t i = 0; i < n_buffers; ++i)
        buf_available[i] = true;
}

void checkCounters(final_assert_type my_t) {
    ASSERT(output_counter == max_counter, "not all tokens were passed through pipeline");
    switch(my_t) {
        case assert_nonpointer:
            ASSERT(pointer_specialized_calls+first_pointer_specialized_calls+second_pointer_specialized_calls == 0, "non-pointer filters specialized to pointer");
            ASSERT(non_pointer_specialized_calls == 3*max_counter, "bad count for non-pointer filters");
            if(check_intbuffer) {
                for(int i = 1; i <= max_counter; ++i) {
                    int j = i*i;
                    bool found_val = false;
                    for(int k = 0; k < max_counter; ++k) {
                        if(intbuffer[k] == j) {
                            found_val = true;
                            break;
                        }
                    }
                    ASSERT(found_val, "Missing value in output array" );
                }
            }
            break;
        case assert_firstpointer:
            ASSERT(pointer_specialized_calls == max_counter &&  // input filter extra invocation
                    first_pointer_specialized_calls == max_counter &&
                    non_pointer_specialized_calls == max_counter &&
                    second_pointer_specialized_calls == 0, "incorrect specialization for firstpointer");
            break;
        case assert_secondpointer:
            ASSERT(pointer_specialized_calls == max_counter &&
                    first_pointer_specialized_calls == 0 &&
                    non_pointer_specialized_calls == max_counter &&  // input filter
                    second_pointer_specialized_calls == max_counter, "incorrect specialization for firstpointer");
            break;
        case assert_allpointer:
            ASSERT(non_pointer_specialized_calls+first_pointer_specialized_calls+second_pointer_specialized_calls == 0, "pointer filters specialized to non-pointer");
            ASSERT(pointer_specialized_calls == 3*max_counter, "bad count for pointer filters");
            break;
        case no_pointer_counts:
            break;
    }
}

static const tbb::filter::mode filter_table[] = { tbb::filter::parallel, tbb::filter::serial_in_order, tbb::filter::serial_out_of_order};
const unsigned number_of_filter_types = sizeof(filter_table)/sizeof(filter_table[0]);

typedef tbb::filter_t<void, void> filter_chain;
typedef tbb::filter::mode mode_array;

// The filters are passed by value, which forces a temporary copy to be created.  This is
// to reproduce the bug where a filter_chain uses refs to filters, which after a call
// would be references to destructed temporaries.
template<typename type1, typename type2>
void fill_chain( filter_chain &my_chain, mode_array *filter_type, input_filter<type1> i_filter,
         middle_filter<type1, type2> m_filter, output_filter<type2> o_filter ) {
    my_chain = tbb::make_filter<void, type1>(filter_type[0], i_filter) &
        tbb::make_filter<type1, type2>(filter_type[1], m_filter) &
        tbb::make_filter<type2, void>(filter_type[2], o_filter);
}

void run_function_spec() {
    ASSERT(!filter_node_count, NULL);
    REMARK("Testing < void, void > (single filter in pipeline)");
#if __TBB_CPP11_LAMBDAS_PRESENT
    REMARK( " ( + lambdas)");
#endif
    REMARK("\n");
    input_filter<void> i_filter;
    // Test pipeline that contains only one filter
    for( unsigned i = 0; i<number_of_filter_types; i++) {
        tbb::filter_t<void, void> one_filter( filter_table[i], i_filter );
        ASSERT(filter_node_count==1, "some filter nodes left after previous iteration?");
        resetCounters();
        tbb::parallel_pipeline( n_tokens, one_filter );
        // no need to check counters
#if __TBB_CPP11_LAMBDAS_PRESENT
        tbb::atomic<int> counter;
        counter = max_counter;
        // Construct filter using lambda-syntax when parallel_pipeline() is being run;
        tbb::parallel_pipeline( n_tokens,
            tbb::make_filter<void, void>(filter_table[i], [&counter]( tbb::flow_control& control ) {
                    if( counter-- == 0 )
                        control.stop();
                    }
            )
        );
#endif
    }
    ASSERT(!filter_node_count, "filter_node objects leaked");
}

template<typename t1, typename t2>
void run_filter_set(
        input_filter<t1>& i_filter,
        middle_filter<t1,t2>& m_filter,
        output_filter<t2>& o_filter,
        mode_array *filter_type,
        final_assert_type my_t) {
    tbb::filter_t<void, t1> filter1( filter_type[0], i_filter );
    tbb::filter_t<t1, t2> filter2( filter_type[1], m_filter );
    tbb::filter_t<t2, void> filter3( filter_type[2], o_filter );
    ASSERT(filter_node_count==3, "some filter nodes left after previous iteration?");
    resetCounters();
    // Create filters sequence when parallel_pipeline() is being run
    tbb::parallel_pipeline( n_tokens, filter1 & filter2 & filter3 );
    checkCounters(my_t);

    // Create filters sequence partially outside parallel_pipeline() and also when parallel_pipeline() is being run
    tbb::filter_t<void, t2> filter12;
    filter12 = filter1 & filter2;
    resetCounters();
    tbb::parallel_pipeline( n_tokens, filter12 & filter3 );
    checkCounters(my_t);

    tbb::filter_t<void, void> filter123 = filter12 & filter3;
    // Run pipeline twice with the same filter sequence
    for( unsigned i = 0; i<2; i++ ) {
        resetCounters();
        tbb::parallel_pipeline( n_tokens, filter123 );
        checkCounters(my_t);
    }

    // Now copy-construct another filter_t instance, and use it to run pipeline
    {
        tbb::filter_t<void, void> copy123( filter123 );
        resetCounters();
        tbb::parallel_pipeline( n_tokens, copy123 );
        checkCounters(my_t);
    }

    // Construct filters and create the sequence when parallel_pipeline() is being run
    resetCounters();
    tbb::parallel_pipeline( n_tokens,
               tbb::make_filter<void, t1>(filter_type[0], i_filter) &
               tbb::make_filter<t1, t2>(filter_type[1], m_filter) &
               tbb::make_filter<t2, void>(filter_type[2], o_filter) );
    checkCounters(my_t);

    // Construct filters, make a copy, destroy the original filters, and run with the copy
    int cnt = filter_node_count;
    {
        tbb::filter_t<void, void>* p123 = new tbb::filter_t<void,void> (
               tbb::make_filter<void, t1>(filter_type[0], i_filter) &
               tbb::make_filter<t1, t2>(filter_type[1], m_filter) &
               tbb::make_filter<t2, void>(filter_type[2], o_filter) );
        ASSERT(filter_node_count==cnt+5, "filter node accounting error?");
        tbb::filter_t<void, void> copy123( *p123 );
        delete p123;
        ASSERT(filter_node_count==cnt+5, "filter nodes deleted prematurely?");
        resetCounters();
        tbb::parallel_pipeline( n_tokens, copy123 );
        checkCounters(my_t);
    }

    // construct a filter with temporaries
    {
        tbb::filter_t<void, void> my_filter;
        fill_chain<t1,t2>( my_filter, filter_type, i_filter, m_filter, o_filter );
        resetCounters();
        tbb::parallel_pipeline( n_tokens, my_filter );
        checkCounters(my_t);
    }
    ASSERT(filter_node_count==cnt, "scope ended but filter nodes not deleted?");
}

#if __TBB_CPP11_LAMBDAS_PRESENT
template <typename t1, typename t2>
void run_lambdas_test( mode_array *filter_type ) {
    tbb::atomic<int> counter;
    counter = max_counter;
    // Construct filters using lambda-syntax and create the sequence when parallel_pipeline() is being run;
    resetCounters();  // only need the output_counter reset.
    tbb::parallel_pipeline( n_tokens,
        tbb::make_filter<void, t1>(filter_type[0], [&counter]( tbb::flow_control& control ) -> t1 {
                if( --counter < 0 )
                    control.stop();
                return t1(); }
        ) &
        tbb::make_filter<t1, t2>(filter_type[1], []( t1 /*my_storage*/ ) -> t2 {
                return t2(); }
        ) &
        tbb::make_filter<t2, void>(filter_type[2], [] ( t2 ) -> void {
                output_counter++; }
        )
    );
    checkCounters(no_pointer_counts);  // don't have to worry about specializations
    counter = max_counter;
    // pointer filters
    resetCounters();
    tbb::parallel_pipeline( n_tokens,
        tbb::make_filter<void, t1*>(filter_type[0], [&counter]( tbb::flow_control& control ) -> t1* {
                if( --counter < 0 ) {
                    control.stop();
                    return NULL;
                }
                return new(fetchNextBuffer()) t1(); }
        ) &
        tbb::make_filter<t1*, t2*>(filter_type[1], []( t1* my_storage ) -> t2* {
                tbb::tbb_allocator<t1>().destroy(my_storage); // my_storage->~t1();
                return new(my_storage) t2(); }
        ) &
        tbb::make_filter<t2*, void>(filter_type[2], [] ( t2* my_storage ) -> void {
                tbb::tbb_allocator<t2>().destroy(my_storage);  // my_storage->~t2();
                freeBuffer(my_storage);
                output_counter++; }
        )
    );
    checkCounters(no_pointer_counts);
    // first filter outputs pointer
    counter = max_counter;
    resetCounters();
    tbb::parallel_pipeline( n_tokens,
        tbb::make_filter<void, t1*>(filter_type[0], [&counter]( tbb::flow_control& control ) -> t1* {
                if( --counter < 0 ) {
                    control.stop();
                    return NULL;
                }
                return new(fetchNextBuffer()) t1(); }
        ) &
        tbb::make_filter<t1*, t2>(filter_type[1], []( t1* my_storage ) -> t2 {
                tbb::tbb_allocator<t1>().destroy(my_storage);   // my_storage->~t1();
                freeBuffer(my_storage);
                return t2(); }
        ) &
        tbb::make_filter<t2, void>(filter_type[2], [] ( t2 /*my_storage*/) -> void {
                output_counter++; }
        )
    );
    checkCounters(no_pointer_counts);
    // second filter outputs pointer
    counter = max_counter;
    resetCounters();
    tbb::parallel_pipeline( n_tokens,
        tbb::make_filter<void, t1>(filter_type[0], [&counter]( tbb::flow_control& control ) -> t1 {
                if( --counter < 0 ) {
                    control.stop();
                }
                return t1(); }
        ) &
        tbb::make_filter<t1, t2*>(filter_type[1], []( t1 /*my_storage*/ ) -> t2* {
                return new(fetchNextBuffer()) t2(); }
        ) &
        tbb::make_filter<t2*, void>(filter_type[2], [] ( t2* my_storage) -> void {
                tbb::tbb_allocator<t2>().destroy(my_storage);  // my_storage->~t2();
                freeBuffer(my_storage);
                output_counter++; }
        )
    );
    checkCounters(no_pointer_counts);
}
#endif

template<typename type1, typename type2>
void run_function(const char *l1, const char *l2) {
    ASSERT(!filter_node_count, NULL);
    REMARK("Testing < %s, %s >", l1, l2 );
#if __TBB_CPP11_LAMBDAS_PRESENT
    REMARK( " ( + lambdas)");
#endif
    check_intbuffer = (!strcmp(l1,"int") && !strcmp(l2,"int"));
    if(check_intbuffer) REMARK(", check output of filters");
    REMARK("\n");

    Check<type1> check1;  // check constructions/destructions
    Check<type2> check2;  // for type1 or type2 === check_type<T>

    const size_t number_of_filters = 3;

    input_filter<type1> i_filter;
    input_filter<type1*> p_i_filter;

    middle_filter<type1, type2> m_filter;
    middle_filter<type1*, type2> pr_m_filter;
    middle_filter<type1, type2*> rp_m_filter;
    middle_filter<type1*, type2*> pp_m_filter;

    output_filter<type2> o_filter;
    output_filter<type2*> p_o_filter;

    // allocate the buffers for the filters
    unsigned max_size = (sizeof(type1) > sizeof(type2) ) ? sizeof(type1) : sizeof(type2);
    for(unsigned i = 0; i < (unsigned)n_buffers; ++i) {
        buffers[i] = malloc(max_size);
        buf_available[i] = true;
    }

    unsigned limit = 1;
    // Test pipeline that contains number_of_filters filters
    for( unsigned i=0; i<number_of_filters; ++i)
        limit *= number_of_filter_types;
    // Iterate over possible filter sequences
    for( unsigned numeral=0; numeral<limit; ++numeral ) {
        unsigned temp = numeral;
        tbb::filter::mode filter_type[number_of_filter_types];
        for( unsigned i=0; i<number_of_filters; ++i, temp/=number_of_filter_types )
            filter_type[i] = filter_table[temp%number_of_filter_types];

        run_filter_set<type1,type2>(i_filter, m_filter, o_filter, filter_type, assert_nonpointer );
        run_filter_set<type1*,type2>(p_i_filter, pr_m_filter, o_filter, filter_type, assert_firstpointer);
        run_filter_set<type1,type2*>(i_filter, rp_m_filter, p_o_filter, filter_type, assert_secondpointer);
        run_filter_set<type1*,type2*>(p_i_filter, pp_m_filter, p_o_filter, filter_type, assert_allpointer);

#if __TBB_CPP11_LAMBDAS_PRESENT
        run_lambdas_test<type1,type2>(filter_type);
#endif
    }
    ASSERT(!filter_node_count, "filter_node objects leaked");

    for(unsigned i = 0; i < (unsigned)n_buffers; ++i) {
        free(buffers[i]);
    }
}

#include "tbb/task_scheduler_init.h"

int TestMain() {
#if TBB_USE_DEBUG
    // size and copyability.
    REMARK("is_large_object<int>::value=%d\n", tbb::interface6::internal::is_large_object<int>::value);
    REMARK("is_large_object<double>::value=%d\n", tbb::interface6::internal::is_large_object<double>::value);
    REMARK("is_large_object<int *>::value=%d\n", tbb::interface6::internal::is_large_object<int *>::value);
    REMARK("is_large_object<check_type<int> >::value=%d\n", tbb::interface6::internal::is_large_object<check_type<int> >::value);
    REMARK("is_large_object<check_type<int>* >::value=%d\n", tbb::interface6::internal::is_large_object<check_type<int>* >::value);
    REMARK("is_large_object<check_type<short> >::value=%d\n\n", tbb::interface6::internal::is_large_object<check_type<short> >::value);
#endif
    // Test with varying number of threads.
    for( int nthread=MinThread; nthread<=MaxThread; ++nthread ) {
        // Initialize TBB task scheduler
        REMARK("\nTesting with nthread=%d\n", nthread);
        tbb::task_scheduler_init init(nthread);

        // Run test several times with different types
        run_function_spec();
        run_function<size_t,int>("size_t", "int");
        run_function<int,double>("int", "double");
        run_function<size_t,double>("size_t", "double");
        run_function<size_t,bool>("size_t", "bool");
        run_function<int,int>("int","int");
        run_function<check_type<unsigned int>,size_t>("check_type<unsigned int>", "size_t");
        run_function<check_type<unsigned short>,size_t>("check_type<unsigned short>", "size_t");
        run_function<check_type<unsigned int>, check_type<unsigned int> >("check_type<unsigned int>", "check_type<unsigned int>");
        run_function<check_type<unsigned int>, check_type<unsigned short> >("check_type<unsigned int>", "check_type<unsigned short>");
        run_function<check_type<unsigned short>, check_type<unsigned short> >("check_type<unsigned short>", "check_type<unsigned short>");
        run_function<double, check_type<unsigned short> >("double", "check_type<unsigned short>");
    }
    return Harness::Done;
}

