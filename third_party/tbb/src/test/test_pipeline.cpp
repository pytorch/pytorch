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

#include "tbb/tbb_stddef.h"
#include "tbb/pipeline.h"
#include "tbb/spin_mutex.h"
#include "tbb/atomic.h"
#include <cstdlib>
#include <cstdio>
#include "harness.h"

// In the test, variables related to token counting are declared
// as unsigned long to match definition of tbb::internal::Token.

struct Buffer {
    //! Indicates that the buffer is not used.
    static const unsigned long unused = ~0ul;
    unsigned long id;
    //! True if Buffer is in use.
    bool is_busy;
    unsigned long sequence_number;
    Buffer() : id(unused), is_busy(false), sequence_number(unused) {}
};

class waiting_probe {
    size_t check_counter;
public:
    waiting_probe() : check_counter(0) {}
    bool required( ) {
        ++check_counter;
        return !((check_counter+1)&size_t(0x7FFF));
    }
    void probe( ); // defined below
};

static const unsigned MaxStreamSize = 8000;
static const unsigned MaxStreamItemsPerThread = 1000;
//! Maximum number of filters allowed
static const unsigned MaxFilters = 5;
static unsigned StreamSize;
static const unsigned MaxBuffer = 8;
static bool Done[MaxFilters][MaxStreamSize];
static waiting_probe WaitTest;
static unsigned out_of_order_count;

#include "harness_concurrency_tracker.h"

class BaseFilter: public tbb::filter {
    bool* const my_done;
    const bool my_is_last;
    bool my_is_running;
public:
    tbb::atomic<tbb::internal::Token> current_token;
    BaseFilter( tbb::filter::mode type, bool done[], bool is_last ) :
        filter(type),
        my_done(done),
        my_is_last(is_last),
        my_is_running(false),
        current_token()
    {}
    virtual Buffer* get_buffer( void* item ) {
        current_token++;
        return static_cast<Buffer*>(item);
    }
    void* operator()( void* item ) __TBB_override {
        Harness::ConcurrencyTracker ct;
        if( is_serial() )
            ASSERT( !my_is_running, "premature entry to serial stage" );
        my_is_running = true;
        Buffer* b = get_buffer(item);
        if( b ) {
            if( is_ordered() ) {
                if( b->sequence_number == Buffer::unused )
                    b->sequence_number = current_token-1;
                else
                    ASSERT( b->sequence_number==current_token-1, "item arrived out of order" );
            } else if( is_serial() ) {
                if( b->sequence_number != current_token-1 && b->sequence_number != Buffer::unused )
                    out_of_order_count++;
            }
            ASSERT( b->id < StreamSize, NULL );
            ASSERT( !my_done[b->id], "duplicate processing of token?" );
            ASSERT( b->is_busy, NULL );
            my_done[b->id] = true;
            if( my_is_last ) {
                b->id = Buffer::unused;
                b->sequence_number = Buffer::unused;
                __TBB_store_with_release(b->is_busy, false);
            }
        }
        my_is_running = false;
        return b;
    }
};

class InputFilter: public BaseFilter {
    tbb::spin_mutex input_lock;
    Buffer buffer[MaxBuffer];
    const tbb::internal::Token my_number_of_tokens;
public:
    InputFilter( tbb::filter::mode type, tbb::internal::Token ntokens, bool done[], bool is_last ) :
        BaseFilter(type, done, is_last),
        my_number_of_tokens(ntokens)
    {}
    Buffer* get_buffer( void* ) __TBB_override {
        unsigned long next_input;
        unsigned free_buffer = 0;
        { // lock protected scope
            tbb::spin_mutex::scoped_lock lock(input_lock);
            if( current_token>=StreamSize )
                return NULL;
            next_input = current_token++;
            // once in a while, emulate waiting for input; this only makes sense for serial input
            if( is_serial() && WaitTest.required() )
                WaitTest.probe( );
            while( free_buffer<MaxBuffer )
                if( __TBB_load_with_acquire(buffer[free_buffer].is_busy) )
                    ++free_buffer;
                else {
                    buffer[free_buffer].is_busy = true;
                    break;
                }
        }
        ASSERT( free_buffer<my_number_of_tokens, "premature reuse of buffer" );
        Buffer* b = &buffer[free_buffer];
        ASSERT( &buffer[0] <= b, NULL );
        ASSERT( b <= &buffer[MaxBuffer-1], NULL );
        ASSERT( b->id == Buffer::unused, NULL);
        b->id = next_input;
        ASSERT( b->sequence_number == Buffer::unused, NULL);
        return b;
    }
};

//! The struct below repeats layout of tbb::pipeline.
struct hacked_pipeline {
    tbb::filter* filter_list;
    tbb::filter* filter_end;
    tbb::empty_task* end_counter;
    tbb::atomic<tbb::internal::Token> input_tokens;
    tbb::atomic<tbb::internal::Token> token_counter;
    bool end_of_input;
    bool has_thread_bound_filters;

    virtual ~hacked_pipeline();
};

//! The struct below repeats layout of tbb::internal::input_buffer.
struct hacked_input_buffer {
    void* array; // This should be changed to task_info* if ever used
    void* my_sem; // This should be changed to semaphore* if ever used
    tbb::internal::Token array_size;
    tbb::internal::Token low_token;
    tbb::spin_mutex array_mutex;
    tbb::internal::Token high_token;
    bool is_ordered;
    bool is_bound;
};

//! The struct below repeats layout of tbb::filter.
struct hacked_filter {
    tbb::filter* next_filter_in_pipeline;
    hacked_input_buffer* my_input_buffer;
    unsigned char my_filter_mode;
    tbb::filter* prev_filter_in_pipeline;
    tbb::pipeline* my_pipeline;
    tbb::filter* next_segment;

    virtual ~hacked_filter();
};

bool do_hacking_tests = true;
const tbb::internal::Token tokens_before_wraparound = 0xF;

void TestTrivialPipeline( unsigned nthread, unsigned number_of_filters ) {
    // There are 3 filter types: parallel, serial_in_order and serial_out_of_order
    static const tbb::filter::mode filter_table[] = { tbb::filter::parallel, tbb::filter::serial_in_order, tbb::filter::serial_out_of_order};
    const unsigned number_of_filter_types = sizeof(filter_table)/sizeof(filter_table[0]);
    REMARK( "testing with %lu threads and %lu filters\n", nthread, number_of_filters );
    ASSERT( number_of_filters<=MaxFilters, "too many filters" );
    ASSERT( sizeof(hacked_pipeline) == sizeof(tbb::pipeline), "layout changed for tbb::pipeline?" );
    ASSERT( sizeof(hacked_filter) == sizeof(tbb::filter), "layout changed for tbb::filter?" );
    tbb::internal::Token ntokens = nthread<MaxBuffer ? nthread : MaxBuffer;
    // Count maximum iterations number
    unsigned limit = 1;
    for( unsigned i=0; i<number_of_filters; ++i)
        limit *= number_of_filter_types;
    // Iterate over possible filter sequences
    for( unsigned numeral=0; numeral<limit; ++numeral ) {
        // Build pipeline
        tbb::pipeline pipeline;
        if( do_hacking_tests ) {
            // A private member of pipeline is hacked there for sake of testing wrap-around immunity.
            tbb::internal::punned_cast<hacked_pipeline*>(&pipeline)->token_counter = ~tokens_before_wraparound;
        }
        tbb::filter* filter[MaxFilters];
        unsigned temp = numeral;
        // parallelism_limit is the upper bound on the possible parallelism
        unsigned parallelism_limit = 0;
        for( unsigned i=0; i<number_of_filters; ++i, temp/=number_of_filter_types ) {
            tbb::filter::mode filter_type = filter_table[temp%number_of_filter_types];
            const bool is_last = i==number_of_filters-1;
            if( i==0 )
                filter[i] = new InputFilter(filter_type,ntokens,Done[i],is_last);
            else
                filter[i] = new BaseFilter(filter_type,Done[i],is_last);
            pipeline.add_filter(*filter[i]);
            // The ordered buffer of serial filters is hacked as well.
            if ( filter[i]->is_serial() ) {
                if( do_hacking_tests ) {
                    ((hacked_filter*)(void*)filter[i])->my_input_buffer->low_token = ~tokens_before_wraparound;
                    ((hacked_filter*)(void*)filter[i])->my_input_buffer->high_token = ~tokens_before_wraparound;
                }
                parallelism_limit += 1;
            } else {
                parallelism_limit = nthread;
            }
        }
        // Account for clipping of parallelism.
        if( parallelism_limit>nthread )
            parallelism_limit = nthread;
        if( parallelism_limit>ntokens )
            parallelism_limit = (unsigned)ntokens;
        Harness::ConcurrencyTracker::Reset();
        unsigned streamSizeLimit = min( MaxStreamSize, nthread * MaxStreamItemsPerThread );
        for( StreamSize=0; StreamSize<=streamSizeLimit; ) {
            memset( Done, 0, sizeof(Done) );
            for( unsigned i=0; i<number_of_filters; ++i ) {
                static_cast<BaseFilter*>(filter[i])->current_token=0;
            }
            pipeline.run( ntokens );
            ASSERT( !Harness::ConcurrencyTracker::InstantParallelism(), "filter still running?" );
            for( unsigned i=0; i<number_of_filters; ++i )
                ASSERT( static_cast<BaseFilter*>(filter[i])->current_token==StreamSize, NULL );
            for( unsigned i=0; i<MaxFilters; ++i )
                for( unsigned j=0; j<StreamSize; ++j ) {
                    ASSERT( Done[i][j]==(i<number_of_filters), NULL );
                }
            if( StreamSize < min(nthread*8, 32u) ) {
                ++StreamSize;
            } else {
                StreamSize = StreamSize*8/3;
            }
        }
        if( Harness::ConcurrencyTracker::PeakParallelism() < parallelism_limit )
            REMARK( "nthread=%lu ntokens=%lu MaxParallelism=%lu parallelism_limit=%lu\n",
                nthread, ntokens, Harness::ConcurrencyTracker::PeakParallelism(), parallelism_limit );
        for( unsigned i=0; i < number_of_filters; ++i ) {
            delete filter[i];
            filter[i] = NULL;
        }
        pipeline.clear();
    }
}

#include "harness_cpu.h"

static int nthread; // knowing number of threads is necessary to call TestCPUUserTime

void waiting_probe::probe( ) {
    if( nthread==1 ) return;
    REMARK("emulating wait for input\n");
    // Test that threads sleep while no work.
    // The master doesn't sleep so there could be 2 active threads if a worker is waiting for input
    TestCPUUserTime(nthread, 2);
}

#include "tbb/task_scheduler_init.h"

int TestMain () {
    out_of_order_count = 0;
    if( MinThread<1 ) {
        REPORT("must have at least one thread");
        exit(1);
    }
    if( tbb::TBB_runtime_interface_version()>TBB_INTERFACE_VERSION) {
        REMARK("Warning: implementation dependent tests disabled\n");
        do_hacking_tests = false;
    }

    // Test with varying number of threads.
    for( nthread=MinThread; nthread<=MaxThread; ++nthread ) {
        // Initialize TBB task scheduler
        tbb::task_scheduler_init init(nthread);

        // Test pipelines with n filters
        for( unsigned n=0; n<=MaxFilters; ++n )
            TestTrivialPipeline(nthread,n);

        // Test that all workers sleep when no work
        TestCPUUserTime(nthread);
    }
    if( !out_of_order_count )
        REPORT("Warning: out of order serial filter received tokens in order\n");
    return Harness::Done;
}
