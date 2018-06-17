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

#include "../examples/common/utility/utility.h"
#include "tbb/tick_count.h"
//#include <tbb/parallel_for.h>
#include "tbb/task_scheduler_init.h" //for number of threads
#include <functional>

#include "coarse_grained_raii_lru_cache.h"
#define TBB_PREVIEW_CONCURRENT_LRU_CACHE 1
#include "tbb/concurrent_lru_cache.h"

#define HARNESS_CUSTOM_MAIN 1
#define HARNESS_NO_PARSE_COMMAND_LINE 1

#include "../src/test/harness.h"
#include "../src/test/harness_barrier.h"

#include <vector>
#include <algorithm>
#include "tbb/mutex.h"

//TODO: probably move this to separate header utlity file
namespace micro_benchmarking{
namespace utils{
    template <typename type>
    void disable_elimination(type const& v){
        volatile type dummy = v;
        (void) dummy;
    }
    //Busy work and calibration helpers
    unsigned int one_us_iters = 345; // default value

    //TODO: add a CLI parameter for calibration run
    // if user wants to calibrate to microseconds on particular machine, call
    // this at beginning of program; sets one_us_iters to number of iters to
    // busy_wait for approx. 1 us
    void calibrate_busy_wait() {
        const unsigned niter = 1000000;
        tbb::tick_count t0 = tbb::tick_count::now();
        for (volatile unsigned int i=0; i<niter; ++i) continue;
        tbb::tick_count t1 = tbb::tick_count::now();

        one_us_iters = (unsigned int)(niter/(t1-t0).seconds())*1e-6;
    }

    void busy_wait(int us)
    {
        unsigned int iter = us*one_us_iters;
        for (volatile unsigned int i=0; i<iter; ++i) continue;
    }
}
}

struct parameter_pack{
    size_t time_window_sec;
    size_t time_check_granularity_ops;
    size_t cache_lru_history_size;
    size_t time_of_item_use_usec;
    size_t cache_miss_percent;
    int threads_number;
    size_t weight_of_initiation_call_usec;
    bool use_serial_initiation_function;
    parameter_pack(
            size_t a_time_window_sec
            ,size_t a_time_check_granularity_ops
            ,size_t a_cache_lru_history_size
            ,size_t a_time_of_item_use_usec, size_t a_cache_miss_percent
            , int a_threads_number ,size_t a_weight_of_initiation_call_usec
            , bool a_use_serial_initiation_function
    )   :
        time_window_sec(a_time_window_sec)
        ,time_check_granularity_ops(a_time_check_granularity_ops)
        ,cache_lru_history_size(a_cache_lru_history_size)
        ,time_of_item_use_usec(a_time_of_item_use_usec)
        ,cache_miss_percent(a_cache_miss_percent)
        ,threads_number(a_threads_number)
        ,weight_of_initiation_call_usec(a_weight_of_initiation_call_usec)
        ,use_serial_initiation_function(a_use_serial_initiation_function)
    {}
};

struct return_size_t {
    size_t m_weight_of_initiation_call_usec;
    bool use_serial_initiation_function;
    return_size_t(size_t a_weight_of_initiation_call_usec, bool a_use_serial_initiation_function)
        :m_weight_of_initiation_call_usec(a_weight_of_initiation_call_usec), use_serial_initiation_function(a_use_serial_initiation_function)
    {}
    size_t operator()(size_t key){
        static tbb::mutex mtx;
        if (use_serial_initiation_function){
            mtx.lock();
        }
        micro_benchmarking::utils::busy_wait(m_weight_of_initiation_call_usec);
        if (use_serial_initiation_function){
            mtx.unlock();
        }

        return key;
    }
};

template< typename a_cache_type>
struct throughput {
    typedef throughput self_type;
    typedef a_cache_type cache_type;

    parameter_pack m_parameter_pack;


    const size_t per_thread_sample_size ;
    typedef std::vector<size_t> access_sequence_type;
    access_sequence_type m_access_sequence;
    cache_type m_cache;
    Harness::SpinBarrier m_barrier;
    tbb::atomic<size_t> loops_count;

    throughput(parameter_pack a_parameter_pack)
        :m_parameter_pack(a_parameter_pack)
        ,per_thread_sample_size(m_parameter_pack.cache_lru_history_size *(1 +  m_parameter_pack.cache_miss_percent/100))
        ,m_access_sequence(m_parameter_pack.threads_number * per_thread_sample_size )
        ,m_cache(return_size_t(m_parameter_pack.weight_of_initiation_call_usec,m_parameter_pack.use_serial_initiation_function),m_parameter_pack.cache_lru_history_size)

    {
        loops_count=0;
        //TODO: check if changing from generating longer sequence to generating indexes in a specified range (i.e. making per_thread_sample_size fixed) give any change
        std::generate(m_access_sequence.begin(),m_access_sequence.end(),std::rand);
    }

    size_t operator()(){
        struct _{ static void  retrieve_from_cache(self_type* _this, size_t thread_index){
            parameter_pack& p = _this->m_parameter_pack;
            access_sequence_type::iterator const begin_it =_this->m_access_sequence.begin()+ thread_index * _this->per_thread_sample_size;
            access_sequence_type::iterator const end_it = begin_it +  _this->per_thread_sample_size;

            _this->m_barrier.wait();
            tbb::tick_count start = tbb::tick_count::now();

            size_t local_loops_count =0;
            do {
                size_t part_of_the_sample_so_far = (local_loops_count * p.time_check_granularity_ops) % _this->per_thread_sample_size;
                access_sequence_type::iterator const iteration_begin_it = begin_it + part_of_the_sample_so_far;
                access_sequence_type::iterator const iteration_end_it = iteration_begin_it +
                        (std::min)(p.time_check_granularity_ops, _this->per_thread_sample_size - part_of_the_sample_so_far);

                for (access_sequence_type::iterator it = iteration_begin_it; it < iteration_end_it; ++it){
                    typename cache_type::handle h = _this->m_cache[*it];
                    micro_benchmarking::utils::busy_wait(p.time_of_item_use_usec);
                    micro_benchmarking::utils::disable_elimination(h.value());
                }
                ++local_loops_count;
            }while((tbb::tick_count::now()-start).seconds() < p.time_window_sec);
            _this->loops_count+=local_loops_count;
        }};
        m_barrier.initialize(m_parameter_pack.threads_number);

        NativeParallelFor(m_parameter_pack.threads_number,std::bind1st(std::ptr_fun(&_::retrieve_from_cache),this));

        return loops_count * m_parameter_pack.time_check_granularity_ops;
    }
};

int main(int argc,const char** args ){

    size_t time_window_sec = 10;
    size_t cache_lru_history_size = 1000;
    size_t time_check_granularity_ops = 200;
    size_t time_of_item_use_usec = 100;
    size_t cache_miss_percent = 5;
    int threads_number =tbb::task_scheduler_init::default_num_threads();
    size_t weight_of_initiation_call_usec =1000;
    bool use_serial_initiation_function = false;
    bool use_coarse_grained_locked_cache = false;

    parameter_pack p(time_window_sec, time_check_granularity_ops, cache_lru_history_size,time_of_item_use_usec,cache_miss_percent,threads_number,weight_of_initiation_call_usec,use_serial_initiation_function);

    utility::parse_cli_arguments(argc,args,utility::cli_argument_pack()
            .arg(p.cache_lru_history_size,"cache-lru-history-size","")
            .arg(p.time_window_sec,"time-window","time frame for measuring, in seconds")
            .arg(p.threads_number,"n-of-threads","number of threads to run on")
            .arg(p.time_of_item_use_usec,"time-of-item-use","time between  consequent requests to the cache, in microseconds")
            .arg(p.cache_miss_percent,"cache-miss-percent","cache miss percent ")
            .arg(p.weight_of_initiation_call_usec,"initiation-call-weight","time occupied by a single call to initiation function, in microseconds")
            .arg(p.use_serial_initiation_function,"use-serial-initiation-function","limit lock-based serial initiation function")
            .arg(use_coarse_grained_locked_cache,"use-locked-version","use stl coarse grained lock based version")
            );

    typedef tbb::concurrent_lru_cache<size_t,size_t,return_size_t> tbb_cache;
    typedef coarse_grained_raii_lru_cache<size_t,size_t,return_size_t> coarse_grained_locked_cache;

    size_t operations =0;
    if (!use_coarse_grained_locked_cache){
        operations = throughput<tbb_cache>(p)();
    }else{
        operations = throughput<coarse_grained_locked_cache>(p)();
    }
    std::cout<<"operations: "<<operations<<std::endl;
    return 0;
}
