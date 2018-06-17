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

#if _MSC_VER
    #pragma warning (disable: 4503) // Suppress "decorated name length exceeded, name was truncated" warning
#endif

#ifdef TEST_COARSE_GRAINED_LOCK_IMPLEMENTATION
    #include "../perf/coarse_grained_raii_lru_cache.h"
    #define selected_raii_lru_cache_impl coarse_grained_raii_lru_cache
#else
    #define TBB_PREVIEW_CONCURRENT_LRU_CACHE 1
    #include "tbb/concurrent_lru_cache.h"
    #define selected_raii_lru_cache_impl tbb::concurrent_lru_cache
#endif

#include "harness_test_cases_framework.h"
#include "harness.h"
#include "harness_barrier.h"

#include <utility>

#include "tbb/task_scheduler_init.h"

namespace helpers{
    // Busy work and calibration helpers
    unsigned int one_us_iters = 345; // default value

    // if user wants to calibrate to microseconds on particular machine, call
    // this at beginning of program; sets one_us_iters to number of iters to
    // busy_wait for approx. 1 us
//    void calibrate_busy_wait() {
//        tbb::tick_count t0, t1;
//
//        t0 = tbb::tick_count::now();
//        for (volatile unsigned int i=0; i<1000000; ++i) continue;
//        t1 = tbb::tick_count::now();
//
//        one_us_iters = (unsigned int)((1000000.0/(t1-t0).seconds())*0.000001);
//        printf("one_us_iters: %d\n", one_us_iters);
//    }
    void busy_wait(int us)
    {
        unsigned int iter = us*one_us_iters;
        for (volatile unsigned int i=0; i<iter; ++i) continue;
    }
}
namespace helpers{
    template<class T> void ignore( const T& ) { }
    //TODO: add test cases for prevent_optimizing_out function
    template<typename type>
    void prevent_optimizing_out(type volatile const& s){
        volatile const type* dummy = &s;
        ignore(dummy);
    }

    struct empty_fixture{};

    template<typename argument_type>
    struct native_for_concurrent_op_repeated:NoAssign{
        typedef void (*test_function_pointer_type)(argument_type&);

        argument_type& m_counter_ref;
        test_function_pointer_type m_test_function_pointer_type;
        std::size_t m_repeat_number;
        native_for_concurrent_op_repeated(argument_type& counter_ref, test_function_pointer_type action, std::size_t repeat_number)
            :m_counter_ref(counter_ref), m_test_function_pointer_type(action), m_repeat_number(repeat_number)
        {}
        template <typename ignored_parameter_type>
        void operator()(ignored_parameter_type const&)const{
            for (size_t i=0; i<m_repeat_number;++i){
                m_test_function_pointer_type(m_counter_ref);
            }
        }

    };

    template <typename counter_type = size_t>
    struct object_instances_counting_type{
        counter_type * m_p_count;
        object_instances_counting_type(): m_p_count (new counter_type){*m_p_count =1; } //to overcome absence of constructor in tbb::atomic
        ~object_instances_counting_type(){ if (! --(*m_p_count)){delete(m_p_count);}}
        object_instances_counting_type(object_instances_counting_type const& other): m_p_count(other.m_p_count){
            ++(*m_p_count);
        }
        object_instances_counting_type& operator=(object_instances_counting_type other){
            std::swap(this->m_p_count,other.m_p_count);
            return *this;
        }
        size_t instances_count()const {return *m_p_count;}
    };
    typedef object_instances_counting_type<> object_instances_counting_serial_type;
    typedef object_instances_counting_type<tbb::atomic<std::size_t> > object_instances_counting_concurrent_type;

    namespace object_instances_counting_type_test_cases{
        namespace serial_tests{
            TEST_CASE_WITH_FIXTURE(test_object_instances_counting_type_creation,empty_fixture){
                ASSERT(object_instances_counting_serial_type().instances_count()==1,"newly created instance by definition has instances_count equal to 1");
            }
            TEST_CASE_WITH_FIXTURE(test_object_instances_counting_type_copy,empty_fixture){
                object_instances_counting_serial_type source;
                ASSERT(object_instances_counting_serial_type(source).instances_count()==2,"copy should increase ref count");
            }
            TEST_CASE_WITH_FIXTURE(test_object_instances_counting_type_assignment,empty_fixture){
                object_instances_counting_serial_type source;
                object_instances_counting_serial_type assigned;
                assigned = source;
                ASSERT(source.instances_count()==2,"assign should increase ref count");
                ASSERT(assigned.instances_count()==2,"assign should increase ref count");
            }
        }
        namespace concurrent_tests{
            typedef native_for_concurrent_op_repeated<object_instances_counting_concurrent_type>  native_for_concurrent_op;

            struct native_for_single_op_repeated_fixture{
                object_instances_counting_concurrent_type source;
                void run_native_for_and_assert_source_is_unique(native_for_concurrent_op::test_function_pointer_type operation,const char* msg){
                    //TODO: refactor number of threads into separate fixture
                    const size_t number_of_threads = min(4,tbb::task_scheduler_init::default_num_threads());
                    const size_t repeats_per_thread = 1000000;

                    NativeParallelFor(number_of_threads , native_for_concurrent_op(source,operation,repeats_per_thread));
                    ASSERT(source.instances_count()==1,msg);
                }

            };
            TEST_CASE_WITH_FIXTURE(test_object_instances_counting_type_copy,native_for_single_op_repeated_fixture){
                struct _{ static void copy(object_instances_counting_concurrent_type& a_source){
                    object_instances_counting_concurrent_type copy(a_source);
                    helpers::prevent_optimizing_out(copy);
                }};
                run_native_for_and_assert_source_is_unique(&_::copy,"reference counting during copy construction/destruction is not thread safe ?");
            }
            TEST_CASE_WITH_FIXTURE(test_object_instances_counting_type_assignment,native_for_single_op_repeated_fixture){
                struct _{ static void assign(object_instances_counting_concurrent_type& a_source){
                    object_instances_counting_concurrent_type assigned;
                    assigned = a_source;
                    helpers::prevent_optimizing_out(assigned);
                }};
                run_native_for_and_assert_source_is_unique(&_::assign,"reference counting during assigning/destruction is not thread safe ?");
            }

        }
}
}

struct get_lru_cache_type{

    template< typename parameter1, typename parameter2, typename parameter3=void>
    struct apply{
        typedef selected_raii_lru_cache_impl<parameter1,parameter2,parameter3> type;
    };
    template< typename parameter1, typename parameter2>
    struct apply<parameter1,parameter2,void>{
        typedef selected_raii_lru_cache_impl<parameter1,parameter2> type;
    };

};

// these includes are needed for test_task_handle_mv_sem*
#include <vector>
#include <string>
#include <functional>

namespace serial_tests{
    using namespace helpers;
    namespace usability{
    namespace compilation_only{
        TEST_CASE_WITH_FIXTURE(test_creation_and_use_interface,empty_fixture){
            struct dummy_function{static int _(int key){return key;}};
            typedef get_lru_cache_type::apply<int,int>::type cache_type;
            size_t number_of_lru_history_items = 8;
            cache_type cache((&dummy_function::_),(number_of_lru_history_items));
            int dummy_key=0;
            cache_type::handle h = cache[dummy_key];
            int value = h.value();
            (void)value;
        }
    }
    namespace behaviour {
        namespace helpers {
            template <size_t id> struct tag {};
            template< typename tag, typename value_and_key_type>
            struct call_counting_function {
                static int calls_count;
                static value_and_key_type _(value_and_key_type key) {
                    ++calls_count;
                    return key;
                }
            };
            template< typename tag, typename value_and_key_type>
            int call_counting_function<tag, value_and_key_type>::calls_count = 0;
        }

        using std::string;
        struct mv_sem_fixture {
            struct item_init{
                static string init(string key) {
                    return key;
                }
            };
            typedef tbb::concurrent_lru_cache<string, string> cache_type;
            typedef cache_type::handle handle_type;
            cache_type cache;
            mv_sem_fixture() : cache((&item_init::init), 1) {};

            handle_type default_ctor_check;
        };

        TEST_CASE_WITH_FIXTURE(test_task_handle_mv_sem, mv_sem_fixture) {
            handle_type handle;
            handle_type foobar = handle_type();

            //c++03 : handle_move_t assignment
            handle = cache["handle"];

            //c++03 : init ctor from handle_move_t
            handle_type foo = cache["bar"];

            //c++03 : init ctor from handle_move_t
            handle_type handle1(move(handle));

            //c++03 : handle_move_t assignment
            handle = move(handle1);

            ASSERT(!handle_type(), "user-defined to-bool conversion does not work");
            ASSERT(handle, "user-defined to-bool conversion does not work");

            handle = handle_type();
        }

        TEST_CASE_WITH_FIXTURE(test_task_handle_mv_sem_certain_case, mv_sem_fixture) {
            // there is no way to use handle_object as vector argument in C++03
            // because argument must meet requirements of CopyAssignable and
            // CopyConstructible (C++ documentation)
#if __TBB_CPP11_RVALUE_REF_PRESENT
            // retain handle_object to keep an item in the cache if it is still active without aging
            handle_type sheep = cache["sheep"];
            handle_type horse = cache["horse"];
            handle_type bull = cache["bull"];

            std::vector<handle_type> animals;
            animals.reserve(5);
            animals.emplace_back(std::move(sheep));
            animals.emplace_back(std::move(horse));
            animals[0] = std::move(bull);
            // after resize() vec will be full of default constructed handlers with null pointers
            // on item in cache and on cache which item belongs to
            animals.resize(10);
#endif /* __TBB_CPP11_RVALUE_REF_PRESENT */
        }

        TEST_CASE_WITH_FIXTURE(test_cache_returns_only_values_from_value_function,empty_fixture){
            struct dummy_function{static int _(int /*key*/){return 0xDEADBEEF;}};
            typedef get_lru_cache_type::apply<int,int>::type cache_type;
            size_t number_of_lru_history_items = 8;
            int dummy_key=1;
            cache_type cache((&dummy_function::_),(number_of_lru_history_items));
            ASSERT(dummy_function::_(dummy_key)==cache[dummy_key].value(),"cache operator() must return only values obtained from value_function ");
        }

        TEST_CASE_WITH_FIXTURE(test_value_function_called_only_on_cache_miss,empty_fixture){
            typedef helpers::tag<__LINE__> tag;
            typedef helpers::call_counting_function<tag,int> function;
            typedef get_lru_cache_type::apply<int,int>::type cache_type;
            size_t number_of_lru_history_items = 8;
            cache_type cache((&function::_),(number_of_lru_history_items));

            int dummy_key=0;
            cache[dummy_key];
            cache[dummy_key];
            ASSERT(function::calls_count==1,"value function should be called only on a cache miss");
        }
        }
        namespace helpers{
            using ::helpers::object_instances_counting_serial_type;
        }
        namespace helpers{
            template<typename value_type>
            struct clonning_function:NoAssign{
                value_type& m_ref_original;
                clonning_function(value_type& ref_original):m_ref_original(ref_original){}
                template<typename key_type>
                value_type operator()(key_type)const{ return m_ref_original;}
            };
        }
        struct instance_counting_fixture{
            static const size_t number_of_lru_history_items = 8;

            typedef helpers::clonning_function<helpers::object_instances_counting_serial_type> cloner_type;
            typedef get_lru_cache_type::apply<size_t,helpers::object_instances_counting_serial_type,cloner_type>::type cache_type;
            helpers::object_instances_counting_serial_type source;
            cloner_type cloner;
            cache_type cache;

            instance_counting_fixture():cloner((source)),cache(cloner,number_of_lru_history_items){}
        };

        TEST_CASE_WITH_FIXTURE(test_cache_stores_unused_objects,instance_counting_fixture){
            for (size_t i=0;i<number_of_lru_history_items;++i){
                cache[i];
            }
            ASSERT(source.instances_count()> 1,"cache should store some unused objects ");
        }

        TEST_CASE_WITH_FIXTURE(test_cache_stores_no_more_then_X_number_of_unused_objects,instance_counting_fixture){
            for (size_t i=0;i<number_of_lru_history_items+1;++i){
                cache[i];
            }
            ASSERT(source.instances_count()== number_of_lru_history_items+1,"cache should respect number of stored unused objects to number passed in constructor");
        }

        namespace helpers{
            template< typename key_type, typename value_type>
            struct map_searcher:NoAssign{
                typedef std::map<key_type,value_type> map_type;
                map_type & m_map_ref;
                map_searcher(map_type & map_ref): m_map_ref(map_ref) {}
                value_type& operator()(key_type k){
                    typename map_type::iterator it =m_map_ref.find(k);
                    if (it==m_map_ref.end()){
                        it = m_map_ref.insert(it,std::make_pair(k,value_type()));
                    }
                    return it->second;
                }
            };
        }

        struct filled_instance_counting_fixture_with_external_map{
            static const size_t number_of_lru_history_items = 8;

            typedef helpers::map_searcher<size_t,helpers::object_instances_counting_serial_type> map_searcher_type;
            typedef map_searcher_type::map_type objects_map_type;
            typedef get_lru_cache_type::apply<size_t,helpers::object_instances_counting_serial_type,map_searcher_type>::type cache_type;
            map_searcher_type::map_type objects_map;
            cache_type cache;
            filled_instance_counting_fixture_with_external_map():cache(map_searcher_type(objects_map),number_of_lru_history_items){}
            bool is_evicted(size_t k){
                objects_map_type::iterator it =objects_map.find(k);
                ASSERT(it!=objects_map.end(),"no value for key - error in test logic ?");
                return it->second.instances_count()==1;
            }
            void fill_up_cache(size_t lower_bound, size_t upper_bound){
                for (size_t i=lower_bound;i<upper_bound;++i){
                    cache[i];
                }
            }
        };

        TEST_CASE_WITH_FIXTURE(test_cache_should_evict_unused_objects_lru_order,filled_instance_counting_fixture_with_external_map){
            ASSERT(number_of_lru_history_items > 2,"incorrect test setup");
            fill_up_cache(0,number_of_lru_history_items);
            //heat up first element
            cache[0];
            //cause eviction
            cache[number_of_lru_history_items];
            ASSERT(is_evicted(1) && !is_evicted(0),"cache should evict items in lru order");
        }

        TEST_CASE_WITH_FIXTURE(test_live_handler_object_prevents_item_from_eviction,filled_instance_counting_fixture_with_external_map){
            cache_type::handle h = cache[0];
            //cause eviction
            fill_up_cache(1,number_of_lru_history_items+2);
            ASSERT(is_evicted(1) && !is_evicted(0),"cache should not evict items in use");
        }
        TEST_CASE_WITH_FIXTURE(test_live_handler_object_is_ref_counted,filled_instance_counting_fixture_with_external_map){
            cache_type::handle h = cache[0];
            {
                cache_type::handle h1 = cache[0];
            }
            //cause eviction
            fill_up_cache(1,number_of_lru_history_items+2);
            ASSERT(is_evicted(1) && !is_evicted(0),"cache should not evict items in use");
        }
    }
}


namespace concurrency_tests{
    namespace helpers{
        using namespace ::helpers;
    }
    namespace helpers{
        //key_type must be convertible to array index
        template< typename key_type, typename value_type, size_t array_size>
        struct array_searcher:NoAssign{
            typedef value_type array_type[array_size];
            array_type const& m_array_ref;
            array_searcher(array_type const& array_ref): m_array_ref(array_ref) {}
            const value_type& operator()(key_type k)const{
                size_t index = k;
                ASSERT(k < array_size,"incorrect test setup");
                return m_array_ref[index];
            }
        };
    }

    struct filled_instance_counting_fixture_with_external_array{
        static const size_t number_of_lru_history_items = 8;
        static const size_t array_size = 16*number_of_lru_history_items;

        typedef helpers::array_searcher<size_t,helpers::object_instances_counting_concurrent_type,array_size> array_searcher_type;
        typedef array_searcher_type::array_type objects_array_type;
        typedef get_lru_cache_type::apply<size_t,helpers::object_instances_counting_concurrent_type,array_searcher_type>::type cache_type;
        array_searcher_type::array_type objects_array;
        cache_type cache;
        filled_instance_counting_fixture_with_external_array():cache(array_searcher_type(objects_array),number_of_lru_history_items){}
        bool is_evicted(size_t k)const{
            return array_searcher_type(objects_array)(k).instances_count()==1;
        }
        void fill_up_cache(size_t lower_bound, size_t upper_bound){
            for (size_t i=lower_bound;i<upper_bound;++i){
                cache[i];
            }
        }
        size_t number_of_non_evicted_from_cache()const{
            size_t result=0;
            for (size_t i=0; i<array_size; ++i){
                if (!this->is_evicted(i)){
                    ++result;
                }
            }
            return result;
        }
    };


    //TODO: make this more reproducible
    //TODO: split this test case in two parts
    TEST_CASE_WITH_FIXTURE(correctness_of_braces_and_handle_destructor,filled_instance_counting_fixture_with_external_array){
        typedef correctness_of_braces_and_handle_destructor self_type;
        struct _{static void use_cache(self_type& tc){
            for (size_t i=0;i<array_size;++i){
                cache_type::handle h=tc.cache[i];
                helpers::prevent_optimizing_out(h.value());
            }

        }};
        static const size_t repeat_number = 2;
        static const size_t number_of_threads = 4 * tbb::task_scheduler_init::default_num_threads(); //have 4x over subscription
        static const size_t repeats_per_thread = 4;

        for (size_t i=0; i < repeat_number; i++){
            NativeParallelFor(number_of_threads,helpers::native_for_concurrent_op_repeated<self_type>(*this,&_::use_cache,repeats_per_thread));
            fill_up_cache(0,array_size);
            ASSERT(number_of_non_evicted_from_cache()==number_of_lru_history_items,"thread safety is broken for cache ");
        }
    }
}
