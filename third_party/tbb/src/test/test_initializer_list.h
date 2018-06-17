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

#ifndef __TBB_test_initializer_list_H
#define __TBB_test_initializer_list_H
#include "tbb/tbb_config.h"

#if __TBB_INITIALIZER_LISTS_PRESENT
#include <initializer_list>
#include <vector>
#include "harness_defs.h" //for int_to_type

namespace initializer_list_support_tests{
    template<typename container_type, typename element_type>
    void test_constructor(std::initializer_list<element_type> il, container_type const& expected){
        container_type vd (il);
        ASSERT(vd == expected,"initialization via explicit constructor call with init list failed");
    }


    template<typename container_type, typename element_type>
    void test_assignment_operator(std::initializer_list<element_type> il, container_type const& expected){
        container_type va;
        va = il;
        ASSERT(va == expected,"init list operator= failed");
    }

    struct skip_test {
        template<typename container_type, typename element_type>
        static void do_test(std::initializer_list<element_type>, container_type const&) { /* do nothing */ }
    };

    struct test_assign {
        template<typename container_type, typename element_type>
        static void do_test( std::initializer_list<element_type> il, container_type const& expected ) {
            container_type vae;
            vae.assign( il );
            ASSERT( vae == expected, "init list assign(begin,end) failed" );
        }
    };

    struct test_special_insert {
        template<typename container_type, typename element_type>
        static void do_test( std::initializer_list<element_type> il, container_type const& expected ) {
            container_type vd;
            vd.insert( il );
            ASSERT( vd == expected, "inserting with an initializer list failed" );
        }
    };

    template <typename container_type, typename test_assign, typename test_special>
    void TestInitListSupport(std::initializer_list<typename container_type::value_type> il){
        typedef typename container_type::value_type element_type;
        std::vector<element_type> test_seq(il.begin(),il.end());
        container_type expected(test_seq.begin(), test_seq.end());

        test_constructor<container_type,element_type>(il, expected);
        test_assignment_operator<container_type,element_type>(il, expected);
        test_assign::do_test(il, expected);
        test_special::do_test(il, expected);
    }

    template <typename container_type, typename test_special = skip_test>
    void TestInitListSupport(std::initializer_list<typename container_type::value_type> il) {
        TestInitListSupport<container_type, test_assign, test_special>(il);
    }

    template <typename container_type, typename test_special = skip_test>
    void TestInitListSupportWithoutAssign(std::initializer_list<typename container_type::value_type> il){
        TestInitListSupport<container_type, skip_test, test_special>(il);
    }

    //TODO: add test for no leaks, and correct element lifetime
    //the need for macro comes from desire to test different scenarios where initializer sequence is compile time constant
    #define __TBB_TEST_INIT_LIST_SUITE_SINGLE(FUNC_NAME, CONTAINER, ELEMENT_TYPE, INIT_SEQ)                                                           \
    void FUNC_NAME(){                                                                                                                                 \
        typedef ELEMENT_TYPE element_type;                                                                                                            \
        typedef CONTAINER<element_type> container_type;                                                                                               \
        element_type test_seq[] = INIT_SEQ;                                                                                                           \
        container_type expected(test_seq,test_seq + Harness::array_length(test_seq));                                                                 \
                                                                                                                                                      \
        /*test for explicit contructor call*/                                                                                                         \
        container_type vd INIT_SEQ;                                                                                                                   \
        ASSERT(vd == expected,"initialization via explicit constructor call with init list failed");                                                  \
        /*test for explicit contructor call with std::initializer_list*/                                                                              \
                                                                                                                                                      \
        std::initializer_list<element_type> init_list = INIT_SEQ;                                                                                     \
        container_type v1 (init_list);                                                                                                                \
        ASSERT(v1 == expected,"initialization via explicit constructor call with std::initializer_list failed");                                      \
                                                                                                                                                      \
        /*implicit constructor call test*/                                                                                                            \
        container_type v = INIT_SEQ;                                                                                                                  \
        ASSERT(v == expected,"init list constructor failed");                                                                                         \
                                                                                                                                                      \
        /*assignment operator test*/                                                                                                                  \
        /*TODO: count created and destroyed injects to assert that no extra copy of vector was created implicitly*/                                   \
        container_type va;                                                                                                                            \
        va = INIT_SEQ;                                                                                                                                \
        ASSERT(va == expected,"init list operator= failed");                                                                                          \
        /*assign(begin,end) test*/                                                                                                                    \
        container_type vae;                                                                                                                           \
        vae.assign(INIT_SEQ);                                                                                                                         \
        ASSERT(vae == expected,"init list assign(begin,end) failed");                                                                                 \
    }                                                                                                                                                 \

    namespace initializer_list_helpers{
        template<typename T>
        class ad_hoc_container{
            std::vector<T> vec;
            public:
            ad_hoc_container(){}
            template<typename InputIterator>
            ad_hoc_container(InputIterator begin, InputIterator end) : vec(begin,end) {}
            ad_hoc_container(std::initializer_list<T> il) : vec(il.begin(),il.end()) {}
            ad_hoc_container(ad_hoc_container const& other) : vec(other.vec) {}
            ad_hoc_container& operator=(ad_hoc_container const& rhs){ vec=rhs.vec; return *this;}
            ad_hoc_container& operator=(std::initializer_list<T> il){ vec.assign(il.begin(),il.end()); return *this;}
            template<typename InputIterator>
            void assign(InputIterator begin, InputIterator end){ vec.assign(begin,end);}
            void assign(std::initializer_list<T> il){ vec.assign(il.begin(),il.end());}
            friend bool operator==(ad_hoc_container<T> const& lhs, ad_hoc_container<T> const& rhs){ return lhs.vec==rhs.vec;}
        };
    }

    #define AD_HOC_INIT_SEQ {1,2,3,4}
    __TBB_TEST_INIT_LIST_SUITE_SINGLE(TestCompilerSupportInt, initializer_list_helpers::ad_hoc_container, int, AD_HOC_INIT_SEQ )
    #undef AD_HOC_INIT_SEQ

    #if __TBB_CPP11_INIT_LIST_TEST_BROKEN
        void TestCompilerSupportIntPair(){
            REPORT("Known issue: skip initializer_list compiler test for std::pair list elements.\n");
        }
    #else
        #define AD_HOC_PAIR_INIT_SEQ {{1,1}, {2,2},{3,3}, {4,4}}
        #define AD_HOC_INIT_SEQ_PAIR_TYPE std::pair<int,int>
        __TBB_TEST_INIT_LIST_SUITE_SINGLE(TestCompilerSupportIntPair, initializer_list_helpers::ad_hoc_container, AD_HOC_INIT_SEQ_PAIR_TYPE, AD_HOC_PAIR_INIT_SEQ )
        #undef AD_HOC_INIT_SEQ_PAIR_TYPE
        #undef AD_HOC_PAIR_INIT_SEQ
    #endif

    bool TestCompilerForInitializerList();
    namespace  {
        const bool conpiler_init_list_tests_are_run =  TestCompilerForInitializerList();
    }

    //TODO: move this to test_compiler
    bool TestCompilerForInitializerList(){
        TestCompilerSupportInt();
        TestCompilerSupportIntPair();
        tbb::internal::suppress_unused_warning(conpiler_init_list_tests_are_run);
        return true;
    }
} // namespace initializer_list_support_tests

#endif //__TBB_INITIALIZER_LISTS_PRESENT
#endif //__TBB_test_initializer_list_H
