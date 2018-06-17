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

#define _VARIADIC_MAX 10   // Visual Studio 2012
#include "harness.h"
#include "tbb/atomic.h"
#include "harness_checktype.h"

#include "tbb/flow_graph.h"
#include <cstdio>
#include <stdexcept>
#include <vector>

#if __TBB_GCC_STRICT_ALIASING_BROKEN
    #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

// given a tuple, return the type of the element that has the maximum alignment requirement.
// Given a tuple and that type, return the number of elements of the object with the max
// alignment requirement that is at least as big as the largest object in the tuple.

using tbb::flow::tuple_element;
using tbb::flow::tuple_size;
using tbb::flow::cast_to;
using tbb::flow::is_a;

typedef int *int_ptr;
typedef char odd_array_type[15];
typedef char odder_array[17];
typedef check_type<int> counted_array_type[12];
typedef std::vector<double> d_vector;
typedef std::vector<int> i_vector;
typedef i_vector i_vector_array[2];
typedef tbb::flow::tagged_msg<size_t, int, char, double, odd_array_type, odder_array, d_vector, check_type<int>, counted_array_type, i_vector_array> tagged_msg_type;

// test base of tagged_msg
void TestWrapper() {
    using tbb::flow::interface10::internal::Wrapper;
    Wrapper<int> wi(42);
    Wrapper<int> wic(23);

    REMARK("Value of wic is %d\n", wic.value());

    // pointer-type creation
    int point_to_me = 23;
    Wrapper<int_ptr> wip(&point_to_me);
    ASSERT(*(wip.value()) == 23, "Error in wip value");

    odd_array_type ww;
    for(int ii = 0; ii < 15; ++ii) { ww[ii] = char('0' + ii); } ww[14] = 0;

    Wrapper<odd_array_type> ci(ww);
    ASSERT(!strncmp(ci.value(), ww, 14), "odd_array_type ci not properly-constructed" );

    Wrapper<odd_array_type> ci2(ci);

    ASSERT(!strncmp(ci2.value(), ww, 14), "odd_array_type ci2 not properly-constructed" );

    d_vector di;
    di.clear();
    di.push_back(2.0);
    Wrapper<d_vector> dvec(di);
    ASSERT(dvec.value()[0] == 2.0, "incorrect value in vector");

    // test array of non-PODs.
    i_vector_array oia;
    oia[0].clear();
    oia[1].clear();
    oia[0].push_back(3);
    oia[1].push_back(2);
    Wrapper<i_vector_array> ia(oia);
    ASSERT((ia.value()[1])[0] == 2, "integer vector array element[1] misbehaved");
    ASSERT((ia.value()[0])[0] == 3, "integer vector array element[0] misbehaved");
    Wrapper<i_vector_array> iac(ia);
    ASSERT((iac.value()[1])[0] == 2, "integer vector array element[1] misbehaved");
    ASSERT((iac.value()[0])[0] == 3, "integer vector array element[0] misbehaved");

    // counted_array
    counted_array_type cat_orig;
    for(int i = 0; i < 12; ++i) cat_orig[i] = i + 1;
    Wrapper<counted_array_type> cat(cat_orig);
    for(int j = 0; j < 12; ++j)
        ASSERT(1 + j == cat.value()[j], "Error in cat array");

    int i = wi.value();
    ASSERT(i == 42, "Assignment to i failed");
    ASSERT(wi.value() == 42, "Assignment to wi failed");
    double d = wi.value();
    ASSERT(d == 42, "Implicit cast in assign to double failed");
    int_ptr ip = wip.value();
    ASSERT(ip == &(point_to_me), "Error in assignment of pointer");
}

void RunTests() {
    tagged_msg_type def;
    tagged_msg_type i(1,3);
    check_type<int>::check_type_counter = 0;
    int z;
    #if TBB_USE_EXCEPTIONS
    try {
        z = cast_to<int>(def); // disallowed (non-array returning int)
        ASSERT(false, "should not allow cast to int of non-array");
    }
    catch(...) {
        REMARK("cast of non-array to int disallowed (okay)\n");
    }
    #endif
    z = cast_to<int>(i);
    ASSERT(is_a<int>(i), "wrong type for i ( == int)");
    ASSERT(!(is_a<double>(i)), "Wrong type for i ( != double)");
    z = 5;
    z = cast_to<int>(i);

    const int &ref_i(cast_to<int>(i));
    ASSERT(ref_i == 3, "ref_i got wrong value");
    tagged_msg_type j(2,4);
    i = j;
    ASSERT(ref_i == 4, "assign to i did not affect ref_i");

    ASSERT( z == 3, "Error retrieving value from i");

    //updating and retrieving tags
    ASSERT(j.tag() == 2, "Error retrieving tag for j");
    j.set_tag(10);
    ASSERT(j.tag() == 10, "Error updating tag for j");

    tbb::flow::tagged_msg<char, int, char, double> k('a', 4);
    k.set_tag('b');
    ASSERT(k.tag() == 'b', "Error updating char tag");

    tagged_msg_type double_tagged_msg(3, 8.0);
    ASSERT(is_a<double>(double_tagged_msg), "Wrong type for double_tagged_msg (== double)");
    ASSERT(!is_a<char>(double_tagged_msg), "Wrong type for double_tagged_msg (!= char)");
    ASSERT(!is_a<int>(double_tagged_msg), "Wrong type for double_tagged_msg (!= int)");
    tagged_msg_type copytype(double_tagged_msg);
    ASSERT(is_a<double>(copytype), "Wrong type for double_tagged_msg (== double)");
    ASSERT(!is_a<char>(copytype), "Wrong type for double_tagged_msg (!= char)");
    ASSERT(!is_a<int>(copytype), "Wrong type for double_tagged_msg (!= int)");
    tagged_msg_type default_tagged_msg;
    ASSERT(!(is_a<double>(default_tagged_msg)), "wrong type for default ( != double)");
    ASSERT(!(is_a<int>(default_tagged_msg)), "wrong type for default ( != int)");
    ASSERT(!(is_a<bool>(default_tagged_msg)), "wrong type for default ( != bool)");
    check_type<int> c;
    ASSERT(check_type<int>::check_type_counter == 1, "Incorrect number of check_type<int>s created");
    tagged_msg_type cnt_type(4, c);
    ASSERT(check_type<int>::check_type_counter == 2, "Incorrect number of check_type<int>s created");
    ASSERT(is_a<check_type<int> >(cnt_type), "Incorrect type for cnt_type");
    cnt_type = default_tagged_msg;
    ASSERT(check_type<int>::check_type_counter == 1, "Incorrect number of check_type<int>s after reassignment");
    ASSERT(cnt_type.is_default_constructed(), "Assigned check_type<int>s is not default-constructed");
    // having problem with init on gcc 3.4.6 (fxeolin16)  constructor for elements of array not called
    // for this version.
    // counted_array_type counted_array;
    check_type<int> counted_array[12];  // this is okay
    ASSERT(check_type<int>::check_type_counter == 13, "Incorrect number of check_type<int>s after counted_array construction");
    tagged_msg_type counted_array_tagged_msg(5, counted_array);
    // the is_a<>() should return exact type matches.
    ASSERT(!is_a<check_type<int> *>(counted_array_tagged_msg), "Test of is_a for counted_array_tagged_msg fails");
    #if TBB_USE_EXCEPTIONS
    try {
        int *iip = cast_to<int *>(counted_array_tagged_msg);
        ASSERT(false, "did not throw on invalid cast");
        *iip = 2;  // avoids "ipp set but not used" warning
    }
    catch(std::runtime_error &re) {
        REMARK("attempt to cast to invalid type caught %s\n", re.what());
    }
    ASSERT(is_a<counted_array_type>(counted_array_tagged_msg), "testing");
    const check_type<int> *ctip = cast_to<counted_array_type>(counted_array_tagged_msg);

    ASSERT((int)(*ctip) == 0, "ctip incorrect");

    ASSERT(check_type<int>::check_type_counter == 25, "Incorrect number of check_type<int>s after counted_array_tagged_msg construction");
    counted_array_tagged_msg = default_tagged_msg;
    ASSERT(check_type<int>::check_type_counter == 13, "Incorrect number of check_type<int>s after counted_array_tagged_msg destruction");
    ASSERT(counted_array_tagged_msg.is_default_constructed(), "Assigned counted_array_type is not default-constructed");

    default_tagged_msg = double_tagged_msg;
    const double my_dval = cast_to<double>(default_tagged_msg);
    ASSERT(my_dval == 8.0, "did not retrieve correct value from assigned default_tagged_msg");

    {
        odd_array_type my_b;
        for(size_t ii=0; ii < 14;++ii) {
            my_b[ii] = (char)('0' + ii);
        }
        my_b[14] = 0;
        {
            tagged_msg_type odd_array_tagged_msg(6, my_b);
            const char *my_copy = cast_to<odd_array_type>(odd_array_tagged_msg);
            ASSERT(!strncmp(my_b, my_copy, 14), "copied char array not correct value");
            default_tagged_msg = odd_array_tagged_msg;
            try {
                const char *my_copy2 = cast_to<odd_array_type>(default_tagged_msg);
                ASSERT(!strncmp(my_b, my_copy2, 14), "char array from default tagged_msg assign not correct value");
            }
            catch(...) {
                ASSERT(false, "Bad cast");
            }
        }
    }

    ASSERT(!is_a<double>(i), "bad type for i");
    try {
        double y = cast_to<double>(i);
        // use '&' to force eval of RHS (fixes "initialized but not referenced" vs2012 warnings)
        ASSERT(false & (0 != y), "Error: cast to type in tuple did not get exception");
    }
    catch(std::runtime_error &bc) {
        ASSERT(0 == strcmp(bc.what(), "Illegal tagged_msg cast"), "Incorrect std:runtime_error");
    }
    catch(...) {
        ASSERT(false & cast_to<int>(i), "Error: improper exception thrown");
    }

    try {
        int *ip = cast_to<int *>(i);
        ASSERT(false & (NULL!=ip), "Error: non-array cast to pointer type.");
    }
    catch(std::runtime_error &bc) {
        ASSERT(0 == strcmp(bc.what(), "Illegal tagged_msg cast"), "Incorrect std:runtime_error");
    }
    catch(...) {
        ASSERT(false, "did not get runtime_error exception in casting non-array to pointer");
    }

    try {
        bool b = cast_to<bool>(i);
        ASSERT(false & b, "Error: cast against type did not get exception");
    }
    catch(std::runtime_error &bc) {
        ASSERT(0 == strcmp(bc.what(), "Illegal tagged_msg cast"), "Incorrect std:runtime_error");
    }
    catch(...) {
        ASSERT(false, "did not get runtime_error exception casting to disparate types");
    }
    #endif //TBB_USE_EXCEPTIONS
}

int TestMain() {
    TestWrapper();
    ASSERT(check_type<int>::check_type_counter == 0, "After TestWrapper return not all check_type<int>s were destroyed");
    RunTests();
    ASSERT(check_type<int>::check_type_counter == 0, "After RunTests return not all check_type<int>s were destroyed");
    return Harness::Done;
}
