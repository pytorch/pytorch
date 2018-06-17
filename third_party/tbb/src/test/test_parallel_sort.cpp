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

#include "tbb/parallel_sort.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/concurrent_vector.h"
#include "harness.h"
#include <math.h>
#include <vector>
#include <exception>
#include <algorithm>
#include <iterator>
#include <functional>
#include <string>
#include <cstring>

/** Has tightly controlled interface so that we can verify
    that parallel_sort uses only the required interface. */
class Minimal {
    int val;
public:
    Minimal() {}
    void set_val(int i) { val = i; }
    static bool CompareWith (const Minimal &a, const Minimal &b) {
        return (a.val < b.val);
    }
    static bool AreEqual( Minimal &a,  Minimal &b) {
       return a.val == b.val;
    }
};

//! Defines a comparison function object for Minimal
class MinimalCompare {
public:
    bool operator() (const Minimal &a, const Minimal &b) const {
        return Minimal::CompareWith(a,b);
    }
};

//! The default validate; but it uses operator== which is not required
template<typename RandomAccessIterator>
bool Validate(RandomAccessIterator a, RandomAccessIterator b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        ASSERT( a[i] == b[i], NULL );
    }
    return true;
}

//! A Validate specialized to string for debugging-only
template<>
bool Validate<std::string *>(std::string * a, std::string * b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if ( Verbose && a[i] != b[i]) {
          for (size_t j = 0; j < n; j++) {
              REPORT("a[%llu] == %s and b[%llu] == %s\n", static_cast<unsigned long long>(j), a[j].c_str(), static_cast<unsigned long long>(j), b[j].c_str());
          }
        }
        ASSERT( a[i] == b[i], NULL );
    }
    return true;
}

//! A Validate specialized to Minimal since it does not define an operator==
template<>
bool Validate<Minimal *>(Minimal *a, Minimal *b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        ASSERT( Minimal::AreEqual(a[i],b[i]), NULL );
    }
    return true;
}

//! A Validate specialized to concurrent_vector<Minimal> since it does not define an operator==
template<>
bool Validate<tbb::concurrent_vector<Minimal>::iterator>(tbb::concurrent_vector<Minimal>::iterator a,
                                                         tbb::concurrent_vector<Minimal>::iterator b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        ASSERT( Minimal::AreEqual(a[i],b[i]), NULL );
    }
    return true;
}

//! used in Verbose mode for identifying which data set is being used
static std::string test_type;

//! The default initialization routine.
/*! This routine assumes that you can assign to the elements from a float.
    It assumes that iter and sorted_list have already been allocated. It fills
    them according to the current data set (tracked by a local static variable).
    Returns true if a valid test has been setup, or false if there is no test to
    perform.
*/

template < typename RandomAccessIterator, typename Compare >
bool init_iter(RandomAccessIterator iter, RandomAccessIterator sorted_list, size_t n, const Compare &compare, bool reset) {
    static char test_case = 0;
    const char num_cases = 3;

    if (reset) test_case = 0;

    if (test_case < num_cases) {
        // switch on the current test case, filling the iter and sorted_list appropriately
        switch(test_case) {
            case 0:
                /* use sin to generate the values */
                test_type = "sin";
                for (size_t i = 0; i < n; i++)
                    iter[i] = sorted_list[i] = static_cast<typename std::iterator_traits< RandomAccessIterator >::value_type>(sin(float(i)));
                break;
            case 1:
                /* presorted list */
                test_type = "pre-sorted";
                for (size_t i = 0; i < n; i++)
                    iter[i] = sorted_list[i] = static_cast<typename std::iterator_traits< RandomAccessIterator >::value_type>(i);
                break;
            case 2:
                /* reverse-sorted list */
                test_type = "reverse-sorted";
                for (size_t i = 0; i < n; i++)
                    iter[i] = sorted_list[i] = static_cast<typename std::iterator_traits< RandomAccessIterator >::value_type>(n - i);
                break;
        }

        // pre-sort sorted_list for later validity testing
        std::sort(sorted_list, sorted_list + n, compare);
        test_case++;
        return true;
    }
    return false;
}

template < typename T, typename Compare >
bool init_iter(T * iter, T * sorted_list, size_t n, const Compare &compare, bool reset) {
    static char test_case = 0;
    const char num_cases = 3;

    if (reset) test_case = 0;

    if (test_case < num_cases) {
        // switch on the current test case, filling the iter and sorted_list appropriately
        switch(test_case) {
            case 0:
                /* use sin to generate the values */
                test_type = "sin";
                for (size_t i = 0; i < n; i++) {
                    iter[i] = T(sin(float(i)));
                    sorted_list[i] = T(sin(float(i)));
                }
                break;
            case 1:
                /* presorted list */
                test_type = "pre-sorted";
                for (size_t i = 0; i < n; i++) {
                    iter[i] = T(i);
                    sorted_list[i] = T(i);
                }
                break;
            case 2:
                /* reverse-sorted list */
                test_type = "reverse-sorted";
                for (size_t i = 0; i < n; i++) {
                    iter[i] = T(n - i);
                    sorted_list[i] = T(n - i);
                }
                break;
        }

        // pre-sort sorted_list for later validity testing
        std::sort(sorted_list, sorted_list + n, compare);
        test_case++;
        return true;
    }
    return false;
}


//! The initialization routine specialized to the class Minimal
/*! Minimal cannot have floats assigned to it.  This function uses the set_val method
*/

template < >
bool init_iter(Minimal* iter, Minimal * sorted_list, size_t n, const MinimalCompare &compare, bool reset) {
    static char test_case = 0;
    const char num_cases = 3;

    if (reset) test_case = 0;

    if (test_case < num_cases) {
        switch(test_case) {
            case 0:
                /* use sin to generate the values */
                test_type = "sin";
                for (size_t i = 0; i < n; i++) {
                    iter[i].set_val( int( sin( float(i) ) * 1000.f) );
                    sorted_list[i].set_val( int ( sin( float(i) ) * 1000.f) );
                }
                break;
            case 1:
                /* presorted list */
                test_type = "pre-sorted";
                for (size_t i = 0; i < n; i++) {
                    iter[i].set_val( int(i) );
                    sorted_list[i].set_val( int(i) );
                }
                break;
            case 2:
                /* reverse-sorted list */
                test_type = "reverse-sorted";
                for (size_t i = 0; i < n; i++) {
                    iter[i].set_val( int(n-i) );
                    sorted_list[i].set_val( int(n-i) );
                }
                break;
        }
        std::sort(sorted_list, sorted_list + n, compare);
        test_case++;
        return true;
    }
    return false;
}

//! The initialization routine specialized to the class concurrent_vector<Minimal>
/*! Minimal cannot have floats assigned to it.  This function uses the set_val method
*/

template < >
bool init_iter(tbb::concurrent_vector<Minimal>::iterator iter, tbb::concurrent_vector<Minimal>::iterator sorted_list,
               size_t n, const MinimalCompare &compare, bool reset) {
    static char test_case = 0;
    const char num_cases = 3;

    if (reset) test_case = 0;

    if (test_case < num_cases) {
        switch(test_case) {
            case 0:
                /* use sin to generate the values */
                test_type = "sin";
                for (size_t i = 0; i < n; i++) {
                    iter[i].set_val( int( sin( float(i) ) * 1000.f) );
                    sorted_list[i].set_val( int ( sin( float(i) ) * 1000.f) );
                }
                break;
            case 1:
                /* presorted list */
                test_type = "pre-sorted";
                for (size_t i = 0; i < n; i++) {
                    iter[i].set_val( int(i) );
                    sorted_list[i].set_val( int(i) );
                }
                break;
            case 2:
                /* reverse-sorted list */
                test_type = "reverse-sorted";
                for (size_t i = 0; i < n; i++) {
                    iter[i].set_val( int(n-i) );
                    sorted_list[i].set_val( int(n-i) );
                }
                break;
        }
        std::sort(sorted_list, sorted_list + n, compare);
        test_case++;
        return true;
    }
    return false;
}

//! The initialization routine specialized to the class string
/*! strings are created from floats.
*/

template<>
bool init_iter(std::string *iter, std::string *sorted_list, size_t n, const std::less<std::string> &compare, bool reset) {
    static char test_case = 0;
    const char num_cases = 1;

    if (reset) test_case = 0;

    if (test_case < num_cases) {
        switch(test_case) {
            case 0:
                /* use sin to generate the values */
                test_type = "sin";
                for (size_t i = 0; i < n; i++) {
                    char buffer[20];
// Getting rid of secure warning issued by VC 14 and newer
#if _MSC_VER && __STDC_SECURE_LIB__>=200411
                    sprintf_s(buffer, sizeof(buffer), "%f", float(sin(float(i))));
#else
                    sprintf(buffer, "%f", float(sin(float(i))));
#endif
                    sorted_list[i] = iter[i] = std::string(buffer);
                }
                break;
        }
        std::sort(sorted_list, sorted_list + n, compare);
        test_case++;
        return true;
    }
    return false;
}

//! The current number of threads in use (for Verbose only)
static size_t current_p;

//! The current data type being sorted (for Verbose only)
static std::string current_type;

//! The default test routine.
/*! Tests all data set sizes from 0 to N, all grainsizes from 0 to G=10, and selects from
    all possible interfaces to parallel_sort depending on whether a scratch space and
    compare have been provided.
*/
template<typename RandomAccessIterator, typename Compare>
bool parallel_sortTest(size_t n, RandomAccessIterator iter, RandomAccessIterator sorted_list, const Compare *comp) {
    bool passed = true;

    Compare local_comp;

    init_iter(iter, sorted_list, n, local_comp, true);
    do {
        REMARK("%s %s p=%llu n=%llu :",current_type.c_str(), test_type.c_str(),
                   static_cast<unsigned long long>(current_p), static_cast<unsigned long long>(n));
        if (comp != NULL) {
            tbb::parallel_sort(iter, iter + n, local_comp );
         } else {
            tbb::parallel_sort(iter, iter + n );
         }
        if (!Validate(iter, sorted_list, n))
            passed = false;
        REMARK("passed\n");
    } while (init_iter(iter, sorted_list, n, local_comp, false));
    return passed;
}

//! The test routine specialize to Minimal, since it does not have a less defined for it
template<>
bool parallel_sortTest(size_t n, Minimal * iter, Minimal * sorted_list, const MinimalCompare *compare) {
    bool passed = true;

    if (compare == NULL) return passed;

    init_iter(iter, sorted_list, n, *compare, true);
    do {
        REMARK("%s %s p=%llu n=%llu :",current_type.c_str(), test_type.c_str(),
                    static_cast<unsigned long long>(current_p), static_cast<unsigned long long>(n));

        tbb::parallel_sort(iter, iter + n, *compare );

        if (!Validate(iter, sorted_list, n))
            passed = false;
        REMARK("passed\n");
    } while (init_iter(iter, sorted_list, n, *compare, false));
    return passed;
}

//! The test routine specialize to concurrent_vector of Minimal, since it does not have a less defined for it
template<>
bool parallel_sortTest(size_t n, tbb::concurrent_vector<Minimal>::iterator iter,
                       tbb::concurrent_vector<Minimal>::iterator sorted_list, const MinimalCompare *compare) {
    bool passed = true;

    if (compare == NULL) return passed;

    init_iter(iter, sorted_list, n, *compare, true);
    do {
        REMARK("%s %s p=%llu n=%llu :",current_type.c_str(), test_type.c_str(),
                    static_cast<unsigned long long>(current_p), static_cast<unsigned long long>(n));

        tbb::parallel_sort(iter, iter + n, *compare );

        if (!Validate(iter, sorted_list, n))
            passed = false;
        REMARK("passed\n");
    } while (init_iter(iter, sorted_list, n, *compare, false));
    return passed;
}

//! The main driver for the tests.
/*! Minimal, float and string types are used.  All interfaces to parallel_sort that are usable
    by each type are tested.
*/
void Flog() {
    // For each type create:
    // the list to be sorted by parallel_sort (array)
    // the list to be sort by STL sort (array_2)
    // and a less function object

    const size_t N = 50000;

    Minimal *minimal_array = new Minimal[N];
    Minimal *minimal_array_2 = new Minimal[N];
    MinimalCompare minimal_less;

    float *float_array = new float[N];
    float *float_array_2 = new float[N];
    std::less<float> float_less;

    tbb::concurrent_vector<float> float_cv1;
    tbb::concurrent_vector<float> float_cv2;
    float_cv1.grow_to_at_least(N);
    float_cv2.grow_to_at_least(N);

    std::string *string_array = new std::string[N];
    std::string *string_array_2 = new std::string[N];
    std::less<std::string> string_less;

    tbb::concurrent_vector<Minimal> minimal_cv1;
    tbb::concurrent_vector<Minimal> minimal_cv2;
    minimal_cv1.grow_to_at_least(N);
    minimal_cv2.grow_to_at_least(N);


    // run the appropriate tests for each type

    current_type = "Minimal(less)";
    parallel_sortTest(0, minimal_array, minimal_array_2, &minimal_less);
    parallel_sortTest(1, minimal_array, minimal_array_2, &minimal_less);
    parallel_sortTest(10, minimal_array, minimal_array_2, &minimal_less);
    parallel_sortTest(9999, minimal_array, minimal_array_2, &minimal_less);
    parallel_sortTest(50000, minimal_array, minimal_array_2, &minimal_less);

    current_type = "float (no less)";
    parallel_sortTest(0, float_array, float_array_2, static_cast<std::less<float> *>(NULL));
    parallel_sortTest(1, float_array, float_array_2, static_cast<std::less<float> *>(NULL));
    parallel_sortTest(10, float_array, float_array_2, static_cast<std::less<float> *>(NULL));
    parallel_sortTest(9999, float_array, float_array_2, static_cast<std::less<float> *>(NULL));
    parallel_sortTest(50000, float_array, float_array_2, static_cast<std::less<float> *>(NULL));

    current_type = "float (less)";
    parallel_sortTest(0, float_array, float_array_2, &float_less);
    parallel_sortTest(1, float_array, float_array_2, &float_less);
    parallel_sortTest(10, float_array, float_array_2, &float_less);
    parallel_sortTest(9999, float_array, float_array_2, &float_less);
    parallel_sortTest(50000, float_array, float_array_2, &float_less);

    current_type = "concurrent_vector<float> (no less)";
    parallel_sortTest(0, float_cv1.begin(), float_cv2.begin(), static_cast<std::less<float> *>(NULL));
    parallel_sortTest(1, float_cv1.begin(), float_cv2.begin(), static_cast<std::less<float> *>(NULL));
    parallel_sortTest(10, float_cv1.begin(), float_cv2.begin(), static_cast<std::less<float> *>(NULL));
    parallel_sortTest(9999, float_cv1.begin(), float_cv2.begin(), static_cast<std::less<float> *>(NULL));
    parallel_sortTest(50000, float_cv1.begin(), float_cv2.begin(), static_cast<std::less<float> *>(NULL));

    current_type = "concurrent_vector<float> (less)";
    parallel_sortTest(0, float_cv1.begin(), float_cv2.begin(), &float_less);
    parallel_sortTest(1, float_cv1.begin(), float_cv2.begin(), &float_less);
    parallel_sortTest(10, float_cv1.begin(), float_cv2.begin(), &float_less);
    parallel_sortTest(9999, float_cv1.begin(), float_cv2.begin(), &float_less);
    parallel_sortTest(50000, float_cv1.begin(), float_cv2.begin(), &float_less);

    current_type = "string (no less)";
    parallel_sortTest(0, string_array, string_array_2, static_cast<std::less<std::string> *>(NULL));
    parallel_sortTest(1, string_array, string_array_2, static_cast<std::less<std::string> *>(NULL));
    parallel_sortTest(10, string_array, string_array_2, static_cast<std::less<std::string> *>(NULL));
    parallel_sortTest(9999, string_array, string_array_2, static_cast<std::less<std::string> *>(NULL));
    parallel_sortTest(50000, string_array, string_array_2, static_cast<std::less<std::string> *>(NULL));

    current_type = "string (less)";
    parallel_sortTest(0, string_array, string_array_2, &string_less);
    parallel_sortTest(1, string_array, string_array_2, &string_less);
    parallel_sortTest(10, string_array, string_array_2, &string_less);
    parallel_sortTest(9999, string_array, string_array_2, &string_less);
    parallel_sortTest(50000, string_array, string_array_2, &string_less);

    current_type = "concurrent_vector<Minimal> (less)";
    parallel_sortTest(0, minimal_cv1.begin(), minimal_cv2.begin(), &minimal_less);
    parallel_sortTest(1, minimal_cv1.begin(), minimal_cv2.begin(), &minimal_less);
    parallel_sortTest(10, minimal_cv1.begin(), minimal_cv2.begin(), &minimal_less);
    parallel_sortTest(9999, minimal_cv1.begin(), minimal_cv2.begin(), &minimal_less);
    parallel_sortTest(50000, minimal_cv1.begin(), minimal_cv2.begin(), &minimal_less);

    delete [] minimal_array;
    delete [] minimal_array_2;

    delete [] float_array;
    delete [] float_array_2;

    delete [] string_array;
    delete [] string_array_2;
}

const int elements = 10000;

void rand_vec(std::vector<int> &v) {
    for (int i=0; i<elements; ++i) {
        (v.push_back(rand()%elements*10));
    }
}

void range_sort_test() {
    std::vector<int> v;

    typedef std::vector<int>::iterator itor;
    // iterator checks
    rand_vec(v);
    tbb::parallel_sort(v.begin(), v.end());
    for(itor a=v.begin(); a<v.end()-1; ++a) ASSERT(*a <= *(a+1), "v not sorted");
    v.clear();

    rand_vec(v);
    tbb::parallel_sort(v.begin(), v.end(), std::greater<int>());
    for(itor a=v.begin(); a<v.end()-1; ++a) ASSERT(*a >= *(a+1), "v not sorted");
    v.clear();

    // range checks
    rand_vec(v);
    tbb::parallel_sort(v);
    for(itor a=v.begin(); a<v.end()-1; ++a) ASSERT(*a <= *(a+1), "v not sorted");
    v.clear();

    rand_vec(v);
    tbb::parallel_sort(v, std::greater<int>());
    for(itor a=v.begin(); a<v.end()-1; ++a) ASSERT(*a >= *(a+1), "v not sorted");
    v.clear();

    // const range checks
    rand_vec(v);
    tbb::parallel_sort(tbb::blocked_range<std::vector<int>::iterator>(v.begin(), v.end()));
    for(itor a=v.begin(); a<v.end()-1; ++a) ASSERT(*a <= *(a+1), "v not sorted");
    v.clear();

    rand_vec(v);
    tbb::parallel_sort(tbb::blocked_range<std::vector<int>::iterator>(v.begin(), v.end()), std::greater<int>());
    for(itor a=v.begin(); a<v.end()-1; ++a) ASSERT(*a >= *(a+1), "v not sorted");
    v.clear();

    // array tests
    int arr[elements];
    for(int i=0; i<elements; ++i) arr[i] = rand()%(elements*10);
    tbb::parallel_sort(arr);
    for(int i=0; i<elements-1; ++i) ASSERT(arr[i] <= arr[i+1], "arr not sorted");
}

#include <cstdio>
#include "harness_cpu.h"

int TestMain () {
    if( MinThread<1 ) {
        REPORT("Usage: number of threads must be positive\n");
        exit(1);
    }
    for( int p=MinThread; p<=MaxThread; ++p ) {
        if( p>0 ) {
            tbb::task_scheduler_init init( p );
            current_p = p;
            Flog();
            range_sort_test();

            // Test that all workers sleep when no work
            TestCPUUserTime(p);
        }
    }
    return Harness::Done;
}

