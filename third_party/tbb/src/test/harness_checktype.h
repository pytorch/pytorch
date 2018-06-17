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

#ifndef tbb_tests_harness_checktype_H
#define tbb_tests_harness_checktype_H

// type that checks construction and destruction.

#ifndef __HARNESS_CHECKTYPE_DEFAULT_CTOR
    #define __HARNESS_CHECKTYPE_DEFAULT_CTOR 1
#endif

template<class Counter>
class check_type : Harness::NoAfterlife {
    Counter id;
    bool am_ready;
public:
    static tbb::atomic<int> check_type_counter;
    // if only non-default constructors are desired, set __HARNESS_CHECKTYPE_NODEFAULT_CTOR
    check_type(Counter _n
#if __HARNESS_CHECKTYPE_DEFAULT_CTOR
            = 0
#endif
            ) : id(_n), am_ready(false) {
        ++check_type_counter;
    }

    check_type(const check_type& other) : Harness::NoAfterlife(other) {
        other.AssertLive();
        AssertLive();
        id = other.id;
        am_ready = other.am_ready;
        ++check_type_counter;
    }

    operator int() const { return (int)my_id(); }
    check_type& operator++() { ++id; return *this;; }

    ~check_type() {
        AssertLive();
        --check_type_counter;
        ASSERT(check_type_counter >= 0, "too many destructions");
    }

    check_type &operator=(const check_type &other) {
        other.AssertLive();
        AssertLive();
        id = other.id;
        am_ready = other.am_ready;
        return *this;
    }

    Counter my_id() const { AssertLive(); return id; }
    bool is_ready() { AssertLive(); return am_ready; }
    void function() {
        AssertLive();
        if( id == (Counter)0 ) {
            id = (Counter)1;
            am_ready = true;
        }
    }

};

template<class Counter>
tbb::atomic<int> check_type<Counter>::check_type_counter;

// provide a class that for a check_type will initialize the counter on creation, and on
// destruction will check that the constructions and destructions of check_type match.
template<class MyClass>
struct Check {
    Check() {}   // creation does nothing
    ~Check() {}  // destruction checks nothing
};

template<class Counttype>
struct Check<check_type< Counttype > > {
    Check() { check_type<Counttype>::check_type_counter = 0; }
    ~Check() { ASSERT(check_type<Counttype>::check_type_counter == 0, "check_type constructions and destructions don't match"); }
};

#endif  // tbb_tests_harness_checktype_H
