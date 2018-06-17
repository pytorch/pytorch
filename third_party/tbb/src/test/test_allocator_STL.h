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

// Tests for compatibility with the host's STL.

#include "harness.h"

template<typename Container>
void TestSequence(const typename Container::allocator_type &a) {
    Container c(a);
    for( int i=0; i<1000; ++i )
        c.push_back(i*i);
    typename Container::const_iterator p = c.begin();
    for( int i=0; i<1000; ++i ) {
        ASSERT( *p==i*i, NULL );
        ++p;
    }
    // regression test against compilation error for GCC 4.6.2
    c.resize(1000);
}

template<typename Set>
void TestSet(const typename Set::allocator_type &a) {
    Set s(typename Set::key_compare(), a);
    typedef typename Set::value_type value_type;
    for( int i=0; i<100; ++i )
        s.insert(value_type(3*i));
    for( int i=0; i<300; ++i ) {
        ASSERT( s.erase(i)==size_t(i%3==0), NULL );
    }
}

template<typename Map>
void TestMap(const typename Map::allocator_type &a) {
    Map m(typename Map::key_compare(), a);
    typedef typename Map::value_type value_type;
    for( int i=0; i<100; ++i )
        m.insert(value_type(i,i*i));
    for( int i=0; i<100; ++i )
        ASSERT( m.find(i)->second==i*i, NULL );
}

#include <deque>
#include <list>
#include <map>
#include <set>
#include <vector>

#if __TBB_CPP11_RVALUE_REF_PRESENT
struct MoveOperationTracker {
    int my_value;

    MoveOperationTracker( int value = 0 ) : my_value( value ) {}
    MoveOperationTracker(const MoveOperationTracker&) {
        ASSERT( false, "Copy constructor is called" );
    }
    MoveOperationTracker(MoveOperationTracker&& m) __TBB_NOEXCEPT( true ) : my_value( m.my_value ) {
    }
    MoveOperationTracker& operator=(MoveOperationTracker const&) {
        ASSERT( false, "Copy assigment operator is called" );
        return *this;
    }
    MoveOperationTracker& operator=(MoveOperationTracker&& m) __TBB_NOEXCEPT( true ) {
        my_value = m.my_value;
        return *this;
    }

    bool operator==(int value) const {
        return my_value == value;
    }

    bool operator==(const MoveOperationTracker& m) const {
        return my_value == m.my_value;
    }
};
#endif /*  __TBB_CPP11_RVALUE_REF_PRESENT */

template<typename Allocator>
void TestAllocatorWithSTL(const Allocator &a = Allocator() ) {
    typedef typename Allocator::template rebind<int>::other Ai;
    typedef typename Allocator::template rebind<std::pair<const int, int> >::other Acii;
#if _MSC_VER
    typedef typename Allocator::template rebind<const int>::other Aci;
    typedef typename Allocator::template rebind<std::pair<int, int> >::other Aii;
#endif

    // Sequenced containers
    TestSequence<std::deque <int,Ai> >(a);
    TestSequence<std::list  <int,Ai> >(a);
    TestSequence<std::vector<int,Ai> >(a);

#if __TBB_CPP11_RVALUE_REF_PRESENT
    typedef typename Allocator::template rebind<MoveOperationTracker>::other Amot;
    TestSequence<std::deque <MoveOperationTracker, Amot> >(a);
    TestSequence<std::list  <MoveOperationTracker, Amot> >(a);
    TestSequence<std::vector<MoveOperationTracker, Amot> >(a);
#endif

    // Associative containers
    TestSet<std::set     <int, std::less<int>, Ai> >(a);
    TestSet<std::multiset<int, std::less<int>, Ai> >(a);
    TestMap<std::map     <int, int, std::less<int>, Acii> >(a);
    TestMap<std::multimap<int, int, std::less<int>, Acii> >(a);

#if _MSC_VER && _CPPLIB_VER < 650
    // Test compatibility with Microsoft's implementation of std::allocator for some cases that
    // are undefined according to the ISO standard but permitted by Microsoft.
    TestSequence<std::deque <const int,Aci> >(a);
#if _CPPLIB_VER>=500
    TestSequence<std::list  <const int,Aci> >(a);
#endif
    TestSequence<std::vector<const int,Aci> >(a);
    TestSet<std::set<const int, std::less<int>, Aci> >(a);
    TestMap<std::map<int, int, std::less<int>, Aii> >(a);
    TestMap<std::map<const int, int, std::less<int>, Acii> >(a);
    TestMap<std::multimap<int, int, std::less<int>, Aii> >(a);
    TestMap<std::multimap<const int, int, std::less<int>, Acii> >(a);
#endif /* _MSC_VER */
}
