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

/* Some tests in this source file are based on PPL tests provided by Microsoft. */
#include "tbb/parallel_for.h"
#include "tbb/tick_count.h"
#include "harness.h"
#include "test_container_move_support.h"
// Test that unordered containers do not require keys have default constructors.
#define __HARNESS_CHECKTYPE_DEFAULT_CTOR 0
#include "harness_checktype.h"
#undef  __HARNESS_CHECKTYPE_DEFAULT_CTOR
#include "harness_allocator.h"

template<typename T>
struct degenerate_hash {
    size_t operator()(const T& /*a*/) const {
        return 1;
    }
};

// TestInitListSupportWithoutAssign with an empty initializer list causes internal error in Intel Compiler.
#define __TBB_ICC_EMPTY_INIT_LIST_TESTS_BROKEN (__INTEL_COMPILER && __INTEL_COMPILER <= 1500)

typedef local_counting_allocator<debug_allocator<std::pair<const int,int>,std::allocator> > MyAllocator;

#define CheckAllocatorE(t,a,f) CheckAllocator(t,a,f,true,__LINE__)
#define CheckAllocatorA(t,a,f) CheckAllocator(t,a,f,false,__LINE__)
template<typename MyTable>
inline void CheckAllocator(MyTable &table, size_t expected_allocs, size_t expected_frees, bool exact = true, int line = 0) {
    typename MyTable::allocator_type a = table.get_allocator();
    REMARK("#%d checking allocators: items %u/%u, allocs %u/%u\n", line,
        unsigned(a.items_allocated), unsigned(a.items_freed), unsigned(a.allocations), unsigned(a.frees) );
    ASSERT( a.items_allocated == a.allocations, NULL); ASSERT( a.items_freed == a.frees, NULL);
    if(exact) {
        ASSERT( a.allocations == expected_allocs, NULL); ASSERT( a.frees == expected_frees, NULL);
    } else {
        ASSERT( a.allocations >= expected_allocs, NULL); ASSERT( a.frees >= expected_frees, NULL);
        ASSERT( a.allocations - a.frees == expected_allocs - expected_frees, NULL );
    }
}

template<typename T>
struct strip_const { typedef T type; };

template<typename T>
struct strip_const<const T> { typedef T type; };

// value generator for cumap
template <typename K, typename V = std::pair<const K, K> >
struct ValueFactory {
    typedef typename strip_const<K>::type Kstrip;
    static V make(const K &value) { return V(value, value); }
    static Kstrip key(const V &value) { return value.first; }
    static Kstrip get(const V &value) { return (Kstrip)value.second; }
    template< typename U >
    static U convert(const V &value) { return U(value.second); }
};

// generator for cuset
template <typename T>
struct ValueFactory<T, T> {
    static T make(const T &value) { return value; }
    static T key(const T &value) { return value; }
    static T get(const T &value) { return value; }
    template< typename U >
    static U convert(const T &value) { return U(value); }
};

template <typename T>
struct Value : ValueFactory<typename T::key_type, typename T::value_type> {
    template<typename U>
    static bool compare( const typename T::iterator& it, U val ) {
        return (Value::template convert<U>(*it) == val);
    }
};

#if _MSC_VER
#pragma warning(disable: 4189) // warning 4189 -- local variable is initialized but not referenced
#pragma warning(disable: 4127) // warning 4127 -- while (true) has a constant expression in it
#endif

template<typename ContainerType, typename Iterator, typename RangeType>
std::pair<intptr_t,intptr_t> CheckRecursiveRange(RangeType range) {
    std::pair<intptr_t,intptr_t> sum(0, 0); // count, sum
    for( Iterator i = range.begin(), e = range.end(); i != e; ++i ) {
        ++sum.first; sum.second += Value<ContainerType>::get(*i);
    }
    if( range.is_divisible() ) {
        RangeType range2( range, tbb::split() );
        std::pair<intptr_t,intptr_t> sum1 = CheckRecursiveRange<ContainerType,Iterator, RangeType>( range );
        std::pair<intptr_t,intptr_t> sum2 = CheckRecursiveRange<ContainerType,Iterator, RangeType>( range2 );
        sum1.first += sum2.first; sum1.second += sum2.second;
        ASSERT( sum == sum1, "Mismatched ranges after division");
    }
    return sum;
}

template <typename T>
struct SpecialTests {
    static void Test(const char *str) {REMARK("skipped -- specialized %s tests\n", str);}
};

#if __TBB_INITIALIZER_LISTS_PRESENT
template<typename container_type>
bool equal_containers( container_type const& lhs, container_type const& rhs ) {
    if ( lhs.size() != rhs.size() ) {
        return false;
    }
    return std::equal( lhs.begin(), lhs.end(), rhs.begin(), Harness::IsEqual() );
}

#include "test_initializer_list.h"

template <typename Table, typename MultiTable>
void TestInitList( std::initializer_list<typename Table::value_type> il ) {
    using namespace initializer_list_support_tests;
    REMARK("testing initializer_list methods \n");

    TestInitListSupportWithoutAssign<Table,test_special_insert>(il);
    TestInitListSupportWithoutAssign<MultiTable, test_special_insert>( il );

#if __TBB_ICC_EMPTY_INIT_LIST_TESTS_BROKEN
    REPORT( "Known issue: TestInitListSupportWithoutAssign with an empty initializer list is skipped.\n");
#else
    TestInitListSupportWithoutAssign<Table, test_special_insert>( {} );
    TestInitListSupportWithoutAssign<MultiTable, test_special_insert>( {} );
#endif
}
#endif //if __TBB_INITIALIZER_LISTS_PRESENT

template<Harness::StateTrackableBase::StateValue desired_state, typename T>
void check_value_state(/* typename do_check_element_state =*/ tbb::internal::true_type, T const& t, const char* filename, int line )
{
    ASSERT_CUSTOM(is_state_f<desired_state>()(t), "", filename, line);
}

template<Harness::StateTrackableBase::StateValue desired_state, typename T>
void check_value_state(/* typename do_check_element_state =*/ tbb::internal::false_type, T const&, const char* , int ) {/*do nothing*/}
#define ASSERT_VALUE_STATE(do_check_element_state,state,value) check_value_state<state>(do_check_element_state,value,__FILE__,__LINE__)

#if __TBB_CPP11_RVALUE_REF_PRESENT
template<typename T, typename do_check_element_state, typename V>
void test_rvalue_insert(V v1, V v2)
{
    typedef T container_t;

    container_t cont;

    std::pair<typename container_t::iterator, bool> ins = cont.insert(Value<container_t>::make(v1));
    ASSERT(ins.second == true && Value<container_t>::get(*(ins.first)) == v1, "Element 1 has not been inserted properly");
    ASSERT_VALUE_STATE(do_check_element_state(),Harness::StateTrackableBase::MoveInitialized,*ins.first);

    typename container_t::iterator it2 = cont.insert(ins.first, Value<container_t>::make(v2));
    ASSERT(Value<container_t>::get(*(it2)) == v2, "Element 2 has not been inserted properly");
    ASSERT_VALUE_STATE(do_check_element_state(),Harness::StateTrackableBase::MoveInitialized,*it2);

}
#if __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT
// The test does not use variadic templates, but emplace() does.

namespace emplace_helpers {
template<typename container_t, typename arg_t, typename value_t>
std::pair<typename container_t::iterator, bool> call_emplace_impl(container_t& c, arg_t&& k, value_t *){
    // this is a set
    return c.emplace(std::forward<arg_t>(k));
}

template<typename container_t, typename arg_t, typename first_t, typename second_t>
std::pair<typename container_t::iterator, bool> call_emplace_impl(container_t& c, arg_t&& k, std::pair<first_t, second_t> *){
    // this is a map
    return c.emplace(k, std::forward<arg_t>(k));
}

template<typename container_t, typename arg_t>
std::pair<typename container_t::iterator, bool> call_emplace(container_t& c, arg_t&& k){
    typename container_t::value_type * selector = NULL;
    return call_emplace_impl(c, std::forward<arg_t>(k), selector);
}

template<typename container_t, typename arg_t, typename value_t>
typename container_t::iterator call_emplace_hint_impl(container_t& c, typename container_t::const_iterator hint, arg_t&& k, value_t *){
    // this is a set
    return c.emplace_hint(hint, std::forward<arg_t>(k));
}

template<typename container_t, typename arg_t, typename first_t, typename second_t>
typename container_t::iterator call_emplace_hint_impl(container_t& c, typename container_t::const_iterator hint, arg_t&& k, std::pair<first_t, second_t> *){
    // this is a map
    return c.emplace_hint(hint, k, std::forward<arg_t>(k));
}

template<typename container_t, typename arg_t>
typename container_t::iterator call_emplace_hint(container_t& c, typename container_t::const_iterator hint, arg_t&& k){
    typename container_t::value_type * selector = NULL;
    return call_emplace_hint_impl(c, hint, std::forward<arg_t>(k), selector);
}
}
template<typename T, typename do_check_element_state, typename V>
void test_emplace_insert(V v1, V v2){
    typedef T container_t;
    container_t cont;

    std::pair<typename container_t::iterator, bool> ins = emplace_helpers::call_emplace(cont, v1);
    ASSERT(ins.second == true && Value<container_t>::compare(ins.first, v1), "Element 1 has not been inserted properly");
    ASSERT_VALUE_STATE(do_check_element_state(),Harness::StateTrackableBase::DirectInitialized,*ins.first);

    typename container_t::iterator it2 = emplace_helpers::call_emplace_hint(cont, ins.first, v2);
    ASSERT(Value<container_t>::compare(it2, v2), "Element 2 has not been inserted properly");
    ASSERT_VALUE_STATE(do_check_element_state(),Harness::StateTrackableBase::DirectInitialized,*it2);
}
#endif //__TBB_CPP11_VARIADIC_TEMPLATES_PRESENT
#endif // __TBB_CPP11_RVALUE_REF_PRESENT

template<typename T, typename do_check_element_state>
void test_basic(const char * str, do_check_element_state)
{
    T cont;
    const T &ccont(cont);

    // bool empty() const;
    ASSERT(ccont.empty(), "Concurrent container is not empty after construction");

    // size_type size() const;
    ASSERT(ccont.size() == 0, "Concurrent container is not empty after construction");

    // size_type max_size() const;
    ASSERT(ccont.max_size() > 0, "Concurrent container max size is invalid");

    //iterator begin();
    //iterator end();
    ASSERT(cont.begin() == cont.end(), "Concurrent container iterators are invalid after construction");
    ASSERT(ccont.begin() == ccont.end(), "Concurrent container iterators are invalid after construction");
    ASSERT(cont.cbegin() == cont.cend(), "Concurrent container iterators are invalid after construction");

    //std::pair<iterator, bool> insert(const value_type& obj);
    std::pair<typename T::iterator, bool> ins = cont.insert(Value<T>::make(1));
    ASSERT(ins.second == true && Value<T>::get(*(ins.first)) == 1, "Element 1 has not been inserted properly");

#if __TBB_CPP11_RVALUE_REF_PRESENT
    test_rvalue_insert<T,do_check_element_state>(1,2);
#if __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT
    test_emplace_insert<T,do_check_element_state>(1,2);
#endif // __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT
#endif // __TBB_CPP11_RVALUE_REF_PRESENT

    // bool empty() const;
    ASSERT(!ccont.empty(), "Concurrent container is empty after adding an element");

    // size_type size() const;
    ASSERT(ccont.size() == 1, "Concurrent container size is incorrect");

    std::pair<typename T::iterator, bool> ins2 = cont.insert(Value<T>::make(1));

    if (T::allow_multimapping)
    {
        // std::pair<iterator, bool> insert(const value_type& obj);
        ASSERT(ins2.second == true && Value<T>::get(*(ins2.first)) == 1, "Element 1 has not been inserted properly");

        // size_type size() const;
        ASSERT(ccont.size() == 2, "Concurrent container size is incorrect");

        // size_type count(const key_type& k) const;
        ASSERT(ccont.count(1) == 2, "Concurrent container count(1) is incorrect");

        // std::pair<iterator, iterator> equal_range(const key_type& k);
        std::pair<typename T::iterator, typename T::iterator> range = cont.equal_range(1);
        typename T::iterator it = range.first;
        ASSERT(it != cont.end() && Value<T>::get(*it) == 1, "Element 1 has not been found properly");
        unsigned int count = 0;
        for (; it != range.second; it++)
        {
            count++;
            ASSERT(Value<T>::get(*it) == 1, "Element 1 has not been found properly");
        }

        ASSERT(count == 2, "Range doesn't have the right number of elements");
    }
    else
    {
        // std::pair<iterator, bool> insert(const value_type& obj);
        ASSERT(ins2.second == false && ins2.first == ins.first, "Element 1 should not be re-inserted");

        // size_type size() const;
        ASSERT(ccont.size() == 1, "Concurrent container size is incorrect");

        // size_type count(const key_type& k) const;
        ASSERT(ccont.count(1) == 1, "Concurrent container count(1) is incorrect");

        // std::pair<const_iterator, const_iterator> equal_range(const key_type& k) const;
        // std::pair<iterator, iterator> equal_range(const key_type& k);
        std::pair<typename T::iterator, typename T::iterator> range = cont.equal_range(1);
        typename T::iterator it = range.first;
        ASSERT(it != cont.end() && Value<T>::get(*it) == 1, "Element 1 has not been found properly");
        ASSERT(++it == range.second, "Range doesn't have the right number of elements");
    }

    // const_iterator find(const key_type& k) const;
    // iterator find(const key_type& k);
    typename T::iterator it = cont.find(1);
    ASSERT(it != cont.end() && Value<T>::get(*(it)) == 1, "Element 1 has not been found properly");
    ASSERT(ccont.find(1) == it, "Element 1 has not been found properly");

    // iterator insert(const_iterator hint, const value_type& obj);
    typename T::iterator it2 = cont.insert(ins.first, Value<T>::make(2));
    ASSERT(Value<T>::get(*it2) == 2, "Element 2 has not been inserted properly");

    // T(const T& _Umap)
    T newcont = ccont;
    ASSERT(T::allow_multimapping ? (newcont.size() == 3) : (newcont.size() == 2), "Copy construction has not copied the elements properly");

    // size_type unsafe_erase(const key_type& k);
    typename T::size_type size = cont.unsafe_erase(1);
    ASSERT(T::allow_multimapping ? (size == 2) : (size == 1), "Erase has not removed the right number of elements");

    // iterator unsafe_erase(const_iterator position);
    typename T::iterator it4 = cont.unsafe_erase(cont.find(2));
    ASSERT(it4 == cont.end() && cont.size() == 0, "Erase has not removed the last element properly");

    // template<class InputIterator> void insert(InputIterator first, InputIterator last);
    cont.insert(newcont.begin(), newcont.end());
    ASSERT(T::allow_multimapping ? (cont.size() == 3) : (cont.size() == 2), "Range insert has not copied the elements properly");

    // iterator unsafe_erase(const_iterator first, const_iterator last);
    std::pair<typename T::iterator, typename T::iterator> range2 = newcont.equal_range(1);
    newcont.unsafe_erase(range2.first, range2.second);
    ASSERT(newcont.size() == 1, "Range erase has not erased the elements properly");

    // void clear();
    newcont.clear();
    ASSERT(newcont.begin() == newcont.end() && newcont.size() == 0, "Clear has not cleared the container");

#if __TBB_INITIALIZER_LISTS_PRESENT
#if __TBB_CPP11_INIT_LIST_TEMP_OBJS_LIFETIME_BROKEN
    REPORT("Known issue: the test for insert with initializer_list is skipped.\n");
#else
    // void insert(const std::initializer_list<value_type> &il);
    newcont.insert( { Value<T>::make( 1 ), Value<T>::make( 2 ), Value<T>::make( 1 ) } );
    if (T::allow_multimapping) {
        ASSERT(newcont.size() == 3, "Concurrent container size is incorrect");
        ASSERT(newcont.count(1) == 2, "Concurrent container count(1) is incorrect");
        ASSERT(newcont.count(2) == 1, "Concurrent container count(2) is incorrect");
        std::pair<typename T::iterator, typename T::iterator> range = cont.equal_range(1);
        it = range.first;
        ASSERT(it != newcont.end() && Value<T>::get(*it) == 1, "Element 1 has not been found properly");
        unsigned int count = 0;
        for (; it != range.second; it++) {
            count++;
            ASSERT(Value<T>::get(*it) == 1, "Element 1 has not been found properly");
        }
        ASSERT(count == 2, "Range doesn't have the right number of elements");
        range = newcont.equal_range(2); it = range.first;
        ASSERT(it != newcont.end() && Value<T>::get(*it) == 2, "Element 2 has not been found properly");
        count = 0;
        for (; it != range.second; it++) {
            count++;
            ASSERT(Value<T>::get(*it) == 2, "Element 2 has not been found properly");
        }
        ASSERT(count == 1, "Range doesn't have the right number of elements");
    } else {
        ASSERT(newcont.size() == 2, "Concurrent container size is incorrect");
        ASSERT(newcont.count(1) == 1, "Concurrent container count(1) is incorrect");
        ASSERT(newcont.count(2) == 1, "Concurrent container count(2) is incorrect");
        std::pair<typename T::iterator, typename T::iterator> range = newcont.equal_range(1);
        it = range.first;
        ASSERT(it != newcont.end() && Value<T>::get(*it) == 1, "Element 1 has not been found properly");
        ASSERT(++it == range.second, "Range doesn't have the right number of elements");
        range = newcont.equal_range(2); it = range.first;
        ASSERT(it != newcont.end() && Value<T>::get(*it) == 2, "Element 2 has not been found properly");
        ASSERT(++it == range.second, "Range doesn't have the right number of elements");
    }
#endif /* __TBB_CPP11_INIT_LIST_TEMP_OBJS_COMPILATION_BROKEN */
#endif /* __TBB_INITIALIZER_LISTS_PRESENT */

    // T& operator=(const T& _Umap)
    newcont = ccont;
    ASSERT(T::allow_multimapping ? (newcont.size() == 3) : (newcont.size() == 2), "Assignment operator has not copied the elements properly");

    // void rehash(size_type n);
    newcont.rehash(16);
    ASSERT(T::allow_multimapping ? (newcont.size() == 3) : (newcont.size() == 2), "Rehash should not affect the container elements");

    // float load_factor() const;
    // float max_load_factor() const;
    ASSERT(ccont.load_factor() <= ccont.max_load_factor(), "Load factor is invalid");

    // void max_load_factor(float z);
    cont.max_load_factor(16.0f);
    ASSERT(ccont.max_load_factor() == 16.0f, "Max load factor has not been changed properly");

    // hasher hash_function() const;
    ccont.hash_function();

    // key_equal key_eq() const;
    ccont.key_eq();

    cont.clear();
    CheckAllocatorA(cont, 1, 0); // one dummy is always allocated
    for (int i = 0; i < 256; i++)
    {
        std::pair<typename T::iterator, bool> ins3 = cont.insert(Value<T>::make(i));
        ASSERT(ins3.second == true && Value<T>::get(*(ins3.first)) == i, "Element 1 has not been inserted properly");
    }
    ASSERT(cont.size() == 256, "Wrong number of elements have been inserted");
    ASSERT((256 == CheckRecursiveRange<T,typename T::iterator>(cont.range()).first), NULL);
    ASSERT((256 == CheckRecursiveRange<T,typename T::const_iterator>(ccont.range()).first), NULL);

    // size_type unsafe_bucket_count() const;
    ASSERT(ccont.unsafe_bucket_count() == 16, "Wrong number of buckets");

    // size_type unsafe_max_bucket_count() const;
    ASSERT(ccont.unsafe_max_bucket_count() > 65536, "Wrong max number of buckets");

    for (unsigned int i = 0; i < 256; i++)
    {
        typename T::size_type buck = ccont.unsafe_bucket(i);

        // size_type unsafe_bucket(const key_type& k) const;
        ASSERT(buck < 16, "Wrong bucket mapping");
    }

    typename T::size_type bucketSizeSum = 0;
    typename T::size_type iteratorSizeSum = 0;

    for (unsigned int i = 0; i < 16; i++)
    {
        bucketSizeSum += cont.unsafe_bucket_size(i);
        for (typename T::iterator bit = cont.unsafe_begin(i); bit != cont.unsafe_end(i); bit++) iteratorSizeSum++;
    }
    ASSERT(bucketSizeSum == 256, "sum of bucket counts incorrect");
    ASSERT(iteratorSizeSum == 256, "sum of iterator counts incorrect");

    // void swap(T&);
    cont.swap(newcont);
    ASSERT(newcont.size() == 256, "Wrong number of elements after swap");
    ASSERT(newcont.count(200) == 1, "Element with key 200 is not present after swap");
    ASSERT(newcont.count(16) == 1, "Element with key 16 is not present after swap");
    ASSERT(newcont.count(99) == 1, "Element with key 99 is not present after swap");
    ASSERT(T::allow_multimapping ? (cont.size() == 3) : (cont.size() == 2), "Wrong number of elements after swap");

    REMARK("passed -- basic %s tests\n", str);

#if defined (VERBOSE)
    REMARK("container dump debug:\n");
    cont._Dump();
    REMARK("container dump release:\n");
    cont.dump();
    REMARK("\n");
#endif

    SpecialTests<T>::Test(str);
}

template<typename T>
void test_basic(const char * str){
    test_basic<T>(str, tbb::internal::false_type());
}

void test_machine() {
    ASSERT(__TBB_ReverseByte(0)==0, NULL );
    ASSERT(__TBB_ReverseByte(1)==0x80, NULL );
    ASSERT(__TBB_ReverseByte(0xFE)==0x7F, NULL );
    ASSERT(__TBB_ReverseByte(0xFF)==0xFF, NULL );
}

template<typename T>
class FillTable: NoAssign {
    T &table;
    const int items;
    bool my_asymptotic;
    typedef std::pair<typename T::iterator, bool> pairIB;
public:
    FillTable(T &t, int i, bool asymptotic) : table(t), items(i), my_asymptotic(asymptotic) {
        ASSERT( !(items&1) && items > 100, NULL);
    }
    void operator()(int threadn) const {
        if( threadn == 0 ) { // Fill even keys forward (single thread)
            bool last_inserted = true;
            for( int i = 0; i < items; i+=2 ) {
                pairIB pib = table.insert(Value<T>::make(my_asymptotic?1:i));
                ASSERT(Value<T>::get(*(pib.first)) == (my_asymptotic?1:i), "Element not properly inserted");
                ASSERT( last_inserted || !pib.second, "Previous key was not inserted but this is inserted" );
                last_inserted = pib.second;
            }
        } else if( threadn == 1 ) { // Fill even keys backward (single thread)
            bool last_inserted = true;
            for( int i = items-2; i >= 0; i-=2 ) {
                pairIB pib = table.insert(Value<T>::make(my_asymptotic?1:i));
                ASSERT(Value<T>::get(*(pib.first)) == (my_asymptotic?1:i), "Element not properly inserted");
                ASSERT( last_inserted || !pib.second, "Previous key was not inserted but this is inserted" );
                last_inserted = pib.second;
            }
        } else if( !(threadn&1) ) { // Fill odd keys forward (multiple threads)
            for( int i = 1; i < items; i+=2 )
#if __TBB_INITIALIZER_LISTS_PRESENT && !__TBB_CPP11_INIT_LIST_TEMP_OBJS_LIFETIME_BROKEN
                if ( i % 32 == 1 && i + 6 < items ) {
                    if (my_asymptotic) {
                        table.insert({ Value<T>::make(1), Value<T>::make(1), Value<T>::make(1) });
                        ASSERT(Value<T>::get(*table.find(1)) == 1, "Element not properly inserted");
                    }
                    else {
                        table.insert({ Value<T>::make(i), Value<T>::make(i + 2), Value<T>::make(i + 4) });
                        ASSERT(Value<T>::get(*table.find(i)) == i, "Element not properly inserted");
                        ASSERT(Value<T>::get(*table.find(i + 2)) == i + 2, "Element not properly inserted");
                        ASSERT(Value<T>::get(*table.find(i + 4)) == i + 4, "Element not properly inserted");
                    }
                    i += 4;
                } else
#endif
                {
                    pairIB pib = table.insert(Value<T>::make(my_asymptotic ? 1 : i));
                    ASSERT(Value<T>::get(*(pib.first)) == (my_asymptotic ? 1 : i), "Element not properly inserted");
                }
        } else { // Check odd keys backward (multiple threads)
            if (!my_asymptotic) {
                bool last_found = false;
                for( int i = items-1; i >= 0; i-=2 ) {
                    typename T::iterator it = table.find(i);
                    if( it != table.end() ) { // found
                        ASSERT(Value<T>::get(*it) == i, "Element not properly inserted");
                        last_found = true;
                    } else ASSERT( !last_found, "Previous key was found but this is not" );
                }
            }
        }
    }
};

typedef tbb::atomic<unsigned char> AtomicByte;

template<typename ContainerType, typename RangeType>
struct ParallelTraverseBody: NoAssign {
    const int n;
    AtomicByte* const array;
    ParallelTraverseBody( AtomicByte an_array[], int a_n ) :
        n(a_n), array(an_array)
    {}
    void operator()( const RangeType& range ) const {
        for( typename RangeType::iterator i = range.begin(); i!=range.end(); ++i ) {
            int k = static_cast<int>(Value<ContainerType>::key(*i));
            ASSERT( k == Value<ContainerType>::get(*i), NULL );
            ASSERT( 0<=k && k<n, NULL );
            array[k]++;
        }
    }
};

// if multimapping, oddCount is the value that each odd-indexed array element should have.
// not meaningful for non-multimapped case.
void CheckRange( AtomicByte array[], int n, bool allowMultiMapping, int oddCount ) {
    if(allowMultiMapping) {
        for( int k = 0; k<n; ++k) {
            if(k%2) {
                if( array[k] != oddCount ) {
                    REPORT("array[%d]=%d (should be %d)\n", k, int(array[k]), oddCount);
                    ASSERT(false,NULL);
                }
            }
            else {
                if(array[k] != 2) {
                    REPORT("array[%d]=%d\n", k, int(array[k]));
                    ASSERT(false,NULL);
                }
            }
        }
    }
    else {
        for( int k=0; k<n; ++k ) {
            if( array[k] != 1 ) {
                REPORT("array[%d]=%d\n", k, int(array[k]));
                ASSERT(false,NULL);
            }
        }
    }
}

template<typename T>
class CheckTable: NoAssign {
    T &table;
public:
    CheckTable(T &t) : NoAssign(), table(t) {}
    void operator()(int i) const {
        int c = (int)table.count( i );
        ASSERT( c, "must exist" );
    }
};

template<typename T>
void test_concurrent(const char *tablename, bool asymptotic = false) {
#if TBB_USE_ASSERT
    int items = 2000;
#else
    int items = 20000;
#endif
    int nItemsInserted = 0;
    int nThreads = 0;
    T table(items/1000);
    #if __bgp__
    nThreads = 6;
    #else
    nThreads = 16;
    #endif
    if(T::allow_multimapping) {
        // even passes (threads 0 & 1) put N/2 items each
        // odd passes  (threads > 1)   put N/2 if thread is odd, else checks if even.
        items = 4*items / (nThreads + 2);  // approximately same number of items inserted.
        nItemsInserted = items + (nThreads-2) * items / 4;
    }
    else {
        nItemsInserted = items;
    }
    REMARK("%s items == %d\n", tablename, items);
    tbb::tick_count t0 = tbb::tick_count::now();
    NativeParallelFor( nThreads, FillTable<T>(table, items, asymptotic) );
    tbb::tick_count t1 = tbb::tick_count::now();
    REMARK( "time for filling '%s' by %d items = %g\n", tablename, table.size(), (t1-t0).seconds() );
    ASSERT( int(table.size()) == nItemsInserted, NULL);

    if(!asymptotic) {
        AtomicByte* array = new AtomicByte[items];
        memset( array, 0, items*sizeof(AtomicByte) );

        typename T::range_type r = table.range();
        std::pair<intptr_t,intptr_t> p = CheckRecursiveRange<T,typename T::iterator>(r);
        ASSERT((nItemsInserted == p.first), NULL);
        tbb::parallel_for( r, ParallelTraverseBody<T, typename T::const_range_type>( array, items ));
        CheckRange( array, items, T::allow_multimapping, (nThreads - 1)/2 );

        const T &const_table = table;
        memset( array, 0, items*sizeof(AtomicByte) );
        typename T::const_range_type cr = const_table.range();
        ASSERT((nItemsInserted == CheckRecursiveRange<T,typename T::const_iterator>(cr).first), NULL);
        tbb::parallel_for( cr, ParallelTraverseBody<T, typename T::const_range_type>( array, items ));
        CheckRange( array, items, T::allow_multimapping, (nThreads - 1) / 2 );
        delete[] array;

        tbb::parallel_for( 0, items, CheckTable<T>( table ) );
    }

    table.clear();
    CheckAllocatorA(table, items+1, items); // one dummy is always allocated

}

// The helper to call a function only when a doCall == true.
template <bool doCall> struct CallIf {
    template<typename FuncType> void operator() ( FuncType func ) const { func(); }
};
template <> struct CallIf<false> {
    template<typename FuncType> void operator()( FuncType ) const {}
};

#include <vector>
#include <list>
#include <algorithm>

template <typename ValueType>
class TestRange : NoAssign {
    const std::list<ValueType> &my_lst;
    std::vector< tbb::atomic<bool> > &my_marks;
public:
    TestRange( const std::list<ValueType> &lst, std::vector< tbb::atomic<bool> > &marks ) : my_lst( lst ), my_marks( marks ) {
        std::fill( my_marks.begin(), my_marks.end(), false );
    }
    template <typename Range>
    void operator()( const Range &r ) const { doTestRange( r.begin(), r.end() ); }
    template<typename Iterator>
    void doTestRange( Iterator i, Iterator j ) const {
        for ( Iterator it = i; it != j; ) {
            Iterator prev_it = it++;
            typename std::list<ValueType>::const_iterator it2 = std::search( my_lst.begin(), my_lst.end(), prev_it, it, Harness::IsEqual() );
            ASSERT( it2 != my_lst.end(), NULL );
            typename std::list<ValueType>::difference_type dist = std::distance( my_lst.begin( ), it2 );
            ASSERT( !my_marks[dist], NULL );
            my_marks[dist] = true;
        }
    }
};

#if __TBB_CPP11_SMART_POINTERS_PRESENT
// For the sake of simplified testing, make unique_ptr implicitly convertible to/from the pointer
namespace test {
    template<typename T>
    class unique_ptr : public std::unique_ptr<T> {
    public:
        typedef typename std::unique_ptr<T>::pointer pointer;
        unique_ptr( pointer p ) : std::unique_ptr<T>(p) {}
        operator pointer() const { return this->get(); }
    };
}

namespace tbb {
    template<> class tbb_hash< std::shared_ptr<int> > {
    public:
        size_t operator()( const std::shared_ptr<int>& key ) const { return tbb_hasher( *key ); }
    };
    template<> class tbb_hash< const std::shared_ptr<int> > {
    public:
        size_t operator()( const std::shared_ptr<int>& key ) const { return tbb_hasher( *key ); }
    };
    template<> class tbb_hash< std::weak_ptr<int> > {
    public:
        size_t operator()( const std::weak_ptr<int>& key ) const { return tbb_hasher( *key.lock( ) ); }
    };
    template<> class tbb_hash< const std::weak_ptr<int> > {
    public:
        size_t operator()( const std::weak_ptr<int>& key ) const { return tbb_hasher( *key.lock( ) ); }
    };
    template<> class tbb_hash< test::unique_ptr<int> > {
    public:
        size_t operator()( const test::unique_ptr<int>& key ) const { return tbb_hasher( *key ); }
    };
    template<> class tbb_hash< const test::unique_ptr<int> > {
    public:
        size_t operator()( const test::unique_ptr<int>& key ) const { return tbb_hasher( *key ); }
    };
}
#endif /* __TBB_CPP11_SMART_POINTERS_PRESENT */

template <bool, typename Table>
void TestMapSpecificMethods( Table &, const typename Table::value_type & ) { /* do nothing for a common case */ }

template <bool defCtorPresent, typename Table>
class CheckValue : NoAssign {
    Table &my_c;
public:
    CheckValue( Table &c ) : my_c( c ) {}
    void operator()( const typename Table::value_type &value ) {
        typedef typename Table::iterator Iterator;
        typedef typename Table::const_iterator ConstIterator;
        const Table &constC = my_c;
        ASSERT( my_c.count( Value<Table>::key( value ) ) == 1, NULL );
        // find
        ASSERT( Harness::IsEqual()(*my_c.find( Value<Table>::key( value ) ), value), NULL );
        ASSERT( Harness::IsEqual()(*constC.find( Value<Table>::key( value ) ), value), NULL );
        // erase
        ASSERT( my_c.unsafe_erase( Value<Table>::key( value ) ), NULL );
        ASSERT( my_c.count( Value<Table>::key( value ) ) == 0, NULL );
        // insert
        std::pair<Iterator, bool> res = my_c.insert( value );
        ASSERT( Harness::IsEqual()(*res.first, value), NULL );
        ASSERT( res.second, NULL);
        // erase
        Iterator it = res.first;
        it++;
        ASSERT( my_c.unsafe_erase( res.first ) == it, NULL );
        // insert
        ASSERT( Harness::IsEqual()(*my_c.insert( my_c.begin(), value ), value), NULL );
        // equal_range
        std::pair<Iterator, Iterator> r1 = my_c.equal_range( Value<Table>::key( value ) );
        ASSERT( Harness::IsEqual()(*r1.first, value) && ++r1.first == r1.second, NULL );
        std::pair<ConstIterator, ConstIterator> r2 = constC.equal_range( Value<Table>::key( value ) );
        ASSERT( Harness::IsEqual()(*r2.first, value) && ++r2.first == r2.second, NULL );
        TestMapSpecificMethods<defCtorPresent>( my_c, value );
    }
};

#include "tbb/task_scheduler_init.h"

#if __TBB_CPP11_RVALUE_REF_PRESENT
#include "test_container_move_support.h"

struct unordered_move_traits_base {
    enum{ expected_number_of_items_to_allocate_for_steal_move = 3 };

    template <typename unordered_type, typename iterator_type>
    static unordered_type& construct_container(tbb::aligned_space<unordered_type> & storage, iterator_type begin, iterator_type end){
        new (storage.begin()) unordered_type(begin, end);
        return * storage.begin();
    }

    template <typename unordered_type, typename iterator_type, typename allocator_type>
    static unordered_type& construct_container(tbb::aligned_space<unordered_type> & storage, iterator_type begin, iterator_type end, allocator_type const& a ){
        size_t deault_n_of_buckets = 8; //can not use concurrent_unordered_base::n_of_buckets as it is inaccessible
        new (storage.begin()) unordered_type(begin, end, deault_n_of_buckets, typename unordered_type::hasher(), typename unordered_type::key_equal(), a);
        return * storage.begin();
    }

    template<typename unordered_type, typename iterator>
    static bool equal(unordered_type const& c, iterator begin, iterator end){
        bool equal_sizes = ( static_cast<size_t>(std::distance(begin, end)) == c.size() );
        if (!equal_sizes)
            return false;

        for (iterator it = begin; it != end; ++it ){
            if (c.find( Value<unordered_type>::key(*it)) == c.end()){
                return false;
            }
        }
        return true;
    }
};

template<typename container_traits>
void test_rvalue_ref_support(const char* container_name){
    TestMoveConstructor<container_traits>();
    TestMoveAssignOperator<container_traits>();
#if TBB_USE_EXCEPTIONS
    TestExceptionSafetyGuaranteesMoveConstructorWithUnEqualAllocatorMemoryFailure<container_traits>();
    TestExceptionSafetyGuaranteesMoveConstructorWithUnEqualAllocatorExceptionInElementCtor<container_traits>();
#endif //TBB_USE_EXCEPTIONS
    REMARK("passed -- %s move support tests\n", container_name);
}
#endif //__TBB_CPP11_RVALUE_REF_PRESENT

template <bool defCtorPresent, typename Table>
void Examine( Table c, const std::list<typename Table::value_type> &lst ) {
    typedef typename Table::size_type SizeType;
    typedef typename Table::value_type ValueType;

    ASSERT( !c.empty() && c.size() == lst.size() && c.max_size() >= c.size(), NULL );

    std::for_each( lst.begin(), lst.end(), CheckValue<defCtorPresent, Table>( c ) );

    std::vector< tbb::atomic<bool> > marks( lst.size() );

    TestRange<ValueType>( lst, marks ).doTestRange( c.begin(), c.end() );
    ASSERT( std::find( marks.begin(), marks.end(), false ) == marks.end(), NULL );

    TestRange<ValueType>( lst, marks ).doTestRange( c.begin(), c.end() );
    ASSERT( std::find( marks.begin(), marks.end(), false ) == marks.end(), NULL );

    const Table constC = c;
    ASSERT( c.size() == constC.size(), NULL );

    TestRange<ValueType>( lst, marks ).doTestRange( constC.cbegin(), constC.cend() );
    ASSERT( std::find( marks.begin(), marks.end(), false ) == marks.end(), NULL );

    tbb::task_scheduler_init init;

    tbb::parallel_for( c.range(), TestRange<ValueType>( lst, marks ) );
    ASSERT( std::find( marks.begin(), marks.end(), false ) == marks.end(), NULL );

    tbb::parallel_for( constC.range( ), TestRange<ValueType>( lst, marks ) );
    ASSERT( std::find( marks.begin(), marks.end(), false ) == marks.end(), NULL );

    const SizeType bucket_count = c.unsafe_bucket_count();
    ASSERT( c.unsafe_max_bucket_count() >= bucket_count, NULL );
    SizeType counter = SizeType( 0 );
    for ( SizeType i = 0; i < bucket_count; ++i ) {
        const SizeType size = c.unsafe_bucket_size( i );
        typedef typename Table::difference_type diff_type;
        ASSERT( std::distance( c.unsafe_begin( i ), c.unsafe_end( i ) ) == diff_type( size ), NULL );
        ASSERT( std::distance( c.unsafe_cbegin( i ), c.unsafe_cend( i ) ) == diff_type( size ), NULL );
        ASSERT( std::distance( constC.unsafe_begin( i ), constC.unsafe_end( i ) ) == diff_type( size ), NULL );
        ASSERT( std::distance( constC.unsafe_cbegin( i ), constC.unsafe_cend( i ) ) == diff_type( size ), NULL );
        counter += size;
    }
    ASSERT( counter == lst.size(), NULL );

    typedef typename Table::value_type value_type;
    for ( typename std::list<value_type>::const_iterator it = lst.begin(); it != lst.end(); ) {
        const SizeType index = c.unsafe_bucket( Value<Table>::key( *it ) );
        typename std::list<value_type>::const_iterator prev_it = it++;
        ASSERT( std::search( c.unsafe_begin( index ), c.unsafe_end( index ), prev_it, it, Harness::IsEqual() ) != c.unsafe_end( index ), NULL );
    }

    c.rehash( 2 * bucket_count );
    ASSERT( c.unsafe_bucket_count() > bucket_count, NULL );

    ASSERT( c.load_factor() <= c.max_load_factor(), NULL );
    c.max_load_factor( 1.0f );

    Table c2;
    typename std::list<value_type>::const_iterator begin5 = lst.begin();
    std::advance( begin5, 5 );
    c2.insert( lst.begin(), begin5 );
    std::for_each( lst.begin(), begin5, CheckValue<defCtorPresent, Table>( c2 ) );

    c2.swap( c );
    ASSERT( c2.size() == lst.size(), NULL );
    ASSERT( c.size() == 5, NULL );
    std::for_each( lst.begin(), lst.end(), CheckValue<defCtorPresent, Table>( c2 ) );

    c2.clear();
    ASSERT( c2.size() == 0, NULL );

    typename Table::allocator_type a = c.get_allocator();
    value_type *ptr = a.allocate( 1 );
    ASSERT( ptr, NULL );
    a.deallocate( ptr, 1 );

    c.hash_function();
    c.key_eq();
}

template <bool defCtorPresent, typename Table, typename TableDebugAlloc>
void TypeTester( const std::list<typename Table::value_type> &lst ) {
    ASSERT( lst.size() >= 5, "Array should have at least 5 elements" );
    ASSERT( lst.size() <= 100, "The test has O(n^2) complexity so a big number of elements can lead long execution time" );
    // Construct an empty table.
    Table c1;
    c1.insert( lst.begin(), lst.end() );
    Examine<defCtorPresent>( c1, lst );
#if __TBB_INITIALIZER_LISTS_PRESENT && !__TBB_CPP11_INIT_LIST_TEMP_OBJS_LIFETIME_BROKEN
    // Constructor from an initializer_list.
    typename std::list<typename Table::value_type>::const_iterator it = lst.begin();
    Table c2( { *it++, *it++, *it++ } );
    c2.insert( it, lst.end( ) );
    Examine<defCtorPresent>( c2, lst );
#endif
    // Copying constructor.
    Table c3( c1 );
    Examine<defCtorPresent>( c3, lst );
    // Construct with non-default allocator
    TableDebugAlloc c4;
    c4.insert( lst.begin(), lst.end() );
    Examine<defCtorPresent>( c4, lst );
    // Copying constructor for a container with a different allocator type.
    TableDebugAlloc c5( c4 );
    Examine<defCtorPresent>( c5, lst );
    // Construction empty table with n preallocated buckets.
    Table c6( lst.size() );
    c6.insert( lst.begin(), lst.end() );
    Examine<defCtorPresent>( c6, lst );
    TableDebugAlloc c7( lst.size( ) );
    c7.insert( lst.begin(), lst.end() );
    Examine<defCtorPresent>( c7, lst );
    // Construction with a copying iteration range and a given allocator instance.
    Table c8( c1.begin(), c1.end() );
    Examine<defCtorPresent>( c8, lst );
    typename TableDebugAlloc::allocator_type a;
    TableDebugAlloc c9( a );
    c9.insert( c7.begin(), c7.end() );
    Examine<defCtorPresent>( c9, lst );
}

namespace test_select_size_t_constant{
    __TBB_STATIC_ASSERT((tbb::internal::select_size_t_constant<1234,1234>::value == 1234),"select_size_t_constant::value is not compile time constant");
//    There will be two constant used in the test 32 bit and 64 bit one.
//    The 64 bit constant should chosen so that it 32 bit halves adds up to the 32 bit one ( first constant used in the test).
//    % ~0U is used to sum up 32bit halves of the 64 constant.  ("% ~0U" essentially adds the 32-bit "digits", like "%9" adds
//    the digits (modulo 9) of a number in base 10).
//    So iff select_size_t_constant is correct result of the calculation below will be same on both 32bit and 64bit platforms.
    __TBB_STATIC_ASSERT((tbb::internal::select_size_t_constant<0x12345678U,0x091A2B3C091A2B3CULL>::value % ~0U == 0x12345678U),
            "select_size_t_constant have chosen the wrong constant");
}
