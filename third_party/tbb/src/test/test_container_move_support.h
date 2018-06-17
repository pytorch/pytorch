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

#ifndef __TBB_test_container_move_support_H
#define __TBB_test_container_move_support_H

#include "harness.h"
#include "harness_assert.h"
#include "harness_allocator.h"
#include "harness_state_trackable.h"
#include "tbb/atomic.h"
#include "tbb/aligned_space.h"
#include <stdexcept>
#include <string>
#include <functional>

tbb::atomic<size_t> FooCount;
size_t MaxFooCount = 0;

//! Exception for concurrent_container
class Foo_exception : public std::bad_alloc {
public:
    virtual const char *what() const throw() __TBB_override { return "out of Foo limit"; }
    virtual ~Foo_exception() throw() {}
};

struct FooLimit {
    FooLimit(){
        if(MaxFooCount && FooCount >= MaxFooCount)
            __TBB_THROW( Foo_exception() );
    }
};

static const intptr_t initial_value_of_bar = 42;


struct Foo : FooLimit, Harness::StateTrackable<true>{
    typedef Harness::StateTrackable<true> StateTrackable;
    intptr_t my_bar;
public:
    bool is_valid_or_zero() const{
        return is_valid()||(state==ZeroInitialized && !my_bar);
    }
    intptr_t& zero_bar(){
        ASSERT( is_valid_or_zero(), NULL );
        return my_bar;
    }
    intptr_t zero_bar() const{
        ASSERT( is_valid_or_zero(), NULL );
        return my_bar;
    }
    intptr_t& bar(){
        ASSERT( is_valid(), NULL );
        return my_bar;
    }
    intptr_t bar() const{
        ASSERT( is_valid(), NULL );
        return my_bar;
    }
    operator intptr_t() const{
        return this->bar();
    }
    Foo( intptr_t barr ): StateTrackable(0){
        my_bar = barr;
        FooCount++;
    }
    Foo(){
        my_bar = initial_value_of_bar;
        FooCount++;
    }
    Foo( const Foo& foo ): FooLimit(), StateTrackable(foo){
        my_bar = foo.my_bar;
        FooCount++;
    }
#if __TBB_CPP11_RVALUE_REF_PRESENT
    Foo( Foo&& foo ): FooLimit(), StateTrackable(std::move(foo)){
        my_bar = foo.my_bar;
        //TODO: consider not using constant here, instead something like ~my_bar
        foo.my_bar = -1;
        FooCount++;
    }
#endif
    ~Foo(){
        my_bar = ~initial_value_of_bar;
        if(state != ZeroInitialized) --FooCount;
    }
    friend bool operator==(const int &lhs, const Foo &rhs) {
        ASSERT( rhs.is_valid_or_zero(), "comparing invalid objects ?" );
        return lhs == rhs.my_bar;
    }
    friend bool operator==(const Foo &lhs, const int &rhs) {
        ASSERT( lhs.is_valid_or_zero(),   "comparing invalid objects ?" );
        return lhs.my_bar == rhs;
    }
    friend bool operator==(const Foo &lhs, const Foo &rhs) {
        ASSERT( lhs.is_valid_or_zero(),   "comparing invalid objects ?" );
        ASSERT( rhs.is_valid_or_zero(), "comparing invalid objects ?" );
        return lhs.my_bar == rhs.my_bar;
    }
    friend bool operator<(const Foo &lhs, const Foo &rhs) {
        ASSERT( lhs.is_valid_or_zero(),   "comparing invalid objects ?" );
        ASSERT( rhs.is_valid_or_zero(), "comparing invalid objects ?" );
        return lhs.my_bar < rhs.my_bar;
    }
    bool is_const() const {return true;}
    bool is_const() {return false;}
protected:
    char reserve[1];
    Foo& operator=( const Foo& x ) {
        StateTrackable::operator=(x);
        my_bar = x.my_bar;
        return *this;
    }
#if __TBB_CPP11_RVALUE_REF_PRESENT
    Foo& operator=( Foo&& x ) {
        ASSERT( x.is_valid_or_zero(), "bad source for assignment" );
        ASSERT( is_valid_or_zero(), NULL );
        StateTrackable::operator=(std::move(x));
        my_bar = x.my_bar;
        x.my_bar = -1;
        return *this;
    }
#endif
};

struct FooWithAssign: public Foo {
    FooWithAssign()                         : Foo(){}
    FooWithAssign(intptr_t barr)            : Foo(barr){}
    FooWithAssign(FooWithAssign const& f) : Foo(f) {}
    FooWithAssign& operator=(FooWithAssign const& f) { return  static_cast<FooWithAssign&>(Foo::operator=(f)); }


#if __TBB_CPP11_RVALUE_REF_PRESENT
    FooWithAssign(FooWithAssign && f)       : Foo(std::move(f)) {}
    FooWithAssign& operator=(FooWithAssign && f) { return  static_cast<FooWithAssign&>(Foo::operator=(std::move(f))); }
#endif
};

template<typename FooIteratorType>
class FooIteratorBase {
protected:
    intptr_t x_bar;
private:
    FooIteratorType& as_derived(){ return *static_cast<FooIteratorType*>(this);}
public:
    FooIteratorBase(intptr_t x) {
        x_bar = x;
    }
    FooIteratorType &operator++() {
        x_bar++; return as_derived();
    }
    FooIteratorType operator++(int) {
        FooIteratorType tmp(as_derived()); x_bar++; return tmp;
    }
    friend bool operator==(const FooIteratorType & lhs, const FooIteratorType & rhs){ return lhs.x_bar == rhs.x_bar; }
    friend bool operator!=(const FooIteratorType & lhs, const FooIteratorType & rhs){ return !(lhs == rhs); }
};

class FooIterator: public std::iterator<std::input_iterator_tag,FooWithAssign>, public FooIteratorBase<FooIterator> {
public:
    FooIterator(intptr_t x): FooIteratorBase<FooIterator>(x) {}

    FooWithAssign operator*() {
        return FooWithAssign(x_bar);
    }
};

class FooPairIterator: public std::iterator<std::input_iterator_tag, std::pair<FooWithAssign,FooWithAssign> >,  public FooIteratorBase<FooPairIterator> {
public:
    FooPairIterator(intptr_t x): FooIteratorBase<FooPairIterator>(x) {}

    std::pair<FooWithAssign,FooWithAssign> operator*() {
        FooWithAssign foo; foo.bar() = x_bar;

        return std::make_pair(foo, foo);
    }
};

namespace FooTests{
    template<typename Foo_type>
    void TestDefaultConstructor(){
        Foo_type src;
        ASSERT(src.state == Foo::DefaultInitialized, "incorrect state for default constructed Foo (derived) ?");
    }

    template<typename Foo_type>
    void TestDirectConstructor(){
        Foo_type src(1);
        ASSERT(src.state == Foo::DirectInitialized, "incorrect state for direct constructed Foo (derived) ?");
    }

    template<typename Foo_type>
    void TestCopyConstructor(){
        Foo_type src;
        Foo_type dst(src);
        ASSERT(dst.state == Foo::CopyInitialized, "incorrect state for Copy constructed Foo ?");
    }

    template<typename Foo_type>
    void TestAssignOperator(){
        Foo_type src;
        Foo_type dst;
        dst = (src);

        ASSERT(dst.state == Foo::Assigned, "incorrect state for Assigned Foo ?");
    }

#if __TBB_CPP11_RVALUE_REF_PRESENT
    template<typename Foo_type>
    void TestMoveConstructor(){
        Foo_type src;
        Foo_type dst(std::move(src));
        ASSERT(dst.state == Foo::MoveInitialized, "incorrect state for Move constructed Foo ?");
        ASSERT(src.state == Foo::MovedFrom, "incorrect state for Move from  Foo ?");
    }

    template<typename Foo_type>
    void TestMoveAssignOperator(){
        Foo_type src;
        Foo_type dst;
        dst = std::move(src);

        ASSERT(dst.state == Foo::MoveAssigned, "incorrect state for Move Assigned Foo ?");
        ASSERT(src.state == Foo::MovedFrom, "incorrect state for Moved from Foo ?");
    }
#if TBB_USE_EXCEPTIONS
    void TestMoveConstructorException();
#endif //TBB_USE_EXCEPTIONS
#endif //__TBB_CPP11_RVALUE_REF_PRESENT
}

void TestFoo(){
    using namespace FooTests;
    TestDefaultConstructor<Foo>();
    TestDefaultConstructor<FooWithAssign>();
    TestDirectConstructor<Foo>();
    TestDirectConstructor<FooWithAssign>();
    TestCopyConstructor<Foo>();
    TestCopyConstructor<FooWithAssign>();
    TestAssignOperator<FooWithAssign>();
#if __TBB_CPP11_RVALUE_REF_PRESENT
    TestMoveConstructor<Foo>();
    TestMoveConstructor<FooWithAssign>();
    TestMoveAssignOperator<FooWithAssign>();
#if TBB_USE_EXCEPTIONS && !__TBB_CPP11_EXCEPTION_IN_STATIC_TEST_BROKEN
    TestMoveConstructorException();
#endif //TBB_USE_EXCEPTIONS
#endif //__TBB_CPP11_RVALUE_REF_PRESENT
}

//TODO: replace _IN_TEST with separately defined macro IN_TEST(msg,test_name)
#define ASSERT_IN_TEST(p,message,test_name) ASSERT(p, (std::string(test_name) + ": " + message).c_str());
//TODO: move to harness_assert
#define ASSERT_THROWS_IN_TEST(expression, exception_type, message, test_name)                  \
        try{                                                                                   \
                expression;                                                                    \
                ASSERT_IN_TEST(false, "should throw an exception", test_name);                 \
        }catch(exception_type &){                                                              \
        }catch(...){ASSERT_IN_TEST(false, "unexpected exception", test_name);}                 \

#define ASSERT_THROWS(expression, exception_type, message)  ASSERT_THROWS_IN_TEST(expression, exception_type, message, "")

template<Harness::StateTrackableBase::StateValue desired_state, bool allow_zero_initialized_state>
bool is_state(Harness::StateTrackable<allow_zero_initialized_state> const& f){ return f.state == desired_state;}

template<Harness::StateTrackableBase::StateValue desired_state>
struct is_not_state_f {
    template <bool allow_zero_initialized_state>
    bool operator()(Harness::StateTrackable<allow_zero_initialized_state> const& f){ return !is_state<desired_state>(f);}
};

template<Harness::StateTrackableBase::StateValue desired_state>
struct is_state_f {
    template <bool allow_zero_initialized_state>
    bool operator()(Harness::StateTrackable<allow_zero_initialized_state> const& f){ return is_state<desired_state>(f); }
    //TODO: cu_map defines key as a const thus by default it is not moved, instead it is copied. Investigate how std::unordered_map behaves
    template<typename T1, typename T2>
    bool operator()(std::pair<T1, T2> const& p){ return /*is_state<desired_state>(p.first) && */is_state<desired_state>(p.second); }
};

template<typename iterator, typename unary_predicate>
bool all_of(iterator  begin, iterator const& end, unary_predicate p){
    for (; begin != end; ++begin){
        if ( !p(*begin)) return false;
    }
    return true;
}

template<typename container, typename unary_predicate>
bool all_of(container const& c, unary_predicate p){
    return ::all_of( c.begin(), c.end(), p );
}

void TestAllOf(){
    Foo foos[] = {Foo(), Foo(), Foo()};
    ASSERT(::all_of(foos, Harness::end(foos), is_state_f<Foo::DefaultInitialized>()), "all_of returned false while true expected");
    ASSERT(! ::all_of(foos, Harness::end(foos), is_state_f<Foo::CopyInitialized>()), "all_of returned true while false expected  ");
}

template<typename static_counter_allocator_type>
struct track_allocator_memory: NoCopy{
    typedef typename static_counter_allocator_type::counters_t counters_t;

    counters_t previous_state;
    const char* const test_name;
    track_allocator_memory(const char* a_test_name): test_name(a_test_name) { static_counter_allocator_type::init_counters(); }
    ~track_allocator_memory(){verify_no_allocator_memory_leaks();}

    void verify_no_allocator_memory_leaks() const{
        ASSERT_IN_TEST( static_counter_allocator_type::items_allocated == static_counter_allocator_type::items_freed, "memory leak?", test_name );
        ASSERT_IN_TEST( static_counter_allocator_type::allocations == static_counter_allocator_type::frees, "memory leak?", test_name );
    }
    void save_allocator_counters(){ previous_state = static_counter_allocator_type::counters(); }
    void verify_no_more_than_x_memory_items_allocated(size_t  expected_number_of_items_to_allocate){
        counters_t now = static_counter_allocator_type::counters();
        ASSERT_IN_TEST( (now.items_allocated - previous_state.items_allocated) <= expected_number_of_items_to_allocate, "More then excepted memory allocated ?", test_name );
    }
};

#include <vector>
template<int line_n>
struct track_foo_count: NoCopy{
    bool active;
    size_t previous_state;
    const char* const test_name;
    track_foo_count(const char* a_test_name): active(true), previous_state(FooCount), test_name(a_test_name) { }
    ~track_foo_count(){
        if (active){
            this->verify_no_undestroyed_foo_left_and_dismiss();
        }
    }

    //TODO: ideally in most places this check should be replaced with "no foo created or destroyed"
    //TODO: deactivation of the check seems like a hack
    void verify_no_undestroyed_foo_left_and_dismiss() {
        ASSERT_IN_TEST( FooCount == previous_state, "Some instances of Foo were not destroyed ?", test_name );
        active = false;
    }
};

//TODO: inactive mode in these limiters is a temporary workaround for usage in exception type loop of TestException

struct limit_foo_count_in_scope: NoCopy{
    size_t previous_state;
    bool active;
    limit_foo_count_in_scope(size_t new_limit, bool an_active = true): previous_state(MaxFooCount), active(an_active) {
        if (active){
            MaxFooCount = new_limit;
        }
    }
    ~limit_foo_count_in_scope(){
        if (active) {
            MaxFooCount = previous_state;
        }
    }
};

template<typename static_counter_allocator_type>
struct limit_allocated_items_in_scope: NoCopy{
    size_t previous_state;
    bool active;
    limit_allocated_items_in_scope(size_t new_limit, bool an_active = true) : previous_state(static_counter_allocator_type::max_items), active(an_active)  {
        if (active){
            static_counter_allocator_type::set_limits(new_limit);
        }
    }
    ~limit_allocated_items_in_scope(){
        if (active) {
            static_counter_allocator_type::set_limits(previous_state);
        }
    }
};

struct default_container_traits{
    template <typename container_type, typename iterator_type>
    static container_type& construct_container(tbb::aligned_space<container_type> & storage, iterator_type begin, iterator_type end){
        new (storage.begin()) container_type(begin, end);
        return *storage.begin();
    }

    template <typename container_type, typename iterator_type, typename allocator_type>
    static container_type& construct_container(tbb::aligned_space<container_type> & storage, iterator_type begin, iterator_type end, allocator_type const& a){
        new (storage.begin()) container_type(begin, end, a);
        return *storage.begin();
    }
};

struct memory_locations {
    std::vector<const void*> locations;

    template <typename container_type>
    memory_locations(container_type const& source) : locations(source.size()){
        for (typename container_type::const_iterator it = source.begin(); it != source.end(); ++it){locations[std::distance(source.begin(), it)] = & *it;}
    }

    template <typename container_t>
    bool content_location_unchanged(container_t const& dst){
        struct is_same_location{
            static bool compare(typename container_t::value_type const& v,  const void* location){ return &v == location;}
        };

        return std::equal(dst.begin(), dst.end(), locations.begin(), &is_same_location::compare);
    }

    template <typename container_t>
    bool content_location_changed(container_t const& dst){
        struct is_not_same_location{
            static bool compare(typename container_t::value_type const& v,  const void* location){ return &v != location;}
        };

        return std::equal(dst.begin(), dst.end(), locations.begin(), &is_not_same_location::compare);
    }

};

#if __TBB_CPP11_RVALUE_REF_PRESENT
#include <algorithm>
void TestMemoryLocaionsHelper(){
    const size_t test_sequence_len =  15;
    std::vector<char> source(test_sequence_len, 0);
    std::generate_n(source.begin(), source.size(), Harness::FastRandomBody<char>(1));

    memory_locations source_memory_locations((source));

    std::vector<char> copy((source));
    ASSERT(source_memory_locations.content_location_changed(copy), "");

    std::vector<char> alias(std::move(source));
    ASSERT(source_memory_locations.content_location_unchanged(alias), "");
}
namespace FooTests{
#if TBB_USE_EXCEPTIONS
    void TestMoveConstructorException(){
        Foo src;
        const Foo::StateValue source_state_before = src.state;
        ASSERT_THROWS_IN_TEST(
            {
                limit_foo_count_in_scope foo_limit(FooCount);
                Foo f1(std::move(src));
            },
            std::bad_alloc, "", "TestLimitInstancesNumber"
        );
        ASSERT(source_state_before == src.state, "state of source changed while should not?");
    }
#endif //TBB_USE_EXCEPTIONS
}

template<typename container_traits, typename allocator_t>
struct move_fixture : NoCopy{
    typedef  typename allocator_t::value_type element_type;
    typedef  typename container_traits:: template apply<element_type, allocator_t>::type container_t;
    typedef  typename container_traits::init_iterator_type init_iterator_type;
    enum {default_container_size = 100};
    const size_t  container_size;
    tbb::aligned_space<container_t> source_storage;
    container_t & source;
    //check that location of _all_ elements of container under test is changed/unchanged
    memory_locations locations;

    ~move_fixture(){
        source_storage.begin()->~container_t();
    }

    const char* const test_name;
    move_fixture(const char* a_test_name, size_t a_container_size = default_container_size )
    :   container_size(a_container_size)
    ,   source(container_traits::construct_container(source_storage, init_iterator_type(0), init_iterator_type(container_size)))
    ,   locations(source)
    ,   test_name(a_test_name)
    {
        init("move_fixture::move_fixture()");
    }

    move_fixture(const char* a_test_name, allocator_t const& a, size_t a_container_size = default_container_size)
    :   container_size(a_container_size)
    ,   source(container_traits::construct_container(source_storage, init_iterator_type(0), init_iterator_type(container_size), a))
    ,   locations(source)
    ,   test_name(a_test_name)
    {
        init("move_fixture::move_fixture(allocator_t const& a)");
    }

    void init(const std::string& ctor_name){
        verify_size(source, ctor_name.c_str());
        verify_content_equal_to_source(source, "did not properly initialized source? Or can not check container for equality with expected ?: " + ctor_name);
        verify_size(locations.locations, "move_fixture:init ");
    }

    bool content_location_unchanged(container_t const& dst){
        return locations.content_location_unchanged(dst);
    }

    bool content_location_changed(container_t const& dst){
        return locations.content_location_changed(dst);
    }

    template<typename container_type>
    void verify_size(container_type const& dst, const char* a_test_name){
        ASSERT_IN_TEST(container_size == dst.size(), "Did not construct all the elements or allocate enough memory?, while should ?", a_test_name);
    }

    void verify_content_equal_to_source(container_t const& dst, const std::string& msg){
        ASSERT_IN_TEST( container_traits::equal(dst, init_iterator_type(0), init_iterator_type(container_size)), msg.c_str(), test_name);
    }

    void verify_content_equal_to_source(container_t const& dst){
        verify_content_equal_to_source(dst, "content changed during move/copy ?");
    }

    void verify_content_equal_to_source(container_t const& dst, size_t number_of_constructed_items){
        ASSERT_IN_TEST(number_of_constructed_items <= dst.size(), "incorrect test expectation/input parameters?", test_name);
        ASSERT_IN_TEST(std::equal(dst.begin(), dst.begin() + number_of_constructed_items, init_iterator_type(0)), "content changed during move/copy ?", test_name);
    }

    //TODO: better name ? e.g. "content_was_stolen"
    void verify_content_shallow_moved(container_t const& dst){
        verify_size(dst, test_name);
        ASSERT_IN_TEST(content_location_unchanged(dst), "container move constructor actually changed element locations, while should not", test_name);
        ASSERT_IN_TEST(source.empty(), "Moved from container instance should not contain any elements", test_name);
        verify_content_equal_to_source(dst);
    }

    //TODO: better name ? e.g. "element move"
    void verify_content_deep_moved(container_t const& dst){
        verify_size(dst, test_name);
        ASSERT_IN_TEST(content_location_changed(dst),                "container actually did not changed element locations for unequal allocators, while should", test_name);
        ASSERT_IN_TEST(all_of(dst, is_state_f<Foo::MoveInitialized>()), "container did not move construct some elements?", test_name);
        ASSERT_IN_TEST(all_of(source, is_state_f<Foo::MovedFrom>()),    "container did not move all the elements?", test_name);
        verify_content_equal_to_source(dst);
    }

    void verify_part_of_content_deep_moved(container_t const& dst, size_t number_of_constructed_items){
        ASSERT_IN_TEST(content_location_changed(dst),                "Vector actually did not changed element locations for unequal allocators, while should", test_name);
        ASSERT_IN_TEST(::all_of(dst.begin(), dst.begin() + number_of_constructed_items, is_state_f<Foo::MoveInitialized>()), "Vector did not move construct some elements?", test_name);
        if (dst.size() != number_of_constructed_items) {
            ASSERT_IN_TEST(::all_of(dst.begin() + number_of_constructed_items, dst.end(), is_state_f<Foo::ZeroInitialized>()), "Failed to zero-initialize items left not constructed after the exception?", test_name );
        }
        verify_content_equal_to_source(dst, number_of_constructed_items);

        ASSERT_IN_TEST(::all_of(source.begin(), source.begin() + number_of_constructed_items, is_state_f<Foo::MovedFrom>()),  "Vector did not move all the elements?", test_name);
        ASSERT_IN_TEST(::all_of(source.begin() + number_of_constructed_items, source.end(), is_not_state_f<Foo::MovedFrom>()),  "Vector changed elements in source after exception point?", test_name);
    }
};


template <typename T, typename pocma = Harness::false_type>
struct arena_allocator_fixture : NoCopy{
    typedef arena<T, pocma>  allocator_t;
    typedef typename allocator_t::arena_data_t arena_data_t;

    std::vector<tbb::aligned_space<T, 1> > storage;
    arena_data_t arena_data;
    allocator_t allocator;

    arena_allocator_fixture(size_t size_to_allocate)
    :   storage(size_to_allocate)
    ,   arena_data((*storage.begin()).begin(), storage.size())
    ,   allocator(arena_data)
    {}
};

//TODO: add ability to inject debug_allocator into stateful_allocator_fixture::allocator_t
template <typename T, typename pocma = Harness::false_type>
struct two_memory_arenas_fixture : NoCopy{
    typedef arena_allocator_fixture<T, pocma> arena_fixture_t;
    typedef typename arena_fixture_t::allocator_t  allocator_t;

    arena_fixture_t source_arena_fixture;
    arena_fixture_t dst_arena_fixture;

    allocator_t& source_allocator;
    allocator_t& dst_allocator;

    const char* test_name;

    two_memory_arenas_fixture(size_t size_to_allocate, const char* a_test_name)
    :   source_arena_fixture(size_to_allocate)
    ,   dst_arena_fixture(size_to_allocate)
    ,   source_allocator(source_arena_fixture.allocator)
    ,   dst_allocator(dst_arena_fixture.allocator)
    ,   test_name(a_test_name)
    {
        ASSERT_IN_TEST(&*source_arena_fixture.storage.begin() != &*dst_arena_fixture.storage.begin(), "source and destination arena instances should use different memory regions", test_name);
        ASSERT_IN_TEST(source_allocator != dst_allocator, "arenas using different memory regions should not compare equal", test_name);
        ASSERT_IN_TEST(pocma::value == tbb::internal::allocator_traits<allocator_t>::propagate_on_container_move_assignment::value, "This test require proper allocator_traits support", test_name);

        //Some ISO C++11 allocator requirements enforcement:
        allocator_t source_allocator_copy(source_allocator), dst(dst_allocator);
        allocator_t source_previous_state(source_allocator);
        ASSERT_IN_TEST(source_previous_state == source_allocator, "Copy of allocator should compare equal to it's source", test_name);
        dst = std::move(source_allocator_copy);
        ASSERT_IN_TEST(dst == source_previous_state, "Move initialized instance of allocator should compare equal to it's source state before movement", test_name);
    }

    void verify_allocator_was_moved(const allocator_t& result_allocator){
        //TODO: add assert that allocator move constructor/assignment operator was called
        ASSERT_IN_TEST(result_allocator == source_allocator, "allocator was not moved ?", test_name);
        ASSERT_IN_TEST(result_allocator != dst_allocator,    "allocator was not moved ?", test_name);
    }

//    template <typename any_allocator_t>
//    void verify_allocator_was_moved(const any_allocator_t& ){}
};

template <typename pocma = Harness::false_type>
struct std_stateful_allocator : NoCopy {
    typedef stateful_allocator<FooWithAssign, pocma> allocator_t;

    allocator_t source_allocator;
    allocator_t dst_allocator;

    const char* test_name;

    std_stateful_allocator(size_t , const char* a_test_name)
    :   test_name(a_test_name)
    {}

    template <typename any_allocator_t>
    void verify_allocator_was_moved(const any_allocator_t& ){}

};

template<typename container_traits, typename pocma = Harness::false_type, typename T = FooWithAssign>
struct default_stateful_fixture_make_helper{
//    typedef std_stateful_allocator<pocma> allocator_fixture_t;
    typedef two_memory_arenas_fixture<T, pocma> allocator_fixture_t;
    typedef static_shared_counting_allocator<Harness::int_to_type<__LINE__>, typename allocator_fixture_t::allocator_t, std::size_t> allocator_t;

    typedef move_fixture<container_traits, allocator_t> move_fixture_t;
    typedef track_allocator_memory<allocator_t> no_leaks_t;
    typedef track_foo_count<__LINE__> no_foo_leaks_in_fixture_t;
    typedef track_foo_count<__LINE__> no_foo_leaks_in_test_t;

    struct default_stateful_fixture : no_leaks_t, private no_foo_leaks_in_fixture_t, allocator_fixture_t, move_fixture_t, no_foo_leaks_in_test_t {

        default_stateful_fixture(const char* a_test_name)
        :   no_leaks_t(a_test_name)
        ,   no_foo_leaks_in_fixture_t(a_test_name)
        //TODO: calculate needed size more accurately
        //allocate twice more storage to handle case when copy constructor called instead of move one
        ,   allocator_fixture_t(2*4 * move_fixture_t::default_container_size, a_test_name)
        ,   move_fixture_t(a_test_name, allocator_fixture_t::source_allocator)
        ,   no_foo_leaks_in_test_t(a_test_name)
        {
            no_leaks_t::save_allocator_counters();
        }

        void verify_no_more_than_x_memory_items_allocated(){
            no_leaks_t::verify_no_more_than_x_memory_items_allocated(container_traits::expected_number_of_items_to_allocate_for_steal_move);
        }
        using no_foo_leaks_in_test_t::verify_no_undestroyed_foo_left_and_dismiss;
        typedef typename move_fixture_t::container_t::allocator_type allocator_t;
    };

    typedef default_stateful_fixture type;
};

template<typename container_traits>
void TestMoveConstructorSingleArgument(){
    typedef typename default_stateful_fixture_make_helper<container_traits>::type fixture_t;
    typedef typename fixture_t::container_t container_t;

    fixture_t fixture("TestMoveConstructorSingleArgument");

    container_t dst(std::move(fixture.source));

    fixture.verify_content_shallow_moved(dst);
    fixture.verify_allocator_was_moved(dst.get_allocator());
    fixture.verify_no_more_than_x_memory_items_allocated();
    fixture.verify_no_undestroyed_foo_left_and_dismiss();
}

template<typename container_traits>
void TestMoveConstructorWithEqualAllocator(){
    typedef typename default_stateful_fixture_make_helper<container_traits>::type fixture_t;
    typedef typename fixture_t::container_t container_t;

    fixture_t fixture("TestMoveConstructorWithEqualAllocator");

    container_t dst(std::move(fixture.source), fixture.source.get_allocator());

    fixture.verify_content_shallow_moved(dst);
    fixture.verify_no_more_than_x_memory_items_allocated();
    fixture.verify_no_undestroyed_foo_left_and_dismiss();
}

template<typename container_traits>
void TestMoveConstructorWithUnEqualAllocator(){
    typedef typename default_stateful_fixture_make_helper<container_traits>::type fixture_t;
    typedef typename fixture_t::container_t container_t;

    fixture_t fixture("TestMoveConstructorWithUnEqualAllocator");

    container_t dst(std::move(fixture.source), fixture.dst_allocator);

    fixture.verify_content_deep_moved(dst);
}

template<typename container_traits>
void TestMoveConstructor(){
    TestMoveConstructorSingleArgument<container_traits>();
    TestMoveConstructorWithEqualAllocator<container_traits>();
    TestMoveConstructorWithUnEqualAllocator<container_traits>();
}

template<typename container_traits>
void TestMoveAssignOperatorPOCMAStateful(){
    typedef typename default_stateful_fixture_make_helper<container_traits, Harness::true_type>::type fixture_t;
    typedef typename fixture_t::container_t container_t;

    fixture_t fixture("TestMoveAssignOperatorPOCMAStateful");

    container_t dst(fixture.dst_allocator);

    fixture.save_allocator_counters();

    dst = std::move(fixture.source);

    fixture.verify_content_shallow_moved(dst);
    fixture.verify_allocator_was_moved(dst.get_allocator());
    fixture.verify_no_more_than_x_memory_items_allocated();
    fixture.verify_no_undestroyed_foo_left_and_dismiss();
}

template<typename container_traits>
void TestMoveAssignOperatorPOCMANonStateful(){
    typedef std::allocator<FooWithAssign>  allocator_t;

    typedef move_fixture<container_traits, allocator_t> fixture_t;
    typedef typename fixture_t::container_t container_t;

    fixture_t fixture("TestMoveAssignOperatorPOCMANonStateful");

    ASSERT(fixture.source.get_allocator() == allocator_t(), "Incorrect test setup: allocator is stateful while should not?");

    container_t dst;
    dst = std::move(fixture.source);

    fixture.verify_content_shallow_moved(dst);
    //TODO: add an assert that allocator was "moved" when POCMA is set
}

template<typename container_traits>
void TestMoveAssignOperatorNotPOCMAWithUnEqualAllocator(){
    typedef typename default_stateful_fixture_make_helper<container_traits>::type fixture_t;
    typedef typename fixture_t::container_t container_t;

    fixture_t fixture("TestMoveAssignOperatorNotPOCMAWithUnEqualAllocator");

    container_t dst(fixture.dst_allocator);
    dst = std::move(fixture.source);

    fixture.verify_content_deep_moved(dst);
}

template<typename container_traits>
void TestMoveAssignOperatorNotPOCMAWithEqualAllocator(){
    typedef typename default_stateful_fixture_make_helper<container_traits, Harness::false_type>::type fixture_t;
    typedef typename fixture_t::container_t container_t;
    fixture_t fixture("TestMoveAssignOperatorNotPOCMAWithEqualAllocator");

    container_t dst(fixture.source_allocator);
    ASSERT(fixture.source.get_allocator() == dst.get_allocator(), "Incorrect test setup: allocators are not equal while should be?");

    fixture.save_allocator_counters();

    dst = std::move(fixture.source);

    fixture.verify_content_shallow_moved(dst);
    fixture.verify_no_more_than_x_memory_items_allocated();
    fixture.verify_no_undestroyed_foo_left_and_dismiss();
}

template<typename container_traits>
void TestMoveAssignOperator(){
#if __TBB_ALLOCATOR_TRAITS_PRESENT
    TestMoveAssignOperatorPOCMANonStateful<container_traits>();
    TestMoveAssignOperatorPOCMAStateful<container_traits>();
#endif
    TestMoveAssignOperatorNotPOCMAWithUnEqualAllocator<container_traits>();
    TestMoveAssignOperatorNotPOCMAWithEqualAllocator<container_traits>();
}

template<typename container_traits>
void TestConstructorWithMoveIterators(){
    typedef typename default_stateful_fixture_make_helper<container_traits>::type fixture_t;
    typedef typename fixture_t::container_t container_t;

    fixture_t fixture("TestConstructorWithMoveIterators");

    container_t dst(std::make_move_iterator(fixture.source.begin()), std::make_move_iterator(fixture.source.end()), fixture.dst_allocator);

    fixture.verify_content_deep_moved(dst);
}

template<typename container_traits>
void TestAssignWithMoveIterators(){
    typedef typename default_stateful_fixture_make_helper<container_traits>::type fixture_t;
    typedef typename fixture_t::container_t container_t;

    fixture_t fixture("TestAssignWithMoveIterators");

    container_t dst(fixture.dst_allocator);
    dst.assign(std::make_move_iterator(fixture.source.begin()), std::make_move_iterator(fixture.source.end()));

    fixture.verify_content_deep_moved(dst);
}

#if  TBB_USE_EXCEPTIONS
template<typename container_traits>
void TestExceptionSafetyGuaranteesMoveConstructorWithUnEqualAllocatorMemoryFailure(){
    typedef typename default_stateful_fixture_make_helper<container_traits>::type fixture_t;
    typedef typename fixture_t::container_t container_t;
    typedef typename container_t::allocator_type allocator_t;
    const char* test_name = "TestExceptionSafetyGuaranteesMoveConstructorWithUnEqualAllocatorMemoryFailure";
    fixture_t fixture(test_name);

    limit_allocated_items_in_scope<allocator_t> allocator_limit(allocator_t::items_allocated + fixture.container_size/4);
    ASSERT_THROWS_IN_TEST(container_t dst(std::move(fixture.source), fixture.dst_allocator), std::bad_alloc, "", test_name);
}

//TODO: add tests that verify that stealing move constructors/assign operators does not throw exceptions
template<typename container_traits>
void TestExceptionSafetyGuaranteesMoveConstructorWithUnEqualAllocatorExceptionInElementCtor(){
    typedef typename default_stateful_fixture_make_helper<container_traits>::type fixture_t;
    typedef typename fixture_t::container_t container_t;

    const char* test_name = "TestExceptionSafetyGuaranteesMoveConstructorWithUnEqualAllocatorExceptionInElementCtor";
    fixture_t fixture(test_name);

    limit_foo_count_in_scope foo_limit(FooCount + fixture.container_size/4);
    ASSERT_THROWS_IN_TEST(container_t dst(std::move(fixture.source), fixture.dst_allocator), std::bad_alloc, "", test_name);
}
#endif /* TBB_USE_EXCEPTIONS */
#endif//__TBB_CPP11_RVALUE_REF_PRESENT

namespace helper_stuff_tests {
    void inline TestArena(){
        typedef int arena_element;

        arena_element arena_storage[10] = {0};
        typedef arena<arena_element> arena_t;

        arena_t::arena_data_t arena_data(arena_storage,Harness::array_length(arena_storage));
        arena_t a(arena_data);

        ASSERT(a.allocate(1) == arena_storage, "");
        ASSERT(a.allocate(2) == &arena_storage[1], "");
        ASSERT(a.allocate(2) == &arena_storage[2+1], "");
    }

    template<typename static_counting_allocator_type>
    void inline TestStaticCountingAllocatorRebound(){
        static_counting_allocator_type::set_limits(1);
        typedef typename static_counting_allocator_type:: template rebind<std::pair<int,int> >::other rebound_type;
        ASSERT(rebound_type::max_items == static_counting_allocator_type::max_items, "rebound allocator should use the same limits");
        static_counting_allocator_type::set_limits(0);
    }

    void inline TestStatefulAllocator(){
        stateful_allocator<int> a1,a2;
        stateful_allocator<int> copy_of_a1(a1);
        ASSERT(a1 != a2,"non_equal_allocator are designed to simulate stateful allocators");
        ASSERT(copy_of_a1 == a1,"");
    }
}
struct TestHelperStuff{
    TestHelperStuff(){
        using namespace helper_stuff_tests;
        TestFoo();
        TestAllOf();
        TestArena();
        TestStaticCountingAllocatorRebound<static_shared_counting_allocator<int, arena<int> > >();
        TestStatefulAllocator();
#if __TBB_CPP11_RVALUE_REF_PRESENT
        TestMemoryLocaionsHelper();
#endif //__TBB_CPP11_RVALUE_REF_PRESENT
    }
};
static TestHelperStuff TestHelperStuff_s;
#endif /* __TBB_test_container_move_support_H */
