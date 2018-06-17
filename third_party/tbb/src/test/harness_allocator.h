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

// Declarations for simple estimate of the memory being used by a program.
// Not yet implemented for macOS*.
// This header is an optional part of the test harness.
// It assumes that "harness_assert.h" has already been included.

#ifndef tbb_test_harness_allocator_H
#define tbb_test_harness_allocator_H

#include "harness_defs.h"

#if __linux__ || __APPLE__ || __sun
#include <unistd.h>
#elif _WIN32
#include "tbb/machine/windows_api.h"
#endif /* OS specific */
#include <memory>
#include <new>
#include <cstdio>
#include <stdexcept>
#include <utility>
#include __TBB_STD_SWAP_HEADER

#include "tbb/atomic.h"

#if __SUNPRO_CC
using std::printf;
#endif

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
    // Workaround for overzealous compiler warnings in /Wp64 mode
    #pragma warning (push)
#if defined(_Wp64)
    #pragma warning (disable: 4267)
#endif
#if _MSC_VER <= 1600
    #pragma warning (disable: 4355)
#endif
#if _MSC_VER <= 1800
    #pragma warning (disable: 4512)
#endif
#endif

#if TBB_INTERFACE_VERSION >= 7005
// Allocator traits were introduced in 4.2 U5
namespace Harness {
#if __TBB_ALLOCATOR_TRAITS_PRESENT
    using std::true_type;
    using std::false_type;
#else
    using tbb::internal::true_type;
    using tbb::internal::false_type;
#endif //__TBB_ALLOCATOR_TRAITS_PRESENT
}
#endif

template<typename counter_type = size_t>
struct arena_data {
    char * const my_buffer;
    size_t const my_size; //in bytes
    counter_type my_allocated; // in bytes

    template<typename T>
    arena_data(T * a_buffer, size_t a_size) __TBB_NOEXCEPT(true)
    :   my_buffer(reinterpret_cast<char*>(a_buffer))
    ,   my_size(a_size * sizeof(T))
    {
        my_allocated =0;
    }
private:
    void operator=( const arena_data& ); // NoAssign is not used to avoid dependency on harness.h
};

template<typename T, typename pocma = Harness::false_type, typename counter_type = size_t>
struct arena {
    typedef arena_data<counter_type> arena_data_t;
private:
    arena_data_t * my_data;
public:
    typedef T value_type;
    typedef value_type* pointer;
    typedef const value_type* const_pointer;
    typedef value_type& reference;
    typedef const value_type& const_reference;
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    template<typename U> struct rebind {
        typedef arena<U, pocma, counter_type> other;
    };

    typedef pocma propagate_on_container_move_assignment;

    arena(arena_data_t & data) __TBB_NOEXCEPT(true) : my_data(&data) {}

    template<typename U1, typename U2, typename U3>
    friend struct arena;

    template<typename U1, typename U2 >
    arena(arena<U1, U2, counter_type> const& other) __TBB_NOEXCEPT(true) : my_data(other.my_data) {}

    friend void swap(arena & lhs ,arena & rhs){
        std::swap(lhs.my_data, rhs.my_data);
    }

    pointer address(reference x) const {return &x;}
    const_pointer address(const_reference x) const {return &x;}

    //! Allocate space for n objects, starting on a cache/sector line.
    pointer allocate( size_type n, const void* =0) {
        size_t new_size = (my_data->my_allocated += n*sizeof(T));
        ASSERT(my_data->my_allocated <= my_data->my_size,"trying to allocate more than was reserved");
        char* result =  &(my_data->my_buffer[new_size - n*sizeof(T)]);
        return reinterpret_cast<pointer>(result);
    }

    //! Free block of memory that starts on a cache line
    void deallocate( pointer p_arg, size_type n) {
        char* p = reinterpret_cast<char*>(p_arg);
        ASSERT(p >=my_data->my_buffer && p <= my_data->my_buffer + my_data->my_size, "trying to deallocate pointer not from arena ?");
        ASSERT(p + n*sizeof(T) <= my_data->my_buffer + my_data->my_size, "trying to deallocate incorrect number of items?");
        tbb::internal::suppress_unused_warning(p, n);
    }

    //! Largest value for which method allocate might succeed.
    size_type max_size() const throw() {
        return my_data->my_size / sizeof(T);
    }

    //! Copy-construct value at location pointed to by p.
#if __TBB_ALLOCATOR_CONSTRUCT_VARIADIC
    template<typename U, typename... Args>
    void construct(U *p, Args&&... args)
        { ::new((void *)p) U(std::forward<Args>(args)...); }
#else // __TBB_ALLOCATOR_CONSTRUCT_VARIADIC
#if __TBB_CPP11_RVALUE_REF_PRESENT
    void construct( pointer p, value_type&& value ) {::new((void*)(p)) value_type(std::move(value));}
#endif
    void construct( pointer p, const value_type& value ) {::new((void*)(p)) value_type(value);}
#endif // __TBB_ALLOCATOR_CONSTRUCT_VARIADIC

    //! Destroy value at location pointed to by p.
    void destroy( pointer p ) {
        p->~value_type();
        // suppress "unreferenced parameter" warnings by MSVC up to and including 2015
        tbb::internal::suppress_unused_warning(p);
    }

    friend bool operator==(arena const& lhs, arena const& rhs){
        return lhs.my_data == rhs.my_data;
    }

    friend bool operator!=(arena const& lhs, arena const& rhs){
        return !(lhs== rhs);
    }
};

template <typename count_t = tbb::atomic<size_t> >
struct allocator_counters {
    count_t items_allocated;
    count_t items_freed;
    count_t allocations;
    count_t frees;

    friend bool operator==(allocator_counters const & lhs, allocator_counters const & rhs){
        return     lhs.items_allocated == rhs.items_allocated
                && lhs.items_freed == rhs.items_freed
                && lhs.allocations == rhs.allocations
                && lhs.frees == rhs.frees
        ;
    }
};

template <typename base_alloc_t, typename count_t = tbb::atomic<size_t> >
class static_counting_allocator : public base_alloc_t
{
public:
    typedef typename base_alloc_t::pointer pointer;
    typedef typename base_alloc_t::const_pointer const_pointer;
    typedef typename base_alloc_t::reference reference;
    typedef typename base_alloc_t::const_reference const_reference;
    typedef typename base_alloc_t::value_type value_type;
    typedef typename base_alloc_t::size_type size_type;
    typedef typename base_alloc_t::difference_type difference_type;
    template<typename U> struct rebind {
        typedef static_counting_allocator<typename base_alloc_t::template rebind<U>::other,count_t> other;
    };

    typedef allocator_counters<count_t> counters_t;

    static size_t max_items;
    static count_t items_allocated;
    static count_t items_freed;
    static count_t allocations;
    static count_t frees;
    static bool verbose, throwing;

    static_counting_allocator() throw() { }

    static_counting_allocator(const base_alloc_t& src) throw()
    : base_alloc_t(src) { }

    static_counting_allocator(const static_counting_allocator& src) throw()
    : base_alloc_t(src) { }

    template<typename U, typename C>
    static_counting_allocator(const static_counting_allocator<U, C>& src) throw()
    : base_alloc_t(src) { }

    pointer allocate(const size_type n)
    {
        if(verbose) printf("\t+%d|", int(n));
        if(max_items && items_allocated + n >= max_items) {
            if(verbose) printf("items limit hits!");
            if(throwing)
                __TBB_THROW( std::bad_alloc() );
            return NULL;
        }
        pointer p = base_alloc_t::allocate(n, pointer(0));
        allocations++;
        items_allocated += n;
        return p;
    }

    pointer allocate(const size_type n, const void * const)
    {   return allocate(n); }

    void deallocate(const pointer ptr, const size_type n)
    {
        if(verbose) printf("\t-%d|", int(n));
        frees++;
        items_freed += n;
        base_alloc_t::deallocate(ptr, n);
    }

    static counters_t counters(){
        counters_t c = {items_allocated, items_freed, allocations, frees} ;
        return c;
    }

    static void init_counters(bool v = false) {
        verbose = v;
        if(verbose) printf("\n------------------------------------------- Allocations:\n");
        items_allocated = 0;
        items_freed = 0;
        allocations = 0;
        frees = 0;
        max_items = 0;
    }

    static void set_limits(size_type max = 0, bool do_throw = true) {
        max_items = max;
        throwing = do_throw;
    }
};

template <typename base_alloc_t, typename count_t>
size_t static_counting_allocator<base_alloc_t, count_t>::max_items;
template <typename base_alloc_t, typename count_t>
count_t static_counting_allocator<base_alloc_t, count_t>::items_allocated;
template <typename base_alloc_t, typename count_t>
count_t static_counting_allocator<base_alloc_t, count_t>::items_freed;
template <typename base_alloc_t, typename count_t>
count_t static_counting_allocator<base_alloc_t, count_t>::allocations;
template <typename base_alloc_t, typename count_t>
count_t static_counting_allocator<base_alloc_t, count_t>::frees;
template <typename base_alloc_t, typename count_t>
bool static_counting_allocator<base_alloc_t, count_t>::verbose;
template <typename base_alloc_t, typename count_t>
bool static_counting_allocator<base_alloc_t, count_t>::throwing;


template <typename tag, typename count_t = tbb::atomic<size_t> >
class static_shared_counting_allocator_base
{
public:
    typedef allocator_counters<count_t> counters_t;

    static size_t max_items;
    static count_t items_allocated;
    static count_t items_freed;
    static count_t allocations;
    static count_t frees;
    static bool verbose, throwing;

    static counters_t counters(){
        counters_t c = {items_allocated, items_freed, allocations, frees} ;
        return c;
    }

    static void init_counters(bool v = false) {
        verbose = v;
        if(verbose) printf("\n------------------------------------------- Allocations:\n");
        items_allocated = 0;
        items_freed = 0;
        allocations = 0;
        frees = 0;
        max_items = 0;
    }

    static void set_limits(size_t max = 0, bool do_throw = true) {
        max_items = max;
        throwing = do_throw;
    }
};

template <typename tag, typename count_t>
size_t static_shared_counting_allocator_base<tag, count_t>::max_items;

template <typename tag, typename count_t>
count_t static_shared_counting_allocator_base<tag, count_t>::items_allocated;

template <typename tag, typename count_t>
count_t static_shared_counting_allocator_base<tag, count_t>::items_freed;

template <typename tag, typename count_t>
count_t static_shared_counting_allocator_base<tag, count_t>::allocations;

template <typename tag, typename count_t>
count_t static_shared_counting_allocator_base<tag, count_t>::frees;

template <typename tag, typename count_t>
bool static_shared_counting_allocator_base<tag, count_t>::verbose;

template <typename tag, typename count_t>
bool static_shared_counting_allocator_base<tag, count_t>::throwing;

template <typename tag, typename base_alloc_t, typename count_t = tbb::atomic<size_t> >
class static_shared_counting_allocator : public static_shared_counting_allocator_base<tag, count_t>, public base_alloc_t
{
    typedef static_shared_counting_allocator_base<tag, count_t> base_t;
public:
    typedef typename base_alloc_t::pointer pointer;
    typedef typename base_alloc_t::const_pointer const_pointer;
    typedef typename base_alloc_t::reference reference;
    typedef typename base_alloc_t::const_reference const_reference;
    typedef typename base_alloc_t::value_type value_type;
    typedef typename base_alloc_t::size_type size_type;
    typedef typename base_alloc_t::difference_type difference_type;
    template<typename U> struct rebind {
        typedef static_shared_counting_allocator<tag, typename base_alloc_t::template rebind<U>::other, count_t> other;
    };

    static_shared_counting_allocator() throw() { }

    static_shared_counting_allocator(const base_alloc_t& src) throw()
    : base_alloc_t(src) { }

    static_shared_counting_allocator(const static_shared_counting_allocator& src) throw()
    : base_alloc_t(src) { }

    template<typename U, typename C>
    static_shared_counting_allocator(const static_shared_counting_allocator<tag, U, C>& src) throw()
    : base_alloc_t(src) { }

    pointer allocate(const size_type n)
    {
        if(base_t::verbose) printf("\t+%d|", int(n));
        if(base_t::max_items && base_t::items_allocated + n >= base_t::max_items) {
            if(base_t::verbose) printf("items limit hits!");
            if(base_t::throwing)
                __TBB_THROW( std::bad_alloc() );
            return NULL;
        }
        base_t::allocations++;
        base_t::items_allocated += n;
        return base_alloc_t::allocate(n, pointer(0));
    }

    pointer allocate(const size_type n, const void * const)
    {   return allocate(n); }

    void deallocate(const pointer ptr, const size_type n)
    {
        if(base_t::verbose) printf("\t-%d|", int(n));
        base_t::frees++;
        base_t::items_freed += n;
        base_alloc_t::deallocate(ptr, n);
    }
};

template <typename base_alloc_t, typename count_t = tbb::atomic<size_t> >
class local_counting_allocator : public base_alloc_t
{
public:
    typedef typename base_alloc_t::pointer pointer;
    typedef typename base_alloc_t::const_pointer const_pointer;
    typedef typename base_alloc_t::reference reference;
    typedef typename base_alloc_t::const_reference const_reference;
    typedef typename base_alloc_t::value_type value_type;
    typedef typename base_alloc_t::size_type size_type;
    typedef typename base_alloc_t::difference_type difference_type;
    template<typename U> struct rebind {
        typedef local_counting_allocator<typename base_alloc_t::template rebind<U>::other,count_t> other;
    };

    count_t items_allocated;
    count_t items_freed;
    count_t allocations;
    count_t frees;
    size_t max_items;

    void set_counters(const count_t & a_items_allocated, const count_t & a_items_freed, const count_t & a_allocations, const count_t & a_frees, const count_t & a_max_items){
        items_allocated = a_items_allocated;
        items_freed = a_items_freed;
        allocations = a_allocations;
        frees = a_frees;
        max_items = a_max_items;
    }

    template< typename allocator_t>
    void set_counters(const allocator_t & a){
        this->set_counters(a.items_allocated, a.items_freed, a.allocations, a.frees, a.max_items);
    }

    void clear_counters(){
        count_t zero;
        zero = 0;
        this->set_counters(zero,zero,zero,zero,zero);
    }

    local_counting_allocator() throw() {
        this->clear_counters();
    }

    local_counting_allocator(const local_counting_allocator &a) throw()
        : base_alloc_t(a)
        , items_allocated(a.items_allocated)
        , items_freed(a.items_freed)
        , allocations(a.allocations)
        , frees(a.frees)
        , max_items(a.max_items)
    { }

    template<typename U, typename C>
    local_counting_allocator(const static_counting_allocator<U,C> & a) throw() {
        this->set_counters(a);
    }

    template<typename U, typename C>
    local_counting_allocator(const local_counting_allocator<U,C> &a) throw()
        : items_allocated(a.items_allocated)
        , items_freed(a.items_freed)
        , allocations(a.allocations)
        , frees(a.frees)
        , max_items(a.max_items)
    { }

    bool operator==(const local_counting_allocator &a) const
    { return static_cast<const base_alloc_t&>(a) == *this; }

    pointer allocate(const size_type n)
    {
        if(max_items && items_allocated + n >= max_items)
            __TBB_THROW( std::bad_alloc() );
        pointer p = base_alloc_t::allocate(n, pointer(0));
        ++allocations;
        items_allocated += n;
        return p;
    }

    pointer allocate(const size_type n, const void * const)
    { return allocate(n); }

    void deallocate(const pointer ptr, const size_type n)
    {
        ++frees;
        items_freed += n;
        base_alloc_t::deallocate(ptr, n);
    }

    void set_limits(size_type max = 0) {
        max_items = max;
    }
};

template <typename T, template<typename X> class Allocator = std::allocator>
class debug_allocator : public Allocator<T>
{
public:
    typedef Allocator<T> base_allocator_type;
    typedef typename base_allocator_type::value_type value_type;
    typedef typename base_allocator_type::pointer pointer;
    typedef typename base_allocator_type::const_pointer const_pointer;
    typedef typename base_allocator_type::reference reference;
    typedef typename base_allocator_type::const_reference const_reference;
    typedef typename base_allocator_type::size_type size_type;
    typedef typename base_allocator_type::difference_type difference_type;
    template<typename U> struct rebind {
        typedef debug_allocator<U, Allocator> other;
    };

    debug_allocator() throw() { }
    debug_allocator(const debug_allocator &a) throw() : base_allocator_type( a ) { }
    template<typename U>
    debug_allocator(const debug_allocator<U> &a) throw() : base_allocator_type( Allocator<U>( a ) ) { }

    pointer allocate(const size_type n, const void *hint = 0 ) {
        pointer ptr = base_allocator_type::allocate( n, hint );
        std::memset( ptr, 0xE3E3E3E3, n * sizeof(value_type) );
        return ptr;
    }
};

//! Analogous to std::allocator<void>, as defined in ISO C++ Standard, Section 20.4.1
/** @ingroup memory_allocation */
template<template<typename T> class Allocator>
class debug_allocator<void, Allocator> : public Allocator<void> {
public:
    typedef Allocator<void> base_allocator_type;
    typedef typename base_allocator_type::value_type value_type;
    typedef typename base_allocator_type::pointer pointer;
    typedef typename base_allocator_type::const_pointer const_pointer;
    template<typename U> struct rebind {
        typedef debug_allocator<U, Allocator> other;
    };
};

template<typename T1, template<typename X1> class B1, typename T2, template<typename X2> class B2>
inline bool operator==( const debug_allocator<T1,B1> &a, const debug_allocator<T2,B2> &b) {
    return static_cast< B1<T1> >(a) == static_cast< B2<T2> >(b);
}
template<typename T1, template<typename X1> class B1, typename T2, template<typename X2> class B2>
inline bool operator!=( const debug_allocator<T1,B1> &a, const debug_allocator<T2,B2> &b) {
    return static_cast< B1<T1> >(a) != static_cast< B2<T2> >(b);
}

template <typename T, typename pocma = Harness::false_type, template<typename X> class Allocator = std::allocator>
class stateful_allocator : public Allocator<T>
{
    void* unique_pointer;

    template<typename T1, typename pocma1, template<typename X1> class Allocator1>
    friend class  stateful_allocator;
public:
    typedef Allocator<T> base_allocator_type;
    typedef typename base_allocator_type::value_type value_type;
    typedef typename base_allocator_type::pointer pointer;
    typedef typename base_allocator_type::const_pointer const_pointer;
    typedef typename base_allocator_type::reference reference;
    typedef typename base_allocator_type::const_reference const_reference;
    typedef typename base_allocator_type::size_type size_type;
    typedef typename base_allocator_type::difference_type difference_type;
    template<typename U> struct rebind {
        typedef stateful_allocator<U, pocma, Allocator> other;
    };
    typedef pocma propagate_on_container_move_assignment;

    stateful_allocator() throw() : unique_pointer(this) { }

    template<typename U>
    stateful_allocator(const stateful_allocator<U, pocma> &a) throw() : base_allocator_type( Allocator<U>( a ) ),  unique_pointer(a.uniqe_pointer) { }

    friend bool operator==(stateful_allocator const& lhs, stateful_allocator const& rhs){
        return lhs.unique_pointer == rhs.unique_pointer;
    }

    friend bool operator!=(stateful_allocator const& rhs, stateful_allocator const& lhs){
        return !(lhs == rhs);
    }

};

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
    // Workaround for overzealous compiler warnings
    #pragma warning (pop)
#endif // warning 4267,4512,4355 is back

namespace Harness {

    struct IsEqual {
#if __TBB_CPP11_SMART_POINTERS_PRESENT
        template <typename T>
        static bool compare( const std::weak_ptr<T> &t1, const std::weak_ptr<T> &t2 ) {
            // Compare real pointers.
            return t1.lock().get() == t2.lock().get();
        }
        template <typename T>
        static bool compare( const std::unique_ptr<T> &t1, const std::unique_ptr<T> &t2 ) {
            // Compare real values.
            return *t1 == *t2;
        }
        template <typename T1, typename T2>
        static bool compare( const std::pair< const std::weak_ptr<T1>, std::weak_ptr<T2> > &t1,
                const std::pair< const std::weak_ptr<T1>, std::weak_ptr<T2> > &t2 ) {
            // Compare real pointers.
            return t1.first.lock().get() == t2.first.lock().get() &&
                t1.second.lock().get() == t2.second.lock().get();
        }
#endif /* __TBB_CPP11_SMART_POINTERS_PRESENT */
        template <typename T1, typename T2>
        static bool compare( const T1 &t1, const T2 &t2 ) {
            return t1 == t2;
        }
        template <typename T1, typename T2>
        bool operator()( T1 &t1, T2 &t2) const {
            return compare( (const T1&)t1, (const T2&)t2 );
        }
    };

} // Harness
#endif // tbb_test_harness_allocator_H
