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

#ifndef __TBB_concurrent_vector_H
#define __TBB_concurrent_vector_H

#include "tbb/tbb_stddef.h"
#include "tbb/atomic.h"
#include "tbb/cache_aligned_allocator.h"
#include "tbb/blocked_range.h"
#include "tbb/tbb_machine.h"
#include <new>
#include <iterator>

namespace tbb {

template<typename T>
class concurrent_vector;

//! @cond INTERNAL
namespace internal {

    //! Base class of concurrent vector implementation.
    /** @ingroup containers */
    class concurrent_vector_base {
    protected:

        // Basic types declarations
        typedef unsigned long segment_index_t;
        typedef size_t size_type;

        //! Log2 of "min_segment_size".
        static const int lg_min_segment_size = 4;

        //! Minimum size (in physical items) of a segment.
        static const int min_segment_size = segment_index_t(1)<<lg_min_segment_size;

        static segment_index_t segment_index_of( size_t index ) {
            uintptr_t i = index|1<<(lg_min_segment_size-1);
            uintptr_t j = __TBB_Log2(i);
            return segment_index_t(j-(lg_min_segment_size-1));
        }

        static segment_index_t segment_base( segment_index_t k ) {
            return min_segment_size>>1<<k & -min_segment_size;
        }

        static segment_index_t segment_size( segment_index_t k ) {
            segment_index_t result = k==0 ? min_segment_size : min_segment_size/2<<k;
            __TBB_ASSERT( result==segment_base(k+1)-segment_base(k), NULL );
            return result;
        }

        void __TBB_EXPORTED_METHOD internal_reserve( size_type n, size_type element_size, size_type max_size );

        size_type __TBB_EXPORTED_METHOD internal_capacity() const;

        //! Requested size of vector
        atomic<size_type> my_early_size;

        /** Can be zero-initialized. */
        struct segment_t {
            /** Declared volatile because in weak memory model, must have ld.acq/st.rel  */
            void* volatile array;
#if TBB_USE_ASSERT
            ~segment_t() {
                __TBB_ASSERT( !array, "should have been set to NULL by clear" );
            }
#endif /* TBB_USE_ASSERT */
        };

        // Data fields

        //! Pointer to the segments table
        atomic<segment_t*> my_segment;

        //! embedded storage of segment pointers
        segment_t my_storage[2];

        // Methods

        concurrent_vector_base() {
            my_early_size = 0;
            my_storage[0].array = NULL;
            my_storage[1].array = NULL;
            my_segment = my_storage;
        }

        //! An operation on an n-element array starting at begin.
        typedef void(__TBB_EXPORTED_FUNC *internal_array_op1)(void* begin, size_type n );

        //! An operation on n-element destination array and n-element source array.
        typedef void(__TBB_EXPORTED_FUNC *internal_array_op2)(void* dst, const void* src, size_type n );

        void __TBB_EXPORTED_METHOD internal_grow_to_at_least( size_type new_size, size_type element_size, internal_array_op1 init );
        void internal_grow( size_type start, size_type finish, size_type element_size, internal_array_op1 init );
        size_type __TBB_EXPORTED_METHOD internal_grow_by( size_type delta, size_type element_size, internal_array_op1 init );
        void* __TBB_EXPORTED_METHOD internal_push_back( size_type element_size, size_type& index );
        void __TBB_EXPORTED_METHOD internal_clear( internal_array_op1 destroy, bool reclaim_storage );
        void __TBB_EXPORTED_METHOD internal_copy( const concurrent_vector_base& src, size_type element_size, internal_array_op2 copy );
        void __TBB_EXPORTED_METHOD internal_assign( const concurrent_vector_base& src, size_type element_size,
                              internal_array_op1 destroy, internal_array_op2 assign, internal_array_op2 copy );
private:
        //! Private functionality that does not cross DLL boundary.
        class helper;
        friend class helper;
    };

    //! Meets requirements of a forward iterator for STL and a Value for a blocked_range.*/
    /** Value is either the T or const T type of the container.
        @ingroup containers */
    template<typename Container, typename Value>
    class vector_iterator
#if defined(_WIN64) && defined(_MSC_VER)
        // Ensure that Microsoft's internal template function _Val_type works correctly.
        : public std::iterator<std::random_access_iterator_tag,Value>
#endif /* defined(_WIN64) && defined(_MSC_VER) */
    {
        //! concurrent_vector over which we are iterating.
        Container* my_vector;

        //! Index into the vector
        size_t my_index;

        //! Caches my_vector-&gt;internal_subscript(my_index)
        /** NULL if cached value is not available */
        mutable Value* my_item;

        template<typename C, typename T, typename U>
        friend bool operator==( const vector_iterator<C,T>& i, const vector_iterator<C,U>& j );

        template<typename C, typename T, typename U>
        friend bool operator<( const vector_iterator<C,T>& i, const vector_iterator<C,U>& j );

        template<typename C, typename T, typename U>
        friend ptrdiff_t operator-( const vector_iterator<C,T>& i, const vector_iterator<C,U>& j );

        template<typename C, typename U>
        friend class internal::vector_iterator;

#if !defined(_MSC_VER) || defined(__INTEL_COMPILER)
        template<typename T>
        friend class tbb::concurrent_vector;
#else
public: // workaround for MSVC
#endif

        vector_iterator( const Container& vector, size_t index ) :
            my_vector(const_cast<Container*>(&vector)),
            my_index(index),
            my_item(NULL)
        {}

    public:
        //! Default constructor
        vector_iterator() : my_vector(NULL), my_index(~size_t(0)), my_item(NULL) {}

        vector_iterator( const vector_iterator<Container,typename Container::value_type>& other ) :
            my_vector(other.my_vector),
            my_index(other.my_index),
            my_item(other.my_item)
        {}

        vector_iterator operator+( ptrdiff_t offset ) const {
            return vector_iterator( *my_vector, my_index+offset );
        }
        friend vector_iterator operator+( ptrdiff_t offset, const vector_iterator& v ) {
            return vector_iterator( *v.my_vector, v.my_index+offset );
        }
        vector_iterator operator+=( ptrdiff_t offset ) {
            my_index+=offset;
            my_item = NULL;
            return *this;
        }
        vector_iterator operator-( ptrdiff_t offset ) const {
            return vector_iterator( *my_vector, my_index-offset );
        }
        vector_iterator operator-=( ptrdiff_t offset ) {
            my_index-=offset;
            my_item = NULL;
            return *this;
        }
        Value& operator*() const {
            Value* item = my_item;
            if( !item ) {
                item = my_item = &my_vector->internal_subscript(my_index);
            }
            __TBB_ASSERT( item==&my_vector->internal_subscript(my_index), "corrupt cache" );
            return *item;
        }
        Value& operator[]( ptrdiff_t k ) const {
            return my_vector->internal_subscript(my_index+k);
        }
        Value* operator->() const {return &operator*();}

        //! Pre increment
        vector_iterator& operator++() {
            size_t k = ++my_index;
            if( my_item ) {
                // Following test uses 2's-complement wizardry and fact that
                // min_segment_size is a power of 2.
                if( (k& k-concurrent_vector<Container>::min_segment_size)==0 ) {
                    // k is a power of two that is at least k-min_segment_size  
                    my_item= NULL;
                } else {
                    ++my_item;
                }
            }
            return *this;
        }

        //! Pre decrement
        vector_iterator& operator--() {
            __TBB_ASSERT( my_index>0, "operator--() applied to iterator already at beginning of concurrent_vector" );
            size_t k = my_index--;
            if( my_item ) {
                // Following test uses 2's-complement wizardry and fact that
                // min_segment_size is a power of 2.
                if( (k& k-concurrent_vector<Container>::min_segment_size)==0 ) {
                    // k is a power of two that is at least k-min_segment_size  
                    my_item= NULL;
                } else {
                    --my_item;
                }
            }
            return *this;
        }

        //! Post increment
        vector_iterator operator++(int) {
            vector_iterator result = *this;
            operator++();
            return result;
        }

        //! Post decrement
        vector_iterator operator--(int) {
            vector_iterator result = *this;
            operator--();
            return result;
        }

        // STL support

        typedef ptrdiff_t difference_type;
        typedef Value value_type;
        typedef Value* pointer;
        typedef Value& reference;
        typedef std::random_access_iterator_tag iterator_category;
    };

    template<typename Container, typename T, typename U>
    bool operator==( const vector_iterator<Container,T>& i, const vector_iterator<Container,U>& j ) {
        return i.my_index==j.my_index;
    }

    template<typename Container, typename T, typename U>
    bool operator!=( const vector_iterator<Container,T>& i, const vector_iterator<Container,U>& j ) {
        return !(i==j);
    }

    template<typename Container, typename T, typename U>
    bool operator<( const vector_iterator<Container,T>& i, const vector_iterator<Container,U>& j ) {
        return i.my_index<j.my_index;
    }

    template<typename Container, typename T, typename U>
    bool operator>( const vector_iterator<Container,T>& i, const vector_iterator<Container,U>& j ) {
        return j<i;
    }

    template<typename Container, typename T, typename U>
    bool operator>=( const vector_iterator<Container,T>& i, const vector_iterator<Container,U>& j ) {
        return !(i<j);
    }

    template<typename Container, typename T, typename U>
    bool operator<=( const vector_iterator<Container,T>& i, const vector_iterator<Container,U>& j ) {
        return !(j<i);
    }

    template<typename Container, typename T, typename U>
    ptrdiff_t operator-( const vector_iterator<Container,T>& i, const vector_iterator<Container,U>& j ) {
        return ptrdiff_t(i.my_index)-ptrdiff_t(j.my_index);
    }

} // namespace internal
//! @endcond

//! Concurrent vector
/** @ingroup containers */
template<typename T>
class concurrent_vector: private internal::concurrent_vector_base {
public:
    using internal::concurrent_vector_base::size_type;
private:
    template<typename I>
    class generic_range_type: public blocked_range<I> {
    public:
        typedef T value_type;
        typedef T& reference;
        typedef const T& const_reference;
        typedef I iterator;
        typedef ptrdiff_t difference_type;
        generic_range_type( I begin_, I end_, size_t grainsize_ ) : blocked_range<I>(begin_,end_,grainsize_) {}
        generic_range_type( generic_range_type& r, split ) : blocked_range<I>(r,split()) {}
    };

    template<typename C, typename U>
    friend class internal::vector_iterator;
public:
    typedef T& reference;
    typedef const T& const_reference;
    typedef T value_type;
    typedef ptrdiff_t difference_type;

    //! Construct empty vector.
    concurrent_vector() {}

    //! Copy a vector.
    concurrent_vector( const concurrent_vector& vector ) : internal::concurrent_vector_base()
    { internal_copy(vector,sizeof(T),&copy_array); }

    //! Assignment
    concurrent_vector& operator=( const concurrent_vector& vector ) {
        if( this!=&vector )
            internal_assign(vector,sizeof(T),&destroy_array,&assign_array,&copy_array);
        return *this;
    }

    //! Clear and destroy vector.
    ~concurrent_vector() {internal_clear(&destroy_array,/*reclaim_storage=*/true);}

    //------------------------------------------------------------------------
    // Concurrent operations
    //------------------------------------------------------------------------
    //! Grow by "delta" elements.
    /** Returns old size. */
    size_type grow_by( size_type delta ) {
        return delta ? internal_grow_by( delta, sizeof(T), &initialize_array ) : my_early_size.load();
    }

    //! Grow array until it has at least n elements.
    void grow_to_at_least( size_type n ) {
        if( my_early_size<n )
            internal_grow_to_at_least( n, sizeof(T), &initialize_array );
    };

    //! Push item
    size_type push_back( const_reference item ) {
        size_type k;
        new( internal_push_back(sizeof(T),k) ) T(item);
        return k;
    }

    //! Get reference to element at given index.
    /** This method is thread-safe for concurrent reads, and also while growing the vector,
        as long as the calling thread has checked that index&lt;size(). */
    reference operator[]( size_type index ) {
        return internal_subscript(index);
    }

    //! Get const reference to element at given index.
    const_reference operator[]( size_type index ) const {
        return internal_subscript(index);
    }

    //------------------------------------------------------------------------
    // STL support (iterators)
    //------------------------------------------------------------------------
    typedef internal::vector_iterator<concurrent_vector,T> iterator;
    typedef internal::vector_iterator<concurrent_vector,const T> const_iterator;

#if !defined(_MSC_VER) || _CPPLIB_VER>=300
    // Assume ISO standard definition of std::reverse_iterator
    typedef std::reverse_iterator<iterator> reverse_iterator;
    typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
#else
    // Use non-standard std::reverse_iterator
    typedef std::reverse_iterator<iterator,T,T&,T*> reverse_iterator;
    typedef std::reverse_iterator<const_iterator,T,const T&,const T*> const_reverse_iterator;
#endif /* defined(_MSC_VER) && (_MSC_VER<1300) */

    // Forward sequence 
    iterator begin() {return iterator(*this,0);}
    iterator end() {return iterator(*this,size());}
    const_iterator begin() const {return const_iterator(*this,0);}
    const_iterator end() const {return const_iterator(*this,size());}

    // Reverse sequence
    reverse_iterator rbegin() {return reverse_iterator(end());}
    reverse_iterator rend() {return reverse_iterator(begin());}
    const_reverse_iterator rbegin() const {return const_reverse_iterator(end());}
    const_reverse_iterator rend() const {return const_reverse_iterator(begin());}

    //------------------------------------------------------------------------
    // Support for TBB algorithms (ranges)
    //------------------------------------------------------------------------
    typedef generic_range_type<iterator> range_type;
    typedef generic_range_type<const_iterator> const_range_type;

    //! Get range to use with parallel algorithms
    range_type range( size_t grainsize = 1 ) {
        return range_type( begin(), end(), grainsize );
    }

    //! Get const range for iterating with parallel algorithms
    const_range_type range( size_t grainsize = 1 ) const {
        return const_range_type( begin(), end(), grainsize );
    }

    //------------------------------------------------------------------------
    // Size and capacity
    //------------------------------------------------------------------------
    //! Return size of vector.
    size_type size() const {return my_early_size;}

    //! Return false if vector is not empty.
    bool empty() const {return !my_early_size;}

    //! Maximum size to which array can grow without allocating more memory.
    size_type capacity() const {return internal_capacity();}

    //! Allocate enough space to grow to size n without having to allocate more memory later.
    /** Like most of the methods provided for STL compatibility, this method is *not* thread safe.
        The capacity afterwards may be bigger than the requested reservation. */
    void reserve( size_type n ) {
        if( n )
            internal_reserve(n, sizeof(T), max_size());
    }

    //! Upper bound on argument to reserve.
    size_type max_size() const {return (~size_t(0))/sizeof(T);}

    //! Not thread safe
    /** Does not change capacity. */
    void clear() {internal_clear(&destroy_array,/*reclaim_storage=*/false);}
private:
    //! Get reference to element at given index.
    T& internal_subscript( size_type index ) const;

    //! Construct n instances of T, starting at "begin".
    static void __TBB_EXPORTED_FUNC initialize_array( void* begin, size_type n );

    //! Construct n instances of T, starting at "begin".
    static void __TBB_EXPORTED_FUNC copy_array( void* dst, const void* src, size_type n );

    //! Assign n instances of T, starting at "begin".
    static void __TBB_EXPORTED_FUNC assign_array( void* dst, const void* src, size_type n );

    //! Destroy n instances of T, starting at "begin".
    static void __TBB_EXPORTED_FUNC destroy_array( void* begin, size_type n );
};

template<typename T>
T& concurrent_vector<T>::internal_subscript( size_type index ) const {
    __TBB_ASSERT( index<size(), "index out of bounds" );
    segment_index_t k = segment_index_of( index );
    size_type j = index-segment_base(k);
    return static_cast<T*>(my_segment[k].array)[j];
}

template<typename T>
void concurrent_vector<T>::initialize_array( void* begin, size_type n ) {
    T* array = static_cast<T*>(begin);
    for( size_type j=0; j<n; ++j )
        new( &array[j] ) T();
}

template<typename T>
void concurrent_vector<T>::copy_array( void* dst, const void* src, size_type n ) {
    T* d = static_cast<T*>(dst);
    const T* s = static_cast<const T*>(src);
    for( size_type j=0; j<n; ++j )
        new( &d[j] ) T(s[j]);
}

template<typename T>
void concurrent_vector<T>::assign_array( void* dst, const void* src, size_type n ) {
    T* d = static_cast<T*>(dst);
    const T* s = static_cast<const T*>(src);
    for( size_type j=0; j<n; ++j )
        d[j] = s[j];
}

template<typename T>
void concurrent_vector<T>::destroy_array( void* begin, size_type n ) {
    T* array = static_cast<T*>(begin);
    for( size_type j=n; j>0; --j )
        array[j-1].~T();
}

} // namespace tbb

#endif /* __TBB_concurrent_vector_H */
