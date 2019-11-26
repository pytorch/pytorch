//
// Copyright (c) 2016-2018 Martin Moene
//
// https://github.com/martinmoene/any-lite
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// C10
// - Move to `c10` namespace.
// - `struct in_place_t {}` is moved to `c10/util/in_place.h`,
//   so that it can also be used in other backports such as
//   `c10::optional` and `c10::variant`.
// - The following `in_place` functions are removed, because in C++17 `std::in_place`
//   is an object instead.
//   ```
//   template< class T >
//   inline in_place_t in_place( detail::in_place_type_tag<T> = detail::in_place_type_tag<T>() )
//   {
//       return in_place_t();
//   }
//   template< std::size_t K >
//   inline in_place_t in_place( detail::in_place_index_tag<K> = detail::in_place_index_tag<K>() )
//   {
//       return in_place_t();
//   }
//   ```
// - Namespace `any_lite::detail` is renamed to `any_lite::any_lite_detail`, to avoid compile errors
//   such as:
//   ```
//   ../aten/src/ATen/core/Dict.h:223:30: error: reference to ‘detail’ is ambiguous
//      using size_type = typename detail::DictImpl::dict_map_type::size_type;
//                                 ^
//   In file included from ../c10/util/Exception.h:5:0,
//                    from ../c10/core/Device.h:5,
//                    from ../c10/core/Allocator.h:6,
//                    from ../c10/core/CPUAllocator.h:6,
//                    from ../caffe2/core/allocator.h:3,
//                    from ../caffe2/core/storage.h:12,
//                    from ../caffe2/core/tensor.h:5,
//                    from ../caffe2/core/test_utils.h:4,
//                    from ../caffe2/predictor/emulator/data_filler_test.cc:2:
//   ../c10/util/StringUtil.h:15:18: note: candidates are: namespace c10::detail { }
//    namespace detail {
//                     ^
//   In file included from aten/src/ATen/core/TensorBody.h:16:0,
//                    from ../aten/src/ATen/core/Tensor.h:11,
//                    from ../caffe2/core/tensor.h:11,
//                    from ../caffe2/core/test_utils.h:4,
//                    from ../caffe2/predictor/emulator/data_filler_test.cc:2:
//   ../c10/util/any.h:385:18: note:                 namespace c10::any_lite::detail { }
//    namespace detail {
//                     ^
//   ```

#pragma once

#ifndef C10_UTIL_ANY_H_
#define C10_UTIL_ANY_H_

#include <c10/util/in_place.h>

#define any_lite_MAJOR  0
#define any_lite_MINOR  2
#define any_lite_PATCH  0

#define any_lite_VERSION  any_STRINGIFY(any_lite_MAJOR) "." any_STRINGIFY(any_lite_MINOR) "." any_STRINGIFY(any_lite_PATCH)

#define any_STRINGIFY(  x )  any_STRINGIFY_( x )
#define any_STRINGIFY_( x )  #x

// any-lite configuration:

#define any_ANY_DEFAULT  0
#define any_ANY_NONSTD   1
#define any_ANY_STD      2

#if !defined( any_CONFIG_SELECT_ANY )
# define any_CONFIG_SELECT_ANY  ( any_HAVE_STD_ANY ? any_ANY_STD : any_ANY_NONSTD )
#endif

// Control presence of exception handling (try and auto discover):

#ifndef any_CONFIG_NO_EXCEPTIONS
# if defined(__cpp_exceptions) || defined(__EXCEPTIONS) || defined(_CPPUNWIND)
#  define any_CONFIG_NO_EXCEPTIONS  0
# else
#  define any_CONFIG_NO_EXCEPTIONS  1
# endif
#endif

// C++ language version detection (C++20 is speculative):
// Note: VC14.0/1900 (VS2015) lacks too much from C++14.

#ifndef   any_CPLUSPLUS
# if defined(_MSVC_LANG ) && !defined(__clang__)
#  define any_CPLUSPLUS  (_MSC_VER == 1900 ? 201103L : _MSVC_LANG )
# else
#  define any_CPLUSPLUS  __cplusplus
# endif
#endif

#define any_CPP98_OR_GREATER  ( any_CPLUSPLUS >= 199711L )
#define any_CPP11_OR_GREATER  ( any_CPLUSPLUS >= 201103L )
#define any_CPP14_OR_GREATER  ( any_CPLUSPLUS >= 201402L )
#define any_CPP17_OR_GREATER  ( any_CPLUSPLUS >= 201703L )
#define any_CPP20_OR_GREATER  ( any_CPLUSPLUS >= 202000L )

// Use C++17 std::any if available and requested:

#if any_CPP17_OR_GREATER && defined(__has_include )
# if __has_include( <any> )
#  define any_HAVE_STD_ANY  1
# else
#  define any_HAVE_STD_ANY  0
# endif
#else
# define  any_HAVE_STD_ANY  0
#endif

#define  any_USES_STD_ANY  ( (any_CONFIG_SELECT_ANY == any_ANY_STD) || ((any_CONFIG_SELECT_ANY == any_ANY_DEFAULT) && any_HAVE_STD_ANY) )

//
// in_place: code duplicated in any-lite, expected-lite, optional-lite, value-ptr-lite, variant-lite:
//

#ifndef nonstd_lite_HAVE_IN_PLACE_TYPES
#define nonstd_lite_HAVE_IN_PLACE_TYPES  1

// C++17 std::in_place in <utility>:

#if any_CPP17_OR_GREATER

#include <utility>

namespace c10 {

using std::in_place;
using std::in_place_type;
using std::in_place_index;
using std::in_place_t;
using std::in_place_type_t;
using std::in_place_index_t;

#define nonstd_lite_in_place_t(      T)  std::in_place_t
#define nonstd_lite_in_place_type_t( T)  std::in_place_type_t<T>
#define nonstd_lite_in_place_index_t(K)  std::in_place_index_t<K>

#define nonstd_lite_in_place(      T)    std::in_place_t{}
#define nonstd_lite_in_place_type( T)    std::in_place_type_t<T>{}
#define nonstd_lite_in_place_index(K)    std::in_place_index_t<K>{}

} // namespace nonstd

#else // any_CPP17_OR_GREATER

#include <cstddef>

namespace c10 {
namespace detail {

template< class T >
struct in_place_type_tag {};

template< std::size_t K >
struct in_place_index_tag {};

} // namespace detail

template< class T >
inline in_place_t in_place_type( detail::in_place_type_tag<T> = detail::in_place_type_tag<T>() )
{
    return in_place_t();
}

template< std::size_t K >
inline in_place_t in_place_index( detail::in_place_index_tag<K> = detail::in_place_index_tag<K>() )
{
    return in_place_t();
}

// mimic templated typedef:

#define nonstd_lite_in_place_t(      T)  c10::in_place_t(&)( c10::detail::in_place_type_tag<T>  )
#define nonstd_lite_in_place_type_t( T)  c10::in_place_t(&)( c10::detail::in_place_type_tag<T>  )
#define nonstd_lite_in_place_index_t(K)  c10::in_place_t(&)( c10::detail::in_place_index_tag<K> )

#define nonstd_lite_in_place(      T)    c10::in_place_type<T>
#define nonstd_lite_in_place_type( T)    c10::in_place_type<T>
#define nonstd_lite_in_place_index(K)    c10::in_place_index<K>

} // namespace nonstd

#endif // any_CPP17_OR_GREATER
#endif // nonstd_lite_HAVE_IN_PLACE_TYPES

//
// Using std::any:
//

#if any_USES_STD_ANY

#include <any>
#include <utility>

namespace c10 {

    using std::any;
    using std::any_cast;
    using std::make_any;
    using std::swap;
    using std::bad_any_cast;
}

#else // any_USES_STD_ANY

#include <utility>

// Compiler versions:
//
// MSVC++ 6.0  _MSC_VER == 1200 (Visual Studio 6.0)
// MSVC++ 7.0  _MSC_VER == 1300 (Visual Studio .NET 2002)
// MSVC++ 7.1  _MSC_VER == 1310 (Visual Studio .NET 2003)
// MSVC++ 8.0  _MSC_VER == 1400 (Visual Studio 2005)
// MSVC++ 9.0  _MSC_VER == 1500 (Visual Studio 2008)
// MSVC++ 10.0 _MSC_VER == 1600 (Visual Studio 2010)
// MSVC++ 11.0 _MSC_VER == 1700 (Visual Studio 2012)
// MSVC++ 12.0 _MSC_VER == 1800 (Visual Studio 2013)
// MSVC++ 14.0 _MSC_VER == 1900 (Visual Studio 2015)
// MSVC++ 14.1 _MSC_VER >= 1910 (Visual Studio 2017)

#if defined(_MSC_VER ) && !defined(__clang__)
# define any_COMPILER_MSVC_VER      (_MSC_VER )
# define any_COMPILER_MSVC_VERSION  (_MSC_VER / 10 - 10 * ( 5 + (_MSC_VER < 1900 ) ) )
#else
# define any_COMPILER_MSVC_VER      0
# define any_COMPILER_MSVC_VERSION  0
#endif

#define any_COMPILER_VERSION( major, minor, patch )  ( 10 * ( 10 * (major) + (minor) ) + (patch) )

#if defined(__clang__)
# define any_COMPILER_CLANG_VERSION  any_COMPILER_VERSION(__clang_major__, __clang_minor__, __clang_patchlevel__)
#else
# define any_COMPILER_CLANG_VERSION  0
#endif

#if defined(__GNUC__) && !defined(__clang__)
# define any_COMPILER_GNUC_VERSION  any_COMPILER_VERSION(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__)
#else
# define any_COMPILER_GNUC_VERSION  0
#endif

// half-open range [lo..hi):
//#define any_BETWEEN( v, lo, hi ) ( (lo) <= (v) && (v) < (hi) )

// Presence of language and library features:

#define any_HAVE( feature )  ( any_HAVE_##feature )

#ifdef _HAS_CPP0X
# define any_HAS_CPP0X  _HAS_CPP0X
#else
# define any_HAS_CPP0X  0
#endif

#define any_CPP11_90   (any_CPP11_OR_GREATER || any_COMPILER_MSVC_VER >= 1500)
#define any_CPP11_100  (any_CPP11_OR_GREATER || any_COMPILER_MSVC_VER >= 1600)
#define any_CPP11_120  (any_CPP11_OR_GREATER || any_COMPILER_MSVC_VER >= 1800)
#define any_CPP11_140  (any_CPP11_OR_GREATER || any_COMPILER_MSVC_VER >= 1900)

#define any_CPP14_000  (any_CPP14_OR_GREATER)
#define any_CPP17_000  (any_CPP17_OR_GREATER)

// Presence of C++11 language features:

#define any_HAVE_CONSTEXPR_11           any_CPP11_140
#define any_HAVE_DEFAULT_FUNCTION_TEMPLATE_ARG \
                                        any_CPP11_120
#define any_HAVE_INITIALIZER_LIST       any_CPP11_120
#define any_HAVE_NOEXCEPT               any_CPP11_140
#define any_HAVE_NULLPTR                any_CPP11_100
#define any_HAVE_TYPE_TRAITS            any_CPP11_90
#define any_HAVE_STATIC_ASSERT          any_CPP11_100
#define any_HAVE_ADD_CONST              any_CPP11_90
#define any_HAVE_REMOVE_REFERENCE       any_CPP11_90

#define any_HAVE_TR1_ADD_CONST          (!! any_COMPILER_GNUC_VERSION )
#define any_HAVE_TR1_REMOVE_REFERENCE   (!! any_COMPILER_GNUC_VERSION )
#define any_HAVE_TR1_TYPE_TRAITS        (!! any_COMPILER_GNUC_VERSION )

// Presence of C++14 language features:

#define any_HAVE_CONSTEXPR_14           any_CPP14_000

// Presence of C++17 language features:

#define any_HAVE_NODISCARD              any_CPP17_000

// Presence of C++ language features:

#if any_HAVE_CONSTEXPR_11
# define any_constexpr constexpr
#else
# define any_constexpr /*constexpr*/
#endif

#if any_HAVE_CONSTEXPR_14
# define any_constexpr14 constexpr
#else
# define any_constexpr14 /*constexpr*/
#endif

#if any_HAVE_NOEXCEPT
# define any_noexcept noexcept
#else
# define any_noexcept /*noexcept*/
#endif

#if any_HAVE_NULLPTR
# define any_nullptr nullptr
#else
# define any_nullptr NULL
#endif

#if any_HAVE_NODISCARD
# define any_nodiscard [[nodiscard]]
#else
# define any_nodiscard /*[[nodiscard]]*/
#endif

// additional includes:

#if any_CONFIG_NO_EXCEPTIONS
# include <cassert>
#else
# include <typeinfo>
#endif

#if ! any_HAVE_NULLPTR
# include <cstddef>
#endif

#if any_HAVE_INITIALIZER_LIST
# include <initializer_list>
#endif

#if any_HAVE_TYPE_TRAITS
# include <type_traits>
#elif any_HAVE_TR1_TYPE_TRAITS
# include <tr1/type_traits>
#endif

// Method enabling

#if any_CPP11_OR_GREATER

#define any_REQUIRES_0(...) \
    template< bool B = (__VA_ARGS__), typename std::enable_if<B, int>::type = 0 >

#define any_REQUIRES_T(...) \
    , typename = typename std::enable_if< (__VA_ARGS__), c10::any_lite::any_lite_detail::enabler >::type

#define any_REQUIRES_R(R, ...) \
    typename std::enable_if<__VA_ARGS__, R>::type

#define any_REQUIRES_A(...) \
    , typename std::enable_if<__VA_ARGS__, void*>::type = nullptr

#endif

//
// any:
//

namespace c10 {  namespace any_lite {

// C++11 emulation:

namespace std11 {

#if any_HAVE_ADD_CONST

using std::add_const;

#elif any_HAVE_TR1_ADD_CONST

using std::tr1::add_const;

#else

template< class T > struct add_const { typedef const T type; };

#endif // any_HAVE_ADD_CONST

#if any_HAVE_REMOVE_REFERENCE

using std::remove_reference;

#elif any_HAVE_TR1_REMOVE_REFERENCE

using std::tr1::remove_reference;

#else

template< class T > struct remove_reference     { typedef T type; };
template< class T > struct remove_reference<T&> { typedef T type; };

#endif // any_HAVE_REMOVE_REFERENCE

} // namespace std11

namespace any_lite_detail {

// for any_REQUIRES_T

/*enum*/ class enabler{};

} // namespace any_lite_detail

#if ! any_CONFIG_NO_EXCEPTIONS

class bad_any_cast : public std::bad_cast
{
public:
#if any_CPP11_OR_GREATER
    virtual const char* what() const any_noexcept
#else
    virtual const char* what() const throw()
#endif
   {
      return "any-lite: bad any_cast";
   }
};

#endif // any_CONFIG_NO_EXCEPTIONS

class any
{
public:
    any_constexpr any() any_noexcept
    : content( any_nullptr )
    {}

    any( any const & other )
    : content( other.content ? other.content->clone() : any_nullptr )
    {}

#if any_CPP11_OR_GREATER

    any( any && other ) any_noexcept
    : content( std::move( other.content ) )
    {
        other.content = any_nullptr;
    }

    template<
        class ValueType, class T = typename std::decay<ValueType>::type
        any_REQUIRES_T( ! std::is_same<T, any>::value )
    >
    any( ValueType && value ) any_noexcept
    : content( new holder<T>( std::move( value ) ) )
    {}

    template<
        class T, class... Args
        any_REQUIRES_T( std::is_constructible<T, Args&&...>::value )
    >
    explicit any( nonstd_lite_in_place_type_t(T), Args&&... args )
    : content( new holder<T>( T( std::forward<Args>(args)... ) ) )
    {}

    template<
        class T, class U, class... Args
        any_REQUIRES_T( std::is_constructible<T, std::initializer_list<U>&, Args&&...>::value )
    >
    explicit any( nonstd_lite_in_place_type_t(T), std::initializer_list<U> il, Args&&... args )
    : content( new holder<T>( T( il, std::forward<Args>(args)... ) ) )
    {}

#else

    template< class ValueType >
    any( ValueType const & value )
    : content( new holder<ValueType>( value ) )
    {}

#endif // any_CPP11_OR_GREATER

    ~any()
    {
        reset();
    }

    any & operator=( any const & other )
    {
        any( other ).swap( *this );
        return *this;
    }

#if any_CPP11_OR_GREATER

    any & operator=( any && other ) any_noexcept
    {
        any( std::move( other ) ).swap( *this );
        return *this;
    }

    template<
        class ValueType, class T = typename std::decay<ValueType>::type
        any_REQUIRES_T( ! std::is_same<T, any>::value )
    >
    any & operator=( ValueType && value )
    {
        any( std::move( value ) ).swap( *this );
        return *this;
    }

    template< class T, class... Args >
    void emplace( Args && ... args )
    {
        any( T( std::forward<Args>(args)... ) ).swap( *this );
    }

    template<
        class T, class U, class... Args
        any_REQUIRES_T( std::is_constructible<T, std::initializer_list<U>&, Args&&...>::value )
    >
    void emplace( std::initializer_list<U> il, Args&&... args )
    {
        any( T( il, std::forward<Args>(args)... ) ).swap( *this );
    }

#else

    template< class ValueType >
    any & operator=( ValueType const & value )
    {
        any( value ).swap( *this );
        return *this;
    }

#endif // any_CPP11_OR_GREATER

    void reset() any_noexcept
    {
        delete content; content = any_nullptr;
    }

    void swap( any & other ) any_noexcept
    {
        std::swap( content, other.content );
    }

    bool has_value() const any_noexcept
    {
        return content != any_nullptr;
    }

    const std::type_info & type() const any_noexcept
    {
        return has_value() ? content->type() : typeid( void );
    }

    //
    // non-standard:
    //

    template< class ValueType >
    const ValueType * to_ptr() const
    {
        return &( static_cast<holder<ValueType> *>( content )->held );
    }

    template< class ValueType >
    ValueType * to_ptr()
    {
        return &( static_cast<holder<ValueType> *>( content )->held );
    }

private:
    class placeholder
    {
    public:
        virtual ~placeholder()
        {
        }

        virtual std::type_info const & type() const = 0;

        virtual placeholder * clone() const = 0;
    };

    template< typename ValueType >
    class holder : public placeholder
    {
    public:
        holder( ValueType const & value )
        : held( value )
        {}

#if any_CPP11_OR_GREATER
        holder( ValueType && value )
        : held( std::move( value ) )
        {}
#endif

        virtual std::type_info const & type() const
        {
            return typeid( ValueType );
        }

        virtual placeholder * clone() const
        {
            return new holder( held );
        }

        ValueType held;
    };

    placeholder * content;
};

inline void swap( any & x, any & y ) any_noexcept
{
    x.swap( y );
}

#if any_CPP11_OR_GREATER

template< class T, class ...Args >
inline any make_any( Args&& ...args )
{
    return any( nonstd_lite_in_place_type(T), std::forward<Args>(args)...);
}

template< class T, class U, class ...Args >
inline any make_any( std::initializer_list<U> il, Args&& ...args )
{
    return any( nonstd_lite_in_place_type(T), il, std::forward<Args>(args)...);
}

#endif // any_CPP11_OR_GREATER

template<
    class ValueType
#if any_HAVE_DEFAULT_FUNCTION_TEMPLATE_ARG
//  any_REQUIRES_T(...) Allow for VC120 (VS2013):
    , typename = typename std::enable_if< (std::is_reference<ValueType>::value || std::is_copy_constructible<ValueType>::value), c10::any_lite::any_lite_detail::enabler >::type
#endif
>
any_nodiscard inline ValueType any_cast( any const & operand )
{
   const ValueType * result = any_cast< typename std11::add_const< typename std11::remove_reference<ValueType>::type >::type >( &operand );

#if any_CONFIG_NO_EXCEPTIONS
   assert( result );
#else
   if ( ! result )
   {
       throw bad_any_cast();
   }
#endif

   return *result;
}

template<
    class ValueType
#if any_HAVE_DEFAULT_FUNCTION_TEMPLATE_ARG
//  any_REQUIRES_T(...) Allow for VC120 (VS2013):
    , typename = typename std::enable_if< (std::is_reference<ValueType>::value || std::is_copy_constructible<ValueType>::value), c10::any_lite::any_lite_detail::enabler >::type
#endif
>
any_nodiscard inline ValueType any_cast( any & operand )
{
   const ValueType * result = any_cast< typename std11::remove_reference<ValueType>::type >( &operand );

#if any_CONFIG_NO_EXCEPTIONS
   assert( result );
#else
   if ( ! result )
   {
       throw bad_any_cast();
   }
#endif

   return *result;
}

#if any_CPP11_OR_GREATER

template<
    class ValueType
#if any_HAVE_DEFAULT_FUNCTION_TEMPLATE_ARG
    any_REQUIRES_T( std::is_reference<ValueType>::value || std::is_copy_constructible<ValueType>::value )
#endif
>
any_nodiscard inline ValueType any_cast( any && operand )
{
   const ValueType * result = any_cast< typename std11::remove_reference<ValueType>::type >( &operand );

#if any_CONFIG_NO_EXCEPTIONS
   assert( result );
#else
   if ( ! result )
   {
       throw bad_any_cast();
   }
#endif

   return *result;
}

#endif // any_CPP11_OR_GREATER

template< class ValueType >
any_nodiscard inline ValueType const * any_cast( any const * operand ) any_noexcept
{
    return operand != any_nullptr && operand->type() == typeid(ValueType) ? operand->to_ptr<ValueType>() : any_nullptr;
}

template<class ValueType >
any_nodiscard inline ValueType * any_cast( any * operand ) any_noexcept
{
    return operand != any_nullptr && operand->type() == typeid(ValueType) ? operand->to_ptr<ValueType>() : any_nullptr;
}

} // namespace any_lite

using namespace any_lite;

} // namespace nonstd

#endif // any_USES_STD_ANY

#endif // C10_UTIL_ANY_H_
