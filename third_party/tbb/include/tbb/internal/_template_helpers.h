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

#ifndef __TBB_template_helpers_H
#define __TBB_template_helpers_H

#include <utility>
#include <cstddef>

namespace tbb { namespace internal {

//! Enables one or the other code branches
template<bool Condition, typename T = void> struct enable_if {};
template<typename T> struct enable_if<true, T> { typedef T type; };

//! Strips its template type argument from cv- and ref-qualifiers
template<typename T> struct strip                     { typedef T type; };
template<typename T> struct strip<const T>            { typedef T type; };
template<typename T> struct strip<volatile T>         { typedef T type; };
template<typename T> struct strip<const volatile T>   { typedef T type; };
template<typename T> struct strip<T&>                 { typedef T type; };
template<typename T> struct strip<const T&>           { typedef T type; };
template<typename T> struct strip<volatile T&>        { typedef T type; };
template<typename T> struct strip<const volatile T&>  { typedef T type; };
//! Specialization for function pointers
template<typename T> struct strip<T(&)()>             { typedef T(*type)(); };
#if __TBB_CPP11_RVALUE_REF_PRESENT
template<typename T> struct strip<T&&>                { typedef T type; };
template<typename T> struct strip<const T&&>          { typedef T type; };
template<typename T> struct strip<volatile T&&>       { typedef T type; };
template<typename T> struct strip<const volatile T&&> { typedef T type; };
#endif
//! Specialization for arrays converts to a corresponding pointer
template<typename T, std::size_t N> struct strip<T(&)[N]>                { typedef T* type; };
template<typename T, std::size_t N> struct strip<const T(&)[N]>          { typedef const T* type; };
template<typename T, std::size_t N> struct strip<volatile T(&)[N]>       { typedef volatile T* type; };
template<typename T, std::size_t N> struct strip<const volatile T(&)[N]> { typedef const volatile T* type; };

//! Detects whether two given types are the same
template<class U, class V> struct is_same_type      { static const bool value = false; };
template<class W>          struct is_same_type<W,W> { static const bool value = true; };

template<typename T> struct is_ref { static const bool value = false; };
template<typename U> struct is_ref<U&> { static const bool value = true; };

#if __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT
//! std::void_t internal implementation (to avoid GCC < 4.7 "template aliases" absence)
template<typename...> struct void_t { typedef void type; };
#endif

#if __TBB_CPP11_RVALUE_REF_PRESENT && __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT

//! Allows to store a function parameter pack as a variable and later pass it to another function
template< typename... Types >
struct stored_pack;

template<>
struct stored_pack<>
{
    typedef stored_pack<> pack_type;
    stored_pack() {}

    // Friend front-end functions
    template< typename F, typename Pack > friend void call( F&& f, Pack&& p );
    template< typename Ret, typename F, typename Pack > friend Ret call_and_return( F&& f, Pack&& p );

protected:
    // Ideally, ref-qualified non-static methods would be used,
    // but that would greatly reduce the set of compilers where it works.
    template< typename Ret, typename F, typename... Preceding >
    static Ret call( F&& f, const pack_type& /*pack*/, Preceding&&... params ) {
        return std::forward<F>(f)( std::forward<Preceding>(params)... );
    }
    template< typename Ret, typename F, typename... Preceding >
    static Ret call( F&& f, pack_type&& /*pack*/, Preceding&&... params ) {
        return std::forward<F>(f)( std::forward<Preceding>(params)... );
    }
};

template< typename T, typename... Types >
struct stored_pack<T, Types...> : stored_pack<Types...>
{
    typedef stored_pack<T, Types...> pack_type;
    typedef stored_pack<Types...> pack_remainder;
    // Since lifetime of original values is out of control, copies should be made.
    // Thus references should be stripped away from the deduced type.
    typename strip<T>::type leftmost_value;

    // Here rvalue references act in the same way as forwarding references,
    // as long as class template parameters were deduced via forwarding references.
    stored_pack( T&& t, Types&&... types )
    : pack_remainder(std::forward<Types>(types)...), leftmost_value(std::forward<T>(t)) {}

    // Friend front-end functions
    template< typename F, typename Pack > friend void call( F&& f, Pack&& p );
    template< typename Ret, typename F, typename Pack > friend Ret call_and_return( F&& f, Pack&& p );

protected:
    template< typename Ret, typename F, typename... Preceding >
    static Ret call( F&& f, pack_type& pack, Preceding&&... params ) {
        return pack_remainder::template call<Ret>(
            std::forward<F>(f), static_cast<pack_remainder&>(pack),
            std::forward<Preceding>(params)... , pack.leftmost_value
        );
    }
    template< typename Ret, typename F, typename... Preceding >
    static Ret call( F&& f, const pack_type& pack, Preceding&&... params ) {
        return pack_remainder::template call<Ret>(
            std::forward<F>(f), static_cast<const pack_remainder&>(pack),
            std::forward<Preceding>(params)... , pack.leftmost_value
        );
    }
    template< typename Ret, typename F, typename... Preceding >
    static Ret call( F&& f, pack_type&& pack, Preceding&&... params ) {
        return pack_remainder::template call<Ret>(
            std::forward<F>(f), static_cast<pack_remainder&&>(pack),
            std::forward<Preceding>(params)... , std::move(pack.leftmost_value)
        );
    }
};

//! Calls the given function with arguments taken from a stored_pack
template< typename F, typename Pack >
void call( F&& f, Pack&& p ) {
    strip<Pack>::type::template call<void>( std::forward<F>(f), std::forward<Pack>(p) );
}

template< typename Ret, typename F, typename Pack >
Ret call_and_return( F&& f, Pack&& p ) {
    return strip<Pack>::type::template call<Ret>( std::forward<F>(f), std::forward<Pack>(p) );
}

template< typename... Types >
stored_pack<Types...> save_pack( Types&&... types ) {
    return stored_pack<Types...>( std::forward<Types>(types)... );
}

#endif /* __TBB_CPP11_RVALUE_REF_PRESENT && __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT */

#if __TBB_CPP14_INTEGER_SEQUENCE_PRESENT

using std::index_sequence;
using std::make_index_sequence;

#elif __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT && __TBB_CPP11_TEMPLATE_ALIASES_PRESENT

template<std::size_t... S> class index_sequence {};

template<std::size_t N, std::size_t... S>
struct make_index_sequence_impl : make_index_sequence_impl < N - 1, N - 1, S... > {};

template<std::size_t... S>
struct make_index_sequence_impl <0, S...> {
    using type = index_sequence<S...>;
};

template<std::size_t N>
using make_index_sequence = typename tbb::internal::make_index_sequence_impl<N>::type;

#endif /* __TBB_CPP14_INTEGER_SEQUENCE_PRESENT */

} } // namespace internal, namespace tbb

#endif /* __TBB_template_helpers_H */
