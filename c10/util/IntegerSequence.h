#pragma once

#include <c10/util/TypeList.h>

/**
 * This file offers compile-time manipulations of std::integer_sequence objects,
 * allowing metaprograms to work with lists of ints.
 */

namespace c10 {
namespace guts {
namespace iseq {

/**
 * A CompileTimeInteger wraps an integer into a type to lift integers into the type system.
 * CompileTimeInteger<size_t, 2> is a type representing 'size_t 2' that can for example be
 * stored in type lists.
 * This type is mostly used to map between a std::integer_sequence and a typelist<CompileTimeInteger...>
 * so that we can easily port all the algorithms we have available for typelist to std::integer_sequence.
 */
template<class T, T value_>
struct CompileTimeInteger {
    using IntType = T;
    static constexpr T value() {
        return value_;
    }
};

/**
 * Take an integer_sequence/index_sequence and create a typelist using CompileTimeInteger.
 *
 * Example:
 *  iseq_to_typelist_t<std::index_sequence<2, 3>> == typelist<CompileTimeInteger<size_t, 2>, CompileTimeInteger<size_t, 3>>
 */
template<class IntegerSequence>
struct iseq_to_typelist {};
template<class IntType, IntType... integers>
struct iseq_to_typelist<std::integer_sequence<IntType, integers...>> {
    using type = guts::typelist::typelist<CompileTimeInteger<IntType, integers>...>;
};
template<class IntegerSequence>
using iseq_to_typelist_t = typename iseq_to_typelist<IntegerSequence>::type;


/**
 * Take a typelist made from CompileTimeInteger and create an integer_sequence/index_sequence from it.
 *
 * Example:
 *  std::index_sequence<2, 3> == typelist_to_iseq_t<typelist<CompileTimeInteger<size_t, 2>, CompileTimeInteger<size_t, 3>>>
 */
template<class TypeList>
struct typelist_to_iseq {};
template<class... Types>
struct typelist_to_iseq<guts::typelist::typelist<Types...>> {
    using IntType = typename guts::typelist::head_t<guts::typelist::typelist<Types...>>::IntType;
    static_assert(
        guts::conjunction<
            std::is_same<CompileTimeInteger<typename Types::IntType, Types::value()>, Types>...
        >::value, "All types must be instances of CompileTimeInteger"
    );
    static_assert(
        guts::conjunction<
            std::is_same<IntType, typename Types::IntType>...
        >::value, "All types must be based on the same IntType");
    using type = std::integer_sequence<IntType, Types::value()...>;
};
template<>
struct typelist_to_iseq<guts::typelist::typelist<>> {
    using type = std::index_sequence<>;
};
template<class TypeList>
using typelist_to_iseq_t = typename typelist_to_iseq<TypeList>::type;

/**
 * Drop the first n elements from an integer_sequence/index_sequence.
 * Example:
 *  std::index_sequence<3, 5, 6> == drop_t<2, std::index_sequence<5, 8, 3, 5, 6>>
 */
template<class ISeq, size_t N>
using drop_t = typelist_to_iseq_t<typelist::drop_t<iseq_to_typelist_t<ISeq>, N>>;

/**
 * Concatenate multiple integer sequences
 * Example:
 *   concat_t<std::index_sequence<2, 5, 3>, std::index_sequence<4, 2>, std::index_sequence<5>>
 *     == std::index_sequence<2, 5, 3, 4, 2, 5>
 */
template<class... ISeqs>
using concat_t = typelist_to_iseq_t<typelist::concat_t<iseq_to_typelist_t<ISeqs>...>>;

}
}
}
