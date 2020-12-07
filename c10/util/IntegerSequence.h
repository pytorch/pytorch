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
 * Take the first n elements from an integer_sequence/index_sequence.
 * Example:
 *  std::index_sequence<5, 8> == take_t<2, std::index_sequence<5, 8, 3, 5, 6>>
 */
template<class ISeq, size_t N>
using take_t = typelist_to_iseq_t<typelist::take_t<iseq_to_typelist_t<ISeq>, N>>;

/**
 * Drop the first n elements from an integer_sequence/index_sequence.
 * Example:
 *  std::index_sequence<3, 5, 6> == drop_t<2, std::index_sequence<5, 8, 3, 5, 6>>
 */
template<class ISeq, size_t N>
using drop_t = typelist_to_iseq_t<typelist::drop_t<iseq_to_typelist_t<ISeq>, N>>;

/**
 * Make an index_sequence representing a range of values.
 * Begin is inclusive, end is exclusive.
 * Example:
 *  range_t<3, 8> == std::index_sequence<3, 4, 5, 6, 7>
 */
template<size_t begin, size_t end>
struct range {
    static_assert(end >= begin, "end must be larger than begin");
    using type = drop_t<std::make_index_sequence<end>, begin>;
};
template<size_t begin, size_t end>
using range_t = typename range<begin, end>::type;


/**
 * Insert an integer at a given position.
 * Example:
 *   insert_t<std::index_sequence<5, 8, 3, 5, 6>, 2, 4>
 *     == std::index_sequence<5, 8, 4, 3, 5, 6>
 */
template<class ISeq, size_t Pos, typename ISeq::value_type Value>
using insert_t = typelist_to_iseq_t<
    typelist::insert_t<iseq_to_typelist_t<ISeq>, Pos, CompileTimeInteger<typename ISeq::value_type, Value>>>;

/**
 * Insert a list of integers at given positions.
 * This expects to get a typelist with index_sequence<Pos, Value> pairs.
 * Insertions happen in the order they're provided in that list.
 * Example:
 *   insert_all_t<std::index_sequence<5, 8, 3>, typelist<index_sequence<2, 4>, index_sequence<1, 10>>>
 *     == std::index_sequence<5, 10, 8, 4, 3>
 */
template<class ISeq, class Insertions>
struct insert_all {
    static_assert(false_t<ISeq>::value, "In iseq::insert_all_t<ISeq, Insertions>, the Insertions must be a typelist of 2-element index sequences [Pos, Value].");
};
template<class ISeq>
struct insert_all<ISeq, typelist::typelist<>> {
    using type = ISeq;
};
template<class ISeq, typename ISeq::value_type HeadInsertionPos, typename ISeq::value_type HeadInsertionValue, class... TailInsertions>
struct insert_all<ISeq, typelist::typelist<std::integer_sequence<typename ISeq::value_type, HeadInsertionPos, HeadInsertionValue>, TailInsertions...>> {
    using type = typename insert_all<insert_t<ISeq, HeadInsertionPos, HeadInsertionValue>, typelist::typelist<TailInsertions...>>::type;
};
template<class ISeq, class Insertions>
using insert_all_t = typename insert_all<ISeq, Insertions>::type;

/**
 * Removes an integer at a given position.
 * Example:
 *  remove_by_index_t<std::index_sequence<5, 8, 3, 5, 6>, 2>
 *    == std::index_sequence<5, 8, 5, 6>
 */
template<class ISeq, size_t Pos>
using remove_by_index_t = typelist_to_iseq_t<typelist::remove_by_index_t<iseq_to_typelist_t<ISeq>, Pos>>;

/**
 * Remove a list of indices from the list.
 * The indices are removed one after the other in order, so later indices will be evaluated
 * relative to a modified list where the first indices were already removed.
 * This means that if you want all indices to be evaluated relative to the
 * original list, you have to pass in indices in a decreasing order.
 * Example:
 *   remove_all_by_index_t<std::index_sequence<5, 8, 3, 5, 6>, std::index_sequence<2, 3, 1>>
 *     == std::index_sequence<5, 5>
 */
 template<class ISeq, class IndicesToRemove>
 struct remove_all_by_index {
     static_assert(false_t<ISeq>::value, "In iseq::remove_all_by_index_t<ISeq, IndicesToRemove>, both ISeq and the Insertions must be a std::index_sequence.");
 };
 template<class ISeq>
 struct remove_all_by_index<ISeq, std::index_sequence<>> {
     using type = ISeq;
 };
 template<class ISeq, size_t HeadIndexToRemove, size_t... TailIndicesToRemove>
 struct remove_all_by_index<ISeq, std::index_sequence<HeadIndexToRemove, TailIndicesToRemove...>> {
     using type = typename remove_all_by_index<remove_by_index_t<ISeq, HeadIndexToRemove>, std::index_sequence<TailIndicesToRemove...>>::type;
 };
 template<class ISeq, class IndicesToRemove>
 using remove_all_by_index_t = typename remove_all_by_index<ISeq, IndicesToRemove>::type;

/**
 * Concatenate multiple integer sequences
 * Example:
 *   concat_t<std::index_sequence<2, 5, 3>, std::index_sequence<4, 2>, std::index_sequence<5>>
 *     == std::index_sequence<2, 5, 3, 4, 2, 5>
 */
template<class... ISeqs>
using concat_t = typelist_to_iseq_t<typelist::concat_t<iseq_to_typelist_t<ISeqs>...>>;

/**
 * Take an integer sequence and return a typelist of integer sequences where
 * each element is a 2-element integer sequence [index, value].
 * Example:
 *  zip_with_index_t<std::index_sequence<3, 2, 95>>
 *   == typelist<std::index_sequence<0, 3>, std::index_sequence<1, 2>, std::index_sequence<2, 95>>
 */
namespace detail {
    template<class ISeq, class IndexISeq>
    struct zip_with_index_ {
        static_assert(false_t<ISeq>::value, "In iseq::zip_with_index<ISeq>, the ISeq argument must be std::integer_sequence<...>.");
    };
    template<class IntType, IntType... ISeq, IntType... IndexISeq>
    struct zip_with_index_<std::integer_sequence<IntType, ISeq...>, std::integer_sequence<IntType, IndexISeq...>> {
        using type = typelist::typelist<std::integer_sequence<IntType, IndexISeq, ISeq>...>;
    };
}
template<class ISeq>
using zip_with_index_t =
    typename detail::zip_with_index_<ISeq, std::make_integer_sequence<typename ISeq::value_type, ISeq::size()>>::type;

/**
 * Reverse an integer sequence
 * Example:
 *   reverse_t<std::index_sequence<2, 5, 3>> == std::index_sequence<3, 5, 2>
 */
template<class ISeq>
using reverse_t = typelist_to_iseq_t<typelist::reverse_t<iseq_to_typelist_t<ISeq>>>;

/**
  * Set one of the elements of an integer sequence to a different value
  * Example:
  *   set_t<std::index_sequence<2, 5, 3, 4>, 2, 100> == std::index_sequence<2, 5, 100, 4>
  */
template<class ISeq, size_t Index, typename ISeq::value_type NewValue>
using set_t = typelist_to_iseq_t<typelist::set_t<
    iseq_to_typelist_t<ISeq>,
    Index,
    CompileTimeInteger<typename ISeq::value_type, NewValue>>>;

/**
  * Set multiple elements of an integer sequence at once.
  * The ValuesToSet parameter expects a typelist where each element
  * is an index_sequence<Index, Value> representing one element to change.
  *
  * Example:
  *   set_all_t<std::index_sequence<5, 8, 3>, typelist<index_sequence<2, 4>, index_sequence<1, 10>>>
  *     == std::index_sequence<5, 10, 4>
  */
template<class ISeq, class ValuesToSet>
struct set_all {
    static_assert(false_t<ISeq>::value, "In iseq::set_all_t<ISeq, ValuesToSet>, the ValuesToSet must be a typelist of 2-element index sequences [Pos, Value].");
};
template<class ISeq>
struct set_all<ISeq, typelist::typelist<>> {
    using type = ISeq;
};
template<class ISeq, typename ISeq::value_type HeadSetIndex, typename ISeq::value_type HeadValueToSet, class... TailValuesToSet>
struct set_all<ISeq, typelist::typelist<std::integer_sequence<typename ISeq::value_type, HeadSetIndex, HeadValueToSet>, TailValuesToSet...>> {
    using type = typename set_all<set_t<ISeq, HeadSetIndex, HeadValueToSet>, typelist::typelist<TailValuesToSet...>>::type;
};
template<class ISeq, class ValuesToSet>
using set_all_t = typename set_all<ISeq, ValuesToSet>::type;


/**
 * A permutation of length N is an integer sequence containing of the numbers 0..(N-1) in an arbitrary order.
 * It can be interpreted as a permutation function that can reorder integer sequences of length N by their element indices.
 * The permutation [2, 0, 1] is interpreted as a reordering function that reorders elements [0 -> 2, 1 -> 0, 2 -> 1]
 * (i.e. the element previously at position 0 goes to position 2, the element previously at position 1 goes to position 0
 * and the element previously at position 2 goes to position 1).
 *
 * Example:
 *  permutation<std::index_sequence<2, 0, 1>>::template typelist_t<typelist<int64_t, std::string, double>>
 *     == typelist<double, int64_t, std::string>
 */
template<class Indices>
struct permutation {
    static_assert(guts::false_t<Indices>::value, "In permutation<Indices>, Indices must be a std::index_sequence.");
};
template<size_t... Indices>
struct permutation<std::index_sequence<Indices...>> {
private:
    using indices = std::index_sequence<Indices...>;

    // A valid permutation must contain all the numbers 0..(N-1).
    // Let's assert this. Note that asserting this also asserts that each number is only contained
    // once since there's exactly N numbers in the permutation.
    template<class CompileTimeInt>
    using index_exists_in_permutation_ = typelist::contains<
        iseq_to_typelist_t<indices>,
        CompileTimeInt
    >;
    static_assert(
        typelist::all<
            index_exists_in_permutation_,
            iseq_to_typelist_t<std::make_index_sequence<sizeof...(Indices)>>
        >::value,
        "A valid permutation must contain all the numbers 0..(N-1)");

    template<class TypeList>
    struct apply_to_typelist {
        static_assert(false_t<TypeList>(),
            "In iseq::permutation::apply_to_typelist<TypeList>, TypeList must be a guts::typelist::typelist.");
    };
    template<class... Types>
    struct apply_to_typelist<guts::typelist::typelist<Types...>> {
        using typelist = guts::typelist::typelist<Types...>;
        static_assert(sizeof...(Indices) == sizeof...(Types),
            "In iseq::permutation::apply_to_typelist<TypeList>, the permutation length and typelist length must be equal.");
        using type = guts::typelist::typelist<guts::typelist::element_t<Indices, typelist>...>;
    };
public:
    /**
     * Apply the permutation to a typelist, reordering the types in the list.
     * Example:
     *  permutation<std::index_sequence<2, 0, 1>>::template apply_to_typelist_t<typelist<int64_t, std::string, double>>
     *     == typelist<double, int64_t, std::string>
     */
    template<class TypeList>
    using apply_to_typelist_t = typename apply_to_typelist<TypeList>::type;;

    /**
     * Apply the permutation to a tuple, reordering the elements in the tuple.
     * Example:
     *  permutation<std::index_sequence<2, 0, 1>>::apply_to_tuple(std::tuple<int, float, std::string>(2, 1.2f, "hello"))
     *   == std::tuple<std::string, int, float>("hello", 2, 1.2f)
     */
    template<class Tuple>
    static constexpr decltype(auto) apply_to_tuple(Tuple&& t) {
        static_assert(std::tuple_size<Tuple>::value == sizeof...(Indices),
            "In iseq::permutation::apply_to_tuple(Tuple), the permutation size and tuple size must be equal");
        return guts::typelist::to_tuple_t<apply_to_typelist_t<guts::typelist::from_tuple_t<Tuple>>>(
            std::get<Indices>(std::forward<Tuple>(t))...);
    }

private:
    template<class OtherPermutation>
    struct and_then {
        static_assert(guts::false_t<OtherPermutation>::value,
            "In permutation::and_then<OtherPermutation>, OtherPermutation must be an permutation.");
    };
    template<class OtherPermutationIndices>
    struct and_then<permutation<OtherPermutationIndices>> {
        static_assert(std::index_sequence<Indices...>::size() == OtherPermutationIndices::size(),
            "In permutation::and_then(other_permutation), both permutation and other_permutation must have the same length");
        using type = permutation<guts::iseq::typelist_to_iseq_t<
            apply_to_typelist_t<guts::iseq::iseq_to_typelist_t<OtherPermutationIndices>>
        >>;
    };
public:
    /**
     * Combine two permutations into one performing both.
     * The resulting permutation will first perform the `this` permutation and then `OtherPermutation`.
     * Example:
     *  permutation<std::index_sequence<3, 2, 4, 0, 1>>::and_then_t<permutation<std::index_sequence<3, 1, 4, 0, 2>>>
     *    == permutation<std::index_sequence<0, 4, 2, 3, 1>>
     */
    template<class OtherPermutation>
    using and_then_t = typename and_then<OtherPermutation>::type;

    /**
      * Invert the permutation function in the mathematical sense.
      * This means that permutation::and_then<permutation::inverted>
      * is the identity function.
      */
    // Note: The implementation here first builds an arbitraty index sequence
    // with the right length and then overwrites each element with the correct value
    // using set_all_t.
    using inverted_t = permutation<set_all_t<
        std::make_index_sequence<indices::size()>,
        guts::typelist::map_t<guts::iseq::reverse_t, zip_with_index_t<indices>>
    >>;
};

}
}
}
