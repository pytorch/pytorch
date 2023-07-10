// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/integral_constant.hpp"
#include "ck/utility/type.hpp"
#include "ck/utility/functional.hpp"
#include "ck/utility/math.hpp"

namespace ck {

template <index_t, index_t, index_t>
struct static_for;

template <index_t...>
struct Sequence;

template <typename Seq, index_t I>
struct sequence_split;

template <typename>
struct sequence_reverse;

template <typename>
struct sequence_map_inverse;

template <typename>
struct is_valid_sequence_map;

template <index_t I, index_t... Is>
__host__ __device__ constexpr auto sequence_pop_front(Sequence<I, Is...>);

template <typename Seq>
__host__ __device__ constexpr auto sequence_pop_back(Seq);

template <index_t... Is>
struct Sequence
{
    using Type      = Sequence;
    using data_type = index_t;

    static constexpr index_t mSize = sizeof...(Is);

    __host__ __device__ static constexpr auto Size() { return Number<mSize>{}; }

    __host__ __device__ static constexpr auto GetSize() { return Size(); }

    __host__ __device__ static constexpr index_t At(index_t I)
    {
        // the last dummy element is to prevent compiler complain about empty array, when mSize = 0
        const index_t mData[mSize + 1] = {Is..., 0};
        return mData[I];
    }

    template <index_t I>
    __host__ __device__ static constexpr auto At(Number<I>)
    {
        static_assert(I < mSize, "wrong! I too large");

        return Number<At(I)>{};
    }

    template <index_t I>
    __host__ __device__ static constexpr auto Get(Number<I>)
    {
        return At(Number<I>{});
    }

    template <typename I>
    __host__ __device__ constexpr auto operator[](I i) const
    {
        return At(i);
    }

    template <index_t... IRs>
    __host__ __device__ static constexpr auto ReorderGivenNew2Old(Sequence<IRs...> /*new2old*/)
    {
        static_assert(sizeof...(Is) == sizeof...(IRs),
                      "wrong! reorder map should have the same size as Sequence to be rerodered");

        static_assert(is_valid_sequence_map<Sequence<IRs...>>::value, "wrong! invalid reorder map");

        return Sequence<Type::At(Number<IRs>{})...>{};
    }

    // MapOld2New is Sequence<...>
    template <typename MapOld2New>
    __host__ __device__ static constexpr auto ReorderGivenOld2New(MapOld2New)
    {
        static_assert(MapOld2New::Size() == Size(),
                      "wrong! reorder map should have the same size as Sequence to be rerodered");

        static_assert(is_valid_sequence_map<MapOld2New>::value, "wrong! invalid reorder map");

        return ReorderGivenNew2Old(typename sequence_map_inverse<MapOld2New>::type{});
    }

    __host__ __device__ static constexpr auto Reverse()
    {
        return typename sequence_reverse<Type>::type{};
    }

    __host__ __device__ static constexpr auto Front()
    {
        static_assert(mSize > 0, "wrong!");
        return At(Number<0>{});
    }

    __host__ __device__ static constexpr auto Back()
    {
        static_assert(mSize > 0, "wrong!");
        return At(Number<mSize - 1>{});
    }

    __host__ __device__ static constexpr auto PopFront() { return sequence_pop_front(Type{}); }

    __host__ __device__ static constexpr auto PopBack() { return sequence_pop_back(Type{}); }

    template <index_t... Xs>
    __host__ __device__ static constexpr auto PushFront(Sequence<Xs...>)
    {
        return Sequence<Xs..., Is...>{};
    }

    template <index_t... Xs>
    __host__ __device__ static constexpr auto PushFront(Number<Xs>...)
    {
        return Sequence<Xs..., Is...>{};
    }

    template <index_t... Xs>
    __host__ __device__ static constexpr auto PushBack(Sequence<Xs...>)
    {
        return Sequence<Is..., Xs...>{};
    }

    template <index_t... Xs>
    __host__ __device__ static constexpr auto PushBack(Number<Xs>...)
    {
        return Sequence<Is..., Xs...>{};
    }

    template <index_t... Ns>
    __host__ __device__ static constexpr auto Extract(Number<Ns>...)
    {
        return Sequence<Type::At(Number<Ns>{})...>{};
    }

    template <index_t... Ns>
    __host__ __device__ static constexpr auto Extract(Sequence<Ns...>)
    {
        return Sequence<Type::At(Number<Ns>{})...>{};
    }

    template <index_t I, index_t X>
    __host__ __device__ static constexpr auto Modify(Number<I>, Number<X>)
    {
        static_assert(I < Size(), "wrong!");

        using seq_split          = sequence_split<Type, I>;
        constexpr auto seq_left  = typename seq_split::left_type{};
        constexpr auto seq_right = typename seq_split::right_type{}.PopFront();

        return seq_left.PushBack(Number<X>{}).PushBack(seq_right);
    }

    template <typename F>
    __host__ __device__ static constexpr auto Transform(F f)
    {
        return Sequence<f(Is)...>{};
    }

    __host__ __device__ static void Print()
    {
        printf("{");
        printf("size %d, ", index_t{Size()});
        static_for<0, Size(), 1>{}([&](auto i) { printf("%d ", At(i).value); });
        printf("}");
    }
};

// merge sequence
template <typename Seq, typename... Seqs>
struct sequence_merge
{
    using type = typename sequence_merge<Seq, typename sequence_merge<Seqs...>::type>::type;
};

template <index_t... Xs, index_t... Ys>
struct sequence_merge<Sequence<Xs...>, Sequence<Ys...>>
{
    using type = Sequence<Xs..., Ys...>;
};

template <typename Seq>
struct sequence_merge<Seq>
{
    using type = Seq;
};

// generate sequence
template <index_t NSize, typename F>
struct sequence_gen
{
    template <index_t IBegin, index_t NRemain, typename G>
    struct sequence_gen_impl
    {
        static constexpr index_t NRemainLeft  = NRemain / 2;
        static constexpr index_t NRemainRight = NRemain - NRemainLeft;
        static constexpr index_t IMiddle      = IBegin + NRemainLeft;

        using type = typename sequence_merge<
            typename sequence_gen_impl<IBegin, NRemainLeft, G>::type,
            typename sequence_gen_impl<IMiddle, NRemainRight, G>::type>::type;
    };

    template <index_t I, typename G>
    struct sequence_gen_impl<I, 1, G>
    {
        static constexpr index_t Is = G{}(Number<I>{});
        using type                  = Sequence<Is>;
    };

    template <index_t I, typename G>
    struct sequence_gen_impl<I, 0, G>
    {
        using type = Sequence<>;
    };

    using type = typename sequence_gen_impl<0, NSize, F>::type;
};

// arithmetic sequence
template <index_t IBegin, index_t IEnd, index_t Increment>
struct arithmetic_sequence_gen
{
    struct F
    {
        __host__ __device__ constexpr index_t operator()(index_t i) const
        {
            return i * Increment + IBegin;
        }
    };

    using type0 = typename sequence_gen<(IEnd - IBegin) / Increment, F>::type;
    using type1 = Sequence<>;

    static constexpr bool kHasContent =
        (Increment > 0 && IBegin < IEnd) || (Increment < 0 && IBegin > IEnd);

    using type = typename conditional<kHasContent, type0, type1>::type;
};

// uniform sequence
template <index_t NSize, index_t I>
struct uniform_sequence_gen
{
    struct F
    {
        __host__ __device__ constexpr index_t operator()(index_t) const { return I; }
    };

    using type = typename sequence_gen<NSize, F>::type;
};

// reverse inclusive scan (with init) sequence
template <typename, typename, index_t>
struct sequence_reverse_inclusive_scan;

template <index_t I, index_t... Is, typename Reduce, index_t Init>
struct sequence_reverse_inclusive_scan<Sequence<I, Is...>, Reduce, Init>
{
    using old_scan = typename sequence_reverse_inclusive_scan<Sequence<Is...>, Reduce, Init>::type;

    static constexpr index_t new_reduce = Reduce{}(I, old_scan{}.Front());

    using type = typename sequence_merge<Sequence<new_reduce>, old_scan>::type;
};

template <index_t I, typename Reduce, index_t Init>
struct sequence_reverse_inclusive_scan<Sequence<I>, Reduce, Init>
{
    using type = Sequence<Reduce{}(I, Init)>;
};

template <typename Reduce, index_t Init>
struct sequence_reverse_inclusive_scan<Sequence<>, Reduce, Init>
{
    using type = Sequence<>;
};

// split sequence
template <typename Seq, index_t I>
struct sequence_split
{
    static constexpr index_t NSize = Seq{}.Size();

    using range0 = typename arithmetic_sequence_gen<0, I, 1>::type;
    using range1 = typename arithmetic_sequence_gen<I, NSize, 1>::type;

    using left_type  = decltype(Seq::Extract(range0{}));
    using right_type = decltype(Seq::Extract(range1{}));
};

// reverse sequence
template <typename Seq>
struct sequence_reverse
{
    static constexpr index_t NSize = Seq{}.Size();

    using seq_split = sequence_split<Seq, NSize / 2>;
    using type      = typename sequence_merge<
        typename sequence_reverse<typename seq_split::right_type>::type,
        typename sequence_reverse<typename seq_split::left_type>::type>::type;
};

template <index_t I>
struct sequence_reverse<Sequence<I>>
{
    using type = Sequence<I>;
};

template <index_t I0, index_t I1>
struct sequence_reverse<Sequence<I0, I1>>
{
    using type = Sequence<I1, I0>;
};

#if 1
template <typename Reduce, typename Seq, typename... Seqs>
struct sequence_reduce
{
    using type = typename sequence_reduce<Reduce,
                                          Seq,
                                          typename sequence_reduce<Reduce, Seqs...>::type>::type;
};

template <typename Reduce, index_t... Xs, index_t... Ys>
struct sequence_reduce<Reduce, Sequence<Xs...>, Sequence<Ys...>>
{
    using type = Sequence<Reduce{}(Xs, Ys)...>;
};

template <typename Reduce, typename Seq>
struct sequence_reduce<Reduce, Seq>
{
    using type = Seq;
};
#endif

template <typename Values, typename Ids, typename Compare>
struct sequence_sort_impl
{
    template <typename LeftValues,
              typename LeftIds,
              typename RightValues,
              typename RightIds,
              typename MergedValues,
              typename MergedIds,
              typename Comp>
    struct sorted_sequence_merge_impl
    {
        static constexpr bool choose_left = LeftValues::Front() < RightValues::Front();

        static constexpr index_t chosen_value =
            choose_left ? LeftValues::Front() : RightValues::Front();
        static constexpr index_t chosen_id = choose_left ? LeftIds::Front() : RightIds::Front();

        using new_merged_values = decltype(MergedValues::PushBack(Number<chosen_value>{}));
        using new_merged_ids    = decltype(MergedIds::PushBack(Number<chosen_id>{}));

        using new_left_values =
            typename conditional<choose_left, decltype(LeftValues::PopFront()), LeftValues>::type;
        using new_left_ids =
            typename conditional<choose_left, decltype(LeftIds::PopFront()), LeftIds>::type;

        using new_right_values =
            typename conditional<choose_left, RightValues, decltype(RightValues::PopFront())>::type;
        using new_right_ids =
            typename conditional<choose_left, RightIds, decltype(RightIds::PopFront())>::type;

        using merge = sorted_sequence_merge_impl<new_left_values,
                                                 new_left_ids,
                                                 new_right_values,
                                                 new_right_ids,
                                                 new_merged_values,
                                                 new_merged_ids,
                                                 Comp>;
        // this is output
        using merged_values = typename merge::merged_values;
        using merged_ids    = typename merge::merged_ids;
    };

    template <typename LeftValues,
              typename LeftIds,
              typename MergedValues,
              typename MergedIds,
              typename Comp>
    struct sorted_sequence_merge_impl<LeftValues,
                                      LeftIds,
                                      Sequence<>,
                                      Sequence<>,
                                      MergedValues,
                                      MergedIds,
                                      Comp>
    {
        using merged_values = typename sequence_merge<MergedValues, LeftValues>::type;
        using merged_ids    = typename sequence_merge<MergedIds, LeftIds>::type;
    };

    template <typename RightValues,
              typename RightIds,
              typename MergedValues,
              typename MergedIds,
              typename Comp>
    struct sorted_sequence_merge_impl<Sequence<>,
                                      Sequence<>,
                                      RightValues,
                                      RightIds,
                                      MergedValues,
                                      MergedIds,
                                      Comp>
    {
        using merged_values = typename sequence_merge<MergedValues, RightValues>::type;
        using merged_ids    = typename sequence_merge<MergedIds, RightIds>::type;
    };

    template <typename LeftValues,
              typename LeftIds,
              typename RightValues,
              typename RightIds,
              typename Comp>
    struct sorted_sequence_merge
    {
        using merge = sorted_sequence_merge_impl<LeftValues,
                                                 LeftIds,
                                                 RightValues,
                                                 RightIds,
                                                 Sequence<>,
                                                 Sequence<>,
                                                 Comp>;

        using merged_values = typename merge::merged_values;
        using merged_ids    = typename merge::merged_ids;
    };

    static constexpr index_t nsize = Values::Size();

    using split_unsorted_values = sequence_split<Values, nsize / 2>;
    using split_unsorted_ids    = sequence_split<Ids, nsize / 2>;

    using left_unsorted_values = typename split_unsorted_values::left_type;
    using left_unsorted_ids    = typename split_unsorted_ids::left_type;
    using left_sort          = sequence_sort_impl<left_unsorted_values, left_unsorted_ids, Compare>;
    using left_sorted_values = typename left_sort::sorted_values;
    using left_sorted_ids    = typename left_sort::sorted_ids;

    using right_unsorted_values = typename split_unsorted_values::right_type;
    using right_unsorted_ids    = typename split_unsorted_ids::right_type;
    using right_sort = sequence_sort_impl<right_unsorted_values, right_unsorted_ids, Compare>;
    using right_sorted_values = typename right_sort::sorted_values;
    using right_sorted_ids    = typename right_sort::sorted_ids;

    using merged_sorted = sorted_sequence_merge<left_sorted_values,
                                                left_sorted_ids,
                                                right_sorted_values,
                                                right_sorted_ids,
                                                Compare>;

    using sorted_values = typename merged_sorted::merged_values;
    using sorted_ids    = typename merged_sorted::merged_ids;
};

template <index_t ValueX, index_t ValueY, index_t IdX, index_t IdY, typename Compare>
struct sequence_sort_impl<Sequence<ValueX, ValueY>, Sequence<IdX, IdY>, Compare>
{
    static constexpr bool choose_x = Compare{}(ValueX, ValueY);

    using sorted_values =
        typename conditional<choose_x, Sequence<ValueX, ValueY>, Sequence<ValueY, ValueX>>::type;
    using sorted_ids = typename conditional<choose_x, Sequence<IdX, IdY>, Sequence<IdY, IdX>>::type;
};

template <index_t Value, index_t Id, typename Compare>
struct sequence_sort_impl<Sequence<Value>, Sequence<Id>, Compare>
{
    using sorted_values = Sequence<Value>;
    using sorted_ids    = Sequence<Id>;
};

template <typename Compare>
struct sequence_sort_impl<Sequence<>, Sequence<>, Compare>
{
    using sorted_values = Sequence<>;
    using sorted_ids    = Sequence<>;
};

template <typename Values, typename Compare>
struct sequence_sort
{
    using unsorted_ids = typename arithmetic_sequence_gen<0, Values::Size(), 1>::type;
    using sort         = sequence_sort_impl<Values, unsorted_ids, Compare>;

    // this is output
    using type                = typename sort::sorted_values;
    using sorted2unsorted_map = typename sort::sorted_ids;
};

template <typename Values, typename Less, typename Equal>
struct sequence_unique_sort
{
    template <typename RemainValues,
              typename RemainIds,
              typename UniquifiedValues,
              typename UniquifiedIds,
              typename Eq>
    struct sorted_sequence_uniquify_impl
    {
        static constexpr index_t current_value = RemainValues::Front();
        static constexpr index_t current_id    = RemainIds::Front();

        static constexpr bool is_unique_value = (current_value != UniquifiedValues::Back());

        using new_remain_values = decltype(RemainValues::PopFront());
        using new_remain_ids    = decltype(RemainIds::PopFront());

        using new_uniquified_values =
            typename conditional<is_unique_value,
                                 decltype(UniquifiedValues::PushBack(Number<current_value>{})),
                                 UniquifiedValues>::type;

        using new_uniquified_ids =
            typename conditional<is_unique_value,
                                 decltype(UniquifiedIds::PushBack(Number<current_id>{})),
                                 UniquifiedIds>::type;

        using uniquify = sorted_sequence_uniquify_impl<new_remain_values,
                                                       new_remain_ids,
                                                       new_uniquified_values,
                                                       new_uniquified_ids,
                                                       Eq>;

        // this is output
        using uniquified_values = typename uniquify::uniquified_values;
        using uniquified_ids    = typename uniquify::uniquified_ids;
    };

    template <typename UniquifiedValues, typename UniquifiedIds, typename Eq>
    struct sorted_sequence_uniquify_impl<Sequence<>,
                                         Sequence<>,
                                         UniquifiedValues,
                                         UniquifiedIds,
                                         Eq>
    {
        using uniquified_values = UniquifiedValues;
        using uniquified_ids    = UniquifiedIds;
    };

    template <typename SortedValues, typename SortedIds, typename Eq>
    struct sorted_sequence_uniquify
    {
        using uniquify = sorted_sequence_uniquify_impl<decltype(SortedValues::PopFront()),
                                                       decltype(SortedIds::PopFront()),
                                                       Sequence<SortedValues::Front()>,
                                                       Sequence<SortedIds::Front()>,
                                                       Eq>;

        using uniquified_values = typename uniquify::uniquified_values;
        using uniquified_ids    = typename uniquify::uniquified_ids;
    };

    using sort          = sequence_sort<Values, Less>;
    using sorted_values = typename sort::type;
    using sorted_ids    = typename sort::sorted2unsorted_map;

    using uniquify = sorted_sequence_uniquify<sorted_values, sorted_ids, Equal>;

    // this is output
    using type                = typename uniquify::uniquified_values;
    using sorted2unsorted_map = typename uniquify::uniquified_ids;
};

template <typename SeqMap>
struct is_valid_sequence_map : is_same<typename arithmetic_sequence_gen<0, SeqMap::Size(), 1>::type,
                                       typename sequence_sort<SeqMap, math::less<index_t>>::type>
{
};

template <typename SeqMap>
struct sequence_map_inverse
{
    template <typename X2Y, typename WorkingY2X, index_t XBegin, index_t XRemain>
    struct sequence_map_inverse_impl
    {
        static constexpr auto new_y2x =
            WorkingY2X::Modify(X2Y::At(Number<XBegin>{}), Number<XBegin>{});

        using type =
            typename sequence_map_inverse_impl<X2Y, decltype(new_y2x), XBegin + 1, XRemain - 1>::
                type;
    };

    template <typename X2Y, typename WorkingY2X, index_t XBegin>
    struct sequence_map_inverse_impl<X2Y, WorkingY2X, XBegin, 0>
    {
        using type = WorkingY2X;
    };

    using type =
        typename sequence_map_inverse_impl<SeqMap,
                                           typename uniform_sequence_gen<SeqMap::Size(), 0>::type,
                                           0,
                                           SeqMap::Size()>::type;
};

template <index_t... Xs, index_t... Ys>
__host__ __device__ constexpr bool operator==(Sequence<Xs...>, Sequence<Ys...>)
{
    return ((Xs == Ys) && ...);
}

template <index_t... Xs, index_t... Ys>
__host__ __device__ constexpr auto operator+(Sequence<Xs...>, Sequence<Ys...>)
{
    static_assert(sizeof...(Xs) == sizeof...(Ys), "wrong! inconsistent size");

    return Sequence<(Xs + Ys)...>{};
}

template <index_t... Xs, index_t... Ys>
__host__ __device__ constexpr auto operator-(Sequence<Xs...>, Sequence<Ys...>)
{
    static_assert(sizeof...(Xs) == sizeof...(Ys), "wrong! inconsistent size");

    return Sequence<(Xs - Ys)...>{};
}

template <index_t... Xs, index_t... Ys>
__host__ __device__ constexpr auto operator*(Sequence<Xs...>, Sequence<Ys...>)
{
    static_assert(sizeof...(Xs) == sizeof...(Ys), "wrong! inconsistent size");

    return Sequence<(Xs * Ys)...>{};
}

template <index_t... Xs, index_t... Ys>
__host__ __device__ constexpr auto operator/(Sequence<Xs...>, Sequence<Ys...>)
{
    static_assert(sizeof...(Xs) == sizeof...(Ys), "wrong! inconsistent size");

    return Sequence<(Xs / Ys)...>{};
}

template <index_t... Xs, index_t... Ys>
__host__ __device__ constexpr auto operator%(Sequence<Xs...>, Sequence<Ys...>)
{
    static_assert(sizeof...(Xs) == sizeof...(Ys), "wrong! inconsistent size");

    return Sequence<(Xs % Ys)...>{};
}

template <index_t... Xs, index_t Y>
__host__ __device__ constexpr auto operator+(Sequence<Xs...>, Number<Y>)
{
    return Sequence<(Xs + Y)...>{};
}

template <index_t... Xs, index_t Y>
__host__ __device__ constexpr auto operator-(Sequence<Xs...>, Number<Y>)
{
    return Sequence<(Xs - Y)...>{};
}

template <index_t... Xs, index_t Y>
__host__ __device__ constexpr auto operator*(Sequence<Xs...>, Number<Y>)
{
    return Sequence<(Xs * Y)...>{};
}

template <index_t... Xs, index_t Y>
__host__ __device__ constexpr auto operator/(Sequence<Xs...>, Number<Y>)
{
    return Sequence<(Xs / Y)...>{};
}

template <index_t... Xs, index_t Y>
__host__ __device__ constexpr auto operator%(Sequence<Xs...>, Number<Y>)
{
    return Sequence<(Xs % Y)...>{};
}

template <index_t Y, index_t... Xs>
__host__ __device__ constexpr auto operator+(Number<Y>, Sequence<Xs...>)
{
    return Sequence<(Y + Xs)...>{};
}

template <index_t Y, index_t... Xs>
__host__ __device__ constexpr auto operator-(Number<Y>, Sequence<Xs...>)
{
    return Sequence<(Y - Xs)...>{};
}

template <index_t Y, index_t... Xs>
__host__ __device__ constexpr auto operator*(Number<Y>, Sequence<Xs...>)
{
    return Sequence<(Y * Xs)...>{};
}

template <index_t Y, index_t... Xs>
__host__ __device__ constexpr auto operator/(Number<Y>, Sequence<Xs...>)
{
    return Sequence<(Y / Xs)...>{};
}

template <index_t Y, index_t... Xs>
__host__ __device__ constexpr auto operator%(Number<Y>, Sequence<Xs...>)
{
    return Sequence<(Y % Xs)...>{};
}

template <index_t I, index_t... Is>
__host__ __device__ constexpr auto sequence_pop_front(Sequence<I, Is...>)
{
    return Sequence<Is...>{};
}

template <typename Seq>
__host__ __device__ constexpr auto sequence_pop_back(Seq)
{
    static_assert(Seq::Size() > 0, "wrong! cannot pop an empty Sequence!");
    return sequence_pop_front(Seq::Reverse()).Reverse();
}

template <typename... Seqs>
__host__ __device__ constexpr auto merge_sequences(Seqs...)
{
    return typename sequence_merge<Seqs...>::type{};
}

template <typename F, index_t... Xs>
__host__ __device__ constexpr auto transform_sequences(F f, Sequence<Xs...>)
{
    return Sequence<f(Xs)...>{};
}

template <typename F, index_t... Xs, index_t... Ys>
__host__ __device__ constexpr auto transform_sequences(F f, Sequence<Xs...>, Sequence<Ys...>)
{
    static_assert(Sequence<Xs...>::mSize == Sequence<Ys...>::mSize, "Dim not the same");

    return Sequence<f(Xs, Ys)...>{};
}

template <typename F, index_t... Xs, index_t... Ys, index_t... Zs>
__host__ __device__ constexpr auto
transform_sequences(F f, Sequence<Xs...>, Sequence<Ys...>, Sequence<Zs...>)
{
    static_assert(Sequence<Xs...>::mSize == Sequence<Ys...>::mSize &&
                      Sequence<Xs...>::mSize == Sequence<Zs...>::mSize,
                  "Dim not the same");

    return Sequence<f(Xs, Ys, Zs)...>{};
}

template <typename Seq, typename Reduce, index_t Init>
__host__ __device__ constexpr auto reverse_inclusive_scan_sequence(Seq, Reduce, Number<Init>)
{
    return typename sequence_reverse_inclusive_scan<Seq, Reduce, Init>::type{};
}

template <typename Seq, typename Reduce, index_t Init>
__host__ __device__ constexpr auto reverse_exclusive_scan_sequence(Seq, Reduce, Number<Init>)
{
    return reverse_inclusive_scan_sequence(Seq::PopFront(), Reduce{}, Number<Init>{})
        .PushBack(Number<Init>{});
}

template <typename Seq, typename Reduce, index_t Init>
__host__ __device__ constexpr auto inclusive_scan_sequence(Seq, Reduce, Number<Init>)
{
    return reverse_inclusive_scan_sequence(Seq{}.Reverse(), Reduce{}, Number<Init>{}).Reverse();
}

template <typename Seq, index_t... Is>
__host__ __device__ constexpr auto pick_sequence_elements_by_ids(Seq, Sequence<Is...> /* ids */)
{
    return Sequence<Seq::At(Number<Is>{})...>{};
}

#if 1
namespace detail {
template <typename WorkSeq, typename RemainSeq, typename RemainMask>
struct pick_sequence_elements_by_mask_impl
{
    using new_work_seq = typename conditional<RemainMask::Front(),
                                              decltype(WorkSeq::PushBack(RemainSeq::Front())),
                                              WorkSeq>::type;

    using type =
        typename pick_sequence_elements_by_mask_impl<new_work_seq,
                                                     decltype(RemainSeq::PopFront()),
                                                     decltype(RemainMask::PopFront())>::type;
};

template <typename WorkSeq>
struct pick_sequence_elements_by_mask_impl<WorkSeq, Sequence<>, Sequence<>>
{
    using type = WorkSeq;
};

} // namespace detail

template <typename Seq, typename Mask>
__host__ __device__ constexpr auto pick_sequence_elements_by_mask(Seq, Mask)
{
    static_assert(Seq::Size() == Mask::Size(), "wrong!");

    return typename detail::pick_sequence_elements_by_mask_impl<Sequence<>, Seq, Mask>::type{};
}

namespace detail {
template <typename WorkSeq, typename RemainValues, typename RemainIds>
struct modify_sequence_elements_by_ids_impl
{
    using new_work_seq = decltype(WorkSeq::Modify(RemainIds::Front(), RemainValues::Front()));

    using type =
        typename modify_sequence_elements_by_ids_impl<new_work_seq,
                                                      decltype(RemainValues::PopFront()),
                                                      decltype(RemainIds::PopFront())>::type;
};

template <typename WorkSeq>
struct modify_sequence_elements_by_ids_impl<WorkSeq, Sequence<>, Sequence<>>
{
    using type = WorkSeq;
};
} // namespace detail

template <typename Seq, typename Values, typename Ids>
__host__ __device__ constexpr auto modify_sequence_elements_by_ids(Seq, Values, Ids)
{
    static_assert(Values::Size() == Ids::Size() && Seq::Size() >= Values::Size(), "wrong!");

    return typename detail::modify_sequence_elements_by_ids_impl<Seq, Values, Ids>::type{};
}
#endif

template <typename Seq, typename Reduce, index_t Init>
__host__ __device__ constexpr index_t
reduce_on_sequence(Seq, Reduce f, Number<Init> /*initial_value*/)
{
    index_t result = Init;

    for(index_t i = 0; i < Seq::Size(); ++i)
    {
        result = f(result, Seq::At(i));
    }

    return result;
}

// TODO: a generic any_of for any container
template <typename Seq, typename F>
__host__ __device__ constexpr bool sequence_any_of(Seq, F f)
{
    bool flag = false;

    for(index_t i = 0; i < Seq::Size(); ++i)
    {
        flag = flag || f(Seq::At(i));
    }

    return flag;
}

// TODO: a generic all_of for any container
template <typename Seq, typename F>
__host__ __device__ constexpr bool sequence_all_of(Seq, F f)
{
    bool flag = true;

    for(index_t i = 0; i < Seq::Size(); ++i)
    {
        flag = flag && f(Seq::At(i));
    }

    return flag;
}

template <typename Sx, typename Sy>
using sequence_merge_t = typename sequence_merge<Sx, Sy>::type;

template <index_t NSize, index_t I>
using uniform_sequence_gen_t = typename uniform_sequence_gen<NSize, I>::type;

} // namespace ck
